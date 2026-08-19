//! Fuzzy substring search over StringZilla's `szs` kernels, where `sz-find` matches literally.
//!
//! A query is expanded into every string within `--max-distance` edits of it, and those variants
//! are matched exactly by one Aho-Corasick automaton. The edit model therefore lives in the
//! vocabulary and its BM25 weights rather than in a substitution matrix, so `--max-distance` is an
//! edit budget in every mode and digits never share a scoring class with each other.
//!
//! `--cost keyboard` draws substitutions and insertions from physically adjacent keys instead of
//! the whole alphabet, which is what keeps a two-edit ball affordable.
//!
//! Every query's variants are pooled into one dictionary, so the corpus is walked once however many
//! `--pattern` flags are given. BM25 scores that walk: a strictly positive weight per variant makes
//! a positive score mean "matched", and `find_into` runs afterwards, over survivors only, when
//! spans are actually asked for.
//!
//! Exit: 0 matched something, 1 matched nothing, 2 could not run.

use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::szs::{
    AnyBytesTape, Bm25Params, CaseSensitivity, DeviceScope, OverlapPolicy, Substrings,
    SubstringsMatch,
};

use shared::folds::Fold;
use shared::keyboards::Keyboard;
use shared::misspellings;
use shared::*;

// region: Effort

/// How hard the search tries, as an ordered ladder where each rung matches everything the rung
/// below it matched.
///
/// This is the product surface: it fills in what `--max-distance`, `--cost`, `--dictionary` and
/// `--fold` would otherwise each have to be given separately, and those survive as overrides for
/// somebody who already knows what they cost. Declaration order is the ladder, so `Ord` derives it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default, ValueEnum)]
enum Effort {
    /// The query itself, and nothing else.
    Exact,
    /// One fat-finger edit, drawn from physically adjacent keys.
    #[default]
    Typos,
    /// One edit from the whole layout, plus the recorded misspellings.
    Spelling,
    /// Also matches through diacritics and compatibility forms.
    Accents,
    /// Also matches letters that sound alike: `Gaddafi` reaches `Qaddafi`.
    Sounds,
    /// Also carries a non-Latin query onto Latin, so a Han needle reaches a Han corpus.
    Scripts,
    /// Two edits from the whole layout, on top of every fold above.
    Deep,
}

impl Effort {
    fn max_distance(self) -> usize {
        match self {
            Effort::Exact => 0,
            Effort::Deep => 2,
            _ => 1,
        }
    }

    /// Only `Typos` narrows to adjacent keys; every other rung draws from the whole layout, which
    /// is what makes each rung a superset of the one below it.
    fn alphabet(self) -> Alphabet {
        match self {
            Effort::Typos => Alphabet::Keyboard,
            _ => Alphabet::Script,
        }
    }

    fn dictionary(self) -> Dictionary {
        match self {
            Effort::Exact | Effort::Typos => Dictionary::Ignored,
            _ => Dictionary::Known,
        }
    }

    fn folding(self) -> Folding {
        match self {
            Effort::Exact | Effort::Typos | Effort::Spelling => Folding::Untouched,
            Effort::Accents => Folding::Accents,
            Effort::Sounds => Folding::Sounds,
            Effort::Scripts | Effort::Deep => Folding::Scripts,
        }
    }
}

// endregion: Effort

// region: Folding

/// The writing system the queries are written in.
///
/// One detection serves two decisions that were previously taken separately and could disagree: a
/// Cyrillic query picked a Cyrillic keyboard but was never offered `Cyrillic-Latin`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Script {
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Hebrew,
    Arabic,
    Han,
    Hangul,
    Kana,
}

impl Script {
    /// The script one codepoint belongs to, or `None` for digits, punctuation and spacing.
    ///
    /// Ranges rather than a table: the nine scripts a layout or a transform exists for are worth
    /// naming, and everything else is noise a query carries rather than a script it is in.
    fn of_codepoint(codepoint: char) -> Option<Script> {
        match codepoint as u32 {
            0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F => Some(Script::Latin),
            0x0370..=0x03FF | 0x1F00..=0x1FFF => Some(Script::Greek),
            0x0400..=0x04FF | 0x0500..=0x052F => Some(Script::Cyrillic),
            0x0530..=0x058F => Some(Script::Armenian),
            0x0590..=0x05FF => Some(Script::Hebrew),
            0x0600..=0x06FF | 0x0750..=0x077F => Some(Script::Arabic),
            0x3040..=0x30FF | 0x31F0..=0x31FF => Some(Script::Kana),
            0x1100..=0x11FF | 0xAC00..=0xD7AF | 0x3130..=0x318F => Some(Script::Hangul),
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Some(Script::Han),
            _ => None,
        }
    }

    /// The script most of the queries' letters are in, in one pass over them.
    ///
    /// Latin is the fallback rather than an error, since a query of digits and punctuation names no
    /// script and the Latin layout is the one that can type it.
    fn of(patterns: &[String]) -> Script {
        let mut tallies = [0usize; 9];
        for pattern in patterns {
            for codepoint in pattern.chars() {
                if let Some(script) = Script::of_codepoint(codepoint) {
                    tallies[script.index()] += 1;
                }
            }
        }
        // Kana and Han mix freely in Japanese, and only one of them needs transliterating: kana
        // already spells the sound, where a Han character does not. So any Han at all decides,
        // however much kana surrounds it, and kana answers only when no Han appears.
        if tallies[Script::Han.index()] > 0 {
            return Script::Han;
        }
        if tallies[Script::Kana.index()] > 0 {
            return Script::Kana;
        }
        Script::ALL
            .iter()
            .copied()
            .max_by_key(|script| tallies[script.index()])
            .filter(|script| tallies[script.index()] > 0)
            .unwrap_or(Script::Latin)
    }

    /// Every script, in the order ties break: earlier wins, and Latin leads so an ASCII query never
    /// drifts onto a lookalike.
    const ALL: [Script; 9] = [
        Script::Latin,
        Script::Cyrillic,
        Script::Greek,
        Script::Armenian,
        Script::Hebrew,
        Script::Arabic,
        Script::Han,
        Script::Hangul,
        Script::Kana,
    ];

    fn index(self) -> usize {
        match self {
            Script::Latin => 0,
            Script::Cyrillic => 1,
            Script::Greek => 2,
            Script::Armenian => 3,
            Script::Hebrew => 4,
            Script::Arabic => 5,
            Script::Han => 6,
            Script::Hangul => 7,
            Script::Kana => 8,
        }
    }

    /// The XKB layout whose keys carry this script, for `Alphabet::Keyboard`.
    ///
    /// Han and Hangul are typed through an input method rather than off a layout, so their edits
    /// are drawn from the Latin keys their romanization is typed on.
    fn layout(self) -> &'static str {
        match self {
            Script::Latin | Script::Han | Script::Hangul => "us",
            Script::Cyrillic => "ru",
            Script::Greek => "gr",
            Script::Armenian => "am",
            Script::Hebrew => "il",
            Script::Arabic => "ara",
            Script::Kana => "jp",
        }
    }

    /// The transform that carries this script to Latin, and `None` where it already is.
    fn transliteration(self) -> Option<&'static str> {
        match self {
            Script::Latin => None,
            Script::Cyrillic => Some("Cyrillic-Latin"),
            Script::Greek => Some("Greek-Latin"),
            Script::Han => Some("Han-Latin"),
            // No table ships for these yet, so naming one would fail the run rather than widen it.
            Script::Armenian | Script::Hebrew | Script::Arabic | Script::Hangul | Script::Kana => {
                None
            }
        }
    }
}

/// What a rung asks a fold to achieve, before the query's script says which transform delivers it.
///
/// Each level is a superset of the one above, which is what keeps the effort ladder monotone.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
enum Folding {
    /// Corpus and query are matched in the bytes the user typed.
    #[default]
    Untouched,
    /// Diacritics and compatibility forms collapse onto their base letters.
    Accents,
    /// Letters that sound alike collapse onto one spelling.
    Sounds,
    /// A non-Latin query is carried onto Latin before either of the above.
    Scripts,
}

impl Folding {
    /// The transforms this level applies, in the order they feed each other.
    ///
    /// Transliteration leads: `Latin-Phonetic` has nothing to act on until the script transform has
    /// produced a Latin syllable for it.
    fn transforms(self, script: Script) -> Vec<&'static str> {
        let mut chain = Vec::new();
        if self == Folding::Scripts {
            chain.extend(script.transliteration());
        }
        if self >= Folding::Accents {
            chain.push("Latin-ASCII");
        }
        if self >= Folding::Sounds {
            chain.push("Latin-Phonetic");
        }
        chain
    }
}

/// The transforms a run folds through: the ones `--fold` named, or the ones the level implies.
///
/// An empty result is the zero-cost case, and it is what every effort below `Folding::Accents`
/// produces without `--fold` naming anything.
fn resolve_folds(named: &[String], folding: Folding, script: Script) -> Result<Vec<Fold>, Failure> {
    let wanted: Vec<String> = match named.is_empty() {
        false => named.to_vec(),
        true => folding
            .transforms(script)
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    let mut folds = Vec::with_capacity(wanted.len());
    for name in wanted {
        let fold = Fold::load(&name);
        if fold.is_empty() {
            return Err(Failure::Unresolved {
                path: name,
                subject: "CLDR transform".to_string(),
                note: "pass --fold with an embedded transform, such as Han-Latin or Latin-ASCII",
            });
        }
        folds.push(fold);
    }
    Ok(folds)
}

// endregion: Folding

// region: Vocabulary

/// Where an edit's replacement characters come from.
///
/// This is the one knob that decides whether a `d <= 2` ball is a few hundred needles or a few
/// hundred thousand: restricting substitutions and insertions to physically adjacent keys cuts the
/// ball roughly fifteen-fold at every radius.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Alphabet {
    /// Every character the layout carries, which is plain edit distance.
    Script,
    /// Only the keys adjacent to the one being replaced, for fat-finger typos.
    Keyboard,
}

/// How a variant was derived from its query, which is what sets its weight.
///
/// A transposition is the likeliest real typo and a deletion the least informative, so they do not
/// share a score even at equal edit distance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Derivation {
    Exact,
    /// A misspelling somebody wrote down, from hunspell `REP` or Wikipedia's list.
    Recorded,
    Transposition,
    Substitution,
    Deletion,
    Insertion,
}

impl Derivation {
    /// The BM25 weight this derivation earns at `distance` edits.
    ///
    /// Every value is strictly positive, which is what makes a positive score mean "matched": a
    /// zero weight would leave a matched line indistinguishable from an untouched one.
    fn weight(self, distance: usize) -> f32 {
        let base = match self {
            Derivation::Exact => 1.0,
            Derivation::Recorded => 0.95,
            Derivation::Transposition => 0.9,
            Derivation::Substitution => 0.8,
            Derivation::Insertion => 0.7,
            Derivation::Deletion => 0.6,
        };
        let decayed = base / (1.0 + distance as f32);
        // `f32::MIN_POSITIVE` rather than zero, so deep balls stay above the match threshold.
        decayed.max(f32::MIN_POSITIVE)
    }
}

/// The pooled dictionary of every query's variants, ready for one automaton.
///
/// Queries share one automaton rather than one each, so the corpus is walked once however many
/// `--pattern` flags are given. Which query a variant came from is recoverable from the reported
/// `needle_index`, and is carried only once an output mode names it.
struct Vocabulary {
    needles: Vec<String>,
    weights: Vec<f32>,
}

impl Vocabulary {
    /// Build the ball of every query up to `max_distance`, drawing edits from `alphabet`.
    ///
    /// Empty variants are dropped rather than indexed: an empty needle matches at every position and
    /// `Substrings::new` rejects the whole dictionary over one.
    fn build(
        patterns: &[String],
        max_distance: usize,
        alphabet: Alphabet,
        keyboard: &Keyboard,
        dictionary: Dictionary,
    ) -> Self {
        let mut best: HashMap<String, f32> = HashMap::new();

        for pattern in patterns {
            // A recorded misspelling is evidence, not a guess, so it outranks a same-distance edit.
            if dictionary == Dictionary::Known {
                for known in misspellings::variants_of(pattern) {
                    Vocabulary::record(&mut best, &known, Derivation::Recorded, 0);
                }
            }
            if pattern.is_empty() {
                continue;
            }
            let mut frontier: HashSet<Vec<char>> = HashSet::new();
            frontier.insert(pattern.chars().collect());
            let mut seen = frontier.clone();
            Vocabulary::record(&mut best, pattern, Derivation::Exact, 0);

            for distance in 1..=max_distance {
                let mut next: HashSet<Vec<char>> = HashSet::new();
                for source in &frontier {
                    for (variant, derivation) in edits_of(source, alphabet, keyboard) {
                        if variant.is_empty() || seen.contains(&variant) {
                            continue;
                        }
                        let text: String = variant.iter().collect();
                        Vocabulary::record(&mut best, &text, derivation, distance);
                        next.insert(variant);
                    }
                }
                seen.extend(next.iter().cloned());
                frontier = next;
                if frontier.is_empty() {
                    break;
                }
            }
        }

        let mut pairs: Vec<(String, f32)> = best.into_iter().collect();
        pairs.sort_by(|left, right| left.0.cmp(&right.0));
        let mut needles = Vec::with_capacity(pairs.len());
        let mut weights = Vec::with_capacity(pairs.len());
        for (text, weight) in pairs {
            needles.push(text);
            weights.push(weight);
        }
        Self { needles, weights }
    }

    /// Keep the highest weight a variant earns, since the same string can be reached by several
    /// routes and the cheapest explanation is the one a reader would give.
    fn record(
        best: &mut HashMap<String, f32>,
        text: &str,
        derivation: Derivation,
        distance: usize,
    ) {
        let weight = derivation.weight(distance);
        best.entry(text.to_string())
            .and_modify(|held| *held = held.max(weight))
            .or_insert(weight);
    }

    fn is_empty(&self) -> bool {
        self.needles.is_empty()
    }
}

/// Every string one edit away from `source`, paired with how it was reached.
///
/// Deletions and transpositions need no alphabet; substitutions and insertions draw from one,
/// which is where `--cost keyboard` narrows the ball.
fn edits_of(
    source: &[char],
    alphabet: Alphabet,
    keyboard: &Keyboard,
) -> Vec<(Vec<char>, Derivation)> {
    let mut produced = Vec::new();

    for index in 0..source.len() {
        let mut variant = source.to_vec();
        variant.remove(index);
        produced.push((variant, Derivation::Deletion));
    }

    for index in 0..source.len().saturating_sub(1) {
        let mut variant = source.to_vec();
        variant.swap(index, index + 1);
        produced.push((variant, Derivation::Transposition));
    }

    let replacements = |character: char| -> Vec<char> {
        match alphabet {
            Alphabet::Keyboard => keyboard.near(character).to_vec(),
            Alphabet::Script => keyboard.alphabet(),
        }
    };

    for index in 0..source.len() {
        for replacement in replacements(source[index]) {
            if replacement == source[index] {
                continue;
            }
            let mut variant = source.to_vec();
            variant[index] = replacement;
            produced.push((variant, Derivation::Substitution));
        }
    }

    for index in 0..=source.len() {
        // An insertion at the end has no character of its own to sit beside, so it borrows the last
        // one's neighbours; without this the tail of a needle admits no insertions at all.
        let anchor = source[index.min(source.len().saturating_sub(1))];
        for inserted in replacements(anchor) {
            let mut variant = source.to_vec();
            variant.insert(index, inserted);
            produced.push((variant, Derivation::Insertion));
        }
    }

    produced
}

// endregion: Vocabulary

// region: Searching

/// What the vocabulary is built from, resolved once from `Args`.
struct SearchConfig {
    max_distance: usize,
    alphabet: Alphabet,
    case_sensitivity: CaseSensitivity,
    utf8: bool,
    dictionary: Dictionary,
    /// Lowest BM25 score a line may earn and still count as matched.
    floor: f32,
    /// Keep only the best `top` lines per input when set, ranked by score.
    top: Option<usize>,
}

impl SearchConfig {
    /// The rung's settings with every expert flag layered over it, so `--max-distance` and its
    /// neighbours mean the same thing whatever `--effort` was, and the rung only fills the gaps.
    fn resolve(args: &Args, effort: Effort) -> Self {
        let case_sensitivity = args.ignore_case.unwrap_or(CaseSensitivity::Uncased);
        Self {
            max_distance: args.max_distance.unwrap_or(effort.max_distance()),
            alphabet: args.cost.map_or(effort.alphabet(), Cost::alphabet),
            dictionary: args.dictionary.unwrap_or(effort.dictionary()),
            case_sensitivity,
            utf8: args.utf8 || case_sensitivity == CaseSensitivity::Uncased,
            floor: args.min_score.unwrap_or(0.0),
            top: args.top_k,
        }
    }
}

/// The pooled dictionary, the automaton compiled from it, and the device that walks it.
///
/// One engine serves the whole run: every `--pattern` contributes its variants to a single
/// dictionary, so the corpus is walked once rather than once per query.
struct Engine {
    device: DeviceScope,
    automaton: Substrings,
    vocabulary: Vocabulary,
    /// Applied to corpus and query alike before anything is matched.
    folder: Option<Folder>,
}

impl Engine {
    /// Expand every query into its ball and compile the pooled result.
    fn build(
        patterns: &[String],
        config: &SearchConfig,
        keyboard: &Keyboard,
        folds: &[Fold],
        device: DeviceScope,
    ) -> Result<Self, Failure> {
        // The query is folded first, so the ball is built in the domain the corpus will be matched
        // in rather than in the one the user typed.
        // An empty chain is no chain: no automaton is compiled, no rewrite runs, and no offset
        // map can exist to be consulted later.
        let folder = match folds.is_empty() {
            true => None,
            false => Some(Folder::new(&device, folds, config.case_sensitivity)?),
        };
        let patterns: Vec<String> = match &folder {
            Some(folder) => {
                let raw: Vec<&[u8]> = patterns.iter().map(|one| one.as_bytes()).collect();
                folder
                    .apply(&device, &raw)?
                    .into_iter()
                    .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                    .collect()
            }
            None => patterns.to_vec(),
        };
        let vocabulary = Vocabulary::build(
            &patterns,
            config.max_distance,
            config.alphabet,
            keyboard,
            config.dictionary,
        );
        if vocabulary.is_empty() {
            return Err(Failure::Unresolved {
                path: "-".to_string(),
                subject: patterns.join(", "),
                note: "every query was empty; pass a pattern with at least one character",
            });
        }
        let automaton = Substrings::new(&device, &vocabulary.needles, config.case_sensitivity)
            .map_err(|error| engine_failure("multi-pattern search", error))?;
        Ok(Self {
            device,
            automaton,
            vocabulary,
            folder,
        })
    }

    /// Score every line in one BM25 walk.
    ///
    /// Weights are strictly positive, so a positive score is exactly "at least one variant of at
    /// least one query occurs in this line" - no separate counting walk is needed to decide it.
    fn score(&self, lines: &[&[u8]]) -> Result<Vec<f32>, Failure> {
        let mut scores = vec![0.0f32; lines.len()];
        if lines.is_empty() {
            return Ok(scores);
        }

        // Under a fold the corpus is rewritten once and matched in the folded domain; the caller
        // still holds the original lines, which is what gets printed.
        let folded = match &self.folder {
            Some(folder) => Some(folder.apply(&self.device, lines)?),
            None => None,
        };
        let borrowed: Vec<&[u8]>;
        let lines: &[&[u8]] = match &folded {
            Some(owned) => {
                borrowed = owned.iter().map(Vec::as_slice).collect();
                &borrowed
            }
            None => lines,
        };

        // A GPU scope reads its inputs from unified memory and refuses borrowed slices, so the
        // corpus is copied only where it has to be.
        let haystacks = if self.device.is_gpu() {
            AnyBytesTape::from_sequences(lines)
                .map_err(|error| engine_failure("corpus staging", error))?
        } else {
            AnyBytesTape::from_slices(lines)
        };

        // Length normalization divides by the corpus mean, and BM25 refuses a mean that is not
        // positive rather than quietly ignoring it, so an all-empty corpus scores unnormalized.
        let total: usize = lines.iter().map(|line| line.len()).sum();
        let mean = total as f32 / lines.len() as f32;
        let parameters = if mean > 0.0 {
            Bm25Params::normalized(mean)
        } else {
            Bm25Params::unnormalized()
        };

        self.automaton
            .score_bm25_into(
                &self.device,
                &haystacks,
                &self.vocabulary.weights,
                None,
                parameters,
                &mut scores,
            )
            .map_err(|error| engine_failure("BM25 scoring", error))?;
        Ok(scores)
    }

    /// Count and find in one place, over whatever haystacks the caller hands in.
    ///
    /// Deliberately a second, much smaller walk than scoring: sizing a match buffer needs a prior
    /// count, and paying for both over the whole corpus would triple the work to answer a question
    /// only a few hundred lines ever ask.
    fn find(&self, haystacks_bytes: &[&[u8]]) -> Result<Vec<SubstringsMatch>, Failure> {
        let survivors = haystacks_bytes;
        if survivors.is_empty() {
            return Ok(Vec::new());
        }
        let haystacks = if self.device.is_gpu() {
            AnyBytesTape::from_sequences(survivors)
                .map_err(|error| engine_failure("corpus staging", error))?
        } else {
            AnyBytesTape::from_slices(survivors)
        };

        let mut counts = vec![0usize; survivors.len()];
        let total = self
            .automaton
            .count_into(
                &self.device,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &mut counts,
            )
            .map_err(|error| engine_failure("match counting", error))?;

        let mut matches = vec![SubstringsMatch::default(); total];
        let found = self
            .automaton
            .find_into(
                &self.device,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &mut matches,
            )
            .map_err(|error| engine_failure("match location", error))?;
        matches.truncate(found);
        Ok(matches)
    }

    /// Locate the variants inside lines that already scored, always in the caller's own bytes.
    ///
    /// The domain the automaton walked is this method's business alone: an output mode receives
    /// spans it can slice directly and never learns whether a fold ran.
    fn locate(&self, survivors: &[&[u8]]) -> Result<Vec<Located>, Failure> {
        match &self.folder {
            // Matched domain and printed domain agree, so the automaton's own offsets already
            // answer and no map exists to consult.
            None => Ok(self
                .find(survivors)?
                .into_iter()
                .map(Located::verbatim)
                .collect()),
            // The automaton was compiled from folded needles, so it has to be shown folded bytes.
            // Handing it the originals is what made `--show matches` under a fold report a subset.
            Some(folder) => {
                let folded = folder.apply(&self.device, survivors)?;
                let borrowed: Vec<&[u8]> = folded.iter().map(Vec::as_slice).collect();
                let rewrites = folder.rewrites(&self.device, survivors)?;
                Ok(self
                    .find(&borrowed)?
                    .into_iter()
                    .map(|found| rewrites.located(found))
                    .collect())
            }
        }
    }
}

/// One located span, in the original bytes of the line it was found in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Located {
    line_index: usize,
    byte_offset: usize,
    byte_length: usize,
}

impl Located {
    /// A match found in the same bytes that will be printed, so its offsets already answer.
    fn verbatim(found: SubstringsMatch) -> Self {
        Self {
            line_index: found.haystack_index,
            byte_offset: found.byte_offset,
            byte_length: found.byte_length,
        }
    }
}

/// Whether the embedded misspelling table contributes variants.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Dictionary {
    /// The embedded misspellings contribute nothing.
    Ignored,
    /// A recorded misspelling of the query becomes a variant of it.
    Known,
}

/// One transform compiled into a rewrite.
struct Stage {
    automaton: Substrings,
    targets: Vec<String>,
}

/// The transforms a run folds through, in order, applied to corpus and query alike.
///
/// A sequence rather than one merged dictionary, because the transforms feed each other:
/// `Latin-Phonetic` has nothing to act on until `Han-Latin` has produced a Latin syllable for it,
/// and a merged automaton would let only one of them fire at each position.
struct Folder {
    stages: Vec<Stage>,
}

impl Folder {
    /// Transform rules are written in lower case, so a cased fold would leave `Coronavirus` as
    /// `Coronafirus` while `coronavirus` became `koronafirus` - the same word landing in two
    /// domains. The fold therefore matches uncased whenever the search does, and the replacement is
    /// inserted verbatim, which puts both spellings in one place.
    fn new(
        device: &DeviceScope,
        folds: &[Fold],
        case_sensitivity: CaseSensitivity,
    ) -> Result<Self, Failure> {
        let mut stages = Vec::with_capacity(folds.len());
        for fold in folds {
            stages.push(Stage {
                automaton: Substrings::new(device, &fold.sources, case_sensitivity)
                    .map_err(|error| engine_failure("fold", error))?,
                targets: fold.targets.clone(),
            });
        }
        Ok(Self { stages })
    }

    /// Rewrite every haystack through every stage in turn, returning owned bytes since the product
    /// of a rewrite is a new tape.
    fn apply(&self, device: &DeviceScope, lines: &[&[u8]]) -> Result<Vec<Vec<u8>>, Failure> {
        let mut carried: Vec<Vec<u8>> = lines.iter().map(|line| line.to_vec()).collect();
        for stage in &self.stages {
            let borrowed: Vec<&[u8]> = carried.iter().map(Vec::as_slice).collect();
            carried = stage.rewrite(device, &borrowed)?;
        }
        Ok(carried)
    }

    /// Where every stage rewrote these lines, as the map from the folded bytes back to the caller's.
    ///
    /// Built only when an output mode asks for spans, and only over the lines that already scored -
    /// a corpus-wide map of `Han-Latin` would run several times the size of the corpus itself.
    fn rewrites(&self, device: &DeviceScope, lines: &[&[u8]]) -> Result<Rewrites, Failure> {
        let mut stages = Vec::with_capacity(self.stages.len());
        let mut carried: Vec<Vec<u8>> = lines.iter().map(|line| line.to_vec()).collect();
        for stage in &self.stages {
            let borrowed: Vec<&[u8]> = carried.iter().map(Vec::as_slice).collect();
            stages.push(stage.sites(device, &borrowed)?);
            carried = stage.rewrite(device, &borrowed)?;
        }
        Ok(Rewrites { stages })
    }
}

impl Stage {
    /// This stage's rewrite of every haystack.
    fn rewrite(&self, device: &DeviceScope, lines: &[&[u8]]) -> Result<Vec<Vec<u8>>, Failure> {
        if lines.is_empty() {
            return Ok(Vec::new());
        }
        let haystacks = AnyBytesTape::from_sequences(lines)
            .map_err(|error| engine_failure("fold staging", error))?;
        let input_bytes: usize = lines.iter().map(|line| line.len()).sum();
        let bound = self
            .automaton
            .replace_bound(&self.targets, input_bytes)
            .map_err(|error| engine_failure("fold sizing", error))?;

        let mut data = vec![0u8; bound];
        let mut offsets = vec![0usize; lines.len() + 1];
        self.automaton
            .replace_into(
                device,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &self.targets,
                &mut data,
                &mut offsets,
            )
            .map_err(|error| engine_failure("fold", error))?;

        Ok((0..lines.len())
            .map(|index| data[offsets[index]..offsets[index + 1]].to_vec())
            .collect())
    }

    /// Where this stage fires, in the coordinates of the bytes handed to it.
    ///
    /// The same `LeftmostLongest` cover the rewrite uses, so the sites found here are exactly the
    /// substitutions that happened.
    fn sites(&self, device: &DeviceScope, lines: &[&[u8]]) -> Result<StageMap, Failure> {
        let mut sites = Vec::new();
        let mut starts = vec![0usize; lines.len() + 1];
        if lines.is_empty() {
            return Ok(StageMap { sites, starts });
        }

        let haystacks = AnyBytesTape::from_sequences(lines)
            .map_err(|error| engine_failure("fold staging", error))?;
        let mut counts = vec![0usize; lines.len()];
        let total = self
            .automaton
            .count_into(
                device,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &mut counts,
            )
            .map_err(|error| engine_failure("fold counting", error))?;
        let mut found = vec![SubstringsMatch::default(); total];
        let written = self
            .automaton
            .find_into(
                device,
                &haystacks,
                OverlapPolicy::LeftmostLongest,
                &mut found,
            )
            .map_err(|error| engine_failure("fold locating", error))?;
        found.truncate(written);
        // The walk emits in order of match ends and interleaves haystacks, so the ascending order
        // the drift arithmetic needs has to be asked for.
        found.sort_unstable_by_key(|one| (one.haystack_index, one.byte_offset));

        let mut line = 0usize;
        let mut drift: i64 = 0;
        for one in found {
            while line < one.haystack_index {
                starts[line + 1] = sites.len();
                line += 1;
                drift = 0;
            }
            // Under case folding a needle's own length is not the match's - a one-byte needle
            // matches a three-byte Kelvin sign - so the consumed length comes from the match and the
            // produced length from the replacement.
            let produced = self.targets[one.needle_index].len();
            sites.push(Site {
                folded_offset: (one.byte_offset as i64 + drift) as usize,
                folded_length: produced,
                original_offset: one.byte_offset,
                original_length: one.byte_length,
            });
            drift += produced as i64 - one.byte_length as i64;
        }
        for tail in line..lines.len() {
            starts[tail + 1] = sites.len();
        }
        Ok(StageMap { sites, starts })
    }
}

/// One place a stage rewrote, as the two spans that define the map there.
#[derive(Clone, Copy)]
struct Site {
    folded_offset: usize,
    folded_length: usize,
    original_offset: usize,
    original_length: usize,
}

/// Which end of a span an offset is, since a boundary landing inside a rewrite widens outward.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edge {
    Start,
    End,
}

/// One stage's sites, grouped by line.
struct StageMap {
    sites: Vec<Site>,
    /// `sites[starts[line]..starts[line + 1]]` are one line's, in ascending offset order.
    starts: Vec<usize>,
}

impl StageMap {
    /// Where an offset in this stage's output sits in its input.
    ///
    /// Between rewrites the two domains advance in lockstep and the drift alone answers. Strictly
    /// inside one, the offset names a byte of a replacement that no input byte corresponds to, so
    /// the region is claimed whole: a start falls back to its first input byte and an end runs past
    /// its last. Snapping outward is what keeps `end >= start` and keeps a reported span from
    /// naming half a syllable that was never in the file.
    fn backward(&self, line: usize, folded: usize, edge: Edge) -> usize {
        let sites = &self.sites[self.starts[line]..self.starts[line + 1]];
        let above = sites.partition_point(|site| site.folded_offset <= folded);
        let Some(index) = above.checked_sub(1) else {
            return folded;
        };
        let site = sites[index];
        let folded_end = site.folded_offset + site.folded_length;
        let original_end = site.original_offset + site.original_length;
        if folded < folded_end {
            return match edge {
                Edge::Start => site.original_offset,
                Edge::End => original_end,
            };
        }
        if folded == folded_end && site.folded_length == 0 {
            // A deleting rule leaves a zero-width mark, so the offset is both on the site and after
            // it, and only the edge says which of its input boundaries it names.
            return match edge {
                Edge::Start => site.original_offset,
                Edge::End => original_end,
            };
        }
        original_end + (folded - folded_end)
    }
}

/// The whole chain's map, from the bytes the automaton walked back to the caller's own.
struct Rewrites {
    stages: Vec<StageMap>,
}

impl Rewrites {
    /// Carry one offset back through every stage, last applied first.
    fn backward(&self, line: usize, folded: usize, edge: Edge) -> usize {
        self.stages
            .iter()
            .rev()
            .fold(folded, |offset, stage| stage.backward(line, offset, edge))
    }

    /// Carry a whole match back, so the caller receives a span it can slice directly.
    fn located(&self, found: SubstringsMatch) -> Located {
        let line = found.haystack_index;
        let start = self.backward(line, found.byte_offset, Edge::Start);
        let end = self.backward(line, found.byte_offset + found.byte_length, Edge::End);
        Located {
            line_index: line,
            byte_offset: start,
            byte_length: end.saturating_sub(start),
        }
    }
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig {
    line_numbers: bool,
    scores: bool,
    show: Show,
    format: Format,
    /// Prefix each record with its file name, as grep does for multiple inputs.
    prefix: bool,
    summary: bool,
    terminator: Terminator,
}

/// One input's lines and the score each earned.
struct Searched<'a> {
    lines: Vec<&'a [u8]>,
    scores: Vec<f32>,
}

impl<'a> Searched<'a> {
    /// A line matched when it clears the floor, which defaults to any positive score - and a
    /// positive score means "matched" only because every weight is strictly positive.
    fn matched(&self, index: usize, floor: f32) -> bool {
        self.scores[index] > 0.0 && self.scores[index] >= floor
    }
}

/// Cut one input into lines and score all of them in a single walk.
fn search_lines<'a>(
    data: &'a [u8],
    engine: &Engine,
    config: &SearchConfig,
) -> Result<Searched<'a>, Failure> {
    let lines: Vec<&[u8]> = LineIter::new(data, Newlines::from_utf8(config.utf8)).collect();
    let scores = engine.score(&lines)?;
    Ok(Searched { lines, scores })
}

// endregion: Searching

// region: CLI

/// One column a record can carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Field {
    /// The 1-based line number
    LineNumbers,
    /// The BM25 score the line earned
    Scores,
}

/// Which record kind to emit
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, ValueEnum)]
enum Show {
    /// Every matching line.
    #[default]
    Lines,
    /// Only the matched parts of selected lines.
    Matches,
    /// One count per input.
    Count,
    /// The path of every input that matched.
    Files,
    /// The path of every input that did not match.
    FilesWithout,
}

/// How records are rendered
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, ValueEnum)]
enum Format {
    /// The matching line, with any requested fields ahead of it.
    #[default]
    Text,
    /// JSON Lines, one object per match.
    Json,
}

/// Which characters an edit may substitute or insert
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Cost {
    /// Any character of the layout, which is plain edit distance.
    Edit,
    /// Only physically adjacent keys, for fat-finger typos: `xolor` reaches `color`.
    Keyboard,
}

impl Cost {
    fn alphabet(self) -> Alphabet {
        match self {
            Cost::Edit => Alphabet::Script,
            Cost::Keyboard => Alphabet::Keyboard,
        }
    }
}

/// Where the kernels run
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Device {
    /// The GPU where one was built in, the CPU otherwise.
    Auto,
    /// Every core unless `--threads` says otherwise.
    Cpu,
    /// Requires a build with `--features cuda`.
    Gpu,
}

/// Fuzzy (edit-distance / alignment bounded) substring search
#[derive(Parser)]
#[command(name = "sz-fuzzy-find")]
#[command(version, about = "SIMD/GPU-accelerated fuzzy substring search", long_about = None)]
struct Args {
    /// Substring to search for (approximately); omit when using --pattern
    pattern: Option<String>,

    /// Input files (use '-' or omit for stdin)
    inputs: Vec<String>,

    /// Additional query; a line matches if ANY query matches (repeatable)
    #[arg(id = "pattern_flag", long = "pattern", value_name = "PATTERN")]
    extra: Vec<String>,

    /// How hard to try; higher rungs match more and cost more
    #[arg(long, value_enum)]
    effort: Option<Effort>,

    /// Maximum edit distance, in code points under --utf8; overrides --effort
    #[arg(long)]
    max_distance: Option<usize>,

    /// Which characters an edit may substitute or insert
    #[arg(long, value_enum)]
    cost: Option<Cost>,

    /// Keyboard layout the edits follow; detected from the pattern's script by default
    #[arg(long, value_name = "LAYOUT")]
    layout: Option<String>,

    /// Fold corpus and query through a CLDR transform first, such as Han-Latin (repeatable)
    #[arg(long, value_name = "TRANSFORM")]
    fold: Vec<String>,

    /// Also match the recorded misspellings of the query; overrides --effort
    #[arg(long, value_enum, num_args = 0..=1, default_missing_value = "known", value_name = "USE")]
    dictionary: Option<Dictionary>,

    /// Keep only the N best-scoring lines per input, ranked by BM25
    #[arg(long, value_name = "N")]
    top_k: Option<usize>,

    /// Lowest BM25 score a line may earn and still be reported
    #[arg(long, value_name = "SCORE")]
    min_score: Option<f32>,

    /// Where the kernels run
    #[arg(long, value_enum)]
    device: Option<Device>,

    /// CPU thread count, where 0 is every core [default: 0]
    #[arg(long)]
    threads: Option<usize>,

    /// GPU device index [default: 0]
    #[arg(long, requires = "device")]
    gpu_id: Option<usize>,

    /// Case-insensitive search with full Unicode folding; implies --utf8 [default: 1]
    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "1",
        value_name = "ON",
        value_parser = parse_case_sensitivity
    )]
    ignore_case: Option<CaseSensitivity>,

    /// Which columns each record carries, comma-separated; none by default
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        help_heading = "Output Formats"
    )]
    fields: Vec<Field>,

    /// Treat the input as UTF-8 text
    #[arg(long)]
    utf8: bool,

    /// Which record kind to emit
    #[arg(long, value_enum)]
    show: Option<Show>,

    /// Print one line about the whole run on stderr
    #[arg(long)]
    summary: bool,

    /// How records are rendered
    #[arg(long, value_enum, help_heading = "Output Formats")]
    format: Option<Format>,

    /// NUL-terminate each output record instead of newline, for `xargs -0`
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any match was found, 1 otherwise
    #[arg(long, conflicts_with_all = ["show", "format", "null", "fields", "summary"], help_heading = "Output Formats")]
    quiet: bool,
}

/// Read `--ignore-case`'s optional value. Named states rather than a bare `bool`, and the flag
/// keeps the name it has in `sz-find`, `sz-replace`, `sz-dedup` and `sz-sort` while defaulting the
/// other way here: a fuzzy search that respected case would refuse the first thing anyone tries.
fn parse_case_sensitivity(text: &str) -> Result<CaseSensitivity, String> {
    match text {
        "1" | "true" | "yes" | "on" => Ok(CaseSensitivity::Uncased),
        "0" | "false" | "no" | "off" => Ok(CaseSensitivity::Cased),
        other => Err(format!("`{other}` is not 0 or 1")),
    }
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Every constraint clap cannot express, because `conflicts_with` fires on an
/// argument's presence and never on its value.
fn validate(args: &Args) -> Result<(), clap::Error> {
    let device = args.device.unwrap_or(Device::Auto);
    if args.gpu_id.is_some() && device != Device::Gpu {
        return Err(reject("--gpu-id needs --device gpu"));
    }
    if args.threads.is_some() && device == Device::Gpu {
        return Err(reject("--threads is a CPU setting, and --device is gpu"));
    }
    if args.null && args.format == Some(Format::Json) {
        return Err(reject("--format json cannot be combined with --null"));
    }
    if args.fields.contains(&Field::LineNumbers) && args.show == Some(Show::Count) {
        return Err(reject(
            "--line-numbers has no record to number under --show count",
        ));
    }
    Ok(())
}

fn build_device(
    device: Device,
    threads: Option<usize>,
    gpu_id: Option<usize>,
) -> Result<DeviceScope, String> {
    // `DeviceScope::default()` yields a single core; 0 means every core.
    let cores = threads.unwrap_or(0);
    let result = match device {
        Device::Gpu => DeviceScope::gpu_device(gpu_id.unwrap_or(0)),
        Device::Cpu | Device::Auto => DeviceScope::cpu_cores(cores),
    };
    // The library's error is a struct whose `Debug` leaks its variant names into a message
    // a person reads, so only the sentence inside it is passed on.
    result.map_err(|error| match device {
        Device::Gpu => format!("--device gpu is unavailable: {}", error),
        Device::Cpu | Device::Auto => format!("--threads {} is unavailable: {}", cores, error),
    })
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-fuzzy-find", run(&args, &mut output, &mut io::stderr()))
}

/// An engine fails to build on allocation or a device fault, never on a bad argument.
fn engine_failure(engine: &str, error: impl std::fmt::Debug) -> Failure {
    Failure::Io {
        path: engine.to_string(),
        source: io::Error::other(format!("init failed: {:?}", error)),
    }
}

/// The run's output and the notes about it are two different streams, and the caller passes
/// both: `output` carries what the run produced, `notes` carries what it has to say about
/// the run. Only the second may be prose, and only the second goes to stderr, so redirecting
/// stdout gives a file of data rather than data with a sentence appended.
fn run(args: &Args, output: &mut dyn Write, notes: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let (patterns, inputs) =
        resolve_positionals(args.pattern.as_deref(), &args.extra, &args.inputs)
            .map_err(|message| reject(&message))?;

    // The layout decides which characters count as adjacent, so it is resolved before the ball is
    // built and named explicitly when detection would have to guess.
    let script = Script::of(&patterns);
    let layout = args
        .layout
        .clone()
        .unwrap_or_else(|| script.layout().to_string());
    let keyboard = Keyboard::load(&layout);
    let effort = args.effort.unwrap_or_default();
    if keyboard.is_empty() {
        return Err(Failure::Unresolved {
            path: layout,
            subject: "keyboard layout".to_string(),
            note: "pass --layout with one of the embedded names, such as us, de, fr or ru",
        });
    }

    let device = build_device(
        args.device.unwrap_or(Device::Auto),
        args.threads,
        args.gpu_id,
    )
    .map_err(|message| reject(&message))?;

    let folds = resolve_folds(&args.fold, effort.folding(), script)?;

    let config = SearchConfig::resolve(args, effort);
    let engine = Engine::build(&patterns, &config, &keyboard, &folds, device)?;

    let output_config = OutputConfig {
        line_numbers: args.fields.contains(&Field::LineNumbers),
        scores: args.fields.contains(&Field::Scores) || args.top_k.is_some(),
        show: args.show.unwrap_or_default(),
        format: args.format.unwrap_or_default(),
        prefix: inputs.len() > 1,
        summary: args.summary,
        terminator: Terminator::from_null(args.null),
    };

    // A quiet run still searches, so the match count that answers it stays honest.
    let mut discard = io::sink();
    let writer: &mut dyn Write = if args.quiet {
        &mut discard
    } else {
        &mut *output
    };
    let opened = inputs
        .iter()
        .map(|path| (path.as_str(), get_input(Some(path))));
    let outcome = search_inputs(writer, notes, opened, &engine, &config, &output_config)?;

    output.flush().at("-")?;
    if outcome.readable == 0 {
        Ok(Status::Error)
    } else {
        Ok(Status::from_found(outcome.total > 0))
    }
}

/// What the whole run found, for the exit code and `--summary`.
#[derive(Default, PartialEq, Eq, Debug)]
struct Outcome {
    total: usize,
    readable: usize,
}

/// The queries and the input paths, once the positional has been assigned to whichever
/// of the two it belongs to. Without `--pattern` the positional is the needle; with it,
/// every positional is a path.
fn resolve_positionals(
    pattern: Option<&str>,
    extra: &[String],
    inputs: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut inputs = inputs.to_vec();
    let mut patterns: Vec<String> = Vec::new();
    match (pattern, extra.is_empty()) {
        (Some(pattern), true) => patterns.push(pattern.to_string()),
        (Some(path), false) => inputs.insert(0, path.to_string()),
        (None, true) => return Err("no query given; pass a pattern or --pattern".into()),
        (None, false) => {}
    }
    patterns.extend(extra.iter().cloned());
    if inputs.is_empty() {
        inputs.push("-".to_string());
    }
    Ok((patterns, inputs))
}

/// Search every opened input, warning about the ones that could not be opened and continuing.
fn search_inputs<'a>(
    output: &mut dyn Write,
    notes: &mut dyn Write,
    inputs: impl IntoIterator<Item = (&'a str, io::Result<InputSource>)>,
    engine: &Engine,
    config: &SearchConfig,
    out_cfg: &OutputConfig,
) -> Result<Outcome, Failure> {
    let mut outcome = Outcome::default();
    let mut seen = 0;
    for (path, input) in inputs {
        seen += 1;
        let input = match input {
            Ok(input) => input,
            Err(error) => {
                eprintln!("sz-fuzzy-find: {}: {}", path, error);
                continue;
            }
        };
        outcome.readable += 1;
        let found = search_lines(input.as_bytes(), engine, config)?;

        // The scored pass already said which lines matched, so the survivors are gathered once and
        // every output mode reads from them rather than re-testing.
        let mut survivors: Vec<(usize, &[u8], f32)> = found
            .lines
            .iter()
            .enumerate()
            .filter(|(index, _)| found.matched(*index, config.floor))
            .map(|(index, line)| (index, *line, found.scores[index]))
            .collect();

        // Ranking is the only place order stops being the file's own, so it is applied once here
        // and every mode below reads the same list.
        if let Some(top) = config.top {
            survivors.sort_by(|left, right| {
                right
                    .2
                    .partial_cmp(&left.2)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(left.0.cmp(&right.0))
            });
            survivors.truncate(top);
        }
        let count = survivors.len();

        match out_cfg.show {
            Show::Count => write_count(output, out_cfg, path, count).at(path)?,
            Show::Files => {
                if count > 0 {
                    write_path(output, out_cfg, path).at(path)?;
                }
            }
            Show::FilesWithout => {
                if count == 0 {
                    write_path(output, out_cfg, path).at(path)?;
                }
            }
            Show::Lines => {
                for (index, line, score) in &survivors {
                    write_match(output, out_cfg, path, index + 1, line, *score).at(path)?;
                }
            }
            // Spans cost a second walk, so they are located only here, and only over the handful of
            // lines that already scored rather than over the whole corpus.
            Show::Matches => {
                let lines: Vec<&[u8]> = survivors.iter().map(|(_, line, _)| *line).collect();
                for located in engine.locate(&lines)? {
                    let (index, line, score) = survivors[located.line_index];
                    let span =
                        &line[located.byte_offset..located.byte_offset + located.byte_length];
                    write_match(output, out_cfg, path, index + 1, span, score).at(path)?;
                }
            }
        }
        outcome.total += count;
    }

    if out_cfg.summary {
        write_summary(output, notes, out_cfg, &outcome, seen).at("-")?;
    }
    Ok(outcome)
}

/// The one summary closing the run: a record under `--json`, where it belongs to the stream
/// it closes, and otherwise a sentence on stderr, where it cannot be mistaken for a match.
fn write_summary(
    output: &mut dyn Write,
    notes: &mut dyn Write,
    cfg: &OutputConfig,
    outcome: &Outcome,
    inputs: usize,
) -> io::Result<()> {
    if cfg.format == Format::Json {
        return writeln!(
            output,
            r#"{{"type":"summary","data":{{"matched_lines":{},"readable_inputs":{},"total_inputs":{}}}}}"#,
            outcome.total, outcome.readable, inputs
        );
    }
    writeln!(
        notes,
        "matched {} lines in {} of {} inputs",
        outcome.total, outcome.readable, inputs
    )
}

/// One count record per input, in whichever format the run selected.
fn write_count(
    output: &mut dyn Write,
    cfg: &OutputConfig,
    path: &str,
    count: usize,
) -> io::Result<()> {
    if cfg.format == Format::Json {
        output.write_all(br#"{"type":"count","data":{"path":"#)?;
        json_text_field_to(output, path.as_bytes())?;
        write!(output, r#","count":{}}}}}"#, count)?;
        return output.write_all(b"\n");
    }
    if cfg.prefix {
        write!(output, "{}:", path)?;
    }
    write!(output, "{}", count)?;
    output.write_all(&[cfg.terminator.as_byte()])
}

/// One path record, for the two modes that report inputs rather than lines.
fn write_path(output: &mut dyn Write, cfg: &OutputConfig, path: &str) -> io::Result<()> {
    if cfg.format == Format::Json {
        output.write_all(br#"{"type":"path","data":{"path":"#)?;
        json_text_field_to(output, path.as_bytes())?;
        output.write_all(b"}}")?;
        return output.write_all(b"\n");
    }
    output.write_all(path.as_bytes())?;
    output.write_all(&[cfg.terminator.as_byte()])
}

fn write_match(
    output: &mut dyn Write,
    cfg: &OutputConfig,
    path: &str,
    line_no: usize,
    line: &[u8],
    score: f32,
) -> io::Result<()> {
    if cfg.format == Format::Json {
        // Ripgrep's schema, minus `submatches`: fuzzy matching has no exact span.
        output.write_all(br#"{"type":"match","data":{"path":"#)?;
        json_text_field_to(output, path.as_bytes())?;
        output.write_all(br#","lines":"#)?;
        json_text_field_to(output, line)?;
        write!(output, r#","line_number":{}"#, line_no)?;
        if cfg.scores {
            write!(output, r#","score":{:.4}"#, score)?;
        }
        output.write_all(br#","submatches":[]}}"#)?;
        return output.write_all(b"\n");
    }
    if cfg.prefix {
        write!(output, "{}:", path)?;
    }
    if cfg.line_numbers {
        write!(output, "{}:", line_no)?;
    }
    if cfg.scores {
        write!(output, "{:.4}:", score)?;
    }
    output.write_all(line)?;
    output.write_all(&[cfg.terminator.as_byte()])
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn declares_no_short_flags() {
        assert!(Args::command()
            .get_arguments()
            .all(|a| a.get_short().is_none() || matches!(a.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|a| a.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "pattern",
                "effort",
                "max-distance",
                "cost",
                "layout",
                "fold",
                "dictionary",
                "top-k",
                "min-score",
                "device",
                "threads",
                "gpu-id",
                "ignore-case",
                "fields",
                "utf8",
                "show",
                "summary",
                "format",
                "null",
                "quiet",
                "help",
                "version"
            ]
        );
    }

    #[test]
    fn reads_one_file_without_waiting_on_stdin() {
        // The positional is a path once `--pattern` carries the needle, so the input
        // list must not fall back to stdin beside it.
        let positionals = |argv: &[&str]| {
            let args = Args::try_parse_from(argv).unwrap();
            resolve_positionals(args.pattern.as_deref(), &args.extra, &args.inputs).unwrap()
        };
        let (patterns, inputs) = positionals(&["sz-fuzzy-find", "--pattern", "abc", "file.txt"]);
        assert_eq!(patterns, ["abc"]);
        assert_eq!(inputs, ["file.txt"]);
        // With no positional at all, stdin is still the input.
        assert_eq!(positionals(&["sz-fuzzy-find", "--pattern", "abc"]).1, ["-"]);
    }

    #[test]
    fn rejects_settings_that_used_to_be_ignored() {
        for argv in [
            ["sz-fuzzy-find", "--device", "banana", "a"].as_slice(),
            ["sz-fuzzy-find", "--device", "GPU", "a"].as_slice(),
            ["sz-fuzzy-find", "--gpu-id", "1", "a"].as_slice(),
            ["sz-fuzzy-find", "--cost", "phonetic", "a"].as_slice(),
        ] {
            assert!(
                Args::try_parse_from(argv).is_err(),
                "{:?} must be a usage error",
                argv
            );
        }
        // A value-conditional constraint clap cannot express.
        let args = Args::try_parse_from(["sz-fuzzy-find", "--device", "cpu", "--gpu-id", "1", "a"])
            .unwrap();
        assert!(validate(&args).is_err());
    }

    #[test]
    fn accepts_threads_under_the_default_device() {
        // `--threads` used to demand `--device`, which the default already resolves to CPU.
        let args = Args::try_parse_from(["sz-fuzzy-find", "--threads", "2", "a"]).unwrap();
        assert!(validate(&args).is_ok());
        let gpu = Args::try_parse_from(["sz-fuzzy-find", "--device", "gpu", "--threads", "2", "a"])
            .unwrap();
        assert!(validate(&gpu).is_err());
    }

    #[test]
    fn names_its_placeholders_after_the_flags() {
        let mut command = Args::command();
        command.build();
        let placeholder = |id: &str| {
            command
                .get_arguments()
                .find(|a| a.get_id() == id)
                .and_then(|a| a.get_value_names())
                .map(|names| names[0].to_string())
        };
        assert_eq!(placeholder("pattern_flag").as_deref(), Some("PATTERN"));
        assert_eq!(placeholder("layout").as_deref(), Some("LAYOUT"));
    }

    #[test]
    fn closes_a_json_stream_with_its_summary() {
        // `--summary` used to append a prose line after the JSON records.
        let outcome = Outcome {
            total: 9,
            readable: 1,
        };
        let mut cfg = OutputConfig {
            line_numbers: false,
            scores: false,
            show: Show::Lines,
            format: Format::Json,
            prefix: false,
            summary: true,
            terminator: Terminator::Newline,
        };
        let (mut written, mut notes) = (Vec::new(), Vec::new());
        write_summary(&mut written, &mut notes, &cfg, &outcome, 1).unwrap();
        let record = String::from_utf8(written).unwrap();
        assert!(
            record.starts_with(r#"{"type":"summary","data":{"#),
            "{}",
            record
        );
        assert!(record.contains(r#""matched_lines":9"#), "{}", record);
        assert!(notes.is_empty(), "a record belongs to the stream it closes");

        // In text it is prose about the run, so it leaves the record stream alone.
        cfg.format = Format::Text;
        let (mut written, mut notes) = (Vec::new(), Vec::new());
        write_summary(&mut written, &mut notes, &cfg, &outcome, 1).unwrap();
        assert!(written.is_empty(), "prose is not a match");
        assert!(String::from_utf8(notes)
            .unwrap()
            .starts_with("matched 9 lines"));
    }

    // region: Vocabulary

    fn us() -> Keyboard {
        Keyboard::load("us")
    }

    fn ball(pattern: &str, max_distance: usize, alphabet: Alphabet) -> Vocabulary {
        Vocabulary::build(
            &[pattern.to_string()],
            max_distance,
            alphabet,
            &us(),
            Dictionary::Ignored,
        )
    }

    #[test]
    fn embeds_a_keyboard_for_every_script_it_claims() {
        for layout in ["us", "de", "fr", "ru"] {
            assert!(
                !Keyboard::load(layout).is_empty(),
                "{layout} must be in the embedded table"
            );
        }
        assert!(Keyboard::load("no-such-layout").is_empty());
    }

    #[test]
    fn never_admits_an_empty_needle() {
        // `Substrings::new` rejects an entire dictionary over one empty needle, and deleting the
        // only character of a one-character query produces exactly that.
        for pattern in ["a", "ab", "color"] {
            let vocabulary = ball(pattern, 2, Alphabet::Keyboard);
            assert!(
                vocabulary.needles.iter().all(|needle| !needle.is_empty()),
                "{pattern} produced an empty needle"
            );
        }
    }

    #[test]
    fn weights_every_variant_strictly_positive() {
        // A positive score is the match test, so a zero weight would make a matched line
        // indistinguishable from an untouched one.
        let vocabulary = ball("color", 2, Alphabet::Keyboard);
        assert!(vocabulary.weights.iter().all(|weight| *weight > 0.0));
        assert_eq!(vocabulary.needles.len(), vocabulary.weights.len());
    }

    #[test]
    fn scores_the_query_itself_above_its_variants() {
        let vocabulary = ball("color", 1, Alphabet::Keyboard);
        let exact = vocabulary
            .needles
            .iter()
            .position(|needle| needle == "color")
            .expect("the query is its own first variant");
        let best_variant = vocabulary
            .weights
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != exact)
            .map(|(_, weight)| *weight)
            .fold(0.0f32, f32::max);
        assert!(vocabulary.weights[exact] > best_variant);
    }

    #[test]
    fn narrows_the_ball_to_adjacent_keys() {
        // The whole reason `--cost keyboard` exists: it is what keeps a two-edit ball affordable.
        let script = ball("washington", 2, Alphabet::Script).needles.len();
        let keyboard = ball("washington", 2, Alphabet::Keyboard).needles.len();
        assert!(
            keyboard * 5 < script,
            "keyboard ball {keyboard} should be far under the script ball {script}"
        );
    }

    #[test]
    fn pools_every_query_into_one_dictionary() {
        // Several queries share one automaton, and `origins` is what maps a variant back.
        let patterns = vec!["color".to_string(), "flavour".to_string()];
        let pooled =
            Vocabulary::build(&patterns, 1, Alphabet::Keyboard, &us(), Dictionary::Ignored);
        assert!(pooled.needles.iter().any(|needle| needle == "color"));
        assert!(pooled.needles.iter().any(|needle| needle == "flavour"));

        // Pooling is what makes one walk enough, so the dictionary must be the union rather than
        // whichever query happened to be built last.
        let alone = |pattern: &str| {
            Vocabulary::build(
                &[pattern.to_string()],
                1,
                Alphabet::Keyboard,
                &us(),
                Dictionary::Ignored,
            )
            .needles
            .len()
        };
        assert!(pooled.needles.len() > alone("color").max(alone("flavour")));
    }

    #[test]
    fn detects_the_script_the_query_is_written_in() {
        let one = |pattern: &str| Script::of(&[pattern.to_string()]);
        assert_eq!(one("washington"), Script::Latin);
        assert_eq!(one("правительство"), Script::Cyrillic);
        assert_eq!(one("Հայաստան"), Script::Armenian);
        assert_eq!(one("北京大学"), Script::Han);
        // Kana and Han mix in Japanese, and a query carrying both wants the Han reading folded.
        assert_eq!(one("ひらがな"), Script::Kana);
        assert_eq!(one("日本のひらがな"), Script::Han);
        // A query naming no script at all is typed on the layout that can type it.
        assert_eq!(one("2024"), Script::Latin);

        // Every script must name a layout the embedded table actually carries, or `--cost keyboard`
        // would silently have no neighbours to draw on.
        for script in Script::ALL {
            assert!(
                !Keyboard::load(script.layout()).is_empty(),
                "{script:?} names layout {} which is not embedded",
                script.layout()
            );
        }
    }

    // endregion: Vocabulary

    // region: Searching

    fn cpu() -> DeviceScope {
        DeviceScope::cpu_cores(1).expect("a CPU scope is always available")
    }

    fn config(max_distance: usize) -> SearchConfig {
        SearchConfig {
            max_distance,
            alphabet: Alphabet::Script,
            case_sensitivity: CaseSensitivity::Cased,
            utf8: false,
            dictionary: Dictionary::Ignored,
            floor: 0.0,
            top: None,
        }
    }

    fn matching_lines(patterns: &[&str], data: &[u8], config: &SearchConfig) -> Vec<String> {
        let patterns: Vec<String> = patterns.iter().map(|p| p.to_string()).collect();
        let engine = Engine::build(&patterns, config, &us(), &[], cpu()).expect("engine builds");
        let found = search_lines(data, &engine, config).expect("search runs");
        found
            .lines
            .iter()
            .enumerate()
            .filter(|(index, _)| found.matched(*index, 0.0))
            .map(|(_, line)| String::from_utf8_lossy(line).into_owned())
            .collect()
    }

    #[test]
    fn finds_a_one_edit_typo() {
        // The defect the matrix era could not fix: a real one-edit miss under whole-line scoring.
        let data = b"Washington\nWashigton\nunrelated\n";
        let matched = matching_lines(&["Washington"], data, &config(1));
        assert_eq!(matched, ["Washington", "Washigton"]);
    }

    #[test]
    fn keeps_digits_apart() {
        // All ten digits used to share one scoring class, so `--max-distance 0 2024` returned 1999.
        let data = b"in 2024\nin 1999\n";
        let matched = matching_lines(&["2024"], data, &config(0));
        assert_eq!(matched, ["in 2024"]);
    }

    #[test]
    fn matches_any_of_several_queries_in_one_walk() {
        let data = b"the color red\nthe flavour blue\nneither\n";
        let matched = matching_lines(&["color", "flavour"], data, &config(0));
        assert_eq!(matched, ["the color red", "the flavour blue"]);
    }

    #[test]
    fn folds_case_when_asked() {
        let data = b"COLOR\ncolor\n";
        let mut folded = config(0);
        folded.case_sensitivity = CaseSensitivity::Uncased;
        folded.utf8 = true;
        assert_eq!(matching_lines(&["color"], data, &folded).len(), 2);
        assert_eq!(matching_lines(&["color"], data, &config(0)), ["color"]);
    }

    #[test]
    fn locates_spans_inside_the_lines_that_scored() {
        let data = b"the color red\nnothing here\n";
        let patterns = vec!["color".to_string()];
        let config = config(0);
        let engine = Engine::build(&patterns, &config, &us(), &[], cpu()).unwrap();
        let found = search_lines(data, &engine, &config).unwrap();
        let survivors: Vec<&[u8]> = found
            .lines
            .iter()
            .enumerate()
            .filter(|(index, _)| found.matched(*index, 0.0))
            .map(|(_, line)| *line)
            .collect();
        let located = engine.locate(&survivors).unwrap();
        assert_eq!(located.len(), 1);
        assert_eq!(located[0].line_index, 0);
        assert_eq!(located[0].byte_offset, 4);
        assert_eq!(located[0].byte_length, 5);
    }

    #[test]
    fn keeps_matches_when_one_input_is_missing() {
        let patterns = vec!["color".to_string()];
        let config = config(1);
        let engine = Engine::build(&patterns, &config, &us(), &[], cpu()).unwrap();
        let out_cfg = OutputConfig {
            line_numbers: false,
            scores: false,
            show: Show::Lines,
            format: Format::Text,
            prefix: false,
            summary: false,
            terminator: Terminator::Newline,
        };
        let inputs = [
            ("missing.txt", Err(io::Error::from(io::ErrorKind::NotFound))),
            ("present.txt", Ok(InputSource::Buffer(b"colour\n".to_vec()))),
        ];
        let mut written = Vec::new();
        let outcome = search_inputs(
            &mut written,
            &mut io::sink(),
            inputs,
            &engine,
            &config,
            &out_cfg,
        )
        .unwrap();
        assert_eq!(outcome.readable, 1);
        assert_eq!(outcome.total, 1);
        assert_eq!(written, b"colour\n");
    }

    // endregion: Searching

    // region: Effort and Folding

    #[test]
    fn climbs_the_ladder_without_ever_narrowing() {
        // The one property that makes the ladder a ladder: every rung must match everything the
        // rung below it matched, so raising --effort can only ever add lines.
        let data = "color scheme\ncolour scheme\nkolor test\nr\u{e9}sum\u{e9} draft\n\
                    resume draft\nunrelated line\n"
            .as_bytes();
        let ladder = [
            Effort::Exact,
            Effort::Typos,
            Effort::Spelling,
            Effort::Accents,
            Effort::Sounds,
        ];
        let mut previous: Vec<String> = Vec::new();
        for effort in ladder {
            let matched = matched_at(effort, "color", data);
            for line in &previous {
                assert!(
                    matched.contains(line),
                    "{effort:?} dropped {line:?}, which {:?} had matched",
                    ladder[0]
                );
            }
            previous = matched;
        }
    }

    #[test]
    fn spends_nothing_on_folds_below_accents() {
        // The executable form of the zero-cost claim: no fold is resolved, so no automaton is
        // compiled, no rewrite runs and no offset map can exist.
        for effort in [Effort::Exact, Effort::Typos, Effort::Spelling] {
            let folds = resolve_folds(&[], effort.folding(), Script::Latin).unwrap();
            assert!(folds.is_empty(), "{effort:?} resolved a fold");
        }
        for effort in [Effort::Accents, Effort::Sounds] {
            let folds = resolve_folds(&[], effort.folding(), Script::Latin).unwrap();
            assert!(!folds.is_empty(), "{effort:?} resolved no fold");
        }
    }

    #[test]
    fn widens_the_ball_one_rung_at_a_time() {
        let ball = |effort: Effort| {
            Vocabulary::build(
                &["washington".to_string()],
                effort.max_distance(),
                effort.alphabet(),
                &us(),
                effort.dictionary(),
            )
            .needles
            .len()
        };
        // Exact is the query and nothing else; every rung above widens.
        assert_eq!(ball(Effort::Exact), 1);
        assert!(ball(Effort::Typos) > ball(Effort::Exact));
        assert!(ball(Effort::Spelling) > ball(Effort::Typos));
        assert!(ball(Effort::Deep) > ball(Effort::Spelling));
    }

    #[test]
    fn carries_spans_back_through_a_fold() {
        // The defect this wave repairs: `locate` used to walk original bytes against an automaton
        // compiled from folded needles, so it reported only the lines that happened to spell the
        // folded form already.
        let data = "phonetic analysis\nfonetik analysis\n".as_bytes();
        let spans = spans_at(Effort::Sounds, "phonetic", data);
        assert_eq!(spans, ["phonetic", "fonetik"]);
    }

    #[test]
    fn snaps_a_span_outward_when_it_lands_inside_a_rewrite() {
        // `ph` -> `f` makes the folded span shorter than the original, so an offset inside the
        // rewrite has no original byte of its own and must claim the whole region.
        let site = Site {
            folded_offset: 4,
            folded_length: 1,
            original_offset: 4,
            original_length: 2,
        };
        let map = StageMap {
            sites: vec![site],
            starts: vec![0, 1],
        };
        assert_eq!(map.backward(0, 4, Edge::Start), 4);
        assert_eq!(map.backward(0, 4, Edge::End), 6);
        assert_eq!(map.backward(0, 5, Edge::Start), 6);
        // Past the site the domains advance in lockstep again, offset by the drift.
        assert_eq!(map.backward(0, 6, Edge::End), 7);
    }

    #[test]
    fn maps_a_deleting_rule_to_the_bytes_it_removed() {
        // An empty replacement leaves a zero-width mark: the offset is both on the site and after
        // it, and only the edge says which original boundary it names.
        let map = StageMap {
            sites: vec![Site {
                folded_offset: 2,
                folded_length: 0,
                original_offset: 2,
                original_length: 3,
            }],
            starts: vec![0, 1],
        };
        assert_eq!(map.backward(0, 2, Edge::Start), 2);
        assert_eq!(map.backward(0, 2, Edge::End), 5);
    }

    #[test]
    fn chains_transliteration_before_phonetics() {
        // `Latin-Phonetic` has nothing to act on until the script transform has produced a Latin
        // syllable, so the order is a property of the level rather than of the caller.
        assert_eq!(
            Folding::Scripts.transforms(Script::Han),
            ["Han-Latin", "Latin-ASCII", "Latin-Phonetic"]
        );
        assert_eq!(
            Folding::Untouched.transforms(Script::Han),
            Vec::<&str>::new()
        );
        // A Latin query has nothing to transliterate, so the chain is shorter by one.
        assert_eq!(
            Folding::Scripts.transforms(Script::Latin),
            ["Latin-ASCII", "Latin-Phonetic"]
        );
    }

    #[test]
    fn embeds_every_transform_the_ladder_names() {
        for script in Script::ALL {
            for folding in [Folding::Accents, Folding::Sounds, Folding::Scripts] {
                let resolved = resolve_folds(&[], folding, script);
                assert!(
                    resolved.is_ok(),
                    "{script:?} at {folding:?} names a transform that is not embedded"
                );
            }
        }
    }

    /// The lines one effort matches, as owned strings so rungs can be compared against each other.
    fn matched_at(effort: Effort, pattern: &str, data: &[u8]) -> Vec<String> {
        let (engine, config) = engine_at(effort, pattern);
        let found = search_lines(data, &engine, &config).expect("search runs");
        found
            .lines
            .iter()
            .enumerate()
            .filter(|(index, _)| found.matched(*index, 0.0))
            .map(|(_, line)| String::from_utf8_lossy(line).into_owned())
            .collect()
    }

    /// The matched spans one effort locates, sliced from the original lines.
    fn spans_at(effort: Effort, pattern: &str, data: &[u8]) -> Vec<String> {
        let (engine, config) = engine_at(effort, pattern);
        let found = search_lines(data, &engine, &config).expect("search runs");
        let survivors: Vec<&[u8]> = found
            .lines
            .iter()
            .enumerate()
            .filter(|(index, _)| found.matched(*index, 0.0))
            .map(|(_, line)| *line)
            .collect();
        engine
            .locate(&survivors)
            .expect("locate runs")
            .into_iter()
            .map(|one| {
                let line = survivors[one.line_index];
                String::from_utf8_lossy(&line[one.byte_offset..one.byte_offset + one.byte_length])
                    .into_owned()
            })
            .collect()
    }

    fn engine_at(effort: Effort, pattern: &str) -> (Engine, SearchConfig) {
        let config = SearchConfig {
            max_distance: effort.max_distance(),
            alphabet: effort.alphabet(),
            case_sensitivity: CaseSensitivity::Uncased,
            utf8: true,
            dictionary: effort.dictionary(),
            floor: 0.0,
            top: None,
        };
        let folds = resolve_folds(&[], effort.folding(), Script::Latin).expect("folds resolve");
        let patterns = vec![pattern.to_string()];
        let engine =
            Engine::build(&patterns, &config, &us(), &folds, cpu()).expect("engine builds");
        (engine, config)
    }

    // endregion: Effort and Folding
}

// endregion: Tests
