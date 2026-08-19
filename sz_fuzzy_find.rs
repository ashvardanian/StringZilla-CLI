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

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringtape::{BytesTape, BytesTapeView};
use stringzilla::szs::{
    AnyBytesTape, Bm25Params, CaseSensitivity, DeviceScope, OverlapPolicy, Substrings,
    SubstringsMatch, UnifiedAlloc, UnifiedVec,
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

    /// Every script-to-Latin transform the table carries, in the order they feed each other.
    ///
    /// All of them, rather than the one the query happens to be written in: it is the corpus that
    /// decides what is worth folding, and `beijing` should reach 北京 without being typed in Han.
    /// Fusing them is what makes that affordable - the five cost one walk more than the one did.
    ///
    /// `Kana-Latin` leads because six of its rules read a Latin vowel before U+30FC, so it is the
    /// one transform another's output can reach. `Hant-Latin` precedes `Han-Latin` because it is
    /// the specialisation: the 101 characters both spell get the traditional reading, and placing
    /// it second would spend every one of its rules and leave it empty.
    const TRANSLITERATIONS: [&'static str; 5] = [
        "Kana-Latin",
        "Hant-Latin",
        "Han-Latin",
        "Cyrillic-Latin",
        "Greek-Latin",
    ];
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
    fn transforms(self) -> Vec<&'static str> {
        let mut chain = Vec::new();
        if self == Folding::Scripts {
            chain.extend(Script::TRANSLITERATIONS);
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
fn resolve_folds(named: &[String], folding: Folding) -> Result<Vec<Fold>, Failure> {
    let wanted: Vec<String> = match named.is_empty() {
        false => named.to_vec(),
        true => folding
            .transforms()
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    let folds = Fold::load_all(&wanted);
    for (fold, name) in folds.iter().zip(&wanted) {
        if fold.is_empty() {
            return Err(Failure::Unresolved {
                path: name.clone(),
                subject: "CLDR transform".to_string(),
                note: "pass --fold with an embedded transform, such as Han-Latin or Latin-ASCII",
            });
        }
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

/// The corpus in the one shape every kernel here wants: a contiguous byte run plus the offset each
/// line starts at, which is what a tape already is.
///
/// An entry runs to the start of the next one and so carries its own terminator - the offsets are
/// a partition, and a partition cannot leave gaps. Nothing matches into one, since no needle holds
/// a newline, and [`trimmed`] takes it off again for the handful of lines that reach an output mode.
enum Corpus<'a> {
    /// The bytes where the host can already read them: the mapping itself, or a rewrite's product.
    Host {
        data: Cow<'a, [u8]>,
        offsets: Vec<u64>,
    },
    /// One staged copy in unified memory, because a CUDA scope cannot follow a host pointer.
    ///
    /// Staged once for a whole run rather than once per walk: the same corpus serves scoring,
    /// locating, and every fold stage between them.
    Staged(BytesTape<u64, UnifiedAlloc>),
    /// A rewrite's product, left where the kernel that produced it wrote it.
    ///
    /// The slot is a run-owned buffer that outlives any one rewrite, so `lines` says how much of
    /// it this corpus is - the rest is capacity a previous, larger input left behind.
    Rewritten { slot: Rewrite, lines: usize },
}

/// The two buffers `replace_into` fills, which are the two buffers a tape is made of.
///
/// Where they are allocated is the whole point: unified memory lets a CUDA scope read the product
/// back without a copy, and costs a host scope dearly - managed pages are not ordinary memory - so
/// the choice follows the device rather than being made once for both.
enum Rewrite {
    Host {
        data: Vec<u8>,
        offsets: Vec<usize>,
    },
    Unified {
        data: UnifiedVec<u8>,
        offsets: UnifiedVec<usize>,
    },
}

impl Rewrite {
    fn allocate(device: &DeviceScope, bytes: usize, entries: usize) -> Self {
        match device.is_gpu() {
            false => Self::Host {
                data: vec![0u8; bytes],
                offsets: vec![0usize; entries],
            },
            true => {
                let mut data: UnifiedVec<u8> = UnifiedVec::new_in(UnifiedAlloc);
                let mut offsets: UnifiedVec<usize> = UnifiedVec::new_in(UnifiedAlloc);
                data.resize(bytes, 0);
                offsets.resize(entries, 0);
                Self::Unified { data, offsets }
            }
        }
    }

    /// This slot, made large enough for one more rewrite.
    ///
    /// Grows and never shrinks, so a run over many files of similar shape allocates for the first
    /// of them and for none of the rest.
    fn grown_to(&mut self, bytes: usize, entries: usize) {
        match self {
            Self::Host { data, offsets } => {
                if data.len() < bytes {
                    data.resize(bytes, 0);
                }
                if offsets.len() < entries {
                    offsets.resize(entries, 0);
                }
            }
            Self::Unified { data, offsets } => {
                if data.len() < bytes {
                    data.resize(bytes, 0);
                }
                if offsets.len() < entries {
                    offsets.resize(entries, 0);
                }
            }
        }
    }

    /// How many bytes this slot can take, which is what a rewrite is offered before it asks for more.
    fn written_capacity(&self) -> usize {
        self.parts().0.len()
    }

    /// The prefix a rewrite of this shape writes into, since the slot itself may be larger.
    fn parts_upto(&mut self, bytes: usize, entries: usize) -> (&mut [u8], &mut [usize]) {
        match self {
            Self::Host { data, offsets } => (&mut data[..bytes], &mut offsets[..entries]),
            Self::Unified { data, offsets } => (&mut data[..bytes], &mut offsets[..entries]),
        }
    }

    fn parts(&self) -> (&[u8], &[usize]) {
        match self {
            Self::Host { data, offsets } => (data, offsets),
            Self::Unified { data, offsets } => (data, offsets),
        }
    }
}

impl<'a> Corpus<'a> {
    /// Cut an input into lines, copying the bytes only where the device cannot read them in place.
    fn lines_of(device: &DeviceScope, data: &'a [u8], newlines: Newlines) -> Result<Self, Failure> {
        let mut offsets: Vec<u64> = Vec::new();
        let mut end = 0usize;
        for line in named_lines(data, newlines) {
            offsets.push(line.offset as u64);
            end = line.offset + line.whole.len();
        }
        offsets.push(end as u64);
        Self::Host {
            data: Cow::Borrowed(&data[..end]),
            offsets,
        }
        .on(device)
    }

    /// Gather already-selected lines into a corpus of their own.
    ///
    /// Unlike [`Corpus::lines_of`] this always copies, and answers for the few hundred lines an
    /// output mode asks about rather than for a whole input.
    fn gathered<'b>(
        device: &DeviceScope,
        lines: impl Iterator<Item = &'b [u8]> + Clone,
    ) -> Result<Self, Failure> {
        // Cloned rather than collected: the first pass sizes the buffer and the second fills it,
        // so a caller with the lines already in hand never builds a vector of slices to be read
        // twice and dropped.
        let mut data = Vec::with_capacity(lines.clone().map(|line| line.len()).sum());
        let mut offsets = Vec::with_capacity(lines.clone().count() + 1);
        for line in lines {
            offsets.push(data.len() as u64);
            data.extend_from_slice(line);
        }
        offsets.push(data.len() as u64);
        Self::Host {
            data: Cow::Owned(data),
            offsets,
        }
        .on(device)
    }

    /// This corpus where the device can reach it, which for a host scope is where it already is.
    ///
    /// The one place the staging decision is made, so no verb below has to ask again which memory
    /// its haystacks live in.
    fn on(self, device: &DeviceScope) -> Result<Self, Failure> {
        // A rewrite's product is already unified, and a host scope reads unified memory fine, so
        // neither device has anything left to do to it.
        if !device.is_gpu() || matches!(self, Self::Rewritten { .. }) {
            return Ok(self);
        }
        let mut tape: BytesTape<u64, UnifiedAlloc> =
            BytesTape::with_capacity_in(self.bytes().len(), self.len() + 1, UnifiedAlloc)
                .map_err(|error| engine_failure("corpus staging", error))?;
        for index in 0..self.len() {
            tape.push(self.line(index))
                .map_err(|error| engine_failure("corpus staging", error))?;
        }
        Ok(Self::Staged(tape))
    }

    /// What the engines are handed: three words, and no copy at the call site.
    fn haystacks(&self) -> AnyBytesTape<'_> {
        match self {
            // Safety: the offsets ascend, start at zero and end at `data.len()`, since every
            // constructor above builds them as a partition of exactly these bytes.
            Self::Host { data, offsets } => {
                AnyBytesTape::View64(unsafe { BytesTapeView::from_raw_parts(data, offsets) })
            }
            Self::Staged(tape) => AnyBytesTape::View64(tape.view()),
            Self::Rewritten { slot, lines } => {
                let (data, offsets) = slot.parts();
                AnyBytesTape::View64(unsafe {
                    BytesTapeView::from_raw_parts(data, as_u64(&offsets[..lines + 1]))
                })
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Host { offsets, .. } => offsets.len() - 1,
            Self::Staged(tape) => tape.len(),
            Self::Rewritten { lines, .. } => *lines,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every byte the corpus spans, which is what a length-normalized BM25 needs.
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Host { data, .. } => data,
            Self::Staged(tape) => tape.data_slice(),
            Self::Rewritten { slot, lines } => {
                let (data, offsets) = slot.parts();
                &data[..offsets[*lines]]
            }
        }
    }

    /// One entry, terminator and all.
    fn line(&self, index: usize) -> &[u8] {
        match self {
            Self::Host { data, offsets } => {
                &data[offsets[index] as usize..offsets[index + 1] as usize]
            }
            Self::Staged(tape) => &tape[index],
            Self::Rewritten { slot, .. } => {
                let (data, offsets) = slot.parts();
                &data[offsets[index]..offsets[index + 1]]
            }
        }
    }
}

/// The same offsets seen as the width a tape addresses them with.
///
/// `replace_into` writes `usize` and a tape view reads `u64`; on every target this suite builds for
/// those are one type, and the assertion below is what says so.
fn as_u64(offsets: &[usize]) -> &[u64] {
    const _: () = assert!(std::mem::size_of::<usize>() == std::mem::size_of::<u64>());
    unsafe { std::slice::from_raw_parts(offsets.as_ptr().cast::<u64>(), offsets.len()) }
}

impl Corpus<'_> {
    /// The buffer this corpus was written into, given up so the next rewrite can use it again.
    ///
    /// Only a rewrite's product has one; a mapping and a staged tape own nothing a run can pass on.
    fn reclaimed(self) -> Option<Rewrite> {
        match self {
            Self::Rewritten { slot, .. } => Some(slot),
            _ => None,
        }
    }
}

/// An entry without whatever separated it from the next one.
///
/// The offsets partition the input, so a terminator rides along with the line it ends; the split
/// kernel that found it in the first place is what takes it back off.
fn trimmed(entry: &[u8], newlines: Newlines) -> &[u8] {
    LineIter::new(entry, newlines).next().unwrap_or(entry)
}

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

/// Every match in `corpus`, under the leftmost-longest cover.
///
/// Two walks rather than one: sizing the match buffer needs a count, and the engine will not
/// answer both from a single pass. Both callers pay it, so both call this.
fn cover(
    automaton: &Substrings,
    device: &DeviceScope,
    corpus: &Corpus,
    counting: &'static str,
    locating: &'static str,
) -> Result<Vec<SubstringsMatch>, Failure> {
    if corpus.is_empty() {
        return Ok(Vec::new());
    }
    let haystacks = corpus.haystacks();
    let mut counts = vec![0usize; corpus.len()];
    let total = automaton
        .count_into(
            device,
            &haystacks,
            OverlapPolicy::LeftmostLongest,
            &mut counts,
        )
        .map_err(|error| engine_failure(counting, error))?;

    let mut matches = vec![SubstringsMatch::default(); total];
    let written = automaton
        .find_into(
            device,
            &haystacks,
            OverlapPolicy::LeftmostLongest,
            &mut matches,
        )
        .map_err(|error| engine_failure(locating, error))?;
    matches.truncate(written);
    Ok(matches)
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
                let sources = patterns.iter().map(|one| one.as_bytes());
                let folded = folder.apply(&device, &Corpus::gathered(&device, sources)?)?;
                (0..folded.len())
                    .map(|index| String::from_utf8_lossy(folded.line(index)).into_owned())
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
    fn score(&self, corpus: &Corpus) -> Result<Vec<f32>, Failure> {
        let mut scores = vec![0.0f32; corpus.len()];
        if corpus.is_empty() {
            return Ok(scores);
        }

        // Under a fold the corpus is rewritten once and matched in the folded domain; the caller
        // still holds the original lines, which is what gets printed.
        let folded = match &self.folder {
            Some(folder) => Some(folder.apply(&self.device, corpus)?),
            None => None,
        };
        let corpus: &Corpus = folded.as_ref().unwrap_or(corpus);
        let haystacks = corpus.haystacks();

        // Length normalization divides by the corpus mean, and BM25 refuses a mean that is not
        // positive rather than quietly ignoring it, so an all-empty corpus scores unnormalized.
        let mean = corpus.bytes().len() as f32 / corpus.len() as f32;
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
        drop(haystacks);

        // Nothing reads the folded bytes past this point, so the buffer goes back for the next file.
        if let (Some(folder), Some(folded)) = (&self.folder, folded) {
            folder.reclaim(folded);
        }
        Ok(scores)
    }

    /// Count and find in one place, over whatever haystacks the caller hands in.
    ///
    /// Deliberately a second, much smaller walk than scoring: sizing a match buffer needs a prior
    /// count, and paying for both over the whole corpus would triple the work to answer a question
    /// only a few hundred lines ever ask.
    fn find(&self, corpus: &Corpus) -> Result<Vec<SubstringsMatch>, Failure> {
        cover(
            &self.automaton,
            &self.device,
            corpus,
            "match counting",
            "match location",
        )
    }

    /// Locate the variants inside lines that already scored, always in the caller's own bytes.
    ///
    /// The domain the automaton walked is this method's business alone: an output mode receives
    /// spans it can slice directly and never learns whether a fold ran.
    fn locate(&self, survivors: &Corpus) -> Result<Vec<Located>, Failure> {
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
                let (rewrites, folded) = folder.rewrites(&self.device, survivors)?;
                let located: Vec<Located> = self
                    .find(&folded)?
                    .into_iter()
                    .map(|found| rewrites.located(found))
                    .collect();
                folder.reclaim(folded);
                Ok(located)
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

/// One or more transforms compiled into a single rewrite.
struct Layer {
    automaton: Substrings,
    targets: Vec<String>,
}

/// The transforms a run folds through, applied to corpus and query alike.
///
/// Folds that can see each other's output must run in turn - `Latin-Phonetic` has nothing to act
/// on until a script transform has produced a Latin syllable for it - but folds that cannot are
/// one automaton, since running them in sequence would let only one of them fire at each position
/// anyway. So the chain is partitioned into layers, and a layer is one walk.
struct Folder {
    layers: Vec<Layer>,
    /// Buffers the chain hands out and takes back.
    ///
    /// A fold's product is a whole rewritten corpus, and a chain over many inputs would otherwise
    /// allocate one per layer per file. Two slots serve a chain of any length, and they are grown
    /// by the largest input seen rather than sized by the current one.
    spare: RefCell<Vec<Rewrite>>,
}

/// Whether `later` can share `earlier`'s automaton.
///
/// Three ways it cannot: they begin matches at the same character, so a single leftmost-longest
/// walk would have a choice the sequence never offered it; `later` reads what `earlier` writes;
/// or `earlier` reads what `later` writes. Failing the test opens a new layer, which is what the
/// sequence did for every fold regardless, so a false negative costs a pass and never a result.
fn fusable(earlier: &Fold, later: &Fold) -> bool {
    let (before, after) = (&earlier.alphabets, &later.alphabets);
    before.source_heads.is_disjoint(&after.source_heads)
        && after.source_heads.is_disjoint(&before.targets)
        && before.source_heads.is_disjoint(&after.targets)
}

/// The chain with every source claimed by the first fold that carries it.
///
/// A sequence already behaves this way - the earlier fold rewrites the character and the later one
/// never sees it - so spending sources up front is what lets two folds share a walk without
/// changing what either produces. Keyed on the exact bytes: `А` and `а` are different rules with
/// different replacements, and a case-folded key would drop one of every such pair.
fn spend_sources(folds: &[Fold]) -> Vec<Fold> {
    let mut spent: HashSet<&str> = HashSet::new();
    let mut chain = Vec::with_capacity(folds.len());
    for fold in folds {
        // Spending can empty a fold outright, and an empty needle list is not something to
        // compile. An emptied fold is spent, not unresolved.
        let kept = fold.retaining(|source| spent.insert(source));
        if !kept.is_empty() {
            chain.push(kept);
        }
    }
    chain
}

impl Layer {
    /// One automaton over every source of every fold in the layer.
    ///
    /// Concatenated in chain order, because `Substrings` numbers its needles in the order it is
    /// given them and [`Layer::sites`] reads `targets[needle_index]` back out.
    fn compile(
        device: &DeviceScope,
        folds: &[&Fold],
        case_sensitivity: CaseSensitivity,
    ) -> Result<Self, Failure> {
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for fold in folds {
            sources.extend(fold.sources.iter().cloned());
            targets.extend(fold.targets.iter().cloned());
        }
        Ok(Self {
            automaton: Substrings::new(device, &sources, case_sensitivity)
                .map_err(|error| engine_failure("fold", error))?,
            targets,
        })
    }
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
        let chain = spend_sources(folds);
        let mut layers = Vec::new();
        let mut open: Vec<&Fold> = Vec::new();
        for fold in &chain {
            if !open.is_empty() && !open.iter().all(|held| fusable(held, fold)) {
                layers.push(Layer::compile(device, &open, case_sensitivity)?);
                open.clear();
            }
            open.push(fold);
        }
        if !open.is_empty() {
            layers.push(Layer::compile(device, &open, case_sensitivity)?);
        }
        Ok(Self {
            layers,
            spare: RefCell::new(Vec::new()),
        })
    }

    /// One layer per fold and no sources spent, which is what a chain of rewrites has always been.
    ///
    /// Kept as the oracle [`Folder::new`] is differenced against, since fusing folds that can see
    /// each other's output goes wrong quietly - the spans come back mapped to the wrong bytes
    /// rather than the run failing.
    #[cfg(test)]
    fn sequential(
        device: &DeviceScope,
        folds: &[Fold],
        case_sensitivity: CaseSensitivity,
    ) -> Result<Self, Failure> {
        let mut layers = Vec::with_capacity(folds.len());
        for fold in folds {
            layers.push(Layer::compile(device, &[fold], case_sensitivity)?);
        }
        Ok(Self {
            layers,
            spare: RefCell::new(Vec::new()),
        })
    }

    /// A slot to write into, if the run has one to spare.
    fn lend(&self) -> Option<Rewrite> {
        self.spare.borrow_mut().pop()
    }

    /// Take a corpus back once nothing reads it, so its buffer serves the next fold or the next file.
    fn reclaim(&self, corpus: Corpus<'_>) {
        if let Some(slot) = corpus.reclaimed() {
            self.spare.borrow_mut().push(slot);
        }
    }

    /// The bytes each idle slot can hold, so a test can see that a second input reuses them.
    #[cfg(test)]
    fn spare_capacity(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self
            .spare
            .borrow()
            .iter()
            .map(|slot| slot.parts().0.len())
            .collect();
        sizes.sort_unstable();
        sizes
    }

    /// Rewrite every haystack through every stage in turn, returning owned bytes since the product
    /// of a rewrite is a new tape.
    fn apply(&self, device: &DeviceScope, corpus: &Corpus) -> Result<Corpus<'static>, Failure> {
        // A folder is only built from a non-empty chain, so a first stage always exists to produce
        // the owned tape the rest of the chain then rewrites in turn.
        let (first, rest) = self
            .layers
            .split_first()
            .expect("a fold chain is never empty");
        let mut carried = first.rewrite(device, corpus, self.lend())?;
        for layer in rest {
            let next = layer.rewrite(device, &carried, self.lend())?;
            // The corpus just consumed is the buffer the layer after next will write into.
            self.reclaim(carried);
            carried = next;
        }
        Ok(carried)
    }

    /// Where every stage rewrote these lines, as the map from the folded bytes back to the caller's.
    ///
    /// Returned with the folded corpus the last stage produced, so locating never re-runs the chain.
    ///
    /// Built only when an output mode asks for spans, and only over the lines that already scored -
    /// a corpus-wide map of `Han-Latin` would run several times the size of the corpus itself.
    fn rewrites<'a>(
        &self,
        device: &DeviceScope,
        corpus: &Corpus,
    ) -> Result<(Rewrites, Corpus<'a>), Failure> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let (first, rest) = self
            .layers
            .split_first()
            .expect("a fold chain is never empty");
        layers.push(first.sites(device, corpus)?);
        let mut carried = first.rewrite(device, corpus, self.lend())?;
        for layer in rest {
            layers.push(layer.sites(device, &carried)?);
            let next = layer.rewrite(device, &carried, self.lend())?;
            self.reclaim(carried);
            carried = next;
        }
        // The last stage's own output is the folded corpus, so the caller takes it from here rather
        // than running the whole chain a second time to arrive at the same bytes.
        Ok((Rewrites { layers }, carried))
    }
}

impl Layer {
    /// This layer's rewrite of every haystack.
    fn rewrite(
        &self,
        device: &DeviceScope,
        corpus: &Corpus,
        spare: Option<Rewrite>,
    ) -> Result<Corpus<'static>, Failure> {
        if corpus.is_empty() {
            return Corpus::gathered(device, std::iter::empty());
        }
        let haystacks = corpus.haystacks();
        let bound = self
            .automaton
            .replace_bound(&self.targets, corpus.bytes().len())
            .map_err(|error| engine_failure("fold sizing", error))?;

        // `replace_into` writes a contiguous run and the boundary of every entry in it, which is
        // the next corpus already, so the slot is handed on rather than split back into one
        // allocation per line.
        //
        // Sized by what the last rewrite needed rather than by `replace_bound`, which is the
        // widest-expanding needle applied to every input byte and on a Han corpus several times
        // what a rewrite actually produces. A slot that turns out too small is grown once and the
        // rewrite runs again - a refusal leaves the true boundaries behind, so the second attempt
        // asks for the exact size rather than guessing at it.
        let lines = corpus.len();
        let mut slot = spare.unwrap_or_else(|| Rewrite::allocate(device, corpus.bytes().len(), 0));
        slot.grown_to(corpus.bytes().len(), lines + 1);

        let mut capacity = slot.written_capacity();
        if self
            .fill(device, &haystacks, &mut slot, capacity, lines)
            .is_err()
        {
            let needed = slot.parts().1[lines];
            // The bound is the fallback for a backend that refuses before writing any boundary.
            capacity = match needed > capacity && needed <= bound {
                true => needed,
                false => bound,
            };
            slot.grown_to(capacity, lines + 1);
            self.fill(device, &haystacks, &mut slot, capacity, lines)
                .map_err(|error| engine_failure("fold", error))?;
        }
        Ok(Corpus::Rewritten { slot, lines })
    }

    /// One rewrite into the first `capacity` bytes of `slot`, reporting only whether it fit.
    fn fill(
        &self,
        device: &DeviceScope,
        haystacks: &AnyBytesTape<'_>,
        slot: &mut Rewrite,
        capacity: usize,
        lines: usize,
    ) -> Result<usize, stringzilla::szs::Error> {
        let (data, offsets) = slot.parts_upto(capacity, lines + 1);
        self.automaton.replace_into(
            device,
            haystacks,
            OverlapPolicy::LeftmostLongest,
            &self.targets,
            data,
            offsets,
        )
    }

    /// Where this layer fires, in the coordinates of the bytes handed to it.
    ///
    /// The same `LeftmostLongest` cover the rewrite uses, so the sites found here are exactly the
    /// substitutions that happened.
    fn sites(&self, device: &DeviceScope, corpus: &Corpus) -> Result<LayerMap, Failure> {
        let mut sites = Vec::new();
        let mut starts = vec![0usize; corpus.len() + 1];
        if corpus.is_empty() {
            return Ok(LayerMap { sites, starts });
        }

        let mut found = cover(
            &self.automaton,
            device,
            corpus,
            "fold counting",
            "fold locating",
        )?;
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
        for tail in line..corpus.len() {
            starts[tail + 1] = sites.len();
        }
        Ok(LayerMap { sites, starts })
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
struct LayerMap {
    sites: Vec<Site>,
    /// `sites[starts[line]..starts[line + 1]]` are one line's, in ascending offset order.
    starts: Vec<usize>,
}

impl LayerMap {
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
    layers: Vec<LayerMap>,
}

impl Rewrites {
    /// Carry one offset back through every layer, last applied first.
    fn backward(&self, line: usize, folded: usize, edge: Edge) -> usize {
        self.layers
            .iter()
            .rev()
            .fold(folded, |offset, layer| layer.backward(line, offset, edge))
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
    corpus: Corpus<'a>,
    scores: Vec<f32>,
}

impl Searched<'_> {
    /// A line matched when it clears the floor, which defaults to any positive score - and a
    /// positive score means "matched" only because every weight is strictly positive.
    fn matched(&self, index: usize, floor: f32) -> bool {
        self.scores[index] > 0.0 && self.scores[index] >= floor
    }

    /// How many lines matched, without gathering them.
    ///
    /// `--show count`, `files` and `files-without` print a tally or a path and never look at a
    /// line, so on a corpus that mostly matches they would otherwise build a list per input only
    /// to ask for its length.
    fn matched_count(&self, floor: f32, top: Option<usize>) -> usize {
        let matched = (0..self.corpus.len())
            .filter(|index| self.matched(*index, floor))
            .count();
        top.map_or(matched, |top| matched.min(top))
    }

    /// Every line that matched, trimmed of its terminator and paired with the score it earned.
    ///
    /// The scored pass already said which lines these are, so they are gathered once here and
    /// every output mode reads from the list rather than re-testing the corpus.
    fn survivors(&self, floor: f32, newlines: Newlines) -> Vec<(usize, &[u8], f32)> {
        (0..self.corpus.len())
            .filter(|index| self.matched(*index, floor))
            .map(|index| {
                (
                    index,
                    trimmed(self.corpus.line(index), newlines),
                    self.scores[index],
                )
            })
            .collect()
    }
}

/// Cut one input into lines and score all of them in a single walk.
fn search_lines<'a>(
    data: &'a [u8],
    engine: &Engine,
    config: &SearchConfig,
) -> Result<Searched<'a>, Failure> {
    let corpus = Corpus::lines_of(&engine.device, data, Newlines::from_utf8(config.utf8))?;
    let scores = engine.score(&corpus)?;
    Ok(Searched { corpus, scores })
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

    let folds = resolve_folds(&args.fold, effort.folding())?;

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
        // A tally needs no lines behind it, so the modes that print one never gather them.
        let tallied = matches!(out_cfg.show, Show::Count | Show::Files | Show::FilesWithout);
        let mut survivors = match tallied {
            true => Vec::new(),
            false => found.survivors(config.floor, Newlines::from_utf8(config.utf8)),
        };

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
        let count = match tallied {
            true => found.matched_count(config.floor, config.top),
            false => survivors.len(),
        };

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
                let lines = survivors.iter().map(|(_, line, _)| *line);
                let selected = Corpus::gathered(&engine.device, lines)?;
                for located in engine.locate(&selected)? {
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
            .survivors(0.0, Newlines::from_utf8(config.utf8))
            .into_iter()
            .map(|(_, line, _)| String::from_utf8_lossy(line).into_owned())
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
            .survivors(0.0, Newlines::from_utf8(config.utf8))
            .into_iter()
            .map(|(_, line, _)| line)
            .collect();
        let selected = Corpus::gathered(&engine.device, survivors.iter().copied()).unwrap();
        let located = engine.locate(&selected).unwrap();
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
            Effort::Scripts,
            Effort::Deep,
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
            let folds = resolve_folds(&[], effort.folding()).unwrap();
            assert!(folds.is_empty(), "{effort:?} resolved a fold");
        }
        for effort in [Effort::Accents, Effort::Sounds] {
            let folds = resolve_folds(&[], effort.folding()).unwrap();
            assert!(!folds.is_empty(), "{effort:?} resolved no fold");
        }
        // And the claim is about what gets built, not only about what gets named.
        let patterns = vec!["color".to_string()];
        let config = config(1);
        let engine = Engine::build(&patterns, &config, &us(), &[], cpu()).expect("engine builds");
        assert!(
            engine.folder.is_none(),
            "an empty chain compiled an automaton"
        );
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
        let map = LayerMap {
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
        let map = LayerMap {
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
    /// Two folds share a walk only when neither can pick up where the other leaves off.
    #[test]
    fn fuses_only_folds_that_cannot_see_each_other() {
        // The predicate reads a chain whose sources are already spent, since that is the only
        // state it is ever applied to - `Hant-Latin` and `Han-Latin` share 101 characters until
        // the first of them claims them.
        let pair = |first: &str, second: &str| {
            let loaded = Fold::load_all(&[first, second]);
            let chain = spend_sources(&loaded);
            fusable(&chain[0], &chain[1])
        };
        let named = |name: &str| Fold::load(name);
        assert!(pair("Hant-Latin", "Han-Latin"));
        assert!(pair("Cyrillic-Latin", "Greek-Latin"));
        // Six `Kana-Latin` rules are sourced on a Latin vowel before U+30FC, so anything that
        // emits a Latin vowel feeds it.
        assert!(!fusable(&named("Han-Latin"), &named("Kana-Latin")));
        // `Latin-ASCII` strips exactly the tone marks `Han-Latin` emits.
        assert!(!fusable(&named("Han-Latin"), &named("Latin-ASCII")));
        // And phonetics read the Latin that de-accenting produces.
        assert!(!fusable(&named("Latin-ASCII"), &named("Latin-Phonetic")));
    }

    /// Spending a source is what lets folds share a walk, and it is keyed on exact bytes: `А` and
    /// `а` are different rules with different replacements.
    #[test]
    fn spends_a_source_on_the_fold_that_claims_it_first() {
        let folds = Fold::load_all(&["Hant-Latin", "Han-Latin"]);
        let chain = spend_sources(&folds);
        assert_eq!(
            chain[0].sources.len(),
            101,
            "the specialisation keeps all of its rules"
        );
        assert_eq!(
            chain[1].sources.len(),
            folds[1].sources.len() - 101,
            "and the general table loses exactly the ones already spent"
        );

        // Case is not a source of identity here: every rule that differs only by case survives.
        let cyrillic = Fold::load("Cyrillic-Latin");
        let alone = spend_sources(std::slice::from_ref(&cyrillic));
        assert_eq!(alone[0].sources.len(), cyrillic.sources.len());
    }

    /// The buffers a fold writes into outlive the file that sized them.
    #[test]
    fn folds_a_second_file_without_allocating_again() {
        let (engine, config) = engine_at(Effort::Sounds, "phonetic");
        let folder = engine.folder.as_ref().expect("Sounds folds");

        // A wide file first, so the slots are grown to fit it.
        let wide =
            "phonetic analysis of a fairly long line, repeated to give the fold work\n".repeat(64);
        search_lines(wide.as_bytes(), &engine, &config).expect("wide file scores");
        let after_wide = folder.spare_capacity();
        assert!(
            after_wide.iter().all(|bytes| *bytes > 0),
            "no slot came back from the first file: {after_wide:?}"
        );

        // Then a narrow one, which must fit in what the wide one left behind.
        search_lines(b"phonetic\n", &engine, &config).expect("narrow file scores");
        assert_eq!(
            folder.spare_capacity(),
            after_wide,
            "the second file grew a buffer it should have reused"
        );
    }

    /// The partition the shipped tables produce, so a table edit that changes it has to say so.
    #[test]
    fn partitions_the_transliterations_into_one_layer() {
        let device = cpu();
        let named: Vec<String> = [
            "Kana-Latin",
            "Hant-Latin",
            "Han-Latin",
            "Cyrillic-Latin",
            "Greek-Latin",
            "Latin-ASCII",
            "Latin-Phonetic",
        ]
        .iter()
        .map(|one| one.to_string())
        .collect();
        let folds = resolve_folds(&named, Folding::Untouched).expect("folds resolve");
        let folder =
            Folder::new(&device, &folds, CaseSensitivity::Uncased).expect("layers compile");

        // Kana leads alone - six of its rules read a Latin vowel, so anything emitting one feeds
        // it. The four script transforms then share a walk. `Latin-ASCII` strips the tone marks
        // they emit, and phonetics read what it leaves.
        let sizes: Vec<usize> = folder.layers.iter().map(|one| one.targets.len()).collect();
        assert_eq!(sizes.len(), 4, "seven folds, four walks: {sizes:?}");
        assert_eq!(sizes[0], 292, "Kana-Latin, less its 143 duplicated sources");
        assert_eq!(
            sizes[1], 44_774,
            "Hant 101 + Han 44,568 + Cyrillic 52 + Greek 53 - the count that catches a dedupe \
             keyed on case, which would silently drop one rule of every cased pair"
        );
        assert_eq!(sizes[2], 1_223, "Latin-ASCII");
        assert_eq!(sizes[3], 47, "Latin-Phonetic");
    }

    /// The property the whole partition rests on: fusing changes how many walks run, and nothing
    /// else. Checked on the bytes and on the map, because a wrongly fused pair reports spans
    /// against the wrong original rather than failing.
    #[test]
    fn fusing_reproduces_the_sequence_it_replaced() {
        let device = cpu();
        // Named rather than taken from a rung, so the chain holds several transliterations and the
        // partition has something to fuse.
        let named: Vec<String> = [
            "Kana-Latin",
            "Hant-Latin",
            "Han-Latin",
            "Cyrillic-Latin",
            "Greek-Latin",
            "Latin-ASCII",
            "Latin-Phonetic",
        ]
        .iter()
        .map(|one| one.to_string())
        .collect();
        let folds = resolve_folds(&named, Folding::Untouched).expect("folds resolve");

        // Every needle of the small folds and a stride through the large one, then every needle
        // glued to the next - the boundaries a merged automaton could straddle that a sequence
        // never crossed. Sampling keeps all 101 `Hant-Latin` rules, which is where the conflicts
        // that matter live.
        let mut probes: Vec<String> = folds
            .iter()
            .flat_map(|fold| {
                let stride = 1 + fold.sources.len() / 500;
                fold.sources.iter().step_by(stride).cloned()
            })
            .collect();
        let glued: Vec<String> = probes.windows(2).map(|pair| pair.concat()).collect();
        probes.extend(glued);
        let lines: Vec<&[u8]> = probes.iter().map(|one| one.as_bytes()).collect();

        for case_sensitivity in [CaseSensitivity::Cased, CaseSensitivity::Uncased] {
            let fused = Folder::new(&device, &folds, case_sensitivity).expect("layers compile");
            let serial =
                Folder::sequential(&device, &folds, case_sensitivity).expect("stages compile");
            assert!(
                fused.layers.len() < serial.layers.len(),
                "seven folds should not need seven walks, got {}",
                fused.layers.len()
            );

            let corpus = Corpus::gathered(&device, lines.iter().copied()).expect("corpus gathers");
            let by_layer = fused.apply(&device, &corpus).expect("fused folds");
            let by_stage = serial.apply(&device, &corpus).expect("sequential folds");
            assert_eq!(by_layer.len(), by_stage.len());
            for index in 0..by_layer.len() {
                assert_eq!(
                    by_layer.line(index),
                    by_stage.line(index),
                    "line {index} folded differently under fusion"
                );
            }

            // The maps too: an offset carried back must land on the same original byte.
            let (fused_map, _) = fused.rewrites(&device, &corpus).expect("fused map");
            let (serial_map, _) = serial.rewrites(&device, &corpus).expect("sequential map");
            for index in 0..by_layer.len() {
                let folded_length = by_layer.line(index).len();
                for offset in 0..=folded_length {
                    // A start edge answers at any offset; an end edge only ever answers at the far
                    // side of a match, so offset zero is not a question `located` can ask.
                    let edges: &[Edge] = match offset {
                        0 => &[Edge::Start],
                        _ => &[Edge::Start, Edge::End],
                    };
                    for edge in edges {
                        assert_eq!(
                            fused_map.backward(index, offset, *edge),
                            serial_map.backward(index, offset, *edge),
                            "line {index} {:?} offset {offset} maps back differently",
                            String::from_utf8_lossy(by_layer.line(index))
                        );
                    }
                }
            }
        }
    }

    fn chains_transliteration_before_phonetics() {
        let chain = Folding::Scripts.transforms();
        let at = |name: &str| chain.iter().position(|one| *one == name).expect(name);

        // Every embedded transliteration is reached, not just the one the query is written in.
        for name in Script::TRANSLITERATIONS {
            assert!(chain.contains(&name), "{name} is not on the widest rung");
        }
        // `Latin-ASCII` strips the tone marks the transliterations emit, and `Latin-Phonetic` has
        // nothing to act on until a Latin syllable exists, so both follow all of them.
        let accents = at("Latin-ASCII");
        for name in Script::TRANSLITERATIONS {
            assert!(at(name) < accents, "{name} must precede Latin-ASCII");
        }
        assert!(accents < at("Latin-Phonetic"));
        // Kana leads, since anything that emits a Latin vowel feeds it; Hant precedes Han, or
        // spending would leave it with no rules of its own.
        assert_eq!(at("Kana-Latin"), 0);
        assert!(at("Hant-Latin") < at("Han-Latin"));

        assert_eq!(Folding::Untouched.transforms(), Vec::<&str>::new());
    }

    #[test]
    fn embeds_every_transform_the_ladder_names() {
        for folding in [Folding::Accents, Folding::Sounds, Folding::Scripts] {
            let resolved = resolve_folds(&[], folding);
            assert!(
                resolved.is_ok(),
                "{folding:?} names a transform that is not embedded"
            );
        }
    }

    /// The lines one effort matches, as owned strings so rungs can be compared against each other.
    fn matched_at(effort: Effort, pattern: &str, data: &[u8]) -> Vec<String> {
        let (engine, config) = engine_at(effort, pattern);
        let found = search_lines(data, &engine, &config).expect("search runs");
        found
            .survivors(0.0, Newlines::from_utf8(config.utf8))
            .into_iter()
            .map(|(_, line, _)| String::from_utf8_lossy(line).into_owned())
            .collect()
    }

    /// The matched spans one effort locates, sliced from the original lines.
    fn spans_at(effort: Effort, pattern: &str, data: &[u8]) -> Vec<String> {
        let (engine, config) = engine_at(effort, pattern);
        let found = search_lines(data, &engine, &config).expect("search runs");
        let survivors: Vec<&[u8]> = found
            .survivors(0.0, Newlines::from_utf8(config.utf8))
            .into_iter()
            .map(|(_, line, _)| line)
            .collect();
        let selected =
            Corpus::gathered(&engine.device, survivors.iter().copied()).expect("corpus gathers");
        engine
            .locate(&selected)
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
        let folds = resolve_folds(&[], effort.folding()).expect("folds resolve");
        let patterns = vec![pattern.to_string()];
        let engine =
            Engine::build(&patterns, &config, &us(), &folds, cpu()).expect("engine builds");
        (engine, config)
    }

    // endregion: Effort and Folding
}

// endregion: Tests
