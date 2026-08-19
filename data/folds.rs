//! Script and phonetic folds.
//!
//! `Han-Latin`, `Hant-Latin`, `Latin-ASCII`, `Cyrillic-Latin` and `Greek-Latin` are extracted from
//! the Unicode CLDR transform rules at <https://github.com/unicode-org/cldr/tree/main/common/transforms>,
//! under the Unicode License v3, which is permissive and compatible with this crate's Apache-2.0.
//!
//! `Kana-Latin` is CLDR's `Katakana-Latin-BGN` with the hiragana half composed in: CLDR ships
//! `Hiragana-Latin` only as a compound of two transforms, and one fold is one leftmost-longest walk
//! that cannot feed itself, so each hiragana rule is written out against its katakana counterpart.
//!
//! `Latin-Phonetic`, `Pinyin-Fuzzy`, `Cologne-Phonetic` and `Daitch-Mokotoff` are authored here
//! from published algorithms rather than extracted from any data file, so they carry no third-party
//! licence at all. Each is a context-free approximation of a context-sensitive original.
//!
//! The three phonetic models are alternatives to one another, not layers: `Latin-Phonetic` follows
//! Metaphone, `Cologne-Phonetic` follows Kölner Phonetik for German, and `Daitch-Mokotoff` is built
//! for Slavic and Germanic surnames spelled across scripts. Stacking two of them would collapse
//! distinctions neither drops alone, so only the first is reached by `--effort` and the other two
//! are named explicitly with `--fold`.
//!
//! CLDR's rule language is context-sensitive in general - `$vowel { x } $consonant → y` - so only
//! the context-free subset compiles into a plain replace dictionary. That subset is nearly all of
//! `Han-Latin`, which is what makes folding a Chinese corpus to pinyin a single automaton walk.
//!
//! The table is `transform<TAB>source<TAB>target`. A transform's rows are contiguous only by
//! convention and `Latin-ASCII`'s are not, so nothing here may assume one run per name.

use std::collections::HashSet;

pub const FOLDS_TABLE: &str = include_str!("folds.tsv");

/// What a fold reads and what it writes, which is all that decides whether two of them can share
/// one automaton.
///
/// Over codepoints rather than bytes: every multi-byte script shares the UTF-8 continuation range,
/// so a byte-level comparison finds dozens of collisions between any two scripts and concludes
/// that nothing anywhere is independent.
#[derive(Default)]
pub struct Alphabets {
    /// The first codepoint of every source. Two folds that share none can never both begin a match
    /// at one position, which is what lets a single leftmost-longest walk stand in for two.
    pub source_heads: HashSet<char>,
    /// Every codepoint a replacement can produce, so a fold that would act on another's output is
    /// recognizable before either has run.
    pub targets: HashSet<char>,
}

/// One named transform, as the parallel needle and replacement lists a rewrite takes.
pub struct Fold {
    /// Borrowed from the embedded table, so naming a fold costs nothing.
    pub name: &'static str,
    pub sources: Vec<String>,
    pub targets: Vec<String>,
    pub alphabets: Alphabets,
}

impl Fold {
    /// Read several transforms in one pass over the table, answering in the order asked.
    ///
    /// One scan for the whole request rather than one per name: the table is 46,651 rows, and a
    /// fold chain that named seven transforms would otherwise read all of them seven times.
    /// A name the table does not carry comes back empty, which is what the caller reports.
    pub fn load_all<Name: AsRef<str>>(names: &[Name]) -> Vec<Self> {
        let mut folds: Vec<Self> = names.iter().map(|_| Self::empty("")).collect();
        for row in FOLDS_TABLE.lines() {
            let mut columns = row.split('\t');
            let (Some(transform), Some(source), Some(target)) =
                (columns.next(), columns.next(), columns.next())
            else {
                continue;
            };
            if source.is_empty() {
                continue;
            }
            for (index, name) in names.iter().enumerate() {
                if name.as_ref() != transform {
                    continue;
                }
                // The row is where the name gets borrowed from, so recognizing a fold and naming
                // it are the same visit rather than a second scan.
                folds[index].name = transform;
                folds[index].push(source, target);
            }
        }
        folds
    }

    /// Read one transform out of the embedded table, by its CLDR name.
    pub fn load(name: &str) -> Self {
        Self::load_all(&[name]).pop().expect("one name, one fold")
    }

    /// Every transform the table carries, for `--fold`'s error message.
    pub fn known() -> Vec<&'static str> {
        let mut names: Vec<&str> = FOLDS_TABLE
            .lines()
            .filter_map(|row| row.split('\t').next())
            .collect();
        // Sorted before deduplication because a transform's rows are not one contiguous run:
        // `Latin-ASCII` occupies two blocks, and a consecutive-only dedup lists it twice.
        names.sort_unstable();
        names.dedup();
        names
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The same fold with only the sources `keep` accepts, alphabets recomputed to match.
    ///
    /// Used to spend a source across a chain: whichever fold claims it first keeps it, and every
    /// later fold loses it, which is what a sequence of rewrites does anyway.
    pub fn retaining<'a>(&'a self, mut keep: impl FnMut(&'a str) -> bool) -> Self {
        let mut kept = Self::empty(self.name);
        for (source, target) in self.sources.iter().zip(&self.targets) {
            if keep(source.as_str()) {
                kept.push(source, target);
            }
        }
        kept
    }

    fn empty(name: &'static str) -> Self {
        Self {
            name,
            sources: Vec::new(),
            targets: Vec::new(),
            alphabets: Alphabets::default(),
        }
    }

    fn push(&mut self, source: &str, target: &str) {
        if let Some(head) = source.chars().next() {
            self.alphabets.source_heads.insert(head);
        }
        self.alphabets.targets.extend(target.chars());
        self.sources.push(source.to_string());
        self.targets.push(target.to_string());
    }
}
