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
//! The table is `transform<TAB>source<TAB>target`, grouped by transform.

pub const FOLDS_TABLE: &str = include_str!("folds.tsv");

/// One named transform, as the parallel needle and replacement lists a rewrite takes.
pub struct Fold {
    pub sources: Vec<String>,
    pub targets: Vec<String>,
}

impl Fold {
    /// Read one transform out of the embedded table, by its CLDR name.
    pub fn load(name: &str) -> Self {
        let mut sources = Vec::new();
        let mut targets = Vec::new();
        for row in FOLDS_TABLE.lines() {
            let mut columns = row.split('\t');
            let (Some(transform), Some(source), Some(target)) =
                (columns.next(), columns.next(), columns.next())
            else {
                continue;
            };
            if transform != name || source.is_empty() {
                continue;
            }
            sources.push(source.to_string());
            targets.push(target.to_string());
        }
        Self { sources, targets }
    }

    /// Every transform the table carries, for `--fold`'s error message.
    pub fn known() -> Vec<&'static str> {
        let mut names: Vec<&str> = FOLDS_TABLE
            .lines()
            .filter_map(|row| row.split('\t').next())
            .collect();
        names.dedup();
        names
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}
