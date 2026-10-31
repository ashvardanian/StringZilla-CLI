//! Known typo pairs, from the `client9/misspell` dictionary.
//!
//! Source: <https://github.com/client9/misspell>, taken through the CSV mirror at
//! <https://github.com/crate-ci/typos/blob/master/crates/misspell-dict/assets/compatible.csv>.
//! Licensed MIT, Copyright (c) 2015-2017 Nick Galbreath, which is compatible with this crate's
//! Apache-2.0 and imposes no notice on downstream binaries.
//!
//! Hunspell `REP` tables and Wikipedia's list were the obvious larger sources and are deliberately
//! not used: `de_DE_frami` is GPL-only and the Wikipedia list is CC BY-SA, neither of which can be
//! redistributed inside an Apache-2.0 binary. `codespell` is the same problem - its tool is GPL-2
//! but its dictionary is separately CC BY-SA 3.0.
//!
//! An entry whose correction is ambiguous is dropped at extraction: a dictionary that cannot name
//! one replacement cannot feed a rewrite.
//!
//! The table is `language<TAB>misspelling<TAB>correction`.

pub const MISSPELLINGS_TABLE: &str = include_str!("misspellings.tsv");

/// The misspellings a query is the correction for, which become extra variants of it.
///
/// Matching is on the correction rather than on the misspelling: the user types what they meant,
/// and the corpus is what holds the error.
pub fn variants_of(query: &str) -> Vec<String> {
    let mut found: Vec<String> = MISSPELLINGS_TABLE
        .lines()
        .filter_map(|row| {
            let mut columns = row.split('\t');
            let (_, misspelling, correction) = (columns.next()?, columns.next()?, columns.next()?);
            (correction == query).then(|| misspelling.to_string())
        })
        .collect();
    found.sort();
    found.dedup();
    found
}

/// Every language the table carries, for documentation and for `--summary`.
pub fn known_languages() -> Vec<&'static str> {
    let mut names: Vec<&str> = MISSPELLINGS_TABLE
        .lines()
        .filter_map(|row| row.split('\t').next())
        .collect();
    names.dedup();
    names
}
