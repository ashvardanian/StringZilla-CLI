//! Physical key adjacency, derived from xkeyboard-config.
//!
//! Source: <https://gitlab.freedesktop.org/xkeyboard-config/xkeyboard-config>, read from the
//! system's `/usr/share/X11/xkb/symbols` at extraction time. Licensed MIT/X11-style, compatible
//! with this crate's Apache-2.0.
//!
//! Everything about this table lives here: the raw `keyboards.tsv` beside this file, the parser the
//! binaries use, and the extractor `sz-tables` calls to regenerate it.
//!
//! XKB names keys by position rather than by legend - `AE01` through `AB10` name a row and a column
//! - so one adjacency graph over positions serves every layout and a layout is only a permutation of
//! characters onto keys. That is why a table this small covers Latin, Cyrillic, Greek, Hebrew and
//! Arabic alike.

use std::collections::{BTreeMap, BTreeSet};

/// Physical key adjacency for every layout the extractor below could resolve, as
/// `layout<TAB>character<TAB>neighbours` rows.
///
/// XKB names keys by position rather than by legend, so one adjacency graph serves every layout and
/// a layout is only a permutation of characters onto keys. That is why a table this small covers
/// Latin, Cyrillic, Greek, Hebrew and Arabic alike.
pub const KEYBOARDS_TABLE: &str = include_str!("keyboards.tsv");

/// Which keys sit next to which, for one layout.
#[derive(Default)]
pub struct Keyboard {
    neighbours: BTreeMap<char, Vec<char>>,
}

impl Keyboard {
    /// Read one layout out of the embedded table. Unknown names yield an empty keyboard, which the
    /// caller reports rather than silently treating as "no neighbours".
    pub fn load(layout: &str) -> Self {
        let mut neighbours = BTreeMap::new();
        for row in KEYBOARDS_TABLE.lines() {
            let mut columns = row.split('\t');
            let (Some(name), Some(character), Some(touching)) =
                (columns.next(), columns.next(), columns.next())
            else {
                continue;
            };
            if name != layout {
                continue;
            }
            if let Some(character) = character.chars().next() {
                neighbours.insert(character, touching.chars().collect());
            }
        }
        Self { neighbours }
    }

    /// Every layout the table carries, for `--layout`'s error message and for detection.
    pub fn known_layouts() -> Vec<&'static str> {
        let mut names: Vec<&str> = KEYBOARDS_TABLE
            .lines()
            .filter_map(|row| row.split('\t').next())
            .collect();
        names.dedup();
        names
    }

    /// Pick the layout whose alphabet covers most of `pattern`, so a Cyrillic needle reaches a
    /// Cyrillic keyboard without the caller naming one. Ties keep the earlier name, and `us` is
    /// tried first so a plain ASCII needle never drifts onto a lookalike layout.
    pub fn detect(pattern: &str) -> String {
        let wanted: BTreeSet<char> = pattern
            .to_lowercase()
            .chars()
            .filter(|character| character.is_alphabetic())
            .collect();
        let coverage = |layout: &str| {
            Keyboard::load(layout)
                .neighbours
                .keys()
                .filter(|character| wanted.contains(character))
                .count()
        };

        // `us` is the baseline rather than a zero, so every layout that merely ties it - which is
        // every Latin layout on an ASCII needle - leaves the choice alone. Only a script `us` cannot
        // type at all, such as Cyrillic or Greek, covers strictly more and takes over.
        let mut best = ("us".to_string(), coverage("us"));
        for layout in Keyboard::known_layouts() {
            let covered = coverage(layout);
            if covered > best.1 {
                best = (layout.to_string(), covered);
            }
        }
        best.0
    }

    pub fn is_empty(&self) -> bool {
        self.neighbours.is_empty()
    }

    pub fn near(&self, character: char) -> &[char] {
        self.neighbours
            .get(&character)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Every character the layout carries, which is the substitution alphabet under `--cost edit`.
    pub fn alphabet(&self) -> Vec<char> {
        self.neighbours.keys().copied().collect()
    }
}
