//! SIMD/GPU-accelerated fuzzy substring search utility
//!
//! A grep-like tool that combines exact substring matching with bounded fuzzy
//! matching, built entirely on StringZilla's `szs` kernels:
//!
//! - `--match line` (default): **Smith-Waterman** local alignment of the needle
//!   against each line. Threshold via `--max-distance` or `--min-similarity`.
//! - `--match word`: tokenize lines and match the needle against each word using
//!   **Levenshtein** (`--cost edit`) or Smith-Waterman (matrix costs). With
//!   `--utf8`, word mode tokenizes on Unicode alphanumerics and counts edit
//!   distance in code points rather than bytes.
//!
//! Scoring is selectable: `--cost edit` (uniform), `--cost keyboard` (QWERTY key
//! proximity), `--cost phonetic` (articulatory similarity), or `--cost-matrix FILE`.
//! Custom scoring routes through Smith-Waterman, which carries the
//! `byte_to_class[256]` + `class_substitution_costs[32][32]` matrix. That matrix has
//! 32 classes for 52 letters, so Smith-Waterman folds case unconditionally and
//! `--ignore-case` changes only the exact and Levenshtein paths.
//!
//! Execution runs on the CPU multicore backend by default, or the GPU when built
//! with `--features cuda` and invoked with `--device gpu`.
//!
//! # Examples
//!
//! ```bash
//! # Substring fuzzy search, up to 1 edit (Smith-Waterman)
//! sz-fuzzy-find --max-distance 1 color file.txt
//!
//! # Keyboard-aware scoring (fat-finger typos), 80% similarity
//! sz-fuzzy-find --cost keyboard --min-similarity 0.8 color file.txt
//!
//! # Phonetic scoring (sounds-like)
//! sz-fuzzy-find --cost phonetic --min-similarity 0.8 Smith names.txt
//!
//! # Word mode: needle vs each token via Levenshtein
//! sz-fuzzy-find --match word --max-distance 1 colour file.txt
//!
//! # Run on the GPU (requires: cargo build --features cuda)
//! sz-fuzzy-find --device gpu --max-distance 1 needle big.txt
//! ```

use std::borrow::Cow;
use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz;
use stringzilla::szs::{
    DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8, SmithWatermanScores,
};

use shared::*;

// region: Scoring Matrices

/// Diagonal (self) score. Uniform across matrices so `--min-similarity` normalizes
/// consistently: the maximum score a needle of N bytes can earn is `MATCH * N`.
const MATCH: i8 = 8;
const MATCH_I: isize = MATCH as isize;

/// Class indices beyond the 26 letters.
const CLASS_DIGIT: usize = 26;
const CLASS_OTHER: usize = 27;
const CLASS_SPACE: usize = 28;

/// A fully-built scoring scheme: byte→class map, 32×32 class scores, affine gaps.
struct Scheme {
    byte_to_class: [u8; 256],
    costs: [[i8; 32]; 32],
    gap_open: i8,
    gap_extend: i8,
}

/// Shared class assignment: `a-z`/`A-Z` → 0..25 (case-folded), digit → 26,
/// whitespace → 28, everything else → 27. Letters fold case because 26 letter
/// classes already nearly fill the 32-class budget, so Smith-Waterman fuzzy
/// matching is case-insensitive by construction.
fn default_byte_to_class() -> [u8; 256] {
    let mut map = [CLASS_OTHER as u8; 256];
    for b in b'a'..=b'z' {
        map[b as usize] = b - b'a';
    }
    for b in b'A'..=b'Z' {
        map[b as usize] = b - b'A';
    }
    for b in b'0'..=b'9' {
        map[b as usize] = CLASS_DIGIT as u8;
    }
    for &b in &[b' ', b'\t', b'\r', b'\n', 0x0b, 0x0c] {
        map[b as usize] = CLASS_SPACE as u8;
    }
    map
}

#[inline]
fn letter(c: u8) -> usize {
    (c - b'a') as usize
}

fn diagonal_only() -> [[i8; 32]; 32] {
    let mut m = [[0i8; 32]; 32];
    for (class, row) in m.iter_mut().enumerate() {
        row[class] = MATCH;
    }
    m
}

/// `edit`: uniform — diagonal +MATCH, all substitutions 0. With gaps ≈ −MATCH this
/// behaves like substitution/indel-counting edit distance under the `--max-distance` threshold.
fn edit_scheme() -> Scheme {
    Scheme {
        byte_to_class: default_byte_to_class(),
        costs: diagonal_only(),
        gap_open: -MATCH,
        gap_extend: -MATCH,
    }
}

/// Staggered-QWERTY `(x, y)` coordinates per letter class (0=a .. 25=z).
fn keyboard_coords() -> [(f32, f32); 26] {
    let mut c = [(0.0f32, 0.0f32); 26];
    let rows: [(&[u8], f32, f32); 3] = [
        (b"qwertyuiop", 0.0, 0.0),
        (b"asdfghjkl", 0.25, 1.0),
        (b"zxcvbnm", 0.75, 2.0),
    ];
    for (letters, x_off, y) in rows {
        for (i, &ch) in letters.iter().enumerate() {
            c[letter(ch)] = (x_off + i as f32, y);
        }
    }
    c
}

/// `keyboard`: substitution score falls off with Euclidean key distance.
/// self +8, orthogonal neighbor +4, diagonal neighbor +2, two away 0, far −4.
fn keyboard_scheme() -> Scheme {
    let coords = keyboard_coords();
    let mut m = diagonal_only();
    for a in 0..26 {
        for b in 0..26 {
            if a == b {
                continue;
            }
            let (ax, ay) = coords[a];
            let (bx, by) = coords[b];
            let dist = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
            let score = (MATCH as f32 - 4.0 * dist).round().clamp(-8.0, 8.0);
            m[a][b] = score as i8;
        }
    }
    penalize_nonletters(&mut m);
    Scheme {
        byte_to_class: default_byte_to_class(),
        costs: m,
        gap_open: -6,
        gap_extend: -2,
    }
}

/// `phonetic`: articulatory similarity, seeded by Editex (Zobel & Dart) groups.
fn phonetic_scheme() -> Scheme {
    let mut m = diagonal_only();
    let mut set = |a: u8, b: u8, s: i8| {
        m[letter(a)][letter(b)] = s;
        m[letter(b)][letter(a)] = s;
    };
    // Voiced/unvoiced cognates (same place & manner) — strongest similarity.
    for &(a, b) in &[
        (b'b', b'p'),
        (b'd', b't'),
        (b'g', b'k'),
        (b'v', b'f'),
        (b'z', b's'),
        (b'j', b'g'),
    ] {
        set(a, b, 5);
    }
    // Same manner / Editex group.
    for &(a, b) in &[
        (b'm', b'n'),
        (b'l', b'r'),
        (b'c', b'k'),
        (b'c', b'q'),
        (b'k', b'q'),
        (b's', b'x'),
        (b'x', b'z'),
        (b's', b'z'),
    ] {
        set(a, b, 4);
    }
    // Vowels are highly interchangeable.
    let vowels = [b'a', b'e', b'i', b'o', b'u', b'y'];
    for i in 0..vowels.len() {
        for j in (i + 1)..vowels.len() {
            set(vowels[i], vowels[j], 2);
        }
    }
    // Remaining consonant↔consonant pairs are dissimilar; vowels↔consonants and the
    // near-silent h/w stay neutral (0). h/w lean on cheap gaps instead.
    let is_vowel = |c: u8| vowels.contains(&c);
    let silent = |c: u8| c == b'h' || c == b'w';
    for a in b'a'..=b'z' {
        for b in (a + 1)..=b'z' {
            let (ai, bi) = (letter(a), letter(b));
            if m[ai][bi] != 0 || silent(a) || silent(b) || is_vowel(a) || is_vowel(b) {
                continue;
            }
            m[ai][bi] = -2;
            m[bi][ai] = -2;
        }
    }
    penalize_nonletters(&mut m);
    Scheme {
        byte_to_class: default_byte_to_class(),
        costs: m,
        gap_open: -6,
        gap_extend: -2,
    }
}

/// Letters substituting with digits/other/whitespace classes are clearly wrong.
fn penalize_nonletters(m: &mut [[i8; 32]; 32]) {
    for &c in &[CLASS_DIGIT, CLASS_OTHER, CLASS_SPACE] {
        m[c][..26].fill(-4);
    }
    for row in m.iter_mut().take(26) {
        for &c in &[CLASS_DIGIT, CLASS_OTHER, CLASS_SPACE] {
            row[c] = -4;
        }
    }
}

/// Parse a `--cost-matrix FILE`: 256 whitespace-separated class ids, then 32×32
/// whitespace-separated i8 scores (row-major), then optional `gap_open gap_extend`.
fn load_custom_scheme(path: &str) -> io::Result<Scheme> {
    let text = std::fs::read_to_string(path)?;
    let nums: Vec<i64> = text
        .split_whitespace()
        .map(|t| t.parse::<i64>())
        .collect::<Result<_, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}", e)))?;
    if nums.len() < 256 + 32 * 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "cost-matrix needs 256 class ids + 32x32 scores",
        ));
    }
    let mut byte_to_class = [0u8; 256];
    for i in 0..256 {
        byte_to_class[i] = (nums[i] as u8) & 31;
    }
    let mut costs = [[0i8; 32]; 32];
    for r in 0..32 {
        for c in 0..32 {
            costs[r][c] = nums[256 + r * 32 + c] as i8;
        }
    }
    let (gap_open, gap_extend) = if nums.len() >= 256 + 32 * 32 + 2 {
        (nums[256 + 1024] as i8, nums[256 + 1025] as i8)
    } else {
        (-6, -2)
    };
    Ok(Scheme {
        byte_to_class,
        costs,
        gap_open,
        gap_extend,
    })
}

fn build_scheme(cost: Cost, custom_path: Option<&str>) -> Result<Scheme, Failure> {
    match custom_path {
        Some(path) => load_custom_scheme(path).at(path),
        None => Ok(match cost {
            Cost::Edit => edit_scheme(),
            Cost::Keyboard => keyboard_scheme(),
            Cost::Phonetic => phonetic_scheme(),
        }),
    }
}

// endregion: Scoring Matrices

// region: Tokenizing

/// Tokenize a line into maximal runs of ASCII alphanumerics (word matching).
fn tokenize(line: &[u8]) -> impl Iterator<Item = &[u8]> {
    line.split(|byte: &u8| !byte.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
}

/// Tokenize valid UTF-8 into maximal runs of Unicode alphanumerics.
fn tokenize_utf8(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

// endregion: Tokenizing

// region: Queries

/// One prepared query needle, carrying the state every filter needs so that no
/// filter has to re-derive anything per line or per token.
struct Query {
    bytes: Vec<u8>,
    /// Which ASCII letters occur, folded; bit 0 is `a`.
    letters: u32,
    /// Code points, which is what the UTF-8 kernel counts edits in.
    chars: usize,
}

impl Query {
    fn new(pattern: String, ignore_case: bool) -> Self {
        let pattern = if ignore_case {
            pattern.to_lowercase()
        } else {
            pattern
        };
        let chars = pattern.chars().count();
        let bytes = pattern.into_bytes();
        let letters = letter_set(&bytes);
        Self {
            bytes,
            letters,
            chars,
        }
    }

    /// The length the active kernel measures edits against.
    fn length(&self, utf8: bool) -> usize {
        if utf8 {
            self.chars
        } else {
            self.bytes.len()
        }
    }
}

/// Bitmap of the ASCII letters present, case-folded. Non-letters are ignored, so
/// the set is a lower bound on what a match must share.
fn letter_set(text: &[u8]) -> u32 {
    let mut set = 0u32;
    for &byte in text {
        if byte.is_ascii_alphabetic() {
            set |= 1 << (byte.to_ascii_lowercase() - b'a');
        }
    }
    set
}

// endregion: Queries

// region: Filtering

/// Whether a token of `length` can be within `max_distance` edits of the query.
///
/// Levenshtein moves one character at a time, so the lengths cannot differ by more
/// than the budget. Sound: never rejects a true match.
#[inline]
fn accepts_length(query: &Query, length: usize, max_distance: usize, utf8: bool) -> bool {
    query.length(utf8).abs_diff(length) <= max_distance
}

/// Whether a token sharing `letters` can be within `max_distance` edits.
///
/// One substitution can drop a letter from one side and add one to the other, so a
/// symmetric difference of `d` letters needs at least `d / 2` edits. Sound, and two
/// instructions once the query's set is precomputed.
#[inline]
fn accepts_letters(query: &Query, letters: u32, max_distance: usize) -> bool {
    (query.letters ^ letters).count_ones() as usize <= 2 * max_distance
}

/// Lines still needing the fuzzy kernel, having survived cheap rejection.
///
/// Only exact matches are rejected today, so every remaining line is a candidate.
/// The partition filter belongs here: splitting the needle into `k + 1` pieces, a
/// line holding none of them cannot be within `k` edits, because each edit can
/// damage at most one piece. That bound covers substitutions, insertions, and
/// deletions, but not transpositions, which touch two adjacent positions and so
/// need `2k + 1` pieces.
fn select_candidates(lines: &[&[u8]], matched: &[bool], candidates: &mut Vec<usize>) {
    candidates.clear();
    candidates.extend((0..lines.len()).filter(|&index| !matched[index]));
}

/// Both cheap bounds, in increasing cost order. Byte length stands in for code
/// points outside UTF-8 mode, where the kernel counts bytes anyway.
#[inline]
fn survives(query: &Query, token: &[u8], max_distance: usize, utf8: bool) -> bool {
    let length = if utf8 {
        sz::count_utf8(token)
    } else {
        token.len()
    };
    accepts_length(query, length, max_distance, utf8)
        && accepts_letters(query, letter_set(token), max_distance)
}

// endregion: Filtering

// region: Matching

/// Flag a line when any of its tokens is accepted.
///
/// Tokens arrive grouped by line, so the grouping is `runs`, one entry per line
/// holding its owning line and the token index one past its last. Recording an
/// owner per token instead would cost one `usize` per token across the file.
fn mark_lines_by_run<T: Copy>(
    row: &[T],
    runs: &[(usize, usize)],
    matched: &mut [bool],
    accept: impl Fn(T) -> bool,
) {
    let mut start = 0;
    for &(line, end) in runs {
        if row[start..end].iter().any(|&value| accept(value)) {
            matched[line] = true;
        }
        start = end;
    }
}

/// Does `line` contain `needle` exactly (case-folded when `ignore_case`)?
#[inline]
fn exact_contains(line: &[u8], needle: &[u8], ignore_case: bool) -> bool {
    if ignore_case {
        sz::utf8_uncased_search(line, needle).is_some()
    } else {
        sz::find(line, needle).is_some()
    }
}

/// Minimum Smith-Waterman score for a needle of `len` bytes to count as a match.
/// `--min-similarity s` ⇒ `s · MATCH · len`; otherwise `--max-distance k` ⇒ `(len − k) · MATCH`.
fn score_threshold(len: usize, min_similarity: Option<f64>, k: usize) -> isize {
    match min_similarity {
        Some(s) => (s * MATCH_I as f64 * len as f64).ceil() as isize,
        None => (len as isize - k as isize) * MATCH_I,
    }
}

/// Configuration resolved once from CLI args.
struct MatchConfig {
    ignore_case: bool,
    word: bool,
    cost_is_edit: bool,
    max_distance: usize,
    min_similarity: Option<f64>,
    utf8: bool,
}

impl MatchConfig {
    /// Edit budget for one query, so `--min-similarity` reaches the Levenshtein path
    /// rather than being ignored there.
    fn budget(&self, query: &Query) -> usize {
        match self.min_similarity {
            Some(similarity) => {
                ((1.0 - similarity) * query.length(self.utf8) as f64).floor() as usize
            }
            None => self.max_distance,
        }
    }
}

/// Lowercase a token when folding, borrowing it otherwise.
fn fold(token: &str, ignore_case: bool) -> Cow<'_, [u8]> {
    if ignore_case {
        Cow::Owned(token.to_lowercase().into_bytes())
    } else {
        Cow::Borrowed(token.as_bytes())
    }
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig {
    line_numbers: bool,
    count: bool,
    /// Prefix each record with its file name, as grep does for multiple inputs.
    prefix: bool,
    json: bool,
    summary: bool,
    terminator: Terminator,
}

/// The kernels, built once for the whole run.
struct Engines {
    device: DeviceScope,
    sw: SmithWatermanScores,
    lev: LevenshteinDistances,
    lev_utf8: LevenshteinDistancesUtf8,
}

/// One input's lines, which of them matched, and how many word-mode kernels had to
/// skip because `--utf8` was promised and the line was not well-formed.
struct Searched<'a> {
    lines: Vec<&'a [u8]>,
    matched: Vec<bool>,
    malformed: usize,
}

/// Search one input's bytes; returns the lines and per-line match flags.
fn search_lines<'a>(
    data: &'a [u8],
    queries: &[Query],
    eng: &Engines,
    cfg: &MatchConfig,
) -> Searched<'a> {
    let lines: Vec<&[u8]> = LineIter::new(data, Newlines::from_utf8(cfg.utf8)).collect();
    let mut matched = vec![false; lines.len()];
    let malformed = if cfg.utf8 && cfg.word {
        lines
            .iter()
            .filter(|line| std::str::from_utf8(line).is_err())
            .count()
    } else {
        0
    };

    // Reused across queries: every one of these is sized by the line count, and
    // reallocating them per query is pure churn.
    let mut candidates: Vec<usize> = Vec::new();
    let mut haystacks: Vec<&[u8]> = Vec::new();
    let mut token_runs: Vec<(usize, usize)> = Vec::new();
    let mut tokens: Vec<Cow<'a, [u8]>> = Vec::new();

    for q in queries {
        let needle = q.bytes.as_slice();
        let len = needle.len();
        if len == 0 {
            continue;
        }
        let thr = score_threshold(len, cfg.min_similarity, cfg.max_distance);
        let budget = cfg.budget(q);

        // Exact-first: cheap, and it covers the distance-0 case.
        for (i, line) in lines.iter().enumerate() {
            if !matched[i] && exact_contains(line, needle, cfg.ignore_case) {
                matched[i] = true;
            }
        }

        select_candidates(&lines, &matched, &mut candidates);
        if candidates.is_empty() {
            continue;
        }

        // One query row (the needle) against many candidate columns; `compute`
        // returns a 1×N matrix, so `.row(0)` is the per-candidate score/distance.
        let query = [needle];

        if cfg.word {
            // Gather every token of every candidate, remembering its line. UTF-8
            // mode tokenizes on Unicode alphanumerics; `--utf8` promises
            // well-formed input, so malformed lines only match via exact search.
            // The length and letter bounds are Levenshtein properties. A score floor
            // with class costs admits matches they would reject, so they apply only
            // in edit mode; elsewhere every token reaches the kernel as before.
            let filtering = cfg.cost_is_edit;
            token_runs.clear();
            tokens.clear();
            for &index in &candidates {
                if cfg.utf8 {
                    if let Ok(text) = std::str::from_utf8(lines[index]) {
                        tokens.extend(
                            tokenize_utf8(text)
                                .map(|token| fold(token, cfg.ignore_case))
                                .filter(|token| !filtering || survives(q, token, budget, true)),
                        );
                    }
                } else {
                    tokens.extend(
                        tokenize(lines[index])
                            .filter(|token| !filtering || survives(q, token, budget, false))
                            .map(Cow::Borrowed),
                    );
                }
                token_runs.push((index, tokens.len()));
            }
            if tokens.is_empty() {
                continue;
            }
            let views: Vec<&[u8]> = tokens.iter().map(|token| token.as_ref()).collect();
            if !cfg.cost_is_edit {
                let scores = eng
                    .sw
                    .compute(&eng.device, &query[..], &views)
                    .expect("smith-waterman compute failed");
                mark_lines_by_run(scores.row(0), &token_runs, &mut matched, |score| {
                    score >= thr
                });
            } else if cfg.utf8 {
                // Code-point-level distances; both sides validated above.
                let needle = std::str::from_utf8(needle).expect("patterns are UTF-8 arguments");
                let views: Vec<&str> = views
                    .iter()
                    .map(|t| std::str::from_utf8(t).expect("tokens cut from validated lines"))
                    .collect();
                let dists = eng
                    .lev_utf8
                    .compute(&eng.device, &[needle][..], &views[..])
                    .expect("utf8 levenshtein compute failed");
                mark_lines_by_run(dists.row(0), &token_runs, &mut matched, |distance| {
                    distance <= budget
                });
            } else {
                let dists = eng
                    .lev
                    .compute(&eng.device, &query[..], &views)
                    .expect("levenshtein compute failed");
                mark_lines_by_run(dists.row(0), &token_runs, &mut matched, |distance| {
                    distance <= budget
                });
            }
        } else {
            // Substring: best local alignment of the needle within each line.
            haystacks.clear();
            haystacks.extend(candidates.iter().map(|&index| lines[index]));
            // One line per result, so every run holds a single entry.
            token_runs.clear();
            token_runs.extend(
                candidates
                    .iter()
                    .enumerate()
                    .map(|(position, &line)| (line, position + 1)),
            );
            let scores = eng
                .sw
                .compute(&eng.device, &query[..], &haystacks)
                .expect("smith-waterman compute failed");
            mark_lines_by_run(scores.row(0), &token_runs, &mut matched, |score| {
                score >= thr
            });
        }
    }

    Searched {
        lines,
        matched,
        malformed,
    }
}

// endregion: Matching

// region: CLI

/// What the needle is matched against
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Match {
    Line,
    Word,
}

/// One column a record can carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Field {
    /// The 1-based line number
    LineNumbers,
}

/// Which record kind to emit
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Show {
    Lines,
    Count,
}

/// How records are rendered
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    Text,
    Json,
}

/// Which scoring model scores a substitution
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Cost {
    Edit,
    Keyboard,
    Phonetic,
}

/// Where the kernels run
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Device {
    Auto,
    Cpu,
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

    /// Maximum edit distance, in code points under --utf8 [default: 1]
    #[arg(long)]
    max_distance: Option<usize>,

    /// Minimum normalized similarity; 1.0 is exact
    #[arg(long, conflicts_with = "max_distance", value_parser = parse_similarity)]
    min_similarity: Option<f64>,

    /// Which scoring model scores a substitution
    #[arg(long, value_enum, conflicts_with = "cost_matrix")]
    cost: Option<Cost>,

    /// Load a scoring matrix from FILE instead of a built-in model
    #[arg(long)]
    cost_matrix: Option<String>,

    /// What the needle is matched against
    #[arg(long = "match", value_enum, value_name = "MATCH")]
    match_against: Option<Match>,

    /// Where the kernels run
    #[arg(long, value_enum)]
    device: Option<Device>,

    /// CPU thread count, where 0 is every core [default: 0]
    #[arg(long)]
    threads: Option<usize>,

    /// GPU device index [default: 0]
    #[arg(long, requires = "device")]
    gpu_id: Option<usize>,

    /// Case-insensitive search with full Unicode folding; implies --utf8
    #[arg(long)]
    ignore_case: bool,

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

    /// Print one line about the whole run, on stderr
    #[arg(long)]
    summary: bool,

    /// How records are rendered
    #[arg(long, value_enum, help_heading = "Output Formats")]
    format: Option<Format>,

    /// NUL-terminate each output record instead of newline
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any match was found, 1 otherwise
    #[arg(long, conflicts_with_all = ["show", "format", "null", "fields", "summary"], help_heading = "Output Formats")]
    quiet: bool,
}

/// A similarity outside 0..=1 is silently useless: `2.0` matches nothing, `-1.0` and
/// `nan` match everything.
fn parse_similarity(text: &str) -> Result<f64, String> {
    let value: f64 = text
        .parse()
        .map_err(|_| format!("`{}` is not a number", text))?;
    if !(0.0..=1.0).contains(&value) {
        return Err(format!("`{}` is outside 0.0..=1.0", text));
    }
    Ok(value)
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

    let cost = args.cost.unwrap_or(Cost::Edit);
    let cost_is_edit = args.cost_matrix.is_none() && cost == Cost::Edit;

    let device = build_device(
        args.device.unwrap_or(Device::Auto),
        args.threads,
        args.gpu_id,
    )
    .map_err(|message| reject(&message))?;
    let scheme = build_scheme(cost, args.cost_matrix.as_deref())?;
    let sw = SmithWatermanScores::new(
        &device,
        &scheme.byte_to_class,
        &scheme.costs,
        scheme.gap_open,
        scheme.gap_extend,
    )
    .map_err(|error| engine_failure("Smith-Waterman", error))?;

    // Standard unit-cost Levenshtein for word edit mode, byte- and code-point-level.
    let lev = LevenshteinDistances::new(&device, 0, 1, 1, 1)
        .map_err(|error| engine_failure("Levenshtein", error))?;
    let lev_utf8 = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1)
        .map_err(|error| engine_failure("UTF-8 Levenshtein", error))?;

    let engines = Engines {
        device,
        sw,
        lev,
        lev_utf8,
    };

    let (patterns, inputs) =
        resolve_positionals(args.pattern.as_deref(), &args.extra, &args.inputs)
            .map_err(|message| reject(&message))?;
    let queries: Vec<Query> = patterns
        .into_iter()
        .map(|pattern| Query::new(pattern, args.ignore_case))
        .collect();

    let cfg = MatchConfig {
        ignore_case: args.ignore_case,
        word: args.match_against == Some(Match::Word),
        cost_is_edit,
        max_distance: args.max_distance.unwrap_or(1),
        min_similarity: args.min_similarity,
        utf8: args.utf8 || args.ignore_case,
    };
    let output_config = OutputConfig {
        line_numbers: args.fields.contains(&Field::LineNumbers),
        count: args.show == Some(Show::Count),
        prefix: inputs.len() > 1,
        json: args.format == Some(Format::Json),
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
    let outcome = search_inputs(
        writer,
        notes,
        opened,
        &queries,
        &engines,
        &cfg,
        &output_config,
    )
    .at("-")?;

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
    malformed: usize,
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
    queries: &[Query],
    engines: &Engines,
    cfg: &MatchConfig,
    out_cfg: &OutputConfig,
) -> io::Result<Outcome> {
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
        let found = search_lines(input.as_bytes(), queries, engines, cfg);
        outcome.malformed += found.malformed;

        let mut count = 0usize;
        for (index, line) in found.lines.iter().enumerate() {
            if !found.matched[index] {
                continue;
            }
            count += 1;
            if !out_cfg.count {
                write_match(output, out_cfg, path, index + 1, line)?;
            }
        }
        if out_cfg.count {
            write_count(output, out_cfg, path, count)?;
        }
        outcome.total += count;
    }

    if out_cfg.summary {
        write_summary(output, notes, out_cfg, &outcome, seen)?;
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
    if cfg.json {
        return writeln!(
            output,
            r#"{{"type":"summary","data":{{"matched_lines":{},"readable_inputs":{},"total_inputs":{},"malformed_lines":{}}}}}"#,
            outcome.total, outcome.readable, inputs, outcome.malformed
        );
    }
    writeln!(
        notes,
        "matched {} lines in {} of {} inputs, skipping {} malformed lines",
        outcome.total, outcome.readable, inputs, outcome.malformed
    )
}

/// One count record per input, in whichever format the run selected.
fn write_count(
    output: &mut dyn Write,
    cfg: &OutputConfig,
    path: &str,
    count: usize,
) -> io::Result<()> {
    if cfg.json {
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

fn write_match(
    output: &mut dyn Write,
    cfg: &OutputConfig,
    path: &str,
    line_no: usize,
    line: &[u8],
) -> io::Result<()> {
    if cfg.json {
        // Ripgrep's schema, minus `submatches`: fuzzy matching has no exact span.
        output.write_all(br#"{"type":"match","data":{"path":"#)?;
        json_text_field_to(output, path.as_bytes())?;
        output.write_all(br#","lines":"#)?;
        json_text_field_to(output, line)?;
        write!(output, r#","line_number":{},"submatches":[]}}}}"#, line_no)?;
        return output.write_all(b"\n");
    }
    if cfg.prefix {
        write!(output, "{}:", path)?;
    }
    if cfg.line_numbers {
        write!(output, "{}:", line_no)?;
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
                "max-distance",
                "min-similarity",
                "cost",
                "cost-matrix",
                "match",
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
            ["sz-fuzzy-find", "--min-similarity", "2.0", "a"].as_slice(),
            ["sz-fuzzy-find", "--min-similarity", "nan", "a"].as_slice(),
            ["sz-fuzzy-find", "--device", "banana", "a"].as_slice(),
            ["sz-fuzzy-find", "--device", "GPU", "a"].as_slice(),
            ["sz-fuzzy-find", "--gpu-id", "1", "a"].as_slice(),
            ["sz-fuzzy-find", "--cost", "edit", "--cost-matrix", "f", "a"].as_slice(),
            [
                "sz-fuzzy-find",
                "--min-similarity",
                "0.5",
                "--max-distance",
                "1",
                "a",
            ]
            .as_slice(),
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
        assert_eq!(placeholder("match_against").as_deref(), Some("MATCH"));
    }

    #[test]
    fn closes_a_json_stream_with_its_summary() {
        // `--summary` used to append a prose line after the JSON records.
        let outcome = Outcome {
            total: 9,
            readable: 1,
            malformed: 0,
        };
        let mut cfg = OutputConfig {
            line_numbers: false,
            count: false,
            prefix: false,
            json: true,
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
        cfg.json = false;
        let (mut written, mut notes) = (Vec::new(), Vec::new());
        write_summary(&mut written, &mut notes, &cfg, &outcome, 1).unwrap();
        assert!(written.is_empty(), "prose is not a match");
        assert!(String::from_utf8(notes)
            .unwrap()
            .starts_with("matched 9 lines"));
    }

    #[test]
    fn keeps_matches_when_one_input_is_missing() {
        let engines = engines_for(edit_scheme());
        let cfg = MatchConfig {
            ignore_case: false,
            word: false,
            cost_is_edit: true,
            max_distance: 1,
            min_similarity: None,
            utf8: false,
        };
        let out_cfg = OutputConfig {
            line_numbers: false,
            count: false,
            prefix: false,
            json: false,
            summary: false,
            terminator: Terminator::Newline,
        };
        let inputs = [
            ("missing.txt", Err(io::Error::from(io::ErrorKind::NotFound))),
            ("present.txt", Ok(InputSource::Buffer(b"colour\n".to_vec()))),
        ];
        let queries = vec![Query::new("color".to_string(), false)];

        let mut written = Vec::new();
        let outcome = search_inputs(
            &mut written,
            &mut io::sink(),
            inputs,
            &queries,
            &engines,
            &cfg,
            &out_cfg,
        )
        .unwrap();
        assert_eq!(outcome.readable, 1);
        assert_eq!(outcome.total, 1);
        assert_eq!(written, b"colour\n");
    }

    #[test]
    fn folds_case_on_the_levenshtein_path() {
        // `--match word --cost edit --ignore-case` used to compare unfolded tokens.
        let engines = engines_for(edit_scheme());
        let cfg = |ignore_case: bool| MatchConfig {
            ignore_case,
            word: true,
            cost_is_edit: true,
            max_distance: 0,
            min_similarity: None,
            utf8: true,
        };
        let data = "COLOUR\n".as_bytes();
        let folded = vec![Query::new("colour".to_string(), true)];
        let literal = vec![Query::new("colour".to_string(), false)];
        assert!(search_lines(data, &folded, &engines, &cfg(true)).matched[0]);
        assert!(!search_lines(data, &literal, &engines, &cfg(false)).matched[0]);
    }

    #[test]
    fn counts_lines_skipped_as_malformed_under_utf8() {
        let engines = engines_for(edit_scheme());
        let cfg = MatchConfig {
            ignore_case: false,
            word: true,
            cost_is_edit: true,
            max_distance: 1,
            min_similarity: None,
            utf8: true,
        };
        let data = b"colour\n\xff\xfe bad\n";
        let queries = vec![Query::new("color".to_string(), false)];
        assert_eq!(search_lines(data, &queries, &engines, &cfg).malformed, 1);
    }

    #[test]
    fn spends_min_similarity_as_an_edit_budget() {
        // In word+edit mode `--min-similarity` used to be dropped entirely.
        let cfg = MatchConfig {
            ignore_case: false,
            word: true,
            cost_is_edit: true,
            max_distance: 9,
            min_similarity: Some(0.8),
            utf8: false,
        };
        let query = Query::new("colour".to_string(), false);
        assert_eq!(cfg.budget(&query), 1);
    }

    fn engines_for(scheme: Scheme) -> Engines {
        let device = cpu();
        let sw = SmithWatermanScores::new(
            &device,
            &scheme.byte_to_class,
            &scheme.costs,
            scheme.gap_open,
            scheme.gap_extend,
        )
        .unwrap();
        let lev = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
        let lev_utf8 = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
        Engines {
            device,
            sw,
            lev,
            lev_utf8,
        }
    }

    #[test]
    fn scores_keyboard_neighbors_above_distant_keys() {
        let s = keyboard_scheme();
        let near = s.costs[letter(b't')][letter(b'y')]; // adjacent
        let far = s.costs[letter(b't')][letter(b'p')]; // far
        assert!(near > far, "adjacent {} should beat distant {}", near, far);
        assert_eq!(s.costs[letter(b't')][letter(b't')], MATCH);
    }

    #[test]
    fn scores_phonetic_cognates_above_unrelated() {
        let s = phonetic_scheme();
        let cognate = s.costs[letter(b'b')][letter(b'p')]; // voiced/unvoiced pair
        let unrelated = s.costs[letter(b'b')][letter(b'z')];
        assert!(cognate > unrelated);
        assert_eq!(s.costs[letter(b's')][letter(b'z')], 4);
    }

    #[test]
    fn folds_case_in_byte_to_class_map() {
        let map = default_byte_to_class();
        assert_eq!(map[b'A' as usize], map[b'a' as usize]);
        assert_eq!(map[b'Z' as usize], map[b'z' as usize]);
        assert_eq!(map[b'5' as usize] as usize, CLASS_DIGIT);
        assert_eq!(map[b' ' as usize] as usize, CLASS_SPACE);
    }

    #[test]
    fn tokenizes_on_punctuation() {
        let tokens: Vec<&[u8]> = tokenize(b"the colour, red!").collect();
        assert_eq!(tokens, vec![b"the".as_slice(), b"colour", b"red"]);
    }

    #[test]
    fn tokenizes_unicode_words_whole() {
        let tokens: Vec<&str> = tokenize_utf8("café, naïve! 42").collect();
        assert_eq!(tokens, vec!["café", "naïve", "42"]);
        // The byte tokenizer stops at the first non-ASCII byte instead.
        let ascii: Vec<&[u8]> = tokenize("café".as_bytes()).collect();
        assert_eq!(ascii, vec![b"caf".as_slice()]);
    }

    #[test]
    fn computes_score_threshold_from_length_and_ratio() {
        assert_eq!(score_threshold(5, None, 1), 32); // (5-1)*8
        assert_eq!(score_threshold(5, Some(0.8), 1), 32); // ceil(0.8*8*5)
    }

    fn cpu() -> DeviceScope {
        DeviceScope::cpu_cores(1).expect("cpu device")
    }

    #[test]
    fn scores_fuzzy_substring_via_smith_waterman() {
        let s = edit_scheme();
        let device = cpu();
        let sw = SmithWatermanScores::new(
            &device,
            &s.byte_to_class,
            &s.costs,
            s.gap_open,
            s.gap_extend,
        )
        .unwrap();
        let query = vec![b"color".as_slice()];
        let lines = vec![b"the colour red".as_slice(), b"nothing here".as_slice()];
        let scores = sw.compute(&device, &query, &lines).unwrap();
        // "colour" is within ~1 edit of "color"; threshold for k=1 is (5-1)*8 = 32.
        assert!(
            scores[(0, 0)] >= 32,
            "colour score {} should pass",
            scores[(0, 0)]
        );
        assert!(
            scores[(0, 1)] < 32,
            "unrelated score {} should fail",
            scores[(0, 1)]
        );
    }

    #[test]
    fn measures_word_levenshtein_distance() {
        let device = cpu();
        let lev = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
        let query = vec![b"colour".as_slice()];
        let tokens = vec![b"color".as_slice(), b"zzzzz".as_slice()];
        let dists = lev.compute(&device, &query, &tokens).unwrap();
        assert_eq!(dists[(0, 0)], 1); // colour -> color is one deletion
        assert!(dists[(0, 1)] >= 4);
    }

    #[test]
    fn counts_code_points_not_bytes_in_utf8_word_mode() {
        let device = cpu();
        let lev_utf8 = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
        let dists = lev_utf8.compute(&device, &["naïve"], &["naive"]).unwrap();
        assert_eq!(dists[(0, 0)], 1); // ï -> i is one substitution, not two byte edits

        let lev = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
        let byte_dists = lev
            .compute(&device, &["naïve".as_bytes()], &[b"naive".as_slice()])
            .unwrap();
        assert_eq!(byte_dists[(0, 0)], 2);
    }
}

// endregion: Tests
