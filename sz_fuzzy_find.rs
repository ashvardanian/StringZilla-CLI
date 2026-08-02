//! SIMD/GPU-accelerated fuzzy substring search utility
//!
//! A grep-like tool that combines exact substring matching with bounded fuzzy
//! matching, built entirely on StringZilla's `szs` kernels:
//!
//! - Default (substring): **Smith-Waterman** local alignment of the needle against
//!   each line. Threshold via `-k` (converted) or `--min-similarity`.
//! - `-w/--word`: tokenize lines and match the needle against each word using
//!   **Levenshtein** (`--cost edit`, integer `-k`) or Smith-Waterman (matrix costs).
//!   With `--utf8`, word mode tokenizes on Unicode alphanumerics and counts edit
//!   distance in code points rather than bytes.
//!
//! Scoring is selectable: `--cost edit` (uniform), `--cost keyboard` (QWERTY key
//! proximity), `--cost phonetic` (articulatory similarity), or `--cost-matrix FILE`.
//! Custom scoring routes through Smith-Waterman, which carries the
//! `byte_to_class[256]` + `class_substitution_costs[32][32]` matrix.
//!
//! Execution runs on the CPU multicore backend by default, or the GPU when built
//! with `--features cuda` and invoked with `--device gpu`.
//!
//! # Examples
//!
//! ```bash
//! # Substring fuzzy search, up to 1 edit (Smith-Waterman)
//! sz-fuzzy-find -k 1 color file.txt
//!
//! # Keyboard-aware scoring (fat-finger typos), 80% similarity
//! sz-fuzzy-find --cost keyboard --min-similarity 0.8 color file.txt
//!
//! # Phonetic scoring (sounds-like)
//! sz-fuzzy-find --cost phonetic --min-similarity 0.8 Smith names.txt
//!
//! # Word mode: needle vs each token via Levenshtein
//! sz-fuzzy-find -w -k 1 colour file.txt
//!
//! # Run on the GPU (requires: cargo build --features cuda)
//! sz-fuzzy-find --device gpu -k 1 needle big.txt
//! ```

use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz;
use stringzilla::szs::{
    DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8, SmithWatermanScores,
};

mod shared;
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

/// Selected scoring model.
#[derive(Clone)]
enum Cost {
    Edit,
    Keyboard,
    Phonetic,
    Custom,
}

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
    for i in 0..32 {
        m[i][i] = MATCH;
    }
    m
}

/// `edit`: uniform — diagonal +MATCH, all substitutions 0. With gaps ≈ −MATCH this
/// behaves like substitution/indel-counting edit distance under the `-k` threshold.
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
    for a in 0..26 {
        for &c in &[CLASS_DIGIT, CLASS_OTHER, CLASS_SPACE] {
            m[a][c] = -4;
            m[c][a] = -4;
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

fn build_scheme(cost: &Cost, custom_path: Option<&str>) -> io::Result<Scheme> {
    Ok(match cost {
        Cost::Edit => edit_scheme(),
        Cost::Keyboard => keyboard_scheme(),
        Cost::Phonetic => phonetic_scheme(),
        Cost::Custom => load_custom_scheme(custom_path.expect("custom requires --cost-matrix"))?,
    })
}

// endregion: Scoring Matrices

// region: Matching

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

/// Lines still needing the fuzzy kernel, having survived cheap rejection.
///
/// Only exact matches are rejected today, so every remaining line is a candidate
/// and the kernel sees all of them. A partition filter belongs here: splitting the
/// needle into `k + 1` pieces, a line holding none of them cannot be within `k`
/// substitutions, insertions, or deletions.
fn select_candidates(lines: &[&[u8]], matched: &[bool], candidates: &mut Vec<usize>) {
    candidates.clear();
    candidates.extend((0..lines.len()).filter(|&index| !matched[index]));
}

/// Flag `matched[owners[index]]` for every kernel result `row[index]` that `accept`s.
fn mark_lines<T: Copy>(
    row: &[T],
    owners: &[usize],
    matched: &mut [bool],
    accept: impl Fn(T) -> bool,
) {
    for (index, &value) in row.iter().enumerate() {
        if accept(value) {
            matched[owners[index]] = true;
        }
    }
}

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
/// `--min-similarity s` ⇒ `s · MATCH · len`; otherwise `-k` ⇒ `(len − k) · MATCH`.
fn score_threshold(len: usize, min_similarity: Option<f64>, k: usize) -> isize {
    match min_similarity {
        Some(s) => (s * MATCH_I as f64 * len as f64).ceil() as isize,
        None => (len as isize - k as isize) * MATCH_I,
    }
}

/// Configuration resolved once from CLI args.
struct Config {
    ignore_case: bool,
    word: bool,
    cost_is_edit: bool,
    max_distance: usize,
    min_similarity: Option<f64>,
    line_numbers: bool,
    count: bool,
    utf8: bool,
    prefix: bool,
    json: bool,
    quiet: bool,
    terminator: Terminator,
}

/// One prepared query needle.
struct Query {
    bytes: Vec<u8>,
}

/// The kernels, built once for the whole run.
struct Engines {
    device: DeviceScope,
    sw: SmithWatermanScores,
    lev: LevenshteinDistances,
    lev_utf8: LevenshteinDistancesUtf8,
}

/// Search one input's bytes; returns the lines and per-line match flags.
fn search_lines<'a>(
    data: &'a [u8],
    queries: &[Query],
    eng: &Engines,
    cfg: &Config,
) -> (Vec<&'a [u8]>, Vec<bool>) {
    let lines: Vec<&[u8]> = LineIter::new(data, Newlines::from_utf8(cfg.utf8)).collect();
    let mut matched = vec![false; lines.len()];

    // Reused across queries: every one of these is sized by the line count, and
    // reallocating them per query is pure churn.
    let mut candidates: Vec<usize> = Vec::new();
    let mut haystacks: Vec<&[u8]> = Vec::new();
    let mut token_runs: Vec<(usize, usize)> = Vec::new();
    let mut tokens: Vec<&[u8]> = Vec::new();

    for q in queries {
        let needle = q.bytes.as_slice();
        let len = needle.len();
        if len == 0 {
            continue;
        }
        let thr = score_threshold(len, cfg.min_similarity, cfg.max_distance);

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
            token_runs.clear();
            tokens.clear();
            for &index in &candidates {
                if cfg.utf8 {
                    let Ok(text) = std::str::from_utf8(lines[index]) else {
                        continue;
                    };
                    tokens.extend(tokenize_utf8(text).map(|token| token.as_bytes()));
                } else {
                    tokens.extend(tokenize(lines[index]));
                }
                token_runs.push((index, tokens.len()));
            }
            if tokens.is_empty() {
                continue;
            }
            if !cfg.cost_is_edit {
                let scores = eng
                    .sw
                    .compute(&eng.device, &query[..], &tokens)
                    .expect("smith-waterman compute failed");
                mark_lines_by_run(scores.row(0), &token_runs, &mut matched, |score| score >= thr);
            } else if cfg.utf8 {
                // Code-point-level distances; both sides validated above.
                let needle = std::str::from_utf8(needle).expect("patterns are UTF-8 arguments");
                let tokens: Vec<&str> = tokens
                    .iter()
                    .map(|t| std::str::from_utf8(t).expect("tokens cut from validated lines"))
                    .collect();
                let dists = eng
                    .lev_utf8
                    .compute(&eng.device, &[needle][..], &tokens[..])
                    .expect("utf8 levenshtein compute failed");
                mark_lines_by_run(dists.row(0), &token_runs, &mut matched, |distance| {
                    distance <= cfg.max_distance
                });
            } else {
                let dists = eng
                    .lev
                    .compute(&eng.device, &query[..], &tokens)
                    .expect("levenshtein compute failed");
                mark_lines_by_run(dists.row(0), &token_runs, &mut matched, |distance| {
                    distance <= cfg.max_distance
                });
            }
        } else {
            // Substring: best local alignment of the needle within each line.
            haystacks.clear();
            haystacks.extend(candidates.iter().map(|&index| lines[index]));
            let scores = eng
                .sw
                .compute(&eng.device, &query[..], &haystacks)
                .expect("smith-waterman compute failed");
            mark_lines(scores.row(0), &candidates, &mut matched, |s| s >= thr);
        }
    }

    (lines, matched)
}

// endregion: Matching

// region: CLI

/// Fuzzy (edit-distance / alignment bounded) substring search
#[derive(Parser)]
#[command(name = "sz-fuzzy-find")]
#[command(version, about = "SIMD/GPU-accelerated fuzzy substring search", long_about = None)]
struct Args {
    /// Substring to search for (approximately); omit when using -e
    pattern: Option<String>,

    /// Input files (use '-' or omit for stdin)
    inputs: Vec<String>,

    /// Additional query; a line matches if ANY query matches (repeatable)
    #[arg(short = 'e', long = "pattern")]
    extra: Vec<String>,

    /// Maximum edit distance (used directly in -w edit mode, or as a similarity floor)
    #[arg(short = 'k', long = "max-distance", default_value = "1")]
    max_distance: usize,

    /// Minimum normalized similarity 0..1 (overrides -k); 1.0 == exact
    #[arg(long = "min-similarity")]
    min_similarity: Option<f64>,

    /// Scoring model: edit, keyboard, or phonetic
    #[arg(long = "cost", default_value = "edit")]
    cost: String,

    /// Load a custom scoring matrix from FILE (implies a custom cost model)
    #[arg(long = "cost-matrix")]
    cost_matrix: Option<String>,

    /// Word mode: match the needle against each token rather than the whole line
    #[arg(short = 'w', long = "word")]
    word: bool,

    /// Execution device: auto, cpu, or gpu
    #[arg(long = "device", default_value = "auto")]
    device: String,

    /// CPU thread count (0 = all cores)
    #[arg(long = "threads")]
    threads: Option<usize>,

    /// GPU device index (with --device gpu)
    #[arg(long = "gpu-id", default_value = "0")]
    gpu_id: usize,

    /// Case-insensitive search (full Unicode case folding)
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Show line numbers
    #[arg(short = 'n', long)]
    line_numbers: bool,

    /// Count matching lines only
    #[arg(short = 'c', long)]
    count: bool,

    /// Enable UTF-8 mode: Unicode newlines, and code-point edit distances in -w mode
    #[arg(long)]
    utf8: bool,

    /// Emit JSON Lines in the ripgrep-compatible schema
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// NUL-terminate each output record instead of newline
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 on any match, 1 on none
    #[arg(short = 'q', long, conflicts_with_all = ["count", "json", "null", "line_numbers"], help_heading = "Output Formats")]
    quiet: bool,
}

fn build_device(args: &Args) -> Result<DeviceScope, String> {
    let result = match args.device.as_str() {
        "gpu" => DeviceScope::gpu_device(args.gpu_id),
        "cpu" => DeviceScope::cpu_cores(args.threads.unwrap_or(0)),
        // `DeviceScope::default()` yields a single core; 0 means every core.
        _ => DeviceScope::cpu_cores(args.threads.unwrap_or(0)),
    };
    result.map_err(|e| format!("{:?}", e))
}

fn parse_cost(args: &Args) -> Result<Cost, String> {
    if args.cost_matrix.is_some() {
        return Ok(Cost::Custom);
    }
    match args.cost.as_str() {
        "edit" => Ok(Cost::Edit),
        "keyboard" => Ok(Cost::Keyboard),
        "phonetic" => Ok(Cost::Phonetic),
        other => Err(format!(
            "unknown --cost '{}' (edit|keyboard|phonetic)",
            other
        )),
    }
}

fn main() {
    let args = Args::parse();

    let cost = parse_cost(&args).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        process::exit(2);
    });
    let cost_is_edit = matches!(cost, Cost::Edit);

    let device = build_device(&args).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not initialize device '{}': {}",
            args.device, e
        );
        process::exit(2);
    });

    let scheme = build_scheme(&cost, args.cost_matrix.as_deref()).unwrap_or_else(|e| {
        eprintln!("Error reading cost matrix: {}", e);
        process::exit(2);
    });

    let sw = SmithWatermanScores::new(
        &device,
        &scheme.byte_to_class,
        &scheme.costs,
        scheme.gap_open,
        scheme.gap_extend,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: Smith-Waterman init failed: {:?}", e);
        process::exit(2);
    });

    // Standard unit-cost Levenshtein for -w edit mode, byte- and code-point-level.
    let lev = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap_or_else(|e| {
        eprintln!("Error: Levenshtein init failed: {:?}", e);
        process::exit(2);
    });
    let lev_utf8 = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap_or_else(|e| {
        eprintln!("Error: UTF-8 Levenshtein init failed: {:?}", e);
        process::exit(2);
    });

    let engines = Engines {
        device,
        sw,
        lev,
        lev_utf8,
    };

    // With -e supplying the queries, a positional is an input path, as in grep.
    let mut inputs = args.inputs.clone();
    let mut patterns: Vec<String> = Vec::new();
    match (&args.pattern, args.extra.is_empty()) {
        (Some(pattern), true) => patterns.push(pattern.clone()),
        (Some(path), false) => inputs.insert(0, path.clone()),
        (None, true) => {
            eprintln!("Error: no query given; pass a pattern or -e PATTERN");
            process::exit(ExitCode::Error as i32);
        }
        (None, false) => {}
    }
    patterns.extend(args.extra.iter().cloned());
    if inputs.is_empty() {
        inputs.push("-".to_string());
    }
    let queries: Vec<Query> = patterns
        .into_iter()
        .map(|p| Query {
            bytes: p.into_bytes(),
        })
        .collect();

    let cfg = Config {
        ignore_case: args.ignore_case,
        word: args.word,
        cost_is_edit,
        max_distance: args.max_distance,
        min_similarity: args.min_similarity,
        line_numbers: args.line_numbers,
        count: args.count,
        utf8: args.utf8 || args.ignore_case,
        prefix: inputs.len() > 1,
        json: args.json,
        quiet: args.quiet,
        terminator: Terminator::from_null(args.null),
    };

    let mut output = get_output(None).unwrap_or_else(|e| {
        eprintln!("Error opening output: {}", e);
        process::exit(2);
    });

    let mut total = 0usize;
    for name in &inputs {
        let input = match get_input(Some(name.as_str())) {
            Ok(input) => input,
            Err(e) => {
                eprintln!("Error reading {}: {}", name, e);
                process::exit(2);
            }
        };
        let (lines, matched) = search_lines(input.as_bytes(), &queries, &engines, &cfg);

        let mut count = 0usize;
        for (i, line) in lines.iter().enumerate() {
            if !matched[i] {
                continue;
            }
            count += 1;
            if cfg.count {
                continue;
            }
            if let Err(e) = write_match(&mut output, &cfg, name, i + 1, line) {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    process::exit(0);
                }
                eprintln!("Error writing output: {}", e);
                process::exit(2);
            }
        }
        if cfg.count && !cfg.quiet {
            let _ = if cfg.prefix {
                writeln!(output, "{}:{}", name, count)
            } else {
                writeln!(output, "{}", count)
            };
        }
        total += count;
    }

    // Flush explicitly: process::exit skips the BufWriter's Drop, which would
    // otherwise discard everything we buffered.
    let _ = output.flush();

    // grep convention: exit 0 if any line matched, 1 otherwise.
    process::exit(if total > 0 { 0 } else { 1 });
}

fn write_match(
    output: &mut dyn Write,
    cfg: &Config,
    name: &str,
    line_no: usize,
    line: &[u8],
) -> io::Result<()> {
    if cfg.quiet {
        return Ok(());
    }
    if cfg.json {
        // Ripgrep's schema, minus `submatches`: fuzzy matching has no exact span.
        output.write_all(br#"{"type":"match","data":{"path":"#)?;
        json_text_field_to(output, name.as_bytes())?;
        output.write_all(br#","lines":"#)?;
        json_text_field_to(output, line)?;
        write!(output, r#","line_number":{},"submatches":[]}}}}"#, line_no)?;
        return output.write_all(b"\n");
    }
    if cfg.prefix {
        write!(output, "{}:", name)?;
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
