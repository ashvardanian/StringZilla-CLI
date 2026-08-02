//! SIMD-accelerated line deduplication utility
//!
//! Remove duplicate lines from files, keeping the first occurrence of each unique line.
//! Uses StringZilla's SIMD hash for fast deduplication with minimal memory overhead.
//!
//! # Algorithm
//!
//! This implementation uses an open-addressed flat hash set with linear probing:
//!
//! ```text
//! AppendOnlyFlatHashSet: Vec<LineEntry>
//!     slot[0]: { hash: 0x1234,    offset: 0,         length: 10       }
//!     slot[1]: { hash: 0,         offset: u64::MAX,  length: u64::MAX }  ← empty
//!     slot[2]: { hash: 0xABCD,    offset: 15,        length: 8        }
//!     ...
//!
//! LineEntry = 24 bytes (hash: u64, offset: u64, length: u64)
//! ```
//!
//! ## Open Addressing with Linear Probing
//!
//! - Insert: `slot = hash & mask`, probe forward until empty slot
//! - Lookup: `slot = hash & mask`, probe forward checking hash matches
//! - Empty slots marked by `offset = u64::MAX` (impossible for real files)
//!
//! ## Growth Strategy
//!
//! - Start with 1024 slots (24 KB)
//! - Grow 2x when load factor exceeds 60%
//! - Rehash all entries into new larger array
//!
//! ## In-Place Compaction
//!
//! When modifying a file in-place, the algorithm uses two pointers:
//!
//! ```text
//! read_pos ────────────────────────────►
//!     ┌────────┬────────┬────────┬────────┬────────┐
//!     │ line A │ line B │ line A │ line C │ line B │  (input)
//!     └────────┴────────┴────────┴────────┴────────┘
//!
//! write_pos ───────────────►
//!     ┌────────┬────────┬────────┐
//!     │ line A │ line B │ line C │  (compacted output)
//!     └────────┴────────┴────────┘
//! ```
//!
//! **Invariant**: `write_pos ≤ read_pos` always holds, ensuring we never overwrite
//! unread data. After compaction, the file is truncated to the new length.
//!
//! ## Case-Insensitive Mode
//!
//! For `-i` mode, uses proper Unicode case folding via `utf8_uncased_fold()`:
//!
//! 1. **Hashing**: Case-fold line into scratch buffer, then hash the folded form
//! 2. **Collision check**: Use `utf8_uncased_order()` directly on original
//!    lines (no re-folding needed for comparison)
//!
//! # Examples
//!
//! ```bash
//! # Write unique lines to stdout
//! sz-dedup file.txt
//!
//! # Rewrite the file in place
//! sz-dedup --in-place file.txt
//!
//! # Case-insensitive deduplication (full Unicode support)
//! sz-dedup -i file.txt
//!
//! # Output to different file (streaming mode)
//! sz-dedup file.txt -o unique.txt
//!
//! # From stdin to stdout
//! cat file.txt | sz-dedup
//!
//! # Show count of unique lines
//! sz-dedup -c file.txt
//! ```

use std::cmp::Ordering;
use std::io::{self, Write};

use clap::Parser;
use stringzilla::sz;

mod shared;
use shared::*;

// region: AppendOnlyFlatHashSet

/// Entry in the flat hash set. 24 bytes total.
#[derive(Clone, Copy)]
struct LineEntry {
    hash: u64,
    offset: u64,
    length: u64,
}

impl LineEntry {
    /// Sentinel value for empty slots (offset=u64::MAX is impossible for real files)
    const EMPTY: Self = Self {
        hash: 0,
        offset: u64::MAX,
        length: u64::MAX,
    };

    #[inline]
    fn is_empty(&self) -> bool {
        self.offset == u64::MAX
    }
}

impl Default for LineEntry {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Open-addressed hash set with linear probing.
/// Grows 2x when load factor exceeds 60%.
struct AppendOnlyFlatHashSet {
    slots: Vec<LineEntry>,
    populated_count: usize,
}

impl AppendOnlyFlatHashSet {
    /// Create a new hash set with initial capacity of 1024 slots (24 KB)
    fn new() -> Self {
        const INITIAL_CAPACITY: usize = 1024;
        Self {
            slots: vec![LineEntry::default(); INITIAL_CAPACITY],
            populated_count: 0,
        }
    }

    /// Mask for fast modulo via bitwise AND (slots.len() - 1)
    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Entries carrying the given hash, probing forward from its home slot until an
    /// empty one. The load factor stays under 60%, so an empty slot always ends it.
    #[inline]
    fn find(&self, hash: u64) -> impl Iterator<Item = &LineEntry> {
        let mask = self.mask();
        let home = (hash as usize) & mask;
        (0..self.slots.len())
            .map(move |step| &self.slots[(home + step) & mask])
            .take_while(|entry| !entry.is_empty())
            .filter(move |entry| entry.hash == hash)
    }

    /// Whether `line` was already recorded, comparing it against the bytes each
    /// same-hash entry points at in `data`.
    #[inline]
    fn contains(&self, hash: u64, line: &[u8], data: &[u8], ignore_case: bool) -> bool {
        self.find(hash).any(|entry| {
            let start = entry.offset as usize;
            let existing = &data[start..start + entry.length as usize];
            lines_equal(existing, line, ignore_case)
        })
    }

    /// Insert a new entry. Grows the table if load factor > 60%.
    #[inline]
    fn insert(&mut self, hash: u64, offset: u64, length: u64) {
        // Grow if load factor > 60%
        if self.populated_count * 100 > self.slots.len() * 60 {
            self.grow();
        }

        self.insert_no_grow(hash, offset, length);
    }

    /// Insert without checking load factor (used during rehash)
    #[inline]
    fn insert_no_grow(&mut self, hash: u64, offset: u64, length: u64) {
        let mask = self.mask();
        let mut slot = (hash as usize) & mask;
        while !self.slots[slot].is_empty() {
            slot = (slot + 1) & mask;
        }
        self.slots[slot] = LineEntry {
            hash,
            offset,
            length,
        };
        self.populated_count += 1;
    }

    /// Double the capacity and rehash all entries
    fn grow(&mut self) {
        let new_cap = self.slots.len() * 2;
        let old_slots = std::mem::replace(&mut self.slots, vec![LineEntry::default(); new_cap]);
        self.populated_count = 0;

        for entry in old_slots {
            if !entry.is_empty() {
                self.insert_no_grow(entry.hash, entry.offset, entry.length);
            }
        }
    }
}

// endregion: AppendOnlyFlatHashSet

// region: Hash and Comparison Utilities

/// Compute hash for a line, using case-folding if ignore_case is true.
#[inline]
fn compute_hash(line: &[u8], ignore_case: bool, scratch: &mut Vec<u8>) -> u64 {
    if ignore_case {
        // UTF-8 case folding can expand characters (e.g., ß → ss), max ~3x
        scratch.clear();
        scratch.resize(line.len().saturating_mul(3).max(64), 0);
        let folded_len = sz::utf8_uncased_fold(line, &mut scratch[..]);
        sz::hash(&scratch[..folded_len])
    } else {
        sz::hash(line)
    }
}

/// Check if two lines are equal, using case-insensitive comparison if needed.
#[inline]
fn lines_equal(a: &[u8], b: &[u8], ignore_case: bool) -> bool {
    if ignore_case {
        sz::utf8_uncased_order(a, b) == Ordering::Equal
    } else {
        a == b
    }
}

// endregion: Hash and Comparison Utilities

// region: Deduplication Functions

/// Deduplicate lines in-place, compacting the buffer.
///
/// When `utf8` is true, handles all Unicode newlines (LF, CR, CRLF, NEL, LS, PS).
/// When `utf8` is false, only handles LF newlines.
fn dedup_in_place(data: &mut [u8], ignore_case: bool, utf8: bool) -> DedupCounts {
    // Compaction only overwrites `[0..write_pos]`, which is always behind `line_start`,
    // so the tail `data[line_start..]` is intact — find each newline lazily there with
    // no up-front offset buffer.
    let mut seen = AppendOnlyFlatHashSet::new();
    let mut scratch = Vec::new();
    let mut write_pos: usize = 0;
    let mut unique_count: usize = 0;
    let mut total_count: usize = 0;
    let mut line_start: usize = 0;

    while line_start < data.len() {
        // Next newline in the untouched tail; the final line has none (len 0).
        let (line_end, newline_len) = if utf8 {
            // UTF-8 aware: first of the 7 Unicode newline chars, CRLF as one run.
            match sz::Utf8Newlines::new(&data[line_start..]).next() {
                Some(run) => (offset_within(data, run), run.len()),
                None => (data.len(), 0),
            }
        } else {
            match sz::find(&data[line_start..], b"\n") {
                Some(offset) => (line_start + offset, 1),
                None => (data.len(), 0),
            }
        };

        let line_len = line_end - line_start;
        let line = &data[line_start..line_end];
        let hash = compute_hash(line, ignore_case, &mut scratch);

        // Check for duplicate against already-written lines in [0..write_pos].
        let is_duplicate = seen.contains(hash, line, data, ignore_case);

        if !is_duplicate {
            seen.insert(hash, write_pos as u64, line_len as u64);
            if write_pos != line_start {
                data.copy_within(line_start..line_end, write_pos);
            }
            write_pos += line_len;
            // Preserve the original newline sequence.
            if newline_len > 0 {
                data.copy_within(line_end..line_end + newline_len, write_pos);
                write_pos += newline_len;
            }
            unique_count += 1;
        }

        total_count += 1;
        line_start = line_end + newline_len;
    }

    DedupCounts {
        total: total_count,
        unique: unique_count,
        compacted_bytes: write_pos,
    }
}

/// How many lines were read, how many survived deduplication, and how many bytes
/// the survivors occupy.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct DedupCounts {
    total: usize,
    unique: usize,
    /// Length to truncate the file to after in-place compaction. The streaming
    /// path rewrites nothing and leaves it zero.
    compacted_bytes: usize,
}

impl DedupCounts {
    /// Whether any duplicate was dropped, which is what `-q` reports.
    fn dropped_any(self) -> bool {
        self.unique < self.total
    }
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    json: bool,
    terminator: Terminator,
    /// Input name carried into the JSON envelope.
    path: &'a str,
}

/// Write one surviving line. `index` is zero-based; records report it one-based.
fn write_line(
    output: &mut dyn Write,
    config: &OutputConfig,
    line: &[u8],
    index: usize,
) -> io::Result<()> {
    if config.json {
        output.write_all(br#"{"type":"line","data":{"path":"#)?;
        json_text_field_to(output, config.path.as_bytes())?;
        output.write_all(br#","lines":"#)?;
        json_text_field_to(output, line)?;
        write!(output, r#","line_number":{}}}}}"#, index + 1)?;
        return output.write_all(b"\n");
    }
    output.write_all(line)?;
    output.write_all(&[config.terminator.as_byte()])
}

/// Deduplicate lines, writing to output stream (for stdin or explicit output file).
/// When `utf8` is true, handles all Unicode newlines (LF, CR, CRLF, NEL, LS, PS).
/// When `utf8` is false, only handles LF newlines.
/// Output is normalized to LF newlines.
fn dedup_to_writer(
    data: &[u8],
    output: &mut dyn Write,
    ignore_case: bool,
    utf8: bool,
    config: &OutputConfig,
) -> io::Result<DedupCounts> {
    let mut seen = AppendOnlyFlatHashSet::new();
    let mut scratch = Vec::new();
    let mut counts = DedupCounts::default();

    let lines = LineIter::new(data, Newlines::from_utf8(utf8));

    for line in lines {
        counts.total += 1;
        let line_offset = offset_within(data, line);
        let hash = compute_hash(line, ignore_case, &mut scratch);

        let is_duplicate = seen.contains(hash, line, data, ignore_case);

        if !is_duplicate {
            seen.insert(hash, line_offset as u64, line.len() as u64);
            write_line(output, config, line, counts.unique)?;
            counts.unique += 1;
        }
    }

    output.flush()?;
    Ok(counts)
}

// endregion: Deduplication Functions

// region: CLI

/// Deduplicate lines in files
#[derive(Parser)]
#[command(name = "sz-dedup")]
#[command(version, about = "SIMD-accelerated line deduplication", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output file (use '-' or omit for stdout, or omit for in-place)
    #[arg(short, long)]
    output: Option<String>,

    /// Case-insensitive deduplication (full Unicode case folding)
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Rewrite the input file in place instead of writing to stdout
    #[arg(long, conflicts_with_all = ["output", "quiet"])]
    in_place: bool,

    /// Show count of unique lines
    #[arg(short = 'c', long)]
    count: bool,

    /// Enable UTF-8 mode (handle Unicode newlines: CR, CRLF, NEL, LS, PS)
    #[arg(long)]
    utf8: bool,

    /// Emit JSON Lines, one record per emitted line
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// NUL-terminate each output line instead of newline
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress output; exit 0 if any duplicate was dropped, 1 otherwise
    #[arg(short = 'q', long, conflicts_with_all = ["output", "json", "null"], help_heading = "Output Formats")]
    quiet: bool,
}

fn main() {
    let args = Args::parse();

    // UTF-8 mode is implicit when case-insensitive (case folding requires UTF-8)
    let utf8_mode = args.utf8 || args.ignore_case;

    let mut stdout = io::stdout();

    // In-place is opt-in: the default writes to stdout like every other binary,
    // so `sz-dedup file | head` cannot destroy the input.
    let in_place_path = if args.in_place {
        let Some(path) = args.input.as_deref().filter(|path| *path != "-") else {
            eprintln!("Error: --in-place requires a file argument (cannot rewrite stdin)");
            ExitCode::Error.exit(&mut stdout);
        };
        Some(path)
    } else {
        None
    };

    let config = OutputConfig {
        json: args.json,
        terminator: Terminator::from_null(args.null),
        path: args.input.as_deref().unwrap_or("-"),
    };

    let counts = if let Some(input_path) = in_place_path {
        // In-place mode: mutable mmap, compact, truncate
        let mut input = get_input_mutable(input_path).unwrap_or_else(|error| {
            exit_with_error(
                &mut stdout,
                &error,
                "Error opening file for in-place modification",
            )
        });

        let data = input.as_mut_bytes().unwrap();
        let counts = dedup_in_place(data, args.ignore_case, utf8_mode);

        if let Err(error) = input.truncate_and_flush(counts.compacted_bytes as u64) {
            exit_with_error(&mut stdout, &error, "Error truncating file");
        }

        counts
    } else {
        // Streaming mode: read-only input, write to output
        let input = get_input(args.input.as_deref())
            .unwrap_or_else(|error| exit_with_error(&mut stdout, &error, "Error reading input"));

        let data = input.as_bytes();

        // `-q` reports through the exit code alone, so the lines go nowhere.
        let mut output: Box<dyn Write> = if args.quiet {
            Box::new(io::sink())
        } else {
            get_output(args.output.as_deref()).unwrap_or_else(|error| {
                exit_with_error(&mut stdout, &error, "Error opening output")
            })
        };

        dedup_to_writer(data, &mut output, args.ignore_case, utf8_mode, &config)
            .unwrap_or_else(|error| exit_on_write_error(&mut output, &error, "Error deduplicating"))
    };

    // The summary always terminates a `--json` run. In-place mode has no per-line
    // stream at all, so it is the entire report there.
    if args.json {
        let _ = stdout.write_all(br#"{"type":"summary","data":{"path":"#);
        let _ = json_text_field_to(&mut stdout, config.path.as_bytes());
        let _ = writeln!(
            stdout,
            r#","unique_lines":{},"total_lines":{}}}}}"#,
            counts.unique, counts.total
        );
    } else if args.count {
        eprintln!("{} unique lines", counts.unique);
    }

    if args.quiet {
        ExitCode::from_found(counts.dropped_any()).exit(&mut stdout);
    }
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config() -> OutputConfig<'static> {
        OutputConfig {
            json: false,
            terminator: Terminator::Newline,
            path: "-",
        }
    }

    #[test]
    fn inserts_and_finds_entries_by_hash() {
        let mut set = AppendOnlyFlatHashSet::new();
        set.insert(123, 0, 10);
        set.insert(456, 20, 5);
        set.insert(123, 50, 8); // Same hash, different entry

        assert_eq!(set.populated_count, 3);
        assert_eq!(set.find(123).count(), 2);
        assert_eq!(set.find(456).count(), 1);
        assert_eq!(set.find(789).count(), 0);
    }

    #[test]
    fn grows_and_rehashes_beyond_capacity() {
        let mut set = AppendOnlyFlatHashSet::new();
        // Insert more than 60% of initial capacity to trigger growth
        for i in 1..=700 {
            set.insert(i, i * 10, i);
        }
        assert!(set.slots.len() > 1024);
        assert_eq!(set.populated_count, 700);

        // Verify all entries are still findable
        for i in 1..=700 {
            assert_eq!(set.find(i).count(), 1);
        }
    }

    #[test]
    fn dedups_repeated_lines_in_place() {
        let mut data = b"line1\nline2\nline1\nline3\n".to_vec();
        let counts = dedup_in_place(&mut data, false, false);

        assert_eq!(counts.unique, 3);
        let result = String::from_utf8(data[..counts.compacted_bytes].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn dedups_in_place_ignoring_case() {
        let mut data = b"Hello\nhello\nworld\nWORLD\n".to_vec();
        let counts = dedup_in_place(&mut data, true, true);

        assert_eq!(counts.unique, 2);
        let result = String::from_utf8(data[..counts.compacted_bytes].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["Hello", "world"]);
    }

    #[test]
    fn dedups_in_place_folding_unicode() {
        let mut data = "MÜNCHEN\nmünchen\nberlin\n".as_bytes().to_vec();
        let counts = dedup_in_place(&mut data, true, true);

        assert_eq!(counts.unique, 2);
        let result = String::from_utf8(data[..counts.compacted_bytes].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["MÜNCHEN", "berlin"]);
    }

    #[test]
    fn collapses_all_duplicate_lines_in_place() {
        let mut data = b"dup\ndup\ndup\ndup\n".to_vec();
        let counts = dedup_in_place(&mut data, false, false);

        assert_eq!(counts.unique, 1);
        assert_eq!(&data[..counts.compacted_bytes], b"dup\n");
    }

    #[test]
    fn keeps_unique_lines_in_place() {
        let mut data = b"a\nb\nc\n".to_vec();
        let original_len = data.len();
        let counts = dedup_in_place(&mut data, false, false);

        assert_eq!(counts.unique, 3);
        assert_eq!(counts.compacted_bytes, original_len);
    }

    #[test]
    fn dedups_streaming_to_writer() {
        let data = b"line1\nline2\nline1\nline3\n";
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, false, false, &text_config()).unwrap();

        assert_eq!(counts.unique, 3);
        let result = String::from_utf8(output).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn dedups_to_writer_ignoring_case() {
        let data = b"Hello\nhello\nHELLO\nworld\n";
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, true, true, &text_config()).unwrap();

        assert_eq!(counts.unique, 2);
        let result = String::from_utf8(output).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["Hello", "world"]);
    }

    #[test]
    fn preserves_first_occurrence_casing() {
        let data = b"First\nfirst\nFIRST\n";
        let mut output = Vec::new();

        dedup_to_writer(data, &mut output, true, true, &text_config()).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert_eq!(result, "First\n");
    }

    #[test]
    fn dedups_empty_input() {
        let data = b"";
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, false, false, &text_config()).unwrap();

        assert_eq!(counts.unique, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn dedups_repeated_blank_lines() {
        let data = b"\n\n\ntext\n\n";
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, false, false, &text_config()).unwrap();

        assert_eq!(counts.unique, 2); // "" and "text"
    }

    #[test]
    fn detects_empty_line_entries() {
        assert!(LineEntry::EMPTY.is_empty());
        assert!(LineEntry::default().is_empty());

        let occupied = LineEntry {
            hash: 123,
            offset: 0,
            length: 10,
        };
        assert!(!occupied.is_empty());
    }
}

// endregion: Tests
