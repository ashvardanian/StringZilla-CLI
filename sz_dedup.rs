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
//! ## In-Place Rewriting
//!
//! `--in-place` writes the surviving lines to a temporary file beside the input and then
//! copies them back over the original, so the file keeps its identity — symlinks and
//! hardlinks to it survive — and an interrupted run leaves the input intact. Each line
//! keeps the terminator it arrived with, so nothing but the duplicates changes.
//!
//! ## Case-Insensitive Mode
//!
//! `--ignore-case` uses proper Unicode case folding via `utf8_uncased_fold()`:
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
//! sz-dedup --ignore-case file.txt
//!
//! # Output to different file (streaming mode)
//! sz-dedup file.txt --output unique.txt
//!
//! # From stdin to stdout
//! cat file.txt | sz-dedup
//!
//! # Show one line about the whole run
//! sz-dedup --summary file.txt
//! ```

use std::cmp::Ordering;
use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz;

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

/// How many lines were read and how many survived deduplication.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct DedupCounts {
    total: usize,
    unique: usize,
}

impl DedupCounts {
    /// Whether any duplicate was dropped, which is what `--quiet` reports.
    fn dropped_any(self) -> bool {
        self.unique < self.total
    }
}

/// Lines paired with the span they occupy, terminator included, so a caller that
/// rewrites the input can reproduce CR, CRLF, NEL, LS and PS rather than flatten them.
struct TerminatedLines<'a> {
    data: &'a [u8],
    lines: LineIter<'a>,
    pending: Option<&'a [u8]>,
}

impl<'a> TerminatedLines<'a> {
    fn new(data: &'a [u8], newlines: Newlines) -> Self {
        let mut lines = LineIter::new(data, newlines);
        let pending = lines.next();
        Self {
            data,
            lines,
            pending,
        }
    }
}

impl<'a> Iterator for TerminatedLines<'a> {
    type Item = (&'a [u8], &'a [u8]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let line = self.pending.take()?;
        let start = offset_within(self.data, line);
        self.pending = self.lines.next();
        let end = match self.pending {
            Some(next) => offset_within(self.data, next),
            None => self.data.len(),
        };
        Some((line, &self.data[start..end]))
    }
}

/// How a surviving line is written out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rendering {
    /// The line plus the requested terminator, normalizing the input's own.
    Terminated(Terminator),
    /// One JSON record per line, closed by a summary record.
    Json,
    /// The line exactly as it appeared, terminator and all.
    Verbatim,
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    rendering: Rendering,
    /// Input name carried into the JSON envelope.
    path: &'a str,
}

/// Write one surviving line. `index` is zero-based; records report it one-based.
fn write_line(
    output: &mut dyn Write,
    config: &OutputConfig,
    line: &[u8],
    span: &[u8],
    index: usize,
) -> io::Result<()> {
    match config.rendering {
        Rendering::Json => write_line_record(output, config.path, line, index),
        Rendering::Verbatim => output.write_all(span),
        Rendering::Terminated(terminator) => {
            output.write_all(line)?;
            output.write_all(&[terminator.as_byte()])
        }
    }
}

/// Write the summary record that closes a JSON stream.
fn write_summary_json(output: &mut dyn Write, path: &str, counts: DedupCounts) -> io::Result<()> {
    output.write_all(br#"{"type":"summary","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    writeln!(
        output,
        r#","unique_lines":{},"total_lines":{}}}}}"#,
        counts.unique, counts.total
    )
}

/// Deduplicate `data` into `output`, keeping the first occurrence of each line.
/// When `utf8` is true, handles all Unicode newlines (LF, CR, CRLF, NEL, LS, PS).
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

    for (line, span) in TerminatedLines::new(data, Newlines::from_utf8(utf8)) {
        counts.total += 1;
        let line_offset = offset_within(data, line);
        let hash = compute_hash(line, ignore_case, &mut scratch);

        let is_duplicate = seen.contains(hash, line, data, ignore_case);

        if !is_duplicate {
            seen.insert(hash, line_offset as u64, line.len() as u64);
            write_line(output, config, line, span, counts.unique)?;
            counts.unique += 1;
        }
    }

    if config.rendering == Rendering::Json {
        write_summary_json(output, config.path, counts)?;
    }
    output.flush()?;
    Ok(counts)
}

// endregion: Deduplication Functions

// region: CLI

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
}

/// Deduplicate lines in files
#[derive(Parser)]
#[command(name = "sz-dedup")]
#[command(version, about = "SIMD-accelerated line deduplication", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Write to this file instead of stdout
    #[arg(long, conflicts_with_all = ["in_place", "dry_run"])]
    output: Option<String>,

    /// Rewrite the input file, swapping the result in atomically once it is on disk
    #[arg(long, conflicts_with_all = ["dry_run", "null", "quiet"])]
    in_place: bool,

    /// Deduplicate the input and write nothing
    #[arg(long)]
    dry_run: bool,

    /// Fold case when comparing lines; implies --utf8
    #[arg(long)]
    ignore_case: bool,

    /// Treat the input as UTF-8 text
    #[arg(long)]
    utf8: bool,

    /// Render records as plain lines or as JSON Lines
    #[arg(long, value_enum, default_value_t = Format::Text, help_heading = "Output Formats")]
    format: Format,

    /// Print one line about the whole run on stdout
    #[arg(long, conflicts_with = "dry_run", help_heading = "Output Formats")]
    summary: bool,

    /// NUL-terminate each output record instead of newline
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any duplicate was dropped, 1 otherwise
    #[arg(long, conflicts_with_all = ["output", "dry_run", "null", "summary"], help_heading = "Output Formats")]
    quiet: bool,
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Every constraint that depends on an argument's *value*, which clap cannot declare.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.format == Format::Json {
        if args.null {
            return Err(reject("--format json cannot be combined with --null"));
        }
        if args.quiet {
            return Err(reject("--format json cannot be combined with --quiet"));
        }
    }
    if args.in_place && args.input.as_deref().is_none_or(|path| path == "-") {
        return Err(reject(
            "--in-place requires a file argument (cannot rewrite stdin)",
        ));
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    // Every byte this run prints goes here, so the records stay in one order.
    let mut output = stdout_writer();
    report("sz-dedup", run(&args, &mut output))
}

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    // Case folding is a Unicode operation, so it brings the Unicode newline set with it.
    let utf8_mode = args.utf8 || args.ignore_case;
    let name = args.input.as_deref().unwrap_or("-");

    let input = get_input(args.input.as_deref()).at(name)?;
    let data = input.as_bytes();

    // In-place is opt-in: the default writes to stdout like every other binary,
    // so `sz-dedup file | head` cannot destroy the input.
    let counts = if args.in_place {
        let path = args.input.as_deref().expect("validated");
        let config = OutputConfig {
            rendering: Rendering::Verbatim,
            path: name,
        };
        write_replacing("sz-dedup", path, |output| {
            dedup_to_writer(data, output, args.ignore_case, utf8_mode, &config)
        })?
    } else if args.dry_run || args.quiet {
        let config = OutputConfig {
            rendering: Rendering::Terminated(Terminator::from_null(args.null)),
            path: name,
        };
        dedup_to_writer(data, &mut io::sink(), args.ignore_case, utf8_mode, &config).at(name)?
    } else {
        let config = OutputConfig {
            rendering: match args.format {
                Format::Json => Rendering::Json,
                Format::Text => Rendering::Terminated(Terminator::from_null(args.null)),
            },
            path: name,
        };
        // Through a temporary like `--in-place`, so an interrupted run leaves the previous
        // file rather than a half-written one, and naming the input as the output does not
        // truncate the mapping this run is still reading from.
        match args.output.as_deref().filter(|path| *path != "-") {
            Some(path) => write_creating("sz-dedup", path, |output| {
                dedup_to_writer(data, output, args.ignore_case, utf8_mode, &config)
            })?,
            None => dedup_to_writer(data, output, args.ignore_case, utf8_mode, &config).at("-")?,
        }
    };

    if args.format == Format::Json {
        // A run with no record stream still owes its one summary record.
        if args.in_place || args.dry_run {
            write_summary_json(output, name, counts).at("-")?;
        }
    } else if args.summary || args.dry_run {
        println!("{} unique lines of {}", counts.unique, counts.total);
    }
    output.flush().at("-")?;

    Ok(if args.quiet {
        Status::from_found(counts.dropped_any())
    } else {
        Status::from_found(counts.unique > 0)
    })
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn text_config() -> OutputConfig<'static> {
        OutputConfig {
            rendering: Rendering::Terminated(Terminator::Newline),
            path: "-",
        }
    }

    fn verbatim_config() -> OutputConfig<'static> {
        OutputConfig {
            rendering: Rendering::Verbatim,
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
    fn folds_unicode_case_when_comparing_lines() {
        let data = "MÜNCHEN\nmünchen\nberlin\n".as_bytes();
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, true, true, &text_config()).unwrap();

        assert_eq!(counts.unique, 2);
        assert_eq!(output, "MÜNCHEN\nberlin\n".as_bytes());
    }

    #[test]
    fn keeps_every_terminator_the_input_used() {
        // The in-place rendering: a rewritten file must differ from the original only
        // by the lines that were dropped.
        let data = "a\r\nb\u{2028}a\r\nc".as_bytes();
        let mut output = Vec::new();

        let counts = dedup_to_writer(data, &mut output, false, true, &verbatim_config()).unwrap();

        assert_eq!(counts.total, 4);
        assert_eq!(output, "a\r\nb\u{2028}c".as_bytes());
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
    fn declares_no_short_flags() {
        let mut command = Args::command();
        command.build();
        assert!(command
            .get_arguments()
            .all(|argument| argument.get_short().is_none()
                || matches!(argument.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "output",
                "in-place",
                "dry-run",
                "ignore-case",
                "utf8",
                "format",
                "summary",
                "null",
                "quiet",
                "help",
                "version",
            ]
        );
    }

    /// Parse and then apply the value-conditional checks, as `run` does.
    fn accepts(flags: &[&str]) -> bool {
        let arguments = ["sz-dedup", "f"].into_iter().chain(flags.iter().copied());
        Args::try_parse_from(arguments).is_ok_and(|args| validate(&args).is_ok())
    }

    #[test]
    fn declares_the_conflicts_that_used_to_pass_silently() {
        assert!(accepts(&["--in-place"]));
        for flags in [
            vec!["--in-place", "--null"],
            vec!["--in-place", "--output", "o"],
            vec!["--in-place", "--dry-run"],
            vec!["--quiet", "--summary"],
            vec!["--quiet", "--format", "json"],
            vec!["--null", "--format", "json"],
            vec!["--summary", "--dry-run"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
        assert!(
            accepts(&["--summary", "--format", "json"]),
            "--summary names the record json already emits"
        );
    }

    #[test]
    fn refuses_to_rewrite_stdin_in_place() {
        for arguments in [
            vec!["sz-dedup", "--in-place"],
            vec!["sz-dedup", "--in-place", "-"],
        ] {
            let args = Args::try_parse_from(&arguments).unwrap();
            assert!(validate(&args).is_err(), "expected {:?} to fail", arguments);
        }
    }

    #[test]
    fn rewrites_an_empty_file_as_a_no_op() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("empty.txt");
        fs::write(&path, b"").unwrap();
        let config = verbatim_config();

        let counts = write_replacing("sz-dedup", path.to_str().unwrap(), |output| {
            dedup_to_writer(b"", output, false, false, &config)
        })
        .unwrap();

        assert_eq!(counts, DedupCounts::default());
        assert_eq!(fs::read(&path).unwrap(), b"");
    }

    #[test]
    fn closes_a_json_stream_with_its_summary() {
        let data = b"a\na\n";
        let mut output = Vec::new();
        let config = OutputConfig {
            rendering: Rendering::Json,
            path: "f.txt",
        };

        dedup_to_writer(data, &mut output, false, false, &config).unwrap();

        let text = String::from_utf8(output).unwrap();
        let records: Vec<_> = text.lines().collect();
        assert_eq!(records.len(), 2);
        assert!(records[1].contains(r#""type":"summary""#));
        assert!(records[1].contains(r#""unique_lines":1,"total_lines":2"#));
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
