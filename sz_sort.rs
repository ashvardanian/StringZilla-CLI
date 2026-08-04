//! SIMD-accelerated line sorting utility
//!
//! A faster, Unicode-correct replacement for `sort`, built on StringZilla's
//! `argsort` — case-insensitive folding and reversal happen inside the kernel.
//! Comparison is unsigned byte-wise, which for valid UTF-8 is identical to Unicode
//! code-point order, so `--utf8` only affects newline handling.
//!
//! Lines are held in a `BytesCowsAuto` borrowing the input buffer, which stores a
//! packed offset and length per line and picks their widths from the data size and
//! the longest line. A `Vec<&[u8]>` would spend 16 bytes per line on fat pointers
//! against 5 or 6 for the packed entry, which on a large file dominates the input
//! itself. `argsort_by` reaches the lines through a callback, so the kernel never
//! needs a materialized slice array either way.
//!
//! # Examples
//!
//! ```bash
//! # Sort a file to stdout
//! sz-sort file.txt
//!
//! # Reverse (descending) order
//! sz-sort --reverse file.txt
//!
//! # Sort and drop duplicate lines (like `sort -u`)
//! sz-sort --unique file.txt
//!
//! # Case-insensitive sort (full Unicode case folding)
//! sz-sort --ignore-case file.txt
//!
//! # Write to a file instead of stdout, or back into the input
//! sz-sort file.txt --output sorted.txt
//! sz-sort file.txt --in-place
//!
//! # Verify a file is already sorted (exit 1 if not)
//! sz-sort --check file.txt
//! ```

use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringtape::{BytesCowsAuto, StringTapeError};
use stringzilla::sz;

use shared::*;

// region: Sorting

/// How lines are ordered: which comparison, and in which direction.
#[derive(Clone, Copy)]
struct SortOrder {
    ignore_case: bool,
    reverse: bool,
}

impl SortOrder {
    /// Compare two lines in the requested direction. Case-insensitive comparison
    /// uses StringZilla's on-the-fly Unicode folding — no materialized keys, and
    /// reversal leaves `Equal` alone, so adjacency stays the same relation.
    #[inline]
    fn compare(self, left: &[u8], right: &[u8]) -> Ordering {
        let ordering = if self.ignore_case {
            sz::utf8_uncased_order(left, right)
        } else {
            left.cmp(right)
        };
        if self.reverse {
            ordering.reverse()
        } else {
            ordering
        }
    }

    /// Whether `left` is allowed to precede `right`.
    #[inline]
    fn holds(self, left: &[u8], right: &[u8]) -> bool {
        self.compare(left, right).is_le()
    }

    /// The same order stated for `argsort`, which folds and reverses inside the
    /// kernel rather than through a comparator.
    fn argsort_options(self) -> sz::ArgsortOptions {
        let mut options = sz::ArgsortOptions::default();
        if self.ignore_case {
            options = options.uncased();
        }
        if self.reverse {
            options = options.reversed();
        }
        options
    }
}

/// A re-runnable view of one buffer's lines.
///
/// `BytesCowsAuto::from_iter_and_data` walks its input twice — once to size the
/// offset and length types, once to record them — so it takes a `Clone` iterable
/// rather than a one-shot iterator. Splitting twice costs one extra SIMD pass and
/// avoids materializing the slices at all.
#[derive(Clone, Copy)]
struct Lines<'a> {
    data: &'a [u8],
    newlines: Newlines,
}

impl<'a> IntoIterator for Lines<'a> {
    type Item = &'a [u8];
    type IntoIter = LineIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        LineIter::new(self.data, self.newlines)
    }
}

/// Collect the input's lines into packed (offset, length) entries borrowing `data`.
fn collect_lines(data: &[u8], newlines: Newlines) -> Result<BytesCowsAuto<'_>, StringTapeError> {
    BytesCowsAuto::from_iter_and_data(Lines { data, newlines }, Cow::Borrowed(data))
}

/// The line at `index`. Indices come from a permutation as long as `lines`, so an
/// index outside it is a bug here rather than a bad input.
#[inline]
fn line_at<'a>(lines: &'a BytesCowsAuto<'a>, index: usize) -> &'a [u8] {
    lines.get(index).expect("permutation index within lines")
}

/// Compute the sorted permutation directly via StringZilla's `argsort`, which folds
/// case and reverses inside the kernel — so the case-insensitive path never
/// materializes a folded key per line.
fn sorted_order(
    lines: &BytesCowsAuto<'_>,
    order: SortOrder,
) -> Result<Vec<sz::SortedIdx>, sz::Status> {
    let mut permutation = vec![0usize; lines.len()];
    sz::argsort_by(
        |index| line_at(lines, index),
        &mut permutation,
        order.argsort_options(),
    )?;
    Ok(permutation)
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    format: Format,
    /// Drop lines equal to the one before them, which sorting made adjacent.
    unique: bool,
    terminator: Terminator,
    /// Input name carried into the JSON envelope.
    path: &'a str,
}

/// Write one sorted line. `position` is zero-based; records report it one-based.
fn write_line(
    output: &mut dyn Write,
    config: &OutputConfig,
    line: &[u8],
    position: usize,
) -> io::Result<()> {
    if config.format == Format::Json {
        return write_line_record(output, config.path, line, position);
    }
    output.write_all(line)?;
    output.write_all(&[config.terminator.as_byte()])
}

/// Write the summary record that closes a JSON stream.
fn write_summary_json(
    output: &mut dyn Write,
    path: &str,
    total: usize,
    written: usize,
) -> io::Result<()> {
    output.write_all(br#"{"type":"summary","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    writeln!(
        output,
        r#","written_lines":{},"total_lines":{}}}}}"#,
        written, total
    )
}

/// Write the lines in permutation order, dropping adjacent equals under `--unique`.
/// Returns how many lines were written.
fn write_sorted(
    lines: &BytesCowsAuto<'_>,
    permutation: &[sz::SortedIdx],
    order: SortOrder,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut previous: Option<&[u8]> = None;
    let mut emitted = 0;
    for &index in permutation {
        let line = line_at(lines, index);
        if config.unique
            && previous.is_some_and(|kept| order.compare(kept, line) == Ordering::Equal)
        {
            continue;
        }
        previous = Some(line);
        write_line(output, config, line, emitted)?;
        emitted += 1;
    }
    if config.format == Format::Json {
        write_summary_json(output, config.path, lines.len(), emitted)?;
    }
    output.flush()?;
    Ok(emitted)
}

/// Check whether `lines` are already in sorted order. Returns the 1-based index
/// of the first line that breaks the order, or `None` if fully sorted.
fn check_sorted(lines: &BytesCowsAuto<'_>, order: SortOrder) -> Option<usize> {
    (1..lines.len())
        .find(|&index| !order.holds(line_at(lines, index - 1), line_at(lines, index)))
        .map(|index| index + 1)
}

// endregion: Sorting

// region: CLI

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
}

/// Sort lines in a file or stream
#[derive(Parser)]
#[command(name = "sz-sort")]
#[command(version, about = "SIMD-accelerated line sorting", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Write to this file instead of stdout
    #[arg(long, conflicts_with_all = ["in_place", "dry_run"])]
    output: Option<String>,

    /// Rewrite the input file, swapping the result in atomically once it is on disk
    #[arg(long, conflicts_with_all = ["dry_run", "null", "quiet"])]
    in_place: bool,

    /// Sort the input and write nothing
    #[arg(long)]
    dry_run: bool,

    /// Reverse the result (descending order)
    #[arg(long)]
    reverse: bool,

    /// Drop duplicate lines, keeping one of each (like `sort -u`)
    #[arg(long)]
    unique: bool,

    /// Fold case when comparing lines; implies --utf8
    #[arg(long)]
    ignore_case: bool,

    /// Report through the exit code whether the input is already sorted
    #[arg(long, conflicts_with_all = ["output", "in_place", "dry_run", "unique", "null", "format", "summary"])]
    check: bool,

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

    /// Suppress all output; exit 0 if any line was sorted, 1 otherwise
    #[arg(long, conflicts_with_all = ["output", "dry_run", "null", "summary", "format"], help_heading = "Output Formats")]
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
        if args.in_place {
            return Err(reject("--format json cannot be combined with --in-place"));
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
    report("sz-sort", run(&args, &mut output))
}

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let name = args.input.as_deref().unwrap_or("-");
    let input = get_input(args.input.as_deref()).at(name)?;
    let data = input.as_bytes();

    let order = SortOrder {
        ignore_case: args.ignore_case,
        reverse: args.reverse,
    };
    // Case folding is a Unicode operation, so it brings the Unicode newline set with it.
    let newlines = Newlines::from_utf8(args.utf8 || args.ignore_case);
    let lines = collect_lines(data, newlines)
        .map_err(|error| io::Error::other(format!("indexing lines: {:?}", error)))
        .at(name)?;

    if args.check {
        return Ok(match check_sorted(&lines, order) {
            None => Status::Success,
            Some(line_number) => {
                if !args.quiet {
                    eprintln!("sz-sort: {}:{}: disorder", name, line_number);
                }
                Status::NoResult
            }
        });
    }

    let permutation = sorted_order(&lines, order)
        .map_err(|status| io::Error::other(format!("sorting: {:?}", status)))
        .at(name)?;
    let config = OutputConfig {
        format: args.format,
        unique: args.unique,
        terminator: Terminator::from_null(args.null),
        path: name,
    };

    let emitted = if args.dry_run || args.quiet {
        write_sorted(&lines, &permutation, order, &config, &mut io::sink()).at(name)?
    } else if args.in_place {
        let path = args.input.as_deref().expect("validated");
        write_replacing("sz-sort", path, |output| {
            write_sorted(&lines, &permutation, order, &config, output)
        })?
    } else {
        // Through a temporary like `--in-place`, so an interrupted run leaves the previous
        // file rather than a half-written one, and naming the input as the output does not
        // truncate the mapping this run is still reading from.
        match args.output.as_deref().filter(|path| *path != "-") {
            Some(path) => write_creating("sz-sort", path, |output| {
                write_sorted(&lines, &permutation, order, &config, output)
            })?,
            None => write_sorted(&lines, &permutation, order, &config, output).at("-")?,
        }
    };

    if args.format == Format::Json {
        // The record stream went to a sink, so its closing summary still owes stdout.
        if args.dry_run {
            write_summary_json(output, name, lines.len(), emitted).at("-")?;
        }
    } else if args.summary || args.dry_run {
        println!("{} lines read, {} written", lines.len(), emitted);
    }
    output.flush().at("-")?;
    Ok(Status::from_found(emitted > 0))
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config(unique: bool) -> OutputConfig<'static> {
        OutputConfig {
            format: Format::Text,
            unique,
            terminator: Terminator::Newline,
            path: "-",
        }
    }

    fn lines_of(data: &[u8]) -> BytesCowsAuto<'_> {
        collect_lines(data, Newlines::Lf).unwrap()
    }

    fn sort_to_string(data: &[u8], reverse: bool, unique: bool, ignore_case: bool) -> String {
        let lines = lines_of(data);
        let order = SortOrder {
            ignore_case,
            reverse,
        };
        let permutation = sorted_order(&lines, order).unwrap();
        let mut out = Vec::new();
        write_sorted(&lines, &permutation, order, &text_config(unique), &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn sorts_lines_ascending() {
        assert_eq!(
            sort_to_string(b"banana\napple\ncherry\n", false, false, false),
            "apple\nbanana\ncherry\n"
        );
    }

    #[test]
    fn sorts_lines_descending() {
        assert_eq!(
            sort_to_string(b"apple\nbanana\ncherry\n", true, false, false),
            "cherry\nbanana\napple\n"
        );
    }

    #[test]
    fn sorts_and_deduplicates_lines() {
        assert_eq!(
            sort_to_string(b"b\na\nb\nc\na\n", false, true, false),
            "a\nb\nc\n"
        );
    }

    #[test]
    fn sorts_lines_ignoring_case() {
        // Folding orders "Apple" < "BANANA" < "cherry"; original casing preserved.
        assert_eq!(
            sort_to_string(b"cherry\nApple\nBANANA\n", false, false, true),
            "Apple\nBANANA\ncherry\n"
        );
    }

    #[test]
    fn deduplicates_lines_ignoring_case() {
        assert_eq!(
            sort_to_string(b"Hello\nhello\nWORLD\nworld\n", false, true, true),
            "Hello\nWORLD\n"
        );
    }

    #[test]
    fn sorts_utf8_in_codepoint_order() {
        // 'a' (U+0061) < 'á' (U+00E1) < 'é' (U+00E9) in code-point and unsigned-byte
        // order; StringZilla's `sz_order` compares bytes unsigned, matching `LC_ALL=C
        // sort` and the UTF-8 guarantee that byte order reproduces code-point order.
        assert_eq!(
            sort_to_string("é\na\ná\n".as_bytes(), false, false, false),
            "a\ná\né\n"
        );
    }

    #[test]
    fn reports_first_unsorted_line() {
        let sorted = lines_of(b"a\nb\nc\n");
        assert_eq!(
            check_sorted(
                &sorted,
                SortOrder {
                    ignore_case: false,
                    reverse: false
                }
            ),
            None
        );

        let unsorted = lines_of(b"a\nc\nb\n");
        assert_eq!(
            check_sorted(
                &unsorted,
                SortOrder {
                    ignore_case: false,
                    reverse: false
                }
            ),
            Some(3)
        );
    }

    #[test]
    fn accepts_descending_order_in_check() {
        let desc = lines_of(b"c\nb\na\n");
        assert_eq!(
            check_sorted(
                &desc,
                SortOrder {
                    ignore_case: false,
                    reverse: true
                }
            ),
            None
        );
    }

    #[test]
    fn checks_sorted_order_ignoring_case() {
        // "Apple" < "BANANA" < "cherry" under folding, regardless of input casing.
        let folded = lines_of(b"Apple\nBANANA\ncherry\n");
        assert_eq!(
            check_sorted(
                &folded,
                SortOrder {
                    ignore_case: true,
                    reverse: false
                }
            ),
            None
        );
        let not_folded = lines_of(b"BANANA\nApple\n");
        assert_eq!(
            check_sorted(
                &not_folded,
                SortOrder {
                    ignore_case: true,
                    reverse: false
                }
            ),
            Some(2)
        );
    }

    #[test]
    fn sorts_empty_input_to_empty() {
        assert_eq!(sort_to_string(b"", false, false, false), "");
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
                "reverse",
                "unique",
                "ignore-case",
                "check",
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
        let arguments = ["sz-sort", "f"].into_iter().chain(flags.iter().copied());
        Args::try_parse_from(arguments).is_ok_and(|args| validate(&args).is_ok())
    }

    #[test]
    fn keeps_check_alone_and_rejects_the_flags_it_discards() {
        assert!(accepts(&["--check"]));
        for flags in [
            vec!["--check", "--output", "o"],
            vec!["--check", "--in-place"],
            vec!["--check", "--dry-run"],
            vec!["--check", "--unique"],
            vec!["--check", "--null"],
            vec!["--check", "--format", "json"],
            vec!["--check", "--summary"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
    }

    #[test]
    fn declares_the_destination_conflicts() {
        assert!(accepts(&["--in-place"]));
        for flags in [
            vec!["--in-place", "--output", "o"],
            vec!["--in-place", "--dry-run"],
            vec!["--in-place", "--null"],
            vec!["--in-place", "--format", "json"],
            vec!["--null", "--format", "json"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
        let args = Args::try_parse_from(["sz-sort", "--in-place"]).unwrap();
        assert!(validate(&args).is_err(), "--in-place cannot rewrite stdin");
    }

    #[test]
    fn takes_quiet_alone_and_rejects_what_it_would_suppress() {
        assert!(accepts(&["--quiet"]), "--quiet must stand on its own");
        for flags in [
            vec!["--quiet", "--format", "json"],
            vec!["--quiet", "--null"],
            vec!["--quiet", "--summary"],
            vec!["--quiet", "--output", "o"],
            vec!["--quiet", "--dry-run"],
            vec!["--quiet", "--in-place"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
    }

    #[test]
    fn rejects_summary_where_dry_run_already_prints_it() {
        assert!(!accepts(&["--summary", "--dry-run"]));
    }

    #[test]
    fn closes_a_json_stream_with_its_summary() {
        assert!(accepts(&["--summary", "--format", "json"]));
        let lines = lines_of(b"b\na\n");
        let order = SortOrder {
            ignore_case: false,
            reverse: false,
        };
        let permutation = sorted_order(&lines, order).unwrap();
        let config = OutputConfig {
            format: Format::Json,
            unique: false,
            terminator: Terminator::Newline,
            path: "f.txt",
        };
        let mut output = Vec::new();

        write_sorted(&lines, &permutation, order, &config, &mut output).unwrap();

        let text = String::from_utf8(output).unwrap();
        let records: Vec<_> = text.lines().collect();
        assert_eq!(records.len(), 3);
        assert!(records[2].contains(r#""type":"summary""#));
        assert!(records[2].contains(r#""written_lines":2,"total_lines":2"#));
    }
}

// endregion: Tests
