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
//! sz-sort -r file.txt
//!
//! # Sort and drop duplicate lines (like `sort -u`)
//! sz-sort -u file.txt
//!
//! # Case-insensitive sort (full Unicode case folding)
//! sz-sort -i file.txt
//!
//! # Write to a file instead of stdout
//! sz-sort file.txt -o sorted.txt
//!
//! # Verify a file is already sorted (exit 1 if not)
//! sz-sort -c file.txt
//! ```

use std::borrow::Cow;
use std::cmp::Ordering;
use std::io::{self, Write};

use clap::Parser;
use stringtape::{BytesCowsAuto, StringTapeError};
use stringzilla::sz;

mod shared;
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
    json: bool,
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
    if config.json {
        output.write_all(br#"{"type":"line","data":{"path":"#)?;
        json_text_field_to(output, config.path.as_bytes())?;
        output.write_all(br#","lines":"#)?;
        json_text_field_to(output, line)?;
        write!(output, r#","line_number":{}}}}}"#, position + 1)?;
        return output.write_all(b"\n");
    }
    output.write_all(line)?;
    output.write_all(&[config.terminator.as_byte()])
}

/// Write the lines in permutation order, dropping adjacent equals under `--unique`.
fn write_sorted(
    lines: &BytesCowsAuto<'_>,
    permutation: &[sz::SortedIdx],
    order: SortOrder,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut previous: Option<&[u8]> = None;
    let mut written = 0;
    for &index in permutation {
        let line = line_at(lines, index);
        if config.unique
            && previous.is_some_and(|kept| order.compare(kept, line) == Ordering::Equal)
        {
            continue;
        }
        previous = Some(line);
        write_line(output, config, line, written)?;
        written += 1;
    }
    output.flush()
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

/// Sort lines in a file or stream
#[derive(Parser)]
#[command(name = "sz-sort")]
#[command(version, about = "SIMD-accelerated line sorting", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output file (use '-' or omit for stdout)
    #[arg(short, long)]
    output: Option<String>,

    /// Reverse the result (descending order)
    #[arg(short, long)]
    reverse: bool,

    /// Drop duplicate lines, keeping one of each (like `sort -u`)
    #[arg(short, long)]
    unique: bool,

    /// Case-insensitive sort (full Unicode case folding)
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Check whether the input is already sorted; exit 1 if not (no output)
    #[arg(short = 'c', long)]
    check: bool,

    /// Enable UTF-8 mode (handle Unicode newlines: CR, CRLF, NEL, LS, PS)
    #[arg(long)]
    utf8: bool,

    /// Emit JSON Lines, one record per emitted line
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// NUL-terminate each output line instead of newline
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress output; with --check, report order through the exit code only
    #[arg(
        short = 'q',
        long,
        requires = "check",
        conflicts_with = "output",
        help_heading = "Output Formats"
    )]
    quiet: bool,
}

fn main() {
    let args = Args::parse();
    let mut stdout = io::stdout();

    // UTF-8 mode is implicit when case-insensitive (case folding requires UTF-8).
    let utf8_mode = args.utf8 || args.ignore_case;

    let input = get_input(args.input.as_deref())
        .unwrap_or_else(|error| exit_with_error(&mut stdout, &error, "Error reading input"));
    let data = input.as_bytes();

    let order = SortOrder {
        ignore_case: args.ignore_case,
        reverse: args.reverse,
    };
    let lines = collect_lines(data, Newlines::from_utf8(utf8_mode)).unwrap_or_else(|error| {
        eprintln!("Error indexing lines: {:?}", error);
        ExitCode::Error.exit(&mut stdout)
    });
    let name = args.input.as_deref().unwrap_or("-");

    if args.check {
        match check_sorted(&lines, order) {
            None => ExitCode::Success.exit(&mut stdout),
            Some(line_number) => {
                if !args.quiet {
                    eprintln!("sz-sort: {}:{}: disorder", name, line_number);
                }
                ExitCode::NoResult.exit(&mut stdout);
            }
        }
    }

    let permutation = sorted_order(&lines, order).unwrap_or_else(|status| {
        eprintln!("Error sorting: {:?}", status);
        ExitCode::Error.exit(&mut stdout)
    });

    let mut output = get_output(args.output.as_deref())
        .unwrap_or_else(|error| exit_with_error(&mut stdout, &error, "Error opening output"));

    let config = OutputConfig {
        json: args.json,
        unique: args.unique,
        terminator: Terminator::from_null(args.null),
        path: name,
    };

    if let Err(error) = write_sorted(&lines, &permutation, order, &config, &mut output) {
        exit_on_write_error(&mut output, &error, "Error writing output");
    }
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config(unique: bool) -> OutputConfig<'static> {
        OutputConfig {
            json: false,
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
}

// endregion: Tests
