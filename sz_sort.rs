//! SIMD-accelerated line sorting utility
//!
//! A faster, Unicode-correct replacement for `sort`, built on StringZilla's
//! `argsort` (case-insensitive folding and reversal happen inside the kernel).
//! Comparison is unsigned byte-wise, which for valid UTF-8 is identical to Unicode
//! code-point order (UTF-8 preserves code-point ordering under unsigned byte
//! comparison), so `--utf8` only affects newline handling.
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

use std::cmp::Ordering;
use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz;

mod shared;
use shared::*;

// region: Sorting

/// Compare two lines under the active ordering. Case-insensitive comparison uses
/// StringZilla's on-the-fly Unicode folding — no materialized keys.
#[inline]
fn line_order(a: &[u8], b: &[u8], ignore_case: bool) -> Ordering {
    if ignore_case {
        sz::utf8_uncased_order(a, b)
    } else {
        a.cmp(b)
    }
}

/// Compute the sorted permutation of `lines` directly via StringZilla's `argsort`,
/// which folds case and reverses inside the kernel — so the case-insensitive path
/// no longer materializes a folded key per line.
fn sorted_order(lines: &[&[u8]], ignore_case: bool, reverse: bool) -> Vec<sz::SortedIdx> {
    let mut order = vec![0usize; lines.len()];
    let mut options = sz::ArgsortOptions::default();
    if ignore_case {
        options = options.uncased();
    }
    if reverse {
        options = options.reversed();
    }
    if let Err(status) = sz::argsort_by(|i| lines[i], &mut order, options) {
        eprintln!("Error: sort failed ({:?})", status);
        process::exit(1);
    }
    order
}

/// Write the sorted lines, skipping adjacent fold-equal lines when `unique` is set.
fn write_sorted(
    lines: &[&[u8]],
    order: &[sz::SortedIdx],
    unique: bool,
    ignore_case: bool,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut prev: Option<&[u8]> = None;
    for &i in order {
        let line = lines[i];
        if unique {
            if prev.is_some_and(|p| line_order(p, line, ignore_case) == Ordering::Equal) {
                continue;
            }
            prev = Some(line);
        }
        output.write_all(line)?;
        output.write_all(b"\n")?;
    }
    output.flush()
}

/// Check whether `lines` are already in sorted order. Returns the 1-based index
/// of the first line that breaks the order, or `None` if fully sorted.
fn check_sorted(lines: &[&[u8]], ignore_case: bool, reverse: bool) -> Option<usize> {
    for i in 1..lines.len() {
        let ord = line_order(lines[i - 1], lines[i], ignore_case);
        let in_order = if reverse {
            ord != Ordering::Less
        } else {
            ord != Ordering::Greater
        };
        if !in_order {
            return Some(i + 1);
        }
    }
    None
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
}

fn main() {
    let args = Args::parse();

    // UTF-8 mode is implicit when case-insensitive (case folding requires UTF-8).
    let utf8_mode = args.utf8 || args.ignore_case;

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };
    let data = input.as_bytes();

    let lines: Vec<&[u8]> = LineIter::new(data, utf8_mode).collect();

    if args.check {
        match check_sorted(&lines, args.ignore_case, args.reverse) {
            None => process::exit(0),
            Some(line_no) => {
                let name = args.input.as_deref().unwrap_or("-");
                eprintln!("sz-sort: {}:{}: disorder", name, line_no);
                process::exit(1);
            }
        }
    }

    let order = sorted_order(&lines, args.ignore_case, args.reverse);

    let mut output = match get_output(args.output.as_deref()) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("Error opening output: {}", e);
            process::exit(1);
        }
    };

    if let Err(e) = write_sorted(&lines, &order, args.unique, args.ignore_case, &mut output) {
        if e.kind() == io::ErrorKind::BrokenPipe {
            process::exit(0);
        }
        eprintln!("Error writing output: {}", e);
        process::exit(1);
    }
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_of(data: &[u8]) -> Vec<&[u8]> {
        LineIter::new(data, false).collect()
    }

    fn sort_to_string(data: &[u8], reverse: bool, unique: bool, ignore_case: bool) -> String {
        let lines = lines_of(data);
        let order = sorted_order(&lines, ignore_case, reverse);
        let mut out = Vec::new();
        write_sorted(&lines, &order, unique, ignore_case, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn basic_sort() {
        assert_eq!(
            sort_to_string(b"banana\napple\ncherry\n", false, false, false),
            "apple\nbanana\ncherry\n"
        );
    }

    #[test]
    fn reverse_sort() {
        assert_eq!(
            sort_to_string(b"apple\nbanana\ncherry\n", true, false, false),
            "cherry\nbanana\napple\n"
        );
    }

    #[test]
    fn unique_sort() {
        assert_eq!(
            sort_to_string(b"b\na\nb\nc\na\n", false, true, false),
            "a\nb\nc\n"
        );
    }

    #[test]
    fn ignore_case_sort() {
        // Folding orders "Apple" < "BANANA" < "cherry"; original casing preserved.
        assert_eq!(
            sort_to_string(b"cherry\nApple\nBANANA\n", false, false, true),
            "Apple\nBANANA\ncherry\n"
        );
    }

    #[test]
    fn ignore_case_unique() {
        assert_eq!(
            sort_to_string(b"Hello\nhello\nWORLD\nworld\n", false, true, true),
            "Hello\nWORLD\n"
        );
    }

    #[test]
    fn utf8_byte_order_is_codepoint_order() {
        // 'a' (U+0061) < 'á' (U+00E1) < 'é' (U+00E9) in code-point and unsigned-byte
        // order; StringZilla's `sz_order` compares bytes unsigned, matching `LC_ALL=C
        // sort` and the UTF-8 guarantee that byte order reproduces code-point order.
        assert_eq!(
            sort_to_string("é\na\ná\n".as_bytes(), false, false, false),
            "a\ná\né\n"
        );
    }

    #[test]
    fn check_detects_disorder() {
        let sorted = lines_of(b"a\nb\nc\n");
        assert_eq!(check_sorted(&sorted, false, false), None);

        let unsorted = lines_of(b"a\nc\nb\n");
        assert_eq!(check_sorted(&unsorted, false, false), Some(3));
    }

    #[test]
    fn check_reverse() {
        let desc = lines_of(b"c\nb\na\n");
        assert_eq!(check_sorted(&desc, false, true), None);
    }

    #[test]
    fn ignore_case_check() {
        // "Apple" < "BANANA" < "cherry" under folding, regardless of input casing.
        let folded = lines_of(b"Apple\nBANANA\ncherry\n");
        assert_eq!(check_sorted(&folded, true, false), None);
        let not_folded = lines_of(b"BANANA\nApple\n");
        assert_eq!(check_sorted(&not_folded, true, false), Some(2));
    }

    #[test]
    fn empty_input() {
        assert_eq!(sort_to_string(b"", false, false, false), "");
    }
}

// endregion: Tests
