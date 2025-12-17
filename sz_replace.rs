//! SIMD-accelerated substring replacement utility
//!
//! A simpler alternative to sed/awk for substring replacement.
//! Uses StringZilla for fast searching and replacing.
//!
//! # Examples
//!
//! ```bash
//! # Replace all occurrences
//! sz-replace foo bar file.txt
//!
//! # Replace in-place
//! sz-replace -i foo bar file.txt
//!
//! # Case-insensitive replacement
//! sz-replace -i foo bar file.txt
//!
//! # Show count of replacements
//! sz-replace --count foo bar file.txt
//!
//! # From stdin
//! cat file.txt | sz-replace foo bar
//! ```

use std::borrow::Cow;
use std::fs;
use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz::find;

mod shared;
use shared::*;

/// Replace substrings in files
#[derive(Parser)]
#[command(name = "sz-replace")]
#[command(version, about = "SIMD-accelerated substring replacement", long_about = None)]
struct Args {
    /// Substring to search for
    pattern: String,

    /// Replacement string
    replacement: String,

    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output file (default: stdout, conflicts with --in-place)
    #[arg(short = 'o', long, conflicts_with = "in_place")]
    output: Option<String>,

    /// Replace in-place (modifies input file)
    #[arg(short = 'i', long = "in-place")]
    in_place: bool,

    /// Case-insensitive search
    #[arg(short = 'I', long)]
    ignore_case: bool,

    /// Show count of replacements made
    #[arg(short = 'c', long)]
    count: bool,

    /// Dry run - show what would be replaced without making changes
    #[arg(short = 'n', long = "dry-run")]
    dry_run: bool,

    /// Enable UTF-8 mode (validate input)
    #[arg(long)]
    utf8: bool,
}

/// Replace all occurrences of pattern with replacement
fn replace_all(
    data: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
) -> (Vec<u8>, usize) {
    if pattern.is_empty() {
        return (data.to_vec(), 0);
    }

    let mut result = Vec::with_capacity(data.len());
    let mut pos = 0;
    let mut count = 0;

    // For case-insensitive, we need to work with lowercased versions
    // Use Cow to avoid unnecessary allocations when ignore_case is false
    let search_data: Cow<[u8]> = if ignore_case {
        Cow::Owned(data.to_ascii_lowercase())
    } else {
        Cow::Borrowed(data)
    };
    let search_pattern: Cow<[u8]> = if ignore_case {
        Cow::Owned(pattern.to_ascii_lowercase())
    } else {
        Cow::Borrowed(pattern)
    };

    while pos < data.len() {
        if let Some(found) = find(&search_data[pos..], &search_pattern) {
            let match_pos = pos + found;

            // Copy everything before the match
            result.extend_from_slice(&data[pos..match_pos]);

            // Add replacement
            result.extend_from_slice(replacement);

            count += 1;
            pos = match_pos + pattern.len();
        } else {
            // No more matches, copy rest of data
            result.extend_from_slice(&data[pos..]);
            break;
        }
    }

    (result, count)
}

fn main() {
    let args = Args::parse();

    if args.pattern.is_empty() {
        eprintln!("Error: pattern cannot be empty");
        process::exit(1);
    }

    if args.in_place && args.input.is_none() {
        eprintln!("Error: --in-place requires an input file (cannot use stdin)");
        process::exit(1);
    }

    if args.in_place && args.input == Some("-".to_string()) {
        eprintln!("Error: --in-place cannot be used with stdin");
        process::exit(1);
    }

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };

    let data = input.as_bytes();

    // Validate UTF-8 if requested
    if args.utf8 {
        if let Err(e) = validate_utf8(data) {
            eprintln!("Error: {}", e);
            process::exit(1);
        }

        // Also validate replacement string
        if let Err(e) = validate_utf8(args.replacement.as_bytes()) {
            eprintln!("Error in replacement string: {}", e);
            process::exit(1);
        }
    }

    // Perform replacement
    let (result, count) = replace_all(
        data,
        args.pattern.as_bytes(),
        args.replacement.as_bytes(),
        args.ignore_case,
    );

    // Show count if requested
    if args.count {
        eprintln!("Replaced {} occurrence(s)", count);
    }

    // Handle dry run
    if args.dry_run {
        eprintln!("Dry run: would replace {} occurrence(s)", count);
        process::exit(0);
    }

    // Write output
    if args.in_place {
        // Write back to input file
        let input_path = args.input.as_ref().unwrap();
        if let Err(e) = fs::write(input_path, &result) {
            eprintln!("Error writing to file: {}", e);
            process::exit(1);
        }
    } else {
        // Write to output file or stdout
        let mut output = match get_output(args.output.as_deref()) {
            Ok(output) => output,
            Err(e) => {
                eprintln!("Error opening output: {}", e);
                process::exit(1);
            }
        };

        if let Err(e) = output.write_all(&result) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                process::exit(0);
            }
            eprintln!("Error writing output: {}", e);
            process::exit(1);
        }

        if let Err(e) = output.flush() {
            if e.kind() != io::ErrorKind::BrokenPipe {
                eprintln!("Error flushing output: {}", e);
                process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_all_basic() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi world hi");
    }

    #[test]
    fn test_replace_all_no_matches() {
        let data = b"hello world";
        let (result, count) = replace_all(data, b"foo", b"bar", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn test_replace_all_case_insensitive() {
        let data = b"Hello HELLO hello";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn test_replace_all_empty_pattern() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"", b"x", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_replace_all_longer_replacement() {
        let data = b"a b a";
        let (result, count) = replace_all(data, b"a", b"foo", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"foo b foo");
    }

    #[test]
    fn test_replace_all_shorter_replacement() {
        let data = b"hello hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi hi");
    }

    #[test]
    fn test_replace_overlapping() {
        let data = b"aaa";
        let (result, count) = replace_all(data, b"aa", b"b", false);

        // Should replace first match, then continue after it (non-overlapping)
        assert_eq!(count, 1);
        assert_eq!(result, b"ba");
    }

    #[test]
    fn test_replace_utf8_content() {
        let data = "héllo wörld héllo".as_bytes();
        let (result, count) = replace_all(data, "héllo".as_bytes(), "hi".as_bytes(), false);

        assert_eq!(count, 2);
        assert_eq!(result, "hi wörld hi".as_bytes());
    }

    #[test]
    fn test_replace_at_boundaries() {
        // Pattern at start
        let data = b"hello world";
        let (result, count) = replace_all(data, b"hello", b"hi", false);
        assert_eq!(count, 1);
        assert_eq!(result, b"hi world");

        // Pattern at end
        let data = b"hello world";
        let (result, count) = replace_all(data, b"world", b"there", false);
        assert_eq!(count, 1);
        assert_eq!(result, b"hello there");
    }

    #[test]
    fn test_replace_entire_content() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"hello", b"goodbye", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"goodbye");
    }

    #[test]
    fn test_replace_with_empty() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"", false);

        assert_eq!(count, 2);
        assert_eq!(result, b" world ");
    }

    #[test]
    fn test_replace_case_insensitive_preserves_replacement() {
        // Case insensitive finds, but replacement is literal
        let data = b"HELLO hello HeLLo";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn test_replace_single_char() {
        let data = b"a b a c a";
        let (result, count) = replace_all(data, b"a", b"x", false);

        assert_eq!(count, 3);
        assert_eq!(result, b"x b x c x");
    }

    #[test]
    fn test_replace_newlines() {
        let data = b"line1\nline2\nline1\n";
        let (result, count) = replace_all(data, b"line1", b"first", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"first\nline2\nfirst\n");
    }
}
