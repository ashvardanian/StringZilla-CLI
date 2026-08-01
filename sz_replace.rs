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

use std::fs;
use std::io::{self, Write};

use clap::Parser;
use stringzilla::sz::{find, utf8_uncased_search};

mod shared;
use shared::*;

/// Replace substrings in files
#[derive(Parser)]
#[command(name = "sz-replace")]
#[command(version, about = "SIMD-accelerated substring replacement", long_about = None)]
#[command(group(clap::ArgGroup::new("sink").multiple(true).args(["output", "in_place", "dry_run"])))]
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
    #[arg(long = "in-place")]
    in_place: bool,

    /// Case-insensitive search (full Unicode case folding)
    #[arg(short = 'i', long)]
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

    /// Emit a JSON Lines summary of the replacement; requires -o, --in-place, or -n
    #[arg(long, requires = "sink", help_heading = "Output Formats")]
    json: bool,
}

/// Replace all occurrences of `pattern` with `replacement`, streaming the output
/// into `out` (no full-file buffer). Returns the replacement count.
///
/// Case-insensitive matching uses StringZilla's full-Unicode `utf8_uncased_search`
/// (no whole-file `to_ascii_lowercase` copy), and advances by the matched length —
/// which case folding may make differ from the pattern length (e.g. ß ↔ SS).
fn replace_all_to(
    data: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
    out: &mut dyn Write,
) -> io::Result<usize> {
    if pattern.is_empty() {
        out.write_all(data)?;
        return Ok(0);
    }

    let mut pos = 0;
    let mut count = 0;
    while pos < data.len() {
        let matched = if ignore_case {
            utf8_uncased_search(&data[pos..], pattern)
        } else {
            find(&data[pos..], pattern).map(|off| (off, pattern.len()))
        };

        match matched {
            Some((offset, match_len)) => {
                let match_pos = pos + offset;
                out.write_all(&data[pos..match_pos])?;
                out.write_all(replacement)?;
                count += 1;
                pos = match_pos + match_len;
            }
            None => {
                out.write_all(&data[pos..])?;
                break;
            }
        }
    }
    Ok(count)
}

fn main() {
    let args = Args::parse();

    let mut stdout = io::stdout();

    if args.pattern.is_empty() {
        eprintln!("Error: pattern cannot be empty");
        ExitCode::Error.exit(&mut stdout);
    }

    if args.in_place && args.input.is_none() {
        eprintln!("Error: --in-place requires an input file (cannot use stdin)");
        ExitCode::Error.exit(&mut stdout);
    }

    if args.in_place && args.input == Some("-".to_string()) {
        eprintln!("Error: --in-place cannot be used with stdin");
        ExitCode::Error.exit(&mut stdout);
    }

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut stdout, &error, "Error reading input"),
    };

    let data = input.as_bytes();
    let pattern = args.pattern.as_bytes();
    let replacement = args.replacement.as_bytes();

    // Dry run: count via a sink — no output buffer materialized.
    if args.dry_run {
        let count = replace_all_to(
            data,
            pattern,
            replacement,
            args.ignore_case,
            &mut io::sink(),
        )
        .expect("writing to a sink cannot fail");
        if args.json {
            write_summary_json(&mut stdout, &args, count, true);
        } else {
            eprintln!("Dry run: would replace {} occurrence(s)", count);
        }
        ExitCode::Success.exit(&mut stdout);
    }

    let count = if args.in_place {
        // In-place needs the whole transformed buffer before rewriting the file.
        let input_path = args.input.as_ref().unwrap();
        let mut buf = Vec::with_capacity(data.len());
        let count = replace_all_to(data, pattern, replacement, args.ignore_case, &mut buf)
            .expect("writing to a Vec cannot fail");
        if let Err(error) = fs::write(input_path, &buf) {
            exit_with_error(&mut stdout, &error, "Error writing to file");
        }
        count
    } else {
        // Stream directly to stdout / -o — no full-output buffer.
        let mut output = match get_output(args.output.as_deref()) {
            Ok(output) => output,
            Err(error) => exit_with_error(&mut stdout, &error, "Error opening output"),
        };
        let count = match replace_all_to(data, pattern, replacement, args.ignore_case, &mut *output)
        {
            Ok(count) => count,
            Err(error) => exit_on_write_error(&mut *output, &error, "Error writing output"),
        };
        if let Err(error) = output.flush() {
            if error.kind() != io::ErrorKind::BrokenPipe {
                exit_with_error(&mut stdout, &error, "Error flushing output");
            }
        }
        count
    };

    if args.json {
        write_summary_json(&mut stdout, &args, count, false);
    } else if args.count {
        eprintln!("Replaced {} occurrence(s)", count);
    }
}

/// Write the single summary record. The transformed bytes never share stdout with
/// it, because `--json` requires `-o`, `-i`, or `-n`.
fn write_summary_json(output: &mut dyn Write, args: &Args, replacements: usize, dry_run: bool) {
    let path = args.input.as_deref().unwrap_or("-");
    let _ = output.write_all(br#"{"type":"summary","data":{"path":"#);
    let _ = json_text_field_to(output, path.as_bytes());
    let _ = writeln!(
        output,
        r#","replacements":{},"dry_run":{}}}}}"#,
        replacements, dry_run
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: run the streaming replace into a buffer and return it.
    fn replace_all(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        ignore_case: bool,
    ) -> (Vec<u8>, usize) {
        let mut buf = Vec::new();
        let count = replace_all_to(data, pattern, replacement, ignore_case, &mut buf).unwrap();
        (buf, count)
    }

    #[test]
    fn replaces_all_occurrences() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi world hi");
    }

    #[test]
    fn leaves_text_unchanged_without_matches() {
        let data = b"hello world";
        let (result, count) = replace_all(data, b"foo", b"bar", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn replaces_all_ignoring_case() {
        let data = b"Hello HELLO hello";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn ignores_empty_pattern() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"", b"x", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn replaces_with_longer_string() {
        let data = b"a b a";
        let (result, count) = replace_all(data, b"a", b"foo", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"foo b foo");
    }

    #[test]
    fn replaces_with_shorter_string() {
        let data = b"hello hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi hi");
    }

    #[test]
    fn replaces_non_overlapping_matches() {
        let data = b"aaa";
        let (result, count) = replace_all(data, b"aa", b"b", false);

        // Should replace first match, then continue after it (non-overlapping)
        assert_eq!(count, 1);
        assert_eq!(result, b"ba");
    }

    #[test]
    fn replaces_multibyte_utf8_pattern() {
        let data = "héllo wörld héllo".as_bytes();
        let (result, count) = replace_all(data, "héllo".as_bytes(), "hi".as_bytes(), false);

        assert_eq!(count, 2);
        assert_eq!(result, "hi wörld hi".as_bytes());
    }

    #[test]
    fn replaces_at_start_and_end() {
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
    fn replaces_whole_input() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"hello", b"goodbye", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"goodbye");
    }

    #[test]
    fn deletes_pattern_with_empty_replacement() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"", false);

        assert_eq!(count, 2);
        assert_eq!(result, b" world ");
    }

    #[test]
    fn keeps_literal_replacement_when_ignoring_case() {
        // Case insensitive finds, but replacement is literal
        let data = b"HELLO hello HeLLo";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn replaces_single_character() {
        let data = b"a b a c a";
        let (result, count) = replace_all(data, b"a", b"x", false);

        assert_eq!(count, 3);
        assert_eq!(result, b"x b x c x");
    }

    #[test]
    fn replaces_across_multiple_lines() {
        let data = b"line1\nline2\nline1\n";
        let (result, count) = replace_all(data, b"line1", b"first", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"first\nline2\nfirst\n");
    }
}
