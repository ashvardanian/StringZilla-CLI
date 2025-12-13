//! SIMD-accelerated substring search utility
//!
//! A grep-like tool with simpler syntax, using StringZilla for fast searching.
//! Unlike grep, uses literal substring matching (not regex) for maximum speed.
//!
//! # Examples
//!
//! ```bash
//! # Find lines containing "error"
//! sz-find error log.txt
//!
//! # Case-insensitive search
//! sz-find -i ERROR log.txt
//!
//! # Show line numbers
//! sz-find -n error log.txt
//!
//! # Count matches only
//! sz-find -c error log.txt
//!
//! # Show context lines
//! sz-find -C 2 error log.txt
//!
//! # From stdin
//! cat log.txt | sz-find error
//! ```

use std::collections::VecDeque;
use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz::{find, rfind, utf8_case_insensitive_find};

mod shared;
use shared::*;

/// Search for substring in files
#[derive(Parser)]
#[command(name = "sz-find")]
#[command(version, about = "SIMD-accelerated substring search", long_about = None)]
struct Args {
    /// Substring to search for
    pattern: String,

    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Case-insensitive search
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Show line numbers
    #[arg(short = 'n', long)]
    line_number: bool,

    /// Count matching lines only
    #[arg(short = 'c', long)]
    count: bool,

    /// Show only filenames with matches
    #[arg(short = 'l', long)]
    files_with_matches: bool,

    /// Lines of context before match
    #[arg(short = 'B', long, default_value = "0")]
    before_context: usize,

    /// Lines of context after match
    #[arg(short = 'A', long, default_value = "0")]
    after_context: usize,

    /// Lines of context before and after match
    #[arg(short = 'C', long)]
    context: Option<usize>,

    /// Enable UTF-8 mode for proper Unicode handling
    #[arg(long)]
    utf8: bool,

    /// Allow pattern to match across line boundaries
    #[arg(short = 'm', long)]
    multiline: bool,
}

/// Check if a line contains a pattern
fn line_contains(line: &[u8], pattern: &[u8], ignore_case: bool) -> bool {
    if ignore_case {
        utf8_case_insensitive_find(line, pattern).is_some()
    } else {
        find(line, pattern).is_some()
    }
}

/// Search for pattern in data using intra-line matching (lazy iteration)
fn search_intraline<'a, I>(
    iter: I,
    pattern: &[u8],
    output: &mut dyn Write,
    args: &Args,
    before_context: usize,
    after_context: usize,
) -> io::Result<usize>
where
    I: Iterator<Item = &'a [u8]>,
{
    let mut match_count = 0;
    let mut context_before: VecDeque<(usize, Vec<u8>)> =
        VecDeque::with_capacity(before_context + 1);
    let mut pending_after: usize = 0;
    let mut last_printed_line: Option<usize> = None;

    for (line_num, line) in iter.enumerate() {
        let line_num_1based = line_num + 1;
        let is_match = line_contains(line, pattern, args.ignore_case);

        if is_match {
            match_count += 1;

            if args.count || args.files_with_matches {
                // Just counting, don't print
                continue;
            }

            // Print buffered context_before lines (that haven't been printed yet)
            for (ctx_line_num, ctx_line) in context_before.iter() {
                if last_printed_line.map_or(true, |lp| *ctx_line_num > lp) {
                    print_line(output, ctx_line, *ctx_line_num, args, false)?;
                    last_printed_line = Some(*ctx_line_num);
                }
            }

            // Print matching line
            print_line(output, line, line_num_1based, args, true)?;
            last_printed_line = Some(line_num_1based);

            // Set pending_after counter for following lines
            pending_after = after_context;
        } else if pending_after > 0 {
            // Print as after-context
            print_line(output, line, line_num_1based, args, false)?;
            last_printed_line = Some(line_num_1based);
            pending_after -= 1;
        }

        // Maintain rolling context_before buffer
        if before_context > 0 {
            if context_before.len() >= before_context {
                context_before.pop_front();
            }
            context_before.push_back((line_num_1based, line.to_vec()));
        }
    }

    Ok(match_count)
}

/// Search for pattern in data using multi-line matching (whole buffer search)
fn search_multiline(
    data: &[u8],
    pattern: &[u8],
    output: &mut dyn Write,
    args: &Args,
    before_context: usize,
    after_context: usize,
) -> io::Result<usize> {
    let mut match_count = 0;
    let mut pos = 0;
    let mut last_printed_end: usize = 0;

    while pos < data.len() {
        // Find next match
        let match_result = if args.ignore_case {
            utf8_case_insensitive_find(&data[pos..], pattern).map(|(off, len)| (pos + off, len))
        } else {
            find(&data[pos..], pattern).map(|off| (pos + off, pattern.len()))
        };

        let (match_offset, match_len) = match match_result {
            Some(m) => m,
            None => break,
        };

        match_count += 1;

        if args.count || args.files_with_matches {
            pos = match_offset + match_len;
            continue;
        }

        // Find line boundaries around the match
        let line_start = if match_offset == 0 {
            0
        } else {
            rfind(&data[..match_offset], b"\n")
                .map(|i| i + 1)
                .unwrap_or(0)
        };

        // Find end of the line containing the end of the match
        let match_end = match_offset + match_len;
        let line_end = find(&data[match_end..], b"\n")
            .map(|i| match_end + i)
            .unwrap_or(data.len());

        // Expand for before_context
        let mut output_start = line_start;
        if before_context > 0 {
            let mut count = 0;
            let mut search_pos = line_start;
            while count < before_context && search_pos > 0 {
                search_pos = if search_pos <= 1 {
                    0
                } else {
                    rfind(&data[..search_pos - 1], b"\n")
                        .map(|i| i + 1)
                        .unwrap_or(0)
                };
                count += 1;
            }
            output_start = search_pos;
        }

        // Expand for after_context
        let mut output_end = line_end;
        if after_context > 0 {
            let mut count = 0;
            let mut search_pos = line_end;
            while count < after_context && search_pos < data.len() {
                if let Some(next_nl) = find(&data[search_pos + 1..], b"\n") {
                    search_pos = search_pos + 1 + next_nl;
                } else {
                    search_pos = data.len();
                    break;
                }
                count += 1;
            }
            output_end = search_pos;
        }

        // Avoid printing overlapping regions
        let actual_start = output_start.max(last_printed_end);
        if actual_start < output_end {
            // Print with line numbers if requested
            if args.line_number {
                // Count line number at actual_start
                let line_num = data[..actual_start].iter().filter(|&&b| b == b'\n').count() + 1;
                let region = &data[actual_start..output_end];
                for (i, line) in region.split(|&b| b == b'\n').enumerate() {
                    if !line.is_empty() || i == 0 {
                        write!(output, "{}:", line_num + i)?;
                        output.write_all(line)?;
                        output.write_all(b"\n")?;
                    }
                }
            } else {
                output.write_all(&data[actual_start..output_end])?;
                if output_end < data.len() && data[output_end] != b'\n' {
                    output.write_all(b"\n")?;
                }
            }
            last_printed_end = output_end;
        }

        pos = match_end;
    }

    Ok(match_count)
}

/// Print a single line with optional line numbers
fn print_line(
    output: &mut dyn Write,
    line: &[u8],
    line_num: usize,
    args: &Args,
    _is_match: bool,
) -> io::Result<()> {
    if args.line_number {
        write!(output, "{}:", line_num)?;
    }
    output.write_all(line)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() {
    let args = Args::parse();

    // Handle -C flag (context on both sides)
    let (before_context, after_context) = if let Some(ctx) = args.context {
        (ctx, ctx)
    } else {
        (args.before_context, args.after_context)
    };

    if args.pattern.is_empty() {
        eprintln!("Error: pattern cannot be empty");
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
    let pattern = args.pattern.as_bytes();
    let mut output = io::stdout();

    let search_result = if args.multiline {
        // Multi-line mode: pattern can span lines
        search_multiline(
            data,
            pattern,
            &mut output,
            &args,
            before_context,
            after_context,
        )
    } else {
        // Intra-line mode: pattern must be within a single line
        if args.utf8 {
            search_intraline(
                Utf8LineIterator::new(data),
                pattern,
                &mut output,
                &args,
                before_context,
                after_context,
            )
        } else {
            search_intraline(
                LineIterator::new(data),
                pattern,
                &mut output,
                &args,
                before_context,
                after_context,
            )
        }
    };

    let match_count = match search_result {
        Ok(count) => count,
        Err(e) => {
            if e.kind() == io::ErrorKind::BrokenPipe {
                process::exit(0);
            }
            eprintln!("Error searching: {}", e);
            process::exit(1);
        }
    };

    // Handle special output modes
    if args.count {
        println!("{}", match_count);
    } else if args.files_with_matches && match_count > 0 {
        let filename = args.input.as_deref().unwrap_or("-");
        println!("{}", filename);
    }

    // Exit with status 1 if no matches found (like grep)
    if match_count == 0 {
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_contains() {
        let line = b"hello world";
        assert!(line_contains(line, b"hello", false));
        assert!(line_contains(line, b"world", false));
        assert!(!line_contains(line, b"foo", false));
    }

    #[test]
    fn test_line_contains_case_insensitive() {
        let line = b"Hello World";
        assert!(line_contains(line, b"hello", true));
        assert!(line_contains(line, b"WORLD", true));
        assert!(!line_contains(line, b"foo", true));
    }

    #[test]
    fn test_search_basic() {
        let data = b"line1\nerror here\nline3\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count =
            search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 1);
        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("error here"));
        assert!(!result.contains("line1"));
    }

    #[test]
    fn test_search_with_line_numbers() {
        let data = b"line1\nerror here\nline3\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: true,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("2:error here"));
    }

    #[test]
    fn test_search_count() {
        let data = b"error1\nok\nerror2\nerror3\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: true,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count =
            search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 3);
    }

    #[test]
    fn test_search_pattern_at_start() {
        let data = b"error at start\nmiddle line\nend\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count =
            search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 1);
        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("error at start"));
    }

    #[test]
    fn test_search_pattern_at_end() {
        let data = b"first\nsecond\nthird has error\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count =
            search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 1);
        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("third has error"));
    }

    #[test]
    fn test_search_with_before_context() {
        let data = b"line1\nline2\nerror here\nline4\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 1,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        search_intraline(LineIterator::new(data), b"error", &mut output, &args, 1, 0).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("line2"));
        assert!(result.contains("error here"));
    }

    #[test]
    fn test_search_with_after_context() {
        let data = b"line1\nerror here\nline3\nline4\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 1,
            context: None,
            utf8: false,
            multiline: false,
        };

        search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 1).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("error here"));
        assert!(result.contains("line3"));
    }

    #[test]
    fn test_search_no_matches() {
        let data = b"nothing to find here\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "missing".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count = search_intraline(
            LineIterator::new(data),
            b"missing",
            &mut output,
            &args,
            0,
            0,
        )
        .unwrap();

        assert_eq!(count, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn test_search_multiple_matches_per_line() {
        // The search function counts lines with matches, not occurrences
        let data = b"error error error\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "error".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: false,
        };

        let count =
            search_intraline(LineIterator::new(data), b"error", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 1); // Only 1 line matches
    }

    #[test]
    fn test_multiline_search() {
        let data = b"hello\nworld\nfoo bar\n";
        let mut output = Vec::new();
        let args = Args {
            pattern: "hello\nworld".to_string(),
            input: None,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: false,
            multiline: true,
        };

        let count = search_multiline(data, b"hello\nworld", &mut output, &args, 0, 0).unwrap();

        assert_eq!(count, 1);
        let result = String::from_utf8(output).unwrap();
        assert!(result.contains("hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_case_insensitive_unicode() {
        let data = "Straße ist hier\n".as_bytes();
        let mut output = Vec::new();
        let args = Args {
            pattern: "STRASSE".to_string(),
            input: None,
            ignore_case: true,
            line_number: false,
            count: false,
            files_with_matches: false,
            before_context: 0,
            after_context: 0,
            context: None,
            utf8: true,
            multiline: false,
        };

        let count = search_intraline(
            Utf8LineIterator::new(data),
            "STRASSE".as_bytes(),
            &mut output,
            &args,
            0,
            0,
        )
        .unwrap();

        assert_eq!(count, 1);
    }
}
