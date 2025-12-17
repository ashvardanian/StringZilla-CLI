//! SIMD-accelerated row extraction utility
//!
//! A simpler, faster replacement for `sed -n 'Np'`, `head`, `tail`, and `awk 'NR==N'`.
//! Uses StringZilla for fast line scanning.
//!
//! # Examples
//!
//! ```bash
//! # Extract line 5
//! sz-rows -r 5 file.txt
//!
//! # Extract lines 10-20
//! sz-rows -r 10-20 file.txt
//!
//! # Extract first 10 lines (like head -n 10)
//! sz-rows -r 1-10 file.txt
//!
//! # Extract last 10 lines (like tail -n 10)
//! sz-rows --tail 10 file.txt
//!
//! # Extract multiple specific lines
//! sz-rows -r 1,5,10 file.txt
//!
//! # Extract every 5th line
//! sz-rows --every 5 file.txt
//!
//! # From stdin
//! cat file.txt | sz-rows -r 5-10
//! ```

use std::collections::HashSet;
use std::io::{self, Write};
use std::process;

use clap::Parser;

mod shared;
use shared::*;

/// Extract rows from text files
#[derive(Parser)]
#[command(name = "sz-rows")]
#[command(version, about = "SIMD-accelerated row extraction (like sed -n, head, tail)", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Row(s) to extract: single (5), list (1,5,10), or range (10-20)
    #[arg(short = 'r', long = "rows", conflicts_with_all = ["tail", "every"])]
    rows: Option<String>,

    /// Extract last N lines (like tail -n)
    #[arg(long = "tail", conflicts_with_all = ["rows", "every"])]
    tail: Option<usize>,

    /// Extract every Nth line
    #[arg(long = "every", conflicts_with_all = ["rows", "tail"])]
    every: Option<usize>,

    /// Show line numbers in output
    #[arg(short = 'n', long = "line-numbers")]
    line_numbers: bool,

    /// Enable UTF-8 mode (validate input)
    #[arg(long)]
    utf8: bool,
}

/// Represents which rows to extract
enum RowSelector {
    /// Specific row indices (0-based)
    Indices(HashSet<usize>),
    /// Range of rows (0-based, inclusive)
    Range(usize, usize),
    /// Last N rows
    Tail(usize),
    /// Every Nth row (1 = all, 2 = every other, etc.)
    Every(usize),
}

/// Parse row specification into a RowSelector
fn parse_rows(spec: &str) -> Result<RowSelector, String> {
    // Check if it's a simple range (no commas)
    if !spec.contains(',') && spec.contains('-') {
        let parts: Vec<&str> = spec.split('-').collect();
        if parts.len() == 2 {
            let start: usize = parts[0]
                .trim()
                .parse()
                .map_err(|_| format!("Invalid row number: {}", parts[0]))?;
            let end: usize = parts[1]
                .trim()
                .parse()
                .map_err(|_| format!("Invalid row number: {}", parts[1]))?;
            if start == 0 || end == 0 {
                return Err("Row numbers start at 1".to_string());
            }
            if start > end {
                return Err(format!("Invalid range: {} > {}", start, end));
            }
            return Ok(RowSelector::Range(start - 1, end - 1)); // Convert to 0-based
        }
    }

    // Parse as list of indices (possibly with ranges)
    let mut indices = HashSet::new();

    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            // Range within list: "2-5"
            let parts: Vec<&str> = part.split('-').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid range: {}", part));
            }
            let start: usize = parts[0]
                .parse()
                .map_err(|_| format!("Invalid row number: {}", parts[0]))?;
            let end: usize = parts[1]
                .parse()
                .map_err(|_| format!("Invalid row number: {}", parts[1]))?;
            if start == 0 || end == 0 {
                return Err("Row numbers start at 1".to_string());
            }
            if start > end {
                return Err(format!("Invalid range: {} > {}", start, end));
            }
            for i in start..=end {
                indices.insert(i - 1); // Convert to 0-based
            }
        } else {
            // Single row: "5"
            let num: usize = part
                .parse()
                .map_err(|_| format!("Invalid row number: {}", part))?;
            if num == 0 {
                return Err("Row numbers start at 1".to_string());
            }
            indices.insert(num - 1); // Convert to 0-based
        }
    }

    if indices.is_empty() {
        return Err("No rows specified".to_string());
    }

    Ok(RowSelector::Indices(indices))
}

/// Extract rows using indices or range
fn extract_rows_by_selector(
    data: &[u8],
    selector: &RowSelector,
    show_line_numbers: bool,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut count = 0;

    match selector {
        RowSelector::Indices(indices) => {
            let max_index = *indices.iter().max().unwrap_or(&0);
            for (i, line) in LineIterator::new(data).enumerate() {
                if i > max_index {
                    break; // No need to continue past the last requested index
                }
                if indices.contains(&i) {
                    if show_line_numbers {
                        write!(output, "{}:", i + 1)?;
                    }
                    output.write_all(line)?;
                    output.write_all(b"\n")?;
                    count += 1;
                }
            }
        }
        RowSelector::Range(start, end) => {
            for (i, line) in LineIterator::new(data).enumerate() {
                if i > *end {
                    break;
                }
                if i >= *start {
                    if show_line_numbers {
                        write!(output, "{}:", i + 1)?;
                    }
                    output.write_all(line)?;
                    output.write_all(b"\n")?;
                    count += 1;
                }
            }
        }
        RowSelector::Tail(n) => {
            // Collect line positions, then output last N
            let lines: Vec<_> = LineIterator::new(data).collect();
            let start = lines.len().saturating_sub(*n);
            for (i, line) in lines[start..].iter().enumerate() {
                if show_line_numbers {
                    write!(output, "{}:", start + i + 1)?;
                }
                output.write_all(line)?;
                output.write_all(b"\n")?;
                count += 1;
            }
        }
        RowSelector::Every(n) => {
            for (i, line) in LineIterator::new(data).enumerate() {
                if (i + 1) % n == 0 {
                    if show_line_numbers {
                        write!(output, "{}:", i + 1)?;
                    }
                    output.write_all(line)?;
                    output.write_all(b"\n")?;
                    count += 1;
                }
            }
        }
    }

    output.flush()?;
    Ok(count)
}

fn main() {
    let args = Args::parse();

    // Determine row selector
    let selector = if let Some(ref rows) = args.rows {
        match parse_rows(rows) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: {}", e);
                process::exit(1);
            }
        }
    } else if let Some(n) = args.tail {
        if n == 0 {
            eprintln!("Error: --tail must be at least 1");
            process::exit(1);
        }
        RowSelector::Tail(n)
    } else if let Some(n) = args.every {
        if n == 0 {
            eprintln!("Error: --every must be at least 1");
            process::exit(1);
        }
        RowSelector::Every(n)
    } else {
        eprintln!("Error: must specify --rows, --tail, or --every");
        process::exit(1);
    };

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };

    let data = input.as_bytes();

    let mut output = io::stdout();

    if let Err(e) = extract_rows_by_selector(data, &selector, args.line_numbers, &mut output) {
        if e.kind() == io::ErrorKind::BrokenPipe {
            process::exit(0);
        }
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rows_single() {
        match parse_rows("5").unwrap() {
            RowSelector::Indices(indices) => {
                assert!(indices.contains(&4)); // 0-based
                assert_eq!(indices.len(), 1);
            }
            _ => panic!("Expected Indices"),
        }
    }

    #[test]
    fn test_parse_rows_list() {
        match parse_rows("1,5,10").unwrap() {
            RowSelector::Indices(indices) => {
                assert!(indices.contains(&0));
                assert!(indices.contains(&4));
                assert!(indices.contains(&9));
                assert_eq!(indices.len(), 3);
            }
            _ => panic!("Expected Indices"),
        }
    }

    #[test]
    fn test_parse_rows_range() {
        match parse_rows("5-10").unwrap() {
            RowSelector::Range(start, end) => {
                assert_eq!(start, 4); // 0-based
                assert_eq!(end, 9);
            }
            _ => panic!("Expected Range"),
        }
    }

    #[test]
    fn test_parse_rows_errors() {
        assert!(parse_rows("0").is_err()); // 0 not allowed
        assert!(parse_rows("10-5").is_err()); // invalid range
        assert!(parse_rows("abc").is_err()); // not a number
    }

    #[test]
    fn test_extract_single_row() {
        let data = b"line1\nline2\nline3\nline4\n";
        let mut output = Vec::new();

        let mut indices = HashSet::new();
        indices.insert(1); // line2
        let selector = RowSelector::Indices(indices);

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 1);
        assert_eq!(output, b"line2\n");
    }

    #[test]
    fn test_extract_range() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(1, 3); // lines 2-4

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline3\nline4\n");
    }

    #[test]
    fn test_extract_tail() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = RowSelector::Tail(2);

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"line4\nline5\n");
    }

    #[test]
    fn test_extract_every() {
        let data = b"line1\nline2\nline3\nline4\nline5\nline6\n";
        let mut output = Vec::new();

        let selector = RowSelector::Every(2); // every 2nd line

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline4\nline6\n");
    }

    #[test]
    fn test_extract_with_line_numbers() {
        let data = b"line1\nline2\nline3\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(0, 1); // lines 1-2

        extract_rows_by_selector(data, &selector, true, &mut output).unwrap();

        assert_eq!(output, b"1:line1\n2:line2\n");
    }

    #[test]
    fn test_extract_multiple_indices() {
        let data = b"a\nb\nc\nd\ne\n";
        let mut output = Vec::new();

        let mut indices = HashSet::new();
        indices.insert(0); // a
        indices.insert(2); // c
        indices.insert(4); // e
        let selector = RowSelector::Indices(indices);

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"a\nc\ne\n");
    }

    #[test]
    fn test_extract_beyond_file() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(0, 100); // Request more than exists

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 2); // Only 2 lines exist
    }

    #[test]
    fn test_tail_larger_than_file() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = RowSelector::Tail(100);

        let count = extract_rows_by_selector(data, &selector, false, &mut output).unwrap();

        assert_eq!(count, 2); // Return all lines
        assert_eq!(output, b"line1\nline2\n");
    }
}
