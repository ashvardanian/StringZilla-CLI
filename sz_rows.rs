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

use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};

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

    /// Enable UTF-8 mode (split on Unicode newlines: CR, CRLF, NEL, LS, PS)
    #[arg(long)]
    utf8: bool,

    /// Emit JSON Lines, one record per output line
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// NUL-terminate each output line instead of newline
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,
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

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    json: bool,
    terminator: Terminator,
    show_line_numbers: bool,
    /// Input name carried into the JSON envelope.
    path: &'a str,
}

/// Write one extracted row. `index` is zero-based; records report it one-based.
fn write_row(
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

    if config.show_line_numbers {
        write!(output, "{}:", index + 1)?;
    }
    output.write_all(line)?;
    output.write_all(&[config.terminator.as_byte()])
}

/// Extract rows using indices or range
fn extract_rows_by_selector(
    data: &[u8],
    selector: &RowSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut count = 0;

    match selector {
        RowSelector::Indices(indices) => {
            let max_index = *indices.iter().max().unwrap_or(&0);
            for (index, line) in LineIter::new(data, newlines).enumerate() {
                if index > max_index {
                    break; // No need to continue past the last requested index
                }
                if indices.contains(&index) {
                    write_row(output, config, line, index)?;
                    count += 1;
                }
            }
        }
        RowSelector::Range(start, end) => {
            for (index, line) in LineIter::new(data, newlines).enumerate() {
                if index > *end {
                    break;
                }
                if index >= *start {
                    write_row(output, config, line, index)?;
                    count += 1;
                }
            }
        }
        RowSelector::Tail(wanted) => {
            // Keep only the last N lines in a ring buffer — O(n) memory, not O(file).
            let wanted = *wanted;
            let mut ring: VecDeque<(usize, &[u8])> = VecDeque::with_capacity(wanted);
            for (index, line) in LineIter::new(data, newlines).enumerate() {
                if wanted > 0 {
                    if ring.len() == wanted {
                        ring.pop_front();
                    }
                    ring.push_back((index, line));
                }
            }
            for (index, line) in ring {
                write_row(output, config, line, index)?;
                count += 1;
            }
        }
        RowSelector::Every(stride) => {
            for (index, line) in LineIter::new(data, newlines).enumerate() {
                if (index + 1) % stride == 0 {
                    write_row(output, config, line, index)?;
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
    let mut output = io::stdout();

    // Determine row selector
    let selector = if let Some(ref rows) = args.rows {
        match parse_rows(rows) {
            Ok(selector) => selector,
            Err(message) => {
                eprintln!("Error: {}", message);
                ExitCode::Error.exit(&mut output);
            }
        }
    } else if let Some(wanted) = args.tail {
        if wanted == 0 {
            eprintln!("Error: --tail must be at least 1");
            ExitCode::Error.exit(&mut output);
        }
        RowSelector::Tail(wanted)
    } else if let Some(stride) = args.every {
        if stride == 0 {
            eprintln!("Error: --every must be at least 1");
            ExitCode::Error.exit(&mut output);
        }
        RowSelector::Every(stride)
    } else {
        eprintln!("Error: must specify --rows, --tail, or --every");
        ExitCode::Error.exit(&mut output);
    };

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    let data = input.as_bytes();

    let config = OutputConfig {
        json: args.json,
        terminator: Terminator::from_null(args.null),
        show_line_numbers: args.line_numbers,
        path: args.input.as_deref().unwrap_or("-"),
    };

    if let Err(error) = extract_rows_by_selector(data, &selector, Newlines::from_utf8(args.utf8), &config, &mut output) {
        exit_on_write_error(&mut output, &error, "Error");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_config() -> OutputConfig<'static> {
        OutputConfig {
            json: false,
            terminator: Terminator::Newline,
            show_line_numbers: false,
            path: "-",
        }
    }

    #[test]
    fn parses_single_row_index() {
        match parse_rows("5").unwrap() {
            RowSelector::Indices(indices) => {
                assert!(indices.contains(&4)); // 0-based
                assert_eq!(indices.len(), 1);
            }
            _ => panic!("Expected Indices"),
        }
    }

    #[test]
    fn parses_comma_separated_row_list() {
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
    fn parses_row_range() {
        match parse_rows("5-10").unwrap() {
            RowSelector::Range(start, end) => {
                assert_eq!(start, 4); // 0-based
                assert_eq!(end, 9);
            }
            _ => panic!("Expected Range"),
        }
    }

    #[test]
    fn rejects_invalid_row_specs() {
        assert!(parse_rows("0").is_err()); // 0 not allowed
        assert!(parse_rows("10-5").is_err()); // invalid range
        assert!(parse_rows("abc").is_err()); // not a number
    }

    #[test]
    fn extracts_single_row_by_index() {
        let data = b"line1\nline2\nline3\nline4\n";
        let mut output = Vec::new();

        let mut indices = HashSet::new();
        indices.insert(1); // line2
        let selector = RowSelector::Indices(indices);

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 1);
        assert_eq!(output, b"line2\n");
    }

    #[test]
    fn extracts_row_range() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(1, 3); // lines 2-4

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline3\nline4\n");
    }

    #[test]
    fn extracts_last_n_rows() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = RowSelector::Tail(2);

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"line4\nline5\n");
    }

    #[test]
    fn extracts_every_nth_row() {
        let data = b"line1\nline2\nline3\nline4\nline5\nline6\n";
        let mut output = Vec::new();

        let selector = RowSelector::Every(2); // every 2nd line

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline4\nline6\n");
    }

    #[test]
    fn prefixes_rows_with_line_numbers() {
        let data = b"line1\nline2\nline3\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(0, 1); // lines 1-2
        let mut config = text_config();
        config.show_line_numbers = true;

        extract_rows_by_selector(data, &selector, Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"1:line1\n2:line2\n");
    }

    #[test]
    fn terminates_rows_with_nul() {
        let data = b"a\nb\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.terminator = Terminator::Null;

        extract_rows_by_selector(data, &RowSelector::Range(0, 1), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"a\0b\0");
    }

    #[test]
    fn emits_json_rows_with_line_numbers() {
        let data = b"a\nb\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.json = true;

        extract_rows_by_selector(data, &RowSelector::Range(1, 1), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                r#"{"type":"line","data":{"path":{"text":"-"},"#,
                r#""lines":{"text":"b"},"line_number":2}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn extracts_multiple_indexed_rows() {
        let data = b"a\nb\nc\nd\ne\n";
        let mut output = Vec::new();

        let mut indices = HashSet::new();
        indices.insert(0); // a
        indices.insert(2); // c
        indices.insert(4); // e
        let selector = RowSelector::Indices(indices);

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"a\nc\ne\n");
    }

    #[test]
    fn clamps_range_to_available_rows() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = RowSelector::Range(0, 100); // Request more than exists

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 2); // Only 2 lines exist
    }

    #[test]
    fn returns_all_rows_when_tail_exceeds_length() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = RowSelector::Tail(100);

        let count = extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 2); // Return all lines
        assert_eq!(output, b"line1\nline2\n");
    }
}
