//! SIMD-accelerated column extraction utility
//!
//! A simpler, faster replacement for `cut -f` and `awk '{print $N}'`.
//! Uses StringZilla for fast delimiter scanning.
//!
//! # Examples
//!
//! ```bash
//! # Extract second column (tab-delimited by default)
//! sz-cols -f 2 data.tsv
//!
//! # Extract multiple columns
//! sz-cols -f 1,3,5 data.tsv
//!
//! # Use comma as delimiter (CSV)
//! sz-cols -d ',' -f 2 data.csv
//!
//! # Extract column range
//! sz-cols -f 2-5 data.tsv
//!
//! # From stdin
//! cat data.tsv | sz-cols -f 2
//! ```

use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz::{FindSplits, MatcherType};

mod shared;
use shared::*;

/// Extract columns from delimited text
#[derive(Parser)]
#[command(name = "sz-cols")]
#[command(version, about = "SIMD-accelerated column extraction (like cut -f)", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Field(s) to extract: single (2), list (1,3,5), or range (2-5)
    #[arg(short = 'f', long = "fields", required = true)]
    fields: String,

    /// Field delimiter (default: tab)
    #[arg(short = 'd', long = "delimiter", default_value = "\t")]
    delimiter: String,

    /// Output delimiter (default: same as input delimiter)
    #[arg(short = 'D', long = "output-delimiter")]
    output_delimiter: Option<String>,

    /// Only output lines with at least N fields
    #[arg(long = "min-fields")]
    min_fields: Option<usize>,

    /// Enable UTF-8 mode (validate input)
    #[arg(long)]
    utf8: bool,
}

/// Parse field specification into a list of 0-based field indices
fn parse_fields(spec: &str) -> Result<Vec<usize>, String> {
    let mut fields = Vec::new();

    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            // Range: "2-5"
            let parts: Vec<&str> = part.split('-').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid range: {}", part));
            }
            let start: usize = parts[0]
                .parse()
                .map_err(|_| format!("Invalid field number: {}", parts[0]))?;
            let end: usize = parts[1]
                .parse()
                .map_err(|_| format!("Invalid field number: {}", parts[1]))?;
            if start == 0 || end == 0 {
                return Err("Field numbers start at 1".to_string());
            }
            if start > end {
                return Err(format!("Invalid range: {} > {}", start, end));
            }
            for i in start..=end {
                fields.push(i - 1); // Convert to 0-based
            }
        } else {
            // Single field: "2"
            let num: usize = part
                .parse()
                .map_err(|_| format!("Invalid field number: {}", part))?;
            if num == 0 {
                return Err("Field numbers start at 1".to_string());
            }
            fields.push(num - 1); // Convert to 0-based
        }
    }

    if fields.is_empty() {
        return Err("No fields specified".to_string());
    }

    Ok(fields)
}

/// Split a line into `out` on the delimiter (SIMD `FindSplits`), reusing the
/// caller's buffer to avoid a per-line allocation. Separator semantics give the
/// trailing empty field on a trailing delimiter for free; an empty line has no
/// fields (matching `cut`).
fn split_fields<'a>(line: &'a [u8], delimiter: &'a [u8], out: &mut Vec<&'a [u8]>) {
    out.clear();
    if line.is_empty() {
        return;
    }
    out.extend(FindSplits::new(line, MatcherType::Find(delimiter)));
}

/// Extract specified columns from data
fn extract_cols(
    data: &[u8],
    field_indices: &[usize],
    delimiter: &[u8],
    output_delimiter: &[u8],
    min_fields: Option<usize>,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut line_count = 0;
    let mut fields: Vec<&[u8]> = Vec::new(); // reused across lines

    for line in LineIter::new(data, false) {
        split_fields(line, delimiter, &mut fields);

        // Skip lines with too few fields if min_fields is set
        if let Some(min) = min_fields {
            if fields.len() < min {
                continue;
            }
        }

        let mut first = true;
        for &idx in field_indices {
            if !first {
                output.write_all(output_delimiter)?;
            }
            first = false;

            if idx < fields.len() {
                output.write_all(fields[idx])?;
            }
            // If field doesn't exist, output empty string (like cut)
        }
        output.write_all(b"\n")?;
        line_count += 1;
    }

    output.flush()?;
    Ok(line_count)
}

fn main() {
    let args = Args::parse();

    // Parse field specification
    let field_indices = match parse_fields(&args.fields) {
        Ok(indices) => indices,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };

    let data = input.as_bytes();

    let delimiter = args.delimiter.as_bytes();
    let output_delimiter = args
        .output_delimiter
        .as_ref()
        .map(|s| s.as_bytes())
        .unwrap_or(delimiter);

    let mut output = io::stdout();

    if let Err(e) = extract_cols(
        data,
        &field_indices,
        delimiter,
        output_delimiter,
        args.min_fields,
        &mut output,
    ) {
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
    fn parse_fields_single() {
        assert_eq!(parse_fields("2").unwrap(), vec![1]); // 0-based
        assert_eq!(parse_fields("1").unwrap(), vec![0]);
    }

    #[test]
    fn parse_fields_list() {
        assert_eq!(parse_fields("1,3,5").unwrap(), vec![0, 2, 4]);
        assert_eq!(parse_fields("2, 4").unwrap(), vec![1, 3]); // with spaces
    }

    #[test]
    fn parse_fields_range() {
        assert_eq!(parse_fields("2-5").unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(parse_fields("1-3").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn parse_fields_mixed() {
        assert_eq!(parse_fields("1,3-5,7").unwrap(), vec![0, 2, 3, 4, 6]);
    }

    #[test]
    fn parse_fields_errors() {
        assert!(parse_fields("0").is_err()); // 0 not allowed
        assert!(parse_fields("5-2").is_err()); // invalid range
        assert!(parse_fields("abc").is_err()); // not a number
    }

    fn fields<'a>(line: &'a [u8], delimiter: &'a [u8]) -> Vec<&'a [u8]> {
        let mut out = Vec::new();
        split_fields(line, delimiter, &mut out);
        out
    }

    #[test]
    fn split_fields_tab() {
        assert_eq!(
            fields(b"a\tb\tc", b"\t"),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
    }

    #[test]
    fn split_fields_comma() {
        assert_eq!(
            fields(b"one,two,three", b","),
            vec![b"one".as_slice(), b"two".as_slice(), b"three".as_slice()]
        );
    }

    #[test]
    fn split_fields_multi_char_delimiter() {
        assert_eq!(
            fields(b"a::b::c", b"::"),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
    }

    #[test]
    fn extract_cols_basic() {
        let data = b"a\tb\tc\n1\t2\t3\n";
        let mut output = Vec::new();

        let count = extract_cols(data, &[1], b"\t", b"\t", None, &mut output).unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }

    #[test]
    fn extract_cols_multiple() {
        let data = b"a\tb\tc\td\n";
        let mut output = Vec::new();

        extract_cols(data, &[0, 2], b"\t", b",", None, &mut output).unwrap();

        assert_eq!(output, b"a,c\n");
    }

    #[test]
    fn extract_cols_missing_field() {
        let data = b"a\tb\n";
        let mut output = Vec::new();

        extract_cols(data, &[0, 2], b"\t", b"\t", None, &mut output).unwrap();

        // Field 3 (index 2) doesn't exist, should output empty
        assert_eq!(output, b"a\t\n");
    }

    #[test]
    fn extract_cols_min_fields() {
        let data = b"a\tb\tc\na\n1\t2\t3\n";
        let mut output = Vec::new();

        let count = extract_cols(data, &[1], b"\t", b"\t", Some(3), &mut output).unwrap();

        // Only lines with 3+ fields
        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }
}
