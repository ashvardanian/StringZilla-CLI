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

    /// Emit JSON Lines, one record per output line
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// NUL-terminate each output record; -D still separates fields within a record
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,
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

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    json: bool,
    terminator: Terminator,
    /// Separates fields within one record; unused under `--json`, which nests them.
    output_delimiter: &'a [u8],
    /// Input name carried into the JSON envelope.
    path: &'a str,
}

/// Write one extracted record as text, honoring the record terminator.
fn write_record_text(
    output: &mut dyn Write,
    config: &OutputConfig,
    fields: &[&[u8]],
    field_indices: &[usize],
) -> io::Result<()> {
    let mut first = true;
    for &index in field_indices {
        if !first {
            output.write_all(config.output_delimiter)?;
        }
        first = false;

        if index < fields.len() {
            output.write_all(fields[index])?;
        }
        // If field doesn't exist, output empty string (like cut)
    }
    output.write_all(&[config.terminator.as_byte()])
}

/// Write one extracted record as a JSON Lines object, with the fields as an array.
fn write_record_json(
    output: &mut dyn Write,
    config: &OutputConfig,
    fields: &[&[u8]],
    field_indices: &[usize],
    line_number: usize,
) -> io::Result<()> {
    output.write_all(br#"{"type":"line","data":{"path":"#)?;
    json_text_field_to(output, config.path.as_bytes())?;
    output.write_all(br#","fields":["#)?;
    let mut first = true;
    for &index in field_indices {
        if !first {
            output.write_all(b",")?;
        }
        first = false;
        let field = fields.get(index).copied().unwrap_or(b"");
        json_text_field_to(output, field)?;
    }
    write!(output, r#"],"line_number":{}}}}}"#, line_number)?;
    output.write_all(b"\n")
}

/// Extract specified columns from data
fn extract_cols(
    data: &[u8],
    field_indices: &[usize],
    delimiter: &[u8],
    min_fields: Option<usize>,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut line_count = 0;
    let mut line_number = 0;
    let mut fields: Vec<&[u8]> = Vec::new(); // reused across lines

    for line in LineIter::new(data, Newlines::Lf) {
        line_number += 1;
        split_fields(line, delimiter, &mut fields);

        // Skip lines with too few fields if min_fields is set
        if let Some(min) = min_fields {
            if fields.len() < min {
                continue;
            }
        }

        if config.json {
            write_record_json(output, config, &fields, field_indices, line_number)?;
        } else {
            write_record_text(output, config, &fields, field_indices)?;
        }
        line_count += 1;
    }

    output.flush()?;
    Ok(line_count)
}

fn main() {
    let args = Args::parse();

    // Parse field specification
    let mut output = io::stdout();

    let field_indices = match parse_fields(&args.fields) {
        Ok(indices) => indices,
        Err(message) => {
            eprintln!("Error: {}", message);
            ExitCode::Error.exit(&mut output);
        }
    };

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    let data = input.as_bytes();

    let delimiter = args.delimiter.as_bytes();
    let output_delimiter = args
        .output_delimiter
        .as_ref()
        .map(|delimiter| delimiter.as_bytes())
        .unwrap_or(delimiter);

    let config = OutputConfig {
        json: args.json,
        terminator: Terminator::from_null(args.null),
        output_delimiter,
        path: args.input.as_deref().unwrap_or("-"),
    };

    if let Err(error) = extract_cols(
        data,
        &field_indices,
        delimiter,
        args.min_fields,
        &config,
        &mut output,
    ) {
        exit_on_write_error(&mut output, &error, "Error");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_field_index() {
        assert_eq!(parse_fields("2").unwrap(), vec![1]); // 0-based
        assert_eq!(parse_fields("1").unwrap(), vec![0]);
    }

    #[test]
    fn parses_comma_separated_field_list() {
        assert_eq!(parse_fields("1,3,5").unwrap(), vec![0, 2, 4]);
        assert_eq!(parse_fields("2, 4").unwrap(), vec![1, 3]); // with spaces
    }

    #[test]
    fn parses_field_range() {
        assert_eq!(parse_fields("2-5").unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(parse_fields("1-3").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn parses_mixed_field_list_and_range() {
        assert_eq!(parse_fields("1,3-5,7").unwrap(), vec![0, 2, 3, 4, 6]);
    }

    #[test]
    fn rejects_invalid_field_specs() {
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
    fn splits_fields_on_tab() {
        assert_eq!(
            fields(b"a\tb\tc", b"\t"),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
    }

    #[test]
    fn splits_fields_on_comma() {
        assert_eq!(
            fields(b"one,two,three", b","),
            vec![b"one".as_slice(), b"two".as_slice(), b"three".as_slice()]
        );
    }

    #[test]
    fn splits_fields_on_multi_char_delimiter() {
        assert_eq!(
            fields(b"a::b::c", b"::"),
            vec![b"a".as_slice(), b"b".as_slice(), b"c".as_slice()]
        );
    }

    fn text_config(output_delimiter: &'static [u8]) -> OutputConfig<'static> {
        OutputConfig {
            json: false,
            terminator: Terminator::Newline,
            output_delimiter,
            path: "-",
        }
    }

    #[test]
    fn extracts_single_column() {
        let data = b"a\tb\tc\n1\t2\t3\n";
        let mut output = Vec::new();

        let count = extract_cols(data, &[1], b"\t", None, &text_config(b"\t"), &mut output).unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }

    #[test]
    fn extracts_and_reorders_multiple_columns() {
        let data = b"a\tb\tc\td\n";
        let mut output = Vec::new();

        extract_cols(data, &[0, 2], b"\t", None, &text_config(b","), &mut output).unwrap();

        assert_eq!(output, b"a,c\n");
    }

    #[test]
    fn emits_empty_field_when_missing() {
        let data = b"a\tb\n";
        let mut output = Vec::new();

        extract_cols(data, &[0, 2], b"\t", None, &text_config(b"\t"), &mut output).unwrap();

        // Field 3 (index 2) doesn't exist, should output empty
        assert_eq!(output, b"a\t\n");
    }

    #[test]
    fn skips_rows_below_min_fields() {
        let data = b"a\tb\tc\na\n1\t2\t3\n";
        let mut output = Vec::new();

        let count =
            extract_cols(data, &[1], b"\t", Some(3), &text_config(b"\t"), &mut output).unwrap();

        // Only lines with 3+ fields
        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }

    #[test]
    fn terminates_records_with_nul() {
        let data = b"a\tb\n1\t2\n";
        let mut output = Vec::new();
        let mut config = text_config(b"\t");
        config.terminator = Terminator::Null;

        extract_cols(data, &[0], b"\t", None, &config, &mut output).unwrap();

        assert_eq!(output, b"a\0\x31\0");
    }

    #[test]
    fn emits_json_records_with_fields() {
        let data = b"a\tb\tc\n";
        let mut output = Vec::new();
        let mut config = text_config(b"\t");
        config.json = true;

        extract_cols(data, &[0, 2], b"\t", None, &config, &mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                r#"{"type":"line","data":{"path":{"text":"-"},"#,
                r#""fields":[{"text":"a"},{"text":"c"}],"line_number":1}}"#,
                "\n"
            )
        );
    }
}
