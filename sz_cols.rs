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

use std::io::{self, Read, Write};

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

    /// Enable UTF-8 mode (split rows on Unicode newlines: CR, CRLF, NEL, LS, PS)
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
/// caller's buffer to avoid a per-line allocation. `limit` stops the scan once that
/// many fields are in hand, which turns a whole-line scan into a first-delimiter scan
/// on a wide record; `None` splits the whole line. Separator semantics give the
/// trailing empty field on a trailing delimiter for free; an empty line has no
/// fields (matching `cut`).
fn split_fields<'a>(
    line: &'a [u8],
    delimiter: &'a [u8],
    limit: Option<usize>,
    out: &mut Vec<&'a [u8]>,
) {
    out.clear();
    if line.is_empty() {
        return;
    }
    let splits = FindSplits::new(line, MatcherType::Find(delimiter));
    match limit {
        Some(limit) => out.extend(splits.take(limit)),
        None => out.extend(splits),
    }
}

/// Which fields to extract and how far each line has to be split, decided once from `Args`.
#[derive(Clone, Copy)]
struct FieldSelection<'a> {
    /// Zero-based field indices, in output order.
    indices: &'a [usize],
    /// How many leading fields to split out, or `None` for the whole line.
    limit: Option<usize>,
    /// Separates fields within an input line.
    delimiter: &'a [u8],
    /// Drop lines carrying fewer fields than this.
    min_fields: Option<usize>,
}

impl<'a> FieldSelection<'a> {
    /// Select `indices`, splitting only as far as the furthest of them reaches.
    /// `--min-fields` tests the total field count, so it forfeits that early stop.
    fn new(indices: &'a [usize], delimiter: &'a [u8], min_fields: Option<usize>) -> Self {
        FieldSelection {
            indices,
            limit: min_fields
                .is_none()
                .then(|| indices.iter().copied().max().map_or(0, |index| index + 1)),
            delimiter,
            min_fields,
        }
    }
}

/// What [`extract_cols`] carries between windows, so a second call resumes where the
/// first stopped. `Copy` and lifetime-free, and it allocates nothing.
#[derive(Clone, Copy, Default)]
struct ColsState {
    /// Input lines seen so far, which `--json` reports.
    line_number: usize,
    /// Records written so far. Nothing in the run reads it; the tests do, to compare a
    /// streamed extraction against a whole-buffer one.
    emitted: usize,
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

/// The selected field, or the empty one a line too short to hold it stands in for,
/// which is what `cut` emits.
#[inline]
fn field_at<'a>(fields: &[&'a [u8]], index: usize) -> &'a [u8] {
    fields.get(index).copied().unwrap_or_default()
}

/// Write one extracted record as text, honoring the record terminator.
fn write_record_text(
    output: &mut dyn Write,
    config: &OutputConfig,
    fields: &[&[u8]],
    field_indices: &[usize],
) -> io::Result<()> {
    for (position, &index) in field_indices.iter().enumerate() {
        if position > 0 {
            output.write_all(config.output_delimiter)?;
        }
        output.write_all(field_at(fields, index))?;
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
    for (position, &index) in field_indices.iter().enumerate() {
        if position > 0 {
            output.write_all(b",")?;
        }
        json_text_field_to(output, field_at(fields, index))?;
    }
    write!(output, r#"],"line_number":{}}}}}"#, line_number)?;
    output.write_all(b"\n")
}

/// Extract the selected columns from every complete line in `data`, resuming from `state`.
fn extract_cols(
    data: &[u8],
    state: &mut ColsState,
    selection: &FieldSelection,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut fields: Vec<&[u8]> = Vec::new(); // reused across lines

    for line in LineIter::new(data, newlines) {
        state.line_number += 1;
        split_fields(line, selection.delimiter, selection.limit, &mut fields);

        // Skip lines with too few fields if min_fields is set
        if selection.min_fields.is_some_and(|min| fields.len() < min) {
            continue;
        }

        if config.json {
            write_record_json(
                output,
                config,
                &fields,
                selection.indices,
                state.line_number,
            )?;
        } else {
            write_record_text(output, config, &fields, selection.indices)?;
        }
        state.emitted += 1;
    }

    Ok(())
}

// region: Streaming

/// Drive [`extract_cols`] over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
fn stream_cols<R: Read>(
    refill: &mut Refill<R>,
    state: &mut ColsState,
    selection: &FieldSelection,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<()> {
    refill.for_each_window(newlines.into(), |window| {
        extract_cols(window, state, selection, newlines, config, output)
    })
}

// endregion: Streaming

fn main() {
    let args = Args::parse();

    // Parse field specification
    let mut output = stdout_writer();

    let field_indices = match parse_fields(&args.fields) {
        Ok(indices) => indices,
        Err(message) => {
            eprintln!("Error: {}", message);
            ExitCode::Error.exit(&mut output);
        }
    };

    let input = match get_input_streaming(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    let delimiter = args.delimiter.as_bytes();
    let output_delimiter = args
        .output_delimiter
        .as_ref()
        .map(|delimiter| delimiter.as_bytes())
        .unwrap_or(delimiter);

    let selection = FieldSelection::new(&field_indices, delimiter, args.min_fields);

    let config = OutputConfig {
        json: args.json,
        terminator: Terminator::from_null(args.null),
        output_delimiter,
        path: args.input.as_deref().unwrap_or("-"),
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let mut state = ColsState::default();
    let result = match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => extract_cols(
            source.as_bytes(),
            &mut state,
            &selection,
            newlines,
            &config,
            &mut output,
        ),
        InputWindow::Stream(mut refill) => stream_cols(
            &mut refill,
            &mut state,
            &selection,
            newlines,
            &config,
            &mut output,
        ),
    };

    // One flush per run: flushing inside the loop would issue one per window.
    if let Err(error) = result.and_then(|()| output.flush()) {
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
        split_fields(line, delimiter, None, &mut out);
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

    /// Extract from a whole buffer, returning the output and the record count.
    fn extract(
        data: &[u8],
        selection: &FieldSelection,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut state = ColsState::default();
        let mut output = Vec::new();
        extract_cols(data, &mut state, selection, newlines, config, &mut output).unwrap();
        (output, state.emitted)
    }

    /// Extract the same way, but through a window of exactly `capacity` bytes.
    fn extract_streamed(
        data: &[u8],
        capacity: usize,
        selection: &FieldSelection,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut refill = Refill::new(data, capacity);
        let mut state = ColsState::default();
        let mut output = Vec::new();
        stream_cols(
            &mut refill,
            &mut state,
            selection,
            newlines,
            config,
            &mut output,
        )
        .unwrap();
        (output, state.emitted)
    }

    #[test]
    fn extracts_single_column() {
        let data = b"a\tb\tc\n1\t2\t3\n";

        let (output, count) = extract(
            data,
            &FieldSelection::new(&[1], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }

    #[test]
    fn extracts_and_reorders_multiple_columns() {
        let data = b"a\tb\tc\td\n";

        let (output, _) = extract(
            data,
            &FieldSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &text_config(b","),
        );

        assert_eq!(output, b"a,c\n");
    }

    #[test]
    fn emits_empty_field_when_missing() {
        let data = b"a\tb\n";

        let (output, _) = extract(
            data,
            &FieldSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        // Field 3 (index 2) doesn't exist, should output empty
        assert_eq!(output, b"a\t\n");
    }

    #[test]
    fn skips_rows_below_min_fields() {
        let data = b"a\tb\tc\na\n1\t2\t3\n";

        let (output, count) = extract(
            data,
            &FieldSelection::new(&[1], b"\t", Some(3)),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        // Only lines with 3+ fields
        assert_eq!(count, 2);
        assert_eq!(output, b"b\n2\n");
    }

    #[test]
    fn terminates_records_with_nul() {
        let data = b"a\tb\n1\t2\n";
        let mut config = text_config(b"\t");
        config.terminator = Terminator::Null;

        let (output, _) = extract(
            data,
            &FieldSelection::new(&[0], b"\t", None),
            Newlines::Lf,
            &config,
        );

        assert_eq!(output, b"a\0\x31\0");
    }

    #[test]
    fn emits_json_records_with_fields() {
        let data = b"a\tb\tc\n";
        let mut config = text_config(b"\t");
        config.json = true;

        let (output, _) = extract(
            data,
            &FieldSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &config,
        );

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                r#"{"type":"line","data":{"path":{"text":"-"},"#,
                r#""fields":[{"text":"a"},{"text":"c"}],"line_number":1}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn stops_splitting_once_selected_fields_are_in_hand() {
        let wide: Vec<u8> = {
            let mut line: Vec<u8> = (0..64)
                .map(|column| format!("c{}", column))
                .collect::<Vec<_>>()
                .join("\t")
                .into_bytes();
            line.push(b'\n');
            line
        };

        let early = extract(
            &wide,
            &FieldSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );
        let whole = extract(
            &wide,
            &FieldSelection {
                indices: &[0, 2],
                limit: None,
                delimiter: b"\t",
                min_fields: None,
            },
            Newlines::Lf,
            &text_config(b"\t"),
        );

        assert_eq!(FieldSelection::new(&[0, 2], b"\t", None).limit, Some(3));
        assert_eq!(FieldSelection::new(&[0, 2], b"\t", Some(4)).limit, None);
        assert_eq!(early, whole);
        assert_eq!(early.0, b"c0\tc2\n");
    }

    #[test]
    fn splits_rows_on_unicode_newlines_under_utf8() {
        // Line separators, which only the Unicode newline set breaks on.
        let data = "a\tb\u{2028}c\td\n".as_bytes();
        let selection = FieldSelection::new(&[1], b"\t", None);

        let (output, count) = extract(data, &selection, Newlines::Unicode, &text_config(b"\t"));
        assert_eq!(count, 2);
        assert_eq!(output, b"b\nd\n");

        // Without it the separator is ordinary text inside one wide field.
        let (output, count) = extract(data, &selection, Newlines::Lf, &text_config(b"\t"));
        assert_eq!(count, 1);
        assert_eq!(output, "b\u{2028}c\n".as_bytes());
    }

    #[test]
    fn streams_identically_to_whole_buffer() {
        // A blank line, an over-wide line, and a final line without a terminator.
        let data =
            "a\tb\tc\n\t\t\n\nvery\tlong\trecord\u{2028}that\toutgrows\ta\ttiny\twindow\nx\ty"
                .as_bytes();
        let mut json = text_config(b",");
        json.json = true;

        for (newline_set, newlines) in [("lf", Newlines::Lf), ("unicode", Newlines::Unicode)] {
            for selection in [
                FieldSelection::new(&[1], b"\t", None),
                FieldSelection::new(&[0, 2], b"\t", None),
                FieldSelection::new(&[1], b"\t", Some(3)),
            ] {
                for config in [text_config(b"\t"), text_config(b","), json] {
                    let whole = extract(data, &selection, newlines, &config);
                    for capacity in [7, 13, 64, 4096] {
                        assert_eq!(
                            extract_streamed(data, capacity, &selection, newlines, &config),
                            whole,
                            "{} newlines at capacity {}",
                            newline_set,
                            capacity
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn streams_an_empty_input_without_records() {
        let (output, count) = extract_streamed(
            b"",
            7,
            &FieldSelection::new(&[0], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        assert!(output.is_empty());
        assert_eq!(count, 0);
    }
}
