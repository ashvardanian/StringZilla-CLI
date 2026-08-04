//! SIMD-accelerated column extraction utility
//!
//! A simpler, faster replacement for `cut -f` and `awk '{print $N}'`.
//! Uses StringZilla for fast delimiter scanning.
//!
//! # Examples
//!
//! ```bash
//! # Extract second column (tab-delimited by default)
//! sz-cols --columns 2 data.tsv
//!
//! # Extract multiple columns
//! sz-cols --columns 1,3,5 data.tsv
//!
//! # Use comma as delimiter (CSV)
//! sz-cols --delimiter ',' --columns 2 data.csv
//!
//! # Extract column range
//! sz-cols --columns 2-5 data.tsv
//!
//! # From stdin
//! cat data.tsv | sz-cols --columns 2
//! ```

use std::io::{self, Read, Write};

use clap::{error::ErrorKind, CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{FindSplits, MatcherType};

use shared::*;

/// Extract columns from delimited text
#[derive(Parser)]
#[command(name = "sz-cols")]
#[command(version, about = "SIMD-accelerated column extraction (like cut -f)", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Column(s) to extract: single (2), list (1,3,5), or range (2-5)
    #[arg(long, required = true)]
    columns: String,

    /// Column delimiter (default: tab)
    #[arg(long, default_value = "\t")]
    delimiter: String,

    /// Output delimiter (default: same as input delimiter)
    #[arg(long)]
    output_delimiter: Option<String>,

    /// Only output lines with at least N columns [default: no minimum]
    #[arg(long)]
    min_columns: Option<usize>,

    /// Treat the input as UTF-8 text
    #[arg(long)]
    utf8: bool,

    /// How records are rendered
    #[arg(
        long,
        value_enum,
        default_value = "text",
        help_heading = "Output Formats"
    )]
    format: Format,

    /// NUL-terminate each output record instead of newline
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any record was extracted, 1 otherwise
    #[arg(long, conflicts_with_all = ["format", "null", "output_delimiter"], help_heading = "Output Formats")]
    quiet: bool,
}

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    /// The selected columns, joined by the output delimiter.
    Text,
    /// JSON Lines, one record per output line.
    Json,
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// The constraints clap cannot express: `conflicts_with` fires on a flag's presence,
/// never on its value, and an empty delimiter reaches the splitter unchecked.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.delimiter.is_empty() {
        return Err(reject("--delimiter cannot be empty"));
    }
    if args.format == Format::Json {
        if args.output_delimiter.is_some() {
            return Err(reject(
                "--format json nests columns, so --output-delimiter has no effect",
            ));
        }
        if args.null {
            return Err(reject("--format json cannot be combined with --null"));
        }
    }
    Ok(())
}

/// Parse a column specification into a list of 0-based column indices
/// Read one 1-based column number into the 0-based index the split uses.
///
/// Every message names the token it read, which is why the empty case is separate: `2-`
/// splits into `2` and nothing, and "invalid column number: " names nothing at all.
fn parse_column(token: &str) -> Result<usize, String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("a column number is missing; write both ends of a range, as in `2-5`".into());
    }
    let column: usize = token
        .parse()
        .map_err(|_| format!("`{token}` is not a column number"))?;
    if column == 0 {
        return Err("column numbers start at 1".into());
    }
    Ok(column - 1)
}

fn parse_columns(spec: &str) -> Result<Vec<usize>, String> {
    if spec.trim().is_empty() {
        return Err("no columns given; pass a column, a list, or a range".into());
    }
    let mut columns = Vec::new();

    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            // Range: "2-5"
            let parts: Vec<&str> = part.split('-').collect();
            if parts.len() != 2 {
                return Err(format!("`{part}` is not a range; a range has two ends"));
            }
            let (start, end) = (parse_column(parts[0])?, parse_column(parts[1])?);
            if start > end {
                return Err(format!(
                    "`{part}` runs backwards; a range reads low to high"
                ));
            }
            columns.extend(start..=end);
        } else {
            columns.push(parse_column(part)?);
        }
    }

    if columns.is_empty() {
        return Err("no columns given; pass a column, a list, or a range".into());
    }

    Ok(columns)
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
struct ColumnSelection<'a> {
    /// Zero-based field indices, in output order.
    indices: &'a [usize],
    /// How many leading fields to split out, or `None` for the whole line.
    limit: Option<usize>,
    /// Separates fields within an input line.
    delimiter: &'a [u8],
    /// Drop lines carrying fewer fields than this.
    min_columns: Option<usize>,
}

impl<'a> ColumnSelection<'a> {
    /// Select `indices`, splitting only as far as the furthest of them reaches.
    /// `--min-columns` tests the total column count, so it forfeits that early stop.
    fn new(indices: &'a [usize], delimiter: &'a [u8], min_columns: Option<usize>) -> Self {
        ColumnSelection {
            indices,
            limit: min_columns
                .is_none()
                .then(|| indices.iter().copied().max().map_or(0, |index| index + 1)),
            delimiter,
            min_columns,
        }
    }
}

/// What [`extract_data`] carries between windows, so a second call resumes where the
/// first stopped. `Copy` and lifetime-free, and it allocates nothing.
#[derive(Clone, Copy, Default)]
struct ColsState {
    /// Input lines seen so far, which `--format json` reports.
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
    /// Separates fields within one record; unused under `--format json`, which nests them.
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
    column_indices: &[usize],
) -> io::Result<()> {
    for (position, &index) in column_indices.iter().enumerate() {
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
    column_indices: &[usize],
    line_number: usize,
) -> io::Result<()> {
    output.write_all(br#"{"type":"line","data":{"path":"#)?;
    json_text_field_to(output, config.path.as_bytes())?;
    output.write_all(br#","columns":["#)?;
    for (position, &index) in column_indices.iter().enumerate() {
        if position > 0 {
            output.write_all(b",")?;
        }
        json_text_field_to(output, field_at(fields, index))?;
    }
    write!(output, r#"],"line_number":{}}}}}"#, line_number)?;
    output.write_all(b"\n")
}

/// Extract the selected columns from every complete line in `data`, resuming from `state`.
fn extract_data(
    data: &[u8],
    state: &mut ColsState,
    selection: &ColumnSelection,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<()> {
    let mut fields: Vec<&[u8]> = Vec::new(); // reused across lines

    for line in LineIter::new(data, newlines) {
        state.line_number += 1;
        split_fields(line, selection.delimiter, selection.limit, &mut fields);

        // Skip lines with too few fields if min_columns is set
        if selection.min_columns.is_some_and(|min| fields.len() < min) {
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

/// Drive [`extract_data`] over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
fn extract_stream<R: Read>(
    refill: &mut Refill<R>,
    state: &mut ColsState,
    selection: &ColumnSelection,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<()> {
    refill.for_each_window(newlines.into(), |window| {
        extract_data(window, state, selection, newlines, config, output)
    })
}

// endregion: Streaming

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let column_indices = parse_columns(&args.columns)
        .map_err(|message| Args::command().error(ErrorKind::ValueValidation, message))?;

    let name = args.input.as_deref().unwrap_or("-");
    let input = get_input_streaming(args.input.as_deref()).at(name)?;

    let delimiter = args.delimiter.as_bytes();
    let output_delimiter = args
        .output_delimiter
        .as_ref()
        .map(|delimiter| delimiter.as_bytes())
        .unwrap_or(delimiter);

    let selection = ColumnSelection::new(&column_indices, delimiter, args.min_columns);

    let config = OutputConfig {
        json: args.format == Format::Json,
        terminator: Terminator::from_null(args.null),
        output_delimiter,
        path: name,
    };

    // A quiet run still extracts, so the record count that answers it stays honest.
    let mut discard = io::sink();
    let writer: &mut dyn Write = if args.quiet {
        &mut discard
    } else {
        &mut *output
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let mut state = ColsState::default();
    // Only a pipe streams, so both the reader and the writer of either arm are `-`.
    match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => extract_data(
            source.as_bytes(),
            &mut state,
            &selection,
            newlines,
            &config,
            writer,
        )
        .at("-")?,
        InputWindow::Stream(mut refill) => extract_stream(
            &mut refill,
            &mut state,
            &selection,
            newlines,
            &config,
            writer,
        )
        .at("-")?,
    }

    output.flush().at("-")?;
    Ok(Status::from_found(state.emitted > 0))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-cols", run(&args, &mut output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_token_it_read_in_every_column_error() {
        // A range with a missing end used to report `Invalid column number: `, naming nothing.
        for (spec, expected) in [
            ("2-", "a column number is missing"),
            ("abc", "`abc` is not a column number"),
            ("0", "column numbers start at 1"),
            ("5-2", "runs backwards"),
            ("1-2-3", "is not a range"),
            ("", "no columns given"),
        ] {
            let error = parse_columns(spec).expect_err("must not parse");
            assert!(error.contains(expected), "`{spec}` reported `{error}`");
        }
    }
    #[test]
    fn counts_records_a_quiet_run_never_writes() {
        // `--quiet` extracts into a sink, so the count that becomes the exit status
        // is the same one a printing run would report.
        let selection = ColumnSelection::new(&[0], b"\t", None);
        let config = text_config(b"\t");

        let mut state = ColsState::default();
        let mut discard = io::sink();
        extract_data(
            b"a\tb\n",
            &mut state,
            &selection,
            Newlines::Lf,
            &config,
            &mut discard,
        )
        .unwrap();
        assert!(Status::from_found(state.emitted > 0) == Status::Success);

        let mut state = ColsState::default();
        extract_data(
            b"",
            &mut state,
            &selection,
            Newlines::Lf,
            &config,
            &mut discard,
        )
        .unwrap();
        assert!(Status::from_found(state.emitted > 0) == Status::NoResult);
    }

    #[test]
    fn quiet_rejects_the_flags_it_would_ignore() {
        // Under `--quiet` the exit status is the whole output, so nothing that shapes a
        // record can reach it.
        let rejected = [
            vec!["sz-cols", "--columns", "1", "--quiet", "--format", "json"],
            vec!["sz-cols", "--columns", "1", "--quiet", "--null"],
            vec![
                "sz-cols",
                "--columns",
                "1",
                "--quiet",
                "--output-delimiter",
                ",",
            ],
        ];
        for arguments in rejected {
            assert!(
                Args::try_parse_from(&arguments).is_err(),
                "{:?} must be rejected",
                arguments
            );
        }

        // Which rows survive still steers the status, so the filters compose.
        let accepted = [
            vec!["sz-cols", "--columns", "1", "--quiet", "--min-columns", "3"],
            vec!["sz-cols", "--columns", "1", "--quiet", "--delimiter", ","],
            vec!["sz-cols", "--columns", "1", "--quiet", "--utf8"],
        ];
        for arguments in accepted {
            assert!(
                Args::try_parse_from(&arguments).is_ok(),
                "{:?} must compose",
                arguments
            );
        }
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "columns",
                "delimiter",
                "output-delimiter",
                "min-columns",
                "utf8",
                "format",
                "null",
                "quiet",
                "help",
                "version",
            ]
        );
        assert!(command
            .get_arguments()
            .all(|argument| argument.get_short().is_none()
                || matches!(argument.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn names_the_path_a_failure_happened_on() {
        // A missing input used to report `Error reading input: …`, naming nothing.
        let args = Args::parse_from(["sz-cols", "--columns", "1", "missing.tsv"]);
        let Err(failure) = run(&args, &mut io::sink()) else {
            panic!("a missing input must fail");
        };
        assert!(
            failure.to_string().starts_with("missing.tsv: "),
            "{}",
            failure
        );
    }

    #[test]
    fn rejects_what_json_would_silently_ignore() {
        // An empty delimiter used to reach the splitter, and JSON used to drop both flags.
        let parsed = |arguments: &[&str]| Args::try_parse_from(arguments).unwrap();
        assert!(validate(&parsed(&["sz-cols", "--columns", "1", "--delimiter", ""])).is_err());
        assert!(validate(&parsed(&[
            "sz-cols",
            "--columns",
            "1",
            "--format",
            "json",
            "--output-delimiter",
            ","
        ]))
        .is_err());
        assert!(validate(&parsed(&[
            "sz-cols",
            "--columns",
            "1",
            "--format",
            "json",
            "--null"
        ]))
        .is_err());
        assert!(validate(&parsed(&[
            "sz-cols",
            "--columns",
            "1",
            "--output-delimiter",
            ","
        ]))
        .is_ok());
        assert!(validate(&parsed(&["sz-cols", "--columns", "1", "--null"])).is_ok());
    }

    #[test]
    fn parses_single_column_index() {
        assert_eq!(parse_columns("2").unwrap(), vec![1]); // 0-based
        assert_eq!(parse_columns("1").unwrap(), vec![0]);
    }

    #[test]
    fn parses_comma_separated_column_list() {
        assert_eq!(parse_columns("1,3,5").unwrap(), vec![0, 2, 4]);
        assert_eq!(parse_columns("2, 4").unwrap(), vec![1, 3]); // with spaces
    }

    #[test]
    fn parses_column_range() {
        assert_eq!(parse_columns("2-5").unwrap(), vec![1, 2, 3, 4]);
        assert_eq!(parse_columns("1-3").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn parses_mixed_column_list_and_range() {
        assert_eq!(parse_columns("1,3-5,7").unwrap(), vec![0, 2, 3, 4, 6]);
    }

    #[test]
    fn rejects_invalid_column_specs() {
        assert!(parse_columns("0").is_err()); // 0 not allowed
        assert!(parse_columns("5-2").is_err()); // invalid range
        assert!(parse_columns("abc").is_err()); // not a number
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
        selection: &ColumnSelection,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut state = ColsState::default();
        let mut output = Vec::new();
        extract_data(data, &mut state, selection, newlines, config, &mut output).unwrap();
        (output, state.emitted)
    }

    /// Extract the same way, but through a window of exactly `capacity` bytes.
    fn extract_streamed(
        data: &[u8],
        capacity: usize,
        selection: &ColumnSelection,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut refill = Refill::new(data, capacity);
        let mut state = ColsState::default();
        let mut output = Vec::new();
        extract_stream(
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
            &ColumnSelection::new(&[1], b"\t", None),
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
            &ColumnSelection::new(&[0, 2], b"\t", None),
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
            &ColumnSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        // Field 3 (index 2) doesn't exist, should output empty
        assert_eq!(output, b"a\t\n");
    }

    #[test]
    fn skips_rows_below_min_columns() {
        let data = b"a\tb\tc\na\n1\t2\t3\n";

        let (output, count) = extract(
            data,
            &ColumnSelection::new(&[1], b"\t", Some(3)),
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
            &ColumnSelection::new(&[0], b"\t", None),
            Newlines::Lf,
            &config,
        );

        assert_eq!(output, b"a\0\x31\0");
    }

    #[test]
    fn emits_json_records_with_columns() {
        let data = b"a\tb\tc\n";
        let mut config = text_config(b"\t");
        config.json = true;

        let (output, _) = extract(
            data,
            &ColumnSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &config,
        );

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                r#"{"type":"line","data":{"path":{"text":"-"},"#,
                r#""columns":[{"text":"a"},{"text":"c"}],"line_number":1}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn stops_splitting_once_selected_columns_are_in_hand() {
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
            &ColumnSelection::new(&[0, 2], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );
        let whole = extract(
            &wide,
            &ColumnSelection {
                indices: &[0, 2],
                limit: None,
                delimiter: b"\t",
                min_columns: None,
            },
            Newlines::Lf,
            &text_config(b"\t"),
        );

        assert_eq!(ColumnSelection::new(&[0, 2], b"\t", None).limit, Some(3));
        assert_eq!(ColumnSelection::new(&[0, 2], b"\t", Some(4)).limit, None);
        assert_eq!(early, whole);
        assert_eq!(early.0, b"c0\tc2\n");
    }

    #[test]
    fn splits_rows_on_unicode_newlines_under_utf8() {
        // Line separators, which only the Unicode newline set breaks on.
        let data = "a\tb\u{2028}c\td\n".as_bytes();
        let selection = ColumnSelection::new(&[1], b"\t", None);

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
                ColumnSelection::new(&[1], b"\t", None),
                ColumnSelection::new(&[0, 2], b"\t", None),
                ColumnSelection::new(&[1], b"\t", Some(3)),
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
            &ColumnSelection::new(&[0], b"\t", None),
            Newlines::Lf,
            &text_config(b"\t"),
        );

        assert!(output.is_empty());
        assert_eq!(count, 0);
    }
}
