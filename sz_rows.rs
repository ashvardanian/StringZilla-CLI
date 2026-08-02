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

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use clap::Parser;
use stringzilla::sz;

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
    #[arg(long = "tail", value_parser = parse_at_least_one, conflicts_with_all = ["rows", "every"])]
    tail: Option<NonZeroUsize>,

    /// Extract every Nth line
    #[arg(long = "every", value_parser = parse_at_least_one, conflicts_with_all = ["rows", "tail"])]
    every: Option<NonZeroUsize>,

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

/// Which rows to extract.
enum RowSelector {
    /// Rows a single forward pass picks out.
    Forward(ForwardSelector),
    /// The last N rows, which a forward pass cannot name until the input ends.
    Tail(NonZeroUsize),
}

/// The rows a forward pass picks out, in the order the lines arrive.
enum ForwardSelector {
    /// Specific zero-based row indices, sorted and deduplicated.
    Indices(Vec<usize>),
    /// An inclusive zero-based row range.
    Range(usize, usize),
    /// Every Nth row (1 = all, 2 = every other, and so on).
    Every(NonZeroUsize),
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
            // Convert to 0-based
            return Ok(RowSelector::Forward(ForwardSelector::Range(
                start - 1,
                end - 1,
            )));
        }
    }

    // Parse as list of indices (possibly with ranges)
    let mut indices = Vec::new();

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
            indices.extend((start..=end).map(|row| row - 1)); // Convert to 0-based
        } else {
            // Single row: "5"
            let row: usize = part
                .parse()
                .map_err(|_| format!("Invalid row number: {}", part))?;
            if row == 0 {
                return Err("Row numbers start at 1".to_string());
            }
            indices.push(row - 1); // Convert to 0-based
        }
    }

    if indices.is_empty() {
        return Err("No rows specified".to_string());
    }

    // Rows arrive in order, so the walk reads a sorted list with a cursor rather than
    // hashing every line against a set that is usually one to three elements wide.
    indices.sort_unstable();
    indices.dedup();

    Ok(RowSelector::Forward(ForwardSelector::Indices(indices)))
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

// region: Tail Scan

/// Byte offset at which the last `wanted` LF-terminated lines begin, and how many
/// lines that span covers — `min(wanted, total lines)`. Each backward `rfind` touches
/// one line's worth of bytes, so the cost scales with `wanted`, not with the input.
///
/// A trailing newline terminates the last line rather than starting an empty one,
/// matching [`LineIter`]: `b""` is zero lines and `b"\n"` is one empty line.
fn tail_start_lf(data: &[u8], wanted: NonZeroUsize) -> (usize, usize) {
    if data.is_empty() {
        return (0, 0);
    }

    // The terminator of the final line is not a boundary between two lines.
    let mut limit = data.len() - usize::from(data.ends_with(b"\n"));
    let mut start = 0;
    let mut found = 0;
    while found < wanted.get() {
        let previous = if limit == 0 {
            None
        } else {
            sz::rfind(&data[..limit], b"\n")
        };
        match previous {
            Some(position) => {
                start = position + 1;
                found += 1;
                limit = position;
            }
            // No earlier newline, so the first line of the input is the span's first.
            None => {
                start = 0;
                found += 1;
                break;
            }
        }
    }
    (start, found)
}

// endregion: Tail Scan

// region: Forward Selectors

/// What [`extract_forward_rows`] carries between windows, so a second call resumes where the
/// first stopped. `Copy` and lifetime-free, and it allocates nothing.
#[derive(Clone, Copy, Default)]
struct RowsState {
    /// Absolute zero-based index of the next line to arrive.
    next_index: usize,
    /// How far into a sorted index list the walk has come.
    cursor: usize,
    /// Rows written so far. Nothing in the run reads it; the tests do, to compare a
    /// streamed extraction against a whole-buffer one.
    emitted: usize,
}

impl ForwardSelector {
    /// Whether the row at `index` is wanted, stepping the cursor over it. Rows arrive in
    /// order and the indices are sorted, so the cursor only ever moves forward.
    fn takes(&self, index: usize, state: &mut RowsState) -> bool {
        match self {
            ForwardSelector::Indices(indices) => {
                let wanted = indices.get(state.cursor) == Some(&index);
                state.cursor += usize::from(wanted);
                wanted
            }
            ForwardSelector::Range(start, end) => (*start..=*end).contains(&index),
            ForwardSelector::Every(stride) => (index + 1).is_multiple_of(stride.get()),
        }
    }

    /// Whether no row from here on can match, which is what lets a streamed caller stop
    /// reading rather than drain the rest of its input.
    fn exhausted(&self, state: &RowsState) -> bool {
        match self {
            ForwardSelector::Indices(indices) => state.cursor == indices.len(),
            ForwardSelector::Range(_, end) => state.next_index > *end,
            // A stride keeps matching for as long as lines keep arriving.
            ForwardSelector::Every(_) => false,
        }
    }
}

/// Write the selected rows of `data`, resuming from `state`. [`ControlFlow::Break`] once the
/// selector can never match again, which lets a streamed caller stop draining its reader.
fn extract_forward_rows(
    data: &[u8],
    selector: &ForwardSelector,
    newlines: Newlines,
    config: &OutputConfig,
    state: &mut RowsState,
    output: &mut dyn Write,
) -> io::Result<ControlFlow<()>> {
    for line in LineIter::new(data, newlines) {
        if selector.exhausted(state) {
            return Ok(ControlFlow::Break(()));
        }
        let index = state.next_index;
        state.next_index += 1;
        if selector.takes(index, state) {
            write_row(output, config, line, index)?;
            state.emitted += 1;
        }
    }
    Ok(if selector.exhausted(state) {
        ControlFlow::Break(())
    } else {
        ControlFlow::Continue(())
    })
}

// endregion: Forward Selectors

/// Extract rows using indices, a range, the tail, or a stride, over one whole slice.
fn extract_rows_by_selector(
    data: &[u8],
    selector: &RowSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut state = RowsState::default();

    match selector {
        RowSelector::Forward(forward) => {
            // One slice is the whole input, so there is no next window an early stop saves.
            let _early_stop =
                extract_forward_rows(data, forward, newlines, config, &mut state, output)?;
        }
        RowSelector::Tail(wanted) => match newlines {
            Newlines::Lf => {
                let (start, lines_found) = tail_start_lf(data, *wanted);
                // Absolute numbering costs a newline pass over the skipped prefix,
                // so it is paid only when a record carries a line number.
                let first_index = if config.json || config.show_line_numbers {
                    LineIter::new(&data[..start], newlines).count()
                } else {
                    0
                };
                for (index_in_span, line) in LineIter::new(&data[start..], newlines).enumerate() {
                    write_row(output, config, line, first_index + index_in_span)?;
                    state.emitted += 1;
                }
                debug_assert_eq!(
                    state.emitted, lines_found,
                    "tail span must hold the counted lines"
                );
            }
            // The Unicode newline set has no reverse kernel: `RFindSplits` matches
            // bytes, `sz_utf8_split_newlines` scans forward only, and `rfind_byteset`
            // cannot express the multi-byte NEL, LS, and PS. So the last N lines come
            // from a forward walk into a ring — O(N) memory, not O(file).
            Newlines::Unicode => {
                let wanted = wanted.get();
                let mut ring: VecDeque<(usize, &[u8])> =
                    VecDeque::with_capacity(wanted.min(TAIL_RING_RESERVE));
                for (index, line) in LineIter::new(data, newlines).enumerate() {
                    if ring.len() == wanted {
                        ring.pop_front();
                    }
                    ring.push_back((index, line));
                }
                for (index, line) in ring {
                    write_row(output, config, line, index)?;
                    state.emitted += 1;
                }
            }
        },
    }

    Ok(state.emitted)
}

// region: Streaming

/// Ring slots reserved before a single line arrives. A larger `--tail` still works, growing
/// the deque as lines land rather than trusting the request to be sane.
const TAIL_RING_RESERVE: usize = 4096;

/// The last `wanted` lines seen, each owned because the window it borrowed from is about to
/// be overwritten. An evicted buffer is refilled in place, so the steady state allocates
/// nothing and the footprint stays bounded by `wanted × longest line`.
struct TailRing {
    /// How many lines the ring keeps, which `--tail` sets.
    wanted: NonZeroUsize,
    /// Absolute zero-based index of the oldest retained line.
    first_index: usize,
    /// The retained lines, oldest first.
    lines: VecDeque<Vec<u8>>,
}

impl TailRing {
    /// A ring holding at most `wanted` lines.
    fn new(wanted: NonZeroUsize) -> Self {
        TailRing {
            wanted,
            first_index: 0,
            lines: VecDeque::with_capacity(wanted.get().min(TAIL_RING_RESERVE)),
        }
    }

    /// Retain `line`, evicting the oldest when the ring is full and reusing its buffer.
    fn push(&mut self, line: &[u8]) {
        let mut buffer = if self.lines.len() == self.wanted.get() {
            self.first_index += 1;
            self.lines.pop_front().unwrap_or_default()
        } else {
            Vec::new()
        };
        buffer.clear();
        buffer.extend_from_slice(line);
        self.lines.push_back(buffer);
    }

    /// Write the retained lines in arrival order, numbered absolutely.
    fn write_to(&self, config: &OutputConfig, output: &mut dyn Write) -> io::Result<usize> {
        for (offset, line) in self.lines.iter().enumerate() {
            write_row(output, config, line, self.first_index + offset)?;
        }
        Ok(self.lines.len())
    }
}

/// Drive [`extract_forward_rows`] over a reader, handing it whole-line prefixes of one
/// reused window. The selector's early stop ends the loop, so a bounded request such as
/// `-r 5` stops reading rather than draining the rest of the stream.
fn stream_forward_rows<R: Read>(
    refill: &mut Refill<R>,
    selector: &ForwardSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut state = RowsState::default();
    refill.try_for_each_window(newlines.into(), |window| {
        extract_forward_rows(window, selector, newlines, config, &mut state, output)
    })?;
    Ok(state.emitted)
}

/// Drive the tail ring over a reader. A stream cannot be scanned backward, so the last
/// `wanted` lines are the ones a forward walk still holds when the reader runs out.
fn stream_tail_rows<R: Read>(
    refill: &mut Refill<R>,
    wanted: NonZeroUsize,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut ring = TailRing::new(wanted);
    refill.for_each_window(newlines.into(), |window| {
        for line in LineIter::new(window, newlines) {
            ring.push(line);
        }
        Ok(())
    })?;
    ring.write_to(config, output)
}

/// Extract rows from a reader, choosing the ring for `--tail` and the forward walk for the
/// rest. Line numbers are absolute in both, matching the whole-slice paths.
fn stream_rows<R: Read>(
    refill: &mut Refill<R>,
    selector: &RowSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    match selector {
        RowSelector::Forward(forward) => {
            stream_forward_rows(refill, forward, newlines, config, output)
        }
        RowSelector::Tail(wanted) => stream_tail_rows(refill, *wanted, newlines, config, output),
    }
}

// endregion: Streaming

fn main() {
    let args = Args::parse();
    let mut output = stdout_writer();

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
        RowSelector::Tail(wanted)
    } else if let Some(stride) = args.every {
        RowSelector::Forward(ForwardSelector::Every(stride))
    } else {
        eprintln!("Error: must specify --rows, --tail, or --every");
        ExitCode::Error.exit(&mut output);
    };

    let input = match get_input_streaming(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    let config = OutputConfig {
        json: args.json,
        terminator: Terminator::from_null(args.null),
        show_line_numbers: args.line_numbers,
        path: args.input.as_deref().unwrap_or("-"),
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let result = match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => {
            extract_rows_by_selector(source.as_bytes(), &selector, newlines, &config, &mut output)
        }
        InputWindow::Stream(mut refill) => {
            stream_rows(&mut refill, &selector, newlines, &config, &mut output)
        }
    };

    // One flush per run: flushing inside the loop would issue one per window.
    if let Err(error) = result.and_then(|_| output.flush()) {
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

    fn indices(rows: &[usize]) -> RowSelector {
        RowSelector::Forward(ForwardSelector::Indices(rows.to_vec()))
    }

    fn range(start: usize, end: usize) -> RowSelector {
        RowSelector::Forward(ForwardSelector::Range(start, end))
    }

    fn every(stride: usize) -> RowSelector {
        RowSelector::Forward(ForwardSelector::Every(NonZeroUsize::new(stride).unwrap()))
    }

    fn tail(wanted: usize) -> RowSelector {
        RowSelector::Tail(NonZeroUsize::new(wanted).unwrap())
    }

    /// Extract the same way [`extract_rows_by_selector`] does, but through a window of
    /// exactly `capacity` bytes.
    fn extract_streamed(
        data: &[u8],
        capacity: usize,
        selector: &RowSelector,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut refill = Refill::new(data, capacity);
        let mut output = Vec::new();
        let count = stream_rows(&mut refill, selector, newlines, config, &mut output).unwrap();
        (output, count)
    }

    /// A reader that fails once a budget of bytes has been handed out, so a test can prove
    /// a selector stopped reading rather than merely stopped writing.
    struct BudgetedReader<'a> {
        data: &'a [u8],
        position: usize,
        budget: usize,
    }

    impl Read for BudgetedReader<'_> {
        fn read(&mut self, destination: &mut [u8]) -> io::Result<usize> {
            if self.position >= self.budget {
                return Err(io::Error::other("read past the budget"));
            }
            let available = self.data.len() - self.position;
            let taken = destination
                .len()
                .min(available)
                .min(self.budget - self.position);
            destination[..taken].copy_from_slice(&self.data[self.position..self.position + taken]);
            self.position += taken;
            Ok(taken)
        }
    }

    #[test]
    fn rejects_a_zero_count_without_naming_a_rust_type() {
        for flag in ["--tail", "--every"] {
            let Err(error) = Args::try_parse_from(["sz-rows", flag, "0"]) else {
                panic!("{} 0 must be rejected", flag);
            };
            let rendered = error.to_string();
            assert!(rendered.contains("must be at least 1"), "{}", rendered);
            assert!(!rendered.contains("non-zero type"), "{}", rendered);
            assert_eq!(error.exit_code(), 2, "{}", rendered);

            // Only zero reads differently; every other rejection keeps clap's wording.
            let Err(error) = Args::try_parse_from(["sz-rows", flag, "abc"]) else {
                panic!("{} abc must be rejected", flag);
            };
            assert!(
                error.to_string().contains("invalid digit found in string"),
                "{}",
                error
            );
            assert!(Args::try_parse_from(["sz-rows", flag, "1"]).is_ok());
        }
    }

    #[test]
    fn parses_single_row_index() {
        match parse_rows("5").unwrap() {
            RowSelector::Forward(ForwardSelector::Indices(rows)) => assert_eq!(rows, vec![4]),
            _ => panic!("Expected Indices"),
        }
    }

    #[test]
    fn parses_comma_separated_row_list() {
        // Sorted and deduplicated, whatever order the spec named them in.
        match parse_rows("10,1,5,1").unwrap() {
            RowSelector::Forward(ForwardSelector::Indices(rows)) => assert_eq!(rows, vec![0, 4, 9]),
            _ => panic!("Expected Indices"),
        }
    }

    #[test]
    fn parses_row_range() {
        match parse_rows("5-10").unwrap() {
            RowSelector::Forward(ForwardSelector::Range(start, end)) => {
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

        let selector = indices(&[1]); // line2

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 1);
        assert_eq!(output, b"line2\n");
    }

    #[test]
    fn extracts_row_range() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = range(1, 3); // lines 2-4

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline3\nline4\n");
    }

    #[test]
    fn extracts_last_n_rows() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = tail(2);

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"line4\nline5\n");
    }

    #[test]
    fn extracts_every_nth_row() {
        let data = b"line1\nline2\nline3\nline4\nline5\nline6\n";
        let mut output = Vec::new();

        let selector = every(2); // every 2nd line

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline4\nline6\n");
    }

    #[test]
    fn prefixes_rows_with_line_numbers() {
        let data = b"line1\nline2\nline3\n";
        let mut output = Vec::new();

        let selector = range(0, 1); // lines 1-2
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

        extract_rows_by_selector(data, &range(0, 1), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"a\0b\0");
    }

    #[test]
    fn emits_json_rows_with_line_numbers() {
        let data = b"a\nb\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.json = true;

        extract_rows_by_selector(data, &range(1, 1), Newlines::Lf, &config, &mut output).unwrap();

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

        let selector = indices(&[0, 2, 4]); // a, c, e

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"a\nc\ne\n");
    }

    #[test]
    fn clamps_range_to_available_rows() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = range(0, 100); // Request more than exists

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 2); // Only 2 lines exist
    }

    #[test]
    fn tail_start_matches_forward_line_walk() {
        let inputs: [&[u8]; 8] = [
            b"",
            b"\n",
            b"\n\n",
            b"no trailing newline",
            b"line1\nline2\nline3\n",
            b"line1\nline2\nline3",
            b"a\n\nb\n\n",
            b"\nleading\nblank\n",
        ];

        for data in inputs {
            let lines: Vec<&[u8]> = LineIter::new(data, Newlines::Lf).collect();
            for wanted in 1..=lines.len() + 2 {
                let (start, found) = tail_start_lf(data, NonZeroUsize::new(wanted).unwrap());
                let expected = wanted.min(lines.len());
                assert_eq!(found, expected, "line count for {:?} tail {}", data, wanted);

                let span: Vec<&[u8]> = LineIter::new(&data[start..], Newlines::Lf).collect();
                assert_eq!(
                    span,
                    lines[lines.len() - expected..],
                    "span for {:?} tail {}",
                    data,
                    wanted
                );

                // The prefix carries exactly the lines the span skips, which is what
                // absolute line numbering counts.
                let skipped = LineIter::new(&data[..start], Newlines::Lf).count();
                assert_eq!(
                    skipped,
                    lines.len() - expected,
                    "prefix for {:?} tail {}",
                    data,
                    wanted
                );
            }
        }
    }

    #[test]
    fn numbers_tail_rows_absolutely() {
        let data = b"line1\nline2\nline3\nline4\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.show_line_numbers = true;

        extract_rows_by_selector(data, &tail(2), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"3:line3\n4:line4\n");
    }

    #[test]
    fn extracts_last_n_rows_in_unicode_mode() {
        let data = "line1\u{2028}line2\u{2028}line3\u{2028}".as_bytes();
        let mut output = Vec::new();

        let count = extract_rows_by_selector(
            data,
            &tail(2),
            Newlines::Unicode,
            &text_config(),
            &mut output,
        )
        .unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"line2\nline3\n");
    }

    #[test]
    fn returns_all_rows_when_tail_exceeds_length() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = tail(100);

        let count =
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut output)
                .unwrap();

        assert_eq!(count, 2); // Return all lines
        assert_eq!(output, b"line1\nline2\n");
    }

    // region: Streaming Tests

    #[test]
    fn streams_every_selector_identically_to_the_whole_buffer() {
        let inputs: [&[u8]; 6] = [
            b"",
            b"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\n",
            b"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl",
            b"\n\nblank\n\nlines\n\n",
            b"one",
            b"a-line-far-longer-than-the-window\nshort\nanother-very-long-line-again\n",
        ];
        let selectors = [
            indices(&[0]),
            indices(&[4]),
            indices(&[0, 4, 9]),
            range(1, 3),
            range(9, 19),
            tail(1),
            tail(3),
            tail(1000),
            every(1),
            every(5),
        ];
        let mut numbered = text_config();
        numbered.show_line_numbers = true;
        let mut json = text_config();
        json.json = true;

        for data in inputs {
            for selector in &selectors {
                for config in [&text_config(), &numbered, &json] {
                    for newlines in [Newlines::Lf, Newlines::Unicode] {
                        let mut expected = Vec::new();
                        let expected_count = extract_rows_by_selector(
                            data,
                            selector,
                            newlines,
                            config,
                            &mut expected,
                        )
                        .unwrap();

                        for capacity in [7, 13, 64, 4096] {
                            let (output, count) =
                                extract_streamed(data, capacity, selector, newlines, config);
                            assert_eq!(
                                output, expected,
                                "output for {:?} at capacity {}",
                                data, capacity
                            );
                            assert_eq!(
                                count, expected_count,
                                "count for {:?} at capacity {}",
                                data, capacity
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn streams_a_record_wider_than_the_window() {
        let long = vec![b'x'; 5000];
        let mut data = Vec::new();
        data.extend_from_slice(b"first\n");
        data.extend_from_slice(&long);
        data.extend_from_slice(b"\nlast\n");

        let (output, count) =
            extract_streamed(&data, 7, &range(1, 1), Newlines::Lf, &text_config());

        assert_eq!(count, 1);
        assert_eq!(output.len(), long.len() + 1);
        assert_eq!(&output[..long.len()], &long[..]);
    }

    #[test]
    fn streams_tail_with_absolute_line_numbers() {
        let data = b"line1\nline2\nline3\nline4\n";
        let mut config = text_config();
        config.show_line_numbers = true;

        let (output, count) = extract_streamed(data, 7, &tail(2), Newlines::Lf, &config);

        assert_eq!(count, 2);
        assert_eq!(output, b"3:line3\n4:line4\n");
    }

    #[test]
    fn streamed_tail_bounds_memory_by_the_request() {
        let mut data = Vec::new();
        for index in 0..10_000 {
            data.extend_from_slice(format!("line{}\n", index).as_bytes());
        }

        let mut refill = Refill::new(&data[..], 64);
        let mut ring = TailRing::new(NonZeroUsize::new(3).unwrap());
        refill
            .for_each_window(CutAfter::LineFeed, |window| {
                for line in LineIter::new(window, Newlines::Lf) {
                    ring.push(line);
                }
                Ok(())
            })
            .unwrap();

        // Three retained lines out of ten thousand, however many windows they spanned.
        assert_eq!(ring.lines.len(), 3);
        assert_eq!(ring.first_index, 9_997);
        let retained: usize = ring.lines.iter().map(|line| line.capacity()).sum();
        assert!(retained < 128, "ring held {} bytes", retained);
    }

    #[test]
    fn bounded_selectors_stop_reading_the_stream() {
        let data =
            b"l01\nl02\nl03\nl04\nl05\nl06\nl07\nl08\nl09\nl10\nl11\nl12\nl13\nl14\nl15\nl16\n";
        let window = 40; // Ten lines, so the first window already passes index 4.

        for selector in [indices(&[4]), range(0, 4)] {
            let mut expected = Vec::new();
            extract_rows_by_selector(data, &selector, Newlines::Lf, &text_config(), &mut expected)
                .unwrap();

            let reader = BudgetedReader {
                data,
                position: 0,
                budget: window,
            };
            let mut refill = Refill::new(reader, window);
            let mut output = Vec::new();
            stream_rows(
                &mut refill,
                &selector,
                Newlines::Lf,
                &text_config(),
                &mut output,
            )
            .expect("a bounded selector must stop before a second read");
            assert_eq!(output, expected);
        }

        // The same budget proves itself by failing the selector that must read on.
        let reader = BudgetedReader {
            data,
            position: 0,
            budget: window,
        };
        let mut refill = Refill::new(reader, window);
        let mut output = Vec::new();
        assert!(stream_rows(
            &mut refill,
            &every(5),
            Newlines::Lf,
            &text_config(),
            &mut output
        )
        .is_err());
    }

    // endregion: Streaming Tests
}
