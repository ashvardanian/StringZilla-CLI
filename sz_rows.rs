//! Row selection by index, range, stride or tail, standing in for `sed -n 'Np'`, `head`, `tail`
//! and `awk 'NR==N'`.
//!
//! One selector covers what those four spell four ways, and every selection is answered in a
//! single forward pass: scattered indices are walked in ascending order rather than re-scanned per
//! index, and a stride never materializes the rows it skips.
//!
//! `--tail` is the one selector that cannot be answered forward, since the count is relative to an
//! end the scan has not reached, so it keeps a ring of the last N line spans instead of buffering
//! the input.
//!
//! Exit: 0 wrote a row, 1 ran and selected nothing, 2 could not run.

use std::collections::VecDeque;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use clap::{error::ErrorKind, CommandFactory, Parser, ValueEnum};
use stringzilla::sz;

use shared::*;

/// Extract rows from text files
#[derive(Parser)]
#[command(name = "sz-rows")]
#[command(version, about = "SIMD-accelerated row extraction (like sed -n, head, tail)", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Row(s) to extract: single (5), list (1,5,10), or range (10-20)
    #[arg(long, conflicts_with_all = ["tail", "every"])]
    rows: Option<String>,

    /// Extract last N lines (like tail -n)
    #[arg(long, value_parser = parse_at_least_one, conflicts_with_all = ["rows", "every"])]
    tail: Option<NonZeroUsize>,

    /// Extract every Nth line
    #[arg(long, value_parser = parse_at_least_one, conflicts_with_all = ["rows", "tail"])]
    every: Option<NonZeroUsize>,

    /// Which columns each record carries, comma-separated; none by default
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        help_heading = "Output Formats"
    )]
    fields: Vec<Field>,

    /// How many characters of a line hash to print [default: 8]
    #[arg(long, value_parser = parse_hash_width, help_heading = "Output Formats")]
    hash_width: Option<usize>,

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

    /// NUL-terminate each output record instead of newline, for `xargs -0`
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any row was extracted, 1 otherwise
    #[arg(long, conflicts_with_all = ["format", "null", "fields"], help_heading = "Output Formats")]
    quiet: bool,
}

/// One column a record can carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Field {
    /// The 1-based line number
    LineNumbers,
    /// A hash of the line's content, which survives edits elsewhere in the file
    LineHashes,
    /// A hash of the whole file, as `sz-replace --expect-hash` compares against
    FileHash,
}

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    /// The row itself, terminated by a newline or a NUL.
    Text,
    /// JSON Lines, one record per output line.
    Json,
}

impl Args {
    /// Whether the whole input has to be in hand before the first row is written: a text run
    /// prints the file hash as a header, where JSON names the file on its closing record.
    fn hashes_before_writing(&self) -> bool {
        self.fields.contains(&Field::FileHash) && self.format != Format::Json
    }
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// The constraints clap cannot express: `conflicts_with` fires on a flag's presence,
/// never on its value.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.rows.is_none() && args.tail.is_none() && args.every.is_none() {
        return Err(reject("must specify --rows, --tail, or --every"));
    }
    if args.hash_width.is_some() && !args.fields.contains(&Field::LineHashes) {
        return Err(reject(
            "--hash-width sizes a line hash, so it needs --fields line-hashes",
        ));
    }
    if args.quiet && args.fields.contains(&Field::FileHash) {
        return Err(reject(
            "--quiet prints nothing, so it has nowhere to report --fields file-hash",
        ));
    }
    if args.format == Format::Json {
        if args.fields.contains(&Field::LineNumbers) {
            return Err(reject("--format json already carries a line number"));
        }
        if args.null {
            return Err(reject("--format json cannot be combined with --null"));
        }
    }
    Ok(())
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

/// Read one 1-based row number into the 0-based index the walk uses. Every message names the
/// token it read, which is why the empty case is separate: `2-` splits into `2` and nothing.
fn parse_row(token: &str) -> Result<usize, String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("a row number is missing; write both ends of a range, as in `2-5`".into());
    }
    let row: usize = token
        .parse()
        .map_err(|_| format!("`{token}` is not a row number"))?;
    if row == 0 {
        return Err("row numbers start at 1".into());
    }
    Ok(row - 1)
}

/// Parse row specification into a RowSelector
fn parse_rows(spec: &str) -> Result<RowSelector, String> {
    if spec.trim().is_empty() {
        return Err("no rows given; pass a row, a list, or a range".into());
    }
    // Check if it's a simple range (no commas)
    if !spec.contains(',') && spec.contains('-') {
        let parts: Vec<&str> = spec.split('-').collect();
        if parts.len() == 2 {
            let (start, end) = (parse_row(parts[0])?, parse_row(parts[1])?);
            if start > end {
                return Err(format!(
                    "`{spec}` runs backwards; a range reads low to high"
                ));
            }
            return Ok(RowSelector::Forward(ForwardSelector::Range(start, end)));
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
                return Err(format!("`{part}` is not a range; a range has two ends"));
            }
            let (start, end) = (parse_row(parts[0])?, parse_row(parts[1])?);
            if start > end {
                return Err(format!(
                    "`{part}` runs backwards; a range reads low to high"
                ));
            }
            indices.extend(start..=end);
        } else {
            indices.push(parse_row(part)?);
        }
    }

    if indices.is_empty() {
        return Err("no rows given; pass a row, a list, or a range".into());
    }

    // Rows arrive in order, so the walk reads a sorted list with a cursor rather than
    // hashing every line against a set that is usually one to three elements wide.
    indices.sort_unstable();
    indices.dedup();

    Ok(RowSelector::Forward(ForwardSelector::Indices(indices)))
}

/// How one extracted row is written out: behind the fields the record carries and closed by
/// the requested terminator, or as a JSON record in a stream a summary record closes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rendering {
    Terminated(Terminator),
    Json,
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig<'a> {
    rendering: Rendering,
    /// Which columns each record carries; `Field::FileHash` costs the run the whole input.
    fields: &'a [Field],
    hash_width: usize,
    /// The input's path, carried into the JSON envelope.
    path: &'a str,
}

impl OutputConfig<'_> {
    #[inline]
    fn carries(&self, field: Field) -> bool {
        self.fields.contains(&field)
    }

    /// The hash naming a line, when a record was asked to carry one. Taken from the whole
    /// line rather than the bytes a row prints, so it agrees with `sz-find`.
    #[inline]
    fn naming(&self, line: &NamedLine) -> Option<u64> {
        self.carries(Field::LineHashes).then(|| line.hash())
    }
}

/// Write one extracted row. `index` is zero-based; records report it one-based. `hash` names
/// the line and its terminator together, which only the caller has: a row is printed without
/// whatever ended it.
fn write_row(
    output: &mut dyn Write,
    config: &OutputConfig,
    line: &[u8],
    index: usize,
    hash: Option<u64>,
) -> io::Result<()> {
    let mut buffer = [0u8; HASH_CHARS];
    let named = hash.map(|hash| format_hash(&mut buffer, hash, config.hash_width));

    let terminator = match config.rendering {
        Rendering::Json => return write_line_record(output, config.path, line, index, named),
        Rendering::Terminated(terminator) => terminator,
    };
    if config.carries(Field::LineNumbers) {
        write!(output, "{}:", index + 1)?;
    }
    if let Some(named) = named {
        write!(output, "{named}:")?;
    }
    output.write_all(line)?;
    output.write_all(&[terminator.as_byte()])
}

/// Write the whole file's hash: ahead of the rows in text, where no reader mistakes it for
/// one, and as the summary record that closes a JSON stream.
fn write_file_hash(output: &mut dyn Write, config: &OutputConfig, hash: u64) -> io::Result<()> {
    let mut buffer = [0u8; HASH_CHARS];
    let named = format_hash(&mut buffer, hash, HASH_CHARS);
    match config.rendering {
        Rendering::Terminated(_) => writeln!(output, "{}  {named}", config.path),
        Rendering::Json => {
            write!(output, r#"{{"type":"summary","data":{{"path":"#)?;
            json_text_field_to(output, config.path.as_bytes())?;
            writeln!(output, r#","file_hash":"{named}"}}}}"#)
        }
    }
}

// region: Tail Scan

/// Byte offset at which the last `wanted` LF-terminated lines begin, and how many lines that
/// span covers — `min(wanted, total lines)`. A trailing newline terminates the last line
/// rather than starting an empty one, matching [`LineIter`]: `b""` is zero lines and `b"\n"`
/// is one empty line.
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

/// What [`extract_forward`] carries between windows, so a second call resumes where the
/// first stopped. `Copy` and lifetime-free, and it allocates nothing.
#[derive(Clone, Copy, Default)]
struct RowsState {
    /// Absolute zero-based index of the next line to arrive.
    next_index: usize,
    /// How far into a sorted index list the walk has come.
    cursor: usize,
    /// Rows written so far, which the run returns as its row count.
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
fn extract_forward(
    data: &[u8],
    selector: &ForwardSelector,
    newlines: Newlines,
    config: &OutputConfig,
    state: &mut RowsState,
    output: &mut dyn Write,
) -> io::Result<ControlFlow<()>> {
    for named in named_lines(data, newlines) {
        let line = named.as_cut;
        if selector.exhausted(state) {
            return Ok(ControlFlow::Break(()));
        }
        let index = state.next_index;
        state.next_index += 1;
        if selector.takes(index, state) {
            write_row(output, config, line, index, config.naming(&named))?;
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
fn extract_data(
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
            let _early_stop = extract_forward(data, forward, newlines, config, &mut state, output)?;
        }
        RowSelector::Tail(wanted) => match newlines {
            Newlines::Lf => {
                let (start, lines_found) = tail_start_lf(data, *wanted);
                // Absolute numbering costs a newline pass over the skipped prefix,
                // so it is paid only when a record carries a line number.
                let first_index =
                    if config.rendering == Rendering::Json || config.carries(Field::LineNumbers) {
                        LineIter::new(&data[..start], newlines).count()
                    } else {
                        0
                    };
                for (index_in_span, named) in named_lines(&data[start..], newlines).enumerate() {
                    let line = named.as_cut;
                    write_row(
                        output,
                        config,
                        line,
                        first_index + index_in_span,
                        config.naming(&named),
                    )?;
                    state.emitted += 1;
                }
                debug_assert_eq!(
                    state.emitted, lines_found,
                    "tail span must hold the counted lines"
                );
            }
            // The Unicode newline set has no reverse kernel — `rfind_byteset` cannot
            // express the multi-byte NEL, LS and PS — so the last N lines come from a
            // forward walk into a ring: O(N) memory, not O(file).
            Newlines::Unicode => {
                let wanted = wanted.get();
                let mut ring: VecDeque<(usize, NamedLine)> =
                    VecDeque::with_capacity(wanted.min(TAIL_RING_RESERVE));
                for (index, named) in named_lines(data, newlines).enumerate() {
                    if ring.len() == wanted {
                        ring.pop_front();
                    }
                    ring.push_back((index, named));
                }
                for (index, named) in ring {
                    write_row(output, config, named.as_cut, index, config.naming(&named))?;
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
/// be overwritten. An evicted buffer is refilled in place, so the footprint stays bounded by
/// `wanted × longest line`.
struct TailRing {
    /// How many lines the ring keeps, which `--tail` sets.
    wanted: NonZeroUsize,
    /// Absolute zero-based index of the oldest retained line.
    first_index: usize,
    /// The retained lines, oldest first, each with the hash naming it. The hash is kept
    /// rather than the terminator it covers, since bounding memory is what the ring is for.
    lines: VecDeque<(Vec<u8>, Option<u64>)>,
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
    fn push(&mut self, line: &[u8], hash: Option<u64>) {
        let mut buffer = if self.lines.len() == self.wanted.get() {
            self.first_index += 1;
            self.lines
                .pop_front()
                .map_or_else(Vec::new, |(buffer, _)| buffer)
        } else {
            Vec::new()
        };
        buffer.clear();
        buffer.extend_from_slice(line);
        self.lines.push_back((buffer, hash));
    }

    /// Write the retained lines in arrival order, numbered absolutely.
    fn write_to(&self, config: &OutputConfig, output: &mut dyn Write) -> io::Result<usize> {
        for (offset, (line, hash)) in self.lines.iter().enumerate() {
            write_row(output, config, line, self.first_index + offset, *hash)?;
        }
        Ok(self.lines.len())
    }
}

/// Drive [`extract_forward`] over a walk, one window at a time. The selector's early stop
/// ends the loop, so `--rows 5` stops reading rather than draining the rest of the stream.
fn extract_forward_stream(
    walk: &mut Windows,
    selector: &ForwardSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut state = RowsState::default();
    let mut consumed = 0;
    while let Some((window, _)) = walk.next(newlines.into(), consumed)? {
        consumed = window.len();
        if extract_forward(window, selector, newlines, config, &mut state, output)?.is_break() {
            break;
        }
    }
    Ok(state.emitted)
}

/// Drive the tail ring over a walk. A stream cannot be scanned backward, so the last
/// `wanted` lines are the ones a forward walk still holds when the reader runs out.
fn extract_tail_stream(
    walk: &mut Windows,
    wanted: NonZeroUsize,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut ring = TailRing::new(wanted);
    let mut consumed = 0;
    while let Some((window, _)) = walk.next(newlines.into(), consumed)? {
        consumed = window.len();
        for named in named_lines(window, newlines) {
            ring.push(named.as_cut, config.naming(&named));
        }
    }
    ring.write_to(config, output)
}

/// Extract rows from a walk, choosing the ring for `--tail` and the forward walk for the
/// rest. Line numbers are absolute in both, matching the whole-slice paths.
fn extract_stream(
    walk: &mut Windows,
    selector: &RowSelector,
    newlines: Newlines,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    match selector {
        RowSelector::Forward(forward) => {
            extract_forward_stream(walk, forward, newlines, config, output)
        }
        RowSelector::Tail(wanted) => extract_tail_stream(walk, *wanted, newlines, config, output),
    }
}

/// Read whatever a selector's early stop left in the stream, through the same window. A
/// digest of a prefix is indistinguishable from a digest of the file.
fn drain_remaining(walk: &mut Windows) -> io::Result<()> {
    let mut consumed = 0;
    while let Some((window, _)) = walk.next(CutAfter::Anywhere, consumed)? {
        consumed = window.len();
    }
    Ok(())
}

// endregion: Streaming

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let selector = if let Some(rows) = &args.rows {
        parse_rows(rows)
            .map_err(|message| Args::command().error(ErrorKind::ValueValidation, message))?
    } else if let Some(wanted) = args.tail {
        RowSelector::Tail(wanted)
    } else {
        RowSelector::Forward(ForwardSelector::Every(args.every.expect("validated above")))
    };

    let path = args.input.as_deref().unwrap_or("-");
    let input = if args.hashes_before_writing() {
        get_input(args.input.as_deref())
    } else {
        get_input_streaming(args.input.as_deref())
    }
    .at(path)?;

    let config = OutputConfig {
        rendering: match args.format {
            Format::Json => Rendering::Json,
            Format::Text => Rendering::Terminated(Terminator::from_null(args.null)),
        },
        fields: &args.fields,
        hash_width: args.hash_width.unwrap_or(DEFAULT_HASH_WIDTH),
        path,
    };

    // A quiet run still extracts, so the row count that answers it stays honest.
    let destination = if args.quiet {
        Destination::Discard
    } else {
        Destination::Stdout
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let mut walk = Windows::over(input);
    // Before the first window is filled: a hash installed after that would cover everything
    // but the bytes already read.
    if config.carries(Field::FileHash) {
        walk.hash_stream();
    }
    let mapped_hash = walk
        .whole()
        .filter(|_| config.carries(Field::FileHash))
        .map(content_hash);

    let emitted = destination.write("sz-rows", output, |writer| {
        if let Some(data) = walk.whole() {
            if let (Some(hash), Rendering::Terminated(_)) = (mapped_hash, config.rendering) {
                write_file_hash(writer, &config, hash)?;
            }
            return extract_data(data, &selector, newlines, &config, writer);
        }
        let emitted = extract_stream(&mut walk, &selector, newlines, &config, writer)?;
        if config.carries(Field::FileHash) {
            drain_remaining(&mut walk)?;
        }
        Ok(emitted)
    })?;
    let file_hash = mapped_hash.or_else(|| walk.digest());

    if let (Some(hash), Rendering::Json) = (file_hash, config.rendering) {
        write_file_hash(output, &config, hash).at("-")?;
    }

    output.flush().at("-")?;
    Ok(Status::from_found(emitted > 0))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-rows", run(&args, &mut output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn text_config() -> OutputConfig<'static> {
        OutputConfig {
            rendering: Rendering::Terminated(Terminator::Newline),
            fields: &[],
            hash_width: DEFAULT_HASH_WIDTH,
            path: "-",
        }
    }

    #[test]
    fn names_the_whole_file_the_same_streamed_as_mapped() {
        // The token `sz-replace --expect-hash` wants, from the same read that named the
        // lines. A row selector stops early, so a streamed run has to finish the pipe: a
        // digest of a prefix is indistinguishable from a digest of the file.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("rows.txt");
        let corpus: String = (1..=400).map(|row| format!("line {row}\n")).collect();
        std::fs::write(&path, &corpus).unwrap();

        let args = Args::try_parse_from([
            "sz-rows",
            "--rows",
            "2-3",
            "--format",
            "json",
            "--fields",
            "file-hash",
            path.to_str().unwrap(),
        ])
        .unwrap();
        let mut mapped = Vec::new();
        run(&args, &mut mapped).unwrap();

        let mut buffer = [0u8; HASH_CHARS];
        let expected = format_hash(&mut buffer, content_hash(corpus.as_bytes()), HASH_CHARS);
        let text = String::from_utf8(mapped).unwrap();
        assert!(
            text.contains(&format!(r#""file_hash":"{expected}""#)),
            "{text}"
        );
        // Two rows and the record that closes the file, and no more: a whole-file value has
        // one place to be.
        assert_eq!(text.lines().count(), 3, "{text}");
    }

    #[test]
    fn refuses_a_file_hash_it_would_have_nowhere_to_report() {
        assert!(Args::try_parse_from([
            "sz-rows",
            "--rows",
            "1",
            "--quiet",
            "--fields",
            "file-hash",
            "f",
        ])
        .and_then(|args| validate(&args).map(|_| args))
        .is_err());
    }

    #[test]
    fn names_the_token_it_read_in_every_row_and_column_error() {
        // A range with a missing end used to report `Invalid row number: `, naming nothing.
        for (spec, expected) in [
            ("2-", "a row number is missing"),
            ("abc", "`abc` is not a row number"),
            ("0", "row numbers start at 1"),
            ("5-2", "runs backwards"),
            ("1-2-3", "is not a range"),
            ("", "no rows given"),
        ] {
            let Err(error) = parse_rows(spec) else {
                panic!("`{spec}` must not parse");
            };
            assert!(error.contains(expected), "`{spec}` reported `{error}`");
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

    /// Where a `capacity`-byte window ends: at the last record boundary inside it, or the
    /// first one past it when no record fits, which is what a stream widens to.
    fn window_end(data: &[u8], capacity: usize, newlines: Newlines) -> usize {
        if capacity >= data.len() {
            return data.len();
        }
        last_cut(&data[..capacity], newlines.into()).unwrap_or_else(|| {
            LineSpans::new(data, newlines)
                .next()
                .map_or(data.len(), |span| span.length)
        })
    }

    /// Extract the same way [`extract_data`] does, one `capacity`-byte window at a time.
    /// [`Windows`] is fixed at [`DEFAULT_WINDOW_BYTES`], which no test-sized input reaches,
    /// so the seams are drawn here.
    fn extract_streamed(
        data: &[u8],
        capacity: usize,
        selector: &RowSelector,
        newlines: Newlines,
        config: &OutputConfig,
    ) -> (Vec<u8>, usize) {
        let mut windows = Vec::new();
        let mut rest = data;
        while !rest.is_empty() {
            let end = window_end(rest, capacity, newlines);
            windows.push(&rest[..end]);
            rest = &rest[end..];
        }

        let mut output = Vec::new();
        let count = match selector {
            RowSelector::Forward(forward) => {
                let mut state = RowsState::default();
                for window in windows {
                    let flow =
                        extract_forward(window, forward, newlines, config, &mut state, &mut output)
                            .unwrap();
                    if flow.is_break() {
                        break;
                    }
                }
                state.emitted
            }
            RowSelector::Tail(wanted) => {
                let mut ring = TailRing::new(*wanted);
                for window in windows {
                    for named in named_lines(window, newlines) {
                        ring.push(named.as_cut, config.naming(&named));
                    }
                }
                ring.write_to(config, &mut output).unwrap()
            }
        };
        (output, count)
    }

    /// A reader that fails once a budget of bytes has been handed out, so a test can prove
    /// a selector stopped reading rather than merely stopped writing.
    struct BudgetedReader {
        data: Vec<u8>,
        position: usize,
        budget: usize,
    }

    impl Read for BudgetedReader {
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
    fn counts_rows_a_quiet_run_never_writes() {
        // `--quiet` extracts into a sink, so the count that becomes the exit status
        // is the same one a printing run would report.
        let data = b"line1\nline2\nline3\n";
        let mut discard = io::sink();

        let emitted = extract_data(
            data,
            &range(0, 0),
            Newlines::Lf,
            &text_config(),
            &mut discard,
        )
        .unwrap();
        assert!(Status::from_found(emitted > 0) == Status::Success);

        let emitted = extract_data(
            data,
            &indices(&[998]),
            Newlines::Lf,
            &text_config(),
            &mut discard,
        )
        .unwrap();
        assert!(Status::from_found(emitted > 0) == Status::NoResult);
    }

    #[test]
    fn quiet_rejects_the_flags_it_would_ignore() {
        // Under `--quiet` the exit status is the whole output, so nothing that shapes a
        // record can reach it.
        let rejected = [
            vec!["sz-rows", "--rows", "1", "--quiet", "--format", "json"],
            vec!["sz-rows", "--rows", "1", "--quiet", "--null"],
            vec![
                "sz-rows",
                "--rows",
                "1",
                "--quiet",
                "--fields",
                "line-numbers",
            ],
        ];
        for arguments in rejected {
            assert!(
                Args::try_parse_from(&arguments).is_err(),
                "{:?} must be rejected",
                arguments
            );
        }

        // Which rows are selected still steers the status, so the selectors compose.
        let accepted = [
            vec!["sz-rows", "--quiet", "--tail", "3"],
            vec!["sz-rows", "--quiet", "--every", "5"],
            vec!["sz-rows", "--rows", "1", "--quiet", "--utf8"],
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
    fn declares_no_short_flags() {
        assert!(Args::command()
            .get_arguments()
            .all(|argument| argument.get_short().is_none()
                || matches!(argument.get_short(), Some('h') | Some('V'))));
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
                "rows",
                "tail",
                "every",
                "fields",
                "hash-width",
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
        let args = Args::parse_from(["sz-rows", "--rows", "1", "missing.txt"]);
        let Err(failure) = run(&args, &mut io::sink()) else {
            panic!("a missing input must fail");
        };
        assert!(
            failure.to_string().starts_with("missing.txt: "),
            "{}",
            failure
        );
    }

    #[test]
    fn rejects_what_json_would_silently_ignore() {
        // `--line-numbers` used to be a no-op under JSON, which numbers every record.
        let parsed = |arguments: &[&str]| Args::try_parse_from(arguments).unwrap();
        assert!(validate(&parsed(&[
            "sz-rows",
            "--rows",
            "1",
            "--format",
            "json",
            "--fields",
            "line-numbers"
        ]))
        .is_err());
        assert!(validate(&parsed(&[
            "sz-rows", "--rows", "1", "--format", "json", "--null"
        ]))
        .is_err());
        assert!(validate(&parsed(&[
            "sz-rows",
            "--rows",
            "1",
            "--fields",
            "line-numbers"
        ]))
        .is_ok());
        assert!(validate(&parsed(&["sz-rows", "file.txt"])).is_err());
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
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 1);
        assert_eq!(output, b"line2\n");
    }

    #[test]
    fn extracts_row_range() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = range(1, 3); // lines 2-4

        let count =
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline3\nline4\n");
    }

    #[test]
    fn extracts_last_n_rows() {
        let data = b"line1\nline2\nline3\nline4\nline5\n";
        let mut output = Vec::new();

        let selector = tail(2);

        let count =
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 2);
        assert_eq!(output, b"line4\nline5\n");
    }

    #[test]
    fn extracts_every_nth_row() {
        let data = b"line1\nline2\nline3\nline4\nline5\nline6\n";
        let mut output = Vec::new();

        let selector = every(2); // every 2nd line

        let count =
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"line2\nline4\nline6\n");
    }

    #[test]
    fn prefixes_rows_with_line_numbers() {
        let data = b"line1\nline2\nline3\n";
        let mut output = Vec::new();

        let selector = range(0, 1); // lines 1-2
        let mut config = text_config();
        config.fields = &[Field::LineNumbers];

        extract_data(data, &selector, Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"1:line1\n2:line2\n");
    }

    #[test]
    fn terminates_rows_with_nul() {
        let data = b"a\nb\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.rendering = Rendering::Terminated(Terminator::Null);

        extract_data(data, &range(0, 1), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"a\0b\0");
    }

    #[test]
    fn emits_json_rows_with_line_numbers() {
        let data = b"a\nb\n";
        let mut output = Vec::new();
        let mut config = text_config();
        config.rendering = Rendering::Json;

        extract_data(data, &range(1, 1), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                r#"{"type":"line","data":{"path":{"text":"-"},"#,
                r#""text":{"text":"b"},"line_number":2}}"#,
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
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

        assert_eq!(count, 3);
        assert_eq!(output, b"a\nc\ne\n");
    }

    #[test]
    fn clamps_range_to_available_rows() {
        let data = b"line1\nline2\n";
        let mut output = Vec::new();

        let selector = range(0, 100); // Request more than exists

        let count =
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

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
        config.fields = &[Field::LineNumbers];

        extract_data(data, &tail(2), Newlines::Lf, &config, &mut output).unwrap();

        assert_eq!(output, b"3:line3\n4:line4\n");
    }

    #[test]
    fn extracts_last_n_rows_in_unicode_mode() {
        let data = "line1\u{2028}line2\u{2028}line3\u{2028}".as_bytes();
        let mut output = Vec::new();

        let count = extract_data(
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
            extract_data(data, &selector, Newlines::Lf, &text_config(), &mut output).unwrap();

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
        numbered.fields = &[Field::LineNumbers];
        let mut json = text_config();
        json.rendering = Rendering::Json;

        for data in inputs {
            for selector in &selectors {
                for config in [&text_config(), &numbered, &json] {
                    for newlines in [Newlines::Lf, Newlines::Unicode] {
                        let mut expected = Vec::new();
                        let expected_count =
                            extract_data(data, selector, newlines, config, &mut expected).unwrap();

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
        config.fields = &[Field::LineNumbers];

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
                    ring.push(line, None);
                }
                Ok(())
            })
            .unwrap();

        // Three retained lines out of ten thousand, however many windows they spanned.
        assert_eq!(ring.lines.len(), 3);
        assert_eq!(ring.first_index, 9_997);
        let retained: usize = ring.lines.iter().map(|(line, _)| line.capacity()).sum();
        assert!(retained < 128, "ring held {} bytes", retained);
    }

    #[test]
    fn bounded_selectors_stop_reading_the_stream() {
        // A budget of exactly one window, over an input needing two, so a selector answered
        // by the first proves itself by never asking for the second.
        let mut data = Vec::new();
        while data.len() < DEFAULT_WINDOW_BYTES * 2 {
            data.extend_from_slice(format!("l{:07}\n", data.len()).as_bytes());
        }
        let budget = DEFAULT_WINDOW_BYTES;
        let piped = |data: &[u8]| {
            Windows::over(InputSource::Pipe(Box::new(BudgetedReader {
                data: data.to_vec(),
                position: 0,
                budget,
            })))
        };

        for selector in [indices(&[4]), range(0, 4)] {
            let mut expected = Vec::new();
            extract_data(
                &data,
                &selector,
                Newlines::Lf,
                &text_config(),
                &mut expected,
            )
            .unwrap();

            let mut walk = piped(&data);
            let mut output = Vec::new();
            extract_stream(
                &mut walk,
                &selector,
                Newlines::Lf,
                &text_config(),
                &mut output,
            )
            .expect("a bounded selector must stop before a second read");
            assert_eq!(output, expected);
        }

        // The same budget proves itself by failing the selector that must read on.
        let mut walk = piped(&data);
        let mut output = Vec::new();
        assert!(extract_stream(
            &mut walk,
            &every(5),
            Newlines::Lf,
            &text_config(),
            &mut output
        )
        .is_err());
    }

    // endregion: Streaming Tests
}
