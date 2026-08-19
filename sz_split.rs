//! Split one input into many, standing in for `split` and `csplit`.
//!
//! Four ways to cut: by line count, by byte budget, into a fixed number of shares, or before every
//! line matching a needle. The last has no `split` equivalent and is what makes this a `csplit`
//! replacement too, without the regex.
//!
//! Every chunk is a verbatim byte range of the input. Nothing is re-emitted line by line, so CR,
//! CRLF, NEL, LS and PS survive and an input that never ended in a terminator does not gain one.
//! `--repeat-header` repeats a prefix into each chunk, which is the one case where a chunk is not a
//! single contiguous range.
//!
//! Exit: 0 wrote a chunk, 1 wrote none, 2 could not run.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use clap::{CommandFactory, Parser};
use stringzilla::sz;

use shared::*;

/// Split files into smaller chunks
#[derive(Parser)]
#[command(name = "sz-split")]
#[command(version, about = "SIMD-accelerated file splitting", long_about = None)]
// Exactly one way of deciding where a chunk ends.
#[command(group(clap::ArgGroup::new("chunking").required(true)))]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output prefix for split files
    #[arg(default_value = "x")]
    prefix: String,

    /// Number of lines per output file
    #[arg(long, value_name = "N", value_parser = parse_at_least_one,
          group = "chunking", help_heading = "Chunk Size")]
    chunk_lines: Option<NonZeroUsize>,

    /// Bytes per output file, never splitting a line; accepts K/M/G/T and Ki/Mi/Gi/Ti
    #[arg(long, value_name = "SIZE", value_parser = parse_size,
          group = "chunking", help_heading = "Chunk Size")]
    chunk_bytes: Option<NonZeroUsize>,

    /// Cut into exactly N files of near-equal bytes at line boundaries; needs a seekable input
    #[arg(long, value_name = "N", value_parser = parse_at_least_one,
          group = "chunking", help_heading = "Chunk Size")]
    chunk_count: Option<NonZeroUsize>,

    /// Start a new chunk at each line beginning with LITERAL, as `csplit` does with a regex
    #[arg(long, value_name = "LITERAL", value_parser = parse_pattern,
          group = "chunking", help_heading = "Chunk Size")]
    chunk_pattern: Option<String>,

    /// Fold case when matching --chunk-pattern; implies --utf8
    // Needing a pattern is checked in `validate`: clap's `requires` does not fire for an
    // argument whose group another member has already satisfied.
    #[arg(long)]
    ignore_case: bool,

    /// Repeat the input's first N lines atop every chunk, so each one parses alone
    #[arg(long, value_name = "N", num_args = 0..=1, require_equals = true,
          default_missing_value = "1", value_parser = parse_at_least_one,
          help_heading = "Chunk Size")]
    repeat_header: Option<NonZeroUsize>,

    /// Treat the input as UTF-8 text
    #[arg(long)]
    utf8: bool,

    /// Suffix length, so 2 gives aa, ab, ac
    #[arg(long, value_parser = parse_at_least_one, default_value = "2")]
    suffix_length: NonZeroUsize,

    /// Announce every written chunk on stdout
    #[arg(
        long,
        value_name = "FORMAT",
        value_enum,
        default_value = "none",
        help_heading = "Output Formats"
    )]
    format: Format,

    /// NUL-terminate each announced path instead of newline, for `xargs -0`
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Suppress all output; exit 0 if any chunk was written, 1 otherwise
    #[arg(long, conflicts_with_all = ["format", "null"], help_heading = "Output Formats")]
    quiet: bool,
}

/// How a written chunk is announced.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Format {
    /// Nothing, so a run that asked for no announcement writes no stdout at all.
    None,
    /// The path alone, one per record.
    #[value(alias = "text")]
    Paths,
    /// A JSON Lines record carrying the tallies.
    Json,
}

/// The chunk name suffix for `index` — aa, ab, ac, ... az, ba, bb, ... — or `None` once
/// `index` needs more than `length` characters, where wrapping would reuse an earlier name.
fn generate_suffix(index: usize, length: usize) -> Option<String> {
    let mut suffix = String::with_capacity(length);
    let mut remaining = index;

    for _ in 0..length {
        suffix.insert(0, (b'a' + (remaining % 26) as u8) as char);
        remaining /= 26;
    }

    // A non-zero leftover is the overflow signal: those digits have nowhere to go.
    (remaining == 0).then_some(suffix)
}

/// What the open chunk already holds. Deliberately not the chunk itself: choosing a
/// boundary needs the tallies and nothing else, least of all the file.
#[derive(Clone, Copy, Default)]
struct Filled {
    lines: usize,
    bytes: usize,
    /// Whether any input has reached the chunk yet. Not `bytes == 0`: a repeated header is
    /// written when the chunk opens, so a fresh chunk can already be several bytes long,
    /// and treating it as non-empty makes both budgets refuse to write the first line.
    fresh: bool,
}

/// The prefix of the remaining input that belongs in the open chunk.
struct Fill<'a> {
    /// Bytes to write, a prefix of what was offered.
    span: &'a [u8],
    /// Lines `span` carries, for the manifest tally.
    lines: usize,
    /// Whether the chunk is complete once `span` is written.
    closes: bool,
}

/// A literal needle in the form its search kernel wants. Case folding is decided once, and
/// the uncased metadata is analyzed once rather than on every call.
///
/// Under folding a match is not the needle's length: the kernel compares folded text, so
/// "strasse" matches "Straße" and the span covers the source bytes, not the needle's.
struct Literal<'a> {
    pattern: &'a [u8],
    folded: Option<sz::Utf8UncasedNeedle<'a>>,
}

impl<'a> Literal<'a> {
    /// Analyze `pattern` once. `ignore_case` selects full Unicode case folding.
    fn new(pattern: &'a [u8], ignore_case: bool) -> Self {
        Self {
            pattern,
            folded: ignore_case.then(|| sz::Utf8UncasedNeedle::new(pattern)),
        }
    }

    /// The leftmost match at or after the start of `data`.
    #[inline]
    fn find_in(&self, data: &[u8]) -> Option<Span> {
        match &self.folded {
            Some(needle) => sz::utf8_uncased_search(data, needle)
                .map(|(offset, length)| Span { offset, length }),
            None => sz::find(data, self.pattern).map(|offset| Span {
                offset,
                length: self.pattern.len(),
            }),
        }
    }
}

/// A line-anchored literal delimiter.
///
/// The needle is the pattern alone and the anchor is checked against the byte before a
/// match, rather than folded into the needle as `\n` + pattern. Anchoring inside the needle
/// reads better but cannot survive streaming: a window cuts one byte past an LF, so the
/// newline and the line it introduces always land in different windows and `\n>` is never
/// visible whole. Checking the preceding byte works because every window — and every chunk
/// this mode opens — begins at a line start, so offset 0 is one by construction.
struct Delimiter<'a> {
    needle: Literal<'a>,
}

impl Delimiter<'_> {
    /// Offset in `data` where the next chunk begins: the start of the next line that opens
    /// with the pattern, searching from `from`. `None` when no such line starts in `data`.
    ///
    /// Matches away from a line start are skipped rather than ending the search, so a `>`
    /// inside a FASTA sequence line does not hide the record that follows it.
    fn next_cut(&self, data: &[u8], from: usize, newlines: Newlines) -> Option<usize> {
        let mut at = from;
        loop {
            let found = self.needle.find_in(data.get(at..)?)?;
            let position = at + found.offset;
            if position == 0 || ends_with_terminator(&data[..position], newlines) {
                return Some(position);
            }
            // Advancing past the whole rejected match would skip an anchored match
            // beginning inside it, which a pattern containing a newline can produce.
            at = position + 1;
        }
    }
}

/// How chunk boundaries are chosen, decided once from `Args`.
///
/// An enum rather than a trait: [`SplitConfig`] is `Copy` and threads through every path
/// and test, so a trait object would force a lifetime and an allocation through all of
/// them for a branch taken once per megabyte. It is also how the rest of the codebase
/// answers "one of a fixed set of segmenters" — `LineIter`, `CutAfter`, `InputWindow`.
#[derive(Clone, Copy)]
enum SplitMode<'a> {
    /// A chunk holds this many lines, except possibly the last.
    Lines(NonZeroUsize),
    /// A chunk holds at most this many bytes, and never a partial line. A line longer than
    /// the budget is written whole into a chunk of its own, over budget: refusing to split
    /// a line is the promise, and the budget is the thing that yields.
    Bytes(NonZeroUsize),
    /// A chunk ends where the next line beginning with the delimiter starts.
    Pattern(&'a Delimiter<'a>),
}

impl SplitMode<'_> {
    /// The prefix of `rest` that belongs in a chunk already holding `filled`.
    ///
    /// Never returns both an empty span and `closes == false`: the caller's loop makes
    /// progress on every turn by writing bytes, closing a chunk, or both.
    fn fill<'d>(&self, rest: &'d [u8], filled: Filled, newlines: Newlines) -> Fill<'d> {
        match *self {
            SplitMode::Lines(per_file) => {
                let wanted = per_file.get() - filled.lines;
                let (span, lines) = span_filling_chunk(rest, wanted, newlines);
                Fill {
                    span,
                    lines,
                    closes: filled.lines + lines >= per_file.get(),
                }
            }
            SplitMode::Bytes(budget) => {
                let room = budget.get() - filled.bytes.min(budget.get());
                let span = span_within_budget(rest, room, filled.fresh, newlines);
                Fill {
                    span,
                    lines: count_lines(span, newlines),
                    // A chunk that took everything offered may still have room, and the
                    // next window can fill it; one that stopped short stopped at its limit.
                    closes: span.len() < rest.len(),
                }
            }
            SplitMode::Pattern(delimiter) => {
                // A chunk opens *at* its delimiter, so the match that opened it must not
                // also close it: an empty chunk starts looking one byte in. A chunk that
                // already holds something can close on a match at offset 0, writing
                // nothing — and the turn after that finds the chunk empty and moves on.
                let from = usize::from(filled.fresh).min(rest.len());
                match delimiter.next_cut(rest, from, newlines) {
                    Some(cut) => {
                        let span = &rest[..cut];
                        Fill {
                            span,
                            lines: count_lines(span, newlines),
                            closes: true,
                        }
                    }
                    None => Fill {
                        span: rest,
                        lines: count_lines(rest, newlines),
                        closes: false,
                    },
                }
            }
        }
    }
}

/// Reject a pattern that cannot describe the start of a line: an empty one matches
/// everywhere, and one containing a newline spans two lines rather than opening one.
///
/// Refusing the second is also what keeps streaming simple. Windows cut just past an LF and
/// a delimiter begins at a line start, so a newline-free pattern always lands inside one
/// window; a pattern carrying its own newline would need the window to hold bytes back.
fn parse_pattern(value: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err("must not be empty".to_string());
    }
    if value.contains('\n') {
        return Err(
            "must not contain a newline, since it matches the start of one line".to_string(),
        );
    }
    Ok(value.to_string())
}

/// How the input is cut into chunk files, decided once from `Args`.
#[derive(Clone, Copy)]
struct SplitConfig<'a> {
    /// Prepended to every generated suffix to name a chunk file.
    prefix: &'a str,
    /// Which boundaries end a chunk.
    mode: SplitMode<'a>,
    /// Which byte sequences break a line, and nothing else: chunks are ranges of the input,
    /// so no terminator is ever rewritten.
    newlines: Newlines,
    /// Width of the generated suffix.
    suffix_length: NonZeroUsize,
    /// Lines repeated at the top of every chunk, already taken off the input. Empty unless
    /// `--repeat-header` asked for them, which is the only state where `cat <prefix>*` stops
    /// reproducing the input.
    header: &'a [u8],
    /// How each written chunk is announced.
    format: Format,
    /// What ends an announced path, as `--null` asks. JSON carries its own structure and
    /// ignores it, exactly as it does in every other tool here.
    terminator: Terminator,
    /// Lines `header` carries, so an announcement can say how much of a chunk is not data.
    header_lines: usize,
}

/// The chunk being filled, and the tallies the manifest reports when it closes. The four
/// travel together because they are meaningless apart: closing the file retires all of them.
struct OpenChunk {
    /// Name of the chunk, carried into the manifest.
    name: String,
    /// Data lines written into it, not counting a repeated header.
    lines: usize,
    /// Header lines it opened with, which the manifest adds back into its total.
    header_lines: usize,
    /// Whether any input has been written into it yet, as opposed to a repeated header.
    wrote_data: bool,
    /// Bytes written into it.
    bytes: usize,
    /// The chunk file itself.
    file: BufWriter<File>,
}

impl OpenChunk {
    /// What the chunk holds so far, which is all a boundary decision needs of it.
    #[inline]
    fn filled(&self) -> Filled {
        Filled {
            lines: self.lines,
            bytes: self.bytes,
            fresh: !self.wrote_data,
        }
    }
}

/// What the line splitter carries between windows, so a second call resumes where the
/// first stopped. It owns the chunk still being filled, and allocates only when one opens.
#[derive(Default)]
struct SplitState {
    /// Chunk files opened so far, which names the next one.
    file_index: usize,
    /// The chunk being filled, absent between chunks.
    open: Option<OpenChunk>,
}

/// The error a chunk index outgrowing its suffix width raises, naming the flag to raise
/// and the chunk that has nowhere to go.
fn suffix_exhausted(chunk_number: usize, suffix_length: usize) -> Failure {
    reject(format!(
        "Output file suffixes exhausted: chunk {} does not fit a {}-character --suffix-length",
        chunk_number, suffix_length
    ))
    .into()
}

/// Flush the open chunk and record it in the manifest, readying `state` for the next one.
/// A run under `--format none` writes the record into [`io::sink`], so there is one path here.
fn close_chunk(
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<(), Failure> {
    let Some(mut chunk) = state.open.take() else {
        return Ok(());
    };
    chunk.file.flush().at(&chunk.name)?;
    // The manifest reports the file as it is on disk, header included.
    write_manifest_entry(
        manifest,
        config.format,
        config.terminator,
        &chunk.name,
        chunk.lines + chunk.header_lines,
        chunk.bytes,
        chunk.header_lines,
    )
    .at("-")
}

/// The chunk being filled, opening the next one when none is. GNU `split` stops rather
/// than wrapping onto a name it already wrote, and an exhausted suffix says so here.
fn open_chunk<'a>(
    state: &'a mut SplitState,
    config: &SplitConfig,
) -> Result<&'a mut OpenChunk, Failure> {
    if state.open.is_none() {
        let width = config.suffix_length.get();
        let suffix = generate_suffix(state.file_index, width)
            .ok_or_else(|| suffix_exhausted(state.file_index + 1, width))?;
        let name = format!("{}{}", config.prefix, suffix);
        let mut file = BufWriter::new(File::create(&name).at(&name)?);
        // The header is part of the chunk, so it is written and tallied here rather than
        // treated as free: under a byte budget it has to count, or the budget is not one.
        file.write_all(config.header).at(&name)?;
        state.file_index += 1;
        state.open = Some(OpenChunk {
            name,
            // Data lines only: `--chunk-lines N` promises N lines of data, so the repeated
            // header rides on top rather than eating into the count. Its bytes do count,
            // because `--chunk-bytes` is a cap on the file and a cap has to hold.
            lines: 0,
            header_lines: config.header_lines,
            wrote_data: false,
            bytes: config.header.len(),
            file,
        });
    }
    Ok(state.open.as_mut().expect("just opened"))
}

/// Whether `data` ends with a line terminator of the chosen set. A chunk ending in CR, NEL,
/// LS or PS is terminated, and asking `ends_with(b"\n")` would report it as unfinished and
/// append an LF the input never held.
fn ends_with_terminator(data: &[u8], newlines: Newlines) -> bool {
    match newlines {
        Newlines::Lf => data.ends_with(b"\n"),
        Newlines::Unicode => matches!(
            data,
            [.., b'\n' | b'\r' | 0x0B | 0x0C]
                | [.., 0xC2, 0x85]
                | [.., 0xE2, 0x80, 0xA8]
                | [.., 0xE2, 0x80, 0xA9]
        ),
    }
}

/// Length of the first line of `data`, terminator included, or all of `data` when it holds
/// no terminator at all.
fn first_line_end(data: &[u8], newlines: Newlines) -> usize {
    LineSpans::new(data, newlines)
        .next()
        .map_or(data.len(), |span| span.length)
}

/// The prefix of `data` that fills the open chunk: everything through its `wanted`-th line
/// terminator, or all of `data` when it holds fewer. `wanted` is at least 1, since a chunk
/// closes the moment it fills. Also reports how many lines that prefix carries.
fn span_filling_chunk(data: &[u8], wanted: usize, newlines: Newlines) -> (&[u8], usize) {
    debug_assert!(wanted >= 1, "a full chunk should have been closed already");
    let mut end = 0;
    let mut seen = 0;
    for span in LineSpans::new(data, newlines) {
        end = span.offset + span.length;
        seen += 1;
        if seen == wanted {
            break;
        }
    }
    (&data[..end], seen)
}

/// Lines in `span`, counting an unterminated last line as one, exactly as
/// [`span_filling_chunk`] does for the mode that counts as it goes.
fn count_lines(span: &[u8], newlines: Newlines) -> usize {
    LineSpans::new(span, newlines).count()
}

/// The longest prefix of `data` that ends on a line boundary within `room` bytes.
///
/// Empty when the first line alone exceeds the budget and the chunk already holds
/// something — the caller closes it and offers the line to an empty chunk, where
/// `chunk_is_empty` lets it through whole rather than looping forever. Splitting a line is
/// never an option, so an over-long line produces an over-budget chunk of its own.
fn span_within_budget(data: &[u8], room: usize, chunk_is_empty: bool, newlines: Newlines) -> &[u8] {
    // Everything left fits, so there is nothing to cut: returning it whole keeps the chunk
    // open for the next window instead of closing on an unterminated tail.
    if room >= data.len() {
        return data;
    }
    match last_cut(&data[..room], newlines.into()) {
        Some(position) => &data[..position],
        // No boundary inside the budget: take the whole first line if this chunk is
        // otherwise empty, take nothing if it is not.
        None if chunk_is_empty => &data[..first_line_end(data, newlines)],
        None => &data[..0],
    }
}

/// Append `span` to the open chunk, keeping its tallies honest. Bytes reach the chunk exactly
/// as they arrived: nothing is added, so `cat <prefix>*` reproduces the input.
///
/// The span goes to the kernel in one call however large it is. Feeding it in fixed pieces
/// was measurably faster once and is not any more — over the 5 GB corpus, 800 MB chunks
/// write in the same 0.69 s either way — so the loop that did it is gone.
fn write_span(chunk: &mut OpenChunk, span: &[u8], lines: usize) -> Result<(), Failure> {
    chunk.file.write_all(span).at(&chunk.name)?;
    chunk.lines += lines;
    chunk.bytes += span.len();
    // An empty span is not data: a pattern chunk closing at offset 0 writes nothing, and
    // calling that "written" would make the next chunk refuse its own delimiter.
    chunk.wrote_data |= !span.is_empty();
    Ok(())
}

/// Copy each chunk's byte range out of `data` in one write, which is what splitting is: the
/// bytes reach the chunk exactly as they arrived, whichever terminator ended them.
fn split_by_ranges(
    data: &[u8],
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<(), Failure> {
    let mut rest = data;
    while !rest.is_empty() {
        // The borrow of the open chunk ends with this block, so `close_chunk` can take
        // `state` again below.
        let closes = {
            let chunk = open_chunk(state, config)?;
            let fill = config.mode.fill(rest, chunk.filled(), config.newlines);
            debug_assert!(
                !fill.span.is_empty() || fill.closes,
                "a turn that writes nothing has to close a chunk, or the loop stalls"
            );
            write_span(chunk, fill.span, fill.lines)?;
            rest = &rest[fill.span.len()..];
            fill.closes
        };
        if closes {
            close_chunk(state, config, manifest)?;
        }
    }

    Ok(())
}

/// The error an unmeetable `--chunk-count` raises: every chunk needs a byte of its own, so a
/// count past the input's size can never be honoured, however the boundaries are placed.
fn chunk_count_exceeds_input(wanted: NonZeroUsize, total: usize) -> Failure {
    reject(format!(
        "--chunk-count {} exceeds the input's {} bytes, which cannot yield that many chunks",
        wanted, total
    ))
    .into()
}

/// Where to cut `total` bytes into `wanted` chunks of near-equal size, snapped forward to
/// the next line boundary. Returns the `wanted - 1` interior cut offsets; the last chunk
/// runs to the end. Fails when `wanted` exceeds the input's size, which no placement of
/// boundaries can honour and which a plan would otherwise try to allocate for.
///
/// Each boundary costs one search from its ideal offset rather than a scan of everything
/// before it, so the whole plan is `wanted` searches over a mapped input regardless of how
/// large it is. Snapping can collapse two boundaries onto the same offset when lines are
/// long relative to `total / wanted`; the chunk between them is then empty, which is
/// reported rather than skipped so that the run yields exactly `wanted` files.
fn plan_equal_chunks(
    data: &[u8],
    wanted: NonZeroUsize,
    newlines: Newlines,
) -> Result<Vec<usize>, Failure> {
    let total = data.len();
    if wanted.get() > total {
        return Err(chunk_count_exceeds_input(wanted, total));
    }
    let mut cuts = Vec::with_capacity(wanted.get() - 1);
    let mut previous = 0;
    for index in 1..wanted.get() {
        // Multiplying first keeps the boundaries evenly spaced, where dividing first would
        // truncate each one; the widening is what keeps `total * index` from overflowing.
        let ideal = (total as u128 * index as u128 / wanted.get() as u128) as usize;
        let cut = ideal + first_line_end(&data[ideal..], newlines);
        // Boundaries never move backwards, so a chunk is never handed a negative span.
        previous = cut.max(previous);
        cuts.push(previous);
    }
    Ok(cuts)
}

/// Write `data` as the ranges `cuts` describes, then the tail. Used by `--chunk-count`,
/// whose boundaries are known before any byte is written, so nothing is searched per chunk.
fn split_at_offsets(
    data: &[u8],
    cuts: &[usize],
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<usize, Failure> {
    let mut state = SplitState::default();
    let result = write_ranges(data, cuts, &mut state, config, manifest);
    finish(result, &mut state, config, manifest)
}

fn write_ranges(
    data: &[u8],
    cuts: &[usize],
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<(), Failure> {
    let mut start = 0;
    for end in cuts.iter().copied().chain([data.len()]) {
        let span = &data[start.min(data.len())..end.min(data.len())];
        let lines = count_lines(span, config.newlines);
        write_span(open_chunk(state, config)?, span, lines)?;
        close_chunk(state, config, manifest)?;
        start = end;
    }
    Ok(())
}

/// Retire whatever chunk a run left open, whether it ended or failed, and answer how many
/// chunk files it wrote. A failed run closes its chunk before its error surfaces, so the
/// bytes already accepted reach the disk rather than dying in a buffer.
fn finish(
    result: Result<(), Failure>,
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<usize, Failure> {
    let closed = close_chunk(state, config, manifest);
    result.and(closed).map(|()| state.file_index)
}

/// Largest header the streaming path will buffer. A header is a handful of column names,
/// so anything past this is a file whose first line never ends — and swallowing it would
/// trade the bounded memory a pipe is chosen for against a header nobody asked to repeat.
const MAX_HEADER_BYTES: usize = 8 << 20;

/// The error a header too large for its budget raises. Emitting header-only chunks instead
/// would fill the disk without ever making progress.
fn header_exceeds_budget(header: usize, budget: usize) -> Failure {
    reject(format!(
        "The {}-byte header does not leave room for data in a {}-byte --chunk-bytes budget",
        header, budget
    ))
    .into()
}

/// The first records of an input, taken off before splitting begins.
struct Header<'a> {
    /// The header itself, terminators included exactly as the input wrote them.
    bytes: Vec<u8>,
    /// What is left of the input once the header is removed.
    rest: &'a [u8],
    /// Records the header carries.
    lines: usize,
    /// Whether the last of them ended, as opposed to running out with the input. A
    /// streaming caller reads this to tell a complete header from a truncated window.
    terminated: bool,
}

/// Split `data` into the first `lines` records and the rest.
fn take_header(data: &[u8], lines: NonZeroUsize, newlines: Newlines) -> Header<'_> {
    let (span, seen) = span_filling_chunk(data, lines.get(), newlines);
    Header {
        bytes: span.to_vec(),
        rest: &data[span.len()..],
        lines: seen,
        terminated: ends_with_terminator(span, newlines),
    }
}

/// Pull a header off the front of a stream before splitting begins, growing the window
/// while the header is still incomplete.
fn take_header_streaming<R: Read>(
    refill: &mut Refill<R>,
    lines: NonZeroUsize,
    newlines: Newlines,
) -> Result<(Vec<u8>, usize), Failure> {
    loop {
        // A fresh window holds nothing, and an empty buffer looks like a single
        // unterminated line to the line counter, so fill before asking.
        if !refill.at_eof() && refill.filled().len() < refill.capacity() {
            refill.advance(0).at("-")?;
        }
        let filled = refill.filled();
        let header = take_header(filled, lines, newlines);
        // A trailing CR may be half a CRLF whose LF is still to arrive.
        let ambiguous = header.bytes.ends_with(b"\r") && newlines == Newlines::Unicode;
        // An unterminated last line may simply be a window cut mid-record, so it counts
        // only once the input has ended and no more of it is coming.
        let complete =
            refill.at_eof() || (header.lines >= lines.get() && header.terminated && !ambiguous);
        let consumed = filled.len() - header.rest.len();
        let (bytes, taken) = (header.bytes, header.lines);
        if complete {
            refill.advance(consumed).at("-")?;
            return Ok((bytes, taken));
        }
        if refill.filled().len() >= MAX_HEADER_BYTES {
            return Err(reject(format!(
                "The first {} lines exceed the {} MiB a streamed --repeat-header may buffer",
                lines.get(),
                MAX_HEADER_BYTES >> 20
            ))
            .into());
        }
        refill.grow().at("-")?;
    }
}

/// Split a whole buffer, closing the chunk left open at the end.
fn split_data(
    data: &[u8],
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<usize, Failure> {
    let mut state = SplitState::default();
    let result = split_by_ranges(data, &mut state, config, manifest);
    finish(result, &mut state, config, manifest)
}

// region: Streaming

/// Drive [`split_by_ranges`] over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
fn split_stream<R: Read>(
    refill: &mut Refill<R>,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> Result<usize, Failure> {
    let mut state = SplitState::default();
    // A `Failure` cannot travel through the window loop's `io::Result`, so a window that
    // fails breaks the loop and hands its error back here.
    let mut result = Ok(());
    let read = refill.try_for_each_window(config.newlines.into(), |window| {
        result = split_by_ranges(window, &mut state, config, manifest);
        Ok(match result {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        })
    });
    finish(read.at("-").and(result), &mut state, config, manifest)
}

// endregion: Streaming

/// Write one record naming a file that was written.
fn write_manifest_entry(
    output: &mut dyn Write,
    format: Format,
    terminator: Terminator,
    name: &str,
    lines: usize,
    bytes: usize,
    header_lines: usize,
) -> io::Result<()> {
    match format {
        Format::None => return Ok(()),
        Format::Paths => {
            output.write_all(name.as_bytes())?;
            return output.write_all(&[terminator.as_byte()]);
        }
        Format::Json => {}
    }
    output.write_all(br#"{"type":"file","data":{"path":"#)?;
    json_text_field_to(output, name.as_bytes())?;
    // `header_lines` says how many of `lines` are the repeated header, so a consumer can
    // tell a chunk's data apart from its preamble without re-reading the file.
    writeln!(
        output,
        r#","lines":{},"bytes":{},"header_lines":{}}}}}"#,
        lines, bytes, header_lines
    )
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Reject the combinations clap cannot: it conflicts on an argument's presence and never on
/// its value, so anything conditioned on a value is checked here, once, before any input is
/// read and any chunk file is created.
fn validate(args: &Args) -> Result<(), clap::Error> {
    // Folding is a property of the pattern, so asking for it without one is a mistake worth
    // naming rather than a flag that quietly does nothing.
    if args.ignore_case && args.chunk_pattern.is_none() {
        return Err(reject(
            "--ignore-case folds the --chunk-pattern match, so it needs a --chunk-pattern to fold",
        ));
    }
    if let Some(wanted) = args.chunk_count {
        let width = args.suffix_length.get();
        let nameable = u32::try_from(width)
            .ok()
            .and_then(|width| 26usize.checked_pow(width))
            .unwrap_or(usize::MAX);
        if nameable < wanted.get() {
            return Err(reject(format!(
                "--chunk-count {} needs a wider --suffix-length: {} characters name {} files",
                wanted, width, nameable
            )));
        }
    }
    Ok(())
}

/// The error a `--repeat-header` swallowing the whole input raises: every line became header,
/// so there is nothing left to head.
fn header_exceeds_input(lines: NonZeroUsize) -> Failure {
    reject(format!(
        "--repeat-header={} takes the whole input as header, leaving no data to split",
        lines
    ))
    .into()
}

/// Split the input, answering how the process should exit. Every failure returns rather than
/// exiting, so the open chunk and the manifest are both flushed before the status is set.
fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let path = args.input.as_deref().unwrap_or("-");
    let input = get_input_streaming(args.input.as_deref()).at(path)?;

    // The delimiter outlives the config that borrows it.
    let pattern = args.chunk_pattern.clone().unwrap_or_default();
    let delimiter = Delimiter {
        needle: Literal::new(pattern.as_bytes(), args.ignore_case),
    };

    // `--chunk-count` is the one budget that cannot be decided a window at a time, so it
    // reaches `split_at_offsets` rather than joining `SplitMode`.
    let mode = match (args.chunk_lines, args.chunk_bytes, &args.chunk_pattern) {
        (Some(lines), _, _) => SplitMode::Lines(lines),
        (_, Some(bytes), _) => SplitMode::Bytes(bytes),
        (_, _, Some(_)) => SplitMode::Pattern(&delimiter),
        _ => SplitMode::Lines(NonZeroUsize::MIN),
    };

    // Case folding is a Unicode operation, so it brings the Unicode newline set with it.
    let newlines = Newlines::from_utf8(args.utf8 || args.ignore_case);
    let emitted = {
        let mut discarded = io::sink();
        let manifest: &mut dyn Write = match args.format {
            _ if args.quiet => &mut discarded,
            Format::None => &mut discarded,
            _ => &mut *output,
        };
        // The header is taken off the input before any boundary is chosen, so every mode
        // sees only data and every chunk is opened with the header already in it.
        let mut window = input.into_window(DEFAULT_WINDOW_BYTES);
        let (header, header_lines, taken, exhausted) = match (&mut window, args.repeat_header) {
            (InputWindow::Whole(source), Some(lines)) => {
                let header = take_header(source.as_bytes(), lines, newlines);
                let start = source.as_bytes().len() - header.rest.len();
                (header.bytes, header.lines, start, header.rest.is_empty())
            }
            (InputWindow::Stream(refill), Some(lines)) => {
                let (header, seen) = take_header_streaming(refill, lines, newlines)?;
                let drained = refill.at_eof() && refill.filled().is_empty();
                (header, seen, 0, drained)
            }
            _ => (Vec::new(), 0, 0, false),
        };

        // An empty input legitimately writes nothing, so only a header that ate real data
        // is a mistake worth naming.
        if let Some(lines) = args.repeat_header {
            if exhausted && !header.is_empty() {
                return Err(header_exceeds_input(lines));
            }
        }

        // A header that leaves no room for data would repeat forever without progressing.
        if let SplitMode::Bytes(budget) = mode {
            if header.len() >= budget.get() {
                return Err(header_exceeds_budget(header.len(), budget.get()));
            }
        }

        let config = SplitConfig {
            prefix: &args.prefix,
            mode,
            newlines,
            suffix_length: args.suffix_length,
            format: args.format,
            terminator: Terminator::from_null(args.null),
            header: &header,
            header_lines,
        };

        match window {
            InputWindow::Whole(source) => {
                let data = &source.as_bytes()[taken..];
                match args.chunk_count {
                    Some(wanted) => {
                        let cuts = plan_equal_chunks(data, wanted, newlines)?;
                        split_at_offsets(data, &cuts, &config, manifest)
                    }
                    None => split_data(data, &config, manifest),
                }
            }
            InputWindow::Stream(mut refill) => match args.chunk_count {
                // Only a genuine pipe lands here: a `< file` redirect is mapped above.
                Some(_) => Err(reject(
                    "--chunk-count needs the input's size, which a pipe does not report; \
                     redirect from a file (sz-split --chunk-count N < file), name the file, \
                     or use --chunk-bytes",
                )
                .into()),
                None => split_stream(&mut refill, &config, manifest),
            },
        }?
    };

    output.flush().at("-")?;
    Ok(Status::from_found(emitted > 0))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-split", run(&args, &mut output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn finds_literal_spans_cased_and_folded() {
        let cased = Literal::new(b"ss", false);
        assert_eq!(
            cased.find_in(b"aSSbss"),
            Some(Span {
                offset: 4,
                length: 2
            })
        );

        // Folded, the span covers the source bytes, so "strasse" reaches "Straße".
        let folded = Literal::new("strasse".as_bytes(), true);
        let found = folded.find_in("Bahnhofstraße 5".as_bytes()).unwrap();
        assert_eq!(found.offset, 7);
        assert_eq!(found.length, "straße".len());

        // A match can be longer than the needle: one-byte "k" matches the three-byte
        // KELVIN SIGN, so a span is never assumed to be the pattern's length.
        let kelvin = Literal::new(b"k", true);
        let found = kelvin.find_in("x\u{212A}y".as_bytes()).unwrap();
        assert_eq!(found.offset, 1);
        assert_eq!(found.length, "\u{212A}".len());
        assert!(found.length > kelvin.pattern.len());
    }

    #[test]
    fn rejects_a_zero_count_without_naming_a_rust_type() {
        let zero_lines = vec!["sz-split", "--chunk-lines", "0"];
        let zero_suffix = vec!["sz-split", "--chunk-lines", "1", "--suffix-length", "0"];
        for arguments in [zero_lines, zero_suffix] {
            let Err(error) = Args::try_parse_from(&arguments) else {
                panic!("{:?} must be rejected", arguments);
            };
            let rendered = error.to_string();
            assert!(rendered.contains("must be at least 1"), "{}", rendered);
            assert!(!rendered.contains("non-zero type"), "{}", rendered);
            assert_eq!(error.exit_code(), 2, "{}", rendered);
        }

        // Only zero reads differently; every other rejection keeps clap's wording.
        let Err(error) = Args::try_parse_from(["sz-split", "--chunk-lines", "abc"]) else {
            panic!("--chunk-lines abc must be rejected");
        };
        assert!(
            error.to_string().contains("invalid digit found in string"),
            "{}",
            error
        );
        assert!(Args::try_parse_from(["sz-split", "--chunk-lines", "1"]).is_ok());
    }

    #[test]
    fn declares_no_short_flags() {
        let mut command = Args::command();
        command.build();
        assert!(
            command
                .get_arguments()
                .all(|argument| argument.get_short().is_none()
                    || matches!(argument.get_short(), Some('h') | Some('V'))),
            "only clap's own -h and -V may be short"
        );
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
                "chunk-lines",
                "chunk-bytes",
                "chunk-count",
                "chunk-pattern",
                "ignore-case",
                "repeat-header",
                "utf8",
                "suffix-length",
                "format",
                "null",
                "quiet",
                "help",
                "version",
            ],
            "sz-split's flag surface changed; reconcile it against the suite vocabulary"
        );
    }

    #[test]
    fn requires_exactly_one_chunking_rule() {
        // Naming none used to be impossible, since `--chunk-lines` was required; the error
        // now lists every way to say how big a chunk is.
        let Err(error) = Args::try_parse_from(["sz-split", "trex.txt"]) else {
            panic!("a run without a chunking rule must be rejected");
        };
        let rendered = error.to_string();
        for flag in ["--chunk-lines", "--chunk-bytes", "--chunk-count"] {
            assert!(rendered.contains(flag), "{}", rendered);
        }
        assert_eq!(error.exit_code(), 2, "{}", rendered);

        // Naming two is equally a mistake.
        assert!(
            Args::try_parse_from(["sz-split", "--chunk-lines", "5", "--chunk-count", "2"]).is_err()
        );
        // `lines` names a `--fields` value elsewhere, so the alias is gone.
        assert!(Args::try_parse_from(["sz-split", "--lines", "5"]).is_err());
        assert!(Args::try_parse_from(["sz-split", "--chunk-bytes", "10K"]).is_ok());
    }

    #[test]
    fn rejects_what_clap_cannot_express() {
        let folding = Args::try_parse_from(["sz-split", "--chunk-lines", "5", "--ignore-case"])
            .expect("folding without a pattern parses; `validate` is what rejects it");
        let error = validate(&folding).unwrap_err();
        assert!(error.to_string().contains("--chunk-pattern"), "{}", error);
        assert!(!error.to_string().contains("-i/"), "{}", error);

        // Two characters name 676 chunks, so 677 of them would fail partway through with
        // chunks already on disk.
        let narrow =
            Args::try_parse_from(["sz-split", "--chunk-count", "677", "--suffix-length", "2"])
                .unwrap();
        let error = validate(&narrow).unwrap_err();
        assert!(error.to_string().contains("--suffix-length"), "{}", error);
        let wide =
            Args::try_parse_from(["sz-split", "--chunk-count", "677", "--suffix-length", "3"])
                .unwrap();
        assert!(validate(&wide).is_ok());
    }

    #[test]
    fn fills_chunks_to_a_byte_budget_without_splitting_a_line() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("b.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Bytes(NonZeroUsize::new(8).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            newlines: Newlines::Lf,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        };
        // Four-byte lines: two fit the budget exactly, and the tail is its own chunk.
        let data = b"aaa\nbbb\nccc\nddd\neee\n";
        split_data(data, &config, &mut io::sink()).unwrap();
        assert_eq!(
            read_chunks(&prefix),
            vec!["aaa\nbbb\n", "ccc\nddd\n", "eee\n"]
        );
    }

    #[test]
    fn writes_an_over_long_line_whole_rather_than_splitting_it() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("l.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Bytes(NonZeroUsize::new(5).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            newlines: Newlines::Lf,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        };
        let data = b"ab\nTHIS-LINE-IS-FAR-TOO-LONG\ncd\n";
        split_data(data, &config, &mut io::sink()).unwrap();
        let chunks = read_chunks(&prefix);
        // The long line is over budget and alone, rather than cut in half.
        assert_eq!(chunks[1], "THIS-LINE-IS-FAR-TOO-LONG\n");
        assert_eq!(chunks.concat().as_bytes(), data);
    }

    #[test]
    fn cuts_into_exactly_as_many_chunks_as_asked_for() {
        let temporary = TempDir::new().unwrap();
        let data = b"x\ny\n";
        for wanted in [1usize, 2, 4] {
            let prefix = temporary
                .path()
                .join(format!("n{}.", wanted))
                .display()
                .to_string();
            let config = SplitConfig {
                prefix: &prefix,
                mode: SplitMode::Lines(NonZeroUsize::MIN),
                suffix_length: NonZeroUsize::new(2).unwrap(),
                newlines: Newlines::Lf,
                format: Format::None,
                terminator: Terminator::Newline,
                header: b"",
                header_lines: 0,
            };
            let cuts =
                plan_equal_chunks(data, NonZeroUsize::new(wanted).unwrap(), Newlines::Lf).unwrap();
            split_at_offsets(data, &cuts, &config, &mut io::sink()).unwrap();
            let chunks = read_chunks(&prefix);
            // Asking for more chunks than there are lines still yields that many files,
            // so a downstream loop over the count is safe; the surplus are empty.
            assert_eq!(chunks.len(), wanted, "asked for {}", wanted);
            assert_eq!(chunks.concat().as_bytes(), data, "asked for {}", wanted);
        }

        // Past one chunk per byte the request is unmeetable, and used to allocate and loop
        // `wanted` times before writing anything.
        let error =
            plan_equal_chunks(data, NonZeroUsize::new(1 << 40).unwrap(), Newlines::Lf).unwrap_err();
        assert!(error.to_string().contains("--chunk-count"), "{}", error);
    }

    #[test]
    fn repeats_the_header_atop_every_chunk_without_spending_the_line_budget() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("h.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Lines(NonZeroUsize::new(2).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            newlines: Newlines::Lf,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"id,name\n",
            header_lines: 1,
        };
        // `--chunk-lines 2` promises two lines of data, so the header rides on top rather
        // than counting as one of them.
        split_data(b"1,a\n2,b\n3,c\n4,d\n", &config, &mut io::sink()).unwrap();
        assert_eq!(
            read_chunks(&prefix),
            vec!["id,name\n1,a\n2,b\n", "id,name\n3,c\n4,d\n"]
        );
    }

    #[test]
    fn takes_the_same_header_from_a_buffer_and_a_stream() {
        let data = b"id,name\n1,a\n2,b\n3,c\n".as_slice();
        let lines = NonZeroUsize::new(1).unwrap();
        let whole = take_header(data, lines, Newlines::Lf);

        // A fresh window holds nothing, and tiny capacities force the grow path, so the
        // stream has to agree with the buffer at every one of them.
        for capacity in [1usize, 3, 8, 64] {
            let mut refill = Refill::new(data, capacity);
            let (header, seen) = take_header_streaming(&mut refill, lines, Newlines::Lf).unwrap();
            assert_eq!(header, whole.bytes, "capacity {}", capacity);
            assert_eq!(seen, whole.lines, "capacity {}", capacity);
            // What is left in the window is the start of the data, not of the header.
            assert!(
                whole.rest.starts_with(refill.filled()),
                "capacity {}",
                capacity
            );
        }
    }

    #[test]
    fn counts_the_header_against_a_byte_budget_but_never_a_line_budget() {
        // A byte budget caps the file, so the header it opens with has to count.
        let header = b"id,name\n";
        let budget = NonZeroUsize::new(16).unwrap();
        let filled = Filled {
            lines: 0,
            bytes: header.len(),
            fresh: true,
        };
        let fill = SplitMode::Bytes(budget).fill(b"1,a\n2,b\n3,c\n", filled, Newlines::Lf);
        assert_eq!(fill.span, b"1,a\n2,b\n", "8 header bytes leave 8 of the 16");

        // A line budget counts data, so the same header costs it nothing.
        let fill = SplitMode::Lines(NonZeroUsize::new(2).unwrap()).fill(
            b"1,a\n2,b\n3,c\n",
            Filled::default(),
            Newlines::Lf,
        );
        assert_eq!(fill.span, b"1,a\n2,b\n");
    }

    /// A config over any mode with the Unicode newline set, for the byte-fidelity test.
    fn utf8_config<'a>(prefix: &'a str, mode: SplitMode<'a>) -> SplitConfig<'a> {
        SplitConfig {
            prefix,
            mode,
            newlines: Newlines::Unicode,
            suffix_length: NonZeroUsize::new(2).unwrap(),
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        }
    }

    /// A pattern-mode config; the delimiter is built by the caller because it borrows.
    fn pattern_config<'a>(prefix: &'a str, delimiter: &'a Delimiter<'a>) -> SplitConfig<'a> {
        SplitConfig {
            prefix,
            mode: SplitMode::Pattern(delimiter),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            newlines: Newlines::Lf,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        }
    }

    #[test]
    fn starts_a_chunk_at_every_delimiter_line() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("p.").display().to_string();
        let delimiter = Delimiter {
            needle: Literal::new(b">", false),
        };
        let data = b">seq1\nACGT\n>seq2\nTTTT\n";
        split_data(data, &pattern_config(&prefix, &delimiter), &mut io::sink()).unwrap();
        let chunks = read_chunks(&prefix);
        // The delimiter opens its chunk, and the newline before it closes the previous one.
        assert_eq!(chunks, vec![">seq1\nACGT\n", ">seq2\nTTTT\n"]);
        assert_eq!(chunks.concat().as_bytes(), data);
    }

    #[test]
    fn ignores_the_pattern_away_from_a_line_start() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("a.").display().to_string();
        let delimiter = Delimiter {
            needle: Literal::new(b">", false),
        };
        // The `>` inside `x>y` is not a record start, and an unanchored search would split
        // there. Input beginning with the pattern needs no cut, so no empty chunk opens.
        split_data(
            b">a\nx>y\n>b\n",
            &pattern_config(&prefix, &delimiter),
            &mut io::sink(),
        )
        .unwrap();
        assert_eq!(read_chunks(&prefix), vec![">a\nx>y\n", ">b\n"]);
    }

    #[test]
    fn folds_case_when_matching_a_delimiter() {
        let temporary = TempDir::new().unwrap();
        let data = b"From a\nbody\nFROM b\nbody\nfrom c\nbody\n";
        // Cased, only the lowercase record matches, so the file cuts once.
        for (name, ignore_case, wanted) in [("cased.", false, 2usize), ("folded.", true, 3)] {
            let prefix = temporary.path().join(name).display().to_string();
            let delimiter = Delimiter {
                needle: Literal::new("from ".as_bytes(), ignore_case),
            };
            split_data(data, &pattern_config(&prefix, &delimiter), &mut io::sink()).unwrap();
            let chunks = read_chunks(&prefix);
            assert_eq!(chunks.len(), wanted, "{}", name);
            assert_eq!(chunks.concat().as_bytes(), data, "{}", name);
        }
    }

    #[test]
    fn finds_a_delimiter_split_across_a_window_seam() {
        let temporary = TempDir::new().unwrap();
        let data = b">a\nSEQ\n>b\nSEQ\n>c\n";
        let whole = {
            let prefix = temporary.path().join("whole.").display().to_string();
            let delimiter = Delimiter {
                needle: Literal::new(b">", false),
            };
            split_data(data, &pattern_config(&prefix, &delimiter), &mut io::sink()).unwrap();
            read_chunks(&prefix)
        };

        // Every capacity puts the seam somewhere different, including between the `\n` and
        // the `>` that follows it — the one place an unretained window would lose a split.
        for capacity in [1usize, 2, 3, 5, 7, 16] {
            let prefix = temporary
                .path()
                .join(format!("s{}.", capacity))
                .display()
                .to_string();
            let delimiter = Delimiter {
                needle: Literal::new(b">", false),
            };
            let config = pattern_config(&prefix, &delimiter);
            let mut refill = Refill::new(data.as_slice(), capacity);
            split_stream(&mut refill, &config, &mut io::sink()).unwrap();
            assert_eq!(read_chunks(&prefix), whole, "capacity {}", capacity);
        }
    }

    #[test]
    fn rejects_a_pattern_that_cannot_open_a_line() {
        let Err(error) = Args::try_parse_from(["sz-split", "--chunk-pattern", ""]) else {
            panic!("an empty pattern must be rejected");
        };
        assert!(error.to_string().contains("must not be empty"), "{}", error);
        // A pattern is a line prefix, so a newline inside one contradicts it.
        let Err(error) = Args::try_parse_from(["sz-split", "--chunk-pattern", "a\nb"]) else {
            panic!("a pattern spanning two lines must be rejected");
        };
        assert!(error.to_string().contains("newline"), "{}", error);
        // `--utf8` now only says which byte sequences break a line, so every budget takes it.
        for mode in [
            ["--chunk-lines", "5"],
            ["--chunk-bytes", "1M"],
            ["--chunk-count", "2"],
            ["--chunk-pattern", ">"],
        ] {
            let arguments = ["sz-split", mode[0], mode[1], "--utf8"];
            assert!(Args::try_parse_from(arguments).is_ok(), "{:?}", arguments);
        }
    }

    #[test]
    fn generates_aa_ab_suffix_sequence() {
        assert_eq!(generate_suffix(0, 2).as_deref(), Some("aa"));
        assert_eq!(generate_suffix(1, 2).as_deref(), Some("ab"));
        assert_eq!(generate_suffix(25, 2).as_deref(), Some("az"));
        assert_eq!(generate_suffix(26, 2).as_deref(), Some("ba"));
        assert_eq!(generate_suffix(27, 2).as_deref(), Some("bb"));
    }

    #[test]
    fn refuses_suffixes_wider_than_the_requested_length() {
        // Two characters name 26 * 26 chunks, so index 676 is the first that cannot be named.
        assert_eq!(generate_suffix(675, 2).as_deref(), Some("zz"));
        assert_eq!(generate_suffix(676, 2), None);
        assert_eq!(generate_suffix(25, 1).as_deref(), Some("z"));
        assert_eq!(generate_suffix(26, 1), None);
    }

    /// A two-character-suffix config, the shape every test below splits with.
    fn config(prefix: &str, lines_per_file: usize, newlines: Newlines) -> SplitConfig<'_> {
        SplitConfig {
            prefix,
            mode: SplitMode::Lines(NonZeroUsize::new(lines_per_file).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            newlines,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        }
    }

    /// The chunk files written under `prefix`, in suffix order.
    fn read_chunks(prefix: &str) -> Vec<String> {
        (0..)
            .map_while(|index| generate_suffix(index, 2))
            .map(|suffix| format!("{}{}", prefix, suffix))
            .map_while(|path| fs::read_to_string(path).ok())
            .collect()
    }

    #[test]
    fn splits_input_into_line_chunk_files() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("test_").to_str().unwrap().to_string();

        let data = b"line1\nline2\nline3\nline4\nline5\n";
        split_data(data, &config(&prefix, 2, Newlines::Lf), &mut io::sink()).unwrap();

        // Check first file
        let file1 = fs::read_to_string(format!("{}aa", prefix)).unwrap();
        assert_eq!(file1, "line1\nline2\n");

        // Check second file
        let file2 = fs::read_to_string(format!("{}ab", prefix)).unwrap();
        assert_eq!(file2, "line3\nline4\n");

        // Check third file
        let file3 = fs::read_to_string(format!("{}ac", prefix)).unwrap();
        assert_eq!(file3, "line5\n");
    }

    #[test]
    fn writes_one_line_per_file() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("single_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"a\nb\nc\n";
        split_data(data, &config(&prefix, 1, Newlines::Lf), &mut io::sink()).unwrap();

        assert_eq!(fs::read_to_string(format!("{}aa", prefix)).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(format!("{}ab", prefix)).unwrap(), "b\n");
        assert_eq!(fs::read_to_string(format!("{}ac", prefix)).unwrap(), "c\n");
    }

    #[test]
    fn leaves_an_unterminated_last_line_unterminated() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("notrail_")
            .to_str()
            .unwrap()
            .to_string();

        // Appending the terminator the input never wrote used to make `cat` differ from it.
        let data = b"line1\nline2";
        split_data(data, &config(&prefix, 1, Newlines::Lf), &mut io::sink()).unwrap();
        assert_eq!(read_chunks(&prefix), vec!["line1\n", "line2"]);

        // Under the LF newline set a lone CR is not a terminator, and is still not rewritten.
        let bare = temp_dir.path().join("cr_").to_str().unwrap().to_string();
        split_data(b"\r", &config(&bare, 1, Newlines::Lf), &mut io::sink()).unwrap();
        assert_eq!(fs::read(format!("{}aa", bare)).unwrap(), b"\r");
    }

    #[test]
    fn errors_rather_than_overwriting_when_suffixes_run_out() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("overflow_")
            .to_str()
            .unwrap()
            .to_string();

        // One character names 26 chunks, so a 27th line has nowhere to go.
        let data: Vec<u8> = (0..27)
            .flat_map(|line| format!("line{}\n", line).into_bytes())
            .collect();
        let overflowing = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Lines(NonZeroUsize::new(1).unwrap()),
            suffix_length: NonZeroUsize::new(1).unwrap(),
            newlines: Newlines::Lf,
            format: Format::None,
            terminator: Terminator::Newline,
            header: b"",
            header_lines: 0,
        };

        let error = split_data(&data, &overflowing, &mut io::sink()).unwrap_err();

        let message = error.to_string();
        assert!(message.contains("--suffix-length"), "{}", message);
        assert!(message.contains("chunk 27"), "{}", message);
        // The 26 chunks that did fit keep the lines they were given.
        assert_eq!(
            fs::read_to_string(format!("{}a", prefix)).unwrap(),
            "line0\n"
        );
        assert_eq!(
            fs::read_to_string(format!("{}z", prefix)).unwrap(),
            "line25\n"
        );
    }

    #[test]
    fn splits_on_unicode_newlines_under_utf8() {
        let temp_dir = TempDir::new().unwrap();
        // Line separators, which only the Unicode newline set breaks on.
        let data = "a\u{2028}b\u{2028}c\n".as_bytes();

        let unicode_prefix = temp_dir
            .path()
            .join("unicode_")
            .to_str()
            .unwrap()
            .to_string();
        split_data(
            data,
            &config(&unicode_prefix, 1, Newlines::Unicode),
            &mut io::sink(),
        )
        .unwrap();
        // Each chunk keeps the terminator the input broke on, so nothing is rewritten.
        assert_eq!(
            read_chunks(&unicode_prefix),
            vec!["a\u{2028}", "b\u{2028}", "c\n"]
        );

        let byte_prefix = temp_dir.path().join("byte_").to_str().unwrap().to_string();
        split_data(
            data,
            &config(&byte_prefix, 1, Newlines::Lf),
            &mut io::sink(),
        )
        .unwrap();
        assert_eq!(read_chunks(&byte_prefix), vec!["a\u{2028}b\u{2028}c\n"]);
    }

    #[test]
    fn preserves_every_terminator_under_utf8_in_every_mode() {
        let temporary = TempDir::new().unwrap();
        let delimiter = Delimiter {
            needle: Literal::new(b"c", false),
        };
        // CRLF, a line separator and a CRLF-terminated tail; then an unterminated tail and a
        // lone CR, both of which used to gain an LF the input never held.
        for (which, data) in ["a\r\nb\u{2028}c\r\n", "a\r\nb", "\r"].iter().enumerate() {
            let data = data.as_bytes();
            for (name, mode) in [
                ("lines", SplitMode::Lines(NonZeroUsize::new(2).unwrap())),
                ("bytes", SplitMode::Bytes(NonZeroUsize::new(8).unwrap())),
                ("pattern", SplitMode::Pattern(&delimiter)),
            ] {
                let whole_prefix = temporary
                    .path()
                    .join(format!("w{}{}.", which, name))
                    .display()
                    .to_string();
                split_data(data, &utf8_config(&whole_prefix, mode), &mut io::sink()).unwrap();
                let expected = read_chunks(&whole_prefix);
                assert_eq!(expected.concat().as_bytes(), data, "{} {}", which, name);

                // The whole-buffer run is the oracle: a seam anywhere must not change it.
                for capacity in [1usize, 2, 3, 5, 8, 64] {
                    let prefix = temporary
                        .path()
                        .join(format!("s{}{}{}.", which, name, capacity))
                        .display()
                        .to_string();
                    let mut refill = Refill::new(data, capacity);
                    split_stream(&mut refill, &utf8_config(&prefix, mode), &mut io::sink())
                        .unwrap();
                    assert_eq!(
                        read_chunks(&prefix),
                        expected,
                        "{} {} at capacity {}",
                        which,
                        name,
                        capacity
                    );
                }
            }

            // `--chunk-count` needs the input's size, so it has no streaming path to agree with.
            let prefix = temporary
                .path()
                .join(format!("c{}.", which))
                .display()
                .to_string();
            let config = utf8_config(&prefix, SplitMode::Lines(NonZeroUsize::MIN));
            let wanted = NonZeroUsize::new(data.len().clamp(1, 2)).unwrap();
            let cuts = plan_equal_chunks(data, wanted, Newlines::Unicode).unwrap();
            split_at_offsets(data, &cuts, &config, &mut io::sink()).unwrap();
            assert_eq!(read_chunks(&prefix).concat().as_bytes(), data, "{}", which);
        }
    }

    #[test]
    fn streams_identically_to_whole_buffer() {
        // A blank line, an over-wide line, and a final line without a terminator.
        let data =
            "line1\n\nline3\u{2028}a line far wider than a seven byte window\nlast".as_bytes();
        let temp_dir = TempDir::new().unwrap();

        for (newline_set, newlines) in [("lf", Newlines::Lf), ("unicode", Newlines::Unicode)] {
            for lines_per_file in [1, 2, 3] {
                let whole_prefix = temp_dir
                    .path()
                    .join(format!("whole_{}_{}_", newline_set, lines_per_file))
                    .to_str()
                    .unwrap()
                    .to_string();
                let mut whole_manifest = Vec::new();
                split_data(
                    data,
                    &config(&whole_prefix, lines_per_file, newlines),
                    &mut whole_manifest,
                )
                .unwrap();
                let expected_chunks = read_chunks(&whole_prefix);
                // Only the chunk name differs between the two runs, so drop it before comparing.
                let expected_manifest = String::from_utf8(whole_manifest)
                    .unwrap()
                    .replace(&whole_prefix, "");

                for capacity in [7, 13, 64, 4096] {
                    let prefix = temp_dir
                        .path()
                        .join(format!(
                            "streamed_{}_{}_{}_",
                            newline_set, lines_per_file, capacity
                        ))
                        .to_str()
                        .unwrap()
                        .to_string();
                    let mut streamed_manifest = Vec::new();
                    let mut refill = Refill::new(data, capacity);
                    split_stream(
                        &mut refill,
                        &config(&prefix, lines_per_file, newlines),
                        &mut streamed_manifest,
                    )
                    .unwrap();

                    assert_eq!(
                        read_chunks(&prefix),
                        expected_chunks,
                        "{} newlines at capacity {}",
                        newline_set,
                        capacity
                    );
                    assert_eq!(
                        String::from_utf8(streamed_manifest)
                            .unwrap()
                            .replace(&prefix, ""),
                        expected_manifest,
                        "{} newlines at capacity {}",
                        newline_set,
                        capacity
                    );
                }
            }
        }
    }

    #[test]
    fn streams_an_empty_input_without_chunks() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("empty_").to_str().unwrap().to_string();

        let mut refill = Refill::new(&b""[..], 7);
        split_stream(
            &mut refill,
            &config(&prefix, 2, Newlines::Lf),
            &mut io::sink(),
        )
        .unwrap();

        assert!(read_chunks(&prefix).is_empty());
    }
}
