//! SIMD-accelerated file splitting utility
//!
//! A faster replacement for `split` with UTF-8 awareness.
//! Uses StringZilla for fast line detection.
//!
//! # Examples
//!
//! ```bash
//! # Split by line count
//! sz-split -l 1000 large.txt output_prefix
//!
//! # Split on Unicode newlines rather than LF alone
//! sz-split --utf8 -l 1000 utf8_file.txt output
//!
//! # From stdin
//! cat large.txt | sz-split -l 1000 output
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::num::NonZeroUsize;

use clap::Parser;
use stringzilla::sz::{self, StringZillableBinary};

mod shared;
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
    #[arg(short = 'l', long = "chunk-lines", alias = "lines", value_name = "N",
          value_parser = parse_at_least_one, group = "chunking", help_heading = "Chunk Size")]
    chunk_lines: Option<NonZeroUsize>,

    /// Bytes per output file, never splitting a line; accepts K/M/G/T and Ki/Mi/Gi/Ti
    #[arg(long, value_name = "SIZE", value_parser = parse_size,
          group = "chunking", conflicts_with = "utf8", help_heading = "Chunk Size")]
    chunk_bytes: Option<NonZeroUsize>,

    /// Cut into exactly N files of near-equal bytes at line boundaries; needs a seekable input
    #[arg(long, value_name = "N", value_parser = parse_at_least_one,
          group = "chunking", conflicts_with = "utf8", help_heading = "Chunk Size")]
    chunks: Option<NonZeroUsize>,

    /// Start a new chunk at each line beginning with LITERAL, as `csplit` does with a
    /// regex — FASTA `>`, mbox `From `, SQL `CREATE TABLE`, Markdown `## `
    #[arg(short = 'p', long, value_name = "LITERAL", value_parser = parse_pattern,
          group = "chunking", conflicts_with = "utf8", help_heading = "Chunk Size")]
    pattern: Option<String>,

    /// Fold case when matching --pattern, with full Unicode folding
    ///
    /// Checked in `main` rather than with clap's `requires`, which does not fire for an
    /// argument that belongs to a group another member has already satisfied.
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Repeat the input's first N lines atop every chunk, so each one parses alone.
    /// Breaks `cat <prefix>*` reproducing the input, which is why it is opt-in.
    #[arg(long, value_name = "N", num_args = 0..=1, require_equals = true,
          default_missing_value = "1", value_parser = parse_at_least_one,
          help_heading = "Chunk Size")]
    repeat_header: Option<NonZeroUsize>,

    /// Enable UTF-8 mode (split on Unicode newlines: CR, CRLF, NEL, LS, PS; chunk lines end with LF)
    #[arg(long)]
    utf8: bool,

    /// Suffix length (default: 2, gives aa, ab, ac...)
    #[arg(long, value_parser = parse_at_least_one, default_value = "2")]
    suffix_length: NonZeroUsize,

    /// Emit a JSON Lines manifest of the files written, one record each
    #[arg(long, conflicts_with = "null", help_heading = "Output Formats")]
    json: bool,

    /// Print each written path NUL-terminated, for `xargs -0`
    #[arg(short = '0', long, help_heading = "Output Formats")]
    null: bool,
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
    fn next_cut(&self, data: &[u8], from: usize) -> Option<usize> {
        let mut at = from;
        loop {
            let found = self.needle.find_in(data.get(at..)?)?;
            let position = at + found.offset;
            if position == 0 || data[position - 1] == b'\n' {
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
    fn fill<'d>(&self, rest: &'d [u8], filled: Filled) -> Fill<'d> {
        match *self {
            SplitMode::Lines(per_file) => {
                let wanted = per_file.get() - filled.lines;
                let (span, lines) = span_filling_chunk(rest, wanted);
                Fill {
                    span,
                    lines,
                    closes: filled.lines + lines >= per_file.get(),
                }
            }
            SplitMode::Bytes(budget) => {
                let room = budget.get() - filled.bytes.min(budget.get());
                let span = span_within_budget(rest, room, filled.fresh);
                Fill {
                    span,
                    lines: count_lines(span),
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
                match delimiter.next_cut(rest, from) {
                    Some(cut) => {
                        let span = &rest[..cut];
                        Fill {
                            span,
                            lines: count_lines(span),
                            closes: true,
                        }
                    }
                    None => Fill {
                        span: rest,
                        lines: count_lines(rest),
                        closes: false,
                    },
                }
            }
        }
    }

    /// Lines a chunk holds, when the mode is expressed that way. The line-at-a-time path
    /// asks per line rather than per span, so it reads the budget directly.
    fn line_capacity(&self) -> NonZeroUsize {
        match *self {
            SplitMode::Lines(per_file) => per_file,
            // `--utf8` pairs only with a line budget; the others are rejected at the CLI.
            _ => unreachable!("only a line budget reaches the line-at-a-time path"),
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
        return Err("must not contain a newline, since it matches the start of one line".to_string());
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
    /// Width of the generated suffix.
    suffix_length: NonZeroUsize,
    /// Lines repeated at the top of every chunk, already taken off the input. Empty unless
    /// `--repeat-header` asked for them, which is the only state where `cat <prefix>*` stops
    /// reproducing the input.
    header: &'a [u8],
    /// How each written chunk is announced.
    report: Report,
    /// Lines `header` carries, so the manifest can say how much of a chunk is not data.
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
    /// Whether the bytes written so far end with a newline. The range path copies the
    /// input verbatim, so an unterminated final line reaches the chunk unterminated, and
    /// [`close_chunk`] is where the output's trailing newline is restored.
    ends_with_newline: bool,
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

/// What [`split_by_lines`] carries between windows, so a second call resumes where the
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
fn suffix_exhausted(chunk_number: usize, suffix_length: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "Output file suffixes exhausted: chunk {} does not fit a {}-character --suffix-length",
            chunk_number, suffix_length
        ),
    )
}

/// Create one chunk file, naming it in any failure the caller reports.
fn create_chunk(filename: &str) -> io::Result<BufWriter<File>> {
    let file = File::create(filename).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("Failed to create output file '{}': {}", filename, error),
        )
    })?;
    Ok(BufWriter::new(file))
}

/// Flush the open chunk and record it in the manifest, readying `state` for the next one.
/// A run without `--json` writes the record into [`io::sink`], so there is one path here.
fn close_chunk(
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let Some(mut chunk) = state.open.take() else {
        return Ok(());
    };
    // Output is normalized to end with a newline, whatever the input's last line did.
    if !chunk.ends_with_newline {
        chunk.file.write_all(b"\n")?;
        chunk.bytes += 1;
        chunk.ends_with_newline = true;
    }
    chunk.file.flush()?;
    // The manifest reports the file as it is on disk, header included.
    write_manifest_entry(
        manifest,
        config.report,
        &chunk.name,
        chunk.lines + chunk.header_lines,
        chunk.bytes,
        chunk.header_lines,
    )
}

/// The chunk being filled, opening the next one when none is. GNU `split` stops rather
/// than wrapping onto a name it already wrote, and an exhausted suffix says so here.
fn open_chunk<'a>(
    state: &'a mut SplitState,
    config: &SplitConfig,
) -> io::Result<&'a mut OpenChunk> {
    if state.open.is_none() {
        let width = config.suffix_length.get();
        let suffix = generate_suffix(state.file_index, width)
            .ok_or_else(|| suffix_exhausted(state.file_index + 1, width))?;
        let name = format!("{}{}", config.prefix, suffix);
        let mut file = create_chunk(&name)?;
        // The header is part of the chunk, so it is written and tallied here rather than
        // treated as free: under a byte budget it has to count, or the budget is not one.
        file.write_all(config.header)?;
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
            ends_with_newline: config.header.is_empty() || config.header.ends_with(b"\n"),
            file,
        });
    }
    Ok(state.open.as_mut().expect("just opened"))
}

/// Write every complete line in `data` into chunk files, resuming from `state`. The chunk
/// left open at the end is the caller's to [`close_chunk`].
///
/// A chunk is a contiguous range of the input, so LF mode copies that range whole. Under
/// `--utf8` it is not: every Unicode terminator is rewritten to LF, which only a pass that
/// sees each line can do.
fn split_by_lines(
    data: &[u8],
    state: &mut SplitState,
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    match newlines {
        Newlines::Lf => split_by_ranges(data, state, config, manifest),
        Newlines::Unicode => split_line_by_line(data, state, newlines, config, manifest),
    }
}

/// The prefix of `data` that fills the open chunk: everything through its `wanted`-th
/// newline, or all of `data` when it holds fewer. `wanted` is at least 1, since a chunk
/// closes the moment it fills. Also reports how many lines that prefix carries.
fn span_filling_chunk(data: &[u8], wanted: usize) -> (&[u8], usize) {
    debug_assert!(wanted >= 1, "a full chunk should have been closed already");
    let mut seen = 0;
    for newline in data.sz_matches(b"\n") {
        seen += 1;
        if seen == wanted {
            return (&data[..offset_within(data, newline) + newline.len()], seen);
        }
    }
    // The input ran out first: the tail is the chunk, and an unterminated last line still
    // counts as a line, as it does for the line-at-a-time path below. Counting as we go
    // rather than asking twice keeps this to one pass over the tail.
    (data, seen + usize::from(!data.ends_with(b"\n")))
}

/// Lines in `span`, counting an unterminated last line as one, exactly as
/// [`span_filling_chunk`] does for the mode that counts as it goes.
fn count_lines(span: &[u8]) -> usize {
    if span.is_empty() {
        return 0;
    }
    span.sz_matches(b"\n").count() + usize::from(!span.ends_with(b"\n"))
}

/// The longest prefix of `data` that ends on a line boundary within `room` bytes.
///
/// Empty when the first line alone exceeds the budget and the chunk already holds
/// something — the caller closes it and offers the line to an empty chunk, where
/// `chunk_is_empty` lets it through whole rather than looping forever. Splitting a line is
/// never an option, so an over-long line produces an over-budget chunk of its own.
fn span_within_budget(data: &[u8], room: usize, chunk_is_empty: bool) -> &[u8] {
    // Everything left fits, so there is nothing to cut: returning it whole keeps the chunk
    // open for the next window instead of closing on an unterminated tail.
    if room >= data.len() {
        return data;
    }
    match sz::rfind(&data[..room], b"\n") {
        Some(position) => &data[..position + 1],
        // No boundary inside the budget: take the whole first line if this chunk is
        // otherwise empty, take nothing if it is not.
        None if chunk_is_empty => match sz::find(data, b"\n") {
            Some(position) => &data[..position + 1],
            None => data,
        },
        None => &data[..0],
    }
}

/// Append `span` to the open chunk, keeping its tallies and its end-of-chunk state honest.
///
/// The span goes to the kernel in one call however large it is. Feeding it in fixed pieces
/// was measurably faster once and is not any more — over the 5 GB corpus, 800 MB chunks
/// write in the same 0.69 s either way — so the loop that did it is gone.
fn write_span(chunk: &mut OpenChunk, span: &[u8], lines: usize) -> io::Result<()> {
    chunk.file.write_all(span)?;
    chunk.lines += lines;
    chunk.bytes += span.len();
    // An empty span says nothing about how the chunk ends, and must not overwrite what
    // `close_chunk` reads to decide whether to restore the trailing newline.
    if !span.is_empty() {
        chunk.ends_with_newline = span.ends_with(b"\n");
        chunk.wrote_data = true;
    }
    Ok(())
}

/// Copy each chunk's byte range out of `data` in one write, which is what splitting on LF
/// is: the bytes reach the chunk exactly as they arrived.
fn split_by_ranges(
    data: &[u8],
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut rest = data;
    while !rest.is_empty() {
        // The borrow of the open chunk ends with this block, so `close_chunk` can take
        // `state` again below.
        let closes = {
            let chunk = open_chunk(state, config)?;
            let fill = config.mode.fill(rest, chunk.filled());
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

/// Write one line at a time, re-terminating each with LF. Only `--utf8` needs this, where
/// the seven Unicode terminators are normalized away and no range of the input would do.
fn split_line_by_line(
    data: &[u8],
    state: &mut SplitState,
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    for line in LineIter::new(data, newlines) {
        let filled = {
            let chunk = open_chunk(state, config)?;
            chunk.file.write_all(line)?;
            chunk.file.write_all(b"\n")?;
            chunk.lines += 1;
            chunk.bytes += line.len() + 1;
            chunk.wrote_data = true;
            chunk.lines >= config.mode.line_capacity().get()
        };
        if filled {
            close_chunk(state, config, manifest)?;
        }
    }

    Ok(())
}

/// Where to cut `total` bytes into `wanted` chunks of near-equal size, snapped forward to
/// the next line boundary. Returns the `wanted - 1` interior cut offsets; the last chunk
/// runs to the end.
///
/// Each boundary costs one search from its ideal offset rather than a scan of everything
/// before it, so the whole plan is `wanted` searches over a mapped input regardless of how
/// large it is. Snapping can collapse two boundaries onto the same offset when lines are
/// long relative to `total / wanted`; the chunk between them is then empty, which is
/// reported rather than skipped so that the run yields exactly `wanted` files.
fn plan_equal_chunks(data: &[u8], wanted: NonZeroUsize) -> Vec<usize> {
    let total = data.len();
    let mut cuts = Vec::with_capacity(wanted.get().saturating_sub(1));
    let mut previous = 0;
    for index in 1..wanted.get() {
        // Multiplying first keeps the boundaries evenly spaced; dividing first would
        // truncate each one and drift by up to `wanted` bytes by the last.
        let ideal = total * index / wanted.get();
        let cut = match sz::find(&data[ideal..], b"\n") {
            Some(position) => ideal + position + 1,
            None => total,
        };
        // Boundaries never move backwards, so a chunk is never handed a negative span.
        previous = cut.max(previous);
        cuts.push(previous);
    }
    cuts
}

/// Write `data` as the ranges `cuts` describes, then the tail. Used by `--chunks`, whose
/// boundaries are known before any byte is written, so no per-chunk searching is needed.
fn split_at_offsets(
    data: &[u8],
    cuts: &[usize],
    state: &mut SplitState,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut start = 0;
    for end in cuts.iter().copied().chain([data.len()]) {
        let span = &data[start.min(data.len())..end.min(data.len())];
        write_span(open_chunk(state, config)?, span, count_lines(span))?;
        close_chunk(state, config, manifest)?;
        start = end;
    }
    Ok(())
}

/// Largest header the streaming path will buffer. A header is a handful of column names,
/// so anything past this is a file whose first line never ends — and swallowing it would
/// trade the bounded memory a pipe is chosen for against a header nobody asked to repeat.
const MAX_HEADER_BYTES: usize = 8 << 20;

/// The error a header too large for its budget raises. Emitting header-only chunks instead
/// would fill the disk without ever making progress.
fn header_exceeds_budget(header: usize, budget: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "The {}-byte header does not leave room for data in a {}-byte --chunk-bytes budget",
            header, budget
        ),
    )
}

/// The first records of an input, taken off before splitting begins.
struct Header<'a> {
    /// The header itself, with terminators normalized as the mode normalizes them.
    bytes: Vec<u8>,
    /// What is left of the input once the header is removed.
    rest: &'a [u8],
    /// Records the header carries.
    lines: usize,
    /// Whether the last of them ended, as opposed to running out with the input. A
    /// streaming caller reads this to tell a complete header from a truncated window.
    terminated: bool,
}

/// Split `data` into the first `lines` records and the rest. Under `--utf8` the header is
/// normalized like every other line, so a chunk reads the same whichever mode wrote it.
fn take_header(data: &[u8], lines: NonZeroUsize, newlines: Newlines) -> Header<'_> {
    match newlines {
        Newlines::Lf => {
            let (span, seen) = span_filling_chunk(data, lines.get());
            Header {
                bytes: span.to_vec(),
                rest: &data[span.len()..],
                lines: seen,
                terminated: span.ends_with(b"\n"),
            }
        }
        Newlines::Unicode => {
            // Terminators are normalized to LF here as they are everywhere in this mode, so
            // the header cannot report whether its last line was terminated by inspecting
            // the bytes it produced. Where the *next* line starts answers both questions:
            // it is where the rest begins, and its existence is what proves the header's
            // last line ended rather than being cut off by a window seam.
            let mut header = Vec::new();
            let mut seen = 0;
            let mut lines_iter = LineIter::new(data, newlines).peekable();
            let mut consumed = 0;
            while let Some(line) = lines_iter.next() {
                header.extend_from_slice(line);
                header.push(b'\n');
                seen += 1;
                consumed = match lines_iter.peek() {
                    Some(next) => offset_within(data, next),
                    None => data.len(),
                };
                if seen == lines.get() {
                    break;
                }
            }
            Header {
                bytes: header,
                rest: &data[consumed.min(data.len())..],
                lines: seen,
                terminated: consumed < data.len(),
            }
        }
    }
}

/// Pull a header off the front of a stream before splitting begins, growing the window
/// while the header is still incomplete.
fn take_header_streaming<R: Read>(
    refill: &mut Refill<R>,
    lines: NonZeroUsize,
    newlines: Newlines,
) -> io::Result<(Vec<u8>, usize)> {
    loop {
        // A fresh window holds nothing, and an empty buffer looks like a single
        // unterminated line to the line counter, so fill before asking.
        if !refill.at_eof() && refill.filled().len() < refill.capacity() {
            refill.advance(0)?;
        }
        let filled = refill.filled();
        let header = take_header(filled, lines, newlines);
        // An unterminated last line may simply be a window cut mid-record, so it counts
        // only once the input has ended and no more of it is coming.
        let complete = refill.at_eof() || (header.lines >= lines.get() && header.terminated);
        let consumed = filled.len() - header.rest.len();
        let (bytes, taken) = (header.bytes, header.lines);
        if complete {
            refill.advance(consumed)?;
            return Ok((bytes, taken));
        }
        if refill.filled().len() >= MAX_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "The first {} lines exceed the {} MiB a streamed --repeat-header may buffer",
                    lines.get(),
                    MAX_HEADER_BYTES >> 20
                ),
            ));
        }
        refill.grow()?;
    }
}

/// Split a whole buffer, closing the chunk left open at the end.
fn split_buffer(
    data: &[u8],
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut state = SplitState::default();
    split_by_lines(data, &mut state, newlines, config, manifest)?;
    close_chunk(&mut state, config, manifest)
}

// region: Streaming

/// Drive [`split_by_lines`] over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
fn stream_split<R: Read>(
    refill: &mut Refill<R>,
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut state = SplitState::default();
    refill.for_each_window(newlines.into(), |window| {
        split_by_lines(window, &mut state, newlines, config, manifest)
    })?;
    close_chunk(&mut state, config, manifest)
}

// endregion: Streaming

/// How a written chunk is announced.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Report {
    /// Nothing, which is what a run that asked for neither format gets.
    Silent,
    /// A JSON Lines record carrying the tallies.
    Json,
    /// The path alone, NUL-terminated, for `xargs -0`.
    NullPath,
}

/// Write one manifest record naming a file that was written.
fn write_manifest_entry(
    output: &mut dyn Write,
    report: Report,
    name: &str,
    lines: usize,
    bytes: usize,
    header_lines: usize,
) -> io::Result<()> {
    match report {
        Report::Silent => return Ok(()),
        Report::NullPath => {
            output.write_all(name.as_bytes())?;
            return output.write_all(&[0]);
        }
        Report::Json => {}
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

fn main() {
    let args = Args::parse();
    let mut output = stdout_writer();

    let input = match get_input_streaming(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    // `--chunks` is the one budget that cannot be decided a window at a time, so it names
    // its own mode below rather than joining `SplitMode`.
    // The delimiter outlives the config that borrows it.
    // Folding is a property of the pattern, so asking for it without one is a mistake
    // worth naming rather than a flag that quietly does nothing.
    if args.ignore_case && args.pattern.is_none() {
        let error = io::Error::new(
            io::ErrorKind::InvalidInput,
            "-i/--ignore-case folds the --pattern match, so it needs a --pattern to fold",
        );
        exit_with_error(&mut output, &error, "Error splitting file");
    }

    let pattern = args.pattern.clone().unwrap_or_default();
    let delimiter = Delimiter {
        needle: Literal::new(pattern.as_bytes(), args.ignore_case),
    };

    let mode = match (args.chunk_lines, args.chunk_bytes, &args.pattern) {
        (Some(lines), _, _) => SplitMode::Lines(lines),
        (_, Some(bytes), _) => SplitMode::Bytes(bytes),
        (_, _, Some(_)) => SplitMode::Pattern(&delimiter),
        // `--chunks` reaches `split_at_offsets`, which never asks the mode anything.
        _ => SplitMode::Lines(NonZeroUsize::MIN),
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let result = {
        // Silent unless `--json`, so existing scripts see no new stdout output.
        let report = match (args.json, args.null) {
            (true, _) => Report::Json,
            (_, true) => Report::NullPath,
            _ => Report::Silent,
        };
        let mut discarded = io::sink();
        let manifest: &mut dyn Write = if report == Report::Silent {
            &mut discarded
        } else {
            &mut output
        };
        // The header is taken off the input before any boundary is chosen, so every mode
        // sees only data and every chunk is opened with the header already in it.
        let mut window = input.into_window(DEFAULT_WINDOW_BYTES);
        let (header, header_lines, taken) = match (&mut window, args.repeat_header) {
            (InputWindow::Whole(source), Some(lines)) => {
                let header = take_header(source.as_bytes(), lines, newlines);
                let start = source.as_bytes().len() - header.rest.len();
                (header.bytes, header.lines, start)
            }
            (InputWindow::Stream(refill), Some(lines)) => {
                match take_header_streaming(refill, lines, newlines) {
                    Ok((header, seen)) => (header, seen, 0),
                    Err(error) => exit_with_error(&mut output, &error, "Error reading header"),
                }
            }
            _ => (Vec::new(), 0, 0),
        };

        let config = SplitConfig {
            prefix: &args.prefix,
            mode,
            suffix_length: args.suffix_length,
            report,
            header: &header,
            header_lines,
        };

        // A header that leaves no room for data would repeat forever without progressing.
        if let SplitMode::Bytes(budget) = mode {
            if header.len() >= budget.get() {
                let error = header_exceeds_budget(header.len(), budget.get());
                exit_with_error(&mut output, &error, "Error splitting file");
            }
        }

        match window {
            InputWindow::Whole(source) => {
                let data = &source.as_bytes()[taken..];
                match args.chunks {
                    Some(wanted) => {
                        let cuts = plan_equal_chunks(data, wanted);
                        let mut state = SplitState::default();
                        split_at_offsets(data, &cuts, &mut state, &config, manifest)
                    }
                    None => split_buffer(data, newlines, &config, manifest),
                }
            }
            InputWindow::Stream(mut refill) => match args.chunks {
                // Only a genuine pipe lands here: a `< file` redirect is mapped above.
                Some(_) => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "--chunks needs the input's size, which a pipe does not report; \
                     redirect from a file (sz-split --chunks N < file), name the file, \
                     or use --chunk-bytes",
                )),
                None => stream_split(&mut refill, newlines, &config, manifest),
            },
        }
    };

    if let Err(error) = result.and_then(|()| output.flush()) {
        exit_with_error(&mut output, &error, "Error splitting file");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn rejects_a_zero_count_without_naming_a_rust_type() {
        let zero_lines = vec!["sz-split", "-l", "0"];
        let zero_suffix = vec!["sz-split", "-l", "1", "--suffix-length", "0"];
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
        let Err(error) = Args::try_parse_from(["sz-split", "-l", "abc"]) else {
            panic!("-l abc must be rejected");
        };
        assert!(
            error.to_string().contains("invalid digit found in string"),
            "{}",
            error
        );
        assert!(Args::try_parse_from(["sz-split", "-l", "1"]).is_ok());
    }

    #[test]
    fn pins_this_binarys_short_flags() {
        // Each binary owns its own surface — `shared.rs` holds no clap code, so that a
        // flag added for one tool cannot silently reshape the other ten. Nothing enforces
        // agreement across them, so each pins its own inventory here and divergence shows
        // up as a failing test rather than as an audit.
        let mut shorts: Vec<char> = Args::command()
            .get_arguments()
            .filter_map(|argument| argument.get_short())
            .collect();
        shorts.sort_unstable();
        assert_eq!(
            shorts,
            // `-h` and `-V` are clap's own and are not listed here.
            vec!['0', 'i', 'l', 'p'],
            "sz-split's short flags changed; check the suite-wide meanings before keeping it"
        );
    }

    #[test]
    fn requires_exactly_one_chunking_rule() {
        // Naming none used to be impossible, since `-l` was required; the error now lists
        // every way to say how big a chunk is.
        let Err(error) = Args::try_parse_from(["sz-split", "f.txt"]) else {
            panic!("a run without a chunking rule must be rejected");
        };
        let rendered = error.to_string();
        for flag in ["--chunk-lines", "--chunk-bytes", "--chunks"] {
            assert!(rendered.contains(flag), "{}", rendered);
        }
        assert_eq!(error.exit_code(), 2, "{}", rendered);

        // Naming two is equally a mistake.
        assert!(Args::try_parse_from(["sz-split", "-l", "5", "--chunks", "2"]).is_err());
        // `--lines` stays as an alias, so shipped invocations keep working.
        assert!(Args::try_parse_from(["sz-split", "--lines", "5"]).is_ok());
        assert!(Args::try_parse_from(["sz-split", "--chunk-bytes", "10K"]).is_ok());
    }

    #[test]
    fn fills_chunks_to_a_byte_budget_without_splitting_a_line() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("b.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Bytes(NonZeroUsize::new(8).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            report: Report::Silent,
            header: b"",
            header_lines: 0,
        };
        // Four-byte lines: two fit the budget exactly, and the tail is its own chunk.
        let data = b"aaa\nbbb\nccc\nddd\neee\n";
        split_buffer(data, Newlines::Lf, &config, &mut io::sink()).unwrap();
        assert_eq!(read_chunks(&prefix), vec!["aaa\nbbb\n", "ccc\nddd\n", "eee\n"]);
    }

    #[test]
    fn writes_an_over_long_line_whole_rather_than_splitting_it() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("l.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Bytes(NonZeroUsize::new(5).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            report: Report::Silent,
            header: b"",
            header_lines: 0,
        };
        let data = b"ab\nTHIS-LINE-IS-FAR-TOO-LONG\ncd\n";
        split_buffer(data, Newlines::Lf, &config, &mut io::sink()).unwrap();
        let chunks = read_chunks(&prefix);
        // The long line is over budget and alone, rather than cut in half.
        assert_eq!(chunks[1], "THIS-LINE-IS-FAR-TOO-LONG\n");
        assert_eq!(chunks.concat().as_bytes(), data);
    }

    #[test]
    fn cuts_into_exactly_as_many_chunks_as_asked_for() {
        let temporary = TempDir::new().unwrap();
        let data = b"x\ny\n";
        for wanted in [1usize, 2, 5] {
            let prefix = temporary
                .path()
                .join(format!("n{}.", wanted))
                .display()
                .to_string();
            let config = SplitConfig {
                prefix: &prefix,
                mode: SplitMode::Lines(NonZeroUsize::MIN),
                suffix_length: NonZeroUsize::new(2).unwrap(),
                report: Report::Silent,
                header: b"",
                header_lines: 0,
            };
            let cuts = plan_equal_chunks(data, NonZeroUsize::new(wanted).unwrap());
            let mut state = SplitState::default();
            split_at_offsets(data, &cuts, &mut state, &config, &mut io::sink()).unwrap();
            let chunks = read_chunks(&prefix);
            // Asking for more chunks than there are lines still yields that many files,
            // so a downstream loop over the count is safe; the surplus are empty.
            assert_eq!(chunks.len(), wanted, "asked for {}", wanted);
            assert_eq!(chunks.concat().as_bytes(), data, "asked for {}", wanted);
        }
    }

    #[test]
    fn repeats_the_header_atop_every_chunk_without_spending_the_line_budget() {
        let temporary = TempDir::new().unwrap();
        let prefix = temporary.path().join("h.").display().to_string();
        let config = SplitConfig {
            prefix: &prefix,
            mode: SplitMode::Lines(NonZeroUsize::new(2).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            report: Report::Silent,
            header: b"id,name\n",
            header_lines: 1,
        };
        // `--chunk-lines 2` promises two lines of data, so the header rides on top rather
        // than counting as one of them.
        split_buffer(b"1,a\n2,b\n3,c\n4,d\n", Newlines::Lf, &config, &mut io::sink()).unwrap();
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
            assert!(whole.rest.starts_with(refill.filled()), "capacity {}", capacity);
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
        let fill = SplitMode::Bytes(budget).fill(b"1,a\n2,b\n3,c\n", filled);
        assert_eq!(fill.span, b"1,a\n2,b\n", "8 header bytes leave 8 of the 16");

        // A line budget counts data, so the same header costs it nothing.
        let fill = SplitMode::Lines(NonZeroUsize::new(2).unwrap())
            .fill(b"1,a\n2,b\n3,c\n", Filled::default());
        assert_eq!(fill.span, b"1,a\n2,b\n");
    }

    /// A pattern-mode config; the delimiter is built by the caller because it borrows.
    fn pattern_config<'a>(prefix: &'a str, delimiter: &'a Delimiter<'a>) -> SplitConfig<'a> {
        SplitConfig {
            prefix,
            mode: SplitMode::Pattern(delimiter),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            report: Report::Silent,
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
        split_buffer(
            data,
            Newlines::Lf,
            &pattern_config(&prefix, &delimiter),
            &mut io::sink(),
        )
        .unwrap();
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
        split_buffer(
            b">a\nx>y\n>b\n",
            Newlines::Lf,
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
            split_buffer(
                data,
                Newlines::Lf,
                &pattern_config(&prefix, &delimiter),
                &mut io::sink(),
            )
            .unwrap();
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
            split_buffer(
                data,
                Newlines::Lf,
                &pattern_config(&prefix, &delimiter),
                &mut io::sink(),
            )
            .unwrap();
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
            stream_split(&mut refill, Newlines::Lf, &config, &mut io::sink()).unwrap();
            assert_eq!(read_chunks(&prefix), whole, "capacity {}", capacity);
        }
    }

    #[test]
    fn rejects_a_pattern_that_matches_everywhere_or_conflicts() {
        let Err(error) = Args::try_parse_from(["sz-split", "-p", ""]) else {
            panic!("an empty pattern must be rejected");
        };
        assert!(error.to_string().contains("must not be empty"), "{}", error);
        // A pattern is a line prefix, so a newline inside one contradicts it.
        let Err(error) = Args::try_parse_from(["sz-split", "-p", "a\nb"]) else {
            panic!("a pattern spanning two lines must be rejected");
        };
        assert!(error.to_string().contains("newline"), "{}", error);
        // `--utf8` rewrites terminators, which no byte-exact split may do. Every budget
        // but the line count reaches a path that copies ranges, so all of them refuse it.
        assert!(Args::try_parse_from(["sz-split", "-p", ">", "--utf8"]).is_err());
        assert!(Args::try_parse_from(["sz-split", "--chunk-bytes", "1M", "--utf8"]).is_err());
        assert!(Args::try_parse_from(["sz-split", "--chunks", "2", "--utf8"]).is_err());
        assert!(Args::try_parse_from(["sz-split", "-l", "5", "--utf8"]).is_ok());
        // Folding without a pattern parses, and `main` rejects it: clap's `requires` does
        // not fire for an argument whose group another member has already satisfied.
        assert!(Args::try_parse_from(["sz-split", "-l", "5", "-i"]).is_ok());
        assert!(Args::try_parse_from(["sz-split", "-p", ">", "-i"]).is_ok());
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
    fn config(prefix: &str, lines_per_file: usize) -> SplitConfig<'_> {
        SplitConfig {
            prefix,
            mode: SplitMode::Lines(NonZeroUsize::new(lines_per_file).unwrap()),
            suffix_length: NonZeroUsize::new(2).unwrap(),
            report: Report::Silent,
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
        split_buffer(data, Newlines::Lf, &config(&prefix, 2), &mut io::sink()).unwrap();

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
        split_buffer(data, Newlines::Lf, &config(&prefix, 1), &mut io::sink()).unwrap();

        assert_eq!(fs::read_to_string(format!("{}aa", prefix)).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(format!("{}ab", prefix)).unwrap(), "b\n");
        assert_eq!(fs::read_to_string(format!("{}ac", prefix)).unwrap(), "c\n");
    }

    #[test]
    fn appends_newline_to_last_split_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("notrail_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"line1\nline2";
        split_buffer(data, Newlines::Lf, &config(&prefix, 1), &mut io::sink()).unwrap();

        assert_eq!(
            fs::read_to_string(format!("{}aa", prefix)).unwrap(),
            "line1\n"
        );
        assert_eq!(
            fs::read_to_string(format!("{}ab", prefix)).unwrap(),
            "line2\n"
        );
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
            report: Report::Silent,
            header: b"",
            header_lines: 0,
        };

        let error = split_buffer(&data, Newlines::Lf, &overflowing, &mut io::sink()).unwrap_err();

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
        split_buffer(
            data,
            Newlines::Unicode,
            &config(&unicode_prefix, 1),
            &mut io::sink(),
        )
        .unwrap();
        // Chunk lines are normalized to LF, whatever terminator the input broke on.
        assert_eq!(read_chunks(&unicode_prefix), vec!["a\n", "b\n", "c\n"]);

        let byte_prefix = temp_dir.path().join("byte_").to_str().unwrap().to_string();
        split_buffer(
            data,
            Newlines::Lf,
            &config(&byte_prefix, 1),
            &mut io::sink(),
        )
        .unwrap();
        assert_eq!(read_chunks(&byte_prefix), vec!["a\u{2028}b\u{2028}c\n"]);
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
                split_buffer(
                    data,
                    newlines,
                    &config(&whole_prefix, lines_per_file),
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
                    stream_split(
                        &mut refill,
                        newlines,
                        &config(&prefix, lines_per_file),
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
        stream_split(
            &mut refill,
            Newlines::Lf,
            &config(&prefix, 2),
            &mut io::sink(),
        )
        .unwrap();

        assert!(read_chunks(&prefix).is_empty());
    }
}
