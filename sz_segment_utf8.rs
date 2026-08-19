//! Unicode text segmentation, following UAX-29 and UAX-14. Nothing in coreutils does this, and
//! ICU ships the break iterator as a library with no user-facing binary.
//!
//! Seven boundaries, in two families that are easy to confuse. The tilers — `graphemes`, `words`,
//! `sentences`, `linebreaks` — assign every byte to exactly one segment, so concatenating the
//! output reproduces the input and a word count has to filter for segments carrying a letter or
//! digit. The splitters — `whitespace`, `delimiters`, `newlines` — discard what they split on.
//!
//! `--chunk-bytes` packs consecutive segments into records under a budget without splitting one,
//! and is defined only over the tilers, since a packed span cannot reinsert a discarded separator.
//! A segment larger than the budget is emitted whole.
//!
//! Exit: 0 emitted a record, 1 emitted none, 2 could not run, which includes any named input
//! that could not be read, whatever its readable neighbours produced.

use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{
    StringZillableUnary, Utf8Graphemes, Utf8Linebreaks, Utf8Segments, Utf8Sentences,
    Utf8SplitDelimiters, Utf8SplitNewlines, Utf8SplitWhitespaces, Utf8Wordbreaks,
};

use shared::*;

// region: CLI

/// Segment text into Unicode grapheme clusters, words, sentences, or line breaks
#[derive(Parser)]
#[command(name = "sz-segment-utf8")]
#[command(version, about = "SIMD-accelerated Unicode text segmentation", long_about = None)]
struct Args {
    /// Input files or directories (use '-' or omit for stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// Which boundary splits the input; there is no byte mode, the input is always UTF-8
    #[arg(long, required = true, value_name = "BOUNDARY")]
    by: By,

    /// Keep zero-length segments, which only the splitting boundaries produce
    #[arg(long)]
    keep_empty: bool,

    /// Which record kind to emit (default: segments)
    #[arg(long, help_heading = "Output Formats")]
    show: Option<Show>,

    /// How records are rendered (default: text)
    #[arg(long, help_heading = "Output Formats")]
    format: Option<Format>,

    /// Which columns each record carries, comma-separated; none by default
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        help_heading = "Output Formats"
    )]
    fields: Vec<Field>,

    /// NUL-terminate each output record instead of newline; segments may contain newlines
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Pack consecutive segments into records of at most N bytes, never splitting one [default: one segment per record]
    #[arg(
        long,
        value_name = "N",
        value_parser = parse_size,
        help_heading = "Output Formats"
    )]
    chunk_bytes: Option<NonZeroUsize>,

    /// Suppress all output; exit 0 if any segment was produced, 1 otherwise
    #[arg(long, conflicts_with_all = ["show", "format", "fields", "null"], help_heading = "Output Formats")]
    quiet: bool,

    /// Filter walked files by type (e.g., rust, py, js); named files are always segmented
    #[arg(long = "type", help_heading = "Traversal")]
    file_type: Option<Vec<String>>,

    /// Filter walked files by glob (e.g., "*.rs"); named files are always segmented
    #[arg(long, help_heading = "Traversal")]
    glob: Option<Vec<String>>,

    /// Maximum directory depth [default: unlimited]
    #[arg(long, help_heading = "Traversal")]
    max_depth: Option<usize>,

    /// Include hidden files and directories
    #[arg(long, help_heading = "Traversal")]
    hidden: bool,

    /// Don't respect .gitignore files
    #[arg(long, help_heading = "Traversal")]
    no_ignore: bool,

    /// Follow symbolic links
    #[arg(long, help_heading = "Traversal")]
    follow: bool,
}

/// Which boundary splits the input.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum By {
    /// UAX-29 grapheme clusters: user-perceived characters, including emoji sequences
    Graphemes,
    /// UAX-29 word boundaries; tiles the input, so spaces and punctuation are segments too
    Words,
    /// UAX-29 sentences; no abbreviation dictionary, so "Dr. Smith" splits after "Dr."
    Sentences,
    /// UAX-14 line-break opportunities, including soft wraps; not lines - use newlines
    Linebreaks,
    /// Split on whitespace runs, dropping the separators
    Whitespace,
    /// Split on any Unicode punctuation, symbol, or separator
    Delimiters,
    /// Split on hard line terminators (LF, CR, CRLF, NEL, LS, PS), dropping them
    Newlines,
}

/// One column a record can carry, prefixed before its text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Field {
    /// `start<TAB>end<TAB>`, as byte offsets into the input
    ByteSpan,
}

/// Which record kind the run emits.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Show {
    /// One record per segment
    Segments,
    /// The number of segments in each input, one record per input
    Count,
}

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Segment bytes, one record per line
    Text,
    /// JSON Lines, one object per record
    Json,
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Reject the combinations clap cannot, because they turn on a value rather than a flag.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.show == Some(Show::Count) && !args.fields.is_empty() {
        return Err(reject("--show count cannot be combined with --fields"));
    }
    if args.format == Some(Format::Json) {
        if args.null {
            return Err(reject("--format json cannot be combined with --null"));
        }
        if args.fields.contains(&Field::ByteSpan) {
            return Err(reject(
                "--format json already carries offsets, so --fields byte-span is not allowed",
            ));
        }
    }
    if args.keep_empty && args.by.tiles() {
        return Err(reject(
            "--keep-empty requires --by whitespace, --by delimiters, or --by newlines",
        ));
    }
    if args.chunk_bytes.is_some() && !args.by.tiles() {
        return Err(reject(
            "--chunk-bytes requires --by graphemes, words, sentences, or linebreaks",
        ));
    }
    Ok(())
}

// endregion: CLI

// region: Segmentation Modes

impl By {
    /// Whether segments tile the input, so every byte belongs to exactly one of them.
    /// Only tiling boundaries can be chunked, because only they leave no bytes between
    /// segments, and only the others can yield an empty segment.
    fn tiles(self) -> bool {
        matches!(
            self,
            By::Graphemes | By::Words | By::Sentences | By::Linebreaks
        )
    }

    /// Where a window may be cut so the streamed records match the whole input's. Grapheme,
    /// word and line-break segments never cross a hard line terminator, so a cut at any of
    /// the seven is invisible. Sentences run through VT and FF — `Sentence_Break=Sp` rather
    /// than `Sep` — so only a paragraph separator returns their automaton to its start.
    fn cut_after(self) -> CutAfter {
        match self {
            By::Sentences => CutAfter::ParagraphSeparators,
            _ => CutAfter::LineTerminators,
        }
    }
}

/// Which segmenter runs and which of its segments count. `--keep-empty` lives here rather
/// than beside the output flags because it decides which segments exist, not how they print.
#[derive(Clone, Copy)]
struct Segmentation {
    by: By,
    /// Keep zero-length segments, which only the splitting boundaries produce.
    keep_empty: bool,
}

impl Segmentation {
    /// Read the segmenter and the empty-segment rule the flags name.
    fn from_args(args: &Args) -> Self {
        Segmentation {
            by: args.by,
            keep_empty: args.keep_empty,
        }
    }

    /// Whether a window cut at [`By::cut_after`] yields the records the whole input
    /// yields, which is what lets this segmentation read a pipe in windows.
    ///
    /// `--by delimiters` segments the same bytes differently depending on where they sit
    /// in the buffer — it truncates a token starting within four bytes of a 64-byte boundary
    /// — and streaming shifts every offset, so the two paths would disagree. The splitters
    /// close each window with an empty segment, which the whole input yields only at its
    /// end, and `--keep-empty` is what makes those visible.
    fn window_stable(self) -> bool {
        match self.by {
            By::Graphemes | By::Words | By::Sentences | By::Linebreaks => true,
            By::Whitespace | By::Newlines => !self.keep_empty,
            By::Delimiters => false,
        }
    }
}

/// Whether this run can read a pipe one window at a time. Chunking re-slices the source
/// span, so it needs the whole input however stable the segmenter is on a window.
fn can_stream(segmentation: Segmentation, config: &OutputConfig) -> bool {
    config.chunk_bytes.is_none() && segmentation.window_stable()
}

/// Iterator over one segmenter's output. The seven boundaries are seven distinct types —
/// four alias [`Utf8Segments`] with different kernels, three alias the splitters — so a
/// runtime choice between them needs an enum, exactly as [`LineIter`] does for newlines.
/// Every variant yields borrowed subslices of the input, so no segment is ever copied.
//
// Each variant buffers `ITERATORS_DEFAULT_STEPS` offset and length pairs inline. The
// iterator is built once per file (not per segment), so the size gap is a one-time
// stack cost, not a hot-path allocation.
#[allow(clippy::large_enum_variant)]
enum SegmentIter<'a> {
    Graphemes(Utf8Graphemes<'a>),
    Wordbreaks(Utf8Wordbreaks<'a>),
    Sentences(Utf8Sentences<'a>),
    Linebreaks(Utf8Linebreaks<'a>),
    SplitWhitespaces(Utf8SplitWhitespaces<'a>),
    SplitDelimiters(Utf8SplitDelimiters<'a>),
    SplitNewlines(Utf8SplitNewlines<'a>),
}

impl<'a> SegmentIter<'a> {
    /// Create a segment iterator over the chosen [`By`].
    fn new(data: &'a [u8], by: By) -> Self {
        match by {
            By::Graphemes => SegmentIter::Graphemes(Utf8Segments::new(data)),
            By::Words => SegmentIter::Wordbreaks(Utf8Segments::new(data)),
            By::Sentences => SegmentIter::Sentences(Utf8Segments::new(data)),
            By::Linebreaks => SegmentIter::Linebreaks(Utf8Segments::new(data)),
            By::Whitespace => SegmentIter::SplitWhitespaces(data.sz_utf8_split_whitespaces()),
            By::Delimiters => SegmentIter::SplitDelimiters(data.sz_utf8_split_delimiters()),
            By::Newlines => SegmentIter::SplitNewlines(data.sz_utf8_split_newlines()),
        }
    }
}

impl<'a> Iterator for SegmentIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SegmentIter::Graphemes(iter) => iter.next(),
            SegmentIter::Wordbreaks(iter) => iter.next(),
            SegmentIter::Sentences(iter) => iter.next(),
            SegmentIter::Linebreaks(iter) => iter.next(),
            SegmentIter::SplitWhitespaces(iter) => iter.next(),
            SegmentIter::SplitDelimiters(iter) => iter.next(),
            SegmentIter::SplitNewlines(iter) => iter.next(),
        }
    }
}

// endregion: Segmentation Modes

// region: Output

/// How each record is laid out on the wire.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Render {
    /// The segment bytes alone.
    Plain,
    /// `start<TAB>end<TAB>` followed by the segment bytes.
    Offsets,
    /// One JSON object per record.
    Json,
}

/// What the run puts on stdout.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Report {
    /// One record per segment.
    Records,
    /// The number of segments, once every input has been read.
    Count,
    /// Nothing at all — the exit code carries whether anything was found.
    Quiet,
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig {
    report: Report,
    render: Render,
    terminator: Terminator,
    /// Pack consecutive segments up to this many bytes; tiling boundaries only.
    chunk_bytes: Option<NonZeroUsize>,
}

impl OutputConfig {
    fn from_args(args: &Args) -> Self {
        let report = match (args.quiet, args.show) {
            (true, _) => Report::Quiet,
            (false, Some(Show::Count)) => Report::Count,
            (false, _) => Report::Records,
        };
        let render = if args.format == Some(Format::Json) {
            Render::Json
        } else if args.fields.contains(&Field::ByteSpan) {
            Render::Offsets
        } else {
            Render::Plain
        };
        Self {
            report,
            render,
            terminator: Terminator::from_null(args.null),
            chunk_bytes: args.chunk_bytes,
        }
    }
}

/// Write one record. `path_json` is the pre-escaped path, built once per file so the
/// JSON path is never re-escaped per segment.
fn write_record(
    output: &mut dyn Write,
    config: &OutputConfig,
    path_json: &[u8],
    segment: &[u8],
    start: usize,
    end: usize,
) -> io::Result<()> {
    match config.render {
        Render::Plain => {
            output.write_all(segment)?;
            output.write_all(&[config.terminator.as_byte()])
        }
        Render::Offsets => {
            write!(output, "{}\t{}\t", start, end)?;
            output.write_all(segment)?;
            output.write_all(&[config.terminator.as_byte()])
        }
        Render::Json => {
            output.write_all(br#"{"type":"segment","data":{"path":"#)?;
            output.write_all(path_json)?;
            output.write_all(br#","text":"#)?;
            json_text_field_to(output, segment)?;
            write!(output, r#","start":{},"end":{}}}}}"#, start, end)?;
            output.write_all(b"\n")
        }
    }
}

/// Write one input's segment count, as a record when JSON was asked for.
fn write_count(
    output: &mut dyn Write,
    config: &OutputConfig,
    path_json: &[u8],
    records: usize,
) -> io::Result<()> {
    match config.render {
        Render::Json => {
            output.write_all(br#"{"type":"count","data":{"path":"#)?;
            output.write_all(path_json)?;
            write!(output, r#","count":{}}}}}"#, records)?;
            output.write_all(b"\n")
        }
        _ => {
            write!(output, "{}", records)?;
            output.write_all(&[config.terminator.as_byte()])
        }
    }
}

// endregion: Output

// region: Segmentation

/// Segment a whole input. A `--chunk-bytes` budget packs the segments into chunks, which
/// re-slices the source span and so needs every byte of the input at once.
fn segment_data(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    output: &mut dyn Write,
) -> io::Result<usize> {
    match config.chunk_bytes {
        Some(chunk_bytes) => write_chunks(
            data,
            segmentation,
            config,
            path_json,
            chunk_bytes.get(),
            output,
        ),
        None => write_segments(data, segmentation, config, path_json, 0, output),
    }
}

/// Segment `data` and write the records, returning how many were emitted. `base` is where
/// `data` starts in the input, so a streamed window still reports absolute offsets.
fn write_segments(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    base: usize,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut records = 0;
    for segment in SegmentIter::new(data, segmentation.by) {
        if segment.is_empty() && !segmentation.keep_empty {
            continue;
        }
        records += 1;
        if config.report != Report::Records {
            continue;
        }
        let start = base + offset_within(data, segment);
        write_record(
            output,
            config,
            path_json,
            segment,
            start,
            start + segment.len(),
        )?;
    }
    Ok(records)
}

/// Pack consecutive segments into chunks of at most `chunk_bytes`, never splitting one.
/// Tiling boundaries only, so consecutive segments are contiguous and a chunk is a source
/// span. A segment larger than the budget becomes its own chunk: preserving segments wins
/// over honoring the budget, since the alternative is a broken grapheme or sentence.
fn write_chunks(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    chunk_bytes: usize,
    output: &mut dyn Write,
) -> io::Result<usize> {
    debug_assert!(
        segmentation.by.tiles(),
        "chunking slices the source span, so it needs a tiling boundary"
    );

    let flush = |output: &mut dyn Write, start: usize, end: usize| -> io::Result<()> {
        if config.report != Report::Records {
            return Ok(());
        }
        write_record(output, config, path_json, &data[start..end], start, end)
    };

    let mut records = 0;
    let mut chunk_start = 0;
    let mut chunk_end = 0;

    for segment in SegmentIter::new(data, segmentation.by) {
        // Tiling leaves no gaps, so this segment begins where the previous one ended.
        let segment_end = chunk_end + segment.len();
        if chunk_end > chunk_start && segment_end - chunk_start > chunk_bytes {
            flush(output, chunk_start, chunk_end)?;
            records += 1;
            chunk_start = chunk_end;
        }
        chunk_end = segment_end;
    }

    if chunk_end > chunk_start {
        flush(output, chunk_start, chunk_end)?;
        records += 1;
    }
    Ok(records)
}

// endregion: Segmentation

// region: Streaming

/// Segment a pipe one window at a time, cutting each window where [`By::cut_after`] says
/// the automaton restarts. `base` tracks where the window starts in the input, so
/// `--fields byte-span` and `--format json` report the absolute positions the mapped path does.
fn segment_stream(
    refill: &mut Refill<impl io::Read>,
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut records = 0;
    let mut base = 0;
    refill.for_each_window(segmentation.by.cut_after(), |window| {
        records += write_segments(window, segmentation, config, path_json, base, output)?;
        base += window.len();
        Ok(())
    })?;
    Ok(records)
}

// endregion: Streaming

// region: Inputs

/// Segment one input, mapping it into memory when it is a file and windowing it when it
/// is a pipe.
fn segment_input(
    path: Option<&str>,
    segmentation: Segmentation,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let input = if can_stream(segmentation, config) {
        get_input_streaming(path)?
    } else {
        get_input(path)?
    };
    let shown = path.unwrap_or("-");

    // Escape the path once per file, never per segment.
    let mut path_json = Vec::new();
    if config.render == Render::Json {
        json_text_field_to(&mut path_json, shown.as_bytes())?;
    }

    let records = match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => {
            segment_data(source.as_bytes(), segmentation, config, &path_json, output)?
        }
        InputWindow::Stream(mut refill) => {
            segment_stream(&mut refill, segmentation, config, &path_json, output)?
        }
    };

    if config.report == Report::Count {
        write_count(output, config, &path_json, records)?;
    }
    Ok(records)
}

/// Collect the files named by the inputs, walking directories with ignore support.
fn resolve_inputs(
    inputs: &[String],
    globs: Option<&[glob::Pattern]>,
    traversal: &TraversalOptions<'_>,
) -> Vec<String> {
    let mut resolved = Vec::new();
    for input in inputs {
        let path = Path::new(input);
        if input == "-" || !path.is_dir() {
            resolved.push(input.clone());
            continue;
        }
        for entry in walker(path, traversal, "sz-segment-utf8") {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    eprintln!("sz-segment-utf8: {}", error);
                    continue;
                }
            };
            if !is_readable_entry(&entry) {
                continue;
            }
            // The walker has no glob filter of its own, so `--glob` is applied here.
            if !glob_selects(globs, &entry) {
                continue;
            }
            resolved.push(entry.path().display().to_string());
        }
    }
    resolved
}

// endregion: Inputs

/// The status a finished run reports, from whether any named input went unread and whether
/// any record survived the filter.
fn outcome(inputs: usize, readable: usize, records: usize) -> Status {
    Status::of(readable < inputs, records > 0)
}

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;
    let segmentation = Segmentation::from_args(args);
    let config = OutputConfig::from_args(args);

    let traversal = TraversalOptions {
        hidden: args.hidden,
        no_ignore: args.no_ignore,
        follow: args.follow,
        max_depth: args.max_depth,
        file_type: args.file_type.as_deref(),
    };
    let globs = args
        .glob
        .as_deref()
        .map(compile_globs)
        .transpose()
        .map_err(reject)?;
    let inputs = resolve_inputs(&args.inputs, globs.as_deref(), &traversal);
    let mut records = 0;
    let mut readable = 0;

    for input in &inputs {
        let path = (input != "-").then_some(input.as_str());
        match segment_input(path, segmentation, &config, output) {
            Ok(count) => {
                records += count;
                readable += 1;
            }
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => return Err(error).at(input),
            Err(error) => eprintln!("sz-segment-utf8: {}: {}", input, error),
        }
    }

    output.flush().at("-")?;
    Ok(outcome(inputs.len(), readable, records))
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-segment-utf8", run(&args, &mut output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(data: &[u8], by: By) -> Vec<&[u8]> {
        SegmentIter::new(data, by).collect()
    }

    /// The default segmentation for `by`, which drops the empty segments a splitter emits.
    fn segmenting(by: By) -> Segmentation {
        Segmentation {
            by,
            keep_empty: false,
        }
    }

    fn rendered(data: &[u8], by: By, config: &OutputConfig) -> (String, usize) {
        let (bytes, records) = whole(data, segmenting(by), config);
        (String::from_utf8(bytes).unwrap(), records)
    }

    /// Render the whole input in one call, the way a mapped file is read.
    fn whole(data: &[u8], segmentation: Segmentation, config: &OutputConfig) -> (Vec<u8>, usize) {
        let mut output = Vec::new();
        let records =
            segment_data(data, segmentation, config, br#"{"text":"-"}"#, &mut output).unwrap();
        (output, records)
    }

    /// Render the same input through `capacity`-byte windows, the way a pipe is read.
    fn streamed(
        data: &[u8],
        segmentation: Segmentation,
        config: &OutputConfig,
        capacity: usize,
    ) -> (Vec<u8>, usize) {
        let mut output = Vec::new();
        let mut refill = Refill::new(data, capacity);
        let records = segment_stream(
            &mut refill,
            segmentation,
            config,
            br#"{"text":"-"}"#,
            &mut output,
        )
        .unwrap();
        (output, records)
    }

    const ALL_BOUNDARIES: [By; 7] = [
        By::Graphemes,
        By::Words,
        By::Sentences,
        By::Linebreaks,
        By::Whitespace,
        By::Delimiters,
        By::Newlines,
    ];

    fn by_name(by: By) -> &'static str {
        match by {
            By::Graphemes => "graphemes",
            By::Words => "words",
            By::Sentences => "sentences",
            By::Linebreaks => "linebreaks",
            By::Whitespace => "whitespace",
            By::Delimiters => "delimiters",
            By::Newlines => "newlines",
        }
    }

    /// Every hard terminator, both line endings, blank lines and a missing final newline,
    /// within two-byte codepoints — the UAX-29 word and sentence kernels fault on wider
    /// ones in unoptimized builds, which is where `cargo test` runs.
    const SEAM_CORPUS: &[u8] = concat!(
        "Hello world. Dr. Smith left.\n",
        "A wrapped\nsentence here. And another.\n",
        "crlf line\r\n",
        "bare cr line\rnext\n",
        "vt\u{0b}ff\u{0c} end\n",
        "nel\u{85} end\n",
        "Привет мир, café naïve.\n",
        "tabs\tand   spaces\n",
        "\n\n",
        "no trailing newline at all"
    )
    .as_bytes();

    #[test]
    fn streamed_windows_match_the_whole_input() {
        for by in ALL_BOUNDARIES {
            for keep_empty in [false, true] {
                for render in [Render::Plain, Render::Offsets, Render::Json] {
                    let segmentation = Segmentation { by, keep_empty };
                    let mut config = plain();
                    config.render = render;
                    if !can_stream(segmentation, &config) {
                        continue;
                    }
                    let reference = whole(SEAM_CORPUS, segmentation, &config);
                    // Tiny windows maximize seam crossings; 4096 spans the whole corpus.
                    for capacity in [7, 13, 64, 4096] {
                        assert_eq!(
                            streamed(SEAM_CORPUS, segmentation, &config, capacity),
                            reference,
                            "{} keep_empty={} capacity={}",
                            by_name(by),
                            keep_empty,
                            capacity
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn streamed_offsets_stay_absolute() {
        // Offsets are window-relative before the running base is added, so the last
        // record of a multi-window run is where an unadded base would show up.
        let mut config = plain();
        config.render = Render::Offsets;
        let (bytes, records) = streamed(SEAM_CORPUS, segmenting(By::Newlines), &config, 16);
        let text = String::from_utf8(bytes).unwrap();
        let last = text.lines().next_back().unwrap();
        let start: usize = last.split('\t').next().unwrap().parse().unwrap();
        assert_eq!(
            start,
            SEAM_CORPUS.len() - "no trailing newline at all".len()
        );
        assert_eq!(records, text.lines().count());
    }

    #[test]
    fn grows_the_window_past_a_line_wider_than_it() {
        let mut data = vec![b'x'; 5000];
        data.extend_from_slice(b"\nshort\n");
        let config = plain();
        assert_eq!(
            streamed(&data, segmenting(By::Newlines), &config, 8),
            whole(&data, segmenting(By::Newlines), &config)
        );
    }

    /// Every cut candidate placed inside a sentence: VT and FF run through, NEL, LS, PS and
    /// both line endings end one, a bare CR sits mid-sentence, and the abbreviations put
    /// SB8's unbounded right context across the seams the tiny capacities create.
    const SENTENCE_CORPUS: &[u8] = concat!(
        "Dr. Smith met Mrs. Jones.\n",
        "A sentence spanning\u{0B}a vertical tab stays whole. ",
        "Another spanning\u{0C}a form feed stays whole too.\n",
        "crlf ends this one.\r\n",
        "bare\rcr sits inside this one. ",
        "nel ends this one.\u{0085}",
        "ls ends this one.\u{2028}",
        "ps ends this one.\u{2029}",
        "Mr. Brown left.\r\n",
        "no terminator at the end"
    )
    .as_bytes();

    // LS and PS are three bytes wide, and the UAX-29 sentence kernel faults on those in
    // unoptimized builds, so this one runs under `cargo test --release`.
    #[test]
    #[cfg_attr(
        debug_assertions,
        ignore = "sentence kernel needs an optimized build here"
    )]
    fn streams_sentences_across_every_cut_candidate() {
        let config = plain();
        let segmentation = segmenting(By::Sentences);
        assert!(can_stream(segmentation, &config));
        let reference = whole(SENTENCE_CORPUS, segmentation, &config);
        for capacity in [7, 13, 64, 4096] {
            assert_eq!(
                streamed(SENTENCE_CORPUS, segmentation, &config, capacity),
                reference,
                "capacity={}",
                capacity
            );
        }
    }

    #[test]
    fn sentences_run_through_a_vertical_tab_and_form_feed() {
        // VT and FF are `Sentence_Break=Sp`, so one sentence spans them. Offering them as
        // cut points reports two sentences where the whole input reports one.
        let config = plain();
        for fixture in [&b"one\x0Btwo. Done."[..], &b"one\x0Ctwo. Done."[..]] {
            assert_eq!(segments(fixture, By::Sentences).len(), 2);
            assert_eq!(last_cut(fixture, CutAfter::LineTerminators), Some(4));
            assert_eq!(last_cut(fixture, CutAfter::ParagraphSeparators), None);
            let reference = whole(fixture, segmenting(By::Sentences), &config);
            // Every capacity puts the seam at a different byte, including right on the
            // vertical tab and the form feed.
            for capacity in 1..=fixture.len() + 2 {
                assert_eq!(
                    streamed(fixture, segmenting(By::Sentences), &config, capacity),
                    reference,
                    "capacity={}",
                    capacity
                );
            }
        }
    }

    #[test]
    fn other_modes_break_right_after_a_form_feed() {
        // Grapheme, word and line-break automata restart at every hard terminator, which is
        // what lets those boundaries cut on the wider set.
        for by in [By::Graphemes, By::Words, By::Linebreaks] {
            assert_eq!(by.cut_after(), CutAfter::LineTerminators);
            let mut boundary = 0;
            let breaks = segments(b"one\x0ctwo", by).iter().any(|segment| {
                boundary += segment.len();
                boundary == 4
            });
            assert!(
                breaks,
                "{} must break right after the form feed",
                by_name(by)
            );
        }
        assert_eq!(By::Sentences.cut_after(), CutAfter::ParagraphSeparators);
    }

    fn plain() -> OutputConfig {
        OutputConfig {
            report: Report::Records,
            render: Render::Plain,
            terminator: Terminator::Newline,
            chunk_bytes: None,
        }
    }

    #[test]
    fn tiling_modes_cover_every_byte() {
        let data = "Hi there. Bye!".as_bytes();
        for by in [By::Graphemes, By::Words, By::Sentences] {
            let total: usize = segments(data, by).iter().map(|segment| segment.len()).sum();
            assert_eq!(total, data.len(), "segments must tile the input");
        }
    }

    #[test]
    fn breaks_sentences_at_paragraph_separators() {
        // UAX-29 rule SB4 breaks after a paragraph separator, so a hard-wrapped
        // sentence is two sentences. Verified identical to ICU's break iterator.
        let data = "A wrapped\nsentence here. And another.".as_bytes();
        let found = segments(data, By::Sentences);
        assert_eq!(
            found,
            vec![&b"A wrapped\n"[..], b"sentence here. ", b"And another."]
        );
    }

    #[test]
    fn splits_sentences_after_abbreviations() {
        // No abbreviation dictionary, per the standard. This is documented behavior,
        // not a defect, and is where `punkt` and `pysbd` genuinely do better.
        let data = "Dr. Smith left.".as_bytes();
        assert_eq!(
            segments(data, By::Sentences),
            vec![&b"Dr. "[..], b"Smith left."]
        );
    }

    #[test]
    fn drops_empty_split_segments_by_default() {
        let data = b"a  b";
        let (text, records) = rendered(data, By::Whitespace, &plain());
        assert_eq!(text, "a\nb\n");
        assert_eq!(records, 2);

        let keeping = Segmentation {
            by: By::Whitespace,
            keep_empty: true,
        };
        let (bytes, records) = whole(data, keeping, &plain());
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text, "a\n\nb\n");
        assert_eq!(records, 3);
    }

    #[test]
    fn terminates_records_with_nul() {
        let mut config = plain();
        config.terminator = Terminator::Null;
        let (text, _) = rendered(b"a b", By::Whitespace, &config);
        assert_eq!(text, "a\0b\0");
    }

    #[test]
    fn reports_offsets_that_reconstruct_the_input() {
        let data = "héllo wörld".as_bytes();
        let base = data.as_ptr() as usize;
        for segment in SegmentIter::new(data, By::Graphemes) {
            let start = segment.as_ptr() as usize - base;
            assert_eq!(&data[start..start + segment.len()], segment);
        }
    }

    #[test]
    fn chunks_never_exceed_the_budget_when_segments_fit() {
        let data = "aaaa bbbb cccc dddd".as_bytes();
        let mut config = plain();
        config.chunk_bytes = NonZeroUsize::new(10);
        let (text, records) = rendered(data, By::Words, &config);
        for chunk in text.lines() {
            assert!(chunk.len() <= 10, "chunk {:?} exceeds budget", chunk);
        }
        assert!(records > 1);
        assert_eq!(text.replace('\n', ""), "aaaa bbbb cccc dddd");
    }

    #[test]
    fn oversized_segment_becomes_its_own_chunk() {
        // A single segment larger than the budget is emitted whole rather than split.
        let data = "abcdefghijklmnop".as_bytes();
        let mut config = plain();
        config.chunk_bytes = NonZeroUsize::new(4);
        let (text, records) = rendered(data, By::Words, &config);
        assert_eq!(records, 1);
        assert_eq!(text, "abcdefghijklmnop\n");
    }

    #[test]
    fn emits_json_with_offsets() {
        let mut config = plain();
        config.render = Render::Json;
        let (text, records) = rendered(b"hi", By::Words, &config);
        assert_eq!(records, 1);
        assert_eq!(
            text,
            concat!(
                r#"{"type":"segment","data":{"path":{"text":"-"},"#,
                r#""text":{"text":"hi"},"start":0,"end":2}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn yields_nothing_for_empty_input() {
        let (text, records) = rendered(b"", By::Graphemes, &plain());
        assert!(text.is_empty());
        assert_eq!(records, 0);
    }

    #[test]
    fn counts_as_a_record_carrying_its_path() {
        let mut config = plain();
        config.report = Report::Count;
        config.render = Render::Json;
        let mut output = Vec::new();
        write_count(&mut output, &config, br#"{"text":"book.txt"}"#, 7).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "{\"type\":\"count\",\"data\":{\"path\":{\"text\":\"book.txt\"},\"count\":7}}\n"
        );

        // `--null` terminates every record, and a count is a record.
        let mut config = plain();
        config.report = Report::Count;
        config.terminator = Terminator::Null;
        let mut output = Vec::new();
        write_count(&mut output, &config, b"", 7).unwrap();
        assert_eq!(output, b"7\0");
    }

    #[test]
    fn separates_an_empty_filter_from_an_unreadable_input() {
        // A `--glob` that matched nothing resolves to no inputs at all, which is a run that
        // completed and found nothing — not a run that failed. That distinction is the whole
        // point of this test: it once exited 2 with no message.
        assert!(outcome(0, 0, 0) == Status::NoResult);

        // An input that could not be read is a run that did not complete, whether it was the
        // only one or had readable neighbours that produced records.
        assert!(outcome(2, 0, 0) == Status::Error);
        assert!(outcome(2, 1, 0) == Status::Error);
        assert!(outcome(2, 1, 5) == Status::Error);

        // Everything readable, so only the record count decides.
        assert!(outcome(2, 2, 0) == Status::NoResult);
        assert!(outcome(2, 2, 5) == Status::Success);
    }

    #[test]
    fn counts_compose_with_null_but_not_with_offsets() {
        let parsed = |arguments: &[&str]| Args::try_parse_from(arguments).unwrap();
        let counting = ["sz-segment-utf8", "--by", "words", "--show", "count"];
        assert!(validate(&parsed(&[&counting[..], &["--null"]].concat())).is_ok());
        assert!(validate(&parsed(
            &[&counting[..], &["--fields", "byte-span"]].concat()
        ))
        .is_err());
    }

    #[test]
    fn parses_the_segmentation_boundary() {
        let args = Args::try_parse_from(["sz-segment-utf8", "--by", "words"]).unwrap();
        assert!(args.by == By::Words);
        assert!(Args::try_parse_from(["sz-segment-utf8"]).is_err());
        assert!(Args::try_parse_from(["sz-segment-utf8", "--by", "wordbreaks"]).is_err());
        assert!(Args::try_parse_from(["sz-segment-utf8", "--by", "words", "-c"]).is_err());
    }

    #[test]
    fn declares_no_short_flags() {
        assert!(Args::command()
            .get_arguments()
            .all(|a| a.get_short().is_none() || matches!(a.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|a| a.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "by",
                "keep-empty",
                "show",
                "format",
                "fields",
                "null",
                "chunk-bytes",
                "quiet",
                "type",
                "glob",
                "max-depth",
                "hidden",
                "no-ignore",
                "follow",
                "help",
                "version",
            ]
        );
    }
}
