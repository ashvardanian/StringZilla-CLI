//! SIMD-accelerated Unicode text segmentation
//!
//! Splits text into grapheme clusters, words, sentences, or line-break opportunities
//! following the Unicode standard (UAX-29 and UAX-14). Nothing in coreutils does this,
//! and ICU ships the break iterator as a library with no user-facing binary.
//!
//! # Examples
//!
//! ```bash
//! # One sentence per line
//! sz-segment --sentences book.txt
//!
//! # Count UAX-29 words (not the same as `wc -w`)
//! sz-segment --wordbreaks -c book.txt
//!
//! # NUL-delimited grapheme clusters, safe for xargs
//! sz-segment --graphemes -0 emoji.txt
//!
//! # Byte offsets back into the source, for retrieval pipelines
//! sz-segment --sentences --offsets book.txt
//!
//! # Pack sentences into 2 KB chunks for embedding
//! sz-segment --sentences --chunk-bytes 2000 --json book.txt
//! ```

use std::io::{self, Write};
use std::path::Path;

use clap::{ArgGroup, Parser};
use ignore::WalkBuilder;
use stringzilla::sz::{
    StringZillableUnary, Utf8Graphemes, Utf8Linebreaks, Utf8Segments, Utf8Sentences,
    Utf8SplitDelimiters, Utf8SplitNewlines, Utf8SplitWhitespaces, Utf8Wordbreaks,
};

mod shared;
use shared::*;

// region: CLI

/// Segment text into Unicode grapheme clusters, words, sentences, or line breaks
#[derive(Parser)]
#[command(name = "sz-segment")]
#[command(version, about = "SIMD-accelerated Unicode text segmentation", long_about = None)]
#[command(group(ArgGroup::new("mode").required(true).multiple(false).args([
    "graphemes",
    "wordbreaks",
    "sentences",
    "linebreaks",
    "split_whitespaces",
    "split_delimiters",
    "split_newlines",
])))]
struct Args {
    /// Input files or directories (use '-' for stdin, default: stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// UAX-29 grapheme clusters: user-perceived characters, including emoji sequences
    #[arg(long, help_heading = "Segmentation Modes")]
    graphemes: bool,

    /// UAX-29 word boundaries; tiles the input, so spaces and punctuation are segments too
    #[arg(long, help_heading = "Segmentation Modes")]
    wordbreaks: bool,

    /// UAX-29 sentences; no abbreviation dictionary, so "Dr. Smith" splits after "Dr."
    #[arg(long, help_heading = "Segmentation Modes")]
    sentences: bool,

    /// UAX-14 line-break opportunities, including soft wraps; not lines - use --split-newlines
    #[arg(long, help_heading = "Segmentation Modes")]
    linebreaks: bool,

    /// Split on whitespace runs, dropping the separators
    #[arg(long, help_heading = "Segmentation Modes")]
    split_whitespaces: bool,

    /// Split on any Unicode punctuation, symbol, or separator; takes no delimiter argument
    #[arg(long, help_heading = "Segmentation Modes")]
    split_delimiters: bool,

    /// Split on hard line terminators (LF, CR, CRLF, NEL, LS, PS), dropping them
    #[arg(long, help_heading = "Segmentation Modes")]
    split_newlines: bool,

    /// Keep zero-length segments from the --split-* modes (default: drop them)
    #[arg(long, help_heading = "Segmentation Modes")]
    keep_empty: bool,

    /// NUL-terminate each record instead of newline; needed when segments contain newlines
    #[arg(
        short = '0',
        long,
        conflicts_with = "json",
        help_heading = "Output Formats"
    )]
    null: bool,

    /// Prefix each record with `start<TAB>end<TAB>`, as byte offsets into the input
    #[arg(long, conflicts_with = "json", help_heading = "Output Formats")]
    offsets: bool,

    /// Emit JSON Lines, one object per segment, with byte offsets
    #[arg(long, help_heading = "Output Formats")]
    json: bool,

    /// Print the number of segments instead of the segments themselves
    #[arg(short = 'c', long, conflicts_with_all = ["null", "offsets"], help_heading = "Output Formats")]
    count: bool,

    /// Pack consecutive segments into records of at most N bytes, never splitting one
    #[arg(
        long,
        value_name = "N",
        value_parser = parse_chunk_bytes,
        conflicts_with_all = ["split_whitespaces", "split_delimiters", "split_newlines"],
        help_heading = "Output Formats"
    )]
    chunk_bytes: Option<usize>,

    /// Suppress all output; exit 0 if any segment was produced, 1 otherwise
    #[arg(short = 'q', long, conflicts_with_all = ["json", "count", "offsets", "null"], help_heading = "Output Formats")]
    quiet: bool,

    /// Maximum directory depth (default: unlimited)
    #[arg(long, help_heading = "Traversal")]
    max_depth: Option<usize>,

    /// Include hidden files and directories
    #[arg(long, help_heading = "Traversal")]
    hidden: bool,

    /// Don't respect .gitignore files
    #[arg(long, help_heading = "Traversal")]
    no_ignore: bool,
}

/// Parse a `--chunk-bytes` budget, rejecting zero as unsatisfiable.
fn parse_chunk_bytes(value: &str) -> Result<usize, String> {
    let budget: usize = value
        .parse()
        .map_err(|_| format!("`{}` is not a number", value))?;
    if budget == 0 {
        return Err("must be at least 1".to_string());
    }
    Ok(budget)
}

// endregion: CLI

// region: Segmentation Modes

/// Which segmenter splits the input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// UAX-29 grapheme clusters.
    Graphemes,
    /// UAX-29 word boundaries.
    Wordbreaks,
    /// UAX-29 sentences.
    Sentences,
    /// UAX-14 line-break opportunities.
    Linebreaks,
    /// Runs between Unicode whitespace.
    SplitWhitespaces,
    /// Runs between Unicode punctuation, symbols, and separators.
    SplitDelimiters,
    /// Runs between hard line terminators.
    SplitNewlines,
}

impl Mode {
    /// Pick the mode named by the flags. Clap's `ArgGroup` guarantees exactly one.
    fn from_args(args: &Args) -> Self {
        if args.graphemes {
            Mode::Graphemes
        } else if args.wordbreaks {
            Mode::Wordbreaks
        } else if args.sentences {
            Mode::Sentences
        } else if args.linebreaks {
            Mode::Linebreaks
        } else if args.split_whitespaces {
            Mode::SplitWhitespaces
        } else if args.split_delimiters {
            Mode::SplitDelimiters
        } else {
            Mode::SplitNewlines
        }
    }

    /// Whether segments tile the input, so every byte belongs to exactly one of them.
    /// Only tiling modes can be chunked, because only they leave no bytes between segments.
    fn tiles(self) -> bool {
        matches!(
            self,
            Mode::Graphemes | Mode::Wordbreaks | Mode::Sentences | Mode::Linebreaks
        )
    }

    /// Where a window may be cut so the streamed records match the whole input's. Grapheme,
    /// word and line-break segments never cross a hard line terminator, so a cut at any of
    /// the seven is invisible. Sentences run through VT and FF — `Sentence_Break=Sp` rather
    /// than `Sep` — so only a paragraph separator returns their automaton to its start.
    fn cut_after(self) -> CutAfter {
        match self {
            Mode::Sentences => CutAfter::ParagraphSeparators,
            _ => CutAfter::LineTerminators,
        }
    }
}

/// Which segmenter runs and which of its segments count. `--keep-empty` lives here rather
/// than beside the output flags because it decides which segments exist, not how they print.
#[derive(Clone, Copy)]
struct Segmentation {
    mode: Mode,
    /// Keep zero-length segments, which only the `--split-*` modes produce.
    keep_empty: bool,
}

impl Segmentation {
    /// Read the segmenter and the empty-segment rule the flags name.
    fn from_args(args: &Args) -> Self {
        Segmentation {
            mode: Mode::from_args(args),
            keep_empty: args.keep_empty,
        }
    }

    /// Whether a window cut at [`Mode::cut_after`] yields the records the whole input
    /// yields, which is what lets this segmentation read a pipe in windows.
    ///
    /// `--split-delimiters` segments the same bytes differently depending on where they sit
    /// in the buffer — it truncates a token starting within four bytes of a 64-byte boundary
    /// — and streaming shifts every offset, so the two paths would disagree. The splitters
    /// close each window with an empty segment, which the whole input yields only at its
    /// end, and `--keep-empty` is what makes those visible.
    fn window_stable(self) -> bool {
        match self.mode {
            Mode::Graphemes | Mode::Wordbreaks | Mode::Sentences | Mode::Linebreaks => true,
            Mode::SplitWhitespaces | Mode::SplitNewlines => !self.keep_empty,
            Mode::SplitDelimiters => false,
        }
    }
}

/// Whether this run can read a pipe one window at a time. Chunking re-slices the source
/// span, so it needs the whole input however stable the segmenter is on a window.
fn can_stream(segmentation: Segmentation, config: &OutputConfig) -> bool {
    config.chunk_bytes.is_none() && segmentation.window_stable()
}

/// Iterator over one segmenter's output. The seven modes are seven distinct types —
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
    /// Create a segment iterator over the chosen [`Mode`].
    fn new(data: &'a [u8], mode: Mode) -> Self {
        match mode {
            Mode::Graphemes => SegmentIter::Graphemes(Utf8Segments::new(data)),
            Mode::Wordbreaks => SegmentIter::Wordbreaks(Utf8Segments::new(data)),
            Mode::Sentences => SegmentIter::Sentences(Utf8Segments::new(data)),
            Mode::Linebreaks => SegmentIter::Linebreaks(Utf8Segments::new(data)),
            Mode::SplitWhitespaces => {
                SegmentIter::SplitWhitespaces(data.sz_utf8_split_whitespaces())
            }
            Mode::SplitDelimiters => SegmentIter::SplitDelimiters(data.sz_utf8_split_delimiters()),
            Mode::SplitNewlines => SegmentIter::SplitNewlines(data.sz_utf8_split_newlines()),
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

/// How each record is rendered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// The segment bytes alone.
    Plain,
    /// `start<TAB>end<TAB>` followed by the segment bytes.
    Offsets,
    /// One JSON object per segment.
    Json,
}

/// What the run puts on stdout.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Report {
    /// One record per segment, rendered this way.
    Records(Format),
    /// The number of segments, once every input has been read.
    Count,
    /// Nothing at all — the exit code carries whether anything was found.
    Quiet,
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig {
    report: Report,
    terminator: Terminator,
    /// Pack consecutive segments up to this many bytes; tiling modes only.
    chunk_bytes: Option<usize>,
}

impl OutputConfig {
    fn from_args(args: &Args) -> Self {
        // Clap's conflicts keep `-q`, `-c`, `--json` and `--offsets` exclusive.
        let report = if args.quiet {
            Report::Quiet
        } else if args.count {
            Report::Count
        } else if args.json {
            Report::Records(Format::Json)
        } else if args.offsets {
            Report::Records(Format::Offsets)
        } else {
            Report::Records(Format::Plain)
        };
        Self {
            report,
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
    format: Format,
    path_json: &[u8],
    segment: &[u8],
    start: usize,
    end: usize,
) -> io::Result<()> {
    match format {
        Format::Plain => {
            output.write_all(segment)?;
            output.write_all(&[config.terminator.as_byte()])
        }
        Format::Offsets => {
            write!(output, "{}\t{}\t", start, end)?;
            output.write_all(segment)?;
            output.write_all(&[config.terminator.as_byte()])
        }
        Format::Json => {
            output.write_all(br#"{"type":"segment","data":{"path":"#)?;
            output.write_all(path_json)?;
            output.write_all(br#","text":"#)?;
            json_text_field_to(output, segment)?;
            write!(output, r#","start":{},"end":{}}}}}"#, start, end)?;
            output.write_all(b"\n")
        }
    }
}

// endregion: Output

// region: Segmentation

/// Segment a whole input. A `--chunk-bytes` budget packs the segments into chunks, which
/// re-slices the source span and so needs every byte of the input at once.
fn segment_whole(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    output: &mut dyn Write,
) -> io::Result<usize> {
    match config.chunk_bytes {
        Some(chunk_bytes) => {
            write_chunks(data, segmentation, config, path_json, chunk_bytes, output)
        }
        None => segment_data(data, segmentation, config, path_json, 0, output),
    }
}

/// Segment `data` and write the records, returning how many were emitted. `base` is where
/// `data` starts in the input, so a streamed window still reports absolute offsets.
fn segment_data(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    base: usize,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut records = 0;
    for segment in SegmentIter::new(data, segmentation.mode) {
        if segment.is_empty() && !segmentation.keep_empty {
            continue;
        }
        records += 1;
        let Report::Records(format) = config.report else {
            continue;
        };
        let start = base + offset_within(data, segment);
        write_record(
            output,
            config,
            format,
            path_json,
            segment,
            start,
            start + segment.len(),
        )?;
    }
    Ok(records)
}

/// Pack consecutive segments into chunks of at most `chunk_bytes`, never splitting one.
/// Tiling modes only, so consecutive segments are contiguous and a chunk is a source span.
/// A segment larger than the budget becomes its own chunk: preserving segments wins over
/// honoring the budget, since the alternative is emitting a broken grapheme or sentence.
fn write_chunks(
    data: &[u8],
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    chunk_bytes: usize,
    output: &mut dyn Write,
) -> io::Result<usize> {
    debug_assert!(
        segmentation.mode.tiles(),
        "chunking slices the source span, so it needs a tiling mode"
    );

    let flush = |output: &mut dyn Write, start: usize, end: usize| -> io::Result<()> {
        let Report::Records(format) = config.report else {
            return Ok(());
        };
        write_record(
            output,
            config,
            format,
            path_json,
            &data[start..end],
            start,
            end,
        )
    };

    let mut records = 0;
    let mut chunk_start = 0;
    let mut chunk_end = 0;

    for segment in SegmentIter::new(data, segmentation.mode) {
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

/// Segment a pipe one window at a time, cutting each window where [`Mode::cut_after`] says
/// the mode's automaton restarts. `base` tracks where the window starts in the input, so
/// `--offsets` and `--json` report the same absolute positions the whole-slice path reports.
fn segment_stream(
    refill: &mut Refill<impl io::Read>,
    segmentation: Segmentation,
    config: &OutputConfig,
    path_json: &[u8],
    output: &mut dyn Write,
) -> io::Result<usize> {
    let mut records = 0;
    let mut base = 0;
    refill.for_each_window(segmentation.mode.cut_after(), |window| {
        records += segment_data(window, segmentation, config, path_json, base, output)?;
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
    let display = path.unwrap_or("-");

    // Escape the path once per file, never per segment.
    let mut path_json = Vec::new();
    if config.report == Report::Records(Format::Json) {
        json_text_field_to(&mut path_json, display.as_bytes())?;
    }

    match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => {
            segment_whole(source.as_bytes(), segmentation, config, &path_json, output)
        }
        InputWindow::Stream(mut refill) => {
            segment_stream(&mut refill, segmentation, config, &path_json, output)
        }
    }
}

/// Collect the files named by the inputs, walking directories with ignore support.
fn resolve_inputs(args: &Args) -> Vec<String> {
    let mut resolved = Vec::new();
    for input in &args.inputs {
        let path = Path::new(input);
        if input != "-" && path.is_dir() {
            let mut builder = WalkBuilder::new(path);
            builder
                .hidden(!args.hidden)
                .git_ignore(!args.no_ignore)
                .git_global(!args.no_ignore)
                .git_exclude(!args.no_ignore);
            if let Some(depth) = args.max_depth {
                builder.max_depth(Some(depth));
            }
            for entry in builder.build().flatten() {
                if is_readable_entry(&entry) {
                    resolved.push(entry.path().display().to_string());
                }
            }
        } else {
            resolved.push(input.clone());
        }
    }
    resolved
}

// endregion: Inputs

fn main() {
    let args = Args::parse();
    let segmentation = Segmentation::from_args(&args);
    let config = OutputConfig::from_args(&args);

    let inputs = resolve_inputs(&args);
    let mut output = stdout_writer();
    let mut records = 0;

    for input in &inputs {
        let path = if input == "-" {
            None
        } else {
            Some(input.as_str())
        };
        match segment_input(path, segmentation, &config, &mut output) {
            Ok(count) => records += count,
            Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {
                ExitCode::Success.exit(&mut output)
            }
            Err(error) => {
                eprintln!("Error reading {}: {}", input, error);
                ExitCode::Error.exit(&mut output);
            }
        }
    }

    if config.report == Report::Count {
        if let Err(error) = writeln!(output, "{}", records) {
            exit_on_write_error(&mut output, &error, "Error writing output");
        }
    }

    if let Err(error) = output.flush() {
        exit_on_write_error(&mut output, &error, "Error writing output");
    }

    if config.report == Report::Quiet {
        ExitCode::from_found(records > 0).exit(&mut output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(data: &[u8], mode: Mode) -> Vec<&[u8]> {
        SegmentIter::new(data, mode).collect()
    }

    /// The default segmentation for `mode`, which drops the empty segments a splitter emits.
    fn segmenting(mode: Mode) -> Segmentation {
        Segmentation {
            mode,
            keep_empty: false,
        }
    }

    fn rendered(data: &[u8], mode: Mode, config: &OutputConfig) -> (String, usize) {
        let (bytes, records) = whole(data, segmenting(mode), config);
        (String::from_utf8(bytes).unwrap(), records)
    }

    /// Render the whole input in one call, the way a mapped file is read.
    fn whole(data: &[u8], segmentation: Segmentation, config: &OutputConfig) -> (Vec<u8>, usize) {
        let mut output = Vec::new();
        let records =
            segment_whole(data, segmentation, config, br#"{"text":"-"}"#, &mut output).unwrap();
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

    const ALL_MODES: [Mode; 7] = [
        Mode::Graphemes,
        Mode::Wordbreaks,
        Mode::Sentences,
        Mode::Linebreaks,
        Mode::SplitWhitespaces,
        Mode::SplitDelimiters,
        Mode::SplitNewlines,
    ];

    fn mode_name(mode: Mode) -> &'static str {
        match mode {
            Mode::Graphemes => "graphemes",
            Mode::Wordbreaks => "wordbreaks",
            Mode::Sentences => "sentences",
            Mode::Linebreaks => "linebreaks",
            Mode::SplitWhitespaces => "split-whitespaces",
            Mode::SplitDelimiters => "split-delimiters",
            Mode::SplitNewlines => "split-newlines",
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
        for mode in ALL_MODES {
            for keep_empty in [false, true] {
                for format in [Format::Plain, Format::Offsets, Format::Json] {
                    let segmentation = Segmentation { mode, keep_empty };
                    let mut config = plain();
                    config.report = Report::Records(format);
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
                            mode_name(mode),
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
        config.report = Report::Records(Format::Offsets);
        let (bytes, records) = streamed(SEAM_CORPUS, segmenting(Mode::SplitNewlines), &config, 16);
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
            streamed(&data, segmenting(Mode::SplitNewlines), &config, 8),
            whole(&data, segmenting(Mode::SplitNewlines), &config)
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
        let segmentation = segmenting(Mode::Sentences);
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
            assert_eq!(segments(fixture, Mode::Sentences).len(), 2);
            assert_eq!(last_cut(fixture, CutAfter::LineTerminators), Some(4));
            assert_eq!(last_cut(fixture, CutAfter::ParagraphSeparators), None);
            let reference = whole(fixture, segmenting(Mode::Sentences), &config);
            // Every capacity puts the seam at a different byte, including right on the
            // vertical tab and the form feed.
            for capacity in 1..=fixture.len() + 2 {
                assert_eq!(
                    streamed(fixture, segmenting(Mode::Sentences), &config, capacity),
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
        // what lets those modes cut on the wider set.
        for mode in [Mode::Graphemes, Mode::Wordbreaks, Mode::Linebreaks] {
            assert_eq!(mode.cut_after(), CutAfter::LineTerminators);
            let mut boundary = 0;
            let breaks = segments(b"one\x0ctwo", mode).iter().any(|segment| {
                boundary += segment.len();
                boundary == 4
            });
            assert!(
                breaks,
                "{} must break right after the form feed",
                mode_name(mode)
            );
        }
        assert_eq!(Mode::Sentences.cut_after(), CutAfter::ParagraphSeparators);
    }

    fn plain() -> OutputConfig {
        OutputConfig {
            report: Report::Records(Format::Plain),
            terminator: Terminator::Newline,
            chunk_bytes: None,
        }
    }

    #[test]
    fn tiling_modes_cover_every_byte() {
        let data = "Hi there. Bye!".as_bytes();
        for mode in [Mode::Graphemes, Mode::Wordbreaks, Mode::Sentences] {
            let total: usize = segments(data, mode)
                .iter()
                .map(|segment| segment.len())
                .sum();
            assert_eq!(total, data.len(), "segments must tile the input");
        }
    }

    #[test]
    fn breaks_sentences_at_paragraph_separators() {
        // UAX-29 rule SB4 breaks after a paragraph separator, so a hard-wrapped
        // sentence is two sentences. Verified identical to ICU's break iterator.
        let data = "A wrapped\nsentence here. And another.".as_bytes();
        let found = segments(data, Mode::Sentences);
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
            segments(data, Mode::Sentences),
            vec![&b"Dr. "[..], b"Smith left."]
        );
    }

    #[test]
    fn drops_empty_split_segments_by_default() {
        let data = b"a  b";
        let (text, records) = rendered(data, Mode::SplitWhitespaces, &plain());
        assert_eq!(text, "a\nb\n");
        assert_eq!(records, 2);

        let keeping = Segmentation {
            mode: Mode::SplitWhitespaces,
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
        let (text, _) = rendered(b"a b", Mode::SplitWhitespaces, &config);
        assert_eq!(text, "a\0b\0");
    }

    #[test]
    fn reports_offsets_that_reconstruct_the_input() {
        let data = "héllo wörld".as_bytes();
        let base = data.as_ptr() as usize;
        for segment in SegmentIter::new(data, Mode::Graphemes) {
            let start = segment.as_ptr() as usize - base;
            assert_eq!(&data[start..start + segment.len()], segment);
        }
    }

    #[test]
    fn chunks_never_exceed_the_budget_when_segments_fit() {
        let data = "aaaa bbbb cccc dddd".as_bytes();
        let mut config = plain();
        config.chunk_bytes = Some(10);
        let (text, records) = rendered(data, Mode::Wordbreaks, &config);
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
        config.chunk_bytes = Some(4);
        let (text, records) = rendered(data, Mode::Wordbreaks, &config);
        assert_eq!(records, 1);
        assert_eq!(text, "abcdefghijklmnop\n");
    }

    #[test]
    fn emits_json_with_offsets() {
        let mut config = plain();
        config.report = Report::Records(Format::Json);
        let (text, records) = rendered(b"hi", Mode::Wordbreaks, &config);
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
        let (text, records) = rendered(b"", Mode::Graphemes, &plain());
        assert!(text.is_empty());
        assert_eq!(records, 0);
    }
}
