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
    #[arg(short = '0', long, conflicts_with = "json", help_heading = "Output Formats")]
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
    let budget: usize = value.parse().map_err(|_| format!("`{}` is not a number", value))?;
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
            Mode::SplitWhitespaces => SegmentIter::SplitWhitespaces(data.sz_utf8_split_whitespaces()),
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

/// What the run reports once every segment has been seen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Summary {
    /// Nothing; the records were the output.
    None,
    /// The number of records.
    Count,
    /// Nothing at all — the exit code carries whether anything was found.
    Quiet,
}

/// Everything the writer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct OutputConfig {
    /// How to render each record, or `None` when records are not emitted.
    format: Option<Format>,
    summary: Summary,
    terminator: Terminator,
    /// Pack consecutive segments up to this many bytes; tiling modes only.
    chunk_bytes: Option<usize>,
    /// Keep zero-length segments, which only the `--split-*` modes produce.
    keep_empty: bool,
}

impl OutputConfig {
    fn from_args(args: &Args) -> Self {
        // Clap's conflicts keep `-q`, `-c`, `--json` and `--offsets` exclusive.
        let (format, summary) = if args.quiet {
            (None, Summary::Quiet)
        } else if args.count {
            (None, Summary::Count)
        } else if args.json {
            (Some(Format::Json), Summary::None)
        } else if args.offsets {
            (Some(Format::Offsets), Summary::None)
        } else {
            (Some(Format::Plain), Summary::None)
        };
        Self {
            format,
            summary,
            terminator: Terminator::from_null(args.null),
            chunk_bytes: args.chunk_bytes,
            keep_empty: args.keep_empty,
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

/// Segment `data` and write the records, returning how many were emitted.
fn segment_data(
    data: &[u8],
    mode: Mode,
    config: &OutputConfig,
    path_json: &[u8],
    output: &mut dyn Write,
) -> io::Result<usize> {
    if let Some(chunk_bytes) = config.chunk_bytes {
        return write_chunks(data, mode, config, path_json, chunk_bytes, output);
    }

    let mut records = 0;
    for segment in SegmentIter::new(data, mode) {
        if segment.is_empty() && !config.keep_empty {
            continue;
        }
        records += 1;
        let Some(format) = config.format else {
            continue;
        };
        let start = offset_within(data, segment);
        write_record(output, config, format, path_json, segment, start, start + segment.len())?;
    }
    Ok(records)
}

/// Pack consecutive segments into chunks of at most `chunk_bytes`, never splitting one.
/// Tiling modes only, so consecutive segments are contiguous and a chunk is a source span.
/// A segment larger than the budget becomes its own chunk: preserving segments wins over
/// honoring the budget, since the alternative is emitting a broken grapheme or sentence.
fn write_chunks(
    data: &[u8],
    mode: Mode,
    config: &OutputConfig,
    path_json: &[u8],
    chunk_bytes: usize,
    output: &mut dyn Write,
) -> io::Result<usize> {
    debug_assert!(
        mode.tiles(),
        "chunking slices the source span, so it needs a tiling mode"
    );

    let flush = |output: &mut dyn Write, start: usize, end: usize| -> io::Result<()> {
        let Some(format) = config.format else {
            return Ok(());
        };
        write_record(output, config, format, path_json, &data[start..end], start, end)
    };

    let mut records = 0;
    let mut chunk_start = 0;
    let mut chunk_end = 0;

    for segment in SegmentIter::new(data, mode) {
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

/// Segment one input, mapping it into memory when it is a file.
fn segment_input(
    path: Option<&str>,
    mode: Mode,
    config: &OutputConfig,
    output: &mut dyn Write,
) -> io::Result<usize> {
    let input = get_input(path)?;
    let display = path.unwrap_or("-");

    // Escape the path once per file, never per segment.
    let mut path_json = Vec::new();
    if config.format == Some(Format::Json) {
        json_text_field_to(&mut path_json, display.as_bytes())?;
    }

    segment_data(input.as_bytes(), mode, config, &path_json, output)
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

// endregion: Segmentation

fn main() {
    let args = Args::parse();
    let mode = Mode::from_args(&args);
    let config = OutputConfig::from_args(&args);

    let inputs = resolve_inputs(&args);
    let mut output = stdout_writer();
    let mut records = 0;

    for input in &inputs {
        let path = if input == "-" { None } else { Some(input.as_str()) };
        match segment_input(path, mode, &config, &mut output) {
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

    if config.summary == Summary::Count {
        if let Err(error) = writeln!(output, "{}", records) {
            exit_on_write_error(&mut output, &error, "Error writing output");
        }
    }

    if let Err(error) = output.flush() {
        exit_on_write_error(&mut output, &error, "Error writing output");
    }

    if config.summary == Summary::Quiet {
        ExitCode::from_found(records > 0).exit(&mut output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments(data: &[u8], mode: Mode) -> Vec<&[u8]> {
        SegmentIter::new(data, mode).collect()
    }

    fn rendered(data: &[u8], mode: Mode, config: &OutputConfig) -> (String, usize) {
        let mut output = Vec::new();
        let records = segment_data(data, mode, config, br#"{"text":"-"}"#, &mut output).unwrap();
        (String::from_utf8(output).unwrap(), records)
    }

    fn plain() -> OutputConfig {
        OutputConfig {
            format: Some(Format::Plain),
            summary: Summary::None,
            terminator: Terminator::Newline,
            chunk_bytes: None,
            keep_empty: false,
        }
    }

    #[test]
    fn tiling_modes_cover_every_byte() {
        let data = "Hi there. Bye!".as_bytes();
        for mode in [Mode::Graphemes, Mode::Wordbreaks, Mode::Sentences] {
            let total: usize = segments(data, mode).iter().map(|segment| segment.len()).sum();
            assert_eq!(total, data.len(), "segments must tile the input");
        }
    }

    #[test]
    fn breaks_sentences_at_paragraph_separators() {
        // UAX-29 rule SB4 breaks after a paragraph separator, so a hard-wrapped
        // sentence is two sentences. Verified identical to ICU's break iterator.
        let data = "A wrapped\nsentence here. And another.".as_bytes();
        let found = segments(data, Mode::Sentences);
        assert_eq!(found, vec![&b"A wrapped\n"[..], b"sentence here. ", b"And another."]);
    }

    #[test]
    fn splits_sentences_after_abbreviations() {
        // No abbreviation dictionary, per the standard. This is documented behavior,
        // not a defect, and is where `punkt` and `pysbd` genuinely do better.
        let data = "Dr. Smith left.".as_bytes();
        assert_eq!(segments(data, Mode::Sentences), vec![&b"Dr. "[..], b"Smith left."]);
    }

    #[test]
    fn drops_empty_split_segments_by_default() {
        let data = b"a  b";
        let (text, records) = rendered(data, Mode::SplitWhitespaces, &plain());
        assert_eq!(text, "a\nb\n");
        assert_eq!(records, 2);

        let mut keeping = plain();
        keeping.keep_empty = true;
        let (text, records) = rendered(data, Mode::SplitWhitespaces, &keeping);
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
        config.format = Some(Format::Json);
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
