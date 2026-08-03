//! SIMD-accelerated word count utility
//!
//! A faster replacement for `wc` with proper UTF-8 support and directory traversal.
//! Uses StringZilla for SIMD-accelerated counting operations.
//!
//! Counting measures rather than searches, so an empty input is a successful answer:
//! a row of zeros and exit 0, as `wc` reports it, where the tools that search exit 1
//! on finding nothing.
//!
//! # Examples
//!
//! ```bash
//! # Count single file
//! sz-count file.txt
//!
//! # Count multiple files
//! sz-count src/*.rs
//!
//! # Count directory recursively
//! sz-count src/
//!
//! # Human-readable output
//! sz-count --format human src/
//!
//! # UTF-8 mode (count characters, Unicode whitespace/newlines)
//! sz-count --utf8 --fields lines,words,chars docs/
//!
//! # Just the line count, as a bare integer for scripts
//! sz-count --fields lines file.txt
//!
//! # Match `wc` byte-for-byte
//! sz-count --posix file.txt
//! ```

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use clap::{error::ErrorKind, CommandFactory, Parser, ValueEnum};
use ignore::{Walk, WalkBuilder};
use stringzilla::sz;
use stringzilla::sz::StringZillableUnary;

mod shared;
use shared::*;

// region: CLI

/// Count lines, words, bytes, and characters in files and directories
#[derive(Parser)]
#[command(name = "sz-count")]
#[command(version, about = "SIMD-accelerated word count", long_about = None)]
struct Args {
    /// Input files or directories (use '-' for stdin, default: stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// Which measurements to emit
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "lines,words,bytes",
        help_heading = "Fields"
    )]
    fields: Vec<Field>,

    /// Treat the input as UTF-8 text
    #[arg(long, help_heading = "Counting Modes")]
    utf8: bool,

    /// Match `wc` byte-for-byte: newline-terminated lines and the C `isspace` set
    #[arg(long, conflicts_with = "utf8", help_heading = "Counting Modes")]
    posix: bool,

    /// How records are rendered
    #[arg(
        long,
        value_enum,
        default_value = "table",
        help_heading = "Output Formats"
    )]
    format: Format,

    /// Suppress all output; exit 0 if anything was counted, 1 otherwise
    #[arg(long, conflicts_with_all = ["format", "fields"], help_heading = "Output Formats")]
    quiet: bool,

    /// Filter walked files by type (e.g., rust, py, js); named files are always counted
    #[arg(long = "type", help_heading = "Traversal")]
    file_type: Option<Vec<String>>,

    /// Filter walked files by glob (e.g., "*.rs"); named files are always counted
    #[arg(long, help_heading = "Traversal")]
    glob: Option<Vec<String>>,

    /// Maximum directory depth (default: unlimited)
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

/// One measurement a row can carry, in the order [`HEADERS`] lists them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Field {
    Lines,
    Words,
    Bytes,
    Chars,
    MaxLineLength,
}

/// How rows are rendered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    /// Aligned columns with comma-grouped digits.
    Table,
    /// Aligned columns with plain digits.
    Plain,
    /// Aligned columns with K/M/G/T suffixes.
    Human,
    /// JSON Lines with untruncated paths and bare integers.
    Json,
}

/// The one constraint clap cannot express: `conflicts_with` fires on a flag's
/// presence, never on its value.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.posix && args.fields.contains(&Field::Chars) {
        return Err(Args::command().error(
            ErrorKind::ArgumentConflict,
            "--posix counts bytes, so it cannot emit --fields chars",
        ));
    }
    Ok(())
}

// endregion: CLI

// region: Counting Modes

/// How lines, words, and characters are delimited.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// `wc`-compatible: lines are newline bytes, so an unterminated final line
    /// does not count, and words are separated by the C `isspace` set.
    Posix,
    /// Byte-level, with an unterminated final line counted.
    Ascii,
    /// Unicode newlines and whitespace, with code point counting.
    Unicode,
}

impl Mode {
    /// Pick the mode named by the flags. Clap keeps `--posix` and `--utf8` exclusive.
    #[inline]
    fn from_args(posix: bool, utf8: bool) -> Self {
        if posix {
            Mode::Posix
        } else if utf8 {
            Mode::Unicode
        } else {
            Mode::Ascii
        }
    }

    /// Which newline set delimits lines.
    #[inline]
    fn newlines(self) -> Newlines {
        match self {
            Mode::Posix | Mode::Ascii => Newlines::Lf,
            Mode::Unicode => Newlines::Unicode,
        }
    }

    /// Whether a final line without a terminator counts as a line.
    /// `wc -l` counts newline bytes, so it does not.
    #[inline]
    fn counts_unterminated_line(self) -> bool {
        match self {
            Mode::Posix => false,
            Mode::Ascii | Mode::Unicode => true,
        }
    }

    /// Whether words are separated by Unicode whitespace rather than the C `isspace` set.
    #[inline]
    fn separates_words_by_unicode(self) -> bool {
        match self {
            Mode::Posix | Mode::Ascii => false,
            Mode::Unicode => true,
        }
    }

    /// Whether every newline this mode breaks lines on is also a word separator,
    /// which is what lets [`Plan::WordsOnly`] scan the buffer without line bounds.
    #[inline]
    fn newlines_separate_words(self) -> bool {
        match (self.newlines(), self.separates_words_by_unicode()) {
            // LF sits in the C `isspace` set.
            (Newlines::Lf, false) => is_posix_space(b'\n'),
            // All seven Unicode newlines sit in the 25-code-point `White_Space` set.
            (Newlines::Unicode, true) => true,
            // Mixing the two sets would need its own proof.
            _ => false,
        }
    }
}

/// C's `isspace` set: space plus the 0x09..=0x0D control run, which is
/// `\t \n \v \f \r`. Rust's `is_ascii_whitespace` omits `\v`.
#[inline]
fn is_posix_space(byte: u8) -> bool {
    byte == b' ' || (0x09..=0x0D).contains(&byte)
}

// endregion: Counting Modes

// region: Counts and Fields

/// Every measurement's header, in column order.
const HEADERS: &[&str] = &["lines", "words", "bytes", "chars", "maxline"];

/// How many measurements [`Counts`] carries, and so the most columns a table has.
/// [`Counts::values`] and [`Fields::selectors`] are sized by it, so a sixth
/// measurement that reaches only one of the three fails to compile.
const FIELD_COUNT: usize = HEADERS.len();

/// Longest name the table prints in full; anything longer is truncated from the left.
const NAME_WIDTH: usize = 40;

/// Columns a tree branch glyph occupies ahead of an entry's name, as in `├─ `.
const TREE_GLYPH_WIDTH: usize = 3;

/// One input's measurements. Every field is always computed cheaply enough to
/// keep this a plain record; [`Fields`] decides which of them reach the output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    lines: usize,
    words: usize,
    bytes: usize,
    chars: usize,
    max_line_length: usize,
}

impl Counts {
    fn add(&mut self, other: &Counts) {
        self.lines += other.lines;
        self.words += other.words;
        self.bytes += other.bytes;
        self.chars += other.chars;
        self.max_line_length = self.max_line_length.max(other.max_line_length);
    }

    /// Every measurement in column order, matching [`HEADERS`].
    fn values(self) -> [usize; FIELD_COUNT] {
        [
            self.lines,
            self.words,
            self.bytes,
            self.chars,
            self.max_line_length,
        ]
    }

    /// The selected measurements in column order, paired with their headers.
    /// Backed by stack arrays, so iterating a row allocates nothing.
    fn columns(self, fields: Fields) -> impl Iterator<Item = (&'static str, usize)> {
        HEADERS
            .iter()
            .copied()
            .zip(self.values())
            .zip(fields.selectors())
            .filter(|(_, selected)| *selected)
            .map(|(column, _)| column)
    }
}

/// Which measurements to compute and print. Mirrors [`Counts`] field for field.
#[derive(Clone, Copy)]
struct Fields {
    lines: bool,
    words: bool,
    bytes: bool,
    chars: bool,
    max_line_length: bool,
}

impl Fields {
    /// Read the selectors `--fields` names.
    fn from_selection(selection: &[Field]) -> Self {
        let holds = |field| selection.contains(&field);
        Self {
            lines: holds(Field::Lines),
            words: holds(Field::Words),
            bytes: holds(Field::Bytes),
            chars: holds(Field::Chars),
            max_line_length: holds(Field::MaxLineLength),
        }
    }

    /// Which measurements are selected, in column order, matching [`HEADERS`].
    fn selectors(self) -> [bool; FIELD_COUNT] {
        [
            self.lines,
            self.words,
            self.bytes,
            self.chars,
            self.max_line_length,
        ]
    }

    /// Whether any selected field needs the bytes themselves. Bytes alone are the
    /// length a stat reports, so every other selector is what calls for a pass.
    fn reads_content(self) -> bool {
        self.lines || self.words || self.chars || self.max_line_length
    }
}

// endregion: Counts and Fields

// region: Counting

/// Which pass over the data the selected fields call for. Characters are orthogonal,
/// having their own whole-buffer pass, and bytes are the buffer length, so the five
/// selectors collapse to four plans.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Plan {
    /// Nothing to scan: the length answers everything asked for.
    Size,
    /// One line pass, for the line count and the longest line.
    LineExtents,
    /// One whitespace pass over the whole buffer, with no line bounds needed.
    WordsOnly,
    /// One line pass that also counts the words each line holds.
    LinesAndWords,
}

impl Plan {
    /// Pick the pass the selected fields call for.
    fn of(fields: Fields) -> Self {
        let lines = fields.lines || fields.max_line_length;
        match (lines, fields.words) {
            (true, true) => Plan::LinesAndWords,
            (true, false) => Plan::LineExtents,
            (false, true) => Plan::WordsOnly,
            (false, false) => Plan::Size,
        }
    }
}

/// Everything the counting passes need, decided once from `Args`, mirroring the
/// [`RenderConfig`] the printer is handed.
#[derive(Clone, Copy)]
struct Counter {
    mode: Mode,
    fields: Fields,
    plan: Plan,
    /// Where a streamed window may end without the counters losing anything.
    cut: CutAfter,
}

/// The running tallies plus the two facts a window boundary would otherwise lose:
/// whether the bytes so far end inside a word, and whether they end with a newline.
#[derive(Default)]
struct CountState {
    counts: Counts,
    inside_word: bool,
    ends_with_newline: bool,
}

impl Counter {
    /// Derive the pass to take and the weakest cut it allows, once per run.
    fn new(mode: Mode, fields: Fields) -> Self {
        let plan = Plan::of(fields);
        let cut = match plan {
            // A line pass has to see whole lines.
            Plan::LineExtents | Plan::LinesAndWords => mode.newlines().into(),
            // `count_utf8` and the Unicode word splitter decode, so they need whole
            // characters; the byte-level word loop reads one byte at a time and does not.
            Plan::Size | Plan::WordsOnly => {
                if fields.chars || mode.separates_words_by_unicode() {
                    CutAfter::Characters
                } else {
                    CutAfter::Anywhere
                }
            }
        };
        Counter {
            mode,
            fields,
            plan,
            cut,
        }
    }

    /// Count the selected metrics for given data, taking only the passes they call for.
    fn count_data(self, data: &[u8]) -> Counts {
        let mut state = CountState::default();
        self.count_window(data, &mut state);
        self.finish(state)
    }

    /// Fold one window into the running tallies. Every plan but [`Plan::WordsOnly`] needs
    /// the window to hold whole lines, and every decoding pass needs it to hold whole
    /// characters; [`Counter::cut`] is what guarantees both.
    fn count_window(self, window: &[u8], state: &mut CountState) {
        state.counts.bytes += window.len();
        if self.fields.chars {
            // One pass over the whole window, so whitespace and newlines are included.
            // Skipped entirely when no selector asked for characters.
            state.counts.chars += sz::count_utf8(window);
        }

        match self.plan {
            Plan::Size => {}
            Plan::LineExtents => count_line_extents(window, self.mode, &mut state.counts),
            Plan::WordsOnly => {
                // Every newline is itself a separator, so no word run spans a line break
                // and the whole window counts as one line.
                debug_assert!(self.mode.newlines_separate_words());
                state.counts.words +=
                    count_words_resuming(window, self.mode, &mut state.inside_word);
            }
            Plan::LinesAndWords => count_lines_and_words(window, self.mode, &mut state.counts),
        }

        if let Some(&last) = window.last() {
            state.ends_with_newline = last == b'\n';
        }
    }

    /// Report the tallies, applying `wc`'s rule that a final line without a
    /// terminator is not a line.
    fn finish(self, state: CountState) -> Counts {
        let mut counts = state.counts;
        // `LineIter` credits an unterminated final line; `wc -l` counts newline bytes only.
        if !self.mode.counts_unterminated_line() && !state.ends_with_newline {
            counts.lines = counts.lines.saturating_sub(1);
        }
        counts
    }
}

/// Count lines and measure the longest one.
#[inline]
fn count_line_extents(data: &[u8], mode: Mode, counts: &mut Counts) {
    for line in LineIter::new(data, mode.newlines()) {
        counts.lines += 1;
        counts.max_line_length = counts.max_line_length.max(line.len());
    }
}

/// Count lines, measure the longest one, and count the words they hold in one pass.
#[inline]
fn count_lines_and_words(data: &[u8], mode: Mode, counts: &mut Counts) {
    for line in LineIter::new(data, mode.newlines()) {
        counts.lines += 1;
        counts.max_line_length = counts.max_line_length.max(line.len());
        counts.words += count_words(line, mode);
    }
}

/// Count maximal runs of non-whitespace in one line.
///
/// The byte loop is scalar by measurement: over a 200 MB slice on 2026-08-02 it holds
/// 1.27 GB/s, where a scan alternating `sz::find_byteset` over a byte set built once
/// holds 0.35 GB/s. One kernel call costs 13.7 ns because the Ice Lake byteset kernel
/// rebuilds its lookup vectors per call, and a 9.5-byte word leaves 7.5 ns to beat.
#[inline]
fn count_words(line: &[u8], mode: Mode) -> usize {
    if mode.separates_words_by_unicode() {
        return line.sz_utf8_split_whitespaces().skip_empty().count();
    }

    let mut words = 0;
    let mut inside_word = false;
    for &byte in line {
        let separates = is_posix_space(byte);
        words += usize::from(!separates && !inside_word);
        inside_word = !separates;
    }
    words
}

/// Count the runs [`count_words`] counts, discounting one the previous window left open
/// and reporting whether this window ends inside one, so a word a window boundary splits
/// is counted once. Seeding the byte loop itself would cost it its vectorization, so the
/// resume is a correction on its result rather than a different loop.
#[inline]
fn count_words_resuming(data: &[u8], mode: Mode, inside_word: &mut bool) -> usize {
    let (Some(&first), Some(&last)) = (data.first(), data.last()) else {
        return 0;
    };

    if mode.separates_words_by_unicode() {
        // The splitter cuts on separators, so a run is empty exactly where two separators
        // met, and the outer two runs report which end of the window sits inside a word.
        let mut runs = data.sz_utf8_split_whitespaces();
        let mut words = 0;
        let mut ends_inside_word = false;
        if let Some(opening) = runs.next() {
            ends_inside_word = !opening.is_empty();
            words += usize::from(ends_inside_word && !*inside_word);
        }
        for run in runs {
            ends_inside_word = !run.is_empty();
            words += usize::from(ends_inside_word);
        }
        *inside_word = ends_inside_word;
        return words;
    }

    let words = count_words(data, mode) - usize::from(*inside_word && !is_posix_space(first));
    *inside_word = !is_posix_space(last);
    words
}

/// The size the kernel keeps for a regular file, when it is worth trusting.
/// Synthetic files misreport it in both directions — `/proc` says zero, `/sys` says
/// a whole page — so the claim stands only when the byte it calls last reads back,
/// which also excludes empty files, free to read anyway. Opening rather than stat'ing
/// keeps an unreadable file failing the way reading it would.
fn trusted_file_size(path: &Path) -> io::Result<Option<usize>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let Ok(size) = usize::try_from(metadata.len()) else {
        return Ok(None);
    };
    if !metadata.is_file() || size == 0 {
        return Ok(None);
    }
    let mut last_byte = [0u8; 1];
    let probe = file
        .seek(SeekFrom::Start(metadata.len() - 1))
        .and_then(|_| file.read(&mut last_byte));
    if !matches!(probe, Ok(1)) {
        return Ok(None);
    }
    Ok(Some(size))
}

// endregion: Counting

// region: Streaming

impl Counter {
    /// Count a single file, mapping it only when a selected field reads its bytes.
    fn count_file(self, path: &Path) -> io::Result<Counts> {
        if !self.fields.reads_content() {
            if let Some(bytes) = trusted_file_size(path)? {
                return Ok(Counts {
                    bytes,
                    ..Counts::default()
                });
            }
        }
        let input = open_input(path)?;
        Ok(self.count_data(input.as_bytes()))
    }

    /// Count a stream through one reused window, so peak memory is the window rather than
    /// the input. Only a true pipe reaches this; every other source is mapped or buffered.
    fn count_stream<R: Read>(self, mut refill: Refill<R>) -> io::Result<Counts> {
        let mut state = CountState::default();
        refill.for_each_window(self.cut, |window| {
            self.count_window(window, &mut state);
            Ok(())
        })?;
        Ok(self.finish(state))
    }

    /// Count stdin, mapping a redirect the way a named file is mapped. A pipe has neither a
    /// length to stat nor a slice to scan, so a plan that reads nothing drains it and every
    /// other plan windows it.
    fn count_stdin(self) -> io::Result<Counts> {
        match get_input_streaming(None)?.into_window(DEFAULT_WINDOW_BYTES) {
            InputWindow::Whole(source) => Ok(self.count_data(source.as_bytes())),
            InputWindow::Stream(refill) if !self.fields.reads_content() => Ok(Counts {
                bytes: drained_length(refill)?,
                ..Counts::default()
            }),
            InputWindow::Stream(refill) => self.count_stream(refill),
        }
    }
}

/// Read a stream to its end and report how many bytes went past, keeping none of them.
/// Retaining nothing makes every window a fresh read, so the cost stays one window.
fn drained_length<R: Read>(mut refill: Refill<R>) -> io::Result<usize> {
    let mut bytes = 0;
    refill.for_each_window(CutAfter::Anywhere, |window| {
        bytes += window.len();
        Ok(())
    })?;
    Ok(bytes)
}

// endregion: Streaming

// region: Rendering

/// Everything the printer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct RenderConfig {
    fields: Fields,
    format: Format,
}

/// Format number with K/M/G/T suffixes
fn format_human(value: usize) -> String {
    if value < 1_000 {
        value.to_string()
    } else if value < 1_000_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else if value < 1_000_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value < 1_000_000_000_000 {
        format!("{:.1}G", value as f64 / 1_000_000_000.0)
    } else {
        format!("{:.1}T", value as f64 / 1_000_000_000_000.0)
    }
}

/// Render one measurement into `buffer`, which the aligned table pads to width.
/// Grouped numbers borrow the buffer; the other two forms have to allocate.
fn format_number<'a>(
    value: usize,
    config: &RenderConfig,
    buffer: &'a mut [u8; 26],
) -> Cow<'a, str> {
    match config.format {
        Format::Human => Cow::Owned(format_human(value)),
        Format::Table => Cow::Borrowed(format_grouped_number(buffer, value)),
        Format::Plain | Format::Json => Cow::Owned(value.to_string()),
    }
}

/// Truncate path to fit within max_width, keeping the rightmost part.
/// Counts characters, not bytes, so multi-byte paths neither panic nor mis-measure.
fn truncate_path(path: &str, max_width: usize) -> Cow<'_, str> {
    let characters = path.chars().count();
    if characters <= max_width {
        return Cow::Borrowed(path);
    }
    let keep = max_width.saturating_sub(3); // Reserve 3 characters for "..."
    let tail: String = path.chars().skip(characters - keep).collect();
    Cow::Owned(format!("...{}", tail))
}

/// Every width one table needs: the name column, and each selected count column.
struct TableWidths {
    name: usize,
    columns: [usize; FIELD_COUNT],
}

/// Width of each selected column: the widest of its header and any of its values.
/// A fixed width silently misaligns the moment a count outgrows it.
fn column_widths<'a>(
    rows: impl Iterator<Item = &'a Counts>,
    config: &RenderConfig,
) -> [usize; FIELD_COUNT] {
    let mut widths = [0usize; FIELD_COUNT];
    for (index, (header, _)) in Counts::default().columns(config.fields).enumerate() {
        widths[index] = header.len();
    }
    let mut buffer = [0u8; 26];
    for counts in rows {
        for (index, (_, value)) in counts.columns(config.fields).enumerate() {
            widths[index] = widths[index].max(format_number(value, config, &mut buffer).len());
        }
    }
    widths
}

/// Width of the name column: the widest name any row prints, its tree glyphs included and
/// [`NAME_WIDTH`] the bound past which [`truncate_path`] takes over. Padding every table to
/// that bound instead strands one short name a screen away from its own counts.
fn name_width<'a>(names: impl Iterator<Item = (&'a str, usize)>) -> usize {
    names
        .map(|(name, prefix)| prefix + name.chars().count().min(NAME_WIDTH - prefix))
        .max()
        .unwrap_or(0)
}

/// Write the header row
fn print_header(
    output: &mut dyn Write,
    config: &RenderConfig,
    widths: &TableWidths,
) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = widths.name)?;
    for ((header, _), width) in Counts::default()
        .columns(config.fields)
        .zip(&widths.columns)
    {
        write!(output, " {:>width$}", header, width = width)?;
    }
    output.write_all(b"\n")
}

/// Write one row of measurements against a left-aligned name, itself prefixed by
/// any tree glyphs. Writing the two parts separately keeps the row allocation-free.
fn print_row(
    output: &mut dyn Write,
    prefix: &str,
    name: &str,
    counts: &Counts,
    config: &RenderConfig,
    widths: &TableWidths,
) -> io::Result<()> {
    let prefix_width = prefix.chars().count();
    let name = truncate_path(name, NAME_WIDTH.saturating_sub(prefix_width));
    output.write_all(prefix.as_bytes())?;
    write!(
        output,
        "{:width$}",
        name,
        width = widths.name.saturating_sub(prefix_width)
    )?;
    let mut buffer = [0u8; 26];
    for ((_, value), width) in counts.columns(config.fields).zip(&widths.columns) {
        write!(
            output,
            " {:>width$}",
            format_number(value, config, &mut buffer),
            width = width
        )?;
    }
    output.write_all(b"\n")
}

/// Write the separator line above the totals
fn print_separator(
    output: &mut dyn Write,
    config: &RenderConfig,
    widths: &TableWidths,
) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = widths.name)?;
    for (_, width) in Counts::default()
        .columns(config.fields)
        .zip(&widths.columns)
    {
        write!(output, " {:─>width$}", "", width = width)?;
    }
    output.write_all(b"\n")
}

/// Write the unnamed totals row
fn print_totals(
    output: &mut dyn Write,
    counts: &Counts,
    config: &RenderConfig,
    widths: &TableWidths,
) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = widths.name)?;
    let mut buffer = [0u8; 26];
    for ((_, value), width) in counts.columns(config.fields).zip(&widths.columns) {
        write!(
            output,
            " {:>width$}",
            format_number(value, config, &mut buffer),
            width = width
        )?;
    }
    output.write_all(b"\n")
}

/// Write one `count` record, with the untruncated path and bare integers.
fn write_counts_json(
    output: &mut dyn Write,
    path: &str,
    counts: &Counts,
    fields: Fields,
) -> io::Result<()> {
    output.write_all(br#"{"type":"count","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    for (header, value) in counts.columns(fields) {
        write!(output, r#","{}":{}"#, header, value)?;
    }
    output.write_all(b"}}\n")
}

/// Write the final `total` record, which also reports how many inputs were read.
fn write_total_json(
    output: &mut dyn Write,
    files: usize,
    counts: &Counts,
    fields: Fields,
) -> io::Result<()> {
    write!(output, r#"{{"type":"total","data":{{"files":{}"#, files)?;
    for (header, value) in counts.columns(fields) {
        write!(output, r#","{}":{}"#, header, value)?;
    }
    output.write_all(b"}}\n")
}

/// One counted input.
struct Row {
    /// Full path, which JSON reports untruncated.
    path: String,
    /// What the table prints, which the tree view strips to a relative name.
    name: String,
    counts: Counts,
}

/// Everything one run has to print.
struct Report {
    rows: Vec<Row>,
    total: Counts,
    /// The directory the rows hang off, when a lone directory was named.
    root: Option<String>,
}

/// Write the whole report.
fn render(output: &mut dyn Write, report: &Report, config: &RenderConfig) -> io::Result<()> {
    if config.format == Format::Json {
        for row in &report.rows {
            write_counts_json(output, &row.path, &row.counts, config.fields)?;
        }
        if report.rows.len() > 1 || report.root.is_some() {
            write_total_json(output, report.rows.len(), &report.total, config.fields)?;
        }
        return Ok(());
    }

    if let ([row], None) = (report.rows.as_slice(), &report.root) {
        let mut columns = row.counts.columns(config.fields);
        if let (Some((_, value)), None) = (columns.next(), columns.next()) {
            // Exactly one selector and exactly one input: a bare integer, for `n=$(...)`.
            return writeln!(output, "{}", value);
        }
        let widths = TableWidths {
            name: name_width(std::iter::once((row.name.as_str(), 0))),
            columns: column_widths(std::iter::once(&row.counts), config),
        };
        print_header(output, config, &widths)?;
        return print_row(output, "", &row.name, &row.counts, config, &widths);
    }

    let prefix = if report.root.is_some() {
        TREE_GLYPH_WIDTH
    } else {
        0
    };
    let widths = TableWidths {
        name: name_width(
            report
                .root
                .iter()
                .map(|root| (root.as_str(), 0))
                .chain(report.rows.iter().map(|row| (row.name.as_str(), prefix))),
        ),
        columns: column_widths(
            report
                .rows
                .iter()
                .map(|row| &row.counts)
                .chain([&report.total]),
            config,
        ),
    };

    print_header(output, config, &widths)?;
    if let Some(root) = &report.root {
        print_row(output, "", root, &report.total, config, &widths)?;
    }
    for (index, row) in report.rows.iter().enumerate() {
        let branch = match (&report.root, index + 1 == report.rows.len()) {
            (None, _) => "",
            (Some(_), true) => "└─ ",
            (Some(_), false) => "├─ ",
        };
        print_row(output, branch, &row.name, &row.counts, config, &widths)?;
    }
    print_separator(output, config, &widths)?;
    print_totals(output, &report.total, config, &widths)
}

// endregion: Rendering

// region: Input Processing

/// Walk `path`, skipping hidden and ignored entries unless the flags say otherwise.
fn walk(path: &Path, args: &Args) -> Walk {
    let mut builder = WalkBuilder::new(path);
    builder
        .hidden(!args.hidden)
        .git_ignore(!args.no_ignore)
        .git_global(!args.no_ignore)
        .git_exclude(!args.no_ignore)
        .follow_links(args.follow);
    if let Some(depth) = args.max_depth {
        builder.max_depth(Some(depth));
    }
    if let Some(names) = &args.file_type {
        let mut types = ignore::types::TypesBuilder::new();
        types.add_defaults();
        for name in names {
            types.select(name);
        }
        match types.build() {
            Ok(matcher) => {
                builder.types(matcher);
            }
            Err(error) => eprintln!("sz-count: invalid --type: {}", error),
        }
    }
    builder.build()
}

/// Compile each `--glob` once, so a malformed one is reported here rather than
/// silently matching nothing on every file of the walk.
fn compile_globs(patterns: &[String]) -> Vec<glob::Pattern> {
    patterns
        .iter()
        .filter_map(|pattern| match glob::Pattern::new(pattern) {
            Ok(compiled) => Some(compiled),
            Err(error) => {
                eprintln!("sz-count: invalid glob '{}': {}", pattern, error);
                None
            }
        })
        .collect()
}

/// Whether a walked file passes the `--glob` filter, which matches either the whole
/// path or the file name, as `sz-find` does.
fn glob_selects(globs: &Option<Vec<glob::Pattern>>, path: &Path) -> bool {
    let Some(globs) = globs else {
        return true;
    };
    let path_text = path.to_string_lossy();
    let name_text = path.file_name().unwrap_or_default().to_string_lossy();
    globs
        .iter()
        .any(|pattern| pattern.matches(&path_text) || pattern.matches(&name_text))
}

/// The files a directory holds, sorted, with every walk failure warned about.
fn walk_files(
    path: &Path,
    args: &Args,
    globs: &Option<Vec<glob::Pattern>>,
    failures: &mut usize,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for result in walk(path, args) {
        match result {
            Ok(entry) if is_readable_entry(&entry) && glob_selects(globs, entry.path()) => {
                paths.push(entry.path().to_path_buf())
            }
            Ok(_) => {}
            Err(error) => {
                eprintln!("sz-count: {}", error);
                *failures += 1;
            }
        }
    }
    paths.sort();
    paths
}

/// What one command-line input turned out to name.
enum Resolved {
    Stdin,
    File(PathBuf),
    Directory(PathBuf),
}

/// Resolve every input, warning on the ones that cannot be stat'ed and counting them,
/// so a missing file no longer abandons the inputs beside it.
fn resolve_inputs(args: &Args, failures: &mut usize) -> Vec<Resolved> {
    let mut resolved = Vec::new();
    for input in &args.inputs {
        if input == "-" {
            resolved.push(Resolved::Stdin);
            continue;
        }
        let path = Path::new(input);
        // One stat answers both questions, and reports why an unreadable path is
        // unreadable rather than calling every failure a missing file.
        match fs::metadata(path) {
            Ok(metadata) if metadata.is_dir() => resolved.push(Resolved::Directory(path.into())),
            Ok(_) => resolved.push(Resolved::File(path.into())),
            Err(error) => {
                eprintln!("sz-count: {}: {}", input, error);
                *failures += 1;
            }
        }
    }
    resolved
}

/// Count `path`, folding it into `report` or warning about why it could not be read.
fn push_row(
    report: &mut Report,
    counter: Counter,
    path: &Path,
    name: String,
    failures: &mut usize,
) {
    match counter.count_file(path) {
        Ok(counts) => {
            report.total.add(&counts);
            report.rows.push(Row {
                path: path.to_string_lossy().into_owned(),
                name,
                counts,
            });
        }
        Err(error) => {
            eprintln!("sz-count: {}: {}", path.display(), error);
            *failures += 1;
        }
    }
}

/// Count every input, in the order they were named.
fn gather(args: &Args, counter: Counter, failures: &mut usize) -> Report {
    let globs = args.glob.as_deref().map(compile_globs);
    let resolved = resolve_inputs(args, failures);
    let mut report = Report {
        rows: Vec::new(),
        total: Counts::default(),
        root: None,
    };

    // A lone directory gets the tree view, with per-file rows under a summary.
    if let [Resolved::Directory(directory)] = resolved.as_slice() {
        let text = directory.to_string_lossy().into_owned();
        report.root = Some(if text.ends_with('/') {
            text
        } else {
            format!("{}/", text)
        });
        for path in walk_files(directory, args, &globs, failures) {
            let name = path
                .strip_prefix(directory)
                .unwrap_or(&path)
                .display()
                .to_string();
            push_row(&mut report, counter, &path, name, failures);
        }
        return report;
    }

    for input in &resolved {
        match input {
            Resolved::Stdin => match counter.count_stdin() {
                Ok(counts) => {
                    report.total.add(&counts);
                    report.rows.push(Row {
                        path: "-".to_string(),
                        name: "-".to_string(),
                        counts,
                    });
                }
                Err(error) => {
                    eprintln!("sz-count: -: {}", error);
                    *failures += 1;
                }
            },
            Resolved::File(path) => {
                let name = path.display().to_string();
                push_row(&mut report, counter, path, name, failures);
            }
            Resolved::Directory(directory) => {
                for path in walk_files(directory, args, &globs, failures) {
                    let name = path.display().to_string();
                    push_row(&mut report, counter, &path, name, failures);
                }
            }
        }
    }
    report
}

/// The status a finished run reports. Nothing readable did not complete; nothing
/// counted completed and found nothing.
fn outcome(report: &Report, failures: usize) -> ExitCode {
    if !report.rows.is_empty() {
        ExitCode::Success
    } else if failures > 0 {
        ExitCode::Error
    } else {
        ExitCode::NoResult
    }
}

// endregion: Input Processing

fn run(output: &mut dyn Write) -> io::Result<ExitCode> {
    let args = Args::parse();
    if let Err(error) = validate(&args) {
        error.exit();
    }

    let mode = Mode::from_args(args.posix, args.utf8);
    let fields = Fields::from_selection(&args.fields);
    let counter = Counter::new(mode, fields);

    let mut failures = 0;
    let report = gather(&args, counter, &mut failures);
    let code = outcome(&report, failures);

    if code == ExitCode::Success && !args.quiet {
        let config = RenderConfig {
            fields,
            format: args.format,
        };
        render(output, &report, &config)?;
    }
    Ok(code)
}

fn main() {
    let mut output = stdout_writer();
    match run(&mut output) {
        Ok(code) => code.exit(&mut output),
        Err(error) => exit_on_write_error(&mut output, &error, "sz-count"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every selector set, in the bit order [`Counts::columns`] lists them.
    fn fields_from_bits(bits: u8) -> Fields {
        Fields {
            lines: bits & 0b00001 != 0,
            words: bits & 0b00010 != 0,
            bytes: bits & 0b00100 != 0,
            chars: bits & 0b01000 != 0,
            max_line_length: bits & 0b10000 != 0,
        }
    }

    fn counted(data: &[u8], mode: Mode) -> Counts {
        Counter::new(mode, fields_from_bits(0b11111)).count_data(data)
    }

    /// Inputs whose newlines, whitespace, and terminators exercise every plan seam.
    const EDGE_INPUTS: [&[u8]; 12] = [
        b"",
        b"\n",
        b"a\nb",
        b"abc",
        b"ab\nabcd\nabc\n",
        b"a\x0Bb\x0Cc\rd e\n",
        // Seven Unicode terminators, then a non-breaking space between two words.
        "a\u{0B}b\u{0C}c\rd\u{85}e\u{2028}f\u{2029}g\n".as_bytes(),
        "a\u{A0}b c".as_bytes(),
        // Words long enough that a small window lands inside one.
        b"alpha beta gamma delta epsilon\nzeta eta\n",
        // Two-, three-, and four-byte characters for a window to land inside.
        "\u{E9}\u{4E2D}\u{6587} \u{1F600}word\u{A0}tail\n".as_bytes(),
        // CRLF pairs for a window to land between.
        b"line one\r\nline two\r\n",
        b"  \t\n \n  ",
    ];

    #[test]
    fn counts_chars_over_the_whole_buffer() {
        // Whitespace and newlines are characters too, so this matches `wc -m`.
        let counts = counted("a b\né\n".as_bytes(), Mode::Unicode);
        assert_eq!(counts.chars, 6);
        assert_eq!(counts.bytes, 7);
    }

    #[test]
    fn treats_vertical_tab_as_a_word_separator() {
        // Rust's `is_ascii_whitespace` omits `\v`; C's `isspace` includes it.
        assert_eq!(counted(b"a\x0Bb\x0Cc\rd e\n", Mode::Ascii).words, 5);
        assert_eq!(counted(b"a\x0Bb\x0Cc\rd e\n", Mode::Posix).words, 5);
    }

    #[test]
    fn posix_lines_count_newline_bytes() {
        // `wc -l` counts terminators, so an unterminated final line does not count.
        assert_eq!(counted(b"a\nb", Mode::Posix).lines, 1);
        assert_eq!(counted(b"a\nb", Mode::Ascii).lines, 2);
        assert_eq!(counted(b"a\nb\n", Mode::Posix).lines, 2);
        assert_eq!(counted(b"abc", Mode::Posix).lines, 0);
        assert_eq!(counted(b"", Mode::Posix).lines, 0);
    }

    #[test]
    fn posix_mode_ignores_unicode_newlines() {
        // Seven Unicode terminators: `--utf8` breaks on all of them, `--posix` on none.
        let data = "a\u{0B}b\u{0C}c\rd\u{85}e\u{2028}f\u{2029}g\n".as_bytes();
        assert_eq!(counted(data, Mode::Unicode).lines, 7);
        assert_eq!(counted(data, Mode::Posix).lines, 1);
    }

    #[test]
    fn separates_words_by_unicode_whitespace() {
        // A non-breaking space separates words under `--utf8` but not in ASCII mode.
        let data = "a\u{A0}b c".as_bytes();
        assert_eq!(counted(data, Mode::Unicode).words, 3);
        assert_eq!(counted(data, Mode::Ascii).words, 2);
    }

    #[test]
    fn reports_the_longest_line() {
        assert_eq!(counted(b"ab\nabcd\nabc\n", Mode::Ascii).max_line_length, 4);
        assert_eq!(counted(b"", Mode::Ascii).max_line_length, 0);
    }

    #[test]
    fn skips_the_character_pass_when_unselected() {
        let fields = Fields {
            lines: true,
            words: false,
            bytes: false,
            chars: false,
            max_line_length: false,
        };
        let counter = Counter::new(Mode::Unicode, fields);
        assert_eq!(counter.count_data("é\n".as_bytes()).chars, 0);
    }

    #[test]
    fn maps_selectors_onto_four_plans() {
        assert_eq!(Plan::of(fields_from_bits(0b00100)), Plan::Size);
        assert_eq!(Plan::of(fields_from_bits(0b01000)), Plan::Size);
        assert_eq!(Plan::of(fields_from_bits(0b00001)), Plan::LineExtents);
        assert_eq!(Plan::of(fields_from_bits(0b10000)), Plan::LineExtents);
        assert_eq!(Plan::of(fields_from_bits(0b00010)), Plan::WordsOnly);
        assert_eq!(Plan::of(fields_from_bits(0b00011)), Plan::LinesAndWords);
        assert_eq!(Plan::of(fields_from_bits(0b11111)), Plan::LinesAndWords);

        // Only bytes: the length answers it, so the content stays unread.
        assert!(!fields_from_bits(0b00100).reads_content());
        // Characters ride on `Plan::Size` but still need their own pass.
        assert!(fields_from_bits(0b01100).reads_content());
    }

    #[test]
    fn every_newline_also_separates_words() {
        // `Plan::WordsOnly` skips line iteration, which is sound only while every
        // newline is itself a separator, so no word run can span a line break.
        for mode in [Mode::Posix, Mode::Ascii, Mode::Unicode] {
            assert!(mode.newlines_separate_words());
            assert_eq!(count_words(b"a\nb", mode), 2);
        }
        for newline in [
            '\n', '\u{0B}', '\u{0C}', '\r', '\u{85}', '\u{2028}', '\u{2029}',
        ] {
            let data = format!("a{}b", newline);
            assert_eq!(
                count_words(data.as_bytes(), Mode::Unicode),
                2,
                "U+{:04X} must separate words under --utf8",
                newline as u32
            );
        }
    }

    #[test]
    fn subset_plans_agree_with_the_full_plan() {
        // Every field set lands on one of four plans, and each must report what the
        // all-fields plan reports for the fields it was asked for.
        for mode in [Mode::Posix, Mode::Ascii, Mode::Unicode] {
            for data in EDGE_INPUTS {
                let full = counted(data, mode);
                for bits in 1u8..0b100000 {
                    let fields = fields_from_bits(bits);
                    let subset = Counter::new(mode, fields).count_data(data);
                    let context = format!("bits {:05b} over {:?}", bits, data);
                    assert_eq!(subset.bytes, full.bytes, "bytes, {}", context);
                    if fields.lines {
                        assert_eq!(subset.lines, full.lines, "lines, {}", context);
                    }
                    if fields.words {
                        assert_eq!(subset.words, full.words, "words, {}", context);
                    }
                    if fields.chars {
                        assert_eq!(subset.chars, full.chars, "chars, {}", context);
                    }
                    if fields.max_line_length {
                        assert_eq!(
                            subset.max_line_length, full.max_line_length,
                            "maxline, {}",
                            context
                        );
                    }
                }
            }
        }
    }

    /// Count the same data through a window of `capacity` bytes, the way a pipe is counted.
    fn streamed(data: &[u8], capacity: usize, mode: Mode, fields: Fields) -> Counts {
        Counter::new(mode, fields)
            .count_stream(Refill::new(data, capacity))
            .unwrap()
    }

    #[test]
    fn streams_the_same_counts_as_a_whole_buffer() {
        // Every capacity puts the seam at a different byte, so one sweep covers a window
        // ending inside a word, inside a multi-byte character, and inside a CRLF pair.
        for mode in [Mode::Posix, Mode::Ascii, Mode::Unicode] {
            for data in EDGE_INPUTS {
                for bits in 1u8..0b100000 {
                    let fields = fields_from_bits(bits);
                    let whole = Counter::new(mode, fields).count_data(data);
                    for capacity in 1..=data.len() + 2 {
                        assert_eq!(
                            streamed(data, capacity, mode, fields),
                            whole,
                            "bits {:05b} over {:?} at capacity {}",
                            bits,
                            data,
                            capacity
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn streams_the_same_counts_at_tiny_capacities() {
        // Longer than one window in every mode, so the tallies have to survive many seams.
        let data = "one two\u{A0}three\u{2028}four five\r\nsix\u{1F600}seven eight\n".as_bytes();
        for mode in [Mode::Posix, Mode::Ascii, Mode::Unicode] {
            let fields = fields_from_bits(0b11111);
            let whole = Counter::new(mode, fields).count_data(data);
            for capacity in [7, 13] {
                assert_eq!(
                    streamed(data, capacity, mode, fields),
                    whole,
                    "capacity {}",
                    capacity
                );
            }
        }
    }

    #[test]
    fn resumes_a_word_split_across_windows() {
        // The run "two" straddles the seam, and counting it in both windows would double it.
        for mode in [Mode::Posix, Mode::Ascii, Mode::Unicode] {
            let mut inside_word = false;
            let opening = count_words_resuming(b"one tw", mode, &mut inside_word);
            assert!(inside_word);
            let closing = count_words_resuming(b"o three ", mode, &mut inside_word);
            assert!(!inside_word);
            assert_eq!(opening + closing, 3);
        }
    }

    #[test]
    fn constrains_the_cut_only_where_a_pass_decodes() {
        let cut_for = |bits, mode| Counter::new(mode, fields_from_bits(bits)).cut;
        // Words alone over bytes: nothing decodes, so the window carries nothing.
        assert_eq!(cut_for(0b00010, Mode::Ascii), CutAfter::Anywhere);
        assert_eq!(cut_for(0b00010, Mode::Posix), CutAfter::Anywhere);
        // The Unicode splitter and the character pass both decode.
        assert_eq!(cut_for(0b00010, Mode::Unicode), CutAfter::Characters);
        assert_eq!(cut_for(0b01000, Mode::Ascii), CutAfter::Characters);
        // Anything measuring lines has to see whole ones.
        assert_eq!(cut_for(0b00001, Mode::Ascii), CutAfter::LineFeed);
        assert_eq!(cut_for(0b10000, Mode::Unicode), CutAfter::LineTerminators);
        assert_eq!(cut_for(0b11111, Mode::Posix), CutAfter::LineFeed);
    }

    #[test]
    fn sizes_a_file_without_reading_it() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"a b\nc\n").unwrap();
        file.flush().unwrap();
        let counter = Counter::new(Mode::Ascii, fields_from_bits(0b00100));
        let sized = counter.count_file(file.path()).unwrap();
        assert_eq!(sized.bytes, 6);

        // A zero-length file reports an untrusted size, so it falls through to a read.
        let empty = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(trusted_file_size(empty.path()).unwrap(), None);
        let read = counter.count_file(empty.path()).unwrap();
        assert_eq!(read.bytes, 0);
    }

    #[test]
    fn distrusts_the_size_synthetic_files_report() {
        // `/proc` reports zero bytes and `/sys` a whole page, neither of which is the
        // content length, so both have to reach the read path.
        let counter = Counter::new(Mode::Ascii, fields_from_bits(0b00100));
        for path in ["/proc/version", "/sys/devices/system/cpu/online"] {
            let path = Path::new(path);
            if !path.exists() {
                continue;
            }
            let expected = fs::read(path).unwrap().len();
            assert_eq!(trusted_file_size(path).unwrap(), None, "{}", path.display());
            let measured = counter.count_file(path).unwrap();
            assert_eq!(measured.bytes, expected, "{}", path.display());
        }
    }

    #[test]
    fn drains_a_stream_without_holding_it() {
        // Longer than one window, so the count has to survive across reads.
        let capacity = 4 << 10;
        let data = vec![b'x'; capacity * 2 + 7];
        assert_eq!(
            drained_length(Refill::new(&data[..], capacity)).unwrap(),
            data.len()
        );
        assert_eq!(drained_length(Refill::new(&b""[..], capacity)).unwrap(), 0);
    }

    #[test]
    fn defaults_to_the_wc_field_trio() {
        let args = Args::parse_from(["sz-count", "file.txt"]);
        let fields = Fields::from_selection(&args.fields);
        assert_eq!(fields.selectors(), [true, true, true, false, false]);
    }

    #[test]
    fn utf8_leaves_the_default_fields_alone() {
        // `--utf8` used to append characters silently; they are asked for by name now.
        let args = Args::parse_from(["sz-count", "--utf8", "file.txt"]);
        let fields = Fields::from_selection(&args.fields);
        assert_eq!(fields.selectors(), [true, true, true, false, false]);
    }

    #[test]
    fn reads_one_comma_separated_field_list() {
        let args = Args::parse_from(["sz-count", "--fields", "lines,max-line-length", "file.txt"]);
        let fields = Fields::from_selection(&args.fields);
        assert_eq!(fields.selectors(), [true, false, false, false, true]);
        assert!(Args::try_parse_from(["sz-count", "--fields", "maxline"]).is_err());
    }

    #[test]
    fn rejects_characters_under_posix() {
        // `--posix` promises byte-for-byte `wc`, which never counts code points.
        let args = Args::parse_from(["sz-count", "--posix", "--fields", "chars", "file.txt"]);
        assert!(validate(&args).is_err());
        let args = Args::parse_from(["sz-count", "--posix", "file.txt"]);
        assert!(validate(&args).is_ok());
    }

    #[test]
    fn reports_the_exit_status_of_a_finished_run() {
        let directory = tempfile::tempdir().unwrap();
        let counted = |arguments: &[&str], failures: &mut usize| {
            let args = Args::parse_from(arguments);
            let fields = Fields::from_selection(&args.fields);
            gather(&args, Counter::new(Mode::Ascii, fields), failures)
        };
        let directory_path = directory.path().to_str().unwrap();

        // An empty directory completed and found nothing.
        let mut failures = 0;
        let report = counted(&["sz-count", "--quiet", directory_path], &mut failures);
        assert!(outcome(&report, failures) == ExitCode::NoResult);

        // A readable file found something, however quiet the run.
        let file = directory.path().join("a.txt");
        fs::write(&file, b"a b\n").unwrap();
        let mut failures = 0;
        let report = counted(
            &["sz-count", "--quiet", file.to_str().unwrap()],
            &mut failures,
        );
        assert!(outcome(&report, failures) == ExitCode::Success);

        // A missing input warns and still counts its neighbour, so only the run that
        // read nothing at all is incomplete.
        let mut failures = 0;
        let report = counted(
            &[
                "sz-count",
                "--quiet",
                "no-such-file",
                file.to_str().unwrap(),
            ],
            &mut failures,
        );
        assert_eq!(report.rows.len(), 1);
        assert!(outcome(&report, failures) == ExitCode::Success);

        let mut failures = 0;
        let report = counted(&["sz-count", "--quiet", "no-such-file"], &mut failures);
        assert!(outcome(&report, failures) == ExitCode::Error);
    }

    #[test]
    fn quiet_rejects_the_flags_it_would_ignore() {
        // Under `--quiet` the exit status is the whole output, so nothing that shapes a
        // measurement or a table can reach it.
        let rejected = [
            vec!["sz-count", "--quiet", "--format", "json"],
            vec!["sz-count", "--quiet", "--fields", "lines"],
        ];
        for arguments in rejected {
            assert!(
                Args::try_parse_from(&arguments).is_err(),
                "{:?} must be rejected",
                arguments
            );
        }

        // What the input is, and which files are walked, both still steer the status.
        let accepted = [
            vec!["sz-count", "--quiet", "--utf8"],
            vec!["sz-count", "--quiet", "--posix"],
            vec!["sz-count", "--quiet", "--glob", "*.rs"],
            vec!["sz-count", "--quiet", "--max-depth", "1"],
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
                "fields",
                "utf8",
                "posix",
                "format",
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
        assert!(command
            .get_arguments()
            .all(|argument| argument.get_short().is_none()
                || matches!(argument.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn selects_columns_in_table_order() {
        let counts = Counts {
            lines: 1,
            words: 2,
            bytes: 3,
            chars: 4,
            max_line_length: 5,
        };
        let fields = Fields {
            lines: true,
            words: false,
            bytes: true,
            chars: false,
            max_line_length: true,
        };
        let selected: Vec<_> = counts.columns(fields).collect();
        assert_eq!(selected, vec![("lines", 1), ("bytes", 3), ("maxline", 5)]);
    }

    #[test]
    fn truncates_multibyte_paths_on_character_boundaries() {
        // Byte-slicing here used to panic; the path is 60 non-ASCII characters.
        let path = "é".repeat(60);
        let truncated = truncate_path(&path, 40);
        assert!(truncated.starts_with("..."));
        assert_eq!(truncated.chars().count(), 40);
        assert_eq!(truncate_path("short.txt", 40), "short.txt");
    }

    #[test]
    fn sizes_the_name_column_to_the_names() {
        // Short names sit beside their counts rather than a column away from them.
        assert_eq!(name_width(std::iter::once(("tiny.txt", 0))), 8);
        // A tree entry's branch glyphs count toward the width its row occupies.
        assert_eq!(
            name_width(std::iter::once(("tiny.txt", TREE_GLYPH_WIDTH))),
            11
        );
        // The widest row sets the column, and truncation bounds it at `NAME_WIDTH`.
        let names = [("a.txt", 0), ("nested/directory/b.txt", 0)];
        assert_eq!(name_width(names.into_iter()), 22);
        let long = "é".repeat(60);
        assert_eq!(name_width(std::iter::once((long.as_str(), 0))), NAME_WIDTH);
        // Empty input renders a header alone, so the column collapses rather than pads.
        assert_eq!(name_width(std::iter::empty()), 0);
    }

    #[test]
    fn writes_json_with_untruncated_paths() {
        let counts = Counts {
            lines: 2,
            words: 3,
            bytes: 14,
            chars: 0,
            max_line_length: 7,
        };
        let fields = Fields {
            lines: true,
            words: true,
            bytes: true,
            chars: false,
            max_line_length: false,
        };
        let long_path = format!("{}/file.txt", "nested".repeat(10));
        let mut output = Vec::new();
        write_counts_json(&mut output, &long_path, &counts, fields).unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.starts_with(r#"{"type":"count","#), "{}", text);
        assert!(text.contains(&long_path), "path must not be truncated");
        assert!(text.contains(r#""lines":2,"words":3,"bytes":14}}"#));
        assert!(!text.contains("chars"));
    }
}
