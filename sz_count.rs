//! SIMD-accelerated word count utility
//!
//! A faster replacement for `wc` with proper UTF-8 support and directory traversal.
//! Uses StringZilla for SIMD-accelerated counting operations.
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
//! sz-count -H src/
//!
//! # UTF-8 mode (count characters, Unicode whitespace/newlines)
//! sz-count --utf8 docs/
//!
//! # Just the line count, as a bare integer for scripts
//! sz-count -l file.txt
//!
//! # Match `wc` byte-for-byte
//! sz-count --posix file.txt
//! ```

use std::borrow::Cow;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use ignore::WalkBuilder;
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

    /// Print the line count
    #[arg(short = 'l', long, help_heading = "Fields")]
    lines: bool,

    /// Print the word count
    #[arg(short = 'w', long, help_heading = "Fields")]
    words: bool,

    /// Print the byte count
    #[arg(short = 'c', long, help_heading = "Fields")]
    bytes: bool,

    /// Print the character count (UTF-8 code points)
    #[arg(short = 'm', long, help_heading = "Fields")]
    chars: bool,

    /// Print the length of the longest line, in bytes
    #[arg(short = 'L', long, help_heading = "Fields")]
    max_line_length: bool,

    /// Enable UTF-8 mode (count characters, use Unicode whitespace/newlines)
    #[arg(long, help_heading = "Counting Modes")]
    utf8: bool,

    /// Match `wc` byte-for-byte: newline-terminated lines and the C `isspace` set
    #[arg(long, conflicts_with = "utf8", help_heading = "Counting Modes")]
    posix: bool,

    /// Human-readable output (use K/M/G/T suffixes)
    #[arg(short = 'H', long, help_heading = "Output Formats")]
    human_readable: bool,

    /// Emit JSON Lines with untruncated paths and bare integers
    #[arg(long, conflicts_with_all = ["human_readable", "no_thousands_separator"], help_heading = "Output Formats")]
    json: bool,

    /// Print plain digits in the table, without comma grouping
    #[arg(long, conflicts_with = "human_readable", help_heading = "Output Formats")]
    no_thousands_separator: bool,

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
}

/// C's `isspace` set: space plus the 0x09..=0x0D control run, which is
/// `\t \n \v \f \r`. Rust's `is_ascii_whitespace` omits `\v`.
#[inline]
fn is_posix_space(byte: u8) -> bool {
    byte == b' ' || (0x09..=0x0D).contains(&byte)
}

// endregion: Counting Modes

// region: Counts and Fields

/// How many measurements [`Counts`] carries, and so the most columns a table has.
const FIELD_COUNT: usize = 5;

/// Width of the leading name column in the aligned table.
const NAME_WIDTH: usize = 40;

/// One input's measurements. Every field is always computed cheaply enough to
/// keep this a plain record; [`Fields`] decides which of them reach the output.
#[derive(Debug, Clone, Copy, Default)]
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

    /// The selected measurements in column order, paired with their headers.
    /// Backed by a stack array, so iterating a row allocates nothing.
    fn columns(&self, fields: Fields) -> impl Iterator<Item = (&'static str, usize)> {
        [
            ("lines", self.lines, fields.lines),
            ("words", self.words, fields.words),
            ("bytes", self.bytes, fields.bytes),
            ("chars", self.chars, fields.chars),
            ("maxline", self.max_line_length, fields.max_line_length),
        ]
        .into_iter()
        .filter(|(_, _, selected)| *selected)
        .map(|(header, value, _)| (header, value))
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
    /// Read the selectors, falling back to `wc`'s default trio when none are given.
    /// `--utf8` adds characters to that default, as it always has.
    fn from_args(args: &Args, mode: Mode) -> Self {
        let selected = Self {
            lines: args.lines,
            words: args.words,
            bytes: args.bytes,
            chars: args.chars,
            max_line_length: args.max_line_length,
        };
        if selected.count() > 0 {
            return selected;
        }
        Self {
            lines: true,
            words: true,
            bytes: true,
            chars: mode == Mode::Unicode,
            max_line_length: false,
        }
    }

    /// How many measurements are selected.
    fn count(self) -> usize {
        [
            self.lines,
            self.words,
            self.bytes,
            self.chars,
            self.max_line_length,
        ]
        .iter()
        .filter(|selected| **selected)
        .count()
    }
}

// endregion: Counts and Fields

// region: Counting

/// Count all metrics for given data using StringZilla iterators (single-pass)
fn count_data(data: &[u8], mode: Mode, fields: Fields) -> Counts {
    let mut counts = Counts {
        bytes: data.len(),
        // One pass over the whole buffer, so whitespace and newlines are included.
        // Skipped entirely when no selector asked for characters.
        chars: if fields.chars { sz::count_utf8(data) } else { 0 },
        ..Counts::default()
    };

    for line in LineIter::new(data, mode.newlines()) {
        counts.lines += 1;
        counts.max_line_length = counts.max_line_length.max(line.len());
        counts.words += count_words(line, mode);
    }

    // `LineIter` credits an unterminated final line; `wc -l` counts newline bytes only.
    if !mode.counts_unterminated_line() && counts.lines > 0 && !data.ends_with(b"\n") {
        counts.lines -= 1;
    }

    counts
}

/// Count maximal runs of non-whitespace in one line.
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

/// Count a single file
fn count_file(path: &Path, mode: Mode, fields: Fields) -> io::Result<Counts> {
    let input = open_input(path)?;
    Ok(count_data(input.as_bytes(), mode, fields))
}

/// Count stdin
fn count_stdin(mode: Mode, fields: Fields) -> io::Result<Counts> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    Ok(count_data(&buffer, mode, fields))
}

// endregion: Counting

// region: Rendering

/// Everything the printer needs, decided once from `Args`.
#[derive(Clone, Copy)]
struct RenderConfig {
    fields: Fields,
    human_readable: bool,
    thousands_separator: bool,
    json: bool,
}

impl RenderConfig {
    fn from_args(args: &Args, fields: Fields) -> Self {
        Self {
            fields,
            human_readable: args.human_readable,
            thousands_separator: !args.no_thousands_separator,
            json: args.json,
        }
    }
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
fn format_number<'a>(value: usize, config: &RenderConfig, buffer: &'a mut [u8; 26]) -> Cow<'a, str> {
    if config.human_readable {
        return Cow::Owned(format_human(value));
    }
    if !config.thousands_separator {
        return Cow::Owned(value.to_string());
    }
    Cow::Borrowed(format_grouped_number(buffer, value))
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

/// Write the header row
fn print_header(output: &mut dyn Write, config: &RenderConfig, widths: &[usize]) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = NAME_WIDTH)?;
    for ((header, _), width) in Counts::default().columns(config.fields).zip(widths) {
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
    widths: &[usize],
) -> io::Result<()> {
    let prefix_width = prefix.chars().count();
    let name = truncate_path(name, NAME_WIDTH.saturating_sub(prefix_width));
    output.write_all(prefix.as_bytes())?;
    write!(
        output,
        "{:width$}",
        name,
        width = NAME_WIDTH.saturating_sub(prefix_width)
    )?;
    let mut buffer = [0u8; 26];
    for ((_, value), width) in counts.columns(config.fields).zip(widths) {
        write!(
            output,
            " {:>width$}",
            format_number(value, config, &mut buffer),
            width = width
        )?;
    }
    output.write_all(b"\n")
}

/// Write a tree branch. Depth is bounded by the walk, and only the branch glyph
/// varies, so the indent comes from a static slice rather than a built string.
fn print_tree_line(
    output: &mut dyn Write,
    name: &str,
    counts: &Counts,
    config: &RenderConfig,
    widths: &[usize],
    is_last: bool,
    depth: usize,
) -> io::Result<()> {
    for _ in 0..depth {
        output.write_all(b"   ")?;
    }
    let branch = if is_last { "└─ " } else { "├─ " };
    print_row(output, branch, name, counts, config, widths)
}

/// Write the separator line above the totals
fn print_separator(
    output: &mut dyn Write,
    config: &RenderConfig,
    widths: &[usize],
) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = NAME_WIDTH)?;
    for (_, width) in Counts::default().columns(config.fields).zip(widths) {
        write!(output, " {:─>width$}", "", width = width)?;
    }
    output.write_all(b"\n")
}

/// Write the unnamed totals row
fn print_totals(
    output: &mut dyn Write,
    counts: &Counts,
    config: &RenderConfig,
    widths: &[usize],
) -> io::Result<()> {
    write!(output, "{:>width$}", "", width = NAME_WIDTH)?;
    let mut buffer = [0u8; 26];
    for ((_, value), width) in counts.columns(config.fields).zip(widths) {
        write!(
            output,
            " {:>width$}",
            format_number(value, config, &mut buffer),
            width = width
        )?;
    }
    output.write_all(b"\n")
}

/// Write one `counts` record, with the untruncated path and bare integers.
fn write_counts_json(output: &mut dyn Write, path: &str, counts: &Counts, fields: Fields) -> io::Result<()> {
    output.write_all(br#"{"type":"counts","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    for (header, value) in counts.columns(fields) {
        write!(output, r#","{}":{}"#, header, value)?;
    }
    output.write_all(b"}}\n")
}

/// Write the final `total` record, which also reports how many inputs were read.
fn write_total_json(output: &mut dyn Write, files: usize, counts: &Counts, fields: Fields) -> io::Result<()> {
    write!(output, r#"{{"type":"total","data":{{"files":{}"#, files)?;
    for (header, value) in counts.columns(fields) {
        write!(output, r#","{}":{}"#, header, value)?;
    }
    output.write_all(b"}}\n")
}

/// Print one input's counts, honoring the bare-integer rule for a single selector.
fn report_single(name: &str, counts: &Counts, config: &RenderConfig) -> io::Result<()> {
    let mut output = stdout_writer();
    if config.json {
        return write_counts_json(&mut output, name, counts, config.fields);
    }
    if config.fields.count() == 1 {
        // Exactly one selector and exactly one input: a bare integer, for `n=$(...)`.
        let (_, value) = counts.columns(config.fields).next().unwrap_or(("", 0));
        return writeln!(output, "{}", value);
    }
    let widths = column_widths(std::iter::once(counts), config);
    print_header(&mut output, config, &widths)?;
    print_row(&mut output, "", name, counts, config, &widths)?;
    output.flush()
}

// endregion: Rendering

// region: Input Processing

/// Collect the files a directory holds, honoring the ignore and depth options.
fn walk_files(path: &Path, args: &Args) -> Vec<PathBuf> {
    let mut builder = WalkBuilder::new(path);
    builder
        .hidden(!args.hidden)
        .git_ignore(!args.no_ignore)
        .git_global(!args.no_ignore)
        .git_exclude(!args.no_ignore);
    if let Some(depth) = args.max_depth {
        builder.max_depth(Some(depth));
    }
    builder
        .build()
        .flatten()
        .filter(is_readable_entry)
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

/// Process a directory recursively
fn process_directory(path: &Path, mode: Mode, config: &RenderConfig, args: &Args) -> io::Result<()> {
    let mut entries = Vec::new();
    let mut total = Counts::default();

    // Build walker with ignore support
    let mut builder = WalkBuilder::new(path);
    builder
        .hidden(!args.hidden) // Skip hidden files unless --hidden is set
        .git_ignore(!args.no_ignore) // Respect .gitignore unless --no-ignore is set
        .git_global(!args.no_ignore)
        .git_exclude(!args.no_ignore);

    if let Some(depth) = args.max_depth {
        builder.max_depth(Some(depth));
    }

    for result in builder.build() {
        let entry = result.map_err(io::Error::other)?;
        if !is_readable_entry(&entry) {
            continue;
        }
        let entry_path = entry.path();
        match count_file(entry_path, mode, config.fields) {
            Ok(counts) => {
                total.add(&counts);
                entries.push((entry_path.to_path_buf(), counts));
            }
            Err(error) => {
                eprintln!("Warning: failed to read {}: {}", entry_path.display(), error);
            }
        }
    }

    if entries.is_empty() {
        eprintln!("No files found in {}", path.display());
        return Ok(());
    }

    entries.sort_by(|left, right| left.0.cmp(&right.0));

    let mut output = stdout_writer();
    if config.json {
        for (entry_path, counts) in &entries {
            write_counts_json(&mut output, &entry_path.to_string_lossy(), counts, config.fields)?;
        }
        return write_total_json(&mut output, entries.len(), &total, config.fields);
    }

    let widths = column_widths(
        entries.iter().map(|(_, counts)| counts).chain([&total]),
        config,
    );
    print_header(&mut output, config, &widths)?;

    // Print directory summary first
    let path_text = path.to_string_lossy();
    let directory_display: Cow<'_, str> = if path_text.ends_with('/') {
        path_text
    } else {
        Cow::Owned(format!("{}/", path_text))
    };
    print_row(&mut output, "", &directory_display, &total, config, &widths)?;

    for (index, (entry_path, counts)) in entries.iter().enumerate() {
        let is_last = index == entries.len() - 1;
        let name = entry_path
            .strip_prefix(path)
            .unwrap_or(entry_path)
            .display()
            .to_string();
        print_tree_line(&mut output, &name, counts, config, &widths, is_last, 0)?;
    }

    print_separator(&mut output, config, &widths)?;
    print_totals(&mut output, &total, config, &widths)?;
    output.flush()
}

/// Process multiple files
fn process_multiple_files(paths: &[PathBuf], mode: Mode, config: &RenderConfig) -> io::Result<()> {
    let mut results = Vec::new();
    let mut total = Counts::default();

    for path in paths {
        match count_file(path, mode, config.fields) {
            Ok(counts) => {
                total.add(&counts);
                results.push((path.clone(), counts));
            }
            Err(error) => {
                eprintln!("Warning: failed to read {}: {}", path.display(), error);
            }
        }
    }

    if results.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "No files found"));
    }

    let mut output = stdout_writer();
    if config.json {
        for (path, counts) in &results {
            write_counts_json(&mut output, &path.to_string_lossy(), counts, config.fields)?;
        }
        return write_total_json(&mut output, results.len(), &total, config.fields);
    }

    let widths = column_widths(
        results.iter().map(|(_, counts)| counts).chain([&total]),
        config,
    );
    print_header(&mut output, config, &widths)?;
    for (path, counts) in &results {
        print_row(&mut output, "", &path.to_string_lossy(), counts, config, &widths)?;
    }
    print_separator(&mut output, config, &widths)?;
    print_totals(&mut output, &total, config, &widths)?;
    output.flush()
}

// endregion: Input Processing

fn main() {
    let args = Args::parse();
    let mode = Mode::from_args(args.posix, args.utf8);
    let fields = Fields::from_args(&args, mode);
    let config = RenderConfig::from_args(&args, fields);
    let mut stdout = io::stdout();

    // Handle stdin
    if args.inputs.len() == 1 && args.inputs[0] == "-" {
        match count_stdin(mode, fields) {
            Ok(counts) => {
                if let Err(error) = report_single("-", &counts, &config) {
                    exit_on_write_error(&mut stdout, &error, "Error writing output");
                }
            }
            Err(error) => exit_with_error(&mut stdout, &error, "Error reading stdin"),
        }
        return;
    }

    // Resolve paths
    let mut files = Vec::new();
    let mut directories = Vec::new();

    for input in &args.inputs {
        let path = Path::new(input);
        if !path.exists() {
            eprintln!("Error: {} does not exist", input);
            ExitCode::Error.exit(&mut stdout);
        }

        if path.is_dir() {
            directories.push(path.to_path_buf());
        } else {
            files.push(path.to_path_buf());
        }
    }

    let outcome = if directories.is_empty() && files.len() == 1 {
        match count_file(&files[0], mode, fields) {
            Ok(counts) => report_single(&files[0].display().to_string(), &counts, &config),
            Err(error) => exit_with_error(
                &mut stdout,
                &error,
                &format!("Error reading {}", files[0].display()),
            ),
        }
    } else if directories.len() == 1 && files.is_empty() {
        // A lone directory gets the tree view, with per-file rows under a summary.
        process_directory(&directories[0], mode, &config, &args)
    } else {
        // Anything else is a flat list: directories contribute the files they hold.
        for directory in &directories {
            files.extend(walk_files(directory, &args));
        }
        process_multiple_files(&files, mode, &config)
    };

    if let Err(error) = outcome {
        exit_on_write_error(&mut stdout, &error, "Error");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted(data: &[u8], mode: Mode) -> Counts {
        let fields = Fields {
            lines: true,
            words: true,
            bytes: true,
            chars: true,
            max_line_length: true,
        };
        count_data(data, mode, fields)
    }

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
        assert_eq!(count_data("é\n".as_bytes(), Mode::Unicode, fields).chars, 0);
    }

    #[test]
    fn defaults_to_the_wc_field_trio() {
        let args = Args::parse_from(["sz-count", "file.txt"]);
        let fields = Fields::from_args(&args, Mode::Ascii);
        assert_eq!(fields.count(), 3);
        assert!(fields.lines && fields.words && fields.bytes);
        assert!(!fields.chars && !fields.max_line_length);
    }

    #[test]
    fn utf8_adds_chars_to_the_default_fields() {
        let args = Args::parse_from(["sz-count", "--utf8", "file.txt"]);
        assert_eq!(Fields::from_args(&args, Mode::Unicode).count(), 4);
    }

    #[test]
    fn selectors_replace_the_defaults() {
        let args = Args::parse_from(["sz-count", "-l", "file.txt"]);
        let fields = Fields::from_args(&args, Mode::Ascii);
        assert_eq!(fields.count(), 1);
        assert!(fields.lines && !fields.words && !fields.bytes);
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
        assert!(text.contains(&long_path), "path must not be truncated");
        assert!(text.contains(r#""lines":2,"words":3,"bytes":14}}"#));
        assert!(!text.contains("chars"));
    }
}
