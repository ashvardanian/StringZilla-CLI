//! Literal substring search, standing in for `grep -F` and `rg -F`.
//!
//! Patterns are substrings, never regular expressions, which is what makes a single SIMD pass
//! enough. `--ignore-case` folds through StringZilla's full-Unicode `utf8_uncased_search`, so it
//! answers a wider question than `grep -i` — "Straße" matches `strasse` — and a match can be a
//! different length from the needle.
//!
//! `--fields line-hashes,file-hash` is the read half of the editing handshake: a line's name comes
//! from its own bytes and survives edits elsewhere, and the file's token is what
//! `sz-replace --expect-hash` compares against. The file token is written only after the whole
//! input has been read, so it names what was actually seen.
//!
//! Exit: 0 found something, 1 ran and found nothing, 2 could not run, which includes any
//! named input that could not be read, whatever its readable neighbours produced.

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Range};
use std::path::Path;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{find, rfind, utf8_uncased_search, StringZillableBinary, Utf8UncasedNeedle};

use shared::*;

// region: CLI

/// SIMD-accelerated substring search
#[derive(Parser)]
#[command(name = "sz-find")]
#[command(version, about = "SIMD-accelerated substring search", long_about = None)]
struct Args {
    /// Substring to search for
    pattern: String,

    /// Input files or directories (use '-' or omit for stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// Emit matching lines, matched parts, counts or file names
    #[arg(
        long,
        value_enum,
        conflicts_with = "quiet",
        help_heading = "Output Formats"
    )]
    show: Option<Show>,

    /// Render records as text, JSON Lines or vim-compatible locations
    #[arg(
        long,
        value_enum,
        default_value = "text",
        help_heading = "Output Formats"
    )]
    format: Format,

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

    /// Match the pattern anywhere or only as a whole word
    #[arg(
        long = "match",
        value_name = "MATCH",
        value_enum,
        default_value = "substring",
        help_heading = "Matching"
    )]
    match_kind: Match,

    /// Case-insensitive search, which implies --utf8
    #[arg(long, help_heading = "Matching")]
    ignore_case: bool,

    /// Lines of context before match
    #[arg(long, default_value = "0", help_heading = "Output Formats")]
    before_context: usize,

    /// Lines of context after match
    #[arg(long, default_value = "0", help_heading = "Output Formats")]
    after_context: usize,

    /// Lines of context before and after match, overriding both [default: 0]
    #[arg(long, conflicts_with_all = ["before_context", "after_context"], help_heading = "Output Formats")]
    context: Option<usize>,

    /// Treat the input as UTF-8 text
    #[arg(long, help_heading = "Matching")]
    utf8: bool,

    /// Allow pattern to match across line boundaries
    #[arg(long, conflicts_with = "invert_match", help_heading = "Matching")]
    multiline: bool,

    /// Invert match: show non-matching lines
    #[arg(long, help_heading = "Matching")]
    invert_match: bool,

    /// Stop after NUM matches [default: no limit]
    #[arg(long, value_parser = parse_at_least_one, help_heading = "Matching")]
    max_matches: Option<NonZeroUsize>,

    /// Suppress all output; exit 0 if any match was found, 1 otherwise
    #[arg(
        long,
        conflicts_with_all = [
            "format", "null", "fields", "heading", "color", "max_line_length", "context",
            "before_context", "after_context",
        ],
        help_heading = "Output Formats"
    )]
    quiet: bool,

    /// Report totals for the whole run, on stderr unless `--format json` makes them a record
    #[arg(long, help_heading = "Output Formats")]
    summary: bool,

    /// Colorize output (auto, always, never)
    #[arg(long, default_value = "auto", help_heading = "Output Formats")]
    color: ColorChoice,

    /// Group matches by file with filename header
    #[arg(long, help_heading = "Output Formats")]
    heading: bool,

    /// Trim printed lines to NUM units, on a character boundary [default: no limit]
    #[arg(long, value_parser = parse_at_least_one, help_heading = "Output Formats")]
    max_line_length: Option<NonZeroUsize>,

    /// NUL-terminate each output record instead of newline, for `xargs -0`
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Filter walked files by type (e.g., rust, py, js); named files are always searched
    #[arg(long = "type", help_heading = "Traversal")]
    file_type: Option<Vec<String>>,

    /// Filter walked files by glob (e.g., "*.rs"); named files are always searched
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

    /// Search binary files (don't skip them)
    #[arg(long, help_heading = "Traversal")]
    binary: bool,
}

/// One column a record can carry. `--show` picks which records the run emits; `--fields`
/// picks what each of them says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Field {
    /// The 1-based line number
    LineNumbers,
    /// The 1-based column of the first match on the line
    ColumnNumbers,
    /// The line's byte offset from the start of the input
    ByteOffset,
    /// A hash of the line's content, which survives edits elsewhere in the file
    LineHashes,
    /// A hash of the whole file, as `sz-replace --expect-hash` compares against
    FileHash,
}

impl Field {
    /// The bit this field takes in [`OutputConfig::fields`].
    #[inline]
    fn bit(self) -> u8 {
        match self {
            Field::LineNumbers => 1,
            Field::ColumnNumbers => 2,
            Field::ByteOffset => 4,
            Field::LineHashes => 8,
            Field::FileHash => 16,
        }
    }

    /// The bits of every named field, which is the whole `--fields` set in one word.
    fn bits(fields: &[Field]) -> u8 {
        fields.iter().fold(0, |bits, field| bits | field.bit())
    }
}

/// Which record kind the run emits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Show {
    /// Every selected line
    #[default]
    Lines,
    /// Only the matched parts of selected lines
    Matches,
    /// One count per input
    Count,
    /// The path of every input that matched
    Files,
    /// The path of every input that did not match
    FilesWithout,
}

/// How records are rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Format {
    /// The matching line, with any requested fields ahead of it.
    #[default]
    Text,
    /// JSON Lines, one object per match plus begin and end records.
    Json,
    /// `path:line:column:text`, which editors follow back to the match.
    Vimgrep,
}

/// What the needle is matched against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Match {
    /// A literal substring, matched anywhere in the line.
    #[default]
    Substring,
    /// The same substring, but only where a whole word.
    Word,
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(ErrorKind::ArgumentConflict, message)
}

/// Whether the run names each file once rather than on every record, which is where a
/// whole-file value can go.
fn names_files_once(args: &Args) -> bool {
    args.format == Format::Json || args.heading
}

/// Everything clap cannot express, in one place: its conflicts fire on a flag's presence,
/// never on its value. Every pair rejected here is inert by construction, not merely for
/// some inputs, which is why silence would misreport what the run did.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.pattern.is_empty() {
        return Err(reject("pattern cannot be empty"));
    }

    // `--quiet` stops at the first match, where a cap changes nothing — except under
    // `--summary`, which keeps the tally running and whose totals the cap does bound.
    if args.quiet && args.max_matches.is_some() && !args.summary {
        return Err(reject(
            "--quiet stops at the first match, so it cannot be combined with --max-matches",
        ));
    }

    let show = args.show.unwrap_or_default();
    if args.invert_match && show == Show::Matches {
        return Err(reject(
            "--invert-match selects the lines that hold no match, so --show matches has nothing to print",
        ));
    }

    // A record naming a whole file carries no position, no line and no trimmed text.
    if matches!(show, Show::Count | Show::Files | Show::FilesWithout) {
        let decorations = [
            (!args.fields.is_empty(), "--fields"),
            (args.heading, "--heading"),
            (args.max_line_length.is_some(), "--max-line-length"),
            (
                args.context.is_some() || args.before_context > 0 || args.after_context > 0,
                "--context",
            ),
        ];
        if let Some((_, flag)) = decorations.into_iter().find(|(present, _)| *present) {
            let name = show.to_possible_value().unwrap();
            return Err(reject(format!(
                "--show {} emits one record per file, so it cannot be combined with {}",
                name.get_name(),
                flag
            )));
        }
    }

    // Sizing a hash that is never printed is inert for every value of the flag.
    if args.hash_width.is_some() && !args.fields.contains(&Field::LineHashes) {
        return Err(reject(
            "--hash-width sizes a line hash, so it needs --fields line-hashes",
        ));
    }

    // A multiline match crosses lines, so no single line holds a column to point at, and
    // every record would report column one.
    if args.multiline && args.fields.contains(&Field::ColumnNumbers) {
        return Err(reject(
            "--multiline matches across lines, so --fields column-numbers has no column \
             to resolve",
        ));
    }

    if args.fields.contains(&Field::FileHash) && !names_files_once(args) {
        return Err(reject(
            "--fields file-hash names a whole file, so it needs --format json or --heading",
        ));
    }

    // `--color auto` is the default, so any other value is an explicit request.
    let colored = !matches!(args.color, ColorChoice::Auto);
    // JSON emits `line_number` and `absolute_offset` on every record, so asking for those
    // again is redundant — the two hashes are not, since neither is emitted unasked. Only
    // some values conflict, so the message names the one that did rather than the flag: the
    // documented agent recipe passes `--format json --fields line-hashes,file-hash`, which
    // a message blaming `--fields` reads as forbidden.
    if args.format == Format::Json {
        if let Some(field) = args.fields.iter().find(|field| {
            matches!(
                field,
                Field::LineNumbers | Field::ColumnNumbers | Field::ByteOffset
            )
        }) {
            let value = field.to_possible_value().unwrap();
            return Err(reject(format!(
                "--format json already carries {0}, so --fields {0} is not allowed",
                value.get_name()
            )));
        }
    }
    let carried = match args.format {
        // JSON escapes nothing, names its file in every record and reproduces whole lines.
        Format::Json => [
            (args.null, "--null"),
            (args.heading, "--heading"),
            (colored, "--color"),
            (args.max_line_length.is_some(), "--max-line-length"),
        ]
        .into_iter()
        .find(|(present, _)| *present)
        .map(|(_, flag)| flag),
        // Vimgrep names its file, line and column in every record, and a fifth column would
        // be a different format wearing its name.
        Format::Vimgrep => [
            (args.heading, "--heading"),
            (!args.fields.is_empty(), "--fields"),
            (colored, "--color"),
        ]
        .into_iter()
        .find(|(present, _)| *present)
        .map(|(_, flag)| flag),
        Format::Text => return Ok(()),
    };
    if let Some(flag) = carried {
        let name = args.format.to_possible_value().unwrap();
        return Err(reject(format!(
            "--format {} cannot be combined with {}",
            name.get_name(),
            flag
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
enum ColorChoice {
    #[default]
    Auto,
    Always,
    Never,
}

impl std::str::FromStr for ColorChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            Ok(ColorChoice::Auto)
        } else if s.eq_ignore_ascii_case("always") || s.eq_ignore_ascii_case("yes") {
            Ok(ColorChoice::Always)
        } else if s.eq_ignore_ascii_case("never") || s.eq_ignore_ascii_case("no") {
            Ok(ColorChoice::Never)
        } else {
            Err(format!("Invalid color choice: {}", s))
        }
    }
}

// endregion: CLI

// region: Output Configuration

/// ANSI color codes
#[derive(Clone, Copy)]
struct Colors {
    path: &'static str,
    line_number: &'static str,
    column: &'static str,
    byte_offset: &'static str,
    hash: &'static str,
    match_highlight: &'static str,
    reset: &'static str,
    separator: &'static str,
}

impl Colors {
    fn enabled() -> Self {
        Self {
            path: "\x1b[1;35m",            // Bold Magenta
            line_number: "\x1b[32m",       // Green
            column: "\x1b[32m",            // Green (same as line number)
            byte_offset: "\x1b[36m",       // Cyan
            hash: "\x1b[33m",              // Yellow
            match_highlight: "\x1b[1;31m", // Bold red
            reset: "\x1b[0m",
            separator: "\x1b[36m", // Cyan
        }
    }

    fn disabled() -> Self {
        Self {
            path: "",
            line_number: "",
            column: "",
            byte_offset: "",
            hash: "",
            match_highlight: "",
            reset: "",
            separator: "",
        }
    }
}

/// Output format mode
#[derive(Clone, Copy, PartialEq)]
enum OutputFormat {
    Standard, // Default: file:line:text or file:line:col:text
    Heading,  // Group by file with filename header
    Vimgrep,  // file:line:col:text (every match on separate line)
    Json,     // JSON Lines format (ripgrep-compatible)
}

/// How many inputs the run reads, which is what decides whether a record names its file.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Inputs {
    One,
    Many,
}

/// Everything the line emitters read, resolved once from the flags.
/// Only the printing paths hold one; [`Reporting::Tally`] has no emitter.
#[derive(Clone, Copy)]
struct OutputConfig {
    output_format: OutputFormat,
    colors: Colors,
    /// The `--fields` columns each record carries, packed by [`Field::bits`].
    fields: u8,
    /// How many characters of a line hash are printed.
    hash_width: usize,
    show: Show,
    inputs: Inputs,
    max_line_length: Option<usize>,
    /// What ends every emitted record.
    terminator: u8,
}

impl OutputConfig {
    /// Whether the named `--fields` column is printed.
    #[inline]
    fn carries(&self, field: Field) -> bool {
        self.fields & field.bit() != 0
    }

    /// Whether a record names the file it came from, which a heading writes once instead.
    #[inline]
    fn shows_path(&self) -> bool {
        self.inputs == Inputs::Many && self.output_format != OutputFormat::Heading
    }

    /// Whether a record carries its line number, which `path:line:column:text` does unasked.
    #[inline]
    fn line_numbers(&self) -> bool {
        self.carries(Field::LineNumbers) || self.output_format == OutputFormat::Vimgrep
    }

    /// Whether a record carries the column of its first match.
    #[inline]
    fn column_numbers(&self) -> bool {
        self.carries(Field::ColumnNumbers) || self.output_format == OutputFormat::Vimgrep
    }

    /// Whether a record is reduced to the matched parts of its line.
    #[inline]
    fn only_matches(&self) -> bool {
        self.show == Show::Matches
    }

    /// Whether a printed matching line has its matches wrapped in color codes.
    /// `--show matches`, JSON and vimgrep reproduce the match text themselves.
    #[inline]
    fn highlights(&self) -> bool {
        !self.colors.match_highlight.is_empty()
            && !self.only_matches()
            && !matches!(
                self.output_format,
                OutputFormat::Json | OutputFormat::Vimgrep
            )
    }

    /// Whether a record carries a prefix that has to be resolved per line, which is what
    /// makes `--multiline` print its region line by line rather than as one slice.
    ///
    /// Columns are absent because a multiline region holds none to resolve, which
    /// `validate` refuses outright rather than printing column one for every line.
    #[inline]
    fn annotates_lines(&self) -> bool {
        self.line_numbers() || self.carries(Field::ByteOffset) || self.carries(Field::LineHashes)
    }

    /// Collapse the output flags into the form every emitter reads. `is_terminal` answers
    /// what `--color auto` asks of the destination, which only `main` can see.
    fn new(args: &Args, show: Show, inputs: Inputs, is_terminal: bool) -> Self {
        let output_format = match args.format {
            Format::Json => OutputFormat::Json,
            Format::Vimgrep => OutputFormat::Vimgrep,
            Format::Text if args.heading => OutputFormat::Heading,
            Format::Text => OutputFormat::Standard,
        };

        // JSON carries its own structure, so escape codes would corrupt it. `--quiet` and
        // the tallying selectors need no test here: they never build an `OutputConfig`.
        let use_color = match args.color {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => is_terminal && output_format != OutputFormat::Json,
        };
        let colors = if use_color {
            Colors::enabled()
        } else {
            Colors::disabled()
        };
        Self {
            output_format,
            colors,
            fields: Field::bits(&args.fields),
            hash_width: args.hash_width.unwrap_or(DEFAULT_HASH_WIDTH),
            show,
            inputs,
            max_line_length: args.max_line_length.map(NonZeroUsize::get),
            terminator: Terminator::from_null(args.null).as_byte(),
        }
    }
}

/// What `--summary` reports, summed over the walk, which runs on this thread alone.
#[derive(Default)]
struct Summary {
    files_searched: usize,
    files_matched: usize,
    lines_searched: usize,
    matches_found: usize,
    bytes_searched: usize,
}

// endregion: Output Configuration

// region: Matching

/// Whether the Unicode rules — the newline set, word boundaries and character
/// units — are in force. `--ignore-case` folds by Unicode, so it turns them on too.
#[inline]
fn uses_unicode(utf8: bool, ignore_case: bool) -> bool {
    utf8 || ignore_case
}

/// Check if byte is a word boundary character
#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The character ending at `position`, absent when the bytes before it are not UTF-8.
fn character_before(data: &[u8], position: usize) -> Option<char> {
    let mut start = position.saturating_sub(4);
    while start < position && (data[start] & 0xC0) == 0x80 {
        start += 1;
    }
    std::str::from_utf8(&data[start..position])
        .ok()?
        .chars()
        .next_back()
}

/// The character starting at `position`, absent when the bytes there are not UTF-8.
fn character_at(data: &[u8], position: usize) -> Option<char> {
    let chunk = &data[position..(position + 4).min(data.len())];
    let text = match std::str::from_utf8(chunk) {
        Ok(text) => text,
        Err(error) => std::str::from_utf8(&chunk[..error.valid_up_to()]).ok()?,
    };
    text.chars().next()
}

/// Check if position is at a word boundary
fn is_word_boundary(data: &[u8], start: usize, end: usize, unicode: bool) -> bool {
    let word = |character: char| character.is_alphanumeric() || character == '_';
    let before = match unicode.then(|| character_before(data, start)).flatten() {
        Some(character) => word(character),
        None => start > 0 && is_word_byte(data[start - 1]),
    };
    let after = match unicode.then(|| character_at(data, end)).flatten() {
        Some(character) => word(character),
        None => end < data.len() && is_word_byte(data[end]),
    };
    !before && !after
}

/// The number of characters in `data`, which is its continuation-byte-free byte count.
#[inline]
fn character_count(data: &[u8]) -> usize {
    data.iter().filter(|byte| (*byte & 0xC0) != 0x80).count()
}

/// Whether a budgeted write took a whole run or stopped inside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fit {
    Whole,
    Cut,
}

/// Trim `line` to `limit` characters under [`Newlines::Unicode`], to `limit` bytes otherwise,
/// returning the kept prefix, what it spent of the limit, and whether anything was dropped.
fn trim_to_limit(line: &[u8], limit: usize, newlines: Newlines) -> (&[u8], usize, Fit) {
    if newlines == Newlines::Lf {
        // The unit is the byte here, so the kept prefix is its own count.
        let (kept, trimmed) = truncate_at_character(line, limit);
        let fit = if trimmed { Fit::Cut } else { Fit::Whole };
        return (kept, kept.len(), fit);
    }
    let mut seen = 0;
    for (index, byte) in line.iter().enumerate() {
        if (byte & 0xC0) != 0x80 {
            if seen == limit {
                return (&line[..index], seen, Fit::Cut);
            }
            seen += 1;
        }
    }
    (line, seen, Fit::Whole)
}

/// Information about a single match
#[derive(Clone, Copy)]
struct MatchInfo {
    /// 0-based byte offset where match starts
    offset: usize,
    /// Length of the match in bytes
    length: usize,
}

impl MatchInfo {
    /// 1-based column number, in characters under [`Newlines::Unicode`] and in bytes otherwise.
    #[inline]
    fn column(&self, line: &[u8], newlines: Newlines) -> usize {
        if newlines == Newlines::Unicode {
            character_count(&line[..self.offset]) + 1
        } else {
            self.offset + 1
        }
    }

    /// The matched bytes, borrowed from the line the match was found in.
    #[inline]
    fn text<'a>(&self, line: &'a [u8]) -> &'a [u8] {
        &line[self.offset..self.offset + self.length]
    }
}

/// The pattern in the form its search kernel takes.
///
/// The uncased variant borrows an already analyzed needle, so the metadata is
/// computed once rather than on every call.
#[derive(Clone, Copy)]
enum Needle<'a> {
    Cased(&'a [u8]),
    Uncased(&'a Utf8UncasedNeedle<'a>),
}

impl<'a> Needle<'a> {
    /// Pick the uncased needle when one was built, the raw pattern otherwise.
    #[inline]
    fn new(pattern: &'a [u8], uncased: Option<&'a Utf8UncasedNeedle<'a>>) -> Self {
        match uncased {
            Some(uncased) => Needle::Uncased(uncased),
            None => Needle::Cased(pattern),
        }
    }
}

/// Zero-allocation iterator over pattern matches in data.
///
/// Uses StringZilla's SIMD-accelerated search functions directly.
/// For case-insensitive matching, uses `utf8_uncased_search` which
/// may return matches of different lengths than the pattern.
struct MatchIter<'a> {
    data: &'a [u8],
    needle: Needle<'a>,
    pos: usize,
    whole_word: bool,
    unicode: bool,
}

impl<'a> MatchIter<'a> {
    #[inline]
    fn new(data: &'a [u8], needle: Needle<'a>, whole_word: bool, unicode: bool) -> Self {
        Self {
            data,
            needle,
            pos: 0,
            whole_word,
            unicode,
        }
    }
}

impl<'a> Iterator for MatchIter<'a> {
    type Item = MatchInfo;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.data.len() {
            let remaining = &self.data[self.pos..];

            let (offset, len) = match self.needle {
                Needle::Uncased(needle) => utf8_uncased_search(remaining, needle)?,
                Needle::Cased(pattern) => (find(remaining, pattern)?, pattern.len()),
            };

            let abs_start = self.pos + offset;
            let abs_end = abs_start + len;

            // Check word boundary if needed
            if self.whole_word && !is_word_boundary(self.data, abs_start, abs_end, self.unicode) {
                self.pos = abs_start + 1;
                continue;
            }

            self.pos = abs_end;
            return Some(MatchInfo {
                offset: abs_start,
                length: len,
            });
        }
        None
    }
}

/// The needle plus the boundary rule every search call shares.
struct Matcher<'a> {
    pattern: &'a [u8],
    /// Present only under `--ignore-case`, holding the analysis every search shares.
    uncased_needle: Option<Utf8UncasedNeedle<'a>>,
    whole_word: bool,
    unicode: bool,
}

impl Matcher<'_> {
    /// The needle in the form the kernels take, borrowing the cached analysis.
    #[inline]
    fn needle(&self) -> Needle<'_> {
        Needle::new(self.pattern, self.uncased_needle.as_ref())
    }

    /// Iterate every match within `data`.
    #[inline]
    fn matches<'d>(&'d self, data: &'d [u8]) -> MatchIter<'d> {
        MatchIter::new(data, self.needle(), self.whole_word, self.unicode)
    }
}

/// Whether anything reads how many lines a run walked over, which only a whole-buffer
/// search pays a pass over its input to know.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LineCounting {
    Ignored,
    Reported,
}

/// Whether a line is in the result, and the first match a column points at when one exists.
#[derive(Clone, Copy)]
enum Selection {
    Rejected,
    /// Absent under `--invert-match`, which selects the lines that hold no match.
    Selected(Option<MatchInfo>),
}

/// What every search reporting reads: the needle, the line split, the selection rules, and
/// whether the lines walked over are counted.
struct Search<'a> {
    matcher: Matcher<'a>,
    newlines: Newlines,
    multiline: bool,
    invert_match: bool,
    max_matches: Option<usize>,
    line_counting: LineCounting,
}

impl Search<'_> {
    /// Select the line, resolving its first match once for the column and the highlight.
    #[inline]
    fn select(&self, line: &[u8]) -> Selection {
        match self.matcher.matches(line).next() {
            Some(found) if !self.invert_match => Selection::Selected(Some(found)),
            None if self.invert_match => Selection::Selected(None),
            _ => Selection::Rejected,
        }
    }
}

// endregion: Matching

// region: Search Paths

/// Lines of context requested on each side of a match.
#[derive(Clone, Copy)]
struct Context {
    before: usize,
    after: usize,
}

impl Context {
    /// No surrounding lines, which is what every reporting but `PrintContext` carries.
    #[inline]
    fn none() -> Self {
        Context {
            before: 0,
            after: 0,
        }
    }
}

/// What the per-line loop must remember between lines, decided once from the flags.
///
/// The split axis is the carried state, not the flag: output format, `--ignore-case`,
/// `--invert-match` and `--match word` all change what an emitter writes, not what
/// the loop holds.
#[derive(Clone, Copy)]
enum Reporting {
    /// Two counters. Serves `--quiet` and the three record kinds that name no line.
    Tally { stop_at_first: bool },
    /// Counters plus a heading bit. Serves every format without context lines.
    Print(OutputConfig),
    /// Counters, heading, a look-behind ring and group separators. Serves the context flags.
    PrintContext(OutputConfig, Context),
}

/// Where a file's hash is printed, which is what decides whether it is computed at all.
#[derive(Clone, Copy, PartialEq, Eq)]
enum HashPlacement {
    Absent,
    /// Beside the path in a heading, over the bytes the heading names.
    Heading,
    /// On the JSON record that closes the file, once every window is read.
    ClosingRecord,
}

impl Reporting {
    /// Pick the reporting: existence only, plain printing, or printing with context.
    fn choose(
        args: &Args,
        show: Show,
        context: Context,
        inputs: Inputs,
        is_terminal: bool,
    ) -> Self {
        // `--show count` must see every line. Existence answers need only one hit — except
        // under `--summary`, whose totals would otherwise describe a prefix of the file.
        let tally = |stop_at_first: bool| Reporting::Tally {
            stop_at_first: stop_at_first && !args.summary,
        };
        if args.quiet {
            return tally(true);
        }
        match show {
            Show::Count => tally(false),
            Show::Files | Show::FilesWithout => tally(true),
            Show::Lines | Show::Matches => {
                let config = OutputConfig::new(args, show, inputs, is_terminal);
                if context.before == 0 && context.after == 0 {
                    Reporting::Print(config)
                } else {
                    Reporting::PrintContext(config, context)
                }
            }
        }
    }

    /// The emitter this reporting prints through, absent for the tally.
    #[inline]
    fn config(&self) -> Option<&OutputConfig> {
        match self {
            Reporting::Tally { .. } => None,
            Reporting::Print(config) | Reporting::PrintContext(config, _) => Some(config),
        }
    }

    /// Where this run prints a file's hash, which is what decides whether it is computed at
    /// all: a file that matched nothing never pays for a token nothing names.
    #[inline]
    fn hash_placement(&self) -> HashPlacement {
        match self.config() {
            Some(config) if config.carries(Field::FileHash) => match config.output_format {
                OutputFormat::Heading => HashPlacement::Heading,
                OutputFormat::Json => HashPlacement::ClosingRecord,
                OutputFormat::Standard | OutputFormat::Vimgrep => HashPlacement::Absent,
            },
            _ => HashPlacement::Absent,
        }
    }

    /// The surrounding lines this reporting prints, none for the two that print none.
    #[inline]
    fn context(&self) -> Context {
        match self {
            Reporting::PrintContext(_, context) => *context,
            Reporting::Tally { .. } | Reporting::Print(_) => Context::none(),
        }
    }
}

// endregion: Search Paths

// region: Line Loops

/// Why a search stopped with input still unread.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stop {
    /// `--max-matches` is satisfied, which ends the walk over the remaining inputs too.
    MaxMatches,
    /// Only existence was asked for, and the first match answered it.
    FirstMatch,
}

/// What one file contributed to the run, and where its search stopped.
struct FileResult {
    match_count: usize,
    lines_searched: usize,
    bytes_searched: usize,
    flow: ControlFlow<Stop>,
}

impl FileResult {
    /// Whether `--max-matches` ended this file, and with it the whole run.
    #[inline]
    fn capped(&self) -> bool {
        self.flow == ControlFlow::Break(Stop::MaxMatches)
    }
}

/// A line held for look-behind: its number, and its span within the current window,
/// which the whole-slice paths open at byte zero.
struct ContextLine {
    line_number: usize,
    span: Range<usize>,
    /// Where the line's terminator ends, so a held line can still be named.
    whole_end: usize,
}

/// The look-behind ring, pending after-context and group separator that the context
/// flags carry from one line to the next.
struct ContextState {
    lines: Context,
    /// Window offsets rather than copies, and `lines.before` entries is the exact ceiling.
    /// Empty under `--after-context` alone, which is what lets that reporting stream.
    behind: VecDeque<ContextLine>,
    pending_after: usize,
    last_printed_line: Option<usize>,
}

impl ContextState {
    /// A ring sized to the look-behind actually requested, allocating nothing at zero.
    fn new(lines: Context) -> Self {
        ContextState {
            lines,
            behind: VecDeque::with_capacity(lines.before),
            pending_after: 0,
            last_printed_line: None,
        }
    }

    /// Whether the line still has to be printed, given how far the output has reached.
    #[inline]
    fn is_unprinted(&self, line_number: usize) -> bool {
        self.last_printed_line.is_none_or(|last| line_number > last)
    }

    /// The first look-behind line a group opening here would print, absent when the ring
    /// holds none still unprinted. Entries run forward, so the first is the oldest.
    fn first_unprinted_behind(&self) -> Option<usize> {
        self.behind
            .iter()
            .map(|buffered| buffered.line_number)
            .find(|line_number| self.is_unprinted(*line_number))
    }
}

/// What a search reporting remembers between lines, and between windows once the input
/// streams: the counters, the once-per-file heading bit and the context ring.
struct FindState {
    /// How many lines of this file were selected.
    match_count: usize,
    /// How many lines of this file were walked over, selected or not.
    lines_searched: usize,
    /// Offset of the current window's first byte within the whole input, which keeps
    /// `--fields byte-offset` and the JSON and vimgrep formats absolute across a streamed run.
    window_base: usize,
    /// Whether the once-per-file header the format calls for has been written.
    printed_heading: bool,
    /// The look-behind ring and separators the context flags carry between lines.
    context: ContextState,
}

impl FindState {
    /// Start a file at line zero, byte zero, with nothing printed yet.
    fn new(lines: Context) -> Self {
        FindState {
            match_count: 0,
            lines_searched: 0,
            window_base: 0,
            printed_heading: false,
            context: ContextState::new(lines),
        }
    }

    /// Whether `--max-matches` is already satisfied, where every reporting stops reporting.
    #[inline]
    fn exhausted(&self, search: &Search) -> bool {
        search
            .max_matches
            .is_some_and(|max| self.match_count >= max)
    }

    /// The totals this file contributes, over `bytes_searched` bytes of input.
    fn result(&self, bytes_searched: usize, flow: ControlFlow<Stop>) -> FileResult {
        FileResult {
            match_count: self.match_count,
            lines_searched: self.lines_searched,
            bytes_searched,
            flow,
        }
    }
}

/// Run one window through the chosen reporting, folding its lines into `state`.
/// Breaks once the reporting has reported everything it will, which ends a streamed read.
fn search_window(
    window: &[u8],
    path: &str,
    search: &Search,
    reporting: &Reporting,
    state: &mut FindState,
    output: &mut dyn Write,
    heading_hash: Option<&[u8]>,
) -> io::Result<ControlFlow<Stop>> {
    match reporting {
        Reporting::Tally { stop_at_first } => {
            Ok(tally_window(window, search, *stop_at_first, state))
        }
        Reporting::Print(config) => {
            print_window(window, path, search, config, state, output, heading_hash)
        }
        Reporting::PrintContext(config, _) => {
            print_context_window(window, path, search, config, state, output, heading_hash)
        }
    }
}

/// Count matching lines, stopping at the first when only existence is reported.
/// Carries two counters, so it allocates nothing and resolves no offsets.
fn tally_window(
    window: &[u8],
    search: &Search,
    stop_at_first: bool,
    state: &mut FindState,
) -> ControlFlow<Stop> {
    // Counting reads nothing a line's terminator could tell it, so it takes the cheaper
    // iterator: `named_lines` looks one line ahead to find where each one ends.
    for line in LineIter::new(window, search.newlines) {
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        state.lines_searched += 1;
        if state.exhausted(search) {
            return ControlFlow::Break(Stop::MaxMatches);
        }

        if matches!(search.select(line), Selection::Selected(_)) {
            state.match_count += 1;
            if stop_at_first {
                return ControlFlow::Break(Stop::FirstMatch);
            }
        }
    }
    ControlFlow::Continue(())
}

/// Print matching lines, carrying counters and the once-per-file heading bit.
fn print_window(
    window: &[u8],
    path: &str,
    search: &Search,
    config: &OutputConfig,
    state: &mut FindState,
    output: &mut dyn Write,
    heading_hash: Option<&[u8]>,
) -> io::Result<ControlFlow<Stop>> {
    let mut emitter = Emitter {
        output,
        path,
        search,
        config,
        heading_hash,
    };
    for named in named_lines(window, search.newlines) {
        let line = named.as_cut;
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        state.lines_searched += 1;
        if state.exhausted(search) {
            return Ok(ControlFlow::Break(Stop::MaxMatches));
        }

        let Selection::Selected(first_match) = search.select(line) else {
            continue;
        };
        state.match_count += 1;

        let record = LineRecord {
            as_cut: line,
            whole: named.whole,
            line_number: state.lines_searched,
            byte_offset: state.window_base + named.offset,
            is_match: true,
            first_match,
        };
        emitter.file_heading(&mut state.printed_heading)?;
        emitter.match_line(record)?;
    }
    Ok(ControlFlow::Continue(()))
}

/// Print matching lines with their surrounding context, carrying a look-behind ring
/// of window offsets, the pending after-context count, and separators.
fn print_context_window(
    window: &[u8],
    path: &str,
    search: &Search,
    config: &OutputConfig,
    state: &mut FindState,
    output: &mut dyn Write,
    heading_hash: Option<&[u8]>,
) -> io::Result<ControlFlow<Stop>> {
    let mut emitter = Emitter {
        output,
        path,
        search,
        config,
        heading_hash,
    };
    for named in named_lines(window, search.newlines) {
        let line = named.as_cut;
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        state.lines_searched += 1;
        if state.exhausted(search) {
            return Ok(ControlFlow::Break(Stop::MaxMatches));
        }

        let line_number = state.lines_searched;
        let line_start = named.offset;
        let byte_offset = state.window_base + line_start;

        if let Selection::Selected(first_match) = search.select(line) {
            state.match_count += 1;
            emitter.file_heading(&mut state.printed_heading)?;

            // The group opens at its own first line — the oldest look-behind line still
            // unprinted, or the matching line when the ring holds none — and a gap
            // between that line and the last printed one divides the two with `--`,
            // which is where `grep -C` writes its divider.
            let group_start = state
                .context
                .first_unprinted_behind()
                .unwrap_or(line_number);
            let follows_a_gap = state
                .context
                .last_printed_line
                .is_some_and(|last| group_start > last + 1);
            if follows_a_gap {
                emitter.group_separator()?;
            }

            // Print the buffered look-behind lines, sliced from the current window.
            for buffered in state.context.behind.iter() {
                if state.context.is_unprinted(buffered.line_number) {
                    emitter.line(LineRecord {
                        as_cut: &window[buffered.span.start..buffered.span.end],
                        whole: &window[buffered.span.start..buffered.whole_end],
                        line_number: buffered.line_number,
                        byte_offset: state.window_base + buffered.span.start,
                        is_match: false,
                        first_match: None,
                    })?;
                    state.context.last_printed_line = Some(buffered.line_number);
                }
            }

            emitter.match_line(LineRecord {
                as_cut: line,
                whole: named.whole,
                line_number,
                byte_offset,
                is_match: true,
                first_match,
            })?;
            state.context.last_printed_line = Some(line_number);
            state.context.pending_after = state.context.lines.after;
        } else if state.context.pending_after > 0 {
            emitter.line(LineRecord {
                as_cut: line,
                whole: named.whole,
                line_number,
                byte_offset,
                is_match: false,
                first_match: None,
            })?;
            state.context.last_printed_line = Some(line_number);
            state.context.pending_after -= 1;
        }

        // Maintain the rolling look-behind ring.
        if state.context.lines.before > 0 {
            if state.context.behind.len() >= state.context.lines.before {
                state.context.behind.pop_front();
            }
            state.context.behind.push_back(ContextLine {
                line_number,
                span: line_start..line_start + line.len(),
                whole_end: line_start + named.whole.len(),
            });
        }
    }
    Ok(ControlFlow::Continue(()))
}

/// A 1-based line number for positions visited in increasing order.
///
/// Each answer costs one scan of the gap since the previous position, so a whole-buffer
/// search stays linear in its input. Counting from byte zero per match makes it quadratic.
#[derive(Default)]
struct LineCursor {
    counted_through: usize,
    newlines_before: usize,
}

impl LineCursor {
    /// The line number at `position`, which must not precede the previous answer.
    fn line_at(&mut self, data: &[u8], position: usize) -> usize {
        debug_assert!(
            position >= self.counted_through,
            "positions must move forward"
        );
        self.newlines_before += data[self.counted_through..position]
            .sz_matches(b"\n")
            .count();
        self.counted_through = position;
        self.newlines_before + 1
    }
}

/// The span a match prints: the line holding it, widened by the surrounding lines `context`
/// asks for. Ends at the newline terminating the last line, which the caller carries.
fn printed_region(
    data: &[u8],
    match_start: usize,
    match_end: usize,
    context: Context,
) -> (usize, usize) {
    let line_start = rfind(&data[..match_start], b"\n").map_or(0, |offset| offset + 1);
    let line_end = find(&data[match_end..], b"\n").map_or(data.len(), |offset| match_end + offset);

    let mut start = line_start;
    for _ in 0..context.before {
        if start == 0 {
            break;
        }
        start = rfind(&data[..start - 1], b"\n").map_or(0, |offset| offset + 1);
    }

    let mut end = line_end;
    for _ in 0..context.after {
        if end >= data.len() {
            break;
        }
        match find(&data[end + 1..], b"\n") {
            Some(offset) => end += 1 + offset,
            None => {
                end = data.len();
                break;
            }
        }
    }
    (start, end)
}

/// Search using multi-line matching (whole buffer search)
fn search_multiline(
    data: &[u8],
    path: &str,
    search: &Search,
    reporting: &Reporting,
    output: &mut dyn Write,
) -> io::Result<FileResult> {
    // The emitter, the tally's early stop and the context width hold for the whole file.
    let stop_at_first = matches!(
        reporting,
        Reporting::Tally {
            stop_at_first: true
        }
    );
    let context = reporting.context();
    // Groups are divided only where surrounding lines are printed, as in line mode.
    let separates_groups = context.before > 0 || context.after > 0;
    let placement = reporting.hash_placement();
    // Reborrowed rather than moved, so the sink is still reachable for the closing
    // JSON record once every region is printed.
    let mut emitter = reporting.config().map(|config| Emitter {
        output: &mut *output,
        path,
        search,
        config,
        heading_hash: (placement == HashPlacement::Heading).then_some(data),
    });

    // Each region carries its own surrounding lines, so this reporting keeps no look-behind
    // ring, and the whole file is one window that counts its lines up front.
    let mut state = FindState::new(Context::none());
    if search.line_counting == LineCounting::Reported {
        state.lines_searched = data.sz_matches(b"\n").count() + 1;
    }
    let mut scanned_through = 0;
    // One past the terminator of the last printed region, absent until one is printed.
    let mut last_printed_end: Option<usize> = None;
    // Regions arrive in increasing order, which is what lets the line number carry.
    let mut line_numbers = LineCursor::default();

    for found in search.matcher.matches(data) {
        if state.exhausted(search) {
            break;
        }
        state.match_count += 1;
        let match_end = found.offset + found.length;
        scanned_through = match_end;

        let Some(emitter) = emitter.as_mut() else {
            if stop_at_first {
                break;
            }
            continue;
        };
        // The same once-per-file header line mode writes, so `--heading` names the file
        // and `--format json` opens the record a `{"type":"end"}` will close.
        emitter.file_heading(&mut state.printed_heading)?;

        let (output_start, output_end) = printed_region(data, found.offset, match_end, context);

        // Avoid printing overlapping regions
        let actual_start = output_start.max(last_printed_end.unwrap_or(0));
        // `output_end` addresses the newline terminating the region's last line, so the
        // region reaches one past it and carries that terminator. A region of exactly one
        // newline is a blank line, which is context like any other.
        let region_end = (output_end + 1).min(data.len());
        if actual_start < region_end {
            // A region that does not abut the previous one opens a group, and `grep -C`
            // divides groups with `--`.
            let follows_a_gap = last_printed_end.is_some_and(|end| actual_start > end);
            if separates_groups && follows_a_gap {
                emitter.group_separator()?;
            }

            let region = &data[actual_start..region_end];
            if emitter.config.annotates_lines()
                || emitter.config.output_format != OutputFormat::Standard
            {
                let line_number = line_numbers.line_at(data, actual_start);
                for (index, named) in named_lines(region, Newlines::Lf).enumerate() {
                    emitter.line(LineRecord {
                        as_cut: named.as_cut,
                        whole: named.whole,
                        line_number: line_number + index,
                        byte_offset: actual_start + named.offset,
                        is_match: true,
                        // A multiline region holds no column to point at, which `validate`
                        // refuses to ask for.
                        first_match: None,
                    })?;
                }
            } else {
                emitter.region(region)?;
            }
            last_printed_end = Some(region_end);
        }
    }

    let closes_the_file = state.printed_heading && placement == HashPlacement::ClosingRecord;
    let file_hash = closes_the_file.then(|| content_hash(data));
    print_json_end(output, path, reporting, &state, file_hash)?;

    // `--max-matches` caps this file, and reaching the cap ends the walk over the rest —
    // but only with input still unread, which is where the line paths stop as well.
    let flow = if state.exhausted(search) && scanned_through < data.len() {
        ControlFlow::Break(Stop::MaxMatches)
    } else {
        ControlFlow::Continue(())
    };
    Ok(state.result(data.len(), flow))
}

// endregion: Line Loops

// region: Streaming

/// Whether a true pipe can be searched through a bounded window rather than drained whole.
///
/// `--multiline` searches the whole input backward, and the `--before-context` ring holds
/// window-relative spans that dangle once the window moves, so those two keep every byte.
fn can_stream(search: &Search, reporting: &Reporting) -> bool {
    !search.multiline
        // A heading is written on the first match, long before a stream reaches the bytes a
        // file's hash covers. JSON names the file again when it closes it, which a stream
        // does reach, so only the heading has to hold the input.
        && reporting.hash_placement() != HashPlacement::Heading
        && match reporting {
            Reporting::Tally { .. } | Reporting::Print(_) => true,
            Reporting::PrintContext(_, context) => context.before == 0,
        }
}

/// One input's windows, however they arrive.
///
/// A run walks [`Windows`], which hands a mapped file over whole and a pipe over in pieces.
/// The tests walk a [`Refill`] instead, since choosing its capacity is the only way to put a
/// seam at a chosen byte.
trait Windowing {
    /// Every byte at once, where the input can give them at once.
    fn whole(&self) -> Option<&[u8]>;
    /// The next window, once `consumed` bytes of the previous one are finished with.
    fn window(&mut self, cut: CutAfter, consumed: usize) -> io::Result<Option<&[u8]>>;
    /// What a streamed walk hashed, once it has been read to its end.
    fn digest(&self) -> Option<u64>;
}

impl Windowing for Windows {
    fn whole(&self) -> Option<&[u8]> {
        Windows::whole(self)
    }

    fn window(&mut self, cut: CutAfter, consumed: usize) -> io::Result<Option<&[u8]>> {
        Ok(self.next(cut, consumed)?.map(|(window, _)| window))
    }

    fn digest(&self) -> Option<u64> {
        Windows::digest(self)
    }
}

/// Search one input window by window, folding every line into one file's totals.
///
/// A whole input yields a single window and a pipe as many as it takes, so the loop is all
/// that separates the two. A match cannot cross a newline here — the paths search within
/// each line — so cutting at a line ending needs no carry, not even under `--ignore-case`.
fn search_windows<W: Windowing>(
    walk: &mut W,
    path: &str,
    search: &Search,
    reporting: &Reporting,
    output: &mut dyn Write,
) -> io::Result<FileResult> {
    if search.multiline {
        // A multiline match does cross a newline, which is why `can_stream` keeps it whole.
        let data = walk
            .whole()
            .expect("a multiline search reads its input whole");
        return search_multiline(data, path, search, reporting, output);
    }

    let placement = reporting.hash_placement();
    // A whole input names itself from its own bytes; a stream from what it hashed as it read,
    // which is why the walk was told to hash before it was asked for a window.
    let arrives_whole = walk.whole().is_some();
    // Only an input that arrived whole holds the bytes a heading names inline.
    let heading_hash = placement == HashPlacement::Heading && arrives_whole;
    let mut state = FindState::new(reporting.context());

    // A run that asked for the file's hash keeps *reading* after the search has what it
    // needs, since a digest of a prefix looks exactly like a digest of the file — but it
    // stops *searching*, or the totals would count one line per remaining window.
    let hashing = placement != HashPlacement::Absent && !arrives_whole;
    let cut: CutAfter = search.newlines.into();
    let mut flow = ControlFlow::Continue(());
    let mut consumed = 0;
    while let Some(window) = walk.window(cut, consumed)? {
        if flow.is_continue() {
            // An input that arrives whole is one window, so the window is what a heading names.
            flow = search_window(
                window,
                path,
                search,
                reporting,
                &mut state,
                output,
                heading_hash.then_some(window),
            )?;
        }
        // Counted before the flow is honoured, so a stop still reports the window it read.
        consumed = window.len();
        state.window_base += consumed;
        if flow.is_break() && !hashing {
            break;
        }
    }

    let closes_the_file = state.printed_heading && placement == HashPlacement::ClosingRecord;
    let file_hash = closes_the_file
        .then(|| walk.whole().map(content_hash).or_else(|| walk.digest()))
        .flatten();
    print_json_end(output, path, reporting, &state, file_hash)?;
    // An early stop leaves the rest of a pipe unread, so the total describes what was
    // searched rather than what the writer still holds.
    Ok(state.result(state.window_base, flow))
}

// endregion: Streaming

// region: Line Output

/// Whether a printed line has its matches wrapped in the highlight color.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Coloring {
    Plain,
    Matches,
}

/// One line as an emitter sees it: its bytes, where it sits, and how it was selected.
#[derive(Clone, Copy)]
struct LineRecord<'a> {
    /// The line as the newline set cut it, which is what the record prints and matches
    /// against. Same slice, same word as [`NamedLine::as_cut`].
    as_cut: &'a [u8],
    /// The line and its terminator, which is what names it.
    whole: &'a [u8],
    line_number: usize,
    byte_offset: usize,
    /// False for a surrounding context line, which prints `-` where a match prints `:`.
    is_match: bool,
    /// The match a column points at, found once where the line was selected.
    first_match: Option<MatchInfo>,
}

/// Everything a printed line reads beyond the line itself: the sink, the file it came
/// from, the pattern its matches are resolved against, and the resolved output flags.
struct Emitter<'a, 'p> {
    output: &'a mut dyn Write,
    path: &'a str,
    search: &'a Search<'p>,
    config: &'a OutputConfig,
    /// The bytes a heading names beside the path, hashed on the first record that prints one.
    heading_hash: Option<&'a [u8]>,
}

impl Emitter<'_, '_> {
    /// Whether the record holds match positions to point at. `--invert-match` selects the
    /// lines that hold none, leaving columns and `--show matches` nothing to resolve.
    #[inline]
    fn positions_in(&self, record: LineRecord) -> bool {
        record.is_match && !self.search.invert_match
    }

    /// Emit the once-per-file header the format calls for, on its first printed match.
    fn file_heading(&mut self, printed: &mut bool) -> io::Result<()> {
        if *printed {
            return Ok(());
        }
        let (path, config, heading_hash) = (self.path, self.config, self.heading_hash);
        match config.output_format {
            OutputFormat::Heading => {
                write!(
                    self.output,
                    "{}{}{}",
                    config.colors.path, path, config.colors.reset
                )?;
                // Beside the path, which is the only place a whole-file value belongs in a
                // stream of per-line records.
                if let Some(data) = heading_hash {
                    // Two spaces rather than `:`, which would read as a `file:line` prefix,
                    // and outside the colour span as every other column's separator is.
                    let mut buffer = [0u8; HASH_CHARS];
                    write!(
                        self.output,
                        "  {}{}{}",
                        config.colors.hash,
                        format_hash(&mut buffer, content_hash(data), HASH_CHARS),
                        config.colors.reset
                    )?;
                }
                self.output.write_all(b"\n")?;
                *printed = true;
            }
            // JSON names the file again when it closes it, and that is where the file's
            // hash goes: an opening record is written long before a stream reaches the bytes
            // it would cover.
            OutputFormat::Json => {
                self.output
                    .write_all(br#"{"type":"begin","data":{"path":"#)?;
                json_text_field_to(self.output, path.as_bytes())?;
                self.output.write_all(b"}}\n")?;
                *printed = true;
            }
            OutputFormat::Standard | OutputFormat::Vimgrep => {}
        }
        Ok(())
    }

    /// The `--` divider `grep -C` writes between groups of lines that do not adjoin.
    /// JSON Lines has no such record, and every line it emits carries its own number.
    fn group_separator(&mut self) -> io::Result<()> {
        let config = self.config;
        if config.output_format == OutputFormat::Json {
            return Ok(());
        }
        writeln!(
            self.output,
            "{}--{}",
            config.colors.separator, config.colors.reset
        )
    }

    /// Print a matching line, coloring its matches where the format allows.
    fn match_line(&mut self, record: LineRecord) -> io::Result<()> {
        // The color decision travels beside the record rather than inside it: the column
        // resolves against the line as it was found, and an escape code ahead of a match
        // would shift that column by its own length.
        let coloring = if self.config.highlights() && self.positions_in(record) {
            Coloring::Matches
        } else {
            Coloring::Plain
        };
        self.write_line(record, coloring)
    }

    /// Print one line with the prefixes its format asks for.
    fn line(&mut self, record: LineRecord) -> io::Result<()> {
        self.write_line(record, Coloring::Plain)
    }

    /// Print the line behind the prefixes the format asks for, resolving every position
    /// against `record.as_cut`.
    fn write_line(&mut self, record: LineRecord, coloring: Coloring) -> io::Result<()> {
        let (path, search, config) = (self.path, self.search, self.config);
        let positioned = self.positions_in(record);

        match config.output_format {
            OutputFormat::Json => self.json_line(record),
            OutputFormat::Vimgrep => {
                // Vimgrep puts every match on its own `file:line:column:text` line.
                if positioned {
                    for found in search.matcher.matches(record.as_cut) {
                        write!(
                            self.output,
                            "{}:{}:{}:",
                            path,
                            record.line_number,
                            found.column(record.as_cut, search.newlines)
                        )?;
                        if config.only_matches() {
                            self.output.write_all(found.text(record.as_cut))?;
                        } else {
                            self.write_body(record.as_cut, coloring)?;
                        }
                        self.output.write_all(&[config.terminator])?;
                    }
                } else if record.is_match {
                    // An inverted match holds no position, so the line prints at column one.
                    write!(self.output, "{}:{}:1:", path, record.line_number)?;
                    self.write_body(record.as_cut, coloring)?;
                    self.output.write_all(&[config.terminator])?;
                }
                Ok(())
            }
            OutputFormat::Standard | OutputFormat::Heading => {
                self.write_prefix(record)?;
                self.write_content(record, coloring)?;
                self.output.write_all(&[config.terminator])
            }
        }
    }

    /// Print one line in JSON Lines format (ripgrep-compatible), writing straight to the
    /// sink so that no record is staged in a `String` first.
    fn json_line(&mut self, record: LineRecord) -> io::Result<()> {
        let (path, search) = (self.path, self.search);
        if !record.is_match {
            self.output
                .write_all(br#"{"type":"context","data":{"path":"#)?;
            json_text_field_to(self.output, path.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.as_cut)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{}"#,
                record.line_number, record.byte_offset
            )?;
            self.json_line_hash(record.whole)?;
            self.output.write_all(b"}}\n")
        } else if !self.positions_in(record) {
            // An inverted match holds no position, so it carries no submatches.
            self.output
                .write_all(br#"{"type":"match","data":{"path":"#)?;
            json_text_field_to(self.output, path.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.as_cut)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{}"#,
                record.line_number, record.byte_offset
            )?;
            self.json_line_hash(record.whole)?;
            self.output.write_all(br#","submatches":[]}}"#)?;
            self.output.write_all(b"\n")
        } else {
            self.output
                .write_all(br#"{"type":"match","data":{"path":"#)?;
            json_text_field_to(self.output, path.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.as_cut)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{}"#,
                record.line_number, record.byte_offset
            )?;
            self.json_line_hash(record.whole)?;
            self.output.write_all(br#","submatches":["#)?;

            for (index, found) in search.matcher.matches(record.as_cut).enumerate() {
                if index > 0 {
                    self.output.write_all(b",")?;
                }
                self.output.write_all(br#"{"match":"#)?;
                json_text_field_to(self.output, found.text(record.as_cut))?;
                write!(
                    self.output,
                    r#","start":{},"end":{}}}"#,
                    found.offset,
                    found.offset + found.length
                )?;
            }

            self.output.write_all(b"]}}\n")
        }
    }

    /// Write the `"line_hash"` field every JSON record carries.
    ///
    /// Unconditional, as `line_number` and `absolute_offset` already are: a machine schema
    /// that changes shape with a flag makes every consumer handle both.
    fn json_line_hash(&mut self, line: &[u8]) -> io::Result<()> {
        if !self.config.carries(Field::LineHashes) {
            return Ok(());
        }
        let mut buffer = [0u8; HASH_CHARS];
        write!(
            self.output,
            r#","line_hash":"{}""#,
            format_hash(&mut buffer, content_hash(line), self.config.hash_width)
        )
    }

    /// Write the columns that come before a record's text: where it is, and what it is
    /// named. Repeated for every record `--show matches` emits from one line, so a consumer
    /// counting fields reads the same shape on every row.
    fn write_prefix(&mut self, record: LineRecord) -> io::Result<()> {
        let (path, search, config) = (self.path, self.search, self.config);
        let colors = config.colors;
        // A context line is set off by `-` where a matching line uses `:`.
        let separator = if record.is_match { ":" } else { "-" };
        if config.shows_path() {
            write!(
                self.output,
                "{}{}{}{}",
                colors.path, path, colors.reset, separator
            )?;
        }
        if config.line_numbers() {
            write!(
                self.output,
                "{}{}{}{}",
                colors.line_number, record.line_number, colors.reset, separator
            )?;
        }
        if config.column_numbers() && self.positions_in(record) {
            let column = record
                .first_match
                .map_or(1, |found| found.column(record.as_cut, search.newlines));
            write!(
                self.output,
                "{}{}{}{}",
                colors.column, column, colors.reset, separator
            )?;
        }
        if config.carries(Field::ByteOffset) {
            write!(
                self.output,
                "{}{}{}{}",
                colors.byte_offset, record.byte_offset, colors.reset, separator
            )?;
        }
        // Last of the prefix, because it names the line rather than locating it, and
        // because leaving the positional columns contiguous keeps `cut -d:` working.
        if config.carries(Field::LineHashes) {
            let mut buffer = [0u8; HASH_CHARS];
            write!(
                self.output,
                "{}{}{}{}",
                colors.hash,
                // The line as it was read, never `body`: that may carry highlight codes or
                // be trimmed, and neither is what the name refers to.
                format_hash(&mut buffer, content_hash(record.whole), config.hash_width),
                colors.reset,
                separator
            )?;
        }
        Ok(())
    }

    /// Write a line's content, reduced to the matches themselves under `--show matches`.
    fn write_content(&mut self, record: LineRecord, coloring: Coloring) -> io::Result<()> {
        let (search, config) = (self.search, self.config);
        if !(config.only_matches() && self.positions_in(record)) {
            return self.write_body(record.as_cut, coloring);
        }
        for (index, found) in search.matcher.matches(record.as_cut).enumerate() {
            if index > 0 {
                self.output.write_all(&[config.terminator])?;
                // Every match is its own record, so it carries the whole prefix rather than
                // trailing the first one's.
                self.write_prefix(record)?;
            }
            if !config.colors.match_highlight.is_empty() {
                self.output
                    .write_all(config.colors.match_highlight.as_bytes())?;
            }
            self.output.write_all(found.text(record.as_cut))?;
            if !config.colors.reset.is_empty() {
                self.output.write_all(config.colors.reset.as_bytes())?;
            }
        }
        Ok(())
    }

    /// Write a whole line, trimmed to `--max-line-length` on a character boundary and ending
    /// in the mark that tells its reader the rest was dropped.
    ///
    /// An escape code is not a character anyone reads, so it spends none of the limit.
    fn write_body(&mut self, line: &[u8], coloring: Coloring) -> io::Result<()> {
        match self.write_spans(line, coloring)? {
            Fit::Whole => Ok(()),
            Fit::Cut => self.output.write_all(b" [...]"),
        }
    }

    /// Write the line's runs against one shared budget, wrapping every match in the highlight
    /// color where one was asked for, and stopping where the budget runs out.
    fn write_spans(&mut self, line: &[u8], coloring: Coloring) -> io::Result<Fit> {
        let (search, config) = (self.search, self.config);
        let mut budget = config.max_line_length;
        if coloring == Coloring::Plain {
            return self.write_run(line, &mut budget);
        }
        let mut cursor = 0;
        for found in search.matcher.matches(line) {
            if self.write_run(&line[cursor..found.offset], &mut budget)? == Fit::Cut {
                return Ok(Fit::Cut);
            }
            self.output
                .write_all(config.colors.match_highlight.as_bytes())?;
            let fit = self.write_run(found.text(line), &mut budget)?;
            // Written even where the budget ran out inside the match, so a trimmed line
            // never leaves the terminal colored.
            self.output.write_all(config.colors.reset.as_bytes())?;
            if fit == Fit::Cut {
                return Ok(Fit::Cut);
            }
            cursor = found.offset + found.length;
        }
        self.write_run(&line[cursor..], &mut budget)
    }

    /// Write as much of `run` as the budget still allows, spending what it wrote.
    fn write_run(&mut self, run: &[u8], budget: &mut Option<usize>) -> io::Result<Fit> {
        let Some(limit) = *budget else {
            self.output.write_all(run)?;
            return Ok(Fit::Whole);
        };
        let (kept, spent, fit) = trim_to_limit(run, limit, self.search.newlines);
        *budget = Some(limit - spent);
        self.output.write_all(kept)?;
        Ok(fit)
    }

    /// Write a whole multi-line region verbatim, which is what `--multiline` prints when
    /// no per-line prefix is asked for.
    fn region(&mut self, region: &[u8]) -> io::Result<()> {
        let (path, config) = (self.path, self.config);
        if config.shows_path() {
            write!(
                self.output,
                "{}{}{}:",
                config.colors.path, path, config.colors.reset
            )?;
        }
        self.output.write_all(region)?;
        // An input whose last line is unterminated still prints as a whole line.
        if !region.ends_with(b"\n") {
            self.output.write_all(&[config.terminator])?;
        }
        Ok(())
    }
}

/// Close the JSON Lines record for a file that opened one, once every window is done.
fn print_json_end(
    output: &mut dyn Write,
    path: &str,
    reporting: &Reporting,
    state: &FindState,
    file_hash: Option<u64>,
) -> io::Result<()> {
    let Some(config) = reporting.config() else {
        return Ok(());
    };
    if config.output_format != OutputFormat::Json || !state.printed_heading {
        return Ok(());
    }
    output.write_all(br#"{"type":"end","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    // On the closing record rather than the opening one, because a file's hash covers bytes
    // the opening record is written long before a stream reaches. Naming it here is what
    // lets a piped search report one without holding the pipe in memory.
    if let Some(hash) = file_hash {
        let mut buffer = [0u8; HASH_CHARS];
        write!(
            output,
            r#","file_hash":"{}""#,
            format_hash(&mut buffer, hash, HASH_CHARS)
        )?;
    }
    write!(
        output,
        r#","stats":{{"matches":{},"lines_searched":{}}}}}}}"#,
        state.match_count, state.lines_searched
    )?;
    output.write_all(b"\n")
}

// endregion: Line Output

// region: Input Processing

/// Check if data appears to be binary (contains NUL bytes in first 8KB)
fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    find(&data[..check_len], b"\0").is_some()
}

/// The path stdin reports itself under, in records and in diagnostics.
const STDIN_NAME: &str = "-";

/// The run's totals and the questions the exit code asks of them.
#[derive(Default)]
struct Outcome {
    summary: Summary,
    tally: Tally,
    max_reached: bool,
}

/// What every input of this run is searched and reported with.
struct Session<'a> {
    search: Search<'a>,
    reporting: Reporting,
    show: Show,
    format: Format,
    terminator: u8,
    inputs: Inputs,
    binary: bool,
}

impl Session<'_> {
    /// Whether the input is binary and `--binary` did not ask for it.
    #[inline]
    fn skips(&self, data: &[u8]) -> bool {
        !self.binary && is_binary(data)
    }

    /// Fold one input's totals into the run and emit the record `--show` asks for.
    fn report(
        &self,
        output: &mut dyn Write,
        path: &str,
        result: &FileResult,
        outcome: &mut Outcome,
    ) -> io::Result<()> {
        let summary = &mut outcome.summary;
        summary.lines_searched += result.lines_searched;
        summary.matches_found += result.match_count;
        let matched = result.match_count > 0;
        if matched {
            summary.files_matched += 1;
        }

        match self.show {
            Show::Lines | Show::Matches => {
                if matched {
                    outcome.tally.produced();
                }
            }
            Show::Count => {
                if matched {
                    outcome.tally.produced();
                }
                // Only files with matches are reported over a walk, as `grep -c` and
                // `rg -c` do: listing every `reporting:0` buries the answer.
                if matched || self.inputs == Inputs::One {
                    self.write_count(output, path, result.match_count)?;
                }
            }
            Show::Files => {
                if matched {
                    outcome.tally.produced();
                    self.write_path(output, path)?;
                }
            }
            Show::FilesWithout => {
                if !matched {
                    outcome.tally.produced();
                    self.write_path(output, path)?;
                }
            }
        }
        Ok(())
    }

    /// Emit one input's matching-line count.
    fn write_count(&self, output: &mut dyn Write, path: &str, count: usize) -> io::Result<()> {
        if self.format == Format::Json {
            output.write_all(br#"{"type":"count","data":{"path":"#)?;
            json_text_field_to(output, path.as_bytes())?;
            return writeln!(output, r#","count":{}}}}}"#, count);
        }
        if self.inputs == Inputs::Many {
            write!(output, "{}:{}", path, count)?;
        } else {
            write!(output, "{}", count)?;
        }
        output.write_all(&[self.terminator])
    }

    /// Emit one input's path.
    fn write_path(&self, output: &mut dyn Write, path: &str) -> io::Result<()> {
        if self.format == Format::Json {
            output.write_all(br#"{"type":"file","data":{"path":"#)?;
            json_text_field_to(output, path.as_bytes())?;
            return output.write_all(b"}}\n");
        }
        output.write_all(path.as_bytes())?;
        output.write_all(&[self.terminator])
    }
}

/// Search one input, whatever the walk resolved it to, and fold what it found into the run.
fn search_input(
    session: &Session,
    input: &Input,
    output: &mut dyn Write,
    outcome: &mut Outcome,
) -> io::Result<()> {
    let (search, reporting) = (&session.search, &session.reporting);
    let path = input.display_name();
    let source = match input {
        // A redirect maps, and a true pipe streams unless the reporting reads lines a window
        // has already dropped.
        Input::Stdin if can_stream(search, reporting) => get_input_streaming(None),
        Input::Stdin => get_input(None),
        Input::File(entry) => open_input(entry.path()),
    };
    let source = match source {
        Ok(source) => source,
        Err(error) => {
            eprintln!("sz-find: {}: {}", path, error);
            outcome.tally.failed();
            return Ok(());
        }
    };
    outcome.summary.files_searched += 1;

    let mut windows = Windows::over(source);
    // Before the sniff below, which fills the first window: a hash installed after that would
    // miss everything it read.
    if reporting
        .config()
        .is_some_and(|config| config.carries(Field::FileHash))
    {
        windows.hash_stream();
    }
    // A whole input is sniffed from its own bytes, a pipe from its first fill, which
    // `Windows::next` leaves in place for the search that follows — so `--binary` reads the
    // same either way. `CutAfter::Anywhere` consumes nothing and cuts nothing short.
    let binary = match windows.whole() {
        Some(data) => session.skips(data),
        None => {
            let filled = windows.next(CutAfter::Anywhere, 0)?;
            session.skips(filled.map_or(&[][..], |(window, _)| window))
        }
    };
    if binary {
        // Passed over rather than searched, but a whole input was measured getting here.
        outcome.summary.bytes_searched += windows.whole().map_or(0, <[u8]>::len);
        return Ok(());
    }

    let result = search_windows(&mut windows, &path, search, reporting, output)?;
    outcome.summary.bytes_searched += result.bytes_searched;
    outcome.max_reached |= result.capped();
    session.report(output, &path, &result, outcome)
}

/// Report the run's totals. Under `--format json` they are one more record on the stream
/// they describe, so it still parses line by line. In text they are prose, and prose goes to
/// stderr — a paragraph in the middle of `sz-find … > matches.txt` is not a match.
fn print_summary(
    output: &mut dyn Write,
    notes: &mut dyn Write,
    format: Format,
    summary: &Summary,
    elapsed: std::time::Duration,
) -> io::Result<()> {
    let bytes = summary.bytes_searched;
    if format == Format::Json {
        return writeln!(
            output,
            r#"{{"type":"summary","data":{{"files_searched":{},"files_matched":{},"lines_searched":{},"matches_found":{},"bytes_searched":{},"seconds":{:.3}}}}}"#,
            summary.files_searched,
            summary.files_matched,
            summary.lines_searched,
            summary.matches_found,
            bytes,
            elapsed.as_secs_f64()
        );
    }
    writeln!(notes)?;
    writeln!(notes, "Summary:")?;
    writeln!(notes, "  Files searched: {}", summary.files_searched)?;
    writeln!(notes, "  Files matched:  {}", summary.files_matched)?;
    writeln!(notes, "  Lines searched: {}", summary.lines_searched)?;
    writeln!(notes, "  Matches found:  {}", summary.matches_found)?;
    writeln!(
        notes,
        "  Bytes searched: {} ({:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    )?;
    writeln!(notes, "  Time elapsed:   {:.3}s", elapsed.as_secs_f64())?;
    if elapsed.as_secs_f64() > 0.0 {
        writeln!(
            notes,
            "  Throughput:     {:.2} MB/s",
            (bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64()
        )?;
    }
    Ok(())
}

// endregion: Input Processing

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    // Every byte this run prints goes here, so the records stay in one order.
    let mut output = stdout_writer();
    report("sz-find", run(&args, &mut output, &mut io::stderr()))
}

/// Search every input in turn and answer with the status the run earned.
/// The run's output and the notes about it are two different streams, and the caller passes
/// both: `output` carries what the run produced, `notes` carries what it has to say about
/// the run. Only the second may be prose, and only the second goes to stderr, so redirecting
/// stdout gives a file of data rather than data with a sentence appended.
fn run(args: &Args, output: &mut dyn Write, notes: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;
    let started = std::time::Instant::now();
    let show = args.show.unwrap_or_default();

    // `--context` sets both sides at once, and conflicts with setting either alone.
    let context = match args.context {
        Some(lines) => Context {
            before: lines,
            after: lines,
        },
        None => Context {
            before: args.before_context,
            after: args.after_context,
        },
    };

    let inputs = if args.inputs.len() > 1
        || args
            .inputs
            .iter()
            .any(|input| input != "-" && Path::new(input).is_dir())
    {
        Inputs::Many
    } else {
        Inputs::One
    };

    let pattern = args.pattern.as_bytes();
    let unicode = uses_unicode(args.utf8, args.ignore_case);
    let session = Session {
        search: Search {
            matcher: Matcher {
                pattern,
                uncased_needle: args.ignore_case.then(|| Utf8UncasedNeedle::new(pattern)),
                whole_word: args.match_kind == Match::Word,
                unicode,
            },
            newlines: Newlines::from_utf8(unicode),
            multiline: args.multiline,
            invert_match: args.invert_match,
            max_matches: args.max_matches.map(NonZeroUsize::get),
            line_counting: if args.summary || args.format == Format::Json {
                LineCounting::Reported
            } else {
                LineCounting::Ignored
            },
        },
        // One dispatch decision for the whole run: what the per-line loop remembers.
        reporting: Reporting::choose(args, show, context, inputs, io::stdout().is_terminal()),
        show,
        format: args.format,
        terminator: Terminator::from_null(args.null).as_byte(),
        inputs,
        binary: args.binary,
    };

    // Compile each `--glob` once, so a malformed one is reported here rather than
    // silently matching nothing on every file of the walk.
    let globs = args
        .glob
        .as_deref()
        .map(compile_globs)
        .transpose()
        .map_err(reject)?;

    let traversal = TraversalOptions {
        hidden: args.hidden,
        no_ignore: args.no_ignore,
        follow: args.follow,
        max_depth: args.max_depth,
        file_type: args.file_type.as_deref(),
    };

    let mut outcome = Outcome::default();
    // Lazy, so `--max-matches` stops the walk rather than merely stopping the printing.
    for resolved in shared::inputs(&args.inputs, &traversal, globs.as_deref(), "sz-find") {
        if outcome.max_reached {
            break;
        }
        match resolved {
            // Only writes of stdout can fail here, and stdout is named `-`; an input that
            // failed to open was already reported and folded into `outcome`.
            Ok(input) => search_input(&session, &input, output, &mut outcome).at(STDIN_NAME)?,
            Err(failure) => {
                eprintln!("sz-find: {}", failure);
                outcome.tally.failed();
            }
        }
    }

    if args.summary {
        print_summary(
            output,
            notes,
            args.format,
            &outcome.summary,
            started.elapsed(),
        )
        .at(STDIN_NAME)?;
    }
    output.flush().at(STDIN_NAME)?;

    Ok(outcome.tally.status())
}

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use std::io::Read;

    /// The same walk over a [`Refill`] of a chosen capacity, which is the only way to put a
    /// window seam at a chosen byte.
    impl<R: Read> Windowing for Refill<R> {
        fn whole(&self) -> Option<&[u8]> {
            None
        }

        fn window(&mut self, cut: CutAfter, consumed: usize) -> io::Result<Option<&[u8]>> {
            if !self.advance(consumed)? {
                return Ok(None);
            }
            loop {
                if self.at_eof() {
                    return Ok(Some(self.filled()));
                }
                match last_cut(self.filled(), cut) {
                    Some(end) => return Ok(Some(&self.filled()[..end])),
                    None => self.grow()?,
                }
            }
        }

        fn digest(&self) -> Option<u64> {
            Refill::digest(self)
        }
    }

    /// One walk over `data` handed over whole, as a mapped file arrives.
    fn whole(data: &[u8]) -> Windows {
        Windows::over(InputSource::Buffer(data.to_vec()))
    }

    fn make_search(pattern: &[u8]) -> Search<'_> {
        Search {
            matcher: Matcher {
                pattern,
                uncased_needle: None,
                whole_word: false,
                unicode: false,
            },
            newlines: Newlines::Lf,
            multiline: false,
            invert_match: false,
            max_matches: None,
            line_counting: LineCounting::Reported,
        }
    }

    fn make_config() -> OutputConfig {
        OutputConfig {
            output_format: OutputFormat::Standard,
            colors: Colors::disabled(),
            fields: 0,
            hash_width: DEFAULT_HASH_WIDTH,
            show: Show::Lines,
            inputs: Inputs::One,
            max_line_length: None,
            terminator: b'\n',
        }
    }

    /// Search a whole slice, returning what it printed and what it counted.
    fn search_whole(
        data: &[u8],
        search: &Search,
        reporting: &Reporting,
    ) -> (Vec<u8>, usize, usize) {
        let mut output = Vec::new();
        let result =
            search_windows(&mut whole(data), "test.txt", search, reporting, &mut output).unwrap();
        (output, result.match_count, result.lines_searched)
    }

    /// Search the same way, but through a window of exactly `capacity` bytes.
    fn search_streamed(
        data: &[u8],
        capacity: usize,
        search: &Search,
        reporting: &Reporting,
    ) -> (Vec<u8>, usize, usize) {
        let mut refill = Refill::new(data, capacity);
        // As `search_stdin` does, and before anything is read: a hash installed later would
        // miss the window that installing it came after.
        if reporting
            .config()
            .is_some_and(|config| config.carries(Field::FileHash))
        {
            refill.hash_stream();
        }
        let mut output = Vec::new();
        let result =
            search_windows(&mut refill, "test.txt", search, reporting, &mut output).unwrap();
        (output, result.match_count, result.lines_searched)
    }

    /// A reader that fails once a budget of bytes has been handed out, so a test can prove
    /// a reporting stopped reading rather than merely stopped printing.
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

    /// Lines that put every seam case in reach of a tiny window: multi-byte characters,
    /// CRLF, blank lines, a line wider than the smallest capacity, and no trailing newline.
    const SEAM_CORPUS: &[u8] =
        "alpha error one\nbeta ok\n\ngamma error two\r\nδέλτα error três\nlong \
         xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx error tail\nepsilon ok\nzeta error last"
            .as_bytes();

    /// Test helper: check if pattern matches in line
    fn has_match(line: &[u8], pattern: &[u8], ignore_case: bool, whole_word: bool) -> bool {
        let uncased = ignore_case.then(|| Utf8UncasedNeedle::new(pattern));
        MatchIter::new(
            line,
            Needle::new(pattern, uncased.as_ref()),
            whole_word,
            false,
        )
        .next()
        .is_some()
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
        // Built first: `--help` and `--version` are added when the command is finalized.
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "show",
                "format",
                "fields",
                "hash-width",
                "match",
                "ignore-case",
                "before-context",
                "after-context",
                "context",
                "utf8",
                "multiline",
                "invert-match",
                "max-matches",
                "quiet",
                "summary",
                "color",
                "heading",
                "max-line-length",
                "null",
                "type",
                "glob",
                "max-depth",
                "hidden",
                "no-ignore",
                "follow",
                "binary",
                "help",
                "version",
            ]
        );
    }

    #[test]
    fn selects_one_record_kind() {
        let show = |value: &str| {
            Args::try_parse_from(["sz-find", "--show", value, "error"]).map(|args| args.show)
        };
        assert_eq!(show("matches").unwrap(), Some(Show::Matches));
        assert_eq!(show("files-without").unwrap(), Some(Show::FilesWithout));
        assert!(show("both").is_err());
        // No default, which is what lets `--quiet` conflict with it.
        assert_eq!(
            Args::try_parse_from(["sz-find", "error"]).unwrap().show,
            None
        );
    }

    #[test]
    fn rejects_flags_that_would_silently_override_each_other() {
        let accepts = |flags: &[&str]| {
            let mut argv = vec!["sz-find"];
            argv.extend_from_slice(flags);
            argv.push("e");
            Args::try_parse_from(argv)
                .map_err(|_| ())
                .and_then(|args| validate(&args).map_err(|_| ()))
                .is_ok()
        };

        // Each second flag is inert by construction under the first.
        for flags in [
            vec!["--quiet", "--format", "json"],
            vec!["--quiet", "--null"],
            vec!["--quiet", "--fields", "line-numbers"],
            vec!["--quiet", "--fields", "line-hashes"],
            vec!["--quiet", "--heading"],
            vec!["--quiet", "--color", "always"],
            vec!["--quiet", "--max-line-length", "10"],
            vec!["--quiet", "--context", "2"],
            vec!["--quiet", "--max-matches", "3"],
            vec!["--quiet", "--show", "count"],
            vec!["--format", "json", "--null"],
            vec!["--format", "json", "--heading"],
            vec!["--format", "json", "--color", "always"],
            vec!["--format", "json", "--max-line-length", "10"],
            // JSON emits the position of every record unasked, so naming it is inert.
            vec!["--format", "json", "--fields", "line-numbers"],
            vec!["--format", "json", "--fields", "byte-offset"],
            vec!["--format", "vimgrep", "--heading"],
            vec!["--format", "vimgrep", "--fields", "line-numbers"],
            vec!["--format", "vimgrep", "--fields", "line-hashes"],
            vec!["--format", "vimgrep", "--color", "always"],
            vec!["--show", "count", "--fields", "line-numbers"],
            vec!["--show", "count", "--heading"],
            vec!["--show", "files", "--fields", "line-numbers"],
            vec!["--show", "files", "--context", "2"],
            vec!["--show", "matches", "--invert-match"],
            vec!["--multiline", "--invert-match"],
            vec!["--context", "2", "--after-context", "1"],
        ] {
            assert!(!accepts(&flags), "accepted {:?}", flags);
        }

        // A count is a record, so `--null` terminates it and JSON carries it. A cap is
        // inert under `--quiet` alone, and bounds the totals `--summary` reports.
        assert!(accepts(&["--show", "count", "--null"]));
        assert!(accepts(&["--show", "count", "--format", "json"]));
        assert!(accepts(&["--quiet", "--summary", "--max-matches", "2"]));
        // A whole-file hash is the one field JSON does not already emit, so it is the one
        // `--fields` value that form accepts.
        assert!(accepts(&["--format", "json", "--fields", "file-hash"]));
        // Neither hash is emitted unasked, so both are meaningful under JSON.
        assert!(accepts(&["--format", "json", "--fields", "line-hashes"]));
        assert!(accepts(&[
            "--format",
            "json",
            "--fields",
            "line-hashes,file-hash"
        ]));
        assert!(accepts(&["--heading", "--fields", "file-hash"]));
        // Plain text and vimgrep have nowhere to put it but every record.
        assert!(!accepts(&["--fields", "file-hash"]));
        assert!(!accepts(&["--format", "vimgrep", "--fields", "file-hash"]));

        assert!(validate(&Args::try_parse_from(["sz-find", ""]).unwrap()).is_err());
        // Zero is a whole-run no-op rather than a limit.
        assert!(Args::try_parse_from(["sz-find", "--max-matches", "0", "e"]).is_err());
        assert!(Args::try_parse_from(["sz-find", "--max-line-length", "0", "e"]).is_err());
    }

    #[test]
    fn bounds_the_summary_totals_under_quiet() {
        // `--quiet` prints no line, so the cap only ever shows in what `--summary` reports.
        let directory = tempfile::TempDir::new().unwrap();
        let log = directory.path().join("log.txt");
        std::fs::write(&log, b"error one\nerror two\nerror three\n").unwrap();

        let args = Args::try_parse_from([
            "sz-find",
            "--quiet",
            "--summary",
            "--max-matches",
            "2",
            "error",
            log.to_str().unwrap(),
        ])
        .unwrap();
        assert!(validate(&args).is_ok());

        let (mut printed, mut notes) = (Vec::new(), Vec::new());
        assert!(matches!(
            run(&args, &mut printed, &mut notes),
            Ok(Status::Success)
        ));
        assert!(printed.is_empty(), "--quiet prints no line");
        let notes = String::from_utf8(notes).unwrap();
        assert!(notes.contains("Matches found:  2"), "{notes}");
    }

    #[test]
    fn emits_the_summary_as_a_record_under_json() {
        let summary = Summary {
            files_searched: 2,
            files_matched: 1,
            lines_searched: 9,
            matches_found: 3,
            bytes_searched: 40,
        };
        let mut printed = Vec::new();
        print_summary(
            &mut printed,
            &mut io::sink(),
            Format::Json,
            &summary,
            std::time::Duration::from_millis(1500),
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(printed).unwrap(),
            "{\"type\":\"summary\",\"data\":{\"files_searched\":2,\"files_matched\":1,\
             \"lines_searched\":9,\"matches_found\":3,\"bytes_searched\":40,\"seconds\":1.500}}\n"
        );
    }

    #[test]
    fn tallies_silently_under_quiet() {
        let args = Args::try_parse_from(["sz-find", "--quiet", "error", "log.txt"]).unwrap();
        let reporting = Reporting::choose(&args, Show::Lines, Context::none(), Inputs::One, false);
        assert!(matches!(
            reporting,
            Reporting::Tally {
                stop_at_first: true
            }
        ));
        assert!(reporting.config().is_none());
    }

    /// A session reporting `show` records in `format`, over a single unnamed input.
    fn make_session(show: Show, format: Format, null: bool) -> Session<'static> {
        Session {
            search: make_search(b"error"),
            reporting: Reporting::Tally {
                stop_at_first: false,
            },
            show,
            format,
            terminator: Terminator::from_null(null).as_byte(),
            inputs: Inputs::One,
            binary: false,
        }
    }

    fn report_one(session: &Session, match_count: usize) -> Vec<u8> {
        let mut output = Vec::new();
        let mut outcome = Outcome::default();
        let result = FileResult {
            match_count,
            lines_searched: 9,
            bytes_searched: 40,
            flow: ControlFlow::Continue(()),
        };
        session
            .report(&mut output, "log.txt", &result, &mut outcome)
            .unwrap();
        assert_eq!(
            outcome.tally.status(),
            Status::from_found(match_count > 0 || session.show == Show::FilesWithout)
        );
        output
    }

    #[test]
    fn emits_a_count_record_rather_than_a_bare_integer_in_json() {
        let text = report_one(&make_session(Show::Count, Format::Text, false), 3);
        assert_eq!(text, b"3\n");
        let json = report_one(&make_session(Show::Count, Format::Json, false), 3);
        assert_eq!(
            json,
            br#"{"type":"count","data":{"path":{"text":"log.txt"},"count":3}}"#
                .iter()
                .copied()
                .chain(*b"\n")
                .collect::<Vec<u8>>()
        );
    }

    #[test]
    fn terminates_every_record_with_nul_under_null() {
        assert_eq!(
            report_one(&make_session(Show::Files, Format::Text, true), 2),
            b"log.txt\0"
        );
        assert_eq!(
            report_one(&make_session(Show::FilesWithout, Format::Text, true), 0),
            b"log.txt\0"
        );
        assert_eq!(
            report_one(&make_session(Show::Count, Format::Text, true), 2),
            b"2\0"
        );

        // `--null` reaches the printed lines too, where grep's `-Z` stops at file names.
        let mut config = make_config();
        config.terminator = 0;
        let (printed, ..) = search_whole(
            b"error one\nplain\nerror two\n",
            &make_search(b"error"),
            &Reporting::Print(config),
        );
        assert_eq!(printed, b"error one\0error two\0");
    }

    #[test]
    fn honours_stdin_in_any_position() {
        // Each input is dispatched on its own, so `-` names stdin wherever it appears
        // rather than becoming a literal reporting once a second input follows it.
        let args = Args::try_parse_from(["sz-find", "error", "log.txt", "-"]).unwrap();
        assert_eq!(args.inputs, ["log.txt", "-"]);
        // Records give stdin the same token that selects it.
        assert_eq!(STDIN_NAME, "-");
    }

    #[test]
    fn searches_past_a_missing_input_but_still_reports_the_failure() {
        let directory = tempfile::TempDir::new().unwrap();
        let present = directory.path().join("present.txt");
        std::fs::write(&present, b"error here\n").unwrap();
        let missing = directory.path().join("missing.txt");

        let mut output = Vec::new();
        let args = Args::try_parse_from([
            "sz-find",
            "error",
            missing.to_str().unwrap(),
            present.to_str().unwrap(),
        ])
        .unwrap();
        // The readable neighbour is still searched and its matches still printed, but an input
        // that could not be read is a run that did not complete.
        assert!(matches!(
            run(&args, &mut output, &mut io::sink()),
            Ok(Status::Error)
        ));
        assert!(
            !output.is_empty(),
            "the readable input must still be searched"
        );

        let args = Args::try_parse_from(["sz-find", "error", missing.to_str().unwrap()]).unwrap();
        assert!(matches!(
            run(&args, &mut io::sink(), &mut io::sink()),
            Ok(Status::Error)
        ));
    }

    #[test]
    fn reads_word_boundaries_and_columns_by_character_under_utf8() {
        // `é` closes no word under Unicode rules, so `caf` is not a whole word there.
        let line = "café".as_bytes();
        assert!(is_word_boundary(line, 0, 3, false));
        assert!(!is_word_boundary(line, 0, 3, true));

        // Six characters precede the match, in eleven bytes.
        let found = MatchInfo {
            offset: 11,
            length: 5,
        };
        let line = "δέλτα error".as_bytes();
        assert_eq!(found.column(line, Newlines::Lf), 12);
        assert_eq!(found.column(line, Newlines::Unicode), 7);
        assert_eq!(
            trim_to_limit(line, 5, Newlines::Unicode),
            (&line[..10], 5, Fit::Cut)
        );
        assert_eq!(
            trim_to_limit(line, 6, Newlines::Lf),
            (&line[..6], 6, Fit::Cut)
        );
    }

    #[test]
    fn detects_substring_matches() {
        let line = b"hello world";
        assert!(has_match(line, b"hello", false, false));
        assert!(has_match(line, b"world", false, false));
        assert!(!has_match(line, b"foo", false, false));
    }

    #[test]
    fn matches_ignoring_case() {
        let line = b"Hello World";
        assert!(has_match(line, b"hello", true, false));
        assert!(has_match(line, b"WORLD", true, false));
        assert!(!has_match(line, b"foo", true, false));
    }

    #[test]
    fn matches_only_at_word_boundaries() {
        let line = b"hello world";
        assert!(has_match(line, b"hello", false, true));
        assert!(has_match(line, b"world", false, true));
        assert!(!has_match(line, b"ello", false, true)); // Not at word boundary
        assert!(!has_match(line, b"worl", false, true)); // Not at word boundary
    }

    #[test]
    fn finds_standalone_word_after_partial_match() {
        // Test that word boundary check finds matches after non-boundary matches
        let line = b"fn_name fn";
        assert!(has_match(line, b"fn", false, true)); // Should find standalone "fn"
    }

    #[test]
    fn inverts_line_match_selection() {
        let line = b"hello world";
        let mut search = make_search(b"hello");
        assert!(matches!(search.select(line), Selection::Selected(_)));

        search.invert_match = true;
        assert!(matches!(search.select(line), Selection::Rejected));
        assert!(matches!(
            search.select(b"goodbye world"),
            Selection::Selected(_)
        ));
    }

    #[test]
    fn detects_binary_from_null_bytes() {
        assert!(is_binary(b"hello\0world"));
        assert!(!is_binary(b"hello world"));
        assert!(!is_binary(b"hello\nworld\n"));
    }

    #[test]
    fn reports_matching_line() {
        let data = b"line1\nerror here\nline3\n";
        let search = make_search(b"error");
        let reporting = Reporting::Print(make_config());
        let mut output = Vec::new();

        let result = search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        assert_eq!(result.match_count, 1);
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("error here"));
    }

    #[test]
    fn includes_surrounding_context_lines() {
        let data = b"line1\nline2\nerror here\nline4\nline5\n";
        let search = make_search(b"error");
        let context = Context {
            before: 1,
            after: 1,
        };
        let reporting = Reporting::PrintContext(make_config(), context);
        let mut output = Vec::new();

        search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("line2"));
        assert!(output_str.contains("error here"));
        assert!(output_str.contains("line4"));
    }

    #[test]
    fn stops_after_max_count() {
        let data = b"error1\nerror2\nerror3\nerror4\n";
        let mut search = make_search(b"error");
        search.max_matches = Some(2);
        let reporting = Reporting::Print(make_config());
        let mut output = Vec::new();

        let result = search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        assert_eq!(result.match_count, 2);
        assert!(result.capped());
    }

    #[test]
    fn counts_without_printing() {
        let data = b"error1\nok\nerror2\n";
        let search = make_search(b"error");
        let reporting = Reporting::Tally {
            stop_at_first: false,
        };
        let mut output = Vec::new();

        let result = search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        assert_eq!(result.match_count, 2);
        assert_eq!(result.lines_searched, 3);
        assert!(output.is_empty());
    }

    #[test]
    fn stops_the_tally_at_the_first_match() {
        let data = b"ok\nerror1\nerror2\nerror3\n";
        let search = make_search(b"error");
        let reporting = Reporting::Tally {
            stop_at_first: true,
        };
        let mut output = Vec::new();

        let result = search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        assert_eq!(result.match_count, 1);
        assert_eq!(result.lines_searched, 2);
    }

    #[test]
    fn streams_every_path_that_reads_no_earlier_line() {
        let search = make_search(b"error");
        assert!(can_stream(
            &search,
            &Reporting::Tally {
                stop_at_first: true
            }
        ));
        assert!(can_stream(&search, &Reporting::Print(make_config())));
        assert!(can_stream(
            &search,
            &Reporting::PrintContext(
                make_config(),
                Context {
                    before: 0,
                    after: 3
                }
            )
        ));
        assert!(!can_stream(
            &search,
            &Reporting::PrintContext(
                make_config(),
                Context {
                    before: 1,
                    after: 0
                }
            )
        ));
    }

    #[test]
    fn wraps_matches_in_color_codes() {
        let search = make_search(b"hello");
        let mut config = make_config();
        config.colors = Colors::enabled();

        let (output, ..) = search_whole(b"hello world hello\n", &search, &Reporting::Print(config));
        assert_eq!(
            output,
            b"\x1b[1;31mhello\x1b[0m world \x1b[1;31mhello\x1b[0m\n"
        );

        // A line without a match is printed as it stands, escapes and all absent.
        let (output, ..) = search_whole(
            b"hello\nnothing here\n",
            &make_search(b"nothing"),
            &Reporting::Print(config),
        );
        assert_eq!(output, b"\x1b[1;31mnothing\x1b[0m here\n");
    }

    /// The trimming budget counts what a reader sees, so color changes where a line is cut
    /// but never how much of it survives.
    #[test]
    fn trims_colored_and_plain_lines_to_the_same_text() {
        let data = b"one error two error three error four\n";
        let search = make_search(b"error");
        for limit in [1usize, 4, 9, 12, 20, 33, 36, 40] {
            let mut config = make_config();
            config.max_line_length = Some(limit);
            let (plain, ..) = search_whole(data, &search, &Reporting::Print(config));

            config.colors = Colors::enabled();
            let (colored, ..) = search_whole(data, &search, &Reporting::Print(config));
            let visible: Vec<u8> = String::from_utf8(colored)
                .unwrap()
                .replace("\x1b[1;31m", "")
                .replace("\x1b[0m", "")
                .into_bytes();
            assert_eq!(visible, plain, "limit {}", limit);
        }
    }

    #[test]
    fn reports_the_same_column_with_and_without_color() {
        // Two matches on one line, so the colored form carries an escape code ahead of
        // each. `grep -bo` puts the first at byte 5, which is column 6.
        let data = b"lead error one error two\n";
        let search = make_search(b"error");
        let mut config = make_config();
        config.fields = Field::bits(&[Field::ColumnNumbers]);

        let (plain, ..) = search_whole(data, &search, &Reporting::Print(config));
        assert_eq!(plain, b"6:lead error one error two\n");

        config.colors = Colors::enabled();
        let (colored, ..) = search_whole(data, &search, &Reporting::Print(config));
        let colored = String::from_utf8(colored).unwrap();
        assert!(
            colored.starts_with("\x1b[32m6\x1b[0m:"),
            "column resolved against the highlighted line: {:?}",
            colored
        );
    }

    #[test]
    fn matches_across_newlines_in_multiline_mode() {
        let data = b"hello\nworld\nfoo bar\n";
        let mut search = make_search(b"hello\nworld");
        search.multiline = true;
        let reporting = Reporting::Print(make_config());
        let mut output = Vec::new();

        let result = search_windows(
            &mut whole(data),
            "test.txt",
            &search,
            &reporting,
            &mut output,
        )
        .unwrap();

        assert_eq!(result.match_count, 1);
    }

    #[test]
    fn keeps_blank_context_lines_in_multiline_mode() {
        // The blank lines either side of the match are context that `grep -C 1` prints.
        let data = b"alpha\n\nbeta MATCH here\n\ngamma\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let context = Context {
            before: 1,
            after: 1,
        };

        let reporting = Reporting::PrintContext(make_config(), context);
        let (printed, ..) = search_whole(data, &search, &reporting);
        assert_eq!(printed, b"\nbeta MATCH here\n\n");

        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineNumbers]);
        let reporting = Reporting::PrintContext(config, context);
        let (numbered, ..) = search_whole(data, &search, &reporting);
        assert_eq!(numbered, b"2:\n3:beta MATCH here\n4:\n");
    }

    #[test]
    fn terminates_every_multiline_region() {
        // Two matches far apart print as two whole lines, not one run-on line. The first
        // sits at byte zero, where the search for the line start has nothing to scan.
        let data = b"MATCH l1\nl2\nl3\nl4\nl5 MATCH\nl6\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let reporting = Reporting::Print(make_config());

        let (printed, ..) = search_whole(data, &search, &reporting);
        assert_eq!(printed, b"MATCH l1\nl5 MATCH\n");
    }

    #[test]
    fn prints_a_context_region_that_is_only_a_blank_line() {
        // The after-context of the first match and the before-context of the second are
        // one blank line each, so neither region holds anything but its terminator.
        let data = b"MATCH one\n\n\nMATCH two\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let context = Context {
            before: 1,
            after: 1,
        };
        let reporting = Reporting::PrintContext(make_config(), context);

        let (printed, ..) = search_whole(data, &search, &reporting);
        assert_eq!(printed, b"MATCH one\n\n\nMATCH two\n");
    }

    #[test]
    fn divides_multiline_groups_that_do_not_adjoin() {
        // Two matches four lines apart print as two groups, and `grep -C 1` divides
        // groups with `--`. Line mode and multiline mode answer alike.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta\nepsilon\nzeta MATCH two\neta\n";
        let mut search = make_search(b"MATCH");
        let context = Context {
            before: 1,
            after: 1,
        };
        let reporting = Reporting::PrintContext(make_config(), context);

        let (by_line, ..) = search_whole(data, &search, &reporting);
        search.multiline = true;
        let (by_region, ..) = search_whole(data, &search, &reporting);
        assert_eq!(
            by_region,
            b"alpha\nbeta MATCH one\ngamma\n--\nepsilon\nzeta MATCH two\neta\n"
        );
        assert_eq!(by_region, by_line);
    }

    #[test]
    fn joins_groups_that_adjoin() {
        // The after-context of the first match and the before-context of the second are
        // consecutive lines, so the whole file prints as one group, as `grep -C 1` does.
        // Line mode and multiline mode answer alike.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta\nepsilon MATCH two\nzeta\n";
        let mut search = make_search(b"MATCH");
        let context = Context {
            before: 1,
            after: 1,
        };
        let reporting = Reporting::PrintContext(make_config(), context);

        let (by_line, ..) = search_whole(data, &search, &reporting);
        search.multiline = true;
        let (by_region, ..) = search_whole(data, &search, &reporting);
        assert_eq!(by_region, data);
        assert_eq!(by_line, by_region);
    }

    #[test]
    fn divides_line_groups_exactly_where_grep_does() {
        // Every expectation below is GNU grep 3.11's own output for the same input and
        // the same context widths. A group opens at the first line it will print, not
        // at its matching line, so groups whose surrounding lines adjoin take no divider.
        let two_apart: &[u8] = b"one\ntwo MATCH\nthree\nfour MATCH\nfive\n";
        let three_apart: &[u8] = b"one\ntwo MATCH\nthree\nfour\nfive MATCH\nsix\n";
        let four_apart: &[u8] = b"one\ntwo MATCH\nthree\nfour\nfive\nsix MATCH\nseven\n";
        let at_both_ends: &[u8] = b"one MATCH\ntwo\nthree\nfour MATCH\nfive\n";
        let adjacent: &[u8] = b"one\ntwo MATCH\nthree MATCH\nfour\n";
        let far_apart: &[u8] =
            b"one\ntwo\nthree MATCH\nfour\nfive\nsix\nseven MATCH\neight\nnine\n";

        let cases: [(&[u8], usize, usize, &[u8]); 11] = [
            // Adjoining contexts print as one block.
            (two_apart, 1, 0, b"one\ntwo MATCH\nthree\nfour MATCH\n"),
            (two_apart, 0, 1, b"two MATCH\nthree\nfour MATCH\nfive\n"),
            (
                three_apart,
                1,
                1,
                b"one\ntwo MATCH\nthree\nfour\nfive MATCH\nsix\n",
            ),
            (
                at_both_ends,
                1,
                1,
                b"one MATCH\ntwo\nthree\nfour MATCH\nfive\n",
            ),
            // Overlapping contexts print each line once.
            (adjacent, 2, 2, b"one\ntwo MATCH\nthree MATCH\nfour\n"),
            (
                far_apart,
                2,
                2,
                b"one\ntwo\nthree MATCH\nfour\nfive\nsix\nseven MATCH\neight\nnine\n",
            ),
            // A line falling between the groups divides them.
            (three_apart, 1, 0, b"one\ntwo MATCH\n--\nfour\nfive MATCH\n"),
            (
                four_apart,
                1,
                1,
                b"one\ntwo MATCH\nthree\n--\nfive\nsix MATCH\nseven\n",
            ),
            (at_both_ends, 1, 0, b"one MATCH\n--\nthree\nfour MATCH\n"),
            (
                at_both_ends,
                0,
                1,
                b"one MATCH\ntwo\n--\nfour MATCH\nfive\n",
            ),
            (
                far_apart,
                1,
                1,
                b"two\nthree MATCH\nfour\n--\nsix\nseven MATCH\neight\n",
            ),
        ];

        for (data, before, after, expected) in cases {
            let search = make_search(b"MATCH");
            let reporting = Reporting::PrintContext(make_config(), Context { before, after });
            let (printed, ..) = search_whole(data, &search, &reporting);
            assert_eq!(
                String::from_utf8_lossy(&printed),
                String::from_utf8_lossy(expected),
                "-B {} -A {}",
                before,
                after
            );
        }
    }

    #[test]
    fn divides_no_multiline_groups_without_context() {
        // Without any context lines there are no groups, so no divider is written.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta\nepsilon\nzeta MATCH two\neta\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let reporting = Reporting::Print(make_config());

        let (printed, ..) = search_whole(data, &search, &reporting);
        assert_eq!(printed, b"beta MATCH one\nzeta MATCH two\n");
    }

    #[test]
    fn counts_lines_forward_without_rescanning() {
        // Newlines at 1, 3, 5, 7 and 9.
        let mut data = b"a\nb\nc\nd\ne\n".to_vec();
        let mut lines = LineCursor::default();
        assert_eq!(lines.line_at(&data, 0), 1);
        assert_eq!(lines.line_at(&data, 4), 3);

        // Rewrite the prefix already counted: an implementation that rescans from byte
        // zero would see the two added newlines and report 7 rather than 5.
        data[0] = b'\n';
        data[2] = b'\n';
        assert_eq!(lines.line_at(&data, 8), 5);
        assert_eq!(lines.line_at(&data, data.len()), 6);
    }

    #[test]
    fn numbers_every_multiline_region_from_the_running_count() {
        let mut data = Vec::new();
        for index in 0..1000 {
            data.extend_from_slice(if index % 5 == 0 {
                b"line MATCH\n".as_slice()
            } else {
                b"line plain\n".as_slice()
            });
        }
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineNumbers]);
        let reporting = Reporting::Print(config);

        let (printed, matches, _) = search_whole(&data, &search, &reporting);
        assert_eq!(matches, 200);
        let text = String::from_utf8(printed).unwrap();
        let numbers: Vec<&str> = text
            .lines()
            .map(|line| line.split(':').next().unwrap())
            .collect();
        assert_eq!(numbers.first().copied(), Some("1"));
        assert_eq!(numbers.last().copied(), Some("996"));
        assert_eq!(numbers.len(), 200);
    }

    #[test]
    fn never_streams_a_multiline_search() {
        let mut search = make_search(b"error");
        let reporting = Reporting::Print(make_config());
        assert!(can_stream(&search, &reporting));

        search.multiline = true;
        assert!(!can_stream(&search, &reporting));
    }

    #[test]
    fn streams_the_same_output_at_tiny_capacities() {
        let config = make_config();
        let paths = [
            Reporting::Tally {
                stop_at_first: false,
            },
            Reporting::Print(config),
            Reporting::PrintContext(
                config,
                Context {
                    before: 0,
                    after: 2,
                },
            ),
        ];

        for newlines in [Newlines::Lf, Newlines::Unicode] {
            for reporting in &paths {
                let mut search = make_search(b"error");
                search.newlines = newlines;
                let expected = search_whole(SEAM_CORPUS, &search, reporting);
                for capacity in [1, 2, 3, 7, 13, 64, 4096] {
                    let streamed = search_streamed(SEAM_CORPUS, capacity, &search, reporting);
                    assert_eq!(streamed, expected, "capacity {}", capacity);
                }
            }
        }
    }

    #[test]
    fn keeps_line_numbers_and_offsets_absolute_across_windows() {
        let mut config = make_config();
        config.fields = Field::bits(&[
            Field::LineNumbers,
            Field::ColumnNumbers,
            Field::ByteOffset,
            Field::LineHashes,
        ]);
        let reporting = Reporting::Print(config);
        let search = make_search(b"error");

        let (expected, ..) = search_whole(SEAM_CORPUS, &search, &reporting);
        for capacity in [1, 7, 13, 64] {
            let (streamed, ..) = search_streamed(SEAM_CORPUS, capacity, &search, &reporting);
            assert_eq!(streamed, expected, "capacity {}", capacity);
        }
        // The last line starts well past any of the tiny windows, so a window-relative
        // offset would report it near zero — and the hash is the same either way, since it
        // names the line's content rather than where the reader happened to be.
        let text = String::from_utf8(expected).unwrap();
        let last = text.lines().last().unwrap();
        let offset = SEAM_CORPUS.len() - b"zeta error last".len();
        let mut buffer = [0u8; HASH_CHARS];
        let name = format_hash(
            &mut buffer,
            content_hash(b"zeta error last"),
            DEFAULT_HASH_WIDTH,
        );
        assert_eq!(last, format!("8:6:{offset}:{name}:zeta error last"));
    }

    #[test]
    fn opens_and_closes_one_json_record_per_streamed_file() {
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        let reporting = Reporting::Print(config);
        let search = make_search(b"error");

        let (streamed, ..) = search_streamed(SEAM_CORPUS, 7, &search, &reporting);
        let text = String::from_utf8(streamed).unwrap();
        assert_eq!(text.matches(r#""type":"begin""#).count(), 1);
        assert_eq!(text.matches(r#""type":"end""#).count(), 1);
    }

    #[test]
    fn opens_and_closes_one_json_record_per_multiline_file() {
        // A JSON consumer reads one record shape, whether or not `--multiline` was given.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta MATCH two\n";
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        let reporting = Reporting::Print(config);
        let mut search = make_search(b"MATCH");

        let record_types = |printed: Vec<u8>| -> Vec<String> {
            String::from_utf8(printed)
                .unwrap()
                .lines()
                .map(|record| record.split('"').nth(3).unwrap().to_string())
                .collect()
        };
        let by_line = record_types(search_whole(data, &search, &reporting).0);
        search.multiline = true;
        let by_region = record_types(search_whole(data, &search, &reporting).0);

        assert_eq!(by_region, ["begin", "match", "match", "end"]);
        assert_eq!(by_region, by_line);
    }

    #[test]
    fn names_the_file_once_per_multiline_run() {
        // `--heading` prints the path as a header rather than as a per-line prefix, and
        // multiline mode reads the same writer, so the header appears there too.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta MATCH two\n";
        let mut config = make_config();
        config.output_format = OutputFormat::Heading;
        let reporting = Reporting::Print(config);
        let mut search = make_search(b"MATCH");
        search.multiline = true;

        let (printed, ..) = search_whole(data, &search, &reporting);
        assert_eq!(printed, b"test.txt\nbeta MATCH one\ndelta MATCH two\n");
    }

    #[test]
    fn stops_reading_the_stream_at_the_first_match() {
        let data = b"error one\nerror two\nerror three\nerror four\n";
        let search = make_search(b"error");
        let reporting = Reporting::Tally {
            stop_at_first: true,
        };
        // A budget of one window proves nothing past it was ever requested.
        let reader = BudgetedReader {
            data,
            position: 0,
            budget: 10,
        };
        let mut refill = Refill::new(reader, 10);
        let mut output = Vec::new();

        let result =
            search_windows(&mut refill, "test.txt", &search, &reporting, &mut output).unwrap();

        assert_eq!(result.match_count, 1);
        assert_eq!(result.lines_searched, 1);
    }

    #[test]
    fn stops_reading_the_stream_at_the_max_count() {
        let data = b"error one\nerror two\nerror three\nerror four\n";
        let mut search = make_search(b"error");
        search.max_matches = Some(1);
        let reporting = Reporting::Print(make_config());
        // Two lines fit the budget: the second is where the max-count test fires.
        let reader = BudgetedReader {
            data,
            position: 0,
            budget: 20,
        };
        let mut refill = Refill::new(reader, 20);
        let mut output = Vec::new();

        let result =
            search_windows(&mut refill, "test.txt", &search, &reporting, &mut output).unwrap();

        assert_eq!(result.match_count, 1);
        assert!(result.capped());
        assert_eq!(output, b"error one\n");
    }

    /// Test helper: the hashes a run printed, in order, given one column of them.
    ///
    /// Split on both separators, since a context line ends its columns with `-` where a
    /// matching line ends them with `:`.
    fn printed_hashes(output: &[u8], column: usize) -> Vec<String> {
        String::from_utf8(output.to_vec())
            .unwrap()
            .lines()
            .filter_map(|line| line.split([':', '-']).nth(column).map(str::to_string))
            .collect()
    }

    #[test]
    fn names_a_line_the_same_under_both_newline_sets() {
        // `gamma error two` ends in CRLF. A name covers the line's terminator, and a line
        // spans to the start of the next one in either newline set, so both readings name
        // the same bytes — with no special case anywhere.
        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineHashes]);
        let reporting = Reporting::Print(config);

        let mut lf = make_search(b"error");
        lf.newlines = Newlines::Lf;
        let mut unicode = make_search(b"error");
        unicode.newlines = Newlines::Unicode;

        let (under_lf, ..) = search_whole(SEAM_CORPUS, &lf, &reporting);
        let (under_unicode, ..) = search_whole(SEAM_CORPUS, &unicode, &reporting);

        let expected = {
            let mut buffer = [0u8; HASH_CHARS];
            format_hash(
                &mut buffer,
                content_hash(b"gamma error two\r\n"),
                DEFAULT_HASH_WIDTH,
            )
            .to_string()
        };
        for (mode, output) in [("lf", under_lf), ("unicode", under_unicode)] {
            let named: Vec<_> = printed_hashes(&output, 0);
            assert!(
                named.contains(&expected),
                "under {mode} the CRLF line was named {named:?}, not {expected}"
            );
        }
    }

    #[test]
    fn prints_the_names_the_library_derives() {
        // The producing half of the handshake. What this binary prints is all that
        // `sz-replace --match line-hash` and `--expect-hash` have to go on, and neither can
        // observe the other's arithmetic, so each side is pinned against `shared` instead.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::LineHashes, Field::FileHash]);
        let corpus = b"alpha\nbeta error\ngamma\n";

        let (output, ..) = search_whole(corpus, &make_search(b"error"), &Reporting::Print(config));
        let text = String::from_utf8(output).unwrap();

        let mut buffer = [0u8; HASH_CHARS];
        // A line's name covers its terminator, so this is the whole line, not its text.
        let name = format_hash(
            &mut buffer,
            content_hash(b"beta error\n"),
            DEFAULT_HASH_WIDTH,
        );
        assert!(text.contains(&format!(r#""line_hash":"{name}""#)), "{text}");
        let mut whole = [0u8; HASH_CHARS];
        let file = format_hash(&mut whole, content_hash(corpus), HASH_CHARS);
        assert!(text.contains(&format!(r#""file_hash":"{file}""#)), "{text}");
        // And both are in the form the consuming side parses.
        assert!(parse_hash_prefix(name).is_ok());
        assert!(parse_content_hash(file).is_ok());
    }

    #[test]
    fn names_the_same_line_identically_wherever_it_sits() {
        // Content-derived, not position-derived: the two identical lines share a name and the
        // one between them does not, whatever their line numbers are.
        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineHashes]);
        let reporting = Reporting::Print(config);
        let search = make_search(b"error");

        let corpus = b"dup error\nother error\ndup error\n";
        let (output, ..) = search_whole(corpus, &search, &reporting);
        let named = printed_hashes(&output, 0);

        assert_eq!(named.len(), 3);
        assert_eq!(named[0], named[2]);
        assert_ne!(named[0], named[1]);
    }

    #[test]
    fn names_an_empty_line_and_a_context_line() {
        // Both are lines an agent may want to address, so neither may be skipped: an empty
        // line is nameable, and a context record carries a name as a match record does.
        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineHashes]);
        let reporting = Reporting::PrintContext(
            config,
            Context {
                before: 1,
                after: 0,
            },
        );
        let search = make_search(b"beta");

        let (output, ..) = search_whole(b"alpha\n\nbeta error\n", &search, &reporting);
        let named = printed_hashes(&output, 0);

        assert_eq!(named.len(), 2, "a context line and its match");
        let mut buffer = [0u8; HASH_CHARS];
        assert_eq!(
            named[0],
            format_hash(&mut buffer, content_hash(b"\n"), DEFAULT_HASH_WIDTH)
        );
    }

    #[test]
    fn names_the_line_rather_than_what_it_prints() {
        // The name refers to the bytes in the file, so trimming, highlighting or reducing the
        // record to its matches must not move it — the same rule the column already obeys.
        let search = make_search(b"error");
        let corpus = b"alpha error one\n";

        let mut plain = make_config();
        plain.fields = Field::bits(&[Field::LineHashes]);
        let (expected, ..) = search_whole(corpus, &search, &Reporting::Print(plain));
        let expected = printed_hashes(&expected, 0);

        let mut colored = plain;
        colored.colors = Colors::enabled();
        let mut trimmed = plain;
        trimmed.max_line_length = Some(4);
        let mut reduced = plain;
        reduced.show = Show::Matches;

        for (label, config) in [
            ("colored", colored),
            ("trimmed", trimmed),
            ("reduced", reduced),
        ] {
            let (output, ..) = search_whole(corpus, &search, &Reporting::Print(config));
            let text = String::from_utf8(output).unwrap();
            assert!(
                text.contains(&expected[0]),
                "{label} named the line {text:?}, not {expected:?}"
            );
        }
    }

    #[test]
    fn carries_a_line_hash_in_every_json_record_kind() {
        // A consumer reads one schema, not one per record type, so a `--context` run must not
        // hand it records that sometimes carry the field and sometimes do not.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::LineHashes]);
        let search = make_search(b"beta");

        let reporting = Reporting::PrintContext(
            config,
            Context {
                before: 1,
                after: 1,
            },
        );
        let (output, ..) = search_whole(b"alpha\nbeta error\ngamma\n", &search, &reporting);
        let text = String::from_utf8(output).unwrap();

        for kind in [r#""type":"match""#, r#""type":"context""#] {
            let carried = text
                .lines()
                .filter(|line| line.contains(kind))
                .all(|line| line.contains(r#""line_hash":""#));
            assert!(carried, "a {kind} record carried no line_hash:\n{text}");
        }

        // And the inverted form, whose records carry no submatches to hide behind.
        let mut inverted = make_search(b"beta");
        inverted.invert_match = true;
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::LineHashes]);
        let (output, ..) = search_whole(b"alpha\nbeta\n", &inverted, &Reporting::Print(config));
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(r#""line_hash":""#), "{text}");
        assert!(text.contains(r#""submatches":[]"#), "{text}");
    }

    #[test]
    fn names_the_whole_file_once_beside_its_name() {
        let search = make_search(b"error");
        let corpus = b"alpha error one\nbeta error two\n";
        let mut buffer = [0u8; HASH_CHARS];
        let expected = format_hash(&mut buffer, content_hash(corpus), HASH_CHARS).to_string();

        let mut json = make_config();
        json.output_format = OutputFormat::Json;
        json.fields = Field::bits(&[Field::FileHash]);
        let (output, ..) = search_whole(corpus, &search, &Reporting::Print(json));
        let text = String::from_utf8(output).unwrap();
        assert_eq!(
            text.matches(&format!(r#""file_hash":"{expected}""#))
                .count(),
            1,
            "the file's hash belongs on the one record that names the file:\n{text}"
        );

        let mut heading = make_config();
        heading.output_format = OutputFormat::Heading;
        heading.fields = Field::bits(&[Field::FileHash]);
        let (output, ..) = search_whole(corpus, &search, &Reporting::Print(heading));
        let text = String::from_utf8(output).unwrap();
        assert_eq!(text.matches(&expected).count(), 1, "{text}");
        // The token it prints is the one `sz-replace --expect-hash` accepts.
        assert!(parse_content_hash(&expected).is_ok());
    }

    #[test]
    fn streams_a_file_hash_that_a_closing_record_can_carry() {
        // JSON names the file again when it closes it, which a stream reaches; a heading
        // names it on the first match, which a stream has not read past yet.
        let search = make_search(b"error");

        let mut json = make_config();
        json.output_format = OutputFormat::Json;
        json.fields = Field::bits(&[Field::FileHash]);
        assert!(can_stream(&search, &Reporting::Print(json)));

        let mut heading = make_config();
        heading.output_format = OutputFormat::Heading;
        heading.fields = Field::bits(&[Field::FileHash]);
        assert!(!can_stream(&search, &Reporting::Print(heading)));
    }

    #[test]
    fn counts_the_lines_it_searched_rather_than_the_windows_it_read() {
        // A run that asked for the file's hash keeps reading after the cap is reached. It
        // must not keep *counting*: re-entering the search once per remaining window added
        // one line each time, so a 60,000-line stream reported six.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::FileHash]);
        let mut search = make_search(b"error");
        search.max_matches = Some(1);
        let corpus = b"alpha error\nbeta\ngamma\ndelta\nepsilon\n";

        let (hashed, ..) = search_streamed(corpus, 7, &search, &Reporting::Print(config));
        let mut plain = make_config();
        plain.output_format = OutputFormat::Json;
        let (unhashed, ..) = search_streamed(corpus, 7, &search, &Reporting::Print(plain));

        let searched = |output: Vec<u8>| {
            String::from_utf8(output)
                .unwrap()
                .split(r#""lines_searched":"#)
                .nth(1)
                .and_then(|rest| rest.split('}').next())
                .expect("the closing record reports what was searched")
                .to_string()
        };
        assert_eq!(searched(hashed), searched(unhashed));
    }

    #[test]
    fn opens_a_json_file_record_without_naming_it() {
        // The file's hash goes on the record that closes a file, which a stream reaches;
        // the one that opens it is written long before the bytes it would cover.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::FileHash]);
        let (output, ..) = search_whole(
            b"alpha error\n",
            &make_search(b"error"),
            &Reporting::Print(config),
        );
        let text = String::from_utf8(output).unwrap();
        let begin = text.lines().next().unwrap();
        assert!(begin.contains(r#""type":"begin""#), "{text}");
        assert!(!begin.contains("file_hash"), "{begin}");
        assert!(text.lines().last().unwrap().contains("file_hash"), "{text}");
    }

    #[test]
    fn names_the_whole_file_the_same_streamed_as_mapped() {
        // The hash a piped run reports has to be the hash of the file, not of the windows it
        // happened to be handed.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::FileHash]);
        let reporting = Reporting::Print(config);
        let search = make_search(b"error");
        let mut buffer = [0u8; HASH_CHARS];
        let file = format_hash(&mut buffer, content_hash(SEAM_CORPUS), HASH_CHARS);
        let expected = format!(r#""file_hash":"{file}""#);

        let (whole, ..) = search_whole(SEAM_CORPUS, &search, &reporting);
        assert!(String::from_utf8(whole).unwrap().contains(&expected));

        for capacity in [1, 7, 13, 64] {
            let (streamed, ..) = search_streamed(SEAM_CORPUS, capacity, &search, &reporting);
            let text = String::from_utf8(streamed).unwrap();
            assert!(text.contains(&expected), "capacity {capacity}:\n{text}");
        }
    }

    #[test]
    fn names_the_whole_file_even_when_the_search_stops_early() {
        // `--max-matches` leaves the rest of a pipe unread, and a hash of the prefix would
        // look exactly like a hash of the file. A run that asked for one reads to the end.
        let mut config = make_config();
        config.output_format = OutputFormat::Json;
        config.fields = Field::bits(&[Field::FileHash]);
        let mut search = make_search(b"error");
        search.max_matches = Some(1);

        let (streamed, ..) = search_streamed(SEAM_CORPUS, 7, &search, &Reporting::Print(config));
        let text = String::from_utf8(streamed).unwrap();
        let mut buffer = [0u8; HASH_CHARS];
        let file = format_hash(&mut buffer, content_hash(SEAM_CORPUS), HASH_CHARS);
        assert!(text.contains(&format!(r#""file_hash":"{file}""#)), "{text}");
        assert_eq!(text.matches(r#""type":"match""#).count(), 1);
    }

    #[test]
    fn annotates_every_line_of_a_multiline_region() {
        // Asking only for hashes still has to divide the region into lines; the condition
        // that decides used to list the other two columns by hand and would print a raw blob.
        let mut config = make_config();
        config.fields = Field::bits(&[Field::LineHashes]);
        let mut search = make_search(b"beta\ngamma");
        search.multiline = true;

        let (output, ..) = search_whole(
            b"alpha\nbeta\ngamma\ndelta\n",
            &search,
            &Reporting::Print(config),
        );
        let text = String::from_utf8(output).unwrap();

        let mut buffer = [0u8; HASH_CHARS];
        for line in [&b"beta\n"[..], &b"gamma\n"[..]] {
            let name = format_hash(&mut buffer, content_hash(line), DEFAULT_HASH_WIDTH);
            assert!(
                text.contains(&format!(
                    "{name}:{}",
                    String::from_utf8_lossy(line).trim_end()
                )),
                "the region printed without naming its lines:\n{text}"
            );
        }
    }
}

// endregion: Tests
