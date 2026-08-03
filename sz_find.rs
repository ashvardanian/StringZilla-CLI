//! SIMD-accelerated substring search utility
//!
//! A grep-like tool with simpler syntax, using StringZilla for fast searching.
//! Unlike grep, uses literal substring matching (not regex) for maximum speed.
//! For find-and-replace, use the dedicated `sz-replace` utility.
//!
//! # Examples
//!
//! ```bash
//! # Search single file
//! sz-find error log.txt
//!
//! # Search multiple files
//! sz-find error src/*.rs tests/*.rs
//!
//! # Search directory recursively
//! sz-find error src/
//!
//! # Case-insensitive search
//! sz-find --ignore-case ERROR log.txt
//!
//! # Show line numbers
//! sz-find --line-numbers error log.txt
//!
//! # Count matching lines only
//! sz-find --show count error log.txt
//!
//! # Show context lines
//! sz-find --context 2 error log.txt
//!
//! # Filter by file type
//! sz-find error src/ --type rust
//!
//! # Filter by glob pattern
//! sz-find error src/ --glob "*.rs"
//!
//! # Invert match (show non-matching lines)
//! sz-find --invert-match error log.txt
//!
//! # Match whole words only
//! sz-find --match word error log.txt
//!
//! # Stop after N matches
//! sz-find --max-matches 10 error log.txt
//!
//! # From stdin
//! cat log.txt | sz-find error
//! ```

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Read, Write};
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Range};
use std::path::Path;

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, ValueEnum};
use ignore::WalkBuilder;
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

    /// Lines of context before and after match
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

    /// Stop after NUM matches
    #[arg(long, value_parser = parse_at_least_one, help_heading = "Matching")]
    max_matches: Option<NonZeroUsize>,

    /// Suppress all output; exit 0 if any match was found, 1 otherwise
    #[arg(
        long,
        conflicts_with_all = [
            "format", "null", "line_numbers", "column_numbers", "byte_offsets", "heading",
            "color", "max_line_length", "context", "before_context", "after_context",
        ],
        help_heading = "Output Formats"
    )]
    quiet: bool,

    /// Report totals for the whole run
    #[arg(long, help_heading = "Output Formats")]
    summary: bool,

    /// Colorize output (auto, always, never)
    #[arg(long, default_value = "auto", help_heading = "Output Formats")]
    color: ColorChoice,

    /// Show line numbers
    #[arg(long, help_heading = "Output Formats")]
    line_numbers: bool,

    /// Show column numbers (1-based)
    #[arg(long, help_heading = "Output Formats")]
    column_numbers: bool,

    /// Show byte offset of each line
    #[arg(long, help_heading = "Output Formats")]
    byte_offsets: bool,

    /// Group matches by file with filename header
    #[arg(long, help_heading = "Output Formats")]
    heading: bool,

    /// Trim printed lines to NUM units, on a character boundary
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

    /// Maximum directory depth
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
    /// The name of every input that matched
    Files,
    /// The name of every input that did not match
    FilesWithout,
}

/// How records are rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Format {
    #[default]
    Text,
    Json,
    Vimgrep,
}

/// What the needle is matched against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum Match {
    #[default]
    Substring,
    Word,
}

/// Everything clap cannot express, in one place: its conflicts fire on a flag's presence,
/// never on its value. Every pair rejected here is inert by construction, not merely for
/// some inputs, which is why silence would misreport what the run did.
fn validate(args: &Args) -> Result<(), clap::Error> {
    let reject = |message: String| Args::command().error(ErrorKind::ArgumentConflict, message);
    if args.pattern.is_empty() {
        return Err(reject("pattern cannot be empty".to_string()));
    }

    // `--quiet` stops at the first match, where a cap changes nothing — except under
    // `--summary`, which keeps the tally running and whose totals the cap does bound.
    if args.quiet && args.max_matches.is_some() && !args.summary {
        return Err(reject(
            "--quiet stops at the first match, so it cannot be combined with --max-matches"
                .to_string(),
        ));
    }

    let show = args.show.unwrap_or_default();
    if args.invert_match && show == Show::Matches {
        return Err(reject(
            "--invert-match selects the lines that hold no match, so --show matches has nothing to print".to_string(),
        ));
    }

    // A record naming a whole file carries no position, no line and no trimmed text.
    if matches!(show, Show::Count | Show::Files | Show::FilesWithout) {
        let decorations = [
            (args.line_numbers, "--line-numbers"),
            (args.column_numbers, "--column-numbers"),
            (args.byte_offsets, "--byte-offsets"),
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

    // `--color auto` is the default, so any other value is an explicit request.
    let colored = !matches!(args.color, ColorChoice::Auto);
    let carried: Vec<(bool, &str)> = match args.format {
        // JSON escapes nothing, names its file in every record and reproduces whole lines.
        Format::Json => vec![
            (args.null, "--null"),
            (args.heading, "--heading"),
            (colored, "--color"),
            (args.max_line_length.is_some(), "--max-line-length"),
        ],
        // Vimgrep names its file, line and column in every record.
        Format::Vimgrep => vec![
            (args.heading, "--heading"),
            (args.line_numbers, "--line-numbers"),
            (args.column_numbers, "--column-numbers"),
            (args.byte_offsets, "--byte-offsets"),
            (colored, "--color"),
        ],
        Format::Text => return Ok(()),
    };
    if let Some((_, flag)) = carried.into_iter().find(|(present, _)| *present) {
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
    filename: &'static str,
    line_number: &'static str,
    column: &'static str,
    byte_offset: &'static str,
    match_highlight: &'static str,
    reset: &'static str,
    separator: &'static str,
}

impl Colors {
    fn enabled() -> Self {
        Self {
            filename: "\x1b[1;35m",        // Bold Magenta
            line_number: "\x1b[32m",       // Green
            column: "\x1b[32m",            // Green (same as line number)
            byte_offset: "\x1b[36m",       // Cyan
            match_highlight: "\x1b[1;31m", // Bold red
            reset: "\x1b[0m",
            separator: "\x1b[36m", // Cyan
        }
    }

    fn disabled() -> Self {
        Self {
            filename: "",
            line_number: "",
            column: "",
            byte_offset: "",
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

/// Everything the line emitters read, resolved once from the flags.
/// Only the printing paths hold one; [`SearchPath::Tally`] has no emitter.
#[derive(Clone, Copy)]
struct Printer {
    output_format: OutputFormat,
    colors: Colors,
    show_filename: bool,
    line_numbers: bool,
    column_numbers: bool,
    byte_offsets: bool,
    only_matches: bool,
    max_line_length: Option<usize>,
    /// Whether columns and `--max-line-length` are counted in characters rather than bytes.
    character_units: bool,
    /// What ends every emitted record.
    terminator: u8,
    /// Whether a selected line holds match positions to point at. `--invert-match` selects
    /// the lines that hold none, leaving columns and `--show matches` nothing to resolve.
    locates_matches: bool,
    /// Whether a printed matching line has its matches wrapped in color codes.
    highlight: bool,
}

impl Printer {
    /// Collapse the output flags into the form every emitter reads.
    fn new(args: &Args, show: Show, multiple_inputs: bool) -> Self {
        let output_format = match args.format {
            Format::Json => OutputFormat::Json,
            Format::Vimgrep => OutputFormat::Vimgrep,
            Format::Text if args.heading => OutputFormat::Heading,
            Format::Text => OutputFormat::Standard,
        };

        // JSON carries its own structure, so escape codes would corrupt it. `--quiet` and
        // the tallying selectors need no test here: they never build a `Printer`.
        let use_color = match args.color {
            ColorChoice::Always => true,
            ColorChoice::Never => false,
            ColorChoice::Auto => io::stdout().is_terminal() && output_format != OutputFormat::Json,
        };
        let colors = if use_color {
            Colors::enabled()
        } else {
            Colors::disabled()
        };
        let only_matches = show == Show::Matches;

        Self {
            output_format,
            colors,
            show_filename: multiple_inputs && output_format != OutputFormat::Heading,
            // Vimgrep implies line numbers and columns.
            line_numbers: args.line_numbers || output_format == OutputFormat::Vimgrep,
            column_numbers: args.column_numbers || output_format == OutputFormat::Vimgrep,
            byte_offsets: args.byte_offsets,
            only_matches,
            max_line_length: args.max_line_length.map(NonZeroUsize::get),
            character_units: uses_unicode(args),
            terminator: Terminator::from_null(args.null).as_byte(),
            locates_matches: !args.invert_match,
            // `--show matches`, JSON and vimgrep reproduce the match text themselves,
            // and an inverted line holds no match to wrap.
            highlight: !colors.match_highlight.is_empty()
                && !args.invert_match
                && !only_matches
                && output_format != OutputFormat::Json
                && output_format != OutputFormat::Vimgrep,
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
fn uses_unicode(args: &Args) -> bool {
    args.utf8 || args.ignore_case
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

/// Trim `line` to `limit` characters when `characters`, to `limit` bytes otherwise,
/// returning the kept prefix and whether anything was dropped.
fn trim_to_limit(line: &[u8], limit: usize, characters: bool) -> (&[u8], bool) {
    if !characters {
        return truncate_at_character(line, limit);
    }
    let mut seen = 0;
    for (index, byte) in line.iter().enumerate() {
        if (byte & 0xC0) != 0x80 {
            if seen == limit {
                return (&line[..index], true);
            }
            seen += 1;
        }
    }
    (line, false)
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
    /// 1-based column number, in characters under `--utf8` and in bytes otherwise.
    #[inline]
    fn column(&self, line: &[u8], characters: bool) -> usize {
        if characters {
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

/// What every search path reads: the needle, the line split, and the selection rules.
struct Search<'a> {
    matcher: Matcher<'a>,
    newlines: Newlines,
    multiline: bool,
    invert_match: bool,
    max_matches: Option<usize>,
}

impl Search<'_> {
    /// Whether the line belongs in the result, accounting for `--invert-match`.
    #[inline]
    fn selects(&self, line: &[u8]) -> bool {
        self.matcher.matches(line).next().is_some() != self.invert_match
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
    /// No surrounding lines, which is what every path but `PrintContext` carries.
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
enum SearchPath {
    /// Two counters. Serves `--quiet` and the three record kinds that name no line.
    Tally { stop_at_first: bool },
    /// Counters plus a heading bit. Serves every format without context lines.
    Print(Printer),
    /// Counters, heading, a look-behind ring and group separators. Serves the context flags.
    PrintContext(Printer, Context),
}

impl SearchPath {
    /// Pick the path: existence only, plain printing, or printing with context.
    fn choose(args: &Args, show: Show, context: Context, multiple_inputs: bool) -> Self {
        // `--show count` must see every line. Existence answers need only one hit — except
        // under `--summary`, whose totals would otherwise describe a prefix of the file.
        let tally = |stop_at_first: bool| SearchPath::Tally {
            stop_at_first: stop_at_first && !args.summary,
        };
        if args.quiet {
            return tally(true);
        }
        match show {
            Show::Count => tally(false),
            Show::Files | Show::FilesWithout => tally(true),
            Show::Lines | Show::Matches => {
                let printer = Printer::new(args, show, multiple_inputs);
                if context.before == 0 && context.after == 0 {
                    SearchPath::Print(printer)
                } else {
                    SearchPath::PrintContext(printer, context)
                }
            }
        }
    }

    /// The emitter this path prints through, absent for the tally.
    #[inline]
    fn printer(&self) -> Option<&Printer> {
        match self {
            SearchPath::Tally { .. } => None,
            SearchPath::Print(printer) | SearchPath::PrintContext(printer, _) => Some(printer),
        }
    }

    /// The surrounding lines this path prints, none for the two that print none.
    #[inline]
    fn context(&self) -> Context {
        match self {
            SearchPath::PrintContext(_, context) => *context,
            SearchPath::Tally { .. } | SearchPath::Print(_) => Context::none(),
        }
    }
}

// endregion: Search Paths

// region: Line Loops

/// Search result for a single file
struct FileResult {
    match_count: usize,
    lines_searched: usize,
    bytes_searched: usize,
}

/// A line held for look-behind: its number, and its span within the current window,
/// which the whole-slice paths open at byte zero.
struct ContextLine {
    line_number: usize,
    span: Range<usize>,
}

/// The look-behind ring, pending after-context and group separator that the context
/// flags carry from one line to the next.
struct ContextState {
    lines: Context,
    /// Window offsets rather than copies, and `lines.before` entries is the exact ceiling.
    /// Empty under `--after-context` alone, which is what lets that path stream.
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

/// What a search path remembers between lines, and between windows once the input
/// streams: the counters, the once-per-file heading bit and the context ring.
struct Progress {
    match_count: usize,
    lines_searched: usize,
    /// Offset of the current window's first byte within the whole input, which keeps
    /// `--byte-offsets` and the JSON and vimgrep formats absolute across a streamed run.
    window_base: usize,
    printed_heading: bool,
    /// Grown on the first highlighted line and reused for every later one.
    highlight_buffer: Vec<u8>,
    context: ContextState,
}

impl Progress {
    /// Start a file at line zero, byte zero, with nothing printed yet.
    fn new(lines: Context) -> Self {
        Progress {
            match_count: 0,
            lines_searched: 0,
            window_base: 0,
            printed_heading: false,
            highlight_buffer: Vec::new(),
            context: ContextState::new(lines),
        }
    }

    /// Whether `--max-matches` is already satisfied, where every path stops reporting.
    #[inline]
    fn exhausted(&self, search: &Search) -> bool {
        search
            .max_matches
            .is_some_and(|max| self.match_count >= max)
    }

    /// The totals this file contributes, over `bytes_searched` bytes of input.
    fn result(&self, bytes_searched: usize) -> FileResult {
        FileResult {
            match_count: self.match_count,
            lines_searched: self.lines_searched,
            bytes_searched,
        }
    }
}

/// Search one whole slice, which is the shape a mapped or buffered input already has.
fn search_slice(
    data: &[u8],
    filename: &str,
    search: &Search,
    path: &SearchPath,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<FileResult> {
    if search.multiline {
        return search_multiline(data, filename, search, path, output, max_reached);
    }
    let mut progress = Progress::new(path.context());
    // One window spans the whole input, so there is nothing left to stop for.
    let _ = search_window(
        data,
        filename,
        search,
        path,
        &mut progress,
        output,
        max_reached,
    )?;
    print_json_end(output, filename, path, &progress)?;
    Ok(progress.result(data.len()))
}

/// Run one window through the chosen path, folding its lines into `progress`.
/// Breaks once the path has reported everything it will, which ends a streamed read.
fn search_window(
    window: &[u8],
    filename: &str,
    search: &Search,
    path: &SearchPath,
    progress: &mut Progress,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<ControlFlow<()>> {
    match path {
        SearchPath::Tally { stop_at_first } => Ok(tally_window(
            window,
            search,
            *stop_at_first,
            progress,
            max_reached,
        )),
        SearchPath::Print(printer) => print_window(
            window,
            filename,
            search,
            printer,
            progress,
            output,
            max_reached,
        ),
        SearchPath::PrintContext(printer, _) => print_context_window(
            window,
            filename,
            search,
            printer,
            progress,
            output,
            max_reached,
        ),
    }
}

/// Count matching lines, stopping at the first when only existence is reported.
/// Carries two counters, so it allocates nothing and resolves no offsets.
fn tally_window(
    window: &[u8],
    search: &Search,
    stop_at_first: bool,
    progress: &mut Progress,
    max_reached: &mut bool,
) -> ControlFlow<()> {
    for line in LineIter::new(window, search.newlines) {
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        progress.lines_searched += 1;
        if progress.exhausted(search) {
            *max_reached = true;
            return ControlFlow::Break(());
        }

        if search.selects(line) {
            progress.match_count += 1;
            if stop_at_first {
                return ControlFlow::Break(());
            }
        }
    }
    ControlFlow::Continue(())
}

/// Print matching lines, carrying counters and the once-per-file heading bit.
fn print_window(
    window: &[u8],
    filename: &str,
    search: &Search,
    printer: &Printer,
    progress: &mut Progress,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<ControlFlow<()>> {
    let mut emitter = Emitter {
        output,
        filename,
        search,
        printer,
    };
    for line in LineIter::new(window, search.newlines) {
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        progress.lines_searched += 1;
        if progress.exhausted(search) {
            *max_reached = true;
            return Ok(ControlFlow::Break(()));
        }

        if !search.selects(line) {
            continue;
        }
        progress.match_count += 1;

        let record = LineRecord {
            line,
            line_number: progress.lines_searched,
            // Taken from the slice itself. Accumulating `line.len() + 1` assumes a
            // one-byte terminator and is wrong by one per CR, two per LS or PS.
            byte_offset: progress.window_base + offset_within(window, line),
            is_match: true,
        };
        emitter.file_heading(&mut progress.printed_heading)?;
        emitter.match_line(record, &mut progress.highlight_buffer)?;
    }
    Ok(ControlFlow::Continue(()))
}

/// Print matching lines with their surrounding context, carrying a look-behind ring
/// of window offsets, the pending after-context count, and separators.
fn print_context_window(
    window: &[u8],
    filename: &str,
    search: &Search,
    printer: &Printer,
    progress: &mut Progress,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<ControlFlow<()>> {
    let mut emitter = Emitter {
        output,
        filename,
        search,
        printer,
    };
    for line in LineIter::new(window, search.newlines) {
        // Counted before the limit is tested, so `--summary` includes the line that trips the cap.
        progress.lines_searched += 1;
        if progress.exhausted(search) {
            *max_reached = true;
            return Ok(ControlFlow::Break(()));
        }

        let line_number = progress.lines_searched;
        // Taken from the slice itself. Accumulating `line.len() + 1` assumes a
        // one-byte terminator and is wrong by one per CR, two per LS or PS.
        let line_start = offset_within(window, line);
        let byte_offset = progress.window_base + line_start;

        if search.selects(line) {
            progress.match_count += 1;
            emitter.file_heading(&mut progress.printed_heading)?;

            // The group opens at its own first line — the oldest look-behind line still
            // unprinted, or the matching line when the ring holds none — and a gap
            // between that line and the last printed one divides the two with `--`,
            // which is where `grep -C` writes its divider.
            let group_start = progress
                .context
                .first_unprinted_behind()
                .unwrap_or(line_number);
            let follows_a_gap = progress
                .context
                .last_printed_line
                .is_some_and(|last| group_start > last + 1);
            if follows_a_gap {
                emitter.group_separator()?;
            }

            // Print the buffered look-behind lines, sliced from the current window.
            for buffered in progress.context.behind.iter() {
                if progress.context.is_unprinted(buffered.line_number) {
                    emitter.line(LineRecord {
                        line: &window[buffered.span.start..buffered.span.end],
                        line_number: buffered.line_number,
                        byte_offset: progress.window_base + buffered.span.start,
                        is_match: false,
                    })?;
                    progress.context.last_printed_line = Some(buffered.line_number);
                }
            }

            emitter.match_line(
                LineRecord {
                    line,
                    line_number,
                    byte_offset,
                    is_match: true,
                },
                &mut progress.highlight_buffer,
            )?;
            progress.context.last_printed_line = Some(line_number);
            progress.context.pending_after = progress.context.lines.after;
        } else if progress.context.pending_after > 0 {
            emitter.line(LineRecord {
                line,
                line_number,
                byte_offset,
                is_match: false,
            })?;
            progress.context.last_printed_line = Some(line_number);
            progress.context.pending_after -= 1;
        }

        // Maintain the rolling look-behind ring.
        if progress.context.lines.before > 0 {
            if progress.context.behind.len() >= progress.context.lines.before {
                progress.context.behind.pop_front();
            }
            progress.context.behind.push_back(ContextLine {
                line_number,
                span: line_start..line_start + line.len(),
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

/// Search using multi-line matching (whole buffer search)
fn search_multiline(
    data: &[u8],
    filename: &str,
    search: &Search,
    path: &SearchPath,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<FileResult> {
    // The emitter, the tally's early stop and the context width hold for the whole file.
    let stop_at_first = matches!(
        path,
        SearchPath::Tally {
            stop_at_first: true
        }
    );
    let context = path.context();
    // Groups are divided only where surrounding lines are printed, as in line mode.
    let separates_groups = context.before > 0 || context.after > 0;
    // Reborrowed rather than moved, so the sink is still reachable for the closing
    // JSON record once every region is printed.
    let mut emitter = path.printer().map(|printer| Emitter {
        output: &mut *output,
        filename,
        search,
        printer,
    });

    // Each region carries its own surrounding lines, so this path keeps no look-behind
    // ring, and the whole file is one window that counts its lines up front.
    let mut progress = Progress::new(Context::none());
    progress.lines_searched = data.sz_matches(b"\n").count() + 1;
    let mut scanned_through = 0;
    // One past the terminator of the last printed region, absent until one is printed.
    let mut last_printed_end: Option<usize> = None;
    // Regions arrive in increasing order, which is what lets the line number carry.
    let mut line_numbers = LineCursor::default();

    for found in search.matcher.matches(data) {
        if progress.exhausted(search) {
            break;
        }
        progress.match_count += 1;
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
        emitter.file_heading(&mut progress.printed_heading)?;

        // Find line boundaries around the match
        let line_start = rfind(&data[..found.offset], b"\n").map_or(0, |offset| offset + 1);
        let line_end =
            find(&data[match_end..], b"\n").map_or(data.len(), |offset| match_end + offset);

        // Expand for before context
        let mut output_start = line_start;
        if context.before > 0 {
            let mut expanded = 0;
            let mut search_start = line_start;
            while expanded < context.before && search_start > 0 {
                search_start = if search_start <= 1 {
                    0
                } else {
                    rfind(&data[..search_start - 1], b"\n")
                        .map(|offset| offset + 1)
                        .unwrap_or(0)
                };
                expanded += 1;
            }
            output_start = search_start;
        }

        // Expand for after context
        let mut output_end = line_end;
        if context.after > 0 {
            let mut expanded = 0;
            let mut search_end = line_end;
            while expanded < context.after && search_end < data.len() {
                if let Some(next_newline) = find(&data[search_end + 1..], b"\n") {
                    search_end = search_end + 1 + next_newline;
                } else {
                    search_end = data.len();
                    break;
                }
                expanded += 1;
            }
            output_end = search_end;
        }

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
            if emitter.printer.line_numbers
                || emitter.printer.byte_offsets
                || emitter.printer.output_format != OutputFormat::Standard
            {
                let line_number = line_numbers.line_at(data, actual_start);
                for (index, line) in LineIter::new(region, Newlines::Lf).enumerate() {
                    emitter.line(LineRecord {
                        line,
                        line_number: line_number + index,
                        byte_offset: actual_start + offset_within(region, line),
                        is_match: true,
                    })?;
                }
            } else {
                emitter.region(region)?;
            }
            last_printed_end = Some(region_end);
        }
    }

    print_json_end(output, filename, path, &progress)?;

    // `--max-matches` caps this file, and reaching the cap ends the walk over the rest —
    // but only with input still unread, which is where the line paths stop as well.
    if progress.exhausted(search) && scanned_through < data.len() {
        *max_reached = true;
    }

    Ok(progress.result(data.len()))
}

// endregion: Line Loops

// region: Streaming

/// Whether a true pipe can be searched through a bounded window rather than drained whole.
///
/// `--multiline` searches the whole input backward and `--before-context` reaches back past
/// the window, so those two keep every byte. `--after-context` alone only reaches forward.
fn can_stream(search: &Search, path: &SearchPath) -> bool {
    !search.multiline
        && match path {
            SearchPath::Tally { .. } | SearchPath::Print(_) => true,
            SearchPath::PrintContext(_, context) => context.before == 0,
        }
}

/// Drive the chosen path over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
///
/// A match cannot cross a newline here — the paths search within each line — so cutting
/// at [`last_cut`] needs no carry, not even under `--ignore-case`. `--multiline` is where a
/// match does cross, and it never reaches this driver.
fn search_stream<R: Read>(
    refill: &mut Refill<R>,
    filename: &str,
    search: &Search,
    path: &SearchPath,
    output: &mut dyn Write,
    max_reached: &mut bool,
) -> io::Result<FileResult> {
    debug_assert!(can_stream(search, path), "this path reads earlier lines");

    let mut progress = Progress::new(path.context());
    refill.try_for_each_window(search.newlines.into(), |window| {
        let flow = search_window(
            window,
            filename,
            search,
            path,
            &mut progress,
            output,
            max_reached,
        )?;
        // Counted before the flow is honoured, so a stop still reports the window it read.
        progress.window_base += window.len();
        Ok(flow)
    })?;

    print_json_end(output, filename, path, &progress)?;
    // An early stop leaves the rest of the pipe unread, so the total describes what
    // was searched rather than what the writer still holds.
    Ok(progress.result(progress.window_base))
}

// endregion: Streaming

// region: Line Output

/// One line as an emitter sees it: its bytes, where it sits, and how it was selected.
#[derive(Clone, Copy)]
struct LineRecord<'a> {
    line: &'a [u8],
    line_number: usize,
    byte_offset: usize,
    /// False for a surrounding context line, which prints `-` where a match prints `:`.
    is_match: bool,
}

/// Everything a printed line reads beyond the line itself: the sink, the file it came
/// from, the pattern its matches are resolved against, and the resolved output flags.
struct Emitter<'a, 'p> {
    output: &'a mut dyn Write,
    filename: &'a str,
    search: &'a Search<'p>,
    printer: &'a Printer,
}

impl Emitter<'_, '_> {
    /// Emit the once-per-file header the format calls for, on its first printed match.
    fn file_heading(&mut self, printed: &mut bool) -> io::Result<()> {
        if *printed {
            return Ok(());
        }
        let (filename, printer) = (self.filename, self.printer);
        match printer.output_format {
            OutputFormat::Heading => {
                writeln!(
                    self.output,
                    "{}{}{}",
                    printer.colors.filename, filename, printer.colors.reset
                )?;
                *printed = true;
            }
            OutputFormat::Json => {
                self.output
                    .write_all(br#"{"type":"begin","data":{"path":"#)?;
                json_text_field_to(self.output, filename.as_bytes())?;
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
        let printer = self.printer;
        if printer.output_format == OutputFormat::Json {
            return Ok(());
        }
        writeln!(
            self.output,
            "{}--{}",
            printer.colors.separator, printer.colors.reset
        )
    }

    /// Print a matching line, coloring its matches where the format allows.
    fn match_line(&mut self, record: LineRecord, highlight_buffer: &mut Vec<u8>) -> io::Result<()> {
        let (search, printer) = (self.search, self.printer);
        let highlighted = printer
            .highlight
            .then(|| highlight_line(record.line, search, printer, highlight_buffer))
            .flatten();
        // The colored bytes travel beside the record rather than inside it: the column
        // resolves against the line as it was found, and an escape code spliced in ahead
        // of a match would shift that column by its own length.
        self.write_line(record, highlighted.unwrap_or(record.line))
    }

    /// Print one line with the prefixes its format asks for.
    fn line(&mut self, record: LineRecord) -> io::Result<()> {
        self.write_line(record, record.line)
    }

    /// Print `body` — the line itself, or its highlighted form — behind the prefixes the
    /// format asks for, resolving every position against `record.line`.
    fn write_line(&mut self, record: LineRecord, body: &[u8]) -> io::Result<()> {
        let (filename, search, printer) = (self.filename, self.search, self.printer);
        let colors = printer.colors;
        // A context line is set off by `-` where a matching line uses `:`.
        let separator = if record.is_match { ":" } else { "-" };
        let locates_matches = record.is_match && printer.locates_matches;

        match printer.output_format {
            OutputFormat::Json => self.json_line(record),
            OutputFormat::Vimgrep => {
                // Vimgrep puts every match on its own `file:line:column:text` line.
                if locates_matches {
                    for found in search.matcher.matches(record.line) {
                        write!(
                            self.output,
                            "{}:{}:{}:",
                            filename,
                            record.line_number,
                            found.column(record.line, printer.character_units)
                        )?;
                        if printer.only_matches {
                            self.output.write_all(found.text(record.line))?;
                        } else {
                            self.write_trimmed(body)?;
                        }
                        self.output.write_all(&[printer.terminator])?;
                    }
                } else if record.is_match {
                    // An inverted match holds no position, so the line prints at column one.
                    write!(self.output, "{}:{}:1:", filename, record.line_number)?;
                    self.write_trimmed(body)?;
                    self.output.write_all(&[printer.terminator])?;
                }
                Ok(())
            }
            // `Printer::new` clears `show_filename` under `--heading`, where the name is a
            // header rather than a per-line prefix, so both formats print the same line.
            OutputFormat::Standard | OutputFormat::Heading => {
                if printer.show_filename {
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.filename, filename, colors.reset, separator
                    )?;
                }
                if printer.line_numbers {
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.line_number, record.line_number, colors.reset, separator
                    )?;
                }
                if printer.column_numbers && locates_matches {
                    let column = search
                        .matcher
                        .matches(record.line)
                        .next()
                        .map_or(1, |found| {
                            found.column(record.line, printer.character_units)
                        });
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.column, column, colors.reset, separator
                    )?;
                }
                if printer.byte_offsets {
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.byte_offset, record.byte_offset, colors.reset, separator
                    )?;
                }
                self.write_content(record, body)?;
                self.output.write_all(&[printer.terminator])
            }
        }
    }

    /// Print one line in JSON Lines format (ripgrep-compatible), writing straight to the
    /// sink so that no record is staged in a `String` first.
    fn json_line(&mut self, record: LineRecord) -> io::Result<()> {
        let (filename, search, printer) = (self.filename, self.search, self.printer);
        if !record.is_match {
            self.output
                .write_all(br#"{"type":"context","data":{"path":"#)?;
            json_text_field_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.line)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{}}}}}"#,
                record.line_number, record.byte_offset
            )?;
            self.output.write_all(b"\n")
        } else if !printer.locates_matches {
            // An inverted match holds no position, so it carries no submatches.
            self.output
                .write_all(br#"{"type":"match","data":{"path":"#)?;
            json_text_field_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.line)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{},"submatches":[]}}}}"#,
                record.line_number, record.byte_offset
            )?;
            self.output.write_all(b"\n")
        } else {
            self.output
                .write_all(br#"{"type":"match","data":{"path":"#)?;
            json_text_field_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#","lines":"#)?;
            json_text_field_to(self.output, record.line)?;
            write!(
                self.output,
                r#","line_number":{},"absolute_offset":{},"submatches":["#,
                record.line_number, record.byte_offset
            )?;

            for (index, found) in search.matcher.matches(record.line).enumerate() {
                if index > 0 {
                    self.output.write_all(b",")?;
                }
                self.output.write_all(br#"{"match":"#)?;
                json_text_field_to(self.output, found.text(record.line))?;
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

    /// Write a line's content, reduced to the matches themselves under `--show matches`.
    fn write_content(&mut self, record: LineRecord, body: &[u8]) -> io::Result<()> {
        let (search, printer) = (self.search, self.printer);
        if !(printer.only_matches && record.is_match && printer.locates_matches) {
            return self.write_trimmed(body);
        }
        for (index, found) in search.matcher.matches(record.line).enumerate() {
            if index > 0 {
                self.output.write_all(&[printer.terminator])?;
            }
            if !printer.colors.match_highlight.is_empty() {
                self.output
                    .write_all(printer.colors.match_highlight.as_bytes())?;
            }
            self.output.write_all(found.text(record.line))?;
            if !printer.colors.reset.is_empty() {
                self.output.write_all(printer.colors.reset.as_bytes())?;
            }
        }
        Ok(())
    }

    /// Write a whole line, trimmed to `--max-line-length` on a character boundary.
    /// One long minified line would otherwise flood a terminal or a context window.
    fn write_trimmed(&mut self, line: &[u8]) -> io::Result<()> {
        let Some(limit) = self.printer.max_line_length else {
            return self.output.write_all(line);
        };
        let (kept, trimmed) = trim_to_limit(line, limit, self.printer.character_units);
        self.output.write_all(kept)?;
        if trimmed {
            self.output.write_all(b" [...]")?;
        }
        Ok(())
    }

    /// Write a whole multi-line region verbatim, which is what `--multiline` prints when
    /// no per-line prefix is asked for.
    fn region(&mut self, region: &[u8]) -> io::Result<()> {
        let (filename, printer) = (self.filename, self.printer);
        if printer.show_filename {
            write!(
                self.output,
                "{}{}{}:",
                printer.colors.filename, filename, printer.colors.reset
            )?;
        }
        self.output.write_all(region)?;
        // An input whose last line is unterminated still prints as a whole line.
        if !region.ends_with(b"\n") {
            self.output.write_all(&[printer.terminator])?;
        }
        Ok(())
    }
}

/// Wrap every match in `line` with the highlight color, building into a reused buffer.
/// `None` when the line holds no match, leaving the caller to print the line itself.
fn highlight_line<'b>(
    line: &[u8],
    search: &Search,
    printer: &Printer,
    buffer: &'b mut Vec<u8>,
) -> Option<&'b [u8]> {
    // Peeked before the buffer is touched: a line with no match is printed as it stands.
    let mut matches = search.matcher.matches(line).peekable();
    matches.peek()?;

    buffer.clear();
    // Reserve capacity to minimize allocations during building
    buffer.reserve(line.len() + 64);
    let mut last_end = 0;
    for found in matches {
        buffer.extend_from_slice(&line[last_end..found.offset]);
        buffer.extend_from_slice(printer.colors.match_highlight.as_bytes());
        buffer.extend_from_slice(found.text(line));
        buffer.extend_from_slice(printer.colors.reset.as_bytes());
        last_end = found.offset + found.length;
    }
    buffer.extend_from_slice(&line[last_end..]);
    Some(buffer)
}

/// Close the JSON Lines record for a file that opened one, once every window is done.
fn print_json_end(
    output: &mut dyn Write,
    filename: &str,
    path: &SearchPath,
    progress: &Progress,
) -> io::Result<()> {
    let Some(printer) = path.printer() else {
        return Ok(());
    };
    if printer.output_format != OutputFormat::Json || !progress.printed_heading {
        return Ok(());
    }
    output.write_all(br#"{"type":"end","data":{"path":"#)?;
    json_text_field_to(output, filename.as_bytes())?;
    write!(
        output,
        r#","stats":{{"matches":{},"lines_searched":{}}}}}}}"#,
        progress.match_count, progress.lines_searched
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

/// The name stdin reports itself under, in records and in diagnostics.
const STDIN_NAME: &str = "-";

/// The run's totals and the questions the exit code asks of them.
#[derive(Default)]
struct Outcome {
    summary: Summary,
    max_reached: bool,
    /// Whether the run produced the thing `--show` asks for.
    found: bool,
    read_any: bool,
    failed_any: bool,
}

/// What every input of this run is searched and reported with.
struct Session<'a> {
    search: Search<'a>,
    path: SearchPath,
    show: Show,
    format: Format,
    terminator: u8,
    /// Whether a per-file record carries its file name, as a walk's records must.
    named: bool,
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
        filename: &str,
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
            Show::Lines | Show::Matches => outcome.found |= matched,
            Show::Count => {
                outcome.found |= matched;
                // Only files with matches are reported over a walk, as `grep -c` and
                // `rg -c` do: listing every `path:0` buries the answer.
                if matched || !self.named {
                    self.write_count(output, filename, result.match_count)?;
                }
            }
            Show::Files => {
                outcome.found |= matched;
                if matched {
                    self.write_path(output, filename)?;
                }
            }
            Show::FilesWithout => {
                if !matched {
                    outcome.found = true;
                    self.write_path(output, filename)?;
                }
            }
        }
        Ok(())
    }

    /// Emit one input's matching-line count.
    fn write_count(&self, output: &mut dyn Write, filename: &str, count: usize) -> io::Result<()> {
        if self.format == Format::Json {
            output.write_all(br#"{"type":"count","data":{"path":"#)?;
            json_text_field_to(output, filename.as_bytes())?;
            return writeln!(output, r#","count":{}}}}}"#, count);
        }
        if self.named {
            write!(output, "{}:{}", filename, count)?;
        } else {
            write!(output, "{}", count)?;
        }
        output.write_all(&[self.terminator])
    }

    /// Emit one input's name.
    fn write_path(&self, output: &mut dyn Write, filename: &str) -> io::Result<()> {
        if self.format == Format::Json {
            output.write_all(br#"{"type":"file","data":{"path":"#)?;
            json_text_field_to(output, filename.as_bytes())?;
            return output.write_all(b"}}\n");
        }
        output.write_all(filename.as_bytes())?;
        output.write_all(&[self.terminator])
    }
}

/// Search stdin: a redirect maps, and a true pipe streams through one reused window
/// unless the path reads lines the window has already dropped.
fn search_stdin(
    session: &Session,
    output: &mut dyn Write,
    outcome: &mut Outcome,
) -> io::Result<()> {
    let (search, path) = (&session.search, &session.path);
    let streams = can_stream(search, path);
    let source = if streams {
        get_input_streaming(None)
    } else {
        get_input(None)
    };
    let source = match source {
        Ok(source) => source,
        Err(error) => {
            eprintln!("sz-find: {}: {}", STDIN_NAME, error);
            outcome.failed_any = true;
            return Ok(());
        }
    };
    outcome.read_any = true;
    outcome.summary.files_searched += 1;

    let result = match source.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(source) => {
            let data = source.as_bytes();
            if session.skips(data) {
                return Ok(());
            }
            search_slice(
                data,
                STDIN_NAME,
                search,
                path,
                output,
                &mut outcome.max_reached,
            )?
        }
        InputWindow::Stream(mut refill) => {
            // The first window is filled up front so that `--binary` reads the same on a
            // pipe as it does on a file; `try_for_each_window` refills without consuming.
            refill.advance(0)?;
            if session.skips(refill.filled()) {
                return Ok(());
            }
            search_stream(
                &mut refill,
                STDIN_NAME,
                search,
                path,
                output,
                &mut outcome.max_reached,
            )?
        }
    };

    outcome.summary.bytes_searched += result.bytes_searched;
    session.report(output, STDIN_NAME, &result, outcome)
}

/// Search every file one input names, which is the input itself when it is a file.
fn search_tree(
    session: &Session,
    args: &Args,
    globs: Option<&[glob::Pattern]>,
    input: &str,
    output: &mut dyn Write,
    outcome: &mut Outcome,
) -> io::Result<()> {
    // Checked here rather than left to the walker, whose error names the path twice.
    if let Err(error) = std::fs::metadata(input) {
        eprintln!("sz-find: {}: {}", input, error);
        outcome.failed_any = true;
        return Ok(());
    }

    for entry in build_walker(input, args) {
        if outcome.max_reached {
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                eprintln!("sz-find: {}", error);
                outcome.failed_any = true;
                continue;
            }
        };
        if !is_readable_entry(&entry) {
            continue;
        }

        // The walker has no glob filter of its own, so `--glob` is applied here.
        if let Some(globs) = globs {
            let path_text = entry.path().to_string_lossy();
            let name_text = entry.file_name().to_string_lossy();
            let selected = globs
                .iter()
                .any(|pattern| pattern.matches(&path_text) || pattern.matches(&name_text));
            if !selected {
                continue;
            }
        }

        let source = match open_input(entry.path()) {
            Ok(source) => source,
            Err(error) => {
                eprintln!("sz-find: {}: {}", entry.path().display(), error);
                outcome.failed_any = true;
                continue;
            }
        };
        outcome.read_any = true;
        let data = source.as_bytes();
        outcome.summary.files_searched += 1;
        outcome.summary.bytes_searched += data.len();
        if session.skips(data) {
            continue;
        }

        let filename = entry.path().to_string_lossy();
        let result = search_slice(
            data,
            &filename,
            &session.search,
            &session.path,
            output,
            &mut outcome.max_reached,
        )?;
        session.report(output, &filename, &result, outcome)?;
    }
    Ok(())
}

/// Build the directory walker with all filters
fn build_walker(input: &str, args: &Args) -> ignore::Walk {
    let mut builder = WalkBuilder::new(input);

    builder
        .hidden(!args.hidden)
        .git_ignore(!args.no_ignore)
        .git_global(!args.no_ignore)
        .git_exclude(!args.no_ignore)
        .follow_links(args.follow);

    if let Some(depth) = args.max_depth {
        builder.max_depth(Some(depth));
    }

    // Add file type filters
    if let Some(ref types) = args.file_type {
        let mut types_builder = ignore::types::TypesBuilder::new();
        types_builder.add_defaults();
        for t in types {
            types_builder.select(t);
        }
        match types_builder.build() {
            Ok(types_matcher) => {
                builder.types(types_matcher);
            }
            Err(e) => {
                eprintln!("sz-find: warning: invalid file type filter: {}", e);
            }
        }
    }

    // Note: glob filtering is handled manually in the main loop
    // since WalkBuilder doesn't have a direct glob filter API

    builder.build()
}

/// Report the run's totals, on stdout beside the records they describe. Under
/// `--format json` they are one more record, so the stream still parses line by line.
fn print_summary(
    output: &mut dyn Write,
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
    writeln!(output)?;
    writeln!(output, "Summary:")?;
    writeln!(output, "  Files searched: {}", summary.files_searched)?;
    writeln!(output, "  Files matched:  {}", summary.files_matched)?;
    writeln!(output, "  Lines searched: {}", summary.lines_searched)?;
    writeln!(output, "  Matches found:  {}", summary.matches_found)?;
    writeln!(
        output,
        "  Bytes searched: {} ({:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    )?;
    writeln!(output, "  Time elapsed:   {:.3}s", elapsed.as_secs_f64())?;
    if elapsed.as_secs_f64() > 0.0 {
        writeln!(
            output,
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
    report("sz-find", run(&args, &mut output))
}

/// Search every input in turn and answer with the status the run earned.
fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
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

    // A record names its file once the run reads more than one.
    let named = args.inputs.len() > 1
        || args
            .inputs
            .iter()
            .any(|input| input != "-" && Path::new(input).is_dir());

    let pattern = args.pattern.as_bytes();
    let unicode = uses_unicode(args);
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
        },
        // One dispatch decision for the whole run: what the per-line loop remembers.
        path: SearchPath::choose(args, show, context, named),
        show,
        format: args.format,
        terminator: Terminator::from_null(args.null).as_byte(),
        named,
        binary: args.binary,
    };

    // Compile each `--glob` once, so a malformed one is reported here rather than
    // silently matching nothing on every file of the walk.
    let globs = args.glob.as_ref().map(|globs| {
        globs
            .iter()
            .filter_map(|glob| match glob::Pattern::new(glob) {
                Ok(pattern) => Some(pattern),
                Err(error) => {
                    eprintln!("sz-find: warning: invalid glob '{}': {}", glob, error);
                    None
                }
            })
            .collect::<Vec<_>>()
    });

    let mut outcome = Outcome::default();
    for input in &args.inputs {
        if outcome.max_reached {
            break;
        }
        // Only reads of stdin and writes of stdout reach here, both named `-`; a file that
        // failed to open was already reported and folded into `outcome`.
        if input == "-" {
            search_stdin(&session, output, &mut outcome).at(STDIN_NAME)?;
        } else {
            search_tree(
                &session,
                args,
                globs.as_deref(),
                input,
                output,
                &mut outcome,
            )
            .at(STDIN_NAME)?;
        }
    }

    if args.summary {
        print_summary(output, args.format, &outcome.summary, started.elapsed()).at(STDIN_NAME)?;
    }
    output.flush().at(STDIN_NAME)?;

    // Every input failed to open, so the run did not complete.
    if outcome.failed_any && !outcome.read_any {
        return Ok(Status::Error);
    }
    Ok(Status::from_found(outcome.found))
}

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

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
        }
    }

    fn make_printer() -> Printer {
        Printer {
            output_format: OutputFormat::Standard,
            colors: Colors::disabled(),
            show_filename: false,
            line_numbers: false,
            column_numbers: false,
            byte_offsets: false,
            only_matches: false,
            max_line_length: None,
            character_units: false,
            terminator: b'\n',
            locates_matches: true,
            highlight: false,
        }
    }

    /// Search a whole slice, returning what it printed and what it counted.
    fn search_whole(data: &[u8], search: &Search, path: &SearchPath) -> (Vec<u8>, usize, usize) {
        let mut output = Vec::new();
        let mut max_reached = false;
        let result = search_slice(
            data,
            "test.txt",
            search,
            path,
            &mut output,
            &mut max_reached,
        )
        .unwrap();
        (output, result.match_count, result.lines_searched)
    }

    /// Search the same way, but through a window of exactly `capacity` bytes.
    fn search_streamed(
        data: &[u8],
        capacity: usize,
        search: &Search,
        path: &SearchPath,
    ) -> (Vec<u8>, usize, usize) {
        let mut refill = Refill::new(data, capacity);
        let mut output = Vec::new();
        let mut max_reached = false;
        let result = search_stream(
            &mut refill,
            "test.txt",
            search,
            path,
            &mut output,
            &mut max_reached,
        )
        .unwrap();
        (output, result.match_count, result.lines_searched)
    }

    /// A reader that fails once a budget of bytes has been handed out, so a test can prove
    /// a path stopped reading rather than merely stopped printing.
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
                "line-numbers",
                "column-numbers",
                "byte-offsets",
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
            vec!["--quiet", "--line-numbers"],
            vec!["--quiet", "--column-numbers"],
            vec!["--quiet", "--byte-offsets"],
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
            vec!["--format", "vimgrep", "--heading"],
            vec!["--format", "vimgrep", "--line-numbers"],
            vec!["--format", "vimgrep", "--byte-offsets"],
            vec!["--format", "vimgrep", "--color", "always"],
            vec!["--show", "count", "--line-numbers"],
            vec!["--show", "count", "--heading"],
            vec!["--show", "files", "--line-numbers"],
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

        let mut printed = Vec::new();
        assert!(matches!(run(&args, &mut printed), Ok(Status::Success)));
        let printed = String::from_utf8(printed).unwrap();
        assert!(printed.contains("Matches found:  2"), "{}", printed);
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
        let path = SearchPath::choose(&args, Show::Lines, Context::none(), false);
        assert!(matches!(
            path,
            SearchPath::Tally {
                stop_at_first: true
            }
        ));
        assert!(path.printer().is_none());
    }

    /// A session reporting `show` records in `format`, over a single unnamed input.
    fn make_session(show: Show, format: Format, null: bool) -> Session<'static> {
        Session {
            search: make_search(b"error"),
            path: SearchPath::Tally {
                stop_at_first: false,
            },
            show,
            format,
            terminator: Terminator::from_null(null).as_byte(),
            named: false,
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
        };
        session
            .report(&mut output, "log.txt", &result, &mut outcome)
            .unwrap();
        assert_eq!(
            outcome.found,
            match_count > 0 || session.show == Show::FilesWithout
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
                .chain([b'\n'])
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
        let mut printer = make_printer();
        printer.terminator = 0;
        let (printed, ..) = search_whole(
            b"error one\nplain\nerror two\n",
            &make_search(b"error"),
            &SearchPath::Print(printer),
        );
        assert_eq!(printed, b"error one\0error two\0");
    }

    #[test]
    fn honours_stdin_in_any_position() {
        // Each input is dispatched on its own, so `-` names stdin wherever it appears
        // rather than becoming a literal path once a second input follows it.
        let args = Args::try_parse_from(["sz-find", "error", "log.txt", "-"]).unwrap();
        assert_eq!(args.inputs, ["log.txt", "-"]);
        // Records name stdin with the same token that selects it.
        assert_eq!(STDIN_NAME, "-");
    }

    #[test]
    fn warns_past_a_missing_input_but_fails_when_none_was_readable() {
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
        assert!(matches!(run(&args, &mut output), Ok(Status::Success)));

        let args = Args::try_parse_from(["sz-find", "error", missing.to_str().unwrap()]).unwrap();
        assert!(matches!(run(&args, &mut io::sink()), Ok(Status::Error)));
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
        assert_eq!(found.column(line, false), 12);
        assert_eq!(found.column(line, true), 7);
        assert_eq!(trim_to_limit(line, 5, true), (&line[..10], true));
        assert_eq!(trim_to_limit(line, 6, false), (&line[..6], true));
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
        assert!(search.selects(line));

        search.invert_match = true;
        assert!(!search.selects(line));
        assert!(search.selects(b"goodbye world"));
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
        let path = SearchPath::Print(make_printer());
        let mut max_reached = false;
        let mut output = Vec::new();

        let result = search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
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
        let path = SearchPath::PrintContext(make_printer(), context);
        let mut max_reached = false;
        let mut output = Vec::new();

        search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
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
        let path = SearchPath::Print(make_printer());
        let mut max_reached = false;
        let mut output = Vec::new();

        let result = search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
        )
        .unwrap();

        assert_eq!(result.match_count, 2);
        assert!(max_reached);
    }

    #[test]
    fn counts_without_printing() {
        let data = b"error1\nok\nerror2\n";
        let search = make_search(b"error");
        let path = SearchPath::Tally {
            stop_at_first: false,
        };
        let mut max_reached = false;
        let mut output = Vec::new();

        let result = search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
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
        let path = SearchPath::Tally {
            stop_at_first: true,
        };
        let mut max_reached = false;
        let mut output = Vec::new();

        let result = search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
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
            &SearchPath::Tally {
                stop_at_first: true
            }
        ));
        assert!(can_stream(&search, &SearchPath::Print(make_printer())));
        assert!(can_stream(
            &search,
            &SearchPath::PrintContext(
                make_printer(),
                Context {
                    before: 0,
                    after: 3
                }
            )
        ));
        assert!(!can_stream(
            &search,
            &SearchPath::PrintContext(
                make_printer(),
                Context {
                    before: 1,
                    after: 0
                }
            )
        ));
    }

    #[test]
    fn wraps_matches_in_color_codes() {
        let line = b"hello world hello";
        let search = make_search(b"hello");
        let mut printer = make_printer();
        printer.colors = Colors::enabled();
        printer.highlight = true;

        let mut buffer = Vec::new();
        let highlighted = highlight_line(line, &search, &printer, &mut buffer).unwrap();
        let result = String::from_utf8(highlighted.to_vec()).unwrap();

        assert!(result.contains("\x1b[1;31m")); // Contains highlight
        assert!(result.contains("\x1b[0m")); // Contains reset

        // A line without a match leaves the buffer alone, so the caller prints the line.
        assert!(highlight_line(b"nothing here", &search, &printer, &mut buffer).is_none());
    }

    #[test]
    fn reports_the_same_column_with_and_without_color() {
        // Two matches on one line, so the colored form carries an escape code ahead of
        // each. `grep -bo` puts the first at byte 5, which is column 6.
        let data = b"lead error one error two\n";
        let search = make_search(b"error");
        let mut printer = make_printer();
        printer.column_numbers = true;

        let (plain, ..) = search_whole(data, &search, &SearchPath::Print(printer));
        assert_eq!(plain, b"6:lead error one error two\n");

        printer.colors = Colors::enabled();
        printer.highlight = true;
        let (colored, ..) = search_whole(data, &search, &SearchPath::Print(printer));
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
        let path = SearchPath::Print(make_printer());
        let mut max_reached = false;
        let mut output = Vec::new();

        let result = search_slice(
            data,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
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

        let path = SearchPath::PrintContext(make_printer(), context);
        let (printed, ..) = search_whole(data, &search, &path);
        assert_eq!(printed, b"\nbeta MATCH here\n\n");

        let mut printer = make_printer();
        printer.line_numbers = true;
        let path = SearchPath::PrintContext(printer, context);
        let (numbered, ..) = search_whole(data, &search, &path);
        assert_eq!(numbered, b"2:\n3:beta MATCH here\n4:\n");
    }

    #[test]
    fn terminates_every_multiline_region() {
        // Two matches far apart print as two whole lines, not one run-on line. The first
        // sits at byte zero, where the search for the line start has nothing to scan.
        let data = b"MATCH l1\nl2\nl3\nl4\nl5 MATCH\nl6\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let path = SearchPath::Print(make_printer());

        let (printed, ..) = search_whole(data, &search, &path);
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
        let path = SearchPath::PrintContext(make_printer(), context);

        let (printed, ..) = search_whole(data, &search, &path);
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
        let path = SearchPath::PrintContext(make_printer(), context);

        let (by_line, ..) = search_whole(data, &search, &path);
        search.multiline = true;
        let (by_region, ..) = search_whole(data, &search, &path);
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
        let path = SearchPath::PrintContext(make_printer(), context);

        let (by_line, ..) = search_whole(data, &search, &path);
        search.multiline = true;
        let (by_region, ..) = search_whole(data, &search, &path);
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
            let path = SearchPath::PrintContext(make_printer(), Context { before, after });
            let (printed, ..) = search_whole(data, &search, &path);
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
        let path = SearchPath::Print(make_printer());

        let (printed, ..) = search_whole(data, &search, &path);
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
        let mut printer = make_printer();
        printer.line_numbers = true;
        let path = SearchPath::Print(printer);

        let (printed, matches, _) = search_whole(&data, &search, &path);
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
        let path = SearchPath::Print(make_printer());
        assert!(can_stream(&search, &path));

        search.multiline = true;
        assert!(!can_stream(&search, &path));
    }

    #[test]
    fn streams_the_same_output_at_tiny_capacities() {
        let printer = make_printer();
        let paths = [
            SearchPath::Tally {
                stop_at_first: false,
            },
            SearchPath::Print(printer),
            SearchPath::PrintContext(
                printer,
                Context {
                    before: 0,
                    after: 2,
                },
            ),
        ];

        for newlines in [Newlines::Lf, Newlines::Unicode] {
            for path in &paths {
                let mut search = make_search(b"error");
                search.newlines = newlines;
                let expected = search_whole(SEAM_CORPUS, &search, path);
                for capacity in [1, 2, 3, 7, 13, 64, 4096] {
                    let streamed = search_streamed(SEAM_CORPUS, capacity, &search, path);
                    assert_eq!(streamed, expected, "capacity {}", capacity);
                }
            }
        }
    }

    #[test]
    fn keeps_line_numbers_and_offsets_absolute_across_windows() {
        let mut printer = make_printer();
        printer.line_numbers = true;
        printer.byte_offsets = true;
        printer.column_numbers = true;
        let path = SearchPath::Print(printer);
        let search = make_search(b"error");

        let (expected, ..) = search_whole(SEAM_CORPUS, &search, &path);
        for capacity in [1, 7, 13, 64] {
            let (streamed, ..) = search_streamed(SEAM_CORPUS, capacity, &search, &path);
            assert_eq!(streamed, expected, "capacity {}", capacity);
        }
        // The last line starts well past any of the tiny windows, so a window-relative
        // offset would report it near zero.
        let text = String::from_utf8(expected).unwrap();
        let last = text.lines().last().unwrap();
        let offset = SEAM_CORPUS.len() - b"zeta error last".len();
        assert_eq!(last, format!("8:6:{}:zeta error last", offset));
    }

    #[test]
    fn opens_and_closes_one_json_record_per_streamed_file() {
        let mut printer = make_printer();
        printer.output_format = OutputFormat::Json;
        let path = SearchPath::Print(printer);
        let search = make_search(b"error");

        let (streamed, ..) = search_streamed(SEAM_CORPUS, 7, &search, &path);
        let text = String::from_utf8(streamed).unwrap();
        assert_eq!(text.matches(r#""type":"begin""#).count(), 1);
        assert_eq!(text.matches(r#""type":"end""#).count(), 1);
    }

    #[test]
    fn opens_and_closes_one_json_record_per_multiline_file() {
        // A JSON consumer reads one record shape, whether or not `--multiline` was given.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta MATCH two\n";
        let mut printer = make_printer();
        printer.output_format = OutputFormat::Json;
        let path = SearchPath::Print(printer);
        let mut search = make_search(b"MATCH");

        let record_types = |printed: Vec<u8>| -> Vec<String> {
            String::from_utf8(printed)
                .unwrap()
                .lines()
                .map(|record| record.split('"').nth(3).unwrap().to_string())
                .collect()
        };
        let by_line = record_types(search_whole(data, &search, &path).0);
        search.multiline = true;
        let by_region = record_types(search_whole(data, &search, &path).0);

        assert_eq!(by_region, ["begin", "match", "match", "end"]);
        assert_eq!(by_region, by_line);
    }

    #[test]
    fn names_the_file_once_per_multiline_run() {
        // `--heading` prints the name as a header rather than as a per-line prefix, and
        // multiline mode reads the same writer, so the header appears there too.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta MATCH two\n";
        let mut printer = make_printer();
        printer.output_format = OutputFormat::Heading;
        let path = SearchPath::Print(printer);
        let mut search = make_search(b"MATCH");
        search.multiline = true;

        let (printed, ..) = search_whole(data, &search, &path);
        assert_eq!(printed, b"test.txt\nbeta MATCH one\ndelta MATCH two\n");
    }

    #[test]
    fn stops_reading_the_stream_at_the_first_match() {
        let data = b"error one\nerror two\nerror three\nerror four\n";
        let search = make_search(b"error");
        let path = SearchPath::Tally {
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
        let mut max_reached = false;

        let result = search_stream(
            &mut refill,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
        )
        .unwrap();

        assert_eq!(result.match_count, 1);
        assert_eq!(result.lines_searched, 1);
    }

    #[test]
    fn stops_reading_the_stream_at_the_max_count() {
        let data = b"error one\nerror two\nerror three\nerror four\n";
        let mut search = make_search(b"error");
        search.max_matches = Some(1);
        let path = SearchPath::Print(make_printer());
        // Two lines fit the budget: the second is where the max-count test fires.
        let reader = BudgetedReader {
            data,
            position: 0,
            budget: 20,
        };
        let mut refill = Refill::new(reader, 20);
        let mut output = Vec::new();
        let mut max_reached = false;

        let result = search_stream(
            &mut refill,
            "test.txt",
            &search,
            &path,
            &mut output,
            &mut max_reached,
        )
        .unwrap();

        assert_eq!(result.match_count, 1);
        assert!(max_reached);
        assert_eq!(output, b"error one\n");
    }
}

// endregion: Tests
