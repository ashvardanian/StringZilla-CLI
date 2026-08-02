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
//! sz-find -i ERROR log.txt
//!
//! # Show line numbers
//! sz-find -n error log.txt
//!
//! # Count matches only
//! sz-find -c error log.txt
//!
//! # Show context lines
//! sz-find -C 2 error log.txt
//!
//! # Filter by file type
//! sz-find error src/ -t rust
//!
//! # Filter by glob pattern
//! sz-find error src/ -g "*.rs"
//!
//! # Invert match (show non-matching lines)
//! sz-find -v error log.txt
//!
//! # Match whole words only
//! sz-find -w error log.txt
//!
//! # Stop after N matches
//! sz-find -m 10 error log.txt
//!
//! # From stdin
//! cat log.txt | sz-find error
//! ```

use std::collections::VecDeque;
use std::io::{self, IsTerminal, Read, Write};
use std::ops::{ControlFlow, Range};
use std::path::Path;

use clap::Parser;
use ignore::WalkBuilder;
use stringzilla::sz::{find, rfind, utf8_uncased_search, StringZillableBinary, Utf8UncasedNeedle};

mod shared;
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

    /// Case-insensitive search
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Show line numbers
    #[arg(short = 'n', long)]
    line_number: bool,

    /// Count matching lines only
    #[arg(short = 'c', long)]
    count: bool,

    /// Show only filenames with matches
    #[arg(short = 'l', long)]
    files_with_matches: bool,

    /// Show only filenames without matches
    #[arg(short = 'L', long)]
    files_without_match: bool,

    /// Lines of context before match
    #[arg(short = 'B', long, default_value = "0")]
    before_context: usize,

    /// Lines of context after match
    #[arg(short = 'A', long, default_value = "0")]
    after_context: usize,

    /// Lines of context before and after match
    #[arg(short = 'C', long)]
    context: Option<usize>,

    /// Enable UTF-8 mode for proper Unicode handling
    #[arg(long)]
    utf8: bool,

    /// Allow pattern to match across line boundaries
    #[arg(short = 'U', long)]
    multiline: bool,

    /// Invert match: show non-matching lines
    #[arg(short = 'v', long)]
    invert_match: bool,

    /// Match whole words only
    #[arg(short = 'w', long)]
    word: bool,

    /// Stop after NUM matches
    #[arg(short = 'm', long)]
    max_count: Option<usize>,

    /// Suppress all output (exit code only)
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Show match statistics
    #[arg(long)]
    stats: bool,

    /// Colorize output (auto, always, never)
    #[arg(long, default_value = "auto")]
    color: ColorChoice,

    // Output format options
    /// Show column numbers (1-based)
    #[arg(long)]
    column: bool,

    /// Show byte offset of each line
    #[arg(short = 'b', long)]
    byte_offset: bool,

    /// Group matches by file with filename header
    #[arg(long)]
    heading: bool,

    /// Output in vim-compatible format (file:line:col:text)
    #[arg(long)]
    vimgrep: bool,

    /// Show only the matching part of lines
    #[arg(short = 'o', long)]
    only_matching: bool,

    /// Trim printed lines to NUM bytes, on a character boundary
    #[arg(short = 'M', long)]
    max_columns: Option<usize>,

    /// Output in JSON Lines format (ripgrep-compatible)
    #[arg(long)]
    json: bool,

    // Directory traversal options
    /// Filter by file type (e.g., rust, py, js, cpp, go)
    #[arg(short = 't', long = "type")]
    file_type: Option<Vec<String>>,

    /// Filter by glob pattern (e.g., "*.rs", "*.{c,h}")
    #[arg(short = 'g', long)]
    glob: Option<Vec<String>>,

    /// Maximum directory depth
    #[arg(long)]
    max_depth: Option<usize>,

    /// Include hidden files and directories
    #[arg(long)]
    hidden: bool,

    /// Don't respect .gitignore files
    #[arg(long)]
    no_ignore: bool,

    /// Follow symbolic links
    #[arg(short = 'F', long)]
    follow: bool,

    /// Search binary files (don't skip them)
    #[arg(short = 'a', long)]
    binary: bool,

    /// Print NUL byte after filenames
    #[arg(short = '0', long)]
    null: bool,
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
    line_number: bool,
    column: bool,
    byte_offset: bool,
    only_matching: bool,
    max_columns: Option<usize>,
    /// Whether a selected line holds match positions to point at. `-v` selects the lines
    /// that hold none, leaving columns, `-o` and `--vimgrep` nothing to resolve.
    locates_matches: bool,
    /// Whether a printed matching line has its matches wrapped in color codes.
    highlight: bool,
}

impl Printer {
    /// Collapse the output flags into the form every emitter reads.
    fn new(args: &Args, multiple_inputs: bool) -> Self {
        let output_format = if args.json {
            OutputFormat::Json
        } else if args.vimgrep {
            OutputFormat::Vimgrep
        } else if args.heading {
            OutputFormat::Heading
        } else {
            OutputFormat::Standard
        };

        // JSON carries its own structure, so escape codes would corrupt it. `-q` and
        // `-c` need no test here: a silent run tallies and never builds a `Printer`.
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

        Self {
            output_format,
            colors,
            show_filename: multiple_inputs && output_format != OutputFormat::Heading,
            // Vimgrep implies line numbers and columns.
            line_number: args.line_number || args.vimgrep,
            column: args.column || args.vimgrep,
            byte_offset: args.byte_offset,
            only_matching: args.only_matching,
            max_columns: args.max_columns,
            locates_matches: !args.invert_match,
            // `-o`, `--json` and `--vimgrep` reproduce the match text themselves,
            // and an inverted line holds no match to wrap.
            highlight: !colors.match_highlight.is_empty()
                && !args.invert_match
                && !args.only_matching
                && output_format != OutputFormat::Json
                && output_format != OutputFormat::Vimgrep,
        }
    }
}

/// What `--stats` reports, summed over the walk, which runs on this thread alone.
#[derive(Default)]
struct Stats {
    files_searched: usize,
    files_matched: usize,
    lines_searched: usize,
    matches_found: usize,
    bytes_searched: usize,
}

// endregion: Output Configuration

// region: Matching

/// Check if byte is a word boundary character
#[inline]
fn is_word_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Check if position is at a word boundary
#[inline]
fn is_word_boundary(data: &[u8], start: usize, end: usize) -> bool {
    let before_ok = start == 0 || !is_word_char(data[start - 1]);
    let after_ok = end >= data.len() || !is_word_char(data[end]);
    before_ok && after_ok
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
    /// 1-based column number (for display)
    #[inline]
    fn column(&self) -> usize {
        self.offset + 1
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
}

impl<'a> MatchIter<'a> {
    #[inline]
    fn new(data: &'a [u8], needle: Needle<'a>, whole_word: bool) -> Self {
        Self {
            data,
            needle,
            pos: 0,
            whole_word,
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
            if self.whole_word && !is_word_boundary(self.data, abs_start, abs_end) {
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
    /// Present only under `-i`, holding the needle analysis shared by every search.
    uncased_needle: Option<Utf8UncasedNeedle<'a>>,
    whole_word: bool,
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
        MatchIter::new(data, self.needle(), self.whole_word)
    }
}

/// What every search path reads: the needle, the line split, and the selection rules.
struct Search<'a> {
    matcher: Matcher<'a>,
    newlines: Newlines,
    multiline: bool,
    invert_match: bool,
    max_count: Option<usize>,
}

impl Search<'_> {
    /// Whether the line belongs in the result, accounting for `-v`.
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
/// The split axis is the carried state, not the flag: output format, `-i`, `-o`,
/// `-v` and `-w` all change what an emitter writes, not what the loop holds.
#[derive(Clone, Copy)]
enum SearchPath {
    /// Two counters. Serves `-q`, `-c`, `-l` and `-L`.
    Tally { stop_at_first: bool },
    /// Counters plus a heading bit. Serves every format without context lines.
    Print(Printer),
    /// Counters, heading, a look-behind ring and group separators. Serves `-B`, `-A`, `-C`.
    PrintContext(Printer, Context),
}

impl SearchPath {
    /// Pick the path: existence only, plain printing, or printing with context.
    fn choose(args: &Args, context: Context, multiple_inputs: bool) -> Self {
        let silent =
            args.quiet || args.count || args.files_with_matches || args.files_without_match;
        if silent {
            // `-c` must see every line. `-q`, `-l` and `-L` need only one hit — except
            // under `--stats`, whose totals would otherwise describe a prefix of the file.
            let exists_only = args.quiet || args.files_with_matches || args.files_without_match;
            return SearchPath::Tally {
                stop_at_first: exists_only && !args.count && !args.stats,
            };
        }

        let printer = Printer::new(args, multiple_inputs);
        if context.before == 0 && context.after == 0 {
            SearchPath::Print(printer)
        } else {
            SearchPath::PrintContext(printer, context)
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

/// The look-behind ring, pending after-context and group separator that `-B`, `-A`
/// and `-C` carry from one line to the next.
struct ContextState {
    lines: Context,
    /// Window offsets rather than copies, and `lines.before` entries is the exact ceiling.
    /// Empty under `-A` alone, which is what lets that path stream.
    behind: VecDeque<ContextLine>,
    pending_after: usize,
    last_printed_line: Option<usize>,
    need_separator: bool,
}

impl ContextState {
    /// A ring sized to the look-behind actually requested, allocating nothing at zero.
    fn new(lines: Context) -> Self {
        ContextState {
            lines,
            behind: VecDeque::with_capacity(lines.before),
            pending_after: 0,
            last_printed_line: None,
            need_separator: false,
        }
    }
}

/// What a search path remembers between lines, and between windows once the input
/// streams: the counters, the once-per-file heading bit and the context ring.
struct Progress {
    match_count: usize,
    lines_searched: usize,
    /// Offset of the current window's first byte within the whole input, which keeps
    /// `-b`, `--json` and `--vimgrep` absolute across a streamed run.
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

    /// Whether `-m` is already satisfied, which is where every path stops reporting.
    #[inline]
    fn exhausted(&self, search: &Search) -> bool {
        search.max_count.is_some_and(|max| self.match_count >= max)
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
        // Counted before the limit is tested, so `--stats` includes the line that trips `-m`.
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
        // Counted before the limit is tested, so `--stats` includes the line that trips `-m`.
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
        // Counted before the limit is tested, so `--stats` includes the line that trips `-m`.
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

            // A gap since the last line printed opens a group, and `grep -C` divides
            // groups with `--`.
            let follows_a_gap = progress
                .context
                .last_printed_line
                .is_some_and(|last| line_number > last + 1);
            if follows_a_gap && progress.context.need_separator {
                emitter.group_separator()?;
            }

            // Print the buffered look-behind lines, sliced from the current window.
            for buffered in progress.context.behind.iter() {
                if progress
                    .context
                    .last_printed_line
                    .is_none_or(|last| buffered.line_number > last)
                {
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
            progress.context.need_separator = true;
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
    let mut emitter = path.printer().map(|printer| Emitter {
        output,
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
            if emitter.printer.line_number
                || emitter.printer.byte_offset
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

    // `-m` caps this file, and reaching the cap ends the walk over the remaining files —
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
/// `-U` searches the whole input backward and `-B` reaches back past the window, so those
/// two keep every byte. `-A` alone qualifies; it only reaches forward.
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
/// at [`last_cut`] needs no carry, not even under `-i`. `-U` is the mode where a
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
                    .write_all(br#"{"type":"begin","data":{"path":{"text":""#)?;
                json_escape_to(self.output, filename.as_bytes())?;
                self.output.write_all(b"\"}}}\n")?;
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
        self.line(LineRecord {
            line: highlighted.unwrap_or(record.line),
            ..record
        })
    }

    /// Print one line with the prefixes its format asks for.
    fn line(&mut self, record: LineRecord) -> io::Result<()> {
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
                            found.column()
                        )?;
                        if printer.only_matching {
                            self.output.write_all(found.text(record.line))?;
                        } else {
                            self.write_trimmed(record.line)?;
                        }
                        self.output.write_all(b"\n")?;
                    }
                } else if record.is_match {
                    // An inverted match holds no position, so the line prints at column one.
                    write!(self.output, "{}:{}:1:", filename, record.line_number)?;
                    self.write_trimmed(record.line)?;
                    self.output.write_all(b"\n")?;
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
                if printer.line_number {
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.line_number, record.line_number, colors.reset, separator
                    )?;
                }
                if printer.column && locates_matches {
                    let column = search
                        .matcher
                        .matches(record.line)
                        .next()
                        .map_or(1, |found| found.column());
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.column, column, colors.reset, separator
                    )?;
                }
                if printer.byte_offset {
                    write!(
                        self.output,
                        "{}{}{}{}",
                        colors.byte_offset, record.byte_offset, colors.reset, separator
                    )?;
                }
                self.write_content(record)?;
                self.output.write_all(b"\n")
            }
        }
    }

    /// Print one line in JSON Lines format (ripgrep-compatible), writing straight to the
    /// sink so that no record is staged in a `String` first.
    fn json_line(&mut self, record: LineRecord) -> io::Result<()> {
        let (filename, search, printer) = (self.filename, self.search, self.printer);
        if !record.is_match {
            self.output
                .write_all(br#"{"type":"context","data":{"path":{"text":""#)?;
            json_escape_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#""},"lines":{"text":""#)?;
            json_escape_to(self.output, record.line)?;
            write!(
                self.output,
                r#""}},"line_number":{},"absolute_offset":{}}}}}"#,
                record.line_number, record.byte_offset
            )?;
            self.output.write_all(b"\n")
        } else if !printer.locates_matches {
            // An inverted match holds no position, so it carries no submatches.
            self.output
                .write_all(br#"{"type":"match","data":{"path":{"text":""#)?;
            json_escape_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#""},"lines":{"text":""#)?;
            json_escape_to(self.output, record.line)?;
            write!(
                self.output,
                r#""}},"line_number":{},"absolute_offset":{},"submatches":[]}}}}"#,
                record.line_number, record.byte_offset
            )?;
            self.output.write_all(b"\n")
        } else {
            self.output
                .write_all(br#"{"type":"match","data":{"path":{"text":""#)?;
            json_escape_to(self.output, filename.as_bytes())?;
            self.output.write_all(br#""},"lines":{"text":""#)?;
            json_escape_to(self.output, record.line)?;
            write!(
                self.output,
                r#""}},"line_number":{},"absolute_offset":{},"submatches":["#,
                record.line_number, record.byte_offset
            )?;

            for (index, found) in search.matcher.matches(record.line).enumerate() {
                if index > 0 {
                    self.output.write_all(b",")?;
                }
                self.output.write_all(br#"{"match":{"text":""#)?;
                json_escape_to(self.output, found.text(record.line))?;
                write!(
                    self.output,
                    r#""}},"start":{},"end":{}}}"#,
                    found.offset,
                    found.offset + found.length
                )?;
            }

            self.output.write_all(b"]}}\n")
        }
    }

    /// Write a line's content, reduced to the matches themselves under `-o`.
    fn write_content(&mut self, record: LineRecord) -> io::Result<()> {
        let (search, printer) = (self.search, self.printer);
        if !(printer.only_matching && record.is_match && printer.locates_matches) {
            return self.write_trimmed(record.line);
        }
        for (index, found) in search.matcher.matches(record.line).enumerate() {
            if index > 0 {
                self.output.write_all(b"\n")?;
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

    /// Write a whole line, trimmed to `--max-columns` on a character boundary.
    /// One long minified line would otherwise flood a terminal or a context window.
    fn write_trimmed(&mut self, line: &[u8]) -> io::Result<()> {
        let Some(limit) = self.printer.max_columns else {
            return self.output.write_all(line);
        };
        let (kept, trimmed) = truncate_at_character(line, limit);
        self.output.write_all(kept)?;
        if trimmed {
            self.output.write_all(b" [...]")?;
        }
        Ok(())
    }

    /// Write a whole multi-line region verbatim, which is what `-U` prints when no
    /// per-line prefix is asked for.
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
            self.output.write_all(b"\n")?;
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
    output.write_all(br#"{"type":"end","data":{"path":{"text":""#)?;
    json_escape_to(output, filename.as_bytes())?;
    write!(
        output,
        r#""}},"stats":{{"matches":{},"lines_searched":{}}}}}}}"#,
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

/// Process stdin: a redirect maps, and a true pipe streams through one reused window
/// unless the path reads lines the window has already dropped.
fn process_stdin(
    search: &Search,
    search_path: &SearchPath,
    output: &mut dyn Write,
    stats: &mut Stats,
    max_reached: &mut bool,
) -> io::Result<FileResult> {
    stats.files_searched += 1;

    let result = if can_stream(search, search_path) {
        match get_input_streaming(None)?.into_window(DEFAULT_WINDOW_BYTES) {
            InputWindow::Whole(source) => search_slice(
                source.as_bytes(),
                "(stdin)",
                search,
                search_path,
                output,
                max_reached,
            )?,
            InputWindow::Stream(mut refill) => search_stream(
                &mut refill,
                "(stdin)",
                search,
                search_path,
                output,
                max_reached,
            )?,
        }
    } else {
        let input = get_input(None)?;
        search_slice(
            input.as_bytes(),
            "(stdin)",
            search,
            search_path,
            output,
            max_reached,
        )?
    };

    stats.bytes_searched += result.bytes_searched;
    record_file_result(stats, &result);
    Ok(result)
}

/// Fold one file's outcome into the `--stats` totals.
fn record_file_result(stats: &mut Stats, result: &FileResult) {
    stats.lines_searched += result.lines_searched;
    stats.matches_found += result.match_count;
    if result.match_count > 0 {
        stats.files_matched += 1;
    }
}

/// Build the directory walker with all filters
fn build_walker(inputs: &[String], args: &Args) -> ignore::Walk {
    let mut builder = if inputs.is_empty() || (inputs.len() == 1 && inputs[0] == "-") {
        WalkBuilder::new(".")
    } else {
        let mut builder = WalkBuilder::new(&inputs[0]);
        for input in &inputs[1..] {
            builder.add(input);
        }
        builder
    };

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
                eprintln!("Warning: invalid file type filter: {}", e);
            }
        }
    }

    // Note: glob filtering is handled manually in the main loop
    // since WalkBuilder doesn't have a direct glob filter API

    builder.build()
}

/// Print final statistics
fn print_stats(stats: &Stats, elapsed: std::time::Duration) {
    let bytes = stats.bytes_searched;

    eprintln!();
    eprintln!("Statistics:");
    eprintln!("  Files searched: {}", stats.files_searched);
    eprintln!("  Files matched:  {}", stats.files_matched);
    eprintln!("  Lines searched: {}", stats.lines_searched);
    eprintln!("  Matches found:  {}", stats.matches_found);
    eprintln!(
        "  Bytes searched: {} ({:.2} MB)",
        bytes,
        bytes as f64 / 1_000_000.0
    );
    eprintln!("  Time elapsed:   {:.3}s", elapsed.as_secs_f64());
    if elapsed.as_secs_f64() > 0.0 {
        eprintln!(
            "  Throughput:     {:.2} MB/s",
            (bytes as f64 / 1_000_000.0) / elapsed.as_secs_f64()
        );
    }
}

// endregion: Input Processing

fn main() {
    let args = Args::parse();
    let start_time = std::time::Instant::now();
    // Every byte this run prints goes here, so the records stay in one order.
    let mut output = stdout_writer();

    // Validate arguments
    if args.pattern.is_empty() {
        eprintln!("Error: pattern cannot be empty");
        ExitCode::Error.exit(&mut output);
    }

    if args.files_with_matches && args.files_without_match {
        eprintln!("Error: -l and -L are mutually exclusive");
        ExitCode::Error.exit(&mut output);
    }

    // `-C` sets both sides at once.
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

    // Determine if we're searching multiple files/directories
    let is_stdin = args.inputs.len() == 1 && args.inputs[0] == "-";
    let has_directory = !is_stdin && args.inputs.iter().any(|p| Path::new(p).is_dir());
    let multiple_inputs = args.inputs.len() > 1 || has_directory;

    // One dispatch decision for the whole run: what the per-line loop remembers.
    let search_path = SearchPath::choose(&args, context, multiple_inputs);

    let pattern = args.pattern.as_bytes();
    let search = Search {
        matcher: Matcher {
            pattern,
            uncased_needle: args.ignore_case.then(|| Utf8UncasedNeedle::new(pattern)),
            whole_word: args.word,
        },
        newlines: Newlines::from_utf8(args.utf8),
        multiline: args.multiline,
        invert_match: args.invert_match,
        max_count: args.max_count,
    };

    let mut stats = Stats::default();
    let mut max_reached = false;
    let mut any_match = false;
    let mut file_counts: Vec<(String, usize)> = Vec::new();

    // Handle stdin
    if is_stdin {
        match process_stdin(
            &search,
            &search_path,
            &mut output,
            &mut stats,
            &mut max_reached,
        ) {
            Ok(result) => {
                if result.match_count > 0 {
                    any_match = true;
                }
                if args.count {
                    let written = writeln!(output, "{}", result.match_count);
                    if let Err(error) = written {
                        exit_on_write_error(&mut output, &error, "Error writing output");
                    }
                }
            }
            Err(error) => {
                if error.kind() != io::ErrorKind::BrokenPipe {
                    eprintln!("Error reading stdin: {}", error);
                    ExitCode::Error.exit(&mut output);
                }
            }
        }
    } else {
        // Compile each `-g` glob once, so a malformed one is reported here rather than
        // silently matching nothing on every file of the walk.
        let globs = args.glob.as_ref().map(|globs| {
            globs
                .iter()
                .filter_map(|glob| match glob::Pattern::new(glob) {
                    Ok(pattern) => Some(pattern),
                    Err(error) => {
                        eprintln!("Warning: invalid glob '{}': {}", glob, error);
                        None
                    }
                })
                .collect::<Vec<_>>()
        });

        // Process files and directories
        let walker = build_walker(&args.inputs, &args);

        for result in walker {
            // Check if we've hit global max count
            if max_reached {
                break;
            }

            let entry = match result {
                Ok(entry) => entry,
                Err(error) => {
                    eprintln!("Warning: {}", error);
                    continue;
                }
            };

            // Skip directories
            if !is_readable_entry(&entry) {
                continue;
            }

            // The walker has no glob filter of its own, so `-g` is applied here.
            if let Some(globs) = &globs {
                let path_text = entry.path().to_string_lossy();
                let name_text = entry.file_name().to_string_lossy();
                let selected = globs
                    .iter()
                    .any(|pattern| pattern.matches(&path_text) || pattern.matches(&name_text));
                if !selected {
                    continue;
                }
            }

            let input = match open_input(entry.path()) {
                Ok(input) => input,
                Err(error) => {
                    eprintln!("Warning: {}: {}", entry.path().display(), error);
                    continue;
                }
            };
            let data = input.as_bytes();
            stats.files_searched += 1;
            stats.bytes_searched += data.len();

            // A binary file is skipped unless `-a` asked for it.
            if !args.binary && is_binary(data) {
                continue;
            }

            let filename = entry.path().to_string_lossy();
            let searched = search_slice(
                data,
                &filename,
                &search,
                &search_path,
                &mut output,
                &mut max_reached,
            );
            let result = match searched {
                Ok(result) => result,
                Err(error) if error.kind() == io::ErrorKind::BrokenPipe => {
                    ExitCode::Success.exit(&mut output)
                }
                Err(error) => {
                    eprintln!("Warning: {}: {}", entry.path().display(), error);
                    continue;
                }
            };
            record_file_result(&mut stats, &result);

            if result.match_count > 0 {
                any_match = true;
                // Only files with matches are reported, as `grep -c` and `rg -c` do.
                // Listing every `path:0` buries the answer in a directory walk.
                if args.count {
                    file_counts.push((filename.to_string(), result.match_count));
                }
            }

            // `-l` names the files that matched, `-L` those that did not.
            let named = if result.match_count > 0 {
                args.files_with_matches
            } else {
                args.files_without_match
            };
            if named {
                let terminator = if args.null { "\0" } else { "\n" };
                let written = write!(output, "{}{}", filename, terminator);
                if let Err(error) = written {
                    exit_on_write_error(&mut output, &error, "Error writing output");
                }
            }
        }

        // Print counts for multi-file mode
        for (path, count) in &file_counts {
            let written = if multiple_inputs {
                writeln!(output, "{}:{}", path, count)
            } else {
                writeln!(output, "{}", count)
            };
            if let Err(error) = written {
                exit_on_write_error(&mut output, &error, "Error writing output");
            }
        }
    }

    // Print statistics. The results are stdout and the report is stderr, so the buffer
    // is flushed first to keep the two ordered where they land in one stream.
    if args.stats {
        let _ = output.flush();
        print_stats(&stats, start_time.elapsed());
    }

    // Exit with status 1 if no matches found (like grep). Exiting skips the
    // buffer's `Drop`, so the flush has to happen first.
    if !any_match {
        ExitCode::NoResult.exit(&mut output);
    }
}

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn make_search(pattern: &[u8]) -> Search<'_> {
        Search {
            matcher: Matcher {
                pattern,
                uncased_needle: None,
                whole_word: false,
            },
            newlines: Newlines::Lf,
            multiline: false,
            invert_match: false,
            max_count: None,
        }
    }

    fn make_printer() -> Printer {
        Printer {
            output_format: OutputFormat::Standard,
            colors: Colors::disabled(),
            show_filename: false,
            line_number: false,
            column: false,
            byte_offset: false,
            only_matching: false,
            max_columns: None,
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
        MatchIter::new(line, Needle::new(pattern, uncased.as_ref()), whole_word)
            .next()
            .is_some()
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
        search.max_count = Some(2);
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
        printer.line_number = true;
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
    fn joins_multiline_groups_that_adjoin() {
        // The after-context of the first match and the before-context of the second are
        // consecutive lines, so the whole file prints as one group, as `grep -C 1` does.
        // Line mode measures the gap from the matching line rather than from the group's
        // first line, and divides here where `grep` does not.
        let data = b"alpha\nbeta MATCH one\ngamma\ndelta\nepsilon MATCH two\nzeta\n";
        let mut search = make_search(b"MATCH");
        search.multiline = true;
        let context = Context {
            before: 1,
            after: 1,
        };
        let path = SearchPath::PrintContext(make_printer(), context);

        let (by_region, ..) = search_whole(data, &search, &path);
        assert_eq!(by_region, data);
    }

    #[test]
    fn divides_no_multiline_groups_without_context() {
        // Without `-A`, `-B` or `-C` there are no groups, so no divider is written.
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
        printer.line_number = true;
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
        printer.line_number = true;
        printer.byte_offset = true;
        printer.column = true;
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
        search.max_count = Some(1);
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
