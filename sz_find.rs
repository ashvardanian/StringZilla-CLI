//! SIMD-accelerated substring search utility
//!
//! A grep-like tool with simpler syntax, using StringZilla for fast searching.
//! Unlike grep, uses literal substring matching (not regex) for maximum speed.
//! Passing `-r/--replace` switches to in-place find-and-replace.
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
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use clap::Parser;
use ignore::WalkBuilder;
use memmap2::Mmap;
use stringzilla::sz::{find, rfind, utf8_case_insensitive_find};

mod shared;
use shared::*;

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
    #[arg(short = 'M', long)]
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

    // Replacement options
    /// Replacement string (enables replace mode)
    #[arg(short = 'r', long)]
    replace: Option<String>,

    /// Modify files in-place (requires -r, default for file inputs)
    #[arg(long, requires = "replace")]
    in_place: bool,

    /// Dry run - show what would be replaced without making changes
    #[arg(long, requires = "replace")]
    dry_run: bool,
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

/// ANSI color codes
struct Colors {
    filename: &'static str,
    line_number: &'static str,
    column: &'static str,
    byte_offset: &'static str,
    match_highlight: &'static str,
    context_mark: &'static str,
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
            context_mark: "\x1b[2m",       // Dim
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
            context_mark: "",
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

/// Statistics for the search
#[derive(Default)]
struct Stats {
    files_searched: AtomicUsize,
    files_matched: AtomicUsize,
    lines_searched: AtomicUsize,
    matches_found: AtomicUsize,
    bytes_searched: AtomicUsize,
}

/// Search configuration (derived from Args for efficiency)
struct SearchConfig<'a> {
    pattern: &'a [u8],
    ignore_case: bool,
    line_number: bool,
    count: bool,
    files_with_matches: bool,
    files_without_match: bool,
    before_context: usize,
    after_context: usize,
    utf8: bool,
    multiline: bool,
    invert_match: bool,
    whole_word: bool,
    max_count: Option<usize>,
    quiet: bool,
    colors: Colors,
    show_filename: bool,
    // New output options
    column: bool,
    byte_offset: bool,
    only_matching: bool,
    output_format: OutputFormat,
}

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
}

/// Zero-allocation iterator over pattern matches in data.
///
/// Uses StringZilla's SIMD-accelerated search functions directly.
/// For case-insensitive matching, uses `utf8_case_insensitive_find` which
/// may return matches of different lengths than the pattern.
struct MatchIter<'a> {
    data: &'a [u8],
    pattern: &'a [u8],
    pos: usize,
    ignore_case: bool,
    whole_word: bool,
}

impl<'a> MatchIter<'a> {
    #[inline]
    fn new(data: &'a [u8], pattern: &'a [u8], ignore_case: bool, whole_word: bool) -> Self {
        Self {
            data,
            pattern,
            pos: 0,
            ignore_case,
            whole_word,
        }
    }
}

impl<'a> Iterator for MatchIter<'a> {
    type Item = MatchInfo;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.data.len() {
            let remaining = &self.data[self.pos..];

            let (offset, len) = if self.ignore_case {
                utf8_case_insensitive_find(remaining, self.pattern)?
            } else {
                let off = find(remaining, self.pattern)?;
                (off, self.pattern.len())
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

/// Create a match iterator from SearchConfig
#[inline]
fn matches<'a>(data: &'a [u8], config: &'a SearchConfig) -> MatchIter<'a> {
    MatchIter::new(data, config.pattern, config.ignore_case, config.whole_word)
}

/// Write JSON-escaped bytes directly to output (no intermediate String allocation)
fn json_escape_to(output: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    for &byte in data {
        match byte {
            b'"' => output.write_all(b"\\\"")?,
            b'\\' => output.write_all(b"\\\\")?,
            b'\n' => output.write_all(b"\\n")?,
            b'\r' => output.write_all(b"\\r")?,
            b'\t' => output.write_all(b"\\t")?,
            b if b < 0x20 => write!(output, "\\u{:04x}", b)?,
            b => output.write_all(&[b])?,
        }
    }
    Ok(())
}

/// Replace all occurrences in-place using two-pointer compaction.
///
/// **Case-sensitive only** - for byte-level replacement where pattern.len() == match.len().
/// Returns (new_length, replacement_count).
///
/// Invariant: `write_pos <= read_pos` ensures we never overwrite unread data.
fn replace_in_place(
    data: &mut [u8],
    pattern: &[u8],
    replacement: &[u8],
    whole_word: bool,
) -> (usize, usize) {
    debug_assert!(
        replacement.len() <= pattern.len(),
        "In-place replacement requires replacement.len() <= pattern.len()"
    );

    if pattern.is_empty() {
        return (data.len(), 0);
    }

    let mut read_pos: usize = 0;
    let mut write_pos: usize = 0;
    let mut count: usize = 0;

    // Use raw find() to avoid borrow conflict (can't iterate and mutate simultaneously)
    loop {
        let remaining = &data[read_pos..];
        let offset = match find(remaining, pattern) {
            Some(off) => off,
            None => break,
        };

        let abs_start = read_pos + offset;
        let abs_end = abs_start + pattern.len();

        // Check word boundary if needed
        // Note: During compaction, data[write_pos..read_pos] is undefined ("gap").
        // For "before" byte: if match starts at read_pos and there's a gap, use write_pos-1
        // For "after" byte: always at abs_end which is in original region
        if whole_word {
            let before_ok = if abs_start == 0 {
                true // Start of buffer
            } else if abs_start == read_pos && write_pos < read_pos && write_pos > 0 {
                // Match at start of remaining, gap exists - check compacted byte
                !is_word_char(data[write_pos - 1])
            } else {
                !is_word_char(data[abs_start - 1])
            };
            let after_ok = abs_end >= data.len() || !is_word_char(data[abs_end]);

            if !before_ok || !after_ok {
                // Not a word boundary - copy up to and past this non-match
                let skip_to = abs_start + 1;
                let len = skip_to - read_pos;
                if write_pos != read_pos {
                    data.copy_within(read_pos..skip_to, write_pos);
                }
                write_pos += len;
                read_pos = skip_to;
                continue;
            }
        }

        // Copy data between last position and this match
        if abs_start > read_pos {
            let len = abs_start - read_pos;
            if write_pos != read_pos {
                data.copy_within(read_pos..abs_start, write_pos);
            }
            write_pos += len;
        }

        // Write replacement
        data[write_pos..write_pos + replacement.len()].copy_from_slice(replacement);
        write_pos += replacement.len();
        count += 1;

        read_pos = abs_end;
    }

    // Copy remaining data after last match
    if read_pos < data.len() {
        let len = data.len() - read_pos;
        if write_pos != read_pos {
            data.copy_within(read_pos..data.len(), write_pos);
        }
        write_pos += len;
    }

    (write_pos, count)
}

/// Replace all occurrences, returning a new buffer.
///
/// **Case-sensitive only** - used when replacement is longer than pattern.
fn replace_to_buffer(
    data: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    whole_word: bool,
) -> (Vec<u8>, usize) {
    if pattern.is_empty() {
        return (data.to_vec(), 0);
    }

    let mut result = Vec::with_capacity(data.len());
    let mut last_end = 0;
    let mut count = 0;

    // Use MatchIter for case-sensitive search (no allocations)
    for m in MatchIter::new(data, pattern, false, whole_word) {
        // Copy data before this match
        result.extend_from_slice(&data[last_end..m.offset]);
        // Add replacement
        result.extend_from_slice(replacement);
        count += 1;
        last_end = m.offset + m.length;
    }

    // Copy remaining data after last match
    result.extend_from_slice(&data[last_end..]);

    (result, count)
}

/// Count matches without modification (for dry-run)
fn count_matches(data: &[u8], pattern: &[u8], whole_word: bool) -> usize {
    MatchIter::new(data, pattern, false, whole_word).count()
}

/// Process replacement for a single file
fn process_file_replace(
    path: &Path,
    pattern: &[u8],
    replacement: &[u8],
    whole_word: bool,
    in_place: bool,
    dry_run: bool,
    skip_binary: bool,
) -> io::Result<(usize, bool)> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Skip binary files
    if skip_binary && is_binary(&mmap) {
        return Ok((0, false));
    }

    if dry_run {
        let count = count_matches(&mmap, pattern, whole_word);
        return Ok((count, false));
    }

    let can_shrink_in_place = replacement.len() <= pattern.len();

    if in_place && can_shrink_in_place {
        // In-place modification using mutable mmap (shrinking or same size)
        drop(mmap);
        let mut input = get_input_mutable(path.to_str().unwrap())?;
        let data = input.as_mut_bytes().unwrap();
        let (new_len, count) = replace_in_place(data, pattern, replacement, whole_word);
        input.truncate_and_flush(new_len as u64)?;
        Ok((count, true))
    } else {
        // Buffer approach: read, replace, write back
        let (result, count) = replace_to_buffer(&mmap, pattern, replacement, whole_word);
        if count > 0 && in_place {
            drop(mmap);
            std::fs::write(path, &result)?;
        }
        Ok((count, count > 0))
    }
}

/// Check if line matches (considering invert_match)
/// TODO: Add sz_find_word_boundary() to StringZilla for SIMD-accelerated word-aware search
#[inline]
fn line_matches(line: &[u8], config: &SearchConfig) -> bool {
    let has_match = matches(line, config).next().is_some();
    if config.invert_match {
        !has_match
    } else {
        has_match
    }
}

/// Highlight matches in a line, writing to a reusable buffer.
///
/// Returns the number of bytes written. If 0, no highlighting was needed
/// (caller should use the original line directly for zero-copy output).
fn highlight_line(line: &[u8], config: &SearchConfig, buffer: &mut Vec<u8>) -> usize {
    buffer.clear();

    // Return 0 if no highlighting needed
    if config.colors.match_highlight.is_empty() || config.invert_match {
        return 0;
    }

    // Reserve capacity to minimize allocations during building
    buffer.reserve(line.len() + 64);

    let mut last_end = 0;
    let mut found_match = false;

    for m in matches(line, config) {
        found_match = true;
        // Add text before this match
        buffer.extend_from_slice(&line[last_end..m.offset]);
        // Add highlighted match
        buffer.extend_from_slice(config.colors.match_highlight.as_bytes());
        buffer.extend_from_slice(&line[m.offset..m.offset + m.length]);
        buffer.extend_from_slice(config.colors.reset.as_bytes());
        last_end = m.offset + m.length;
    }

    // If no matches found, return 0
    if !found_match {
        buffer.clear();
        return 0;
    }

    // Add remaining text after last match
    buffer.extend_from_slice(&line[last_end..]);
    buffer.len()
}

/// Search result for a single file
struct FileResult {
    path: String,
    match_count: usize,
    lines_searched: usize,
}

/// Search a memory-mapped file
fn search_mmap(
    data: &[u8],
    filename: &str,
    config: &SearchConfig,
    output: &mut dyn Write,
    max_reached: &AtomicBool,
) -> io::Result<FileResult> {
    if config.multiline {
        search_multiline(data, filename, config, output, max_reached)
    } else if config.utf8 {
        search_intraline(
            data,
            Utf8LineIterator::new(data),
            filename,
            config,
            output,
            max_reached,
        )
    } else {
        search_intraline(
            data,
            LineIterator::new(data),
            filename,
            config,
            output,
            max_reached,
        )
    }
}

/// Context line info: (line_number, start_offset, end_offset)
/// The actual line data is sliced from the original buffer when needed.
type ContextLine = (usize, usize, usize);

/// Search using intra-line matching (lazy iteration)
///
/// `data` is the original buffer - lines from the iterator are slices into this buffer.
/// This allows storing offsets in the context buffer instead of copying line data.
fn search_intraline<'a, I>(
    data: &'a [u8],
    iter: I,
    filename: &str,
    config: &SearchConfig,
    output: &mut dyn Write,
    max_reached: &AtomicBool,
) -> io::Result<FileResult>
where
    I: Iterator<Item = &'a [u8]>,
{
    let mut match_count = 0;
    let mut lines_searched = 0;
    let mut byte_offset: usize = 0;
    let mut context_before: VecDeque<ContextLine> =
        VecDeque::with_capacity(config.before_context + 1);
    let mut pending_after: usize = 0;
    let mut last_printed_line: Option<usize> = None;
    let mut need_separator = false;
    let mut printed_heading = false;
    // Reusable buffer for highlighting (avoids per-line allocation)
    let mut highlight_buffer = Vec::with_capacity(512);

    for (line_num, line) in iter.enumerate() {
        lines_searched += 1;
        let line_num_1based = line_num + 1;
        let current_byte_offset = byte_offset;

        // Advance byte offset past this line (including newline)
        byte_offset += line.len() + 1;

        // Check if we've hit max count
        if let Some(max) = config.max_count {
            if match_count >= max {
                max_reached.store(true, Ordering::Relaxed);
                break;
            }
        }

        let is_match = line_matches(line, config);

        if is_match {
            match_count += 1;

            if config.quiet
                || config.count
                || config.files_with_matches
                || config.files_without_match
            {
                // Maintain context buffer even when not printing (store offsets, not copies)
                if config.before_context > 0 {
                    if context_before.len() >= config.before_context {
                        context_before.pop_front();
                    }
                    context_before.push_back((
                        line_num_1based,
                        current_byte_offset,
                        current_byte_offset + line.len(),
                    ));
                }
                continue;
            }

            // Print heading (filename header) on first match for this file
            if config.output_format == OutputFormat::Heading && !printed_heading {
                writeln!(
                    output,
                    "{}{}{}",
                    config.colors.filename, filename, config.colors.reset
                )?;
                printed_heading = true;
            }

            // Print JSON begin message on first match
            if config.output_format == OutputFormat::Json && !printed_heading {
                output.write_all(br#"{"type":"begin","data":{"path":{"text":""#)?;
                json_escape_to(output, filename.as_bytes())?;
                output.write_all(b"\"}}}\n")?;
                printed_heading = true;
            }

            // Print separator between non-contiguous match groups
            if config.before_context > 0 || config.after_context > 0 {
                if let Some(last) = last_printed_line {
                    if line_num_1based > last + 1 && need_separator {
                        if config.output_format != OutputFormat::Json {
                            writeln!(
                                output,
                                "{}--{}",
                                config.colors.separator, config.colors.reset
                            )?;
                        }
                    }
                }
            }

            // Print buffered context_before lines (slice from original data)
            for &(ctx_line_num, ctx_start, ctx_end) in context_before.iter() {
                if last_printed_line.map_or(true, |lp| ctx_line_num > lp) {
                    let ctx_line = &data[ctx_start..ctx_end];
                    print_line(
                        output,
                        ctx_line,
                        ctx_line_num,
                        ctx_start,
                        filename,
                        config,
                        false,
                    )?;
                    last_printed_line = Some(ctx_line_num);
                }
            }

            // Print matching line (highlight only for standard/heading modes)
            // Use highlight_buffer to avoid per-line allocation
            let needs_highlight = config.output_format != OutputFormat::Json
                && config.output_format != OutputFormat::Vimgrep
                && !config.only_matching;

            let line_to_print: &[u8] =
                if needs_highlight && highlight_line(line, config, &mut highlight_buffer) > 0 {
                    &highlight_buffer
                } else {
                    line
                };

            print_line(
                output,
                line_to_print,
                line_num_1based,
                current_byte_offset,
                filename,
                config,
                true,
            )?;
            last_printed_line = Some(line_num_1based);
            pending_after = config.after_context;
            need_separator = true;
        } else if pending_after > 0 {
            // Print as after-context
            print_line(
                output,
                line,
                line_num_1based,
                current_byte_offset,
                filename,
                config,
                false,
            )?;
            last_printed_line = Some(line_num_1based);
            pending_after -= 1;
        }

        // Maintain rolling context_before buffer (store offsets, not copies)
        if config.before_context > 0 {
            if context_before.len() >= config.before_context {
                context_before.pop_front();
            }
            context_before.push_back((
                line_num_1based,
                current_byte_offset,
                current_byte_offset + line.len(),
            ));
        }
    }

    // Print JSON end message if we printed any matches
    if config.output_format == OutputFormat::Json && printed_heading {
        output.write_all(br#"{"type":"end","data":{"path":{"text":""#)?;
        json_escape_to(output, filename.as_bytes())?;
        write!(
            output,
            r#""}},"stats":{{"matches":{},"lines_searched":{}}}}}}}"#,
            match_count, lines_searched
        )?;
        output.write_all(b"\n")?;
    }

    Ok(FileResult {
        path: filename.to_string(),
        match_count,
        lines_searched,
    })
}

/// Search using multi-line matching (whole buffer search)
fn search_multiline(
    data: &[u8],
    filename: &str,
    config: &SearchConfig,
    output: &mut dyn Write,
    max_reached: &AtomicBool,
) -> io::Result<FileResult> {
    let mut match_count = 0;
    let mut pos = 0;
    let mut last_printed_end: usize = 0;
    let lines_searched = count_byte(data, b'\n') + 1;

    while pos < data.len() {
        // Check max count
        if let Some(max) = config.max_count {
            if match_count >= max {
                max_reached.store(true, Ordering::Relaxed);
                break;
            }
        }

        // Find next match
        let match_result = if config.ignore_case {
            utf8_case_insensitive_find(&data[pos..], config.pattern)
                .map(|(off, len)| (pos + off, len))
        } else {
            find(&data[pos..], config.pattern).map(|off| (pos + off, config.pattern.len()))
        };

        let (match_offset, match_len) = match match_result {
            Some(m) => m,
            None => break,
        };

        // Check word boundary
        if config.whole_word && !is_word_boundary(data, match_offset, match_offset + match_len) {
            pos = match_offset + 1;
            continue;
        }

        match_count += 1;

        if config.quiet || config.count || config.files_with_matches || config.files_without_match {
            pos = match_offset + match_len;
            continue;
        }

        // Find line boundaries around the match
        let line_start = if match_offset == 0 {
            0
        } else {
            rfind(&data[..match_offset], b"\n")
                .map(|i| i + 1)
                .unwrap_or(0)
        };

        let match_end = match_offset + match_len;
        let line_end = find(&data[match_end..], b"\n")
            .map(|i| match_end + i)
            .unwrap_or(data.len());

        // Expand for before_context
        let mut output_start = line_start;
        if config.before_context > 0 {
            let mut count = 0;
            let mut search_pos = line_start;
            while count < config.before_context && search_pos > 0 {
                search_pos = if search_pos <= 1 {
                    0
                } else {
                    rfind(&data[..search_pos - 1], b"\n")
                        .map(|i| i + 1)
                        .unwrap_or(0)
                };
                count += 1;
            }
            output_start = search_pos;
        }

        // Expand for after_context
        let mut output_end = line_end;
        if config.after_context > 0 {
            let mut count = 0;
            let mut search_pos = line_end;
            while count < config.after_context && search_pos < data.len() {
                if let Some(next_nl) = find(&data[search_pos + 1..], b"\n") {
                    search_pos = search_pos + 1 + next_nl;
                } else {
                    search_pos = data.len();
                    break;
                }
                count += 1;
            }
            output_end = search_pos;
        }

        // Avoid printing overlapping regions
        let actual_start = output_start.max(last_printed_end);
        if actual_start < output_end {
            if config.line_number
                || config.byte_offset
                || config.output_format != OutputFormat::Standard
            {
                let line_num = count_byte(&data[..actual_start], b'\n') + 1;
                let region = &data[actual_start..output_end];
                let mut line_byte_offset = actual_start;
                for (i, line) in region.split(|&b| b == b'\n').enumerate() {
                    if !line.is_empty() || i == 0 {
                        print_line(
                            output,
                            line,
                            line_num + i,
                            line_byte_offset,
                            filename,
                            config,
                            true,
                        )?;
                    }
                    line_byte_offset += line.len() + 1;
                }
            } else {
                if config.show_filename {
                    write!(
                        output,
                        "{}{}{}:",
                        config.colors.filename, filename, config.colors.reset
                    )?;
                }
                output.write_all(&data[actual_start..output_end])?;
                if output_end < data.len() && data[output_end] != b'\n' {
                    output.write_all(b"\n")?;
                }
            }
            last_printed_end = output_end;
        }

        pos = match_end;
    }

    Ok(FileResult {
        path: filename.to_string(),
        match_count,
        lines_searched,
    })
}

/// Print a single line with optional filename, line numbers, columns, byte offsets
fn print_line(
    output: &mut dyn Write,
    line: &[u8],
    line_num: usize,
    byte_offset: usize,
    filename: &str,
    config: &SearchConfig,
    is_match: bool,
) -> io::Result<()> {
    // Use different separator for context vs match lines
    let sep = if is_match { ":" } else { "-" };

    match config.output_format {
        OutputFormat::Json => print_line_json(
            output,
            line,
            line_num,
            byte_offset,
            filename,
            config,
            is_match,
        ),
        OutputFormat::Vimgrep => {
            // Vimgrep: output each match on its own line with file:line:col:text
            if is_match && !config.invert_match {
                for m in matches(line, config) {
                    write!(output, "{}:{}:{}:", filename, line_num, m.column())?;
                    if config.only_matching {
                        output.write_all(&line[m.offset..m.offset + m.length])?;
                    } else {
                        output.write_all(line)?;
                    }
                    output.write_all(b"\n")?;
                }
            } else if is_match {
                // Inverted match - just print the line
                write!(output, "{}:{}:1:", filename, line_num)?;
                output.write_all(line)?;
                output.write_all(b"\n")?;
            }
            Ok(())
        }
        OutputFormat::Heading => {
            // Heading mode: filename is printed separately as header
            // Just print line_num:col:text
            if config.line_number {
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.line_number, line_num, config.colors.reset, sep
                )?;
            }
            if config.column && is_match && !config.invert_match {
                let col = matches(line, config).next().map_or(1, |m| m.column());
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.column, col, config.colors.reset, sep
                )?;
            }
            if config.byte_offset {
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.byte_offset, byte_offset, config.colors.reset, sep
                )?;
            }
            print_line_content(output, line, config, is_match)?;
            output.write_all(b"\n")?;
            Ok(())
        }
        OutputFormat::Standard => {
            // Standard: file:line:col:byte:text
            if config.show_filename {
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.filename, filename, config.colors.reset, sep
                )?;
            }
            if config.line_number {
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.line_number, line_num, config.colors.reset, sep
                )?;
            }
            if config.column && is_match && !config.invert_match {
                let col = matches(line, config).next().map_or(1, |m| m.column());
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.column, col, config.colors.reset, sep
                )?;
            }
            if config.byte_offset {
                write!(
                    output,
                    "{}{}{}{}",
                    config.colors.byte_offset, byte_offset, config.colors.reset, sep
                )?;
            }
            print_line_content(output, line, config, is_match)?;
            output.write_all(b"\n")?;
            Ok(())
        }
    }
}

/// Print line content (handles only_matching mode)
fn print_line_content(
    output: &mut dyn Write,
    line: &[u8],
    config: &SearchConfig,
    is_match: bool,
) -> io::Result<()> {
    if config.only_matching && is_match && !config.invert_match {
        // Only print matching parts
        let mut first = true;
        for m in matches(line, config) {
            if !first {
                output.write_all(b"\n")?;
            }
            first = false;
            if !config.colors.match_highlight.is_empty() {
                output.write_all(config.colors.match_highlight.as_bytes())?;
            }
            output.write_all(&line[m.offset..m.offset + m.length])?;
            if !config.colors.reset.is_empty() {
                output.write_all(config.colors.reset.as_bytes())?;
            }
        }
        Ok(())
    } else {
        output.write_all(line)
    }
}

/// Print a line in JSON Lines format (ripgrep-compatible)
/// Writes directly to output without intermediate String allocations.
fn print_line_json(
    output: &mut dyn Write,
    line: &[u8],
    line_num: usize,
    byte_offset: usize,
    filename: &str,
    config: &SearchConfig,
    is_match: bool,
) -> io::Result<()> {
    if !is_match {
        // Context line in JSON format
        output.write_all(br#"{"type":"context","data":{"path":{"text":""#)?;
        json_escape_to(output, filename.as_bytes())?;
        output.write_all(br#""},"lines":{"text":""#)?;
        json_escape_to(output, line)?;
        write!(
            output,
            r#""}},"line_number":{},"absolute_offset":{}}}}}"#,
            line_num, byte_offset
        )?;
        output.write_all(b"\n")
    } else if config.invert_match {
        // Inverted match - no submatches
        output.write_all(br#"{"type":"match","data":{"path":{"text":""#)?;
        json_escape_to(output, filename.as_bytes())?;
        output.write_all(br#""},"lines":{"text":""#)?;
        json_escape_to(output, line)?;
        write!(
            output,
            r#""}},"line_number":{},"absolute_offset":{},"submatches":[]}}}}"#,
            line_num, byte_offset
        )?;
        output.write_all(b"\n")
    } else {
        // Regular match with submatches - write directly to avoid Vec allocation
        output.write_all(br#"{"type":"match","data":{"path":{"text":""#)?;
        json_escape_to(output, filename.as_bytes())?;
        output.write_all(br#""},"lines":{"text":""#)?;
        json_escape_to(output, line)?;
        write!(
            output,
            r#""}},"line_number":{},"absolute_offset":{},"submatches":["#,
            line_num, byte_offset
        )?;

        let mut first = true;
        for m in matches(line, config) {
            if !first {
                output.write_all(b",")?;
            }
            first = false;
            output.write_all(br#"{"match":{"text":""#)?;
            json_escape_to(output, &line[m.offset..m.offset + m.length])?;
            write!(
                output,
                r#""}},"start":{},"end":{}}}"#,
                m.offset,
                m.offset + m.length
            )?;
        }

        output.write_all(b"]}}\n")
    }
}

/// Check if data appears to be binary (contains NUL bytes in first 8KB)
fn is_binary(data: &[u8]) -> bool {
    let check_len = data.len().min(8192);
    find(&data[..check_len], b"\0").is_some()
}

/// Process a single file
fn process_file(
    path: &Path,
    config: &SearchConfig,
    output: &mut dyn Write,
    stats: &Stats,
    max_reached: &AtomicBool,
    skip_binary: bool,
) -> io::Result<Option<FileResult>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    stats.files_searched.fetch_add(1, Ordering::Relaxed);
    stats
        .bytes_searched
        .fetch_add(mmap.len(), Ordering::Relaxed);

    // Skip binary files unless -a is specified
    if skip_binary && is_binary(&mmap) {
        return Ok(None);
    }

    let filename = path.to_string_lossy();
    let result = search_mmap(&mmap, &filename, config, output, max_reached)?;

    stats
        .lines_searched
        .fetch_add(result.lines_searched, Ordering::Relaxed);
    stats
        .matches_found
        .fetch_add(result.match_count, Ordering::Relaxed);

    if result.match_count > 0 {
        stats.files_matched.fetch_add(1, Ordering::Relaxed);
    }

    Ok(Some(result))
}

/// Process stdin
fn process_stdin(
    config: &SearchConfig,
    output: &mut dyn Write,
    stats: &Stats,
    max_reached: &AtomicBool,
) -> io::Result<FileResult> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;

    stats.files_searched.fetch_add(1, Ordering::Relaxed);
    stats
        .bytes_searched
        .fetch_add(buffer.len(), Ordering::Relaxed);

    let result = search_mmap(&buffer, "(stdin)", config, output, max_reached)?;

    stats
        .lines_searched
        .fetch_add(result.lines_searched, Ordering::Relaxed);
    stats
        .matches_found
        .fetch_add(result.match_count, Ordering::Relaxed);

    if result.match_count > 0 {
        stats.files_matched.fetch_add(1, Ordering::Relaxed);
    }

    Ok(result)
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
    let files = stats.files_searched.load(Ordering::Relaxed);
    let matched = stats.files_matched.load(Ordering::Relaxed);
    let lines = stats.lines_searched.load(Ordering::Relaxed);
    let matches = stats.matches_found.load(Ordering::Relaxed);
    let bytes = stats.bytes_searched.load(Ordering::Relaxed);

    eprintln!();
    eprintln!("Statistics:");
    eprintln!("  Files searched: {}", files);
    eprintln!("  Files matched:  {}", matched);
    eprintln!("  Lines searched: {}", lines);
    eprintln!("  Matches found:  {}", matches);
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

/// Run replacement mode (when -r flag is provided)
fn run_replace_mode(args: &Args, replacement: &str) {
    // Case-insensitive replacement is not supported (UTF-8 match lengths vary)
    if args.ignore_case {
        eprintln!("Error: case-insensitive replacement (-i -r) is not supported");
        eprintln!("       UTF-8 case folding can produce matches of different lengths,");
        eprintln!("       making in-place replacement unsafe. Use case-sensitive replacement.");
        process::exit(1);
    }

    let pattern = args.pattern.as_bytes();
    let replacement = replacement.as_bytes();
    let is_stdin = args.inputs.len() == 1 && args.inputs[0] == "-";

    // In-place is default for file inputs (unless stdin)
    let in_place = !is_stdin;

    if is_stdin {
        // Stdin mode: read, replace, write to stdout
        let mut buffer = Vec::new();
        if let Err(e) = io::stdin().read_to_end(&mut buffer) {
            eprintln!("Error reading stdin: {}", e);
            process::exit(1);
        }

        let (result, count) = replace_to_buffer(&buffer, pattern, replacement, args.word);

        if args.dry_run {
            eprintln!("Dry run: would replace {} occurrence(s)", count);
            process::exit(0);
        }

        if let Err(e) = io::stdout().write_all(&result) {
            if e.kind() != io::ErrorKind::BrokenPipe {
                eprintln!("Error writing output: {}", e);
                process::exit(1);
            }
        }

        if !args.quiet {
            eprintln!("Replaced {} occurrence(s)", count);
        }
    } else {
        // File/directory mode
        let walker = build_walker(&args.inputs, args);
        let mut total_count = 0;
        let mut files_modified = 0;

        for result in walker {
            let entry = match result {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Warning: {}", e);
                    continue;
                }
            };

            // Skip directories
            if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                continue;
            }

            // Apply glob filters
            if let Some(ref globs) = args.glob {
                let path_str = entry.path().to_string_lossy();
                let mut matches_glob = false;
                for glob in globs {
                    if let Ok(pat) = glob::Pattern::new(glob) {
                        if pat.matches(&path_str)
                            || pat.matches(entry.file_name().to_string_lossy().as_ref())
                        {
                            matches_glob = true;
                            break;
                        }
                    }
                }
                if !matches_glob {
                    continue;
                }
            }

            match process_file_replace(
                entry.path(),
                pattern,
                replacement,
                args.word,
                in_place,
                args.dry_run,
                !args.binary,
            ) {
                Ok((count, modified)) => {
                    if count > 0 {
                        total_count += count;
                        if modified {
                            files_modified += 1;
                        }
                        if !args.quiet {
                            eprintln!(
                                "{}: {} replacement(s){}",
                                entry.path().display(),
                                count,
                                if args.dry_run { " (dry run)" } else { "" }
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: {}: {}", entry.path().display(), e);
                }
            }
        }

        if !args.quiet {
            eprintln!();
            if args.dry_run {
                eprintln!(
                    "Dry run: would replace {} occurrence(s) in {} file(s)",
                    total_count, files_modified
                );
            } else {
                eprintln!(
                    "Replaced {} occurrence(s) in {} file(s)",
                    total_count, files_modified
                );
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    let start_time = std::time::Instant::now();

    // Validate arguments
    if args.pattern.is_empty() {
        eprintln!("Error: pattern cannot be empty");
        process::exit(1);
    }

    if args.files_with_matches && args.files_without_match {
        eprintln!("Error: -l and -L are mutually exclusive");
        process::exit(1);
    }

    // Handle replace mode
    if let Some(ref replacement) = args.replace {
        run_replace_mode(&args, replacement);
        return;
    }

    // Handle -C flag (context on both sides)
    let (before_context, after_context) = if let Some(ctx) = args.context {
        (ctx, ctx)
    } else {
        (args.before_context, args.after_context)
    };

    // Determine if we're searching multiple files/directories
    let is_stdin = args.inputs.len() == 1 && args.inputs[0] == "-";
    let has_directory = !is_stdin && args.inputs.iter().any(|p| Path::new(p).is_dir());
    let multiple_inputs = args.inputs.len() > 1 || has_directory;

    // Determine output format
    let output_format = if args.json {
        OutputFormat::Json
    } else if args.vimgrep {
        OutputFormat::Vimgrep
    } else if args.heading {
        OutputFormat::Heading
    } else {
        OutputFormat::Standard
    };

    // Determine color mode (JSON disables colors)
    let use_color = match args.color {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            io::stdout().is_terminal()
                && !args.quiet
                && !args.count
                && output_format != OutputFormat::Json
        }
    };

    let colors = if use_color {
        Colors::enabled()
    } else {
        Colors::disabled()
    };

    // Vimgrep implies line numbers and column
    let show_line_number = args.line_number || args.vimgrep;
    let show_column = args.column || args.vimgrep;

    // Build search configuration
    let config = SearchConfig {
        pattern: args.pattern.as_bytes(),
        ignore_case: args.ignore_case,
        line_number: show_line_number,
        count: args.count,
        files_with_matches: args.files_with_matches,
        files_without_match: args.files_without_match,
        before_context,
        after_context,
        utf8: args.utf8,
        multiline: args.multiline,
        invert_match: args.invert_match,
        whole_word: args.word,
        max_count: args.max_count,
        quiet: args.quiet,
        colors,
        show_filename: multiple_inputs && !args.count && output_format != OutputFormat::Heading,
        column: show_column,
        byte_offset: args.byte_offset,
        only_matching: args.only_matching,
        output_format,
    };

    let stats = Stats::default();
    let max_reached = AtomicBool::new(false);
    let mut output = io::stdout().lock();
    let mut any_match = false;
    let mut file_counts: Vec<(String, usize)> = Vec::new();

    // Handle stdin
    if is_stdin {
        match process_stdin(&config, &mut output, &stats, &max_reached) {
            Ok(result) => {
                if result.match_count > 0 {
                    any_match = true;
                }
                if args.count {
                    println!("{}", result.match_count);
                }
            }
            Err(e) => {
                if e.kind() != io::ErrorKind::BrokenPipe {
                    eprintln!("Error reading stdin: {}", e);
                    process::exit(1);
                }
            }
        }
    } else {
        // Process files and directories
        let walker = build_walker(&args.inputs, &args);

        for result in walker {
            // Check if we've hit global max count
            if max_reached.load(Ordering::Relaxed) {
                break;
            }

            let entry = match result {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("Warning: {}", e);
                    continue;
                }
            };

            // Skip directories
            if !entry.file_type().map_or(false, |ft| ft.is_file()) {
                continue;
            }

            // Apply glob filters manually since add_custom_ignore_filename doesn't work as expected
            if let Some(ref globs) = args.glob {
                let path_str = entry.path().to_string_lossy();
                let mut matches_glob = false;
                for glob in globs {
                    if let Ok(pattern) = glob::Pattern::new(glob) {
                        if pattern.matches(&path_str)
                            || pattern.matches(entry.file_name().to_string_lossy().as_ref())
                        {
                            matches_glob = true;
                            break;
                        }
                    }
                }
                if !matches_glob {
                    continue;
                }
            }

            match process_file(
                entry.path(),
                &config,
                &mut output,
                &stats,
                &max_reached,
                !args.binary,
            ) {
                Ok(Some(result)) => {
                    if result.match_count > 0 {
                        any_match = true;
                        if args.files_with_matches {
                            let sep = if args.null { "\0" } else { "\n" };
                            print!("{}{}", result.path, sep);
                        }
                    } else if args.files_without_match {
                        let sep = if args.null { "\0" } else { "\n" };
                        print!("{}{}", result.path, sep);
                    }
                    if args.count {
                        file_counts.push((result.path, result.match_count));
                    }
                }
                Ok(None) => {
                    // Binary file skipped
                }
                Err(e) => {
                    if e.kind() == io::ErrorKind::BrokenPipe {
                        process::exit(0);
                    }
                    eprintln!("Warning: {}: {}", entry.path().display(), e);
                }
            }
        }

        // Print counts for multi-file mode
        if args.count && !file_counts.is_empty() {
            for (path, count) in &file_counts {
                if multiple_inputs {
                    println!("{}:{}", path, count);
                } else {
                    println!("{}", count);
                }
            }
        }
    }

    // Print statistics
    if args.stats {
        print_stats(&stats, start_time.elapsed());
    }

    // Exit with status 1 if no matches found (like grep)
    if !any_match && !args.quiet {
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(pattern: &[u8]) -> SearchConfig<'_> {
        SearchConfig {
            pattern,
            ignore_case: false,
            line_number: false,
            count: false,
            files_with_matches: false,
            files_without_match: false,
            before_context: 0,
            after_context: 0,
            utf8: false,
            multiline: false,
            invert_match: false,
            whole_word: false,
            max_count: None,
            quiet: false,
            colors: Colors::disabled(),
            show_filename: false,
            column: false,
            byte_offset: false,
            only_matching: false,
            output_format: OutputFormat::Standard,
        }
    }

    /// Test helper: check if pattern matches in line
    fn has_match(line: &[u8], pattern: &[u8], ignore_case: bool, whole_word: bool) -> bool {
        MatchIter::new(line, pattern, ignore_case, whole_word)
            .next()
            .is_some()
    }

    #[test]
    fn test_match_iter_basic() {
        let line = b"hello world";
        assert!(has_match(line, b"hello", false, false));
        assert!(has_match(line, b"world", false, false));
        assert!(!has_match(line, b"foo", false, false));
    }

    #[test]
    fn test_match_iter_case_insensitive() {
        let line = b"Hello World";
        assert!(has_match(line, b"hello", true, false));
        assert!(has_match(line, b"WORLD", true, false));
        assert!(!has_match(line, b"foo", true, false));
    }

    #[test]
    fn test_match_iter_word_boundary() {
        let line = b"hello world";
        assert!(has_match(line, b"hello", false, true));
        assert!(has_match(line, b"world", false, true));
        assert!(!has_match(line, b"ello", false, true)); // Not at word boundary
        assert!(!has_match(line, b"worl", false, true)); // Not at word boundary
    }

    #[test]
    fn test_match_iter_word_boundary_multiple() {
        // Test that word boundary check finds matches after non-boundary matches
        let line = b"fn_name fn";
        assert!(has_match(line, b"fn", false, true)); // Should find standalone "fn"
    }

    #[test]
    fn test_line_matches_invert() {
        let line = b"hello world";
        let mut config = make_config(b"hello");
        assert!(line_matches(line, &config));

        config.invert_match = true;
        assert!(!line_matches(line, &config));

        let line2 = b"goodbye world";
        assert!(line_matches(line2, &config)); // Inverted: doesn't contain "hello"
    }

    #[test]
    fn test_is_binary() {
        assert!(is_binary(b"hello\0world"));
        assert!(!is_binary(b"hello world"));
        assert!(!is_binary(b"hello\nworld\n"));
    }

    #[test]
    fn test_search_basic() {
        let data = b"line1\nerror here\nline3\n";
        let config = make_config(b"error");
        let max_reached = AtomicBool::new(false);
        let mut output = Vec::new();

        let result = search_mmap(data, "test.txt", &config, &mut output, &max_reached).unwrap();

        assert_eq!(result.match_count, 1);
        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("error here"));
    }

    #[test]
    fn test_search_with_context() {
        let data = b"line1\nline2\nerror here\nline4\nline5\n";
        let mut config = make_config(b"error");
        config.before_context = 1;
        config.after_context = 1;
        let max_reached = AtomicBool::new(false);
        let mut output = Vec::new();

        search_mmap(data, "test.txt", &config, &mut output, &max_reached).unwrap();

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("line2"));
        assert!(output_str.contains("error here"));
        assert!(output_str.contains("line4"));
    }

    #[test]
    fn test_search_max_count() {
        let data = b"error1\nerror2\nerror3\nerror4\n";
        let mut config = make_config(b"error");
        config.max_count = Some(2);
        let max_reached = AtomicBool::new(false);
        let mut output = Vec::new();

        let result = search_mmap(data, "test.txt", &config, &mut output, &max_reached).unwrap();

        assert_eq!(result.match_count, 2);
        assert!(max_reached.load(Ordering::Relaxed));
    }

    #[test]
    fn test_highlight_line() {
        let line = b"hello world hello";
        let mut config = make_config(b"hello");
        config.colors = Colors::enabled();

        let mut buffer = Vec::new();
        let len = highlight_line(line, &config, &mut buffer);
        assert!(len > 0);
        let result = String::from_utf8(buffer).unwrap();

        assert!(result.contains("\x1b[1;31m")); // Contains highlight
        assert!(result.contains("\x1b[0m")); // Contains reset
    }

    #[test]
    fn test_multiline_search() {
        let data = b"hello\nworld\nfoo bar\n";
        let mut config = make_config(b"hello\nworld");
        config.multiline = true;
        let max_reached = AtomicBool::new(false);
        let mut output = Vec::new();

        let result = search_mmap(data, "test.txt", &config, &mut output, &max_reached).unwrap();

        assert_eq!(result.match_count, 1);
    }
}
