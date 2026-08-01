//! Shared utilities for sz-cli tools
//!
//! Provides common functionality for all sz-* command-line utilities including
//! input/output handling, UTF-8 validation, and line iteration.
//!
//! Note: Not all functions are used by every binary. The #[allow(dead_code)]
//! attributes prevent warnings for legitimately shared code.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::process;
use std::sync::OnceLock;

use memmap2::{Mmap, MmapMut};
use stringzilla::sz;
use stringzilla::sz::{FindSplits, StringZillableBinary, StringZillableUnary, Utf8SplitNewlines};

// region: Input Sources

/// Represents the input source - either a memory-mapped file or buffered stdin
#[allow(dead_code)]
pub enum InputSource {
    /// Memory-mapped file for zero-copy access (read-only)
    MappedFile(Mmap),
    /// Mutable memory-mapped file for in-place modification
    MutableMappedFile(MmapMut, File),
    /// Buffered stdin data
    Buffer(Vec<u8>),
}

#[allow(dead_code)]
impl InputSource {
    /// Get the input data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            InputSource::MappedFile(mmap) => &mmap[..],
            InputSource::MutableMappedFile(mmap, _) => &mmap[..],
            InputSource::Buffer(buf) => buf,
        }
    }

    /// Get mutable access to the input data (only for mutable sources)
    pub fn as_mut_bytes(&mut self) -> Option<&mut [u8]> {
        match self {
            InputSource::MutableMappedFile(mmap, _) => Some(&mut mmap[..]),
            InputSource::Buffer(buf) => Some(&mut buf[..]),
            InputSource::MappedFile(_) => None,
        }
    }

    /// Flush changes and truncate file to new length.
    /// Only works for MutableMappedFile; no-op for other variants.
    pub fn truncate_and_flush(&mut self, new_len: u64) -> io::Result<()> {
        match self {
            InputSource::MutableMappedFile(mmap, file) => {
                mmap.flush()?;
                file.set_len(new_len)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Read from a file path or stdin. Only regular files are mapped; pipes and
/// character devices, which `<(cmd)` and `/dev/stdin` resolve to, are buffered.
#[allow(dead_code)]
pub fn get_input(path: Option<&str>) -> io::Result<InputSource> {
    let Some(path) = path.filter(|path| *path != "-") else {
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        return Ok(InputSource::Buffer(buffer));
    };
    open_input(Path::new(path))
}

/// Map `path` for zero-copy access, buffering it when that is not possible.
#[allow(dead_code)]
pub fn open_input(path: &Path) -> io::Result<InputSource> {
    let mut file = File::open(path)?;
    if file.metadata().is_ok_and(|metadata| metadata.is_file()) {
        // An empty file cannot be mapped either, so fall through on any failure.
        if let Ok(mmap) = unsafe { Mmap::map(&file) } {
            return Ok(InputSource::MappedFile(mmap));
        }
    }

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(InputSource::Buffer(buffer))
}

/// Whether a walked entry should be read: any regular file, plus explicitly named
/// non-directories, which is what lets `<(cmd)` and `/dev/stdin` through.
#[allow(dead_code)]
pub fn is_readable_entry(entry: &ignore::DirEntry) -> bool {
    match entry.file_type() {
        Some(kind) if kind.is_file() => true,
        Some(kind) => entry.depth() == 0 && !kind.is_dir(),
        None => false,
    }
}

/// Create a mutable InputSource for in-place file modification
#[allow(dead_code)]
pub fn get_input_mutable(path: &str) -> io::Result<InputSource> {
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    let mmap = unsafe { MmapMut::map_mut(&file)? };
    Ok(InputSource::MutableMappedFile(mmap, file))
}

/// Create an output writer from either a file path or stdout
#[allow(dead_code)]
pub fn get_output(path: Option<&str>) -> io::Result<Box<dyn Write>> {
    match path {
        None | Some("-") => Ok(Box::new(stdout_writer())),
        Some(path) => {
            let file = File::create(path)?;
            Ok(Box::new(BufWriter::new(file)))
        }
    }
}

/// Buffered, locked stdout. `io::Stdout` is line-buffered, costing a syscall per record.
///
/// `process::exit` skips the buffer's `Drop`, so exit through [`ExitCode::exit`].
#[allow(dead_code)]
pub fn stdout_writer() -> BufWriter<io::StdoutLock<'static>> {
    BufWriter::new(io::stdout().lock())
}

// endregion: Input Sources

// region: Line Iteration

/// Which newline set [`LineIter`] splits on.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Newlines {
    /// Byte-level: only LF (`\n`).
    Lf,
    /// All seven Unicode newline characters (LF, VT, FF, CR, NEL, LS, PS),
    /// with CRLF collapsed into a single break.
    Unicode,
}

#[allow(dead_code)]
impl Newlines {
    /// Map a `--utf8` flag to the newline set (LF-only when `false`).
    #[inline]
    pub fn from_utf8(utf8: bool) -> Self {
        if utf8 {
            Newlines::Unicode
        } else {
            Newlines::Lf
        }
    }
}

/// Iterator over lines with terminator semantics — a trailing newline does not
/// yield a final empty line (matching `str::lines`). Newline detection is delegated
/// to StringZilla's native split kernels: byte-level LF (`sz_splits(b"\n")`) or the
/// seven Unicode newline characters plus CRLF-as-one (`sz_utf8_split_newlines`).
/// Those kernels split on *separators* (a trailing delimiter emits a final empty
/// segment), so we drop that single trailing empty to recover terminator semantics.
//
// Boxing the larger variant would add heap indirection on every `next()`; the
// iterator is built once per file (not per line), so the size gap is a one-time
// stack cost, not a hot-path allocation.
#[allow(dead_code, clippy::large_enum_variant)]
pub enum LineIter<'a> {
    Byte(std::iter::Peekable<FindSplits<'a>>),
    Utf8(std::iter::Peekable<Utf8SplitNewlines<'a>>),
}

#[allow(dead_code)]
impl<'a> LineIter<'a> {
    /// Create a line iterator over the chosen [`Newlines`] set.
    pub fn new(data: &'a [u8], newlines: Newlines) -> Self {
        match newlines {
            Newlines::Unicode => LineIter::Utf8(data.sz_utf8_split_newlines().peekable()),
            Newlines::Lf => LineIter::Byte(data.sz_splits(b"\n").peekable()),
        }
    }
}

impl<'a> Iterator for LineIter<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            LineIter::Byte(iter) => drop_trailing_empty(iter),
            LineIter::Utf8(iter) => drop_trailing_empty(iter),
        }
    }
}

/// Yield the next segment, suppressing a single trailing empty segment (the one a
/// separator-splitter emits after a trailing delimiter) for line-terminator semantics.
/// Interior blank lines are preserved — unlike `.skip_empty()`, which drops them all.
#[inline]
fn drop_trailing_empty<'a, I: Iterator<Item = &'a [u8]>>(
    iter: &mut std::iter::Peekable<I>,
) -> Option<&'a [u8]> {
    let line = iter.next()?;
    if line.is_empty() && iter.peek().is_none() {
        return None;
    }
    Some(line)
}

/// Byte offset of `segment` within `data`. Segmenters yield borrowed subslices and
/// expose no offsets accessor, so the pointer difference is the offset.
#[allow(dead_code)]
#[inline]
pub fn offset_within(data: &[u8], segment: &[u8]) -> usize {
    let bounds = data.as_ptr_range();
    debug_assert!(
        bounds.start <= segment.as_ptr() && segment.as_ptr() <= bounds.end,
        "segment must borrow from data"
    );
    segment.as_ptr() as usize - data.as_ptr() as usize
}

// endregion: Line Iteration

// region: Machine-Readable Output

/// What terminates each output record.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Terminator {
    /// A newline, which is lossy for records that contain one.
    Newline,
    /// A NUL byte, which no text record can contain.
    Null,
}

#[allow(dead_code)]
impl Terminator {
    /// Map a `-0` flag to the terminator (newline when `false`).
    #[inline]
    pub fn from_null(null: bool) -> Self {
        if null {
            Terminator::Null
        } else {
            Terminator::Newline
        }
    }

    /// The byte that ends a record.
    #[inline]
    pub fn as_byte(self) -> u8 {
        match self {
            Terminator::Newline => b'\n',
            Terminator::Null => 0,
        }
    }
}

/// Trim `line` to at most `max_bytes`, returning the kept prefix and whether
/// anything was dropped. Lands on a codepoint boundary, so combining marks and
/// emoji sequences can still be split.
#[allow(dead_code)]
pub fn truncate_at_character(line: &[u8], max_bytes: usize) -> (&[u8], bool) {
    if line.len() <= max_bytes {
        return (line, false);
    }
    // A UTF-8 sequence is at most four bytes, so bound the walk: malformed input
    // must not send us scanning back through the whole line.
    let floor = max_bytes.saturating_sub(3);
    let mut end = max_bytes;
    while end > floor && (line[end] & 0xC0) == 0x80 {
        end -= 1;
    }
    (&line[..end], true)
}

/// The bytes JSON must escape. Built once; `Byteset` has no `const` constructor.
fn json_escape_byteset() -> sz::Byteset {
    static ESCAPES: OnceLock<sz::Byteset> = OnceLock::new();
    *ESCAPES.get_or_init(|| {
        let mut set = sz::Byteset::from(b"\"\\".as_slice());
        for control in 0u8..0x20 {
            set.add_u8(control);
        }
        set
    })
}

/// Write JSON-escaped bytes, bulk-writing the run between escapes. Bytes at or
/// above 0x80 pass through, so the result is valid JSON only for valid UTF-8.
#[allow(dead_code)]
pub fn json_escape_to(output: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
    let escapes = json_escape_byteset();
    let mut rest = data;
    loop {
        let Some(offset) = sz::find_byteset(rest, escapes) else {
            return output.write_all(rest);
        };
        output.write_all(&rest[..offset])?;
        let byte = rest[offset];
        rest = &rest[offset + 1..];
        match byte {
            b'"' => output.write_all(br#"\""#)?,
            b'\\' => output.write_all(br"\\")?,
            b'\n' => output.write_all(br"\n")?,
            b'\r' => output.write_all(br"\r")?,
            b'\t' => output.write_all(br"\t")?,
            _ => output.write_all(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX_DIGITS[(byte >> 4) as usize],
                HEX_DIGITS[(byte & 0x0F) as usize],
            ])?,
        }
    }
}

/// Write the `{"text":"…"}` wrapper that every path and line field uses.
#[allow(dead_code)]
pub fn json_text_field_to(output: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    output.write_all(br#"{"text":""#)?;
    json_escape_to(output, data)?;
    output.write_all(br#""}"#)
}

/// Render `value` with `,` between thousands groups. `usize::MAX` is 20 digits
/// plus 6 separators, exactly `buffer`'s length.
#[allow(dead_code)]
pub fn format_grouped_number(buffer: &mut [u8; 26], value: usize) -> &str {
    let mut written = buffer.len();
    let mut digits_in_group = 0;
    let mut remaining = value;
    loop {
        if digits_in_group == 3 {
            written -= 1;
            buffer[written] = b',';
            digits_in_group = 0;
        }
        written -= 1;
        buffer[written] = b'0' + (remaining % 10) as u8;
        digits_in_group += 1;
        remaining /= 10;
        if remaining == 0 {
            break;
        }
    }
    // Only ASCII digits and commas were written.
    core::str::from_utf8(&buffer[written..]).unwrap_or("")
}

/// Write `value` with `,` between thousands groups, without allocating.
#[allow(dead_code)]
pub fn write_grouped_number(output: &mut dyn Write, value: usize) -> io::Result<()> {
    let mut buffer = [0u8; 26];
    output.write_all(format_grouped_number(&mut buffer, value).as_bytes())
}

// endregion: Machine-Readable Output

// region: Process Exit Conventions

/// Process exit status, on `grep`'s model.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ExitCode {
    Success = 0,  // Ran, and produced a result
    NoResult = 1, // Ran, but found nothing
    Error = 2,    // Did not run to completion
}

#[allow(dead_code)]
impl ExitCode {
    /// Map a "found something" flag to the success/no-result pair.
    #[inline]
    pub fn from_found(found: bool) -> Self {
        if found {
            ExitCode::Success
        } else {
            ExitCode::NoResult
        }
    }

    /// Flush `output` and terminate the process with this status.
    /// `process::exit` skips `BufWriter`'s `Drop`, so the flush is not optional.
    pub fn exit(self, output: &mut dyn Write) -> ! {
        let _ = output.flush();
        process::exit(self as i32)
    }
}

/// Report a write failure and terminate. A broken pipe is a normal downstream
/// close, so it exits successfully; anything else exits [`ExitCode::Error`].
#[allow(dead_code)]
pub fn exit_on_write_error(output: &mut dyn Write, error: &io::Error, message: &str) -> ! {
    if error.kind() == io::ErrorKind::BrokenPipe {
        ExitCode::Success.exit(output);
    }
    eprintln!("{}: {}", message, error);
    ExitCode::Error.exit(output);
}

/// Report a fatal error and terminate with [`ExitCode::Error`].
#[allow(dead_code)]
pub fn exit_with_error(output: &mut dyn Write, error: &io::Error, message: &str) -> ! {
    eprintln!("{}: {}", message, error);
    ExitCode::Error.exit(output);
}

// endregion: Process Exit Conventions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(data: &[u8], utf8: bool) -> Vec<&[u8]> {
        LineIter::new(data, Newlines::from_utf8(utf8)).collect()
    }

    #[test]
    fn splits_lines_on_lf() {
        assert_eq!(lines(b"a\nb\nc\n", false), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn splits_lf_lines_without_trailing_newline() {
        assert_eq!(lines(b"a\nb\nc", false), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn splits_lines_on_all_unicode_newlines() {
        // LF, CR, CRLF, NEL, LINE/PARAGRAPH SEPARATOR — CRLF counts as one break.
        let data = "a\nb\r\nc\u{0085}d\u{2028}e\u{2029}".as_bytes();
        assert_eq!(lines(data, true), vec![&b"a"[..], b"b", b"c", b"d", b"e"]);
    }

    #[test]
    fn splits_unicode_lines_without_trailing_newline() {
        assert_eq!(lines(b"a\r\nb\r\nc", true), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn preserves_interior_blank_lines() {
        // Terminator semantics: a trailing newline drops only the *final* empty line;
        // interior blank lines are kept (unlike `.skip_empty()`).
        assert_eq!(lines(b"a\n\nb\n", false), vec![&b"a"[..], b"", b"b"]);
        assert_eq!(lines(b"a\r\n\r\nb\r\n", true), vec![&b"a"[..], b"", b"b"]);
    }

    #[test]
    fn yields_no_lines_on_empty_input() {
        assert!(lines(b"", false).is_empty());
        assert!(lines(b"", true).is_empty());
    }

    fn escaped(data: &[u8]) -> String {
        let mut output = Vec::new();
        json_escape_to(&mut output, data).unwrap();
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn escapes_json_quotes_and_backslashes() {
        assert_eq!(escaped(br#"say "hi""#), r#"say \"hi\""#);
        assert_eq!(escaped(br"back\slash"), r"back\\slash");
    }

    #[test]
    fn escapes_json_control_bytes() {
        assert_eq!(escaped(b"a\nb\rc\td"), r"a\nb\rc\td");
        assert_eq!(escaped(b"\x00\x1f"), r"\u0000\u001f");
    }

    #[test]
    fn passes_high_bytes_through_json_escaping() {
        // Valid UTF-8 stays verbatim, matching the existing `sz-find --json` behavior.
        assert_eq!(escaped("é".as_bytes()), "é");
        assert_eq!(escaped(b""), "");
    }

    #[test]
    fn wraps_json_text_field() {
        let mut output = Vec::new();
        json_text_field_to(&mut output, b"a\"b").unwrap();
        assert_eq!(output, br#"{"text":"a\"b"}"#);
    }

    fn grouped(value: usize) -> String {
        let mut output = Vec::new();
        write_grouped_number(&mut output, value).unwrap();
        String::from_utf8(output).unwrap()
    }

    #[test]
    fn reports_offsets_of_borrowed_segments() {
        // The offset must come from the slice, not from accumulated line lengths:
        // CRLF and the Unicode separators are not one byte wide.
        let data = "a\r\nb\u{2028}hello\n".as_bytes();
        let lines: Vec<&[u8]> = LineIter::new(data, Newlines::Unicode).collect();
        let offsets: Vec<usize> = lines
            .iter()
            .map(|line| offset_within(data, line))
            .collect();
        assert_eq!(offsets, vec![0, 3, 7]);
        assert_eq!(&data[offsets[2]..offsets[2] + 5], b"hello");
    }

    #[test]
    fn truncates_without_splitting_characters() {
        // "héllo" is 6 bytes; cutting at 2 must not split the 2-byte "é".
        let line = "héllo".as_bytes();
        assert_eq!(truncate_at_character(line, 2), ("h".as_bytes(), true));
        assert_eq!(truncate_at_character(line, 3), ("hé".as_bytes(), true));
        assert_eq!(truncate_at_character(line, 6), (line, false));
        assert_eq!(truncate_at_character(line, 99), (line, false));
        assert_eq!(truncate_at_character(b"", 4), (&b""[..], false));
    }

    #[test]
    fn groups_numbers_by_thousands() {
        assert_eq!(grouped(0), "0");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1000), "1,000");
        assert_eq!(grouped(26_804_246), "26,804,246");
        assert_eq!(grouped(usize::MAX), "18,446,744,073,709,551,615");
    }
}

// endregion: Tests
