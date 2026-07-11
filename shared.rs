//! Shared utilities for sz-cli tools
//!
//! Provides common functionality for all sz-* command-line utilities including
//! input/output handling, UTF-8 validation, and line iteration.
//!
//! Note: Not all functions are used by every binary. The #[allow(dead_code)]
//! attributes prevent warnings for legitimately shared code.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};

use memmap2::{Mmap, MmapMut};
use stringzilla::sz::{FindSplits, StringZillableBinary, StringZillableUnary, Utf8SplitNewlines};

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

/// Create an InputSource from either a file path or stdin (read-only)
#[allow(dead_code)]
pub fn get_input(path: Option<&str>) -> io::Result<InputSource> {
    match path {
        None | Some("-") => {
            let mut buffer = Vec::new();
            io::stdin().read_to_end(&mut buffer)?;
            Ok(InputSource::Buffer(buffer))
        }
        Some(path) => {
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            Ok(InputSource::MappedFile(mmap))
        }
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
        None | Some("-") => Ok(Box::new(BufWriter::new(io::stdout()))),
        Some(path) => {
            let file = File::create(path)?;
            Ok(Box::new(BufWriter::new(file)))
        }
    }
}

/// Iterator over lines with terminator semantics — a trailing newline does not
/// yield a final empty line (matching `str::lines`). Newline detection is delegated
/// to StringZilla's native split kernels: byte-level LF (`sz_splits(b"\n")`) or all
/// eight Unicode newlines incl. CRLF (`sz_utf8_split_newlines`), selected by `utf8`.
/// Those kernels split on *separators* (a trailing delimiter emits a final empty
/// segment), so we drop that single trailing empty to recover terminator semantics.
#[allow(dead_code)]
pub enum LineIter<'a> {
    Byte(std::iter::Peekable<FindSplits<'a>>),
    Utf8(std::iter::Peekable<Utf8SplitNewlines<'a>>),
}

#[allow(dead_code)]
impl<'a> LineIter<'a> {
    /// Create a line iterator; `utf8` selects all-Unicode-newlines vs LF-only.
    pub fn new(data: &'a [u8], utf8: bool) -> Self {
        if utf8 {
            LineIter::Utf8(data.sz_utf8_split_newlines().peekable())
        } else {
            LineIter::Byte(data.sz_splits(b"\n").peekable())
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


#[cfg(test)]
mod tests {
    use super::*;

    fn lines(data: &[u8], utf8: bool) -> Vec<&[u8]> {
        LineIter::new(data, utf8).collect()
    }

    #[test]
    fn lines_lf() {
        assert_eq!(lines(b"a\nb\nc\n", false), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn lines_lf_no_trailing_newline() {
        assert_eq!(lines(b"a\nb\nc", false), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn lines_utf8_all_newlines() {
        // LF, CR, CRLF, NEL, LINE/PARAGRAPH SEPARATOR — CRLF counts as one break.
        let data = "a\nb\r\nc\u{0085}d\u{2028}e\u{2029}".as_bytes();
        assert_eq!(lines(data, true), vec![&b"a"[..], b"b", b"c", b"d", b"e"]);
    }

    #[test]
    fn lines_utf8_no_trailing_newline() {
        assert_eq!(lines(b"a\r\nb\r\nc", true), vec![&b"a"[..], b"b", b"c"]);
    }

    #[test]
    fn lines_preserve_interior_blanks() {
        // Terminator semantics: a trailing newline drops only the *final* empty line;
        // interior blank lines are kept (unlike `.skip_empty()`).
        assert_eq!(lines(b"a\n\nb\n", false), vec![&b"a"[..], b"", b"b"]);
        assert_eq!(lines(b"a\r\n\r\nb\r\n", true), vec![&b"a"[..], b"", b"b"]);
    }

    #[test]
    fn lines_empty_input() {
        assert!(lines(b"", false).is_empty());
        assert!(lines(b"", true).is_empty());
    }
}
