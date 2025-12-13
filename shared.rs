//! Shared utilities for sz-cli tools
//!
//! Provides common functionality for all sz-* command-line utilities including
//! input/output handling, UTF-8 validation, and line iteration.
//!
//! Note: Not all functions are used by every binary. The #[allow(dead_code)]
//! attributes prevent warnings for legitimately shared code.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::str;

use memmap2::Mmap;
use stringzilla::sz::find;

/// Represents the input source - either a memory-mapped file or buffered stdin
#[allow(dead_code)]
pub enum InputSource {
    /// Memory-mapped file for zero-copy access
    MappedFile(Mmap),
    /// Buffered stdin data
    Buffer(Vec<u8>),
}

#[allow(dead_code)]
impl InputSource {
    /// Get the input data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            InputSource::MappedFile(mmap) => &mmap[..],
            InputSource::Buffer(buf) => buf,
        }
    }
}

/// Create an InputSource from either a file path or stdin
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

/// Validate that data is valid UTF-8
#[allow(dead_code)]
pub fn validate_utf8(data: &[u8]) -> Result<(), io::Error> {
    str::from_utf8(data).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid UTF-8 at byte {}: {}", e.valid_up_to(), e),
        )
    })?;
    Ok(())
}

/// Iterator over lines in a byte slice, using StringZilla for fast newline search
#[allow(dead_code)]
pub struct LineIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> LineIterator<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Find the next newline starting from the given position
    #[inline]
    fn find_newline(&self, start: usize) -> Option<usize> {
        if start >= self.data.len() {
            return None;
        }
        let remaining = &self.data[start..];
        find(remaining, b"\n").map(|i| start + i)
    }
}

impl<'a> Iterator for LineIterator<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }

        let line_start = self.pos;
        let line_end = self.find_newline(self.pos).unwrap_or(self.data.len());

        // Move position past the newline
        self.pos = if line_end < self.data.len() {
            line_end + 1
        } else {
            line_end
        };

        Some(&self.data[line_start..line_end])
    }
}

/// Count lines in data using SIMD-accelerated search
#[allow(dead_code)]
pub fn count_lines(data: &[u8]) -> usize {
    let mut count = 0;
    let mut pos = 0;

    while pos < data.len() {
        if let Some(found) = find(&data[pos..], b"\n") {
            count += 1;
            pos += found + 1;
        } else {
            break;
        }
    }

    // If data doesn't end with newline but has content, count the last line
    if !data.is_empty() && (data.len() == 0 || data[data.len() - 1] != b'\n') {
        count += 1;
    }

    count
}

/// Count words in data (whitespace-separated sequences)
#[allow(dead_code)]
pub fn count_words(data: &[u8]) -> usize {
    let mut count = 0;
    let mut in_word = false;

    for &byte in data {
        let is_whitespace = byte.is_ascii_whitespace();
        if !is_whitespace && !in_word {
            count += 1;
            in_word = true;
        } else if is_whitespace {
            in_word = false;
        }
    }

    count
}

/// Count UTF-8 characters (code points) in data
#[allow(dead_code)]
pub fn count_chars_utf8(data: &[u8]) -> Result<usize, io::Error> {
    let text = str::from_utf8(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid UTF-8: {}", e)))?;
    Ok(text.chars().count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_iterator() {
        let data = b"line1\nline2\nline3\n";
        let lines: Vec<_> = LineIterator::new(data).collect();

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], b"line1");
        assert_eq!(lines[1], b"line2");
        assert_eq!(lines[2], b"line3");
    }

    #[test]
    fn test_line_iterator_no_trailing_newline() {
        let data = b"line1\nline2\nline3";
        let lines: Vec<_> = LineIterator::new(data).collect();

        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2], b"line3");
    }

    #[test]
    fn test_count_lines() {
        assert_eq!(count_lines(b"line1\nline2\nline3\n"), 3);
        assert_eq!(count_lines(b"line1\nline2\nline3"), 3);
        assert_eq!(count_lines(b"single"), 1);
        assert_eq!(count_lines(b""), 0);
    }

    #[test]
    fn test_count_words() {
        assert_eq!(count_words(b"hello world"), 2);
        assert_eq!(count_words(b"  hello   world  "), 2);
        assert_eq!(count_words(b"one\ntwo\tthree"), 3);
        assert_eq!(count_words(b""), 0);
    }

    #[test]
    fn test_count_chars_utf8() {
        assert_eq!(count_chars_utf8(b"hello").unwrap(), 5);
        assert_eq!(count_chars_utf8("héllo".as_bytes()).unwrap(), 5);
        assert_eq!(count_chars_utf8("こんにちは".as_bytes()).unwrap(), 5);
        assert!(count_chars_utf8(&[0xFF, 0xFE]).is_err()); // Invalid UTF-8
    }

    #[test]
    fn test_validate_utf8() {
        assert!(validate_utf8(b"hello").is_ok());
        assert!(validate_utf8("héllo".as_bytes()).is_ok());
        assert!(validate_utf8(&[0xFF, 0xFE]).is_err());
    }
}
