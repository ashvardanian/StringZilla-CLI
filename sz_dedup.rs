//! SIMD-accelerated line deduplication utility
//!
//! Remove duplicate lines from files, keeping the first occurrence of each unique line.
//! Uses StringZilla's SIMD hash for fast deduplication with minimal memory overhead.
//!
//! # Algorithm
//!
//! This implementation uses an open-addressed flat hash set with linear probing:
//!
//! ```text
//! AppendOnlyFlatHashSet: Vec<LineEntry>
//!     slot[0]: { hash: 0x1234,    offset: 0,         length: 10       }
//!     slot[1]: { hash: 0,         offset: u64::MAX,  length: u64::MAX }  ← empty
//!     slot[2]: { hash: 0xABCD,    offset: 15,        length: 8        }
//!     ...
//!
//! LineEntry = 24 bytes (hash: u64, offset: u64, length: u64)
//! ```
//!
//! ## Open Addressing with Linear Probing
//!
//! - Insert: `slot = hash & mask`, probe forward until empty slot
//! - Lookup: `slot = hash & mask`, probe forward checking hash matches
//! - Empty slots marked by `offset = u64::MAX` (impossible for real files)
//!
//! ## Growth Strategy
//!
//! - Start with 1024 slots (24 KB)
//! - Grow 2x when load factor exceeds 60%
//! - Rehash all entries into new larger array
//!
//! ## In-Place Compaction
//!
//! When modifying a file in-place, the algorithm uses two pointers:
//!
//! ```text
//! read_pos ────────────────────────────►
//!     ┌────────┬────────┬────────┬────────┬────────┐
//!     │ line A │ line B │ line A │ line C │ line B │  (input)
//!     └────────┴────────┴────────┴────────┴────────┘
//!
//! write_pos ───────────────►
//!     ┌────────┬────────┬────────┐
//!     │ line A │ line B │ line C │  (compacted output)
//!     └────────┴────────┴────────┘
//! ```
//!
//! **Invariant**: `write_pos ≤ read_pos` always holds, ensuring we never overwrite
//! unread data. After compaction, the file is truncated to the new length.
//!
//! ## Case-Insensitive Mode
//!
//! For `-i` mode, uses proper Unicode case folding via `utf8_case_fold()`:
//!
//! 1. **Hashing**: Case-fold line into scratch buffer, then hash the folded form
//! 2. **Collision check**: Use `utf8_case_insensitive_order()` directly on original
//!    lines (no re-folding needed for comparison)
//!
//! # Examples
//!
//! ```bash
//! # Remove duplicates in-place (modifies file directly)
//! sz-dedup file.txt
//!
//! # Case-insensitive deduplication (full Unicode support)
//! sz-dedup -i file.txt
//!
//! # Output to different file (streaming mode)
//! sz-dedup file.txt -o unique.txt
//!
//! # From stdin to stdout
//! cat file.txt | sz-dedup
//!
//! # Show count of unique lines
//! sz-dedup -c file.txt
//! ```

use std::cmp::Ordering;
use std::io::{self, Write};
use std::process;

use clap::Parser;
use stringzilla::sz::{self, find_newline_utf8};

mod shared;
use shared::*;

// region: AppendOnlyFlatHashSet

/// Entry in the flat hash set. 24 bytes total.
#[derive(Clone, Copy)]
struct LineEntry {
    hash: u64,
    offset: u64,
    length: u64,
}

impl LineEntry {
    /// Sentinel value for empty slots (offset=u64::MAX is impossible for real files)
    const EMPTY: Self = Self {
        hash: 0,
        offset: u64::MAX,
        length: u64::MAX,
    };

    #[inline]
    fn is_empty(&self) -> bool {
        self.offset == u64::MAX
    }
}

impl Default for LineEntry {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Open-addressed hash set with linear probing.
/// Grows 2x when load factor exceeds 60%.
struct AppendOnlyFlatHashSet {
    slots: Vec<LineEntry>,
    populated_count: usize,
}

impl AppendOnlyFlatHashSet {
    /// Create a new hash set with initial capacity of 1024 slots (24 KB)
    fn new() -> Self {
        const INITIAL_CAPACITY: usize = 1024;
        Self {
            slots: vec![LineEntry::default(); INITIAL_CAPACITY],
            populated_count: 0,
        }
    }

    /// Mask for fast modulo via bitwise AND (slots.len() - 1)
    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Find all entries with the given hash.
    /// Returns an iterator that probes from the hash's home slot until an empty slot.
    #[inline]
    fn find(&self, hash: u64) -> FindIter<'_> {
        let mask = self.mask();
        FindIter {
            slots: &self.slots,
            target_hash: hash,
            slot: (hash as usize) & mask,
            done: false,
        }
    }

    /// Insert a new entry. Grows the table if load factor > 60%.
    #[inline]
    fn insert(&mut self, hash: u64, offset: u64, length: u64) {
        // Grow if load factor > 60%
        if self.populated_count * 100 > self.slots.len() * 60 {
            self.grow();
        }

        self.insert_no_grow(hash, offset, length);
    }

    /// Insert without checking load factor (used during rehash)
    #[inline]
    fn insert_no_grow(&mut self, hash: u64, offset: u64, length: u64) {
        let mask = self.mask();
        let mut slot = (hash as usize) & mask;
        while !self.slots[slot].is_empty() {
            slot = (slot + 1) & mask;
        }
        self.slots[slot] = LineEntry {
            hash,
            offset,
            length,
        };
        self.populated_count += 1;
    }

    /// Double the capacity and rehash all entries
    fn grow(&mut self) {
        let new_cap = self.slots.len() * 2;
        let old_slots = std::mem::replace(&mut self.slots, vec![LineEntry::default(); new_cap]);
        self.populated_count = 0;

        for entry in old_slots {
            if !entry.is_empty() {
                self.insert_no_grow(entry.hash, entry.offset, entry.length);
            }
        }
    }
}

/// Iterator over entries matching a specific hash
struct FindIter<'a> {
    slots: &'a [LineEntry],
    target_hash: u64,
    slot: usize,
    done: bool,
}

impl<'a> Iterator for FindIter<'a> {
    type Item = &'a LineEntry;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let mask = self.slots.len() - 1;
        loop {
            let entry = &self.slots[self.slot];
            self.slot = (self.slot + 1) & mask;

            if entry.is_empty() {
                self.done = true;
                return None;
            }

            if entry.hash == self.target_hash {
                return Some(entry);
            }
        }
    }
}

// endregion: AppendOnlyFlatHashSet

// region: Hash and Comparison Utilities

/// Compute hash for a line, using case-folding if ignore_case is true.
#[inline]
fn compute_hash(line: &[u8], ignore_case: bool, scratch: &mut Vec<u8>) -> u64 {
    if ignore_case {
        // UTF-8 case folding can expand characters (e.g., ß → ss), max ~3x
        scratch.clear();
        scratch.resize(line.len().saturating_mul(3).max(64), 0);
        let folded_len = sz::utf8_case_fold(line, &mut scratch[..]);
        sz::hash(&scratch[..folded_len])
    } else {
        sz::hash(line)
    }
}

/// Check if two lines are equal, using case-insensitive comparison if needed.
#[inline]
fn lines_equal(a: &[u8], b: &[u8], ignore_case: bool) -> bool {
    if ignore_case {
        sz::utf8_case_insensitive_order(a, b) == Ordering::Equal
    } else {
        a == b
    }
}

// endregion: Hash and Comparison Utilities

// region: Deduplication Functions

/// Deduplicate lines in-place, compacting the buffer.
///
/// Returns (new_length, unique_count).
/// When `utf8` is true, handles all Unicode newlines (LF, CR, CRLF, NEL, LS, PS).
/// When `utf8` is false, only handles LF newlines.
fn dedup_in_place(data: &mut [u8], ignore_case: bool, utf8: bool) -> (usize, usize) {
    let mut seen = AppendOnlyFlatHashSet::new();
    let mut scratch = Vec::new();

    let mut read_pos: usize = 0;
    let mut write_pos: usize = 0;
    let mut unique_count: usize = 0;

    while read_pos < data.len() {
        // Find end of current line
        let (line_end, newline_len) = if utf8 {
            // UTF-8 aware: handles LF, CR, CRLF, NEL, LS, PS
            match find_newline_utf8(&data[read_pos..]) {
                Some(span) => (read_pos + span.offset, span.length),
                None => (data.len(), 0),
            }
        } else {
            // Byte-level: only LF
            match sz::find(&data[read_pos..], b"\n") {
                Some(pos) => (read_pos + pos, 1),
                None => (data.len(), 0),
            }
        };

        let line_len = line_end - read_pos;

        // Compute hash
        let hash = compute_hash(&data[read_pos..line_end], ignore_case, &mut scratch);

        // Check for duplicate against already-written lines in [0..write_pos]
        let is_duplicate = seen.find(hash).any(|entry| {
            let existing = &data[entry.offset as usize..(entry.offset + entry.length) as usize];
            let current = &data[read_pos..line_end];
            lines_equal(existing, current, ignore_case)
        });

        if !is_duplicate {
            // Record this line's position (in compacted buffer)
            seen.insert(hash, write_pos as u64, line_len as u64);

            // Copy line to write position if needed
            if write_pos != read_pos {
                data.copy_within(read_pos..line_end, write_pos);
            }
            write_pos += line_len;

            // Copy original newline sequence if present
            if newline_len > 0 {
                data.copy_within(line_end..line_end + newline_len, write_pos);
                write_pos += newline_len;
            }

            unique_count += 1;
        }

        // Advance read position past line and newline
        read_pos = line_end + newline_len;
    }

    (write_pos, unique_count)
}

/// Deduplicate lines, writing to output stream (for stdin or explicit output file).
/// When `utf8` is true, handles all Unicode newlines (LF, CR, CRLF, NEL, LS, PS).
/// When `utf8` is false, only handles LF newlines.
/// Output is normalized to LF newlines.
fn dedup_to_writer(
    data: &[u8],
    output: &mut dyn Write,
    ignore_case: bool,
    utf8: bool,
) -> io::Result<usize> {
    let mut seen = AppendOnlyFlatHashSet::new();
    let mut scratch = Vec::new();
    let mut unique_count = 0;

    let lines = LineIter::new(data, utf8);

    for line in lines {
        let line_offset = line.as_ptr() as usize - data.as_ptr() as usize;
        let hash = compute_hash(line, ignore_case, &mut scratch);

        let is_duplicate = seen.find(hash).any(|entry| {
            let existing = &data[entry.offset as usize..(entry.offset + entry.length) as usize];
            lines_equal(existing, line, ignore_case)
        });

        if !is_duplicate {
            seen.insert(hash, line_offset as u64, line.len() as u64);
            output.write_all(line)?;
            output.write_all(b"\n")?;
            unique_count += 1;
        }
    }

    output.flush()?;
    Ok(unique_count)
}

// endregion: Deduplication Functions

// region: CLI

/// Deduplicate lines in files
#[derive(Parser)]
#[command(name = "sz-dedup")]
#[command(version, about = "SIMD-accelerated line deduplication", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output file (use '-' or omit for stdout, or omit for in-place)
    #[arg(short, long)]
    output: Option<String>,

    /// Case-insensitive deduplication (full Unicode case folding)
    #[arg(short = 'i', long)]
    ignore_case: bool,

    /// Show count of unique lines
    #[arg(short = 'c', long)]
    count: bool,

    /// Enable UTF-8 mode (handle Unicode newlines: CR, CRLF, NEL, LS, PS)
    #[arg(long)]
    utf8: bool,
}

fn main() {
    let args = Args::parse();

    // UTF-8 mode is implicit when case-insensitive (case folding requires UTF-8)
    let utf8_mode = args.utf8 || args.ignore_case;

    // Determine mode: in-place (file, no output) vs streaming (stdin or explicit output)
    let in_place =
        args.output.is_none() && args.input.is_some() && args.input.as_deref() != Some("-");

    let unique_count = if in_place {
        // In-place mode: mutable mmap, compact, truncate
        let input_path = args.input.as_ref().unwrap();

        let mut input = match get_input_mutable(input_path) {
            Ok(input) => input,
            Err(e) => {
                eprintln!("Error opening file for in-place modification: {}", e);
                process::exit(1);
            }
        };

        let data = input.as_mut_bytes().unwrap();
        let (new_len, unique_count) = dedup_in_place(data, args.ignore_case, utf8_mode);

        if let Err(e) = input.truncate_and_flush(new_len as u64) {
            eprintln!("Error truncating file: {}", e);
            process::exit(1);
        }

        unique_count
    } else {
        // Streaming mode: read-only input, write to output
        let input = match get_input(args.input.as_deref()) {
            Ok(input) => input,
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                process::exit(1);
            }
        };

        let data = input.as_bytes();

        let mut output = match get_output(args.output.as_deref()) {
            Ok(output) => output,
            Err(e) => {
                eprintln!("Error opening output: {}", e);
                process::exit(1);
            }
        };

        match dedup_to_writer(data, &mut output, args.ignore_case, utf8_mode) {
            Ok(count) => count,
            Err(e) => {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    process::exit(0);
                }
                eprintln!("Error deduplicating: {}", e);
                process::exit(1);
            }
        }
    };

    if args.count {
        eprintln!("{} unique lines", unique_count);
    }
}

// endregion: CLI

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_hash_set_basic() {
        let mut set = AppendOnlyFlatHashSet::new();
        set.insert(123, 0, 10);
        set.insert(456, 20, 5);
        set.insert(123, 50, 8); // Same hash, different entry

        assert_eq!(set.populated_count, 3);
        assert_eq!(set.find(123).count(), 2);
        assert_eq!(set.find(456).count(), 1);
        assert_eq!(set.find(789).count(), 0);
    }

    #[test]
    fn flat_hash_set_growth() {
        let mut set = AppendOnlyFlatHashSet::new();
        // Insert more than 60% of initial capacity to trigger growth
        for i in 1..=700 {
            set.insert(i, i * 10, i);
        }
        assert!(set.slots.len() > 1024);
        assert_eq!(set.populated_count, 700);

        // Verify all entries are still findable
        for i in 1..=700 {
            assert_eq!(set.find(i).count(), 1);
        }
    }

    #[test]
    fn dedup_in_place_basic() {
        let mut data = b"line1\nline2\nline1\nline3\n".to_vec();
        let (new_len, count) = dedup_in_place(&mut data, false, false);

        assert_eq!(count, 3);
        let result = String::from_utf8(data[..new_len].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn dedup_in_place_case_insensitive() {
        let mut data = b"Hello\nhello\nworld\nWORLD\n".to_vec();
        let (new_len, count) = dedup_in_place(&mut data, true, true);

        assert_eq!(count, 2);
        let result = String::from_utf8(data[..new_len].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["Hello", "world"]);
    }

    #[test]
    fn dedup_in_place_unicode() {
        let mut data = "MÜNCHEN\nmünchen\nberlin\n".as_bytes().to_vec();
        let (new_len, count) = dedup_in_place(&mut data, true, true);

        assert_eq!(count, 2);
        let result = String::from_utf8(data[..new_len].to_vec()).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["MÜNCHEN", "berlin"]);
    }

    #[test]
    fn dedup_in_place_all_duplicates() {
        let mut data = b"dup\ndup\ndup\ndup\n".to_vec();
        let (new_len, count) = dedup_in_place(&mut data, false, false);

        assert_eq!(count, 1);
        assert_eq!(&data[..new_len], b"dup\n");
    }

    #[test]
    fn dedup_in_place_no_change() {
        let mut data = b"a\nb\nc\n".to_vec();
        let original_len = data.len();
        let (new_len, count) = dedup_in_place(&mut data, false, false);

        assert_eq!(count, 3);
        assert_eq!(new_len, original_len);
    }

    #[test]
    fn dedup_to_writer_basic() {
        let data = b"line1\nline2\nline1\nline3\n";
        let mut output = Vec::new();

        let count = dedup_to_writer(data, &mut output, false, false).unwrap();

        assert_eq!(count, 3);
        let result = String::from_utf8(output).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["line1", "line2", "line3"]);
    }

    #[test]
    fn dedup_to_writer_case_insensitive() {
        let data = b"Hello\nhello\nHELLO\nworld\n";
        let mut output = Vec::new();

        let count = dedup_to_writer(data, &mut output, true, true).unwrap();

        assert_eq!(count, 2);
        let result = String::from_utf8(output).unwrap();
        let lines: Vec<_> = result.lines().collect();
        assert_eq!(lines, vec!["Hello", "world"]);
    }

    #[test]
    fn dedup_preserves_first() {
        let data = b"First\nfirst\nFIRST\n";
        let mut output = Vec::new();

        dedup_to_writer(data, &mut output, true, true).unwrap();

        let result = String::from_utf8(output).unwrap();
        assert_eq!(result, "First\n");
    }

    #[test]
    fn dedup_empty() {
        let data = b"";
        let mut output = Vec::new();

        let count = dedup_to_writer(data, &mut output, false, false).unwrap();

        assert_eq!(count, 0);
        assert!(output.is_empty());
    }

    #[test]
    fn dedup_empty_lines() {
        let data = b"\n\n\ntext\n\n";
        let mut output = Vec::new();

        let count = dedup_to_writer(data, &mut output, false, false).unwrap();

        assert_eq!(count, 2); // "" and "text"
    }

    #[test]
    fn line_entry_empty() {
        assert!(LineEntry::EMPTY.is_empty());
        assert!(LineEntry::default().is_empty());

        let occupied = LineEntry {
            hash: 123,
            offset: 0,
            length: 10,
        };
        assert!(!occupied.is_empty());
    }
}

// endregion: Tests
