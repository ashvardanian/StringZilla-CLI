//! Shared utilities for sz-cli tools
//!
//! Provides common functionality for all sz-* command-line utilities including
//! input/output handling, UTF-8 validation, and line iteration.
//!
//! Note: Not all functions are used by every binary. The #[allow(dead_code)]
//! attributes prevent warnings for legitimately shared code.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, Write};
use std::num::{NonZeroUsize, ParseIntError};
use std::ops::ControlFlow;
use std::path::Path;
use std::process;
use std::sync::OnceLock;

use memmap2::Mmap;
use stringzilla::sz;
use stringzilla::sz::{FindSplits, StringZillableBinary, StringZillableUnary, Utf8SplitNewlines};

// region: Input Sources

/// Represents the input source - either a memory-mapped file or buffered stdin
#[allow(dead_code)]
pub enum InputSource {
    /// Memory-mapped file for zero-copy access (read-only)
    MappedFile(Mmap),
    /// Buffered stdin data
    Buffer(Vec<u8>),
    /// An undrained pipe on stdin, the one source with no whole slice. Produced only
    /// by [`get_input_streaming`], so [`get_input`]'s callers never observe it.
    Pipe(io::StdinLock<'static>),
}

/// The two shapes an input takes: one whole slice, or a window to refill.
#[allow(dead_code)]
pub enum InputWindow {
    /// A source that hands over all of its bytes at once.
    Whole(InputSource),
    /// A pipe, read through one window reused for the whole stream.
    Stream(Refill<io::StdinLock<'static>>),
}

#[allow(dead_code)]
impl InputSource {
    /// Get the input data as a byte slice
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            InputSource::MappedFile(mmap) => &mmap[..],
            InputSource::Buffer(buf) => buf,
            InputSource::Pipe(_) => {
                debug_assert!(
                    false,
                    "an undrained pipe has no slice; branch on `into_window`"
                );
                &[]
            }
        }
    }

    /// Consume the source into whichever shape it can provide: one slice for every source
    /// but a true pipe, which streams through a `capacity`-byte window.
    pub fn into_window(self, capacity: usize) -> InputWindow {
        match self {
            InputSource::Pipe(reader) => InputWindow::Stream(Refill::new(reader, capacity)),
            source => InputWindow::Whole(source),
        }
    }
}

/// Read from a file path or stdin. Only regular files are mapped; pipes and
/// character devices, which `<(cmd)` and `/dev/stdin` resolve to, are buffered.
#[allow(dead_code)]
pub fn get_input(path: Option<&str>) -> io::Result<InputSource> {
    let Some(path) = path.filter(|path| *path != "-") else {
        // A redirect resolves to a regular file; only true pipes need buffering.
        if let Some(mmap) = map_stdin() {
            return Ok(InputSource::MappedFile(mmap));
        }
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        return Ok(InputSource::Buffer(buffer));
    };
    open_input(Path::new(path))
}

/// Read from a file path or stdin, leaving a true pipe undrained for [`Refill`].
/// Every other source maps or buffers exactly as [`get_input`] does, so a caller
/// branches once on [`InputSource::into_window`] and streams only the pipe.
#[allow(dead_code)]
pub fn get_input_streaming(path: Option<&str>) -> io::Result<InputSource> {
    let Some(path) = path.filter(|path| *path != "-") else {
        // A redirect resolves to a regular file; only true pipes need streaming.
        if let Some(mmap) = map_stdin() {
            return Ok(InputSource::MappedFile(mmap));
        }
        widen_stdin_pipe();
        return Ok(InputSource::Pipe(io::stdin().lock()));
    };
    open_input(Path::new(path))
}

/// Map descriptor 0 when it is a regular file read from its start.
/// `ManuallyDrop` keeps stdin open.
#[cfg(unix)]
fn map_stdin() -> Option<Mmap> {
    use std::io::Seek;
    use std::mem::ManuallyDrop;
    use std::os::fd::FromRawFd;

    let mut stdin = ManuallyDrop::new(unsafe { File::from_raw_fd(0) });
    stdin.metadata().ok()?.is_file().then_some(())?;
    // A mapping starts at byte zero, so a descriptor another command already read
    // from — `{ head -n 1; tool; } < file` — has to keep the buffered path.
    (stdin.stream_position().ok()? == 0).then_some(())?;
    unsafe { Mmap::map(&*stdin) }.ok()
}

/// Map descriptor 0 when it is a regular file. Unsupported outside Unix.
#[cfg(not(unix))]
fn map_stdin() -> Option<Mmap> {
    None
}

/// Kernel pipe capacity asked for on descriptor 0, matched to [`DEFAULT_WINDOW_BYTES`] so one
/// window fill drains a full pipe. The 64 KiB default is narrower than the 128 KiB block a
/// writer such as `cat` hands over at once, so the writer stalls mid-block and both sides pay a
/// wakeup per fragment. A request above `/proc/sys/fs/pipe-max-size` is refused outright rather
/// than granted in part, so this stays well under the 1 MiB that ships as that limit.
#[cfg(target_os = "linux")]
const PIPE_CAPACITY_BYTES: std::ffi::c_int = 256 << 10;

/// Widen descriptor 0's pipe buffer to [`PIPE_CAPACITY_BYTES`], rounded up by the kernel to a
/// power of two and to at least one page. Best effort: `F_SETPIPE_SZ` answers `EPERM` above
/// `/proc/sys/fs/pipe-max-size` without `CAP_SYS_RESOURCE`, `EBUSY` below the bytes already
/// queued, and `EINVAL` on a descriptor that is no pipe. Each of those keeps the default
/// capacity, which streams correctly and only more slowly, so the result is discarded.
///
/// Widening spends from the per-user page budget in `/proc/sys/fs/pipe-user-pages-soft`,
/// commonly 16384 pages, beyond which fresh pipes open at a single page. One process widens
/// one descriptor here; keep it that way and never loop this over many descriptors.
#[cfg(target_os = "linux")]
fn widen_stdin_pipe() {
    use std::ffi::c_int;

    /// `F_LINUX_SPECIFIC_BASE + 7`, absent from every other platform's `fcntl.h`.
    const F_SETPIPE_SZ: c_int = 1031;

    extern "C" {
        fn fcntl(descriptor: c_int, command: c_int, ...) -> c_int;
    }

    // Safety: `fcntl` reads the descriptor and the two integers, borrowing no memory.
    unsafe { fcntl(0, F_SETPIPE_SZ, PIPE_CAPACITY_BYTES) };
}

/// Widen descriptor 0's pipe buffer. `F_SETPIPE_SZ` is Linux-only, so every other platform
/// streams through whatever capacity it gives a pipe.
#[cfg(not(target_os = "linux"))]
fn widen_stdin_pipe() {}

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

/// Rewrite `path` through a sibling temporary file, renamed over it once the write has
/// reached the disk, then copied back over the original inode so symlinks and hardlinks
/// survive. In-place editing that truncates first loses the file outright when the process
/// dies mid-write; here the complete result exists before the target is touched.
#[allow(dead_code)]
pub fn write_replacing<T>(
    path: &str,
    write: impl FnOnce(&mut dyn Write) -> io::Result<T>,
) -> io::Result<T> {
    // Resolve first: editing through a symlink must change the file it names, not replace
    // the link with a regular file.
    let resolved = std::fs::canonicalize(path)?;
    // Opening for writing now is the authoritative permission check — a read-only target
    // fails here rather than being silently rewritten — and this handle is where the new
    // content lands, so the inode and any hardlinks to it survive.
    let mut target = OpenOptions::new().write(true).open(&resolved)?;
    let directory = resolved.parent().unwrap_or(Path::new("."));
    let (mut temporary, temporary_path) = create_temporary(directory)?;

    let written = (|| {
        let mut writer = BufWriter::new(&mut temporary);
        let value = write(&mut writer)?;
        writer.flush()?;
        drop(writer);
        temporary.sync_all()?;
        temporary.seek(io::SeekFrom::Start(0))?;
        Ok(value)
    })();

    let value = match written {
        Ok(value) => value,
        Err(error) => {
            let _ = std::fs::remove_file(&temporary_path);
            return Err(error);
        }
    };

    // The one window where the target is neither the old content nor the new. The complete
    // result is already durable in the temporary, so a failure here keeps it and names it
    // rather than deleting the only copy.
    let copied = (|| {
        target.set_len(0)?;
        io::copy(&mut temporary, &mut target)?;
        target.sync_all()
    })();

    match copied {
        Ok(()) => {
            let _ = std::fs::remove_file(&temporary_path);
            Ok(value)
        }
        Err(error) => Err(io::Error::new(
            error.kind(),
            format!(
                "{error}; the rewritten content is in {}",
                temporary_path.display()
            ),
        )),
    }
}

/// Create a fresh temporary in `directory`, readable and writable by its owner alone.
/// `create_new` refuses both an existing path and a symlink, so a planted link cannot
/// redirect the write; the nonce keeps concurrent runs in one directory apart.
fn create_temporary(directory: &Path) -> io::Result<(File, std::path::PathBuf)> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);

    for attempt in 0..u16::MAX {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.subsec_nanos());
        let candidate = directory.join(format!(
            ".sz.{}.{nonce:08x}.{attempt:04x}.tmp",
            process::id()
        ));
        match options.open(&candidate) {
            Ok(file) => return Ok((file, candidate)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a temporary file",
    ))
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

/// A byte range, in the coordinates of the buffer it was found in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub struct Span {
    pub offset: usize,
    pub length: usize,
}

/// Iterator over lines *with* their terminators, as offsets into the input.
///
/// [`LineIter`] drops terminators, so a caller that copies its output cannot reproduce the
/// input: CRLF, NEL, LS and PS all become whatever the caller writes back. Each span here
/// runs from a line's first byte to the first byte of the next line, so concatenating every
/// span reproduces the input exactly.
#[allow(dead_code)]
pub struct LineSpans<'a> {
    data: &'a [u8],
    lines: LineIter<'a>,
    pending: Option<&'a [u8]>,
}

#[allow(dead_code)]
impl<'a> LineSpans<'a> {
    pub fn new(data: &'a [u8], newlines: Newlines) -> Self {
        let mut lines = LineIter::new(data, newlines);
        let pending = lines.next();
        Self {
            data,
            lines,
            pending,
        }
    }
}

impl Iterator for LineSpans<'_> {
    type Item = Span;

    #[inline]
    fn next(&mut self) -> Option<Span> {
        let line = self.pending.take()?;
        let offset = offset_within(self.data, line);
        self.pending = self.lines.next();
        // The terminator is whatever separates this line from the next, so the span ends
        // where the next line begins — or at the input's end for the final line.
        let end = match self.pending {
            Some(next) => offset_within(self.data, next),
            None => self.data.len(),
        };
        Some(Span {
            offset,
            length: end - offset,
        })
    }
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

// region: Streaming Windows

/// Starting [`Refill`] capacity: large enough to amortize the read syscall, small enough
/// that the filled window stays in L2 between the read and the scan.
#[allow(dead_code)]
pub const DEFAULT_WINDOW_BYTES: usize = 256 << 10;

/// A caller-driven byte window over a reader: one allocation per run, reused for the
/// whole stream, with the caller choosing how much carries by choosing `consumed`.
///
/// This exists rather than a [`std::io::BufRead`] because `fill_buf` carries __no fill
/// guarantee__ — it may hand back four bytes and offers no way to demand more.
/// [`Refill::advance`] fills to capacity or EOF, so a caller can demand a whole record
/// and get one.
#[allow(dead_code)]
pub struct Refill<R> {
    reader: R,
    buffer: Box<[u8]>,
    valid: usize,
    reached_eof: bool,
}

#[allow(dead_code)]
impl<R: Read> Refill<R> {
    /// Window `reader` through `capacity` bytes, rounded up to one byte. `reader` should be
    /// the raw stream: a `BufReader` under it would stage every byte a second time.
    pub fn new(reader: R, capacity: usize) -> Self {
        Refill {
            reader,
            buffer: vec![0u8; capacity.max(1)].into_boxed_slice(),
            valid: 0,
            reached_eof: false,
        }
    }

    /// The bytes currently in the window.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        &self.buffer[..self.valid]
    }

    /// Whether the reader has reported end of input.
    #[inline]
    pub fn at_eof(&self) -> bool {
        self.reached_eof
    }

    /// The window size, which [`Refill::grow`] is the only way to change.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Drop the first `consumed` bytes, slide the rest down, and read until the window is
    /// full or the reader is exhausted. `Ok(false)` once EOF is reached and nothing is left.
    ///
    /// A caller that retains nothing passes `filled().len()`, which degenerates to a
    /// zero-length slide and costs no memmove at all.
    pub fn advance(&mut self, consumed: usize) -> io::Result<bool> {
        debug_assert!(
            consumed <= self.valid,
            "cannot consume past the filled window"
        );
        let consumed = consumed.min(self.valid);
        self.buffer.copy_within(consumed..self.valid, 0);
        self.valid -= consumed;
        self.fill()?;
        Ok(self.valid > 0)
    }

    /// Double the window and read into the new room, so peak memory tracks the widest record
    /// rather than the input. A record longer than the window makes [`last_cut`] return
    /// `None`, and cutting mid-record would split a match or report it twice.
    ///
    /// Only a reader with bytes left can answer: at end of input the window would double
    /// forever without a byte to show for it, which is why every caller tests [`Refill::at_eof`]
    /// before the cut rather than after it.
    pub fn grow(&mut self) -> io::Result<()> {
        debug_assert!(!self.reached_eof, "growing past end of input reads nothing");
        let capacity = self.buffer.len().saturating_mul(2);
        let mut grown = vec![0u8; capacity].into_boxed_slice();
        grown[..self.valid].copy_from_slice(&self.buffer[..self.valid]);
        self.buffer = grown;
        self.fill()
    }

    /// Hand `body` one window after another, each ending where `cut` says a record does, until
    /// the reader runs dry or `body` answers [`ControlFlow::Break`]. A record wider than the
    /// window widens it, so `body` always sees whole records.
    ///
    /// End of input is tested __before__ the cut, and that order is load-bearing: the last
    /// window has nothing following it, so every byte in it is a whole record, and a cut that
    /// reported a shorter prefix — [`CutAfter::Characters`] over a truncated sequence — would
    /// leave a remainder no [`Refill::grow`] can complete.
    pub fn try_for_each_window(
        &mut self,
        cut: CutAfter,
        mut body: impl FnMut(&[u8]) -> io::Result<ControlFlow<()>>,
    ) -> io::Result<()> {
        let mut consumed = 0;
        while self.advance(consumed)? {
            let end = if self.at_eof() {
                self.filled().len()
            } else {
                match last_cut(self.filled(), cut) {
                    Some(end) => end,
                    // A record wider than the window: widen it, retaining everything.
                    None => {
                        self.grow()?;
                        consumed = 0;
                        continue;
                    }
                }
            };
            if body(&self.filled()[..end])?.is_break() {
                return Ok(());
            }
            consumed = end;
        }
        Ok(())
    }

    /// Hand `body` every window, for a caller that reads its input to the end.
    pub fn for_each_window(
        &mut self,
        cut: CutAfter,
        mut body: impl FnMut(&[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        self.try_for_each_window(cut, |window| {
            body(window).map(|()| ControlFlow::Continue(()))
        })
    }

    /// Read until the window is full or the reader is exhausted, retrying interruptions.
    fn fill(&mut self) -> io::Result<()> {
        while !self.reached_eof && self.valid < self.buffer.len() {
            match self.reader.read(&mut self.buffer[self.valid..]) {
                Ok(0) => self.reached_eof = true,
                Ok(read) => self.valid += read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

/// Where a window may be cut for a given segmenter. A cut is safe exactly where that
/// segmenter's automaton returns to its initial state, which is after the characters
/// the mode mandates an unconditional break for. The variants run weakest first: a
/// consumer that reads bytes one at a time cuts anywhere, and each one below reads
/// further ahead than the last.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum CutAfter {
    /// Every byte, so the window is consumed whole and nothing carries.
    Anywhere,
    /// Between characters, so a multi-byte UTF-8 sequence is never split.
    Characters,
    /// LF (`\n`) alone, which is where a byte-level line splitter breaks.
    LineFeed,
    /// Every Unicode mandatory line terminator: LF, CRLF, VT, FF, NEL, LS, PS.
    LineTerminators,
    /// Paragraph separators only. VT and FF carry `Sentence_Break=Sp` rather than
    /// `Sep`, so a sentence continues across them.
    ParagraphSeparators,
}

impl From<Newlines> for CutAfter {
    /// A line splitter cuts wherever it breaks, so its newline set is its cut set.
    #[inline]
    fn from(newlines: Newlines) -> Self {
        match newlines {
            Newlines::Lf => CutAfter::LineFeed,
            Newlines::Unicode => CutAfter::LineTerminators,
        }
    }
}

/// Byte offset one past the last complete cut point in `data` — the point where a window
/// cuts without splitting a record. `None` when `data` holds no complete record.
///
/// A trailing bare CR is never reported: its LF may arrive in the next window, and cutting
/// between them would yield one extra record.
#[allow(dead_code)]
pub fn last_cut(data: &[u8], cut: CutAfter) -> Option<usize> {
    match cut {
        CutAfter::Anywhere => (!data.is_empty()).then_some(data.len()),
        CutAfter::Characters => {
            let end = whole_character_prefix(data);
            (end > 0).then_some(end)
        }
        CutAfter::LineFeed => sz::rfind(data, b"\n").map(|position| position + 1),
        CutAfter::LineTerminators => last_unicode_cut(data, line_terminator_tail_byteset()),
        CutAfter::ParagraphSeparators => last_unicode_cut(data, paragraph_separator_tail_byteset()),
    }
}

/// Length of the window prefix that ends between characters. The window's last character
/// may be incomplete, so trimming to one byte short of the end is what hands it to the
/// next window whole — the same bounded backward walk [`truncate_at_character`] takes.
#[inline]
fn whole_character_prefix(data: &[u8]) -> usize {
    match data.len().checked_sub(1) {
        Some(before_last) => truncate_at_character(data, before_last).0.len(),
        None => 0,
    }
}

/// The bytes that can close a Unicode line terminator: LF, VT, FF and CR stand alone,
/// `85` closes NEL (`C2 85`), and `A8`/`A9` close LS/PS (`E2 80 A8`/`E2 80 A9`).
fn line_terminator_tail_byteset() -> sz::Byteset {
    static TAILS: OnceLock<sz::Byteset> = OnceLock::new();
    *TAILS.get_or_init(|| sz::Byteset::from(b"\n\x0B\x0C\r\x85\xA8\xA9".as_slice()))
}

/// The same tails without VT and FF, which end a line but not a sentence.
fn paragraph_separator_tail_byteset() -> sz::Byteset {
    static TAILS: OnceLock<sz::Byteset> = OnceLock::new();
    *TAILS.get_or_init(|| sz::Byteset::from(b"\n\r\x85\xA8\xA9".as_slice()))
}

/// Reverse-scan for a terminator whose whole encoding lies inside `data`, rejecting a tail
/// byte that only looks like one — `85`, `A8` and `A9` also close unrelated codepoints.
/// `tails` selects the cut set, so the VT and FF arm fires only for the line terminators.
fn last_unicode_cut(data: &[u8], tails: sz::Byteset) -> Option<usize> {
    let mut searched = data.len();
    while let Some(position) = sz::rfind_byteset(&data[..searched], tails) {
        let complete = match data[position] {
            // CR terminates on its own, unless the LF that would join it is still coming.
            b'\r' => position + 1 < data.len(),
            b'\n' | 0x0B | 0x0C => true,
            0x85 => position >= 1 && data[position - 1] == 0xC2,
            _ => position >= 2 && data[position - 2] == 0xE2 && data[position - 1] == 0x80,
        };
        if complete {
            return Some(position + 1);
        }
        searched = position;
    }
    None
}

// endregion: Streaming Windows

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

/// Write the `{"text":"…"}` wrapper that every path and line field uses, falling back to
/// `{"bytes":"<base64>"}` when the slice is not valid UTF-8.
///
/// Passing invalid bytes through raw would emit JSON no decoder accepts, and text tools do
/// meet non-UTF-8 input. The two-arm shape is ripgrep's, which this envelope already follows.
#[allow(dead_code)]
pub fn json_text_field_to(output: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    if std::str::from_utf8(data).is_err() {
        output.write_all(br#"{"bytes":""#)?;
        base64_to(output, data)?;
        return output.write_all(br#""}"#);
    }
    output.write_all(br#"{"text":""#)?;
    json_escape_to(output, data)?;
    output.write_all(br#""}"#)
}

/// Standard base64 with padding, written without allocating.
fn base64_to(output: &mut dyn Write, data: &[u8]) -> io::Result<()> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = [0u8; 4];
    for group in data.chunks(3) {
        let bits = group.iter().enumerate().fold(0u32, |bits, (index, byte)| {
            bits | (u32::from(*byte) << (16 - 8 * index))
        });
        for (index, slot) in encoded.iter_mut().enumerate() {
            *slot = if index <= group.len() {
                ALPHABET[(bits >> (18 - 6 * index)) as usize & 0x3F]
            } else {
                b'='
            };
        }
        output.write_all(&encoded)?;
    }
    Ok(())
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

// endregion: Machine-Readable Output

// region: Argument Parsing

/// Parse a count that has to be at least 1. Clap's stock [`NonZeroUsize`] parser answers
/// "number would be zero for non-zero type", naming a Rust type where the bound belongs.
/// Every other input keeps the stock wording, so only the zero case reads differently.
#[allow(dead_code)]
pub fn parse_at_least_one(value: &str) -> Result<NonZeroUsize, String> {
    let count: usize = value
        .parse()
        .map_err(|error: ParseIntError| error.to_string())?;
    NonZeroUsize::new(count).ok_or_else(|| "must be at least 1".to_string())
}

/// Parse a byte budget: bare digits count bytes, a bare letter or a `B` suffix is decimal
/// (`K` is 1000), and an `i` is binary (`Ki` is 1024) — the distinction `ls -h` and `df -h`
/// draw. Matching is case-insensitive, so `10mb`, `10MB` and `10Mb` agree.
///
/// Zero shares [`parse_at_least_one`]'s wording, since a budget of nothing is unsatisfiable
/// for the same reason a count of nothing is. Overflow is reported rather than saturated:
/// silently clamping `99E` to one chunk would look like it worked.
#[allow(dead_code)]
pub fn parse_size(value: &str) -> Result<NonZeroUsize, String> {
    let trimmed = value.trim();
    let digits_len = trimmed
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, unit) = trimmed.split_at(digits_len);

    let unit = unit.trim();
    let scale: usize = match unit.to_ascii_uppercase().as_str() {
        "" | "B" => 1,
        "K" | "KB" => 1_000,
        "KI" | "KIB" => 1 << 10,
        "M" | "MB" => 1_000_000,
        "MI" | "MIB" => 1 << 20,
        "G" | "GB" => 1_000_000_000,
        "GI" | "GIB" => 1 << 30,
        "T" | "TB" => 1_000_000_000_000,
        "TI" | "TIB" => 1 << 40,
        // Quote what was typed rather than the folded form, so the message points at the
        // input the reader can see.
        _ => {
            return Err(format!(
                "`{}` is not a size suffix; use K, M, G, T for powers of 1000 or Ki, Mi, Gi, Ti for powers of 1024",
                unit
            ))
        }
    };

    // `digits` holds only ASCII digits by construction, so a parse failure here means the
    // count outran `usize` rather than that it was malformed — the same answer the
    // multiply below gives, and one a reader can act on.
    let too_large = || format!("`{}` is larger than this platform can address", trimmed);
    let count: usize = match digits {
        "" => return Err(format!("`{}` is not a number", trimmed)),
        digits => digits.parse().map_err(|_| too_large())?,
    };
    let bytes = count.checked_mul(scale).ok_or_else(too_large)?;
    NonZeroUsize::new(bytes).ok_or_else(|| "must be at least 1".to_string())
}

// endregion: Argument Parsing

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
    fn encodes_invalid_utf8_as_base64_bytes() {
        let mut output = Vec::new();
        json_text_field_to(&mut output, b"a\xffb").unwrap();
        assert_eq!(output, br#"{"bytes":"Yf9i"}"#);

        // Padding: one and two leftover bytes.
        for (data, expected) in [
            (b"\xff".as_slice(), br#"{"bytes":"/w=="}"#.as_slice()),
            (b"\xff\xfe".as_slice(), br#"{"bytes":"//4="}"#.as_slice()),
        ] {
            let mut output = Vec::new();
            json_text_field_to(&mut output, data).unwrap();
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn wraps_json_text_field() {
        let mut output = Vec::new();
        json_text_field_to(&mut output, b"a\"b").unwrap();
        assert_eq!(output, br#"{"text":"a\"b"}"#);
    }

    fn grouped(value: usize) -> String {
        let mut buffer = [0u8; 26];
        format_grouped_number(&mut buffer, value).to_string()
    }

    #[test]
    fn line_spans_reproduce_the_input() {
        // Every terminator width is different, which is the whole reason spans exist:
        // LF is one byte, CRLF two, LS three.
        let data = "a\r\nb\u{2028}c\nd".as_bytes();
        let spans: Vec<Span> = LineSpans::new(data, Newlines::Unicode).collect();

        let joined: Vec<u8> = spans
            .iter()
            .flat_map(|span| &data[span.offset..span.offset + span.length])
            .copied()
            .collect();
        assert_eq!(joined, data);
        assert_eq!(
            spans.iter().map(|span| span.length).collect::<Vec<_>>(),
            [3, 4, 2, 1]
        );
    }

    #[test]
    fn reports_offsets_of_borrowed_segments() {
        // The offset must come from the slice, not from accumulated line lengths:
        // CRLF and the Unicode separators are not one byte wide.
        let data = "a\r\nb\u{2028}hello\n".as_bytes();
        let lines: Vec<&[u8]> = LineIter::new(data, Newlines::Unicode).collect();
        let offsets: Vec<usize> = lines.iter().map(|line| offset_within(data, line)).collect();
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
    fn offers_a_whole_slice_for_every_source_but_a_pipe() {
        // `into_window` is total: a buffer answers with its bytes, a pipe with a window.
        let buffered = InputSource::Buffer(b"abc".to_vec());
        assert_eq!(buffered.as_bytes(), b"abc");
        let window = buffered.into_window(DEFAULT_WINDOW_BYTES);
        assert!(matches!(window, InputWindow::Whole(source) if source.as_bytes() == b"abc"));
        let piped = InputSource::Pipe(io::stdin().lock());
        assert!(matches!(piped.into_window(1), InputWindow::Stream(_)));
    }

    /// Drive a window the way a streaming binary does, collecting the lines it yields.
    /// `grow` covers a record wider than the window; EOF covers a record without a terminator.
    fn streamed_lines(data: &[u8], capacity: usize, newlines: Newlines) -> Vec<Vec<u8>> {
        let mut refill = Refill::new(data, capacity);
        let mut collected = Vec::new();
        refill
            .for_each_window(newlines.into(), |window| {
                collected.extend(LineIter::new(window, newlines).map(<[u8]>::to_vec));
                Ok(())
            })
            .unwrap();
        collected
    }

    /// Collect the windows the driver hands out, capping the run so a driver that fails to
    /// make progress fails the test rather than spinning.
    fn driven_windows(data: &[u8], capacity: usize, cut: CutAfter) -> Vec<Vec<u8>> {
        let mut refill = Refill::new(data, capacity);
        let mut windows: Vec<Vec<u8>> = Vec::new();
        refill
            .for_each_window(cut, |window| {
                assert!(windows.len() < 64, "the driver stopped making progress");
                windows.push(window.to_vec());
                Ok(())
            })
            .unwrap();
        windows
    }

    fn whole_lines(data: &[u8], newlines: Newlines) -> Vec<Vec<u8>> {
        LineIter::new(data, newlines).map(<[u8]>::to_vec).collect()
    }

    #[test]
    fn reports_the_cut_past_the_last_terminator() {
        assert_eq!(last_cut(b"a\nb\n", CutAfter::LineFeed), Some(4));
        assert_eq!(last_cut(b"a\nb", CutAfter::LineFeed), Some(2));
        assert_eq!(last_cut(b"abc", CutAfter::LineFeed), None);
        assert_eq!(last_cut(b"", CutAfter::LineFeed), None);
        // The cut lands past a whole CRLF, which the Unicode set treats as one break.
        assert_eq!(last_cut(b"a\r\nb", CutAfter::LineTerminators), Some(3));
        assert_eq!(last_cut(b"a\rb", CutAfter::LineTerminators), Some(2));
        assert_eq!(
            last_cut("a\u{2028}b".as_bytes(), CutAfter::LineTerminators),
            Some(4)
        );
        assert_eq!(
            last_cut("a\u{0085}b".as_bytes(), CutAfter::LineTerminators),
            Some(3)
        );
    }

    #[test]
    fn reads_byte_budgets_with_decimal_and_binary_suffixes() {
        let size = |text: &str| parse_size(text).map(NonZeroUsize::get);
        assert_eq!(size("512"), Ok(512));
        assert_eq!(size("10K"), Ok(10_000));
        assert_eq!(size("10KB"), Ok(10_000));
        assert_eq!(size("10Ki"), Ok(10_240));
        assert_eq!(size("10KiB"), Ok(10_240));
        // The suffix is case-insensitive, and surrounding space is ignored.
        assert_eq!(size("10mb"), size("10MB"));
        assert_eq!(size(" 10M "), size("10M"));
        // A budget of nothing reads like a count of nothing.
        assert_eq!(size("0"), Err("must be at least 1".to_string()));
        // Unknown suffixes and overflow name the offending text rather than a Rust type.
        assert!(size("10Q").unwrap_err().contains("`Q`"));
        assert!(size("10 furlongs").unwrap_err().contains("furlongs"));
        // A count too wide for `usize` is a size problem, not a syntax one, whether it
        // outruns the type on its own or only once the suffix scales it.
        assert!(size("99999999999999999999T")
            .unwrap_err()
            .contains("larger than"));
        assert!(size("99999999999T").unwrap_err().contains("larger than"));
        // A suffix with no count in front of it is the syntax error.
        assert!(size("MB").unwrap_err().contains("is not a number"));
    }

    #[test]
    fn refuses_to_cut_inside_a_terminator() {
        // A trailing CR may be the head of a CRLF whose LF is in the next window.
        assert_eq!(last_cut(b"a\nbc\r", CutAfter::LineTerminators), Some(2));
        assert_eq!(last_cut(b"abc\r", CutAfter::LineTerminators), None);
        // A truncated LINE SEPARATOR is not a terminator; the earlier one is.
        let truncated = "a\u{2028}b\u{2028}".as_bytes();
        assert_eq!(
            last_cut(&truncated[..7], CutAfter::LineTerminators),
            Some(4)
        );
        // `2005` ends in `85` and `00A8` ends in `A8`, yet neither is a newline.
        assert_eq!(
            last_cut("a\u{2005}b\u{00A8}c".as_bytes(), CutAfter::LineTerminators),
            None
        );
    }

    #[test]
    fn maps_every_newline_set_onto_its_cut_set() {
        assert_eq!(CutAfter::from(Newlines::Lf), CutAfter::LineFeed);
        assert_eq!(CutAfter::from(Newlines::Unicode), CutAfter::LineTerminators);
    }

    #[test]
    fn skips_vertical_tab_and_form_feed_for_paragraph_separators() {
        // VT and FF are `Sentence_Break=Sp`, so a sentence runs through them and the
        // paragraph set walks back to the previous separator instead.
        assert_eq!(last_cut(b"a\nb\x0Bc", CutAfter::LineTerminators), Some(4));
        assert_eq!(
            last_cut(b"a\nb\x0Bc", CutAfter::ParagraphSeparators),
            Some(2)
        );
        assert_eq!(last_cut(b"a\nb\x0Cc", CutAfter::LineTerminators), Some(4));
        assert_eq!(
            last_cut(b"a\nb\x0Cc", CutAfter::ParagraphSeparators),
            Some(2)
        );
        assert_eq!(last_cut(b"a\x0Bb", CutAfter::ParagraphSeparators), None);
        // LF, CRLF, NEL, LS and PS stay cut points for both sets.
        assert_eq!(last_cut(b"a\r\nb", CutAfter::ParagraphSeparators), Some(3));
        assert_eq!(
            last_cut("a\u{0085}b".as_bytes(), CutAfter::ParagraphSeparators),
            Some(3)
        );
        assert_eq!(
            last_cut("a\u{2028}b".as_bytes(), CutAfter::ParagraphSeparators),
            Some(4)
        );
        assert_eq!(
            last_cut("a\u{2029}b".as_bytes(), CutAfter::ParagraphSeparators),
            Some(4)
        );
        // The bare-CR deferral and the look-alike rejection hold for both sets.
        assert_eq!(last_cut(b"abc\r", CutAfter::ParagraphSeparators), None);
        assert_eq!(
            last_cut(
                "a\u{2005}b\u{00A8}c".as_bytes(),
                CutAfter::ParagraphSeparators
            ),
            None
        );
    }

    #[test]
    fn leaves_the_window_untouched_on_a_zero_consume() {
        let mut refill = Refill::new(&b"abcdefgh"[..], 4);
        assert!(refill.advance(0).unwrap());
        assert_eq!(refill.filled(), b"abcd");
        // Retaining everything re-reads nothing and slides nothing.
        assert!(refill.advance(0).unwrap());
        assert_eq!(refill.filled(), b"abcd");
    }

    #[test]
    fn tiles_the_input_when_nothing_is_retained() {
        // `consumed == filled().len()` degenerates to a zero-length slide, so the windows
        // are exactly the successive capacity-sized chunks of the input.
        let mut refill = Refill::new(&b"abcdefghij"[..], 4);
        let mut windows = Vec::new();
        let mut consumed = 0;
        while refill.advance(consumed).unwrap() {
            windows.push(refill.filled().to_vec());
            consumed = refill.filled().len();
        }
        assert_eq!(
            windows,
            vec![b"abcd".to_vec(), b"efgh".to_vec(), b"ij".to_vec()]
        );
        assert!(refill.at_eof());
    }

    #[test]
    fn grows_the_window_for_a_record_wider_than_it() {
        let data = b"short\nthis single record is far wider than the starting window\ntail\n";
        let mut refill = Refill::new(&data[..], 8);
        assert!(refill.advance(0).unwrap());
        // "short\n" fits, so the first cut needs no growth.
        assert_eq!(last_cut(refill.filled(), CutAfter::LineFeed), Some(6));
        assert!(refill.advance(6).unwrap());
        assert_eq!(last_cut(refill.filled(), CutAfter::LineFeed), None);
        refill.grow().unwrap();
        assert_eq!(refill.capacity(), 16);
        assert_eq!(
            streamed_lines(data, 8, Newlines::Lf),
            whole_lines(data, Newlines::Lf)
        );
    }

    #[test]
    fn grows_the_window_until_the_record_fits() {
        // Growth carries on for as long as the record does, so a megabyte line out of a
        // four-byte window arrives whole rather than stopping at some width.
        let mut data = vec![b'x'; 1 << 20];
        data.push(b'\n');
        let mut refill = Refill::new(&data[..], 4);
        let cut = CutAfter::LineFeed;
        while refill.advance(0).unwrap() && last_cut(refill.filled(), cut).is_none() {
            refill.grow().unwrap();
        }
        assert!(refill.capacity() >= data.len());
        assert_eq!(
            streamed_lines(&data, 4, Newlines::Lf),
            whole_lines(&data, Newlines::Lf)
        );
    }

    #[test]
    fn streams_a_partial_trailing_record_at_eof() {
        let data = b"alpha\nbeta\ngamma";
        assert_eq!(
            streamed_lines(data, 4, Newlines::Lf),
            vec![b"alpha".to_vec(), b"beta".to_vec(), b"gamma".to_vec()]
        );
        assert_eq!(streamed_lines(b"", 4, Newlines::Lf), Vec::<Vec<u8>>::new());
        assert_eq!(streamed_lines(b"\n", 4, Newlines::Lf), vec![b"".to_vec()]);
    }

    #[test]
    fn streams_the_same_lines_at_tiny_capacities() {
        let line_feed = b"alpha\nbeta\n\ngamma delta\nepsilon";
        let unicode = "alpha\r\nbeta\u{2028}\u{2029}gamma\u{0085}delta\u{000B}omega".as_bytes();
        for capacity in [7, 13] {
            assert_eq!(
                streamed_lines(line_feed, capacity, Newlines::Lf),
                whole_lines(line_feed, Newlines::Lf)
            );
            assert_eq!(
                streamed_lines(unicode, capacity, Newlines::Unicode),
                whole_lines(unicode, Newlines::Unicode)
            );
        }
    }

    #[test]
    fn keeps_a_crlf_whole_across_a_seam() {
        // A window of 5 ends on the CR of "bc\r\n". Cutting there would leave a bare LF to
        // open the next window, yielding one extra line.
        let data = b"a\nbc\r\nd\n";
        assert_eq!(
            streamed_lines(data, 5, Newlines::Unicode),
            vec![b"a".to_vec(), b"bc".to_vec(), b"d".to_vec()]
        );
    }

    #[test]
    fn cuts_anywhere_when_nothing_carries() {
        assert_eq!(last_cut(b"abc", CutAfter::Anywhere), Some(3));
        assert_eq!(last_cut(b"", CutAfter::Anywhere), None);
        // Nothing is retained, so the windows are the successive capacity-sized chunks.
        assert_eq!(
            driven_windows(b"abcdefghij", 4, CutAfter::Anywhere),
            vec![b"abcd".to_vec(), b"efgh".to_vec(), b"ij".to_vec()]
        );
    }

    #[test]
    fn cuts_between_characters_rather_than_inside_one() {
        // "é" is two bytes, so a cut at byte 2 would hand its lead byte to one window and
        // its continuation byte to the next.
        let two_byte = "aé".as_bytes();
        assert_eq!(last_cut(two_byte, CutAfter::Characters), Some(1));
        assert_eq!(last_cut(&two_byte[..2], CutAfter::Characters), Some(1));
        assert_eq!(last_cut(b"ab", CutAfter::Characters), Some(1));
        assert_eq!(last_cut(b"a", CutAfter::Characters), None);
        assert_eq!(last_cut(b"", CutAfter::Characters), None);

        // Two-, three- and four-byte characters, at every capacity a seam can land in:
        // the windows tile the input and each one decodes on its own.
        let data = "aé\u{4E2D}\u{1F600}b".as_bytes();
        for capacity in 1..=data.len() + 2 {
            let windows = driven_windows(data, capacity, CutAfter::Characters);
            assert_eq!(windows.concat(), data, "capacity {}", capacity);
            for window in &windows {
                assert!(
                    std::str::from_utf8(window).is_ok(),
                    "capacity {} split a character",
                    capacity
                );
            }
        }
    }

    #[test]
    fn takes_the_last_window_whole_however_it_ends() {
        // End of input is tested before the cut. Cutting first hands back "caf" and leaves a
        // lone lead byte that no `grow` can complete, because the reader is already dry.
        let truncated = &b"caf\xC3"[..];
        assert_eq!(last_cut(truncated, CutAfter::Characters), Some(3));
        assert_eq!(
            driven_windows(truncated, 64, CutAfter::Characters),
            vec![truncated.to_vec()]
        );
        // The same holds for a final line that never got its terminator.
        assert_eq!(
            driven_windows(b"a\nbc", 64, CutAfter::LineFeed),
            vec![b"a\nbc".to_vec()]
        );
    }

    #[test]
    fn stops_the_run_where_the_body_breaks() {
        // A bounded request stops reading rather than draining the rest of the stream.
        let mut refill = Refill::new(&b"a\nb\nc\nd\n"[..], 4);
        let mut windows = Vec::new();
        refill
            .try_for_each_window(CutAfter::LineFeed, |window| {
                windows.push(window.to_vec());
                Ok(ControlFlow::Break(()))
            })
            .unwrap();
        assert_eq!(windows, vec![b"a\nb\n".to_vec()]);
    }

    #[test]
    fn agrees_with_the_whole_buffer_at_every_capacity() {
        // Every capacity puts the seam at a different byte, so one sweep covers a cut inside
        // a CRLF, inside each multi-byte terminator, and inside a look-alike codepoint.
        let inputs: [&[u8]; 6] = [
            b"a\nbc\r\nd\n",
            b"\r\n\r\na\r\n",
            "\u{2028}a\u{2029}b\u{0085}\u{000B}c".as_bytes(),
            "a\u{2005}b\u{00A8}c\u{2028}d".as_bytes(),
            b"no terminator anywhere in this record",
            b"",
        ];
        for input in inputs {
            for capacity in 1..=input.len() + 2 {
                for newlines in [Newlines::Lf, Newlines::Unicode] {
                    assert_eq!(
                        streamed_lines(input, capacity, newlines),
                        whole_lines(input, newlines),
                        "capacity {}",
                        capacity
                    );
                }
            }
        }
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
