//! Primitives shared by every `sz-*` binary: input sources and windows, line and segment
//! iteration, output terminators, argument parsers, and the error type each tool forwards to
//! `main`.
//!
//! The recurring shape is that a tool should not care whether its input is a mapped file, a
//! buffered stdin, or a pipe it may only read once. [`InputSource`] hides that, and the one place
//! it leaks — a true pipe has no whole slice — is visible in the type rather than in a comment.

use std::borrow::Cow;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, Write};
use std::num::{NonZeroUsize, ParseIntError};
use std::path::Path;
use std::process;
use std::sync::OnceLock;

// The reference tables live beside their own parsers under `data/`, and reach every binary through
// this crate rather than being `mod`-included once per binary.
#[cfg(feature = "fuzzy-find")]
#[path = "data/keyboards.rs"]
pub mod keyboards;

#[cfg(feature = "fuzzy-find")]
#[path = "data/folds.rs"]
pub mod folds;

#[cfg(feature = "fuzzy-find")]
#[path = "data/misspellings.rs"]
pub mod misspellings;

use memmap2::Mmap;
use stringzilla::sz;
use stringzilla::sz::{FindSplits, StringZillableBinary, StringZillableUnary, Utf8SplitNewlines};

// region: Input Sources

/// Where a tool's bytes come from: a mapped file, a drained stdin, or a pipe read once.
pub enum InputSource {
    /// Memory-mapped file for zero-copy access (read-only)
    MappedFile(Mmap),
    /// Buffered stdin data
    Buffer(Vec<u8>),
    /// An undrained stream, the one source with no whole slice. Produced only by
    /// [`get_input_streaming`], so [`get_input`]'s callers never observe it.
    Pipe(Box<dyn Read>),
}

/// The two shapes an input takes: one whole slice, or a window to refill.
pub enum InputWindow {
    /// A source that hands over all of its bytes at once.
    Whole(InputSource),
    /// A stream, read through one window reused for the whole run.
    Stream(Refill<Box<dyn Read>>),
}

impl InputSource {
    /// The whole input as one slice. A [`InputSource::Pipe`] has no whole slice and returns
    /// empty after tripping a debug assertion; use [`InputSource::into_window`] for that case.
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
pub fn get_input_streaming(path: Option<&str>) -> io::Result<InputSource> {
    let Some(path) = path.filter(|path| *path != "-") else {
        // A redirect resolves to a regular file; only true pipes need streaming.
        if let Some(mmap) = map_stdin() {
            return Ok(InputSource::MappedFile(mmap));
        }
        widen_stdin_pipe();
        return Ok(InputSource::Pipe(Box::new(io::stdin().lock())));
    };
    open_input_streaming(Path::new(path))
}

/// Map `path` as [`open_input`] does, except that a source with no mapping streams rather
/// than buffering, which is what keeps a named FIFO from being read whole into memory.
pub fn open_input_streaming(path: &Path) -> io::Result<InputSource> {
    let file = File::open(path)?;
    if !file.metadata().is_ok_and(|metadata| metadata.is_file()) {
        return Ok(InputSource::Pipe(Box::new(file)));
    }
    open_input(path)
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
fn is_readable_entry(entry: &ignore::DirEntry) -> bool {
    match entry.file_type() {
        Some(kind) if kind.is_file() => true,
        Some(kind) => entry.depth() == 0 && !kind.is_dir(),
        None => false,
    }
}

/// Rewrite `path` through a sibling temporary file that replaces it only once the new
/// content has reached the disk, so a crash leaves either the old file or the new one.
///
/// The swap is a rename, which is atomic but gives the path a new inode. That severs any
/// hardlink, so a file carrying more than one link is instead copied back over the original
/// inode — preserving the link at the cost of a window where the file is neither version.
/// Symlinks survive either way, since `path` is resolved before anything is written.
///
/// The path is attached here rather than by the caller, since every failure below is about
/// this one file and the function already knows which.
fn write_replacing<T>(
    tool: &str,
    path: &str,
    write: impl FnOnce(&mut dyn Write) -> io::Result<T>,
) -> Result<T, Failure> {
    (|| {
        // Resolve first: editing through a symlink must change the file it names, not replace
        // the link with a regular file. The input has to exist, so there is no create case.
        let resolved = std::fs::canonicalize(path)?;
        let (file, metadata) = open_target(&resolved)?;
        let target = if metadata.is_file() {
            Target::Replaceable {
                resolved,
                file,
                metadata,
            }
        } else {
            Target::Direct(file)
        };
        write_through_temporary(tool, path, target, write)
    })()
    .at(path)
}

/// Put the result at `path`, creating it if nothing is there yet and swapping it in only
/// once the bytes are on disk — so an interrupted run leaves the previous file, or no file,
/// rather than a truncated one. The temporary is a sibling of the target, so the rename
/// stays inside one filesystem.
///
/// Unlike [`write_replacing`] this is the destination rather than the source, so a target
/// that is not a regular file — `/dev/null`, a FIFO, a terminal — is written straight
/// through. Renaming a regular file over a device node would replace it.
fn write_creating<T>(
    tool: &str,
    path: &str,
    write: impl FnOnce(&mut dyn Write) -> io::Result<T>,
) -> Result<T, Failure> {
    (|| write_through_temporary(tool, path, resolve_target(path)?, write))().at(path)
}

/// Open an existing target for writing, with its metadata.
///
/// A rename needs only write permission on the directory, so it would happily replace a file
/// the caller cannot write. Opening the target settles that question the way the kernel
/// would, before anything is created.
fn open_target(resolved: &Path) -> io::Result<(File, std::fs::Metadata)> {
    let file = OpenOptions::new().write(true).open(resolved)?;
    let metadata = file.metadata()?;
    Ok((file, metadata))
}

/// What a destination turns out to be, which decides whether the result can be swapped in
/// or has to be written where it stands.
enum Target {
    /// A regular file at a name a rename can replace.
    Replaceable {
        resolved: std::path::PathBuf,
        file: File,
        metadata: std::fs::Metadata,
    },
    /// A stream or device — `/dev/null`, `/dev/stdout`, a FIFO, a terminal. There is nothing
    /// to preserve and no name a rename could put the result at.
    Direct(File),
    /// Nothing is there yet, so the result is created at this name.
    Fresh(std::path::PathBuf),
}

/// Decide what `path` denotes.
///
/// Opening comes before resolving, deliberately. `/dev/stdout` and `/dev/fd/N` are openable
/// but resolve to names that denote something else — on macOS `realpath("/dev/stdout")` under
/// a redirect yields `/dev/fd/<the target's basename>`, which does not exist — so a
/// resolve-first order turns a stream to write into a file to create, next to the device
/// nodes. A rename may only replace a name that denotes the very file that was opened, and
/// anything failing that test is written where it stands.
fn resolve_target(path: &str) -> io::Result<Target> {
    match OpenOptions::new().write(true).open(path) {
        Ok(file) => {
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Ok(Target::Direct(file));
            }
            let Ok(resolved) = std::fs::canonicalize(path) else {
                return Ok(Target::Direct(file));
            };
            if !denotes(&resolved, &metadata) {
                return Ok(Target::Direct(file));
            }
            Ok(Target::Replaceable {
                resolved,
                file,
                metadata,
            })
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // A symlink whose target does not exist yet is still the caller's chosen name
            // for the file, so the result lands where the link points rather than over it.
            let target = link_target(Path::new(path));
            let name = target
                .file_name()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "not a file name"))?;
            let directory = target
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            Ok(Target::Fresh(std::fs::canonicalize(directory)?.join(name)))
        }
        Err(error) => Err(error),
    }
}

/// Whether `resolved` names the same file as the one already opened.
fn denotes(resolved: &Path, opened: &std::fs::Metadata) -> bool {
    let Ok(named) = std::fs::metadata(resolved) else {
        return false;
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        named.dev() == opened.dev() && named.ino() == opened.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = opened;
        named.is_file()
    }
}

/// Follow a chain of symlinks that ends somewhere nothing exists, so a write through a
/// dangling link creates its target rather than replacing the link. Bounded, since a link
/// may point at itself.
fn link_target(path: &Path) -> std::path::PathBuf {
    let mut current = path.to_path_buf();
    for _ in 0..40 {
        let Ok(next) = std::fs::read_link(&current) else {
            return current;
        };
        current = match current.parent() {
            Some(directory) if next.is_relative() => directory.join(next),
            _ => next,
        };
    }
    current
}

fn write_through_temporary<T>(
    tool: &str,
    path: &str,
    target: Target,
    write: impl FnOnce(&mut dyn Write) -> io::Result<T>,
) -> io::Result<T> {
    // Nothing to preserve and no name to rename to, so the bytes go in where they stand,
    // exactly as `File::create` would have put them.
    let (resolved, existing) = match target {
        Target::Direct(file) => {
            let mut writer = BufWriter::new(&file);
            let value = write(&mut writer)?;
            writer.flush()?;
            return Ok(value);
        }
        Target::Replaceable {
            resolved,
            file,
            metadata,
        } => (resolved, Some((file, metadata))),
        Target::Fresh(resolved) => (resolved, None),
    };
    let resolved = resolved.as_path();

    let directory = resolved.parent().unwrap_or(Path::new("."));
    // Always private to start with. The window a temporary spends incomplete is exactly the
    // window its content is half-written, and a half-written file can hold a prefix the
    // finished one never does — a redaction that has not reached its subject yet.
    let (mut temporary, temporary_path) = create_temporary(directory, 0o600)?;

    let written = (|| {
        let mut writer = BufWriter::new(&mut temporary);
        let value = write(&mut writer)?;
        writer.flush()?;
        drop(writer);
        // Widened only now the content is complete: a replacement lands on the mode its
        // file already had, and a fresh file lands where `File::create` would have put it.
        let landing = landing_permissions(existing.as_ref().map(|(_, data)| data), directory);
        if let Some(permissions) = landing {
            temporary.set_permissions(permissions)?;
        }
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

    // Only a file that already exists can carry other names for its inode, so a fresh one
    // always takes the atomic path.
    let hardlinked = existing
        .as_ref()
        .is_some_and(|(_, metadata)| is_hardlinked(metadata));
    if !hardlinked {
        return match std::fs::rename(&temporary_path, resolved) {
            Ok(()) => Ok(value),
            Err(error) => {
                let _ = std::fs::remove_file(&temporary_path);
                Err(error)
            }
        };
    }

    eprintln!(
        "{tool}: warning: {path} has other hardlinks, so it is rewritten in place rather \
         than replaced; an interrupted run can leave it truncated"
    );

    // The one window where the file is neither version. The result is already durable in the
    // temporary, so a failure here keeps it and names it rather than deleting the only copy.
    let (mut target, _) = existing.expect("only an existing file reports hardlinks");
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

/// The mode a finished temporary should land on: the one its target already had, or the one
/// an ordinary create would have produced.
#[cfg(unix)]
fn landing_permissions(
    existing: Option<&std::fs::Metadata>,
    directory: &Path,
) -> Option<std::fs::Permissions> {
    use std::os::unix::fs::PermissionsExt;
    Some(match existing {
        Some(metadata) => metadata.permissions(),
        None => std::fs::Permissions::from_mode(0o666 & !cleared_by_umask(directory)),
    })
}

/// Windows carries no mode; a replacement keeps its target's read-only bit and a fresh file
/// inherits the directory's ACL, which is what `File::create` gives it too.
#[cfg(not(unix))]
fn landing_permissions(
    existing: Option<&std::fs::Metadata>,
    _directory: &Path,
) -> Option<std::fs::Permissions> {
    existing.map(|metadata| metadata.permissions())
}

/// The permission bits this process's umask clears.
///
/// POSIX offers no way to *read* a umask — only to set it and be handed back the old value —
/// and doing that would leave every other thread creating mode-zero files for the duration.
/// Asking the filesystem what an ordinary create produces answers the same question and races
/// with nothing. The probe is empty, so the moment it is visible reveals nothing, and it is
/// taken once for the life of the process.
#[cfg(unix)]
fn cleared_by_umask(directory: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    static CLEARED: OnceLock<u32> = OnceLock::new();
    *CLEARED.get_or_init(|| {
        // A directory that cannot be probed cannot be written either, so the run is about to
        // fail on its own; the common default keeps the guess sane until it does.
        const COMMON: u32 = 0o022;
        let Ok((probe, path)) = create_temporary(directory, 0o666) else {
            return COMMON;
        };
        let cleared = probe.metadata().map_or(COMMON, |metadata| {
            0o666 & !(metadata.permissions().mode() & 0o777)
        });
        let _ = std::fs::remove_file(&path);
        cleared
    })
}

/// Whether replacing this file by rename would detach it from other names for the same
/// inode. Platforms that do not report a link count take the atomic path.
fn is_hardlinked(metadata: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::fs::MetadataExt::nlink(metadata) > 1
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

/// Create a fresh temporary in `directory` at `mode`, which the umask still narrows.
/// `create_new` refuses both an existing path and a symlink, so a planted link cannot
/// redirect the write; the nonce keeps concurrent runs in one directory apart.
fn create_temporary(directory: &Path, mode: u32) -> io::Result<(File, std::path::PathBuf)> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, mode);
    #[cfg(not(unix))]
    let _ = mode;

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

/// Whether a whole input has already been handed over.
enum Walked {
    NotYet,
    Done,
}

/// How the bytes arrive, which callers reach only through [`Windows`]'s methods.
enum Walk {
    Whole {
        source: InputSource,
        walked: Walked,
        /// Bytes the caller has finished with, so a prefix taken first leaves the rest.
        taken: usize,
    },
    Stream {
        refill: Refill<Box<dyn Read>>,
        base: usize,
    },
}

/// One walk over an input's bytes, whether it arrived whole or has to be streamed.
///
/// A mapped file yields one window covering everything; a pipe yields as many as it takes.
/// The base offset comes back with each window so a caller that reports absolute positions
/// does not have to accumulate one itself.
pub struct Windows(Walk);

impl Windows {
    /// Open `source` as a walk, streaming only what cannot be handed over at once.
    pub fn over(source: InputSource) -> Self {
        match source.into_window(DEFAULT_WINDOW_BYTES) {
            InputWindow::Whole(held) => Windows(Walk::Whole {
                source: held,
                walked: Walked::NotYet,
                taken: 0,
            }),
            InputWindow::Stream(refill) => Windows(Walk::Stream { refill, base: 0 }),
        }
    }

    /// A walk that reads through a window of `capacity` bytes, the way a pipe is read.
    pub fn streaming(reader: impl Read + 'static, capacity: usize) -> Self {
        Windows(Walk::Stream {
            refill: Refill::new(Box::new(reader), capacity),
            base: 0,
        })
    }

    /// Every byte at once, where the input could give them at once.
    pub fn whole(&self) -> Option<&[u8]> {
        match &self.0 {
            Walk::Whole { source, .. } => Some(source.as_bytes()),
            Walk::Stream { .. } => None,
        }
    }

    /// Hash the bytes as they arrive. Must be asked before the first window, since a hash
    /// installed later would miss what has already been read.
    pub fn hash_stream(&mut self) {
        if let Walk::Stream { refill, .. } = &mut self.0 {
            refill.hash_stream();
        }
    }

    /// What the stream hashed, once it has been walked to the end.
    pub fn digest(&self) -> Option<u64> {
        match &self.0 {
            Walk::Whole { .. } => None,
            Walk::Stream { refill, .. } => refill.digest(),
        }
    }

    /// Bring bytes into view without consuming any, so a caller can judge what it has
    /// before deciding how much to take. Reports whether anything is in view.
    pub fn fill(&mut self) -> io::Result<bool> {
        match &mut self.0 {
            Walk::Whole { source, taken, .. } => Ok(*taken < source.as_bytes().len()),
            Walk::Stream { refill, .. } => refill.advance(0),
        }
    }

    /// What is in view right now, which [`Windows::fill`] and [`Windows::grow`] widen.
    pub fn filled(&self) -> &[u8] {
        match &self.0 {
            Walk::Whole { source, taken, .. } => &source.as_bytes()[*taken..],
            Walk::Stream { refill, .. } => refill.filled(),
        }
    }

    /// Whether everything the input will ever give is already in the window.
    pub fn at_eof(&self) -> bool {
        match &self.0 {
            Walk::Whole { .. } => true,
            Walk::Stream { refill, .. } => refill.at_eof(),
        }
    }

    /// Widen the window, for a caller that needs more than one window's worth in view at
    /// once. A whole input is already as wide as it gets.
    pub fn grow(&mut self) -> io::Result<()> {
        match &mut self.0 {
            Walk::Whole { .. } => Ok(()),
            Walk::Stream { refill, .. } => refill.grow(),
        }
    }

    /// The next window and where it starts, or `None` at the end.
    ///
    /// `consumed` is how much of the previous window the caller finished with: passing less
    /// than all of it is how a scanner keeps a partial record in view across a seam. Cutting
    /// only on `cut` keeps records whole, and a record wider than the window grows it.
    pub fn next(&mut self, cut: CutAfter, consumed: usize) -> io::Result<Option<(&[u8], usize)>> {
        match &mut self.0 {
            Walk::Whole {
                source,
                walked,
                taken,
            } => {
                let data = source.as_bytes();
                let start = *taken + consumed;
                // A whole input cannot grow, so a caller that consumed nothing has already
                // seen everything there is.
                if matches!(walked, Walked::Done) && (consumed == 0 || start >= data.len()) {
                    return Ok(None);
                }
                *taken = start;
                *walked = Walked::Done;
                Ok(Some((&data[start..], start)))
            }
            Walk::Stream { refill, base } => {
                // Consuming nothing says the last window was not enough, so the window has to
                // widen before it is handed back — otherwise the caller sees it again forever.
                let stalled = consumed == 0 && !refill.filled().is_empty();
                *base += consumed;
                let start = *base;
                if !refill.advance(consumed)? {
                    return Ok(None);
                }
                if stalled && !refill.at_eof() {
                    refill.grow()?;
                }
                loop {
                    if refill.at_eof() {
                        return Ok(Some((refill.filled(), start)));
                    }
                    match last_cut(refill.filled(), cut) {
                        Some(end) => return Ok(Some((&refill.filled()[..end], start))),
                        None => refill.grow()?,
                    }
                }
            }
        }
    }
}

/// Where a run's output lands.
///
/// The four rungs every rewriting tool spells out by hand, in one place and one order. A
/// discarding run still writes, so the count that answers `--quiet` stays honest.
pub enum Destination<'a> {
    /// Written and thrown away, for `--quiet` and `--dry-run`.
    Discard,
    /// The caller's standard output.
    Stdout,
    /// A named file that need not exist, as `--output` asks.
    Creating(&'a str),
    /// The input itself, swapped in atomically, as `--in-place` asks.
    Replacing(&'a str),
}

impl Destination<'_> {
    /// Run `body` against this destination.
    ///
    /// `FnOnce` rather than a borrowed writer, because a replaced file is only durable once
    /// the body has returned and the temporary has been flushed, permissioned, synced and
    /// renamed — a caller holding the writer could not be told when that had happened.
    pub fn write<T>(
        self,
        tool: &str,
        stdout: &mut dyn Write,
        body: impl FnOnce(&mut dyn Write) -> io::Result<T>,
    ) -> Result<T, Failure> {
        match self {
            Destination::Discard => body(&mut io::sink()).at("-"),
            Destination::Stdout => body(stdout).at("-"),
            Destination::Creating(path) => write_creating(tool, path, body),
            Destination::Replacing(path) => write_replacing(tool, path, body),
        }
    }
}

/// Buffered, locked stdout. `io::Stdout` is line-buffered, costing a syscall per record.
///
/// `Drop` discards its flush error, so a run must flush explicitly before returning.
pub fn stdout_writer() -> BufWriter<io::StdoutLock<'static>> {
    BufWriter::new(io::stdout().lock())
}

// endregion: Input Sources

// region: Line Iteration

/// Which newline set [`LineIter`] splits on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Newlines {
    /// Byte-level: only LF (`\n`).
    Lf,
    /// All seven Unicode newline characters (LF, VT, FF, CR, NEL, LS, PS),
    /// with CRLF collapsed into a single break.
    Unicode,
}

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
#[allow(clippy::large_enum_variant)]
pub enum LineIter<'a> {
    Byte(std::iter::Peekable<FindSplits<'a>>),
    Utf8(std::iter::Peekable<Utf8SplitNewlines<'a>>),
}

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

/// A line, with the split a rewrite needs.
///
/// `whole` is exactly the bytes that name it — the line and its terminator together. That
/// the terminator is *included* is what makes a name mean the same thing under either
/// newline set: a span runs to the start of the next line, so a CRLF line spans `\r\n` in
/// both, where a body-only view would see `text\r` under [`Newlines::Lf`] and `text` under
/// [`Newlines::Unicode`].
///
/// [`NamedLine::body`] and [`NamedLine::terminator`] split `whole` where a rewrite replaces
/// it, which is a different question: the terminator is copied, never reconstructed, so a
/// CRLF file stays CRLF and a file with no final newline keeps having none.
///
/// The newline sets still disagree about where lines *begin* for VT, FF, NEL, LS and PS,
/// which no naming scheme can reconcile.
#[derive(Clone, Copy, Debug)]
pub struct NamedLine<'a> {
    /// Where the line starts in the input.
    pub offset: usize,
    /// The line and its terminator: the bytes [`NamedLine::hash`] covers.
    pub whole: &'a [u8],
    /// The line as the newline set cut it, which under [`Newlines::Lf`] keeps the carriage
    /// return of a CRLF pair — what a search matches against and what `grep` prints. Named
    /// for the cut rather than for the line, because `whole` and [`NamedLine::body`] are
    /// equally the line and the three differ only in what they include of its ending.
    pub as_cut: &'a [u8],
    /// How much of `whole` precedes the terminator.
    body_len: usize,
}

impl<'a> NamedLine<'a> {
    /// The line's text, without whatever ends it.
    #[inline]
    pub fn body(&self) -> &'a [u8] {
        &self.whole[..self.body_len]
    }

    /// What ends the line: nothing at the end of a file that lacks a final newline.
    #[inline]
    pub fn terminator(&self) -> &'a [u8] {
        &self.whole[self.body_len..]
    }

    /// Where the line ends, terminator included.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.whole.len()
    }

    /// The hash naming this line, which [`format_hash`] renders into the token a caller
    /// passes back. The cost of covering the terminator is that adding a final newline
    /// renames a file's last line, which is the one place the two byte strings differ.
    #[inline]
    pub fn hash(&self) -> u64 {
        content_hash(self.whole)
    }
}

/// Iterate the lines of `data`, each carrying its terminator.
pub fn named_lines(data: &[u8], newlines: Newlines) -> NamedLines<'_> {
    let mut lines = LineIter::new(data, newlines);
    let pending = lines.next();
    NamedLines {
        data,
        lines,
        pending,
    }
}

pub struct NamedLines<'a> {
    data: &'a [u8],
    lines: LineIter<'a>,
    pending: Option<&'a [u8]>,
}

impl<'a> Iterator for NamedLines<'a> {
    type Item = NamedLine<'a>;

    #[inline]
    fn next(&mut self) -> Option<NamedLine<'a>> {
        let line = self.pending.take()?;
        let offset = offset_within(self.data, line);
        self.pending = self.lines.next();
        let end = match self.pending {
            Some(next) => offset_within(self.data, next),
            None => self.data.len(),
        };
        // `Newlines::Lf` leaves the carriage return of a CRLF line on the line and the
        // Unicode set takes it off, so the body is trimmed to agree with both. The name is
        // unaffected either way, since it covers the terminator too.
        let body_len = line.len() - usize::from(line.ends_with(b"\r"));
        Some(NamedLine {
            offset,
            whole: &self.data[offset..end],
            as_cut: line,
            body_len,
        })
    }
}

/// A byte range, in the coordinates of the buffer it was found in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
pub struct LineSpans<'a> {
    data: &'a [u8],
    lines: LineIter<'a>,
    pending: Option<&'a [u8]>,
}

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
pub const DEFAULT_WINDOW_BYTES: usize = 256 << 10;

/// A caller-driven byte window over a reader: one allocation per run, reused for the
/// whole stream, with the caller choosing how much carries by choosing `consumed`.
///
/// This exists rather than a [`std::io::BufRead`] because `fill_buf` carries __no fill
/// guarantee__ — it may hand back four bytes and offers no way to demand more.
/// [`Refill::advance`] fills to capacity or EOF, so a caller can demand a whole record
/// and get one.
pub struct Refill<R> {
    reader: R,
    buffer: Box<[u8]>,
    valid: usize,
    reached_eof: bool,
    /// Hashes the stream as it arrives, when a caller asked for a whole-input hash. Boxed
    /// because `sz::Hasher` is cache-line aligned and would otherwise make a `Refill` far
    /// larger than the other shape an input can take.
    hasher: Option<Box<sz::Hasher>>,
}

impl<R: Read> Refill<R> {
    /// Window `reader` through `capacity` bytes, rounded up to one byte. `reader` should be
    /// the raw stream: a `BufReader` under it would stage every byte a second time.
    pub fn new(reader: R, capacity: usize) -> Self {
        Refill {
            reader,
            buffer: vec![0u8; capacity.max(1)].into_boxed_slice(),
            valid: 0,
            reached_eof: false,
            hasher: None,
        }
    }

    /// Hash the stream as it is read, so a whole-input hash costs no second pass and no
    /// buffer holding the input.
    ///
    /// Accumulated in `Refill::fill`, which is the only place a byte enters the window and
    /// so the only place that sees each one exactly once. Hashing where windows are *handed
    /// out* would double-count every byte a cut retained, re-count the whole window each time
    /// it grew, and miss whatever a caller consumed outside the driver.
    pub fn hash_stream(&mut self) {
        // A hash covers bytes that have not gone past yet, and a window filled before this
        // call has already taken some. Asserted rather than tolerated: the result would be a
        // digest of a suffix, which is indistinguishable from a digest of the input.
        debug_assert!(
            self.valid == 0 && !self.reached_eof,
            "a stream is hashed from its first byte or not at all"
        );
        self.hasher
            .get_or_insert_with(|| Box::new(sz::Hasher::new(0)));
    }

    /// The hash of everything read so far, equal to [`content_hash`] of those bytes.
    ///
    /// A run that stops early has read less than the input holds, so this describes what was
    /// read rather than what the writer still has to give.
    pub fn digest(&self) -> Option<u64> {
        self.hasher.as_ref().map(|state| state.digest())
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

    /// Hand `body` every window, for a caller that reads its input to the end.
    pub fn for_each_window(
        &mut self,
        cut: CutAfter,
        mut body: impl FnMut(&[u8]) -> io::Result<()>,
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
            body(&self.filled()[..end])?;
            consumed = end;
        }
        Ok(())
    }

    /// Read until the window is full or the reader is exhausted, retrying interruptions.
    fn fill(&mut self) -> io::Result<()> {
        while !self.reached_eof && self.valid < self.buffer.len() {
            match self.reader.read(&mut self.buffer[self.valid..]) {
                Ok(0) => self.reached_eof = true,
                Ok(read) => {
                    if let Some(state) = self.hasher.as_mut() {
                        state.update(&self.buffer[self.valid..self.valid + read]);
                    }
                    self.valid += read;
                }
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
pub enum Terminator {
    /// A newline, which is lossy for records that contain one.
    Newline,
    /// A NUL byte, which no text record can contain.
    Null,
}

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

/// Write one `{"type":"line","data":{"path":…,"text":…,"line_number":N}}` record.
///
/// `index` is zero-based; the record reports it one-based, as every tool numbers lines.
pub fn write_line_record(
    output: &mut dyn Write,
    path: &str,
    line: &[u8],
    index: usize,
    hash: Option<&str>,
) -> io::Result<()> {
    output.write_all(br#"{"type":"line","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    output.write_all(br#","text":"#)?;
    json_text_field_to(output, line)?;
    write!(output, r#","line_number":{}"#, index + 1)?;
    // Named the same as `sz-find`'s column, since it is the same value read back the same way.
    if let Some(hash) = hash {
        write!(output, r#","line_hash":"{hash}""#)?;
    }
    output.write_all(b"}}")?;
    output.write_all(b"\n")
}

/// Write the `{"text":"…"}` wrapper that every path and line field uses, falling back to
/// `{"bytes":"<base64>"}` when the slice is not valid UTF-8.
///
/// Passing invalid bytes through raw would emit JSON no decoder accepts, and text tools do
/// meet non-UTF-8 input. The two-arm shape is ripgrep's, which this envelope already follows.
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

// region: Content Hashing

/// How many characters render a whole 64-bit hash in [`format_hash`]. Thirteen carry five
/// bits each, which is one more bit than a `u64` holds; the spare one is a zero at the end.
pub const HASH_CHARS: usize = 13;

/// The shortest prefix a caller may name a line by. Four characters is twenty bits, which
/// stays comfortable within one file; shorter is a typo rather than an abbreviation.
pub const HASH_CHARS_MIN: usize = 4;

/// Crockford's base32: the digits and the lowercase letters, less `i`, `l`, `o` and `u`,
/// which are the four a reader confuses with `1`, `1`, `0` and `v`. Alphanumeric, so a hash
/// never needs shell quoting and never opens with a `-` that reads as a flag.
const HASH_ALPHABET: &[u8; 32] = b"0123456789abcdefghjkmnpqrstvwxyz";

/// How many characters of a line hash print unless `--hash-width` says otherwise. Eight is
/// forty bits, comfortable within one file, and two tokens of an LLM's context per line.
pub const DEFAULT_HASH_WIDTH: usize = 8;

/// Accept a hash width the renderer can actually produce, and that is long enough to be an
/// abbreviation rather than a typo.
pub fn parse_hash_width(value: &str) -> Result<usize, String> {
    let width: usize = value
        .parse()
        .map_err(|_| format!("`{value}` is not a number"))?;
    if !(HASH_CHARS_MIN..=HASH_CHARS).contains(&width) {
        return Err(format!(
            "hash width must be between {HASH_CHARS_MIN} and {HASH_CHARS}"
        ));
    }
    Ok(width)
}

/// Hash the bytes a run read or wrote, as `--expect-hash` compares against.
///
/// StringZilla's 64-bit AES hash under seed zero, whose header promises "the same output on
/// all platforms in both single-shot and incremental modes" — which is what lets a token
/// written on one machine mean the same thing on another, and what lets [`HashingWriter`]
/// accumulate one over a stream rather than buffering it. It is not cryptographic: it
/// answers "did this file change", not "did somebody change it".
pub fn content_hash(data: &[u8]) -> u64 {
    sz::hash(data)
}

/// Render the leading `width` characters of `hash`, most significant first.
///
/// Truncation rather than folding, so a short name is always a prefix of the long one and
/// `sz-find --hash-width 4` agrees with the first four characters of `--hash-width 13`.
pub fn format_hash(buffer: &mut [u8; HASH_CHARS], hash: u64, width: usize) -> &str {
    let width = width.clamp(1, HASH_CHARS);
    // Sixty-five bits are rendered, not sixty-four: appending a zero puts the one symbol
    // that cannot range over the whole alphabet at the *end*, where only a full-width name
    // reaches it. Every shorter name then pins a clean five bits per character.
    let padded = (hash as u128) << 1;
    for (index, slot) in buffer.iter_mut().enumerate() {
        let shift = 5 * (HASH_CHARS - 1 - index);
        *slot = HASH_ALPHABET[((padded >> shift) & 0x1F) as usize];
    }
    // Only alphabet bytes were written.
    core::str::from_utf8(&buffer[..width]).expect("the alphabet is ASCII")
}

/// The bits a name constrains: a hash renders through [`format_hash`] beginning with this
/// name exactly when `hash & mask == value`. A file's token is the case where every bit is
/// pinned, which is why one type serves both.
///
/// Because rendering truncates rather than folds, a name of `n` characters pins the leading
/// `5n` bits, so matching is one AND and one compare per line — no line is ever rendered to
/// be compared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HashPrefix {
    value: u64,
    mask: u64,
}

impl HashPrefix {
    #[inline]
    pub fn matches(&self, hash: u64) -> bool {
        hash & self.mask == self.value
    }
}

/// Parse a name, as `sz-find` prints one: a line's under `--fields line-hashes`, a file's
/// under `--fields file-hash`. One alphabet and one length rule serve both — a full-width
/// name pins a whole hash, a shorter one pins its leading bits.
pub fn parse_hash_prefix(name: &str) -> Result<HashPrefix, String> {
    if !(HASH_CHARS_MIN..=HASH_CHARS).contains(&name.len()) {
        return Err(format!(
            "`{name}` is {} characters; a name is {HASH_CHARS_MIN} to {HASH_CHARS}",
            name.len()
        ));
    }

    let mut padded = 0u128;
    for (index, byte) in name.bytes().enumerate() {
        let lowered = byte.to_ascii_lowercase();
        let Some(digit) = HASH_ALPHABET.iter().position(|entry| *entry == lowered) else {
            return Err(format!(
                "`{name}` is not a name: `{}` is not in the alphabet, which drops \
                 i, l, o and u to keep names legible",
                byte as char
            ));
        };
        // The last character of a full-width name carries the hash's final four bits against
        // the padding zero [`format_hash`] appends, so an odd symbol cannot close one. Every
        // earlier character, and every character of a shorter name, ranges over all thirty-two.
        if index == HASH_CHARS - 1 && digit % 2 == 1 {
            return Err(format!(
                "no name ends with `{}`: the last character carries four bits against a \
                 padding zero, so it is always one of 02468acegjmprtwy",
                byte as char
            ));
        }
        padded |= (digit as u128) << (5 * (HASH_CHARS - 1 - index));
    }

    // Shifted back off the padding bit, so both halves describe the hash itself and matching
    // needs no rendering. `5 * HASH_CHARS` bits wide, one more than the hash it carries.
    let unpinned = 5 * (HASH_CHARS - name.len());
    let mask = (((!0u128 << unpinned) & ((1u128 << (5 * HASH_CHARS)) - 1)) >> 1) as u64;
    Ok(HashPrefix {
        value: (padded >> 1) as u64 & mask,
        mask,
    })
}

/// Parse a whole-file token: a name at full width, as `sz-find --fields file-hash` prints
/// one. Anything shorter is refused rather than zero-extended, so a truncated paste fails
/// loudly instead of standing in for some other file.
pub fn parse_content_hash(value: &str) -> Result<u64, String> {
    if value.len() != HASH_CHARS {
        return Err(format!(
            "`{value}` is {} characters; a file's hash is {HASH_CHARS}, as \
             `sz-find --fields file-hash` prints it",
            value.len()
        ));
    }
    Ok(parse_hash_prefix(value)?.value)
}

/// A writer that hashes every byte on its way through, so a stream can be checksummed
/// without a second pass over it or a buffer holding the whole result.
pub struct HashingWriter<'a> {
    inner: &'a mut dyn Write,
    state: sz::Hasher,
}

impl<'a> HashingWriter<'a> {
    pub fn new(inner: &'a mut dyn Write) -> Self {
        HashingWriter {
            inner,
            state: sz::Hasher::new(0),
        }
    }

    /// The hash of everything written so far, equal to [`content_hash`] of those bytes.
    pub fn digest(&self) -> u64 {
        self.state.digest()
    }
}

impl Write for HashingWriter<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let written = self.inner.write(data)?;
        self.state.update(&data[..written]);
        Ok(written)
    }

    /// Overridden rather than inherited: every caller here writes whole slices, and the
    /// default implementation would chunk the hash updates through partial-write bookkeeping
    /// that never happens.
    fn write_all(&mut self, data: &[u8]) -> io::Result<()> {
        self.inner.write_all(data)?;
        self.state.update(data);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

// endregion: Content Hashing

// region: Argument Parsing

/// Parse a count that has to be at least 1. Clap's stock [`NonZeroUsize`] parser answers
/// "number would be zero for non-zero type", naming a Rust type where the bound belongs.
/// Every other input keeps the stock wording, so only the zero case reads differently.
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

// region: Directory Traversal

/// Which files a walk sees. The five flags every directory-accepting tool shares, lifted out
/// of `Args` so a walk can be built without knowing what a CLI is.
#[derive(Clone, Default)]
pub struct TraversalOptions<'a> {
    pub hidden: bool,
    pub no_ignore: bool,
    pub follow: bool,
    pub max_depth: Option<usize>,
    pub file_type: Option<&'a [String]>,
}

/// Build a walk of `root` under `options`.
///
/// An unusable `--type` is reported and dropped rather than aborting the walk, so one bad
/// filter does not cost the caller every other input; `tool` names the reporter.
pub fn walker(root: &Path, options: &TraversalOptions<'_>, tool: &str) -> ignore::Walk {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .hidden(!options.hidden)
        .git_ignore(!options.no_ignore)
        .git_global(!options.no_ignore)
        .git_exclude(!options.no_ignore)
        .follow_links(options.follow)
        .max_depth(options.max_depth)
        // Sorted as it reads, per directory, so a walk is reproducible without collecting it.
        .sort_by_file_path(Path::cmp);

    if let Some(names) = options.file_type {
        let mut types = ignore::types::TypesBuilder::new();
        types.add_defaults();
        for name in names {
            types.select(name);
        }
        match types.build() {
            Ok(matcher) => {
                builder.types(matcher);
            }
            Err(error) => eprintln!("{tool}: warning: invalid --type: {error}"),
        }
    }
    builder.build()
}

/// Compile each glob once, so a malformed one is refused here rather than silently matching
/// nothing on every file of a walk — a dropped pattern and a pattern that selects nothing are
/// different answers, and only the second is a result.
///
/// The message comes back rather than being printed, because only the caller can render a
/// usage error the way clap renders a parse failure.
pub fn compile_globs(patterns: &[String]) -> Result<Vec<glob::Pattern>, String> {
    patterns
        .iter()
        .map(|pattern| {
            glob::Pattern::new(pattern)
                .map_err(|error| format!("invalid glob '{}': {}", pattern, error))
        })
        .collect()
}

/// Whether a walked entry passes a glob filter, which matches either the whole path or the file
/// name. No filter accepts everything.
///
/// Takes the entry rather than its path because [`ignore::DirEntry::file_name`] falls back to the
/// whole path where there is no final component, which `Path::file_name` reports as nothing at
/// all — so a walk rooted at `.` or `/` filters on the name the user typed.
fn glob_selects(globs: Option<&[glob::Pattern]>, entry: &ignore::DirEntry) -> bool {
    let Some(globs) = globs else {
        return true;
    };
    let path_text = entry.path().to_string_lossy();
    let name_text = entry.file_name().to_string_lossy();
    globs
        .iter()
        .any(|pattern| pattern.matches(&path_text) || pattern.matches(&name_text))
}

/// One resolved input: the standard input, or a file the walk reached.
///
/// A named file and a walked file are the same thing, because `ignore::Walk` over a plain path
/// yields one depth-zero entry for it. The entry owns the path the walk already allocated, so
/// holding it costs nothing beyond what the walk spent.
pub enum Input {
    Stdin,
    File(ignore::DirEntry),
}

impl Input {
    /// Where this input lives. The standard input answers `-`, as every tool prints it.
    pub fn path(&self) -> &Path {
        match self {
            Input::Stdin => Path::new("-"),
            Input::File(entry) => entry.path(),
        }
    }

    /// The name a record carries, lossy where a path is not UTF-8.
    pub fn display_name(&self) -> Cow<'_, str> {
        self.path().to_string_lossy()
    }

    /// The file's length, asked for only when a caller needs it: on Unix this is a `stat` the
    /// walk did not already pay for, so it stays out of the walk.
    pub fn size(&self) -> io::Result<u64> {
        match self {
            Input::Stdin => Ok(0),
            Input::File(entry) => entry.metadata().map(|data| data.len()).map_err(|error| {
                error
                    .into_io_error()
                    .unwrap_or_else(|| io::Error::other("could not measure the file"))
            }),
        }
    }
}

/// Every input the names resolve to, in the order they were named, walking directories as it
/// goes.
///
/// Lazy, and failures arrive as items rather than through a counter the caller has to thread:
/// a consumer that stops early never walks the rest, and one that wants them all collects.
pub fn inputs<'a>(
    names: &'a [String],
    traversal: &'a TraversalOptions<'a>,
    globs: Option<&'a [glob::Pattern]>,
    tool: &'a str,
) -> impl Iterator<Item = Result<Input, Failure>> + 'a {
    names.iter().flat_map(move |name| {
        let stdin = (name == "-").then(|| Ok(Input::Stdin));
        // A name the filesystem cannot answer for is reported here rather than left to the
        // walker, whose message already carries the path and would print it twice.
        let unreadable = (name != "-")
            .then(|| fs::metadata(Path::new(name)).err())
            .flatten()
            .map(|source| {
                Err(Failure::Io {
                    path: name.clone(),
                    source,
                })
            });
        let walked = (name != "-" && unreadable.is_none()).then(|| {
            // `--glob` and `--type` filter what a walk finds, never what the caller asked for
            // by name, so a named file carries neither.
            let named = !Path::new(name).is_dir();
            let options = TraversalOptions {
                file_type: if named { None } else { traversal.file_type },
                ..traversal.clone()
            };
            walker(Path::new(name), &options, tool).filter_map(move |entry| match entry {
                Ok(entry) if !is_readable_entry(&entry) => None,
                Ok(entry) if !named && !glob_selects(globs, &entry) => None,
                Ok(entry) => Some(Ok(Input::File(entry))),
                Err(error) => Some(Err(Failure::Io {
                    path: name.clone(),
                    source: io::Error::other(error.to_string()),
                })),
            })
        });
        stdin
            .into_iter()
            .chain(unreadable)
            .chain(walked.into_iter().flatten())
    })
}

// endregion: Directory Traversal

// region: Failure and Exit Conventions

/// Process exit status, on `grep`'s model.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Success = 0,  // Ran, and produced a result
    NoResult = 1, // Ran, but found nothing
    Error = 2,    // Did not run to completion
}

/// What a refused precondition reports, distinct from [`Status::NoResult`] (ran, found
/// nothing) and [`Status::Error`] (could not run). A caller reads it as "look again", not
/// as "fix your arguments", and those are different recoveries. Not a [`Status`], because a
/// run that declined produced nothing and so is only ever reached from an `Err`.
pub const REFUSED_EXIT_CODE: u8 = 3;

/// What a run has seen, folded as it goes, so no caller counts on the side.
///
/// A failure is anything that stopped an input being read or written; `produced` is whatever
/// that tool calls a result — a row, a record, a chunk. The pair answers the exit code.
#[derive(Default, Clone, Copy, Debug)]
pub struct Tally {
    failures: usize,
    produced: bool,
}

impl Tally {
    /// Record an input that could not be read.
    #[inline]
    pub fn failed(&mut self) {
        self.failures += 1;
    }

    /// Record that the run produced something.
    #[inline]
    pub fn produced(&mut self) {
        self.produced = true;
    }

    /// Whether anything failed, which some tools also report in prose.
    #[inline]
    pub fn failures(&self) -> usize {
        self.failures
    }

    /// The exit status this run has earned.
    #[inline]
    pub fn status(self) -> Status {
        Status::of(self.failures > 0, self.produced)
    }
}

impl Status {
    /// The status a finished run reports.
    ///
    /// An input that could not be read means the run did not complete, whatever its readable
    /// siblings produced — the answer `grep` gives, and the only one a caller can act on
    /// without re-reading stderr.
    #[inline]
    pub fn of(any_input_failed: bool, produced: bool) -> Self {
        if any_input_failed {
            Status::Error
        } else {
            Status::from_found(produced)
        }
    }

    /// Map a "found something" flag to the success/no-result pair.
    #[inline]
    pub fn from_found(found: bool) -> Self {
        if found {
            Status::Success
        } else {
            Status::NoResult
        }
    }
}

/// Anything that ends a run early.
///
/// There is deliberately no broken-pipe variant: keeping the [`io::ErrorKind`] intact lets
/// [`report`] recognise a closed downstream in one place instead of at every write site.
pub enum Failure {
    /// An I/O failure, naming the path it happened to.
    Io { path: String, source: io::Error },
    /// A usage error, already rendered by clap so it matches a parse failure exactly.
    Usage(clap::Error),
    /// A precondition naming a file's contents that no longer holds: the file changed
    /// between the read that produced the caller's hash and this run.
    Stale {
        path: String,
        expected: u64,
        actual: u64,
    },
    /// A name that had to pick out one thing and picked out several. Reported rather than
    /// resolved, because choosing among candidates is what the caller came to decide.
    Ambiguous {
        path: String,
        subject: String,
        matches: usize,
        /// How this caller suggests narrowing it. Supplied by the binary, since the flag
        /// that widens the selection differs between them.
        note: &'static str,
    },
    /// A name or a pattern that picked out nothing. Distinct from [`Failure::Ambiguous`],
    /// which found too many rather than none. Which recovery applies depends on what was
    /// looked for — a name that resolves nowhere means the file moved, a substring that
    /// matches nothing means the text is not there — so the caller supplies it in `note`.
    Unresolved {
        path: String,
        subject: String,
        /// How this caller suggests recovering, since only it knows what was looked for.
        note: &'static str,
    },
}

impl From<clap::Error> for Failure {
    fn from(error: clap::Error) -> Self {
        Failure::Usage(error)
    }
}

// Deferring `Debug` to `Display` keeps `unwrap()` in tests readable: `io::Error`'s own
// `Debug` prints a struct dump where the message is what the caller wants to see.
impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Io { path, source } => write!(formatter, "{path}: {source}"),
            Failure::Usage(error) => write!(formatter, "{error}"),
            // Both hashes are printed because the recovery is to compare them against what
            // the caller still holds, and neither alone says which read went out of date.
            Failure::Stale {
                path,
                expected,
                actual,
            } => {
                let (mut found, mut wanted) = ([0u8; HASH_CHARS], [0u8; HASH_CHARS]);
                write!(
                    formatter,
                    "{path}: content is {}, not the expected {}; re-read it before editing",
                    format_hash(&mut found, *actual, HASH_CHARS),
                    format_hash(&mut wanted, *expected, HASH_CHARS)
                )
            }
            Failure::Ambiguous {
                path,
                subject,
                matches,
                note,
            } => write!(
                formatter,
                "{path}: `{subject}` matches {matches} places, not one; {note}"
            ),
            Failure::Unresolved {
                path,
                subject,
                note,
            } => write!(formatter, "{path}: `{subject}` {note}"),
        }
    }
}

/// Attach the path an [`io::Result`] failed on.
///
/// There is no blanket `From<io::Error> for Failure`, so a bare `?` on an I/O call will not
/// compile — which is what makes every failure name its file.
pub trait At<T> {
    fn at(self, path: impl Into<String>) -> Result<T, Failure>;
}

impl<T> At<T> for io::Result<T> {
    #[inline]
    fn at(self, path: impl Into<String>) -> Result<T, Failure> {
        self.map_err(|source| Failure::Io {
            path: path.into(),
            source,
        })
    }
}

/// Turn a finished run into the process's exit status.
///
/// The one place that decides what a closed downstream means, and the only place that knows
/// the mapping from [`Status`] to a number.
pub fn report(tool: &str, outcome: Result<Status, Failure>) -> process::ExitCode {
    match outcome {
        Ok(status) => process::ExitCode::from(status as u8),
        // A downstream that closed early is a normal end, not a failure.
        Err(Failure::Io { source, .. }) if source.kind() == io::ErrorKind::BrokenPipe => {
            process::ExitCode::SUCCESS
        }
        // clap picks the stream and the code: 0 for `--help`, 2 for a real usage error.
        Err(Failure::Usage(error)) => {
            let _ = error.print();
            process::ExitCode::from(error.exit_code() as u8)
        }
        // A refused precondition ran correctly and declined, so it reports neither "found
        // nothing" nor "could not run".
        Err(
            failure @ (Failure::Stale { .. }
            | Failure::Ambiguous { .. }
            | Failure::Unresolved { .. }),
        ) => {
            note(tool, &failure);
            process::ExitCode::from(REFUSED_EXIT_CODE)
        }
        // Named rather than caught, so a new variant is a compile error instead of a
        // silent exit 2.
        Err(failure @ Failure::Io { .. }) => {
            note(tool, &failure);
            process::ExitCode::from(Status::Error as u8)
        }
    }
}

/// Write one diagnostic to stderr. Unlike `eprintln!`, a failed write is not a panic.
fn note(tool: &str, failure: &Failure) {
    let _ = writeln!(io::stderr(), "{tool}: {failure}");
}

// endregion: Failure and Exit Conventions

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
    fn rewrites_the_input_through_a_temporary_file() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("lines.txt");
        fs::write(&path, b"b\na\n").unwrap();

        write_replacing("sz-test", path.to_str().unwrap(), |output| {
            output.write_all(b"a\nb\n")
        })
        .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"a\nb\n");
        let leftovers: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(leftovers, ["lines.txt"]);
    }
    #[test]
    #[cfg(unix)]
    fn swaps_by_rename_but_keeps_a_hardlinked_inode() {
        use std::os::unix::fs::MetadataExt;

        let directory = tempfile::TempDir::new().unwrap();

        // Unlinked: the swap is a rename, so the path gets a new inode and the replacement
        // is atomic.
        let plain = directory.path().join("plain.txt");
        fs::write(&plain, b"b\na\n").unwrap();
        let before = fs::metadata(&plain).unwrap().ino();
        write_replacing("sz-test", plain.to_str().unwrap(), |output| {
            output.write_all(b"a\nb\n")
        })
        .unwrap();
        assert_ne!(fs::metadata(&plain).unwrap().ino(), before);

        // Hardlinked: renaming would strand the other name on the old content, so the inode
        // is rewritten instead and both names see the result.
        let first = directory.path().join("first.txt");
        let second = directory.path().join("second.txt");
        fs::write(&first, b"b\na\n").unwrap();
        fs::hard_link(&first, &second).unwrap();
        let before = fs::metadata(&first).unwrap().ino();
        write_replacing("sz-test", first.to_str().unwrap(), |output| {
            output.write_all(b"a\nb\n")
        })
        .unwrap();
        assert_eq!(fs::metadata(&first).unwrap().ino(), before);
        assert_eq!(fs::read(&second).unwrap(), b"a\nb\n");
    }
    #[test]
    fn leaves_the_input_untouched_when_the_rewrite_fails() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("lines.txt");
        fs::write(&path, b"original\n").unwrap();

        let failed: Result<(), Failure> =
            write_replacing("sz-test", path.to_str().unwrap(), |output| {
                output.write_all(b"partial\n")?;
                Err(io::Error::other("interrupted"))
            });

        assert!(failed.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original\n");
    }

    #[test]
    fn creates_an_output_file_through_a_temporary() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("fresh.txt");

        write_creating("sz-test", path.to_str().unwrap(), |output| {
            output.write_all(b"written\n")
        })
        .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"written\n");
        // The temporary is gone, so the directory holds only what was asked for.
        let mut names: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, vec![std::ffi::OsString::from("fresh.txt")]);
    }

    #[test]
    fn keeps_the_previous_output_when_the_write_fails() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("out.txt");
        fs::write(&path, b"previous\n").unwrap();

        let failed: Result<(), Failure> =
            write_creating("sz-test", path.to_str().unwrap(), |output| {
                output.write_all(b"partial\n")?;
                Err(io::Error::other("interrupted"))
            });

        // `File::create` would have truncated it before the closure ever ran.
        assert!(failed.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous\n");
    }

    #[test]
    #[cfg(unix)]
    fn preserves_the_permissions_of_an_existing_output_file() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("out.txt");
        fs::write(&path, b"previous\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();

        write_creating("sz-test", path.to_str().unwrap(), |output| {
            output.write_all(b"written\n")
        })
        .unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
    }

    #[test]
    #[cfg(unix)]
    fn keeps_a_temporary_private_while_it_is_incomplete() {
        // The window a temporary spends unfinished is the window its content is a prefix of
        // what was meant, so it is nobody's business until the rename. Checked from inside
        // the write, because by the time the call returns there is no temporary to look at.
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("fresh.txt");
        let mut seen = None;

        write_creating("sz-test", path.to_str().unwrap(), |output| {
            output.write_all(b"half a secret")?;
            seen = fs::read_dir(directory.path())
                .unwrap()
                .filter_map(|entry| entry.ok())
                .find(|entry| entry.file_name().to_string_lossy().starts_with(".sz."))
                .map(|entry| entry.metadata().unwrap().permissions().mode() & 0o777);
            Ok(())
        })
        .unwrap();

        assert_eq!(
            seen,
            Some(0o600),
            "the temporary was readable before it was finished"
        );
    }

    #[test]
    #[cfg(unix)]
    fn leaves_a_fresh_output_file_to_the_umask() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::TempDir::new().unwrap();
        let ours = directory.path().join("ours.txt");
        let theirs = directory.path().join("theirs.txt");

        write_creating("sz-test", ours.to_str().unwrap(), |output| {
            output.write_all(b"written\n")
        })
        .unwrap();
        // What `File::create` produces under the same umask, which is what `--output` used
        // to do and must keep doing.
        File::create(&theirs).unwrap();

        let mode_of = |path: &Path| fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode_of(&ours), mode_of(&theirs));
    }

    #[test]
    #[cfg(unix)]
    fn writes_through_a_name_that_does_not_denote_what_it_opens() {
        // `/dev/stdout` opens fine and, under a redirect to a regular file, resolves to
        // `/dev/fd/<the target's basename>` — a name that does not exist. Deciding by
        // `is_file` alone would send that down the rename path and fail; the round-trip
        // test is what catches it.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("out.txt");
        fs::write(&path, b"seed\n").unwrap();
        let opened = OpenOptions::new().write(true).open(&path).unwrap();
        let metadata = opened.metadata().unwrap();

        assert!(denotes(&path, &metadata));
        assert!(!denotes(Path::new("/dev/fd/out.txt"), &metadata));
        assert!(!denotes(directory.path(), &metadata));
    }

    #[test]
    #[cfg(unix)]
    fn creates_the_target_of_a_dangling_symlink_rather_than_replacing_the_link() {
        let directory = tempfile::TempDir::new().unwrap();
        let link = directory.path().join("link.txt");
        let target = directory.path().join("target.txt");
        std::os::unix::fs::symlink("target.txt", &link).unwrap();

        write_creating("sz-test", link.to_str().unwrap(), |output| {
            output.write_all(b"written\n")
        })
        .unwrap();

        assert!(fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(&target).unwrap(), b"written\n");
    }

    #[test]
    #[cfg(unix)]
    fn writes_straight_through_a_target_that_is_not_a_regular_file() {
        // Renaming a regular file over `/dev/null` would replace the device node for every
        // process on the machine, so this path must never reach the temporary. A run that
        // could create the temporary in `/dev` is exactly the run where a regression does
        // that damage, so it declines to be the one that finds out.
        let probe = Path::new("/dev/.sz-write-probe");
        if File::create(probe).is_ok() {
            let _ = fs::remove_file(probe);
            return;
        }

        write_creating("sz-test", "/dev/null", |output| {
            output.write_all(b"discarded\n")
        })
        .unwrap();

        assert!(!fs::metadata("/dev/null").unwrap().is_file());
    }

    #[test]
    fn resolves_a_named_file_and_a_directory_through_one_walk() {
        let directory = tempfile::TempDir::new().unwrap();
        let root = directory.path();
        fs::write(root.join("one.txt"), b"a\n").unwrap();
        fs::write(root.join("two.log"), b"b\n").unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        fs::write(root.join("nested/three.txt"), b"c\n").unwrap();

        let traversal = TraversalOptions {
            hidden: false,
            no_ignore: true,
            follow: false,
            max_depth: None,
            file_type: None,
        };

        // A named file yields exactly itself: `ignore::Walk` over a plain path is one entry,
        // so there is no separate code path for it.
        let named = vec![root.join("one.txt").display().to_string()];
        let found: Vec<_> = inputs(&named, &traversal, None, "sz-test")
            .map(|input| input.unwrap().display_name().into_owned())
            .collect();
        assert_eq!(found.len(), 1);
        assert!(found[0].ends_with("one.txt"));

        // A directory yields every file under it, and stdin rides the same stream.
        let mixed = vec!["-".to_string(), root.display().to_string()];
        let mut names: Vec<String> = inputs(&mixed, &traversal, None, "sz-test")
            .map(|input| input.unwrap().display_name().into_owned())
            .collect();
        names.sort();
        assert_eq!(names.len(), 4, "stdin plus three files: {:?}", names);
        assert!(names.contains(&"-".to_string()));
        assert!(names.iter().any(|name| name.ends_with("three.txt")));
    }

    #[test]
    fn a_glob_selects_without_hiding_a_walk_failure() {
        let directory = tempfile::TempDir::new().unwrap();
        let root = directory.path();
        fs::write(root.join("keep.txt"), b"a\n").unwrap();
        fs::write(root.join("drop.log"), b"b\n").unwrap();

        let traversal = TraversalOptions {
            hidden: false,
            no_ignore: true,
            follow: false,
            max_depth: None,
            file_type: None,
        };
        let globs = vec![glob::Pattern::new("*.txt").unwrap()];
        let names = vec![root.display().to_string()];

        let kept: Vec<String> = inputs(&names, &traversal, Some(&globs), "sz-test")
            .map(|input| input.unwrap().display_name().into_owned())
            .collect();
        assert_eq!(kept.len(), 1, "{:?}", kept);
        assert!(kept[0].ends_with("keep.txt"));
    }

    #[test]
    fn a_named_file_survives_a_filter_that_would_exclude_it() {
        // Every tool's help promises that `--glob` and `--type` filter walked files and that
        // a named file is always taken. Filtering the name the caller typed would make the
        // flag mean the opposite of what it says.
        let directory = tempfile::TempDir::new().unwrap();
        let named = directory.path().join("kept.log");
        fs::write(&named, b"a\n").unwrap();
        fs::write(directory.path().join("walked.log"), b"b\n").unwrap();
        fs::write(directory.path().join("walked.txt"), b"c\n").unwrap();

        let traversal = TraversalOptions {
            hidden: false,
            no_ignore: true,
            follow: false,
            max_depth: None,
            file_type: None,
        };
        let globs = vec![glob::Pattern::new("*.txt").unwrap()];

        let by_name = vec![named.display().to_string()];
        let taken: Vec<_> = inputs(&by_name, &traversal, Some(&globs), "sz-test").collect();
        assert_eq!(taken.len(), 1, "a named file is not filtered by --glob");

        let by_walk = vec![directory.path().display().to_string()];
        let found: Vec<String> = inputs(&by_walk, &traversal, Some(&globs), "sz-test")
            .map(|input| input.unwrap().display_name().into_owned())
            .collect();
        assert_eq!(found.len(), 1, "a walk is filtered: {:?}", found);
        assert!(found[0].ends_with("walked.txt"));
    }

    #[test]
    fn an_unreadable_input_arrives_as_an_item_not_a_counter() {
        let traversal = TraversalOptions {
            hidden: false,
            no_ignore: true,
            follow: false,
            max_depth: None,
            file_type: None,
        };
        let names = vec!["no-such-path-anywhere".to_string()];
        let outcomes: Vec<_> = inputs(&names, &traversal, None, "sz-test").collect();
        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0].is_err(),
            "a missing input is an item in the stream, not a side channel"
        );
    }

    #[test]
    fn walks_a_whole_input_as_one_window() {
        let mut windows = Windows::over(InputSource::Buffer(b"alpha\nbeta\n".to_vec()));
        assert_eq!(windows.whole(), Some(&b"alpha\nbeta\n"[..]));
        let (window, base) = windows.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        assert_eq!(window, b"alpha\nbeta\n");
        assert_eq!(base, 0);
        let length = window.len();
        assert!(windows.next(CutAfter::LineFeed, length).unwrap().is_none());
    }

    #[test]
    fn takes_a_prefix_from_a_whole_input_and_leaves_the_rest() {
        // `--repeat-header` consumes a header before the body loop starts, and a mapped file
        // must behave like a pipe there: the rest of the input is still to come.
        let mut windows = Windows::over(InputSource::Buffer(b"head\nbody\ntail\n".to_vec()));
        let (first, base) = windows.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        assert_eq!(first, b"head\nbody\ntail\n");
        assert_eq!(base, 0);

        let (rest, base) = windows.next(CutAfter::LineFeed, 5).unwrap().unwrap();
        assert_eq!(rest, b"body\ntail\n");
        assert_eq!(base, 5);

        let length = rest.len();
        assert!(windows.next(CutAfter::LineFeed, length).unwrap().is_none());
    }

    #[test]
    fn ends_a_whole_walk_rather_than_repeating_a_window_it_cannot_widen() {
        // Consuming nothing asks for more; a mapped input has no more, so the walk ends
        // instead of handing back the same bytes forever.
        let mut windows = Windows::over(InputSource::Buffer(b"only\n".to_vec()));
        assert!(windows.next(CutAfter::LineFeed, 0).unwrap().is_some());
        assert!(windows.next(CutAfter::LineFeed, 0).unwrap().is_none());
    }

    #[test]
    fn walks_a_stream_in_windows_that_report_their_offsets() {
        // Every window cuts on a line, and the bases tile the input without a gap or an
        // overlap — which is what lets a caller report absolute positions without counting.
        let data: Vec<u8> = (0..4000)
            .flat_map(|n| format!("line {n}\n").into_bytes())
            .collect();
        let mut windows = Windows(Walk::Stream {
            refill: Refill::new(
                Box::new(io::Cursor::new(data.clone())) as Box<dyn Read>,
                1024,
            ),
            base: 0,
        });

        let mut seen = Vec::new();
        let mut consumed = 0;
        while let Some((window, base)) = windows.next(CutAfter::LineFeed, consumed).unwrap() {
            assert_eq!(base, seen.len(), "windows must tile the input");
            assert!(window.ends_with(b"\n"), "a window must cut on a line");
            seen.extend_from_slice(window);
            consumed = window.len();
        }
        assert_eq!(seen, data);
    }

    #[test]
    fn keeps_a_partial_record_in_view_when_less_than_a_window_is_consumed() {
        // A scanner that must see past what it emits consumes less than it was given; the
        // unconsumed tail has to reappear at the head of the next window, or a match
        // straddling the seam is lost.
        let data: Vec<u8> = (0..2000)
            .flat_map(|n| format!("line {n}\n").into_bytes())
            .collect();
        let mut windows = Windows(Walk::Stream {
            refill: Refill::new(
                Box::new(io::Cursor::new(data.clone())) as Box<dyn Read>,
                512,
            ),
            base: 0,
        });

        let (first, _) = windows.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        let held_back = 20.min(first.len());
        let tail: Vec<u8> = first[first.len() - held_back..].to_vec();
        let consumed = first.len() - held_back;

        let (second, base) = windows.next(CutAfter::LineFeed, consumed).unwrap().unwrap();
        assert!(
            second.starts_with(&tail),
            "the unconsumed tail must lead the next window"
        );
        assert_eq!(
            base, consumed,
            "the base advances by what was consumed, not by what was seen"
        );
    }

    #[test]
    fn grows_a_window_for_a_record_that_never_cuts() {
        let long = vec![b'x'; 5000];
        let mut windows = Windows(Walk::Stream {
            refill: Refill::new(
                Box::new(io::Cursor::new(long.clone())) as Box<dyn Read>,
                256,
            ),
            base: 0,
        });
        let (window, _) = windows.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        assert_eq!(
            window.len(),
            long.len(),
            "an unterminated record grows the window"
        );
    }

    #[test]
    fn every_destination_runs_the_body_and_only_one_keeps_the_bytes() {
        let directory = tempfile::TempDir::new().unwrap();
        let created = directory.path().join("made.txt");
        let replaced = directory.path().join("existing.txt");
        fs::write(&replaced, b"before\n").unwrap();

        // A discarding run still writes, so a count taken inside the body stays honest.
        let mut stdout = Vec::new();
        let counted = Destination::Discard
            .write("sz-test", &mut stdout, |output| {
                output.write_all(b"thrown away\n")?;
                Ok(7)
            })
            .unwrap();
        assert_eq!(counted, 7);
        assert!(stdout.is_empty());

        let mut stdout = Vec::new();
        Destination::Stdout
            .write("sz-test", &mut stdout, |output| {
                output.write_all(b"to stdout\n")
            })
            .unwrap();
        assert_eq!(stdout, b"to stdout\n");

        let mut stdout = Vec::new();
        Destination::Creating(created.to_str().unwrap())
            .write("sz-test", &mut stdout, |output| output.write_all(b"made\n"))
            .unwrap();
        assert_eq!(fs::read(&created).unwrap(), b"made\n");
        assert!(stdout.is_empty(), "a file destination never touches stdout");

        let mut stdout = Vec::new();
        Destination::Replacing(replaced.to_str().unwrap())
            .write("sz-test", &mut stdout, |output| {
                output.write_all(b"after\n")
            })
            .unwrap();
        assert_eq!(fs::read(&replaced).unwrap(), b"after\n");
    }

    #[test]
    fn a_failed_body_leaves_the_previous_file_alone() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("keep.txt");
        fs::write(&path, b"original\n").unwrap();

        let mut stdout = Vec::new();
        let outcome = Destination::Replacing(path.to_str().unwrap()).write::<()>(
            "sz-test",
            &mut stdout,
            |output| {
                output.write_all(b"partial")?;
                Err(io::Error::other("stopped halfway"))
            },
        );
        assert!(outcome.is_err());
        assert_eq!(
            fs::read(&path).unwrap(),
            b"original\n",
            "a run that failed must not have replaced anything"
        );
    }

    #[test]
    fn a_tally_answers_the_exit_code_without_a_counter_on_the_side() {
        assert_eq!(Tally::default().status(), Status::NoResult);

        let mut produced = Tally::default();
        produced.produced();
        assert_eq!(produced.status(), Status::Success);

        // A failure outranks a result: a partial answer is not a whole one.
        let mut partial = Tally::default();
        partial.produced();
        partial.failed();
        assert_eq!(partial.status(), Status::Error);
        assert_eq!(partial.failures(), 1);
    }

    #[test]
    fn reports_a_refused_precondition_as_its_own_code() {
        // `report`'s catch-all would swallow a missing arm and quietly exit 2, which the
        // compiler cannot object to, so the distinction is only ever as real as this test.
        let code = |outcome| format!("{:?}", report("sz-test", outcome));

        let stale = Failure::Stale {
            path: "notes.md".into(),
            expected: 0x91bc_0d2f_5a7e_3c11,
            actual: 0x3f2a_1c88_de10_b4e7,
        };
        let ambiguous = Failure::Ambiguous {
            path: "notes.md".into(),
            subject: "k3f9m2qx".into(),
            matches: 3,
            note: "lengthen the name",
        };
        let unresolved = Failure::Unresolved {
            path: "notes.md".into(),
            subject: "k3f9m2qx".into(),
            note: "re-read the file",
        };
        let expected = format!("{:?}", process::ExitCode::from(REFUSED_EXIT_CODE));

        assert_eq!(code(Err(stale)), expected);
        assert_eq!(code(Err(ambiguous)), expected);
        assert_eq!(code(Err(unresolved)), expected);
        // And the codes it must stay distinct from.
        assert_ne!(code(Ok(Status::NoResult)), expected);
        assert_ne!(
            code(Err(Failure::Io {
                path: "notes.md".into(),
                source: io::Error::other("broken"),
            })),
            expected
        );
    }

    #[test]
    fn names_both_hashes_when_a_precondition_is_refused() {
        // The message is the agent's recovery instruction, so it has to carry what it held
        // and what is there now, at the width `--expect-hash` accepts.
        let message = Failure::Stale {
            path: "notes.md".into(),
            expected: 0x91bc_0d2f_5a7e_3c11,
            actual: 0x3f2a_1c88_de10_b4e7,
        }
        .to_string();

        let mut buffer = [0u8; HASH_CHARS];
        let actual = format_hash(&mut buffer, 0x3f2a_1c88_de10_b4e7, HASH_CHARS).to_string();
        assert!(message.contains("notes.md"), "{message}");
        assert!(
            message.contains(format_hash(&mut buffer, 0x91bc_0d2f_5a7e_3c11, HASH_CHARS)),
            "{message}"
        );
        assert!(message.contains(&actual), "{message}");
        assert!(parse_content_hash(&actual).is_ok());
    }

    #[test]
    fn splits_every_line_where_its_name_was_taken() {
        // The invariant the whole rewrite path rests on: `whole` is what was hashed, it is
        // exactly `body` plus the terminator, and concatenating it over every line
        // reproduces the input — so a rewrite copies terminators rather than rebuilding them.
        for input in [
            &b"alpha\nbeta\ngamma\n"[..],
            &b"alpha\r\nbeta\r\n"[..],
            &b"alpha\nbeta"[..],
            &b"\n\n\n"[..],
            &b""[..],
            "α\nβ\u{2028}γ\n".as_bytes(),
        ] {
            for newlines in [Newlines::Lf, Newlines::Unicode] {
                let mut rebuilt = Vec::new();
                for line in named_lines(input, newlines) {
                    assert_eq!(
                        line.body().len() + line.terminator().len(),
                        line.whole.len()
                    );
                    assert_eq!(line.hash(), content_hash(line.whole));
                    rebuilt.extend_from_slice(line.whole);
                }
                assert_eq!(rebuilt, input, "{input:?} under {newlines:?}");
            }
        }
    }

    #[test]
    fn splits_a_crlf_line_the_same_under_both_newline_sets() {
        // `Newlines::Lf` leaves the carriage return on the line and the Unicode set takes it
        // off, so without the trim the two modes would disagree about which bytes are named
        // — and a name issued by one tool would not resolve in another.
        let input = b"alpha\r\nbeta\r\n";
        let cut = |newlines| {
            named_lines(input, newlines)
                .map(|line| {
                    (
                        line.body().to_vec(),
                        line.terminator().to_vec(),
                        line.hash(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(cut(Newlines::Lf), cut(Newlines::Unicode));
        let first = &cut(Newlines::Lf)[0];
        assert_eq!(
            (first.0.as_slice(), first.1.as_slice()),
            (&b"alpha"[..], &b"\r\n"[..])
        );
    }

    #[test]
    fn leaves_a_final_line_without_a_terminator_without_one() {
        let lines: Vec<_> = named_lines(b"alpha\nbeta", Newlines::Lf).collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].terminator().is_empty());
        assert_eq!(lines[1].end(), 10);
        // The one cost of naming the terminator too: adding a final newline renames the
        // last line, because those genuinely are different bytes.
        let terminated: Vec<_> = named_lines(b"alpha\nbeta\n", Newlines::Lf).collect();
        assert_ne!(lines[1].hash(), terminated[1].hash());
    }

    #[test]
    fn matches_a_name_against_the_hash_it_renders_from() {
        // The mask arithmetic has to agree with the renderer at every width, or a name
        // printed by one tool resolves to nothing in another.
        let mut buffer = [0u8; HASH_CHARS];
        for seed in 0..64u64 {
            let hash = content_hash(&seed.to_le_bytes());
            for width in HASH_CHARS_MIN..=HASH_CHARS {
                let name = format_hash(&mut buffer, hash, width).to_string();
                let prefix = parse_hash_prefix(&name).expect("a rendered name parses");
                assert!(prefix.matches(hash), "`{name}` did not match {hash:#018x}");
                // And it declines a hash that differs inside the bits it pins.
                assert!(!prefix.matches(hash ^ (1 << 63)));
            }
        }
    }

    #[test]
    fn pins_five_bits_for_every_character_of_a_name() {
        // The reason the rendering pads to sixty-five bits. With the spare bit at the front
        // a four-character name pinned nineteen bits, not twenty, and half the alphabet
        // could never open a name; with it at the back every prefix is a clean five per
        // character and only the full width is constrained.
        let mut buffer = [0u8; HASH_CHARS];
        let hash = content_hash(b"    return 0;\n");
        for width in HASH_CHARS_MIN..=HASH_CHARS {
            let name = format_hash(&mut buffer, hash, width).to_string();
            let pinned = parse_hash_prefix(&name)
                .expect("a rendered name parses")
                .mask
                .count_ones() as usize;
            assert_eq!(pinned, (5 * width).min(u64::BITS as usize), "`{name}`");
        }
    }

    #[test]
    fn reads_a_name_in_either_case() {
        let lower = parse_hash_prefix("6vzvxbws").unwrap();
        assert_eq!(parse_hash_prefix("6VZVXBWS").unwrap(), lower);
    }

    #[test]
    fn refuses_a_name_that_could_never_have_been_printed() {
        // Each of these is a different mistake, and the message has to say which.
        for (name, expected) in [
            ("abc", "4 to 13"),
            ("abcdefghijklmn", "4 to 13"),
            ("6vzvxbwi", "not in the alphabet"),
            ("zzzzzzzzzzzzz", "last character"),
        ] {
            let error = parse_hash_prefix(name).expect_err("`{name}` must not parse");
            assert!(
                error.contains(expected),
                "`{name}` reported `{error}`, which does not mention `{expected}`"
            );
        }
    }

    #[test]
    fn renders_a_hash_at_the_width_asked_for() {
        let mut buffer = [0u8; HASH_CHARS];
        assert_eq!(format_hash(&mut buffer, 0, HASH_CHARS), "0000000000000");
        // Every bit set. Twelve characters of five bits each, then the last four against the
        // padding zero — which is why the closing symbol is `y` (30) rather than `z` (31).
        assert_eq!(
            format_hash(&mut buffer, u64::MAX, HASH_CHARS),
            "zzzzzzzzzzzzy"
        );
        for width in 1..=HASH_CHARS {
            assert_eq!(
                format_hash(&mut buffer, 0x0123_4567_89ab_cdef, width).len(),
                width
            );
        }
    }

    #[test]
    fn renders_a_short_hash_as_a_prefix_of_the_long_one() {
        // Truncation, not folding: `--hash-width 4` must agree with the first four
        // characters of the whole rendering, or two tools reading one file disagree.
        let mut buffer = [0u8; HASH_CHARS];
        let whole = format_hash(&mut buffer, 0x0123_4567_89ab_cdef, HASH_CHARS).to_string();
        for width in HASH_CHARS_MIN..=HASH_CHARS {
            assert_eq!(
                format_hash(&mut buffer, 0x0123_4567_89ab_cdef, width),
                &whole[..width]
            );
        }
    }

    #[test]
    fn never_renders_a_character_a_reader_confuses() {
        let mut buffer = [0u8; HASH_CHARS];
        for value in 0..64u64 {
            let rendered = format_hash(
                &mut buffer,
                value.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                HASH_CHARS,
            );
            assert!(
                !rendered.contains(['i', 'l', 'o', 'u']),
                "{rendered} carries a character Crockford's alphabet drops"
            );
        }
    }

    #[test]
    fn names_a_crlf_line_the_same_however_it_was_read() {
        // The name covers the terminator, and a span runs to the start of the next line in
        // either newline set, so both readings name the same bytes with no special case.
        let named = |newlines| {
            named_lines(b"alpha\r\nbeta\r\n", newlines)
                .map(|line| line.hash())
                .collect::<Vec<_>>()
        };
        assert_eq!(named(Newlines::Lf), named(Newlines::Unicode));
    }

    #[test]
    fn names_the_same_content_the_same_wherever_it_sits() {
        assert_eq!(
            content_hash(b"    return 0;\n"),
            content_hash(b"    return 0;\n")
        );
        assert_ne!(
            content_hash(b"    return 0;\n"),
            content_hash(b"    return 1;\n")
        );
        // An empty line is nameable rather than a special case.
        let mut buffer = [0u8; HASH_CHARS];
        assert_eq!(format_hash(&mut buffer, content_hash(b"\n"), 8).len(), 8);
    }

    #[test]
    fn hashes_a_windowed_read_as_one_slice() {
        // The property the whole streamed-hash design rests on: a hash accumulated while
        // windowing has to equal the hash of the bytes those windows came from, at every
        // capacity — including the ones where a cut retains a carry and where a record
        // wider than the window forces it to grow.
        let input = b"alpha\nbeta\n\ngamma delta epsilon zeta\nend";
        for capacity in [1, 2, 7, 13, 64, 4096] {
            let mut refill = Refill::new(&input[..], capacity);
            refill.hash_stream();
            refill
                .for_each_window(CutAfter::LineFeed, |_| Ok(()))
                .unwrap();
            assert_eq!(
                refill.digest(),
                Some(content_hash(input)),
                "capacity {capacity}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "hashed from its first byte")]
    #[cfg(debug_assertions)]
    fn refuses_to_hash_a_stream_it_has_already_read_from() {
        // A digest of a suffix is indistinguishable from a digest of the input, so the one
        // ordering that produces it is a failure rather than a tolerated case.
        let mut refill = Refill::new(&b"alpha\nbeta\n"[..], 8);
        refill.advance(0).unwrap();
        refill.hash_stream();
    }

    #[test]
    fn hashes_only_what_a_stopped_read_took() {
        // An early stop leaves the rest of the input unread, so the digest describes what was
        // read. A caller that needs the whole input has to keep reading, not ask twice.
        let input = b"alpha\nbeta\ngamma\n";
        let mut walk = Windows::streaming(io::Cursor::new(input.to_vec()), 6);
        walk.hash_stream();
        // One window, then the caller stops asking — which is how an early stop is spelled.
        walk.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        let stopped = walk.digest().unwrap();
        assert_ne!(stopped, content_hash(input));
        assert_eq!(stopped, content_hash(b"alpha\n"));
    }

    #[test]
    fn hashes_nothing_when_it_was_not_asked_to() {
        let mut refill = Refill::new(&b"alpha\n"[..], 8);
        refill
            .for_each_window(CutAfter::LineFeed, |_| Ok(()))
            .unwrap();
        assert_eq!(refill.digest(), None);
    }

    #[test]
    fn hashes_a_stream_as_one_slice() {
        // The whole chaining design rests on this: the hash accumulated while writing must
        // equal the hash of the file those writes produced.
        let pieces: [&[u8]; 4] = [b"alpha\n", b"", b"beta\ngamma\n", b"delta"];
        let mut sink = Vec::new();
        let mut hashing = HashingWriter::new(&mut sink);
        for piece in pieces {
            hashing.write_all(piece).unwrap();
        }
        let digest = hashing.digest();

        let joined: Vec<u8> = pieces.concat();
        assert_eq!(digest, content_hash(&joined));
        assert_eq!(sink, joined);
    }

    #[test]
    fn hashes_an_empty_stream_as_an_empty_slice() {
        let mut sink = Vec::new();
        let hashing = HashingWriter::new(&mut sink);
        assert_eq!(hashing.digest(), content_hash(b""));
    }

    #[test]
    fn parses_a_content_hash_token_in_either_case() {
        assert_eq!(
            parse_content_hash("0000006ynpzey").unwrap(),
            parse_content_hash("0000006YNPZEY").unwrap()
        );
        let mut buffer = [0u8; HASH_CHARS];
        let value = content_hash(b"round trip");
        let token = format_hash(&mut buffer, value, HASH_CHARS).to_string();
        assert_eq!(parse_content_hash(&token).unwrap(), value);
    }

    #[test]
    fn refuses_a_content_hash_that_is_not_a_whole_name() {
        // Zero-extending a truncated paste would silently compare against another file, and
        // a line name is a prefix of one — accepting it would edit against the wrong subject.
        for token in [
            "",
            "6ynpzey",
            "0000006ynpzeyy",
            "0000006ynpzei",
            "0000006ynpzez",
        ] {
            assert!(
                parse_content_hash(token).is_err(),
                "`{token}` must not parse as a content hash"
            );
        }
    }

    #[test]
    fn pins_the_hash_of_known_bytes() {
        // Deliberately brittle. Every hash this suite has ever printed is a promise about
        // `sz::hash`, so a dependency bump that changes it must fail the build rather than
        // silently reissue different names for the same lines.
        let mut buffer = [0u8; HASH_CHARS];
        assert_eq!(
            format_hash(&mut buffer, content_hash(b""), HASH_CHARS),
            "0sq616b9mh94c"
        );
        assert_eq!(
            format_hash(
                &mut buffer,
                content_hash(b"the quick brown fox\n"),
                HASH_CHARS
            ),
            "vhk9ejjrdp5sm"
        );
        assert_eq!(
            format_hash(&mut buffer, content_hash(b"    return 0;\n"), HASH_CHARS),
            "wzehkqztd78hg"
        );
    }

    #[test]
    fn writes_a_one_based_line_record() {
        let mut output = Vec::new();
        write_line_record(&mut output, "trex.txt", b"hello", 0, None).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "{\"type\":\"line\",\"data\":{\"path\":{\"text\":\"trex.txt\"},\"text\":{\"text\":\"hello\"},\"line_number\":1}}\n"
        );
    }

    #[test]
    fn names_a_line_record_when_one_was_asked_for() {
        // The same key `sz-find` prints, so a record from either tool reads the same way.
        let mut output = Vec::new();
        write_line_record(&mut output, "trex.txt", b"hello", 0, Some("2xg6k171")).unwrap();
        assert!(String::from_utf8(output)
            .unwrap()
            .contains(r#""line_number":1,"line_hash":"2xg6k171""#));
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
        let piped = InputSource::Pipe(Box::new(io::stdin().lock()));
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
    fn stops_the_run_where_the_caller_stops_asking() {
        // A bounded request stops reading rather than draining the rest of the stream.
        let mut walk = Windows::streaming(io::Cursor::new(b"a\nb\nc\nd\n".to_vec()), 4);
        let (first, base) = walk.next(CutAfter::LineFeed, 0).unwrap().unwrap();
        assert_eq!(first, b"a\nb\n");
        assert_eq!(base, 0);
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
