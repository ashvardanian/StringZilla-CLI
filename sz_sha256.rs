//! SHA256 checksums for many files at once, standing in for `sha256sum`.
//!
//! Each SHA256 block feeds the next, so one file cannot be hashed faster by adding cores or
//! widening a register. A *batch* of files can: AVX-512 advances sixteen independent states per
//! instruction, which measures 2.50 GB/s per core against 1.33 GB/s for a single stream.
//!
//! # Algorithm
//!
//! Two widths, tuned apart. A worker keeps `io_width` files reading at once — wider than the
//! sixteen lanes a call advances — and hashes whichever sixteen have a chunk in hand, taking them
//! off a ready queue rather than waiting on any named file. A file that is slow to read costs only
//! itself. Eight live lanes hash at 1.33 GB/s against sixteen lanes' 2.50, so the extra IO width
//! is what keeps groups full and the cliff out of reach until the queue itself runs dry.
//!
//! ```text
//! slots  A B C D E F G H ...   io_width files, all reading
//!        │ │   │   │   │
//!        ▼ ▼   ▼   ▼   ▼       whichever have a chunk in hand
//!        ready queue ──▶ take 16 ──▶ one multistate step ──▶ release
//! ```
//!
//! # Reading
//!
//! A queue depth of one reads at 0.6 GB/s where a depth of sixteen reads at 8-9 GB/s, so the
//! kernel interface, not the hash, is what a checksummer is limited by. Three fleets sit behind
//! one trait, probed at startup: `io_uring` with `O_DIRECT`, Linux AIO with `O_DIRECT` where the
//! ring is forbidden, and blocking reads everywhere else. The last is the portable floor and the
//! reference the other two are checked against. Nothing in that trait mentions hashing, so a
//! cipher over the same machinery reuses it whole.
//!
//! # Widths and Counts
//!
//! One rule, so a reader never has to work out which kind of number they are holding: a byte
//! count that describes a **file** is `u64`, because a file may be larger than a 32-bit address
//! space and `Metadata::len` reports `u64` on every platform; a byte count that describes
//! **memory** is `usize`, because it is bounded by an allocation this process made; and an
//! **index** — into the input list, the slots, the lanes — is `usize`.
//!
//! Exit: 0 hashed a file, or under `--check` every line verified; 1 nothing to hash, a checksum
//! did not match, or `--strict` met a line that is not a checksum line; 2 could not run, which
//! under `--check` includes a manifest holding no checksum lines at all.

#![deny(unsafe_code)]

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz;

use shared::*;

// region: CLI

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    /// `sha256sum`'s own layout, byte for byte: lowercase digest, two spaces, then the path.
    #[value(alias = "text")]
    Coreutils,
    /// The BSD layout `sha256sum --tag` writes: `SHA256 (path) = digest`.
    Bsd,
    /// Aligned columns, with each file's byte count beside its digest.
    Table,
    /// JSON Lines, one record per file.
    Json,
}

/// Which kernel interface reads the files, declared fastest first and compared as declared.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, ValueEnum)]
enum IoBackend {
    /// The fastest interface this kernel offers, probed at startup.
    Auto,
    /// `io_uring` with `O_DIRECT`, on Linux where the kernel permits it.
    Uring,
    /// Linux AIO with `O_DIRECT`, for kernels that forbid `io_uring`.
    Aio,
    /// Blocking reads through the page cache; portable, and the correctness reference.
    Blocking,
}

/// Compute SHA256 checksums of files and directories
#[derive(Parser)]
#[command(name = "sz-sha256")]
#[command(version, about = "SIMD-accelerated SHA256 checksums", long_about = None)]
struct Args {
    /// Input files or directories (use '-' or omit for stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// Read checksums from the named file and verify them, as `sha256sum --check` does
    #[arg(long, value_name = "FILE")]
    check: Option<PathBuf>,

    /// Under --check, pass over files the list names but the filesystem does not have
    #[arg(long, requires = "check")]
    ignore_missing: bool,

    /// Under --check, fail the run if the list holds a line that is not a checksum line
    #[arg(long, requires = "check")]
    strict: bool,

    /// How records are rendered
    #[arg(
        long,
        value_enum,
        default_value = "coreutils",
        help_heading = "Output Formats"
    )]
    format: Format,

    /// Suppress all output; exit 0 if anything hashed and verified, 1 otherwise
    #[arg(long, conflicts_with_all = ["format", "null", "output"], help_heading = "Output Formats")]
    quiet: bool,

    /// Write to this file instead of stdout
    #[arg(long, value_name = "FILE", help_heading = "Output Formats")]
    output: Option<String>,

    /// Report totals for the whole run on stderr: files, bytes, elapsed, throughput
    #[arg(long, help_heading = "Output Formats")]
    summary: bool,

    /// NUL-terminate each output record instead of newline, for `xargs -0`
    #[arg(long, help_heading = "Output Formats")]
    null: bool,

    /// Total read-buffer budget shared by every worker
    #[arg(
        long,
        value_name = "SIZE",
        value_parser = parse_size,
        default_value = "1Gi",
        help_heading = "Performance"
    )]
    memory: NonZeroUsize,

    /// Worker threads, each keeping sixteen hash lanes busy [default: 4, past which gains stall]
    #[arg(long, value_name = "N", value_parser = parse_at_least_one, help_heading = "Performance")]
    threads: Option<NonZeroUsize>,

    /// Which kernel interface reads the files
    #[arg(long, value_enum, default_value = "auto", help_heading = "Performance")]
    io_backend: IoBackend,

    /// Files with reads in flight per worker; wider than the sixteen lanes a call advances
    #[arg(long, value_name = "N", value_parser = parse_at_least_one, help_heading = "Performance")]
    io_width: Option<NonZeroUsize>,

    /// Blocks read at once per file; a depth of one costs an order of magnitude
    #[arg(long, value_name = "N", value_parser = parse_at_least_one, help_heading = "Performance")]
    io_depth: Option<NonZeroUsize>,

    /// Bytes per read; overrides the size derived from --memory
    #[arg(long, value_name = "SIZE", value_parser = parse_size, help_heading = "Performance")]
    io_chunk: Option<NonZeroUsize>,

    /// Filter inputs by type (e.g., rust, py, js)
    #[arg(long = "type", help_heading = "Traversal")]
    file_type: Option<Vec<String>>,

    /// Filter inputs by glob (e.g., "*.rs")
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
}

fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Every constraint that depends on an argument's value, which clap cannot declare.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.format == Format::Json && args.null {
        return Err(reject("--format json cannot be combined with --null"));
    }
    // Verification reports its own outcome per line, so a record layout has nothing to render.
    if args.check.is_some() && args.format != Format::Coreutils {
        return Err(reject(
            "--check reports its own layout, so --format cannot apply",
        ));
    }
    if let Some(chunk) = args.io_chunk {
        if chunk.get() % DIRECT_IO_ALIGNMENT != 0 {
            return Err(reject(format!(
                "--io-chunk must be a multiple of {}",
                DIRECT_IO_ALIGNMENT
            )));
        }
    }
    Ok(())
}

// endregion: CLI

// region: Multistate Hashing

/// Lanes one `multistate` call advances. Fixed by the instruction set rather than chosen: AVX-512
/// compresses sixteen states at once, and a group of eight measures exactly what a single stream
/// does. Widening past sixteen measured no faster, so this is not a tuning knob.
const LANE_COUNT: usize = 16;

/// Lanes below which the batched kernel is the slower answer.
///
/// A `multistate` call costs the same whether one lane or sixteen carry data, so `k` lanes deliver
/// `2.50 * k / 16` GB/s against a single stream's flat 1.33. The two meet between eight, at 1.25,
/// and nine, at 1.41. Below that the single-stream path wins, and at one live lane it wins by
/// sevenfold — which is the shape a lone large file would otherwise take.
const LANES_WORTH_A_GROUP: usize = 9;

/// Alignment `O_DIRECT` demands of a buffer address, a file offset and a read length. Disks report
/// a 512-byte logical sector and a 4096-byte physical one, so the larger satisfies both.
const DIRECT_IO_ALIGNMENT: usize = 4096;

/// One SHA256 digest.
type Digest = [u8; 32];

/// One file's outcome. Deliberately holds no path: a worker records what it found by the file's
/// input position, and the naming happens once at report time where the path is already at hand.
#[derive(Clone, Copy)]
struct Hashed {
    digest: Digest,
    bytes: u64,
}

/// What became of one file, indexed by its position in the input.
type Outcome = Result<Hashed, io::Error>;

/// Why a slot stopped.
///
/// An errno rather than an `io::Error`, because a worker may not allocate and
/// `io::Error::new(kind, string)` does. The path is attached at report time, which is also the
/// only place `shared::At` can turn it into a `Failure`.
#[derive(Clone, Copy)]
enum Stopped {
    /// The file delivered its last byte.
    Ended,
    /// A read came back with this errno.
    Errno(i32),
    /// A read short of the end returned zero: the file shrank underneath the run.
    Shrank,
}

impl Stopped {
    fn into_error(self) -> Option<io::Error> {
        match self {
            Stopped::Ended => None,
            Stopped::Errno(errno) => Some(io::Error::from_raw_os_error(errno)),
            Stopped::Shrank => Some(io::Error::from(io::ErrorKind::UnexpectedEof)),
        }
    }
}

/// A file waiting to be hashed: where it entered the input, its path, and its size.
///
/// The size rides along because the run already measured every file to order the queue, and a
/// worker that re-measured would `stat` each file a second time for nothing.
type Assignment<'a> = (usize, &'a Path, u64);

// endregion: Multistate Hashing

// region: Fleet and Slots

/// A set of files being read at once, each delivering its chunks in order.
///
/// The width here is IO width — files with reads outstanding — and is deliberately wider than the
/// lanes one hashing call advances. The fleet never dictates which file is served next: it
/// publishes those whose next in-order chunk has landed and lets the caller take them in whatever
/// order it likes. That is the whole difference from a per-round reader, which could report
/// nothing until every one of its files had answered.
///
/// Nothing in this trait mentions hashing, so a cipher over the same machinery reuses it whole.
trait Fleet {
    /// Slots, i.e. files that may have reads outstanding at once.
    fn width(&self) -> usize;

    /// The next slot that may be handed a file: holding none, and with nothing outstanding.
    fn claim(&mut self) -> Option<usize>;

    /// Begin reading `path` into `slot`, of known `size`, and queue its first reads.
    fn open(&mut self, slot: usize, position: usize, path: &Path, size: u64) -> io::Result<()>;

    /// Submit what is queued and collect what has landed, sleeping until `want` reads do. Slots
    /// whose next in-order chunk arrived become ready; slots that ended or failed become retired.
    ///
    /// A `want` of zero submits and takes only what is already there, which is how a caller hands
    /// the disk its next reads without stopping to wait for them.
    fn reap(&mut self, want: usize) -> io::Result<()>;

    /// Take the longest-waiting ready slot. It leaves the ring, so it cannot be taken twice into
    /// one group — which is what keeps a batched call's states distinct.
    fn take(&mut self) -> Option<usize>;

    /// Slots holding a chunk right now.
    fn ready(&self) -> usize;

    /// The chunk `slot` is holding, valid until the matching [`Fleet::release`].
    fn chunk(&self, slot: usize) -> &[u8];

    /// Give back the chunk taken from `slot`, resubmit the block it occupied, and republish the
    /// slot when the chunk behind it is already buffered.
    fn release(&mut self, slot: usize) -> io::Result<()>;

    /// Take a slot whose file ended or failed, for reporting.
    fn take_retired(&mut self) -> Option<usize>;

    /// What `slot` recorded: input position, bytes delivered, and why it stopped.
    fn record(&self, slot: usize) -> (usize, u64, Stopped);

    /// Give `slot` back once its record is written.
    fn recycle(&mut self, slot: usize);

    /// Slots holding a file that still owes bytes.
    fn open_slots(&self) -> usize;

    /// Reads outstanding across every slot, including slots already retired.
    fn outstanding(&self) -> usize;

    /// Where `slot`'s file entered the input, if it still holds one.
    fn live_position(&self, slot: usize) -> Option<usize>;
}

// endregion: Fleet and Slots

// region: Scheduling

/// Hash every file `queue` yields, keeping up to `open_cap` of them reading at once and advancing
/// sixteen of them per call the instant sixteen chunks are in hand.
///
/// The body is a flat ladder in strict priority order: hash a full group if one exists, wait only
/// where waiting can widen the group, otherwise finish what is in hand. Nothing here allocates —
/// every structure is built once per worker and the loop only moves indices between them.
fn hash_with_fleet<'a>(
    fleet: &mut dyn Fleet,
    states: &mut [sz::Sha256],
    staging: &mut [sz::Sha256; LANE_COUNT],
    members: &mut [usize; LANE_COUNT],
    queue: &mut dyn Iterator<Item = Assignment<'a>>,
    open_cap: usize,
    report: &mut dyn FnMut(usize, Outcome),
) {
    let mut exhausted = false;

    loop {
        if !exhausted {
            exhausted = !fill_from(fleet, states, queue, open_cap, report);
        }
        report_retired(fleet, states, report);

        // Every open file either holds its next chunk or has a read outstanding for it, so this is
        // exactly how many files a completion could still add to the group.
        let ready = fleet.ready();
        let awaited = fleet.open_slots() - ready;

        // A full group is the best a call ever gets, so it fires without looking further.
        if ready >= LANE_COUNT {
            if let Err(error) = advance_group(fleet, states, staging, members, LANE_COUNT) {
                abandon(fleet, report, error.kind());
                break;
            }
            // Every release queued a read against a block the group just gave back. This is what
            // hands them over, so the disk fills while the next group hashes rather than waiting
            // for a loop that may not need to sleep for thousands of chunks.
            if let Err(error) = fleet.reap(0) {
                abandon(fleet, report, error.kind());
                break;
            }
            continue;
        }
        // Short group, but a landing read could still widen it. This is not the old barrier: that
        // one waited on one named file's next block while others held data it refused to look at.
        if awaited > 0 {
            // Nothing below a full group can be hashed, so one wakeup per completion would be one
            // syscall per completion. Sleeping for the whole shortfall batches them instead.
            if let Err(error) = fleet.reap(LANE_COUNT - ready) {
                abandon(fleet, report, error.kind());
                break;
            }
            continue;
        }
        // Short group and nothing can widen it, so this group is final. Below nine lanes
        // `advance_group` hashes one stream at a time, which beats a mostly-empty group.
        if ready > 0 {
            if let Err(error) = advance_group(fleet, states, staging, members, ready) {
                abandon(fleet, report, error.kind());
                break;
            }
            continue;
        }
        // No open file and no chunk, but reads are still out against slots whose file already
        // ended. They must land before those slots are reusable.
        if fleet.outstanding() > 0 {
            if let Err(error) = fleet.reap(1) {
                abandon(fleet, report, error.kind());
                break;
            }
            continue;
        }
        if exhausted {
            break;
        }
    }
}

/// Claim files into free slots up to `open_cap`. Reports false once the shared cursor is dry.
fn fill_from<'a>(
    fleet: &mut dyn Fleet,
    states: &mut [sz::Sha256],
    queue: &mut dyn Iterator<Item = Assignment<'a>>,
    open_cap: usize,
    report: &mut dyn FnMut(usize, Outcome),
) -> bool {
    while fleet.open_slots() < open_cap {
        // A slot first, a file second. Reversing the two would pull a file off the shared cursor
        // with nowhere to put it, which is a file no other worker will ever see.
        let Some(slot) = fleet.claim() else {
            return true;
        };
        let Some((position, path, size)) = queue.next() else {
            return false;
        };
        match fleet.open(slot, position, path, size) {
            Ok(()) => states[slot] = sz::Sha256::new(), // reset in place rather than allocate
            // An unreadable file is reported where it stands and the slot takes the next one, so
            // one bad path does not cost a slot for the rest of the run.
            Err(error) => report(position, Err(error)),
        }
    }
    true
}

/// Report and recycle every slot whose file ended or failed.
fn report_retired(
    fleet: &mut dyn Fleet,
    states: &[sz::Sha256],
    report: &mut dyn FnMut(usize, Outcome),
) {
    while let Some(slot) = fleet.take_retired() {
        let (position, bytes, why) = fleet.record(slot);
        let outcome = match why.into_error() {
            // A state advanced through the batched kernel finalizes through the single-state path
            // identically, which is what lets one file retire without draining the group.
            None => Ok(Hashed {
                digest: states[slot].digest(),
                bytes,
            }),
            Some(error) => Err(error),
        };
        report(position, outcome);
        fleet.recycle(slot);
    }
}

/// Advance `count` ready slots by one chunk each, then hand their blocks back.
fn advance_group(
    fleet: &mut dyn Fleet,
    states: &mut [sz::Sha256],
    staging: &mut [sz::Sha256; LANE_COUNT],
    members: &mut [usize; LANE_COUNT],
    count: usize,
) -> io::Result<()> {
    // Taking is what keeps the group's states distinct: a slot sits on the ready ring at most once
    // and cannot rejoin until `release` does. Three fleets implement `ready`, so a ring that comes
    // up short fails the group rather than the process.
    if count > LANE_COUNT {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    for lane in 0..count {
        let Some(slot) = fleet.take() else {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        };
        members[lane] = slot;
        staging[lane] = states[slot];
    }

    {
        let held: &dyn Fleet = fleet;
        let group = &mut staging[..count];
        if count >= LANES_WORTH_A_GROUP {
            if sz::sha256_multistate_update_by(group, |lane| held.chunk(members[lane])).is_err() {
                return Err(io::Error::from(io::ErrorKind::InvalidInput));
            }
        } else {
            // A state advanced either way is bit-identical, so the splice is free to take mid-file.
            for (lane, state) in group.iter_mut().enumerate() {
                state.update(held.chunk(members[lane]));
            }
        }
    }

    for lane in 0..count {
        let slot = members[lane];
        states[slot] = staging[lane];
        fleet.release(slot)?;
    }
    Ok(())
}

/// A failure of the submission interface belongs to no single file, so every file still in hand is
/// charged with it rather than whichever one happens to sit in slot zero.
fn abandon(fleet: &mut dyn Fleet, report: &mut dyn FnMut(usize, Outcome), kind: io::ErrorKind) {
    for slot in 0..fleet.width() {
        if let Some(position) = fleet.live_position(slot) {
            report(position, Err(io::Error::from(kind)));
        }
    }
}

// endregion: Scheduling

// region: Blocking Fleet

/// The portable fleet: one blocking read per slot per reap.
///
/// Compiled on every platform and never feature-gated, so it is the one implementation guaranteed
/// to exist — which makes it the answer the faster interfaces are checked against. Queue depth
/// collapses to one, but the IO width still governs how many files are open, so the ready ring
/// still reaches sixteen and the hashing still runs at full width.
struct BlockingFleet {
    chunk_bytes: usize,
    /// One allocation carved by slot, as the direct fleets carve theirs. The only array here that
    /// is not simply `slots` is this one, because the bytes want to be contiguous and the
    /// bookkeeping does not.
    buffers: Vec<u8>,
    slots: Vec<Slot>,
    ready: Vec<usize>,
    retired: Vec<usize>,
    open_slots: usize,
}

/// A slot of the portable fleet, which needs no block ring because it reads one chunk at a time.
struct Slot {
    file: Option<File>,
    /// Where this file entered the input.
    position: usize,
    size: u64,
    delivered: u64,
    /// Bytes the last read left in this slot's buffer.
    filled: usize,
    state: SlotState,
    published: bool,
}

impl Slot {
    fn free() -> Self {
        Slot {
            file: None,
            position: usize::MAX,
            size: 0,
            delivered: 0,
            filled: 0,
            state: SlotState::Free,
            published: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Open,
    Retired(StoppedTag),
}

/// `Stopped` without its errno, so `SlotState` stays comparable; the errno rides alongside.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StoppedTag {
    Ended,
    Failed(i32),
    Shrank,
}

impl StoppedTag {
    fn widen(self) -> Stopped {
        match self {
            StoppedTag::Ended => Stopped::Ended,
            StoppedTag::Failed(errno) => Stopped::Errno(errno),
            StoppedTag::Shrank => Stopped::Shrank,
        }
    }
}

impl BlockingFleet {
    fn new(chunk_bytes: usize, io_width: usize) -> Self {
        let io_width = io_width.max(1);
        BlockingFleet {
            chunk_bytes,
            buffers: vec![0; chunk_bytes * io_width],
            slots: (0..io_width).map(|_| Slot::free()).collect(),
            ready: Vec::with_capacity(io_width),
            retired: Vec::with_capacity(io_width),
            open_slots: 0,
        }
    }

    fn stop(&mut self, slot: usize, why: StoppedTag) {
        self.slots[slot].state = SlotState::Retired(why);
        self.slots[slot].published = false;
        self.ready.retain(|held| *held != slot);
        self.open_slots -= 1;
        self.retired.push(slot);
    }
}

impl Fleet for BlockingFleet {
    fn width(&self) -> usize {
        self.slots.len()
    }

    fn claim(&mut self) -> Option<usize> {
        (0..self.slots.len()).find(|slot| self.slots[*slot].state == SlotState::Free)
    }

    fn open(&mut self, slot: usize, position: usize, path: &Path, size: u64) -> io::Result<()> {
        let file = File::open(path)?;
        self.slots[slot] = Slot {
            file: Some(file),
            position,
            size,
            state: SlotState::Open,
            ..Slot::free()
        };
        self.open_slots += 1;
        if size == 0 {
            self.stop(slot, StoppedTag::Ended);
        }
        Ok(())
    }

    fn reap(&mut self, want: usize) -> io::Result<()> {
        // Nothing is queued here and nothing lands on its own, so a flush has nothing to give
        // the disk and nothing to collect.
        if want == 0 {
            return Ok(());
        }
        for slot in 0..self.slots.len() {
            if self.slots[slot].state != SlotState::Open || self.slots[slot].published {
                continue;
            }
            let Some(file) = self.slots[slot].file.as_mut() else {
                continue;
            };
            let window = &mut self.buffers[slot * self.chunk_bytes..(slot + 1) * self.chunk_bytes];

            // A short read is normal near end of file, so fill the window rather than treating the
            // first partial answer as the end.
            let mut filled = 0;
            while filled < window.len() {
                match file.read(&mut window[filled..]) {
                    Ok(0) => break,
                    Ok(taken) => filled += taken,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        let errno = error.raw_os_error().unwrap_or(libc::EIO);
                        self.stop(slot, StoppedTag::Failed(errno));
                        break;
                    }
                }
            }
            if self.slots[slot].state != SlotState::Open {
                continue;
            }
            self.slots[slot].filled = filled;
            self.slots[slot].published = true;
            self.ready.push(slot);
        }
        Ok(())
    }

    fn take(&mut self) -> Option<usize> {
        let slot = self.ready.first().copied()?;
        self.ready.remove(0);
        self.slots[slot].published = false;
        Some(slot)
    }

    fn ready(&self) -> usize {
        self.ready.len()
    }

    fn chunk(&self, slot: usize) -> &[u8] {
        let start = slot * self.chunk_bytes;
        &self.buffers[start..start + self.slots[slot].filled]
    }

    fn release(&mut self, slot: usize) -> io::Result<()> {
        let taken = self.slots[slot].filled as u64;
        self.slots[slot].delivered += taken;
        self.slots[slot].filled = 0;
        if taken == 0 && self.slots[slot].delivered < self.slots[slot].size {
            self.stop(slot, StoppedTag::Shrank);
            return Ok(());
        }
        if self.slots[slot].delivered >= self.slots[slot].size {
            self.stop(slot, StoppedTag::Ended);
        }
        Ok(())
    }

    fn take_retired(&mut self) -> Option<usize> {
        self.retired.pop()
    }

    fn record(&self, slot: usize) -> (usize, u64, Stopped) {
        let why = match self.slots[slot].state {
            SlotState::Retired(tag) => tag.widen(),
            _ => Stopped::Ended,
        };
        (self.slots[slot].position, self.slots[slot].delivered, why)
    }

    fn recycle(&mut self, slot: usize) {
        self.slots[slot] = Slot::free();
    }

    fn open_slots(&self) -> usize {
        self.open_slots
    }

    fn outstanding(&self) -> usize {
        0
    }

    fn live_position(&self, slot: usize) -> Option<usize> {
        match self.slots[slot].state {
            SlotState::Open => Some(self.slots[slot].position),
            _ => None,
        }
    }
}

// endregion: Blocking Fleet

// region: Linux Kernel Readers

/// The asynchronous readers, and the only part of the program that touches raw memory.
///
/// The lint above the file is the containment: `unsafe` is permitted here and nowhere else, so a
/// pointer that escapes into the scheduler fails the build rather than a review. Both interfaces
/// share the arena, the lane bookkeeping and the tail handling; they differ only in how a read is
/// handed to the kernel and how a finished one is collected.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod direct {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::{AsRawFd, RawFd};
    use std::path::Path;

    use super::{Fleet, Stopped, DIRECT_IO_ALIGNMENT};

    /// Reads in flight per lane. One buffer feeds the hasher while the rest fill, which is the
    /// whole reason these interfaces are here: a depth of one reads at 0.6 GB/s where four reach
    /// 8-9.
    pub const MAX_DEPTH: usize = 4;

    /// Below a megabyte `O_DIRECT` is a loss: it turns off readahead and the page cache, which is
    /// exactly the size at which both were about to pay for themselves.
    const DIRECT_IO_WORTH_IT: u64 = 1 << 20;

    /// How a batch of reads reaches the kernel and comes back.
    ///
    /// The two implementations are `io_uring` and the older AIO interface. Everything above this
    /// trait — the arena, the block ring, alignment, the tail block — is shared.
    pub trait Submitter {
        /// Offer one read. Reports false when the queue is full, which asks the caller to stop
        /// filling and submit what it has.
        fn queue(
            &mut self,
            index: usize,
            descriptor: RawFd,
            offset: u64,
            length: usize,
            buffer: *mut u8,
        ) -> io::Result<bool>;

        /// Offer the arena for pinning, once, before any read is queued.
        ///
        /// `io_uring` can register the buffers up front and skip pinning each one per read;
        /// interfaces that cannot simply ignore this. The arena outlives the submitter, so the
        /// range stays valid for as long as the registration does.
        fn register(&mut self, _base: *mut u8, _length: usize) {}

        /// Hand every queued read to the kernel and collect what has finished, appending
        /// `(block index, result)` to `landed`, and sleeping until `want` of them land. A `want`
        /// past what is outstanding is capped rather than refused, so no caller can wait forever.
        fn reap(&mut self, want: usize, landed: &mut Vec<(usize, i32)>) -> io::Result<()>;
    }

    /// Retry a kernel call that a signal interrupted.
    ///
    /// Both `io_uring_enter` and `io_getevents` return `EINTR` when a signal lands while they are
    /// waiting, and that is a resumption request rather than a failure. Treating it as an error
    /// would abandon whichever file the lane was reading — a run that a stray `SIGCHLD` could
    /// spoil, which is exactly the sort of thing a checksummer may never do.
    fn resuming<T>(mut call: impl FnMut() -> io::Result<T>) -> io::Result<T> {
        loop {
            match call() {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                other => return other,
            }
        }
    }

    /// What one block of one slot's ring is doing.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Block {
        Free,
        InFlight,
        /// Bytes the kernel wrote. The tail block asks for a page-rounded length, so this may
        /// reach past the end of the file and is clamped against the file's size when handed out.
        Ready(u32),
    }

    /// How a read names the block it fills.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Pinning {
        /// The arena is pinned with the kernel, so a read names it by index rather than by
        /// address and skips the pinning an ordinary read repeats.
        Pinned,
        /// Each read hands the kernel an address to pin for the duration of that read.
        Unpinned,
    }

    /// What a slot is doing, which is what decides whether it may be handed a file.
    #[derive(Clone, Copy)]
    enum SlotState {
        Free,
        Open,
        Retired(Stopped),
    }

    /// One file being streamed through its own ring of blocks.
    ///
    /// Ordering is the ring, not the completion order: reads are submitted at `next_submit` and
    /// handed onward at `next_take`, both stepping by one. The kernel may finish block three
    /// before block two, and block three simply waits — it occupies a block no other file can
    /// use, so an out-of-order completion costs nothing but its own buffer.
    struct Slot {
        /// Held here rather than in an array beside this one, and dropped only once the slot's
        /// last read has landed: an AIO control block carries a bare descriptor number, so closing
        /// early and having the number reused would read the wrong file into the buffer.
        file: Option<File>,
        /// Where this file entered the input, or `usize::MAX` when the slot holds none.
        position: usize,
        size: u64,
        /// Next byte to ask the disk for, block-aligned except past the end of the file.
        submit_at: u64,
        /// Bytes handed onward, which is what decides when the file is finished.
        delivered: u64,
        blocks: [Block; MAX_DEPTH],
        next_submit: u8,
        next_take: u8,
        /// Reads submitted and not yet reaped, whatever the slot's state. A slot is claimable
        /// only at zero, which is the whole of the retire-refill race.
        outstanding: u8,
        state: SlotState,
        /// True exactly while this slot's index sits on the ready ring.
        published: bool,
        /// True between `take` and `release`, while a caller holds a borrow of its chunk.
        borrowed: bool,
        direct: bool,
    }

    impl Slot {
        fn free() -> Self {
            Slot {
                file: None,
                position: usize::MAX,
                size: 0,
                submit_at: 0,
                delivered: 0,
                blocks: [Block::Free; MAX_DEPTH],
                next_submit: 0,
                next_take: 0,
                outstanding: 0,
                state: SlotState::Free,
                published: false,
                borrowed: false,
                direct: false,
            }
        }
    }

    /// A fixed-capacity queue of slot indices, allocated once and never grown.
    ///
    /// First-in-first-out rather than a stack, so a slot that has been waiting is not passed over
    /// by one that just landed — which is what stops a wide fleet starving its own tail.
    struct SlotRing {
        entries: Box<[u16]>,
        head: usize,
        length: usize,
    }

    impl SlotRing {
        fn with_capacity(capacity: usize) -> Self {
            SlotRing {
                entries: vec![0u16; capacity.max(1)].into_boxed_slice(),
                head: 0,
                length: 0,
            }
        }

        fn push(&mut self, slot: usize) {
            debug_assert!(self.length < self.entries.len());
            let at = (self.head + self.length) % self.entries.len();
            self.entries[at] = slot as u16;
            self.length += 1;
        }

        fn pop(&mut self) -> Option<usize> {
            if self.length == 0 {
                return None;
            }
            let slot = self.entries[self.head] as usize;
            self.head = (self.head + 1) % self.entries.len();
            self.length -= 1;
            Some(slot)
        }
    }

    /// Streams `io_width` files at once through one submission interface.
    ///
    /// The width here is IO width — files with reads outstanding — and is deliberately wider than
    /// the lanes one hashing call advances. The fleet never dictates which file is served next: it
    /// publishes those whose next in-order chunk has landed and lets the caller take them in any
    /// order. That is the whole difference from a per-round reader, which could report nothing
    /// until every one of its files had answered.
    pub struct Streamer<S: Submitter> {
        submitter: S,
        arena: BufferArena,
        slots: Box<[Slot]>,
        chunk_bytes: usize,
        depth: usize,
        open_slots: usize,
        outstanding: usize,
        ready: SlotRing,
        /// Entries on `ready` whose slot has since failed. Counted rather than removed, because a
        /// queue cannot cheaply drop from the middle; `take` skips them.
        stale: usize,
        retired: SlotRing,
        /// Where the next `claim` starts scanning, so refilling a wide fleet stays amortized.
        rotor: usize,
        landed: Vec<(usize, i32)>,
    }

    impl<S: Submitter> Streamer<S> {
        pub fn new(
            submitter: S,
            chunk_bytes: usize,
            io_width: usize,
            depth: usize,
        ) -> io::Result<Self> {
            let depth = depth.clamp(1, MAX_DEPTH);
            let io_width = io_width.max(1);
            let blocks = io_width * depth;
            let arena = BufferArena::new(blocks, chunk_bytes)?;
            let mut submitter = submitter;
            submitter.register(arena.base_ptr(), blocks * chunk_bytes);
            Ok(Streamer {
                submitter,
                arena,
                slots: (0..io_width)
                    .map(|_| Slot::free())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                chunk_bytes,
                depth,
                open_slots: 0,
                outstanding: 0,
                ready: SlotRing::with_capacity(io_width),
                stale: 0,
                retired: SlotRing::with_capacity(io_width),
                rotor: 0,
                landed: Vec::with_capacity(blocks),
            })
        }

        fn block_index(&self, slot: usize, ring: usize) -> usize {
            slot * self.depth + ring
        }

        /// Queue every read this slot has room for, which is what holds the queue depth up.
        fn submit_reads(&mut self, slot: usize) -> io::Result<()> {
            while self.slots[slot].submit_at < self.slots[slot].size {
                let ring = self.slots[slot].next_submit as usize;
                if self.slots[slot].blocks[ring] != Block::Free {
                    break;
                }
                let remaining = self.slots[slot].size - self.slots[slot].submit_at;
                // The tail block asks for a page-aligned length that may reach past the end of the
                // file, and lets the kernel's short count say where the file really stopped.
                // `O_DIRECT` constrains the length asked for, never the length returned.
                let want = if self.slots[slot].direct {
                    (remaining as usize)
                        .min(self.chunk_bytes)
                        .next_multiple_of(DIRECT_IO_ALIGNMENT)
                        .min(self.chunk_bytes)
                } else {
                    (remaining as usize).min(self.chunk_bytes)
                };

                let Some(descriptor) = self.slots[slot].file.as_ref().map(|file| file.as_raw_fd())
                else {
                    break;
                };
                let index = self.block_index(slot, ring);
                let buffer = self.arena.write_ptr(index);
                if !self.submitter.queue(
                    index,
                    descriptor,
                    self.slots[slot].submit_at,
                    want,
                    buffer,
                )? {
                    break;
                }
                self.slots[slot].blocks[ring] = Block::InFlight;
                self.slots[slot].next_submit = ((ring + 1) % self.depth) as u8;
                self.slots[slot].outstanding += 1;
                self.outstanding += 1;
                self.slots[slot].submit_at += (want as u64).min(remaining);
            }
            Ok(())
        }

        /// Publish `slot` if — and only if — its next in-order chunk is buffered, it is not already
        /// published, and nobody is holding its previous one.
        ///
        /// Called from the only two moments at which the head block can become ready: a completion
        /// filed against the slot, and the end of `release`. That is the whole publication rule,
        /// and it is what keeps a slot off the ring twice.
        fn publish(&mut self, slot: usize) {
            let entry = &mut self.slots[slot];
            if entry.published || entry.borrowed || !matches!(entry.state, SlotState::Open) {
                return;
            }
            if !matches!(entry.blocks[entry.next_take as usize], Block::Ready(_)) {
                return;
            }
            entry.published = true;
            self.ready.push(slot);
        }

        /// End `slot`, whether the file ran out or a read failed.
        ///
        /// The descriptor and any outstanding reads are deliberately left alone: they must land
        /// before the slot is reusable, and cancelling them would cost more than letting them.
        fn stop(&mut self, slot: usize, why: Stopped) {
            debug_assert!(matches!(self.slots[slot].state, SlotState::Open));
            self.slots[slot].state = SlotState::Retired(why);
            // Tombstone a chunk already on the ring: it belongs to a file about to be reported,
            // so hashing it would spend a lane on a digest nobody reads.
            if self.slots[slot].published {
                self.slots[slot].published = false;
                self.stale += 1;
            }
            self.open_slots -= 1;
            self.retired.push(slot);
        }
    }

    impl<S: Submitter> Fleet for Streamer<S> {
        fn width(&self) -> usize {
            self.slots.len()
        }

        fn claim(&mut self) -> Option<usize> {
            for step in 0..self.slots.len() {
                let slot = (self.rotor + step) % self.slots.len();
                // Free is not enough: a slot whose file ended may still have reads in the kernel,
                // and handing it a new file would let a stale completion land in that file's
                // buffer. Silently, as a wrong digest rather than an error.
                if matches!(self.slots[slot].state, SlotState::Free)
                    && self.slots[slot].outstanding == 0
                {
                    self.rotor = (slot + 1) % self.slots.len();
                    return Some(slot);
                }
            }
            None
        }

        fn open(&mut self, slot: usize, position: usize, path: &Path, size: u64) -> io::Result<()> {
            // Below a megabyte `O_DIRECT` is a loss: it turns off readahead and the page cache,
            // which is exactly the size at which both were about to pay for themselves.
            let mut opened = None;
            if size >= DIRECT_IO_WORTH_IT {
                match OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(path)
                {
                    Ok(file) => opened = Some((file, true)),
                    // tmpfs, some network filesystems and compressed btrfs refuse `O_DIRECT`
                    // outright. The buffered path reads the same bytes, so this is a flag rather
                    // than a failure.
                    Err(error) if error.raw_os_error() == Some(libc::EINVAL) => {}
                    Err(error) => return Err(error),
                }
            }
            let (file, direct) = match opened {
                Some(pair) => pair,
                None => (File::open(path)?, false),
            };

            self.slots[slot] = Slot {
                file: Some(file),
                position,
                size,
                direct,
                state: SlotState::Open,
                ..Slot::free()
            };
            self.open_slots += 1;

            // An empty file owes no bytes, so it retires here rather than occupying a lane to be
            // handed a zero-length chunk.
            if size == 0 {
                self.stop(slot, Stopped::Ended);
                return Ok(());
            }
            self.submit_reads(slot)
        }

        fn reap(&mut self, want: usize) -> io::Result<()> {
            self.landed.clear();
            // Only a read still outstanding can land, so this is what stops a wait outliving the
            // reads it is waiting on.
            self.submitter
                .reap(want.min(self.outstanding), &mut self.landed)?;

            // Indexed rather than drained, so the vector keeps the capacity it was built with.
            for entry in 0..self.landed.len() {
                let (index, result) = self.landed[entry];
                let slot = index / self.depth;
                let ring = index % self.depth;
                self.slots[slot].outstanding -= 1;
                self.outstanding -= 1;

                match self.slots[slot].state {
                    // The tail of a read that was in flight when the file ended or failed. Its
                    // block is freed and nothing else happens: the slot holds no other file to
                    // confuse it with, and takes none until this count reaches zero.
                    SlotState::Retired(_) | SlotState::Free => {
                        self.slots[slot].blocks[ring] = Block::Free;
                        if self.slots[slot].outstanding == 0
                            && matches!(self.slots[slot].state, SlotState::Free)
                        {
                            self.slots[slot].file = None;
                        }
                    }
                    SlotState::Open if result < 0 => {
                        self.slots[slot].blocks[ring] = Block::Free;
                        // Charged to the one file it happened to; the others keep their states,
                        // since slots share nothing but the call that advances them.
                        self.stop(slot, Stopped::Errno(-result));
                    }
                    SlotState::Open => {
                        self.slots[slot].blocks[ring] = Block::Ready(result as u32);
                        self.publish(slot);
                    }
                }
            }
            Ok(())
        }

        fn take(&mut self) -> Option<usize> {
            loop {
                let slot = self.ready.pop()?;
                // A tombstone: published, then failed on a deeper block. Its record is already
                // queued for reporting, so its chunk is dropped rather than served.
                if !self.slots[slot].published {
                    self.stale -= 1;
                    continue;
                }
                self.slots[slot].published = false;
                self.slots[slot].borrowed = true;
                return Some(slot);
            }
        }

        fn ready(&self) -> usize {
            self.ready.length - self.stale
        }

        fn chunk(&self, slot: usize) -> &[u8] {
            let entry = &self.slots[slot];
            let Block::Ready(filled) = entry.blocks[entry.next_take as usize] else {
                return &[];
            };
            // The tail block asked for a page-rounded length that may reach past the end of the
            // file, so the file's own size caps the slice.
            let usable = (filled as u64).min(entry.size - entry.delivered) as usize;
            self.arena
                .chunk(self.block_index(slot, entry.next_take as usize), usable)
        }

        fn release(&mut self, slot: usize) -> io::Result<()> {
            let taken = self.chunk(slot).len() as u64;
            let ring = self.slots[slot].next_take as usize;
            {
                let entry = &mut self.slots[slot];
                entry.borrowed = false;
                entry.blocks[ring] = Block::Free;
                entry.next_take = ((ring + 1) % self.depth) as u8;
                entry.delivered += taken;
            }

            // A zero-length read short of the end means the file shrank underneath the run, and a
            // digest of a file that is no longer there would be a lie.
            if taken == 0 && self.slots[slot].delivered < self.slots[slot].size {
                self.stop(slot, Stopped::Shrank);
                return Ok(());
            }
            if self.slots[slot].delivered >= self.slots[slot].size {
                self.stop(slot, Stopped::Ended);
                return Ok(());
            }
            self.submit_reads(slot)?; // the freed block goes straight back to the disk
            self.publish(slot); // the chunk behind it may already be here
            Ok(())
        }

        fn take_retired(&mut self) -> Option<usize> {
            self.retired.pop()
        }

        fn record(&self, slot: usize) -> (usize, u64, Stopped) {
            let why = match self.slots[slot].state {
                SlotState::Retired(why) => why,
                _ => Stopped::Ended,
            };
            (self.slots[slot].position, self.slots[slot].delivered, why)
        }

        fn recycle(&mut self, slot: usize) {
            self.slots[slot].state = SlotState::Free;
            self.slots[slot].position = usize::MAX;
            if self.slots[slot].outstanding == 0 {
                self.slots[slot].file = None;
            }
        }

        fn open_slots(&self) -> usize {
            self.open_slots
        }

        fn outstanding(&self) -> usize {
            self.outstanding
        }

        fn live_position(&self, slot: usize) -> Option<usize> {
            match self.slots[slot].state {
                SlotState::Open => Some(self.slots[slot].position),
                _ => None,
            }
        }
    }
    /// An allocator that rounds every request up to a page boundary.
    ///
    /// `O_DIRECT` rejects a buffer that is not sector-aligned, and a page satisfies every sector
    /// size in use. Expressing it as an allocator rather than a hand-rolled `alloc`/`dealloc` pair
    /// is what lets the arena be an ordinary vector, with an ordinary `Drop`.
    #[derive(Clone, Copy)]
    struct PageAligned;

    // SAFETY: every request is forwarded to the global allocator under a layout that only widens
    // the alignment, and the same widening is applied when the block is handed back.
    unsafe impl allocator_api2::alloc::Allocator for PageAligned {
        fn allocate(
            &self,
            layout: std::alloc::Layout,
        ) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
            let paged = layout
                .align_to(DIRECT_IO_ALIGNMENT)
                .map_err(|_| allocator_api2::alloc::AllocError)?;
            // Zeroed rather than raw, so the pages are pre-faulted and the first read into a block
            // does not take a minor fault inside the hot loop.
            allocator_api2::alloc::Global.allocate_zeroed(paged)
        }

        unsafe fn deallocate(&self, pointer: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
            let paged = layout
                .align_to(DIRECT_IO_ALIGNMENT)
                .expect("the widening that allocated this cannot fail to repeat");
            unsafe { allocator_api2::alloc::Global.deallocate(pointer, paged) }
        }
    }

    /// One page-aligned allocation carved into equal blocks — the only heap a worker's reads ever
    /// touch, and an ordinary vector rather than a raw pointer and a hand-written `Drop`.
    pub struct BufferArena {
        storage: allocator_api2::vec::Vec<u8, PageAligned>,
        block: usize,
    }

    impl BufferArena {
        fn new(blocks: usize, block: usize) -> io::Result<Self> {
            let bytes = blocks
                .checked_mul(block)
                .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
            let mut storage = allocator_api2::vec::Vec::new_in(PageAligned);
            storage
                .try_reserve_exact(bytes)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
            storage.resize(bytes, 0);
            Ok(BufferArena { storage, block })
        }

        /// The whole allocation, for a submitter that pins it once rather than per read.
        fn base_ptr(&self) -> *mut u8 {
            self.storage.as_ptr() as *mut u8
        }

        /// Where a read writes. Raw, because only a submitter hands it to the kernel — but taken
        /// from a bounds-checked slice rather than computed against the base.
        fn write_ptr(&self, index: usize) -> *mut u8 {
            let start = index * self.block;
            self.storage[start..start + self.block].as_ptr() as *mut u8
        }

        /// The bytes a finished read left behind. Plain slicing, because the arena is a vector:
        /// the bound check is the proof, and no pointer arithmetic is involved.
        fn chunk(&self, index: usize, length: usize) -> &[u8] {
            debug_assert!(length <= self.block);
            let start = index * self.block;
            &self.storage[start..start + length]
        }
    }

    /// Submits through `io_uring`, the interface every kernel since 5.1 offers.
    pub struct UringSubmitter {
        ring: io_uring::IoUring,
        queued: usize,
        in_flight: usize,
        /// How a read names the block it fills, settled once when the arena is offered.
        arena: Pinning,
    }

    impl UringSubmitter {
        /// Build a submitter, or report why this kernel cannot supply one.
        ///
        /// Probing by attempting the real thing is deliberate: `io_uring_setup` is blocked by
        /// Docker's default seccomp profile and by `/proc/sys/kernel/io_uring_disabled`, and both
        /// answer with an error rather than with a capability bit.
        pub fn probe(io_width: usize, depth: usize) -> io::Result<Self> {
            let entries = (io_width.max(1) * depth.clamp(1, MAX_DEPTH))
                .next_power_of_two()
                .max(64) as u32;
            // `single_issuer` lets the kernel skip the locking a shared ring needs, and
            // `coop_taskrun` stops it interrupting this thread to deliver completions the loop is
            // about to poll for anyway. Kernels before 6.0 refuse both, and a ring without them is
            // the same ring, so a refusal falls back rather than failing.
            //
            // `IORING_SETUP_IOPOLL` is deliberately absent, and not for the reason it usually is:
            // it costs no kernel thread, unlike `SQPOLL`. It is wrong here on three counts. It is
            // a ring-level promise that every descriptor was opened `O_DIRECT`, which this ring
            // cannot make — files below `DIRECT_IO_WORTH_IT` open buffered by design, as does any
            // filesystem that refuses the flag. It forces `IORING_ENTER_GETEVENTS` onto every
            // submission, which is exactly the non-blocking flush the scheduler relies on to keep
            // the disk fed while a group hashes. And it busy-polls in this thread, spending the
            // cycles that are the product: a core hashes 2.15 GB/s only while it is hashing.
            debug_assert!(
                entries as usize >= io_width.max(1) * depth.clamp(1, MAX_DEPTH),
                "the queue must hold one entry per arena block: a refused push leaves a slot \
                 with no read outstanding and no ready chunk, and nothing revisits it"
            );
            let ring = io_uring::IoUring::builder()
                .setup_single_issuer()
                .setup_coop_taskrun()
                .setup_submit_all()
                .build(entries)
                .or_else(|_| io_uring::IoUring::new(entries))?;
            Ok(UringSubmitter {
                ring,
                queued: 0,
                in_flight: 0,
                arena: Pinning::Unpinned,
            })
        }
    }

    impl Submitter for UringSubmitter {
        fn register(&mut self, base: *mut u8, length: usize) {
            let region = [libc::iovec {
                iov_base: base.cast(),
                iov_len: length,
            }];
            // SAFETY: the arena outlives this submitter, so the pinned range stays valid for as
            // long as the registration does.
            // A refusal is a slower answer rather than a failure: the kernel caps one registered
            // buffer at a gigabyte, so a wide arena simply stays unpinned.
            self.arena = match unsafe { self.ring.submitter().register_buffers(&region) } {
                Ok(()) => Pinning::Pinned,
                Err(_) => Pinning::Unpinned,
            };
        }

        fn queue(
            &mut self,
            index: usize,
            descriptor: RawFd,
            offset: u64,
            length: usize,
            buffer: *mut u8,
        ) -> io::Result<bool> {
            // One registered region covers the whole arena, so every block is an interior pointer
            // into buffer zero — a registration the reads need not start at.
            let file = io_uring::types::Fd(descriptor);
            let entry = match self.arena {
                Pinning::Pinned => io_uring::opcode::ReadFixed::new(file, buffer, length as u32, 0)
                    .offset(offset)
                    .build(),
                Pinning::Unpinned => io_uring::opcode::Read::new(file, buffer, length as u32)
                    .offset(offset)
                    .build(),
            }
            .user_data(index as u64);

            // SAFETY: the buffer is one block of an arena that outlives the ring, the length is at
            // most one block, and nothing else names the block until its completion lands.
            if unsafe { self.ring.submission().push(&entry) }.is_err() {
                return Ok(false);
            }
            self.queued += 1;
            Ok(true)
        }

        fn reap(&mut self, want: usize, landed: &mut Vec<(usize, i32)>) -> io::Result<()> {
            // One `io_uring_enter` that submits what is queued and waits for as many completions
            // as were asked for. Asking for none when nothing can arrive keeps it off an empty
            // ring; asking for more than is pending would never return.
            let pending = self.in_flight + self.queued;
            let want = want.min(pending);
            if self.queued > 0 || want > 0 {
                match resuming(|| self.ring.submit_and_wait(want)) {
                    Ok(taken) => {
                        let taken = taken.min(self.queued);
                        self.in_flight += taken;
                        self.queued -= taken;
                    }
                    // `EBUSY` says the completion queue is backlogged and nothing was taken. The
                    // entries stay queued, and the drain below is what makes room for them.
                    Err(error) if error.raw_os_error() == Some(libc::EBUSY) => {}
                    Err(error) => return Err(error),
                }
            }
            let mut completion = self.ring.completion();
            completion.sync();
            for entry in &mut completion {
                landed.push((entry.user_data() as usize, entry.result()));
                self.in_flight -= 1;
            }
            Ok(())
        }
    }

    /// The control block one AIO read is described by.
    ///
    /// Declared here because `libc` ships neither this nor `io_event`, and glibc wraps none of the
    /// four entry points. The field order is the kernel's `struct iocb` from `linux/aio_abi.h` and
    /// must match it exactly — a misplaced field would corrupt silently rather than fail, which is
    /// what the cross-backend digest test defends against.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Iocb {
        data: u64,
        key: u32,
        read_write_flags: i32,
        opcode: u16,
        request_priority: i16,
        descriptor: u32,
        buffer: u64,
        length: u64,
        offset: i64,
        reserved: u64,
        flags: u32,
        result_descriptor: u32,
    }

    /// One finished AIO read.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct IoEvent {
        data: u64,
        object: u64,
        result: i64,
        result2: i64,
    }

    /// `IOCB_CMD_PREAD`, the only command this reader ever issues.
    const IOCB_CMD_PREAD: u16 = 0;

    /// Submits through the older AIO interface, for kernels where `io_uring` is unavailable.
    ///
    /// AIO is only genuinely asynchronous against `O_DIRECT` — a buffered descriptor makes
    /// `io_submit` block until the read completes — which suits a reader that is already direct
    /// and aligned everywhere it can be.
    pub struct AioSubmitter {
        context: u64,
        /// One control block per arena block, reused rather than rebuilt, so submitting allocates
        /// nothing. `pointers` holds the batch handed to the next `io_submit`.
        blocks: Vec<Iocb>,
        pointers: Vec<*mut Iocb>,
        events: Vec<IoEvent>,
        in_flight: usize,
    }

    impl AioSubmitter {
        pub fn probe(io_width: usize, depth: usize) -> io::Result<Self> {
            let capacity = io_width.max(1) * depth.clamp(1, MAX_DEPTH);
            let mut context: u64 = 0;
            // SAFETY: `context` is the zero-initialised handle the call is specified to fill.
            let status = unsafe {
                libc::syscall(libc::SYS_io_setup, capacity as libc::c_long, &mut context)
            };
            if status < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(AioSubmitter {
                context,
                blocks: vec![Iocb::default(); capacity],
                pointers: Vec::with_capacity(capacity),
                events: vec![IoEvent::default(); capacity],
                in_flight: 0,
            })
        }
    }

    impl AioSubmitter {
        /// Submit what is queued. AIO keeps these separate at the kernel boundary, so the two
        /// calls remain two here — but the trait only ever asks once.
        fn submit_queued(&mut self) -> io::Result<()> {
            while !self.pointers.is_empty() {
                // SAFETY: every pointer names a live control block owned by `self`, and the count
                // matches the slice handed over.
                let accepted = unsafe {
                    libc::syscall(
                        libc::SYS_io_submit,
                        self.context,
                        self.pointers.len() as libc::c_long,
                        self.pointers.as_mut_ptr(),
                    )
                };
                if accepted < 0 {
                    let error = io::Error::last_os_error();
                    if error.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    // `EAGAIN` means the context is full rather than broken, so what was accepted
                    // stands and the rest waits for a completion to free a slot.
                    if error.raw_os_error() == Some(libc::EAGAIN) && self.in_flight > 0 {
                        return Ok(());
                    }
                    self.pointers.clear();
                    return Err(error);
                }
                let accepted = accepted as usize;
                self.in_flight += accepted;
                self.pointers.drain(..accepted);
            }
            Ok(())
        }
    }

    impl Submitter for AioSubmitter {
        fn queue(
            &mut self,
            index: usize,
            descriptor: RawFd,
            offset: u64,
            length: usize,
            buffer: *mut u8,
        ) -> io::Result<bool> {
            if self.pointers.len() == self.pointers.capacity() {
                return Ok(false);
            }
            self.blocks[index] = Iocb {
                data: index as u64,
                opcode: IOCB_CMD_PREAD,
                descriptor: descriptor as u32,
                buffer: buffer as u64,
                length: length as u64,
                offset: offset as i64,
                ..Iocb::default()
            };
            self.pointers.push(&mut self.blocks[index] as *mut Iocb);
            Ok(true)
        }

        fn reap(&mut self, want: usize, landed: &mut Vec<(usize, i32)>) -> io::Result<()> {
            self.submit_queued()?;
            if self.in_flight == 0 {
                return Ok(());
            }
            let minimum = want.min(self.in_flight);
            let collected = resuming(|| {
                // SAFETY: `events` has room for its own length and the kernel fills at most that.
                let collected = unsafe {
                    libc::syscall(
                        libc::SYS_io_getevents,
                        self.context,
                        minimum as libc::c_long,
                        self.events.len() as libc::c_long,
                        self.events.as_mut_ptr(),
                        std::ptr::null_mut::<libc::timespec>(),
                    )
                };
                if collected < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(collected)
            })?;
            for event in &self.events[..collected as usize] {
                landed.push((event.data as usize, event.result as i32));
            }
            self.in_flight -= collected as usize;
            Ok(())
        }
    }

    impl Drop for AioSubmitter {
        fn drop(&mut self) {
            // SAFETY: `context` came from the matching `io_setup` and is destroyed once.
            unsafe { libc::syscall(libc::SYS_io_destroy, self.context) };
        }
    }
}

/// Build the fastest fleet this build and this kernel can supply.
///
/// Selection is a runtime probe rather than a compile-time choice, because a Linux binary that
/// assumes the ring exists fails inside a container that forbids it.
fn select_fleet(plan: &HashPlan) -> Box<dyn Fleet> {
    #[cfg(target_os = "linux")]
    {
        if plan.allows(IoBackend::Uring) {
            let built =
                direct::UringSubmitter::probe(plan.io_width, plan.depth).and_then(|submitter| {
                    direct::Streamer::new(submitter, plan.chunk_bytes, plan.io_width, plan.depth)
                });
            if let Ok(fleet) = built {
                return Box::new(fleet);
            }
        }
        if plan.allows(IoBackend::Aio) {
            let built =
                direct::AioSubmitter::probe(plan.io_width, plan.depth).and_then(|submitter| {
                    direct::Streamer::new(submitter, plan.chunk_bytes, plan.io_width, plan.depth)
                });
            if let Ok(fleet) = built {
                return Box::new(fleet);
            }
        }
    }
    Box::new(BlockingFleet::new(plan.chunk_bytes, plan.io_width))
}

// endregion: Linux Kernel Readers

// region: Workers

/// Everything the fast path is tuned by, lifted out of [`Args`] so hashing can be driven without
/// knowing what a CLI is.
#[derive(Clone, Copy)]
struct HashPlan {
    backend: IoBackend,
    chunk_bytes: usize,
    threads: usize,
    /// Files a worker keeps reads outstanding against. Deliberately wider than [`LANE_COUNT`]:
    /// the ready ring must be able to run dry on some files and still hand over sixteen lanes.
    io_width: usize,
    /// Blocks a file is read through at once. One feeds the hasher while the rest fill.
    depth: usize,
    /// Files a worker may hold open at once, which is what keeps the shared cursor fair.
    open_cap: usize,
}

/// Workers for `files` files totalling `bytes`, on `cores` cores.
///
/// One per core, bounded by the files there are and by a byte floor. `W` workers over `F` files
/// deliver `max(0.156 * F, 1.33 * W)` GB/s below a full group each and `2.50 * W` above it, and
/// neither term ever falls as `W` rises — so there is no file count at which fewer workers win.
/// The middle regime is flat only because the single-stream fallback exists; without it four
/// eight-lane groups would lose to four single streams.
fn workers_for(files: usize, bytes: u64, cores: usize) -> usize {
    /// Below this a worker cannot repay a fork and a join, let alone a ring and an arena.
    const BYTES_WORTH_A_WORKER: u64 = 32 << 20;
    let affordable = (bytes / BYTES_WORTH_A_WORKER).max(1).min(usize::MAX as u64) as usize;
    cores.min(files.max(1)).min(affordable).max(1)
}

/// Files one worker may hold open at once.
///
/// This — not the lane count, not the fleet width — is what makes the shared cursor fair. A worker
/// that opened its full width before its first hash would take a thirty-two file run two workers
/// at a time and leave the rest spinning in the join. Dividing the run by the worker count first
/// gives four workers over thirty-two files eight slots each.
///
/// It caps concurrency, not entitlement: which files a worker gets is still decided one atomic
/// increment at a time, so a worker on slow media simply cycles fewer through the same slots.
fn open_cap_for(files: usize, workers: usize, io_width: usize) -> usize {
    files.div_ceil(workers.max(1)).clamp(1, io_width)
}

impl HashPlan {
    /// Whether `backend` may be tried: naming one skips the interfaces above it and still falls
    /// through to those below when the kernel refuses it.
    fn allows(&self, backend: IoBackend) -> bool {
        self.backend <= backend
    }

    /// Derive the widths from the memory budget. Throughput is flat from 1 KiB to 1 MiB per lane,
    /// so the chunk is a memory decision rather than a speed one, and it is what shrinks first:
    /// queue depth is what the disk actually cares about.
    fn from_args(args: &Args, files: usize, bytes: u64) -> Self {
        let cores = args
            .threads
            .map(NonZeroUsize::get)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, NonZeroUsize::get));
        let threads = workers_for(files, bytes, cores);
        let depth = args.io_depth.map(NonZeroUsize::get).unwrap_or(4);

        // Never below the lane count, or a worker could not assemble a group at all.
        let io_width = args
            .io_width
            .map(NonZeroUsize::get)
            .unwrap_or(2 * LANE_COUNT)
            .max(LANE_COUNT);

        let derived = args.memory.get() / (threads * io_width * depth).max(1);
        let chunk_bytes = args
            .io_chunk
            .map(NonZeroUsize::get)
            .unwrap_or_else(|| derived.clamp(DIRECT_IO_ALIGNMENT, 1 << 20));

        HashPlan {
            backend: args.io_backend,
            chunk_bytes: chunk_bytes.next_multiple_of(DIRECT_IO_ALIGNMENT),
            threads,
            io_width,
            depth,
            open_cap: open_cap_for(files, threads, io_width),
        }
    }
}

/// The files, ordered largest first, drawn from by every worker through one cursor.
///
/// A static shard would give each of N workers its own ragged tail, so N partial groups would drop
/// to the single-stream rate. One cursor leaves the whole run exactly one tail, and a worker stuck
/// behind slow media simply claims fewer files.
///
/// Descending by size does two jobs: the leftovers at the end are the smallest files, so the one
/// partial group is also the cheapest, and neighbours in size order make groups whose lanes finish
/// together — a group costs as much as its longest lane.
struct WorkQueue<'a> {
    order: &'a [usize],
    paths: &'a [&'a Path],
    sizes: &'a [u64],
    cursor: &'a AtomicUsize,
}

impl<'a> Iterator for WorkQueue<'a> {
    type Item = Assignment<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // One atomic per file, not per block: a few nanoseconds against a file that takes
        // milliseconds to read.
        let at = self.cursor.fetch_add(1, Ordering::Relaxed);
        let position = *self.order.get(at)?;
        Some((position, self.paths[position], self.sizes[position]))
    }
}

/// Re-lay a size-descending order so that any contiguous run of it is balanced.
///
/// Workers claim contiguously from one cursor, and a worker fills every slot it is allowed before
/// coming back — so from a plainly descending list the first worker takes the largest files and
/// the rest finish early and spin at the join. Dealing the descending list into `workers` piles
/// round-robin and concatenating them means the first worker's run holds the 1st, 5th, 9th …
/// largest rather than the top four, which is the classic longest-processing-time deal and evens
/// the totals without any coordination at claim time.
fn dealt_round_robin(descending: Vec<usize>, workers: usize) -> Vec<usize> {
    if workers <= 1 {
        return descending;
    }
    let mut dealt = Vec::with_capacity(descending.len());
    for pile in 0..workers {
        dealt.extend(descending.iter().skip(pile).step_by(workers).copied());
    }
    dealt
}

/// Hash every path, returning outcomes in the order the caller supplied them.
fn hash_paths(paths: &[&Path], sizes: &[u64], plan: &HashPlan) -> Vec<Option<Outcome>> {
    let mut order: Vec<usize> = (0..paths.len()).collect();
    order.sort_by(|left, right| sizes[*right].cmp(&sizes[*left]).then(left.cmp(right)));
    let order = dealt_round_robin(order, plan.threads);

    let cursor = AtomicUsize::new(0);
    let collected: Mutex<Vec<(usize, Outcome)>> = Mutex::new(Vec::with_capacity(paths.len()));

    // One worker body, run either on this thread or across the pool. Everything it needs is
    // built inside it, once, and reused for as many files as it manages to claim.
    let run_worker = |local: &mut Vec<(usize, Outcome)>| {
        let mut queue = WorkQueue {
            order: &order,
            paths,
            sizes,
            cursor: &cursor,
        };
        let mut fleet = select_fleet(plan);
        let mut states = vec![sz::Sha256::new(); fleet.width()];
        let mut staging = [sz::Sha256::new(); LANE_COUNT];
        let mut members = [0usize; LANE_COUNT];
        hash_with_fleet(
            fleet.as_mut(),
            &mut states,
            &mut staging,
            &mut members,
            &mut queue,
            plan.open_cap,
            &mut |position, outcome| local.push((position, outcome)),
        );
    };

    // A machine that will not describe its own cores still hashes, on one of them: the pool is
    // how the work is spread, never what makes it correct.
    let topology = (plan.threads > 1)
        .then(forkunion::Topology::new)
        .and_then(Result::ok);
    match topology {
        None => {
            let mut local = Vec::with_capacity(paths.len());
            run_worker(&mut local);
            collected
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend(local);
        }
        // Spawned as late as possible: a `forkunion` worker spins on `PAUSE` from the moment it
        // is created, so any setup left between the spawn and the broadcast is burnt on every
        // core but this one.
        Some(topology) => {
            let mut pool = forkunion::spawn(&topology, plan.threads);
            pool.broadcast(|_thread_index, _compute_domain_index| {
                // Sized up front, since a worker that claims more than its share would otherwise
                // grow this vector mid-run. Growing is off the hot loop either way — once per
                // file, never per chunk — but the capacity costs nothing to state.
                let mut local = Vec::with_capacity(paths.len() / plan.threads + plan.io_width);
                run_worker(&mut local);
                // One lock per worker, at the end of its run, rather than one per file. The
                // vector is append-only, so a poisoned lock still hands back coherent records.
                collected
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .extend(local);
            });
        }
    }

    // Scatter back into input order, which is the only order a manifest may be written in.
    let mut outcomes: Vec<Option<Outcome>> = (0..paths.len()).map(|_| None).collect();
    for (position, outcome) in collected
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
    {
        outcomes[position] = Some(outcome);
    }
    outcomes
}

// endregion: Workers

// region: Rendering

/// How a run renders its records.
struct OutputConfig {
    format: Format,
    terminator: Terminator,
}

/// Write `digest` as the lowercase hex `sha256sum` emits.
fn write_digest_to(output: &mut dyn Write, digest: &Digest) -> io::Result<()> {
    let mut text = [0u8; 64];
    for (index, byte) in digest.iter().enumerate() {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        text[index * 2] = HEX[(byte >> 4) as usize];
        text[index * 2 + 1] = HEX[(byte & 0xF) as usize];
    }
    output.write_all(&text)
}

/// Render every outcome, reporting how many records were written.
fn write_digests_to(
    output: &mut dyn Write,
    paths: &[&Path],
    outcomes: &[Option<Outcome>],
    config: &OutputConfig,
) -> io::Result<usize> {
    let mut written = 0;
    for (position, outcome) in outcomes.iter().enumerate() {
        let Some(Ok(hashed)) = outcome else {
            continue;
        };
        let path = paths[position].to_string_lossy();
        match config.format {
            // Two spaces between digest and path is what `sha256sum` writes and what its own
            // `--check` expects back, so this stays byte-for-byte rather than merely similar.
            Format::Coreutils => {
                write_digest_to(output, &hashed.digest)?;
                output.write_all(b"  ")?;
                output.write_all(path.as_bytes())?;
            }
            // What `sha256sum --tag` writes, and what its `--check` reads back.
            Format::Bsd => {
                output.write_all(b"SHA256 (")?;
                output.write_all(path.as_bytes())?;
                output.write_all(b") = ")?;
                write_digest_to(output, &hashed.digest)?;
            }
            Format::Table => {
                write_digest_to(output, &hashed.digest)?;
                let mut buffer = [0u8; 26];
                let bytes = format_grouped_number(&mut buffer, hashed.bytes as usize);
                write!(output, "  {:>15}  ", bytes)?;
                output.write_all(path.as_bytes())?;
            }
            Format::Json => {
                output.write_all(br#"{"type":"file","data":{"path":"#)?;
                json_text_field_to(output, path.as_bytes())?;
                output.write_all(br#","digest":""#)?;
                write_digest_to(output, &hashed.digest)?;
                write!(output, r#"","bytes":{}}}}}"#, hashed.bytes)?;
            }
        }
        output.write_all(&[config.terminator.as_byte()])?;
        written += 1;
    }
    Ok(written)
}

/// Report totals for the whole run, on stderr where they cannot contaminate a manifest.
fn write_summary_to(
    notes: &mut dyn Write,
    files: usize,
    bytes: u64,
    elapsed: std::time::Duration,
) -> io::Result<()> {
    let seconds = elapsed.as_secs_f64();
    let rate = if seconds > 0.0 {
        bytes as f64 / seconds
    } else {
        0.0
    };
    writeln!(
        notes,
        "{} file{} hashed, {}, {:.2} s — {}/s",
        files,
        if files == 1 { "" } else { "s" },
        scaled(bytes as f64),
        seconds,
        scaled(rate)
    )
}

/// A byte count in the largest unit that leaves a digit before the point, so a run of a few
/// megabytes does not report as `0.00 GB`.
fn scaled(bytes: f64) -> String {
    const UNITS: [(f64, &str); 4] = [(1e9, "GB"), (1e6, "MB"), (1e3, "kB"), (1.0, "B")];
    for (scale, name) in UNITS {
        if bytes >= scale {
            return format!("{:.2} {}", bytes / scale, name);
        }
    }
    format!("{:.2} B", bytes)
}

// endregion: Rendering

// region: Checking

/// One line of a checksum list: the digest it claims, and the file it claims it for.
struct CheckLine {
    digest: Digest,
    path: PathBuf,
}

/// Parse one line of a `sha256sum` list.
///
/// Both spacings the reference tool writes are accepted: two spaces for text mode, and a space
/// then `*` for the binary mode `-b` produces. Anything else is not a checksum line, and the
/// caller counts it rather than failing the run — a list may carry comments or blank lines.
fn parse_check_line(line: &[u8]) -> Option<CheckLine> {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let line = line.strip_suffix(b"\r").unwrap_or(line);

    // A BSD line names its algorithm first, so it is recognised by that prefix rather than by
    // a digest at column zero, and the two layouts cannot be confused.
    let (hex, path) = match line.strip_prefix(b"SHA256 (") {
        Some(tagged) => {
            let closing = tagged.windows(4).rposition(|window| window == b") = ")?;
            let (path, rest) = tagged.split_at(closing);
            (rest.get(4..)?, path)
        }
        None => {
            if line.len() < 67 {
                return None;
            }
            let (hex, rest) = line.split_at(64);
            let path = rest
                .strip_prefix(b"  ")
                .or_else(|| rest.strip_prefix(b" *"))?;
            (hex, path)
        }
    };
    if hex.len() != 64 || path.is_empty() {
        return None;
    }

    let mut digest = [0u8; 32];
    for (index, pair) in hex.chunks_exact(2).enumerate() {
        let value = |byte: u8| match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        };
        digest[index] = (value(pair[0])? << 4) | value(pair[1])?;
    }

    Some(CheckLine {
        digest,
        path: PathBuf::from(String::from_utf8_lossy(path).into_owned()),
    })
}

/// Read a checksum list, keeping only the lines that are checksum lines.
fn read_check_list(path: &Path) -> Result<(Vec<CheckLine>, usize), Failure> {
    let name = path.to_string_lossy().into_owned();
    let text = fs::read(path).at(name)?;
    let mut lines = Vec::new();
    let mut unreadable = 0;
    for line in text.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        match parse_check_line(line) {
            Some(entry) => lines.push(entry),
            None => unreadable += 1,
        }
    }
    Ok((lines, unreadable))
}

/// What verifying one line found.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Matched,
    Mismatched,
    Unreadable,
    /// Named by the list, absent from the filesystem, and passed over under `--ignore-missing`.
    /// Neither reported nor counted, which is what makes a partial download verifiable.
    Skipped,
}

/// Report each line's verdict in the layout `sha256sum --check` uses, and count the failures.
fn write_check_report_to(
    output: &mut dyn Write,
    lines: &[CheckLine],
    verdicts: &[Verdict],
    quiet: bool,
) -> io::Result<usize> {
    let mut failed = 0;
    for (line, verdict) in lines.iter().zip(verdicts) {
        let label = match verdict {
            Verdict::Skipped => continue,
            Verdict::Matched => "OK",
            Verdict::Mismatched => "FAILED",
            Verdict::Unreadable => "FAILED open or read",
        };
        if *verdict != Verdict::Matched {
            failed += 1;
        }
        if quiet {
            continue;
        }
        writeln!(output, "{}: {}", line.path.display(), label)?;
    }
    Ok(failed)
}

// endregion: Checking

// region: Input Processing

/// Every input the names resolve to, measured once.
///
/// The size rides along because the queue is ordered by it: a worker that re-measured would
/// `stat` each file a second time, and on a run of many small files those syscalls already
/// outnumber the reads.
fn measured(input: Input) -> Result<(Input, u64), Failure> {
    let size = input.size().at(input.display_name())?;
    Ok((input, size))
}

/// Hash the standard input, which no fleet can serve.
///
/// One state and one buffer: a stream of unknown length has no second file to share a lane
/// with, and nothing to gain from a wider read than the hasher consumes.
fn hash_stdin(chunk_bytes: usize) -> Outcome {
    let mut state = sz::Sha256::new();
    let mut buffer = vec![0u8; chunk_bytes];
    let mut bytes = 0u64;
    let mut input = io::stdin().lock();
    loop {
        match input.read(&mut buffer) {
            Ok(0) => break,
            Ok(taken) => {
                state.update(&buffer[..taken]);
                bytes += taken as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(Hashed {
        digest: state.digest(),
        bytes,
    })
}

/// Hash every source, returning outcomes in the order they were named.
///
/// Files go to the fleet together, so one pass still fills every lane; the standard input is
/// spliced back into its own position afterwards.
fn hash_sources(sources: &[(Input, u64)], plan: &HashPlan) -> Vec<Option<Outcome>> {
    let mut paths = Vec::with_capacity(sources.len());
    let mut sizes = Vec::with_capacity(sources.len());
    let mut positions = Vec::with_capacity(sources.len());
    for (position, (input, size)) in sources.iter().enumerate() {
        if matches!(input, Input::File(_)) {
            paths.push(input.path());
            sizes.push(*size);
            positions.push(position);
        }
    }

    let mut outcomes: Vec<Option<Outcome>> = (0..sources.len()).map(|_| None).collect();
    for (index, outcome) in hash_paths(&paths, &sizes, plan).into_iter().enumerate() {
        outcomes[positions[index]] = outcome;
    }
    for (position, (input, _)) in sources.iter().enumerate() {
        if matches!(input, Input::Stdin) {
            outcomes[position] = Some(hash_stdin(plan.chunk_bytes));
        }
    }
    outcomes
}

// endregion: Input Processing

/// Verify a checksum list, reporting each line and failing the run if any did not match.
fn run_check(
    args: &Args,
    list: &Path,
    output: &mut dyn Write,
    notes: &mut dyn Write,
) -> Result<Status, Failure> {
    let (lines, unreadable) = read_check_list(list)?;
    if lines.is_empty() {
        eprintln!("sz-sha256: {}: no checksum lines found", list.display());
        return Ok(Status::Error);
    }

    let paths: Vec<&Path> = lines.iter().map(|line| line.path.as_path()).collect();
    // A checksum list names files rather than walking to them, so this is their only measurement.
    let sizes: Vec<u64> = paths
        .iter()
        .map(|path| fs::metadata(path).map(|data| data.len()).unwrap_or(0))
        .collect();
    let plan = HashPlan::from_args(args, paths.len(), sizes.iter().sum());
    let started = std::time::Instant::now();
    let outcomes = hash_paths(&paths, &sizes, &plan);
    let elapsed = started.elapsed();

    let mut total_bytes = 0u64;
    let verdicts: Vec<Verdict> = lines
        .iter()
        .zip(&outcomes)
        .map(|(line, outcome)| match outcome {
            Some(Ok(hashed)) => {
                total_bytes += hashed.bytes;
                if hashed.digest == line.digest {
                    Verdict::Matched
                } else {
                    Verdict::Mismatched
                }
            }
            _ if args.ignore_missing && !line.path.exists() => Verdict::Skipped,
            _ => Verdict::Unreadable,
        })
        .collect();

    let failed = write_check_report_to(output, &lines, &verdicts, args.quiet).at("-")?;
    output.flush().at("-")?;

    if unreadable > 0 {
        eprintln!(
            "sz-sha256: warning: {} line(s) of {} are not checksum lines",
            unreadable,
            list.display()
        );
    }
    if failed > 0 {
        eprintln!(
            "sz-sha256: {} of {} files failed verification",
            failed,
            lines.len()
        );
    }
    if args.summary {
        write_summary_to(notes, lines.len(), total_bytes, elapsed).at("-")?;
    }

    // Skipping every line leaves a run that verified nothing, which is a failure to do the job
    // rather than a clean bill of health.
    let checked = verdicts
        .iter()
        .filter(|verdict| **verdict != Verdict::Skipped)
        .count();
    if checked == 0 {
        eprintln!("sz-sha256: {}: no file was verified", list.display());
        return Ok(Status::NoResult);
    }
    // A malformed line is a run that completed and found the answer to be no, so it reports
    // the same way a mismatch does — but only where the caller asked for it to count at all.
    if args.strict && unreadable > 0 {
        return Ok(Status::NoResult);
    }

    // A mismatch is a run that completed and found the answer to be no, which is exit 1 rather
    // than exit 2 — the same distinction `grep` draws and `sha256sum --check` keeps.
    Ok(Status::from_found(failed == 0))
}

fn run(args: &Args, output: &mut dyn Write, notes: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    if let Some(list) = args.check.as_deref() {
        return run_check(args, list, output, notes);
    }

    let traversal = TraversalOptions {
        hidden: args.hidden,
        no_ignore: args.no_ignore,
        follow: args.follow,
        max_depth: args.max_depth,
        file_type: args.file_type.as_deref(),
    };

    let globs = args
        .glob
        .as_deref()
        .map(compile_globs)
        .transpose()
        .map_err(reject)?;

    let mut tally = Tally::default();
    let mut sources = Vec::new();
    for resolved in inputs(&args.inputs, &traversal, globs.as_deref(), "sz-sha256") {
        match resolved.and_then(measured) {
            Ok(source) => sources.push(source),
            Err(failure) => {
                eprintln!("sz-sha256: {}", failure);
                tally.failed();
            }
        }
    }
    if sources.is_empty() {
        return Ok(tally.status());
    }

    let paths: Vec<&Path> = sources.iter().map(|(input, _)| input.path()).collect();
    // The standard input cannot be sized before it is read, so it is no part of the budget the
    // widths are derived from.
    let sized = sources
        .iter()
        .filter(|(input, _)| matches!(input, Input::File(_)))
        .map(|(_, size)| *size);
    let plan = HashPlan::from_args(args, sources.len(), sized.sum());
    let started = std::time::Instant::now();
    let outcomes = hash_sources(&sources, &plan);
    let elapsed = started.elapsed();

    // Naming happens here rather than in a worker: `.at` is the only way to build a
    // `Failure::Io`, and the path is already at hand where an allocation costs nothing.
    let mut hashed_count = 0;
    let mut total_bytes = 0u64;
    for (position, outcome) in outcomes.iter().enumerate() {
        match outcome {
            Some(Ok(entry)) => {
                hashed_count += 1;
                total_bytes += entry.bytes;
            }
            Some(Err(error)) => {
                let named = io::Error::new(error.kind(), error.to_string());
                let failure = Failure::Io {
                    path: paths[position].to_string_lossy().into_owned(),
                    source: named,
                };
                eprintln!("sz-sha256: {}", failure);
                tally.failed();
            }
            None => {}
        }
    }

    let written = if args.quiet {
        hashed_count
    } else {
        let config = OutputConfig {
            format: args.format,
            terminator: Terminator::from_null(args.null),
        };
        let destination = match args.output.as_deref().filter(|path| *path != "-") {
            Some(path) => Destination::Creating(path),
            None => Destination::Stdout,
        };
        destination.write("sz-sha256", output, |file| {
            write_digests_to(file, &paths, &outcomes, &config)
        })?
    };
    output.flush().at("-")?;

    if args.summary {
        write_summary_to(notes, hashed_count, total_bytes, elapsed).at("-")?;
    }

    if written > 0 {
        tally.produced();
    }
    Ok(tally.status())
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-sha256", run(&args, &mut output, &mut io::stderr()))
}

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes, so a test file's digest is stable across runs without
    /// storing a fixture beside it.
    fn scratch_bytes(length: usize, seed: u8) -> Vec<u8> {
        (0..length)
            .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    /// A directory of files with the given lengths, returned with the directory that owns them.
    fn scratch_files(lengths: &[usize]) -> (tempfile::TempDir, Vec<PathBuf>) {
        let directory = tempfile::TempDir::new().unwrap();
        let paths = lengths
            .iter()
            .enumerate()
            .map(|(index, length)| {
                let path = directory.path().join(format!("file{}.bin", index));
                fs::write(&path, scratch_bytes(*length, index as u8)).unwrap();
                path
            })
            .collect();
        (directory, paths)
    }

    /// Sizes for a set of scratch files. The tool itself gets these from the walk, which is
    /// where the one and only measurement happens; a test that builds its own list measures here.
    fn measure(paths: &[PathBuf]) -> Vec<u64> {
        paths
            .iter()
            .map(|path| fs::metadata(path).map(|data| data.len()).unwrap_or(0))
            .collect()
    }

    fn borrowed(paths: &[PathBuf]) -> Vec<&Path> {
        paths.iter().map(PathBuf::as_path).collect()
    }

    fn plan_with(chunk_bytes: usize, threads: usize) -> HashPlan {
        HashPlan {
            backend: IoBackend::Blocking,
            chunk_bytes,
            threads,
            io_width: 2 * LANE_COUNT,
            depth: 4,
            open_cap: 2 * LANE_COUNT,
        }
    }

    fn digests_via(paths: &[PathBuf], chunk_bytes: usize) -> Vec<Digest> {
        hash_paths(
            &borrowed(paths),
            &measure(paths),
            &plan_with(chunk_bytes, 1),
        )
        .into_iter()
        .map(|outcome| outcome.unwrap().unwrap().digest)
        .collect()
    }

    /// The digest each file would get on its own, which every concurrent arrangement must match.
    fn digests_directly(paths: &[PathBuf]) -> Vec<Digest> {
        paths
            .iter()
            .map(|path| sz::Sha256::hash(&fs::read(path).unwrap()))
            .collect()
    }

    #[test]
    fn hashes_the_known_vectors() {
        let empty = sz::Sha256::hash(b"");
        assert_eq!(
            hex(&empty),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let abc = sz::Sha256::hash(b"abc");
        assert_eq!(
            hex(&abc),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn hex(digest: &Digest) -> String {
        let mut text = Vec::new();
        write_digest_to(&mut text, digest).unwrap();
        String::from_utf8(text).unwrap()
    }

    #[test]
    fn matches_single_shot_hashing_across_chunk_sizes() {
        let (_directory, paths) = scratch_files(&[0, 1, 63, 64, 65, 4095, 4096, 100_000]);
        let expected = digests_directly(&paths);

        // Chunking must not shift a digest, whatever boundary it falls on.
        for chunk_bytes in [DIRECT_IO_ALIGNMENT, 8192, 1 << 16] {
            assert_eq!(
                digests_via(&paths, chunk_bytes),
                expected,
                "chunk {}",
                chunk_bytes
            );
        }
    }

    #[test]
    fn hashes_an_empty_file_without_idling_the_group() {
        let (_directory, paths) = scratch_files(&[0, 0, 0]);
        let empty = sz::Sha256::hash(b"");
        assert_eq!(digests_via(&paths, DIRECT_IO_ALIGNMENT), vec![empty; 3]);
    }

    #[test]
    fn hashes_files_of_uneven_lengths_alike() {
        // A one-byte file beside a multi-megabyte one shares a window, and the short lane must
        // retire and refill without disturbing the long one.
        let (_directory, paths) = scratch_files(&[1, 3 << 20, 2, 1 << 20, 5]);
        let expected = digests_directly(&paths);
        assert_eq!(digests_via(&paths, 1 << 16), expected);
    }

    #[test]
    fn preserves_input_order_under_concurrency() {
        let (_directory, paths) = scratch_files(&[900, 100, 500, 700, 300]);
        let expected = digests_directly(&paths);
        assert_eq!(digests_via(&paths, DIRECT_IO_ALIGNMENT), expected);
    }

    #[test]
    fn keeps_more_files_than_lanes_moving() {
        let lengths: Vec<usize> = (0..LANE_COUNT * 3 + 7).map(|index| index * 1000).collect();
        let (_directory, paths) = scratch_files(&lengths);
        let expected = digests_directly(&paths);
        assert_eq!(digests_via(&paths, DIRECT_IO_ALIGNMENT), expected);
    }

    #[test]
    fn agrees_with_itself_across_thread_counts() {
        // Enough files that several workers genuinely overlap, and uneven enough that they
        // finish out of order — which is exactly when a scatter by input position earns itself.
        let lengths: Vec<usize> = (0..70).map(|index| (index * 7919) % 300_000).collect();
        let (_directory, paths) = scratch_files(&lengths);
        let expected = digests_directly(&paths);

        for threads in [1, 2, 4, 8] {
            let digests: Vec<Digest> = hash_paths(
                &borrowed(&paths),
                &measure(&paths),
                &plan_with(1 << 16, threads),
            )
            .into_iter()
            .map(|outcome| outcome.unwrap().unwrap().digest)
            .collect();
            assert_eq!(digests, expected, "threads {}", threads);
        }
    }

    #[test]
    fn hashes_largest_files_first_whatever_the_input_order() {
        // The queue reorders by size to keep groups even, so the ordering must not leak into
        // the results the caller sees.
        let (_directory, paths) = scratch_files(&[10, 900_000, 50, 400_000, 3]);
        let expected = digests_directly(&paths);
        let digests: Vec<Digest> =
            hash_paths(&borrowed(&paths), &measure(&paths), &plan_with(1 << 16, 2))
                .into_iter()
                .map(|outcome| outcome.unwrap().unwrap().digest)
                .collect();
        assert_eq!(digests, expected);
    }

    #[test]
    fn reports_an_unreadable_file_without_abandoning_the_others() {
        let (directory, paths) = scratch_files(&[10, 20]);
        let mut all = paths.clone();
        all.insert(1, directory.path().join("absent.bin"));

        let outcomes = hash_paths(
            &borrowed(&all),
            &measure(&all),
            &plan_with(DIRECT_IO_ALIGNMENT, 1),
        );

        assert_eq!(outcomes.len(), 3);
        assert!(matches!(outcomes[0], Some(Ok(_))));
        assert!(matches!(outcomes[1], Some(Err(_))));
        assert!(matches!(outcomes[2], Some(Ok(_))));
    }

    #[test]
    fn writes_coreutils_format_byte_for_byte() {
        let (_directory, paths) = scratch_files(&[7]);
        let outcomes = hash_paths(
            &borrowed(&paths),
            &measure(&paths),
            &plan_with(DIRECT_IO_ALIGNMENT, 1),
        );
        let config = OutputConfig {
            format: Format::Coreutils,
            terminator: Terminator::Newline,
        };

        let mut rendered = Vec::new();
        assert_eq!(
            write_digests_to(&mut rendered, &borrowed(&paths), &outcomes, &config).unwrap(),
            1
        );

        let expected = format!(
            "{}  {}\n",
            hex(&sz::Sha256::hash(&fs::read(&paths[0]).unwrap())),
            paths[0].display()
        );
        assert_eq!(String::from_utf8(rendered).unwrap(), expected);
    }

    #[test]
    fn terminates_records_with_nul_under_null() {
        let (_directory, paths) = scratch_files(&[3]);
        let outcomes = hash_paths(
            &borrowed(&paths),
            &measure(&paths),
            &plan_with(DIRECT_IO_ALIGNMENT, 1),
        );
        let config = OutputConfig {
            format: Format::Coreutils,
            terminator: Terminator::Null,
        };

        let mut rendered = Vec::new();
        write_digests_to(&mut rendered, &borrowed(&paths), &outcomes, &config).unwrap();
        assert_eq!(rendered.last(), Some(&0));
    }

    #[test]
    fn rounds_a_chunk_up_to_the_alignment() {
        let args = Args::try_parse_from(["sz-sha256", "--memory", "1Ki", "f"]).unwrap();
        let plan = HashPlan::from_args(&args, 1, 1 << 20);
        assert_eq!(plan.chunk_bytes % DIRECT_IO_ALIGNMENT, 0);
    }

    #[test]
    fn agrees_with_blocking_on_every_available_backend() {
        // Files either side of the megabyte where `O_DIRECT` switches on, and none of them a
        // multiple of the alignment, so the tail block is exercised on every backend. A backend
        // the kernel refuses falls through to the next, which is a slower answer rather than a
        // different one — so every arm must still match.
        let (_directory, paths) = scratch_files(&[1, 4095, 1 << 20, (2 << 20) + 777, 300_000]);
        let expected = digests_directly(&paths);

        for backend in [
            IoBackend::Blocking,
            IoBackend::Uring,
            IoBackend::Aio,
            IoBackend::Auto,
        ] {
            let plan = HashPlan {
                backend,
                chunk_bytes: 1 << 16,
                threads: 1,
                io_width: 2 * LANE_COUNT,
                depth: 4,
                open_cap: 2 * LANE_COUNT,
            };
            let digests: Vec<Digest> = hash_paths(&borrowed(&paths), &measure(&paths), &plan)
                .into_iter()
                .map(|outcome| outcome.unwrap().unwrap().digest)
                .collect();
            assert_eq!(digests, expected, "backend {:?}", backend);
        }
    }

    #[test]
    fn recycles_slots_when_files_outnumber_the_io_width() {
        // Far more files than slots, so every slot must retire and refill many times over. A
        // stale completion landing in a recycled slot would show up here as a wrong digest.
        let lengths: Vec<usize> = (0..90).map(|index| 5000 + index * 313).collect();
        let (_directory, paths) = scratch_files(&lengths);
        let mut plan = plan_with(1 << 12, 1);
        plan.io_width = LANE_COUNT;
        plan.open_cap = LANE_COUNT;

        let digests: Vec<Digest> = hash_paths(&borrowed(&paths), &measure(&paths), &plan)
            .into_iter()
            .map(|outcome| outcome.unwrap().unwrap().digest)
            .collect();
        assert_eq!(digests, digests_directly(&paths));
    }

    #[test]
    fn agrees_at_io_widths_above_and_below_the_lane_count() {
        let lengths: Vec<usize> = (0..40).map(|index| 20_000 + index * 1111).collect();
        let (_directory, paths) = scratch_files(&lengths);
        let expected = digests_directly(&paths);

        // Below the lane count a worker can never fill a group and always hashes single stream;
        // above it, groups form. Both must agree, on every backend that probes.
        for io_width in [LANE_COUNT / 2, LANE_COUNT, 4 * LANE_COUNT] {
            for backend in [IoBackend::Blocking, IoBackend::Uring, IoBackend::Aio] {
                let mut plan = plan_with(1 << 13, 1);
                plan.backend = backend;
                plan.io_width = io_width;
                plan.open_cap = io_width;
                let digests: Vec<Digest> = hash_paths(&borrowed(&paths), &measure(&paths), &plan)
                    .into_iter()
                    .map(|outcome| outcome.unwrap().unwrap().digest)
                    .collect();
                assert_eq!(
                    digests, expected,
                    "width {} backend {:?}",
                    io_width, backend
                );
            }
        }
    }

    #[test]
    fn agrees_across_the_group_and_single_stream_paths() {
        // Below nine live lanes the window hashes one stream per lane, and above it in a group.
        // Both must
        // produce the same digests, which is what makes the switch safe to take mid-file.
        for count in [1, 8, 9, 16, 40] {
            let lengths: Vec<usize> = (0..count).map(|index| 200_000 + index * 1000).collect();
            let (_directory, paths) = scratch_files(&lengths);
            assert_eq!(
                digests_via(&paths, 1 << 16),
                digests_directly(&paths),
                "{} files",
                count
            );
        }
    }

    #[test]
    fn parses_a_checksum_line_in_both_spacings() {
        let digest = sz::Sha256::hash(b"abc");
        let text = hex(&digest);

        // Text mode writes two spaces; `sha256sum -b` writes one space and a star.
        for separator in ["  ", " *"] {
            let line = format!("{}{}some/file.bin", text, separator);
            let parsed = parse_check_line(line.as_bytes()).unwrap();
            assert_eq!(parsed.digest, digest);
            assert_eq!(parsed.path, PathBuf::from("some/file.bin"));
        }
    }

    #[test]
    fn skips_a_checksum_line_it_cannot_parse() {
        // A list may carry comments and blank lines, and neither is a failure.
        for line in [
            &b"# a comment"[..],
            b"",
            b"not-hex-at-all  file.bin",
            b"abcd  short-digest",
        ] {
            assert!(parse_check_line(line).is_none());
        }
    }

    #[test]
    fn reports_a_mismatch_through_the_exit_code() {
        let (_directory, paths) = scratch_files(&[1000, 2000]);
        let lines: Vec<CheckLine> = paths
            .iter()
            .map(|path| CheckLine {
                digest: sz::Sha256::hash(&fs::read(path).unwrap()),
                path: path.clone(),
            })
            .collect();

        let clean = [Verdict::Matched, Verdict::Matched];
        let mut rendered = Vec::new();
        assert_eq!(
            write_check_report_to(&mut rendered, &lines, &clean, false).unwrap(),
            0
        );
        assert!(String::from_utf8(rendered).unwrap().contains(": OK"));

        let dirty = [Verdict::Matched, Verdict::Mismatched];
        let mut rendered = Vec::new();
        assert_eq!(
            write_check_report_to(&mut rendered, &lines, &dirty, false).unwrap(),
            1
        );
        assert!(String::from_utf8(rendered).unwrap().contains(": FAILED"));
    }

    #[test]
    fn reads_the_bsd_layout_it_writes() {
        let (_directory, paths) = scratch_files(&[64, 4095]);
        let digests = digests_directly(&paths);

        let mut rendered = Vec::new();
        let outcomes: Vec<Option<Outcome>> = digests
            .iter()
            .zip(&paths)
            .map(|(digest, path)| {
                Some(Ok(Hashed {
                    digest: *digest,
                    bytes: std::fs::metadata(path).unwrap().len(),
                }))
            })
            .collect();
        let config = OutputConfig {
            format: Format::Bsd,
            terminator: Terminator::from_null(false),
        };
        write_digests_to(&mut rendered, &borrowed(&paths), &outcomes, &config).unwrap();

        let text = String::from_utf8(rendered).unwrap();
        assert!(text.starts_with("SHA256 ("), "{}", text);
        for (line, digest) in text.lines().zip(&digests) {
            let parsed = parse_check_line(line.as_bytes()).expect("a BSD line is a checksum line");
            assert_eq!(&parsed.digest, digest);
        }
    }

    #[test]
    fn reads_a_path_holding_the_bsd_separator() {
        // `) = ` inside a name is legal, so the parser must split on the last one rather than
        // the first, or the digest is read out of the middle of the path.
        let line = b"SHA256 (od d) = ity) = 0000000000000000000000000000000000000000000000000000000000000000";
        let parsed = parse_check_line(line).expect("a name may hold the separator");
        assert_eq!(parsed.path, PathBuf::from("od d) = ity"));
        assert_eq!(parsed.digest, [0u8; 32]);
    }

    #[test]
    fn hashes_the_standard_input_like_a_named_file() {
        // The stream path is a different kernel from the fleet's, so it has to be pinned to the
        // same answer rather than assumed to reach it.
        let (_directory, paths) = scratch_files(&[300_000]);
        let expected = digests_directly(&paths)[0];
        let contents = std::fs::read(&paths[0]).unwrap();

        let mut state = sz::Sha256::new();
        for window in contents.chunks(4096) {
            state.update(window);
        }
        assert_eq!(state.digest(), expected);
    }

    #[test]
    fn skips_a_missing_line_without_counting_it() {
        let lines = vec![
            CheckLine {
                digest: [1u8; 32],
                path: PathBuf::from("present"),
            },
            CheckLine {
                digest: [2u8; 32],
                path: PathBuf::from("absent"),
            },
        ];
        let verdicts = [Verdict::Matched, Verdict::Skipped];
        let mut rendered = Vec::new();
        assert_eq!(
            write_check_report_to(&mut rendered, &lines, &verdicts, false).unwrap(),
            0
        );
        let text = String::from_utf8(rendered).unwrap();
        assert!(text.contains("present: OK"), "{}", text);
        assert!(!text.contains("absent"), "{}", text);
    }

    #[test]
    fn declares_no_short_flags() {
        let command = Args::command();
        assert!(command
            .get_arguments()
            .all(|argument| argument.get_short().is_none()
                || matches!(argument.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect();
        assert_eq!(
            longs,
            [
                "check",
                "ignore-missing",
                "strict",
                "format",
                "quiet",
                "output",
                "summary",
                "null",
                "memory",
                "threads",
                "io-backend",
                "io-width",
                "io-depth",
                "io-chunk",
                "type",
                "glob",
                "max-depth",
                "hidden",
                "no-ignore",
                "follow",
                "help",
                "version",
            ]
        );
    }

    #[test]
    fn writes_json_lines_with_escaped_paths() {
        let (_directory, paths) = scratch_files(&[5]);
        let outcomes = hash_paths(
            &borrowed(&paths),
            &measure(&paths),
            &plan_with(DIRECT_IO_ALIGNMENT, 1),
        );
        let config = OutputConfig {
            format: Format::Json,
            terminator: Terminator::Newline,
        };

        let mut rendered = Vec::new();
        write_digests_to(&mut rendered, &borrowed(&paths), &outcomes, &config).unwrap();
        let text = String::from_utf8(rendered).unwrap();
        assert!(text.starts_with(r#"{"type":"file","data":{"path":"#));
        assert!(text.contains(r#""bytes":5"#));
    }

    /// Parse and then apply the value-conditional checks, as `run` does.
    fn accepts(flags: &[&str]) -> bool {
        let arguments = ["sz-sha256", "f"].into_iter().chain(flags.iter().copied());
        Args::try_parse_from(arguments).is_ok_and(|args| validate(&args).is_ok())
    }

    #[test]
    fn declares_the_conflicts_that_used_to_pass_silently() {
        for flags in [
            vec!["--format", "json", "--null"],
            vec!["--io-chunk", "100"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
        assert!(accepts(&["--format", "json"]));
        assert!(accepts(&["--io-chunk", "8Ki"]));
    }
}

// endregion: Tests
