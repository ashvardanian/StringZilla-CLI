//! SIMD-accelerated substring replacement utility
//!
//! A simpler alternative to sed/awk for substring replacement.
//! Uses StringZilla for fast searching and replacing.
//!
//! # Examples
//!
//! ```bash
//! # Replace all occurrences
//! sz-replace foo bar file.txt
//!
//! # Replace in-place
//! sz-replace --in-place foo bar file.txt
//!
//! # Case-insensitive replacement
//! sz-replace --ignore-case foo bar file.txt
//!
//! # Show one line about the whole run
//! sz-replace --summary foo bar file.txt
//!
//! # Refuse the edit unless the file is still what was read
//! sz-replace --in-place --expect-hash 2544218206cee57b foo bar file.txt
//!
//! # From stdin
//! cat file.txt | sz-replace foo bar
//! ```

use std::io::{self, Read, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{find, utf8_uncased_fold, utf8_uncased_search};

use shared::*;

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
}

/// How the pattern is interpreted.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Match {
    /// A literal substring, matched anywhere
    Substring,
    /// A line name, as `sz-find --fields line-hashes` prints one
    LineHash,
}

/// Where the replacement goes, relative to what the pattern matched.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// In place of the match.
    Over,
    /// Inserted after it, leaving it alone.
    After,
    /// Inserted before it.
    Before,
}

/// How many of the pattern's matches a run acts on.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Occurrences {
    /// Every match, as `sed s///g` does
    All,
    /// The leftmost match only, as `sed s///` does
    First,
    /// Exactly one match; refuse and change nothing if there are more
    One,
}

/// Replace substrings in files
#[derive(Parser)]
#[command(name = "sz-replace")]
#[command(version, about = "SIMD-accelerated substring replacement", long_about = None)]
struct Args {
    /// Substring to search for
    pattern: String,

    /// Replacement string
    replacement: String,

    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Write to this file instead of stdout
    #[arg(long, conflicts_with_all = ["in_place", "dry_run", "quiet"])]
    output: Option<String>,

    /// Rewrite the input file, swapping the result in atomically once it is on disk
    #[arg(long, conflicts_with_all = ["dry_run", "quiet"])]
    in_place: bool,

    /// Report what would be replaced without writing anything
    #[arg(long, conflicts_with_all = ["quiet", "summary"])]
    dry_run: bool,

    /// Refuse the edit unless the input still hashes to this, as `sz-find --fields file-hash` prints it
    #[arg(long, value_name = "HASH", value_parser = parse_content_hash)]
    expect_hash: Option<u64>,

    /// Read the pattern as a literal substring or as a line name
    #[arg(
        long = "match",
        value_name = "MATCH",
        value_enum,
        default_value = "substring",
        help_heading = "Matching"
    )]
    match_kind: Match,

    /// Insert the replacement after the matched line rather than over it
    #[arg(long, conflicts_with = "before", help_heading = "Placement")]
    after: bool,

    /// Insert the replacement before the matched line rather than over it
    #[arg(long, help_heading = "Placement")]
    before: bool,

    /// Split lines on the Unicode newline set rather than LF alone
    #[arg(long, help_heading = "Matching")]
    utf8: bool,

    /// How many matches to act on
    #[arg(long, value_enum, default_value_t = Occurrences::All, help_heading = "Placement")]
    occurrences: Occurrences,

    /// Fold case when searching; matching is byte-literal, so there is no --utf8 to pair it with
    #[arg(long)]
    ignore_case: bool,

    /// Render the summary as text or as a JSON record
    #[arg(long, value_enum, default_value_t = Format::Text, help_heading = "Output Formats")]
    format: Format,

    /// Print one line about the whole run on stdout
    #[arg(long, help_heading = "Output Formats")]
    summary: bool,

    /// Suppress all output; exit 0 if anything was replaced, 1 otherwise
    #[arg(long, conflicts_with = "summary", help_heading = "Output Formats")]
    quiet: bool,
}

impl Args {
    /// Whether this run must see the whole input before it may write.
    ///
    /// Three questions cannot be answered from a window: whether the input still hashes to
    /// what the caller expects, whether a pattern occurs exactly once, and which line a name
    /// picks out. Read by the code that chooses how to open the input *and* by the code that
    /// decides what to edit, so the two cannot drift into answering different questions.
    fn needs_whole_input(&self) -> bool {
        self.occurrences == Occurrences::One
            || self.match_kind == Match::LineHash
            || self.expect_hash.is_some()
    }
}

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Every constraint that depends on an argument's *value*, which clap cannot declare.
fn validate(args: &Args) -> Result<(), clap::Error> {
    if args.pattern.is_empty() {
        return Err(reject("pattern cannot be empty"));
    }
    if args.match_kind == Match::LineHash {
        // A name is read in either case already, and there is no substring to fold.
        if args.ignore_case {
            return Err(reject(
                "--match line-hash cannot be combined with --ignore-case",
            ));
        }
        // An insertion with nothing to insert is inert by construction, except on a file's
        // last unterminated line, where it would quietly add a terminator nobody asked for.
        if (args.after || args.before) && args.replacement.is_empty() {
            return Err(reject(
                "--after and --before insert a line, so the replacement cannot be empty; \
                 an empty replacement deletes the line it names instead",
            ));
        }
        // Reported here rather than by the value parser, since the pattern is only a name
        // under this mode and is a perfectly good substring otherwise.
        if let Err(message) = parse_hash_prefix(&args.pattern) {
            return Err(reject(message));
        }
    } else if args.after || args.before {
        return Err(reject(
            "--after and --before place a line, so they need --match line-hash",
        ));
    } else if args.utf8 {
        return Err(reject(
            "--utf8 chooses how lines are split, so it needs --match line-hash",
        ));
    }
    if args.format == Format::Json {
        if args.quiet {
            return Err(reject("--format json cannot be combined with --quiet"));
        }
        // The summary record must never share stdout with the transformed bytes.
        let diverted =
            args.in_place || args.dry_run || args.output.as_deref().is_some_and(|path| path != "-");
        if !diverted {
            return Err(reject(
                "--format json requires --output, --in-place or --dry-run",
            ));
        }
    }
    if args.in_place && args.input.as_deref().is_none_or(|path| path == "-") {
        return Err(reject(
            "--in-place requires a file argument (cannot rewrite stdin)",
        ));
    }
    Ok(())
}

/// The next match in `rest`, as an offset and the length the match occupies — which case
/// folding can make differ from the pattern's own length (ß ↔ ss, İ ↔ i).
///
/// Case-insensitive matching uses StringZilla's full-Unicode `utf8_uncased_search`, with no
/// whole-file `to_ascii_lowercase` copy.
#[inline]
fn next_match(rest: &[u8], pattern: &[u8], ignore_case: bool) -> Option<(usize, usize)> {
    if ignore_case {
        utf8_uncased_search(rest, pattern)
    } else {
        find(rest, pattern).map(|offset| (offset, pattern.len()))
    }
}

/// How many times `pattern` occurs, counting the same leftmost, non-overlapping matches
/// [`replace_to`] would rewrite. Its own pass, because `--occurrences one` has to settle
/// whether the target is unique before any destination is opened.
fn count_matches(data: &[u8], pattern: &[u8], ignore_case: bool) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    let mut rest = data;
    let mut count = 0;
    while let Some((offset, matched_len)) = next_match(rest, pattern, ignore_case) {
        count += 1;
        rest = &rest[offset + matched_len..];
    }
    count
}

/// Replace up to `limit` occurrences of `pattern` with `replacement`, streaming the output
/// into `out` (no full-file buffer). Returns the replacement count.
///
/// Matching is leftmost-first and non-overlapping, as `sed s///g` is, and advances by the
/// matched length rather than the pattern's so a folded match of a different width keeps the
/// cursor in step.
fn replace_to(
    data: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
    limit: usize,
    out: &mut dyn Write,
) -> io::Result<usize> {
    // An empty pattern matches at every position without consuming anything, so it
    // has no replacement to make.
    if pattern.is_empty() || limit == 0 {
        out.write_all(data)?;
        return Ok(0);
    }

    let mut rest = data;
    let mut count = 0;
    while count < limit {
        let Some((offset, matched_len)) = next_match(rest, pattern, ignore_case) else {
            break;
        };
        out.write_all(&rest[..offset])?;
        out.write_all(replacement)?;
        count += 1;
        rest = &rest[offset + matched_len..];
    }
    out.write_all(rest)?;
    Ok(count)
}

/// One replacement of a byte range, resolved against the input before anything is written.
///
/// An insertion is a zero-width span. Splices come out of resolution already in order and
/// disjoint, which is the shape a batch of edits would need too.
struct Splice {
    start: usize,
    end: usize,
    text: Vec<u8>,
}

/// What a run writes, however its target was addressed.
enum Edit<'a> {
    /// Every match of a literal pattern, up to the limit.
    Substring {
        pattern: &'a [u8],
        replacement: &'a [u8],
        ignore_case: bool,
        limit: usize,
    },
    /// Named lines, already resolved.
    Lines(Vec<Splice>),
}

/// Where the bytes to transform come from. Owned, so the borrow of the mapping lasts
/// exactly as long as the run that is reading it.
enum Source {
    /// The whole input at once: a mapped file, or a pipe a run had to hold.
    Whole(InputSource),
    /// A pipe, read one window at a time.
    Windows(Refill<io::StdinLock<'static>>),
}

impl Source {
    /// The whole input, for the decisions that cannot be made without it.
    fn whole(&self) -> Option<&[u8]> {
        match self {
            Source::Whole(held) => Some(held.as_bytes()),
            Source::Windows(_) => None,
        }
    }
}

/// Everything the replacement needs that does not depend on where it is written, so the
/// destinations cannot disagree about what they are producing or whether it is hashed.
struct Substitution<'a> {
    source: Source,
    edit: Edit<'a>,
    /// Whether the produced bytes are hashed on their way out. A run that reports no hash
    /// pays for none, which is what keeps the plain pipe as fast as it was.
    hashed: bool,
}

/// The most bytes one match can occupy, which is how far a window has to see past what it
/// emits.
///
/// Folding makes this larger than the pattern — `ss` matches `ẞ`, three bytes — but not
/// unboundedly so. A matching span folds to exactly what the pattern folds to, every
/// character occupies at most four UTF-8 bytes, and every character folds to at least one:
/// so no match can run past four times the folded length.
fn longest_match(pattern: &[u8], ignore_case: bool) -> usize {
    if !ignore_case {
        return pattern.len();
    }
    // Folding expands by at most three, which is the bound `sz-dedup` sizes its own scratch by.
    let mut scratch = vec![0u8; pattern.len().saturating_mul(3).max(64)];
    utf8_uncased_fold(pattern, &mut scratch[..]).saturating_mul(4)
}

/// Replace within one window, emitting only what is settled and reporting how far that got.
///
/// `carry` is how many trailing bytes could still be the opening of a match the next window
/// completes; at end of input nothing is pending and it is zero. A match that *starts* in
/// the carry is left for the next window, and one that starts before it and runs past it is
/// whole here and emitted.
///
/// Returns `(replacements, consumed)`. A `consumed` of zero means the window is too narrow
/// to settle anything, which is the caller's cue to widen it.
fn replace_window(
    window: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
    limit: usize,
    carry: usize,
    out: &mut dyn Write,
) -> io::Result<(usize, usize)> {
    let settled = window.len().saturating_sub(carry);
    let (mut base, mut count) = (0, 0);
    while count < limit {
        let Some((offset, matched)) = next_match(&window[base..], pattern, ignore_case) else {
            break;
        };
        let start = base + offset;
        if start >= settled {
            break;
        }
        out.write_all(&window[base..start])?;
        out.write_all(replacement)?;
        count += 1;
        base = start + matched;
    }
    // A match reaching past the settled point is still whole, so it carries the boundary
    // with it rather than being scanned a second time.
    let consumed = base.max(settled);
    out.write_all(&window[base..consumed])?;
    Ok((count, consumed))
}

/// Replace through a window at a time, returning the replacement count and the bytes read.
///
/// Driven by hand rather than through [`Refill::for_each_window`], because a scanner whose
/// match may straddle a seam has to *see* further than it *emits* — one byte less than the
/// longest match stays pending at the end of every window — and a cut can only say where to
/// stop.
fn replace_stream<R: Read>(
    refill: &mut Refill<R>,
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
    limit: usize,
    destination: &mut dyn Write,
) -> io::Result<(usize, usize)> {
    let pending = longest_match(pattern, ignore_case).saturating_sub(1);
    let (mut count, mut bytes, mut consumed) = (0, 0, 0);
    while refill.advance(consumed)? {
        // Nothing is pending once the input has ended: the last window is whole.
        let carry = if refill.at_eof() { 0 } else { pending };
        let (made, took) = replace_window(
            refill.filled(),
            pattern,
            replacement,
            ignore_case,
            limit.saturating_sub(count),
            carry,
            destination,
        )?;
        count += made;
        bytes += took;
        consumed = took;
        // A window no wider than what is pending settles nothing, so it doubles instead of
        // handing the same bytes back forever.
        if consumed == 0 {
            refill.grow()?;
        }
    }
    Ok((count, bytes))
}

/// What a finished run produced: how many replacements it made, how many bytes it read, and
/// the hash of what it wrote when one was asked for.
struct Produced {
    replacements: usize,
    bytes: usize,
    hash: Option<u64>,
}

impl Substitution<'_> {
    /// Write the result into `destination`, hashing it on the way out when asked — the hash
    /// of what was written is the token the next edit expects.
    fn write_to(&mut self, destination: &mut dyn Write) -> io::Result<Produced> {
        if !self.hashed {
            let (replacements, bytes) = self.emit(destination)?;
            return Ok(Produced {
                replacements,
                bytes,
                hash: None,
            });
        }
        let mut hashing = HashingWriter::new(destination);
        let (replacements, bytes) = self.emit(&mut hashing)?;
        Ok(Produced {
            replacements,
            bytes,
            hash: Some(hashing.digest()),
        })
    }

    fn emit(&mut self, destination: &mut dyn Write) -> io::Result<(usize, usize)> {
        let edit = &self.edit;
        match &mut self.source {
            Source::Whole(held) => {
                let data = held.as_bytes();
                match edit {
                    Edit::Substring {
                        pattern,
                        replacement,
                        ignore_case,
                        limit,
                    } => {
                        let count = replace_to(
                            data,
                            pattern,
                            replacement,
                            *ignore_case,
                            *limit,
                            destination,
                        )?;
                        Ok((count, data.len()))
                    }
                    // One pass, since the splices are already ordered and disjoint.
                    Edit::Lines(splices) => {
                        let mut cursor = 0;
                        for splice in splices {
                            destination.write_all(&data[cursor..splice.start])?;
                            destination.write_all(&splice.text)?;
                            cursor = splice.end;
                        }
                        destination.write_all(&data[cursor..])?;
                        Ok((splices.len(), data.len()))
                    }
                }
            }
            Source::Windows(refill) => {
                // A name is resolved against every candidate before anything is written, so
                // a run that addresses lines never reaches here without the whole input.
                let Edit::Substring {
                    pattern,
                    replacement,
                    ignore_case,
                    limit,
                } = edit
                else {
                    unreachable!("line addressing holds the whole input")
                };
                replace_stream(
                    refill,
                    pattern,
                    replacement,
                    *ignore_case,
                    *limit,
                    destination,
                )
            }
        }
    }
}

/// Write `text` as a whole line, taking `terminator` from the line it joins.
///
/// The rule that makes every line operation fall out of one replacement argument: an empty
/// replacement leaves nothing to terminate and so removes the line, a bare newline leaves an
/// empty line behind, and ordinary text is terminated the way its neighbours are — including
/// in a CRLF file, where writing the caller's bytes verbatim would leave the one line ending
/// in a bare LF.
fn as_line(text: &str, terminator: &[u8]) -> Vec<u8> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut written = joined(text, terminator);
    written.extend_from_slice(terminator);
    written
}

/// `text` with its own line breaks rendered as `terminator`, and none added at the end.
///
/// Separate from [`as_line`] because the two are different questions: an insertion after a
/// file's last unterminated line wants the breaks *inside* the text and none after it, and
/// asking `as_line` for that by passing an empty terminator would erase them instead.
fn joined(text: &str, terminator: &[u8]) -> Vec<u8> {
    let mut written = Vec::with_capacity(text.len() + terminator.len());
    for (index, piece) in text
        .strip_suffix('\n')
        .unwrap_or(text)
        .split('\n')
        .enumerate()
    {
        if index > 0 {
            written.extend_from_slice(terminator);
        }
        written.extend_from_slice(piece.strip_suffix('\r').unwrap_or(piece).as_bytes());
    }
    written
}

/// Resolve a line name into the splices that carry out the edit.
///
/// Every candidate is found before anything is written, so a name too short to be unique can
/// only ever cause a refused edit, never an edit to the wrong line.
fn resolve_lines(args: &Args, data: &[u8], path: &str) -> Result<Vec<Splice>, Failure> {
    let newlines = Newlines::from_utf8(args.utf8);
    let name = parse_hash_prefix(&args.pattern).expect("validated");
    let placement = match (args.after, args.before) {
        (true, _) => Placement::After,
        (_, true) => Placement::Before,
        _ => Placement::Over,
    };

    // Only a file's final line can lack a terminator, so the one it borrows is the file's
    // last — carried out of the pass that finds the candidates rather than costing a second.
    let mut prevailing: &[u8] = b"\n";
    let mut matched = Vec::new();
    for line in named_lines(data, newlines) {
        if !line.terminator().is_empty() {
            prevailing = line.terminator();
        }
        if name.matches(line.hash()) {
            matched.push(line);
        }
    }

    // A name that picks out nothing is always a mistake, whatever `--occurrences` says: a
    // substring may legitimately be absent, but a name is an address, and addressing a line
    // that is not there means the file has moved since the name was issued. Reporting it as
    // a no-op would hand back exit 0 and an unchanged file, which is the silent failure the
    // whole scheme exists to prevent.
    if matched.is_empty() {
        return Err(Failure::Unresolved {
            path: path.to_string(),
            subject: args.pattern.clone(),
            note: "names no line here; re-read the file",
        });
    }

    let limit = match args.occurrences {
        Occurrences::All => usize::MAX,
        Occurrences::First => 1,
        Occurrences::One if matched.len() == 1 => 1,
        Occurrences::One => {
            return Err(Failure::Ambiguous {
                path: path.to_string(),
                subject: args.pattern.clone(),
                matches: matched.len(),
                note: "ask sz-find for a longer name with --hash-width, \
                       or name a neighbouring line",
            })
        }
    };

    Ok(matched
        .into_iter()
        .take(limit)
        .map(|line| {
            // A rewrite ends the way the line it replaces ended, so the last line of a file
            // without a final newline still has none afterwards. An insertion needs a
            // terminator of its own to stand as a line, and borrows one when the line it is
            // anchored to has none to lend.
            let borrowed = if line.terminator().is_empty() {
                prevailing
            } else {
                line.terminator()
            };
            match placement {
                // The breaks *inside* the text are rendered with the terminator the file
                // uses; only the trailing one follows the line being replaced, which the
                // last line of a file may not have.
                Placement::Over => Splice {
                    start: line.offset,
                    end: line.end(),
                    text: if line.terminator().is_empty() {
                        joined(&args.replacement, borrowed)
                    } else {
                        as_line(&args.replacement, line.terminator())
                    },
                },
                // Past the line's terminator, so the new line follows it whole. A line that
                // has none is the file's last, and then the terminator goes in front of the
                // text instead of behind it, so the file still ends without one.
                Placement::After => Splice {
                    start: line.end(),
                    end: line.end(),
                    text: if line.terminator().is_empty() {
                        // The anchor is the file's last line and carries no terminator, so
                        // one goes in first to end it, and the new text carries none — which
                        // leaves the file ending as it did.
                        let mut written = borrowed.to_vec();
                        written.extend_from_slice(&joined(&args.replacement, borrowed));
                        written
                    } else {
                        as_line(&args.replacement, borrowed)
                    },
                },
                Placement::Before => Splice {
                    start: line.offset,
                    end: line.offset,
                    text: as_line(&args.replacement, borrowed),
                },
            }
        })
        .collect())
}

/// The summary of a finished run, in whichever shape was asked for.
struct Summary<'a> {
    path: &'a str,
    replacements: usize,
    dry_run: bool,
    /// What the input hashed to, and what the output did. Both are present exactly when the
    /// run was asked to report them.
    hashes: Option<(u64, u64)>,
}

/// Write the single summary record. The transformed bytes never share stdout with it,
/// because `--format json` demands a destination of its own.
fn write_summary_json(output: &mut dyn Write, summary: &Summary) -> io::Result<()> {
    output.write_all(br#"{"type":"summary","data":{"path":"#)?;
    json_text_field_to(output, summary.path.as_bytes())?;
    write!(
        output,
        r#","replacements":{},"dry_run":{}"#,
        summary.replacements, summary.dry_run
    )?;
    if let Some((before, after)) = summary.hashes {
        let (mut was, mut is) = ([0u8; HASH_CHARS], [0u8; HASH_CHARS]);
        write!(
            output,
            r#","hash_before":"{}","hash_after":"{}""#,
            format_hash(&mut was, before, HASH_CHARS),
            format_hash(&mut is, after, HASH_CHARS)
        )?;
    }
    output.write_all(b"}}\n")
}

/// Write the one-line text summary.
///
/// The caller hands it stderr. The transformed bytes are the run's output, whether they go
/// to a file, to stdout, or through a pipe, and a sentence about the run is not part of them
/// — `sz-replace --summary a b f > out.txt` must leave `out.txt` a file of text.
fn write_summary_text(output: &mut dyn Write, summary: &Summary) -> io::Result<()> {
    let verb = if summary.dry_run {
        "Would replace"
    } else {
        "Replaced"
    };
    write!(output, "{} {} occurrence(s)", verb, summary.replacements)?;
    if let Some((_, after)) = summary.hashes {
        let tense = if summary.dry_run { "would be" } else { "is" };
        let mut buffer = [0u8; HASH_CHARS];
        write!(
            output,
            "; content {tense} {}",
            format_hash(&mut buffer, after, HASH_CHARS)
        )?;
    }
    output.write_all(b"\n")
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    // Every byte this run prints goes here, so the records stay in one order.
    let mut output = stdout_writer();
    report("sz-replace", run(&args, &mut output, &mut io::stderr()))
}

/// The run's output and the notes about it are two different streams, and the caller passes
/// both: `output` carries what the run produced, `notes` carries what it has to say about
/// the run. Only the second may be prose, and only the second goes to stderr, so redirecting
/// stdout gives a file of data rather than data with a sentence appended.
fn run(args: &Args, output: &mut dyn Write, notes: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let name = args.input.as_deref().unwrap_or("-");
    let counting_only = args.dry_run || args.quiet;
    let reports_hashes = args.summary || args.dry_run || args.format == Format::Json;

    // A run reads an unbounded input into memory only when it must know something about the
    // whole of it before it may write, *and* its destination cannot take the write back.
    // Every other combination has a way out: a mapped file is already whole, and a temporary
    // can be discarded once the answer arrives at the end.
    // Whatever it writes to. An answer that depended on how the caller happened to open the
    // file would be worse than no answer, and a mapped file is unaffected either way.
    let input = if args.needs_whole_input() {
        get_input(args.input.as_deref())
    } else {
        get_input_streaming(args.input.as_deref())
    }
    .at(name)?;

    let pattern = args.pattern.as_bytes();
    let replacement = args.replacement.as_bytes();

    let source = match input.into_window(DEFAULT_WINDOW_BYTES) {
        InputWindow::Whole(held) => Source::Whole(held),
        // Hashed as it arrives, before anything reads it, so a streamed run can still report
        // what it read without holding it.
        InputWindow::Stream(mut refill) => {
            if reports_hashes {
                refill.hash_stream();
            }
            Source::Windows(refill)
        }
    };

    // Settled before any destination is opened, so a refused edit has nothing to retract;
    // and the fast path pays for a hash only when one will be reported.
    let mapped_hash = source
        .whole()
        .filter(|_| args.expect_hash.is_some() || reports_hashes)
        .map(content_hash);
    if let Some((expected, actual)) = args.expect_hash.zip(mapped_hash) {
        if actual != expected {
            return Err(Failure::Stale {
                path: name.to_string(),
                expected,
                actual,
            });
        }
    }

    let edit = match source.whole() {
        Some(data) if args.match_kind == Match::LineHash => {
            Edit::Lines(resolve_lines(args, data, name)?)
        }
        Some(data) => Edit::Substring {
            pattern,
            replacement,
            ignore_case: args.ignore_case,
            limit: substring_limit(args, data, name)?,
        },
        // A window is only ever handed over when nothing about the whole input is needed, so
        // there is nothing to count before the limit is known. If that ever stopped being
        // true the modes below would quietly answer a different question — a name would
        // become a substring, `one` would become `first` — so it fails loudly instead.
        None => {
            assert!(
                !args.needs_whole_input(),
                "a run that needs the whole input was handed a window"
            );
            Edit::Substring {
                pattern,
                replacement,
                ignore_case: args.ignore_case,
                limit: match args.occurrences {
                    Occurrences::All => usize::MAX,
                    Occurrences::First | Occurrences::One => 1,
                },
            }
        }
    };

    let mut substitution = Substitution {
        source,
        edit,
        hashed: reports_hashes,
    };

    let produced = if counting_only {
        // Discarding the output still reads the input, so this reports a failed read rather
        // than panicking on it, as its two siblings do.
        substitution.write_to(&mut io::sink()).at(name)?
    } else if args.in_place {
        let path = args.input.as_deref().expect("validated");
        write_replacing("sz-replace", path, |output| substitution.write_to(output))?
    } else if let Some(path) = args.output.as_deref().filter(|path| *path != "-") {
        // Through a temporary like `--in-place`, so an interrupted run leaves the previous
        // file rather than a half-written one — and so naming the input as the output does
        // not truncate the mapping this run is still reading from.
        write_creating("sz-replace", path, |output| substitution.write_to(output))?
    } else {
        // Straight to stdout — no full-output buffer. A pipe closing during the flush ends
        // the run where one closing during the write does.
        substitution.write_to(output).at("-")?
    };

    // A mapped run hashed its input up front; a streamed one accumulated it while reading.
    let hash_before = mapped_hash.or_else(|| match &substitution.source {
        Source::Windows(refill) => refill.digest(),
        Source::Whole(_) => None,
    });
    let summary = Summary {
        path: name,
        replacements: produced.replacements,
        dry_run: args.dry_run,
        hashes: hash_before.zip(produced.hash),
    };
    if args.format == Format::Json {
        write_summary_json(output, &summary).at("-")?;
    } else if args.summary || args.dry_run {
        write_summary_text(notes, &summary).at("-")?;
    }

    output.flush().at("-")?;

    // The stream was produced whether or not it changed; counting alone reports matches.
    Ok(Status::from_found(if counting_only {
        produced.replacements > 0
    } else {
        produced.bytes > 0
    }))
}

/// How many matches the substring mode acts on, refusing where `--occurrences one` asserts
/// more than the file holds.
fn substring_limit(args: &Args, data: &[u8], path: &str) -> Result<usize, Failure> {
    Ok(match args.occurrences {
        Occurrences::All => usize::MAX,
        Occurrences::First => 1,
        // Exactly one, in both directions. Refusing only the ambiguous side would let the
        // assertion pass on a pattern that matches nothing, which is the silent no-op the
        // mode exists to prevent, reached through the other door.
        Occurrences::One => match count_matches(data, args.pattern.as_bytes(), args.ignore_case) {
            1 => 1,
            0 => {
                return Err(Failure::Unresolved {
                    path: path.to_string(),
                    subject: args.pattern.clone(),
                    note: "matches nothing, and --occurrences one asserts exactly one",
                })
            }
            matches => {
                return Err(Failure::Ambiguous {
                    path: path.to_string(),
                    subject: args.pattern.clone(),
                    matches,
                    note: "extend the pattern until it is unique, \
                           or pass --occurrences all or first",
                })
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    /// Test helper: replace every match into a buffer and return it.
    fn replace_all(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        ignore_case: bool,
    ) -> (Vec<u8>, usize) {
        replace_limited(data, pattern, replacement, ignore_case, usize::MAX)
    }

    /// Test helper: the same, stopping after `limit` matches.
    fn replace_limited(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        ignore_case: bool,
        limit: usize,
    ) -> (Vec<u8>, usize) {
        let mut buf = Vec::new();
        let count = replace_to(data, pattern, replacement, ignore_case, limit, &mut buf).unwrap();
        (buf, count)
    }

    /// Test helper: drive a whole run, returning its outcome, what it produced, and what it
    /// had to say about the run — the two streams kept apart, as a caller's shell keeps them.
    ///
    /// Paths are passed absolute rather than by changing directory, since the working
    /// directory is process-wide and these tests run in parallel.
    fn run_with(flags: &[&str]) -> (Result<Status, Failure>, Vec<u8>, Vec<u8>) {
        let mut argv = vec!["sz-replace"];
        argv.extend_from_slice(flags);
        let args = Args::try_parse_from(argv).expect("flags must parse");
        let (mut output, mut notes) = (Vec::new(), Vec::new());
        let outcome = run(&args, &mut output, &mut notes);
        (outcome, output, notes)
    }

    /// Test helper: the token `--expect-hash` accepts for a file's current contents.
    fn token_of(path: &Path) -> String {
        let mut buffer = [0u8; HASH_CHARS];
        let hash = content_hash(&fs::read(path).unwrap());
        format_hash(&mut buffer, hash, HASH_CHARS).to_string()
    }

    #[test]
    fn replaces_all_occurrences() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi world hi");
    }

    #[test]
    fn leaves_text_unchanged_without_matches() {
        let data = b"hello world";
        let (result, count) = replace_all(data, b"foo", b"bar", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello world");
    }

    #[test]
    fn replaces_all_ignoring_case() {
        let data = b"Hello HELLO hello";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn ignores_empty_pattern() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"", b"x", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn replaces_with_longer_string() {
        let data = b"a b a";
        let (result, count) = replace_all(data, b"a", b"foo", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"foo b foo");
    }

    #[test]
    fn replaces_with_shorter_string() {
        let data = b"hello hello";
        let (result, count) = replace_all(data, b"hello", b"hi", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"hi hi");
    }

    #[test]
    fn replaces_non_overlapping_matches() {
        let data = b"aaa";
        let (result, count) = replace_all(data, b"aa", b"b", false);

        // Should replace first match, then continue after it (non-overlapping)
        assert_eq!(count, 1);
        assert_eq!(result, b"ba");
    }

    #[test]
    fn skips_overlapping_matches_in_repeating_pattern() {
        // Matching is leftmost-first and non-overlapping, as in `sed s///g`: the "ABA"
        // starting at offset 2 overlaps the one claimed at offset 0, so it is not a match.
        let data = b"ABABAB";
        let (result, count) = replace_all(data, b"ABA", b"-", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"-BAB");
    }

    #[test]
    fn skips_overlapping_matches_across_lines() {
        // The same rule with a newline inside the pattern — nothing about line
        // boundaries makes the second, overlapping "a\na" eligible.
        let data = b"a\na\na";
        let (result, count) = replace_all(data, b"a\na", b"X", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"X\na");
    }

    #[test]
    fn does_not_rescan_the_replacement() {
        // Output is written past the cursor and never re-examined, so a replacement
        // containing the pattern substitutes once rather than looping forever.
        let data = b"a b";
        let (result, count) = replace_all(data, b"a", b"aa", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"aa b");
    }

    #[test]
    fn takes_no_utf8_flag_without_a_line_to_split() {
        // Substring replacement is byte-level, with no line or codepoint semantics to
        // switch, and `--ignore-case` already folds the full Unicode range — so the newline
        // set only means something once the pattern names a line.
        assert!(!accepts(&["--utf8"]));
        assert!(accepts_pattern(
            "6vzvxbws",
            &["--match", "line-hash", "--utf8"]
        ));
    }

    /// Parse and then apply the value-conditional checks, as `run` does.
    fn accepts(flags: &[&str]) -> bool {
        accepts_pattern("a", flags)
    }

    /// The same, with a pattern the caller chooses — a line name has to be well formed
    /// before the flags around it can be judged.
    fn accepts_pattern(pattern: &str, flags: &[&str]) -> bool {
        let arguments = ["sz-replace", pattern, "b", "f"]
            .into_iter()
            .chain(flags.iter().copied());
        Args::try_parse_from(arguments).is_ok_and(|args| validate(&args).is_ok())
    }

    #[test]
    fn declares_no_short_flags() {
        let mut command = Args::command();
        command.build();
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
                "output",
                "in-place",
                "dry-run",
                "expect-hash",
                "match",
                "after",
                "before",
                "utf8",
                "occurrences",
                "ignore-case",
                "format",
                "summary",
                "quiet",
                "help",
                "version",
            ]
        );
    }

    #[test]
    fn declares_the_conflicts_that_used_to_pass_silently() {
        assert!(accepts(&["--dry-run"]));
        for flags in [
            vec!["--dry-run", "--output", "o"],
            vec!["--dry-run", "--in-place"],
            vec!["--in-place", "--output", "o"],
            vec!["--quiet", "--summary"],
            vec!["--quiet", "--in-place"],
            vec!["--dry-run", "--summary"],
        ] {
            assert!(!accepts(&flags), "expected {:?} to be rejected", flags);
        }
    }

    #[test]
    fn keeps_the_json_summary_off_the_transformed_stream() {
        assert!(!accepts(&["--format", "json"]));
        assert!(!accepts(&["--format", "json", "--output", "-"]));
        assert!(accepts(&["--format", "json", "--dry-run"]));
        assert!(accepts(&["--format", "json", "--output", "o"]));
        assert!(
            accepts(&["--format", "json", "--output", "o", "--summary"]),
            "--summary names the record json already emits"
        );
    }

    #[test]
    fn accepts_a_precondition_alongside_every_destination() {
        // Verifying a token is orthogonal to where the bytes go, and `--dry-run` verifying
        // one without writing is the cheapest probe an agent has.
        for flags in [
            vec!["--expect-hash", "0000006ynpzey"],
            vec!["--expect-hash", "0000006ynpzey", "--dry-run"],
            vec!["--expect-hash", "0000006ynpzey", "--quiet"],
            vec!["--expect-hash", "0000006ynpzey", "--in-place"],
            vec!["--expect-hash", "0000006ynpzey", "--output", "o"],
            vec!["--occurrences", "one"],
            vec!["--occurrences", "first", "--in-place"],
        ] {
            assert!(accepts(&flags), "expected {:?} to be accepted", flags);
        }
    }

    #[test]
    fn refuses_a_token_that_is_not_a_whole_hash() {
        // Zero-extending a truncated paste would compare against a different file, and a
        // line name is a prefix of a whole one, so the parser refuses both before the run
        // begins rather than after it has written.
        for token in ["", "6ynpzey", "0000006ynpzeyy", "0000006ynpzez"] {
            let parsed =
                Args::try_parse_from(["sz-replace", "a", "b", "f", "--expect-hash", token]);
            assert!(parsed.is_err(), "`{token}` must not parse");
        }
        assert!(Args::try_parse_from([
            "sz-replace",
            "a",
            "b",
            "f",
            "--expect-hash",
            "0000006YNPZEY"
        ])
        .is_ok());
    }

    #[test]
    fn replaces_only_the_leftmost_match_under_first() {
        let data = b"a b a b a";
        let (result, count) = replace_limited(data, b"a", b"X", false, 1);

        assert_eq!(count, 1);
        assert_eq!(result, b"X b a b a");
    }

    #[test]
    fn counts_the_matches_it_would_rewrite() {
        // The count `--occurrences one` decides on has to be the same one the rewrite makes,
        // or a run refuses what it could have done or does what it should have refused.
        for (data, pattern, ignore_case) in [
            (&b"a b a b a"[..], &b"a"[..], false),
            (&b"aaa"[..], &b"aa"[..], false),
            (&b"ABABAB"[..], &b"ABA"[..], false),
            ("xẞẞy".as_bytes(), &b"ss"[..], true),
            (&b"nothing here"[..], &b"zzz"[..], false),
        ] {
            let (_, replaced) = replace_all(data, pattern, b".", ignore_case);
            assert_eq!(
                count_matches(data, pattern, ignore_case),
                replaced,
                "counting disagreed with rewriting on {:?}",
                String::from_utf8_lossy(data)
            );
        }
    }

    #[test]
    fn replaces_multibyte_utf8_pattern() {
        let data = "héllo wörld héllo".as_bytes();
        let (result, count) = replace_all(data, "héllo".as_bytes(), "hi".as_bytes(), false);

        assert_eq!(count, 2);
        assert_eq!(result, "hi wörld hi".as_bytes());
    }

    #[test]
    fn replaces_at_start_and_end() {
        // Pattern at start
        let data = b"hello world";
        let (result, count) = replace_all(data, b"hello", b"hi", false);
        assert_eq!(count, 1);
        assert_eq!(result, b"hi world");

        // Pattern at end
        let data = b"hello world";
        let (result, count) = replace_all(data, b"world", b"there", false);
        assert_eq!(count, 1);
        assert_eq!(result, b"hello there");
    }

    #[test]
    fn replaces_whole_input() {
        let data = b"hello";
        let (result, count) = replace_all(data, b"hello", b"goodbye", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"goodbye");
    }

    #[test]
    fn deletes_pattern_with_empty_replacement() {
        let data = b"hello world hello";
        let (result, count) = replace_all(data, b"hello", b"", false);

        assert_eq!(count, 2);
        assert_eq!(result, b" world ");
    }

    #[test]
    fn keeps_literal_replacement_when_ignoring_case() {
        // Case insensitive finds, but replacement is literal
        let data = b"HELLO hello HeLLo";
        let (result, count) = replace_all(data, b"hello", b"hi", true);

        assert_eq!(count, 3);
        assert_eq!(result, b"hi hi hi");
    }

    #[test]
    fn replaces_single_character() {
        let data = b"a b a c a";
        let (result, count) = replace_all(data, b"a", b"x", false);

        assert_eq!(count, 3);
        assert_eq!(result, b"x b x c x");
    }

    #[test]
    fn replaces_on_every_line_of_a_multi_line_input() {
        // A single-line pattern, matched independently on each line of the input.
        let data = b"line1\nline2\nline1\n";
        let (result, count) = replace_all(data, b"line1", b"first", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"first\nline2\nfirst\n");
    }

    #[test]
    fn replaces_pattern_spanning_lines() {
        // Replacement is a byte-level substitution with no line concept, so a newline
        // inside the pattern is matched like any other byte and the terminator it
        // straddles is consumed with it.
        let data = b"alpha\nbeta\ngamma\n";
        let (result, count) = replace_all(data, b"alpha\nbeta", b"X", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"X\ngamma\n");
    }

    #[test]
    fn replaces_every_occurrence_of_a_pattern_spanning_lines() {
        let data = b"a\nb\nc\na\nb\n";
        let (result, count) = replace_all(data, b"a\nb", b"X", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"X\nc\nX\n");
    }

    #[test]
    fn replaces_pattern_spanning_lines_ignoring_case() {
        let data = b"ALPHA\nBeta\n";
        let (result, count) = replace_all(data, b"alpha\nbeta", b"X", true);

        assert_eq!(count, 1);
        assert_eq!(result, b"X\n");
    }

    #[test]
    fn expands_one_line_into_several() {
        // The replacement is written verbatim, so newlines in it split the line.
        let data = b"alpha\nbeta\n";
        let (result, count) = replace_all(data, b"beta", b"B1\nB2", false);

        assert_eq!(count, 1);
        assert_eq!(result, b"alpha\nB1\nB2\n");
    }

    #[test]
    fn joins_several_lines_into_one() {
        let data = b"alpha\nbeta\n";
        let (result, count) = replace_all(data, b"\n", b" ", false);

        assert_eq!(count, 2);
        assert_eq!(result, b"alpha beta ");
    }

    #[test]
    fn does_not_match_crlf_with_an_lf_pattern() {
        // Matching is literal, so a pattern written with LF does not span a CRLF
        // terminator. This is the behaviour, not an oversight — `\r` is a byte like
        // any other, and silently ignoring it would make the tool non-literal.
        let data = b"a\r\nb";
        let (result, count) = replace_all(data, b"a\nb", b"X", false);

        assert_eq!(count, 0);
        assert_eq!(result, b"a\r\nb");
    }

    #[test]
    fn advances_by_the_folded_match_length_when_it_is_longer() {
        // Case folding makes the matched span longer than the pattern: "ss" folds
        // "ẞ" (U+1E9E, 3 bytes). Advancing by `pattern.len()` instead would leave the
        // trailing continuation byte of the first ẞ in the output and desynchronize
        // every later match, so this is what pins that line of `replace_all_to`.
        let data = "xẞẞy".as_bytes();
        let (result, count) = replace_all(data, b"ss", b".", true);

        assert_eq!(count, 2);
        assert_eq!(result, b"x..y");
    }

    #[test]
    fn advances_by_the_folded_match_length_when_it_is_shorter() {
        // And the other direction: a 3-byte pattern matching a 2-byte span.
        let data = b"xssy";
        let (result, count) = replace_all(data, "ẞ".as_bytes(), b".", true);

        assert_eq!(count, 1);
        assert_eq!(result, b"x.y");
    }

    #[test]
    fn folds_a_one_byte_pattern_onto_a_longer_match() {
        // Turkish dotted capital I (U+0130, 2 bytes) folds to "i" plus a combining dot,
        // so a one-byte pattern claims a two-byte span.
        let data = "xİİy".as_bytes();
        let (result, count) = replace_all(data, b"i", b".", true);

        assert_eq!(count, 2);
        assert_eq!(result, b"x..y");
    }

    #[test]
    fn folds_a_ligature_onto_its_ascii_expansion() {
        let data = "xﬁy".as_bytes();
        let (result, count) = replace_all(data, b"fi", b".", true);

        assert_eq!(count, 1);
        assert_eq!(result, b"x.y");
    }

    #[test]
    fn takes_the_pattern_literally() {
        // No escape processing on the arguments: a backslash-n typed at the shell is
        // two bytes, and a newline in the pattern has to arrive as one (`$'a\nb'`).
        let args = Args::try_parse_from(["sz-replace", r"a\nb", r"c\td", "file.txt"])
            .expect("literal backslashes must parse");

        assert_eq!(args.pattern, r"a\nb");
        assert_eq!(args.replacement, r"c\td");
    }

    /// Test helper: the name `sz-find --fields line-hashes` prints for a whole line.
    fn name_of(line: &[u8]) -> String {
        let mut buffer = [0u8; HASH_CHARS];
        format_hash(&mut buffer, content_hash(line), DEFAULT_HASH_WIDTH).to_string()
    }

    /// Test helper: apply a line-addressed edit to `content`, returning what it wrote.
    fn line_edit(content: &[u8], name: &str, replacement: &str, flags: &[&str]) -> Vec<u8> {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, content).unwrap();
        let file = path.to_str().unwrap().to_string();
        let mut argv = vec!["--match", "line-hash"];
        argv.extend_from_slice(flags);
        argv.extend_from_slice(&[name, replacement, &file]);
        let (outcome, printed, _notes) = run_with(&argv);
        outcome.expect("the edit applies");
        printed
    }

    /// Test helper: replace through a window of `capacity` bytes, as a pipe would.
    fn replace_windowed(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        limit: usize,
        capacity: usize,
    ) -> (Vec<u8>, usize) {
        replace_windowed_folding(data, pattern, replacement, false, limit, capacity)
    }

    /// The same, with folding, which widens what a window has to see past what it emits.
    fn replace_windowed_folding(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        ignore_case: bool,
        limit: usize,
        capacity: usize,
    ) -> (Vec<u8>, usize) {
        let mut refill = Refill::new(data, capacity);
        let mut written = Vec::new();
        let (count, bytes) = replace_stream(
            &mut refill,
            pattern,
            replacement,
            ignore_case,
            limit,
            &mut written,
        )
        .unwrap();
        assert_eq!(bytes, data.len(), "the stream read every byte");
        (written, count)
    }

    #[test]
    fn holds_the_input_for_every_question_a_window_cannot_answer() {
        // The three modes below once depended on how the caller opened the file: a pipe took
        // the streaming path, where the whole input is unavailable, and each fell through to
        // a default that answered a different question and exited 0. Asserted against the
        // method `run` itself reads, so a copy cannot drift from it.
        let needs = |flags: &[&str]| {
            let mut argv = vec!["sz-replace", "old", "new"];
            argv.extend_from_slice(flags);
            Args::try_parse_from(argv)
                .expect("flags must parse")
                .needs_whole_input()
        };

        for flags in [
            vec!["--expect-hash", "0000006ynpzey"],
            vec!["--occurrences", "one"],
            vec!["--match", "line-hash"],
        ] {
            assert!(needs(&flags), "{flags:?} must hold the input");
            // And still does once the destination could have been discarded, which is what
            // the escape hatch keyed on.
            for sink in [
                vec!["--output", "o.txt"],
                vec!["--dry-run"],
                vec!["--quiet"],
                vec!["--in-place"],
            ] {
                let mut both = flags.clone();
                both.extend_from_slice(&sink);
                assert!(needs(&both), "{both:?} must hold the input");
            }
        }

        // And the questions a window can answer are still answered from one.
        assert!(!needs(&[]));
        assert!(!needs(&["--summary"]));
        assert!(!needs(&["--ignore-case"]));
        assert!(!needs(&["--occurrences", "first"]));
    }

    #[test]
    fn reports_the_same_hash_streamed_as_mapped() {
        // A streamed run accumulates the hash of its input while reading it, so the token it
        // reports is the token the mapped run would have reported.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"alpha\nbeta\n").unwrap();

        let (outcome, _printed, notes) =
            run_with(&["beta", "BETA", path.to_str().unwrap(), "--summary"]);
        outcome.unwrap();
        let text = String::from_utf8(notes).unwrap();
        let mut buffer = [0u8; HASH_CHARS];
        let after = format_hash(&mut buffer, content_hash(b"alpha\nBETA\n"), HASH_CHARS);
        assert!(text.contains(after), "{text}");
    }

    #[test]
    fn streams_a_match_that_straddles_a_window_seam() {
        // The property the carry exists for. At capacity 2 a three-byte pattern cannot fit
        // in one window at all, so every match crosses a seam.
        for capacity in 1..=12 {
            let (written, count) =
                replace_windowed(b"xxabcxxabc", b"abc", b"<>", usize::MAX, capacity);
            assert_eq!(written, b"xx<>xx<>", "capacity {capacity}");
            assert_eq!(count, 2, "capacity {capacity}");
        }
    }

    #[test]
    fn streams_the_same_bytes_the_whole_buffer_writes() {
        // Exhaustive over short inputs and patterns rather than sampled: a seam bug shows up
        // only when a match happens to sit on one, and which offsets those are depends on
        // the window size.
        let corpus: &[&[u8]] = &[
            b"",
            b"a",
            b"aaaa",
            b"abab",
            b"ababab",
            b"xxabcxxabcxx",
            b"aaa
aaa
",
            b"the quick brown fox",
            b"abcabcabcabc",
        ];
        for data in corpus {
            for pattern in [&b"a"[..], b"ab", b"abc", b"aaa", b"zzz"] {
                let (expected, wanted) = replace_all(data, pattern, b"<>", false);
                for capacity in 1..=16 {
                    let (written, count) =
                        replace_windowed(data, pattern, b"<>", usize::MAX, capacity);
                    assert_eq!(
                        written, expected,
                        "{data:?} / {pattern:?} at capacity {capacity}"
                    );
                    assert_eq!(count, wanted);
                }
            }
        }
    }

    #[test]
    fn bounds_a_folded_match_by_what_the_pattern_folds_to() {
        // A match spans at most four times the folded pattern: every character is at most
        // four UTF-8 bytes and folds to at least one. Checked against the widest real
        // expansions rather than only against the arithmetic.
        assert_eq!(longest_match(b"abc", false), 3);
        for (pattern, haystack) in [
            (&b"ss"[..], "ẞ".as_bytes()),
            (&b"fi"[..], "ﬁ".as_bytes()),
            (&b"ffi"[..], "ﬃ".as_bytes()),
            (&b"i"[..], "İ".as_bytes()),
            (&b"k"[..], "\u{212a}".as_bytes()),
        ] {
            assert!(
                haystack.len() <= longest_match(pattern, true),
                "{:?} matched {} bytes, past the bound of {}",
                String::from_utf8_lossy(pattern),
                haystack.len(),
                longest_match(pattern, true)
            );
        }
    }

    #[test]
    fn streams_a_folded_match_that_straddles_a_window_seam() {
        // The case that used to hold the whole input: a match wider than its pattern, cut in
        // half by a window boundary.
        let data = "xxẞxxﬃx".as_bytes();
        let (expected, wanted) = replace_all(data, b"ss", b"<>", true);
        for capacity in 1..=20 {
            let (written, count) =
                replace_windowed_folding(data, b"ss", b"<>", true, usize::MAX, capacity);
            assert_eq!(written, expected, "capacity {capacity}");
            assert_eq!(count, wanted, "capacity {capacity}");
        }
    }

    #[test]
    fn streams_folded_matches_the_same_as_the_whole_buffer() {
        let corpus: &[&str] = &[
            "",
            "ẞ",
            "xẞx",
            "ssẞss",
            "ẞẞẞ",
            "aﬃb",
            "İİ",
            "straße Straße STRASSE",
        ];
        for data in corpus {
            for pattern in [&b"ss"[..], b"i", b"fi", b"strasse"] {
                let (expected, wanted) = replace_all(data.as_bytes(), pattern, b"<>", true);
                for capacity in 1..=24 {
                    let (written, count) = replace_windowed_folding(
                        data.as_bytes(),
                        pattern,
                        b"<>",
                        true,
                        usize::MAX,
                        capacity,
                    );
                    assert_eq!(
                        written,
                        expected,
                        "{data:?} / {:?} at capacity {capacity}",
                        String::from_utf8_lossy(pattern)
                    );
                    assert_eq!(count, wanted);
                }
            }
        }
    }

    #[test]
    fn stops_a_streamed_run_at_the_limit_it_was_given() {
        for capacity in 1..=8 {
            let (written, count) = replace_windowed(b"aXaXaX", b"a", b"<>", 1, capacity);
            assert_eq!(written, b"<>XaXaX", "capacity {capacity}");
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn rewrites_deletes_and_blanks_from_one_replacement_argument() {
        // Three operations, no flags: the terminator is part of what a name covers, so an
        // empty replacement leaves nothing to terminate and a bare newline leaves a line
        // holding nothing.
        let seed = b"alpha\nbeta\ngamma\n";
        let beta = name_of(b"beta\n");
        assert_eq!(line_edit(seed, &beta, "BETA", &[]), b"alpha\nBETA\ngamma\n");
        assert_eq!(line_edit(seed, &beta, "", &[]), b"alpha\ngamma\n");
        assert_eq!(line_edit(seed, &beta, "\n", &[]), b"alpha\n\ngamma\n");
    }

    #[test]
    fn joins_an_inserted_multi_line_replacement_at_the_end_of_a_file() {
        // The anchor is the file's last line and carries no terminator, so one goes in
        // front of the text and none behind it — but the breaks *inside* the text still
        // have to survive, which asking for "no terminator" once erased.
        assert_eq!(
            line_edit(b"alpha\nbeta", &name_of(b"beta"), "X\nY", &["--after"]),
            b"alpha\nbeta\nX\nY"
        );
    }

    #[test]
    fn borrows_the_terminator_the_file_ends_with() {
        // A file may mix terminators; the one an unterminated last line borrows is the one
        // nearest it, not the first the file happens to use.
        assert_eq!(
            line_edit(
                b"alpha\r\nbeta\ngamma",
                &name_of(b"gamma"),
                "NEW",
                &["--after"]
            ),
            b"alpha\r\nbeta\ngamma\nNEW"
        );
    }

    #[test]
    fn refuses_an_insertion_with_nothing_to_insert() {
        // Inert by construction everywhere but a file's last unterminated line, where it
        // would quietly add a terminator instead of doing nothing.
        for placement in ["--after", "--before"] {
            let refused = Args::try_parse_from([
                "sz-replace",
                "--match",
                "line-hash",
                placement,
                "6vzvxbws",
                "",
                "f",
            ])
            .expect("an empty replacement parses");
            assert!(
                validate(&refused).is_err(),
                "{placement} with an empty replacement must be refused"
            );
            // The same argument without a placement is how a line is deleted.
            let accepted =
                Args::try_parse_from(["sz-replace", "--match", "line-hash", "6vzvxbws", "", "f"])
                    .unwrap();
            assert!(validate(&accepted).is_ok());
        }
    }

    #[test]
    fn places_a_new_line_on_the_side_it_was_asked_for() {
        let seed = b"alpha\nbeta\n";
        let alpha = name_of(b"alpha\n");
        assert_eq!(
            line_edit(seed, &alpha, "MIDDLE", &["--after"]),
            b"alpha\nMIDDLE\nbeta\n"
        );
        assert_eq!(
            line_edit(seed, &name_of(b"beta\n"), "MIDDLE", &["--before"]),
            b"alpha\nMIDDLE\nbeta\n"
        );
    }

    #[test]
    fn keeps_the_terminator_the_line_already_had() {
        // Copied, never rebuilt, so an edited line in a CRLF file does not become the one
        // line in it ending with a bare LF — including when the replacement spans lines.
        let seed = b"alpha\r\nbeta\r\ngamma\r\n";
        let beta = name_of(b"beta\r\n");
        assert_eq!(
            line_edit(seed, &beta, "BETA", &[]),
            b"alpha\r\nBETA\r\ngamma\r\n"
        );
        assert_eq!(
            line_edit(seed, &beta, "B1\nB2", &[]),
            b"alpha\r\nB1\r\nB2\r\ngamma\r\n"
        );
        assert_eq!(line_edit(seed, &beta, "", &[]), b"alpha\r\ngamma\r\n");
    }

    #[test]
    fn names_a_crlf_line_the_same_under_either_newline_set() {
        // The name covers the terminator, and a line spans to the start of the next one in
        // both sets, so a name issued by one reading resolves under the other.
        let seed = b"alpha\r\nbeta\r\n";
        let beta = name_of(b"beta\r\n");
        for flags in [&[][..], &["--utf8"][..]] {
            assert_eq!(
                line_edit(seed, &beta, "BETA", flags),
                b"alpha\r\nBETA\r\n",
                "under {flags:?}"
            );
        }
    }

    #[test]
    fn keeps_the_breaks_inside_a_replacement_for_a_file_with_no_final_newline() {
        // The last line of a file has no terminator to lend, and asking for "no terminator"
        // once erased the replacement's own line breaks as well as the trailing one.
        assert_eq!(
            line_edit(b"alpha\nbeta", &name_of(b"beta"), "B1\nB2", &[]),
            b"alpha\nB1\nB2"
        );
    }

    #[test]
    fn leaves_a_file_without_a_final_newline_without_one() {
        let seed = b"alpha\nbeta";
        let beta = name_of(b"beta");
        assert_eq!(line_edit(seed, &beta, "BETA", &[]), b"alpha\nBETA");
        assert_eq!(line_edit(seed, &beta, "", &[]), b"alpha\n");
        // An insertion needs a terminator of its own, and borrows the file's; the file
        // still ends without one.
        assert_eq!(
            line_edit(seed, &beta, "gamma", &["--after"]),
            b"alpha\nbeta\ngamma"
        );
        assert_eq!(
            line_edit(seed, &beta, "mid", &["--before"]),
            b"alpha\nmid\nbeta"
        );
    }

    #[test]
    fn names_the_same_content_wherever_it_sits() {
        // A name identifies content, so identical lines share one — and `--occurrences`
        // decides what happens to the set, exactly as it does for a substring.
        let seed = b"dup\nother\ndup\n";
        let dup = name_of(b"dup\n");
        assert_eq!(line_edit(seed, &dup, "X", &[]), b"X\nother\nX\n");
        assert_eq!(
            line_edit(seed, &dup, "X", &["--occurrences", "first"]),
            b"X\nother\ndup\n"
        );
    }

    #[test]
    fn refuses_an_ambiguous_or_absent_name_without_writing() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"dup\nother\ndup\n").unwrap();
        let file = path.to_str().unwrap();

        let (outcome, printed, _notes) = run_with(&[
            "--match",
            "line-hash",
            "--occurrences",
            "one",
            "--in-place",
            &name_of(b"dup\n"),
            "X",
            file,
        ]);
        let Err(Failure::Ambiguous { matches, .. }) = outcome else {
            panic!("a name matching two lines must be refused under --occurrences one");
        };
        assert_eq!(matches, 2);
        assert!(printed.is_empty());

        let (outcome, _, _notes) = run_with(&[
            "--match",
            "line-hash",
            "--occurrences",
            "one",
            "--in-place",
            "00000000",
            "X",
            file,
        ]);
        assert!(matches!(outcome, Err(Failure::Unresolved { .. })));
        assert_eq!(fs::read(&path).unwrap(), b"dup\nother\ndup\n");
    }

    #[test]
    fn a_short_name_can_only_refuse_never_mis_target() {
        // Every candidate is found before a byte is written, so a name too short to be
        // unique refuses rather than picking one.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"alpha\nbeta\n").unwrap();
        let file = path.to_str().unwrap();
        let four = &name_of(b"beta\n")[..HASH_CHARS_MIN];

        let (outcome, _, _notes) = run_with(&[
            "--match",
            "line-hash",
            "--occurrences",
            "one",
            "--in-place",
            four,
            "BETA",
            file,
        ]);
        match outcome {
            Ok(_) => assert_eq!(fs::read(&path).unwrap(), b"alpha\nBETA\n"),
            Err(_) => assert_eq!(fs::read(&path).unwrap(), b"alpha\nbeta\n"),
        }
    }

    #[test]
    fn a_name_survives_an_edit_elsewhere_where_a_line_number_would_not() {
        // The property that justifies naming lines by content at all.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"alpha\nbeta\ngamma\n").unwrap();
        let file = path.to_str().unwrap();
        let gamma = name_of(b"gamma\n");

        // Two insertions above it. Its line number is now 5; its name has not moved.
        for text in ["one", "two"] {
            let (outcome, _, _notes) = run_with(&[
                "--in-place",
                "--match",
                "line-hash",
                "--after",
                &name_of(b"alpha\n"),
                text,
                file,
            ]);
            outcome.unwrap();
        }
        let (outcome, _, _notes) = run_with(&[
            "--in-place",
            "--match",
            "line-hash",
            "--occurrences",
            "one",
            &gamma,
            "GAMMA",
            file,
        ]);
        outcome.unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"alpha\ntwo\none\nbeta\nGAMMA\n");
    }

    #[test]
    fn a_name_stops_resolving_once_its_line_has_changed() {
        // The name is its own precondition: nothing else has to guard the line it points at.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.txt");
        fs::write(&path, b"alpha\nbeta\n").unwrap();
        let file = path.to_str().unwrap();
        let beta = name_of(b"beta\n");

        let edit = |text: &str| {
            run_with(&[
                "--in-place",
                "--match",
                "line-hash",
                "--occurrences",
                "one",
                &beta,
                text,
                file,
            ])
        };
        edit("done").0.unwrap();
        let (outcome, _, _notes) = edit("again");

        assert!(matches!(outcome, Err(Failure::Unresolved { .. })));
        assert_eq!(fs::read(&path).unwrap(), b"alpha\ndone\n");
    }

    #[test]
    fn chains_two_edits_by_the_reported_hash() {
        // The whole point of reporting a hash: the token an edit hands back is the token the
        // next one passes, so a sequence of edits costs one read rather than one per edit.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\ngamma\n").unwrap();
        let file = path.to_str().unwrap();

        let first = token_of(&path);
        let (outcome, printed, _notes) = run_with(&[
            "beta",
            "BETA",
            file,
            "--in-place",
            "--expect-hash",
            &first,
            "--format",
            "json",
        ]);
        assert_eq!(outcome.unwrap(), Status::Success);
        let second = String::from_utf8(printed)
            .unwrap()
            .split(r#""hash_after":""#)
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("the record names the hash it wrote")
            .to_string();
        assert_eq!(second, token_of(&path), "the reported hash is the file's");

        // The chained token still holds, so the second edit lands without a re-read.
        let (outcome, _, _notes) = run_with(&[
            "gamma",
            "GAMMA",
            file,
            "--in-place",
            "--expect-hash",
            &second,
        ]);
        assert_eq!(outcome.unwrap(), Status::Success);
        assert_eq!(fs::read(&path).unwrap(), b"alpha\nBETA\nGAMMA\n");

        // Replaying the first token now names a file that has moved on twice.
        let (outcome, _, _notes) = run_with(&[
            "alpha",
            "ALPHA",
            file,
            "--in-place",
            "--expect-hash",
            &first,
        ]);
        assert!(matches!(outcome, Err(Failure::Stale { .. })));
        assert_eq!(fs::read(&path).unwrap(), b"alpha\nBETA\nGAMMA\n");
    }

    #[test]
    fn refuses_every_destination_when_the_precondition_fails() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        let out = directory.path().join("out.md");
        fs::write(&path, b"alpha\n").unwrap();
        fs::write(&out, b"previous\n").unwrap();
        let (file, target) = (path.to_str().unwrap(), out.to_str().unwrap());
        let stale = "0000000000000";

        for flags in [
            vec!["alpha", "A", file, "--in-place", "--expect-hash", stale],
            vec![
                "alpha",
                "A",
                file,
                "--output",
                target,
                "--expect-hash",
                stale,
            ],
            vec!["alpha", "A", file, "--dry-run", "--expect-hash", stale],
            vec!["alpha", "A", file, "--expect-hash", stale],
        ] {
            let (outcome, printed, _notes) = run_with(&flags);
            assert!(
                matches!(outcome, Err(Failure::Stale { .. })),
                "expected {flags:?} to be refused"
            );
            assert!(printed.is_empty(), "a refused run printed {printed:?}");
        }

        // Nothing anywhere moved, and no temporary was left to be cleaned up.
        assert_eq!(fs::read(&path).unwrap(), b"alpha\n");
        assert_eq!(fs::read(&out).unwrap(), b"previous\n");
        let mut names: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, ["notes.md", "out.md"]);
    }

    #[test]
    fn refuses_an_ambiguous_target_without_writing() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"beta\ngamma\nbeta\n").unwrap();
        let file = path.to_str().unwrap();

        let (outcome, _, _notes) =
            run_with(&["beta", "B", file, "--in-place", "--occurrences", "one"]);
        let Err(Failure::Ambiguous { matches, .. }) = outcome else {
            panic!("two matches under --occurrences one must be refused");
        };
        assert_eq!(matches, 2);
        assert_eq!(fs::read(&path).unwrap(), b"beta\ngamma\nbeta\n");

        // One match is what the mode asserts, so it goes through.
        let (outcome, _, _notes) =
            run_with(&["gamma", "G", file, "--in-place", "--occurrences", "one"]);
        assert_eq!(outcome.unwrap(), Status::Success);
        assert_eq!(fs::read(&path).unwrap(), b"beta\nG\nbeta\n");
    }

    #[test]
    fn acts_on_as_many_matches_as_the_mode_names() {
        let directory = tempfile::TempDir::new().unwrap();
        let seed = b"beta\ngamma\nbeta\n";

        for (mode, expected) in [
            (None, &b"B\ngamma\nB\n"[..]),
            (Some("all"), &b"B\ngamma\nB\n"[..]),
            (Some("first"), &b"B\ngamma\nbeta\n"[..]),
        ] {
            let path = directory
                .path()
                .join(format!("{}.md", mode.unwrap_or("default")));
            fs::write(&path, seed).unwrap();
            let file = path.to_str().unwrap();
            let mut flags = vec!["beta", "B", file, "--in-place"];
            if let Some(mode) = mode {
                flags.extend_from_slice(&["--occurrences", mode]);
            }
            run_with(&flags).0.unwrap();
            // The default and `all` agree byte for byte: the selector adds modes rather than
            // changing the one that was already there.
            assert_eq!(fs::read(&path).unwrap(), expected, "under {mode:?}");
        }
    }

    #[test]
    fn writes_the_output_file_through_a_temporary() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        let out = directory.path().join("out.md");
        fs::write(&path, b"alpha\n").unwrap();

        run_with(&[
            "alpha",
            "ALPHA",
            path.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .0
        .unwrap();

        assert_eq!(fs::read(&out).unwrap(), b"ALPHA\n");
        let mut names: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        assert_eq!(names, ["notes.md", "out.md"], "a temporary was left behind");
    }

    #[test]
    fn replaces_a_file_named_as_its_own_output() {
        // The input is mapped while the output is opened, so truncating in place would have
        // pulled the pages out from under the read that is still producing the result.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\n").unwrap();
        let file = path.to_str().unwrap();

        run_with(&["beta", "BETA", file, "--output", file])
            .0
            .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"alpha\nBETA\n");
    }

    #[test]
    fn reports_the_hash_of_what_it_wrote() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\n").unwrap();
        let file = path.to_str().unwrap();
        let before = token_of(&path);

        let (outcome, printed, _notes) =
            run_with(&["beta", "BETA", file, "--in-place", "--format", "json"]);
        outcome.unwrap();

        let record = String::from_utf8(printed).unwrap();
        // Accumulated while writing, so it must equal a plain hash of the finished file.
        assert!(
            record.contains(&format!(r#""hash_after":"{}""#, token_of(&path))),
            "{record}"
        );
        assert!(
            record.contains(&format!(r#""hash_before":"{before}""#)),
            "{record}"
        );
    }

    #[test]
    fn reports_the_hash_a_dry_run_would_have_written() {
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\n").unwrap();
        let file = path.to_str().unwrap();

        let (outcome, printed, _notes) =
            run_with(&["beta", "BETA", file, "--dry-run", "--format", "json"]);
        outcome.unwrap();

        // Untouched, but the record still names what the edit would have produced.
        assert_eq!(fs::read(&path).unwrap(), b"alpha\nbeta\n");
        let record = String::from_utf8(printed).unwrap();
        let mut buffer = [0u8; HASH_CHARS];
        let after = format_hash(&mut buffer, content_hash(b"alpha\nBETA\n"), HASH_CHARS);
        assert!(
            record.contains(&format!(r#""hash_after":"{after}""#)),
            "{record}"
        );
    }

    #[test]
    fn keeps_the_summary_out_of_the_bytes_it_summarizes() {
        // `sz-replace --summary a b f > out.txt` has to leave `out.txt` a file of text. The
        // sentence describes the run, so it goes to the stream a shell keeps separate.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\n").unwrap();

        let (outcome, printed, notes) =
            run_with(&["beta", "BETA", path.to_str().unwrap(), "--summary"]);
        outcome.unwrap();

        assert_eq!(String::from_utf8(printed).unwrap(), "alpha\nBETA\n");
        let mut buffer = [0u8; HASH_CHARS];
        let after = format_hash(&mut buffer, content_hash(b"alpha\nBETA\n"), HASH_CHARS);
        assert_eq!(
            String::from_utf8(notes).unwrap().trim(),
            format!("Replaced 1 occurrence(s); content is {after}")
        );
    }

    #[test]
    fn leaves_the_unreported_run_unhashed() {
        // The benchmark path: no summary, no json, so nothing asks for a hash and the run
        // does not pay for one. Only the absence of a reported hash is observable here.
        let directory = tempfile::TempDir::new().unwrap();
        let path = directory.path().join("notes.md");
        fs::write(&path, b"alpha\nbeta\n").unwrap();

        let (outcome, printed, _notes) = run_with(&["beta", "BETA", path.to_str().unwrap()]);
        outcome.unwrap();
        assert_eq!(printed, b"alpha\nBETA\n");
    }
}
