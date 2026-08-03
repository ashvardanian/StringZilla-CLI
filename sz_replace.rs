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
//! # From stdin
//! cat file.txt | sz-replace foo bar
//! ```

use std::io::{self, Write};

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{find, utf8_uncased_search};

use shared::*;

/// How records are rendered.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
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

/// Replace all occurrences of `pattern` with `replacement`, streaming the output
/// into `out` (no full-file buffer). Returns the replacement count.
///
/// Case-insensitive matching uses StringZilla's full-Unicode `utf8_uncased_search`
/// (no whole-file `to_ascii_lowercase` copy), and advances by the matched length —
/// which case folding may make differ from the pattern length (e.g. ß ↔ SS).
fn replace_all_to(
    data: &[u8],
    pattern: &[u8],
    replacement: &[u8],
    ignore_case: bool,
    out: &mut dyn Write,
) -> io::Result<usize> {
    // An empty pattern matches at every position without consuming anything, so it
    // has no replacement to make.
    if pattern.is_empty() {
        out.write_all(data)?;
        return Ok(0);
    }

    let next_match = |rest: &[u8]| {
        if ignore_case {
            utf8_uncased_search(rest, pattern)
        } else {
            find(rest, pattern).map(|offset| (offset, pattern.len()))
        }
    };

    let mut rest = data;
    let mut count = 0;
    while let Some((offset, matched_len)) = next_match(rest) {
        out.write_all(&rest[..offset])?;
        out.write_all(replacement)?;
        count += 1;
        rest = &rest[offset + matched_len..];
    }
    out.write_all(rest)?;
    Ok(count)
}

/// Write the single summary record. The transformed bytes never share stdout with it,
/// because `--format json` demands a destination of its own.
fn write_summary_json(
    output: &mut dyn Write,
    path: &str,
    replacements: usize,
    dry_run: bool,
) -> io::Result<()> {
    output.write_all(br#"{"type":"summary","data":{"path":"#)?;
    json_text_field_to(output, path.as_bytes())?;
    writeln!(
        output,
        r#","replacements":{},"dry_run":{}}}}}"#,
        replacements, dry_run
    )
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    // Every byte this run prints goes here, so the records stay in one order.
    let mut output = stdout_writer();
    report("sz-replace", run(&args, &mut output))
}

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let name = args.input.as_deref().unwrap_or("-");
    let input = get_input(args.input.as_deref()).at(name)?;

    let data = input.as_bytes();
    let pattern = args.pattern.as_bytes();
    let replacement = args.replacement.as_bytes();
    let counting_only = args.dry_run || args.quiet;

    let count = if counting_only {
        replace_all_to(
            data,
            pattern,
            replacement,
            args.ignore_case,
            &mut io::sink(),
        )
        .expect("writing to a sink cannot fail")
    } else if args.in_place {
        let path = args.input.as_deref().expect("validated");
        write_replacing("sz-replace", path, |output| {
            replace_all_to(data, pattern, replacement, args.ignore_case, output)
        })
        .at(path)?
    } else {
        // Stream directly to stdout / --output — no full-output buffer. A pipe closing
        // during the flush ends the run where one closing during the write does.
        let target = args.output.as_deref().unwrap_or("-");
        let mut opened;
        let destination: &mut dyn Write = match target {
            "-" => &mut *output,
            path => {
                opened = get_output(Some(path)).at(path)?;
                &mut *opened
            }
        };
        let count =
            replace_all_to(data, pattern, replacement, args.ignore_case, destination).at(target)?;
        destination.flush().at(target)?;
        count
    };

    if args.format == Format::Json {
        write_summary_json(output, name, count, args.dry_run).at("-")?;
    } else if args.summary || args.dry_run {
        let verb = if args.dry_run {
            "Would replace"
        } else {
            "Replaced"
        };
        println!("{} {} occurrence(s)", verb, count);
    }

    output.flush().at("-")?;

    // The stream was produced whether or not it changed; counting alone reports matches.
    Ok(Status::from_found(if counting_only {
        count > 0
    } else {
        !data.is_empty()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: run the streaming replace into a buffer and return it.
    fn replace_all(
        data: &[u8],
        pattern: &[u8],
        replacement: &[u8],
        ignore_case: bool,
    ) -> (Vec<u8>, usize) {
        let mut buf = Vec::new();
        let count = replace_all_to(data, pattern, replacement, ignore_case, &mut buf).unwrap();
        (buf, count)
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
    fn takes_no_utf8_flag() {
        // Replacement is a byte-level substitution with no line or codepoint semantics to
        // switch, and `--ignore-case` already folds the full Unicode range, so there is no
        // UTF-8 mode left to ask for.
        let Err(error) = Args::try_parse_from(["sz-replace", "--utf8", "a", "b", "file.txt"])
        else {
            panic!("--utf8 must be rejected");
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
    }

    /// Parse and then apply the value-conditional checks, as `run` does.
    fn accepts(flags: &[&str]) -> bool {
        let arguments = ["sz-replace", "a", "b", "f"]
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
}
