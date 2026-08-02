//! SIMD-accelerated file splitting utility
//!
//! A faster replacement for `split` with UTF-8 awareness.
//! Uses StringZilla for fast line detection.
//!
//! # Examples
//!
//! ```bash
//! # Split by line count
//! sz-split -l 1000 large.txt output_prefix
//!
//! # Split on Unicode newlines rather than LF alone
//! sz-split --utf8 -l 1000 utf8_file.txt output
//!
//! # From stdin
//! cat large.txt | sz-split -l 1000 output
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::num::NonZeroUsize;

use clap::Parser;

mod shared;
use shared::*;

/// Split files into smaller chunks
#[derive(Parser)]
#[command(name = "sz-split")]
#[command(version, about = "SIMD-accelerated file splitting", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Output prefix for split files
    #[arg(default_value = "x")]
    prefix: String,

    /// Number of lines per output file
    #[arg(short = 'l', long, required = true)]
    lines: NonZeroUsize,

    /// Enable UTF-8 mode (split on Unicode newlines: CR, CRLF, NEL, LS, PS; chunk lines end with LF)
    #[arg(long)]
    utf8: bool,

    /// Suffix length (default: 2, gives aa, ab, ac...)
    #[arg(long, default_value = "2")]
    suffix_length: NonZeroUsize,

    /// Emit a JSON Lines manifest of the files written, one record each
    #[arg(long, help_heading = "Output Formats")]
    json: bool,
}

/// The chunk name suffix for `index` — aa, ab, ac, ... az, ba, bb, ... — or `None` once
/// `index` needs more than `length` characters, where wrapping would reuse an earlier name.
fn generate_suffix(index: usize, length: usize) -> Option<String> {
    let mut suffix = String::with_capacity(length);
    let mut remaining = index;

    for _ in 0..length {
        suffix.insert(0, (b'a' + (remaining % 26) as u8) as char);
        remaining /= 26;
    }

    // A non-zero leftover is the overflow signal: those digits have nowhere to go.
    (remaining == 0).then_some(suffix)
}

/// How the input is cut into chunk files, decided once from `Args`.
#[derive(Clone, Copy)]
struct SplitConfig<'a> {
    /// Prepended to every generated suffix to name a chunk file.
    prefix: &'a str,
    /// Lines each chunk file holds, except possibly the last.
    lines_per_file: NonZeroUsize,
    /// Width of the generated suffix.
    suffix_length: NonZeroUsize,
}

/// The chunk being filled, and the tallies the manifest reports when it closes. The four
/// travel together because they are meaningless apart: closing the file retires all of them.
struct OpenChunk {
    /// Name of the chunk, carried into the manifest.
    name: String,
    /// Lines written into it.
    lines: usize,
    /// Bytes written into it.
    bytes: usize,
    /// The chunk file itself.
    file: BufWriter<File>,
}

/// What [`split_by_lines`] carries between windows, so a second call resumes where the
/// first stopped. It owns the chunk still being filled, and allocates only when one opens.
#[derive(Default)]
struct SplitState {
    /// Chunk files opened so far, which names the next one.
    file_index: usize,
    /// The chunk being filled, absent between chunks.
    open: Option<OpenChunk>,
}

/// The error a chunk index outgrowing its suffix width raises, naming the flag to raise
/// and the chunk that has nowhere to go.
fn suffix_exhausted(chunk_number: usize, suffix_length: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "Output file suffixes exhausted: chunk {} does not fit a {}-character --suffix-length",
            chunk_number, suffix_length
        ),
    )
}

/// Create one chunk file, naming it in any failure the caller reports.
fn create_chunk(filename: &str) -> io::Result<BufWriter<File>> {
    let file = File::create(filename).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("Failed to create output file '{}': {}", filename, error),
        )
    })?;
    Ok(BufWriter::new(file))
}

/// Flush the open chunk and record it in the manifest, readying `state` for the next one.
/// A run without `--json` writes the record into [`io::sink`], so there is one path here.
fn close_chunk(state: &mut SplitState, manifest: &mut dyn Write) -> io::Result<()> {
    let Some(mut chunk) = state.open.take() else {
        return Ok(());
    };
    chunk.file.flush()?;
    write_manifest_entry(manifest, &chunk.name, chunk.lines, chunk.bytes)
}

/// Write every complete line in `data` into chunk files, resuming from `state`. The chunk
/// left open at the end is the caller's to [`close_chunk`].
fn split_by_lines(
    data: &[u8],
    state: &mut SplitState,
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    for line in LineIter::new(data, newlines) {
        let chunk = match state.open {
            Some(ref mut chunk) => chunk,
            // Between chunks: open the next one into the empty slot. GNU `split` stops
            // here rather than wrapping onto a name it already wrote.
            ref mut slot => {
                let width = config.suffix_length.get();
                let suffix = generate_suffix(state.file_index, width)
                    .ok_or_else(|| suffix_exhausted(state.file_index + 1, width))?;
                let name = format!("{}{}", config.prefix, suffix);
                let file = create_chunk(&name)?;
                state.file_index += 1;
                slot.insert(OpenChunk {
                    name,
                    lines: 0,
                    bytes: 0,
                    file,
                })
            }
        };

        // Output is normalized to end with a newline, whatever the input's last line did.
        chunk.file.write_all(line)?;
        chunk.file.write_all(b"\n")?;
        chunk.lines += 1;
        chunk.bytes += line.len() + 1;

        if chunk.lines >= config.lines_per_file.get() {
            close_chunk(state, manifest)?;
        }
    }

    Ok(())
}

/// Split a whole buffer, closing the chunk left open at the end.
fn split_buffer(
    data: &[u8],
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut state = SplitState::default();
    split_by_lines(data, &mut state, newlines, config, manifest)?;
    close_chunk(&mut state, manifest)
}

// region: Streaming

/// Drive [`split_by_lines`] over a reader, handing it whole-line prefixes of one reused
/// window so that a pipe costs bounded memory rather than the input's size.
fn stream_split<R: Read>(
    refill: &mut Refill<R>,
    newlines: Newlines,
    config: &SplitConfig,
    manifest: &mut dyn Write,
) -> io::Result<()> {
    let mut state = SplitState::default();
    refill.for_each_window(newlines.into(), |window| {
        split_by_lines(window, &mut state, newlines, config, manifest)
    })?;
    close_chunk(&mut state, manifest)
}

// endregion: Streaming

/// Write one manifest record naming a file that was written.
fn write_manifest_entry(
    output: &mut dyn Write,
    name: &str,
    lines: usize,
    bytes: usize,
) -> io::Result<()> {
    output.write_all(br#"{"type":"file","data":{"path":"#)?;
    json_text_field_to(output, name.as_bytes())?;
    writeln!(output, r#","lines":{},"bytes":{}}}}}"#, lines, bytes)
}

fn main() {
    let args = Args::parse();
    let mut output = stdout_writer();

    let input = match get_input_streaming(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut output, &error, "Error reading input"),
    };

    let config = SplitConfig {
        prefix: &args.prefix,
        lines_per_file: args.lines,
        suffix_length: args.suffix_length,
    };

    let newlines = Newlines::from_utf8(args.utf8);
    let result = {
        // Silent unless `--json`, so existing scripts see no new stdout output.
        let mut discarded = io::sink();
        let manifest: &mut dyn Write = if args.json {
            &mut output
        } else {
            &mut discarded
        };
        match input.into_window(DEFAULT_WINDOW_BYTES) {
            InputWindow::Whole(source) => {
                split_buffer(source.as_bytes(), newlines, &config, manifest)
            }
            InputWindow::Stream(mut refill) => {
                stream_split(&mut refill, newlines, &config, manifest)
            }
        }
    };

    if let Err(error) = result.and_then(|()| output.flush()) {
        exit_with_error(&mut output, &error, "Error splitting file");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn generates_aa_ab_suffix_sequence() {
        assert_eq!(generate_suffix(0, 2).as_deref(), Some("aa"));
        assert_eq!(generate_suffix(1, 2).as_deref(), Some("ab"));
        assert_eq!(generate_suffix(25, 2).as_deref(), Some("az"));
        assert_eq!(generate_suffix(26, 2).as_deref(), Some("ba"));
        assert_eq!(generate_suffix(27, 2).as_deref(), Some("bb"));
    }

    #[test]
    fn refuses_suffixes_wider_than_the_requested_length() {
        // Two characters name 26 * 26 chunks, so index 676 is the first that cannot be named.
        assert_eq!(generate_suffix(675, 2).as_deref(), Some("zz"));
        assert_eq!(generate_suffix(676, 2), None);
        assert_eq!(generate_suffix(25, 1).as_deref(), Some("z"));
        assert_eq!(generate_suffix(26, 1), None);
    }

    /// A two-character-suffix config, the shape every test below splits with.
    fn config(prefix: &str, lines_per_file: usize) -> SplitConfig<'_> {
        SplitConfig {
            prefix,
            lines_per_file: NonZeroUsize::new(lines_per_file).unwrap(),
            suffix_length: NonZeroUsize::new(2).unwrap(),
        }
    }

    /// The chunk files written under `prefix`, in suffix order.
    fn read_chunks(prefix: &str) -> Vec<String> {
        (0..)
            .map_while(|index| generate_suffix(index, 2))
            .map(|suffix| format!("{}{}", prefix, suffix))
            .map_while(|path| fs::read_to_string(path).ok())
            .collect()
    }

    #[test]
    fn splits_input_into_line_chunk_files() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("test_").to_str().unwrap().to_string();

        let data = b"line1\nline2\nline3\nline4\nline5\n";
        split_buffer(data, Newlines::Lf, &config(&prefix, 2), &mut io::sink()).unwrap();

        // Check first file
        let file1 = fs::read_to_string(format!("{}aa", prefix)).unwrap();
        assert_eq!(file1, "line1\nline2\n");

        // Check second file
        let file2 = fs::read_to_string(format!("{}ab", prefix)).unwrap();
        assert_eq!(file2, "line3\nline4\n");

        // Check third file
        let file3 = fs::read_to_string(format!("{}ac", prefix)).unwrap();
        assert_eq!(file3, "line5\n");
    }

    #[test]
    fn writes_one_line_per_file() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("single_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"a\nb\nc\n";
        split_buffer(data, Newlines::Lf, &config(&prefix, 1), &mut io::sink()).unwrap();

        assert_eq!(fs::read_to_string(format!("{}aa", prefix)).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(format!("{}ab", prefix)).unwrap(), "b\n");
        assert_eq!(fs::read_to_string(format!("{}ac", prefix)).unwrap(), "c\n");
    }

    #[test]
    fn appends_newline_to_last_split_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("notrail_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"line1\nline2";
        split_buffer(data, Newlines::Lf, &config(&prefix, 1), &mut io::sink()).unwrap();

        assert_eq!(
            fs::read_to_string(format!("{}aa", prefix)).unwrap(),
            "line1\n"
        );
        assert_eq!(
            fs::read_to_string(format!("{}ab", prefix)).unwrap(),
            "line2\n"
        );
    }

    #[test]
    fn errors_rather_than_overwriting_when_suffixes_run_out() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("overflow_")
            .to_str()
            .unwrap()
            .to_string();

        // One character names 26 chunks, so a 27th line has nowhere to go.
        let data: Vec<u8> = (0..27)
            .flat_map(|line| format!("line{}\n", line).into_bytes())
            .collect();
        let overflowing = SplitConfig {
            prefix: &prefix,
            lines_per_file: NonZeroUsize::new(1).unwrap(),
            suffix_length: NonZeroUsize::new(1).unwrap(),
        };

        let error = split_buffer(&data, Newlines::Lf, &overflowing, &mut io::sink()).unwrap_err();

        let message = error.to_string();
        assert!(message.contains("--suffix-length"), "{}", message);
        assert!(message.contains("chunk 27"), "{}", message);
        // The 26 chunks that did fit keep the lines they were given.
        assert_eq!(
            fs::read_to_string(format!("{}a", prefix)).unwrap(),
            "line0\n"
        );
        assert_eq!(
            fs::read_to_string(format!("{}z", prefix)).unwrap(),
            "line25\n"
        );
    }

    #[test]
    fn splits_on_unicode_newlines_under_utf8() {
        let temp_dir = TempDir::new().unwrap();
        // Line separators, which only the Unicode newline set breaks on.
        let data = "a\u{2028}b\u{2028}c\n".as_bytes();

        let unicode_prefix = temp_dir
            .path()
            .join("unicode_")
            .to_str()
            .unwrap()
            .to_string();
        split_buffer(
            data,
            Newlines::Unicode,
            &config(&unicode_prefix, 1),
            &mut io::sink(),
        )
        .unwrap();
        // Chunk lines are normalized to LF, whatever terminator the input broke on.
        assert_eq!(read_chunks(&unicode_prefix), vec!["a\n", "b\n", "c\n"]);

        let byte_prefix = temp_dir.path().join("byte_").to_str().unwrap().to_string();
        split_buffer(
            data,
            Newlines::Lf,
            &config(&byte_prefix, 1),
            &mut io::sink(),
        )
        .unwrap();
        assert_eq!(read_chunks(&byte_prefix), vec!["a\u{2028}b\u{2028}c\n"]);
    }

    #[test]
    fn streams_identically_to_whole_buffer() {
        // A blank line, an over-wide line, and a final line without a terminator.
        let data =
            "line1\n\nline3\u{2028}a line far wider than a seven byte window\nlast".as_bytes();
        let temp_dir = TempDir::new().unwrap();

        for (newline_set, newlines) in [("lf", Newlines::Lf), ("unicode", Newlines::Unicode)] {
            for lines_per_file in [1, 2, 3] {
                let whole_prefix = temp_dir
                    .path()
                    .join(format!("whole_{}_{}_", newline_set, lines_per_file))
                    .to_str()
                    .unwrap()
                    .to_string();
                let mut whole_manifest = Vec::new();
                split_buffer(
                    data,
                    newlines,
                    &config(&whole_prefix, lines_per_file),
                    &mut whole_manifest,
                )
                .unwrap();
                let expected_chunks = read_chunks(&whole_prefix);
                // Only the chunk name differs between the two runs, so drop it before comparing.
                let expected_manifest = String::from_utf8(whole_manifest)
                    .unwrap()
                    .replace(&whole_prefix, "");

                for capacity in [7, 13, 64, 4096] {
                    let prefix = temp_dir
                        .path()
                        .join(format!(
                            "streamed_{}_{}_{}_",
                            newline_set, lines_per_file, capacity
                        ))
                        .to_str()
                        .unwrap()
                        .to_string();
                    let mut streamed_manifest = Vec::new();
                    let mut refill = Refill::new(data, capacity);
                    stream_split(
                        &mut refill,
                        newlines,
                        &config(&prefix, lines_per_file),
                        &mut streamed_manifest,
                    )
                    .unwrap();

                    assert_eq!(
                        read_chunks(&prefix),
                        expected_chunks,
                        "{} newlines at capacity {}",
                        newline_set,
                        capacity
                    );
                    assert_eq!(
                        String::from_utf8(streamed_manifest)
                            .unwrap()
                            .replace(&prefix, ""),
                        expected_manifest,
                        "{} newlines at capacity {}",
                        newline_set,
                        capacity
                    );
                }
            }
        }
    }

    #[test]
    fn streams_an_empty_input_without_chunks() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("empty_").to_str().unwrap().to_string();

        let mut refill = Refill::new(&b""[..], 7);
        stream_split(
            &mut refill,
            Newlines::Lf,
            &config(&prefix, 2),
            &mut io::sink(),
        )
        .unwrap();

        assert!(read_chunks(&prefix).is_empty());
    }
}
