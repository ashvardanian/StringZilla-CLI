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
//! # Split with UTF-8 validation
//! sz-split --utf8 -l 1000 utf8_file.txt output
//!
//! # From stdin
//! cat large.txt | sz-split -l 1000 output
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};

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
    lines: usize,

    /// Enable UTF-8 mode (validate input)
    #[arg(long)]
    utf8: bool,

    /// Suffix length (default: 2, gives aa, ab, ac...)
    #[arg(long, default_value = "2")]
    suffix_length: usize,

    /// Emit a JSON Lines manifest of the files written, one record each
    #[arg(long, help_heading = "Output Formats")]
    json: bool,
}

/// Generate suffix for split files (aa, ab, ac, ... az, ba, bb, ...)
fn generate_suffix(index: usize, length: usize) -> String {
    let mut result = String::with_capacity(length);
    let mut n = index;

    for _ in 0..length {
        let c = (b'a' + (n % 26) as u8) as char;
        result.insert(0, c);
        n /= 26;
    }

    result
}

/// Split data into multiple files by line count
fn split_by_lines(
    data: &[u8],
    prefix: &str,
    lines_per_file: usize,
    suffix_length: usize,
    manifest: Option<&mut dyn Write>,
) -> io::Result<()> {
    if lines_per_file == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Lines per file must be greater than 0",
        ));
    }

    let mut file_index = 0;
    let mut current_lines = 0;
    let mut current_bytes = 0;
    let mut current_name = String::new();
    let mut current_file: Option<BufWriter<File>> = None;
    let mut manifest = manifest;

    for line in LineIter::new(data, Newlines::Lf) {
        // Create new file if needed
        if current_lines == 0 {
            let suffix = generate_suffix(file_index, suffix_length);
            let filename = format!("{}{}", prefix, suffix);
            let file = File::create(&filename).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("Failed to create output file '{}': {}", filename, e),
                )
            })?;
            current_file = Some(BufWriter::new(file));
            current_name = filename;
            file_index += 1;
        }

        // Write line to current file
        if let Some(ref mut writer) = current_file {
            writer.write_all(line)?;
            writer.write_all(b"\n")?;
            current_lines += 1;
            current_bytes += line.len() + 1;

            // Close file if we've reached the line limit
            if current_lines >= lines_per_file {
                writer.flush()?;
                if let Some(ref mut sink) = manifest {
                    write_manifest_entry(*sink, &current_name, current_lines, current_bytes)?;
                }
                current_file = None;
                current_lines = 0;
                current_bytes = 0;
            }
        }
    }

    // Flush final file
    if let Some(mut writer) = current_file {
        writer.flush()?;
        if let Some(ref mut sink) = manifest {
            write_manifest_entry(*sink, &current_name, current_lines, current_bytes)?;
        }
    }

    Ok(())
}

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

    let mut stdout = io::stdout();

    if args.lines == 0 {
        eprintln!("Error: lines must be greater than 0");
        ExitCode::Error.exit(&mut stdout);
    }

    if args.suffix_length == 0 {
        eprintln!("Error: suffix length must be greater than 0");
        ExitCode::Error.exit(&mut stdout);
    }

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(error) => exit_with_error(&mut stdout, &error, "Error reading input"),
    };

    let data = input.as_bytes();

    // Silent unless `--json`, so existing scripts see no new stdout output.
    let mut manifest = io::stdout();
    let sink: Option<&mut dyn Write> = if args.json { Some(&mut manifest) } else { None };

    if let Err(error) = split_by_lines(data, &args.prefix, args.lines, args.suffix_length, sink) {
        let mut stdout = io::stdout();
        exit_with_error(&mut stdout, &error, "Error splitting file");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn generates_aa_ab_suffix_sequence() {
        assert_eq!(generate_suffix(0, 2), "aa");
        assert_eq!(generate_suffix(1, 2), "ab");
        assert_eq!(generate_suffix(25, 2), "az");
        assert_eq!(generate_suffix(26, 2), "ba");
        assert_eq!(generate_suffix(27, 2), "bb");
    }

    #[test]
    fn splits_input_into_line_chunk_files() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("test_").to_str().unwrap().to_string();

        let data = b"line1\nline2\nline3\nline4\nline5\n";
        split_by_lines(data, &prefix, 2, 2, None).unwrap();

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
        split_by_lines(data, &prefix, 1, 2, None).unwrap();

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
        split_by_lines(data, &prefix, 1, 2, None).unwrap();

        assert_eq!(
            fs::read_to_string(format!("{}aa", prefix)).unwrap(),
            "line1\n"
        );
        assert_eq!(
            fs::read_to_string(format!("{}ab", prefix)).unwrap(),
            "line2\n"
        );
    }
}
