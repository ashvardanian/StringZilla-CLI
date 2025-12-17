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
use std::process;

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
) -> io::Result<()> {
    if lines_per_file == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Lines per file must be greater than 0",
        ));
    }

    let mut file_index = 0;
    let mut current_lines = 0;
    let mut current_file: Option<BufWriter<File>> = None;

    for line in LineIterator::new(data) {
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
            file_index += 1;
        }

        // Write line to current file
        if let Some(ref mut writer) = current_file {
            writer.write_all(line)?;
            writer.write_all(b"\n")?;
            current_lines += 1;

            // Close file if we've reached the line limit
            if current_lines >= lines_per_file {
                writer.flush()?;
                current_file = None;
                current_lines = 0;
            }
        }
    }

    // Flush final file
    if let Some(mut writer) = current_file {
        writer.flush()?;
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    if args.lines == 0 {
        eprintln!("Error: lines must be greater than 0");
        process::exit(1);
    }

    if args.suffix_length == 0 {
        eprintln!("Error: suffix length must be greater than 0");
        process::exit(1);
    }

    let input = match get_input(args.input.as_deref()) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };

    let data = input.as_bytes();

    if let Err(e) = split_by_lines(data, &args.prefix, args.lines, args.suffix_length) {
        eprintln!("Error splitting file: {}", e);
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_generate_suffix() {
        assert_eq!(generate_suffix(0, 2), "aa");
        assert_eq!(generate_suffix(1, 2), "ab");
        assert_eq!(generate_suffix(25, 2), "az");
        assert_eq!(generate_suffix(26, 2), "ba");
        assert_eq!(generate_suffix(27, 2), "bb");
    }

    #[test]
    fn test_split_by_lines() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir.path().join("test_").to_str().unwrap().to_string();

        let data = b"line1\nline2\nline3\nline4\nline5\n";
        split_by_lines(data, &prefix, 2, 2).unwrap();

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
    fn test_split_single_line_per_file() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("single_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"a\nb\nc\n";
        split_by_lines(data, &prefix, 1, 2).unwrap();

        assert_eq!(fs::read_to_string(format!("{}aa", prefix)).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(format!("{}ab", prefix)).unwrap(), "b\n");
        assert_eq!(fs::read_to_string(format!("{}ac", prefix)).unwrap(), "c\n");
    }

    #[test]
    fn test_split_no_trailing_newline() {
        let temp_dir = TempDir::new().unwrap();
        let prefix = temp_dir
            .path()
            .join("notrail_")
            .to_str()
            .unwrap()
            .to_string();

        let data = b"line1\nline2";
        split_by_lines(data, &prefix, 1, 2).unwrap();

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
