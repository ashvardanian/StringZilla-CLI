//! SIMD-accelerated word count utility
//!
//! A faster replacement for `wc` with proper UTF-8 support and directory traversal.
//! Uses StringZilla for SIMD-accelerated counting operations.
//!
//! # Examples
//!
//! ```bash
//! # Count single file
//! sz-count file.txt
//!
//! # Count multiple files
//! sz-count src/*.rs
//!
//! # Count directory recursively
//! sz-count src/
//!
//! # Human-readable output
//! sz-count -H src/
//!
//! # UTF-8 mode (count characters, Unicode whitespace/newlines)
//! sz-count --utf8 docs/
//! ```

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;
use ignore::WalkBuilder;
use memmap2::Mmap;
use stringzilla::sz;
use stringzilla::sz::StringZillableUnary;

mod shared;

#[derive(Debug, Clone)]
struct Counts {
    lines: usize,
    words: usize,
    bytes: usize,
    chars: Option<usize>, // Only computed in UTF-8 mode
}

impl Counts {
    fn zero(utf8_mode: bool) -> Self {
        Self {
            lines: 0,
            words: 0,
            bytes: 0,
            chars: if utf8_mode { Some(0) } else { None },
        }
    }

    fn add(&mut self, other: &Counts) {
        self.lines += other.lines;
        self.words += other.words;
        self.bytes += other.bytes;
        match (&mut self.chars, other.chars) {
            (Some(a), Some(b)) => *a += b,
            (None, Some(b)) => self.chars = Some(b),
            _ => {}
        }
    }
}

/// Count lines, words, bytes, and characters in files and directories
#[derive(Parser)]
#[command(name = "sz-count")]
#[command(version, about = "SIMD-accelerated word count", long_about = None)]
struct Args {
    /// Input files or directories (use '-' for stdin, default: stdin)
    #[arg(default_value = "-")]
    inputs: Vec<String>,

    /// Enable UTF-8 mode (count characters, use Unicode whitespace/newlines)
    #[arg(long)]
    utf8: bool,

    /// Human-readable output (use K/M/G/T suffixes)
    #[arg(short = 'H', long)]
    human_readable: bool,

    /// Maximum directory depth (default: unlimited)
    #[arg(long)]
    max_depth: Option<usize>,

    /// Include hidden files and directories
    #[arg(long)]
    hidden: bool,

    /// Don't respect .gitignore files
    #[arg(long)]
    no_ignore: bool,
}

/// Count all metrics for given data using StringZilla iterators (single-pass)
fn count_data(data: &[u8], utf8_mode: bool) -> io::Result<Counts> {
    let bytes = data.len();
    let mut lines = 0;
    let mut words = 0;
    let mut chars = 0;

    if utf8_mode {
        // UTF-8 mode: nested iteration through lines -> words -> chars.
        // `LineIter` gives line-terminator semantics (a trailing newline is not a
        // final empty line); `.skip_empty()` yields whitespace-separated tokens.
        for line in shared::LineIter::new(data, true) {
            lines += 1;
            for word in line.sz_utf8_split_whitespaces().skip_empty() {
                words += 1;
                // Count UTF-8 characters in this word
                chars += sz::count_utf8(word);
            }
        }

        Ok(Counts {
            lines,
            words,
            bytes,
            chars: Some(chars),
        })
    } else {
        // ASCII mode: nested iteration through lines -> words
        for line in shared::LineIter::new(data, false) {
            lines += 1;
            // Split line by ASCII whitespace
            let mut in_word = false;
            for &byte in line {
                let is_whitespace = byte.is_ascii_whitespace();
                if !is_whitespace && !in_word {
                    words += 1;
                    in_word = true;
                } else if is_whitespace {
                    in_word = false;
                }
            }
        }

        Ok(Counts {
            lines,
            words,
            bytes,
            chars: None,
        })
    }
}

/// Count a single file
fn count_file(path: &Path, utf8_mode: bool) -> io::Result<Counts> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    count_data(&mmap, utf8_mode)
}

/// Count stdin
fn count_stdin(utf8_mode: bool) -> io::Result<Counts> {
    let mut buffer = Vec::new();
    io::stdin().read_to_end(&mut buffer)?;
    count_data(&buffer, utf8_mode)
}

/// Format number with K/M/G/T suffixes
fn format_human(n: usize) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else if n < 1_000_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n < 1_000_000_000_000 {
        format!("{:.1}G", n as f64 / 1_000_000_000.0)
    } else {
        format!("{:.1}T", n as f64 / 1_000_000_000_000.0)
    }
}

/// Format number with thousands separators
fn format_number(n: usize, human: bool) -> String {
    if human {
        format_human(n)
    } else {
        let s = n.to_string();
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }
}

/// Truncate path to fit within max_width, keeping the rightmost part
fn truncate_path(path: &str, max_width: usize) -> String {
    if path.len() <= max_width {
        path.to_string()
    } else {
        // Keep rightmost part with "..." prefix
        let keep_len = max_width.saturating_sub(3); // Reserve 3 chars for "..."
        let skip = path.len() - keep_len;
        format!("...{}", &path[skip..])
    }
}

/// Print header row
fn print_header(utf8_mode: bool) {
    if utf8_mode {
        println!(
            "{:>40} {:>7} {:>7} {:>7} {:>7}",
            "", "lines", "words", "bytes", "chars"
        );
    } else {
        println!("{:>40} {:>7} {:>7} {:>7}", "", "lines", "words", "bytes");
    }
}

/// Print a single file's counts
fn print_counts(name: &str, counts: &Counts, human: bool, indent: usize) {
    let indent_str = "   ".repeat(indent);
    let truncated_name = truncate_path(name, 40);
    let lines = format_number(counts.lines, human);
    let words = format_number(counts.words, human);
    let bytes = format_number(counts.bytes, human);

    if let Some(chars) = counts.chars {
        let chars_str = format_number(chars, human);
        println!(
            "{}{:40} {:>7} {:>7} {:>7} {:>7}",
            indent_str, truncated_name, lines, words, bytes, chars_str
        );
    } else {
        println!(
            "{}{:40} {:>7} {:>7} {:>7}",
            indent_str, truncated_name, lines, words, bytes
        );
    }
}

/// Print tree branch
fn print_tree_line(name: &str, counts: &Counts, human: bool, is_last: bool, depth: usize) {
    let mut prefix = String::new();
    for _ in 0..depth {
        prefix.push_str("   ");
    }
    let branch = if is_last { "└─ " } else { "├─ " };
    prefix.push_str(branch);

    let full_name = format!("{}{}", prefix, name);
    let truncated_name = truncate_path(&full_name, 40);
    let lines = format_number(counts.lines, human);
    let words = format_number(counts.words, human);
    let bytes = format_number(counts.bytes, human);

    if let Some(chars) = counts.chars {
        let chars_str = format_number(chars, human);
        println!(
            "{:40} {:>7} {:>7} {:>7} {:>7}",
            truncated_name, lines, words, bytes, chars_str
        );
    } else {
        println!(
            "{:40} {:>7} {:>7} {:>7}",
            truncated_name, lines, words, bytes
        );
    }
}

/// Print separator line for totals
fn print_separator(utf8_mode: bool) {
    if utf8_mode {
        println!("{:>40} {:─>7} {:─>7} {:─>7} {:─>7}", "", "", "", "", "");
    } else {
        println!("{:>40} {:─>7} {:─>7} {:─>7}", "", "", "", "");
    }
}

/// Process a directory recursively
fn process_directory(
    path: &Path,
    utf8_mode: bool,
    human: bool,
    max_depth: Option<usize>,
    hidden: bool,
    no_ignore: bool,
) -> io::Result<()> {
    let mut entries = Vec::new();
    let mut total = Counts::zero(utf8_mode);

    // Build walker with ignore support
    let mut builder = WalkBuilder::new(path);
    builder
        .hidden(!hidden) // Skip hidden files unless --hidden is set
        .git_ignore(!no_ignore) // Respect .gitignore unless --no-ignore is set
        .git_global(!no_ignore)
        .git_exclude(!no_ignore);

    if let Some(depth) = max_depth {
        builder.max_depth(Some(depth));
    }

    // Collect all entries
    for result in builder.build() {
        let entry = result.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        if entry.file_type().map_or(false, |ft| ft.is_file()) {
            let entry_path = entry.path();
            match count_file(entry_path, utf8_mode) {
                Ok(counts) => {
                    total.add(&counts);
                    entries.push((entry_path.to_path_buf(), counts));
                }
                Err(e) => {
                    eprintln!("Warning: failed to read {}: {}", entry_path.display(), e);
                }
            }
        }
    }

    if entries.is_empty() {
        eprintln!("No files found in {}", path.display());
        return Ok(());
    }

    // Sort entries by path
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    // Print header
    print_header(utf8_mode);

    // Print directory summary first
    let path_str = path.display().to_string();
    let dir_display = if path_str.ends_with('/') {
        path_str
    } else {
        format!("{}/", path_str)
    };
    print_counts(&dir_display, &total, human, 0);

    // Print tree structure
    for (i, (entry_path, counts)) in entries.iter().enumerate() {
        let is_last = i == entries.len() - 1;
        let name = entry_path
            .strip_prefix(path)
            .unwrap_or(entry_path)
            .display()
            .to_string();
        print_tree_line(&name, counts, human, is_last, 0);
    }

    // Print separator and totals
    print_separator(utf8_mode);
    let total_lines = format_number(total.lines, human);
    let total_words = format_number(total.words, human);
    let total_bytes = format_number(total.bytes, human);

    if let Some(chars) = total.chars {
        let total_chars = format_number(chars, human);
        println!(
            "{:>40} {:>7} {:>7} {:>7} {:>7}",
            "", total_lines, total_words, total_bytes, total_chars
        );
    } else {
        println!(
            "{:>40} {:>7} {:>7} {:>7}",
            "", total_lines, total_words, total_bytes
        );
    }

    Ok(())
}

/// Process multiple files
fn process_multiple_files(paths: &[PathBuf], utf8_mode: bool, human: bool) -> io::Result<()> {
    let mut results = Vec::new();
    let mut total = Counts::zero(utf8_mode);

    for path in paths {
        match count_file(path, utf8_mode) {
            Ok(counts) => {
                total.add(&counts);
                results.push((path.clone(), counts));
            }
            Err(e) => {
                eprintln!("Warning: failed to read {}: {}", path.display(), e);
            }
        }
    }

    if results.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "No files found"));
    }

    // Print header
    print_header(utf8_mode);

    // Print each file
    for (path, counts) in &results {
        print_counts(&path.display().to_string(), counts, human, 0);
    }

    // Print separator and total
    print_separator(utf8_mode);
    let total_lines = format_number(total.lines, human);
    let total_words = format_number(total.words, human);
    let total_bytes = format_number(total.bytes, human);

    if let Some(chars) = total.chars {
        let total_chars = format_number(chars, human);
        println!(
            "{:>40} {:>7} {:>7} {:>7} {:>7}",
            "", total_lines, total_words, total_bytes, total_chars
        );
    } else {
        println!(
            "{:>40} {:>7} {:>7} {:>7}",
            "", total_lines, total_words, total_bytes
        );
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    // Handle stdin
    if args.inputs.len() == 1 && args.inputs[0] == "-" {
        match count_stdin(args.utf8) {
            Ok(counts) => {
                print_header(args.utf8);
                print_counts("-", &counts, args.human_readable, 0);
            }
            Err(e) => {
                eprintln!("Error reading stdin: {}", e);
                process::exit(1);
            }
        }
        return;
    }

    // Resolve paths
    let mut files = Vec::new();
    let mut dirs = Vec::new();

    for input in &args.inputs {
        let path = Path::new(input);
        if !path.exists() {
            eprintln!("Error: {} does not exist", input);
            process::exit(1);
        }

        if path.is_dir() {
            dirs.push(path.to_path_buf());
        } else {
            files.push(path.to_path_buf());
        }
    }

    // Process based on input type
    if dirs.is_empty() && files.len() == 1 {
        // Single file
        match count_file(&files[0], args.utf8) {
            Ok(counts) => {
                print_header(args.utf8);
                print_counts(
                    &files[0].display().to_string(),
                    &counts,
                    args.human_readable,
                    0,
                );
            }
            Err(e) => {
                eprintln!("Error reading {}: {}", files[0].display(), e);
                process::exit(1);
            }
        }
    } else if dirs.is_empty() {
        // Multiple files
        if let Err(e) = process_multiple_files(&files, args.utf8, args.human_readable) {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    } else if dirs.len() == 1 && files.is_empty() {
        // Single directory
        if let Err(e) = process_directory(
            &dirs[0],
            args.utf8,
            args.human_readable,
            args.max_depth,
            args.hidden,
            args.no_ignore,
        ) {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    } else {
        eprintln!("Error: mixing files and directories is not yet supported");
        process::exit(1);
    }
}
