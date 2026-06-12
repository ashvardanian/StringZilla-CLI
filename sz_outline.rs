//! SIMD-accelerated file outlining utility
//!
//! Extract structural outlines from source files including:
//! - Markdown (`.md`, `.markdown`): headings, code blocks, tables, blockquotes, images
//! - C (`.c`, `.h`): includes, function declarations, function definitions
//!
//! File type is taken from the extension or forced with `-t {md,c,h}`.
//!
//! # Examples
//!
//! ```bash
//! # Outline a Markdown file
//! sz-outline README.md
//!
//! # Verbose output with line numbers
//! sz-outline -v README.md
//!
//! # Most verbose with block details
//! sz-outline -vv src/main.c
//!
//! # Force file type
//! sz-outline -t md document.txt
//! ```

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process;

use clap::Parser;
use memmap2::Mmap;
use stringzilla::sz::find;

mod shared;

// region: Data Structures

/// Verbosity level for output
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Verbosity {
    Names = 0,       // v0: names only
    LineNumbers = 1, // v1: add line numbers and byte offsets
    Detailed = 2,    // v2: add block sizes, inner blocks
}

/// File type determined by extension
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FileType {
    Markdown,
    CSource,
    CHeader,
    Unknown,
}

/// Types of outline elements
#[derive(Debug, Clone, PartialEq, Eq)]
enum ElementKind {
    // Markdown elements
    Heading { level: u8 },
    CodeBlock { language: Option<String> },
    Blockquote,
    Table,
    Image { alt: String },
    Paragraph,

    // C elements
    Include { is_system: bool },
    FunctionDeclaration,
    FunctionDefinition,
}

/// Represents an outline element with position info
#[derive(Debug, Clone)]
struct OutlineElement {
    kind: ElementKind,
    name: String,
    line_number: usize,
    byte_offset: usize,
    byte_length: usize,
    line_count: usize,
    children: Vec<OutlineElement>,
}

impl OutlineElement {
    fn new(kind: ElementKind, name: String, line_number: usize, byte_offset: usize) -> Self {
        Self {
            kind,
            name,
            line_number,
            byte_offset,
            byte_length: 0,
            line_count: 1,
            children: Vec::new(),
        }
    }

    fn with_length(mut self, byte_length: usize, line_count: usize) -> Self {
        self.byte_length = byte_length;
        self.line_count = line_count;
        self
    }
}

// endregion: Data Structures

// region: CLI Interface

/// Extract structural outline from source files
#[derive(Parser)]
#[command(name = "sz-outline")]
#[command(version, about = "SIMD-accelerated file outlining", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    #[arg(default_value = "-")]
    input: String,

    /// Verbosity level: -v for line numbers/offsets, -vv for block details
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,

    /// Force file type detection (md, c, h)
    #[arg(short = 't', long = "type")]
    file_type: Option<String>,

    /// Enable UTF-8 validation
    #[arg(long)]
    utf8: bool,
}

// endregion: CLI Interface

// region: File Type Detection

fn detect_file_type(path: &str) -> FileType {
    let path = Path::new(path);
    match path.extension().and_then(|e| e.to_str()) {
        Some("md") | Some("markdown") => FileType::Markdown,
        Some("c") => FileType::CSource,
        Some("h") => FileType::CHeader,
        _ => FileType::Unknown,
    }
}

fn parse_file_type(type_str: &str) -> FileType {
    match type_str.to_lowercase().as_str() {
        "md" | "markdown" => FileType::Markdown,
        "c" => FileType::CSource,
        "h" => FileType::CHeader,
        _ => FileType::Unknown,
    }
}

// endregion: File Type Detection

// region: Markdown Parser

/// Parse Markdown file and extract outline elements
fn parse_markdown(data: &[u8], verbosity: Verbosity) -> Vec<OutlineElement> {
    let mut elements: Vec<OutlineElement> = Vec::new();
    let mut line_number = 0usize;
    let mut byte_offset = 0usize;

    // State for code blocks
    let mut in_code_block = false;
    let mut code_fence_char: u8 = 0;
    let mut code_block_start: Option<(usize, usize, Option<String>)> = None;

    // State for v2 block tracking
    let mut current_section_idx: Option<usize> = None;
    let mut in_blockquote = false;
    let mut blockquote_start: Option<(usize, usize)> = None;
    let mut in_table = false;
    let mut table_start: Option<(usize, usize)> = None;
    let mut paragraph_start: Option<(usize, usize)> = None;

    for line in shared::LineIterator::new(data) {
        line_number += 1;
        let line_start = byte_offset;
        let line_len = line.len();

        // Handle code blocks
        if let Some((fence_char, lang)) = is_code_fence(line) {
            if in_code_block && fence_char == code_fence_char {
                // End code block
                if let Some((start_line, start_offset, lang)) = code_block_start.take() {
                    let elem = OutlineElement::new(
                        ElementKind::CodeBlock { language: lang },
                        "code".to_string(),
                        start_line,
                        start_offset,
                    )
                    .with_length(
                        byte_offset + line_len - start_offset,
                        line_number - start_line + 1,
                    );

                    if verbosity >= Verbosity::Detailed {
                        if let Some(idx) = current_section_idx {
                            elements[idx].children.push(elem);
                        } else {
                            elements.push(elem);
                        }
                    }
                }
                in_code_block = false;
            } else if !in_code_block {
                // Start code block
                code_fence_char = fence_char;
                code_block_start = Some((line_number, line_start, lang));
                in_code_block = true;
                // End any ongoing paragraph
                finalize_paragraph(
                    &mut elements,
                    &mut paragraph_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
            }
            byte_offset += line_len + 1;
            continue;
        }

        if in_code_block {
            byte_offset += line_len + 1;
            continue;
        }

        let trimmed = trim_start(line, 3);

        // Detect headings
        if let Some((level, text)) = parse_heading(trimmed) {
            // Finalize any ongoing blocks
            finalize_paragraph(
                &mut elements,
                &mut paragraph_start,
                current_section_idx,
                line_number - 1,
                line_start,
                verbosity,
            );
            finalize_blockquote(
                &mut elements,
                &mut in_blockquote,
                &mut blockquote_start,
                current_section_idx,
                line_number - 1,
                line_start,
                verbosity,
            );
            finalize_table(
                &mut elements,
                &mut in_table,
                &mut table_start,
                current_section_idx,
                line_number - 1,
                line_start,
                verbosity,
            );

            let elem = OutlineElement::new(
                ElementKind::Heading { level },
                text,
                line_number,
                line_start,
            )
            .with_length(line_len, 1);

            elements.push(elem);
            current_section_idx = Some(elements.len() - 1);
        }
        // v2: Detect other block types
        else if verbosity >= Verbosity::Detailed {
            // Image detection: ![alt](url)
            if let Some(alt) = parse_image(line) {
                finalize_paragraph(
                    &mut elements,
                    &mut paragraph_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                let elem = OutlineElement::new(
                    ElementKind::Image { alt: alt.clone() },
                    alt,
                    line_number,
                    line_start,
                )
                .with_length(line_len, 1);

                if let Some(idx) = current_section_idx {
                    elements[idx].children.push(elem);
                } else {
                    elements.push(elem);
                }
            }
            // Blockquote: starts with >
            else if trimmed.starts_with(b">") {
                finalize_paragraph(
                    &mut elements,
                    &mut paragraph_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                if !in_blockquote {
                    in_blockquote = true;
                    blockquote_start = Some((line_number, line_start));
                }
            }
            // Table: contains |
            else if find(trimmed, b"|").is_some() && !trimmed.is_empty() {
                finalize_paragraph(
                    &mut elements,
                    &mut paragraph_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                finalize_blockquote(
                    &mut elements,
                    &mut in_blockquote,
                    &mut blockquote_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                if !in_table {
                    in_table = true;
                    table_start = Some((line_number, line_start));
                }
            }
            // Empty line ends blocks
            else if trimmed.is_empty() {
                finalize_paragraph(
                    &mut elements,
                    &mut paragraph_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                finalize_blockquote(
                    &mut elements,
                    &mut in_blockquote,
                    &mut blockquote_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                finalize_table(
                    &mut elements,
                    &mut in_table,
                    &mut table_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
            }
            // Regular text - paragraph
            else {
                finalize_blockquote(
                    &mut elements,
                    &mut in_blockquote,
                    &mut blockquote_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                finalize_table(
                    &mut elements,
                    &mut in_table,
                    &mut table_start,
                    current_section_idx,
                    line_number - 1,
                    line_start,
                    verbosity,
                );
                if paragraph_start.is_none() {
                    paragraph_start = Some((line_number, line_start));
                }
            }
        }

        byte_offset += line_len + 1;
    }

    // Finalize any remaining blocks
    let final_offset = byte_offset;
    finalize_paragraph(
        &mut elements,
        &mut paragraph_start,
        current_section_idx,
        line_number,
        final_offset,
        verbosity,
    );
    finalize_blockquote(
        &mut elements,
        &mut in_blockquote,
        &mut blockquote_start,
        current_section_idx,
        line_number,
        final_offset,
        verbosity,
    );
    finalize_table(
        &mut elements,
        &mut in_table,
        &mut table_start,
        current_section_idx,
        line_number,
        final_offset,
        verbosity,
    );

    elements
}

/// Check if line is a code fence, returns (fence_char, language)
fn is_code_fence(line: &[u8]) -> Option<(u8, Option<String>)> {
    let trimmed = trim_start(line, 3);
    if trimmed.len() < 3 {
        return None;
    }

    let fence_char = trimmed[0];
    if fence_char != b'`' && fence_char != b'~' {
        return None;
    }

    // Count fence characters
    let fence_count = trimmed.iter().take_while(|&&b| b == fence_char).count();
    if fence_count < 3 {
        return None;
    }

    // Extract language (info string)
    let after_fence = &trimmed[fence_count..];
    let lang = if after_fence.is_empty() {
        None
    } else {
        let lang_bytes = trim_both(after_fence);
        if lang_bytes.is_empty() {
            None
        } else {
            Some(String::from_utf8_lossy(lang_bytes).into_owned())
        }
    };

    Some((fence_char, lang))
}

/// Parse heading from line (ATX style)
fn parse_heading(line: &[u8]) -> Option<(u8, String)> {
    if line.is_empty() || line[0] != b'#' {
        return None;
    }

    // Count # characters
    let level = line.iter().take_while(|&&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }

    // Must be followed by space or end of line
    if line.len() > level {
        if line[level] != b' ' && line[level] != b'\t' {
            return None;
        }
    }

    // Extract text
    let text_start = (level + 1).min(line.len());
    let text = trim_both(&line[text_start..]);

    // Remove trailing # characters (optional closing)
    let text = trim_trailing_hashes(text);

    Some((level as u8, String::from_utf8_lossy(text).into_owned()))
}

/// Parse image from line: ![alt](url)
fn parse_image(line: &[u8]) -> Option<String> {
    let pos = find(line, b"![")?;
    let after_bang = &line[pos + 2..];
    let close_bracket = find(after_bang, b"]")?;
    let alt = &after_bang[..close_bracket];
    Some(String::from_utf8_lossy(alt).into_owned())
}

fn finalize_paragraph(
    elements: &mut Vec<OutlineElement>,
    paragraph_start: &mut Option<(usize, usize)>,
    section_idx: Option<usize>,
    end_line: usize,
    end_offset: usize,
    verbosity: Verbosity,
) {
    if verbosity < Verbosity::Detailed {
        return;
    }
    if let Some((start_line, start_offset)) = paragraph_start.take() {
        if end_line >= start_line {
            let elem = OutlineElement::new(
                ElementKind::Paragraph,
                "paragraph".to_string(),
                start_line,
                start_offset,
            )
            .with_length(
                end_offset.saturating_sub(start_offset),
                end_line - start_line + 1,
            );

            if let Some(idx) = section_idx {
                elements[idx].children.push(elem);
            } else {
                elements.push(elem);
            }
        }
    }
}

fn finalize_blockquote(
    elements: &mut Vec<OutlineElement>,
    in_blockquote: &mut bool,
    blockquote_start: &mut Option<(usize, usize)>,
    section_idx: Option<usize>,
    end_line: usize,
    end_offset: usize,
    verbosity: Verbosity,
) {
    if verbosity < Verbosity::Detailed || !*in_blockquote {
        return;
    }
    if let Some((start_line, start_offset)) = blockquote_start.take() {
        let elem = OutlineElement::new(
            ElementKind::Blockquote,
            "blockquote".to_string(),
            start_line,
            start_offset,
        )
        .with_length(
            end_offset.saturating_sub(start_offset),
            end_line - start_line + 1,
        );

        if let Some(idx) = section_idx {
            elements[idx].children.push(elem);
        } else {
            elements.push(elem);
        }
    }
    *in_blockquote = false;
}

fn finalize_table(
    elements: &mut Vec<OutlineElement>,
    in_table: &mut bool,
    table_start: &mut Option<(usize, usize)>,
    section_idx: Option<usize>,
    end_line: usize,
    end_offset: usize,
    verbosity: Verbosity,
) {
    if verbosity < Verbosity::Detailed || !*in_table {
        return;
    }
    if let Some((start_line, start_offset)) = table_start.take() {
        let elem = OutlineElement::new(
            ElementKind::Table,
            "table".to_string(),
            start_line,
            start_offset,
        )
        .with_length(
            end_offset.saturating_sub(start_offset),
            end_line - start_line + 1,
        );

        if let Some(idx) = section_idx {
            elements[idx].children.push(elem);
        } else {
            elements.push(elem);
        }
    }
    *in_table = false;
}

// endregion: Markdown Parser

// region: C Parser

/// Parse C/C++ source file and extract outline elements
fn parse_c(data: &[u8], _verbosity: Verbosity) -> Vec<OutlineElement> {
    let mut elements = Vec::new();
    let mut line_number = 0usize;
    let mut byte_offset = 0usize;

    // State for function body tracking
    let mut brace_depth = 0i32;
    let mut in_function_body = false;
    let mut function_start: Option<(usize, usize, String)> = None;

    // State for multi-line constructs
    let mut in_multiline_comment = false;
    let mut pending_signature: Option<(usize, usize, Vec<u8>)> = None;

    for line in shared::LineIterator::new(data) {
        line_number += 1;
        let line_start = byte_offset;
        let line_len = line.len();

        // Handle multi-line comments
        if in_multiline_comment {
            if find(line, b"*/").is_some() {
                in_multiline_comment = false;
            }
            byte_offset += line_len + 1;
            continue;
        }

        // Check for comment start
        if find(line, b"/*").is_some() && find(line, b"*/").is_none() {
            in_multiline_comment = true;
            byte_offset += line_len + 1;
            continue;
        }

        // Skip single-line comments for parsing
        let effective_line = if let Some(pos) = find(line, b"//") {
            &line[..pos]
        } else {
            line
        };

        let trimmed = trim_both(effective_line);

        // Handle pending multi-line signature
        if let Some((start_line, start_offset, ref mut sig_bytes)) = pending_signature {
            sig_bytes.extend_from_slice(b" ");
            sig_bytes.extend_from_slice(trimmed);

            // Check if signature is complete
            let has_semicolon = find(&sig_bytes, b";").is_some();
            let has_open_brace = find(&sig_bytes, b"{").is_some();

            if has_semicolon {
                // Declaration
                if let Some(sig) = extract_function_signature(&sig_bytes) {
                    elements.push(
                        OutlineElement::new(
                            ElementKind::FunctionDeclaration,
                            sig,
                            start_line,
                            start_offset,
                        )
                        .with_length(
                            byte_offset + line_len - start_offset,
                            line_number - start_line + 1,
                        ),
                    );
                }
                pending_signature = None;
            } else if has_open_brace {
                // Definition - start tracking body
                if let Some(sig) = extract_function_signature(&sig_bytes) {
                    in_function_body = true;
                    brace_depth = count_braces(&sig_bytes);
                    function_start = Some((start_line, start_offset, sig));
                }
                pending_signature = None;
            }

            byte_offset += line_len + 1;
            continue;
        }

        if !in_function_body {
            // Detect #include
            if trimmed.starts_with(b"#include") {
                if let Some((path, is_system)) = parse_include(trimmed) {
                    elements.push(
                        OutlineElement::new(
                            ElementKind::Include { is_system },
                            path,
                            line_number,
                            line_start,
                        )
                        .with_length(line_len, 1),
                    );
                }
            }
            // Skip other preprocessor directives
            else if trimmed.starts_with(b"#") {
                // Skip
            }
            // Look for function signatures
            else if let Some(result) = try_parse_function_line(trimmed) {
                match result {
                    FunctionParseResult::Declaration(sig) => {
                        elements.push(
                            OutlineElement::new(
                                ElementKind::FunctionDeclaration,
                                sig,
                                line_number,
                                line_start,
                            )
                            .with_length(line_len, 1),
                        );
                    }
                    FunctionParseResult::DefinitionStart(sig) => {
                        in_function_body = true;
                        brace_depth = count_braces(trimmed);
                        function_start = Some((line_number, line_start, sig));
                    }
                    FunctionParseResult::Incomplete(sig_bytes) => {
                        pending_signature = Some((line_number, line_start, sig_bytes));
                    }
                }
            }
        } else {
            // Inside function body - track braces
            brace_depth += count_braces(effective_line);

            if brace_depth <= 0 {
                // Function ended
                if let Some((start_line, start_offset, name)) = function_start.take() {
                    elements.push(
                        OutlineElement::new(
                            ElementKind::FunctionDefinition,
                            name,
                            start_line,
                            start_offset,
                        )
                        .with_length(
                            byte_offset + line_len - start_offset,
                            line_number - start_line + 1,
                        ),
                    );
                }
                in_function_body = false;
                brace_depth = 0;
            }
        }

        byte_offset += line_len + 1;
    }

    elements
}

/// Result of attempting to parse a function line
enum FunctionParseResult {
    Declaration(String),
    DefinitionStart(String),
    Incomplete(Vec<u8>),
}

/// Try to parse a line as a function signature
fn try_parse_function_line(line: &[u8]) -> Option<FunctionParseResult> {
    // Must contain '(' for function
    let paren_pos = find(line, b"(")?;

    // Skip if empty before paren
    if paren_pos == 0 {
        return None;
    }

    let before_paren = &line[..paren_pos];

    // Skip control flow statements
    let control_keywords = [
        b"if" as &[u8],
        b"while",
        b"for",
        b"switch",
        b"catch",
        b"return",
    ];
    for kw in control_keywords {
        if ends_with_identifier(before_paren, kw) {
            return None;
        }
    }

    // Must have an identifier
    let last_ident = extract_last_identifier(before_paren)?;

    // Skip macro-like names (all caps)
    if last_ident
        .bytes()
        .all(|b| b.is_ascii_uppercase() || b == b'_')
        && last_ident.len() > 1
    {
        return None;
    }

    // Check for complete signature
    let has_close_paren = find(&line[paren_pos..], b")").is_some();

    if !has_close_paren {
        // Multi-line signature
        return Some(FunctionParseResult::Incomplete(line.to_vec()));
    }

    // Extract signature up to closing paren
    let close_pos = paren_pos + find(&line[paren_pos..], b")").unwrap() + 1;
    let sig = normalize_signature(&line[..close_pos]);

    // Check if declaration or definition
    let after_sig = &line[close_pos..];
    if find(after_sig, b";").is_some() {
        Some(FunctionParseResult::Declaration(sig))
    } else if find(after_sig, b"{").is_some() || find(line, b"{").is_some() {
        Some(FunctionParseResult::DefinitionStart(sig))
    } else {
        // Could be multi-line (attributes, const, etc.)
        Some(FunctionParseResult::Incomplete(line.to_vec()))
    }
}

/// Extract function signature from accumulated bytes
fn extract_function_signature(data: &[u8]) -> Option<String> {
    let paren_pos = find(data, b"(")?;
    let close_pos = paren_pos + find(&data[paren_pos..], b")")?;
    Some(normalize_signature(&data[..close_pos + 1]))
}

/// Parse #include directive
fn parse_include(line: &[u8]) -> Option<(String, bool)> {
    // Skip "#include"
    let after = &line[8..];
    let trimmed = trim_start(after, usize::MAX);

    if trimmed.starts_with(b"<") {
        // System include
        if let Some(end) = find(trimmed, b">") {
            let path = &trimmed[1..end];
            return Some((String::from_utf8_lossy(path).into_owned(), true));
        }
    } else if trimmed.starts_with(b"\"") {
        // Local include
        if let Some(end) = find(&trimmed[1..], b"\"") {
            let path = &trimmed[1..end + 1];
            return Some((String::from_utf8_lossy(path).into_owned(), false));
        }
    }

    None
}

/// Count net brace changes (handling strings/chars)
fn count_braces(line: &[u8]) -> i32 {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut in_char = false;
    let mut prev_escape = false;

    for &byte in line {
        if prev_escape {
            prev_escape = false;
            continue;
        }

        match byte {
            b'\\' => prev_escape = true,
            b'"' if !in_char => in_string = !in_string,
            b'\'' if !in_string => in_char = !in_char,
            b'{' if !in_string && !in_char => depth += 1,
            b'}' if !in_string && !in_char => depth -= 1,
            _ => {}
        }
    }

    depth
}

/// Check if data ends with a given identifier
fn ends_with_identifier(data: &[u8], ident: &[u8]) -> bool {
    let trimmed = trim_both(data);
    if trimmed.len() < ident.len() {
        return false;
    }

    let suffix = &trimmed[trimmed.len() - ident.len()..];
    if suffix != ident {
        return false;
    }

    // Must be word boundary before
    if trimmed.len() > ident.len() {
        let before = trimmed[trimmed.len() - ident.len() - 1];
        if before.is_ascii_alphanumeric() || before == b'_' {
            return false;
        }
    }

    true
}

/// Extract last identifier from data
fn extract_last_identifier(data: &[u8]) -> Option<String> {
    let trimmed = trim_both(data);

    // Find end of identifier (work backwards)
    let mut end = trimmed.len();
    while end > 0 && trimmed[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    if end == 0 {
        return None;
    }

    // Find start of identifier
    let mut start = end;
    while start > 0 {
        let c = trimmed[start - 1];
        if c.is_ascii_alphanumeric() || c == b'_' {
            start -= 1;
        } else {
            break;
        }
    }

    if start == end {
        return None;
    }

    Some(String::from_utf8_lossy(&trimmed[start..end]).into_owned())
}

/// Normalize function signature (collapse whitespace)
fn normalize_signature(data: &[u8]) -> String {
    let s = String::from_utf8_lossy(data);
    let mut result = String::new();
    let mut prev_space = true;

    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(c);
            prev_space = false;
        }
    }

    result.trim().to_string()
}

// endregion: C Parser

// region: String Utilities

/// Trim leading whitespace (up to max_spaces)
fn trim_start(data: &[u8], max_spaces: usize) -> &[u8] {
    let mut count = 0;
    for (i, &byte) in data.iter().enumerate() {
        if byte == b' ' || byte == b'\t' {
            count += 1;
            if count > max_spaces {
                return &data[i..];
            }
        } else {
            return &data[i..];
        }
    }
    &[]
}

/// Trim trailing whitespace
fn trim_end(data: &[u8]) -> &[u8] {
    let mut end = data.len();
    while end > 0 && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &data[..end]
}

/// Trim both ends
fn trim_both(data: &[u8]) -> &[u8] {
    trim_end(trim_start(data, usize::MAX))
}

/// Remove trailing # characters from heading text
fn trim_trailing_hashes(data: &[u8]) -> &[u8] {
    let mut end = data.len();

    // Skip trailing whitespace
    while end > 0 && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    // Skip trailing #
    while end > 0 && data[end - 1] == b'#' {
        end -= 1;
    }

    // Skip whitespace before trailing #
    while end > 0 && data[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    &data[..end]
}

// endregion: String Utilities

// region: Output Formatting

fn format_element(elem: &OutlineElement, verbosity: Verbosity, file_type: FileType) -> String {
    match file_type {
        FileType::Markdown => format_markdown_element(elem, verbosity),
        FileType::CSource | FileType::CHeader => format_c_element(elem, verbosity),
        FileType::Unknown => String::new(),
    }
}

fn format_markdown_element(elem: &OutlineElement, verbosity: Verbosity) -> String {
    let mut output = String::new();

    match &elem.kind {
        ElementKind::Heading { level } => {
            let prefix = "#".repeat(*level as usize);

            match verbosity {
                Verbosity::Names => {
                    output.push_str(&format!("{} {}\n", prefix, elem.name));
                }
                Verbosity::LineNumbers => {
                    output.push_str(&format!(
                        "{} {:40} [L{}, @{}]\n",
                        prefix, elem.name, elem.line_number, elem.byte_offset
                    ));
                }
                Verbosity::Detailed => {
                    let end_line = elem.line_number + elem.line_count - 1;
                    if elem.line_count > 1 {
                        output.push_str(&format!(
                            "{} {:40} [L{}-{}, @{}, {}B]\n",
                            prefix,
                            elem.name,
                            elem.line_number,
                            end_line,
                            elem.byte_offset,
                            elem.byte_length
                        ));
                    } else {
                        output.push_str(&format!(
                            "{} {:40} [L{}, @{}, {}B]\n",
                            prefix, elem.name, elem.line_number, elem.byte_offset, elem.byte_length
                        ));
                    }

                    // Print children
                    for child in &elem.children {
                        output.push_str(&format_child_block(child));
                    }
                }
            }
        }
        ElementKind::CodeBlock { language } => {
            if verbosity >= Verbosity::Detailed {
                let lang_str = language.as_deref().unwrap_or("code");
                output.push_str(&format!(
                    "  - {} [L{}-{}, {}B]\n",
                    lang_str,
                    elem.line_number,
                    elem.line_number + elem.line_count - 1,
                    elem.byte_length
                ));
            }
        }
        _ => {}
    }

    output
}

fn format_child_block(elem: &OutlineElement) -> String {
    let kind_str = match &elem.kind {
        ElementKind::CodeBlock { language } => language
            .as_ref()
            .map(|l| format!("code ({})", l))
            .unwrap_or_else(|| "code".to_string()),
        ElementKind::Blockquote => "blockquote".to_string(),
        ElementKind::Table => "table".to_string(),
        ElementKind::Image { alt } => format!("image: {}", alt),
        ElementKind::Paragraph => "paragraph".to_string(),
        _ => "block".to_string(),
    };

    if elem.line_count > 1 {
        format!(
            "  - {:36} [L{}-{}, {}B]\n",
            kind_str,
            elem.line_number,
            elem.line_number + elem.line_count - 1,
            elem.byte_length
        )
    } else {
        format!(
            "  - {:36} [L{}, {}B]\n",
            kind_str, elem.line_number, elem.byte_length
        )
    }
}

fn format_c_element(elem: &OutlineElement, verbosity: Verbosity) -> String {
    let mut output = String::new();

    match &elem.kind {
        ElementKind::Include { is_system } => {
            let path = if *is_system {
                format!("<{}>", elem.name)
            } else {
                format!("\"{}\"", elem.name)
            };

            match verbosity {
                Verbosity::Names => {
                    output.push_str(&format!("#include {}\n", path));
                }
                Verbosity::LineNumbers | Verbosity::Detailed => {
                    output.push_str(&format!(
                        "#include {:36} [L{}, @{}]\n",
                        path, elem.line_number, elem.byte_offset
                    ));
                }
            }
        }
        ElementKind::FunctionDeclaration => match verbosity {
            Verbosity::Names => {
                output.push_str(&format!("{:44} [declaration]\n", elem.name));
            }
            Verbosity::LineNumbers => {
                output.push_str(&format!(
                    "{:44} [L{}, @{}, declaration]\n",
                    elem.name, elem.line_number, elem.byte_offset
                ));
            }
            Verbosity::Detailed => {
                output.push_str(&format!(
                    "{:44} [L{}, @{}, {}B, declaration]\n",
                    elem.name, elem.line_number, elem.byte_offset, elem.byte_length
                ));
            }
        },
        ElementKind::FunctionDefinition => match verbosity {
            Verbosity::Names => {
                output.push_str(&format!("{:44} [definition]\n", elem.name));
            }
            Verbosity::LineNumbers => {
                let end_line = elem.line_number + elem.line_count - 1;
                output.push_str(&format!(
                    "{:44} [L{}-{}, @{}, definition]\n",
                    elem.name, elem.line_number, end_line, elem.byte_offset
                ));
            }
            Verbosity::Detailed => {
                let end_line = elem.line_number + elem.line_count - 1;
                output.push_str(&format!(
                    "{:44} [L{}-{}, @{}, {}B, {} lines, definition]\n",
                    elem.name,
                    elem.line_number,
                    end_line,
                    elem.byte_offset,
                    elem.byte_length,
                    elem.line_count
                ));
            }
        },
        _ => {}
    }

    output
}

// endregion: Output Formatting

// region: Input Handling

fn get_input(path: &str) -> io::Result<Vec<u8>> {
    if path == "-" {
        let mut buffer = Vec::new();
        io::stdin().read_to_end(&mut buffer)?;
        Ok(buffer)
    } else {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(mmap.to_vec())
    }
}

// endregion: Input Handling

// region: Main

fn main() {
    let args = Args::parse();

    let verbosity = match args.verbose {
        0 => Verbosity::Names,
        1 => Verbosity::LineNumbers,
        _ => Verbosity::Detailed,
    };

    // Determine file type
    let file_type = if let Some(ref ft) = args.file_type {
        parse_file_type(ft)
    } else if args.input != "-" {
        detect_file_type(&args.input)
    } else {
        eprintln!("Error: cannot detect file type from stdin, use --type");
        process::exit(1);
    };

    if file_type == FileType::Unknown {
        eprintln!("Error: unknown file type, use --type to specify (md, c, h)");
        process::exit(1);
    }

    // Get input
    let data = match get_input(&args.input) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };

    // Parse based on file type
    let elements = match file_type {
        FileType::Markdown => parse_markdown(&data, verbosity),
        FileType::CSource | FileType::CHeader => parse_c(&data, verbosity),
        FileType::Unknown => Vec::new(),
    };

    // Output
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for elem in &elements {
        let formatted = format_element(elem, verbosity, file_type);
        if let Err(e) = handle.write_all(formatted.as_bytes()) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                break;
            }
            eprintln!("Error writing output: {}", e);
            process::exit(1);
        }
    }
}

// endregion: Main

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading_detection() {
        assert_eq!(parse_heading(b"# Title"), Some((1, "Title".to_string())));
        assert_eq!(
            parse_heading(b"## Level 2"),
            Some((2, "Level 2".to_string()))
        );
        assert_eq!(
            parse_heading(b"###### Level 6"),
            Some((6, "Level 6".to_string()))
        );
        assert_eq!(parse_heading(b"# Title ##"), Some((1, "Title".to_string())));

        // Invalid
        assert_eq!(parse_heading(b"####### Too many"), None);
        assert_eq!(parse_heading(b"#NoSpace"), None);
        assert_eq!(parse_heading(b"Not a heading"), None);
    }

    #[test]
    fn test_code_fence_detection() {
        assert!(is_code_fence(b"```").is_some());
        assert!(is_code_fence(b"```rust").is_some());
        assert!(is_code_fence(b"~~~").is_some());
        assert!(is_code_fence(b"  ```").is_some());

        assert!(is_code_fence(b"``").is_none());
        assert!(is_code_fence(b"text").is_none());

        // Check language extraction
        let (_, lang) = is_code_fence(b"```rust").unwrap();
        assert_eq!(lang, Some("rust".to_string()));
    }

    #[test]
    fn test_include_parsing() {
        assert_eq!(
            parse_include(b"#include <stdio.h>"),
            Some(("stdio.h".to_string(), true))
        );
        assert_eq!(
            parse_include(b"#include \"myheader.h\""),
            Some(("myheader.h".to_string(), false))
        );
    }

    #[test]
    fn test_brace_counting() {
        assert_eq!(count_braces(b"{"), 1);
        assert_eq!(count_braces(b"}"), -1);
        assert_eq!(count_braces(b"{}"), 0);
        assert_eq!(count_braces(b"{ { } }"), 0);
        assert_eq!(count_braces(b"\"{\""), 0); // In string
        assert_eq!(count_braces(b"'{'"), 0); // In char
    }

    #[test]
    fn test_extract_last_identifier() {
        assert_eq!(
            extract_last_identifier(b"int main"),
            Some("main".to_string())
        );
        assert_eq!(
            extract_last_identifier(b"void *foo"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_last_identifier(b"static int bar"),
            Some("bar".to_string())
        );
    }

    #[test]
    fn test_ends_with_identifier() {
        assert!(ends_with_identifier(b"if", b"if"));
        assert!(ends_with_identifier(b"   if", b"if"));
        assert!(!ends_with_identifier(b"elif", b"if"));
    }

    #[test]
    fn test_markdown_parsing() {
        let md = b"# Title\n\nSome text.\n\n## Section\n\n```rust\ncode\n```\n";
        let elements = parse_markdown(md, Verbosity::Names);

        assert_eq!(elements.len(), 2);
        assert!(matches!(
            elements[0].kind,
            ElementKind::Heading { level: 1 }
        ));
        assert!(matches!(
            elements[1].kind,
            ElementKind::Heading { level: 2 }
        ));
    }

    #[test]
    fn test_c_parsing() {
        let c = b"#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n";
        let elements = parse_c(c, Verbosity::Names);

        assert_eq!(elements.len(), 2);
        assert!(matches!(
            elements[0].kind,
            ElementKind::Include { is_system: true }
        ));
        assert!(matches!(elements[1].kind, ElementKind::FunctionDefinition));
    }
}

// endregion: Tests
