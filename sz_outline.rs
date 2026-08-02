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

use std::borrow::Cow;
use std::io::{self, Write};
use std::path::Path;
use std::process;

use clap::Parser;
use stringzilla::sz::{find, StringZillableUnary};

mod shared;
use shared::{exit_with_error, json_text_field_to, stdout_writer};

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
enum ElementKind<'a> {
    // Markdown elements
    Heading { level: u8 },
    CodeBlock { language: Option<Cow<'a, [u8]>> },
    Blockquote,
    Table,
    Image { alt: Cow<'a, [u8]> },
    Paragraph,

    // C elements
    Include { is_system: bool },
    FunctionDeclaration,
    FunctionDefinition,
}

/// Represents an outline element with position info
#[derive(Debug, Clone)]
struct OutlineElement<'a> {
    kind: ElementKind<'a>,
    name: Cow<'a, [u8]>,
    line_number: usize,
    byte_offset: usize,
    byte_length: usize,
    line_count: usize,
    children: Vec<OutlineElement<'a>>,
}

impl<'a> OutlineElement<'a> {
    fn new(
        kind: ElementKind<'a>,
        name: Cow<'a, [u8]>,
        line_number: usize,
        byte_offset: usize,
    ) -> Self {
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

/// Pending code-block start while scanning: (start line, start byte offset, fence language).
type CodeBlockStart<'a> = (usize, usize, Option<Cow<'a, [u8]>>);

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

    /// Emit JSON Lines, flat records linked by parent_line
    #[arg(long, help_heading = "Output Formats")]
    json: bool,
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
fn parse_markdown<'a>(data: &'a [u8], verbosity: Verbosity) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;
    let mut byte_offset = 0usize;

    // State for code blocks
    let mut in_code_block = false;
    let mut code_fence_char: u8 = 0;
    let mut code_block_start: Option<CodeBlockStart<'a>> = None;

    // State for v2 block tracking
    let mut current_section_idx: Option<usize> = None;
    let mut in_blockquote = false;
    let mut blockquote_start: Option<(usize, usize)> = None;
    let mut in_table = false;
    let mut table_start: Option<(usize, usize)> = None;
    let mut paragraph_start: Option<(usize, usize)> = None;

    for line in shared::LineIter::new(data, shared::Newlines::Lf) {
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
                        Cow::Borrowed(b"code"),
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
                Cow::Borrowed(text),
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
                    ElementKind::Image {
                        alt: Cow::Borrowed(alt),
                    },
                    Cow::Borrowed(alt),
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
fn is_code_fence<'a>(line: &'a [u8]) -> Option<(u8, Option<Cow<'a, [u8]>>)> {
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
            Some(Cow::Borrowed(lang_bytes))
        }
    };

    Some((fence_char, lang))
}

/// Parse heading from line (ATX style)
fn parse_heading(line: &[u8]) -> Option<(u8, &[u8])> {
    if line.is_empty() || line[0] != b'#' {
        return None;
    }

    // Count # characters
    let level = line.iter().take_while(|&&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }

    // Must be followed by space or end of line
    if line.len() > level
        && line[level] != b' ' && line[level] != b'\t' {
            return None;
        }

    // Extract text
    let text_start = (level + 1).min(line.len());
    let text = trim_both(&line[text_start..]);

    // Remove trailing # characters (optional closing)
    let text = trim_trailing_hashes(text);

    Some((level as u8, text))
}

/// Parse image from line: ![alt](url)
fn parse_image(line: &[u8]) -> Option<&[u8]> {
    let pos = find(line, b"![")?;
    let after_bang = &line[pos + 2..];
    let close_bracket = find(after_bang, b"]")?;
    let alt = &after_bang[..close_bracket];
    Some(alt)
}

fn finalize_paragraph<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
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
                Cow::Borrowed(b"paragraph"),
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

fn finalize_blockquote<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
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
            Cow::Borrowed(b"blockquote"),
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

fn finalize_table<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
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
            Cow::Borrowed(b"table"),
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
fn parse_c<'a>(data: &'a [u8], _verbosity: Verbosity) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;
    let mut byte_offset = 0usize;

    // State for function body tracking
    let mut brace_depth = 0i32;
    let mut in_function_body = false;
    let mut function_start: Option<(usize, usize, Cow<'a, [u8]>)> = None;

    // State for multi-line constructs
    let mut in_multiline_comment = false;
    let mut pending_signature: Option<(usize, usize, Vec<u8>)> = None;

    for line in shared::LineIter::new(data, shared::Newlines::Lf) {
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
                // Declaration — the signature was accumulated across lines into a
                // fresh buffer, so it cannot borrow from `data`; keep it owned.
                if let Some(sig) = extract_function_signature(sig_bytes) {
                    elements.push(
                        OutlineElement::new(
                            ElementKind::FunctionDeclaration,
                            Cow::Owned(sig.into_owned()),
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
                if let Some(sig) = extract_function_signature(sig_bytes) {
                    in_function_body = true;
                    brace_depth = count_braces(sig_bytes);
                    function_start = Some((start_line, start_offset, Cow::Owned(sig.into_owned())));
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
                            Cow::Borrowed(path),
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
enum FunctionParseResult<'a> {
    Declaration(Cow<'a, [u8]>),
    DefinitionStart(Cow<'a, [u8]>),
    Incomplete(Vec<u8>),
}

/// Try to parse a line as a function signature
fn try_parse_function_line<'a>(line: &'a [u8]) -> Option<FunctionParseResult<'a>> {
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
        .iter()
        .all(|&b| b.is_ascii_uppercase() || b == b'_')
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
fn extract_function_signature<'a>(data: &'a [u8]) -> Option<Cow<'a, [u8]>> {
    let paren_pos = find(data, b"(")?;
    let close_pos = paren_pos + find(&data[paren_pos..], b")")?;
    Some(normalize_signature(&data[..close_pos + 1]))
}

/// Parse #include directive
fn parse_include(line: &[u8]) -> Option<(&[u8], bool)> {
    // Skip "#include"
    let after = &line[8..];
    let trimmed = trim_start(after, usize::MAX);

    if trimmed.starts_with(b"<") {
        // System include
        if let Some(end) = find(trimmed, b">") {
            let path = &trimmed[1..end];
            return Some((path, true));
        }
    } else if trimmed.starts_with(b"\"") {
        // Local include
        if let Some(end) = find(&trimmed[1..], b"\"") {
            let path = &trimmed[1..end + 1];
            return Some((path, false));
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
fn extract_last_identifier(data: &[u8]) -> Option<&[u8]> {
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

    Some(&trimmed[start..end])
}

/// Normalize a function signature: collapse whitespace runs to single spaces,
/// via StringZilla's SIMD whitespace splitter. Builds one output string (the old
/// char-loop allocated twice: the buffer plus a trimmed copy).
fn normalize_signature<'a>(data: &'a [u8]) -> Cow<'a, [u8]> {
    let mut result: Vec<u8> = Vec::new();
    for token in data.sz_utf8_split_whitespaces().skip_empty() {
        if !result.is_empty() {
            result.push(b' ');
        }
        result.extend_from_slice(token);
    }
    // Borrow the original bytes when collapsing was a no-op (already normalized).
    if result == data {
        Cow::Borrowed(data)
    } else {
        Cow::Owned(result)
    }
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

/// The record kind for one element, matching the human renderer's vocabulary.
fn element_kind_name(kind: &ElementKind<'_>) -> &'static str {
    match kind {
        ElementKind::Heading { .. } => "heading",
        ElementKind::CodeBlock { .. } => "code_block",
        ElementKind::Blockquote => "blockquote",
        ElementKind::Table => "table",
        ElementKind::Image { .. } => "image",
        ElementKind::Paragraph => "paragraph",
        ElementKind::Include { .. } => "include",
        ElementKind::FunctionDeclaration => "function_declaration",
        ElementKind::FunctionDefinition => "function_definition",
    }
}

/// Write one element as a flat JSON Lines record. Children are emitted as their own
/// records carrying `parent_line`, rather than nested, which keeps `jq` filters simple.
fn write_element_json(
    out: &mut dyn Write,
    elem: &OutlineElement<'_>,
    parent_line: Option<usize>,
) -> io::Result<()> {
    out.write_all(br#"{"type":"element","data":{"kind":""#)?;
    out.write_all(element_kind_name(&elem.kind).as_bytes())?;
    out.write_all(br#"","name":"#)?;
    json_text_field_to(out, &elem.name)?;

    match &elem.kind {
        ElementKind::Heading { level } => write!(out, r#","level":{}"#, level)?,
        ElementKind::CodeBlock { language } => match language {
            Some(language) => {
                out.write_all(br#","language":"#)?;
                json_text_field_to(out, language)?;
            }
            None => out.write_all(br#","language":null"#)?,
        },
        ElementKind::Include { is_system } => write!(out, r#","is_system":{}"#, is_system)?,
        _ => {}
    }

    write!(
        out,
        r#","line_number":{},"line_count":{},"byte_offset":{},"byte_length":{}"#,
        elem.line_number, elem.line_count, elem.byte_offset, elem.byte_length
    )?;
    match parent_line {
        Some(line) => write!(out, r#","parent_line":{}"#, line)?,
        None => out.write_all(br#","parent_line":null"#)?,
    }
    out.write_all(b"}}\n")?;

    for child in &elem.children {
        write_element_json(out, child, Some(elem.line_number))?;
    }
    Ok(())
}

fn write_element(
    out: &mut dyn Write,
    elem: &OutlineElement<'_>,
    verbosity: Verbosity,
    file_type: FileType,
) -> io::Result<()> {
    match file_type {
        FileType::Markdown => write_markdown_element(out, elem, verbosity),
        FileType::CSource | FileType::CHeader => write_c_element(out, elem, verbosity),
        FileType::Unknown => Ok(()),
    }
}

fn write_markdown_element(
    out: &mut dyn Write,
    elem: &OutlineElement<'_>,
    verbosity: Verbosity,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&elem.name);
    match &elem.kind {
        ElementKind::Heading { level } => {
            // Headings are levels 1..=6 — slice a static run, no allocation.
            let prefix = &"######"[..(*level as usize).min(6)];
            match verbosity {
                Verbosity::Names => writeln!(out, "{} {}", prefix, name)?,
                Verbosity::LineNumbers => writeln!(
                    out,
                    "{} {:40} [L{}, @{}]",
                    prefix, name, elem.line_number, elem.byte_offset
                )?,
                Verbosity::Detailed => {
                    let end_line = elem.line_number + elem.line_count - 1;
                    if elem.line_count > 1 {
                        writeln!(
                            out,
                            "{} {:40} [L{}-{}, @{}, {}B]",
                            prefix,
                            name,
                            elem.line_number,
                            end_line,
                            elem.byte_offset,
                            elem.byte_length
                        )?;
                    } else {
                        writeln!(
                            out,
                            "{} {:40} [L{}, @{}, {}B]",
                            prefix, name, elem.line_number, elem.byte_offset, elem.byte_length
                        )?;
                    }
                    for child in &elem.children {
                        write_child_block(out, child)?;
                    }
                }
            }
        }
        ElementKind::CodeBlock { language } => {
            if verbosity >= Verbosity::Detailed {
                let lang_cow = language.as_ref().map(|l| String::from_utf8_lossy(l));
                let lang_str = lang_cow.as_deref().unwrap_or("code");
                writeln!(
                    out,
                    "  - {} [L{}-{}, {}B]",
                    lang_str,
                    elem.line_number,
                    elem.line_number + elem.line_count - 1,
                    elem.byte_length
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_child_block(out: &mut dyn Write, elem: &OutlineElement<'_>) -> io::Result<()> {
    // Constant labels borrow; only code(lang)/image build a small owned label for the {:36} pad.
    let kind_str: Cow<str> = match &elem.kind {
        ElementKind::CodeBlock { language } => match language {
            Some(l) => Cow::Owned(format!("code ({})", String::from_utf8_lossy(l))),
            None => Cow::Borrowed("code"),
        },
        ElementKind::Blockquote => Cow::Borrowed("blockquote"),
        ElementKind::Table => Cow::Borrowed("table"),
        ElementKind::Image { alt } => {
            Cow::Owned(format!("image: {}", String::from_utf8_lossy(alt)))
        }
        ElementKind::Paragraph => Cow::Borrowed("paragraph"),
        _ => Cow::Borrowed("block"),
    };

    if elem.line_count > 1 {
        writeln!(
            out,
            "  - {:36} [L{}-{}, {}B]",
            kind_str,
            elem.line_number,
            elem.line_number + elem.line_count - 1,
            elem.byte_length
        )
    } else {
        writeln!(
            out,
            "  - {:36} [L{}, {}B]",
            kind_str, elem.line_number, elem.byte_length
        )
    }
}

fn write_c_element(
    out: &mut dyn Write,
    elem: &OutlineElement<'_>,
    verbosity: Verbosity,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&elem.name);
    match &elem.kind {
        ElementKind::Include { is_system } => {
            let (open, close) = if *is_system { ("<", ">") } else { ("\"", "\"") };
            match verbosity {
                Verbosity::Names => {
                    writeln!(out, "#include {}{}{}", open, name, close)?;
                }
                Verbosity::LineNumbers | Verbosity::Detailed => {
                    // Build the padded `<path>` field (small, per include) for {:36}.
                    let path = format!("{}{}{}", open, name, close);
                    writeln!(
                        out,
                        "#include {:36} [L{}, @{}]",
                        path, elem.line_number, elem.byte_offset
                    )?;
                }
            }
        }
        ElementKind::FunctionDeclaration => match verbosity {
            Verbosity::Names => writeln!(out, "{:44} [declaration]", name)?,
            Verbosity::LineNumbers => writeln!(
                out,
                "{:44} [L{}, @{}, declaration]",
                name, elem.line_number, elem.byte_offset
            )?,
            Verbosity::Detailed => writeln!(
                out,
                "{:44} [L{}, @{}, {}B, declaration]",
                name, elem.line_number, elem.byte_offset, elem.byte_length
            )?,
        },
        ElementKind::FunctionDefinition => match verbosity {
            Verbosity::Names => writeln!(out, "{:44} [definition]", name)?,
            Verbosity::LineNumbers => {
                let end_line = elem.line_number + elem.line_count - 1;
                writeln!(
                    out,
                    "{:44} [L{}-{}, @{}, definition]",
                    name, elem.line_number, end_line, elem.byte_offset
                )?;
            }
            Verbosity::Detailed => {
                let end_line = elem.line_number + elem.line_count - 1;
                writeln!(
                    out,
                    "{:44} [L{}-{}, @{}, {}B, {} lines, definition]",
                    name,
                    elem.line_number,
                    end_line,
                    elem.byte_offset,
                    elem.byte_length,
                    elem.line_count
                )?;
            }
        },
        _ => {}
    }
    Ok(())
}

// endregion: Output Formatting

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

    // Get input — mmap is borrowed, not copied.
    let input = match shared::get_input(Some(&args.input)) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("Error reading input: {}", e);
            process::exit(1);
        }
    };
    let data = input.as_bytes();

    // Parse based on file type
    let elements = match file_type {
        FileType::Markdown => parse_markdown(data, verbosity),
        FileType::CSource | FileType::CHeader => parse_c(data, verbosity),
        FileType::Unknown => Vec::new(),
    };

    // Output
    let mut handle = stdout_writer();

    for elem in &elements {
        let written = if args.json {
            write_element_json(&mut handle, elem, None)
        } else {
            write_element(&mut handle, elem, verbosity, file_type)
        };
        if let Err(error) = written {
            if error.kind() == io::ErrorKind::BrokenPipe {
                break;
            }
            exit_with_error(&mut handle, &error, "Error writing output");
        }
    }
}

// endregion: Main

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_markdown_heading_levels() {
        assert_eq!(parse_heading(b"# Title"), Some((1, b"Title".as_slice())));
        assert_eq!(
            parse_heading(b"## Level 2"),
            Some((2, b"Level 2".as_slice()))
        );
        assert_eq!(
            parse_heading(b"###### Level 6"),
            Some((6, b"Level 6".as_slice()))
        );
        assert_eq!(parse_heading(b"# Title ##"), Some((1, b"Title".as_slice())));

        // Invalid
        assert_eq!(parse_heading(b"####### Too many"), None);
        assert_eq!(parse_heading(b"#NoSpace"), None);
        assert_eq!(parse_heading(b"Not a heading"), None);
    }

    #[test]
    fn detects_code_fences_and_language() {
        assert!(is_code_fence(b"```").is_some());
        assert!(is_code_fence(b"```rust").is_some());
        assert!(is_code_fence(b"~~~").is_some());
        assert!(is_code_fence(b"  ```").is_some());

        assert!(is_code_fence(b"``").is_none());
        assert!(is_code_fence(b"text").is_none());

        // Check language extraction
        let (_, lang) = is_code_fence(b"```rust").unwrap();
        assert_eq!(lang, Some(Cow::Borrowed(b"rust".as_slice())));
    }

    #[test]
    fn parses_c_include_directives() {
        assert_eq!(
            parse_include(b"#include <stdio.h>"),
            Some((b"stdio.h".as_slice(), true))
        );
        assert_eq!(
            parse_include(b"#include \"myheader.h\""),
            Some((b"myheader.h".as_slice(), false))
        );
    }

    #[test]
    fn counts_braces_ignoring_strings_and_chars() {
        assert_eq!(count_braces(b"{"), 1);
        assert_eq!(count_braces(b"}"), -1);
        assert_eq!(count_braces(b"{}"), 0);
        assert_eq!(count_braces(b"{ { } }"), 0);
        assert_eq!(count_braces(b"\"{\""), 0); // In string
        assert_eq!(count_braces(b"'{'"), 0); // In char
    }

    #[test]
    fn extracts_last_identifier_from_signature() {
        assert_eq!(
            extract_last_identifier(b"int main"),
            Some(b"main".as_slice())
        );
        assert_eq!(
            extract_last_identifier(b"void *foo"),
            Some(b"foo".as_slice())
        );
        assert_eq!(
            extract_last_identifier(b"static int bar"),
            Some(b"bar".as_slice())
        );
    }

    #[test]
    fn detects_trailing_identifier() {
        assert!(ends_with_identifier(b"if", b"if"));
        assert!(ends_with_identifier(b"   if", b"if"));
        assert!(!ends_with_identifier(b"elif", b"if"));
    }

    #[test]
    fn parses_markdown_headings_into_elements() {
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
    fn parses_c_includes_and_functions() {
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
