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

use clap::Parser;
use stringzilla::sz::{find, StringZillableUnary};

mod shared;
use shared::*;

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

/// Line number and byte offset of one point in the input.
#[derive(Clone, Copy, Debug)]
struct Position {
    line: usize,
    offset: usize,
}

/// A Markdown block that accumulates consecutive lines of one kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Block {
    Paragraph,
    Blockquote,
    Table,
}

impl Block {
    /// The element kind this block becomes once closed.
    fn kind(self) -> ElementKind<'static> {
        match self {
            Block::Paragraph => ElementKind::Paragraph,
            Block::Blockquote => ElementKind::Blockquote,
            Block::Table => ElementKind::Table,
        }
    }

    /// The element name this block becomes once closed.
    fn label(self) -> &'static [u8] {
        match self {
            Block::Paragraph => b"paragraph",
            Block::Blockquote => b"blockquote",
            Block::Table => b"table",
        }
    }
}

/// The one block currently accumulating lines. At most one is open at a time, so
/// block spans cannot overlap.
#[derive(Clone, Copy, Debug)]
struct OpenBlock {
    block: Block,
    start: Position,
}

/// A fenced code block being scanned: the fence character that closes it, its info
/// string, and where it opened.
#[derive(Clone, Debug)]
struct OpenCodeBlock<'a> {
    marker: u8,
    language: Option<Cow<'a, [u8]>>,
    start: Position,
}

/// A code fence line: the fence character and its info string.
#[derive(Clone, Debug)]
struct Fence<'a> {
    marker: u8,
    language: Option<Cow<'a, [u8]>>,
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

    /// Emit JSON Lines, flat records linked by parent_line
    #[arg(long, help_heading = "Output Formats")]
    json: bool,
}

// endregion: CLI Interface

// region: File Type Detection

/// File type from the path's extension, or `None` when the extension is not outlined.
fn detect_file_type(path: &str) -> Option<FileType> {
    match Path::new(path).extension()?.to_str()? {
        "md" | "markdown" => Some(FileType::Markdown),
        "c" => Some(FileType::CSource),
        "h" => Some(FileType::CHeader),
        _ => None,
    }
}

/// File type from the `--type` argument, or `None` when the name is not outlined.
fn parse_file_type(name: &str) -> Option<FileType> {
    match name.to_lowercase().as_str() {
        "md" | "markdown" => Some(FileType::Markdown),
        "c" => Some(FileType::CSource),
        "h" => Some(FileType::CHeader),
        _ => None,
    }
}

// endregion: File Type Detection

// region: Markdown Parser

/// Parse Markdown file and extract outline elements
fn parse_markdown<'a>(data: &'a [u8], verbosity: Verbosity) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;
    let mut current_section: Option<usize> = None;
    let mut open_code_block: Option<OpenCodeBlock<'a>> = None;
    let mut open_block: Option<OpenBlock> = None;

    for line in LineIter::new(data, Newlines::Lf) {
        line_number += 1;
        let line_start = offset_within(data, line);
        let line_len = line.len();
        let start = Position {
            line: line_number,
            offset: line_start,
        };
        // A block interrupted by this line ends on the previous line, at this line's byte.
        let interrupted = Position {
            line: line_number - 1,
            offset: line_start,
        };

        // Code fences: the same fence character closes what it opened, and a
        // different one inside the block is content.
        if let Some(fence) = is_code_fence(line) {
            if let Some(closed) = open_code_block.take_if(|open| open.marker == fence.marker) {
                let element = OutlineElement::new(
                    ElementKind::CodeBlock {
                        language: closed.language,
                    },
                    Cow::Borrowed(b"code"),
                    closed.start.line,
                    closed.start.offset,
                )
                .with_length(
                    line_start + line_len - closed.start.offset,
                    line_number - closed.start.line + 1,
                );
                if verbosity >= Verbosity::Detailed {
                    push_element(&mut elements, current_section, element);
                }
            } else if open_code_block.is_none() {
                close_block(&mut elements, &mut open_block, current_section, interrupted);
                open_code_block = Some(OpenCodeBlock {
                    marker: fence.marker,
                    language: fence.language,
                    start,
                });
            }
            continue;
        }

        if open_code_block.is_some() {
            continue;
        }

        let trimmed = trim_start(line, 3);

        if let Some((level, text)) = parse_heading(trimmed) {
            close_block(&mut elements, &mut open_block, current_section, interrupted);
            elements.push(
                OutlineElement::new(
                    ElementKind::Heading { level },
                    Cow::Borrowed(text),
                    line_number,
                    line_start,
                )
                .with_length(line_len, 1),
            );
            current_section = Some(elements.len() - 1);
        }
        // Blocks below headings are only reported at the most verbose level.
        else if verbosity >= Verbosity::Detailed {
            // Image: `![alt](url)`
            if let Some(alt) = parse_image(line) {
                close_block(&mut elements, &mut open_block, current_section, interrupted);
                let element = OutlineElement::new(
                    ElementKind::Image {
                        alt: Cow::Borrowed(alt),
                    },
                    Cow::Borrowed(alt),
                    line_number,
                    line_start,
                )
                .with_length(line_len, 1);
                push_element(&mut elements, current_section, element);
            }
            // Blockquote: starts with `>`
            else if trimmed.starts_with(b">") {
                open_or_extend(
                    &mut elements,
                    &mut open_block,
                    current_section,
                    Block::Blockquote,
                    start,
                );
            }
            // Table: contains `|`
            else if !trimmed.is_empty() && find(trimmed, b"|").is_some() {
                open_or_extend(
                    &mut elements,
                    &mut open_block,
                    current_section,
                    Block::Table,
                    start,
                );
            }
            // Blank line ends the open block
            else if trimmed.is_empty() {
                close_block(&mut elements, &mut open_block, current_section, interrupted);
            }
            // Anything else is paragraph text
            else {
                open_or_extend(
                    &mut elements,
                    &mut open_block,
                    current_section,
                    Block::Paragraph,
                    start,
                );
            }
        }
    }

    // A block still open at end of input ends there, whether or not the file
    // ends with a newline.
    let end_of_input = Position {
        line: line_number,
        offset: data.len(),
    };
    close_block(
        &mut elements,
        &mut open_block,
        current_section,
        end_of_input,
    );
    elements
}

/// Record an element under the current section, or at the top level when there is none.
fn push_element<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
    section: Option<usize>,
    element: OutlineElement<'a>,
) {
    match section {
        Some(index) => elements[index].children.push(element),
        None => elements.push(element),
    }
}

/// Close the open block, if any, recording it as ending at `end`.
fn close_block<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
    open: &mut Option<OpenBlock>,
    section: Option<usize>,
    end: Position,
) {
    let Some(OpenBlock { block, start }) = open.take() else {
        return;
    };
    debug_assert!(
        end.line >= start.line,
        "a block closes on or after the line it opened on"
    );
    let element = OutlineElement::new(
        block.kind(),
        Cow::Borrowed(block.label()),
        start.line,
        start.offset,
    )
    .with_length(
        end.offset.saturating_sub(start.offset),
        end.line - start.line + 1,
    );
    push_element(elements, section, element);
}

/// Keep accumulating into the open block when it is already `block`, otherwise close
/// it — ending on the previous line — and open a fresh one at `start`.
fn open_or_extend<'a>(
    elements: &mut Vec<OutlineElement<'a>>,
    open: &mut Option<OpenBlock>,
    section: Option<usize>,
    block: Block,
    start: Position,
) {
    if open.is_some_and(|current| current.block == block) {
        return;
    }
    let interrupted = Position {
        line: start.line - 1,
        offset: start.offset,
    };
    close_block(elements, open, section, interrupted);
    *open = Some(OpenBlock { block, start });
}

/// Read a code fence line — three or more backticks or tildes, plus an info string.
fn is_code_fence(line: &[u8]) -> Option<Fence<'_>> {
    let trimmed = trim_start(line, 3);
    let marker = *trimmed.first()?;
    if marker != b'`' && marker != b'~' {
        return None;
    }
    let width = trimmed.iter().take_while(|&&byte| byte == marker).count();
    (width >= 3).then(|| Fence {
        marker,
        language: Some(trimmed[width..].trim_ascii())
            .filter(|info| !info.is_empty())
            .map(Cow::Borrowed),
    })
}

/// Parse heading from line (ATX style)
fn parse_heading(line: &[u8]) -> Option<(u8, &[u8])> {
    if line.first() != Some(&b'#') {
        return None;
    }

    // Count # characters
    let level = line.iter().take_while(|&&byte| byte == b'#').count();
    if level > 6 {
        return None;
    }

    // Must be followed by space or end of line
    if line.len() > level && line[level] != b' ' && line[level] != b'\t' {
        return None;
    }

    // Extract text, dropping the optional closing run of `#`
    let text_start = (level + 1).min(line.len());
    let text = trim_trailing_hashes(line[text_start..].trim_ascii());
    Some((level as u8, text))
}

/// Parse image from line: ![alt](url)
fn parse_image(line: &[u8]) -> Option<&[u8]> {
    let position = find(line, b"![")?;
    let after_bang = &line[position + 2..];
    let close_bracket = find(after_bang, b"]")?;
    Some(&after_bang[..close_bracket])
}

// endregion: Markdown Parser

// region: C Parser

/// Parse C/C++ source file and extract outline elements
fn parse_c<'a>(data: &'a [u8], _verbosity: Verbosity) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;

    // State for function body tracking
    let mut brace_depth = 0i32;
    let mut in_function_body = false;
    let mut function_start: Option<(usize, usize, Cow<'a, [u8]>)> = None;

    // State for multi-line constructs
    let mut in_multiline_comment = false;
    let mut pending_signature: Option<(usize, usize, Vec<u8>)> = None;

    for line in LineIter::new(data, Newlines::Lf) {
        line_number += 1;
        let line_start = offset_within(data, line);
        let line_end = line_start + line.len();

        // Handle multi-line comments
        if in_multiline_comment {
            if find(line, b"*/").is_some() {
                in_multiline_comment = false;
            }
            continue;
        }

        // Check for comment start
        if find(line, b"/*").is_some() && find(line, b"*/").is_none() {
            in_multiline_comment = true;
            continue;
        }

        // Skip single-line comments for parsing
        let effective_line = match find(line, b"//") {
            Some(position) => &line[..position],
            None => line,
        };

        let trimmed = effective_line.trim_ascii();

        // Handle pending multi-line signature
        if let Some((start_line, start_offset, ref mut signature_bytes)) = pending_signature {
            signature_bytes.extend_from_slice(b" ");
            signature_bytes.extend_from_slice(trimmed);

            // Check if signature is complete
            let has_semicolon = find(&signature_bytes, b";").is_some();
            let has_open_brace = find(&signature_bytes, b"{").is_some();

            if has_semicolon {
                // Declaration — the signature was accumulated across lines into a
                // fresh buffer, so it cannot borrow from `data`; keep it owned.
                if let Some(signature) = extract_function_signature(signature_bytes) {
                    elements.push(
                        OutlineElement::new(
                            ElementKind::FunctionDeclaration,
                            Cow::Owned(signature.into_owned()),
                            start_line,
                            start_offset,
                        )
                        .with_length(line_end - start_offset, line_number - start_line + 1),
                    );
                }
                pending_signature = None;
            } else if has_open_brace {
                // Definition - start tracking body
                if let Some(signature) = extract_function_signature(signature_bytes) {
                    in_function_body = true;
                    brace_depth = count_braces(signature_bytes);
                    function_start =
                        Some((start_line, start_offset, Cow::Owned(signature.into_owned())));
                }
                pending_signature = None;
            }

            continue;
        }

        if !in_function_body {
            // Detect #include
            if let Some((path, is_system)) = parse_include(trimmed) {
                elements.push(
                    OutlineElement::new(
                        ElementKind::Include { is_system },
                        Cow::Borrowed(path),
                        line_number,
                        line_start,
                    )
                    .with_length(line.len(), 1),
                );
            }
            // Skip other preprocessor directives
            else if trimmed.starts_with(b"#") {
                // Skip
            }
            // Look for function signatures
            else if let Some(result) = try_parse_function_line(trimmed) {
                match result {
                    FunctionParseResult::Declaration(signature) => {
                        elements.push(
                            OutlineElement::new(
                                ElementKind::FunctionDeclaration,
                                signature,
                                line_number,
                                line_start,
                            )
                            .with_length(line.len(), 1),
                        );
                    }
                    FunctionParseResult::DefinitionStart(signature) => {
                        in_function_body = true;
                        brace_depth = count_braces(trimmed);
                        function_start = Some((line_number, line_start, signature));
                    }
                    FunctionParseResult::Incomplete(signature_bytes) => {
                        pending_signature = Some((line_number, line_start, signature_bytes));
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
                        .with_length(line_end - start_offset, line_number - start_line + 1),
                    );
                }
                in_function_body = false;
                brace_depth = 0;
            }
        }
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
fn try_parse_function_line(line: &[u8]) -> Option<FunctionParseResult<'_>> {
    // Must contain '(' for function
    let open_paren = find(line, b"(")?;

    // Skip if empty before paren
    if open_paren == 0 {
        return None;
    }

    let before_paren = &line[..open_paren];

    // Skip control flow statements
    let control_keywords = [
        b"if" as &[u8],
        b"while",
        b"for",
        b"switch",
        b"catch",
        b"return",
    ];
    for keyword in control_keywords {
        if ends_with_identifier(before_paren, keyword) {
            return None;
        }
    }

    // Must have an identifier
    let last_identifier = extract_last_identifier(before_paren)?;

    // Skip macro-like names (all caps)
    if last_identifier
        .iter()
        .all(|&byte| byte.is_ascii_uppercase() || byte == b'_')
        && last_identifier.len() > 1
    {
        return None;
    }

    // A signature that does not close on this line continues on the next
    let Some(close_paren) = find(&line[open_paren..], b")") else {
        return Some(FunctionParseResult::Incomplete(line.to_vec()));
    };

    // Extract signature up to closing paren
    let signature_end = open_paren + close_paren + 1;
    let signature = normalize_signature(&line[..signature_end]);

    // Check if declaration or definition
    let after_signature = &line[signature_end..];
    if find(after_signature, b";").is_some() {
        Some(FunctionParseResult::Declaration(signature))
    } else if find(after_signature, b"{").is_some() || find(line, b"{").is_some() {
        Some(FunctionParseResult::DefinitionStart(signature))
    } else {
        // Could be multi-line (attributes, const, etc.)
        Some(FunctionParseResult::Incomplete(line.to_vec()))
    }
}

/// Extract function signature from accumulated bytes
fn extract_function_signature(data: &[u8]) -> Option<Cow<'_, [u8]>> {
    let open_paren = find(data, b"(")?;
    let close_paren = open_paren + find(&data[open_paren..], b")")?;
    Some(normalize_signature(&data[..close_paren + 1]))
}

/// Parse `#include` directive into the path and whether it is a system header
fn parse_include(line: &[u8]) -> Option<(&[u8], bool)> {
    let trimmed = line.strip_prefix(b"#include")?.trim_ascii_start();

    if let Some(system) = trimmed.strip_prefix(b"<") {
        let end = find(system, b">")?;
        Some((&system[..end], true))
    } else if let Some(local) = trimmed.strip_prefix(b"\"") {
        let end = find(local, b"\"")?;
        Some((&local[..end], false))
    } else {
        None
    }
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
fn ends_with_identifier(data: &[u8], identifier: &[u8]) -> bool {
    let trimmed = data.trim_ascii();
    let Some(before_suffix) = trimmed.len().checked_sub(identifier.len()) else {
        return false;
    };
    if &trimmed[before_suffix..] != identifier {
        return false;
    }

    // Must be word boundary before
    match before_suffix.checked_sub(1) {
        Some(index) => !trimmed[index].is_ascii_alphanumeric() && trimmed[index] != b'_',
        None => true,
    }
}

/// Extract last identifier from data
fn extract_last_identifier(data: &[u8]) -> Option<&[u8]> {
    let trimmed = data.trim_ascii();
    let width = trimmed
        .iter()
        .rev()
        .take_while(|&&byte| byte.is_ascii_alphanumeric() || byte == b'_')
        .count();
    (width > 0).then(|| &trimmed[trimmed.len() - width..])
}

/// Normalize a function signature: collapse whitespace runs to single spaces,
/// via StringZilla's SIMD whitespace splitter, into a single output buffer.
fn normalize_signature(data: &[u8]) -> Cow<'_, [u8]> {
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

/// Trim leading spaces and tabs, at most `max_indent` of them — Markdown treats a
/// fourth leading space as indented code rather than as indentation.
fn trim_start(data: &[u8], max_indent: usize) -> &[u8] {
    let indent = data
        .iter()
        .take_while(|&&byte| byte == b' ' || byte == b'\t')
        .count()
        .min(max_indent);
    &data[indent..]
}

/// Remove a heading's optional closing run of `#` characters and the space before it
fn trim_trailing_hashes(data: &[u8]) -> &[u8] {
    let text = data.trim_ascii_end();
    let hashes = text.iter().rev().take_while(|&&byte| byte == b'#').count();
    text[..text.len() - hashes].trim_ascii_end()
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
    element: &OutlineElement<'_>,
    parent_line: Option<usize>,
) -> io::Result<()> {
    out.write_all(br#"{"type":"element","data":{"kind":""#)?;
    out.write_all(element_kind_name(&element.kind).as_bytes())?;
    out.write_all(br#"","name":"#)?;
    json_text_field_to(out, &element.name)?;

    match &element.kind {
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
        element.line_number, element.line_count, element.byte_offset, element.byte_length
    )?;
    match parent_line {
        Some(line) => write!(out, r#","parent_line":{}"#, line)?,
        None => out.write_all(br#","parent_line":null"#)?,
    }
    out.write_all(b"}}\n")?;

    for child in &element.children {
        write_element_json(out, child, Some(element.line_number))?;
    }
    Ok(())
}

fn write_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    verbosity: Verbosity,
    file_type: FileType,
) -> io::Result<()> {
    match file_type {
        FileType::Markdown => write_markdown_element(out, element, verbosity),
        FileType::CSource | FileType::CHeader => write_c_element(out, element, verbosity),
    }
}

fn write_markdown_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    verbosity: Verbosity,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&element.name);
    match &element.kind {
        ElementKind::Heading { level } => {
            // Headings are levels 1..=6 — slice a static run, no allocation.
            let prefix = &"######"[..(*level as usize).min(6)];
            match verbosity {
                Verbosity::Names => writeln!(out, "{} {}", prefix, name)?,
                Verbosity::LineNumbers => writeln!(
                    out,
                    "{} {:40} [L{}, @{}]",
                    prefix, name, element.line_number, element.byte_offset
                )?,
                Verbosity::Detailed => {
                    let end_line = element.line_number + element.line_count - 1;
                    if element.line_count > 1 {
                        writeln!(
                            out,
                            "{} {:40} [L{}-{}, @{}, {}B]",
                            prefix,
                            name,
                            element.line_number,
                            end_line,
                            element.byte_offset,
                            element.byte_length
                        )?;
                    } else {
                        writeln!(
                            out,
                            "{} {:40} [L{}, @{}, {}B]",
                            prefix,
                            name,
                            element.line_number,
                            element.byte_offset,
                            element.byte_length
                        )?;
                    }
                    for child in &element.children {
                        write_child_block(out, child)?;
                    }
                }
            }
        }
        ElementKind::CodeBlock { language } => {
            if verbosity >= Verbosity::Detailed {
                let info = language
                    .as_ref()
                    .map(|bytes| String::from_utf8_lossy(bytes));
                let label = info.as_deref().unwrap_or("code");
                writeln!(
                    out,
                    "  - {} [L{}-{}, {}B]",
                    label,
                    element.line_number,
                    element.line_number + element.line_count - 1,
                    element.byte_length
                )?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_child_block(out: &mut dyn Write, element: &OutlineElement<'_>) -> io::Result<()> {
    // Constant labels borrow; only code and image build a small owned label for the {:36} pad.
    let label: Cow<str> = match &element.kind {
        ElementKind::CodeBlock { language } => match language {
            Some(bytes) => Cow::Owned(format!("code ({})", String::from_utf8_lossy(bytes))),
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

    if element.line_count > 1 {
        writeln!(
            out,
            "  - {:36} [L{}-{}, {}B]",
            label,
            element.line_number,
            element.line_number + element.line_count - 1,
            element.byte_length
        )
    } else {
        writeln!(
            out,
            "  - {:36} [L{}, {}B]",
            label, element.line_number, element.byte_length
        )
    }
}

fn write_c_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    verbosity: Verbosity,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&element.name);
    match &element.kind {
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
                        path, element.line_number, element.byte_offset
                    )?;
                }
            }
        }
        ElementKind::FunctionDeclaration => match verbosity {
            Verbosity::Names => writeln!(out, "{:44} [declaration]", name)?,
            Verbosity::LineNumbers => writeln!(
                out,
                "{:44} [L{}, @{}, declaration]",
                name, element.line_number, element.byte_offset
            )?,
            Verbosity::Detailed => writeln!(
                out,
                "{:44} [L{}, @{}, {}B, declaration]",
                name, element.line_number, element.byte_offset, element.byte_length
            )?,
        },
        ElementKind::FunctionDefinition => match verbosity {
            Verbosity::Names => writeln!(out, "{:44} [definition]", name)?,
            Verbosity::LineNumbers => {
                let end_line = element.line_number + element.line_count - 1;
                writeln!(
                    out,
                    "{:44} [L{}-{}, @{}, definition]",
                    name, element.line_number, end_line, element.byte_offset
                )?;
            }
            Verbosity::Detailed => {
                let end_line = element.line_number + element.line_count - 1;
                writeln!(
                    out,
                    "{:44} [L{}-{}, @{}, {}B, {} lines, definition]",
                    name,
                    element.line_number,
                    end_line,
                    element.byte_offset,
                    element.byte_length,
                    element.line_count
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
    let mut handle = stdout_writer();

    let verbosity = match args.verbose {
        0 => Verbosity::Names,
        1 => Verbosity::LineNumbers,
        _ => Verbosity::Detailed,
    };

    // Determine file type
    let detected = match args.file_type {
        Some(ref name) => parse_file_type(name),
        None if args.input != "-" => detect_file_type(&args.input),
        None => {
            eprintln!("Error: cannot detect file type from stdin, use --type");
            ExitCode::Error.exit(&mut handle);
        }
    };
    let Some(file_type) = detected else {
        eprintln!("Error: unknown file type, use --type to specify (md, c, h)");
        ExitCode::Error.exit(&mut handle);
    };

    // Get input — mmap is borrowed, not copied.
    let input = get_input(Some(&args.input))
        .unwrap_or_else(|error| exit_with_error(&mut handle, &error, "Error reading input"));
    let data = input.as_bytes();

    // Parse based on file type
    let elements = match file_type {
        FileType::Markdown => parse_markdown(data, verbosity),
        FileType::CSource | FileType::CHeader => parse_c(data, verbosity),
    };

    for element in &elements {
        let written = if args.json {
            write_element_json(&mut handle, element, None)
        } else {
            write_element(&mut handle, element, verbosity, file_type)
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
        let fence = is_code_fence(b"```rust").unwrap();
        assert_eq!(fence.marker, b'`');
        assert_eq!(fence.language, Some(Cow::Borrowed(b"rust".as_slice())));
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

        // Lines shorter than the directive, and other directives, are rejected
        assert_eq!(parse_include(b"#inc"), None);
        assert_eq!(parse_include(b"#include"), None);
        assert_eq!(parse_include(b"#define STDIO 1"), None);
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
    fn markdown_block_spans_never_overlap() {
        // A quote line ends the table above it, so the two spans stay disjoint
        // instead of the quote nesting inside a still-open table.
        let markdown = b"# Head\n\n| a | b |\n| - | - |\n> quoted\n\ntail\n";
        let elements = parse_markdown(markdown, Verbosity::Detailed);
        let blocks = &elements[0].children;

        for pair in blocks.windows(2) {
            let (earlier, later) = (&pair[0], &pair[1]);
            assert!(
                earlier.byte_offset + earlier.byte_length <= later.byte_offset,
                "{:?} at {}..{} overlaps {:?} at {}",
                earlier.kind,
                earlier.byte_offset,
                earlier.byte_offset + earlier.byte_length,
                later.kind,
                later.byte_offset
            );
            assert!(
                earlier.line_number + earlier.line_count <= later.line_number,
                "{:?} and {:?} share a line",
                earlier.kind,
                later.kind
            );
        }

        let kinds: Vec<&ElementKind<'_>> = blocks.iter().map(|block| &block.kind).collect();
        assert_eq!(
            kinds,
            vec![
                &ElementKind::Table,
                &ElementKind::Blockquote,
                &ElementKind::Paragraph
            ]
        );
    }

    #[test]
    fn markdown_block_at_end_of_input_stops_at_the_last_byte() {
        // Without a trailing newline the final block must not claim a byte past the end.
        let markdown = b"# Head\n\nparagraph";
        let elements = parse_markdown(markdown, Verbosity::Detailed);
        let paragraph = &elements[0].children[0];

        assert!(matches!(paragraph.kind, ElementKind::Paragraph));
        assert_eq!(
            paragraph.byte_offset + paragraph.byte_length,
            markdown.len()
        );

        // With the trailing newline the block owns it, and still ends at the last byte.
        let terminated = b"# Head\n\nparagraph\n";
        let elements = parse_markdown(terminated, Verbosity::Detailed);
        let paragraph = &elements[0].children[0];
        assert_eq!(
            paragraph.byte_offset + paragraph.byte_length,
            terminated.len()
        );
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
