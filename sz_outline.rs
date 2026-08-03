//! SIMD-accelerated file outlining utility
//!
//! Extract structural outlines from source files including:
//! - Markdown (`.md`, `.markdown`): headings, code blocks, tables, blockquotes, images
//! - C (`.c`, `.h`): includes, function declarations, function definitions
//!
//! The parser is taken from the extension or forced with `--language {md,c,h}`.
//!
//! # Examples
//!
//! ```bash
//! # Outline a Markdown file
//! sz-outline README.md
//!
//! # Add line numbers and byte offsets
//! sz-outline --detail positions README.md
//!
//! # Add block sizes and nested blocks
//! sz-outline --detail blocks src/main.c
//!
//! # Force the parser
//! sz-outline --language md document.txt
//! ```

use std::borrow::Cow;
use std::io::{self, Write};
use std::path::Path;

use clap::{CommandFactory, Parser, ValueEnum};
use stringzilla::sz::{find, StringZillableUnary};

use shared::*;

// region: Data Structures

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

/// How much of each element the human renderer prints
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, ValueEnum)]
enum Detail {
    Headings,
    Positions,
    Blocks,
}

/// Which parser reads the input
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Language {
    Md,
    C,
    H,
}

/// How records are rendered
#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum Format {
    Text,
    Json,
}

/// Extract structural outline from source files
#[derive(Parser)]
#[command(name = "sz-outline")]
#[command(version, about = "SIMD-accelerated file outlining", long_about = None)]
struct Args {
    /// Input file (use '-' or omit for stdin)
    input: Option<String>,

    /// Force the parser instead of detecting it from the extension
    #[arg(long, value_enum, required_unless_present = "input")]
    language: Option<Language>,

    /// How much of each element to print
    #[arg(long, value_enum, default_value = "headings")]
    detail: Detail,

    /// Treat the input as UTF-8 text
    #[arg(long)]
    utf8: bool,

    /// How records are rendered
    #[arg(
        long,
        value_enum,
        default_value = "text",
        help_heading = "Output Formats"
    )]
    format: Format,

    /// Suppress all output; exit 0 if any element was found, 1 otherwise
    #[arg(long, conflicts_with_all = ["detail", "format"], help_heading = "Output Formats")]
    quiet: bool,
}

// endregion: CLI Interface

// region: Language Detection

/// Language from the path's extension, or `None` when the extension is not outlined.
fn detect_language(path: &str) -> Option<Language> {
    match Path::new(path).extension()?.to_str()? {
        "md" | "markdown" => Some(Language::Md),
        "c" => Some(Language::C),
        "h" => Some(Language::H),
        _ => None,
    }
}

// endregion: Language Detection

// region: Markdown Parser

/// Parse Markdown file and extract outline elements
fn parse_markdown<'a>(data: &'a [u8], newlines: Newlines) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;
    let mut current_section: Option<usize> = None;
    let mut open_code_block: Option<OpenCodeBlock<'a>> = None;
    let mut open_block: Option<OpenBlock> = None;

    for line in LineIter::new(data, newlines) {
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
                push_element(&mut elements, current_section, element);
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
        // Image: `![alt](url)`
        else if let Some(alt) = parse_image(line) {
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
fn parse_c<'a>(data: &'a [u8], newlines: Newlines) -> Vec<OutlineElement<'a>> {
    let mut elements: Vec<OutlineElement<'a>> = Vec::new();
    let mut line_number = 0usize;

    // State for function body tracking
    let mut brace_depth = 0i32;
    let mut in_function_body = false;
    let mut function_start: Option<(usize, usize, Cow<'a, [u8]>)> = None;

    // State for multi-line constructs
    let mut in_multiline_comment = false;
    let mut pending_signature: Option<(usize, usize, Vec<u8>)> = None;

    for line in LineIter::new(data, newlines) {
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

/// Whether `detail` prints this element. Below `blocks` only the structural elements
/// print — headings, includes and functions — so both renderers drop the same records.
fn selects(kind: &ElementKind<'_>, detail: Detail) -> bool {
    detail >= Detail::Blocks
        || matches!(
            kind,
            ElementKind::Heading { .. }
                | ElementKind::Include { .. }
                | ElementKind::FunctionDeclaration
                | ElementKind::FunctionDefinition
        )
}

/// Write one element as a flat JSON Lines record. Children are emitted as their own
/// records carrying `parent_line`, rather than nested, which keeps `jq` filters simple.
fn write_element_json(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    parent_line: Option<usize>,
    detail: Detail,
) -> io::Result<()> {
    out.write_all(br#"{"type":"element","data":{"kind":""#)?;
    out.write_all(element_kind_name(&element.kind).as_bytes())?;
    out.write_all(br#"","text":"#)?;
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
        if selects(&child.kind, detail) {
            write_element_json(out, child, Some(element.line_number), detail)?;
        }
    }
    Ok(())
}

fn write_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    detail: Detail,
    language: Language,
    column: usize,
) -> io::Result<()> {
    match language {
        Language::Md => write_markdown_element(out, element, detail, column),
        Language::C | Language::H => write_c_element(out, element, detail, column),
    }
}

fn write_markdown_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    detail: Detail,
    column: usize,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&element.name);
    match &element.kind {
        ElementKind::Heading { level } => {
            // Headings are levels 1..=6 — slice a static run, no allocation.
            let prefix = &"######"[..(*level as usize).min(6)];
            match detail {
                Detail::Headings => writeln!(out, "{} {}", prefix, name)?,
                Detail::Positions => writeln!(
                    out,
                    "{} {:width$} [L{}, @{}]",
                    prefix,
                    name,
                    element.line_number,
                    element.byte_offset,
                    width = column.saturating_sub(prefix.len() + 1)
                )?,
                Detail::Blocks => {
                    let end_line = element.line_number + element.line_count - 1;
                    if element.line_count > 1 {
                        writeln!(
                            out,
                            "{} {:width$} [L{}-{}, @{}, {}B]",
                            prefix,
                            name,
                            element.line_number,
                            end_line,
                            element.byte_offset,
                            element.byte_length,
                            width = column.saturating_sub(prefix.len() + 1)
                        )?;
                    } else {
                        writeln!(
                            out,
                            "{} {:width$} [L{}, @{}, {}B]",
                            prefix,
                            name,
                            element.line_number,
                            element.byte_offset,
                            element.byte_length,
                            width = column.saturating_sub(prefix.len() + 1)
                        )?;
                    }
                    for child in &element.children {
                        write_child_block(out, child, column)?;
                    }
                }
            }
        }
        // A block before the first heading has no parent, and prints in the same
        // branch column as one that does.
        _ => write_child_block(out, element, column)?,
    }
    Ok(())
}

/// The branch a child block hangs off, written once so the column it occupies is the
/// string's own length rather than a number that can drift from it.
const CHILD_BRANCH: &str = "  - ";

/// The keyword an include prints behind, on the same terms.
const INCLUDE_KEYWORD: &str = "#include ";

/// Where the bracketed detail column starts: one past the longest name the run will print,
/// its prefix included. A fixed column instead leaves every short name stranded from its
/// own detail, and shoves the details of a long one out of the column the rest share.
fn detail_column(elements: &[OutlineElement<'_>], detail: Detail) -> usize {
    let mut column = 0;
    for element in elements {
        let name = String::from_utf8_lossy(&element.name).chars().count();
        column = column.max(match &element.kind {
            ElementKind::Heading { level } => (*level as usize).min(6) + 1 + name,
            // The path prints inside its `<>` or `""` delimiters.
            ElementKind::Include { .. } => INCLUDE_KEYWORD.len() + name + 2,
            ElementKind::FunctionDeclaration | ElementKind::FunctionDefinition => name,
            _ if detail >= Detail::Blocks => {
                CHILD_BRANCH.len() + child_label(element).chars().count()
            }
            _ => 0,
        });
        if detail >= Detail::Blocks {
            for child in &element.children {
                column = column.max(CHILD_BRANCH.len() + child_label(child).chars().count());
            }
        }
    }
    column
}

/// The label a child block prints under its heading, owned only for the two kinds that
/// carry a name of their own.
fn child_label<'a>(element: &'a OutlineElement<'_>) -> Cow<'a, str> {
    match &element.kind {
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
    }
}

fn write_child_block(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    column: usize,
) -> io::Result<()> {
    let label = child_label(element);

    if element.line_count > 1 {
        writeln!(
            out,
            "{}{:width$} [L{}-{}, {}B]",
            CHILD_BRANCH,
            label,
            element.line_number,
            element.line_number + element.line_count - 1,
            element.byte_length,
            width = column.saturating_sub(CHILD_BRANCH.len())
        )
    } else {
        writeln!(
            out,
            "{}{:width$} [L{}, {}B]",
            CHILD_BRANCH,
            label,
            element.line_number,
            element.byte_length,
            width = column.saturating_sub(CHILD_BRANCH.len())
        )
    }
}

fn write_c_element(
    out: &mut dyn Write,
    element: &OutlineElement<'_>,
    detail: Detail,
    column: usize,
) -> io::Result<()> {
    // `Cow<[u8]>` is not `Display`; `from_utf8_lossy` borrows for valid UTF-8.
    let name = String::from_utf8_lossy(&element.name);
    match &element.kind {
        ElementKind::Include { is_system } => {
            let (open, close) = if *is_system { ("<", ">") } else { ("\"", "\"") };
            match detail {
                Detail::Headings => {
                    writeln!(out, "{}{}{}{}", INCLUDE_KEYWORD, open, name, close)?;
                }
                Detail::Positions | Detail::Blocks => {
                    // Build the delimited path, so the pad measures the field the reader sees.
                    let path = format!("{}{}{}", open, name, close);
                    writeln!(
                        out,
                        "{}{:width$} [L{}, @{}]",
                        INCLUDE_KEYWORD,
                        path,
                        element.line_number,
                        element.byte_offset,
                        width = column.saturating_sub(INCLUDE_KEYWORD.len())
                    )?;
                }
            }
        }
        ElementKind::FunctionDeclaration => match detail {
            Detail::Headings => writeln!(out, "{:width$} [declaration]", name, width = column)?,
            Detail::Positions => writeln!(
                out,
                "{:width$} [L{}, @{}, declaration]",
                name,
                element.line_number,
                element.byte_offset,
                width = column
            )?,
            Detail::Blocks => writeln!(
                out,
                "{:width$} [L{}, @{}, {}B, declaration]",
                name,
                element.line_number,
                element.byte_offset,
                element.byte_length,
                width = column
            )?,
        },
        ElementKind::FunctionDefinition => match detail {
            Detail::Headings => writeln!(out, "{:width$} [definition]", name, width = column)?,
            Detail::Positions => {
                let end_line = element.line_number + element.line_count - 1;
                writeln!(
                    out,
                    "{:width$} [L{}-{}, @{}, definition]",
                    name,
                    element.line_number,
                    end_line,
                    element.byte_offset,
                    width = column
                )?;
            }
            Detail::Blocks => {
                let end_line = element.line_number + element.line_count - 1;
                writeln!(
                    out,
                    "{:width$} [L{}-{}, @{}, {}B, {} lines, definition]",
                    name,
                    element.line_number,
                    end_line,
                    element.byte_offset,
                    element.byte_length,
                    element.line_count,
                    width = column
                )?;
            }
        },
        _ => {}
    }
    Ok(())
}

// endregion: Output Formatting

// region: Main

/// Render a validation failure the way clap renders a parse failure: same `error:` prefix,
/// same usage block, same exit code. The kind is never displayed, so one kind serves all.
fn reject(message: impl std::fmt::Display) -> clap::Error {
    Args::command().error(clap::error::ErrorKind::ArgumentConflict, message)
}

/// Every constraint that depends on an argument's *value*, which clap cannot declare.
fn validate(args: &Args) -> Result<(), clap::Error> {
    let path = args.input.as_deref().unwrap_or("-");
    if args.language.or_else(|| detect_language(path)).is_none() {
        return Err(reject(format!(
            "cannot outline '{}', use --language (md, c, h)",
            path
        )));
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = Args::parse();
    let mut output = stdout_writer();
    report("sz-outline", run(&args, &mut output))
}

fn run(args: &Args, output: &mut dyn Write) -> Result<Status, Failure> {
    validate(args)?;

    let path = args.input.as_deref().unwrap_or("-");
    let language = args
        .language
        .or_else(|| detect_language(path))
        .expect("validated");

    // The mmap is borrowed, not copied.
    let input = get_input(Some(path)).at(path)?;
    let newlines = Newlines::from_utf8(args.utf8);
    let elements = match language {
        Language::Md => parse_markdown(input.as_bytes(), newlines),
        Language::C | Language::H => parse_c(input.as_bytes(), newlines),
    };

    // The exit code answers what was printed, not what was parsed, so a document
    // whose every element `--detail` drops exits 1 rather than 0.
    let outlined = elements
        .iter()
        .filter(|element| selects(&element.kind, args.detail));
    if args.quiet {
        return Ok(Status::from_found(outlined.count() > 0));
    }

    let column = detail_column(&elements, args.detail);
    let mut emitted = false;
    for element in outlined {
        emitted = true;
        match args.format {
            Format::Json => write_element_json(output, element, None, args.detail),
            Format::Text => write_element(output, element, args.detail, language, column),
        }
        .at("-")?;
    }
    output.flush().at("-")?;
    Ok(Status::from_found(emitted))
}

// endregion: Main

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn declares_no_short_flags() {
        assert!(Args::command()
            .get_arguments()
            .all(|a| a.get_short().is_none() || matches!(a.get_short(), Some('h') | Some('V'))));
    }

    #[test]
    fn declares_the_expected_flags() {
        let mut command = Args::command();
        command.build();
        let longs: Vec<_> = command
            .get_arguments()
            .filter_map(|a| a.get_long())
            .collect();
        assert_eq!(
            longs,
            ["language", "detail", "utf8", "format", "quiet", "help", "version"]
        );
    }

    #[test]
    fn requires_a_language_only_for_stdin() {
        assert!(Args::try_parse_from(["sz-outline", "README.md"]).is_ok());
        assert!(Args::try_parse_from(["sz-outline", "--language", "md"]).is_ok());
        let Err(error) = Args::try_parse_from(["sz-outline"]) else {
            panic!("stdin needs --language");
        };
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
        // A typo is a usage error rather than a runtime message naming the flag you used.
        let Err(error) = Args::try_parse_from(["sz-outline", "--language", "rust", "a.txt"]) else {
            panic!("unknown language must be rejected");
        };
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn breaks_lines_on_unicode_newlines_only_under_utf8() {
        let markdown = "# One\u{2028}# Two\n".as_bytes();
        assert_eq!(parse_markdown(markdown, Newlines::Unicode).len(), 2);
        // LF-only splitting swallows the separator into the first heading's text.
        assert_eq!(parse_markdown(markdown, Newlines::Lf).len(), 1);
    }

    #[test]
    fn defaults_the_rendering_options_and_still_conflicts_with_quiet() {
        let args = Args::try_parse_from(["sz-outline", "README.md"]).unwrap();
        assert_eq!(args.detail, Detail::Headings);
        assert_eq!(args.format, Format::Text);
        // Defaulted values are not "present", so the conflicts survive the defaults.
        assert!(Args::try_parse_from(["sz-outline", "--quiet", "README.md"]).is_ok());
        for flag in [["--detail", "blocks"], ["--format", "json"]] {
            let argv = ["sz-outline", "--quiet", flag[0], flag[1], "README.md"];
            assert!(Args::try_parse_from(argv).is_err(), "{:?}", argv);
        }
    }

    #[test]
    fn rejects_an_extension_it_cannot_outline() {
        let args = Args::try_parse_from(["sz-outline", "notes.txt"]).unwrap();
        assert!(validate(&args).is_err());
        assert!(validate(&Args::try_parse_from(["sz-outline", "notes.md"]).unwrap()).is_ok());
    }

    #[test]
    fn detail_decides_what_either_renderer_emits() {
        // Both the exit code and `--format json` used to ignore `--detail` entirely.
        let markdown = b"just a paragraph\n";
        let elements = parse_markdown(markdown, Newlines::Lf);
        assert_eq!(elements.len(), 1);
        assert!(!selects(&elements[0].kind, Detail::Headings));
        assert!(selects(&elements[0].kind, Detail::Blocks));

        let records = |detail| {
            let markdown = b"# Head\n\n```rust\ncode\n```\n";
            let elements = parse_markdown(markdown, Newlines::Lf);
            let mut printed = Vec::new();
            for element in elements.iter().filter(|e| selects(&e.kind, detail)) {
                write_element_json(&mut printed, element, None, detail).unwrap();
            }
            String::from_utf8(printed).unwrap().lines().count()
        };
        assert_eq!(records(Detail::Headings), 1);
        assert_eq!(records(Detail::Blocks), 2);
    }

    #[test]
    fn records_every_block_regardless_of_detail() {
        // The parser is unconditional; `--detail` filters at the renderer instead.
        let markdown = b"# Head\n\n```rust\ncode\n```\n\n> quoted\n";
        let elements = parse_markdown(markdown, Newlines::Lf);
        let kinds: Vec<&ElementKind<'_>> = elements[0].children.iter().map(|c| &c.kind).collect();
        assert!(
            matches!(kinds[0], ElementKind::CodeBlock { .. }),
            "{:?}",
            kinds
        );
        assert!(matches!(kinds[1], ElementKind::Blockquote), "{:?}", kinds);

        // Yet the heading-only rendering still prints one line per heading.
        let mut printed = Vec::new();
        write_element(
            &mut printed,
            &elements[0],
            Detail::Headings,
            Language::Md,
            detail_column(&elements, Detail::Headings),
        )
        .unwrap();
        assert_eq!(printed, b"# Head\n");
    }

    #[test]
    fn aligns_details_past_the_longest_name() {
        // A heading wider than any fixed column, beside one far narrower, and a child
        // block whose branch glyphs count toward the same column.
        let data = b"# Short\n\nA paragraph.\n\n## A heading long enough to outgrow a fixed forty-column pad\n";
        let elements = parse_markdown(data, Newlines::Lf);
        let column = detail_column(&elements, Detail::Blocks);
        let mut printed = Vec::new();
        for element in &elements {
            write_element(&mut printed, element, Detail::Blocks, Language::Md, column).unwrap();
        }

        let text = String::from_utf8(printed).unwrap();
        let details: Vec<usize> = text
            .lines()
            .map(|line| line.find(" [").expect("every row carries its detail"))
            .collect();
        assert!(details.len() >= 3, "{}", text);
        assert!(
            details.windows(2).all(|pair| pair[0] == pair[1]),
            "details start in different columns:\n{}",
            text
        );
        // Sized to the longest name, not to a constant.
        assert_eq!(details[0], column, "{}", text);
    }

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
        let elements = parse_markdown(md, Newlines::Lf);

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
        let elements = parse_markdown(markdown, Newlines::Lf);
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
        let elements = parse_markdown(markdown, Newlines::Lf);
        let paragraph = &elements[0].children[0];

        assert!(matches!(paragraph.kind, ElementKind::Paragraph));
        assert_eq!(
            paragraph.byte_offset + paragraph.byte_length,
            markdown.len()
        );

        // With the trailing newline the block owns it, and still ends at the last byte.
        let terminated = b"# Head\n\nparagraph\n";
        let elements = parse_markdown(terminated, Newlines::Lf);
        let paragraph = &elements[0].children[0];
        assert_eq!(
            paragraph.byte_offset + paragraph.byte_length,
            terminated.len()
        );
    }

    #[test]
    fn parses_c_includes_and_functions() {
        let c = b"#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n";
        let elements = parse_c(c, Newlines::Lf);

        assert_eq!(elements.len(), 2);
        assert!(matches!(
            elements[0].kind,
            ElementKind::Include { is_system: true }
        ));
        assert!(matches!(elements[1].kind, ElementKind::FunctionDefinition));
    }
}

// endregion: Tests
