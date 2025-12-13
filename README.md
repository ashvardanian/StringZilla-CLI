# StringZilla 🦖 Command-Line Interface

![StringZilla CLI banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-CLI.png?raw=true)

Most text processing command-line utilities have obscure syntax, limited portability across operating systems, can't deal with larger-than-memory datasets, and are not actively leveraging modern SIMD capabilities, such as AVX-512 on x86 and SVE on ARM.
This utility is written in Rust, leveraging StringZilla for both pipe and file-based text processing across Linux, macOS, and Windows.
Just one command to install it from Crates.io:

```bash
cargo install stringzilla-cli
```

It provides the following subcommands:

- `sz-find`: find all inclusions of a substring in a file similar to `grep`, but with a saner syntax
- `sz-outline`: provide LLM with an outline of a file for Markdown, HTML, C, C++, and Python source files
- `sz-replace`: replace all inclusions of a substring in a file... with a saner syntax than `sed` and `awk`; safe for larger-than-memory files
- `sz-count`: 3x faster `wc` word count, that can actually handle UTF-8 properly
- `sz-split`: 4x faster `split` file splitting, that won't break UTF-8 characters or lines
- `sz-dedup`: deduplicate lines; safe for larger-than-memory files
- `sz-cols`: extract columns from delimited text; replaces `cut -f` and `awk '{print $N}'` with simpler syntax
- `sz-rows`: extract rows by index or range; replaces `sed -n`, `head`, `tail`, and `awk 'NR==N'`
- :soon: `sz-sort`: sort lines
- :soon: `sz-context`: provide LLM context for a given token substring match in a file with minimal token waste
- :soon: `sz-fuzzy-find`: combination of exact and Levenshtein-bounded substring search

## Installation

```bash
cargo install --git https://github.com/ashvardanian/StringZillaCLI  # install from GitHub
cargo install --path . --force                                      # or install from local clone
```

## `sz-find`: Unicode Aware Substring Search

A `grep`-like tool using literal substring matching (not regex) for maximum speed.
Unlike `grep` and `ripgrep`, `sz-find` performs __full Unicode-compliant case folding__ for case-insensitive search, correctly handling all 1M+ defined characters.

```bash
# Basic search
sz-find "error" log.txt

# Case-insensitive search (with full Unicode case folding)
sz-find -i "error" log.txt

# Show line numbers
sz-find -n "pattern" file.txt

# Count matches only
sz-find -c "pattern" file.txt

# Context lines (like grep -B/-A/-C)
sz-find -B 2 -A 2 "error" log.txt

# Multi-line patterns (pattern can span lines)
sz-find -m "hello\nworld" file.txt

# UTF-8 mode (handles Unicode newlines: NEL, LINE SEPARATOR, etc.)
sz-find --utf8 "pattern" file.txt
```

There is partial support for Unicode case folding in `ripgrep` (`rg -i`).
The only tool seemingly implementing full folding is `pcre2`, designed for RegEx, rather than substring search.
The first is comparably fast, the second one is __orders of magnitude slower__.

Here's what Unicode compliance on a mixed dataset means for German queries, where the Eszett (ß) character is commonly used, and folds to "ss".

```bash
$ rg -c -i "strasse" xlsum.csv # 183 results
$ sz-find -c -i "strasse" xlsum.csv # 205 results, 22 more

$ rg -c -i "gross" xlsum.csv # 5412 results
$ sz-find -c -i "gross" xlsum.csv # 5418 results, +6 more

$ rg -c -i "weiss" xlsum.csv # 350 results
$ sz-find -c -i "weiss" xlsum.csv # 352 results, +2 more
```

Ligatures from PDFs and word processors (ﬁ, ﬂ, ﬀ, ﬃ, ﬄ) are also folded correctly:

```bash
$ rg -c -i "fi" xlsum.csv # 464678 results
$ sz-find -c -i "fi" xlsum.csv # 464699 results, +21 more

$ rg -c -i "ffi" xlsum.csv # 162155 results
$ sz-find -c -i "ffi" xlsum.csv # 162157 results, +2 more
```

Turkish dotted/dotless I (İ/I/i/ı) folding is also a common pitfall:

```bash
$ rg -c -i "işi" xlsum.csv # 23957 results
$ sz-find -c -i "işi" xlsum.csv # 25065 results, +1108 more
```

The difference becomes significant when searching legal documents, German/Swiss news, Turkish text, PDF-extracted content, or any content with typographic ligatures.

## `sz-outline`: File Outliner for LLMs

Extract structural outlines from source files for LLM context windows.
When feeding large files to language models, you often need a high-level overview without the full content.
`sz-outline` extracts headings, function signatures, includes, and other structural elements.

```bash
# Outline a Markdown file (headings only)
sz-outline README.md

# With line numbers and byte offsets (-v)
sz-outline -v README.md

# Detailed mode with child blocks (-vv)
sz-outline -vv README.md

# Outline C/C++ source (includes + function signatures)
sz-outline src/main.c

# Force file type detection
sz-outline -t md document.txt

# Read from stdin
cat file.md | sz-outline -t md -
```

There are several verbosity levels supported:

__Default__ - Names only:

```
$ sz-outline README.md
# StringZilla CLI
## Installation
## sz-find: Unicode Aware Substring Search
### Performance vs ripgrep
## sz-outline: File Outliner for LLMs
```

__`-v`__ - Add line numbers and byte offsets:

```
$ sz-outline -v README.md
# StringZilla CLI                         [L1, @0]
## Installation                           [L27, @892]
## sz-find: Unicode Aware Substring Search [L34, @1045]
### Performance vs ripgrep                [L127, @4521]
```

__`-vv`__ - Detailed with child blocks (code blocks, tables, images):

```
$ sz-outline -vv README.md
# StringZilla CLI                         [L1, @0, 45B]
  - paragraph                             [L3-5, 312B]
  - code (bash)                           [L9-11, 89B]
## Installation                           [L27, @892, 156B]
  - code (bash)                           [L29-32, 245B]
```

For C source files, `sz-outline` extracts includes and function signatures:

```
$ sz-outline -v src/parser.c
#include <stdio.h>                        [L1, @0]
#include <stdlib.h>                       [L2, @19]
#include "parser.h"                       [L3, @39]
static int parse_token(const char *input) [L12-45, @156, definition]
int parse_file(FILE *fp)                  [L47-123, @892, definition]
void cleanup(void)                        [L125-130, @2341, definition]
```

Function signatures are normalized (whitespace collapsed) and categorized as declarations (`;`) or definitions (`{}`).

## `sz-count`: Word Count

The `wc` utility on Linux can be used to count the number of lines, words, and bytes in a file.
Using SIMD-accelerated character and character-set search, StringZilla can be noticeably faster, even with slow SSDs.

```bash
$ time wc enwik9.txt
  13147025 129348346 1000000000 enwik9.txt

real    0m3.562s
user    0m3.470s
sys     0m0.092s

$ time sz-count --wc enwik9.txt
  13147025 139132610 1000000000 enwik9.txt # Note: word count differs due to stricter ASCII whitespace handling

real    0m1.165s
user    0m1.121s
sys     0m0.044s
```

## `sz-split`: Split File into Smaller Ones

The `split` utility on Linux can be used to split a file into smaller ones.
The current prototype only splits by line counts.

```bash
$ time split -l 100000 enwik9.txt ...

real    0m6.424s
user    0m0.179s
sys     0m0.663s

$ time sz_split -l 100000 enwik9.txt ...

real    0m1.482s
user    0m1.020s
sys     0m0.460s
```

## `sz-cols`: Extract Columns

The `cut` utility and `awk '{print $N}'` are commonly used to extract columns from delimited text.
`sz-cols` provides a simpler, more intuitive syntax with SIMD-accelerated delimiter scanning.

```bash
# Replaces: cut -f2 data.tsv
# Replaces: awk -F'\t' '{print $2}' data.tsv
$ sz-cols -f 2 data.tsv

# Extract multiple columns with custom output delimiter
# Replaces: cut -f1,3 -d',' --output-delimiter=';' data.csv
$ sz-cols -f 1,3 -d ',' -D ';' data.csv

# Extract a range of columns
# Replaces: cut -f2-5 data.tsv
$ sz-cols -f 2-5 data.tsv

# Mixed selection: specific columns and ranges
$ sz-cols -f 1,3-5,8 data.tsv
```

## `sz-rows`: Extract Rows

The `sed -n 'Np'`, `head -n N`, `tail -n N`, and `awk 'NR==N'` commands are commonly used to extract specific lines.
`sz-rows` unifies all these use cases with a single, intuitive interface.

```bash
# Extract line 5
# Replaces: sed -n '5p' file.txt
# Replaces: awk 'NR==5' file.txt
$ sz-rows -r 5 file.txt

# Extract lines 10-20
# Replaces: sed -n '10,20p' file.txt
$ sz-rows -r 10-20 file.txt

# Extract first 10 lines
# Replaces: head -n 10 file.txt
$ sz-rows -r 1-10 file.txt

# Extract last 10 lines
# Replaces: tail -n 10 file.txt
$ sz-rows --tail 10 file.txt

# Extract specific lines
# Replaces: sed -n '1p;5p;10p' file.txt
$ sz-rows -r 1,5,10 file.txt

# Extract every 5th line
# Replaces: awk 'NR % 5 == 0' file.txt
$ sz-rows --every 5 file.txt

# Show line numbers in output
$ sz-rows -n -r 5-10 file.txt
```
