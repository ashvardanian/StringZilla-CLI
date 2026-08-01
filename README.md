# StringZilla 🦖 Command-Line Interface

![StringZilla CLI banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-CLI.png?raw=true)

Most text processing command-line utilities have obscure syntax, limited portability across operating systems, can't deal with larger-than-memory datasets, and are not actively leveraging modern SIMD capabilities, such as AVX-512 on x86 and SVE on ARM.
This utility is written in Rust, leveraging StringZilla for both pipe and file-based text processing across Linux, macOS, and Windows.
Install it straight from the GitHub repository:

```bash
cargo install --git https://github.com/ashvardanian/StringZilla-CLI
```

It provides the following subcommands:

- `sz-find`: find all inclusions of a substring in a file similar to `grep`, but with a saner syntax
- `sz-replace`: replace a substring in a file or stream; a literal-match alternative to `sed s///`
- `sz-outline`: provide an LLM with an outline of a Markdown or C/C++ source file
- `sz-count`: 3x faster `wc` word count, that can actually handle UTF-8 properly
- `sz-dedup`: deduplicate lines; safe for larger-than-memory files
- `sz-split`: 4x faster `split` file splitting, that won't break UTF-8 characters or lines
- `sz-cols`: extract columns from delimited text; replaces `cut -f` and `awk '{print $N}'` with simpler syntax
- `sz-rows`: extract rows by index or range; replaces `sed -n`, `head`, `tail`, and `awk 'NR==N'`
- `sz-sort`: sort lines; Unicode-correct and `sort -u`-style deduplication
- `sz-fuzzy-find`: exact + fuzzy substring search (Smith-Waterman / Levenshtein) with keyboard & phonetic scoring, on CPU or GPU

## Installation

```bash
cargo install --git https://github.com/ashvardanian/StringZilla-CLI # install from GitHub
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

# Match across line boundaries (pattern can span lines)
sz-find -M "hello\nworld" file.txt

# UTF-8 mode (handles Unicode newlines: NEL, LINE SEPARATOR, etc.)
sz-find --utf8 "pattern" file.txt
```

Beyond the flags above, `sz-find` also covers most of the `grep`/`ripgrep` surface: whole-word matching (`-w`), inverted matches (`-v`), only-matching output (`-o`), recursive directory walking with `.gitignore` awareness, type/glob filters (`-t`, `-g`), and `--json`/`--vimgrep` output.
Passing `-r/--replace` turns it into an in-place find-and-replace; for stream-oriented replacement use the dedicated `sz-replace` below.

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
File type is inferred from the extension — Markdown (`.md`, `.markdown`) and C (`.c`, `.h`) — or forced with `-t {md,c,h}` (required when reading from stdin).

```bash
# Outline a Markdown file (headings only)
sz-outline README.md

# With line numbers and byte offsets (-v)
sz-outline -v README.md

# Detailed mode with child blocks (-vv)
sz-outline -vv README.md

# Outline C source (includes + function signatures)
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

The `wc` utility on Linux counts lines, words, and bytes.
A word is a maximal run of non-whitespace, exactly as in `wc -w`; `sz-count` uses the same rule, so it is a drop-in replacement rather than a different measurement.
Output is a labelled, aligned table; pass `--utf8` to also count Unicode code points, or `-H` for human-readable suffixes.

The comparison worth making is against `wc` in a __UTF-8 locale__, because that is the only configuration computing the same thing.
GNU `wc` switches behavior on `LC_CTYPE`: in the C locale it walks bytes and gates word starts on `isprint`, so an entire run of non-ASCII text opens no word at all.

```bash
$ time LC_ALL=C.UTF-8 wc -lwmc xlsum.csv   # ✅ correct under a UTF-8 locale
    51627  26804246 167899744 256000000 xlsum.csv
real    0m2.037s

$ time sz-count --utf8 xlsum.csv           # ✅ same words and characters
                                          lines      words       bytes       chars
xlsum.csv                                51,629 26,804,246 256,000,000 167,899,744
real    0m0.337s

$ time LC_ALL=C wc -lwc xlsum.csv          # ❌ non-ASCII runs open no word at all
    51627  16806679 256000000 xlsum.csv
real    0m2.058s
```

Unlike `wc`, enabling Unicode costs nothing here — `--utf8` runs at the same speed as the default.
Line counts diverge too, because `wc` counts terminators where `sz-count` counts lines, and `--posix` restores `wc`'s reading exactly:

| Lines counted in                           | `wc` | `sz-count` | `--utf8` | `--posix` |
| ------------------------------------------ | ---: | ---------: | -------: | --------: |
| `a\nb`, no final newline                   |    1 |          2 |        2 |         1 |
| Text broken by CR, VT, FF, NEL, LS, and PS |    1 |          1 |        7 |         1 |

What `--posix` deliberately does not restore is the C-locale word gate above, since dropping non-ASCII words is a bug rather than a convention.
For scripting, `-l`, `-w`, `-c`, `-m`, and `-L` select individual fields with `wc`'s own meanings, and a single selector over a single input prints a bare integer:

```bash
$ lines=$(sz-count -l xlsum.csv)   # 51628 — no header, no commas, no padding
$ sz-count --json xlsum.csv        # JSON Lines, bare integers, untruncated paths
```

## `sz-split`: Split File into Smaller Ones

The `split` utility on Linux can be used to split a file into smaller ones.
The current prototype only splits by line counts.

```bash
$ time split -l 100000 enwik9.txt ...

real    0m6.424s
user    0m0.179s
sys     0m0.663s

$ time sz-split -l 100000 enwik9.txt ...

real    0m1.482s
user    0m1.020s
sys     0m0.460s
```

## `sz-replace`: Substring Replacement

A literal-match alternative to `sed 's/old/new/g'` with the same Unicode case-folding engine as `sz-find`.
Reads from a file or stdin and writes to stdout, a file, or back in-place.

```bash
# Replace on a stream (writes to stdout)
sz-replace "old" "new" file.txt

# Edit a file in-place
sz-replace -i "old" "new" file.txt

# Case-insensitive replace (full Unicode case folding)
sz-replace -I "straße" "strasse" file.txt

# Preview without writing, and report the number of replacements
sz-replace -n -c "old" "new" file.txt
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

## `sz-sort`: Sort Lines

A memory-lean, Unicode-correct `sort` built on StringZilla's `argsort`.
Comparison is unsigned byte-wise, which for valid UTF-8 is exactly Unicode code-point order, so `--utf8` only governs newline handling and output is byte-identical to `LC_ALL=C sort`.

```bash
# Sort to stdout (replaces: sort file.txt)
$ sz-sort file.txt

# Reverse / descending (replaces: sort -r)
$ sz-sort -r file.txt

# Sort and drop duplicates (replaces: sort -u)
$ sz-sort -u file.txt

# Case-insensitive sort with full Unicode case folding (replaces: sort -f)
$ sz-sort -i file.txt

# Write to a file (replaces: sort -o sorted.txt file.txt)
$ sz-sort file.txt -o sorted.txt

# Check whether the input is already sorted; exit 1 if not (replaces: sort -c)
$ sz-sort -c file.txt
```

Lines are held as packed offset and length pairs borrowing the input, about a third the footprint of a fat pointer each.

```bash
# Build a vocabulary from a multilingual news corpus: 26.8 M words, 256 MB
$ sz-segment --split-whitespaces xlsum.csv | sz-sort -u > /dev/null
```

| Operation                     | GNU `sort --parallel=1` |           `sz-sort` |
| ----------------------------- | ----------------------: | ------------------: |
| Sort                          |         9.45 s, 1509 MB | __5.44 s, 1248 MB__ |
| Case-insensitive, `-f` / `-i` |        13.38 s, 1509 MB | __8.95 s, 1248 MB__ |
| Deduplicating, `-u`           |         9.26 s, 1509 MB | __4.99 s, 1248 MB__ |

## `sz-dedup`: Deduplicate Lines

Drop repeated lines, keeping the first of each, __without sorting__.
`uniq` collapses only adjacent duplicates and so needs sorted input, and `sort -u` gets there by discarding the original order; the idiom that actually preserves order is `awk '!seen[$0]++'`.

```bash
# Unique lines to stdout, first occurrence kept, input order preserved
$ sz-dedup file.txt

# Rewrite the file in place instead
$ sz-dedup --in-place file.txt

# Case-insensitive, with full Unicode case folding
$ sz-dedup -i file.txt

# Report whether anything was dropped, through the exit code alone
$ sz-dedup -q file.txt
```

Lines are hashed with StringZilla's SIMD hash into an open-addressed table that holds one entry per __distinct__ line, so memory follows the number of unique lines rather than the length of the input.

```bash
$ sz-segment --split-whitespaces xlsum.csv | sz-dedup > /dev/null
```

| Tool                   |     Order |  Time and peak RSS |
| ---------------------- | --------: | -----------------: |
| `awk '!seen[$0]++'`    | preserved |    14.05 s, 742 MB |
| `sort -u --parallel=1` |      lost |    9.37 s, 1509 MB |
| `sz-dedup`             | preserved | __1.62 s, 351 MB__ |

`uniq` is absent from the table because on unsorted input it returns almost every line.

## `sz-fuzzy-find`: Fuzzy Substring Search

`sz-find` matches literally; `sz-fuzzy-find` adds typo tolerance, built on StringZilla's `szs` similarity kernels (CPU multicore by default, GPU optional).
By default a line matches when some substring of it aligns to the query within `-k` edits, found via Smith-Waterman local alignment.
The exact substring is checked first via StringZilla's `find`, so only lines without an exact hit reach the kernel.

```bash
# Find "color" allowing up to 1 edit — also matches "colour", "colur", "kolor"
$ sz-fuzzy-find -k 1 color file.txt

# Several queries at once (a line matches if ANY query matches)
$ sz-fuzzy-find -k 1 -e foo -e bar file.txt

# Word mode: match the needle against each token (Levenshtein), not the whole line
$ sz-fuzzy-find -w -k 1 colour file.txt

# Count matching lines only
$ sz-fuzzy-find -c -k 1 needle file.txt
```

### Scoring models (`--cost`)

Beyond uniform edit distance, scoring can reflect *how* characters get confused.
These route through Smith-Waterman with a `byte_to_class[256]` + `class_substitution_costs[32][32]` matrix, and use a normalized `--min-similarity` (0..1, where 1.0 is exact) instead of `-k`:

```bash
# Keyboard proximity: fat-finger typos (adjacent keys cost less) — "xolor" matches "color"
$ sz-fuzzy-find --cost keyboard --min-similarity 0.8 color file.txt

# Phonetic: sounds-alike (Editex-style articulatory groups) — "fonetik" matches "phonetic"
$ sz-fuzzy-find --cost phonetic --min-similarity 0.75 phonetic file.txt

# Custom 256→class map + 32×32 score matrix
$ sz-fuzzy-find --cost-matrix my_costs.txt --min-similarity 0.8 needle file.txt
```

The keyboard matrix uses staggered-QWERTY Euclidean key distance; the phonetic matrix is seeded by voiced/unvoiced cognates (`b/p`, `d/t`, …) and Editex letter groups. Smith-Waterman fuzzy matching folds ASCII case (the 32-class budget leaves no room to distinguish case per letter).

### Execution device (`--device`)

```bash
$ sz-fuzzy-find --device cpu --threads 8 -k 1 needle big.txt   # CPU, 8 threads
$ sz-fuzzy-find --device gpu -k 1 needle big.txt               # GPU (see build note)
```

CPU multicore is the default. The GPU path requires a CUDA build:

```bash
# On systems with gcc > 14 + CUDA 12.x, point nvcc at a supported host compiler:
$ CUDAHOSTCXX=g++-14 cargo install --git https://github.com/ashvardanian/StringZilla-CLI --features cuda
```

With `-k 0`, `sz-fuzzy-find` degrades to exact substring search, matching `sz-find`.
