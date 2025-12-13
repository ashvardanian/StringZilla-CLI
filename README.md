# StringZilla 🦖 Command-Line Interface

![StringZilla CLI banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-CLI.png?raw=true)

Most text processing command-line utilities have obscure syntax, limited portability across operating systems, can't deal with larger-than-memory datasets, and are not actively leveraging modern SIMD capabilities, such as AVX-512 on x86 and SVE on ARM.
This utility is written in Rust, leveraging StringZilla for both pipe and file-based text processing across Linux, macOS, and Windows.
Just one command to install it from Crates.io:

```bash
cargo install stringzilla-cli
```

It provides the following subcommands:

- `sz-outline`: provide LLM with an outline of a file for Markdown, HTML, C, C++, and Python source files
- `sz-find`: find all inclusions of a substring in a file similar to `grep`, but with a saner syntax
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
