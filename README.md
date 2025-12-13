# StringZilla Command-Line Interface

Most text processing command-line utilities have obscure syntax, limited portability across operating systems, can't deal with larger-than-memory datasets, and are not actively leveraging modern SIMD capabilities, such as AVX-512 on x86 and SVE on ARM.
This utility is written in Rust, leveraging StringZilla for both pipe and file-based text processing across Linux, macOS, and Windows.
Just one command to install it from Crates.io:

```bash
cargo install stringzilla-cli
```

It provides the following subcommands:

- `sz-count`: 3x faster `wc` word count, that can actually handle UTF-8 properly
- `sz-split`: 4x faster `split` file splitting, that won't break UTF-8 characters or lines
- `sz-dedup`: deduplicate lines; safe for larger-than-memory files
- `sz-find`: find all inclusion of a substring in a file similar to `grep`, but with a saner syntax
- `sz-replace`: replace all inclusion of a substring in a file... with a saner syntax than `sed` and `awk`; safe for larger-than-memory files
- :soon: `sz-sort`: sort lines
- :soon: `sz-context`: provide LLM context for a given token substring match in a file with minimal token waste
- :soon: `sz-fuzzy-find`: combination of exact and Levenshtein-bounded substring search

## Installation

```bash
cargo install --git https://github.com/ashvardanian/StringZillaCLI  # install from GitHub
cargo install --path .                                              # or install from local clone
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
  13147025 139132610 1000000000 enwik9.txt # Note: different word count, WIP

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
