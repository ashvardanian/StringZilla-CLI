# StringZilla 🦖 Command-Line Interface

![StringZilla CLI banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla-CLI-v1.jpg?raw=true)

Most text processing command-line utilities have obscure syntax, limited portability across operating systems, can't deal with larger-than-memory datasets, and are not actively leveraging modern SIMD capabilities, such as AVX-512 on x86 and SVE on ARM.
This utility is written in Rust, leveraging StringZilla for both pipe and file-based text processing across Linux, macOS, and Windows.
Install it straight from the GitHub repository:

```bash
cargo install --git https://github.com/ashvardanian/StringZilla-CLI --tag v0.1.0 --locked # pinned release
cargo install --git https://github.com/ashvardanian/StringZilla-CLI --locked              # or the tip of main
cargo install --path . --force --locked                                                   # or a local clone
```

Coding agents can pull the bundled skills from the same repository, which teach them the [multi-pass editing workflow](#multi-pass-agentic-file-editing) and where Unicode folding changes an answer:

```bash
/plugin marketplace add ashvardanian/StringZilla-CLI       # Claude Code, then /plugin install stringzilla-cli@stringzilla
codex plugin marketplace add ashvardanian/StringZilla-CLI  # Codex, then /plugins
```

Codex and Pi also read `.agents/skills/` up to the repository root, so a clone needs neither command.

It provides the following subcommands:

- [`sz-find`](#sz-find-unicode-aware-substring-search): literal substring search with full Unicode case folding, so `--ignore-case strasse` finds "Straße"
- [`sz-replace`](#sz-replace-substring-replacement): literal find-and-replace with the same folding, 18x faster than GNU `sed s///g`
- [`sz-segment-utf8`](#sz-segment-utf8-unicode-text-segmentation): UAX-29 and UAX-14 grapheme, word, & sentence bounds without ICU
- [`sz-dedup`](#sz-dedup-deduplicate-lines): drops repeats in input order without sorting, 4x faster than `awk '!seen[$0]++'`
- [`sz-count`](#sz-count-word-count): `wc` counts about 4x faster than GNU, with `--fields chars` for code points
- [`sz-sort`](#sz-sort-sort-lines): stable sort in `LC_ALL=C` byte order, 3-4x faster than GNU `sort`
- [`sz-rows`](#sz-rows-extract-rows): rows by index, range, stride, or tail, replacing `sed -n`, `head`, `tail`, and `awk NR==N`
- [`sz-cols`](#sz-cols-extract-columns): columns in the order you name them — `--columns 3,1`, which `cut` cannot do
- [`sz-split`](#sz-split-split-file-into-smaller-ones): splits by lines, bytes, or a delimiter line, 5x faster than `csplit`
- [`sz-outline`](#sz-outline-file-outliner-for-llms): experimental tool for sampling file sections for LLM contexts

<details>
<summary>In the examples below it's compared to the following tools on macOS</summary>

```bash
$ alias bsd-grep=/usr/bin/grep                # BSD grep 2.6.0
$ alias gnu-grep=/opt/homebrew/bin/ggrep      # GNU grep 3.12
$ alias ripgrep=/opt/homebrew/bin/rg          # ripgrep 15.2.0
$ alias bsd-wc=/usr/bin/wc                    # BSD wc, Apple text_cmds-199
$ alias gnu-wc=/opt/homebrew/bin/gwc          # GNU coreutils 9.11
$ alias uu-wc=/opt/homebrew/bin/uu-wc         # uutils coreutils 0.9.0
$ alias bsd-sed=/usr/bin/sed                  # BSD sed, Apple text_cmds-199
$ alias gnu-sed=/opt/homebrew/bin/gsed        # GNU sed 4.10
$ alias sd=/opt/homebrew/bin/sd               # sd 1.0.0
$ alias bsd-cut=/usr/bin/cut                  # BSD cut, Apple text_cmds-199
$ alias gnu-cut=/opt/homebrew/bin/gcut        # GNU coreutils 9.11
$ alias bsd-awk=/usr/bin/awk                  # BSD awk, Apple awk-40
$ alias gnu-awk=/opt/homebrew/bin/gawk        # GNU awk 5.4.1
$ alias mawk=/opt/homebrew/bin/mawk           # mawk 1.3.4
$ alias xsv=/opt/homebrew/bin/xsv             # xsv 0.13.0
$ alias apple-perl=/usr/bin/perl              # Perl 5.34, Unicode 13.0
$ alias perl=/opt/homebrew/bin/perl           # Perl 5.42.2, Unicode 16.0
$ alias bsd-tr=/usr/bin/tr                    # BSD tr, Apple text_cmds-199
$ alias gnu-tr=/opt/homebrew/bin/gtr          # GNU coreutils 9.11
$ alias bsd-sort=/usr/bin/sort                # BSD sort, Apple text_cmds-199
$ alias gnu-sort=/opt/homebrew/bin/gsort      # GNU coreutils 9.11
$ alias uu-sort=/opt/homebrew/bin/uu-sort     # uutils coreutils 0.9.0
$ alias bsd-split=/usr/bin/split              # BSD split, Apple text_cmds-199
$ alias gnu-split=/opt/homebrew/bin/gsplit    # GNU coreutils 9.11
$ alias bsd-csplit=/usr/bin/csplit            # BSD csplit, Apple text_cmds-199
$ alias gnu-csplit=/opt/homebrew/bin/gcsplit  # GNU coreutils 9.11
```

</details>

## Tools

### `sz-find`: Unicode Aware Substring Search

A `grep`-like tool using literal substring matching (not regex) for maximum speed.
Unlike `grep` and `ripgrep`, `sz-find` performs __full Unicode-compliant case folding__ for case-insensitive search, correctly handling all 1M+ defined characters.

```bash
$ sz-find "error" log.txt                                      # literal substring search (replaces: grep -F, rg -F)
$ sz-find --ignore-case "error" log.txt                        # full Unicode folding (replaces: grep -i, rg -i)
$ sz-find --fields line-numbers "pattern" file.txt             # number every match (replaces: grep -n)
$ sz-find --show count "pattern" file.txt                      # count only (replaces: grep -c)
$ sz-find --before-context 2 --after-context 2 "error" log.txt # surrounding lines (replaces: grep -B/-A/-C)
$ sz-find --multiline $'hello\nworld' file.txt                 # match across line boundaries
$ sz-find --utf8 "pattern" file.txt                            # break lines on the Unicode newline set
```

`--show` picks which records a run emits; `--fields` picks which columns each of them carries, comma-separated: `line-numbers`, `column-numbers`, `byte-offset`, `line-hashes`, `file-hash`.

Beyond the flags above, `sz-find` also covers most of the `grep`/`ripgrep` surface: whole-word matching (`--match word`), inverted matches (`--invert-match`), only-matching output (`--show matches`), recursive directory walking with `.gitignore` awareness, type/glob filters (`--type`, `--glob`), and `--format json`/`--format vimgrep` output.

Case folding is partial in `ripgrep` and GNU `grep`, and no better in the BSD `grep` macOS ships.
The only tool seemingly implementing full folding is `pcre2`, designed for RegEx rather than substring search, and __orders of magnitude slower__.

Here's what that means for German queries, where the Eszett (ß) folds to "ss":

```bash
$ bsd-grep -c -i                      "strasse" xlsum.csv # ⚠️ 183 results, 36.13 s — 0.14 GB/s
$ gnu-grep -c -i                      "strasse" xlsum.csv # ⚠️ 183 results,  2.11 s — 2.4 GB/s
$ ripgrep  -c -i                      "strasse" xlsum.csv # ⚠️ 183 results,  0.60 s — 8.4 GB/s
$ sz-find  --show count --ignore-case "strasse" xlsum.csv # ✅ 205 results,  1.00 s — 5.0 GB/s, +22 more
```

Turkish dotted/dotless I (İ/I/i/ı) is the same pitfall, and the more common one on this corpus.
The "İ" folds to "i" plus a combining dot, so only a query ending at it diverges — as a prefix search for "IŞİD" does:

```bash
$ bsd-grep -c -i                      "işi" xlsum.csv # ⚠️ 23,957 results, 35.99 s — 0.14 GB/s
$ gnu-grep -c -i                      "işi" xlsum.csv # ⚠️ 29,942 results,  0.54 s — 9.4 GB/s, over-matches
$ ripgrep  -c -i                      "işi" xlsum.csv # ⚠️ 23,957 results,  0.58 s — 8.6 GB/s
$ sz-find  --show count --ignore-case "işi" xlsum.csv # ✅ 25,065 results,  2.22 s — 2.3 GB/s, +1,108 more
```

GNU grep folds the dotless "ı" into "i", sweeping in "atışı", "tartışıldı" and "Dışişleri" — none of which contain the query — while still missing "IŞİD".
Its extra results are the wrong ones twice over.
The difference matters for legal documents, German/Swiss news, Turkish text, or any content mixing scripts.

### `sz-count`: Word Count

The `wc` utility on Linux counts lines, words, and bytes.
A word is a maximal run of non-whitespace, exactly as in `wc -w`; `sz-count` uses the same rule, so it is a drop-in replacement rather than a different measurement.
Output is a labelled, aligned table; `--fields` chooses the columns outright rather than adding to them, so counting code points alongside the defaults is `--fields lines,words,bytes,chars`. Pass `--format human` for human-readable suffixes.

The comparison worth making is in a __UTF-8 locale__, because that is the only configuration computing the same thing.
BSD `wc` never counts code points at all, and GNU `wc` pays for them:

```bash
$ bsd-wc   -lwc                                     xlsum.csv # ⚠️ 527,203,024 words,            no chars,  8.84 s — 0.57 GB/s
$ gnu-wc   -lwmc                                    xlsum.csv # ⚠️ 523,228,738 words, 3,278,614,398 chars, 32.60 s — 0.15 GB/s
$ uu-wc    -lwmc                                    xlsum.csv # ✅ 523,228,731 words, 3,278,614,398 chars,  9.56 s — 0.52 GB/s
$ sz-count --utf8 --fields lines,words,bytes,chars  xlsum.csv # ✅ 523,228,731 words, 3,278,614,398 chars,  8.82 s — 0.57 GB/s
```

Bytes and characters agree exactly; GNU drifts by 7 words in 523 million over a handful of Unicode separators.
The locale moves the answer more than the tool does — under `LC_CTYPE=C` the same GNU run reports 527,203,024.

Dropping Unicode is where the byte-level default earns its keep, counting the same lines, words, and bytes without decoding:

```bash
$ bsd-wc   -lwc xlsum.csv # 🐌 8.84 s — 0.57 GB/s
$ uu-wc    -lwc xlsum.csv # 🐌 9.56 s — 0.52 GB/s
$ gnu-wc   -lwc xlsum.csv # 🐌 5.64 s — 0.89 GB/s
$ sz-count      xlsum.csv # ⚡ 1.19 s — 4.2 GB/s, 4.7x faster than GNU
```

The table itself is labelled and sized to its contents:

```bash
$ sz-count --utf8 --fields lines,words,bytes,chars xlsum.csv
              lines       words         bytes         chars
xlsum.csv 1,004,792 523,228,731 5,011,972,099 3,278,614,398
```

Line counts diverge too, because `wc` counts terminators where `sz-count` counts lines, and `--posix` restores `wc`'s reading exactly:

| Lines counted in                           | `wc` | `sz-count` | `--utf8` | `--posix` |
| ------------------------------------------ | ---: | ---------: | -------: | --------: |
| `a\nb`, no final newline                   |    1 |          2 |        2 |         1 |
| Text broken by CR, VT, FF, NEL, LS, and PS |    1 |          1 |        7 |         1 |

What `--posix` does not restore is locale-dependent word splitting, since the answer changing with the environment is a bug rather than a convention.
For scripting, `--fields` selects individual fields — `lines`, `words`, `bytes`, `chars`, and `max-line-length` — with `wc`'s own meanings, and a single selector over a single input prints a bare integer:

```bash
$ lines=$(sz-count --fields lines xlsum.csv) # 1004598 — no header, no commas, no padding
$ sz-count --format json xlsum.csv           # JSON Lines, bare integers, untruncated paths
```

The table truncates long paths and groups digits for humans; `--format json` does neither, so a pipeline or a model gets the path and the number intact.

### `sz-replace`: Substring Replacement

A literal-match alternative to `sed 's/old/new/g'`, with __full Unicode case folding__ on `--ignore-case`.
Reads from a file or stdin and writes to stdout, a file, or back in-place.

```bash
$ sz-replace "old" "new" file.txt                      # to stdout (replaces: sed 's/old/new/g')
$ sz-replace --in-place "old" "new" file.txt           # rewrite the file (replaces: sed -i)
$ sz-replace --ignore-case "strasse" "street" file.txt # also rewrites "Straße"
$ sz-replace --dry-run "old" "new" file.txt            # count what would change, without writing
$ sz-replace --occurrences first "old" "new" file.txt  # the leftmost match in the file alone
$ sz-replace --occurrences one "old" "new" file.txt    # exactly one, or exit 3 and change nothing
```

`--occurrences all` is the default and is what the tool has always done.
Note `first` means the first match in the _file_, where `sed 's/old/new/'` means the first on every _line_.
`one` turns "I expect this to be unique" from an assumption into an assertion, which is what makes it safe to hand a pattern to something that cannot look at the file first.

Substituting one literal for another over the same 5 GB, every tool below writes byte-identical output:

```bash
$ bsd-sed    's/error/fault/g'            < xlsum.csv # 🐌 27.86 s — 0.18 GB/s
$ gnu-sed    's/error/fault/g'            < xlsum.csv # 🐌  6.35 s — 0.79 GB/s
$ sd         -s error fault               < xlsum.csv # 🐌  2.50 s — 2.0 GB/s
$ perl       -pe 's/error/fault/g'        < xlsum.csv # 🐌  1.97 s — 2.6 GB/s
$ ripgrep    --passthru -F error -r fault < xlsum.csv # 🐌  0.68 s — 7.4 GB/s
$ sz-replace error fault                  < xlsum.csv # ⚡  0.36 s — 13.8 GB/s
```

Given "strasse", `sz-replace --ignore-case` rewrites "Bahnhofstraße", where `gnu-sed -I`, `sd` and `ripgrep -i` all leave it untouched.

`--match line-hash` reads the pattern as a line name rather than a substring, and `--expect-hash` refuses an edit built on a stale read — see [Workflows](#workflows).

### `sz-cols`: Extract Columns

The `cut` utility and `awk '{print $N}'` are commonly used to extract columns from delimited text.
`sz-cols` provides a simpler, more intuitive syntax with SIMD-accelerated delimiter scanning.

```bash
$ sz-cols --columns 2 data.tsv                                          # one column (replaces: cut -f2, awk -F'\t' '{print $2}')
$ sz-cols --columns 1,3 --delimiter ',' --output-delimiter ';' data.csv # re-delimited (replaces: cut --output-delimiter)
$ sz-cols --columns 2-5 data.tsv                                        # a range (replaces: cut -f2-5)
$ sz-cols --columns 1,3-5,8 data.tsv                                    # columns and ranges mixed (replaces: cut -f1,3-5,8)
$ sz-cols --columns 3,1 data.tsv                                        # reordered, which `cut` cannot do at all
```

Extracting one comma-separated field from the same 5 GB, where every tool but `xsv` writes byte-identical output:

```bash
$ bsd-awk -F, '{print $2}'          xlsum.csv # 🐌 47.93 s — 0.10 GB/s
$ bsd-cut -d, -f2                   xlsum.csv # 🐌 23.75 s — 0.21 GB/s
$ gnu-awk -F, '{print $2}'          xlsum.csv # 🐌  3.68 s — 1.4 GB/s
$ xsv     select 2                  xlsum.csv # ⚠️  2.82 s — 1.8 GB/s, quote-aware
$ mawk    -F, '{print $2}'          xlsum.csv # ⚡  1.28 s — 3.9 GB/s
$ sz-cols --delimiter , --columns 2 xlsum.csv # ⚡  0.42 s — 12 GB/s
$ gnu-cut -d, -f2                   xlsum.csv # ⚡  0.33 s — 15 GB/s
```

GNU `cut` takes this one, and the margin is thin because the corpus barely asks a column tool for anything: lines average 5 KB over three real columns, so the first two fields cover a quarter of each line and the rest is skipped to find the newline.
That makes the row a memory-bandwidth race rather than a splitting one.

Reordering is the part `cut` cannot do at all — it is specified to emit fields in the order the __file__ holds them, so `-f 3,1` and `-f 1,3` are one command, and `-f 1,1,3` collapses to two columns.
That leaves `awk`, which pays for splitting the whole record:

```bash
$ bsd-awk -F, -v OFS='\t' '{print $3,$1}'                       xlsum.csv # 🐌 50.27 s — 0.10 GB/s
$ gnu-awk -F, -v OFS='\t' '{print $3,$1}'                       xlsum.csv # 🐌  4.59 s — 1.1 GB/s
$ xsv     select 3,1                                            xlsum.csv # ⚠️  2.33 s — 2.2 GB/s, quote-aware
$ mawk    -F, -v OFS='\t' '{print $3,$1}'                       xlsum.csv # 🐌  1.39 s — 3.6 GB/s
$ sz-cols --delimiter , --output-delimiter '\t' --columns 3,1   xlsum.csv # ⚡  0.43 s — 12 GB/s
```

Output matches `mawk` byte for byte, and costs nothing over the single-field row, since fields are emitted from borrowed slices in the order named rather than by rebuilding the record.

Splitting on a byte is not the same as parsing CSV, though.
Most fields in this corpus are quoted and contain commas, so `xsv`, which honours the quoting, disagrees with every other row above on roughly four lines in five.

### `sz-rows`: Extract Rows

The `sed -n 'Np'`, `head -n N`, `tail -n N`, and `awk 'NR==N'` commands are commonly used to extract specific lines.
`sz-rows` unifies all these use cases with a single, intuitive interface.

```bash
$ sz-rows --rows 5 file.txt                   # one line (replaces: sed -n '5p', awk 'NR==5')
$ sz-rows --rows 10-20 file.txt               # a range (replaces: sed -n '10,20p')
$ sz-rows --rows 1-10 file.txt                # the first ten (replaces: head -n 10)
$ sz-rows --tail 10 file.txt                  # the last ten (replaces: tail -n 10)
$ sz-rows --rows 1,5,10 file.txt              # scattered lines in one pass (replaces: sed -n '1p;5p;10p')
$ sz-rows --every 5 file.txt                  # every fifth (replaces: awk 'NR % 5 == 0')
$ sz-rows --fields line-numbers --rows 5-10 file.txt # numbered output, as grep -n writes it
$ sz-rows --fields line-hashes,file-hash --rows 5-10 file.txt # named output, to edit against
```

Reaching a range deep in the same 5 GB, and sampling every thousandth line:

```bash
$ gnu-sed -n '1000000,1000020p'        xlsum.csv # 🐌 3.77 s
$ bsd-sed -n '1000000,1000020p'        xlsum.csv # 🐌 0.77 s
$ mawk    'NR>=1000000 && NR<=1000020' xlsum.csv # ⚡ 0.33 s
$ sz-rows --rows 1000000-1000020       xlsum.csv # ⚡ 0.30 s

$ gnu-awk 'NR % 1000 == 0'             xlsum.csv # 🐌 1.38 s
$ mawk    'NR % 1000 == 0'             xlsum.csv # ⚡ 0.34 s
$ sz-rows --every 1000                 xlsum.csv # ⚡ 0.30 s
```

The margin over `mawk` is thin; the outlier is GNU `sed`, five times slower here than the BSD `sed` macOS ships.

`--tail` is the exception: like `tail -n`, it seeks from the end rather than scanning, so both return in a millisecond regardless of file size.

### `sz-segment-utf8`: Unicode Text Segmentation

Splitting text into words or sentences the way Unicode defines them is something no standard command-line tool does, and the one that attempts characters gets the Indic scripts wrong.
Coreutils has nothing, ICU ships `genbrk` and `uconv` but neither segments text, and the NLP libraries that do are abbreviation heuristics rather than the standard.
`sz-segment-utf8` exposes StringZilla's UAX-29 and UAX-14 kernels directly.

```bash
$ sz-segment-utf8 --by sentences book.txt                               # one UAX-29 sentence per line
$ sz-segment-utf8 --by graphemes --show count emoji.txt                 # count user-perceived characters
$ sz-segment-utf8 --by sentences --fields byte-span doc.txt             # start and end, for citing back to source
$ sz-segment-utf8 --by sentences --chunk-bytes 2000 --format json c.txt # 2 KB records for an embedder
```

Seven modes, each named for the iterator behind it, over the same 5 GB corpus.
Single-threaded, warm cache, and CPU-bound — user time equals wall time in every row, and every mode costs more than simply reading the file:

| Mode              | Yields                                        |      Segments | Throughput |
| ----------------- | --------------------------------------------- | ------------: | ---------: |
| `--by graphemes`  | UAX-29 user-perceived characters              | 3,099,176,696 |   218 MB/s |
| `--by linebreaks` | UAX-14 wrap opportunities, __not lines__      |   581,578,192 |   231 MB/s |
| `--by words`      | UAX-29 word boundaries, __tiling__            | 1,222,591,864 |   339 MB/s |
| `--by sentences`  | UAX-29 sentences                              |    27,327,870 |   472 MB/s |
| `--by whitespace` | Runs between the 25 Unicode spaces            |   523,228,731 |   681 MB/s |
| `--by delimiters` | Runs between punctuation, symbols, separators |   535,725,987 |  1023 MB/s |
| `--by newlines`   | Runs between hard line terminators            |     1,004,731 |  4876 MB/s |

Cost tracks the work each rule demands rather than the number of boundaries found: `--by delimiters` yields more segments than `--by whitespace` and still runs half again as fast, while `--by graphemes` is the slowest of all despite the whole file being nothing but characters.

The tiling modes assign every byte to exactly one segment, so `--by words` returns whitespace and punctuation as segments of their own — a word count has to filter for segments containing a letter or digit.
`--by linebreaks` reports where a renderer _may_ wrap, so use `--by newlines` to split on actual terminators.

Graphemes are the one mode a stock tool will attempt, since Perl's `\X` is a UAX-29 clusterer, and the answer depends on which Unicode release it was built against:

```bash
$ apple-perl     -CSD -nE '$n += ()=/\X/g; END{say $n}' xlsum.csv # ⚠️ Unicode 13: 3,114,363,758 clusters, 235.6 s —  21 MB/s
$ perl           -CSD -nE '$n += ()=/\X/g; END{say $n}' xlsum.csv # ⚠️ Unicode 16: 3,095,814,378 clusters, 209.7 s —  24 MB/s
$ sz-segment-utf8 --by graphemes --show count           xlsum.csv # ✅ Unicode 17: 3,099,176,696 clusters,  22.3 s — 225 MB/s
```

Unicode 15.1 added rule GB9c, joining an Indic consonant to the next across a virama.
Perl 5.34 lacks it and splits every conjunct; Perl 5.42 has it but merges whole chains, reading "র্বত্য" as one cluster where the linker binds only the first pair.
`sz-segment-utf8` agrees with ICU 78 boundary for boundary.

Splitting on whitespace is the one job the shell already has tools for, and only one of them handles both the ideographic and the no-break space below:

```bash
$ printf 'a\u3000b\u00a0c d\n' | gnu-awk '{print NF}'                         # ⚠️ 2 tokens
$ printf 'a\u3000b\u00a0c d\n' | gnu-tr -s '[:space:]' '\n'                   # ⚠️ 3 tokens
$ printf 'a\u3000b\u00a0c d\n' | bsd-tr -s '[:space:]' '\n'                   # ✅ 4 tokens
$ printf 'a\u3000b\u00a0c d\n' | sz-segment-utf8 --by whitespace --show count # ✅ 4 tokens
```

Segments frequently contain newlines, so the default one-per-line output is lossy — pass `--null` for NUL-delimited records, or `--format json` for JSON Lines carrying offsets.
`--chunk-bytes N` packs consecutive segments into records of at most N bytes without splitting one, and is defined only over the tiling modes, since the `whitespace`, `delimiters`, and `newlines` modes discard separators that a packed span would reinsert.
A segment larger than the budget is emitted whole.

UAX-29 sentences are the standard applied deterministically, with no dictionary: `"Dr. Smith went to Washington."` breaks after `"Dr. "`, and rule SB4 breaks at a hard wrap.
Both match ICU exactly: over a 2 MB multilingual sample its break iterators return the same 9,260,355 graphemes and 81,510 sentences.
Both are also places `punkt` or `pysbd` read more naturally — the trade is spec-correct segmentation across every script, not better English.

### `sz-sort`: Sort Lines

A stable, Unicode-correct `sort` built on StringZilla's `argsort`.
Comparison is unsigned byte-wise, which for valid UTF-8 is exactly Unicode code-point order, so `--utf8` only governs newline handling and output is byte-identical to `LC_ALL=C sort`.
The sort is always stable, so the comparisons below give `sort` its `-s` flag, which drops the last-resort whole-line comparison it would otherwise pay for.

```bash
$ sz-sort file.txt                     # to stdout (replaces: sort)
$ sz-sort --reverse file.txt           # descending (replaces: sort -r)
$ sz-sort --unique file.txt            # sorted and deduplicated (replaces: sort -u)
$ sz-sort --ignore-case file.txt       # full Unicode folding (replaces: sort -f)
$ sz-sort file.txt --output sorted.txt # to a file (replaces: sort -o)
$ sz-sort --check file.txt             # exit 1 unless already sorted (replaces: sort -c)
```

Lines are held as packed offset and length pairs borrowing the input rather than copies of it.
Sorting rewards short records, so the corpus is first split into one word per line:

```bash
$ sz-segment-utf8 --by whitespace xlsum.csv > xlsum-words.txt  # 523,228,731 lines, 12.75 s

$ gnu-sort -s    --parallel=1 xlsum-words.txt # 🐌 262.94 s —  8192 MB
$ uu-sort  -s    --parallel=1 xlsum-words.txt # 🐌  82.34 s —  5518 MB
$ sz-sort                     xlsum-words.txt # ⚡  63.51 s — 25740 MB

$ gnu-sort -s -f --parallel=1 xlsum-words.txt # 🐌 442.78 s —  8194 MB
$ sz-sort  --ignore-case      xlsum-words.txt # ⚡ 136.85 s — 25740 MB

$ gnu-sort -s -u --parallel=1 xlsum-words.txt # 🐌 208.42 s —  8194 MB
$ sz-sort  --unique           xlsum-words.txt # ⚡  56.55 s — 27020 MB
```

Both alternatives are pinned to one thread; left to itself `uu-sort` spreads over every core and finishes in a third of the time.
Resident sizes are not comparable as printed, since `sz-sort` maps the input where the others stream.
Its index is the smallest of the three per line, but unbounded: GNU caps its buffer and spills to temporary files, which is most of why it trails.

### `sz-dedup`: Deduplicate Lines

Drop repeated lines, keeping the first of each, __without sorting__.
`uniq` collapses only adjacent duplicates and so needs sorted input, and `sort -u` gets there by discarding the original order; the idiom that actually preserves order is `awk '!seen[$0]++'`.

```bash
$ sz-dedup file.txt               # first of each, input order kept (replaces: awk '!seen[$0]++')
$ sz-dedup --in-place file.txt    # rewrite the file instead
$ sz-dedup --ignore-case file.txt # full Unicode folding
$ sz-dedup --quiet file.txt       # report through the exit code alone
```

Lines are hashed with StringZilla's SIMD hash into an open-addressed table that holds one entry per __distinct__ line, so memory follows the number of unique lines rather than the length of the input.

Deduplication rewards short lines, so the corpus is first split into one word per line:

```bash
$ sz-segment-utf8 --by whitespace xlsum.csv > xlsum-words.txt # 523,228,731 lines, 12.75 s

$ gnu-sort -s -u --parallel=1         xlsum-words.txt # 🐌 208.42 s — 8194 MB, order lost
$ perl -ne 'print unless $seen{$_}++' xlsum-words.txt # 🐌  98.67 s — 2912 MB
$ mawk '!seen[$0]++'                  xlsum-words.txt # 🐌  79.19 s — 1800 MB
$ sz-dedup                            xlsum-words.txt # ⚡  20.36 s — 6317 MB
```

`sz-dedup` is the only entry that both preserves order and never sorts.
Resident size flatters the others: it maps its input, where `awk` and `perl` stream and pay only for their tables.

`uniq` is absent because on unsorted input it returns almost every line.

### `sz-split`: Split File into Smaller Ones

A chunk can be a line count, a byte budget, a share of the whole, or a delimiter line.
Only the first has an equivalent in `split`; the last is what `csplit` exists for.

```bash
$ sz-split --chunk-lines 100000   large.csv part.               # 100k lines per chunk (replaces: split -l)
$ sz-split --chunk-bytes 100MB    large.csv part.               # 100 MB shards, never splitting a line
$ sz-split --chunk-count 16       large.csv part.               # one chunk per core, cut at line ends
$ sz-split --chunk-pattern '>'    seqs.fa   rec.                # a new chunk at each line starting with >
$ sz-split --repeat-header --chunk-bytes 100MB large.csv        # every shard keeps the CSV header
$ sz-split --chunk-lines 100000 --format json large.csv part.   # a record per chunk written
```

A line is never split, whatever the budget: one longer than `--chunk-bytes` becomes an over-budget chunk of its own.
A chunk is a contiguous range of the input, so LF mode copies that range whole rather than re-emitting each line:

```bash
$ bsd-split -l 200000 xlsum.csv            bsd. # 🐌 3.11 s — 1.6 GB/s
$ gnu-split -l 200000 xlsum.csv            gnu. # ⚡ 0.71 s — 7.1 GB/s
$ sz-split  --chunk-lines 200000 xlsum.csv sz.  # ⚡ 0.70 s — 7.2 GB/s
```

That matters most where lines are short and numerous, and BSD `split` never finished this one:

```bash
$ sz-segment-utf8 --by whitespace xlsum.csv > xlsum-words.txt # 523,228,731 lines of 9.6 bytes

$ gnu-split -l 35000000 xlsum-words.txt            gnu. # 🐌 4.93 s — 1.0 GB/s
$ sz-split  --chunk-lines 35000000 xlsum-words.txt sz.  # ⚡ 3.65 s — 1.4 GB/s
```

Splitting around a delimiter is where the gap is widest, since `csplit` runs a regex engine over every line.
All three write byte-identical files over 600 MB of FASTA in 2,000 records:

```bash
$ bsd-csplit -f b. -n 4 seqs.fa '/^>/'               '{1998}' # 🐌 1.85 s — 0.33 GB/s
$ gnu-csplit -z -f g. -b '%04d' seqs.fa '/^>/'       '{*}'    # 🐌 1.19 s — 0.51 GB/s
$ sz-split   --chunk-pattern '>' --suffix-length 4 seqs.fa s. # ⚡ 0.22 s — 2.8 GB/s
```

The pattern is literal and anchored to a line start, so a `>` inside a sequence line is not a boundary; `--ignore-case` folds case for it.
`--chunk-count N` needs a file rather than a pipe, since it asks the input for its size.
`--repeat-header` is the one flag that stops `cat <prefix>*` reproducing the input.

### `sz-outline`: File Outliner for LLMs

> [!WARNING]
> This one is being reimplemented and is excluded from the default build.
> Enable it with `cargo build --release --features outline`.
> It is being designed without a prior-art reference, so expect it to change a lot even in minor releases.

Extract structural outlines from source files for LLM context windows.
When feeding large files to language models, you often need a high-level overview without the full content.
`sz-outline` extracts headings, function signatures, includes, and other structural elements.
File type is inferred from the extension — Markdown (`.md`, `.markdown`) and C (`.c`, `.h`) — or forced with `--language {md,c,h}` (required when reading from stdin).

```bash
$ sz-outline README.md                        # headings only
$ sz-outline --detail positions README.md     # add line numbers and byte offsets
$ sz-outline --detail blocks README.md        # add child blocks and their sizes
$ sz-outline src/main.c                       # includes and function signatures
$ sz-outline --language md document.txt       # force the parser, whatever the extension
$ cat file.md | sz-outline --language md -    # from a pipe, where the type cannot be inferred
```

There are several detail levels supported:

__Default__ — names only:

```bash
$ sz-outline README.md
# StringZilla 🦖 Command-Line Interface
## Tools
### `sz-find`: Unicode Aware Substring Search
### `sz-count`: Word Count
### `sz-replace`: Substring Replacement
...
```

__`--detail positions`__ — line numbers and byte offsets:

```bash
$ sz-outline --detail positions README.md
# StringZilla 🦖 Command-Line Interface           [L1, @0]
## Tools                                         [L62, @4151]
### `sz-find`: Unicode Aware Substring Search    [L64, @4161]
### `sz-count`: Word Count                       [L109, @7488]
...
```

__`--detail blocks`__ — child blocks too, with their sizes:

```bash
$ sz-outline --detail blocks README.md
# StringZilla 🦖 Command-Line Interface           [L1, @0, 41B]
  - image: StringZilla CLI banner                [L3, 125B]
  - paragraph                                    [L5-7, 436B]
  - code (bash)                                  [L9-13, 338B]
  - paragraph                                    [L15, 39B]
...
```

For C source files, `sz-outline` extracts includes and function signatures:

```bash
$ sz-outline --detail positions src/parser.c
#include <stdio.h>                        [L1, @0]
#include <stdlib.h>                       [L2, @19]
#include "parser.h"                       [L3, @39]
static int parse_token(const char *input) [L12-45, @156, definition]
int parse_file(FILE *fp)                  [L47-123, @892, definition]
void cleanup(void)                        [L125-130, @2341, definition]
```

Every bracketed detail starts in one column, sized to the longest name the run prints rather than to a fixed width.

Function signatures are normalized (whitespace collapsed) and categorized as declarations (`;`) or definitions (`{}`).

### `sz-fuzzy-find`: Fuzzy Substring Search

> [!WARNING]
> This one is being reimplemented and is excluded from the default build.
> Enable it with `cargo build --release --features fuzzy-find`.

`sz-find` matches literally; `sz-fuzzy-find` adds typo tolerance, built on StringZilla's `szs` similarity kernels, multicore by default and GPU-capable.
Exact hits are claimed first with StringZilla's `find`, so only the remainder reaches the kernel.

Three behaviours are known to be wrong today and are what the rewrite fixes.
`--max-distance` is an edit budget only under `--match word`; elsewhere it becomes a Smith-Waterman score floor, so `Washigton` at one edit is missed while `WaShInGtOn` at four matches.
All ten digits share one scoring class, so `--max-distance 0 2024` also returns `1999`.
And every non-exact line reaches the kernel, which is why a 50 MB corpus takes 180 ms where `ugrep -Z1` takes 3 ms.

```bash
# Find "color" allowing up to 1 edit — also matches "colour", "colur", "kolor"
$ sz-fuzzy-find --max-distance 1 color file.txt

# Several queries at once (a line matches if ANY query matches)
$ sz-fuzzy-find --max-distance 1 --pattern foo --pattern bar file.txt

# Word mode: match the needle against each token (Levenshtein), not the whole line
$ sz-fuzzy-find --match word --max-distance 1 colour file.txt

# Count matching lines only
$ sz-fuzzy-find --show count --max-distance 1 needle file.txt
```

Word mode carries the edit-distance semantics, and agrees with `agrep` line for line.
Over 50 MB of multilingual news, one query, 225 matching lines:

| Tool                                          |       Time | Semantics                                        |
| --------------------------------------------- | ---------: | ------------------------------------------------ |
| `tre-agrep -E 1`                              |     2.90 s | edit distance                                    |
| `fzf --filter`                                |     0.43 s | subsequence — 3,508 lines, not the same question |
| `sz-fuzzy-find --match word --max-distance 1` | __0.21 s__ | edit distance                                    |

`fzf` is there for scale rather than parity: it matches characters in order with gaps, so it finds `W-a-s-h-i-n-g-t-o-n` and misses `Washigton`.

#### Scoring models (`--cost`)

Beyond uniform edit distance, scoring can reflect _how_ characters get confused.
These route through Smith-Waterman with a `byte_to_class[256]` + `class_substitution_costs[32][32]` matrix, and use a normalized `--min-similarity` (0..1, where 1.0 is exact) instead of `--max-distance`:

```bash
# Keyboard proximity: fat-finger typos (adjacent keys cost less) — "xolor" matches "color"
$ sz-fuzzy-find --cost keyboard --min-similarity 0.8 color file.txt

# Phonetic: sounds-alike (Editex-style articulatory groups) — "fonetik" matches "phonetic"
$ sz-fuzzy-find --cost phonetic --min-similarity 0.75 phonetic file.txt

# Custom 256→class map + 32×32 score matrix
$ sz-fuzzy-find --cost-matrix my_costs.txt --min-similarity 0.8 needle file.txt
```

The keyboard matrix uses staggered-QWERTY Euclidean key distance; the phonetic matrix is seeded by voiced/unvoiced cognates (`b/p`, `d/t`, …) and Editex letter groups. Smith-Waterman fuzzy matching folds ASCII case (the 32-class budget leaves no room to distinguish case per letter).

#### Execution device (`--device`)

```bash
$ sz-fuzzy-find --device cpu --threads 8 --max-distance 1 needle big.txt # CPU, 8 threads
$ sz-fuzzy-find --device gpu --max-distance 1 needle big.txt             # GPU (see build note)
```

Every core is used unless `--threads` says otherwise. The GPU path requires a CUDA build:

```bash
# On systems with gcc > 14 + CUDA 12.x, point nvcc at a supported host compiler:
$ CUDAHOSTCXX=g++-14 cargo install --git https://github.com/ashvardanian/StringZilla-CLI --features cuda --locked
```

## Workflows

### Multi-Pass Agentic File Editing

A program that reads a file, decides what to change, and writes it back some time later has no way to know the file is still what it read — a formatter on save, a second process, or a person in an editor invalidates the plan and the edit lands anyway.
Line numbers have the same problem one level down: change line 10 and every number below it moves, so a plan made before the first edit is wrong by the second.

The three ways a script rewrites one line it has already read — the first is what a language model writes when asked to edit a file, the second what it writes when asked for a one-liner:

```bash
$ python3 -c "p = Path('parser.c'); p.write_text(p.read_text().replace(old, new))"
$ sed -i 's|// TODO: handle EOF|if (at_eof(s)) return 1;|' parser.c
$ sz-replace --in-place --expect-hash ak5pq35xy89tr \
             --match line-hash --occurrences one bbcvyaqk '    if (at_eof(s)) return 1;' parser.c
```

They differ only once the file stops being what the read said it was:

| Situation                                    | `python -c`          | `sed -i`             | `sz-replace`              |
| -------------------------------------------- | -------------------- | -------------------- | ------------------------- |
| The line is where the read left it           | rewrites it, exit 0  | rewrites it, exit 0  | rewrites it, exit 0       |
| Something else wrote the file after the read | __overwrites it__, 0 | __overwrites it__, 0 | __refuses__, exit 3       |
| The named text occurs twice                  | __rewrites both__, 0 | __rewrites both__, 0 | __refuses__, exit 3       |
| The line is no longer there                  | silent no-op, exit 0 | silent no-op, exit 0 | __refuses__, exit 3       |
| The next edit to the same file               | needs a fresh read   | needs a fresh read   | reuses the returned token |

Three of the four answer by writing and exiting 0, which is why a plan of edits built from one read lands whether or not it is still true.

Two names solve the two halves, and both come out of the same read.
Everything below runs against one small file:

```bash
$ printf 'static int parse(state_t *s) {\n    // TODO: handle EOF\n    return 0;\n}\n' > parser.c
```

__One read names the file and every line it prints.__
The file's name is a token for the whole content; a line's name is derived from its own bytes, so it survives edits elsewhere where a number would shift:

```bash
$ sz-find --heading --fields line-numbers,line-hashes,file-hash TODO parser.c
parser.c  ak5pq35xy89tr
2:bbcvyaqk:    // TODO: handle EOF
```

Or, for a program reading the output, one record per file and one per line:

```bash
$ sz-find --fields line-hashes,file-hash --format json TODO parser.c
{"type":"begin","data":{"path":{"text":"parser.c"}}}
{"type":"match","data":{"path":{"text":"parser.c"},"lines":{"text":"    // TODO: handle EOF"},"line_number":2,"absolute_offset":31,"line_hash":"bbcvyaqk","submatches":[{"match":{"text":"TODO"},"start":7,"end":11}]}}
{"type":"end","data":{"path":{"text":"parser.c"},"file_hash":"ak5pq35xy89tr","stats":{"matches":1,"lines_searched":4}}}
```

__The edit names both.__
`--expect-hash` is the file token; the pattern is the line name; `--occurrences one` asserts the name picks out exactly one line:

```bash
$ sz-replace --in-place --expect-hash ak5pq35xy89tr \
             --match line-hash --occurrences one bbcvyaqk '    if (at_eof(s)) return 1;' \
             --format json parser.c
{"type":"summary","data":{"path":{"text":"parser.c"},"replacements":1,"dry_run":false,"hash_before":"ak5pq35xy89tr","hash_after":"nxmqh0zya0nce"}}
```

__The token it hands back is the one the next edit passes__, so a run of edits costs one read rather than one per edit:

```bash
$ sz-find --fields line-hashes at_eof parser.c
ytyymgew:    if (at_eof(s)) return 1;

$ sz-replace --in-place --expect-hash nxmqh0zya0nce \
             --match line-hash --after ytyymgew '    flush(s);' --format json parser.c
{"type":"summary","data":{"path":{"text":"parser.c"},"replacements":1,"dry_run":false,"hash_before":"nxmqh0zya0nce","hash_after":"tm5feseweghxa"}}
```

__Both names refuse rather than guess.__
A spent file token means the file moved; a name that picks out nothing means the line did.
Either way the file is untouched and the exit code says to look again:

```bash
$ sz-replace --in-place --expect-hash ak5pq35xy89tr --match line-hash bbcvyaqk x parser.c
sz-replace: parser.c: content is tm5feseweghxa, not the expected ak5pq35xy89tr; re-read it before editing
$ echo $?
3

$ sz-replace --in-place --match line-hash bbcvyaqk x parser.c
sz-replace: parser.c: `bbcvyaqk` names no line here; re-read the file
$ echo $?
3
```

Exit code 3 is distinct from 1 (ran, found nothing) and 2 (could not run), because the recovery is to look at the file again rather than to fix the arguments.
Which look depends on the message: a spent file token or a name that resolves nowhere means re-reading, while a name that resolves in several places means asking `sz-find` for a longer one with `--hash-width`.

The loop, for a program driving it:

1. `sz-find --fields line-hashes,file-hash --format json <pattern> <file>` — read `line_hash` off each `match`, and `file_hash` off the `end` record, which is written once the whole file has been read.
2. `sz-replace --in-place --expect-hash <file_hash> --match line-hash --occurrences one <line_hash> <new text> --format json <file>`.
3. Take `hash_after` from the summary record and pass it as the next `--expect-hash`. Repeat from 2.
4. On exit 3, go back to 1. Nothing was written.

Both names refuse rather than guess, and they cover different things.
The line name guards the line being edited and costs nothing, because a line that has changed no longer answers to its name.
`--expect-hash` guards everything else in the file — a line inserted above yours, or a second copy of your line appearing and making the name ambiguous.

A name identifies content rather than position, so identical lines share one, and `--occurrences` decides what happens to the set exactly as it does for a substring.
Every candidate is found before a byte is written, so __a name too short to be unique can only ever cause a refused edit, never an edit to the wrong line__ — which is what makes the default eight characters safe.
`--hash-width` goes up to 13 for the whole 64 bits, and because rendering truncates rather than folds, a short name is always a prefix of the long one.

Both names are written in the same alphabet — Crockford's base32, less `i`, `l`, `o` and `u`, so nothing needs shell quoting and no character is confusable with another.
A file token is simply a name at the full 13 characters; anything shorter is a line name, and `--expect-hash` refuses it rather than zero-extending a truncated paste into a match against some other file.
Each character pins five bits, so eight characters is forty.

A name covers the line's terminator as well as its text.
That is what makes a CRLF line name the same under `--utf8` as under the default LF reading, and it is why every line operation falls out of one replacement argument:

```bash
$ sz-replace --match line-hash bbcvyaqk '    return 1;' parser.c   # rewrite
$ sz-replace --match line-hash bbcvyaqk '' parser.c                # delete
$ sz-replace --match line-hash bbcvyaqk $'\n' parser.c             # blank, keeping the line
```

The cost of covering the terminator is that adding a final newline renames the last line of a file.
The two newline sets still disagree about where a line _begins_ at VT, FF, NEL, LS and PS, which no naming scheme can reconcile: pass `--utf8` to both the read and the edit, or to neither, or a name issued under one will refuse under the other.

The hash is StringZilla's 64-bit AES hash under seed zero, identical across platforms.
It answers "did this file change", not "did somebody change it" — it is fast, not cryptographic.
