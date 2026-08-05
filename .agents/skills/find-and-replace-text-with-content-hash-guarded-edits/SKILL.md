---
name: find-and-replace-text-with-content-hash-guarded-edits
description: >-
  Find text with `sz-find` and rewrite it with `sz-replace`, where one read names both the file and
  every line by its own bytes, so each write refuses instead of clobbering when the file is no
  longer what was read. Prefer this over `sed -i`, `sd`, or a Python `read_text().replace()`
  one-liner whenever more than one edit is planned, another process or a person may be writing the
  same file, or the target text may occur more than once. Patterns are literal substrings, never
  regular expressions.
when_to_use: >-
  Triggers: sz-find, sz-replace, find and replace, search and replace, rename a symbol,
  line hash, file hash, --expect-hash, --match line-hash, --occurrences one, exit code 3,
  content-addressed edit, stale read, concurrent edit, editing without line numbers,
  refuse instead of overwrite, multi-pass file editing, "edit this file in several places",
  "apply these edits safely", "another process is writing that file", "replace this everywhere".
allowed-tools: Bash(sz-find:*) Bash(sz-replace:*)
---

# Find and Replace With Content-Hash Guarded Edits

A read tells you what a file said, not what it still says, and line numbers shift the moment anything above them changes.
A line name is derived from the line's own bytes, so it survives edits elsewhere; a file token names the whole content.
Both come out of one read, and `sz-replace` refuses rather than guesses when either no longer holds.

## Finding

```bash
sz-find error log.txt                                      # literal substring, no regex
sz-find --fields line-numbers,column-numbers error log.txt # which columns each record carries
sz-find --show count error log.txt                         # or matches, files, files-without
sz-find --match word --invert-match error log.txt          # whole-word, negated
sz-find --before-context 2 --after-context 2 error log.txt
sz-find --glob '*.rs' --format json error src/             # recursive, .gitignore-aware
```

`--show` picks which records a run emits, `--fields` picks their columns, and `--format json` gives one JSON Lines record per match.

## Replacing, Guarded

```bash
# 1. Read once. `line_hash` is on each `match`; `file_hash` is on the `end` record.
sz-find --fields line-hashes,file-hash --format json TODO parser.c

# 2. Edit, naming both. `--occurrences one` asserts the name picks out exactly one line.
sz-replace --in-place --expect-hash ak5pq35xy89tr \
           --match line-hash --occurrences one bbcvyaqk '    if (at_eof(s)) return 1;' \
           --format json parser.c

# 3. Pass the summary's `hash_after` as the next `--expect-hash`, and repeat from 2.
# 4. On exit 3, go back to 1. Nothing was written.
```

One read covers a whole run of edits, because each edit hands back the token the next one passes.
Exit 3 means look at the file again, 2 means fix the arguments, 1 means the run found nothing:

| Message                            | What to do                                   |
| ---------------------------------- | -------------------------------------------- |
| `content is X, not the expected Y` | the file moved — re-read                     |
| `names no line here`               | the line moved or is gone — re-read          |
| `matches N places, not one`        | `--hash-width` up to 13, or name a neighbour |

## Every Line Operation Is One Argument

A name covers the line's terminator as well as its text:

```bash
sz-replace --match line-hash bbcvyaqk '    return 1;' parser.c            # rewrite
sz-replace --match line-hash bbcvyaqk '' parser.c                         # delete
sz-replace --match line-hash bbcvyaqk $'\n' parser.c                      # blank, keep the line
sz-replace --match line-hash --after bbcvyaqk '    flush(s);' parser.c    # insert below
sz-replace --match line-hash --before bbcvyaqk '    assert(s);' parser.c  # insert above
```

## Rules Worth Not Rediscovering

- Names are Crockford's base32 less `i`, `l`, `o` and `u`, so nothing needs shell quoting.
  Eight characters by default, 13 for the whole 64 bits, and a short name is always a prefix of the long one.
- A file token is a name at the full 13 characters; `--expect-hash` refuses anything shorter.
- A name too short to be unique can only cause a refused edit, never an edit to the wrong line.
- Adding a final newline renames the last line of a file.
- `sz-replace` edits one file per invocation and has no `--glob`.
  Fan out on the read side, and accept that such a sweep is unguarded because `--expect-hash` takes a single hash:

  ```bash
  sz-find --show files --glob '*.rs' OLD src/ | xargs -n1 sz-replace --in-place --format json OLD NEW
  ```

For non-ASCII text, `--ignore-case` and `--utf8` have semantics worth knowing before use — load `segment-tokenize-and-case-fold-unicode-text`.
