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
sz-find electrified paddocks.toml                                      # literal substring, no regex
sz-find --fields line-numbers,column-numbers electrified paddocks.toml # which columns each record carries
sz-find --show count electrified paddocks.toml                         # or matches, files, files-without
sz-find --match word --invert-match electrified paddocks.toml          # whole-word, negated
sz-find --before-context 2 --after-context 2 electrified paddocks.toml
sz-find --glob '*.toml' --format json electrified .             # recursive, .gitignore-aware
```

`--show` picks which records a run emits, `--fields` picks their columns, and `--format json` gives one JSON Lines record per match.

## Replacing, Guarded

```bash
# 0. Everything below writes, so run it somewhere disposable.
cd "$(mktemp -d)"
printf '[paddock.raptor]\nfeed_kg = 40.5\nelectrified = true\n\n[paddock.trex]\nfeed_kg = 250.0\nelectrified = false\n' > paddocks.toml

# 1. Read once. `line_hash` is on each `match`; `file_hash` is on the `end` record.
sz-find --fields line-hashes,file-hash --format json electrified paddocks.toml

# 2. Edit, naming both. `--occurrences one` asserts the name picks out exactly one line.
sz-replace --in-place --expect-hash d9xqrcystscnp \
           --match line-hash --occurrences one 5j5wg73n 'electrified = true' \
           --format json paddocks.toml

# 3. Pass the summary's `hash_after` as the next `--expect-hash`, and repeat from 2.
# 4. On exit 3, go back to 1. Nothing was written.
```

One read covers a whole run of edits, because each edit hands back the token the next one passes.
Exit 3 means look at the file again, 2 means fix the arguments, 1 means the run found nothing:

| Message                            | What to do                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `content is X, not the expected Y` | the file moved — re-read                                                                             |
| `names no line here`               | the line moved or is gone — re-read                                                                  |
| `matches N places, not one`        | name a neighbouring line; `--hash-width` separates a prefix collision but never byte-identical lines |

## Every Line Operation Is One Argument

A name covers the line's terminator as well as its text:

```bash
sz-replace --match line-hash jx0ehpn9 'feed_kg = 45.0' paddocks.toml       # rewrite
sz-replace --match line-hash jx0ehpn9 '' paddocks.toml                     # delete
sz-replace --match line-hash jx0ehpn9 $'\n' paddocks.toml                  # blank, keep the line
sz-replace --match line-hash --after jx0ehpn9 'fence_volts = 10000' paddocks.toml  # insert below
sz-replace --match line-hash --before jx0ehpn9 '# reviewed 1993-06-11' paddocks.toml  # insert above
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
  # Count first. `--dry-run` writes nothing and reports what each file would lose.
  sz-find --show files --glob '*.rs' OLD src/ | xargs -n1 sz-replace --dry-run --format json OLD NEW

  # Only then drop --dry-run. Nothing here checks that a file is still what was read,
  # so use it for a mechanical rename and never for a planned edit.
  sz-find --show files --glob '*.rs' OLD src/ | xargs -n1 sz-replace --in-place --format json OLD NEW
  ```

For non-ASCII text, `--ignore-case` and `--utf8` have semantics worth knowing before use — load `segment-tokenize-and-case-fold-unicode-text`.
