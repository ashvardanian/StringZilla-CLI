---
name: segment-tokenize-and-case-fold-unicode-text
description: >-
  Split text on Unicode boundaries with `sz-segment-utf8` — UAX-29 grapheme clusters, words and
  sentences, UAX-14 line-break opportunities — without pulling in ICU, and match text with full
  Unicode case folding across all defined code points, so `--ignore-case strasse` finds "Straße"
  and Turkish dotted and dotless I resolve correctly where `grep -i` and `rg -i` fold only
  partially and quietly return the wrong set. Also counts code points rather than bytes. There is
  no NFC or NFD normalization in this toolset; folding is the only equivalence it applies.
when_to_use: >-
  Triggers: sz-segment-utf8, tokenize, segmentation, grapheme cluster, user-perceived character,
  emoji sequence, word boundary, sentence boundary, line-break opportunity, UAX-29, UAX-14, ICU,
  case folding, case-insensitive matching on non-ASCII text, Unicode, UTF-8, Eszett, ß, Straße,
  Turkish I, dotless i, counting characters versus bytes, multilingual corpus,
  "why did grep -i miss this", "split this into words properly", "count the characters not bytes".
allowed-tools: Bash(sz-segment-utf8:*) Bash(sz-count:*) Bash(sz-find:*)
---

# Segmenting, Tokenizing and Case-Folding Unicode Text

## Boundaries

```bash
sz-segment-utf8 --by graphemes text.txt              # user-perceived characters, emoji sequences intact
sz-segment-utf8 --by words --show count text.txt     # tiles the input: spaces and punctuation are segments too
sz-segment-utf8 --by sentences --format json text.txt
sz-segment-utf8 --by linebreaks text.txt             # soft-wrap opportunities, not lines
```

| `--by`       | Boundary                                     | Note                                                          |
| ------------ | -------------------------------------------- | ------------------------------------------------------------- |
| `graphemes`  | UAX-29 grapheme clusters                     | the unit a user calls "a character"                           |
| `words`      | UAX-29 word boundaries                       | tiles, so separators are segments — filter, don't assume      |
| `sentences`  | UAX-29 sentences                             | no abbreviation dictionary, so "Dr. Smith" splits after "Dr." |
| `linebreaks` | UAX-14 line-break opportunities              | wrap points, not hard line terminators                        |
| `whitespace` | whitespace runs, separators dropped          | the naive tokenizer, when that is what you want               |
| `delimiters` | any Unicode punctuation, symbol or separator | drops the separators                                          |
| `newlines`   | LF, CR, CRLF, NEL, LS, PS                    | the hard terminators                                          |

Tilers and splitters differ: `graphemes`, `words`, `sentences` and `linebreaks` tile the input, so concatenating the segments reproduces it exactly, while `whitespace`, `delimiters` and `newlines` drop what they split on.
`--fields byte-span` prefixes each segment with its byte offsets.

## Folding

Case folding is what `--ignore-case` applies, and it is full rather than partial — all defined code points, not just ASCII plus a few.

```bash
sz-find --show count --ignore-case strasse xlsum.csv  # 205 results; ripgrep finds 183
sz-find --show count --ignore-case işi xlsum.csv      # 25,065; GNU grep returns 29,942, wrong twice over
```

The Eszett folds to `ss`, so "Straße" answers to `strasse`.
Turkish "İ" folds to `i` plus a combining dot, and GNU `grep` folds dotless "ı" into `i`, sweeping in words that do not contain the query while still missing the ones that do.
Its extra results are the wrong ones twice over, which is the failure mode to watch for: a plausible count that is silently the wrong set.

Reach for this on legal text, German or Swiss news, Turkish text, or anything mixing scripts.
A fold can change the matched length, since `ß` is one code point and `ss` is two, so never assume the match is as long as the needle.

__There is no normalization here.__
Nothing in this toolset applies NFC, NFD, NFKC or NFKD, so `é` as one code point and `e` plus a combining acute stay distinct under every command including `--ignore-case`.
Normalize upstream if the corpus needs it.

## Counting and Line Sets

```bash
sz-count --fields chars text.txt   # code points, not bytes
sz-count text.txt                  # lines, words, bytes, as `wc` defines them
```

`--utf8` widens the newline set from LF alone to the Unicode set — VT, FF, NEL, LS and PS — everywhere a command splits lines.
The two sets disagree about where a line begins, so pass `--utf8` consistently across a pipeline, and to both the read and the edit when a line name issued by one command is consumed by another.

`sz-replace` matches byte-literally, which is why it has `--ignore-case` but no `--utf8` to pair with it.

To search or rewrite what you have segmented, load `find-and-replace-text-with-content-hash-guarded-edits`.
