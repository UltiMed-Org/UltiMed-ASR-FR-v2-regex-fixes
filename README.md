# UltiMed-ASR-FR-v2-regex-fixes

Cheap post-processing for French medical ASR: lists of rules that rewrite the misspellings the [UltiMed parakeet model](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-onnx) still makes on French medical vocabulary into the right word, for example `veine la faxine` becomes `venlafaxine`. Plain text substitution after decoding: no GPU, no extra model. They also help models that were not finetuned.

Each rule list ships in two equivalent forms: regexes (`rules/`) and lookup tables (`lookup/`), for apps whose regex engine has no lookbehind. Both give the same output on every text (see "Equivalence" below).

This repository, its scripts and its docs were written with Claude Code. The rules and their build history moved here on 2026-11-15 from the model repository (`regex-fixes/`) and from `08_drug_asr_rules/` of [UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts).

License: AGPL-3.0-or-later for everything in this repository (rules, lookup tables, scripts, tests), see `LICENSE`. The build scripts were first published in UltiMed-ASR-FR-v1-scripts under Apache-2.0 and the v1 rule files in the model repository under CC BY 4.0; those earlier copies keep their original license, this repository is the same author's relicensed continuation.

## The rules

These are the **v2** rules, built on 2026-11-13 from the v2 model alone (the "mix7" weight soup, fp32 ONNX) transcribing every clip of the v2 UltiMed corpus ([Olicorne/UltiMed-ASR-FR-v2](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2), 602,776 clips). The rules are learned from the v2 labels, so the replacement spelling is the one the v2 labels use. Each subset's file was built by `scripts/07_build_dataset_rules.sh`; [scripts/README.md](scripts/README.md) has the full build log, the guards and the evaluation.

| file | what it fixes | variants | rules (merged) |
|---|---|---|---|
| `manual_fix_rules.jsonl` | hand-written fixes (`post-opératoire`, `lodose` to `lowdose`, ...), dataset-agnostic, unchanged since v1 | 1,943 | 1,905 |
| `drug_fix_rules.jsonl` | drug names, learned on the drugs subset | 4,854 | 2,388 |
| `dictionary_fix_rules.jsonl` | medical terms, learned on the dictionary subset, then filtered against French news 2024 | 42,827 | 18,137 |
| `parhaf_fix_rules.jsonl` | medical terms, learned on the PARHAF subset | 3,960 | 2,103 |
| `parrot_fix_rules.jsonl` | medical terms, learned on the PARROT subset (test only) | 154 | 121 |
| `acronyms_fix_rules.jsonl` | medical terms, learned on the acronyms subset | 43 | 43 |

The four term files replace v1's single `term_fix_rules.jsonl`. They overlap: 1,116 variants appear in both the dictionary and PARHAF files, and 35 variants get a different replacement depending on the file (19 dictionary vs PARHAF, e.g. `trocarde` becomes `trocart de` in one and `trocart d'` in the other; 12 dictionary vs drugs; 3 PARHAF vs drugs; 1 dictionary vs PARROT). So the order matters: apply the manual file first (a hand fix wins over a learned one), then the drug file. TODO: decide and measure the order of the four term files, or merge them into one checked list as v1 did.

### Measured effect

The v2 model's hypotheses on each subset's test split, each subset scored with its own rule file (WER without rules, then with rules, then clips better / worse):

| test split | no rules | v2 rules |
|---|---|---|
| drugs (2,059 clips) | 3.14 % | 1.77 % (767 / 0) |
| dictionary (49,891) | 2.39 % | 1.89 % (8,693 / 3) |
| PARHAF (5,489) | 2.76 % | 2.47 % (894 / 0) |
| PARROT (1,548) | 3.68 % | 3.37 % (190 / 0) |
| acronyms (151) | 2.90 % | 2.72 % (5 / 0) |

These rules are learned on ALL splits, test included, on purpose: the list is meant to fix every mistake the model made on the dataset. So these numbers show how many seen mistakes the rules fix, not how well they generalize. A build that held the test split out gives 2.37 % on drugs, 2.04 % on dictionary and 2.58 % on PARHAF.

Overcorrection, per file (drugs, dictionary, PARHAF and acronyms; TODO for PARROT): on the v2 labels these files rewrite 0 lines; on 100,000 held-out French news sentences (2023) the dictionary file rewrites 10, PARHAF 2, drugs 1, acronyms 0; on 1,662 French VoxPopuli references, 0. TODO: overcorrection and WER of all files applied together, in the final order.

TODO: real speech. No v2 transcripts of FLEURS-fr, Common Voice fr or the author's own recordings existed when the rules were built, so their effect on real speech with the v2 model is not measured yet.

## Layout

| path | content |
|---|---|
| `rules/` | the six v2 rule files (regex form) |
| `lookup/` | the same six files in lookup-table form, generated from `rules/` by `scripts/08_build_lookup.py` |
| `rejected/` | the candidates each v2 build rejected, with the reason, and `drug_asr_errors.json` (the misspellings the drug build started from) |
| `inputs/` | the hand-written `manual_fixes.tsv` and the generated `convention_fixes.tsv`, source of `manual_fix_rules.jsonl` |
| `v1/` | the v1 outputs, kept for comparison: `drug_fix_rules.jsonl`, `term_fix_rules.jsonl` (dictionary + PARHAF + PARROT in one list), their rejected lists and the per-model `drug_asr_errors*.json` |
| `scripts/` | the build chain (01 to 07, see [scripts/README.md](scripts/README.md)), `08_build_lookup.py` and `apply_lookup.py` |
| `tests/` | `test_drug_asr_rules.py` (the build scripts) and `test_lookup.py` (regex vs lookup equivalence), plus a fixture of 299 real v2 transcripts |

The build scripts read the corpus manifests, the drug lexicon and a few helpers from a [UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts) checkout. By default they look for it next to this repository; set `ULTIMED_SCRIPTS_DIR` to point elsewhere. Applying the rules, building the lookups and `test_lookup.py` need nothing outside this repository.

## Regex form (`rules/*.jsonl`)

One JSON rule per line, applied top to bottom, file after file. For each rule:

- compile `pattern` case-insensitively (`new RegExp(pattern, 'giu')` in JavaScript, `re.IGNORECASE` in Python; the patterns only use syntax both engines share);
- replace every match with `replacement`;
- when the matched text starts with a capital and the replacement does not, capitalise the replacement's first letter.

A rule with `"name": true` is also a French surname: its pattern already skips a match right after a title like `docteur`, so the surname in `docteur X` is left alone. Merged rules carry a `variants` list (one alternation per replacement); unmerged ones a single `variant` plus build statistics.

With tens of thousands of rules a naive loop over every rule for every sentence is slow: index each rule under a literal anchor of its variants and only try the rules whose anchor occurs in the text. Two implementations do this: `compile_rules` in `scripts/02_build_fix_rules.py` (Python) and `scripts/bench/regex-rescore.mjs` in the [parakeet-ultra ONNX repo](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) (JavaScript). [parakeet_web](https://parakeetweb.olicorne.org/) applies them in the browser.

## Lookup form (`lookup/*.lookup.json`)

The patterns need lookbehind (word bounds, French elisions, the title guard), which some regex engines lack. A lookup file holds the same rules as plain data, so a consumer only walks characters and reads dictionaries. It is valid JSON with one rule per line (the two rules below are illustrations):

```json
{"format": "ultimed-fix-rules-lookup", "version": 1,
 "source": "drug_fix_rules.jsonl", "source_sha256": "...",
 "spec": {"word_chars": [[48, 57], [65, 90], ...], "separators": "\t\n ... -", "whitespace": "...",
          "apostrophes": "'’", "case_map": {"A": "a", ...}, "accent_map": {"é": "e", ...},
          "elision": "dlnjmtsc", "titles": ["docteur", "docteure", "dr", "dr.", "pr", ...]},
 "rules": [
  {"to": "mirtazapine", "keys": ["myrtazapine", "mire tazapine"]},
  {"to": "Humira", "keys": ["mirat"], "not_after": true},
  ...]}
```

`spec` spells out every character set the patterns relied on, computed by asking Python's `re` itself, so the consumer needs no Unicode tables: `word_chars` is a list of inclusive code point ranges, `case_map` folds case the way `re.IGNORECASE` does (46 entries, all Latin except the Kelvin sign), `accent_map` folds only the accents the rule classes list (so `é` becomes `e` but `ã` stays `ã`, because the patterns write `a` as `[aàâä]`).

A key is a variant folded the way the regex matched it: each text character goes through `case_map` then `accent_map`, one run of `separators` (whitespace and hyphens) between two words is one space, and an apostrophe (`'` or `’`) is `'` and may be followed in the text by whitespace the key does not show. For example, the texts `Myrtazapine`, `mire-tazapine` and `mire  tazapine` all fold to a key of the first rule above, and `d’ hexaméthasone` folds to `d'hexamethasone`.

Matching rules, which `scripts/apply_lookup.py` (no regex, under 300 lines with its docs and CLI) implements and is the reference for:

1. Files apply in order, and within a file rules apply one after the other, each over the whole text, so a rule sees the output of the rules before it.
2. A match starts on a word character that does not follow a word character, a hyphen or an apostrophe, except an apostrophe after a French elision (a lone `d l n j m t s c`, or `qu`, as in `jusqu'aménorée`). It must end where the text's word characters end (a key covers whole words), and not before a hyphen.
3. A key ending on an apostrophe swallows the whitespace after it, minus one space if a word character or hyphen follows.
4. Within one rule, matches are found left to right and never overlap. At one start position the rule's FIRST key (in its `keys` order) that matches wins, not the longest. The bounds always look at the text as it was before that rule ran.
5. A rule with `"not_after": true` does not match right after a title from `spec.titles` followed by one space (the surname guard of `"name": true`).
6. When the matched text starts with an uppercase letter and `to` starts with a lowercase one, the replacement's first letter is uppercased.

Only rules that match change anything, so a consumer can build one dictionary from every key to its `(rule index, key rank)` pairs, find all matches in one pass, apply the lowest-index matching rule, and repeat with the rules after it. That is what `apply_lookup.py` does:

```bash
uv run scripts/apply_lookup.py lookup/manual_fix_rules.lookup.json lookup/drug_fix_rules.lookup.json \
  --in hyps.txt --out fixed.txt   # one text per line, lookups in the order given
```

The lookup files are about 7 times smaller than the regex files, because they drop the patterns and the build statistics:

| file | rules | keys | `not_after` rules | regex file (bytes) | lookup file (bytes) |
|---|---|---|---|---|---|
| `manual_fix_rules` | 1,905 | 1,943 | 0 | 748,383 | 111,996 |
| `drug_fix_rules` | 2,388 | 4,854 | 23 | 1,119,211 | 146,485 |
| `dictionary_fix_rules` | 18,137 | 42,827 | 389 | 9,939,748 | 1,354,187 |
| `parhaf_fix_rules` | 2,103 | 3,960 | 36 | 1,037,024 | 130,236 |
| `parrot_fix_rules` | 121 | 154 | 0 | 56,212 | 8,607 |
| `acronyms_fix_rules` | 43 | 43 | 1 | 20,892 | 4,195 |

### How the lookups are built

`scripts/08_build_lookup.py rules/*.jsonl` writes `lookup/<name>.lookup.json`. Nothing is reinterpreted: each rule's pattern is first rebuilt from its variants with the same functions that wrote it (`variant_pattern` in 02, the merge scaffolding of 03) and must equal the stored pattern byte for byte, so the keys describe exactly what the regex matches. A rule that does not rebuild stops the build instead of being approximated; none of the 24,697 v2 rules fails. The lookup records the SHA-256 of its source, and `05_build_manual_rules.py` and `07_build_dataset_rules.sh` write the lookup form each time they write rules.

### Equivalence

`tests/test_lookup.py` checks that every lookup gives exactly the output of `apply_rules` on its regex file:

- the doctests, and that `lookup/` is up to date with `rules/` (rebuilt into a temporary dir and compared byte for byte);
- 54 hand-written edge texts (elisions, curly apostrophes, hyphens, whitespace runs, capitals, titles, `aujourd'hui`-like words) on a small rule set built for them;
- 3,000 random texts per file, made of the file's own variants (randomly upper-cased, accented, cut short or re-separated) glued by spaces, hyphens, apostrophes, elisions and titles;
- 299 real v2 transcripts (`tests/fixtures/hyps_sample.txt`: for each file, lines that file changes, plus 100 lines no file changes), or any corpus via `ULTIMED_LOOKUP_CORPUS` (a `.jsonl` with a `hyp` field, or a `.txt`) and `ULTIMED_LOOKUP_LIMIT`.

All pass with 0 mismatches (2026-11-15). On 20,093 v2 transcripts (every 30th line of the full corpus), the regex and lookup outputs were also identical for all six files (manual 175 lines changed, drug 479, dictionary 3,419, PARHAF 559, PARROT 35, acronyms 12). The reference applier is slower than the indexed regexes in Python (about 0.8 to 1.9 ms per line per file, vs 0.1 to 1.2 ms for `compile_rules`); it is written for clarity, not speed.

## Tests

```bash
uv run tests/test_lookup.py
cd tests && uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz --with wordfreq test_drug_asr_rules.py
```

`test_drug_asr_rules.py` skips the two checks that need the UltiMed-ASR-FR-v1-scripts checkout when it is missing.
