# UltiMed-ASR-FR-v2-regex-fixes

> [!WARNING]
> **Release in progress.** UltiMed v2 is being released right now. As long as this notice is here, files, links and numbers in this repository are still being changed.

Part of [UltiMed-Org](https://github.com/UltiMed-Org), which lists every UltiMed repository, dataset and model.

Cheap post-processing for French medical ASR: lists of rules that rewrite the misspellings the [UltiMed parakeet model](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-onnx) still makes on French medical vocabulary into the right word, for example `veine la faxine` becomes `venlafaxine`. Plain text substitution after decoding: no GPU, no extra model. They also help models that were not finetuned.

Each rule list ships in two equivalent forms: regexes (`rules/`) and lookup tables (`lookup/`), for apps whose regex engine has no lookbehind. Both give the same output on every text (see "Equivalence" below).

This repository, its scripts and its docs were written with Claude Code. The rules and their build history moved here on 2026-11-15 from the model repository (`regex-fixes/`) and from `08_drug_asr_rules/` of [UltiMed-ASR-FR-v2-scripts](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts).

License: AGPL-3.0-or-later for everything in this repository (rules, lookup tables, scripts, tests), see `LICENSE`. The build scripts were first published in UltiMed-ASR-FR-v1-scripts under Apache-2.0 and the v1 rule files in the model repository under CC BY 4.0; those earlier copies keep their original license, this repository is the same author's relicensed continuation.

## The rules

These are the **v2** rules, built on 2026-11-17 (rebuilt on 2026-11-18 against the corrected labels and re-mined on 2026-11-20 on the unified labels, see [scripts/README.md](scripts/README.md)) from the v2 model alone (the 70/30 blend of the "mix7" weight soup with the v1 weights, fp32 ONNX) transcribing every clip of the v2 UltiMed corpus ([Olicorne/UltiMed-ASR-FR-v2](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2), 602,776 clips). The rules are learned from the v2 labels, so the replacement spelling is the one the v2 labels use. That includes the casing: a name or acronym that the labels write the same way every time comes out the same way in the replacement (`Gleason`, `angio-IRM`), see `scripts/09_apply_resolutions.py`. Each subset's file was built by `scripts/07_build_dataset_rules.sh`; [scripts/README.md](scripts/README.md) has the full build log, the guards and the evaluation.

| file | what it fixes | variants | rules (merged) |
|---|---|---|---|
| `manual_fix_rules.jsonl` | hand-written fixes and decisions (`post-opératoire`, `scanner lodose` to `scanner low-dose`, ...), dataset-agnostic, built by `scripts/05_build_manual_rules.py` from `inputs/` | 2,156 | 2,059 |
| `drug_fix_rules.jsonl` | drug names, learned on the drugs subset | 5,057 | 2,447 |
| `dictionary_fix_rules.jsonl` | medical terms, learned on the dictionary subset, then filtered against French news 2024 | 43,575 | 18,261 |
| `parhaf_fix_rules.jsonl` | medical terms, learned on the PARHAF subset | 4,146 | 2,172 |
| `parrot_fix_rules.jsonl` | medical terms, learned on the PARROT subset (test only) | 147 | 117 |
| `acronyms_fix_rules.jsonl` | medical terms, learned on the acronyms subset | 43 | 43 |

`rules/confusion_groups.jsonl` is not a rule file: it lists the 905 heard forms the v2 model writes for two or more different words (`étuximab` for `cetuximab` or `rituximab`), which no regex can fix, with the candidates and their counts, for an acoustic second pass. No transcript or clip id is in it. It is tied to the model that produced the hypotheses and must be regenerated for each new model; see `scripts/10_build_confusion_groups.py` and [scripts/README.md](scripts/README.md). Script and docs written with Claude Code.

The four term files replace v1's single `term_fix_rules.jsonl`. They overlap: 1,192 variants appear in both the dictionary and PARHAF files, and 35 variants get a different replacement depending on the file (19 dictionary vs PARHAF, e.g. `trocarde` becomes `trocart de` in one and `trocart d'` in the other; 12 dictionary vs drugs; 3 PARHAF vs drugs; 1 dictionary vs PARROT). The author decided those conflicts (33 distinct variants, some in two pairs) by hand on 2026-11-15 (`inputs/conflict_resolutions.tsv`, some only in a context: `lodose` becomes `low-dose` after `scanner` but `Lodoz` elsewhere, `trocarde` becomes `trocart d'` before a vowel). `scripts/09_apply_resolutions.py` removes them from the learned files and the decisions ship as manual rules, so no variant gets two replacements any more. The order still matters for chained rewrites: apply the drug file first, then the term files, and the manual file LAST, so its spelling conventions (`post-opératoire` -> `postopératoire`, `bévacizumab` -> `bevacizumab`) also respell what a learned rule wrote. (It used to go first so that a hand fix would win, e.g. `lodose` -> `lowdose` over the learned `Lodoz`; that variant is now decided in `conflict_resolutions.tsv` and removed from the learned files.) The order used for every with-rules number published for v2 (the benchmark tables and the real-voice page of the model repository) is: drugs, dictionary, PARHAF, PARROT, acronyms, then manual. Since no variant gets two replacements any more, the order of the term files only matters for chained rewrites between them; other orders were not measured. TODO: merge them into one checked list as v1 did.

### Measured effect

The v2 model (fp32 ONNX, beam 5) on each subset's test split, scored against the corrected labels (2026-11-20, after the owner brand casing pass), WER without rules and then with all six files applied in the order drug, dictionary, PARHAF, PARROT, acronyms, manual (rules at e994c28; the rules at 8512d62 give the same scores):

| test split | words | no rules | v2 rules |
|---|---|---|---|
| drugs | 65,574 | 3.56 % | 2.15 % |
| drug sentences | 3,537 | 3.68 % | 2.83 % |
| dictionary | 239,173 | 2.42 % | 1.89 % |
| PARHAF | 204,212 | 2.74 % | 2.40 % |
| PARROT | 95,035 | 3.40 % | 3.14 % |
| acronyms | 3,435 | 2.85 % | 2.65 % |

**These with-rules numbers are optimistic.** The committed rules are learned on ALL splits, test included, on purpose: the list is meant to fix every mistake the model made on the dataset. So the table shows how many seen mistakes the rules fix, not how well they generalize. A control build learned on train+val only keeps 72 % of the gain on dictionary, 61 % on PARHAF, 61 % on drugs and 20 % on acronyms. The PARROT rules are learned on its test split only (PARROT has no other split), so its gain is entirely in-sample.

Overcorrection (2026-11-20, measured with Claude Code by the ultra repo's `evalrules.py`, rerun after the owner's decisions of the same day and again once those labels were relabelled; a line counts when the rules change it at all), per file: on the 602,776 v2 labels every file rewrites 0 lines; on 100,000 held-out French news sentences (2023) the dictionary file rewrites 14, PARHAF 4, drugs, PARROT and acronyms 0, manual 33; on 1,662 French VoxPopuli references, 0 for every file. All six files applied together, in the order above: 0 of 602,776 labels, 47 of 100,000 news sentences (0.05 %) and 0 of 1,662 VoxPopuli references. Before the relabelling it was 12 labels, all the manual file respelling label text the unification missed (`QuantiFERON` -> `Quantiferon`, `re-épithélialisation` -> `réépithélialisation`, `d'Ostéogenèse Imparfaite` -> `d'ostéogenèse imparfaite`, `anti-Thyroglobuline` -> `antithyroglobuline`, ...); the owner decided the rule is right for all 12 and the labels now carry those spellings. Before those decisions it was 19 labels: the 6 `d'Herpes gestationis` / `zoster` / `simplex` and the 1 `nitro-imidazole` are no longer touched. Two thirds of the news rewrites are those conventions too (`cardio-vasculaire` -> `cardiovasculaire`, `ultra-rapide` -> `ultrarapide`); the rest are learned rules hitting proper names (`Halutz` -> `Hallux`, `Enner` -> `Scanner`).

TODO: real speech. No v2 transcripts of FLEURS-fr, Common Voice fr or the author's own recordings existed when the rules were built, so their effect on real speech with the v2 model is not measured yet.

## Layout

| path | content |
|---|---|
| `rules/` | the six v2 rule files (regex form), and `confusion_groups.jsonl` (ambiguous heard forms, not rules) |
| `lookup/` | the same six files in lookup-table form, generated from `rules/` by `scripts/08_build_lookup.py` |
| `rejected/` | the candidates each v2 build rejected, with the reason, and `drug_asr_errors.json` (the misspellings the drug build started from); not published: they hold fragments of dataset transcripts and nothing reads them, so they stay local and gitignored |
| `inputs/` | the author's `conflict_resolutions.tsv` (variants the learned files disagreed on), the hand-written `manual_fixes.tsv` and the generated `convention_fixes.tsv`, source of `manual_fix_rules.jsonl` |
| `v1/` | the v1 outputs, kept for comparison: `drug_fix_rules.jsonl`, `term_fix_rules.jsonl` (dictionary + PARHAF + PARROT in one list) and the per-model `drug_asr_errors*.json` (their rejected lists stay local, like `rejected/`) |
| `scripts/` | the build chain (01 to 07 and 09, plus 10 for the confusion groups, see [scripts/README.md](scripts/README.md)), `08_build_lookup.py` and `apply_lookup.py` |
| `tests/` | `test_drug_asr_rules.py` (the build scripts) and `test_lookup.py` (regex vs lookup equivalence, on synthetic texts; no real transcript is committed) |

The build scripts read the corpus manifests, the drug lexicon and a few helpers from a [UltiMed-ASR-FR-v2-scripts](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts) checkout. By default they look for it next to this repository, in a folder still named `UltiMed-ASR-FR-v1-scripts`; set `ULTIMED_SCRIPTS_DIR` to point elsewhere. Applying the rules, building the lookups and `test_lookup.py` need nothing outside this repository.

## Regex form (`rules/*.jsonl`)

One JSON rule per line, applied top to bottom, file after file. For each rule:

- compile `pattern` case-insensitively (`new RegExp(pattern, 'giu')` in JavaScript, `re.IGNORECASE` in Python; the patterns only use syntax both engines share);
- replace every match with `replacement`;
- when the matched text starts with a capital and the replacement does not, capitalise the replacement's first letter.

A rule with `"before_vowel": true` (only in the manual file) ends on the whitespace before a word starting with a vowel or `h` instead of a word end, and its replacement eats that whitespace (`trocarde aspiration` becomes `trocart d'aspiration`). A rule with `"name": true` is also a French surname: its pattern already skips a match right after a title like `docteur`, so the surname in `docteur X` is left alone. Merged rules carry a `variants` list (one alternation per replacement); unmerged ones a single `variant` plus build statistics.

With tens of thousands of rules a naive loop over every rule for every sentence is slow: index each rule under a literal anchor of its variants and only try the rules whose anchor occurs in the text. Two implementations do this: `compile_rules` in `scripts/02_build_fix_rules.py` (Python) and `scripts/bench/regex-rescore.mjs` in the [parakeet-ultra ONNX repo](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) (JavaScript). [parakeet_web](https://parakeetweb.olicorne.org/) applies them in the browser.

## Lookup form (`lookup/*.lookup.json`)

The patterns need lookbehind (word bounds, French elisions, the title guard), which some regex engines lack. A lookup file holds the same rules as plain data, so a consumer only walks characters and reads dictionaries. It is valid JSON with one rule per line (the two rules below are illustrations):

```json
{"format": "ultimed-fix-rules-lookup", "version": 2,
 "source": "drug_fix_rules.jsonl", "source_sha256": "...",
 "spec": {"word_chars": [[48, 57], [65, 90], ...], "separators": "\t\n ... -", "whitespace": "...",
          "apostrophes": "'’", "case_map": {"A": "a", ...}, "accent_map": {"é": "e", ...},
          "elision": "dlnjmtsc", "vowels": "AEHIOUYaehiouyÀÂ...", "titles": ["docteur", "docteure", "dr", "dr.", "pr", ...]},
 "rules": [
  {"to": "mirtazapine", "keys": ["myrtazapine", "mire tazapine"]},
  {"to": "Humira", "keys": ["mirat"], "not_after": true},
  ...]}
```

`spec` spells out every character set the patterns relied on, computed by asking Python's `re` itself (case-insensitive extras only among the code points of Unicode 3.2, so the file is the same whatever Python builds it; see `scripts/README.md`), so the consumer needs no Unicode tables: `word_chars` is a list of inclusive code point ranges, `case_map` folds case the way `re.IGNORECASE` does (46 entries, all Latin except the Kelvin sign), `accent_map` folds only the accents the rule classes list (so `é` becomes `e` but `ã` stays `ã`, because the patterns write `a` as `[aàâä]`).

A key is a variant folded the way the regex matched it: each text character goes through `case_map` then `accent_map`, one run of `separators` (whitespace and hyphens) between two words is one space, and an apostrophe (`'` or `’`) is `'` and may be followed in the text by whitespace the key does not show. For example, the texts `Myrtazapine`, `mire-tazapine` and `mire  tazapine` all fold to a key of the first rule above, and `d’ hexaméthasone` folds to `d'hexamethasone`.

Matching rules, which `scripts/apply_lookup.py` (no regex, under 300 lines with its docs and CLI) implements and is the reference for:

1. Files apply in order, and within a file rules apply one after the other, each over the whole text, so a rule sees the output of the rules before it.
2. A match starts on a word character that does not follow a word character, a hyphen or an apostrophe, except an apostrophe after a French elision (a lone `d l n j m t s c`, or `qu`, as in `jusqu'aménorée`). It must end where the text's word characters end (a key covers whole words), and not before a hyphen.
3. A key ending on an apostrophe swallows the whitespace after it, minus one space if a word character or hyphen follows.
4. Within one rule, matches are found left to right and never overlap. At one start position the rule's FIRST key (in its `keys` order) that matches wins, not the longest. The bounds always look at the text as it was before that rule ran.
5. A rule with `"not_after": true` does not match right after a title from `spec.titles` followed by one space (the surname guard of `"name": true`).
6. A rule with `"before_vowel": true` (version 2; version 1 files had none) matches only when whitespace then a character of `spec.vowels` follow the key, and the match runs up to that character, the whitespace included, so `to` (ending on an apostrophe) is glued to the next word.
7. When the matched text starts with an uppercase letter and `to` starts with a lowercase one, the replacement's first letter is uppercased.

Only rules that match change anything, so a consumer can build one dictionary from every key to its `(rule index, key rank)` pairs, find all matches in one pass, apply the lowest-index matching rule, and repeat with the rules after it. That is what `apply_lookup.py` does:

```bash
uv run scripts/apply_lookup.py lookup/drug_fix_rules.lookup.json lookup/manual_fix_rules.lookup.json \
  --in hyps.txt --out fixed.txt   # one text per line, lookups in the order given
```

The lookup files are about 7 times smaller than the regex files, because they drop the patterns and the build statistics:

| file | rules | keys | `not_after` rules | regex file (bytes) | lookup file (bytes) |
|---|---|---|---|---|---|
| `manual_fix_rules` | 2,059 | 2,156 | 0 | 810,058 | 121,389 |
| `drug_fix_rules` | 2,447 | 5,057 | 20 | 1,153,644 | 152,505 |
| `dictionary_fix_rules` | 18,261 | 43,575 | 396 | 10,079,406 | 1,376,358 |
| `parhaf_fix_rules` | 2,172 | 4,146 | 35 | 1,076,131 | 136,574 |
| `parrot_fix_rules` | 117 | 147 | 1 | 55,413 | 8,581 |
| `acronyms_fix_rules` | 43 | 43 | 0 | 19,868 | 4,374 |

### How the lookups are built

`uv run scripts/08_build_lookup.py rules/*_fix_rules.jsonl` writes `lookup/<name>.lookup.json` for the six rule files. Do not pass `rules/*.jsonl`: that glob also catches `confusion_groups.jsonl`, which is not a rule file and stops the build with `KeyError: 'variant'`. Nothing is reinterpreted: each rule's pattern is first rebuilt from its variants with the same functions that wrote it (`variant_pattern` in 02, the merge scaffolding of 03) and must equal the stored pattern byte for byte, so the keys describe exactly what the regex matches. A rule that does not rebuild stops the build instead of being approximated; none of the 25,099 v2 rules fails. The lookup records the SHA-256 of its source, and `05_build_manual_rules.py` and `07_build_dataset_rules.sh` write the lookup form each time they write rules.

### Equivalence

`tests/test_lookup.py` checks that every lookup gives exactly the output of `apply_rules` on its regex file:

- the doctests, and that `lookup/` is up to date with `rules/` (rebuilt into a temporary dir and compared byte for byte);
- 67 hand-written edge texts (elisions, curly apostrophes, hyphens, whitespace runs, capitals, titles, `aujourd'hui`-like words) on a small rule set built for them;
- 3,000 random texts per file, made of the file's own variants (randomly upper-cased, accented, cut short or re-separated) glued by spaces, hyphens, apostrophes, elisions and titles;
- a synthetic corpus per file, generated at test time with a fixed seed: the file's own variants (a quarter of them mangled as above) placed in 15 invented French carrier sentences, at the start and end of the text, after an elision (`l'`, `d’`, `qu'`), next to punctuation, capitalised, after a title, and two side by side. Each file gets as many sentences as it has variants, at least 3,000 and at most 10,000 (manual 3,000, drug 5,057, dictionary 10,000, PARHAF 4,146, PARROT 3,000, acronyms 3,000), so every variant of every file but the dictionary is used at least once (possibly mangled). No real transcript is committed: to check real ones, point `ULTIMED_LOOKUP_CORPUS` at a corpus (a `.jsonl` with a `hyp` field, or a `.txt` with one text per line), optionally with `ULTIMED_LOOKUP_LIMIT`, and every file runs on it.

All pass with 0 mismatches (2026-11-15). On 20,093 v2 transcripts (every 30th line of the full corpus), the regex and lookup outputs were also identical for all six files (checked on the files as they were before the 2026-11-15 conflict decisions and convention respelling, not rerun since; manual 175 lines changed, drug 479, dictionary 3,419, PARHAF 559, PARROT 35, acronyms 12). The reference applier is slower than the indexed regexes in Python (about 0.8 to 1.9 ms per line per file, vs 0.1 to 1.2 ms for `compile_rules`); it is written for clarity, not speed.

## Tests

```bash
uv run tests/test_lookup.py
cd tests && uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz --with wordfreq test_drug_asr_rules.py
```

`test_drug_asr_rules.py` skips the two checks that need the UltiMed-ASR-FR-v2-scripts checkout when it is missing.
