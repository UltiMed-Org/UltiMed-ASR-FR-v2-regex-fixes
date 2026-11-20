# Build scripts: regex fixes for words the ASR model misspells

This is the build history of the rule files, moved here from `08_drug_asr_rules/` of [UltiMed-ASR-FR-v1-scripts](https://github.com/thiswillbeyourgithub/UltiMed-ASR-FR-v1-scripts) on 2026-11-15. Paths below are relative to the root of this repo: the scripts are in `scripts/`, the v2 rule files in `rules/` (their rejected lists and `drug_asr_errors.json` go to `rejected/`, which is gitignored: they hold fragments of dataset transcripts and nothing reads them), the hand-written TSVs in `inputs/`, and the v1 outputs (`drug_fix_rules.jsonl`, `term_fix_rules.jsonl`, the `drug_asr_errors*.json` files) in `v1/`. Paths starting with `99_hf_release/` or `01_dictionnary/` are inside a UltiMed-ASR-FR-v1-scripts checkout, which the scripts expect next to this repo or at `$ULTIMED_SCRIPTS_DIR`. `08_build_lookup.py` (the lookup-table form of the rules) is new here and described in the top-level README.

Goal: transcribe every clip of the UltiMed `drugs` subset with the fine-tuned Parakeet, collect how each drug name gets misspelled (`"mirtazapine": {"myrtazapine": 15, "mire tazapine": 6}`), and turn that into an ordered list of regex rules that fix drug names in ASR output while rarely overcorrecting.

Written with Claude Code.

## 1. Transcribe (NeMo repo, GPU)

Run it after training, when the GPU is free: the trainer uses about 22 GB of the 24 GB card, and CPU inference takes about 3.5 s per clip in fp32 (about 21 hours for the 21.9k drug clips). From the NeMo repo root (`UltiMed-ASR-FR-v1-NeMo_training_scripts`):

```bash
BS=16 ./.venv/bin/python perso/transcribe_manifests.py <final.ckpt or .nemo> drugs_hyps.jsonl \
    ../UltiMed-ASR-FR-v1-scripts/99_hf_release/data/NeMO_files/drugs/{train,val,test}.jsonl
```

It is resumable (rerun the same command after an interruption) and writes `{"audio": <abs path>, "hyp": <text>}` per line.

Caveat: the model was trained on the train split, so its train-clip errors are underestimated. val and test give the honest picture, and train mostly adds more occurrences of the same drugs.

## 2. Collect the misspellings

```bash
uv run scripts/01_extract_drug_errors.py drugs_hyps.jsonl \
    99_hf_release/data/NeMO_files/drugs/{train,val,test}.jsonl
```

This writes `v1/drug_asr_errors.json`, most errors first. Any manifest works (the oli or PARHAF sets too): every drug word in a label counts, whatever the source. The script docstring details how drug words are spotted and how a hypothesis stretch is attributed to one drug.

## 3. Build the rules

```bash
uv run scripts/02_build_fix_rules.py drugs_hyps.jsonl
```

This writes `v1/drug_fix_rules.jsonl` (one rule per line, apply top to bottom) and `drug_fix_rules.rejected.jsonl` (with the reason, for review). Scanning the 600k corpus labels for `real_text` takes about 1.5 min. The guards (`short`, `contains_target`, `other_term` (called `other_drug` in the earlier builds below), `real_text`, `trailing_a` (the variant is real text plus a lone `à`: `patelle à` -> `patella` would turn "la patelle à répétition" into "la patella répétition"), `french_word`, `ambiguous`, `imprecise`, `rare`, `far`, `common_words`) and their thresholds are in the script docstring and `--help`. Both scripts take a repeated `--lexicon <file.jsonl>` (any JSONL with a `term` field, e.g. `01_dictionnary/original_dictionnary.jsonl`): the words of its terms rarer in French than `--max-term-zipf` (default 3.0, wordfreq Zipf) are tracked like drug names, so the same pipeline mines medical-term misspellings on any subset. With the dictionary the lexicon grows from 2,999 drug words to 32,094 ("syndrome de Brugada" adds `brugada`, not `syndrome`). The acronym list adds nothing: no acronym has 5 letters or more without a digit. Several models can feed one rule set: give each model's hyps file as HYPS and its 01 report with a repeated `--errors`, and `merge_reports` sums them. The `rare` and `common_words` thresholds count distinct clips (the most any single model wrote the variant), not the sum: every model transcribes the same audio, so the same slip from several models on one clip is one clip of evidence.

Apply the rules with `fix = compile_rules(rules)` then `fix(text)` from `02_build_fix_rules.py` (`load_rules(path)` reads the JSONL; it is `_util.read_jsonl` under its old name). `compile_rules` indexes each rule by its anchor, the longest accent-folded word of its variant, so a text only runs the few rules whose anchor it contains: same output as `apply_rules(text, rules)` (every rule in order, the reference implementation) but about 150x faster (0.11 ms against 16.9 ms per text, measured with 1.6k rules). You can also copy the patterns into any regex engine that supports lookbehind. Each pattern is case-insensitive: compile it with the `i` flag, plus `u` in JavaScript (`new RegExp(pattern, 'giu')`), and when the matched text starts with a capital and the replacement does not, capitalise the replacement (`Myrtazapine` -> `Mirtazapine`). The word bounds spell out their letter class instead of using `\w`, so the patterns behave the same in Python and JavaScript. `scripts/bench/regex-rescore.mjs` in the [parakeet-tdt-0.6b-v3-ultra-onnx](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx) repo is a JavaScript port, anchor index included (0 mismatches against Python on 41,743 transcripts).

## v2 rules (`rules/`, 2026-11-17, one list per subset)

Rebuilt with Claude Code for the v2 release, from the v2 model alone (the 70/30 blend of mix7 with v1, fp32 ONNX; first built from mix7 alone on 2026-11-13, rebuilt 2026-11-17) transcribing all 602,776 clips of the v2 corpus, and split by subset: `rules/drug_fix_rules.jsonl`, `rules/dictionary_fix_rules.jsonl`, `rules/parhaf_fix_rules.jsonl`, `rules/parrot_fix_rules.jsonl`, `rules/acronyms_fix_rules.jsonl`, each with its `*.rejected.jsonl` in `rejected/` (plus `rejected/drug_asr_errors.json`; local only, not in the repo). Same JSONL format as the v1 files, so they are drop-in. The v1 files above are unchanged. `manual_fix_rules.jsonl` is dataset-agnostic and is not rebuilt from transcripts (05 builds it from `inputs/`): apply it LAST, after the drug and term files (it went first until 2026-11-15, see "Hand-written rules").

Every replacement comes from the v2 labels (02 copies the label text a rule was learned from), so the v1 caveat about old spellings no longer applies. Note the v2 labels themselves carry both spellings of some drugs (`bevacizumab` 363 vs `bévacizumab` 151, `perindopril` 1,055 vs `périndopril` 1,672 in `full.jsonl`), so the accent a rule writes depends on the clip it was learned from.

`07_build_dataset_rules.sh <hyps.jsonl> <out dir> <subset> <split>...` runs the whole chain on one subset: 01 `--hyps-out` keeps the hyps of that subset's clips, 02 (real-text labels = the release-wide manifests of the same splits, the 1M French Wikipedia sentences, `--names`), 03 merge when `MERGE_TEXTS` is set, 04 (`<unk>` cleanup, plus `--guard` when `NEWS_GUARD` is set), 09 (the author's conflict decisions, see below), 08 (the lookup form). The drug list uses no lexicon; the four term lists use `LEXICON=01_dictionnary/original_dictionnary.jsonl` and `NEWS_GUARD` = Leipzig `fr-news-2024-1M.txt`, as v1's `term_fix_rules.jsonl`. The merge (release build only) is checked on the full v2 hyps, `full.jsonl`, French and English Wikipedia and the held-out 2023 news.

```bash
D=99_hf_release/data/NeMO_files GUARDS=.../fr-wikipedia-1M.txt NAMES=.../fr-names.txt \
  MERGE_TEXTS="full.blend70-onnx-fp32.hyp.jsonl $D/full.jsonl .../fr-wikipedia-1M.txt .../en-wikipedia-1M.txt .../fr-news-2023-100K.txt" \
  scripts/07_build_dataset_rules.sh full.blend70-onnx-fp32.hyp.jsonl out/A/drugs drugs train val test
```

Two builds per subset, as in v1:

- **A, release (committed)**: all splits. Its test numbers show coverage, not generalization: the test clips were in the build.
- **B, held-out (not committed)**: train + val only, with the real-text labels limited to train + val too, and no merge. Its test numbers estimate how the rules do on unseen clips. PARROT is test-only in v2, so it has no B build and no held-out number. For drugs a train-only build (T) was also made.

History, build counts of the blend70 rebuild (2026-11-17, the current files are described in "Re-mine on the unified labels" below; measured with Claude Code from the `build.log` of each build; 02 then 03 merge then 04; candidate errors = misspelled occurrences found by 01; "final" is before 09 removes the hand-decided variants, see "Conflict decisions"):

| subset | clips | misspelled | 02 rules (errors covered) | merged | final | rejected |
|---|---|---|---|---|---|---|
| drugs A | 21,900 | 10,500 | 5,000 (9,293) | 2,429 | 2,429 | 652: real_text 227, far 121, short 120, common_words 104, imprecise 48, other_term 22, ambiguous 8, contains_target 1, trailing_a 1 |
| drugs B | 19,841 | 9,506 | 4,689 (8,410) | | 4,689 | 602 |
| drugs T | 17,822 | 8,518 | 4,373 (7,541) | | 4,373 | 550 |
| dictionary A | 534,349 | 194,104 | 43,823 (116,880) | 18,396 | 18,287 | 14,509: real_text 5,589, other_term 2,189, common_words 1,853, imprecise 1,752, short 1,544, far 1,010, ambiguous 422, contains_target 142, trailing_a 5, french_word 3 |
| dictionary B | 484,458 | 176,116 | 41,300 (106,296) | | 41,090 | 13,734 |
| PARHAF A | 43,384 | 17,660 | 4,177 (9,886) | 2,193 | 2,184 | 2,141: real_text 902, other_term 612, short 210, common_words 139, imprecise 133, far 108, ambiguous 20, contains_target 13, trailing_a 3, french_word 1 |
| PARHAF B | 37,895 | 15,395 | 3,767 (8,587) | | 3,747 | 1,970 |
| PARROT A | 1,548 | 699 | 148 (216) | 118 | 118 | 220: other_term 102, real_text 98, common_words 9, far 5, short 4, imprecise 2 |
| acronyms A | 1,595 | 101 | 43 (44) | 43 | 43 | 51: other_term 20, real_text 19, common_words 6, short 5, contains_target 1 |
| acronyms B | 1,444 | 89 | 40 (40) | | 40 | 44 |

History: the first build, from mix7 alone (2026-11-13), had 2,388 drug, 18,137 dictionary, 2,103 PARHAF, 121 PARROT and 43 acronym rules after 04. The blend70 model makes more misspellings on the same clips (drugs 10,500 vs 9,782), hence slightly more rules.

Rebuild on the fixed labels (2026-11-18, Claude Code): on 2026-11-17 the labels were corrected (drug names without accents, e.g. `Nalbuphine`; eponyms capitalised, e.g. `Mounier-Kuhn`; 329 clips re-synthesised with rewritten text; the 67 split manifests re-synced). Build A was rerun with the same command and the same blend70 hyps, minus the 329 re-synthesised clips, whose transcripts predate their new audio (328 have audio newer than the transcription, and the 329th, `054944_0001`, kept its old audio but is in the rewritten set, so it is excluded too). 01 skips a manifest row with no hyp, so those clips simply drop out. Result after 09, against the files committed on 2026-11-17: drugs 2,424 rules (4,989 variants, was 4,986), dictionary 18,273 (was 18,274), PARHAF 2,170 (was 2,171, 4,124 variants, was 4,128), PARROT 117 and acronyms 43 byte-identical, manual unchanged (05 does not read the labels). 76 dictionary targets and 1 PARHAF target changed, all capitalising an eponym: `Mounier-kuhn` becomes `Mounier-Kuhn`, likewise `Gomori-Grocott`, `Lich-Gregoir`, `Chédiak-Higashi`, `Malpighi-Gartner`, `Boucher-Neuhäuser`, `Gerstmann-Sträussler-Scheinker`. No drug target changed: 09 had already respelled and recased them by the label conventions. No new cross-file conflict appeared.

Re-mine on the unified labels (2026-11-20, Claude Code): the numbers in this paragraph describe the rule files committed now; the tables above and below are kept as history. Two things had changed since the 2026-11-18 rebuild. First, 01 was fixed to keep hyphenated drug names with a one-letter tail as one target (`Diffu-K`; before, `Diffu K` or `Diffu` were learned, see "Hyphenated drug names" below). Second, the labels were unified on headword spellings (e.g. `Kussmaul`, `Tornwaldt`, `colestyramine`, `Castleman`, `desferrioxamine`, `lipocaline`): 286 labels were fixed silently and 182 clips redrawn, and in total 524 clips had audio newer than their transcript. Those 524 clips were re-transcribed with the blend70 fp32 ONNX model, which gives fresh hyps for all 602,776 clips, and build A (all splits) and build B (train+val) were rerun with `07_build_dataset_rules.sh`, same command as before.

Result after 09, against the files committed before: drugs 2,447 rules / 5,057 variants (was 2,424 / 4,989; 68 new variants, 0 lost, all hyphenated names such as `Diffu-K`, `Solu-Médrol`, `Néo-Codion`, `Gyno-Pévaryl`, `Chrono-Indocid`); dictionary 18,261 rules / 43,575 variants (was 18,272 / 43,575; 47 variants lost and 47 gained, 15 targets changed to follow the unified spellings); PARHAF 2,173 rules / 4,147 variants (was 2,169 / 4,123; 25 new, 1 lost, 3 targets changed to `Diffu-K`); PARROT (117) and acronyms (43) byte-identical; manual (2,050) unchanged. No cross-file conflict. About ten of the lost dictionary variants were still-valid fixes (e.g. `pylonéphrite` to `pyélonéphrite`, `dysknée` to `dyspnée`) whose only source clip was redrawn and is now transcribed correctly.

Build stages for A (02 rules / merged / before 09 / final / rejected): drugs 5,071 / 2,452 / 2,452 / 2,447 / 659 (10,638 misspelled occurrences in 21,900 clips); dictionary 43,823 / 18,384 / 18,275 / 18,261 / 14,470; PARHAF 4,197 / 2,196 / 2,187 / 2,173 / 2,145; PARROT 148 / 118 / 118 / 117 / 220; acronyms 43 / 43 / 43 / 43 / 51. B final: drugs 4,739, dictionary 41,060, PARHAF 3,735, acronyms 40.

Evaluation with `evalrules.py` on the fresh hyps, test splits, WER (clips better / worse), with the rules as committed (after 09):

| test split | no rules | A (coverage) | B (held-out) |
|---|---|---|---|
| drugs (2,059) | 3.53 % | 2.03 % (838 / 0) | 2.62 % (533 / 0) |
| dictionary (49,891) | 2.54 % | 2.00 % (9,328 / 2) | 2.15 % (7,124 / 15) |
| PARHAF (5,489) | 2.85 % | 2.52 % (986 / 0) | 2.65 % (665 / 2) |
| PARROT (1,548) | 3.64 % | 3.34 % (186 / 0) | none |
| acronyms (151) | 2.96 % | 2.81 % (4 / 0) | 2.93 % (1 / 0) |

Correct v2 labels rewritten (of 602,776): A 0 for every subset; B drugs 0, dictionary 31, PARHAF 2, acronyms 0. The French news, English Wikipedia and VoxPopuli checks were NOT rerun this time; the "rewritten lines" table below still shows the 2026-11-17 files.

For comparison v1 (five models, all splits, so not like for like): `drug_fix_rules.jsonl` 3,963 rules, 2,732 rejected; `term_fix_rules.jsonl` (dictionary + PARHAF + PARROT in one list) 35,876 rules, 25,466 rejected.

Check with the ultra repo's `evalrules.py` (unchanged): the v2 model's hyps on each subset's test split, WER without rules then with each rule set (clips better / worse), and the lines each set rewrites among the 602,776 correct v2 labels, 100k held-out French news (2023), 1M English Wikipedia sentences and the 1,662 VoxPopuli fr references. v1 = the committed `drug_fix_rules.jsonl` for drugs, `term_fix_rules.jsonl` for the others.

| test split | no rules | v1 | A (coverage) | B (held-out) | T (train only) |
|---|---|---|---|---|---|
| drugs (2,059 clips) | 3.53 % | 2.67 % (496 / 1) | 2.05 % (827 / 0) | 2.63 % (529 / 0) | 2.67 % (508 / 0) |
| dictionary (49,891) | 2.54 % | 2.22 % (6,341 / 674) | 2.00 % (9,298 / 2) | 2.15 % (7,091 / 15) | |
| PARHAF (5,489) | 2.85 % | 2.67 % (616 / 7) | 2.53 % (984 / 0) | 2.65 % (664 / 2) | |
| PARROT (1,548) | 3.64 % | 3.50 % (91 / 9) | 3.34 % (186 / 0) | none (test-only) | |
| acronyms (151) | 2.96 % | 2.84 % (3 / 0) | 2.81 % (4 / 0) | 2.93 % (1 / 0) | |

These are the blend70 numbers (2026-11-17, measured with Claude Code), kept as history: the committed files are evaluated in "Re-mine on the unified labels" above. The A column was evaluated on build A as learned, before 09 removed the hand-decided variants (2,424 drug, 18,279 dictionary, 2,177 PARHAF, 117 PARROT, 43 acronym rules); B had 4,681 / 41,072 / 3,732 / 40 rules and T 4,365. History: on the mix7 hyps the same table read 3.14 % -> 1.77 % (A) for drugs, 2.39 % -> 1.89 % for dictionary, 2.76 % -> 2.47 % for PARHAF, 3.68 % -> 3.37 % for PARROT and 2.90 % -> 2.72 % for acronyms.

| rewritten lines | v2 labels | fr news 2023 | en Wikipedia | VoxPopuli fr |
|---|---|---|---|---|
| v1 drug rules | 28 | 2 | 1,097 | 0 |
| drugs A / B / T | 0 / 1 / 4 | 0 / 0 / 0 | 278 / 214 / 215 | 0 / 0 / 0 |
| v1 term rules | 9,496 | 24 | 9,413 | 1 |
| dictionary A / B | 0 / 31 | 14 / 13 | 3,606 / 3,597 | 0 / 0 |
| PARHAF A / B | 0 / 2 | 4 / 4 | 165 / 180 | 0 / 0 |
| PARROT A | 0 | 0 | 1 | 0 |
| acronyms A / B | 0 / 0 | 0 / 0 | 1 / 1 | 0 / 0 |

The v1 term rules rewrite 9,496 correct v2 labels: they carry v1 spellings onto v2 text (`acétoacétate` -> `acéto-acétate`), which is why they had to be rebuilt. English Wikipedia is still rewritten by the drug rules (`tomato` -> `stomato`, `insulator` -> `Insulatard`, `invents` -> `Invanz`): the rules are for French. Held-out fixes look like `Vinclixto` -> `Venclyxto`, `Pegasis` -> `Pegasys`, `céfadroxyl` -> `céfadroxil`, `Rifina` -> `Rifinah`.

Real speech: no v2 transcripts of FLEURS or Common Voice exist yet (that needs the GPU), so the drug rules were only run on the v1 models' hyps (UltiMed and ultra-w4a8, `local/realspeech-ignore_backups/`). v1, A and B change 0 of 1,665 FLEURS-fr clips and 0 of 5,000 unseen CV-fr clips. On the author's 211 own-voice clips they only improve (UltiMed 10.07 % -> v1 9.06 %, A 9.31 %, B 9.34 %, 0 worse); v1 does better there because it was learned from those same v1 models' errors. TODO: rerun this check on v2 hyps of FLEURS-fr / CV-fr / own-voice once the GPU is free.

## Committed rules (2026-10-01, all splits)

The committed `drug_fix_rules.jsonl` (14,377 rules) is learned from ALL splits of the drugs subset (train, val AND test) and from four models: `drug_asr_errors.json` is the released UltiMed model (run 1.7.0, steps 13255 and 14255 averaged), `drug_asr_errors.ultra.json` the base parakeet-ultra (fp32 ONNX), `drug_asr_errors.ultra-int8.json` / `drug_asr_errors.ultra-w4a8.json` its int8 and w4a8 ONNX builds (transcribed on CUDA with onnx-asr; the w4a8 encoder runs with the int8 decoder).

Why test is included: the rules are not a model, they are a list of fixes for every drug-name mistake the models make on this dataset. The claim they support is "fixes every seen mistake without overcorrection", so the test drug numbers below show coverage, not generalization (the held-out train+val build further down shows how well such rules generalize to unseen clips). Overcorrection is measured where it can be: on every correct label of every subset and split, and on texts the rules never saw.

The build is scripted in the parakeet-ultra ONNX model repo (`scripts/drug-rules/`, written with Claude Code): `transcribe-drugs.sh` transcribes the drugs subset with one ONNX variant, `build-rules.sh` runs 01 and 02 on each model and then both checks (`evalrules.py` and `compare-rulesets.sh`):

```bash
# from the parakeet-tdt-0.6b-v3-ultra-onnx repo; each dir holds a drugs_hyps.jsonl
TEXTS="voxpopuli-fr=local/voxpopuli-ignore_backups/fr-validation.txt voxpopuli-en=local/voxpopuli-ignore_backups/en-validation.txt" \
  bash scripts/drug-rules/build-rules.sh <out dir> UltiMed=<dir> ultra-fp32=<dir> ultra-int8=<dir> ultra-w4a8=<dir>
```

Results (test drug clips: WER, then clips better / worse; corpora: texts the rules change, any change is an overcorrection):

| rules | rules count | UltiMed | ultra fp32 | ultra int8 | ultra w4a8 | correct labels changed (all splits) | VoxPopuli fr | VoxPopuli en | fr news (held out) | fr Wikipedia (guard) | en Wikipedia |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 3.69 % | 10.79 % | 11.00 % | 11.59 % | | | | | | |
| previous (train+val, R4) | 13,626 | 2.63 % (584 / 0) | 9.12 % (811 / 0) | 9.35 % (802 / 0) | 9.96 % (795 / 0) | 40 of 602,813 | 6 of 1,662 | 20 of 1,695 | | | |
| previous (all splits, before the elision fix) | 14,377 | 2.04 % (864 / 0) | 8.28 % (1152 / 0) | 8.43 % (1149 / 0) | 9.04 % (1156 / 0) | 0 of 602,813 | 0 of 1,662 | 5 of 1,695 | | | |
| previous (all splits, elision-aware) | 14,377 | 1.86 % (957 / 0) | 8.03 % (1259 / 0) | 8.19 % (1257 / 0) | 8.81 % (1261 / 0) | 0 of 602,813 | 0 of 1,662 | 5 of 1,695 | 14 of 100,000 | 242 of 1,000,000 | 4,935 of 1,000,000 |
| committed (all splits, elision-aware, Wikipedia guard, name titles) | 13,831 | 1.91 % (939 / 0) | 8.12 % (1230 / 0) | 8.28 % (1226 / 0) | 8.92 % (1226 / 0) | 0 of 602,813 | 0 of 1,662 | 0 of 1,695 | 1 of 100,000 | 0 of 1,000,000 | 756 of 1,000,000 |

- The committed build adds French Wikipedia as correct text (2026-10-03, with Claude Code): the 1M-sentence [Leipzig](https://wortschatz.uni-leipzig.de/en/download) `fra_wikipedia_2021_1M` file goes to 02 as an extra `--labels` (a `.txt` file counts one sentence per line; `GUARDS` in the ultra repo's `build-rules.sh`), so a variant that occurs in it is rejected like one found in a corpus label. Leipzig's `fra_news_2023_100K` stays held out to measure what the guard misses (1 sentence left: `l’attaquante` -> `l’Atacand`). Cost: 546 fewer rules and about +0.05 WER on test_drugs (boosted UltiMed int8 2.21 % -> 2.26 %); FLEURS en goes back to its no-rules WER. The same build fixed the tokenizer, which split `l’antivirus` (typographic apostrophe) differently from `l'antivirus`, so a correct text spelled with `’` never blocked a variant spelled with `'`.
- The committed build also keeps names after a title (2026-10-03, with Claude Code). A stress test put every INSEE surname and first name with at least 100 bearers (96,264 + 13,464) in templates like `le docteur X a vu le patient` and found 16 names rewritten by the rules, e.g. `le docteur Mirat` -> `le docteur Humira`, `Heral` -> `Esperal`, `Felden` -> `Feldene`. Dropping those rules would lose real fixes (`mirat` is a genuine ASR spelling of Humira), so 02 `--names` (fed the 247,728 names of `insee-names.py` in the ultra repo) keeps them and only prefixes their pattern with negative lookbehinds for the titles (docteur, Dr, Pr, professeur, M., monsieur, Mme, madame, Mlle...): `sous mirat depuis mai` still becomes `sous Humira`, `M. Mirat` stays. 48 rules carry it (`"name": true`); every number in the table is unchanged, and the name hits drop from 16 to 3, all first names written without a title (`Armonie` -> `Harvoni`, `Kasim` -> `Quasym`, `Lillou` -> `Leeloo`).
- The 5 English VoxPopuli changes of the previous builds (`artificial` -> `artificielles` x3, `strength` -> `Strensiq`, `Cuban` -> `Kuvan`) are expected: the rules are meant for French output only, which is how the benchmarks apply them.
- On boosted UltiMed browser transcripts, test_drugs goes 3.56 % (no rules) -> 2.70 % (R4) -> 2.32 % (before the elision fix) -> 2.19 % (elision-aware) -> 2.24 % (committed, Wikipedia guard) with the fp32 build, and 3.56 % -> 2.34 % -> 2.21 % -> 2.26 % with int8; FLEURS fr is unchanged and FLEURS en moves by at most 0.05 WER. Over the 48 benchmark cell and set pairs re-scored (`compare-rulesets.sh`), the elision fix beat the rules before it on 19 and lost on none; the Wikipedia guard then wins 6 (FLEURS en, back to its no-rules WER) and loses 6 (test_drugs, +0.02 to +0.09).
- The committed build fixes the rule start bound for elided words (2026-10-02, with Claude Code). A pattern used to refuse to start right after an apostrophe, so `12 semaines d'aménorée`, `L’aménorée` or `jusqu'aménorée` were never fixed; it now starts after a French elision (`d' l' n' j' m' t' s' c' qu'`) and still not inside a word like `aujourd'hui`. Same rule count, more matches: the gain is the row above.
- Two 02 fixes came with this build. Words are keyed with hyphens split, like `variant_pattern` matches them, so `sous antidote` in a label now blocks the `sous-antidote` variant (`real_text`; before, it rewrote the label). A one-word variant that is a common French word (`zipf_frequency >= --max-word-zipf`, default 2.5, e.g. `tienne`) is rejected as `french_word`: such words never appeared in a medical label, but VoxPopuli caught them.
- Rejected (in `drug_fix_rules.rejected.jsonl`): common_words 769, far 698, short 525, real_text 426, imprecise 262, other_term 24, ambiguous 22, contains_target 3, french_word 3. Wikipedia moves many rejections to `real_text` and `common_words` (one-off variants made of words Wikipedia uses), which is why `french_word` falls from 38 to 3.

### Earlier train+val build (R4), held out

Before the all-splits decision the rules were built on train+val only, keeping the test split as an honest check. Those numbers are the generalization estimate:

```bash
# per model: drugs_hyps.jsonl from step 1, drugs_hyps.trainval.jsonl = the same minus the test clips
uv run scripts/01_extract_drug_errors.py drugs_hyps.trainval.jsonl \
    99_hf_release/data/NeMO_files/drugs/{train,val}.jsonl --out <drug_asr_errors.json or drug_asr_errors.ultra.json>
uv run scripts/02_build_fix_rules.py {ultimed,ultra,ultra-int8,ultra-w4a8}/drugs_hyps.trainval.jsonl \
    --errors v1/drug_asr_errors.json --errors v1/drug_asr_errors.ultra.json \
    --errors v1/drug_asr_errors.ultra-int8.json --errors v1/drug_asr_errors.ultra-w4a8.json \
    --labels 99_hf_release/data/NeMO_files/train.jsonl --labels 99_hf_release/data/NeMO_files/val.jsonl
```

The `--labels` matter: the default `full.jsonl` includes the test labels. Result: 13,626 rules covering 51,539 train+val errors (rejected: 647 `far`, 483 `short`, 270 `common_words`, 242 `real_text`, 158 `imprecise`, 23 `ambiguous`, 23 `other_drug`, 3 `contains_target`). 1,171 of them restore more than the drug name: a word the misspelling swallowed (`soufflue oxétine` -> `sous fluoxétine`, `souvenent la vaccine` -> `sous venlafaxine`, `paraclasta` -> `par aclasta`) or a glued article (`létoposide` -> `l'étoposide`).

How the defaults were chosen: every candidate rule set was applied in full to held-out test transcripts (word error rate, clips better / worse) and to every correct test label (59,151 across the five subsets, where any change is an overcorrection). On the 2,059 test drug clips:

| rules learned from | settings | rules | UltiMed WER | parakeet-ultra WER | correct labels changed |
|---|---|---|---|---|---|
| none | | 0 | 3.73 % | 10.98 % | |
| UltiMed | `--min-count 2 --min-ratio 0` (earlier default) | 1,623 | 3.00 % (419 better, 0 worse) | | 0 |
| UltiMed | defaults | 5,372 | 2.81 % (521 / 0) | 10.08 % (475 / 0) | 0 |
| parakeet-ultra | `--min-count-words 1` | 7,210 | 2.92 % (448 / 0) | 9.49 % (731 / 0) | 5 |
| both | `--min-count-words 1` | 9,980 | 2.72 % (559 / 0) | 9.44 % (759 / 0) | 5 |
| both | defaults, before swallowed-word targets | 9,807 | 2.73 % (557 / 0) | 9.47 % (753 / 0) | 3 |
| both | defaults | 10,030 | 2.70 % (566 / 0) | 9.45 % (758 / 0) | 3 |

Adding the parakeet-ultra int8 and w4a8 ONNX transcripts (2026-10-01), scored the same way with a slightly different text normalization (so compare within this table only; int8 and w4a8 are the ultra ONNX builds):

| rules learned from | rules | UltiMed | ultra fp32 | ultra int8 | ultra w4a8 | correct labels changed |
|---|---|---|---|---|---|---|
| none | 0 | 3.69 % | 10.79 % | 11.00 % | 11.59 % | |
| UltiMed + ultra fp32 | 10,030 | 2.66 % (570 / 0) | 9.22 % (769 / 0) | 9.47 % (750 / 0) | 10.15 % (713 / 0) | 3 |
| + int8 | 11,691 | 2.65 % (577 / 0) | 9.17 % (788 / 0) | 9.41 % (778 / 0) | 10.08 % (747 / 0) | 3 |
| + int8 + w4a8 (committed) | 13,626 | 2.63 % (584 / 0) | 9.12 % (811 / 0) | 9.35 % (802 / 0) | 9.96 % (795 / 0) | 3 |

On the boosted UltiMed int8 browser transcripts the committed rules bring test_drugs from 3.56 % to 2.70 % (10,030 rules: 2.73 %) and the drug sentences from 3.59 % to 2.82 % (3.02 %); FLEURS French is unchanged and FLEURS English moves by at most 0.05 WER point (one extra rewrite, `program` -> `Prograf`; apply the rules to French only).

- `--min-count` 1 against 2 or 3 and `--min-ratio` 0 to 0.8 all changed 0 labels with the UltiMed rules, so the loosest won; the 0.5 ratio floor only drops garbled one-offs (`reea iutis aeec et are` -> `oméga`) for 3 clips.
- Before the distinct-clip count, a third model pushed the same one-off slips over `--min-count-words` (`lait unique` 3 times = 1 clip x 3 models) and the two overcorrections below came back.
- `--min-count-words 2` removes the two real overcorrections of the parakeet-ultra rules (`café au lait unique` -> `Levunique`, `alpha sur bêta estimé` -> `bétahistine`). The 3 labels still changed name a drug the label spells another way (`méronème` -> `Meronem`, `médroxy-progestérone`, `alpha-calcidol` -> `alfacalcidol`).
- On the UltiMed test_dictionary transcripts the rules also fix 50 clips (WER 2.82 % -> 2.79 %) and worsen none.
- Before `contains_target`, 3 test clips got worse (`anti-TNF-alpha` -> `alpha`).

## Merged rules (`03_merge_rules.py`, 2026-10-04)

02 writes one rule per variant. `03_merge_rules.py` then folds the rules that write the same replacement (and share the name guard) into one rule, `"variants": [...]` with one alternation pattern, at the place of the group's first member. Merged and one-by-one rules are not always equivalent (a short member now fires before the rules that sat between it and the first member), so the script checks the merged file on every text it is given and splits back any group that changes one, until every text comes out exactly as before. The committed files are merged, checked on all hyps (drug and term runs, all models), every label of all splits plus PARROT, French and English Wikipedia (1M sentences each) and the held-out French news: 3,745,969 texts, 0 differences.

| file | rules before | rules after | groups split back | size |
|---|---|---|---|---|
| `drug_fix_rules.jsonl` | 13,831 | 3,428 | 10 | 6.2 MB -> 2.0 MB |
| `term_fix_rules.jsonl` | 78,850 | 25,048 | 25 | 37.6 MB -> 15.3 MB |

The merged file drops the per-variant statistics (`drug`, `precision`, `similarity`, `hyp_occurrences`; `count` is summed): the unmerged build output keeps them. Consumers index a merged rule under the anchor of each of its variants (`compile_rules`, and its JS port `scripts/bench/regex-rescore.mjs` in the ultra model repo, which on the merged drug rules matches Python on the original on all 70,398 drug hyps). Built with Claude Code.

    uv run scripts/03_merge_rules.py <rules.jsonl> <merged.jsonl> --hyps <hyps.jsonl> ... --labels <manifest.jsonl or .txt> ...

## `<unk>` variants (`04_drop_unk_variants.py`, 2026-10-06)

The decoder emits `<unk>` for characters missing from its vocab (`°`, `+`, some apostrophes), and `01`'s tokenizer used to split it into the word `unk`, so a few variants read `unk sophagienne` or `d unk orsum`. No runtime output can match them (onnx-asr glues `<unk>` to the next word, the web decoder deletes it), so they were dead weight. `01` now skips any error span holding `<unk>`, and `04` removed the existing ones from the committed files, rebuilding each merged pattern from its remaining variants exactly as `03` builds it: drug 1 variant (3,428 rules kept), term 42 variants (25,045 -> 25,042 rules). Built with Claude Code.

    uv run scripts/04_drop_unk_variants.py <rules.jsonl> <out.jsonl> [--labels <manifest.jsonl or .txt> ...]

## Elided last letters (2026-10-07)

A rule's end bound lets it fire right before an apostrophe, which real fixes need (`mésangio d'IgA` -> `mésangiaux d'IgA`, `étoposie d'orale`). But `01`'s tokenizer writes the label `fait part d'un` as (`part`, `d'`), so the `real_text` guard never matched it against the variant `part d`, and the redux rebuild shipped `part d` -> `pardee`, which rewrote 16 correct labels into `fait pardee'un`. `02` now also looks a variant up with an apostrophe on an elidable last word (`_elided`), and `04 --labels` drops such variants from files built before. On the committed files it removed one variant, `face l` -> `fas-l` (French Wikipedia has `en face l'usine`, and two UltiMed labels write `face l'odex`): term 25,042 -> 25,041 rules, drug unchanged. A first attempt that refused any match before an apostrophe killed the real fixes instead (on the 10 hyps it changed, 32 -> 42 word errors) and was reverted. Built with Claude Code.

## Hand-written rules (`05_build_manual_rules.py`, 2026-10-07)

`manual_fixes.tsv` lists fixes the author asked for directly (`variant<TAB>replacement`), and `05_build_manual_rules.py` turns it into `manual_fix_rules.jsonl` with the same patterns and merging as the learned rules. Consumers applied it FIRST, before `drug_fix_rules.jsonl` and `term_fix_rules.jsonl`, so a hand fix won on the same words: `lodose` / `lodoses` -> `lowdose` (the learned drug rules turned `lodose` into the brand `Lodoz`), and `ceresta` -> `Seresta` (already learned, kept explicit). Since 2026-11-15 it is applied LAST instead, so the label conventions below also respell what a learned rule wrote; the variants the files fought over are decided in `inputs/conflict_resolutions.tsv` and removed from the learned files (see "Conflict decisions"), so order no longer settles them (`lodose` is now `low-dose` after `scanner`, `Lodoz` elsewhere). These rules are not mined or overcorrection-checked like the others, and the published benchmarks ran without them. Add a line to the TSV and rerun:

```bash
uv run scripts/05_build_manual_rules.py
```

### Label conventions as rules (`06_build_convention_fixes.py`, 2026-10-08)

The UltiMed v2 labels follow one spelling per word (`utils/label_conventions.py`: glued prefix compounds like `postopératoire`, `thoracoabdominopelvien`, unaccented drug INNs like `perindopril`, dictionary accents like `artefact`). A model trained on the v1 labels, or one that still slips, writes the other forms, so the author asked for every fixed term to also be a regex rule. `06_build_convention_fixes.py` takes the hyphen and accent families of the label audit (`06_hotfixes/05_label_variant_audit.py`), runs each form through `apply_label_conventions`, and writes `convention_fixes.tsv` (`post-opératoire` -> `postopératoire`, `bévacizumab` -> `bevacizumab`). A form only becomes a rule when the conventions change nothing but hyphens, spaces or accents. A family is all or nothing: if the conventions keep one form as its own word (`débordé` next to `déborde`), none of its forms gets a rule, because the pattern ignores accents and would hit the kept word too. With `--labels`, every label is converted and any rule that still changes one is dropped (the audit forms are lowercased, so a rule could otherwise turn `METAVIR` into `Metavir`). `05_build_manual_rules.py` reads `manual_fixes.tsv` then `convention_fixes.tsv`, so a hand fix wins on the same pattern. Notations that need context (`à un mois` -> `à 1 mois`, `phase II` -> `phase 2`) stay in the conventions only.

```bash
uv run scripts/06_build_convention_fixes.py path/to/variants.tsv \
    --labels 99_hf_release/data/NeMO_files/full.jsonl --labels 99_hf_release/data/NeMO_files/PARROT/full.jsonl
uv run scripts/05_build_manual_rules.py
```

Caveat (2026-10-08): the drug and term rules ran after these and their replacements came from the v1 labels, so they could write an old spelling back (`bévacizumab`). Since 2026-11-15 the manual file, conventions included, is applied after the learned files, so a learned replacement in an old spelling is respelled on the way out.

Refresh (2026-11-15): the conventions changed (all-caps accent-table hits, scripts commit `dc48a1c`), so 52 rows of `convention_fixes.tsv` no longer matched them (`post-biopsie` -> `postbiopsie` now that the conventions spare `post biopsie`, `protéase` -> `protease` now that they write `protéase`, `iléo-cæcal`, `mélæna`...) and were removed, and the 35 rows a fresh run of 06 found that the file lacked were added (`alpha réductase` -> `alpha-réductase`, `hydroxyvitamine` -> `hydroxy-vitamine`, `d'herpes` -> `d'herpès`): 2,127 -> 2,110 rows, manual file 1,945 -> 1,941 rules. The file was NOT regenerated from scratch: the audit reads the current labels, which already follow most conventions, so most families now hold one form and yield no rule (a fresh run gives 37 rows). Regenerate only from an audit of labels that still hold the old spellings.

Refresh (2026-11-16): seven families got two spellings for one word, because the conventions glued the hyphenated form before the accent table could match it (`crânio-caudale` -> `crâniocaudale` while `cranio-caudale` -> `craniocaudale`; scripts commit `b4b0138` fixes the order). The 7 rows were re-derived through 06 the same way as above (`craniocaudale`, `craniocervicale`, `craniofacial`, `craniorachidienne`, `antithyroperoxydase`, `sternocléidomastoïdien`, `sternocléidomastoïdiens`), no row dropped; only the two sterno rules changed in the manual file (the crânio forms were already caught by the accent-folded pattern of the `cranio-` row), 1,941 -> 1,940 rules. 06 now refuses to write a table where two forms that fold to the same pattern get different replacements, and `tests/test_drug_asr_rules.py` checks the committed file for it.

Recasing and refresh (2026-11-18, Claude Code): after the label fix capitalised the eponyms, 53 manual replacements wrote a form the labels no longer use (`ménière`, labels `Ménière` 391 times). 05 now runs 09's `recase_to_labels` on its output (same two passes and thresholds as for the learned files), with the labels of 02's `DEFAULT_LABELS` (`--labels` to change them, `--no-recase` to skip), so it needs the UltiMed-ASR-FR-v1-scripts manifests like 09. That recased 50 replacements (49 names such as `Ménière`, `Lefèvre`, `Hermansky-Pudlak`, `Corynebacterium`, plus `aminoacyl-tRNA`) and nothing else. Two needed their source edited instead: the `étuximab` decision now reads `after:FOLFOX` (the context words go into the replacement as written, labels `FOLFOX` 278 vs `Folfox` 4), and the two antepartum rows of `convention_fixes.tsv` were stale (the conventions now write `antépartum`, as the labels do 38 times) and were re-derived through 06, which merged them into one rule: 1,998 -> 1,997 rules, 2,069 keys. Left as is: `sars-cov`, whose label forms are split (`SARS-CoV` 3, `Sars-Cov` 2, `SARS-COV` 2, `Sars-COV` 1). Two other rows no longer match the conventions and were not touched: `echoendoscopie` -> `échoendoscopie` (conventions: `écho-endoscopie`) and `heřmansky-pudlak` (conventions: `hermansky-Pudlak`, which the recasing now writes `Hermansky-Pudlak` anyway).

Owner spellings (2026-11-18, Claude Code): the labels now follow the owner's list (`bêta-hCG` and bare `hCG`, `SARS-CoV` / `SARS-CoV-1` / `SARS-CoV-2`, `Teralithe`, `Sjögren`, `Hem-o-lok`, `méti-S` / `méti-R`, see the UltiMed-ASR-FR-v1-scripts BUILD_LOG), so the rules were brought to the same spellings without relearning anything (09 recasing alone, no 07 rebuild). 06 lowercased the first letter of a canonical acronym when the variant started lowercase (`sars cov` became `sARS-CoV`); it now keeps an acronym's capitals (doctest added). `convention_fixes.tsv` re-derived through 06 for the affected families: 2,110 -> 2,132 rows, which also fixes the two stale rows above (`echoendoscopie` now becomes `écho-endoscopie`, `sars-cov` becomes `SARS-CoV`). 05 then gives 2,005 rules and 2,081 keys (was 1,997 and 2,069), the new replacements being `bêta-hCG`, `écho-endoscopie`, `Hem-o-lok`, `méti-R`, `méti-S`, `SARS-CoV`, `SARS-CoV-1`, `SARS-CoV-2`, `Sjögren` and `Teralithe`. 09 on the learned files (same `--labels` as above) respelled `Téralithe` to `Teralithe` (drug, parhaf), `meti-s` to `méti-S` (dictionary, parhaf, also inside `aureus méti-S`) and `sjögren` (parhaf), and recased `bêta-hcg` to `bêta-hCG` (dictionary, parhaf); no rule count changed. Its refusals are the same kinds as before (labels do not back the respelling: `Anophèles`, `Herpès`, `Strongyloïdes`, `antimélanosome`, ...). Lookups rebuilt with 08. No new cross-file conflict: the `chogren` conflict (dictionary `Sjögren` vs parhaf `sjogren`) is gone since both now write `Sjögren`. Left as is: the unhyphenated `hemoloc` / `hemolock` / `hem-o-lock` rows (the labels keep those forms too).

Hem-o-lok (2026-11-18, Claude Code): the owner then made the clip brand `Hem-o-lok` in every spelling (plural `Hem-o-loks`), so the labels no longer keep `hemoloc` / `hemolock` / `hem-o-lock` and the rows left as is above are fixed. 06 only derives same-letter respellings (accents, hyphens, spaces), so `convention_fixes.tsv` was re-derived through 06 for the `-lok` families only (the three `-lock` / `-loc` rows dropped, five singular and plural rows added, 2,132 -> 2,134 rows), and the `-lock` / `-loc` respellings went into `manual_fixes.tsv` as 12 hand rows (separator `hém-o-`, `hémo `, `hémo`, ending `lock` or `loc`, each singular and plural). 05's whole-phrase recasing turns 06's `hem-o-lok` into `Hem-o-lok`; 05 still gives 2,005 rules (rules sharing a replacement merge), now 2,095 keys. 09 on the learned files changed nothing (no rule there wrote a clip form) and 08 reported no conflict. `tests/test_lookup.py` gained nine cases (`Hémolocs` to `Hem-o-loks`, `hémolyse` untouched, ...).

Written with Claude Code.

## Conflict decisions (`09_apply_resolutions.py`, 2026-11-15)

The v2 files overlap, and 35 variant pairs got a different replacement depending on the file (`trocarde`: `trocart de` in dictionary, `trocart d'` in PARHAF; `lodose`: `Lodoz` in drugs, `lowdose` in the manual file). The author decided each of the 33 variants by hand in `inputs/conflict_resolutions.tsv`, `variant<TAB>context<TAB>replacement`:

- context `*`: anywhere (`ostradiol` -> `estradiol`);
- `after:WORDS`: only right after those words, which the rule keeps (`lodose` after `scanner` -> `scanner low-dose`); `after:de` with a replacement starting with `d'` consumes the `de` (`de démolyse` -> `d'hémolyse`), `after:l'` glues (`l'areus` -> `l'Aerius`);
- `before:WORDS`: the mirror of `after:WORDS`, only right before those words, which the rule keeps (`antibial` before `postérieure` -> `en tibiale postérieure`, `glissonne` before `7` -> `gleason 7`);
- `before_vowel`: only when the next word starts with a vowel or `h`, the space included (`trocarde aspiration` -> `trocart d'aspiration`; 02's `before_vowel_pattern`, never merged by 03, `"before_vowel": true` in the lookup);
- an empty replacement: no rule at all for the variant in that context (too ambiguous), and no later line may fix it.

The first line matching a variant wins, so a context line comes before the `*` fallback. `09_apply_resolutions.py SRC DST` removes every variant the TSV names from a learned file (through 04's `drop_unk_variants`, which rebuilds a merged pattern from the remaining variants), and 05 turns the lines into manual rules, read before `manual_fixes.tsv` so a decision beats a hand fix. 07 runs it after 04. It then respells every replacement through the label conventions (06's `convention_fixes`, so only accent, hyphen and space changes count), so a learned rule does not write `bévacizumab` when the conventions say `bevacizumab`. With `--labels`, a respelling is kept only if the labels holding the old replacement as a whole word mostly hold the new one once converted: 02 writes its replacements in the lexicon's lowercase canonical form (they are recased only afterwards, see below), and the conventions glue a lowercase compound they leave alone in the labels' own case (`caldeyro-barcia` would become `caldeyrobarcia`, the labels write `Caldeyro-Barcia`). On the committed files that respelled 134 rules (74 replacements: drugs 42, dictionary 61, PARHAF 31, e.g. `dexaméthasone` -> `dexamethasone`, `Taxotère` -> `Taxotere`, `l'eskétamine` -> `l'esketamine`) and refused 66 (dictionary 58, PARHAF 8: `anti-hbs`, `caldeyro-barcia`, `alvaro-duncan`...). Rules that end up with the same replacement are not re-merged, since a merge must be checked on texts (03). On the committed files: drugs 2,388 -> 2,381 rules (15 variants dropped), dictionary 18,137 -> 18,125 (31), PARHAF 2,103 -> 2,094 (21), PARROT 121 -> 120 (1), acronyms unchanged; the manual file grew 1,905 -> 1,945 rules (1,941 after the convention refresh above). No variant has two replacements across the files any more. `tests/test_lookup.py` (`test_resolutions`) runs 24 sentences through both forms of all six files. Done with Claude Code.

Last, with `--labels`, 09 recases each word of a learned replacement to the labels' casing (`label_casing`, `recase`), since 02's lowercase canonical form loses what the labels write: `gleason` -> `Gleason`, `angio-irm` -> `angio-IRM`, `mcardle` -> `McArdle`. A word is recased only if the labels hold it at least 3 times outside a sentence start (not the text's first word, not right after `. ! ? :` or a newline) and spell it the same way every time; a word the labels also write lowercase mid-sentence stays as it is, and the variants are not touched. Matching is case-insensitive, so this changes the output's casing, not what fires, and the rule, key and byte counts do not move. Running 09 again changes nothing. On the committed files (2026-11-17) it recased 4,253 rules: dictionary 3,817, PARHAF 222, drugs 202, PARROT 8, acronyms 4. Of those, 3,937 only capitalise a name, 206 put an acronym in capitals and the rest are mixed (`CoA`, `IgG`, `DiGeorge`, `ATPase`). Since the check is per word, a compound can come out half recased when one of its words also appears lowercase in the labels (`Charcot-marie-Tooth`: `marie` is also a lowercase word). So a second pass (`label_phrase_casing`, `recase_phrases`) looks every multi-word replacement up as a whole phrase in the labels, in one pass indexed by the first two words, skips the occurrences that start a sentence, and takes the spelling that holds at least 80% of at least 3 occurrences: `Charcot-Marie-Tooth`, `CKD-EPI`, `Caldwell-Luc`. It recased 355 more rules (dictionary 290, drugs 32, PARHAF 32, PARROT 1), for 4,420 in all against the lowercase build (dictionary 3,961, PARHAF 241, drugs 205, PARROT 9, acronyms 4). 670 replacements still mix a capitalised word with a lowercase one that is not a function word (dictionary 577, drugs 46, PARHAF 46, PARROT 1). Most are right as they are (`Escherichia coli`, `Kingella kingae`, `Creutzfeldt-Jakob iatrogène`, `Bactrim fort`); the wrong ones (`Rio branco dilatée`) fail the 80% share or occur fewer than 3 times. `Mounier-kuhn` (labels 11 to 3 against `Mounier-Kuhn`) was one of them until the 2026-11-17 label fix capitalised the eponyms; the rebuild below now writes `Mounier-Kuhn`. `--no-recase` skips both passes. Done with Claude Code.

```bash
uv run scripts/09_apply_resolutions.py rules/drug_fix_rules.jsonl rules/drug_fix_rules.jsonl \
    --labels 99_hf_release/data/NeMO_files/full.jsonl --labels 99_hf_release/data/NeMO_files/PARROT/full.jsonl
uv run scripts/05_build_manual_rules.py
```

## Confusion groups (`10_build_confusion_groups.py`, 2026-11-19)

A regex maps one heard form to one replacement, so 02 drops every variant whose errors split between targets (`ambiguous`) or that the model also writes when it is right (`imprecise`). Those are the cases an acoustic second pass could decide from the audio: `étuximab` was `cetuximab` in two clips and `rituximab` in two others. `10_build_confusion_groups.py` exports them instead of losing them: every folded heard form (02's `_key`) with at least two observed targets, summed over the build folders it is given, one group per line in `rules/confusion_groups.jsonl`:

```json
{"heard": "étuximab", "candidates": [{"text": "cetuximab", "count": 2}, {"text": "rituximab", "count": 2}], "heard_correct_count": 0, "kind": "drug", "sources": ["PARHAF", "dictionary", "drugs"], "regex_fallback": null}
```

- `heard` is the variant as 01 writes it (lowercase, punctuation dropped), the most frequent spelling of the folded form; match it case- and accent-insensitively, as the rule patterns do.
- `candidates` are the targets with their error counts, descending; accent or case siblings are one candidate (01 does not count such a slip as an error). Their spelling is the final label spelling: each goes through 09's `apply_conventions` and `recase_to_labels` like a learned replacement.
- `heard_correct_count` is 02's precision denominator (`count_ngrams` over the hypotheses) minus the error count: how often the model wrote the form where the label had no lexicon word to fix. It is what makes the top of the list (inflections such as `abdominale` for `abdominales`, 630 errors against 8,194 correct uses) a poor target for any rewrite.
- `kind` is `drug` when every candidate comes from a drug-lexicon word rarer in French than 01's `MAX_TERM_ZIPF` (the Zipf test keeps out the everyday words of brand presentations, such as `enfants`), else `term`.
- `sources` are the contributing build folders (subset names); `regex_fallback` is what the committed rule files, applied in README order, make of the bare heard form today, or null. Context rules (`after:`, `before_vowel`) cannot fire on a bare form, so `étuximab` (decided `after:FOLFOX` in `inputs/conflict_resolutions.tsv`) shows null.

The committed file holds counts and strings only, no transcript and no clip id. `--examples-out PATH` additionally writes example rows `{"heard", "ref_target", "hyp", "ref", "clip", "split"}` (up to `--examples-per-candidate`, default 5, per candidate and split), and refuses a path inside the repo. They are recovered by re-running 01's `drug_outcomes` on the hypotheses holding a heard form, with the manifests giving label and split, so the test split's rows can serve as a held-out benchmark for groups mined from a train+val build. `--min-candidate-count` (default 1) filters candidates, `--min-total` (default 3) groups.

**The heard forms belong to one model**: they are what that model's hypotheses contain, so regenerate the file for every new model (and every relabelling).

The committed file (2026-11-20) comes from the five `A_remine` builds, the ones the committed rule files were re-mined from (all splits, unified labels, fresh blend70 hypotheses of all 602,776 clips): 905 groups out of 1,242 forms with 2+ targets (was 939 of 1,281 on the 2026-11-19 `A_fixedlabels` builds), 897 `term` and 8 `drug`, 62 with a non-null `regex_fallback` (was 63). The examples run wrote 8,240 rows (train 5,342, val 1,395, test 1,503). About 7.5 minutes on CPU (timed on the 2026-11-19 run).

```bash
A=.../regex_v2_ignore_backups/A_remine
uv run scripts/10_build_confusion_groups.py $A/drugs $A/dictionary $A/PARHAF $A/PARROT $A/acronyms \
    --examples-out /somewhere/outside/the/repo/confusion_examples_remine.jsonl
```

Pass builds of disjoint clips: folder counts are added. 02's variant regrouping was extracted into `group_variants` so 10 imports it rather than copying it. Written with Claude Code.

## Hyphenated drug names (01, 2026-11-20)

01 tracked `Diffu-K` as two drugs, `Diffu` and `Diffuk`, so `difuca` showed up in the confusion groups with both as candidates and a learned rule would have dropped the K. Root cause: the drug lexicon takes stage 02's presence needles, and `presence_needle` splits on hyphens and drops parts under 3 letters, so `DIFFU-K` only gave `diffu`. 01's tokenizer keeps `diffu-k` whole, which the lexicon never held, so labels writing `Diffu-K` were not spotted at all, and `Diffu K` (the majority label spelling) gave the target `Diffu`. Now `anchor_words` also adds a hyphenated word of the drug's raw term when the needles kept part of it (27 words: `diffu-k`, `solu-medrol`, `co-renitec`, `l-thyroxine`, `gonal-f`, ...), and `rejoin_letter_tail` rejoins a label word followed by a lone letter when the two spell a hyphenated lexicon word (`Diffu K` -> `diffu-k`). Only a one-letter tail: a short head (`coenzyme A déshydrogénase`, `NO synthase`) and longer parts (`loco régionale`) stay apart, so term words like `Sanarelli-Schwartzman` or `écho-endoscopie` behave as before (`tests/test_drug_asr_rules.py::test_hyphenated_short_tail`).

Measured on the `A_fixedlabels` builds (labels and hypotheses unchanged, old vs new 01): the drugs subset gains 141 error occurrences under the new hyphenated keys (`solu-medrol` 15, `diffu-k` 11, ...), PARHAF moves 129 `diffu` errors to `diffu-k` and gains 108 more (`Diffu-K` labels, `solu-medrol` 55), dictionary gains 22 (`solu-medrol` 20), PARROT and acronyms none. The committed rule files were re-mined with it on 2026-11-20 (see "Re-mine on the unified labels"). Written with Claude Code.

## Shared helpers (`_util.py`, 2026-11-20)

The helpers the scripts had each written for themselves live in `scripts/_util.py` (stdlib only, so no PEP 723 list changed): `load_script(name, file)` (load a sibling script, or an absolute path, as a module: the numbered names are not valid identifiers), `read_jsonl` (rows of a JSONL file, blank lines skipped; was 02's `load_rules` and 10's hyps reader), `write_jsonl` (one `ensure_ascii=False` object per line; was spelled out in 02, 03, 04, 05, 09 and 10), and `read_texts(path, field)` (a JSONL field or the lines of a `.txt`; was 02's `_read_texts`, and 05, 06, 09 now read their `--labels` through it instead of their own `json.loads(line)["text"]` loop, which gave the same texts on manifests). Each script loads `_util.py` by path as `ultimed_util` (four lines, the only bootstrap left repeated), never `import _util`: the tests keep their own `tests/_util.py` in `sys.modules` under that name. 02 still exposes `load_rules` and `_read_texts` because scripts outside this repo call them. Output checked byte-identical against the previous commit for 03, 04, 05, 08 and 09 on the same inputs, and `tests/test_drug_asr_rules.py::test_util` runs the doctests. Written with Claude Code.

Left alone because they only look the same: 08's rules reader (it parses the bytes it also hashes for `source_sha256`), 08's `write_lookup` (one JSON document with one rule per line, not JSONL), 01's hyps loop (it keeps each raw line for `--hyps-out`) and its line-by-line reads of the drug entries and term lexicon (one field per row, streamed), 01's `read_jsonl` imported from the dataset repo (manifest reader with its own semantics), and `tests/_util.py`'s `load_script` (registers the module in `sys.modules`, which the scripts' loader deliberately does not).

## Lookup spec independent of the Python version (08, 2026-11-20)

`tests/test_lookup.py::test_lookups_current` passed on Python 3.13 and failed on 3.14: the `spec` of every `lookup/*.json` gained `[42972, 42972]` in `word_chars`. 08 learns what a character class lets through by matching it against all of Unicode with `re.IGNORECASE`, and that answer depends on the interpreter's Unicode database. Python 3.14 ships Unicode 16, which assigned U+A7DC, the capital of U+019B (inside `_WORD`'s literal range), so `[_WORD]` matches it there and not on 3.13. Unicode never turns two existing characters into a new case pair, so the drift only ever comes from newly assigned code points. 08 now takes the literal matches over all of Unicode and the case-insensitive extras only from the code points of Unicode 3.2, the frozen table every Python ships (`unicodedata.ucd_3_2_0`). The six lookup files were rebuilt with 08 alone (no rule changed, byte-identical under 3.13 and 3.14): `word_chars` lost five case partners assigned after 3.2 (U+1E9E capital sharp s, U+2C65, U+2C66, U+2C7E, U+2C7F) and the dictionary `case_map` lost U+1E9E -> `ß`. On those five characters the lookup and the regexes can now disagree; none of them occurs in French text. Written with Claude Code.

## French news guard on the term rules (2026-10-08)

The term rules rewrote 94 of the 100k held-out French news sentences (2023). The committed `term_fix_rules.jsonl` is now cleaned by `04_drop_unk_variants.py --guard` with Leipzig's `fr-news-2024-1M.txt` (minus the 171 sentences it shares with the held-out 2023 file): a variant found in that news is dropped, 35,996 -> 35,876 rules. Effect, measured by the ultra repo's `evalrules.py` (`local/run_logs/term-newsguard-20261008/`): held-out news rewrites 94 -> 24, French Wikipedia 3 -> 1, still 0 of 602,813 labels, and the fixes on UltiMed's dictionary test lose about 0.5 % (UltiMed 10,621 -> 10,563 fixed words, WER 2.13 % unchanged; redux 9.25 -> 9.26 %). Done with Claude Code.

## Medical-term rules (`term_fix_rules.jsonl`, 2026-10-04)

A second, separate rule file for the dictionary terms (`syndrome de Brugada`, `ostéophyte`, `craniopharyngiome`...), built by the same 01 + 02 with `--lexicon 01_dictionnary/original_dictionnary.jsonl`. It is kept apart from `drug_fix_rules.jsonl` on purpose: apply it after the drug rules, or not at all. Built with Claude Code by `term-pilot-round.py <pilot dir> 99` in the ultra repo, from every clip of the dictionary (534,338), PARHAF (43,431) and PARROT (1,549) subsets transcribed by UltiMed (int8 encoder) and base parakeet-ultra (w4a8), with the same guards as the drug rules: the correct labels of all splits (PARROT included), the 1M French Wikipedia sentences, and `--names`.

- 78,845 rules (25,045 once merged, 25,042 after the `<unk>` cleanup, see "Merged rules") covering 848,459 error occurrences; 624 carry the name title guard. Rejected (in `term_fix_rules.rejected.jsonl`): real_text 8,357, imprecise 5,232, common_words 3,266, other_term 2,619, short 2,521, far 2,001, ambiguous 1,316, contains_target 133, trailing_a 16, french_word 5. The 2026-10-05 rebuild added `trailing_a`, which dropped exactly 5 rules (`patelle à` -> `patella`, `pelvique à` -> `pelvica`, `plique à` -> `plica`, `scarpe à` -> `scarpa`, `fibule à` -> `fibula`); the drug rules have no such variant whose remainder is real text, so they are unchanged. The merge was re-checked on the same 3,745,969 texts.
- Overcorrection: 0 of 602,813 correct labels, 13 of 1M French Wikipedia sentences (the guard; chained rules can still reach a few), 76 of 100k French news sentences (held out), 9,177 of 1M English Wikipedia sentences (French only, like the drug rules).
- How much more transcription was worth it (the pilot, rules from chunks 1..k tested on unseen chunk k+1): coverage of the next chunk's term errors rose about 5.5 points per doubling of data, 22.4 % -> 33.7 % (UltiMed) and 41.4 % -> 51.0 % (ultra) from 1 to 4 chunks of about 13k dictionary clips; the full build has about 10 times more.
- Names: 28 of the 109,728 test names are still rewritten, nearly all first names without a title (`Ilyan` -> `ilion`, `Manoé` -> `mannose`) or two-word surnames after a title (`le docteur Le Beller` -> `Le Böhler`, the guard only looks at the word right after the title).
- Acronyms are not covered: the dictionary lexicon only tracks words of 5 letters or more.

## Redux rebuild (2026-10-07, committed)

The committed drug and term files were rebuilt with the 1.58-bit parakeet-redux (int8 encoder) as a third model, so they also fix the mistakes it makes. Every clip of the drugs, dictionary, PARHAF and PARROT subsets was transcribed with redux. The drug rules come from the v7 build (`01` + `02` on the five drug hyps files: UltiMed, ultra fp32 / int8 / w4a8, redux int8). The term rules come from `term-pilot-round.py <pilot dir> 99 --tag=-redux` with the same flags as the 2026-10-04 build. Both were merged by `03` (checked on all hyps, every label, French and English Wikipedia and French news: 4,335,271 texts) and then cleaned by `04 --labels` (the elided `part d` variant, see "Elided last letters"). The driver is `local/helpers/redux-rules-20261006.sh` in the ultra repo. Built with Claude Code.

| file | variants | rules (merged) | before |
|---|---|---|---|
| `drug_fix_rules.jsonl` | 19,032 | 3,963 | 3,428 |
| `term_fix_rules.jsonl` | 138,430 | 35,996 | 25,041 |

WER on the test splits, raw -> drug + term rules (in parentheses: the previous files), `evalrules.py` via `redux-eval-20261005.sh`:

| model | drugs | PARROT | PARHAF | dictionary |
|---|---|---|---|---|
| redux int8 | 14.24 -> 9.92 (11.71) | 11.32 -> 9.36 (10.28) | 9.83 -> 7.72 (8.48) | 12.09 -> 9.24 (10.41) |
| ultra w4a8 | 11.59 -> 8.43 | 8.15 -> 7.25 | 7.83 -> 6.71 | 7.74 -> 6.20 (6.19) |
| UltiMed (drugs: .nemo, else int8) | 3.69 -> 1.83 | 4.20 -> 3.93 | 3.06 -> 2.74 | 2.74 -> 2.13 (2.13) |

- Overcorrection: 0 of 602,813 correct labels. On the 100k held-out French news sentences, 96 are rewritten (75 with the previous files), nearly all by term rules (the drug file alone: 2). Examples: `cryptos` -> `cryptes`, the town `Mamer` -> `Mammaire`, `paqueter` -> `plaquettaire`.
- Worse clips (a fix that raises a clip's WER) stay at 3 to 5 per model and slice, the same as before: they are word-boundary changes like `pyridoxal-phosphate` -> `pyridoxal phosphate` against a hyphenated label.

## Real-speech check (2026-10-04)

Both rule files on real French speech with labels, clip by clip (WER, clips better / worse), by `evalrules.py --refs` in the ultra repo: FLEURS fr (all 1,665 clips, never used in training), 5,000 Common Voice fr clips the finetune never saw (a local train tar minus the rehearsal set), and the author's own 206 dictated medical clips (private).

| set | model | no rules | drug rules | term rules |
|---|---|---|---|---|
| FLEURS fr | UltiMed | 5.11 % | 5.11 % (0 / 0) | 5.11 % (2 / 0) |
| FLEURS fr | ultra w4a8 | 4.19 % | 4.19 % (0 / 0) | 4.19 % (0 / 0) |
| Common Voice fr | UltiMed | 6.02 % | 6.02 % (0 / 0) | 6.02 % (3 / 1) |
| Common Voice fr | ultra w4a8 | 5.43 % | 5.43 % (0 / 0) | 5.42 % (5 / 1) |
| own medical dictation | UltiMed | 10.07 % | 9.06 % (31 / 0) | 9.17 % (28 / 0) |
| own medical dictation | ultra w4a8 | 25.28 % | 23.34 % (63 / 0) | 22.94 % (69 / 0) |

The drug rules make no clip worse. The term rules make 2 of about 13,700 scored clips worse, both already misheard: `éléments finis d'altair` heard `d appler` became `doppler`, and `le jarret` heard `jarré` became `lejars`.

## Tests

```bash
uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz --with wordfreq tests/test_drug_asr_rules.py
uv run tests/test_lookup.py
```

## Smoke test (2026-09-29)

Run 1.7.0 step 7000 on the 205 private oli clips found 211 drug occurrences (198 drugs) and 59 misspellings, for example `phélodipine` (félodipine), `d'hexaméthasone` (dexaméthasone) and `létoposide` (target `l'étoposide`, so the article is kept). 02 kept 57 rules and rejected 2 as `real_text`.
