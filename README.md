# UltiMed-ASR-FR-v2-regex-fixes

> [!WARNING]
> **Release in progress.** UltiMed v2 is being released right now. As long as this notice is here, files, links and numbers in this repository are still being changed.

Part of [UltiMed-Org](https://github.com/UltiMed-Org), which lists every UltiMed repository, dataset and model.

> [!IMPORTANT]
> **Want faster improvements?** This project is entirely self-funded on my minimum-wage salary, and every training run competes for a single consumer GPU. If you or your organisation can donate an **RTX 5090**, or the money to buy one second-hand, it would directly speed up the next versions of the dataset and the fine-tuned models. Reach out via [olicorne.org](https://olicorne.org) or open an issue on this repository.

Cheap post-processing for French medical ASR: lists of rules that rewrite the misspellings the [UltiMed parakeet model](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-UltiMed-v2-onnx) still makes on French medical vocabulary into the right word, for example `veine la faxine` becomes `venlafaxine`. Plain text substitution after decoding: no GPU, no extra model. They also help models that were not finetuned.

Each rule list ships in two equivalent forms: regexes (`rules/`) and lookup tables (`lookup/`), for apps whose regex engine has no lookbehind. Both give the same output on every text (see "Equivalence" below).

This repository, its scripts and its docs were written with Claude Code. The rules and their build history moved here on 2026-11-15 from the model repository (`regex-fixes/`) and from `08_drug_asr_rules/` of [UltiMed-ASR-FR-v2-scripts](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts).

License: AGPL-3.0-or-later for everything in this repository (rules, lookup tables, scripts, tests), see `LICENSE`. The build scripts were first published in UltiMed-ASR-FR-v1-scripts under Apache-2.0 and the v1 rule files in the model repository under CC BY 4.0; those earlier copies keep their original license, this repository is the same author's relicensed continuation.

## The rules

These are the **v2** rules, built on 2026-11-17 (rebuilt on 2026-11-18 against the corrected labels and re-mined on 2026-11-20 on the unified labels, see [scripts/README.md](scripts/README.md)) from the v2 model alone (the 70/30 blend of the "mix7" weight soup with the v1 weights, fp32 ONNX) transcribing every clip of the v2 UltiMed corpus ([Olicorne/UltiMed-ASR-FR-v2](https://huggingface.co/datasets/Olicorne/UltiMed-ASR-FR-v2), 602,776 clips). The rules are learned from the v2 labels, so the replacement spelling is the one the v2 labels use. That includes the casing: a name or acronym that the labels write the same way every time comes out the same way in the replacement (`Gleason`, `angio-IRM`), see `scripts/09_apply_resolutions.py`. Each subset's file was built by `scripts/07_build_dataset_rules.sh`; [scripts/README.md](scripts/README.md) has the full build log, the guards and the evaluation.

The drug and dictionary files also carry the **v1 rules, respelled to v2** (2026-11-23, `scripts/11_import_v1_rules.py`, see "v1 rules imported" below): 127,270 variants the v1 rules fixed, learned from the errors of four v1-era models (the v1 fine-tune and the base parakeet-ultra in fp32, int8 and w4a8), so they hold misspellings the v2 model does not make on the corpus but other models, and v2 on new audio, do. Each imported rule says so (`"source": "v1"`, or `"v1_variants"` on a v2 rule that took some), and the files, their names and their order are unchanged, so apps need no change. All five learned files then also carry **variants found by two other models of the family** (2026-11-23, `scripts/12_import_extra_variants.py`, see "Other models' variants imported" below): 14,825 from the w4a8 quantisation of v2 and 10,332 from the redux fine-tune, tagged `"source": "v2-w4a8"` / `"redux-v2"` (or `"v2-w4a8_variants"` / `"redux-v2_variants"` on a rule that took some).

| file | what it fixes | variants | rules (merged) |
|---|---|---|---|
| `manual_fix_rules.jsonl` | hand-written fixes and decisions (`post-opératoire`, `scanner lodose` to `scanner low-dose`, `Diffu-K`, ...), dataset-agnostic, built by `scripts/05_build_manual_rules.py` from `inputs/` | 2,160 | 2,060 |
| `drug_fix_rules.jsonl` | drug names, learned on the drugs subset, plus the v1 drug rules respelled (15,914 of the variants) and other models' variants (1,147 w4a8, 403 redux) | 22,521 | 4,145 |
| `dictionary_fix_rules.jsonl` | medical terms, learned on the dictionary subset, then filtered against French news 2024, plus the v1 term rules respelled (111,356 of the variants) and other models' variants (12,769 w4a8, 9,460 redux) | 177,160 | 36,177 |
| `parhaf_fix_rules.jsonl` | medical terms, learned on the PARHAF subset, plus other models' variants (872 w4a8, 456 redux) | 5,474 | 2,502 |
| `parrot_fix_rules.jsonl` | medical terms, learned on the PARROT subset (test only), plus other models' variants (29 w4a8, 12 redux) | 188 | 138 |
| `acronyms_fix_rules.jsonl` | medical terms, learned on the acronyms subset, plus other models' variants (8 w4a8, 1 redux) | 52 | 47 |

`rules/confusion_groups.jsonl` is not a rule file: it lists the 905 heard forms the v2 model writes for two or more different words (`étuximab` for `cetuximab` or `rituximab`), which no regex can fix, with the candidates and their counts, for an acoustic second pass. No transcript or clip id is in it. It is tied to the model that produced the hypotheses and must be regenerated for each new model; see `scripts/10_build_confusion_groups.py` and [scripts/README.md](scripts/README.md). Script and docs written with Claude Code.

The four term files replace v1's single `term_fix_rules.jsonl`. They overlap: 1,192 variants appear in both the dictionary and PARHAF files, and 35 variants get a different replacement depending on the file (19 dictionary vs PARHAF, e.g. `trocarde` becomes `trocart de` in one and `trocart d'` in the other; 12 dictionary vs drugs; 3 PARHAF vs drugs; 1 dictionary vs PARROT). The author decided those conflicts (33 distinct variants, some in two pairs) by hand on 2026-11-15 (`inputs/conflict_resolutions.tsv`, some only in a context: `lodose` becomes `low-dose` after `scanner` but `Lodoz` elsewhere, `trocarde` becomes `trocart d'` before a vowel). `scripts/09_apply_resolutions.py` removes them from the learned files and the decisions ship as manual rules, so no variant gets two replacements any more. The order still matters for chained rewrites: apply the drug file first, then the term files, and the manual file LAST, so its spelling conventions (`post-opératoire` -> `postopératoire`, `bévacizumab` -> `bevacizumab`) also respell what a learned rule wrote. (It used to go first so that a hand fix would win, e.g. `lodose` -> `lowdose` over the learned `Lodoz`; that variant is now decided in `conflict_resolutions.tsv` and removed from the learned files.) The order used for every with-rules number published for v2 (the benchmark tables and the real-voice page of the model repository) is: drugs, dictionary, PARHAF, PARROT, acronyms, then manual. Since no variant gets two replacements any more, the order of the term files only matters for chained rewrites between them; other orders were not measured.

### One list?

The numbers in this section were measured on the files before the v1 import (25,099 rules, 55,124 variants); the reasoning does not depend on them.

No merged file ships, on purpose (decided 2026-11-21 with Claude Code). v1's `term_fix_rules.jsonl` was not a merge of separate files: it came out of one build (02 then 03) over the pooled dictionary, PARHAF and PARROT hypotheses. For v2, a generated `all_fix_rules.jsonl` was considered and rejected, because it would only repeat what the six files already give:

- **The six files concatenated in the order above already are that one list.** Both appliers (`compile_rules` and its JavaScript port, `apply_lookup.py`) try the rules in index order and, after every change, retry each later rule whose key the new text contains. So a rewrite in one file that feeds a rule of a later file fires exactly as when the files run one after the other. `tests/test_lookup.py::test_concatenation` checks it on 4,117 synthetic lines (regex and lookup forms, 0 differences). On 185,698 real texts it also gave 0 differences: 83,971 v2 transcripts (all 21,894 drug clips, every 5th PARHAF clip, every 10th dictionary clip), the 100,000 held-out news sentences and 1,727 VoxPopuli fr lines.
- **The only gain would be deduplication, and it is small.** 2,345 of the 55,124 variants (4 %) repeat a variant an earlier file already rewrites to the same replacement: 1,030 dictionary then PARHAF, 640 drugs then dictionary, 510 drugs then PARHAF, 68 dictionary then PARROT, 58 dictionary then manual, a few dozen others. 997 of the 25,099 rules consist only of such repeats. Dropping those 997 changed nothing on the same 185,698 texts. But that can only be tested, never proven: a repeat in a later file still matters whenever a file in between writes its variant.
- **Order-dependent chains between files are rare in practice.** On the same texts, no line had a rule from one file fire only because an earlier file rewrote the text. The 6 chained lines found (all in the drug clips) chain inside the manual file. So in practice the order of the six files only matters for the manual file, which goes last on purpose (see above).
- **It would cost a duplicate.** A 13 MB copy of the same rules (plus its lookup) would have to be regenerated by every step that writes a rule file (05, 07, 09), while the consumers (parakeet_web, Plumed) already load the six files one after the other and would gain nothing from it.

To use one list anyway, concatenate the files in that order (`for f in drug dictionary parhaf parrot acronyms manual; do cat rules/${f}_fix_rules.jsonl; done > all.jsonl`). `uv run scripts/08_build_lookup.py all.jsonl --out all.lookup.json` turns it into a single lookup with the same output.

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

Real speech (2026-11-21, measured with Claude Code). The v2 model (int8 ONNX encoder and decoder, greedy, no phrase boost) transcribed the same real-speech sets as the v1 check of 2026-10-04 in [scripts/README.md](scripts/README.md), and the hypotheses were scored the same way (the normalisation of the ultra repo's `evalrules.py`: lowercase, punctuation dropped, accents and hyphens kept), WER without rules and then with all six files in the order above:

| set | clips | words | no rules | v2 rules | lines changed (better / worse) |
|---|---|---|---|---|---|
| FLEURS fr (all 1,665 local clips) | 1,665 | 42,635 | 4.72 % | 4.72 % | 1 (1 / 0) |
| Common Voice fr (5,000 clips the v2 training never used) | 5,000 | 52,658 | 6.22 % | 6.22 % | 3 (1 / 0) |
| the author's dictated medical sentences (private) | 206 | 3,544 | 8.72 % | 7.65 % | 39 (36 / 3) |
| the author's 15 long dictations (private) | 15 | 1,182 | 7.28 % | 5.92 % | 10 (10 / 0) |

- **General French is left alone, nearly.** On FLEURS the one rewrite is a real fix, `d'Ockland` -> `d'Auckland`. There were 3 until `Le docteur Malar` -> `Le docteur Malaire` (the same sentence read by two speakers: a dictionary rule hit a surname the title guard did not know, since it only guarded INSEE names) was fixed on 2026-11-21 by adding `malar` to `inputs/extra_names.txt`. On the 289 FLEURS-fr validation clips of the model repository's benchmark (same model, greedy int8, that benchmark's own normaliser) the rules change no line: 5.10 % with and without. FLEURS fr was never trained on, but its validation split was monitored to choose the checkpoint. On Common Voice the 3 rewrites are one real fix, `lombrique` -> `lombric`, and two that leave an already wrong word wrong: `Meigné` -> `Maigne` (a village name) and `méthis` -> `méti-S`. None of those 5,000 clips is in the 138,999-clip Common Voice rehearsal set of the v2 training, but Common Voice as a whole is in parakeet v3's pretraining data.
- **The gain is on medical dictation, and it is not in-sample.** No learned file saw any of these recordings; the manual file is hand-written and fires 3 times on the 206 sentences, never on the 15 dictations. By file, the rules that fired on the 206 + 15 recordings: drugs 29 + 8, dictionary 7 + 1, PARHAF 0 + 5, manual 3 + 0, PARROT and acronyms never. Caveat: the 206 sentences are also v2's validation and test sets for the author's voice, so they took part in choosing the released checkpoint (not in building the rules).
- **The 3 sentences made worse are spelling conventions, not mishearings.** Their references were written before the v2 conventions and spell `éculizumab`, `cholécalciférol` and `post-chirurgicale` where the v2 labels, and so the rules, write `eculizumab`, `cholecalciferol` and `postchirurgicale`.
- **The 15 long dictations** are the recordings of the model repository's real-voice stress test page, whose hypotheses were reused as is rather than transcribed again: there the decoder ran in fp32 (that benchmark's setting) rather than int8, and with that page's normaliser the same hypotheses read 7.61 % -> 6.19 %.
- **Small samples.** 206 and 15 recordings of one speaker are enough to show the direction, not a precise size: one word is worth 0.08 points of WER on the 15 dictations and 0.03 on the 206 sentences.
- **The rules are for French.** On the other ten FLEURS languages of the same benchmark run (3,484 clips), they rewrite 57 lines, 40 of them Portuguese `foram` -> `foramen`.

### v1 rules imported (2026-11-23)

The v1 rule files (`v1/drug_fix_rules.jsonl`, 3,963 rules / 19,032 variants, and `v1/term_fix_rules.jsonl`, 35,876 / 138,072) were not learned from the v2 model: they come from the errors of the v1 fine-tune and of the base parakeet-ultra in fp32, int8 and w4a8. So they know misspellings the v2 model does not make on the corpus but that the other models, and v2 itself on new audio, do. Applied as they are, they would write v1 spellings into v2 text (`acétoacétate` where the v2 labels write `acéto-acétate`) and rewrite 9,496 correct v2 labels. `scripts/11_import_v1_rules.py` (written with Claude Code) rebuilds each of them as a v2 rule, reusing the build scripts' own functions:

1. **Respell the replacement** to the spelling the v2 labels use for it (their majority spelling once case, accents, hyphens and spaces are folded away: `acétoacétate` -> `acéto-acétate`, `abdomino-pelvienne` -> `abdominopelvienne`, `Solumédrol` -> `Solu-Médrol`), then the same passes as every v2 file (09: conflict decisions, title guard for `inputs/extra_names.txt`, label conventions, recasing to the labels). 1,883 of the 34,066 v1 replacements changed. A replacement with a word the labels never write is kept only if that word is a headword the builders track (drug lexicon, dictionary terms); 26 replacements (78 variants) had none and were dropped.
2. **Re-guard every variant** with 02's own rules, as if a v2 build had found it: `real_text` against the v2 labels, the 1M French Wikipedia sentences and French news 2024, `short`, `other_term`, `common_words`, `far`, `ambiguous`, the INSEE names title guard; plus, for the import only, any one-word variant that is a French word at all (wordfreq Zipf >= 1.0, stricter than 02's 2.5, since no v2 transcript backs these variants). **v2 wins**: a variant any v2 file already rewrites keeps its v2 replacement and is not imported.
3. **Check on text**: an imported variant is dropped if, alone or with the six files applied in order, it rewrites a correct v2 label or makes a v2 transcript of the corpus (any split) worse against its label.
4. **Merge** into `drug_fix_rules.jsonl` (from the v1 drug file) or `dictionary_fix_rules.jsonl` (from the v1 term file), per replacement, with 03's merge checked on 2.1M texts (v2 transcripts, labels, French Wikipedia, held-out news 2023).

Result: 127,270 variants imported (drugs 15,914, dictionary 111,356), 25,722 dropped: already in a v2 file 20,234, `common_words` 4,305, `real_text` 704, French word 312, no v2 spelling 78, ambiguous 40, made a v2 transcript worse 41, a character the lookup form cannot key (Greek or Cyrillic letters) 6, `contains_target` 2 (the other 4,112 v1 variants are spellings of one another once folded, or appear in both v1 files). The six files grow from 25,099 rules / 55,124 variants to 42,795 / 182,398.

Effect, the six files in order, before and after the import (2026-11-23, measured with Claude Code; the WER uses the normalisation of the ultra repo's `evalrules.py`, on the fp32 ONNX v2 hyps of the 2026-11-20 re-mine for the test splits and on the int8 hyps of the real-speech check below; clips better / worse in brackets):

| set | no rules | before | after |
|---|---|---|---|
| test drugs (2,059 clips) | 3.53 % | 1.93 % (880 / 0) | 1.91 % (884 / 0) |
| test dictionary (49,891) | 2.54 % | 1.98 % (9,782 / 4) | 1.97 % (9,958 / 4) |
| test PARHAF (5,489) | 2.85 % | 2.49 % (1,083 / 0) | 2.48 % (1,093 / 0) |
| test PARROT (1,548) | 3.64 % | 3.32 % (204 / 1) | 3.31 % (206 / 1) |
| test acronyms (151) | 2.96 % | 2.72 % (6 / 0) | 2.72 % (6 / 0) |
| FLEURS fr (1,665) | 4.72 % | 4.72 % (1 / 0) | 4.72 % (1 / 0) |
| Common Voice fr, unseen (5,000) | 6.22 % | 6.22 % (1 / 0) | 6.22 % (2 / 0) |
| the author's dictated medical sentences (206) | 8.69 % | 7.73 % (33 / 3) | 7.51 % (38 / 3) |

| rewritten lines | before | after |
|---|---|---|
| correct v2 labels (602,776) | 0 | 0 |
| French news 2023, held out (100,000) | 47 | 62 |
| VoxPopuli fr references (1,662) | 0 | 0 |

- **The gain is small on the corpus and larger on new audio**, as expected: the v2 files were already learned from v2's own errors on these clips, so the test splits gain 0.01 to 0.02 points, while the author's own dictations (never seen by any rule build) go from 7.73 % to 7.51 %, 5 more sentences fixed (`trimétoprime` -> `triméthoprime`, `respéridone` -> `rispéridone`, `lébastine` -> `l'ébastine`, ...). The 3 worse sentences are the same spelling conventions as before the import (see "Real speech" above). The own-voice hyps were re-transcribed on CPU for this check, so the no-rules WER reads 8.69 % instead of the 8.72 % above.
- **The cost is 15 more rewritten news sentences** (0.015 %), all proper names or rare words that a v1 variant happens to spell: `Kiari` -> `Chiari`, `Tenas` -> `Ténase`, `Perrine Balant` -> `Ballant`, `Koelner` -> `Köllner`, `paqueter` -> `plaquettaire`. Those were left in rather than dropped one by one, since the 2023 news is the held-out measure of what the guards miss; the 2024 news and Wikipedia guards already removed the variants they contain. On Common Voice the 7 new rewrites are of the same kind (`Rainwood` -> `Rinvoq`, `aphase` -> `aphasie`), 1 is a real fix (`champignistes` -> `champignonnistes`), and none changes the WER.
- A first import let `cubis` -> `pubis`, `mélasses` -> `melas` (VoxPopuli) and `locals` -> `local` through, which is why one-word imported variants that are French words are now refused (312 variants, a few of them real fixes such as `ibuprofen`).

### Other models' variants imported (2026-11-23)

The learned files come from the errors of one model, the v2 fp32 blend. Two more models of the family transcribed the whole v2 corpus: the w4a8 quantisation of v2 (`v2-w4a8`, what a phone or a small CPU runs) and the redux fine-tune in int8 (`redux-v2`, about twice v2's WER). Each got a full rule build (07, all splits, the release guards), and `scripts/12_import_extra_variants.py` (written with Claude Code) adds the variants those builds found that the committed files lack, each to its own subset's file (a w4a8 drug variant to `drug_fix_rules.jsonl`, a redux PARHAF variant to `parhaf_fix_rules.jsonl`, ...). It reuses 11's steps 3 to 5 as one shared function (`merge_import`), so the checks are the same as for the v1 import:

1. **Existing rules win**: a variant any of the six files already rewrites (the v1 imports included) is skipped, and w4a8 goes first, so redux cannot claim a variant w4a8 already gave.
2. **Redux is held to more evidence**: a redux variant seen once in its build is kept only if w4a8 or the v1 rules also hold it (`--min-count redux-v2=2`). No such filter for w4a8, whose WER is close to v2's.
3. The current conflict decisions (09), then the same text checks as 11: no imported variant may rewrite a correct v2 label or make a v2 transcript worse, alone or with the six files in order, and 03's merge check on 2.1M texts.

Result: 25,157 variants imported, 14,825 from w4a8 and 10,332 from redux. Redux screening kept 10,344 (10,270 seen at least twice, 74 seen once but corroborated) and dropped 36,559 seen once with no corroboration; 46,771 redux variants were already in a file. The text checks dropped 15 (alone) and 12 (in the stack), and the author's decision removed one more (`aussi toxique` -> `ocytocique`, below). The six files grow from 42,795 rules / 182,398 variants to 45,069 / 207,555.

Effect, six files in order (measured with Claude Code; same normalisation and v2 fp32 hyps as above; better / worse clips in brackets). The w4a8 and redux rows apply the rules to those models' own transcripts of the test splits, which the imported variants were mined from, so they show what the import fixes for those models rather than how it generalizes:

| set | no rules | before | after |
|---|---|---|---|
| test drugs, v2 fp32 | 3.53 % | 1.91 % (884 / 0) | 1.90 % (889 / 0) |
| test dictionary, v2 fp32 | 2.54 % | 1.97 % (9,958 / 4) | 1.96 % (10,076 / 4) |
| test PARHAF, v2 fp32 | 2.85 % | 2.48 % (1,093 / 0) | 2.48 % (1,101 / 0) |
| test PARROT, v2 fp32 | 3.64 % | 3.31 % (206 / 1) | 3.31 % (207 / 1) |
| test acronyms, v2 fp32 | 2.96 % | 2.72 % (6 / 0) | 2.72 % (6 / 0) |
| test drugs, w4a8 | 3.66 % | 2.22 % (799 / 0) | 1.99 % (900 / 0) |
| test dictionary, w4a8 | 2.65 % | 2.11 % (9,402 / 2) | 2.03 % (10,578 / 2) |
| test PARHAF, w4a8 | 2.90 % | 2.54 % (1,085 / 1) | 2.51 % (1,151 / 1) |
| test PARROT, w4a8 | 3.90 % | 3.57 % (209 / 0) | 3.52 % (239 / 0) |
| test acronyms, w4a8 | 2.66 % | 2.42 % (6 / 0) | 2.39 % (7 / 0) |
| test drugs, redux | 6.22 % | 4.42 % (903 / 0) | 4.12 % (1,028 / 0) |
| test dictionary, redux | 5.18 % | 4.17 % (16,246 / 15) | 3.94 % (18,982 / 14) |
| test PARHAF, redux | 4.96 % | 4.21 % (2,024 / 1) | 4.06 % (2,296 / 0) |
| test PARROT, redux | 6.38 % | 5.72 % (420 / 0) | 5.54 % (512 / 0) |
| test acronyms, redux | 4.28 % | 3.89 % (11 / 0) | 3.68 % (16 / 0) |
| FLEURS fr (1,665) | 4.72 % | 4.72 % (1 / 0) | 4.72 % (1 / 0) |
| Common Voice fr, unseen (5,000) | 6.22 % | 6.22 % (2 / 0) | 6.22 % (2 / 0) |
| author's dictated medical sentences (206) | 8.69 % | 7.51 % (38 / 3) | 7.51 % (38 / 3) |
| correct v2 labels (602,776) | 0 | 0 | 0 |
| French news 2023, held out (100,000) | 47 | 62 | 67 |
| VoxPopuli fr references (1,662) | 0 | 0 | 0 |

- **On v2's own transcripts the gain is small**, as for the v1 import: the v2 files already hold v2's errors. No split gets worse and no split has more worse clips than before. On the two other models it is the point of the import: w4a8's drug test split goes from 2.22 % to 1.99 %, redux's dictionary split from 4.17 % to 3.94 %.
- **Real speech does not move**: on FLEURS, Common Voice and the author's dictations the outputs are identical, line for line, with and without the import. Those clips were transcribed by v2 (int8), whose errors the files already held; the import is aimed at the other models' mishearings.
- On the whole corpus (all splits, v2 fp32 transcripts) the import changes 1,994 transcripts, 1,586 of them closer to their label and none further.
- **The cost is 5 more rewritten news sentences** (0.005 %), all proper names or a typo in the source: `Meyerhoff` -> `Meyerhof`, `Burkin Faso` -> `Bourquin Faso`, `Michel Chiret` -> `Chiray`, `musée Esrine` -> `Ezrine`, and `mener des sensibilisation` -> `désensibilisation` (a redux variant seen twice). A sixth, `tout aussi toxique` -> `tout ocytocique`, came from a redux variant inside the merged `ocytocique` rule; `aussi toxique` is everyday French, so the author's decision in `inputs/conflict_resolutions.tsv` (empty replacement: no rule) removed it, which also stops a rerun from importing it again.
- 03's merge check runs on the v2 transcripts, the labels, Wikipedia and the held-out news, not on the w4a8 or redux transcripts, which have only been used to find the variants.

## Layout

| path | content |
|---|---|
| `rules/` | the six v2 rule files (regex form), and `confusion_groups.jsonl` (ambiguous heard forms, not rules) |
| `lookup/` | the same six files in lookup-table form, generated from `rules/` by `scripts/08_build_lookup.py` |
| `rejected/` | the candidates each v2 build rejected, with the reason, and `drug_asr_errors.json` (the misspellings the drug build started from); not published: they hold fragments of dataset transcripts and nothing reads them, so they stay local and gitignored |
| `inputs/` | the author's `conflict_resolutions.tsv` (variants the learned files disagreed on), the hand-written `manual_fixes.tsv` and the generated `convention_fixes.tsv`, source of `manual_fix_rules.jsonl` |
| `v1/` | the v1 outputs, kept for comparison: `drug_fix_rules.jsonl`, `term_fix_rules.jsonl` (dictionary + PARHAF + PARROT in one list) and the per-model `drug_asr_errors*.json` (their rejected lists stay local, like `rejected/`) |
| `scripts/` | the build chain (01 to 07 and 09, plus 10 for the confusion groups and 11 for the v1 import, see [scripts/README.md](scripts/README.md)), `08_build_lookup.py` and `apply_lookup.py` |
| `tests/` | `test_drug_asr_rules.py` (the build scripts) and `test_lookup.py` (regex vs lookup equivalence, on synthetic texts; no real transcript is committed) |

The build scripts read the corpus manifests, the drug lexicon and a few helpers from a [UltiMed-ASR-FR-v2-scripts](https://github.com/UltiMed-Org/UltiMed-ASR-FR-v2-scripts) checkout. By default they look for it next to this repository, in a folder still named `UltiMed-ASR-FR-v1-scripts`; set `ULTIMED_SCRIPTS_DIR` to point elsewhere. Applying the rules, building the lookups and `test_lookup.py` need nothing outside this repository.

## Regex form (`rules/*.jsonl`)

One JSON rule per line, applied top to bottom, file after file. For each rule:

- compile `pattern` case-insensitively (`new RegExp(pattern, 'giu')` in JavaScript, `re.IGNORECASE` in Python; the patterns only use syntax both engines share);
- replace every match with `replacement`;
- when the matched text starts with a capital and the replacement does not, capitalise the replacement's first letter.

A rule with `"before_vowel": true` (only in the manual file) ends on the whitespace before a word starting with a vowel or `h` instead of a word end, and its replacement eats that whitespace (`trocarde aspiration` becomes `trocart d'aspiration`). A rule with `"name": true` is also a French surname (an INSEE name given to 02 `--names`, or one of the hand-kept `inputs/extra_names.txt` that 09 adds, such as `Malar`): its pattern already skips a match right after a title like `docteur`, so the surname in `docteur X` is left alone. Merged rules carry a `variants` list (one alternation per replacement); unmerged ones a single `variant` plus build statistics.

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
| `manual_fix_rules` | 2,060 | 2,160 | 0 | 810,536 | 121,464 |
| `drug_fix_rules` | 4,145 | 22,521 | 55 | 3,059,293 | 465,159 |
| `dictionary_fix_rules` | 36,177 | 177,160 | 803 | 28,966,051 | 4,334,831 |
| `parhaf_fix_rules` | 2,502 | 5,474 | 41 | 1,312,214 | 168,494 |
| `parrot_fix_rules` | 138 | 188 | 1 | 66,823 | 10,060 |
| `acronyms_fix_rules` | 47 | 52 | 0 | 21,999 | 4,663 |

### How the lookups are built

`uv run scripts/08_build_lookup.py rules/*_fix_rules.jsonl` writes `lookup/<name>.lookup.json` for the six rule files. Do not pass `rules/*.jsonl`: that glob also catches `confusion_groups.jsonl`, which is not a rule file and stops the build with `KeyError: 'variant'`. Nothing is reinterpreted: each rule's pattern is first rebuilt from its variants with the same functions that wrote it (`variant_pattern` in 02, the merge scaffolding of 03) and must equal the stored pattern byte for byte, so the keys describe exactly what the regex matches. A rule that does not rebuild stops the build instead of being approximated; none of the 45,069 v2 rules fails. The lookup records the SHA-256 of its source, and `05_build_manual_rules.py` and `07_build_dataset_rules.sh` write the lookup form each time they write rules.

### Equivalence

`tests/test_lookup.py` checks that every lookup gives exactly the output of `apply_rules` on its regex file:

- the doctests, and that `lookup/` is up to date with `rules/` (rebuilt into a temporary dir and compared byte for byte);
- 67 hand-written edge texts (elisions, curly apostrophes, hyphens, whitespace runs, capitals, titles, `aujourd'hui`-like words) on a small rule set built for them;
- 3,000 random texts per file, made of the file's own variants (randomly upper-cased, accented, cut short or re-separated) glued by spaces, hyphens, apostrophes, elisions and titles;
- a synthetic corpus per file, generated at test time with a fixed seed: the file's own variants (a quarter of them mangled as above) placed in 15 invented French carrier sentences, at the start and end of the text, after an elision (`l'`, `d’`, `qu'`), next to punctuation, capitalised, after a title, and two side by side. Each file gets as many sentences as it has variants, at least 3,000 and at most 10,000 (manual 3,000, drug 10,000, dictionary 10,000, PARHAF 5,474, PARROT 3,000, acronyms 3,000), so every variant of every file but the drug and dictionary files is used at least once (possibly mangled). No real transcript is committed: to check real ones, point `ULTIMED_LOOKUP_CORPUS` at a corpus (a `.jsonl` with a `hyp` field, or a `.txt` with one text per line), optionally with `ULTIMED_LOOKUP_LIMIT`, and every file runs on it.
- the six files in order against one list made of them concatenated, regex and lookup alike, on 500 synthetic sentences per file, 1,000 random texts and the conflict-decision sentences (`test_concatenation`, see "One list?").

All pass with 0 mismatches (2026-11-15, rerun 2026-11-23 on the files with every import). On 20,093 v2 transcripts (every 30th line of the full corpus), the regex and lookup outputs were also identical for all six files (checked on the files as they were before the 2026-11-15 conflict decisions and convention respelling, not rerun since; manual 175 lines changed, drug 479, dictionary 3,419, PARHAF 559, PARROT 35, acronyms 12). The reference applier is slower than the indexed regexes in Python (about 0.8 to 1.9 ms per line per file, vs 0.1 to 1.2 ms for `compile_rules`); it is written for clarity, not speed.

## Tests

```bash
uv run tests/test_lookup.py
cd tests && uv run --with click --with loguru --with litellm --with tiktoken --with tqdm --with tenacity --with rapidfuzz --with wordfreq test_drug_asr_rules.py
```

`test_drug_asr_rules.py` skips the two checks that need the UltiMed-ASR-FR-v2-scripts checkout when it is missing.
