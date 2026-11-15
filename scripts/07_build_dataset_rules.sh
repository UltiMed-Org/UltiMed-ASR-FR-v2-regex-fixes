#!/usr/bin/env bash
# Build one subset's fix rules from a whole-corpus transcription: 01 -> 02 -> (03) -> 04 -> 09 -> 08.
# The steps and their guards are the same as the committed v1 files (see README), only the
# input changes: ONE hyps file for every clip of the corpus (e.g. the v2 model on all 602k clips),
# restricted here to one subset and a set of splits by 01 --hyps-out, so each subset gets its own
# rule list instead of one merged blob.
#
# Usage: 07_build_dataset_rules.sh <hyps.jsonl> <out dir> <subset> <split> [<split> ...]
#   subset: a folder of $D (drugs, dictionary, PARHAF, PARROT, acronyms); splits: train val test.
#   The real-text labels (02 --labels, 04 --labels) are the release-wide $D/<split>.jsonl of the SAME
#   splits, so a held-out build (train val) never sees the test labels either.
# Env (all optional):
#   ULTIMED_SCRIPTS_DIR  UltiMed-ASR-FR-v1-scripts checkout (default: a sibling of this repo), read by
#               01/02/06/09 too (drug lexicon, casing table, manifest helpers)
#   D           manifests dir (default $ULTIMED_SCRIPTS_DIR/99_hf_release/data/NeMO_files)
#   LEXICON     extra term list for 01/02 --lexicon (unset = drug names only, as drug_fix_rules.jsonl;
#               01_dictionnary/original_dictionnary.jsonl = the term rules)
#   GUARDS      .txt files of correct outside text that block variants in 02 (e.g. French Wikipedia)
#   NAMES       name lists for 02 --names (a rule that is a name does not fire after a title)
#   NEWS_GUARD  .txt files for 04 --guard (variants found there are dropped after the build)
#   MERGE_TEXTS when set (space-separated hyps .jsonl / manifests / .txt), 03 merges the rules and
#               checks the merge on these texts plus the subset's hyps; unset = no merge
# Outputs in <out dir>: errors.json (01), hyps.jsonl (the subset's hyps), rules.unmerged.jsonl and
# rules.unmerged.rejected.jsonl (02), rules.merged.jsonl (03, if MERGE_TEXTS), rules.unresolved.jsonl (04),
# rules.jsonl (09, final: inputs/conflict_resolutions.tsv variants removed, replacements respelled by
# the label conventions where the labels back it), rules.lookup.json (08, the lookup-table form of rules.jsonl), build.log (every step's full output).
# Written with Claude Code.
set -euo pipefail
HYPS=$(readlink -f "$1") OUT=$(readlink -m "$2") SUB=$3; shift 3
HERE=$(dirname "$(readlink -f "$0")")
SCRIPTS_DIR=$(readlink -f "${ULTIMED_SCRIPTS_DIR:-$HERE/../../UltiMed-ASR-FR-v1-scripts}")
export ULTIMED_SCRIPTS_DIR=$SCRIPTS_DIR
D=$(readlink -f "${D:-$SCRIPTS_DIR/99_hf_release/data/NeMO_files}")
mkdir -p "$OUT"; : > "$OUT/build.log"; cd "$HERE"
mfs=() labels=() lex=() names=() guards=()
for sp in "$@"; do mfs+=("$D/$SUB/$sp.jsonl"); labels+=(--labels "$D/$sp.jsonl"); done
[ -n "${LEXICON:-}" ] && lex=(--lexicon "$(readlink -f "$LEXICON")")
for g in ${GUARDS:-}; do guards+=(--labels "$(readlink -f "$g")"); done
for f in ${NAMES:-}; do names+=(--names "$(readlink -f "$f")"); done
echo "== $SUB [$*] 01 $(date '+%F %T')"
uv run ./01_extract_drug_errors.py "$HYPS" "${mfs[@]}" "${lex[@]}" \
  --out "$OUT/errors.json" --hyps-out "$OUT/hyps.jsonl" 2>&1 | tee -a "$OUT/build.log" | grep -E "lexicon|clips|wrote"
echo "== 02 $(date '+%F %T')"
uv run ./02_build_fix_rules.py "$OUT/hyps.jsonl" --errors "$OUT/errors.json" \
  "${labels[@]}" "${guards[@]}" "${names[@]}" "${lex[@]}" --out "$OUT/rules.unmerged.jsonl" 2>&1 | tee -a "$OUT/build.log" | grep -E "covering|names|wrote"
src=$OUT/rules.unmerged.jsonl
if [ -n "${MERGE_TEXTS:-}" ]; then
  echo "== 03 $(date '+%F %T')"
  pool=(--hyps "$OUT/hyps.jsonl")
  for t in $MERGE_TEXTS; do
    t=$(readlink -f "$t")
    if head -1 "$t" | grep -q '"hyp"'; then pool+=(--hyps "$t"); else pool+=(--labels "$t"); fi
  done
  uv run ./03_merge_rules.py "$src" "$OUT/rules.merged.jsonl" "${pool[@]}" 2>&1 | tee -a "$OUT/build.log" | grep -E "rules|split back"
  src=$OUT/rules.merged.jsonl
fi
echo "== 04 $(date '+%F %T')"
ng=(); for g in ${NEWS_GUARD:-}; do ng+=(--guard "$(readlink -f "$g")"); done
uv run ./04_drop_unk_variants.py "$src" "$OUT/rules.unresolved.jsonl" "${labels[@]}" "${ng[@]}" 2>&1 | tee -a "$OUT/build.log" | grep -E "dropped|occur"
echo "== 09 $(date '+%F %T')"
uv run ./09_apply_resolutions.py "$OUT/rules.unresolved.jsonl" "$OUT/rules.jsonl" "${labels[@]}" 2>&1 | tee -a "$OUT/build.log" | grep -E "dropped|respelled|refused"
echo "== 08 $(date '+%F %T')"
uv run ./08_build_lookup.py "$OUT/rules.jsonl" --out "$OUT/rules.lookup.json" 2>&1 | tee -a "$OUT/build.log" | grep -E "keys|cannot"
echo "== done $SUB $(date '+%F %T')"
