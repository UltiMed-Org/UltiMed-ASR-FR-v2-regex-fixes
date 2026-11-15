#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Post-process a learned rule file: drop the hand-decided variants, respell replacements.

1. Every variant of ``inputs/conflict_resolutions.tsv`` (matched by pattern, so accents and
   hyphens do not matter) is removed from the file: those variants got different
   replacements from different files, and the author's decision ships as manual rules
   instead (05). A merged rule loses just that variant and its pattern is rebuilt by 04's
   ``drop_unk_variants``; a rule left with no variant is dropped.
2. Each replacement goes through the label spelling conventions of
   UltiMed-ASR-FR-v1-scripts (``utils/label_conventions.py``, via 06's
   ``convention_fixes``, so only accent, hyphen and space respellings count): a learned rule
   must not write ``bévacizumab`` when the conventions, and the convention rules, say
   ``bevacizumab``. With ``--labels``, a respelling is kept only if the labels holding the
   old replacement, once converted, mostly hold the new one: the replacements are often
   lowercased (``caldeyro-barcia``), and the conventions glue a lowercase compound they
   leave alone in the labels' own case (``Caldeyro-Barcia``, ``anti-HBs``), like 06's
   label check. ``--no-conventions`` skips this.

Rules with the same replacement are not re-merged: a merge must be checked on texts (03).

Usage: uv run scripts/09_apply_resolutions.py SRC.jsonl DST.jsonl [--resolutions TSV] [--labels MANIFEST ...] [--no-conventions]
(SRC and DST may be the same file). Called by 07_build_dataset_rules.sh after 04.
This file was written by Claude Code.
"""
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b = _load("b", "02_build_fix_rules.py")
u = _load("u", "04_drop_unk_variants.py")
mr = _load("mr", "05_build_manual_rules.py")


def resolved_patterns(lines) -> set[str]:
    """The ``variant_pattern`` of every variant a resolutions TSV decides."""
    return {b.variant_pattern(row[0]) for row in mr.read_rows(lines)}


def drop_resolved(rules: list[dict], patterns: set[str]) -> tuple[list[dict], int]:
    """``(rules without the variants whose pattern is in patterns, variants dropped)``.

    >>> mk = lambda vs, t: u.m.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v in vs])[0][1]
    >>> rules = [mk(["lodose", "lodoz"], "Lodoz"), mk(["périos"], "périosté")]
    >>> out, n = drop_resolved(rules, resolved_patterns(["lodose\\t*\\tLodoz", "périos\\t*\\tpérioste"]))
    >>> n, [(r["variants"], r["pattern"] == b.variant_pattern("lodoz")) for r in out]
    (2, [(['lodoz'], True)])
    """
    return u.drop_unk_variants(rules, bad=lambda v: b.variant_pattern(v) in patterns)


def label_backed(respell: dict[str, str], labels, convert) -> set[str]:
    """The old replacements whose respelling the labels back: of the labels holding one
    (case-insensitively, as whole words: ``Patidexaméthasone`` does not count for
    ``dexaméthasone``), more than half hold the new spelling once ``convert``-ed. A
    replacement no label holds has no evidence against it and is kept.

    >>> glue = lambda t: t.replace("o-b", "ob").replace("é", "e")  # a convention that ignores case
    >>> sorted(label_backed({"caldeyro-barcia": "caldeyrobarcia", "bévacizumab": "bevacizumab", "x": "y"},
    ...                     ["Signe de Caldeyro-Barcia.", "sous bévacizumab", "Bévacizumab", "anti-bévacizumabs"], glue))
    ['bévacizumab', 'x']
    """
    folded = {old.casefold(): (old, re.compile(rf"(?<!\w){re.escape(old.casefold())}(?!\w)")) for old in respell}
    seen, ok = Counter(), Counter()
    for label in labels:
        low = label.casefold()
        hits = [old for f, (old, word) in folded.items() if f in low and word.search(low)]
        if hits:
            conv = convert(label).casefold()
            for old in hits:
                seen[old] += 1
                ok[old] += respell[old].casefold() in conv
    return {old for old in respell if 2 * ok[old] > seen[old] or not seen[old]}


def apply_conventions(rules: list[dict], labels=None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Respell each replacement as the label conventions do, in place (only those
    ``label_backed`` by ``labels`` when given); returns the ``(old, new)`` replacements
    changed, one per rule, and the respellings refused by the labels.

    >>> rules = [{"replacement": "caldeyro-barcia"}, {"replacement": "bévacizumab"}]
    >>> apply_conventions(rules, ["Signe de Caldeyro-Barcia.", "sous bévacizumab"])
    ([('bévacizumab', 'bevacizumab')], [('caldeyro-barcia', 'caldeyrobarcia')])
    """
    cv = _load("cv", "06_build_convention_fixes.py")  # needs UltiMed-ASR-FR-v1-scripts
    reps = {r["replacement"] for r in rules}
    respell = {rep: new for rep in reps if (new := cv.convention_fixes([rep]).get(rep, rep)) != rep}
    refused = []
    if labels is not None:
        backed = label_backed(respell, labels, cv.apply_label_conventions)
        refused = sorted((o, n) for o, n in respell.items() if o not in backed)
        respell = {o: n for o, n in respell.items() if o in backed}
    changed = []
    for r in rules:
        if r["replacement"] in respell:
            changed.append((r["replacement"], respell[r["replacement"]]))
            r["replacement"] = respell[r["replacement"]]
    return changed, refused


@click.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--resolutions", type=click.Path(exists=True, path_type=Path), default=mr.RESOLUTIONS, show_default=True)
@click.option("--labels", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="NeMo manifests (text field): keep only the respellings their converted labels back")
@click.option("--no-conventions", is_flag=True, help="do not respell the replacements")
def main(src: Path, dst: Path, resolutions: Path, labels: tuple[Path, ...], no_conventions: bool) -> None:
    rules = b.load_rules(src)
    n0 = len(rules)
    rules, dropped = drop_resolved(rules, resolved_patterns(resolutions.read_text(encoding="utf-8").splitlines()))
    logger.info(f"{src.name}: dropped {dropped} resolved variants, {n0} -> {len(rules)} rules")
    if not no_conventions:
        texts = (json.loads(line)["text"] for mf in labels for line in open(mf, encoding="utf-8")) if labels else None
        changed, refused = apply_conventions(rules, texts)
        logger.info(f"{len(changed)} replacements respelled by the label conventions: {sorted(set(changed))[:20]}")
        logger.info(f"{len(refused)} respellings refused, the labels do not back them: {refused[:20]}")
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rules), encoding="utf-8")
    logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
