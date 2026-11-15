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
   must not write ``protéase`` when the labels, and the convention rules, say ``protease``.
   ``--no-conventions`` skips this.

Rules with the same replacement are not re-merged: a merge must be checked on texts (03).

Usage: uv run scripts/09_apply_resolutions.py SRC.jsonl DST.jsonl [--resolutions TSV] [--no-conventions]
(SRC and DST may be the same file). Called by 07_build_dataset_rules.sh after 04.
This file was written by Claude Code.
"""
import importlib.util
import json
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


def apply_conventions(rules: list[dict]) -> list[tuple[str, str]]:
    """Respell each replacement as the label conventions do, in place; returns the
    ``(old, new)`` replacements changed."""
    cv = _load("cv", "06_build_convention_fixes.py")  # needs UltiMed-ASR-FR-v1-scripts
    changed, cache = [], {}
    for r in rules:
        rep = r["replacement"]
        if rep not in cache:
            cache[rep] = cv.convention_fixes([rep]).get(rep, rep)
        if cache[rep] != rep:
            changed.append((rep, cache[rep]))
            r["replacement"] = cache[rep]
    return changed


@click.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--resolutions", type=click.Path(exists=True, path_type=Path), default=mr.RESOLUTIONS, show_default=True)
@click.option("--no-conventions", is_flag=True, help="do not respell the replacements")
def main(src: Path, dst: Path, resolutions: Path, no_conventions: bool) -> None:
    rules = b.load_rules(src)
    n0 = len(rules)
    rules, dropped = drop_resolved(rules, resolved_patterns(resolutions.read_text(encoding="utf-8").splitlines()))
    logger.info(f"{src.name}: dropped {dropped} resolved variants, {n0} -> {len(rules)} rules")
    if not no_conventions:
        changed = apply_conventions(rules)
        logger.info(f"{len(changed)} replacements respelled by the label conventions: {sorted(set(changed))[:20]}")
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rules), encoding="utf-8")
    logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
