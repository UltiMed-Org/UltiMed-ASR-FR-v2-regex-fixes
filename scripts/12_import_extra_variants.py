#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Import the variants other models' rule builds found into the v2 files, each into its own subset's file.

Why: the v2 rules are learned from one model's transcripts (the v2 fp32 blend). Other models of the family
(the w4a8 quantisation of v2, the redux fine-tune) misspell the same words in other ways, which v2 can also
produce on new audio. A build of 07 on such a model's transcripts of the whole corpus (build A, all splits,
the same guards as the release build) gives one ``rules.jsonl`` per subset folder; this script adds the
variants the committed files lack, as 11 does for the v1 rules but without the respelling and re-guarding
11 needs: these rules were built by the v2 scripts on v2 labels, so 02's guards and 09's passes already
apply to them.

1. **Read the builds** (``--build SOURCE=DIR``, highest priority first; ``DIR/<subset>/rules.jsonl``), each
   variant a single rule for its subset's file (``SUBSETS``: drugs -> drug, dictionary, PARHAF -> parhaf,
   PARROT -> parrot, acronyms), its count the one of ``rules.unmerged.jsonl``.
2. **Screen** (``screen_build``): ``existing`` (any of the six files, or an earlier build or subset of this
   run, already rewrites it: existing rules win, so no variant gets two replacements), ``noop``, ``unk``,
   ``charset`` (11's ``off_charset``), and ``min_count`` for a source given ``--min-count SOURCE=N``: a
   variant seen fewer than N times in its build is kept only if an earlier ``--build`` or the v1 rules
   (``--v1-dir``) also hold it (corroborated). Then 09's ``resolve`` (conflict decisions, extra-name title
   guard, label conventions, recasing to the labels) on what is left, with the current ``inputs/``.
3. to 5. 11's ``merge_import``: text check alone (no correct label rewritten, no v2 transcript of the corpus
   made worse), 03-checked merge per replacement into the file, tagged ``"source": "<SOURCE>"`` or
   ``"<SOURCE>_variants": [...]``, then the six files in order until no imported variant is involved in a
   rewritten label or a worse transcript.

Rerunning is safe: the variants of the ``--build`` sources are stripped first (other imports, v1's, stay).
Run it after 11 (which strips every import), then 08 for the lookups. Rejections go to ``--rejected``
(default ``rejected/extra_import.rejected.jsonl``, gitignored like the other rejected lists).

    uv run scripts/12_import_extra_variants.py --hyps full.blend70-onnx-fp32.hyp.jsonl \\
        --build v2-w4a8=.../v2-w4a8/A --build redux-v2=.../redux-v2-int8twin/A --min-count redux-v2=2 \\
        --merge-text .../fr-wikipedia-1M.txt --merge-text .../fr-news-2023-100K.txt --workers 6
    uv run scripts/08_build_lookup.py rules/*_fix_rules.jsonl

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
# _util.py is loaded by path, not imported: see its docstring.
_uspec = importlib.util.spec_from_file_location("ultimed_util", _HERE / "_util.py")
util = importlib.util.module_from_spec(_uspec)
_uspec.loader.exec_module(util)
r11 = util.load_script("r11", "11_import_v1_rules.py")
# 11 already loads 09 (and through it 02, 03, 04, 05): reuse those copies.
r9, u, b, mr, m = r11.r9, r11.u, r11.b, r11.mr, r11.m
variants_of = r11.variants_of

_REPO = _HERE.parent
SUBSETS = {"drugs": "drug", "dictionary": "dictionary", "PARHAF": "parhaf", "PARROT": "parrot",
           "acronyms": "acronyms"}  # build subset folder -> the v2 file its variants join, in application order


def singles(r: dict, counts: dict[str, int]) -> list[dict]:
    """One single-variant rule per variant of rule ``r`` (merged or not), pattern rebuilt by 04, count from
    ``counts`` (the build's unmerged file) when it has the variant; an unmerged rule is kept as it is.

    >>> mk = lambda vs, t: m.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v in vs])[0][1]
    >>> [(s["variant"], s["count"], s["pattern"] == b.variant_pattern(s["variant"])) for s in singles(mk(["aa bb", "cc"], "X"), {"cc": 3})]
    [('aa bb', 1, True), ('cc', 3, True)]
    """
    if "variants" not in r:
        return [r]
    out = []
    for v in r["variants"]:
        one = u.drop_unk_variants([r], bad=lambda x, v=v: x != v)[0][0]
        out.append({"variant": v, "pattern": one["pattern"], "replacement": r["replacement"],
                    "count": counts.get(v, 1), **({"name": True} if r.get("name") else {})})
    return out


def read_build(build_dir: Path) -> list[tuple[str, dict]]:
    """``(v2 file, single-variant rule)`` for every variant of a build's ``<subset>/rules.jsonl``, subsets in
    ``SUBSETS`` order (a missing subset folder is skipped)."""
    out = []
    for sub, n in SUBSETS.items():
        d = build_dir / sub
        if not (d / "rules.jsonl").exists():
            logger.warning(f"{d}: no rules.jsonl, skipped")
            continue
        counts = {}
        if (d / "rules.unmerged.jsonl").exists():
            counts = {r["variant"]: r.get("count", 1) for r in util.read_jsonl(d / "rules.unmerged.jsonl")}
        out += [(n, s) for r in util.read_jsonl(d / "rules.jsonl") for s in singles(r, counts)]
    return out


def screen_build(rules: list[tuple[str, dict]], source: str, claimed: set[tuple], corroborated: set[tuple],
                 min_count: int) -> tuple[list[dict], list[dict]]:
    """``(kept, rejected)`` of one build's ``(file, rule)`` pairs; ``claimed`` (the variant keys some rule
    already rewrites) grows with the kept ones, so the first build and subset to hold a variant win it.
    Kept rules carry ``file`` and ``source``, the bookkeeping ``merge_import`` keeps out of the files.

    >>> mk = lambda v, n: {"variant": v, "pattern": b.variant_pattern(v), "replacement": "X", "count": n}
    >>> rules = [("drug", mk("aa", 1)), ("dictionary", mk("aa", 5)), ("drug", mk("bb", 1)), ("drug", mk("cc", 1)),
    ...          ("drug", mk("dd", 2)), ("drug", mk("x", 3)), ("drug", mk("ee", 1))]
    >>> keep, rej = screen_build(rules, "redux-v2", {("ee",)}, {("aa",)}, 2)
    >>> [(r["file"], r["variant"]) for r in keep], [(r["variant"], r["reason"]) for r in rej]
    ([('drug', 'aa'), ('drug', 'dd')], [('aa', 'existing'), ('bb', 'min_count'), ('cc', 'min_count'), ('x', 'noop'), ('ee', 'existing')])
    """
    keep, rejected = [], []
    for n, r in rules:
        v = r["variant"]
        k = b._key(v)
        reason = ("existing" if k in claimed else "noop" if k == b._key(r["replacement"])
                  else "unk" if u.has_unk(v) else "charset" if r11.off_charset(v)
                  else "min_count" if r.get("count", 1) < min_count and k not in corroborated else None)
        if reason:
            rejected.append({"variant": v, "replacement": r["replacement"], "source": source, "file": n,
                             "reason": reason})
            continue
        claimed.add(k)
        keep.append(dict(r, file=n, source=source))
    return keep, rejected


def _pairs_option(values: tuple[str, ...], what: str) -> dict[str, str]:
    """``{"a": "b"}`` from ``("a=b",)``, in order."""
    out = {}
    for x in values:
        k, sep, v = x.partition("=")
        if not sep or not k or not v:
            raise click.BadParameter(f"{x!r} is not {what}")
        out[k] = v
    return out


@click.command()
@click.option("--hyps", "hyps_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="the v2 model's transcripts of every corpus clip (jsonl, fields audio + hyp): the text checks")
@click.option("--build", "builds", multiple=True, required=True,
              help="SOURCE=DIR: a 07 build folder (one subfolder per subset), highest priority first")
@click.option("--min-count", "min_counts", multiple=True,
              help="SOURCE=N: keep a variant of that build seen fewer than N times only if corroborated")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="correct v2 labels (NeMo manifests); default 02's: full.jsonl + PARROT/full.jsonl")
@click.option("--merge-text", "merge_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="extra texts 03's merge is checked on, besides the hyps and labels (.txt or manifest)")
@click.option("--v1-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=_REPO / "v1",
              show_default=True, help="the v1 rules: a variant they hold counts as corroborated")
@click.option("--rules-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=_REPO / "rules",
              show_default=True, help="the v2 rule files, read and rewritten in place")
@click.option("--rejected", "rejected_path", type=click.Path(dir_okay=False, path_type=Path),
              default=_REPO / "rejected" / "extra_import.rejected.jsonl", show_default=True)
@click.option("--workers", type=int, default=1, show_default=True,
              help="processes for the passes over every text (03's merge check, the text checks)")
def main(hyps_path: Path, builds: tuple[str, ...], min_counts: tuple[str, ...], labels_paths: tuple[Path, ...],
         merge_paths: tuple[Path, ...], v1_dir: Path, rules_dir: Path, rejected_path: Path, workers: int) -> None:
    """Add the variants of other models' builds to the v2 rule files, each to its subset's file."""
    build_dirs = {s: Path(d) for s, d in _pairs_option(builds, "SOURCE=DIR").items()}
    thresholds = {s: int(n) for s, n in _pairs_option(min_counts, "SOURCE=N").items()}
    if unknown := set(thresholds) - set(build_dirs):
        raise click.BadParameter(f"--min-count for sources without --build: {sorted(unknown)}")
    labels_paths = labels_paths or b.DEFAULT_LABELS
    labels = [t for p in labels_paths for t in util.read_texts(p, "text")]
    files = {n: util.read_jsonl(rules_dir / f"{n}_fix_rules.jsonl") for n in r11.V2_FILES}
    for n in r11.LEARNED:
        files[n] = r11.strip_imported(files[n], set(build_dirs))

    # 1 and 2. read and screen each build, in priority order
    claimed = {b._key(v) for rs in files.values() for r in rs for v in variants_of(r)}
    corroborated = {b._key(v) for p in sorted(v1_dir.glob("*_fix_rules.jsonl")) for r in util.read_jsonl(p)
                    for v in variants_of(r)}
    keep, rejected = [], []
    for source, d in build_dirs.items():
        rules = read_build(d)
        k, rej = screen_build(rules, source, claimed, corroborated, thresholds.get(source, 1))
        logger.info(f"{source}: {len(rules)} variants, {len(k)} new kept, rejected {dict(Counter(r['reason'] for r in rej))}")
        if source in thresholds:
            low = [r for r in k if r.get("count", 1) < thresholds[source]]
            logger.info(f"{source}: of the kept, {len(k) - len(low)} seen >= {thresholds[source]} times, "
                        f"{len(low)} below but corroborated")
        keep += k
        rejected += rej
        corroborated |= {b._key(r["variant"]) for _, r in rules}
    keep = r9.resolve(keep, mr.RESOLUTIONS.read_text(encoding="utf-8").splitlines(),
                      r9.EXTRA_NAMES.read_text(encoding="utf-8").splitlines(), labels, name="extra import")

    # 3 to 5: 11's check, merge, check in order
    pairs = r11.read_pairs(hyps_path, tuple(Path(p) for p in labels_paths))
    logger.info(f"{len(pairs)} (label, transcript) pairs")
    new = {n: [r for r in keep if r["file"] == n] for n in r11.LEARNED}
    imported, rej = r11.merge_import(files, new, {r["variant"]: r["source"] for r in keep}, labels, pairs,
                                     merge_paths, ("file", "source"), workers)
    rejected += rej

    for n in r11.LEARNED:
        util.write_jsonl(rules_dir / f"{n}_fix_rules.jsonl", files[n])
        logger.info(f"wrote {rules_dir / f'{n}_fix_rules.jsonl'}: {len(files[n])} rules, "
                    f"{sum(len(variants_of(r)) for r in files[n])} variants")
    by_source = Counter(r["source"] for r in keep if r["variant"] in imported)
    by_file = Counter((r["source"], r["file"]) for r in keep if r["variant"] in imported)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    util.write_jsonl(rejected_path, rejected)
    logger.info(f"{len(imported)} variants imported {dict(by_source)} {dict(by_file)}; rejected {len(rejected)} -> "
                f"{rejected_path}: {dict(Counter((r.get('source'), r['reason']) for r in rejected).most_common())}")


if __name__ == "__main__":
    main()
