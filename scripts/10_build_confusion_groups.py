#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
# (dependencies: those of 02_build_fix_rules.py and 09_apply_resolutions.py, which this imports)
"""Export the heard forms that stand for SEVERAL correct words, instead of dropping them.

A regex maps one heard form to one replacement, so 02 rejects a variant whose errors split
between targets (``ambiguous``: no target holds ``--min-share`` of them) or that the model
also writes when it is right (``imprecise``). Those are exactly the cases an acoustic second
pass can decide: ``etuximab`` was ``cétuximab`` in one clip and ``rituximab`` in another,
and only the audio around the span tells which. This script collects, from one or more
build folders of ``07_build_dataset_rules.sh``, every folded heard form (02's ``_key``)
with at least two observed targets, and writes one group per line::

    {"heard": "etuximab",
     "candidates": [{"text": "cétuximab", "count": 12}, {"text": "rituximab", "count": 7}],
     "heard_correct_count": 0, "kind": "drug", "sources": ["drugs", "PARHAF"],
     "regex_fallback": "cétuximab"}

- ``heard``: the variant as 01 writes it from the hypothesis (lowercased, punctuation
  dropped, elisions re-glued; the most frequent spelling of the folded form, like 02's
  ``variant``). The rule patterns match it case- and accent-insensitively, a consumer
  should too.
- ``candidates``: every target 01 attributed to it, error counts summed over the folders,
  count descending. Targets that only differ in accents or case are one candidate (01 does
  not count such a slip as an error, so they cannot be told apart by ear either). The
  spelling is the final label spelling: each target goes through 09's ``apply_conventions``
  and ``recase_to_labels``, exactly as a learned rule's replacement does.
- ``heard_correct_count``: hypothesis occurrences of the heard form (02's ``count_ngrams``,
  the denominator of 02's ``precision``) minus its error count, summed over the folders:
  how often the model wrote it where the label had no lexicon word to correct.
- ``kind``: ``drug`` when every candidate's lexicon word (the 01 report entry it came
  from) is in the drug lexicon (01's ``load_lexicon`` without ``--lexicon``) and is rarer
  in French than 01's ``MAX_TERM_ZIPF``, else ``term``. The Zipf test is there because the
  drug lexicon also holds the everyday words of brand presentations (``enfants`` from
  "Advil enfants"); the subset alone would not do, since drug names also appear in the
  dictionary and PARHAF sentences.
- ``sources``: the build folders (named after their subset) that contributed a candidate.
- ``regex_fallback``: what the committed rule files (``--rules``, applied in order with
  02's ``compile_rules``) turn the bare heard form into today, or null when no rule changes
  it. A non-null value means a regex already picks one candidate.

The output holds no transcript text and no clip id, only counts and the heard / candidate
strings. ``--examples-out`` additionally writes, OUTSIDE the repo (refused otherwise),
example rows ``{"heard", "ref_target", "hyp", "ref", "clip", "split"}``: up to
``--examples-per-candidate`` per candidate and split, so the test split's rows can serve as
a held-out benchmark for groups mined from train+val builds. They are recovered by re-running
01's ``drug_outcomes`` on the clips whose hypothesis holds a heard form, with each folder's
manifests (``<manifests dir>/<folder name>/{train,val,test}.jsonl``) giving the label and
the split.

The heard forms belong to the model that produced the hypotheses: regenerate this file for
every new model.

Usage::

    uv run scripts/10_build_confusion_groups.py BUILD_DIR [BUILD_DIR ...] \\
        [--out rules/confusion_groups.jsonl] [--examples-out /outside/repo/examples.jsonl]

Pass builds of disjoint clips (the subsets of one build A, say): the counts of the folders
are added, so a clip present in two folders counts twice.

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
# _util.py is loaded by path, not imported: see its docstring.
_uspec = importlib.util.spec_from_file_location("ultimed_util", _HERE / "_util.py")
util = importlib.util.module_from_spec(_uspec)
_uspec.loader.exec_module(util)
REPO = _HERE.parent
# 09 carries the whole import chain this script needs: 09 -> 04 (``ar.u``) -> 02 (``ar.b``)
# -> 01 (``ar.b.extract``), each loaded once, plus 09's own label respelling helpers.
ar = util.load_script("ar", "09_apply_resolutions.py")
b = ar.b
extract = b.extract

DEFAULT_OUT = REPO / "rules" / "confusion_groups.jsonl"
# The committed rule files in the order every published number applies them (README).
DEFAULT_RULES = tuple(REPO / "rules" / f"{n}_fix_rules.jsonl"
                      for n in ("drug", "dictionary", "parhaf", "parrot", "acronyms", "manual"))
DEFAULT_MANIFESTS = extract.SCRIPTS_DIR / "99_hf_release" / "data" / "NeMO_files"
SPLITS = ("train", "val", "test")


def read_build(folder: Path) -> tuple[dict, list[dict]]:
    """``(01 report, hypothesis rows)`` of one build folder (``errors.json``, ``hyps.jsonl``)."""
    report = json.loads((folder / "errors.json").read_text(encoding="utf-8"))
    return report, util.read_jsonl(folder / "hyps.jsonl")


def collect_groups(builds: list[tuple[str, dict, list[str]]]) -> dict[tuple, dict]:
    """Heard forms with two or more distinct targets, aggregated over ``builds``.

    ``builds`` holds ``(source name, 01 report, hypothesis texts)``. Returns, per folded heard
    form (02's ``_key``): ``spelling`` (raw spelling counts), ``targets`` (folded target ->
    Counter of raw target spellings weighted by error count), ``drugs`` (folded target ->
    report entries), ``sources`` (folded target -> source names), ``errors`` and
    ``hyp_occurrences`` (summed over the builds).

    >>> rep = {"cétuximab": {"errors": {"etuximab": 3}, "targets": {}},
    ...        "rituximab": {"errors": {"etuximab": 2, "rituxi": 1}, "targets": {}}}
    >>> g = collect_groups([("drugs", rep, ["sous etuximab", "etuximab", "etuximab", "etuximab",
    ...                                     "etuximab", "etuximab"])])
    >>> list(g), dict(g[("etuximab",)]["targets"][("cetuximab",)]), g[("etuximab",)]["hyp_occurrences"]
    ([('etuximab',)], {'cétuximab': 3}, 6)
    """
    per_build = [(src, b.group_variants(report)) for src, report, _ in builds]
    targets_of: dict[tuple, set] = defaultdict(set)
    for _, (by_key, *_rest) in per_build:
        for k, targets in by_key.items():
            targets_of[k].update(b._key(t) for t in targets)
    keys = {k for k, ts in targets_of.items() if len(ts) > 1}
    groups: dict[tuple, dict] = {k: {"spelling": Counter(), "targets": defaultdict(Counter), "drugs": defaultdict(set),
                                     "sources": defaultdict(set), "errors": 0, "hyp_occurrences": 0} for k in keys}
    for (src, (by_key, spelling, drug_of, _clips)), (_, _, hyps) in zip(per_build, builds):
        mine = keys & by_key.keys()
        # 02's precision denominator: how often the folded heard form occurs in the hypotheses.
        hits = b.count_ngrams(hyps, mine)
        for k in mine:
            g = groups[k]
            g["spelling"].update(spelling[k])
            g["errors"] += sum(by_key[k].values())
            g["hyp_occurrences"] += hits[k]
            for t, n in by_key[k].items():
                tk = b._key(t)
                g["targets"][tk][t] += n
                g["drugs"][tk].add(drug_of[k][t])
                g["sources"][tk].add(src)
    return groups


def final_spellings(targets: set[str], label_texts: list[str] | None) -> dict[str, str]:
    """Each target respelled and recased as 09 does a learned rule's replacement.

    Without ``label_texts`` the targets are kept as they are (09 needs the labels to back
    a respelling and to learn the casing).
    """
    if label_texts is None:
        return {t: t for t in targets}
    rows = [{"replacement": t, "orig": t} for t in sorted(targets)]
    ar.apply_conventions(rows, label_texts)
    ar.recase_to_labels(rows, label_texts)
    return {r["orig"]: r["replacement"] for r in rows}


def drug_kind(words: set[str], drug_lex: set[str]) -> str:
    """``drug`` when every lexicon word is a drug-lexicon word rarer in French than 01's
    ``MAX_TERM_ZIPF`` (so ``enfants``, a drug-lexicon word from "Advil enfants", is not).

    >>> drug_kind({"cétuximab", "rituximab"}, {"cetuximab", "rituximab", "enfants"})
    'drug'
    >>> drug_kind({"cétuximab", "enfants"}, {"cetuximab", "enfants"})
    'term'
    """
    from wordfreq import zipf_frequency
    return "drug" if all(extract.fold(w) in drug_lex and zipf_frequency(w.lower(), "fr") < extract.MAX_TERM_ZIPF
                         for w in words) else "term"


def regex_fallback(heard: str, fixes) -> str | None:
    """What the rule files (one compiled ``fix`` each, in order) make of ``heard``, or None.

    >>> rules = [{"variant": "etuximab", "pattern": b.variant_pattern("etuximab"), "replacement": "cétuximab"}]
    >>> regex_fallback("etuximab", [b.compile_rules(rules)]), regex_fallback("rituximab", [b.compile_rules(rules)])
    ('cétuximab', None)
    """
    text = heard
    for fix in fixes:
        text = fix(text)
    return None if text == heard else text


def homophone_key(word: str) -> str:
    """``word`` without the French plural and feminine endings that are not pronounced.

    Two words with the same key sound the same, so audio cannot choose between them:
    ``abdominale`` / ``abdominales`` / ``abdominal``, ``résiduelle`` / ``résiduel``,
    ``opérée`` / ``opéré``, ``inférieure`` / ``inférieur``. Endings that ARE heard keep
    their own key: ``pelvienne`` / ``pelvien``, ``active`` / ``actif``, ``cervicaux`` /
    ``cervical``, ``nerveuse`` / ``nerveux``.

    >>> [homophone_key(w) for w in ("abdominales", "abdominal", "résiduelles", "opérées", "aiguë")]
    ['abdominal', 'abdominal', 'résiduel', 'opéré', 'aigu']
    >>> homophone_key("pelvienne") == homophone_key("pelvien"), homophone_key("cervicaux") == homophone_key("cervical")
    (False, False)
    """
    w = word.lower()
    if len(w) > 3 and w[-1] in "sx" and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == "e" and w[-2] in "éiulr":
        w = w[:-1]
        if w.endswith("ll"):
            w = w[:-1]
    elif len(w) > 3 and w.endswith("uë"):
        w = w[:-1]
    return w


def drop_homophones(heard: str, counts: Counter, *merged: dict[str, set]) -> Counter:
    """``counts`` without the candidates that only differ from ``heard`` by a silent ending
    (the audio cannot tell them apart; agreement is a grammar checker's job), and with
    candidates that sound like each other merged into the most frequent one (counts summed,
    and each ``merged`` dict's sets united into it).

    >>> drop_homophones("abdominale", Counter({"abdominales": 630, "abdominal": 12}))
    Counter()
    >>> drop_homophones("pelviennes", Counter({"pelvienne": 9, "pelviens": 5, "pelvien": 3}))
    Counter({'pelviens': 8})
    """
    hk = homophone_key(heard)
    out: Counter = Counter()
    owner: dict[str, str] = {}
    for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        k = homophone_key(t)
        if k == hk:
            continue
        o = owner.setdefault(k, t)
        out[o] += n
        if o != t:
            for d in merged:
                d[o] |= d[t]
    return out


def build_groups(groups: dict[tuple, dict], spellings: dict[str, str], drug_lex: set[str], fixes,
                 min_candidate_count: int = 1, min_total: int = 3) -> list[dict]:
    """The output rows, most total occurrences first (see the module docstring for the fields).

    Candidates whose final spelling coincides are merged; then ``drop_homophones`` removes
    the candidates that only differ from the heard form by a silent plural or feminine
    ending and merges the ones that sound alike; a candidate below ``min_candidate_count``
    is dropped, then a group needs two candidates and ``min_total`` errors over them.
    """
    out = []
    for k, g in groups.items():
        counts: Counter = Counter()
        drugs: dict[str, set] = defaultdict(set)
        sources: dict[str, set] = defaultdict(set)
        for tk, raw in g["targets"].items():
            text = spellings[raw.most_common(1)[0][0]]
            counts[text] += sum(raw.values())
            drugs[text] |= g["drugs"][tk]
            sources[text] |= g["sources"][tk]
        heard = g["spelling"].most_common(1)[0][0]
        counts = drop_homophones(heard, counts, drugs, sources)
        cands = [(t, n) for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n >= min_candidate_count]
        if len(cands) < 2 or sum(n for _, n in cands) < min_total:
            continue
        out.append({"heard": heard,
                    "candidates": [{"text": t, "count": n} for t, n in cands],
                    "heard_correct_count": max(0, g["hyp_occurrences"] - g["errors"]),
                    "kind": drug_kind(set().union(*(drugs[t] for t, _ in cands)), drug_lex),
                    "sources": sorted(set().union(*(sources[t] for t, _ in cands))),
                    "regex_fallback": regex_fallback(heard, fixes)})
    out.sort(key=lambda r: (-sum(c["count"] for c in r["candidates"]), r["heard"]))
    return out


def collect_examples(folder: Path, rows: list[dict], report: dict, wanted: dict[tuple, dict[tuple, str]],
                     manifests_dir: Path, per_candidate: int, sink: dict) -> None:
    """Add to ``sink`` (``(heard key, candidate text, split)`` -> rows) the clips of ``folder``
    where 01's ``drug_outcomes`` maps a wanted heard form onto one of its candidates.

    ``wanted``: folded heard form -> {folded target: final candidate text}. Only the
    hypotheses holding a wanted heard form are aligned, and 01 runs with a lexicon of just
    the report entries those groups involve: ``drug_outcomes`` treats each lexicon word on
    its own, so the outcomes for those words are the same as with the full lexicon.
    """
    sys.path.insert(0, str(extract.SCRIPTS_DIR / "utils"))
    from nemo_manifest import clip_key, read_jsonl, resolve_audio
    label_of: dict[str, tuple[str, str]] = {}
    for split in SPLITS:
        mf = manifests_dir / folder.name / f"{split}.jsonl"
        if mf.exists():
            for row in read_jsonl(mf):
                label_of[clip_key(resolve_audio(mf.parent, row["audio_filepath"]))] = (row["text"], split)
    by_key, _, drug_of, _ = b.group_variants(report)
    lex = {extract.fold(drug_of[k][t]): None for k in wanted.keys() & by_key.keys() for t in by_key[k]}
    for r in rows:
        key = clip_key(r["audio"])
        if key not in label_of or not b.count_ngrams([r["hyp"]], wanted.keys()):
            continue
        label, split = label_of[key]
        for _, _, variant, target in extract.drug_outcomes(label, r["hyp"], lex):
            hk = b._key(variant) if variant else None
            cand = wanted.get(hk, {}).get(b._key(target))
            if cand is None:
                continue
            bucket = sink.setdefault((hk, cand, split), [])
            if len(bucket) < per_candidate:
                bucket.append({"ref_target": cand, "hyp": r["hyp"], "ref": label, "clip": key, "split": split})


def inside_repo(path: Path) -> bool:
    """True when ``path`` (resolved, symlinks followed) lies inside this repository."""
    return path.resolve().is_relative_to(REPO.resolve())


@click.command()
@click.argument("folders", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out", default=str(DEFAULT_OUT), type=click.Path(path_type=Path), show_default="rules/confusion_groups.jsonl")
@click.option("--rules", "rule_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="rule files giving regex_fallback, applied in the order given (default: the six committed "
                   "files in README order, manual last)")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="manifests whose labels back the final candidate spellings (09's respelling and recasing); "
                   "default 02's DEFAULT_LABELS")
@click.option("--no-respell", is_flag=True, help="keep the targets as 01 wrote them (no label loading)")
@click.option("--min-candidate-count", default=1, show_default=True, help="errors a candidate needs to be kept")
@click.option("--min-total", default=3, show_default=True, help="errors a group needs over its kept candidates")
@click.option("--examples-out", type=click.Path(path_type=Path), default=None,
              help="also write example rows (transcripts, clip ids) here; must be outside the repo")
@click.option("--examples-per-candidate", default=5, show_default=True, help="example rows per candidate and split")
@click.option("--manifests-dir", default=str(DEFAULT_MANIFESTS), type=click.Path(path_type=Path),
              show_default="$ULTIMED_SCRIPTS_DIR/99_hf_release/data/NeMO_files",
              help="holds <folder name>/{train,val,test}.jsonl, for the examples' labels and splits")
def main(folders: tuple[Path, ...], out: Path, rule_paths: tuple[Path, ...], labels_paths: tuple[Path, ...],
         no_respell: bool, min_candidate_count: int, min_total: int, examples_out: Path | None,
         examples_per_candidate: int, manifests_dir: Path) -> None:
    """Write the confusion groups of the build FOLDERS (see the module docstring)."""
    if examples_out is not None and inside_repo(examples_out):
        raise click.UsageError(f"--examples-out {examples_out} is inside the repo: examples hold transcripts, "
                               "write them elsewhere")
    builds = []
    for folder in folders:
        report, rows = read_build(folder)
        builds.append((folder.name, report, rows))
    groups = collect_groups([(src, report, [r["hyp"] for r in rows]) for src, report, rows in builds])
    logger.info(f"{len(groups)} folded heard forms have 2+ targets in {', '.join(f.name for f in folders)}")
    targets = {t for g in groups.values() for raw in g["targets"].values() for t in raw}
    label_texts = None
    if not no_respell:
        label_texts = [t for p in labels_paths or b.DEFAULT_LABELS for t in util.read_texts(p, "text")]
    spellings = final_spellings(targets, label_texts)
    drug_lex = set(extract.load_lexicon())
    fixes = [b.compile_rules(util.read_jsonl(p)) for p in rule_paths or DEFAULT_RULES]
    rows_out = build_groups(groups, spellings, drug_lex, fixes, min_candidate_count, min_total)
    out.parent.mkdir(parents=True, exist_ok=True)
    util.write_jsonl(out, rows_out)
    logger.info(f"wrote {len(rows_out)} groups to {out}")

    kinds = Counter(r["kind"] for r in rows_out)
    with_fallback = sum(r["regex_fallback"] is not None for r in rows_out)
    print(f"\n{len(rows_out)} groups: {dict(kinds)}; {with_fallback} have a regex_fallback (a committed rule fires)")
    print("top 20 by total occurrences:")
    for r in rows_out[:20]:
        cands = ", ".join(f"{c['text']} {c['count']}" for c in r["candidates"])
        print(f"  {r['heard']!r}: {cands} | heard_correct {r['heard_correct_count']} | {r['kind']} "
              f"| fallback {r['regex_fallback']!r} | {','.join(r['sources'])}")

    if examples_out is not None:
        # Folded heard form -> folded target -> final candidate text, for the kept groups only.
        kept = {b._key(r["heard"]) for r in rows_out}
        wanted = {k: {tk: spellings[raw.most_common(1)[0][0]] for tk, raw in groups[k]["targets"].items()}
                  for k in kept}
        heard_of = {b._key(r["heard"]): r["heard"] for r in rows_out}
        sink: dict = {}
        for folder, (_, report, rows) in zip(folders, builds):
            collect_examples(folder, rows, report, wanted, manifests_dir, examples_per_candidate, sink)
        examples_out.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(sink.items(), key=lambda x: (heard_of[x[0][0]], x[0][1], x[0][2]))
        util.write_jsonl(examples_out, ({"heard": heard_of[hk], **e} for (hk, _, _), ex in ordered for e in ex))
        n_split = Counter(split for (_, _, split), ex in sink.items() for _ in ex)
        logger.info(f"wrote {sum(n_split.values())} example rows to {examples_out} (per split: {dict(n_split)})")


if __name__ == "__main__":
    main()
