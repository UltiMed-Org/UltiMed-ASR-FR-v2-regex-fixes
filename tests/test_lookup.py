#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""The lookup form of the rules (``lookup/``, ``scripts/apply_lookup.py``) gives exactly the
output of the regex rules (``rules/``, 02_build_fix_rules.py ``apply_rules``).

- ``test_doctests``: 08_build_lookup.py's doctests.
- ``test_lookups_current``: every ``lookup/*.lookup.json`` is what 08 builds from today's
  ``rules/`` file (so a rule edit without a rebuild fails here).
- ``test_edge_cases``: hand-written rules and texts aimed at each part of the patterns
  (elisions, apostrophe whitespace, hyphens, title guard, case and accent folding, the
  first-alternative-wins order, chained rules, keys ending on an apostrophe).
- ``test_fuzz``: random texts stitched from the real variants with random separators,
  apostrophes, case and accents, against all six rule files.
- ``test_corpus``: the committed sample of real v2-model transcripts
  (``tests/fixtures/hyps_sample.txt``), against all six rule files. Set
  ``ULTIMED_LOOKUP_CORPUS`` to a hyps ``.jsonl`` (field ``hyp``) or a ``.txt`` (one text per
  line) to run the whole file instead, e.g. the 602k-clip whole-corpus transcription;
  ``ULTIMED_LOOKUP_LIMIT=N`` keeps the first N texts.

Each prints lines, changed lines and mismatches per rule file and fails on any mismatch.

    uv run tests/test_lookup.py

This file was written by Claude Code.
"""
from __future__ import annotations

import doctest
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = ("manual", "drug", "dictionary", "parhaf", "parrot", "acronyms")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


build = _load("build_lookup", "08_build_lookup.py")
apply = _load("apply_lookup", "apply_lookup.py")
b = build.b


def _rules(name: str) -> list[dict]:
    return [json.loads(line) for line in (REPO / "rules" / f"{name}_fix_rules.jsonl").open(encoding="utf-8")]


def _lookup_of(rules: list[dict], tmp: Path) -> "apply.Lookup":
    """The Lookup 08 builds from in-memory ``rules`` (through the file format)."""
    lk, bad = build.build_lookup(rules)
    assert not bad, bad
    build.write_lookup(lk, tmp)
    return apply.Lookup.load(tmp)


def _compare(label: str, texts: list[str], ref, lk: "apply.Lookup") -> None:
    """Fail if ``lk.fix`` and ``ref`` disagree on any text; print the counts either way."""
    bad, changed = [], 0
    for t in texts:
        r, g = ref(t), lk.fix(t)
        changed += r != t
        if r != g:
            bad.append((t, r, g))
    print(f"{label}: {len(texts)} lines, {changed} changed, {len(bad)} mismatches")
    assert not bad, bad[:5]


def test_doctests() -> None:
    failed, attempted = doctest.testmod(build)
    assert attempted and not failed, (failed, attempted)
    print(f"08_build_lookup.py: OK ({attempted} cases)")


def test_lookups_current() -> None:
    import tempfile
    for name in FILES:
        src = REPO / "rules" / f"{name}_fix_rules.jsonl"
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "x.json"
            build.main.main([str(src), "--out", str(out)], standalone_mode=False)
            assert out.read_bytes() == build.lookup_path(src).read_bytes(), f"{name}: rerun 08_build_lookup.py"
    print("lookup/: up to date")


def _edge_rules() -> list[dict]:
    pairs = [("mire taz apine", "mirtazapine"), ("myrtazapine", "mirtazapine"), ("hui", "HUI"),
             ("aménorée", "aménorrhée"), ("d'hexaméthasone", "dexaméthasone"), ("prim peran", "Primpéran"),
             ("mirat", "Humira"), ("sko", "SKO"), ("lasilix", "Lasilix"), ("Lasilix", "furosémide"),
             ("ana", "ANA"), ("ana na", "ananas"), ("d'", "de"), ("jusqu' à", "jusqu'à"), ("œdème", "oedème"),
             ("pã", "pa"), ("ad", "AD"), ("ad hoc", "ad-hoc")]
    rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v, t in pairs]
    b.guard_names(rules, {"mirat"})
    merge = _load("merge", "03_merge_rules.py")
    # first alternative wins: "ana" listed before "ana na" in one merged rule
    merged = merge.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": "X"}
                                for v in ("ana", "ana na", "anana")])
    return rules + [r for _, r in merged]


EDGE_TEXTS = [
    "Arrêt de la Mire-taz  apine.", "MYRTAZAPINE puis prim-peran", "aujourd'hui", "prud'hui", "jusqu'hui",
    "12 semaines d'aménorée", "L’aménorée", "l'AMENOREE", "s’ aménorée", "qu'aménorée", "xd'aménorée",
    "sous d' hexamethasone", "sous d'- hexamethasone", "sous d'\thexaméthasone", "d’Hexaméthasone",
    "le docteur Mirat", "Mme Mirat", "Dr. Mirat", "M. Mirat est venu", "sous mirat depuis mai", "docteur  Mirat",
    "xdocteur Mirat", "DOCTEUR Mirat", "KSO et sko, ſko", "lasilix 40", "Lasilix-Lasilix", "ana na ana", "anana",
    "d' accord d'", "d'  ", "d'  x", "d' -x", "d'", "jusqu'  à demain", "jusqu'-à", "Œdème", "pã PÃ pa",
    "ad hoc", "ad  hoc", "ad-hoc ad", "_ad", "ad_", "é ad", "-ad", "ad-", "ad'", "'ad", "l'ad", "c'ad", "ç'ad",
    " ad hoc　", "Ad Hoc", "ÀD", "ana na",
]


def test_edge_cases() -> None:
    import tempfile
    rules = _edge_rules()
    with tempfile.TemporaryDirectory() as d:
        lk = _lookup_of(rules, Path(d) / "edge.json")
    _compare("edge cases", EDGE_TEXTS, lambda t: b.apply_rules(t, rules), lk)


def _fuzz_texts(rules: list[dict], n: int, seed: int) -> list[str]:
    """Texts made of real variants (or pieces of them) glued by random separators."""
    rng = random.Random(seed)
    variants = [v for r in rules for v in (r.get("variants") or [r["variant"]])]
    titles = [t.capitalize() for t in b._TITLES]
    seps = [" ", "  ", "-", " - ", "'", "’", "' ", ". ", ", ", " ", "\t", "", "d'", "l’", "qu'"]
    swaps = {"e": "éÉE", "a": "àÂ", "c": "Ç", "o": "Ô", "i": "Ï", "u": "Ù", "s": "ſ", "k": "K", "n": "ñ"}

    def mangle(v: str) -> str:
        if rng.random() < 0.3:
            v = v.upper() if rng.random() < 0.5 else v.capitalize()
        if rng.random() < 0.3:
            i = rng.randrange(len(v))
            v = v[:i] + rng.choice(swaps.get(v[i].lower(), v[i])) + v[i + 1:]
        if rng.random() < 0.2:
            v = v.replace(" ", rng.choice(["-", "  ", " - "]))
        if rng.random() < 0.1:
            v = v[: rng.randrange(1, len(v) + 1)]
        return v

    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.1:
                parts.append(rng.choice(titles) + " ")
            parts.append(mangle(rng.choice(variants)))
            parts.append(rng.choice(seps))
        out.append("".join(parts))
    return out


def test_fuzz() -> None:
    import tempfile
    for name in FILES:
        rules = _rules(name)
        with tempfile.TemporaryDirectory() as d:
            lk = _lookup_of(rules, Path(d) / "x.json")
        fix = b.compile_rules(rules)  # same output as apply_rules (02's doctests), much faster
        _compare(f"fuzz {name}", _fuzz_texts(rules, 3000, seed=len(name)), fix, lk)


def _corpus() -> list[str]:
    path = os.environ.get("ULTIMED_LOOKUP_CORPUS")
    limit = int(os.environ.get("ULTIMED_LOOKUP_LIMIT", "0")) or None
    if not path:
        return (REPO / "tests" / "fixtures" / "hyps_sample.txt").read_text(encoding="utf-8").splitlines()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["hyp"] if path.endswith(".jsonl") else line.rstrip("\n"))
            if limit and len(out) >= limit:
                break
    return out


def test_corpus() -> None:
    texts = _corpus()
    for name in FILES:
        lk = apply.Lookup.load(REPO / "lookup" / f"{name}_fix_rules.lookup.json")
        _compare(f"corpus {name}", texts, b.compile_rules(_rules(name)), lk)


if __name__ == "__main__":
    test_doctests()
    test_lookups_current()
    test_edge_cases()
    test_fuzz()
    test_corpus()
