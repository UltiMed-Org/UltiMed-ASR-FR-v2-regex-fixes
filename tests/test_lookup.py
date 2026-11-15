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
- ``test_corpus``: by default a synthetic corpus, generated at test time (seeded, so the
  same texts every run) from each rule file's own variants placed in invented French
  carrier sentences (start and end of text, after an elision, next to punctuation,
  capitalised, after a title, two variants side by side), each file checked on its own
  texts. No real transcript is committed. Set ``ULTIMED_LOOKUP_CORPUS`` to a hyps
  ``.jsonl`` (field ``hyp``) or a ``.txt`` (one text per line) to run real transcripts
  instead, against all six rule files, e.g. the 602k-clip whole-corpus transcription;
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


_SWAPS = {"e": "éÉE", "a": "àÂ", "c": "Ç", "o": "Ô", "i": "Ï", "u": "Ù", "s": "ſ", "k": "K", "n": "ñ"}


def _variants(rules: list[dict]) -> list[str]:
    return [v for r in rules for v in (r.get("variants") or [r["variant"]])]


def _mangle(v: str, rng: random.Random) -> str:
    """``v`` randomly upper-cased or capitalised, with one letter swapped for an accented or
    look-alike one, re-separated, or cut short."""
    if rng.random() < 0.3:
        v = v.upper() if rng.random() < 0.5 else v.capitalize()
    if rng.random() < 0.3:
        i = rng.randrange(len(v))
        v = v[:i] + rng.choice(_SWAPS.get(v[i].lower(), v[i])) + v[i + 1:]
    if rng.random() < 0.2:
        v = v.replace(" ", rng.choice(["-", "  ", " - "]))
    if rng.random() < 0.1:
        v = v[: rng.randrange(1, len(v) + 1)]
    return v


def _fuzz_texts(rules: list[dict], n: int, seed: int) -> list[str]:
    """Texts made of real variants (or pieces of them) glued by random separators."""
    rng = random.Random(seed)
    variants = _variants(rules)
    titles = [t.capitalize() for t in b._TITLES]
    seps = [" ", "  ", "-", " - ", "'", "’", "' ", ". ", ", ", " ", "\t", "", "d'", "l’", "qu'"]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.1:
                parts.append(rng.choice(titles) + " ")
            parts.append(_mangle(rng.choice(variants), rng))
            parts.append(rng.choice(seps))
        out.append("".join(parts))
    return out


# Invented carrier sentences for the synthetic corpus ({x}, {y}: variants, {X}, {Y}: capitalised).
CARRIERS = (
    "Le patient présente {x} depuis hier.",
    "{X} a été introduit ce matin.",
    "Traitement habituel : {x}",
    "{x}",
    "On retrouve l'{x} à l'examen clinique.",
    "Pas d’{x} ni de {y} à ce jour.",
    "Il faut savoir qu'{x} reste possible.",
    "Arrêt de {x}, relais par {y}.",
    "Sous {x} {y} depuis mai.",
    "({x}) puis « {y} ».",
    "Bilan : {x}; {y}!",
    "le docteur {X} a vu la patiente",
    "Mme {X} est revenue en consultation pour {y}.",
    "Antécédents : {x}-{y}, {x}?",
    "{X}. {Y}.",
)


def _carrier_texts(rules: list[dict], n: int, seed: int) -> list[str]:
    """``n`` invented sentences, each a ``CARRIERS`` template filled with the file's variants.

    The variants are taken in a seeded shuffle, cycled, so every variant appears once ``n``
    reaches their number; a quarter of them go through ``_mangle`` first.
    """
    rng = random.Random(seed)
    variants = _variants(rules)
    rng.shuffle(variants)
    out = []
    for i in range(n):
        x = variants[i % len(variants)]
        y = rng.choice(variants)
        if rng.random() < 0.25:
            x = _mangle(x, rng)
        out.append(CARRIERS[i % len(CARRIERS)].format(x=x, y=y, X=x.capitalize(), Y=y.capitalize()))
    return out


def test_fuzz() -> None:
    import tempfile
    for name in FILES:
        rules = _rules(name)
        with tempfile.TemporaryDirectory() as d:
            lk = _lookup_of(rules, Path(d) / "x.json")
        fix = b.compile_rules(rules)  # same output as apply_rules (02's doctests), much faster
        _compare(f"fuzz {name}", _fuzz_texts(rules, 3000, seed=len(name)), fix, lk)


def _corpus(path: str) -> list[str]:
    limit = int(os.environ.get("ULTIMED_LOOKUP_LIMIT", "0")) or None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["hyp"] if path.endswith(".jsonl") else line.rstrip("\n"))
            if limit and len(out) >= limit:
                break
    return out


def test_corpus() -> None:
    """Real transcripts from ``ULTIMED_LOOKUP_CORPUS`` against every file, or else each file
    against its own synthetic sentences (its variant count, at least 3,000 and at most 10,000)."""
    path = os.environ.get("ULTIMED_LOOKUP_CORPUS")
    texts = _corpus(path) if path else None
    for seed, name in enumerate(FILES):
        rules = _rules(name)
        lk = apply.Lookup.load(REPO / "lookup" / f"{name}_fix_rules.lookup.json")
        if not path:
            texts = _carrier_texts(rules, min(max(3000, len(_variants(rules))), 10000), seed=seed)
        _compare(f"corpus {name}{'' if path else ' (synthetic)'}", texts, b.compile_rules(rules), lk)


if __name__ == "__main__":
    test_doctests()
    test_lookups_current()
    test_edge_cases()
    test_fuzz()
    test_corpus()
