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
import json
import os
import random
from pathlib import Path

from _util import load_script

REPO = Path(__file__).resolve().parent.parent
# Application order: the learned files, then the hand rules LAST (see 05_build_manual_rules.py).
FILES = ("drug", "dictionary", "parhaf", "parrot", "acronyms", "manual")


build = load_script("08_build_lookup.py", "build_lookup")
apply = load_script("apply_lookup.py", "apply_lookup")
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
    merge = load_script("03_merge_rules.py", "merge")
    # first alternative wins: "ana" listed before "ana na" in one merged rule
    merged = merge.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": "X"}
                                for v in ("ana", "ana na", "anana")])
    # a hand rule that only fires before a vowel or h, then its fallback (05's resolution_rule)
    vowel = {"variant": "trocarde", "pattern": b.before_vowel_pattern("trocarde"), "replacement": "trocart d'",
             "before_vowel": True}
    return rules + [r for _, r in merged] + [vowel, {"variant": "trocarde", "pattern": b.variant_pattern("trocarde"),
                                                     "replacement": "trocart de"}]


EDGE_TEXTS = [
    "Arrêt de la Mire-taz  apine.", "MYRTAZAPINE puis prim-peran", "aujourd'hui", "prud'hui", "jusqu'hui",
    "12 semaines d'aménorée", "L’aménorée", "l'AMENOREE", "s’ aménorée", "qu'aménorée", "xd'aménorée",
    "sous d' hexamethasone", "sous d'- hexamethasone", "sous d'\thexaméthasone", "d’Hexaméthasone",
    "le docteur Mirat", "Mme Mirat", "Dr. Mirat", "M. Mirat est venu", "sous mirat depuis mai", "docteur  Mirat",
    "xdocteur Mirat", "DOCTEUR Mirat", "KSO et sko, ſko", "lasilix 40", "Lasilix-Lasilix", "ana na ana", "anana",
    "d' accord d'", "d'  ", "d'  x", "d' -x", "d'", "jusqu'  à demain", "jusqu'-à", "Œdème", "pã PÃ pa",
    "ad hoc", "ad  hoc", "ad-hoc ad", "_ad", "ad_", "é ad", "-ad", "ad-", "ad'", "'ad", "l'ad", "c'ad", "ç'ad",
    " ad hoc　", "Ad Hoc", "ÀD", "ana na",
    "trocarde aspiration", "Trocarde  Hôpital", "trocarde\toptique", "trocarde de", "trocarde", "trocarde ",
    "trocarde-arme", "trocarde Œil", "trocarde ÿ", "trocarde 5", "trocarde Ãx", "trocarde\u2028 e", "l'trocarde é",
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


# conflict_resolutions.tsv, through every committed file in application order: (text, expected).
RESOLVED = [
    ("scanner lodose thoracique", "scanner low-dose thoracique"), ("un TDM lodose", "un TDM low-dose"),
    ("Scanner lodose", "Scanner low-dose"), ("une imagerie dite lodose", "une imagerie dite low-dose"),
    ("hyper de la lodose lombaire", "hyper de la lordose lombaire"), ("traité par lodose", "traité par Lodoz"),
    ("sous lodose", "sous Lodoz"),
    ("allergie, l'areus soulage", "allergie, l'Aerius soulage"), ("sous areus le soir", "sous Aerius le soir"),
    ("staphylococcus areus", "staphylococcus aureus"),
    ("gyrus singulaire droit", "gyrus cingulaire droit"), ("engagement singulaire", "engagement cingulaire"),
    ("un cas singulaire", "un cas singulaire"),
    ("un trocarde aspiration", "un trocart d'aspiration"), ("un trocarde hémostatique", "un trocart d'hémostatique"),
    ("un trocarde dilatation", "un trocart de dilatation"),
    ("signes de démolyse", "signes d'hémolyse"), ("la démolyse", "la d'hémolyse"),
    ("signe de lasec positif", "signe de Lasègue positif"), ("lasec", "lasec"),
    ("comprimés de déductiles", "comprimés d'Eductyl"), ("une neubopathie", "une neubopathie"),
    ("lésion du périos", "lésion du périoste"), ("traces de protéinur", "traces de protéinur"),
    ("un prictest positif", "un prick test positif"),
    # the block added 2026-11-17 for the 70/30 blend rules
    ("le virus avalan", "le virus Avalon"), ("traité par avalan", "traité par Havlane"),
    ("sous célesthène", "sous celestene"), ("un cerovar rare", "un sérovar rare"),
    ("un chorynebacterium isolé", "un Corynebacterium isolé"),
    ("syndrome de dejent", "syndrome de Dejean"), ("Dejent", "Dejean"),
    ("absence d'esténose", "absence de sténose"), ("une esténose serrée", "une resténose serrée"),
    ("sous hémipénem", "sous imipenem"), ("sous zimipénème", "sous imipenem"),
    ("devant l'lipoalbuminémie", "devant l'hypoalbuminémie"), ("taux d'lipoalbuminémie", "taux d'hypoalbuminémie"),
    ("et lipoalbuminémie", "et l'hypoalbuminémie"),
    ("des polypsessil", "des polypes sessiles"), ("de polypsessil", "de polypes sessiles"),
    ("plusieurs polypsessil", "plusieurs polypes sessiles"),
    ("une vingtaine de polypsessil", "une vingtaine de polypes sessiles"),
    ("un polypsessil", "un polype sessile"),
    ("au prictes", "au prick test"), ("le prictes", "le prick test"), ("un prictes", "un prick test"),
    ("des prictes", "des prick tests"),
    # étuximab: cetuximab after plus/-s/-x, rituximab after -r, else untouched (listed words only)
    ("FOLFOX plus étuximab", "FOLFOX plus cetuximab"), ("sous étuximab", "sous cetuximab"),
    ("traité par étuximab", "traité par rituximab"), ("relais pour étuximab", "relais pour rituximab"),
    ("le étuximab", "le étuximab"), ("dose d'étuximab", "dose d'étuximab"), ("aux étuximab", "aux étuximab"),
    ("étuximab", "étuximab"),
    # antibial and glissonne/glissone, through the before:WORDS context
    ("poplité et antibial postérieure", "poplité et en tibiale postérieure"),
    ("antibial antérieure", "en tibiale antérieure"), ("stade 3 antibial et fémoral", "stade 3 en tibial et fémoral"),
    ("score de Glissone 7", "score de Gleason 7"), ("score de glissonne 7", "score de Gleason 7"),
    ("glissonne 3 plus 4", "Gleason 3 plus 4"), ("Glissone 10", "Gleason 10"),
    ("une capsule de Glissone tendue", "une capsule de Glisson tendue"),
    ("ISUP4, glissone sensible", "ISUP4, glissone sensible"), ("glissonne 34", "glissonne 34"),
    # glison: same decision, only next to score (owner, 2026-11-17); the learned PARHAF rule is gone
    ("score de glison 7", "score de Gleason 7"), ("glison 7 (3+4)", "Gleason 7 (3+4)"),
    ("la capsule de glison", "la capsule de Glisson"),
    # all three: gleason also right after the article "le" (owner, 2026-11-17), not after a word ending in "le"
    ("le glison est de 7", "le Gleason est de 7"), ("Le glissonne à 8", "Le Gleason à 8"),
    ("un glison isolé", "un Gleason isolé"), ("la cellule glison", "la cellule glison"),
    # Gleason and Glisson are names, capitalised; Gleason also right after "un", not after a word ending in "un"
    ("un glison à 7", "un Gleason à 7"), ("Un glissonne 8", "Un Gleason 8"), ("aucun glison", "aucun glison"),
    # learned replacements carry the labels' casing (09 recase): names capitalised, acronyms in capitals
    ("gleasone 7", "Gleason 7"), ("la capsule de gleisson", "la capsule de Glisson"),
    ("60 unités houndsfield", "60 unités Hounsfield"), ("une anio-irm", "une angio-IRM"),
    ("maladie de macardel", "maladie de McArdle"),
    # and the whole phrase's casing when the per-word step leaves it half done (marie, epi, luc also appear lowercase)
    ("maladie de charcot-mary-tooth", "maladie de Charcot-Marie-Tooth"), ("le ckdpi", "le CKD-EPI"),
    ("un abord de caldwell-luk", "un abord de Caldwell-Luc"),
    # the manual file is recased too (05 through 09's recase_to_labels)
    ("maladie de menière", "maladie de Ménière"), ("syndrome d'heřmansky-pudlak", "syndrome d'Hermansky-Pudlak"),
    ("un amino-acyl-trna", "un aminoacyl-tRNA"),
]


def test_resolutions() -> None:
    """The hand decisions of conflict_resolutions.tsv, end to end on the committed files
    (regex and lookup, learned files then manual), with concrete sentences.

    ``antibial`` and ``glissonne``/``glissone``/``glison`` depend on the next word (``before:WORDS``).
    ``étuximab`` depends on the previous word's suffix, which no context can test, so the
    tsv lists the previous words instead."""
    fixes = [b.compile_rules(_rules(n)) for n in FILES]
    lks = [apply.Lookup.load(REPO / "lookup" / f"{n}_fix_rules.lookup.json") for n in FILES]
    bad = []
    for text, want in RESOLVED:
        got, via = text, text
        for f, lk in zip(fixes, lks):
            got, via = f(got), lk.fix(via)
        if got != want or via != want:
            bad.append((text, want, got, via))
    print(f"resolutions: {len(RESOLVED)} sentences, {len(bad)} wrong")
    assert not bad, bad


if __name__ == "__main__":
    test_doctests()
    test_lookups_current()
    test_edge_cases()
    test_fuzz()
    test_corpus()
    test_resolutions()
