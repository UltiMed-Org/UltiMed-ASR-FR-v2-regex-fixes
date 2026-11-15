#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
# (dependencies: those of 02_build_fix_rules.py, which this imports)
"""Write the lookup-table form of regex rule files: ``rules/X.jsonl`` -> ``lookup/X.lookup.json``.

The rule patterns use lookbehind (word bounds, elisions, the title guard of surname rules),
which some app regex engines lack. A lookup file holds the same rules as plain data: per
rule, its replacement and its variants folded into keys, plus a ``spec`` block that spells
out every character set the patterns relied on (word characters, separators, the
case-insensitive and accent foldings), computed here by asking Python's ``re`` itself, so
the consumer needs no Unicode tables. ``apply_lookup.py`` is the reference consumer; the
format and the matching rules are in its docstring and the README.

Nothing is reinterpreted: each rule's pattern is first rebuilt from its variants with 02's
``variant_pattern`` and 03's merge scaffolding and must equal the stored pattern byte for
byte, so the keys describe exactly what the regex matches. A rule that does not rebuild, or
a variant whose shape the lookup grammar cannot express, stops the build with an error
instead of being approximated.

Layout of the output (valid JSON, one rule per line so diffs stay readable)::

    {"format": "ultimed-fix-rules-lookup", "version": 1, "source": ..., "source_sha256": ...,
     "spec": {...},
     "rules": [
      {"to": "mirtazapine", "keys": ["myrtazapine", "mire taz apine"]},
      {"to": "Humira", "keys": ["mirat"], "not_after": true},
      ...]}

Usage::

    uv run scripts/08_build_lookup.py rules/*.jsonl            # -> lookup/<name>.lookup.json
    uv run scripts/08_build_lookup.py IN.jsonl --out OUT.json

This file was written by Claude Code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("b", _HERE / "02_build_fix_rules.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
_mspec = importlib.util.spec_from_file_location("m", _HERE / "03_merge_rules.py")
m = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(m)

DEFAULT_DIR = _HERE.parent / "lookup"
# Every code point but the surrogates, which no decoded text holds: matched once per
# character class to learn what re.IGNORECASE really lets through.
_ALL = "".join(chr(c) for c in range(sys.maxunicode + 1) if not 0xD800 <= c <= 0xDFFF)
_ELISION = "dlnjmtsc"  # the letters of 02's _START elision class
_APOSTROPHES = "'’"


@lru_cache(maxsize=None)
def _matching(pattern: str) -> frozenset[str]:
    """Every character ``pattern`` (one character class) matches, case-insensitively
    (cached: one scan of all of Unicode per class)."""
    return frozenset(re.findall(pattern, _ALL, re.IGNORECASE))


def _ranges(chars: set[str]) -> list[list[int]]:
    """``chars`` as sorted inclusive ``[lo, hi]`` code point ranges."""
    out: list[list[int]] = []
    for c in sorted(map(ord, chars)):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def _variants(r: dict) -> list[str]:
    """A rule's variants, merged (03) or not."""
    return r.get("variants") or [r["variant"]]


def check_pattern(r: dict) -> bool:
    """True if ``r``'s pattern is exactly what 02/03 build from its variants.

    Unmerged rules are ``[title guard] + variant_pattern(v)``; merged ones put the bodies in
    one ``(?:...|...)`` behind the same prefix, except that 04 writes a merged rule left with
    one variant without the group. A ``before_vowel`` rule (05, never merged) ends on
    ``_BEFORE_VOWEL`` instead of ``_END``.

    >>> r = {"variant": "myrtazapine", "pattern": b.variant_pattern("myrtazapine"), "replacement": "x"}
    >>> check_pattern(r), check_pattern(dict(r, pattern=r["pattern"] + "s"))
    (True, False)
    """
    body = lambda v: b.variant_pattern(v)[len(b._START):-len(b._END)]  # noqa: E731
    pre, vs = m._prefix(r), _variants(r)
    if r.get("before_vowel"):
        return "variants" not in r and r["pattern"] == pre + body(vs[0]) + b._BEFORE_VOWEL
    single = pre + body(vs[0]) + b._END
    group = pre + "(?:" + "|".join(map(body, vs)) + ")" + b._END
    return r["pattern"] == (group if "variants" in r else single) or (len(vs) == 1 and r["pattern"] in (single, group))


def build_spec(rules: list[dict]) -> dict:
    """The character sets the patterns rely on, measured with ``re`` itself.

    ``case_map`` sends every character that matches some lowercase letter of the key
    alphabet case-insensitively (``A``, ``É``, the Kelvin sign ``K``, the long ``ſ``) to that
    letter; ``accent_map`` folds the letters of 02's accent classes. Both are needed to match
    keys; the title guard and the elision bound only use ``case_map``.
    """
    alphabet = set(_ELISION + "qu") | {c for t in b._TITLES for c in t} | {c for s in b._ACCENTS.values() for c in s}
    for r in rules:
        for v in _variants(r):
            alphabet |= {c.lower() for c in v if c not in " -'"}
    case_map: dict[str, str] = {}
    for x in sorted(alphabet):
        for c in _matching(re.escape(x)) - {x}:
            assert case_map.setdefault(c, x) == x, f"{c!r} matches both {case_map[c]!r} and {x!r}"
    for x in alphabet:  # a key letter must not itself fold to another one
        assert x not in case_map, x
    accent_map = {c: k for k, s in b._ACCENTS.items() for c in s if c != k}
    assert _matching(f"[{_APOSTROPHES}]") == set(_APOSTROPHES)
    return {
        "word_chars": _ranges(_matching(f"[{b._WORD}]")),
        "separators": "".join(sorted(_matching(r"[\s-]"))),
        "whitespace": "".join(sorted(_matching(r"\s"))),
        "vowels": "".join(sorted(_matching(f"[{b._VOWELS}]"))),
        "apostrophes": _APOSTROPHES,
        "case_map": dict(sorted(case_map.items())),
        "accent_map": accent_map,
        "elision": _ELISION,
        "titles": list(b._TITLES),
    }


def variant_key(v: str, spec: dict) -> str:
    """``v`` folded the way its pattern matches it: see apply_lookup.py's docstring.

    >>> spec = build_spec([{"variant": "x"}])
    >>> variant_key("Phélo-dipine", spec), variant_key("d'  Hexaméthasone", spec), variant_key("Œdème ã", spec)
    ('phelo dipine', "d' hexamethasone", 'œdeme a')

    The ``ã`` above becomes ``a`` because 02 writes it as the class ``[aàâä]``, which does
    not hold ``ã`` itself: the rule fires on "pa" but not on "pã", and so does the key
    (``accent_map`` has no ``ã``).
    """
    words = []
    for w in re.split(r"[\s-]+", v.strip()):
        out = []
        for ch in w:
            if ch == "'":
                out.append("'")
            elif b.fold(ch) in b._ACCENTS:
                out.append(b.fold(ch))  # the pattern's accent class, [eéèêë] for any e
            else:
                # a literal under IGNORECASE: matches exactly what its lowercase matches
                assert _matching(re.escape(ch)) == _matching(re.escape(ch.lower())), ch
                out.append(ch.lower())
        words.append("".join(out))
    return " ".join(words)


def _key_ok(k: str, word: set[str]) -> bool:
    """True if the lookup grammar can express ``k``: words of word characters, apostrophes
    only inside or at the end of a word, never two in a row (02 writes ``'`` as ``['’]\\s*``)."""
    if not k or k[0] not in word or "''" in k or " '" in k or k.endswith(" ") or "  " in k:
        return False
    return all(c in word or c in " '" for c in k)


def build_lookup(rules: list[dict]) -> tuple[dict, list[str]]:
    """The lookup form of ``rules``, plus a description of each rule it cannot express.

    >>> rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v, t in
    ...          [("myrtazapine", "mirtazapine"), ("mirat", "Humira")]]
    >>> b.guard_names(rules, {"mirat"})
    1
    >>> lk, bad = build_lookup(rules)
    >>> lk["rules"], bad
    ([{'to': 'mirtazapine', 'keys': ['myrtazapine']}, {'to': 'Humira', 'keys': ['mirat'], 'not_after': True}], [])
    """
    spec = build_spec(rules)
    word = {chr(c) for lo, hi in spec["word_chars"] for c in range(lo, hi + 1)}
    out, bad = [], []
    for i, r in enumerate(rules):
        keys = [variant_key(v, spec) for v in _variants(r)]
        if not check_pattern(r):
            bad.append(f"rule {i} ({r['replacement']!r}): pattern does not rebuild from its variants")
        for v, k in zip(_variants(r), keys):
            if not _key_ok(k, word) or "’" in v or (r.get("before_vowel") and k.endswith("'")):
                bad.append(f"rule {i} ({r['replacement']!r}): variant {v!r} has no lookup key")
        row = {"to": r["replacement"], "keys": keys}
        if r.get("name"):
            row["not_after"] = True
        if r.get("before_vowel"):
            row["before_vowel"] = True
        out.append(row)
    return {"format": "ultimed-fix-rules-lookup", "version": 2, "spec": spec, "rules": out}, bad


def write_lookup(lookup: dict, path: Path) -> None:
    """Write ``lookup`` as JSON with one rule per line."""
    head = {k: v for k, v in lookup.items() if k != "rules"}
    lines = [json.dumps(r, ensure_ascii=False) for r in lookup["rules"]]
    # The header escapes non-ASCII: its separator lists hold invisible spaces (U+2028...).
    text = json.dumps(head)[:-1] + ',\n "rules": [\n  ' + ",\n  ".join(lines) + "\n ]}\n"
    assert json.loads(text) == lookup
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def lookup_path(src: Path) -> Path:
    """Default output for ``src``: ``lookup/<src stem>.lookup.json``."""
    return DEFAULT_DIR / f"{src.stem}.lookup.json"


def build_lookup_file(src: Path, dst: Path | None = None) -> Path:
    """Write the lookup form of the rule file ``src`` and return where it went.

    Shared by this script's CLI and by ``05_build_manual_rules.py``, which refreshes the
    manual lookup each time it rewrites the manual rules, so the two never drift apart.

    Parameters
    ----------
    src : Path
        A ``*_fix_rules.jsonl`` file.
    dst : Path, optional
        Output file; default ``lookup/<src stem>.lookup.json``.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    click.ClickException
        If any rule cannot be expressed as lookups (each reason is logged first).
    """
    raw = src.read_bytes()
    rules = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    lookup, bad = build_lookup(rules)
    if bad:
        for msg in bad:
            logger.error(f"{src.name}: {msg}")
        raise click.ClickException(f"{src}: {len(bad)} rules cannot be expressed as lookups")
    # source + its hash let a consumer (and tests/test_lookup.py) tell a stale lookup apart.
    lookup = {"format": lookup["format"], "version": lookup["version"], "source": src.name,
              "source_sha256": hashlib.sha256(raw).hexdigest(), "spec": lookup["spec"], "rules": lookup["rules"]}
    dst = dst or lookup_path(src)
    write_lookup(lookup, dst)
    logger.info(f"{src} -> {dst}: {len(rules)} rules, {sum(len(r['keys']) for r in lookup['rules'])} keys, "
                f"{dst.stat().st_size:,} bytes (rules file {len(raw):,})")
    return dst


@click.command()
@click.argument("srcs", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", type=click.Path(path_type=Path), default=None,
              help="output file (one SRC only); default lookup/<SRC stem>.lookup.json")
def main(srcs: tuple[Path, ...], out: Path | None) -> None:
    """Write the lookup form of each rule file SRCS."""
    assert out is None or len(srcs) == 1, "--out takes a single SRC"
    for src in srcs:
        build_lookup_file(src, out)


if __name__ == "__main__":
    main()
