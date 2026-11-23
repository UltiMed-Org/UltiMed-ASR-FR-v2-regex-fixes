#!/usr/bin/env python3
"""Every rule of rules/dictation_rules.csv must turn its example_pre into its example_post.

Format (inherited from the interhop murmure-regex repo): `id,remplacement,regex,example_pre,
example_post`, `# === section ===` comment lines and blank lines ignored. In `remplacement`,
`$N` is group N and `\\n` a newline (no `\\u`: the apps write it literally). Each rule is
checked on its own example and on EXTRA_CASES, and ids must be unique. Stdlib only: ``python
tests/test_dictation_rules.py`` or pytest.

This file was written by Claude Code.
"""
import csv
import re
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "rules" / "dictation_rules.csv"


def load_rules() -> list[dict]:
    lines = [l for l in CSV.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(lines))


def apply(rule: dict, text: str) -> str:
    """Apply one rule like the apps do: $N groups, \\n newline."""
    repl = re.sub(r"\$(\d+)", r"\\g<\1>", rule["remplacement"].replace("\\n", "\n"))
    pattern, flags = rule["regex"], 0
    if pattern.startswith("(?i)"):
        pattern, flags = pattern[4:], re.IGNORECASE
    return re.sub(pattern, repl, text, flags=flags)


# Extra (input, expected) cases per rule id, beyond the CSV's own example.
EXTRA_CASES = {
    "b8cf1d1d-1380": [("a retour à la ligne : b", "a\nb")],
    "7516e9dc-ea5e": [("a saute une ligne: b", "a\nb")],
    "a6f92682-8234": [("a saut de ligne : b", "a\n\nb")],
    "f5b91b4a-5aad": [("a nouveau-tiret : b", "a- b"), ("a nouveaux tirets. b", "a- b")],
    "3f1c9a2e-7b4d": [("a nouvel alinéa : b", "a    b"), ("x alignéa. y", "x    y")],
}


def test_no_uppercase_escape() -> None:
    """Apps do not support \\u and print it (seen: "- \\ubonjour")."""
    bad = [r["id"] for r in load_rules() if "\\u" in r["remplacement"]]
    assert not bad, bad


def test_extra_cases() -> None:
    rules = {r["id"]: r for r in load_rules()}
    bad = [(i, pre, post, apply(rules[i], pre)) for i, cases in EXTRA_CASES.items()
           for pre, post in cases if apply(rules[i], pre) != post]
    assert not bad, "\n".join(map(repr, bad))


def test_examples() -> None:
    rules = load_rules()
    assert rules, "no rules loaded"
    bad = []
    for r in rules:
        expected = r["example_post"].replace("\\n", "\n")
        got = apply(r, r["example_pre"])
        if got != expected:
            bad.append((r["id"], r["example_pre"], expected, got))
    assert not bad, "\n".join(map(repr, bad))


def test_unique_ids() -> None:
    ids = [r["id"] for r in load_rules()]
    assert len(ids) == len(set(ids)), "duplicate ids"


if __name__ == "__main__":
    test_examples()
    test_unique_ids()
    test_no_uppercase_escape()
    test_extra_cases()
    print(f"{len(load_rules())} dictation rules ok")
