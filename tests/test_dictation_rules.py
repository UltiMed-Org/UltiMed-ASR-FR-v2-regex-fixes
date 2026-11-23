#!/usr/bin/env python3
"""Every rule of rules/dictation_rules.csv must turn its example_pre into its example_post.

Format (inherited from the interhop murmure-regex repo): `id,remplacement,regex,example_pre,
example_post`, `# === section ===` comment lines and blank lines ignored. In `remplacement`,
`$N` is group N, `\\n` a newline, and a trailing `\\u` uppercases the next letter. Each rule is
checked on its own example, and ids must be unique. Stdlib only: ``python
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
    """Apply one rule like the apps do: $N groups, \\n newline, trailing \\u uppercases next letter."""
    repl = rule["remplacement"].replace("\\n", "\n")
    upper = repl.endswith("\\u")
    if upper:
        repl = repl[:-2]
    py_repl = re.sub(r"\$(\d+)", r"\\g<\1>", repl)
    pattern = rule["regex"]
    flags = 0
    if pattern.startswith("(?i)"):
        pattern, flags = pattern[4:], re.IGNORECASE
    if not upper:
        return re.sub(pattern, py_repl, text, flags=flags)
    out, pos = [], 0
    for m in re.finditer(pattern, text, flags=flags):
        out.append(text[pos:m.start()] + m.expand(py_repl))
        pos = m.end()
        if pos < len(text):
            out.append(text[pos].upper())
            pos += 1
    return "".join(out) + text[pos:]


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
    print(f"{len(load_rules())} dictation rules ok")
