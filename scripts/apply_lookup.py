#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru"]
# ///
"""Apply fix rules from their lookup-table form (``lookup/*.lookup.json``), with no regex.

The lookup files are written by ``08_build_lookup.py`` from the regex rule files in
``rules/``. They exist for apps whose regex engine has no lookbehind (the rule patterns need
it for their word bounds and title guards): the matching below only walks characters and
reads dicts, so it ports to any language. For every input text the output is identical to
02_build_fix_rules.py ``apply_rules`` on the same rule file; ``tests/test_lookup.py`` checks
that on real transcripts.

What a rule file means, and what this module reproduces exactly:

- Rules apply one after the other, in file order, each over the whole text (so rule 7 sees
  rule 3's output, and a replacement can feed a later rule).
- Within one rule, matches are found left to right, never overlapping; at one start the
  rule's FIRST key (in its ``keys`` order) that matches wins, not the longest. Bounds and
  guards always look at the text as it was before that rule ran.
- A replacement whose rule matched an uppercase first letter gets its own first letter
  uppercased, if it was lowercase (``Mirtazapine`` for ``Myrtazapine``).

A key is a variant folded the way its regex matched it (see ``spec`` in the file):

- each text character maps through ``case_map`` (case-insensitive like Python's
  ``re.IGNORECASE``) then ``accent_map`` (``é`` -> ``e``: only the accents the rule classes
  list, ``ã`` is NOT ``a``);
- one run of ``separators`` (whitespace or hyphens) between two words is one space;
- an apostrophe (``'`` or ``’``) is ``'``, and may be followed in the text by whitespace the
  key does not show (``d' hexaméthasone`` matches ``d'hexamethasone``).

A match must start on a word character that does not follow a word character, a hyphen, or
an apostrophe, except the apostrophe of a French elision (a lone ``d l n j m t s c``, or
``qu``); it must end before a character that is neither a word character nor a hyphen. A
rule with ``not_after`` never starts right after one of ``spec.titles`` plus one space
(``docteur Mirat`` keeps the surname). A rule with ``before_vowel`` (lookup version 2,
hand rules only) matches only when whitespace then a character of ``spec.vowels`` follow
the key, and its match runs to that character, the whitespace included (``trocarde
aspiration`` -> ``trocart d'`` + ``aspiration``). A key ending with an apostrophe swallows
the whitespace after it, minus one space if a word character or hyphen follows.

Usage::

    uv run scripts/apply_lookup.py lookup/drug_fix_rules.lookup.json lookup/manual_fix_rules.lookup.json \\
        --in hyps.txt --out fixed.txt

Lookups apply in the order given (the manual file last), one text per input line.

This file was written by Claude Code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import click
from loguru import logger


@dataclass
class Lookup:
    """One lookup file, loaded for matching.

    Attributes
    ----------
    rules : list[dict]
        The file's rules in application order: ``to``, optional ``not_after`` and ``before_vowel``.
    table : dict[str, list[tuple[int, int]]]
        Key -> ``(rule index, key rank within that rule)`` for every rule holding it, rule
        index ascending (two rules may share a key, and one rule may list a key twice when
        two variants differed only by an accent).
    """

    rules: list[dict]
    table: dict[str, list[tuple[int, int]]]
    word: frozenset[str]
    seps: frozenset[str]
    space: frozenset[str]
    vowels: frozenset[str]
    apostrophes: frozenset[str]
    case_map: dict[str, str]
    accent_map: dict[str, str]
    elision: frozenset[str]
    titles: list[str]
    max_key_len: int
    source: str = ""
    _canon: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Lookup":
        """Read a ``*.lookup.json`` file written by 08_build_lookup.py.

        Parameters
        ----------
        path : Path
            The lookup file.

        Returns
        -------
        Lookup
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["format"] == "ultimed-fix-rules-lookup" and data["version"] in (1, 2), path
        spec = data["spec"]
        table: dict[str, list[tuple[int, int]]] = {}
        for i, r in enumerate(data["rules"]):
            for rank, k in enumerate(r["keys"]):
                table.setdefault(k, []).append((i, rank))
        word = frozenset(chr(c) for lo, hi in spec["word_chars"] for c in range(lo, hi + 1))
        return cls(rules=data["rules"], table=table, word=word, seps=frozenset(spec["separators"]),
                   space=frozenset(spec["whitespace"]), vowels=frozenset(spec.get("vowels", "")), apostrophes=frozenset(spec["apostrophes"]),
                   case_map=spec["case_map"], accent_map=spec["accent_map"],
                   elision=frozenset(spec["elision"]), titles=spec["titles"],
                   max_key_len=max(map(len, table), default=0), source=data.get("source", ""))

    def case(self, ch: str) -> str:
        """``ch`` as the lowercase character a case-insensitive match compares."""
        return self.case_map.get(ch, ch)

    def canon(self, ch: str) -> str:
        """``ch`` as it appears in keys: case, then accent folded."""
        out = self._canon.get(ch)
        if out is None:
            c = self.case(ch)
            out = self._canon[ch] = self.accent_map.get(c, c)
        return out

    def _starts(self, text: str, p: int) -> bool:
        """True if a match may start at ``p`` (the regex ``_START`` bound)."""
        if text[p] not in self.word:
            return False
        if p == 0:
            return True
        prev = text[p - 1]
        if prev in self.word or prev == "-":
            return False
        if prev not in self.apostrophes:
            return True
        # An elided article: one of "dlnjmtsc" standing alone, or "qu" (jusqu', lorsqu').
        if p >= 2 and self.case(text[p - 2]) in self.elision and (p == 2 or text[p - 3] not in self.word):
            return True
        return p >= 3 and self.case(text[p - 3]) == "q" and self.case(text[p - 2]) == "u"

    def _after_title(self, text: str, p: int) -> bool:
        """True if ``p`` follows a title plus one space (the ``not_after`` guard)."""
        for t in self.titles:
            s = p - len(t) - 1
            if s < 0 or text[p - 1] != " " or (s > 0 and text[s - 1] in self.word):
                continue
            if all(self.case(text[s + k]) == t[k] for k in range(len(t))):
                return True
        return False

    def _before_vowel(self, text: str, end: int) -> int:
        """Where a ``before_vowel`` match ending at ``end`` really ends: past the whitespace
        that follows it, if a vowel (or h) comes next, else -1."""
        k = end
        while k < len(text) and text[k] in self.space:
            k += 1
        return k if k > end and k < len(text) and text[k] in self.vowels else -1

    def _parses(self, text: str, p: int) -> list[tuple[str, int]]:
        """Every ``(key, end)`` a key could match from ``p`` on.

        A key's word characters must cover whole runs of text word characters (whatever
        follows a key word, end, separator or apostrophe, is not a word character), so a
        parse is fixed except after an apostrophe, where the key may go on with a word
        (``'x``: the apostrophe takes the whitespace), a separator (``' x``) or stop.
        """
        n, out, stack = len(text), [], [(p, "")]
        while stack:
            i, buf = stack.pop()
            j = i
            while j < n and text[j] in self.word:
                j += 1
            buf += "".join(self.canon(c) for c in text[i:j])
            if len(buf) > self.max_key_len:
                continue
            nxt = text[j] if j < n else ""
            if nxt != "-":
                out.append((buf, j))  # nxt is not a word character: the run is maximal
            if nxt in self.seps:
                k = j
                while k < n and text[k] in self.seps:
                    k += 1
                if k < n and text[k] in self.word:
                    stack.append((k, buf + " "))
            elif nxt and nxt in self.apostrophes:
                k = j + 1
                while k < n and text[k] in self.space:
                    k += 1
                after = text[k] if k < n else ""
                # Key ending on the apostrophe: ['’]\s* then a bound, backtracking one space.
                if not after or (after not in self.word and after != "-"):
                    out.append((buf + "'", k))
                elif k > j + 1:
                    out.append((buf + "'", k - 1))
                if after and after in self.word:
                    stack.append((k, buf + "'"))
                k = j + 1
                while k < n and text[k] in self.seps:
                    k += 1
                if k > j + 1 and k < n and text[k] in self.word:
                    stack.append((k, buf + "' "))
        return out

    def matches(self, text: str, after: int = -1) -> dict[int, list[tuple[int, int, int]]]:
        """Every place a rule past ``after`` matches ``text``.

        Parameters
        ----------
        text : str
        after : int
            Only rules with a larger index count.

        Returns
        -------
        dict[int, list[tuple[int, int, int]]]
            Rule index -> ``(start, rank, end)`` of its matches, guards applied.
        """
        found: dict[int, list[tuple[int, int, int]]] = {}
        for p in range(len(text)):
            if not self._starts(text, p):
                continue
            for key, end in self._parses(text, p):
                for i, rank in self.table.get(key, ()):
                    if i > after and not (self.rules[i].get("not_after") and self._after_title(text, p)):
                        e = self._before_vowel(text, end) if self.rules[i].get("before_vowel") else end
                        if e >= 0:
                            found.setdefault(i, []).append((p, rank, e))
        return found

    def fix(self, text: str, trace: list[int] | None = None) -> str:
        """Apply every rule in order, like 02_build_fix_rules.py ``apply_rules``.

        Only rules that match change anything, so this jumps from one matching rule to the
        next (the lowest index past the last one applied) instead of trying all of them.

        Parameters
        ----------
        text : str
        trace : list[int] | None
            If given, the index of each rule that changed the text is appended.

        Returns
        -------
        str
        """
        last = -1
        while True:
            found = self.matches(text, last)
            if not found:
                return text
            last = min(found)
            best: dict[int, tuple[int, int]] = {}  # start -> (rank, end), first key wins
            for p, rank, end in found[last]:
                if p not in best or rank < best[p][0]:
                    best[p] = (rank, end)
            to, parts, pos = self.rules[last]["to"], [], 0
            for p in sorted(best):
                if p < pos:
                    continue
                rep = to[0].upper() + to[1:] if text[p].isupper() and to[0].islower() else to
                parts += [text[pos:p], rep]
                pos = best[p][1]
            new = "".join(parts) + text[pos:]
            if new != text and trace is not None:
                trace.append(last)
            text = new


@click.command()
@click.argument("lookups", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--in", "inp", type=click.File("r", encoding="utf-8"), default="-", show_default=True,
              help="text file, one text per line")
@click.option("--out", type=click.File("w", encoding="utf-8"), default="-", show_default=True)
def main(lookups: tuple[Path, ...], inp, out) -> None:
    """Fix every line of --in with LOOKUPS, applied in the order given."""
    tables = [Lookup.load(p) for p in lookups]
    changed = n = 0
    for line in inp:
        text = line.rstrip("\n")
        new = text
        for t in tables:
            new = t.fix(new)
        changed += new != text
        n += 1
        out.write(new + "\n")
    logger.info(f"{n} lines, {changed} changed")


if __name__ == "__main__":
    main()
