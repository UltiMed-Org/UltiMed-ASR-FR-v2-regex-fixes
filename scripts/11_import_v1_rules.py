#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Import the v1 rules (``v1/drug_fix_rules.jsonl``, ``v1/term_fix_rules.jsonl``) into the v2 files,
respelled to the v2 labels and re-guarded like a v2 rule.

Why: the v1 rules were learned from several v1-era models (the v1 fine-tune and the base parakeet-ultra
in fp32, int8 and w4a8), so they hold misspellings the v2 model does not make on the corpus but other
models (and v2 on new audio) do. As they are, they carry v1 spellings onto v2 text (``acétoacétate`` where
the v2 labels write ``acéto-acétate``: 9,496 correct v2 labels rewritten), so each one is rebuilt here:

1. **Respell the replacement** to the v2 labels (``label_spellings`` + ``respell``): the labels' majority
   spelling of the same word once case, accents, hyphens, spaces and apostrophes are folded away
   (``compact``), so ``acétoacétate`` -> ``acéto-acétate`` and ``post opératoire`` -> ``postopératoire``.
   The whole replacement is looked up first, then word by word. A word the labels never write is kept
   only if it is a headword the builders track (the drug lexicon of 01, the ``--lexicon`` terms);
   otherwise the rule is dropped (``no_v2_spelling``). Then the same passes as a v2 file, 09's
   ``resolve``: conflict decisions, extra-name title guard, label conventions, recasing to the labels.
2. **Re-guard every variant** with 02's own ``build_rules`` (the v1 variants become a 01-style report,
   so every reason is the one a v2 build applies, not a copy): ``real_text`` against the v2 labels plus
   ``--guard`` texts (French Wikipedia, French news 2024), ``short``, ``other_term``, ``french_word``,
   ``common_words``, ``far``, ``contains_target``, ``ambiguous`` (a variant two v1 rules send to
   different v2 spellings). The v1 counts are only kept for an unmerged v1 rule; a merged rule's
   variants count 1 each, so ``common_words`` drops any of them made only of real words. On top:
   ``v2_variant`` (any v2 file already rewrites it: v2 wins), ``noop`` (it is the target itself),
   ``unk``, ``charset`` (a character 08's lookup form cannot key, e.g. Greek letters), ``french_word_v1``
   (a one-word variant that is a French word, even a rare one: ``V1_MAX_WORD_ZIPF``), and the INSEE ``--names`` title guard of 02.
3. **Check on text** (``bad_variants``), imported rules alone: drop a variant whose rule rewrites a
   correct v2 label (``label_rewrite``) or makes a v2 transcript (``--hyps``, every split) worse against
   its label (``v2_hyp_worse``).
4. **Merge** each imported rule into ``rules/drug_fix_rules.jsonl`` (from the v1 drug file) or
   ``rules/dictionary_fix_rules.jsonl`` (from the v1 term file): inserted in 02's order and merged per
   replacement with the rules already there by 03's ``merge_checked`` (on the hyps, labels and
   ``--merge-text`` texts), so the file names and the order the apps load them in do not change. A rule
   holding only imported variants gets ``"source": "v1"``, a v2 rule that took some gets
   ``"v1_variants": [...]``.
5. **Check the six files in order** (``stack_offenders``): any imported variant still involved in a
   rewritten label or a worse transcript is dropped (04's ``drop_unk_variants``, which rebuilds the
   merged pattern) and the check runs again until clean.

Rerunning is safe: previously imported variants are stripped first (``strip_imported``: every import,
``12_import_extra_variants.py``'s too, so rerun 12 after this), so the import always starts from the
learned v2 files. Rerun it after 07 rebuilds the learned files, then 12, then 08 for the lookups. Steps 3
to 5 are ``merge_import``, which 12 reuses as is; ``--workers N`` splits their passes over the texts
between N processes. The dropped variants and their reason go to ``--rejected`` (default
``rejected/v1_import.rejected.jsonl``, gitignored like the other rejected lists: they hold transcript
fragments).

    uv run scripts/11_import_v1_rules.py --hyps full.blend70-onnx-fp32.hyp.jsonl \\
        --guard .../fr-wikipedia-1M.txt --guard .../fr-news-2024-1M.txt --names .../fr-names.txt \\
        --merge-text .../fr-wikipedia-1M.txt --merge-text .../fr-news-2023-100K.txt
    uv run scripts/08_build_lookup.py rules/*_fix_rules.jsonl

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import click
from loguru import logger
from rapidfuzz.distance import Levenshtein
from wordfreq import zipf_frequency

_HERE = Path(__file__).resolve().parent
# _util.py is loaded by path, not imported: see its docstring.
_uspec = importlib.util.spec_from_file_location("ultimed_util", _HERE / "_util.py")
util = importlib.util.module_from_spec(_uspec)
_uspec.loader.exec_module(util)
r9 = util.load_script("r9", "09_apply_resolutions.py")
# 09 already loads 04 (which loads 02 and 03) and 05: reuse those copies rather than a second one each.
u, b, mr = r9.u, r9.b, r9.mr
m, extract = u.m, r9.b.extract

_REPO = _HERE.parent
TARGETS = {"drug": "drug_fix_rules.jsonl", "term": "dictionary_fix_rules.jsonl"}  # v1 file -> v2 file it joins
V2_FILES = ("drug", "dictionary", "parhaf", "parrot", "acronyms", "manual")  # the application order
LEARNED = V2_FILES[:-1]  # the files a build learns (07) and an import adds to; manual is hand-written
DEFAULT_LEXICON = extract.SCRIPTS_DIR / "01_dictionnary" / "original_dictionnary.jsonl"
_ELISION_HEAD = re.compile(r"^((?:[dlnjmtsc]|qu)['’])(?=\S)", re.IGNORECASE)
_WORD_RE = re.compile(rf"[{b._WORD}]+")
_JOIN = re.compile(r"[\s'’-]+")  # what may separate the words of one spelling in a label
MAX_WORDS = 6  # longest replacement looked up as one phrase, in label words
# A one-word imported variant at least this frequent in French (wordfreq Zipf; 1.0 = about 1 per 100
# million words) is dropped as ``french_word_v1``. 02 uses 2.5 for v2's own variants, but those were
# seen in the v2 model's transcripts of the corpus; the v1 ones only in other models', so a real (if rare)
# French word is not worth the risk: the first import let through ``cubis`` -> ``pubis``, ``mélasses``
# -> ``melas`` (VoxPopuli), ``appendicites`` -> ``appendicite`` and ``locals`` -> ``local``. Costs 312
# of about 127,600 imported variants (2026-11-23), a few of them real fixes (``ibuprofen``).
V1_MAX_WORD_ZIPF = 1.0


def variants_of(r: dict) -> list[str]:
    """A rule's variants, merged (03) or not."""
    return r.get("variants") or [r["variant"]]


@lru_cache(maxsize=None)
def compact(text: str) -> str:
    """Folded letters and digits only: the key two spellings of one word share.

    >>> compact("Acéto-Acétate") == compact("acétoacétate"), compact("post opératoire"), compact("d'Hirschsprung")
    (True, 'postoperatoire', 'dhirschsprung')
    """
    return "".join(ch for ch in b.fold(text) if ch.isalnum())


def split_elision(text: str) -> tuple[str, str]:
    """``("l'", "étoposide")`` for ``l'étoposide``: 02 glues an elided article to some replacements.

    >>> split_elision("l'étoposide"), split_elision("lévothyrox")
    (("l'", 'étoposide'), ('', 'lévothyrox'))
    """
    m_ = _ELISION_HEAD.match(text)
    return (m_.group(1), text[m_.end():]) if m_ else ("", text)


def lookup_keys(target: str) -> set[str]:
    """The ``compact`` keys ``respell`` may look ``target`` up under: whole, then each word.

    >>> sorted(lookup_keys("sous acétoacétate"))
    ['acetoacetate', 'sous', 'sousacetoacetate']
    """
    rest = split_elision(target)[1]
    return {k for k in [compact(rest)] + [compact(split_elision(w)[1]) for w in rest.split()] if k}


def label_spellings(labels, keys: set[str], max_words: int = MAX_WORDS) -> dict[str, Counter]:
    """``compact key -> Counter of the lowercased spellings the labels give it``, for the ``keys`` asked.

    A spelling is a run of up to ``max_words`` label words separated only by spaces, hyphens or
    apostrophes (never punctuation), so ``acéto-acétate``, ``post opératoire`` and ``postopératoire`` are
    all candidates for ``acetoacetate`` / ``postoperatoire``. One pass, pruned by the keys' prefixes.

    >>> labels = ["Cétose à acéto-acétate.", "un acéto-acétate", "Acétoacétate élevé", "ACÉTO-ACÉTATE"]
    >>> label_spellings(labels, {"acetoacetate", "absent"})
    {'acetoacetate': Counter({'acéto-acétate': 3, 'acétoacétate': 1})}
    """
    prefixes = {k[:i] for k in keys for i in range(1, len(k) + 1)}
    out: dict[str, Counter] = defaultdict(Counter)
    for label in labels:
        words = list(_WORD_RE.finditer(label))
        for i, w in enumerate(words):
            key = ""
            for j in range(i, min(i + max_words, len(words))):
                if j > i and not _JOIN.fullmatch(label, words[j - 1].end(), words[j].start()):
                    break
                key += compact(words[j].group(0))
                if key not in prefixes:
                    break
                if key in keys:
                    out[key][label[w.start():words[j].end()].lower()] += 1
    return dict(out)


def _transfer_case(old: str, new: str) -> str:
    """``new`` (lowercase, from the labels) with ``old``'s casing when they differ only by case, else
    ``old``'s capital initial; 09's recasing then applies the labels' own casing where they agree.

    >>> _transfer_case("Kardégic", "kardégic"), _transfer_case("Taxotère", "taxotere"), _transfer_case("x", "y-z")
    ('Kardégic', 'Taxotere', 'y-z')
    """
    if old.lower() == new:
        return old
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def respell(target: str, spellings: dict[str, Counter], headwords: set[str]) -> str | None:
    """``target`` in the v2 labels' spelling, or None when a word of it has none and is no headword.

    >>> sp = {"acetoacetate": Counter({"acéto-acétate": 3, "acétoacétate": 1}), "sous": Counter({"sous": 9})}
    >>> respell("acétoacétate", sp, set()), respell("sous acétoacétate", sp, set()), respell("l'acétoacétate", sp, set())
    ('acéto-acétate', 'sous acéto-acétate', "l'acéto-acétate")
    >>> respell("Zorglub", sp, {"zorglub"}), respell("sous zorglub", sp, set())
    ('Zorglub', None)
    """
    head, rest = split_elision(target)
    if (c := spellings.get(compact(rest))) is not None:
        return head + _transfer_case(rest, c.most_common(1)[0][0])
    out = []
    for w in rest.split():
        h, w2 = split_elision(w)
        if (c := spellings.get(compact(w2))) is not None:
            out.append(h + _transfer_case(w2, c.most_common(1)[0][0]))
        elif compact(w2) in headwords or not compact(w2):
            out.append(w)
        else:
            return None
    return head + " ".join(out)


def headword_keys(lex: dict, lexicons: tuple[Path, ...]) -> set[str]:
    """``compact`` keys of every word 01 tracks (``lex``: drug lexicon plus ``lexicons``) and of each whole term."""
    keys = {compact(w) for w in lex}
    for p in lexicons:
        keys |= {compact(t) for t in util.read_texts(p, "term")}
    return keys - {""}


def v1_report(v1: dict[str, list[dict]], new_target: dict[str, str | None]) -> tuple[dict, dict[tuple, str]]:
    """The v1 rules as a 01-style report for 02's ``build_rules``, plus which v1 file each variant came from.

    One entry per (v1 file, v2 target), its errors the variants with their v1 count when the v1 rule was
    unmerged (it knows it), else 1, so 02's count-based guards (``common_words``) are strict for them. A
    variant both v1 files hold keeps the drug file (applied first).

    >>> rules = {"drug": [{"variants": ["a b", "c"], "replacement": "X"}], "term": [{"variant": "c", "count": 4,
    ...           "replacement": "x"}, {"variant": "d", "replacement": "gone"}]}
    >>> report, origin = v1_report(rules, {"X": "Xx", "x": "Xx", "gone": None})
    >>> report["drug\\tXx"]["errors"], report["term\\tXx"]["errors"], origin[("c",)]
    ({'a b': 1, 'c': 1}, {'c': 4}, 'drug')
    """
    report: dict[str, dict] = {}
    origin: dict[tuple, str] = {}
    for kind in ("term", "drug"):  # drug last so it wins ``origin``
        for r in v1[kind]:
            t = new_target.get(r["replacement"])
            if t is None:
                continue
            e = report.setdefault(f"{kind}\t{t}", {"errors": {}, "targets": {}})
            vs = variants_of(r)
            for v in vs:
                e["errors"][v] = e["errors"].get(v, 0) + (r.get("count", 1) if len(vs) == 1 and "variant" in r else 1)
                e["targets"][v] = t
                origin[b._key(v)] = kind
    return report, origin


_OFF_CHARSET = re.compile(rf"[^{b._WORD}\s'’-]")


def off_charset(variant: str) -> bool:
    """True when ``variant`` holds a character outside the letters the patterns' word bounds know (02's
    ``_WORD``) plus spaces, hyphens and apostrophes: 08 cannot give such a variant a lookup key, so the
    lookup form could not express its rule (v1 learned ``qup iv φ`` -> ``Quviviq``, ``θadiffus``).

    >>> off_charset("qup iv φ"), off_charset("θadiffus"), off_charset("l'acéto-acétate")
    (True, True, False)
    """
    return bool(_OFF_CHARSET.search(variant))


def french_word_v1(variant: str) -> bool:
    """True for a one-word variant at least ``V1_MAX_WORD_ZIPF`` frequent in French (see that constant).

    >>> french_word_v1("cubis"), french_word_v1("myrtazapine"), french_word_v1("lait unique")
    (True, False, False)
    """
    return not re.search(r"[\s'’-]", variant.strip()) and zipf_frequency(variant.lower(), "fr") >= V1_MAX_WORD_ZIPF


def screen(singles: list[dict], v2_keys: set[tuple], origin: dict[tuple, str]) -> tuple[list[dict], list[dict]]:
    """``(kept, rejected)`` of 02's rules built from the v1 report: v2 wins (``v2_variant``: a v2 file
    already rewrites the variant, so it keeps its v2 replacement), ``noop`` (the variant folds to its
    target), ``unk``, ``charset`` (``off_charset``), ``french_word_v1`` (``french_word_v1``). Kept rules are trimmed to what a rule file needs, plus ``v1`` (the v1 file).

    >>> mk = lambda v, t: {"variant": v, "pattern": b.variant_pattern(v), "replacement": t, "count": 1}
    >>> singles = [mk("lodose", "Lodoz"), mk("acéto-acétate", "acéto-acétate"), mk("myrtazapine", "mirtazapine")]
    >>> keep, rej = screen(singles, {("lodose",)}, {("lodose",): "drug", ("aceto", "acetate"): "term", ("myrtazapine",): "drug"})
    >>> [r["variant"] for r in keep], [(r["variant"], r["reason"]) for r in rej]
    (['myrtazapine'], [('lodose', 'v2_variant'), ('acéto-acétate', 'noop')])
    """
    keep, rejected = [], []
    for r in singles:
        k = b._key(r["variant"])
        reason = ("v2_variant" if k in v2_keys else "noop" if k == b._key(r["replacement"])
                  else "unk" if u.has_unk(r["variant"]) else "charset" if off_charset(r["variant"])
                  else "french_word_v1" if french_word_v1(r["variant"]) else None)
        if reason:
            rejected.append({"variant": r["variant"], "replacement": r["replacement"], "v1": origin[k], "reason": reason})
        else:
            keep.append({"variant": r["variant"], "pattern": r["pattern"], "replacement": r["replacement"],
                         "count": r["count"], "v1": origin[k]})
    return keep, rejected


_SOURCE_KEY = "{}_variants"  # the field listing a rule's variants imported from one source: v1_variants, ...


def _source_keys(r: dict) -> dict[str, str]:
    """``source -> field`` for the ``<source>_variants`` fields of rule ``r`` (``variants`` itself excluded)."""
    return {k[:-len("_variants")]: k for k in r if k.endswith("_variants") and k != "variants"}


def provenance(rules: list[dict]) -> dict[str, str]:
    """``variant -> source`` for every imported variant of ``rules``. A rule whose variants all come from one
    import says so with ``"source"`` (``"v1"``, ``"v2-w4a8"``, ...); a rule mixing learned and imported
    variants (or two imports) lists each import's in ``"<source>_variants"`` (``tag_sources`` writes both).

    >>> provenance([{"variants": ["aa", "bb"], "v1_variants": ["bb"]}, {"variant": "cc", "source": "redux-v2"}])
    {'bb': 'v1', 'cc': 'redux-v2'}
    """
    out: dict[str, str] = {}
    for r in rules:
        if r.get("source"):
            out.update(dict.fromkeys(variants_of(r), r["source"]))
        for src, k in _source_keys(r).items():
            out.update(dict.fromkeys(r[k], src))
    return out


def tag_sources(rules: list[dict], origin: dict[str, str]) -> int:
    """Rewrite the provenance fields of ``rules`` in place from ``origin`` (``variant -> source``, as
    ``provenance`` returns), dropping any old ones; returns how many rules hold an imported variant.

    >>> rules = [{"variants": ["aa", "bb", "cc"]}, {"variant": "dd", "v1_variants": ["dd"]}, {"variant": "ee"}]
    >>> tag_sources(rules, {"bb": "v1", "cc": "v2-w4a8", "dd": "v1"}), rules
    (2, [{'variants': ['aa', 'bb', 'cc'], 'v1_variants': ['bb'], 'v2-w4a8_variants': ['cc']}, {'variant': 'dd', 'source': 'v1'}, {'variant': 'ee'}])
    """
    n = 0
    for r in rules:
        for k in _source_keys(r).values():
            del r[k]
        r.pop("source", None)
        by: dict[str, list[str]] = defaultdict(list)
        for v in variants_of(r):
            if v in origin:
                by[origin[v]].append(v)
        if not by:
            continue
        n += 1
        if len(by) == 1 and len(next(iter(by.values()))) == len(variants_of(r)):
            r["source"] = next(iter(by))
        else:
            for src, vs in by.items():
                r[_SOURCE_KEY.format(src)] = vs
    return n


def strip_imported(rules: list[dict], sources=None) -> list[dict]:
    """``rules`` without the variants imported from ``sources`` (default: every import), the remaining
    provenance kept, so a rerun of an import starts from the files it found.

    >>> mk = lambda vs, t: m.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v in vs])[0][1]
    >>> rules = [dict(mk(["aa", "bb"], "X"), v1_variants=["bb"]), dict(mk(["cc"], "Y"), source="v1")]
    >>> [(r["variants"], "v1_variants" in r) for r in strip_imported(rules)]
    [(['aa'], False)]
    >>> rules = [dict(mk(["aa", "bb", "cc"], "X"), v1_variants=["bb"], **{"redux-v2_variants": ["cc"]})]
    >>> [(r["variants"], provenance([r])) for r in strip_imported(rules, {"redux-v2"})]
    [(['aa', 'bb'], {'bb': 'v1'})]
    """
    origin = provenance(rules)
    gone = {v for v, src in origin.items() if sources is None or src in sources}
    out, _ = u.drop_unk_variants(rules, bad=lambda v: v in gone)
    tag_sources(out, {v: src for v, src in origin.items() if v not in gone})
    return out


def _order_key(r: dict) -> tuple:
    """02's rule order (more words, then longer, then more frequent first) of a rule's first variant."""
    v = variants_of(r)[0]
    return (-len(b._key(v)), -len(v), -r.get("count", 0), v)


def insert_ordered(existing: list[dict], new: list[dict]) -> list[dict]:
    """``new`` (any order) placed among ``existing`` (kept in its order) by 02's sort key, before the first
    existing rule that sorts after it, so a longer variant still runs before a shorter one it contains.

    >>> ex = [{"variant": "aa bb", "count": 1}, {"variant": "cc", "count": 1}]
    >>> [r["variant"] for r in insert_ordered(ex, [{"variant": "dddd", "count": 1}, {"variant": "x y z"}])]
    ['x y z', 'aa bb', 'dddd', 'cc']
    """
    new = sorted(new, key=_order_key)
    out, i = [], 0
    for r in existing:
        while i < len(new) and _order_key(new[i]) < _order_key(r):
            out.append(new[i])
            i += 1
        out.append(r)
    return out + new[i:]


def _norm_words(t: str) -> list[str]:
    """evalrules.py's normalisation (lowercase, punctuation out, accents and hyphens kept), as words."""
    return re.sub(r"[^\w'\- ]+", " ", t.lower()).replace("'", " ").split()


def word_errors(ref: str, hyp: str) -> int:
    """Word-level edit distance after ``_norm_words``: the WER numerator for one line.

    >>> word_errors("Sous venlafaxine.", "sous veine la faxine"), word_errors("a b", "A, b")
    (3, 0)
    """
    return Levenshtein.distance(_norm_words(ref), _norm_words(hyp))


def _fired_variants(rule: dict, text: str, mine: set[str]) -> set[str]:
    """The ``mine`` variants of a ``rule`` that fired on a file's input ``text``: those whose own pattern
    matches it. When no variant of the rule matches the input at all, the rule matched what an earlier
    rule of the same file wrote, which variant is not traced, and all of ``mine`` are blamed; when only
    variants that are not ``mine`` match, none is (the first import blamed them all, which dropped some
    innocent variants along with the 27 of 2026-11-23).

    >>> r = {"variants": ["enner", "scaner"], "pattern": "", "replacement": "scanner"}
    >>> sorted(_fired_variants(r, "Enner Valencia", {"scaner"})), sorted(_fired_variants(r, "un scaner", {"scaner"}))
    ([], ['scaner'])
    """
    pre = m._prefix(rule)
    match = {v for v in variants_of(rule) if b._compiled(pre + b.variant_pattern(v)[len(b._START):]).search(text)}
    return match & mine if match else {v for v in variants_of(rule) if v in mine}


def stack_offenders(files: dict[str, list[dict]], imported: set[str], labels: list[str],
                    pairs: list[tuple[str, str]], sources=None, workers: int = 1) -> tuple[dict[str, str], Counter]:
    """Apply the six files in order (``files``, application order) to every label and transcript; return
    ``(imported variant -> reason, line counts)`` for the imported variants that fired where a label was
    rewritten (``label_rewrite``) or a transcript came out worse against its label (``v2_hyp_worse``)
    than with the imported variants removed.

    The baseline without them is computed only for transcripts an imported variant touched, so the check
    costs one pass of the six files plus a second over those lines. The baseline strips the imports of
    ``sources`` (default every import: see ``strip_imported``); ``workers`` splits the lines between
    forked processes (``util.pmap_ranges``).
    """
    fixes = {n: b.compile_rules(rs) for n, rs in files.items()}
    base_files = {n: strip_imported([dict(r) for r in rs], sources) if provenance(rs) else rs
                  for n, rs in files.items()}
    base_fixes = {n: (b.compile_rules(base_files[n]) if base_files[n] is not files[n] else fixes[n]) for n in files}

    def run(text: str, fx) -> tuple[str, set[str]]:
        hit: set[str] = set()
        for n, rs in files.items():
            tr: list[int] = []
            before = text
            text = fx[n](text, tr)
            if fx is fixes:
                for j in tr:
                    hit |= _fired_variants(rs[j], before, imported)
        return text, hit

    def on_labels(lo: int, hi: int) -> tuple[set[str], Counter]:
        hits, st = set(), Counter()
        for t in labels[lo:hi]:
            out, hit = run(t, fixes)
            if out != t:
                st["labels_rewritten"] += 1
                hits |= hit
        return hits, st

    def on_pairs(lo: int, hi: int) -> tuple[set[str], Counter]:
        hits, st = set(), Counter()
        for ref, hyp in pairs[lo:hi]:
            out, hit = run(hyp, fixes)
            if not hit:
                continue
            base = run(hyp, base_fixes)[0]
            if base == out:
                continue
            st["hyps_changed_by_import"] += 1
            e0, e1 = word_errors(ref, base), word_errors(ref, out)
            st["better"] += e1 < e0
            if e1 > e0:
                st["worse"] += 1
                hits |= hit
        return hits, st

    stats = Counter()
    worse, rewrote = set(), set()
    for acc, fn, n in ((rewrote, on_labels, len(labels)), (worse, on_pairs, len(pairs))):
        for hits, st in util.pmap_ranges(fn, n, workers):
            acc |= hits
            stats += st
    return {**dict.fromkeys(worse, "v2_hyp_worse"), **dict.fromkeys(rewrote, "label_rewrite")}, stats


def bad_variants(rules: list[dict], labels: list[str], pairs: list[tuple[str, str]], workers: int = 1) -> dict[str, str]:
    """``variant -> reason`` for the imported single-variant ``rules`` that, applied alone and one at a
    time, rewrite a correct label (``label_rewrite``) or make a transcript worse (``v2_hyp_worse``).

    >>> rules = [{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v, t in
    ...          [("myrtazapine", "mirtazapine"), ("prednisone", "prednisolone"), ("veine la faxine", "venlafaxine")]]
    >>> sorted(bad_variants(rules, ["sous prednisone"], [("sous venlafaxine", "sous veine la faxine"),
    ...                                                   ("le myrtazapine", "le myrtazapine")]).items())
    [('myrtazapine', 'v2_hyp_worse'), ('prednisone', 'label_rewrite')]
    """
    fix = b.compile_rules(rules)

    def on_labels(lo: int, hi: int) -> set[str]:
        out = set()
        for t in labels[lo:hi]:
            tr: list[int] = []
            if fix(t, tr) != t:
                out |= {rules[j]["variant"] for j in tr}
        return out

    def on_pairs(lo: int, hi: int) -> set[str]:
        out = set()
        for ref, hyp in pairs[lo:hi]:
            tr: list[int] = []
            if fix(hyp, tr) == hyp:
                continue
            e0 = word_errors(ref, hyp)
            out |= {rules[j]["variant"] for j in tr if word_errors(ref, b.apply_rules(hyp, [rules[j]])) > e0}
        return out

    rewrote = set().union(*util.pmap_ranges(on_labels, len(labels), workers))
    worse = set().union(*util.pmap_ranges(on_pairs, len(pairs), workers))
    return {**dict.fromkeys(worse, "v2_hyp_worse"), **dict.fromkeys(rewrote, "label_rewrite")}


def _clip_key(path: str) -> tuple[str, ...]:
    """Subset folder and file name: what a hyps line and a manifest row of the same clip share."""
    return Path(path).parts[-2:]


def read_pairs(hyps_path: Path, manifests: tuple[Path, ...]) -> list[tuple[str, str]]:
    """``(label, v2 transcript)`` for every manifest clip the hyps file has."""
    ref = {}
    for mf in manifests:
        for line in mf.open(encoding="utf-8"):
            r = json.loads(line)
            ref[_clip_key(r["audio_filepath"])] = r["text"]
    out = []
    for line in hyps_path.open(encoding="utf-8"):
        h = json.loads(line)
        if (t := ref.get(_clip_key(h["audio"]))) is not None:
            out.append((t, h["hyp"]))
    return out


def merge_import(files: dict[str, list[dict]], new: dict[str, list[dict]], origin: dict[str, str], labels: list[str],
                 pairs: list[tuple[str, str]], merge_paths, meta: tuple[str, ...] = (),
                 workers: int = 1) -> tuple[set[str], list[dict]]:
    """Steps 3 to 5 of an import, shared with ``12_import_extra_variants.py``: check, merge, check in order.

    ``new`` maps a file name of ``files`` (application order, edited in place) to the single-variant rules
    to add to it, ``origin`` each of their variants to its source tag (``"v1"``, ``"v2-w4a8"``, ...). Their
    keys in ``meta`` are bookkeeping, copied to the rejected rows and not into the rule files.

    3. ``bad_variants`` on the ``labels`` and the ``(label, transcript)`` ``pairs``, new rules alone;
    4. per file, ``insert_ordered`` then 03's ``merge_checked`` on the transcripts, labels and ``merge_paths``
       texts, then ``tag_sources`` (earlier imports keep their tags, merging drops them);
    5. ``stack_offenders`` on the six files in order, dropping the new variants involved until clean.

    Returns ``(new variants kept, rejected rows)``.
    """
    rejected: list[dict] = []
    keep = [r for rs in new.values() for r in rs]
    bad = bad_variants(keep, labels, pairs, workers)
    rejected += [{"variant": r["variant"], "replacement": r["replacement"], **{k: r[k] for k in meta if k in r},
                  "reason": bad[r["variant"]]} for r in keep if r["variant"] in bad]
    logger.info(f"{len(bad)} dropped by the text check ({Counter(bad.values())}), {len(keep) - len(bad)} left")

    merge_texts = sorted({h for _, h in pairs} | set(labels)
                         | {t for p in merge_paths for t in util.read_texts(p, "text")})
    imported = {r["variant"] for r in keep} - set(bad)
    sources = {origin[v] for v in imported}
    origin = {**provenance([r for n in new for r in files[n]]), **{v: origin[v] for v in imported}}
    for n, rs in new.items():
        mine = [{k: v for k, v in r.items() if k not in meta} for r in rs if r["variant"] in imported]
        if not mine:
            continue
        combined = insert_ordered(files[n], mine)
        logger.info(f"{n}: {len(files[n])} rules + {len(mine)} imported variants, merge-checking on {len(merge_texts)} texts")
        files[n] = m.merge_checked(combined, merge_texts, workers)
        logger.info(f"{n}: {tag_sources(files[n], origin)} of {len(files[n])} rules hold imported variants")
    del merge_texts

    while True:
        offenders, stats = stack_offenders(files, imported, labels, pairs, sources, workers)
        logger.info(f"six files in order: {dict(stats)}; {len(offenders)} imported variants to drop")
        if not offenders:
            break
        rejected += [{"variant": v, "reason": f"stack_{why}"} for v, why in sorted(offenders.items())]
        imported -= set(offenders)
        for n in new:
            files[n], _ = u.drop_unk_variants(files[n], bad=lambda v: v in offenders)
            tag_sources(files[n], origin)
    return imported, rejected


@click.command()
@click.option("--hyps", "hyps_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="the v2 model's transcripts of every corpus clip (jsonl, fields audio + hyp)")
@click.option("--labels", "labels_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="correct v2 labels (NeMo manifests); default 02's: full.jsonl + PARROT/full.jsonl")
@click.option("--guard", "guard_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="outside correct text (.txt, one sentence per line): French Wikipedia, French news 2024")
@click.option("--names", "names_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="French names (02 --names, e.g. INSEE): such variants do not fire after a title")
@click.option("--merge-text", "merge_paths", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="extra texts 03's merge is checked on, besides the hyps and labels (.txt or manifest)")
@click.option("--lexicon", "lexicons", multiple=True, type=click.Path(exists=True, path_type=Path),
              default=(DEFAULT_LEXICON,), show_default=True, help="term list (01 --lexicon): headwords and other_term")
@click.option("--v1-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=_REPO / "v1",
              show_default=True)
@click.option("--rules-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=_REPO / "rules",
              show_default=True, help="the v2 rule files, read and rewritten in place")
@click.option("--rejected", "rejected_path", type=click.Path(dir_okay=False, path_type=Path),
              default=_REPO / "rejected" / "v1_import.rejected.jsonl", show_default=True)
@click.option("--workers", type=int, default=1, show_default=True,
              help="processes for the passes over every text (03's merge check, the text checks)")
def main(hyps_path: Path, labels_paths: tuple[Path, ...], guard_paths: tuple[Path, ...], names_paths: tuple[Path, ...],
         merge_paths: tuple[Path, ...], lexicons: tuple[Path, ...], v1_dir: Path, rules_dir: Path,
         rejected_path: Path, workers: int) -> None:
    """Respell, re-guard and merge the v1 rules into rules/drug_fix_rules.jsonl and rules/dictionary_fix_rules.jsonl."""
    labels_paths = labels_paths or b.DEFAULT_LABELS
    labels = [t for p in labels_paths for t in util.read_texts(p, "text")]
    v1 = {k: util.read_jsonl(v1_dir / f"{k}_fix_rules.jsonl") for k in TARGETS}
    files = {n: util.read_jsonl(rules_dir / f"{n}_fix_rules.jsonl") for n in V2_FILES}
    others = Counter(src for n in LEARNED for src in provenance(files[n]).values() if src != "v1")
    if others:
        logger.warning(f"also stripping the later imports {dict(others)}: rerun 12_import_extra_variants.py after this")
    for n in LEARNED:
        files[n] = strip_imported(files[n])
    logger.info(f"v1: {', '.join(f'{k} {len(rs)} rules' for k, rs in v1.items())}; {len(labels)} labels")
    rejected: list[dict] = []

    # 1. respell each distinct v1 replacement to the v2 labels' spelling (or a headword), else drop its rules
    targets = sorted({r["replacement"] for rs in v1.values() for r in rs})
    spellings = label_spellings(labels, set().union(*map(lookup_keys, targets)))
    lex = extract.load_lexicon(lexicons)
    heads = headword_keys(lex, lexicons)
    new_target = {t: respell(t, spellings, heads) for t in targets}
    for kind, rs in v1.items():
        rejected += [{"variant": v, "replacement": r["replacement"], "v1": kind, "reason": "no_v2_spelling"}
                     for r in rs if new_target[r["replacement"]] is None for v in variants_of(r)]
    changed = sorted((o, n) for o, n in new_target.items() if n is not None and n != o)
    logger.info(f"{len(targets)} v1 replacements: {len(changed)} respelled to the v2 labels "
                f"(e.g. {changed[:12]}), {sum(n is None for n in new_target.values())} without a v2 spelling")

    # 2. re-guard with 02's build_rules on the respelled report, then v2 wins, then the name guards
    report, origin = v1_report(v1, new_target)
    guard_texts = labels + [t for p in guard_paths for t in util.read_texts(p, "text")]
    singles, rej = b.build_rules(report, [], guard_texts, lex)
    del guard_texts
    rejected += [dict(r, replacement=r["drug"].split("\t")[1], v1=r["drug"].split("\t")[0]) for r in rej]
    keep, rej = screen(singles, {b._key(v) for rs in files.values() for r in rs for v in variants_of(r)}, origin)
    rejected += rej
    if names_paths:
        names = {b.fold(t).strip() for p in names_paths for t in util.read_texts(p, "text")}
        logger.info(f"{b.guard_names(keep, names)} imported variants are French names: kept, but not after a title")
    # 09's passes on the imported rules: conflict decisions, extra names, conventions, recasing to the labels
    keep = r9.resolve(keep, mr.RESOLUTIONS.read_text(encoding="utf-8").splitlines(),
                      r9.EXTRA_NAMES.read_text(encoding="utf-8").splitlines(), labels, name="v1 import")
    logger.info(f"{len(keep)} variants pass the guards")

    # 3 to 5: check alone, merge into the v2 files (03-checked), check the six files in order
    pairs = read_pairs(hyps_path, tuple(Path(p) for p in labels_paths))
    logger.info(f"{len(pairs)} (label, transcript) pairs")
    new = {n: [r for r in keep if r["v1"] == kind] for kind, n in (("drug", "drug"), ("term", "dictionary"))}
    imported, rej = merge_import(files, new, dict.fromkeys((r["variant"] for r in keep), "v1"), labels, pairs,
                                 merge_paths, ("v1",), workers)
    rejected += rej

    for n in LEARNED:
        util.write_jsonl(rules_dir / f"{n}_fix_rules.jsonl", files[n])
        logger.info(f"wrote {rules_dir / f'{n}_fix_rules.jsonl'}: {len(files[n])} rules, "
                    f"{sum(len(variants_of(r)) for r in files[n])} variants")
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    util.write_jsonl(rejected_path, rejected)
    logger.info(f"{len(imported)} v1 variants imported; rejected {len(rejected)} -> {rejected_path}: "
                f"{dict(Counter(r['reason'] for r in rejected).most_common())}")


if __name__ == "__main__":
    main()
