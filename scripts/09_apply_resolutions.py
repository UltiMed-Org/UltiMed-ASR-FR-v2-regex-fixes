#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Post-process a learned rule file: drop the hand-decided variants, respell replacements.

1. Every variant of ``inputs/conflict_resolutions.tsv`` (matched by pattern, so accents and
   hyphens do not matter) is removed from the file: those variants got different
   replacements from different files, and the author's decision ships as manual rules
   instead (05). A merged rule loses just that variant and its pattern is rebuilt by 04's
   ``drop_unk_variants``; a rule left with no variant is dropped.
2. Each replacement goes through the label spelling conventions of
   UltiMed-ASR-FR-v1-scripts (``utils/label_conventions.py``, via 06's
   ``convention_fixes``, so only accent, hyphen and space respellings count): a learned rule
   must not write ``bévacizumab`` when the conventions, and the convention rules, say
   ``bevacizumab``. With ``--labels``, a respelling is kept only if the labels holding the
   old replacement, once converted, mostly hold the new one: 02's replacements are still
   lowercase at this step (``caldeyro-barcia``, step 3 recases them), and the conventions glue a lowercase compound they
   leave alone in the labels' own case (``Caldeyro-Barcia``, ``anti-HBs``), like 06's
   label check. ``--no-conventions`` skips this.
3. With ``--labels``, each word of a replacement is recased to the labels' own casing
   (``label_casing``): ``gleason`` -> ``Gleason``, ``hta`` -> ``HTA``. 02 copies the lexicon's
   lowercase canonical form, whereas the labels write a name capitalised and an acronym in
   capitals. Only a word the labels write the same way every time it is not
   sentence-initial (and at least ``RECASE_MIN`` times) is recased, so a word that also
   appears lowercase mid-sentence is left alone. The variants are untouched.
   Then a multi-word replacement takes the casing the labels give the whole phrase
   (``label_phrase_casing``) when one spelling holds at least ``PHRASE_SHARE`` of its
   non-initial occurrences (at least ``RECASE_MIN``): ``Charcot-marie-Tooth`` ->
   ``Charcot-Marie-Tooth``, which the per-word step leaves half done because ``marie`` is
   also a lowercase word. ``--no-recase`` skips both.
4. Every rule with a variant listed in ``--names`` (default ``inputs/extra_names.txt``, the
   surnames heard in real speech that the INSEE lists given to 02 ``--names`` miss) gets 02's
   title guard (``guard_names``, merged rules included): ``le docteur Malar`` stays, ``os malar``
   still becomes ``os malaire``. A rule already guarded is left alone, so rerunning is harmless.
   This is the one place the list is applied: 07 calls 09 with the default.

Rules with the same replacement are not re-merged: a merge must be checked on texts (03).

Usage: uv run scripts/09_apply_resolutions.py SRC.jsonl DST.jsonl [--resolutions TSV] [--names TXT] [--labels MANIFEST ...] [--no-conventions] [--no-recase]
(SRC and DST may be the same file). Called by 07_build_dataset_rules.sh after 04.
This file was written by Claude Code.
"""
import importlib.util
import re
from collections import Counter
from pathlib import Path

import click
from loguru import logger

_HERE = Path(__file__).resolve().parent
# _util.py is loaded by path, not imported: see its docstring.
_uspec = importlib.util.spec_from_file_location("ultimed_util", _HERE / "_util.py")
util = importlib.util.module_from_spec(_uspec)
_uspec.loader.exec_module(util)
u = util.load_script("u", "04_drop_unk_variants.py")
mr = util.load_script("mr", "05_build_manual_rules.py")
b = u.b  # 04 already loads 02, no third copy of it here
EXTRA_NAMES = mr.INPUTS / "extra_names.txt"


def resolved_patterns(lines) -> set[str]:
    """The ``variant_pattern`` of every variant a resolutions TSV decides."""
    return {b.variant_pattern(row[0]) for row in mr.read_rows(lines)}


def read_names(lines) -> set[str]:
    """The ``fold``-ed names of a names file (one per line, ``#`` comments), as 02 ``--names`` folds them.

    >>> rules = [{"variants": ["mallaire", "malar"], "replacement": "malaire", "pattern": b.variant_pattern("malar")}]
    >>> b.guard_names(rules, read_names(["# a comment", "Malar"])), rules[0]["name"]
    (1, True)
    """
    return {b.fold(row[0]).strip() for row in mr.read_rows(lines)}


def drop_resolved(rules: list[dict], patterns: set[str]) -> tuple[list[dict], int]:
    """``(rules without the variants whose pattern is in patterns, variants dropped)``.

    >>> mk = lambda vs, t: u.m.merge_rules([{"variant": v, "pattern": b.variant_pattern(v), "replacement": t} for v in vs])[0][1]
    >>> rules = [mk(["lodose", "lodoz"], "Lodoz"), mk(["périos"], "périosté")]
    >>> out, n = drop_resolved(rules, resolved_patterns(["lodose\\t*\\tLodoz", "périos\\t*\\tpérioste"]))
    >>> n, [(r["variants"], r["pattern"] == b.variant_pattern("lodoz")) for r in out]
    (2, [(['lodoz'], True)])
    """
    return u.drop_unk_variants(rules, bad=lambda v: b.variant_pattern(v) in patterns)


def label_backed(respell: dict[str, str], labels, convert) -> set[str]:
    """The old replacements whose respelling the labels back: of the labels holding one
    (case-insensitively, as whole words: ``Patidexaméthasone`` does not count for
    ``dexaméthasone``), more than half hold the new spelling once ``convert``-ed. A
    replacement no label holds has no evidence against it and is kept.

    >>> glue = lambda t: t.replace("o-b", "ob").replace("é", "e")  # a convention that ignores case
    >>> sorted(label_backed({"caldeyro-barcia": "caldeyrobarcia", "bévacizumab": "bevacizumab", "x": "y"},
    ...                     ["Signe de Caldeyro-Barcia.", "sous bévacizumab", "Bévacizumab", "anti-bévacizumabs"], glue))
    ['bévacizumab', 'x']
    """
    folded = {old.casefold(): (old, re.compile(rf"(?<!\w){re.escape(old.casefold())}(?!\w)")) for old in respell}
    seen, ok = Counter(), Counter()
    for label in labels:
        low = label.casefold()
        hits = [old for f, (old, word) in folded.items() if f in low and word.search(low)]
        if hits:
            conv = convert(label).casefold()
            for old in hits:
                seen[old] += 1
                ok[old] += respell[old].casefold() in conv
    return {old for old in respell if 2 * ok[old] > seen[old] or not seen[old]}


def apply_conventions(rules: list[dict], labels=None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Respell each replacement as the label conventions do, in place (only those
    ``label_backed`` by ``labels`` when given); returns the ``(old, new)`` replacements
    changed, one per rule, and the respellings refused by the labels.

    >>> rules = [{"replacement": "caldeyro-barcia"}, {"replacement": "bévacizumab"}]
    >>> apply_conventions(rules, ["Signe de Caldeyro-Barcia.", "sous bévacizumab"])
    ([('bévacizumab', 'bevacizumab')], [('caldeyro-barcia', 'caldeyrobarcia')])
    """
    cv = util.load_script("cv", "06_build_convention_fixes.py")  # needs UltiMed-ASR-FR-v1-scripts
    reps = {r["replacement"] for r in rules}
    respell = {rep: new for rep in reps if (new := cv.convention_fixes([rep]).get(rep, rep)) != rep}
    refused = []
    if labels is not None:
        backed = label_backed(respell, labels, cv.apply_label_conventions)
        refused = sorted((o, n) for o, n in respell.items() if o not in backed)
        respell = {o: n for o, n in respell.items() if o in backed}
    changed = []
    for r in rules:
        if r["replacement"] in respell:
            changed.append((r["replacement"], respell[r["replacement"]]))
            r["replacement"] = respell[r["replacement"]]
    return changed, refused



RECASE_MIN = 3  # non-initial label occurrences a word needs before its casing is copied
_WORD_RE = re.compile(rf"[{b._WORD}]+")
_SENTENCE_END = re.compile(r"[.!?:\n]")
_AFTER_WORD = re.compile(rf"[{b._WORD}-]")  # a phrase must not run on into a word or a hyphen
PHRASE_SHARE = 0.8  # share of a phrase's non-initial label occurrences its spelling needs


def label_words(label: str) -> list[tuple[re.Match, bool]]:
    """``label``'s word matches, each with whether it starts a sentence: the text's first
    word, or the first after ``. ! ? :`` or a newline.

    >>> [(m.group(0), s) for m, s in label_words("Vu. Le patient : HTA")]
    [('Vu', True), ('Le', True), ('patient', False), ('HTA', True)]
    """
    out, end = [], 0
    for i, m in enumerate(_WORD_RE.finditer(label)):
        out.append((m, not i or bool(_SENTENCE_END.search(label[end:m.start()]))))
        end = m.end()
    return out


def label_casing(labels, words: set[str], min_count: int = RECASE_MIN) -> dict[str, str]:
    """``lowercased word -> the one cased spelling the labels give it``, for the ``words``
    (lowercase) whose every non-initial occurrence in ``labels`` is spelled the same way,
    seen at least ``min_count`` times (``patient`` maps to itself: its sentence-initial
    capitals are not counted; ``xyz`` is below the threshold). Non-initial: not the text's first word and not right
    after ``. ! ? :`` or a newline (a capital there says nothing about the word).

    >>> labels = ["Score de Gleason 7.", "un Gleason 8", "le Gleason 9", "Le patient. Patient stable",
    ...           "Patient vu, patient stable, patient sorti", "voir HTA", "une HTA", "pas d'HTA", "rare Xyz"]
    >>> label_casing(labels, {"gleason", "patient", "hta", "xyz", "absent"})
    {'gleason': 'Gleason', 'hta': 'HTA', 'patient': 'patient'}
    >>> label_casing(["Titre : Patient", "a Patient", "b Patient", "c Patient"], {"patient"})
    {'patient': 'Patient'}
    """
    seen: dict[str, Counter] = {}
    for label in labels:
        for m, initial in label_words(label):
            w = m.group(0)
            if not initial and w.lower() in words:
                seen.setdefault(w.lower(), Counter())[w] += 1
    return {low: next(iter(c)) for low, c in sorted(seen.items())
            if len(c) == 1 and sum(c.values()) >= min_count}


def recase(rules: list[dict], casing: dict[str, str]) -> list[tuple[str, str]]:
    """Rewrite each replacement's words to their ``casing`` spelling, in place; returns the
    ``(old, new)`` replacements changed, one per rule. Running it twice changes nothing.

    >>> rules = [{"replacement": "score de gleason"}, {"replacement": "patient"}, {"replacement": "l'hta"}]
    >>> recase(rules, {"gleason": "Gleason", "hta": "HTA"}), [r["replacement"] for r in rules]
    ([('score de gleason', 'score de Gleason'), ("l'hta", "l'HTA")], ['score de Gleason', 'patient', "l'HTA"])
    >>> recase(rules, {"gleason": "Gleason", "hta": "HTA"})
    []
    """
    changed = []
    for r in rules:
        new = _WORD_RE.sub(lambda m: casing.get(m.group(0).lower(), m.group(0)), r["replacement"])
        if new != r["replacement"]:
            changed.append((r["replacement"], new))
            r["replacement"] = new
    return changed


def label_phrase_casing(labels, phrases: set[str], min_count: int = RECASE_MIN,
                        share: float = PHRASE_SHARE) -> dict[str, str]:
    """``lowercased phrase -> the cased spelling the labels mostly give it``, for the
    multi-word ``phrases`` (lowercase, words joined by spaces or hyphens). An occurrence is
    the whole phrase, compared without case, not run on into a word or hyphen; one that
    starts a sentence (same definition as ``label_words``) is not counted, since its first
    capital says nothing. Kept when at least ``min_count`` occurrences remain and one
    spelling holds at least ``share`` of them. One pass over the labels: the phrases are
    indexed by their first two words, so a label word only checks the phrases starting
    with it and its neighbour.

    >>> labels = ["la maladie de Charcot-Marie-Tooth", "une Charcot-Marie-Tooth", "pas de Charcot-Marie-Tooth",
    ...           "Charcot-marie-tooth", "vu. Charcot-marie-tooth", "un CKD-EPI", "le CKD-EPI", "le CKD-epi",
    ...           "le Gougerot-Sjogren", "un Gougerot-Sjogren", "un ckd-epi-like"]
    >>> label_phrase_casing(labels, {"charcot-marie-tooth", "ckd-epi", "gougerot-sjogren"})
    {'charcot-marie-tooth': 'Charcot-Marie-Tooth'}
    >>> label_phrase_casing(labels + ["et CKD-EPI", "du CKD-EPI"], {"ckd-epi"})
    {'ckd-epi': 'CKD-EPI'}
    """
    index: dict[tuple[str, str], list[str]] = {}
    for p in sorted(phrases):
        ws = _WORD_RE.findall(p)
        if len(ws) > 1:
            index.setdefault((ws[0], ws[1]), []).append(p)
    seen: dict[str, Counter] = {}
    for label in labels:
        words = label_words(label)
        for (m, initial), (m2, _) in zip(words, words[1:]):
            if initial:
                continue
            for p in index.get((m.group(0).lower(), m2.group(0).lower()), ()):
                s, e = m.start(), m.start() + len(p)
                if label[s:e].lower() == p and not _AFTER_WORD.match(label, e):
                    seen.setdefault(p, Counter())[label[s:e]] += 1
    out = {}
    for p, c in sorted(seen.items()):
        (top, n), total = c.most_common(1)[0], sum(c.values())
        if total >= min_count and n >= share * total:
            out[p] = top
    return out


def recase_phrases(rules: list[dict], casing: dict[str, str]) -> list[tuple[str, str]]:
    """Replace each replacement found whole in ``casing`` (lowercased) by its spelling there,
    in place; returns the ``(old, new)`` changed. Running it twice changes nothing.

    >>> rules = [{"replacement": "Charcot-marie-Tooth"}, {"replacement": "CKD-EPI"}]
    >>> recase_phrases(rules, {"charcot-marie-tooth": "Charcot-Marie-Tooth", "ckd-epi": "CKD-EPI"})
    [('Charcot-marie-Tooth', 'Charcot-Marie-Tooth')]
    >>> recase_phrases(rules, {"charcot-marie-tooth": "Charcot-Marie-Tooth"})
    []
    """
    changed = []
    for r in rules:
        new = casing.get(r["replacement"].lower(), r["replacement"])
        if new != r["replacement"]:
            changed.append((r["replacement"], new))
            r["replacement"] = new
    return changed


def replacement_words(rules: list[dict]) -> set[str]:
    """The lowercased words of every replacement, what ``label_casing`` needs to look at."""
    return {w.lower() for r in rules for w in _WORD_RE.findall(r["replacement"])}


def recase_to_labels(rules: list[dict], texts: list[str]) -> None:
    """Both recasing passes, in place: every word to ``label_casing``, then every multi-word
    replacement to ``label_phrase_casing``. Also used by 05 on the manual file, whose
    replacements come from hand-written TSVs and the lowercase label audit."""
    changed = recase(rules, label_casing(texts, replacement_words(rules)))
    logger.info(f"{len(changed)} replacements recased to the labels' casing: {sorted(set(changed))[:20]}")
    changed = recase_phrases(rules, label_phrase_casing(texts, {r["replacement"].lower() for r in rules}))
    logger.info(f"{len(changed)} replacements recased to the labels' whole-phrase casing: {sorted(set(changed))[:20]}")


@click.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--resolutions", type=click.Path(exists=True, path_type=Path), default=mr.RESOLUTIONS, show_default=True)
@click.option("--names", "names_path", type=click.Path(exists=True, path_type=Path), default=EXTRA_NAMES,
              show_default=True, help="surnames (one per line): their rules do not fire right after a title")
@click.option("--labels", multiple=True, type=click.Path(exists=True, path_type=Path),
              help="NeMo manifests (text field): keep only the respellings their converted labels back")
@click.option("--no-conventions", is_flag=True, help="do not respell the replacements")
@click.option("--no-recase", is_flag=True, help="do not recase the replacements to the labels' casing (words and whole phrases)")
def main(src: Path, dst: Path, resolutions: Path, names_path: Path, labels: tuple[Path, ...], no_conventions: bool,
         no_recase: bool) -> None:
    rules = util.read_jsonl(src)
    n0 = len(rules)
    rules, dropped = drop_resolved(rules, resolved_patterns(resolutions.read_text(encoding="utf-8").splitlines()))
    logger.info(f"{src.name}: dropped {dropped} resolved variants, {n0} -> {len(rules)} rules")
    guarded = b.guard_names(rules, read_names(names_path.read_text(encoding="utf-8").splitlines()))
    logger.info(f"{guarded} more rules kept from firing after a title ({names_path.name})")
    texts = [t for mf in labels for t in util.read_texts(mf, "text")] if labels else None
    if not no_conventions:
        changed, refused = apply_conventions(rules, texts)
        logger.info(f"{len(changed)} replacements respelled by the label conventions: {sorted(set(changed))[:20]}")
        logger.info(f"{len(refused)} respellings refused, the labels do not back them: {refused[:20]}")
    if texts is not None and not no_recase:
        recase_to_labels(rules, texts)
    util.write_jsonl(dst, rules)
    logger.info(f"wrote {dst}")


if __name__ == "__main__":
    main()
