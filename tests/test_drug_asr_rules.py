#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Tests for the rule build scripts in scripts/: the doctests of 01_extract_drug_errors.py (tokenizing,
drug spotting, the character-level mapping inside a multi-word replace, glued elisions)
and of 02_build_fix_rules.py (pattern bounds, rejection reasons, ordering, apply_rules),
03_merge_rules.py, 04_drop_unk_variants.py, 05_build_manual_rules.py, _util.py (the shared JSONL helpers),
06_build_convention_fixes.py and 09_apply_resolutions.py, plus 10_build_confusion_groups.py
(doctests and a run on a tiny synthetic build folder).

    uv run tests/test_drug_asr_rules.py

The ``01 --hyps-out`` and confusion-group CLI tests and 06's doctests need the UltiMed-ASR-FR-v1-scripts checkout
(drug lexicon, ``label_conventions``; see 01's ``SCRIPTS_DIR``) and are skipped without it.

This file was written by Claude Code.
"""
import doctest
from pathlib import Path

from _util import load_script

STAGE = Path(__file__).resolve().parent.parent / "scripts"


def _has_scripts_dir(test: str) -> bool:
    """False (and says so) when the dataset scripts checkout ``test`` needs is missing."""
    d = load_script("01_extract_drug_errors.py").SCRIPTS_DIR
    if not d.is_dir():
        print(f"{test}: SKIPPED, no UltiMed-ASR-FR-v1-scripts at {d} (set ULTIMED_SCRIPTS_DIR)")
    return d.is_dir()


def _run_doctests(name: str, module: str | None = None) -> None:
    mod = load_script(name, module)
    failed, attempted = doctest.testmod(mod)
    assert attempted and not failed, (name, failed, attempted)
    print(f"{name}: OK ({attempted} cases)")


def test_extract_drug_errors() -> None:
    _run_doctests("01_extract_drug_errors.py")


def test_extract_hyps_out() -> None:
    """``01 --hyps-out`` keeps only the hypotheses of the manifests' clips, in manifest
    order: how the v2 build restricts the whole-corpus transcription to one subset's
    splits before 02 reads it."""
    import json
    import tempfile

    from click.testing import CliRunner

    if not _has_scripts_dir("01 --hyps-out"):
        return
    mod = load_script("01_extract_drug_errors.py")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        hyps = [{"audio": str(d / "drugs" / f"{n}.flac"), "hyp": f"hyp {n}"} for n in ("a", "b", "c")]
        (d / "hyps.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hyps))
        (d / "drugs").mkdir()
        rows = [{"audio_filepath": f"{n}.flac", "text": f"label {n}"} for n in ("c", "a", "z")]
        (d / "drugs" / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        res = CliRunner().invoke(mod.main, [str(d / "hyps.jsonl"), str(d / "drugs" / "train.jsonl"),
                                            "--out", str(d / "err.json"), "--hyps-out", str(d / "sub.jsonl")])
        assert res.exit_code == 0, res.output
        got = [json.loads(line)["hyp"] for line in (d / "sub.jsonl").open()]
        assert got == ["hyp c", "hyp a"], got
    print("01 --hyps-out: OK")


def test_collect_accent_target() -> None:
    """A clip whose label is an accent sibling of its bucket's name keeps that label as its
    target: 18 labels write ``périosté`` and 12 ``périoste``, so the bucket is named
    ``périosté``, and the one ``périos`` hyp (label ``périoste``) must not become a rule to
    ``périosté``. Before the fix 01 compared the folded target with the folded key and
    dropped it."""
    mod = load_script("01_extract_drug_errors.py")
    pairs = ([("lésion du périosté", "lésion du périosté")] * 18 + [("lésion du périoste", "lésion du périoste")] * 12
             + [("lésion du périoste", "lésion du périos")])
    report = mod.collect(pairs, {"perioste": None})
    assert list(report) == ["périosté"], report
    assert report["périosté"]["targets"] == {"périos": "périoste"}, report
    print("01 collect accent target: OK")


def test_hyphenated_short_tail() -> None:
    """A drug whose name has a one-letter part (``DIFFU-K``) is one target, ``diffu-k``,
    whether the label writes ``Diffu-K`` or ``Diffu K``. Before the fix stage 02's needles
    only gave ``diffu`` (its hyphen split, ``k`` too short), so ``Diffu-K`` labels were
    never spotted and ``Diffu K`` ones gave the target ``Diffu``, the K dropped. Hyphenated
    term words whose parts are words keep behaving as before."""
    mod = load_script("01_extract_drug_errors.py")
    lex = {"diffu": "Diffu", **dict.fromkeys(mod.anchor_words("DIFFU-K", ["diffu"])),
           **dict.fromkeys(["sanarelli-schwartzman", "echo-endoscopie", "cranio-encephalique", "loco-regionale"])}
    assert set(lex) >= {"diffu", "diffu-k"}, lex
    for label in ("Diffu-K 600 mg le soir", "Diffu K 600 mg le soir"):
        out = mod.drug_outcomes(label, "difuca 600 mg le soir", lex)
        assert out == [("diffu-k", "diffu-k", "difuca", "diffu-k")], (label, out)
        out = mod.drug_outcomes(label, "Diffu-K 600 mg le soir", lex)
        assert out == [("diffu-k", "diffu-k", None, "diffu-k")], (label, out)
    cases = [
        ("phénomène de Sanarelli-Schwartzman", "phénomène de sanarelli chwarzman", "sanarelli chwarzman", "sanarelli-schwartzman"),
        ("une écho-endoscopie haute", "une échoendoscopie haute", "échoendoscopie", "écho-endoscopie"),
        ("un traumatisme cranio-encéphalique", "un traumatisme cranioencéphalique", "cranioencéphalique", "cranio-encéphalique"),
        ("une anesthésie loco-régionale", "une anesthésie locorégionale", "locorégionale", "loco-régionale"),
    ]
    for label, hyp, variant, target in cases:
        out = mod.drug_outcomes(label, hyp, lex)
        assert [o[2:] for o in out] == [(variant, target)], (label, out)
    # Only a one-letter tail is rejoined: longer parts and short heads stay apart, as before.
    assert mod.drug_outcomes("une anesthésie loco régionale", "une anesthésie locorégionale", lex) == []
    lex["a-deshydrogenase"] = None
    assert mod.drug_outcomes("coenzyme A déshydrogénase", "coenzyme a déshydrogénase", lex) == []
    print("01 hyphenated short tail: OK")


def test_util() -> None:
    # Under the name the scripts give it: as "_util" it would replace tests/_util.py in sys.modules.
    _run_doctests("_util.py", "ultimed_util")


def test_build_fix_rules() -> None:
    _run_doctests("02_build_fix_rules.py")


def test_merge_rules() -> None:
    _run_doctests("03_merge_rules.py")


def test_drop_unk_variants() -> None:
    _run_doctests("04_drop_unk_variants.py")


def test_build_manual_rules() -> None:
    _run_doctests("05_build_manual_rules.py")


def test_owner_spellings_20261120() -> None:
    """The headwords unified in the labels on 2026-11-20 are fixed in a model's output too
    (manual_fixes.tsv), while the right spellings and the English peroxisome stay."""
    mod = load_script("05_build_manual_rules.py")
    tsv = mod.INPUTS / "manual_fixes.tsv"
    fix = mod.b.compile_rules(mod.build_manual_rules(tsv.read_text(encoding="utf-8").splitlines()))
    cases = {
        "une tumeur bégnine de la veine de Cokett": "une tumeur bénigne de la veine de Cockett",
        "Cholestyramine et méthyle aminolévulinate": "Colestyramine et méthyl aminolévulinate",
        "réaction de Sanarelli-Schwartzman, bourse de Thornwaldt": "réaction de Sanarelli-Shwartzman, bourse de Tornwaldt",
        "l'idradénome médulaire en injection souscutanée": "l'hidradénome médullaire en injection sous-cutanée",
        "Pneumocystis jiroveci, lipocalycine": "Pneumocystis jirovecii, lipocaline",
        "la tache de Soemmering": "la tache de Sömmerring",
    }
    for heard, want in cases.items():
        assert fix(heard) == want, (heard, fix(heard))
        assert fix(want) == want, want
    kept = "peroxisome proliferator-activated receptor, hidradénome, Pneumocystis jirovecii, diamètre"
    assert fix(kept) == kept


def test_apply_resolutions() -> None:
    _run_doctests("09_apply_resolutions.py")


def test_build_convention_fixes() -> None:
    if _has_scripts_dir("06_build_convention_fixes.py"):
        _run_doctests("06_build_convention_fixes.py")


def test_convention_fixes_agree() -> None:
    """No two rows of the committed ``convention_fixes.tsv`` share a pattern (accents and
    hyphens folded) with different replacements: the first would silently win. Seven
    such groups shipped until 2026-11-16 (``crâniocaudale`` beside ``craniocaudale``)."""
    if not _has_scripts_dir("convention_fixes.tsv agreement"):
        return
    rows = [line.rstrip("\n").split("\t") for line in open(STAGE.parent / "inputs" / "convention_fixes.tsv", encoding="utf-8")
            if not line.startswith("#")]
    clash = load_script("06_build_convention_fixes.py").collisions(dict(rows))
    assert not clash, clash
    print(f"convention_fixes.tsv agreement: OK ({len(rows)} rows)")


def test_build_confusion_groups() -> None:
    _run_doctests("10_build_confusion_groups.py")


def test_confusion_groups_cli() -> None:
    """10 on a synthetic build: ``florbamab`` was heard for ``zorbamab`` (3 clips, one in
    the test split) and ``quorbamab`` (2), and once where the label had no lexicon word, so
    it is one group with two candidates, ``heard_correct_count`` 1, and the committed-style
    rule ``florbamab -> zorbamab`` as fallback. ``blimid`` has a single target and is no
    group. The examples carry the split, and an examples path inside the repo is refused."""
    import json
    import tempfile

    from click.testing import CliRunner

    if not _has_scripts_dir("10 confusion groups"):
        return
    mod = load_script("10_build_confusion_groups.py")
    b = mod.b
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        build, man = d / "drugs", d / "man" / "drugs"
        build.mkdir()
        man.mkdir(parents=True)
        clips = {  # clip: (label, hyp, split)
            "a": ("prendre zorbamab le soir", "prendre florbamab le soir", "train"),
            "b": ("prendre zorbamab le soir", "prendre florbamab le soir", "val"),
            "c": ("arrêt du zorbamab hier", "arrêt du florbamab hier", "test"),
            "d": ("avec quorbamab ce matin", "avec florbamab ce matin", "train"),
            "e": ("avec quorbamab ce matin", "avec florbamab ce matin", "train"),
            "f": ("un florbamab sans faute", "un florbamab sans faute", "train"),
            "g": ("sous blimide", "sous blimid", "train"),
        }
        report = {"zorbamab": {"n_seen": 3, "n_correct": 0, "n_deleted": 0, "errors": {"florbamab": 3}, "targets": {}},
                  "quorbamab": {"n_seen": 2, "n_correct": 0, "n_deleted": 0, "errors": {"florbamab": 2}, "targets": {}},
                  "blimide": {"n_seen": 1, "n_correct": 0, "n_deleted": 0, "errors": {"blimid": 1}, "targets": {}}}
        (build / "errors.json").write_text(json.dumps(report))
        (build / "hyps.jsonl").write_text("".join(json.dumps({"audio": str(man / f"{c}.flac"), "hyp": h}) + "\n"
                                                  for c, (_, h, _) in clips.items()))
        for split in ("train", "val", "test"):
            (man / f"{split}.jsonl").write_text("".join(json.dumps({"audio_filepath": f"{c}.flac", "text": t}) + "\n"
                                                        for c, (t, _, s) in clips.items() if s == split))
        rules = [{"variant": "florbamab", "pattern": b.variant_pattern("florbamab"), "replacement": "zorbamab"}]
        (d / "rules.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rules))
        args = [str(build), "--out", str(d / "groups.jsonl"), "--rules", str(d / "rules.jsonl"), "--no-respell",
                "--manifests-dir", str(d / "man")]
        res = CliRunner().invoke(mod.main, args + ["--examples-out", str(d / "ex.jsonl")])
        assert res.exit_code == 0, res.output
        groups = [json.loads(line) for line in (d / "groups.jsonl").open()]
        assert groups == [{"heard": "florbamab",
                           "candidates": [{"text": "zorbamab", "count": 3}, {"text": "quorbamab", "count": 2}],
                           "heard_correct_count": 1, "kind": "term", "sources": ["drugs"],
                           "regex_fallback": "zorbamab"}], groups
        ex = [json.loads(line) for line in (d / "ex.jsonl").open()]
        assert sorted((e["ref_target"], e["clip"], e["split"]) for e in ex) == [
            ("quorbamab", "drugs/d.flac", "train"), ("quorbamab", "drugs/e.flac", "train"),
            ("zorbamab", "drugs/a.flac", "train"), ("zorbamab", "drugs/b.flac", "val"),
            ("zorbamab", "drugs/c.flac", "test")], ex
        assert all(e["heard"] == "florbamab" for e in ex), ex
        res = CliRunner().invoke(mod.main, args + ["--min-total", "6"])
        assert res.exit_code == 0 and not (d / "groups.jsonl").read_text(), res.output
        res = CliRunner().invoke(mod.main, args + ["--examples-out", str(mod.REPO / "ex.jsonl")])
        assert res.exit_code != 0 and "inside the repo" in res.output, res.output
        assert not (mod.REPO / "ex.jsonl").exists()
    print("10 confusion groups CLI: OK")


if __name__ == "__main__":
    test_extract_drug_errors()
    test_extract_hyps_out()
    test_collect_accent_target()
    test_hyphenated_short_tail()
    test_util()
    test_build_fix_rules()
    test_merge_rules()
    test_drop_unk_variants()
    test_build_manual_rules()
    test_apply_resolutions()
    test_build_convention_fixes()
    test_convention_fixes_agree()
    test_build_confusion_groups()
    test_confusion_groups_cli()
