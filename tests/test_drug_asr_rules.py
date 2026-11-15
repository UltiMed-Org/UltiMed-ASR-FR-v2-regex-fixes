#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["click", "loguru", "litellm", "tiktoken", "tqdm", "tenacity", "rapidfuzz", "wordfreq"]
# ///
"""Tests for the rule build scripts in scripts/: the doctests of 01_extract_drug_errors.py (tokenizing,
drug spotting, the character-level mapping inside a multi-word replace, glued elisions)
and of 02_build_fix_rules.py (pattern bounds, rejection reasons, ordering, apply_rules),
03_merge_rules.py, 04_drop_unk_variants.py, 05_build_manual_rules.py and
06_build_convention_fixes.py.

    uv run tests/test_drug_asr_rules.py

The ``01 --hyps-out`` test and 06's doctests need the UltiMed-ASR-FR-v1-scripts checkout
(drug lexicon, ``label_conventions``; see 01's ``SCRIPTS_DIR``) and are skipped without it.

This file was written by Claude Code.
"""
import doctest
import importlib.util
from pathlib import Path

STAGE = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), STAGE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_scripts_dir(test: str) -> bool:
    """False (and says so) when the dataset scripts checkout ``test`` needs is missing."""
    d = _load("01_extract_drug_errors.py").SCRIPTS_DIR
    if not d.is_dir():
        print(f"{test}: SKIPPED, no UltiMed-ASR-FR-v1-scripts at {d} (set ULTIMED_SCRIPTS_DIR)")
    return d.is_dir()


def _run_doctests(name: str) -> None:
    mod = _load(name)
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
    mod = _load("01_extract_drug_errors.py")
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
    mod = _load("01_extract_drug_errors.py")
    pairs = ([("lésion du périosté", "lésion du périosté")] * 18 + [("lésion du périoste", "lésion du périoste")] * 12
             + [("lésion du périoste", "lésion du périos")])
    report = mod.collect(pairs, {"perioste": None})
    assert list(report) == ["périosté"], report
    assert report["périosté"]["targets"] == {"périos": "périoste"}, report
    print("01 collect accent target: OK")


def test_build_fix_rules() -> None:
    _run_doctests("02_build_fix_rules.py")


def test_merge_rules() -> None:
    _run_doctests("03_merge_rules.py")


def test_drop_unk_variants() -> None:
    _run_doctests("04_drop_unk_variants.py")


def test_build_manual_rules() -> None:
    _run_doctests("05_build_manual_rules.py")


def test_build_convention_fixes() -> None:
    if _has_scripts_dir("06_build_convention_fixes.py"):
        _run_doctests("06_build_convention_fixes.py")


if __name__ == "__main__":
    test_extract_drug_errors()
    test_extract_hyps_out()
    test_collect_accent_target()
    test_build_fix_rules()
    test_merge_rules()
    test_drop_unk_variants()
    test_build_manual_rules()
    test_build_convention_fixes()
