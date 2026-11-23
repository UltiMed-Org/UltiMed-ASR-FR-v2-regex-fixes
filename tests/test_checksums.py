#!/usr/bin/env python3
"""SHA256SUMS must list every published rule file with its current hash.

Fails when a file of rules/ or lookup/ changed (or was added or removed) without
rerunning scripts/13_checksums.sh. Stdlib only: ``python tests/test_checksums.py``
or pytest.

This file was written by Claude Code.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _listed() -> dict[str, str]:
    """{relative path: sha256} from SHA256SUMS ('<hash>  <path>' per line)."""
    out = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        out[path.lstrip("*")] = digest
    return out


def test_sha256sums_up_to_date() -> None:
    listed = _listed()
    published = sorted(
        str(p.relative_to(ROOT))
        for p in [*ROOT.glob("rules/*.jsonl"), *ROOT.glob("lookup/*.json")]
    )
    assert sorted(listed) == published, "file list differs: run scripts/13_checksums.sh"
    for path in published:
        digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert listed[path] == digest, f"{path}: stale hash, run scripts/13_checksums.sh"


if __name__ == "__main__":
    test_sha256sums_up_to_date()
    print("SHA256SUMS ok")
