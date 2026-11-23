#!/usr/bin/env bash
# Write SHA256SUMS at the repository root: the SHA-256 of every published rule file
# (rules/*.jsonl, lookup/*.json), in the standard `sha256sum` format, so an app that
# downloads them can check what it got (`sha256sum -c SHA256SUMS`, or compare with a
# hash pinned in its own code). Run it after any change to rules/ or lookup/;
# tests/test_checksums.py fails while SHA256SUMS is stale.
# Written by Claude Code.
set -euo pipefail
cd "$(dirname "$0")/.."
LC_ALL=C sha256sum rules/*.jsonl rules/*.csv lookup/*.json > SHA256SUMS
echo "SHA256SUMS: $(wc -l < SHA256SUMS) files"
