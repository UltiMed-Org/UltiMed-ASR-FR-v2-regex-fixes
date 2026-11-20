"""Helpers shared by the numbered scripts of this folder (stdlib only, no PEP 723 header).

Every script loads this file by path, under the module name ``ultimed_util``::

    _HERE = Path(__file__).resolve().parent
    _uspec = importlib.util.spec_from_file_location("ultimed_util", _HERE / "_util.py")
    util = importlib.util.module_from_spec(_uspec)
    _uspec.loader.exec_module(util)

and not with ``import _util``, for two reasons. The scripts are themselves loaded by path
(``importlib``) by one another and by the tests, so ``scripts/`` is not on ``sys.path``
there. And the tests have their own ``tests/_util.py``, imported as ``_util`` and kept in
``sys.modules``: a plain ``import _util`` from a script would silently get that one.
Those four lines are the only bootstrap a script repeats; every other sibling module it
needs goes through ``load_script``.

Since this file only uses the standard library, the scripts' PEP 723 dependency lists do
not change, and ``uv run scripts/NN_*.py`` keeps working as before.

This file was written by Claude Code.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Iterable, Iterator

_HERE = Path(__file__).resolve().parent


def load_script(name: str, file: str | Path) -> ModuleType:
    """Execute a script as module ``name`` and return it.

    Parameters
    ----------
    name : str
        Module name, as the importer refers to it (``"b"`` for 02, ``"m"`` for 03, ...).
    file : str or Path
        File name of a sibling script of this folder (the numbered names are not valid
        identifiers, hence no plain import), or an absolute path (``_HERE / file`` is then
        ``file`` itself), e.g. a stage script of the dataset repo.

    Returns
    -------
    ModuleType
        The executed module. It is not registered in ``sys.modules``, like the per-script
        loaders this replaces: each importer gets its own copy.
    """
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_jsonl(path: str | Path) -> list[dict]:
    """Every row of a JSONL file, blank lines skipped (rule files, hypothesis files).

    >>> import tempfile
    >>> p = Path(tempfile.mkdtemp(), "r.jsonl")
    >>> write_jsonl(p, [{"variant": "myrtazapine", "replacement": "mirtazapine"}, {"x": "é"}])
    >>> p.read_text(encoding="utf-8")
    '{"variant": "myrtazapine", "replacement": "mirtazapine"}\\n{"x": "é"}\\n'
    >>> _ = p.write_text(p.read_text(encoding="utf-8") + "\\n", encoding="utf-8")
    >>> read_jsonl(p)
    [{'variant': 'myrtazapine', 'replacement': 'mirtazapine'}, {'x': 'é'}]
    """
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write ``rows`` one JSON object per line, non-ASCII kept as is (UTF-8), in order.

    The whole text is built before the file is opened, so a row that fails to serialise
    leaves no half-written file behind. See ``read_jsonl`` for the doctest.
    """
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    Path(path).write_text(text, encoding="utf-8")


def read_texts(path: str | Path, field: str) -> Iterator[str]:
    """Yield ``field`` of every JSONL row, or every non-empty line of a plain ``.txt``
    file (one sentence per line, e.g. Wikipedia sentences). JSONL lines are not
    filtered: a NeMo manifest has no blank line.

    >>> import tempfile
    >>> d = tempfile.mkdtemp()
    >>> _ = Path(d, "w.txt").write_text("L'abbé est venu.\\n\\ns'il vient\\n", encoding="utf-8")
    >>> list(read_texts(Path(d, "w.txt"), "text"))
    ["L'abbé est venu.", "s'il vient"]
    >>> _ = Path(d, "m.jsonl").write_text('{"text": "a"}\\n', encoding="utf-8")
    >>> list(read_texts(Path(d, "m.jsonl"), "text"))
    ['a']
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if path.suffix == ".txt":
                if line.strip():
                    yield line.rstrip("\n")
            else:
                yield json.loads(line)[field]
