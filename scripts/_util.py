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
import multiprocessing
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Iterator

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


def pmap_ranges(fn: Callable[[int, int], object], n: int, workers: int = 1) -> list:
    """``[fn(lo, hi), ...]`` over ``workers`` contiguous slices of ``range(n)``, in slice order, each in a
    forked process; ``workers <= 1`` (or fewer than two items per worker) runs one ``fn(0, n)`` here.

    ``fn`` reaches the workers through ``fork``, not pickling, so it may be a closure over large data
    (compiled rules, millions of texts) and the scripts need not be importable modules (they are loaded
    by path, see the module docstring); only the results travel back, pickled. Used by the passes that
    run whole rule files over every text: 03's ``merge_checked`` and 11's text checks.

    >>> pmap_ranges(lambda lo, hi: sum(range(lo, hi)), 10, 1), pmap_ranges(lambda lo, hi: (lo, hi), 10, 3)
    ([45], [(0, 3), (3, 6), (6, 10)])
    """
    if workers <= 1 or n < 2 * workers:
        return [fn(0, n)]
    ctx = multiprocessing.get_context("fork")
    queue = ctx.SimpleQueue()
    bounds = [n * k // workers for k in range(workers + 1)]

    def run(k: int) -> None:
        try:
            queue.put((k, fn(bounds[k], bounds[k + 1]), None))
        except BaseException as e:  # noqa: BLE001 (re-raised in the parent)
            queue.put((k, None, repr(e)))

    procs = [ctx.Process(target=run, args=(k,)) for k in range(workers)]
    for p in procs:
        p.start()
    out: dict[int, object] = {}
    try:
        for _ in procs:  # drain before join: a worker blocks on put until its result is read
            k, res, err = queue.get()
            if err is not None:
                raise RuntimeError(f"pmap_ranges worker {k} failed: {err}")
            out[k] = res
    finally:
        for p in procs:
            if len(out) < len(procs):
                p.terminate()
            p.join()
    return [out[k] for k in range(workers)]
