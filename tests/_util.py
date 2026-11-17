"""Shared helper of the test files: load a script of ``scripts/`` as a module.

The test files import it as ``from _util import load_script``, which works both when a
test file is run directly (``uv run tests/test_lookup.py`` puts ``tests/`` on
``sys.path``) and under pytest (which prepends each test file's directory).

This file was written by Claude Code.
"""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(file: str, name: str | None = None):
    """Execute ``scripts/<file>`` as module ``name`` (default: the file stem) and return it.

    The module is registered in ``sys.modules`` first: dataclasses look their module up there.
    """
    name = name or Path(file).stem
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
