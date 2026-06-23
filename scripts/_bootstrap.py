"""Make ``reddit_hype`` importable whether or not the package is pip-installed.

Each script imports this first so ``python scripts/foo.py`` works from a fresh
clone with no install step.
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
