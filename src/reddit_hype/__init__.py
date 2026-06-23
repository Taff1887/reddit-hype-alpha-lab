"""reddit-hype-alpha-lab.

A research engine that scans Reddit for stock hype, measures attention /
sentiment / velocity, and rigorously backtests whether that hype predicts
forward returns — with strict no-lookahead, transaction costs, and walk-forward
validation.

Public entry point is the :func:`reddit_hype.config.load_settings` object that
every module accepts.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings, load_settings  # noqa: E402,F401

__all__ = ["__version__", "Settings", "load_settings"]
