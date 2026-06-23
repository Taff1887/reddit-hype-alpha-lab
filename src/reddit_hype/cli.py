"""Console entry points declared in pyproject.toml."""
from __future__ import annotations

from . import pipeline
from .config import credential_report


def _entry_fetch_reddit() -> None:
    print(credential_report())
    pipeline.step_fetch_reddit()


def _entry_watchlist() -> None:
    wl = pipeline.step_watchlist()
    if not wl.empty:
        cols = ["ticker", "final_hype_alpha_score", "suggested_action", "reason_summary"]
        print(wl[cols].head(20).to_string(index=False))
