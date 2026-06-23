#!/usr/bin/env python
"""Build the tradable ticker universe from FMP (or synthetic in mock mode)."""
import _bootstrap  # noqa: F401

from reddit_hype.config import credential_report
from reddit_hype.pipeline import step_build_universe

if __name__ == "__main__":
    print(credential_report())
    uni = step_build_universe()
    print(f"Universe built: {len(uni)} tickers.")
    print(uni.head(10).to_string(index=False))
