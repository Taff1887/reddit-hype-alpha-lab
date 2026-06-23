#!/usr/bin/env python
"""Fetch daily prices/volume from FMP for mentioned tickers + the benchmark."""
import _bootstrap  # noqa: F401

from reddit_hype.config import credential_report
from reddit_hype.pipeline import step_fetch_prices

if __name__ == "__main__":
    print(credential_report())
    df = step_fetch_prices()
    if df.empty:
        print("No prices fetched.")
    else:
        print(f"Stored {len(df)} price rows for {df['ticker'].nunique()} tickers "
              f"({df['date'].min().date()}..{df['date'].max().date()}).")
