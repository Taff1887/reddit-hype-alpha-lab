#!/usr/bin/env python
"""List the most talked-about stocks on Reddit over the loaded window, with their
realised period return."""
import argparse

import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_most_talked_about

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    df = step_most_talked_about(top_n=args.top)
    if df.empty:
        print("No mentions loaded. Run a backfill first.")
    else:
        cols = [c for c in ["rank", "ticker", "company_name", "total_mentions",
                            "unique_authors", "days_mentioned", "period_return"] if c in df.columns]
        print(df[cols].head(args.top).to_string(index=False))
