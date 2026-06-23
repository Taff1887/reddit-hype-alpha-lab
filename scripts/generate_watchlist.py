#!/usr/bin/env python
"""Generate the daily ranked Reddit hype watchlist CSV."""
import argparse

import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_watchlist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="signal date YYYY-MM-DD (default: latest)")
    args = ap.parse_args()
    wl = step_watchlist(date=args.date)
    if wl.empty:
        print("No watchlist produced. Run build-features first.")
    else:
        cols = ["ticker", "company_name", "final_hype_alpha_score", "suggested_action",
                "reason_summary", "risk_summary"]
        print(wl[cols].head(25).to_string(index=False))
