#!/usr/bin/env python
"""KEYLESS real-data backfill + study, in one command.

Pulls real Reddit history (arctic_shift), a real ticker universe (SEC), and real
prices (Yahoo) — NO API keys, NO manual downloads — then builds features and runs
the hype-decay / reversal study on real data.

Examples:
  python scripts/backfill.py --since 2023-06-01 --until 2023-12-31 \\
      --subreddits wallstreetbets,stocks,smallstreetbets
  python scripts/backfill.py --since 2020-09-01 --until 2021-06-30 --include-comments
"""
import argparse

import _bootstrap  # noqa: F401

from reddit_hype.pipeline import backfill_keyless, step_event_study

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2023-06-01")
    ap.add_argument("--until", default="2023-12-31")
    ap.add_argument("--subreddits", default="wallstreetbets,stocks,smallstreetbets,pennystocks")
    ap.add_argument("--max-per-kind", type=int, default=20000, help="cap kept rows per subreddit per kind")
    ap.add_argument("--include-comments", action="store_true", help="also pull comments (much heavier)")
    args = ap.parse_args()

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    summary = backfill_keyless(
        since=args.since, until=args.until, subreddits=subs,
        max_records_per_kind=args.max_per_kind, include_comments=args.include_comments,
    )
    print("\nBackfill summary:", summary)
    if summary.get("panel_rows"):
        out = step_event_study()
        decay = out.get("decay_attention")
        if decay is not None and not decay.empty:
            print("\n=== Hype-decay / reversal study on REAL data (top vs bottom attention decile) ===")
            print(decay.to_string(index=False))
            print("VERDICT:", decay.attrs.get("verdict"))
        print("\nAlready-ran diagnostic:", out.get("already_ran"))
