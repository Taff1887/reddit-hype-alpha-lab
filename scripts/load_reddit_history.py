#!/usr/bin/env python
"""Backfill reddit_items from historical Reddit dump files.

Drop monthly dumps (Pushshift / Academic Torrents, e.g. RS_2021-01.zst /
RC_2021-01.zst, or per-subreddit *_submissions.zst) into data/raw/reddit/dumps/,
then run this. Records are filtered to configured subreddits + the date window
WHILE streaming. After loading, run extract-mentions -> build-features ->
event-study to study the period on REAL data.
"""
import argparse

import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_load_history
from reddit_hype.reddit_history import write_dumps_readme

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="dump dir (default data/raw/reddit/dumps)")
    ap.add_argument("--since", default=None, help="ISO date floor (default: config data_start)")
    ap.add_argument("--until", default=None, help="ISO date ceiling")
    ap.add_argument("--subreddits", default=None, help="comma-separated override of subreddits")
    ap.add_argument("--max-records", type=int, default=None, help="safety cap on kept rows")
    args = ap.parse_args()

    write_dumps_readme()  # ensure the dumps/ dir + instructions exist
    subs = args.subreddits.split(",") if args.subreddits else None
    df = step_load_history(
        input_dir=args.input_dir, since=args.since, until=args.until,
        subreddits=subs, max_records=args.max_records,
    )
    if df.empty:
        print("No historical items loaded. Put dump files in data/raw/reddit/dumps/ "
              "(see the README there) and check --since/--until + subreddits.")
    else:
        print(f"Loaded {len(df):,} items. By subreddit:")
        print(df["subreddit"].value_counts().head(20).to_string())
        print("\nNext: python scripts/extract_mentions.py && python scripts/fetch_prices.py "
              "&& python scripts/build_features.py")
