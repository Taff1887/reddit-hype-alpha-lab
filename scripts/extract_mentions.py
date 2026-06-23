#!/usr/bin/env python
"""Extract ticker mentions from stored Reddit items (conservative extractor)."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_extract_mentions

if __name__ == "__main__":
    m = step_extract_mentions()
    if m.empty:
        print("No mentions extracted. Did you run fetch-reddit and build-universe?")
    else:
        print(f"Extracted {len(m)} mentions, {m['ticker'].nunique()} distinct tickers.")
        print(m["ticker"].value_counts().head(15).to_string())
