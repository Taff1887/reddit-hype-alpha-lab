#!/usr/bin/env python
"""Fetch recent Reddit posts/comments for the configured subreddits (read-only).

Falls back to deterministic SYNTHETIC data when Reddit credentials are absent.
"""
import _bootstrap  # noqa: F401

from reddit_hype.config import credential_report
from reddit_hype.pipeline import step_fetch_reddit

if __name__ == "__main__":
    print(credential_report())
    df = step_fetch_reddit()
    print(f"Stored {len(df)} Reddit items across {df['subreddit'].nunique()} subreddits.")
