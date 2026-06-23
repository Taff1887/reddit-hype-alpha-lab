"""Historical Reddit dump loader: schema mapping, subreddit + date filtering,
and username hashing — exercised on tiny synthetic NDJSON files (no download)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from reddit_hype.reddit_history import load_dumps

IN_WINDOW = datetime(2024, 6, 1, 14, tzinfo=timezone.utc).timestamp()
OUT_OF_WINDOW = datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp()


def _write(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_load_dumps_maps_filters_and_hashes(tmp_path, settings):
    _write(tmp_path / "RS_test.jsonl", [
        {"id": "abc", "subreddit": "wallstreetbets", "author": "alice",
         "created_utc": IN_WINDOW, "score": 100, "title": "GME squeeze",
         "selftext": "$GME yolo", "num_comments": 50, "upvote_ratio": 0.95,
         "link_flair_text": "DD", "permalink": "/r/wallstreetbets/comments/abc/x"},
        {"id": "zzz", "subreddit": "aww", "author": "carol",          # wrong subreddit
         "created_utc": IN_WINDOW, "score": 9, "title": "puppy", "selftext": ""},
        {"id": "old", "subreddit": "wallstreetbets", "author": "dave",  # out of window
         "created_utc": OUT_OF_WINDOW, "score": 9, "title": "old", "selftext": "$GME"},
    ])
    _write(tmp_path / "RC_test.jsonl", [
        {"id": "c1", "subreddit": "wallstreetbets", "author": "bob",
         "created_utc": IN_WINDOW, "score": 5, "body": "I bought $GME shares",
         "link_id": "t3_abc", "permalink": "/r/wallstreetbets/comments/abc/c1"},
    ])

    df = load_dumps(
        settings, input_dir=tmp_path, since="2023-01-01",
        subreddits=["wallstreetbets"], write=False, merge=False,
    )

    assert len(df) == 2  # out-of-subreddit and out-of-window dropped
    post = df[df["kind"] == "post"].iloc[0]
    comment = df[df["kind"] == "comment"].iloc[0]

    assert post["id"] == "t3_abc" and post["flair"] == "DD"
    assert post["title"] == "GME squeeze" and post["body"] == "$GME yolo"
    assert comment["id"] == "t1_c1" and comment["link_id"] == "t3_abc"

    # usernames must be hashed, never stored raw
    assert post["author_hash"] not in ("alice", None)
    assert comment["author_hash"] not in ("bob", None)
    assert df["synthetic"].eq(False).all()


def test_load_dumps_empty_dir_is_safe(tmp_path, settings):
    import pandas as pd

    # no dump files -> no crash, returns a DataFrame (current reddit_items state)
    df = load_dumps(settings, input_dir=tmp_path / "nope", write=False, merge=False)
    assert isinstance(df, pd.DataFrame)
