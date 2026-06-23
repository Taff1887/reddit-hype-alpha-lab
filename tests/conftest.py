"""Shared test fixtures: a small but realistic end-to-end dataset built through
the real code paths (extractor -> features -> scores -> labels), plus a tiny
hand-crafted price series for exact no-lookahead assertions.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from reddit_hype.config import load_settings
from reddit_hype.fmp_client import MockFmpClient
from reddit_hype.hype_features import build_feature_panel
from reddit_hype.labels import build_labels
from reddit_hype.models import compute_scores
from reddit_hype.reddit_client import REDDIT_ITEM_COLUMNS
from reddit_hype.ticker_extractor import TickerExtractor, build_mention_table

TICKERS = ["NVDA", "AMD", "GME", "CCJ", "MARA"]
# per-ticker baseline items/day; GME gets an engineered spike on one day
BASE_COUNT = {"NVDA": 6, "AMD": 4, "GME": 2, "CCJ": 3, "MARA": 2}
TEXT = {
    "bull": "$%s calls printing, I bought shares today, bullish, holding long. revenue growth strong.",
    "dd": "Deep dive on $%s: revenue up, margins expanding, fundamentals look cheap, catalyst in Q3.",
    "meme": "$%s 🚀🚀 to the moon yolo tendies",
}


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def universe(settings):
    uni = MockFmpClient(settings).universe()
    return uni[uni["ticker"].isin(TICKERS + ["SPY"])].reset_index(drop=True)


@pytest.fixture(scope="session")
def prices(settings):
    p = MockFmpClient(settings).prices(TICKERS + ["SPY"], "2023-01-01", "2030-01-01")
    return p


@pytest.fixture(scope="session")
def signal_days(prices):
    cal = sorted(pd.to_datetime(prices["date"].unique()))
    # leave >=20 trading days of future for forward returns
    return cal[-35:-15]


@pytest.fixture(scope="session")
def items(signal_days):
    rows = []
    uid = 0
    for d in signal_days:
        created = pd.Timestamp(d).tz_localize("UTC") + timedelta(hours=14)
        for t in TICKERS:
            n = BASE_COUNT[t]
            if t == "GME" and d == signal_days[10]:
                n = 25  # engineered attention spike (count only — prices are random)
            for k in range(n):
                uid += 1
                kind = "post" if k % 4 == 0 else "comment"
                style = ["bull", "dd", "meme"][k % 3]
                body = TEXT[style] % t
                rows.append(
                    {
                        "id": f"it_{uid}",
                        "kind": kind,
                        "subreddit": ["wallstreetbets", "stocks", "investing"][k % 3],
                        "author_hash": f"auth_{(uid * 7) % 50}",
                        "created_utc": created.timestamp(),
                        "created_dt": created,
                        "score": 10 + (uid % 30),
                        "title": body if kind == "post" else None,
                        "body": body,
                        "num_comments": 5 if kind == "post" else None,
                        "upvote_ratio": 0.9 if kind == "post" else np.nan,
                        "flair": "DD" if style == "dd" else None,
                        "permalink": f"https://reddit.com/r/x/comments/it_{uid}",
                        "link_id": f"thread_{uid // 3}",
                        "synthetic": True,
                    }
                )
    return pd.DataFrame(rows, columns=REDDIT_ITEM_COLUMNS)


@pytest.fixture(scope="session")
def mentions(items, settings, universe):
    extractor = TickerExtractor.from_settings(settings, universe)
    return build_mention_table(items, extractor, settings)


@pytest.fixture(scope="session")
def scored_panel(mentions, prices, universe, settings):
    panel = build_feature_panel(mentions, prices, universe, settings)
    return compute_scores(panel, settings)


@pytest.fixture(scope="session")
def labeled_panel(scored_panel, prices, settings):
    return build_labels(scored_panel, prices, settings)
