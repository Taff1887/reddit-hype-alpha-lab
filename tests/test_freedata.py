"""Pure parser for the keyless Yahoo price source (no network in the test)."""
from __future__ import annotations

from reddit_hype.freedata import _parse_yahoo_chart


def test_parse_yahoo_chart_maps_ohlcv():
    payload = {
        "chart": {"result": [{
            "timestamp": [1609459200, 1609545600],
            "indicators": {
                "quote": [{"open": [10.0, 11.0], "high": [12.0, 12.5],
                           "low": [9.5, 10.5], "close": [11.0, 12.0],
                           "volume": [1000, 2000]}],
                "adjclose": [{"adjclose": [10.9, 11.9]}],
            },
        }]}
    }
    df = _parse_yahoo_chart(payload, "GME")
    assert list(df["ticker"]) == ["GME", "GME"]
    assert df["close"].tolist() == [11.0, 12.0]
    assert df["adj_close"].tolist() == [10.9, 11.9]
    assert df["dollar_volume"].tolist() == [11.0 * 1000, 12.0 * 2000]
    assert df["date"].is_monotonic_increasing


def test_parse_yahoo_chart_handles_nulls_and_empty():
    assert _parse_yahoo_chart({}, "X").empty
    payload = {"chart": {"result": [{
        "timestamp": [1609459200, 1609545600],
        "indicators": {"quote": [{"close": [None, 12.0], "volume": [None, 5]}], "adjclose": [{"adjclose": [None, 12.0]}]},
    }]}}
    df = _parse_yahoo_chart(payload, "X")
    assert len(df) == 1 and df.iloc[0]["close"] == 12.0  # the null row is dropped
