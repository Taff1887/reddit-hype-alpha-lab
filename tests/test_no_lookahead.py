"""Strict no-lookahead guarantees for forward-return labels.

A hand-crafted price path lets us assert exact entry timing and return values,
plus a property check across the full integration panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reddit_hype.labels import build_labels, validate_no_lookahead


@pytest.fixture()
def hand_prices():
    dates = pd.bdate_range("2024-01-01", periods=12)
    open_ = [10, 10, 10, 10, 20, 21, 22, 23, 24, 25, 26, 27]
    close = [10, 10, 10, 10, 22, 23, 24, 25, 26, 27, 28, 29]
    return pd.DataFrame(
        {
            "ticker": "TST",
            "date": dates,
            "open": open_,
            "high": [c + 1 for c in close],
            "low": [o - 1 for o in open_],
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
            "dollar_volume": [c * 1_000_000 for c in close],
        }
    ), dates


def test_entry_is_next_trading_day_and_returns_exact(hand_prices, settings):
    prices, dates = hand_prices
    panel = pd.DataFrame({"ticker": ["TST"], "date": [dates[3]]})
    out = build_labels(panel, prices, settings)
    row = out.iloc[0]

    # entry is the FIRST trading day strictly after the signal date
    assert pd.Timestamp(row["entry_date"]) == dates[4]
    assert pd.Timestamp(row["entry_date"]) > pd.Timestamp(row["date"])

    # next-open execution: fwd_ret_1d == open-to-close of the entry day
    assert row["fwd_ret_1d"] == pytest.approx(22 / 20 - 1)
    assert row["fwd_oc_1d"] == pytest.approx(22 / 20 - 1)
    assert row["fwd_ret_3d"] == pytest.approx(24 / 20 - 1)
    assert row["fwd_ret_5d"] == pytest.approx(26 / 20 - 1)
    # gap from prior close into the entry open
    assert row["gap_at_entry"] == pytest.approx(20 / 10 - 1)


def test_signal_on_last_day_has_no_future(hand_prices, settings):
    prices, dates = hand_prices
    panel = pd.DataFrame({"ticker": ["TST"], "date": [dates[-1]]})
    out = build_labels(panel, prices, settings)
    assert pd.isna(out.iloc[0]["entry_date"])
    assert pd.isna(out.iloc[0]["fwd_ret_1d"])


def test_validate_no_lookahead_catches_violation():
    bad = pd.DataFrame(
        {"ticker": ["X"], "date": [pd.Timestamp("2024-01-05")],
         "entry_date": [pd.Timestamp("2024-01-05")]}  # same day == violation
    )
    with pytest.raises(AssertionError):
        validate_no_lookahead(bad)


def test_integration_panel_has_no_lookahead(labeled_panel):
    valid = labeled_panel.dropna(subset=["entry_date"])
    assert len(valid) > 0
    assert (pd.to_datetime(valid["entry_date"]) > pd.to_datetime(valid["date"])).all()
    validate_no_lookahead(labeled_panel)  # must not raise
