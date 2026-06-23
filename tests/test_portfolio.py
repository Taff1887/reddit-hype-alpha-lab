"""Portfolio construction: universe filtering, selection caps, and weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reddit_hype.portfolio import (
    _cap_weights,
    apply_universe_filters,
    build_weights,
    select_for_date,
)


def test_select_for_date_respects_caps(scored_panel, settings):
    date = scored_panel["date"].value_counts().idxmax()  # busiest day
    day = scored_panel[scored_panel["date"] == date]
    sel = select_for_date(day, "TopHypeLongOnly", settings)
    top_n = settings.strat("portfolio", "top_n", default=10)
    max_w = settings.strat("portfolio", "max_weight", default=0.2)
    if not sel.empty:
        assert len(sel) <= top_n
        assert sel["weight"].sum() == pytest.approx(1.0)
        assert (sel["weight"] <= max_w + 1e-9).all()


def test_universe_filters_drop_otc_penny_illiquid(settings):
    df = pd.DataFrame(
        {
            "ticker": ["GOOD", "OTCNAME", "PENNY", "ILLIQ"],
            "dollar_volume_20d": [5e6, 5e6, 5e6, 10.0],
            "market_cap": [1e9, 1e9, 1e9, 1e9],
            "is_otc": [False, True, False, False],
            "close": [50.0, 50.0, 0.4, 50.0],
        }
    )
    out = apply_universe_filters(df, settings)
    assert set(out["ticker"]) == {"GOOD"}


def test_build_weights_equal_and_capped(settings):
    sel = pd.DataFrame({"ticker": list("ABC"), "rank_score": [10.0, 1.0, 1.0]})
    eq = build_weights(sel, "equal", max_weight=1.0, settings=settings)
    assert eq["weight"].sum() == pytest.approx(1.0)
    assert eq["weight"].nunique() == 1

    capped = build_weights(sel, "score", max_weight=0.5, settings=settings)
    assert capped["weight"].sum() == pytest.approx(1.0)
    assert (capped["weight"] <= 0.5 + 1e-9).all()


def test_cap_weights_redistributes():
    w = np.array([0.8, 0.1, 0.1])
    out = _cap_weights(w, cap=0.5)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.5 + 1e-9
