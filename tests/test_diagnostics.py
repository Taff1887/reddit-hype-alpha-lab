"""Squeeze-setup score and the hype-decay / reversal study."""
from __future__ import annotations

import pandas as pd

from reddit_hype.backtester import run_backtest
from reddit_hype.diagnostics import hype_decay_study


def test_squeeze_setup_score_bounded_and_present(scored_panel):
    assert "squeeze_setup_score" in scored_panel.columns
    assert "has_short_data" in scored_panel.columns
    s = scored_panel["squeeze_setup_score"].dropna()
    assert (s >= -1e-9).all() and (s <= 1 + 1e-9).all()
    # mock universe carries synthetic short interest, so the flag should be set
    assert bool(scored_panel["has_short_data"].any())


def test_squeeze_score_zero_without_short_data(scored_panel):
    # any rows lacking short data must score exactly 0 (no fabricated squeeze)
    no_data = scored_panel[~scored_panel["has_short_data"].fillna(False)]
    if len(no_data):
        assert (no_data["squeeze_setup_score"].fillna(0) == 0).all()


def test_hype_decay_study_shape_and_verdict(labeled_panel, settings):
    decay = hype_decay_study(labeled_panel, settings, rank_col="attention_zscore")
    assert not decay.empty
    horizons = settings.strat("labels", "horizons", default=[1, 3, 5, 10, 20])
    assert set(decay["horizon_days"]) <= set(horizons)
    assert {"top_mean_raw", "bottom_mean_raw", "spread_raw"}.issubset(decay.columns)
    assert isinstance(decay.attrs.get("verdict"), str) and decay.attrs["verdict"]


def test_squeeze_strategy_runs(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "SqueezeSetup", 5, settings)
    assert isinstance(res.daily_returns, pd.Series)  # may be empty if no setups, must not crash


def test_conditional_battery_runs(labeled_panel, settings):
    from reddit_hype.diagnostics import conditional_battery

    tbl = conditional_battery(labeled_panel, settings, min_dollar_volume=0.0)
    assert "condition" in tbl.columns
    assert {"cond_mean_mktadj", "cond_tstat_nw", "spread_vs_rest"}.issubset(tbl.columns)
    assert tbl["verdict"].notna().all()
