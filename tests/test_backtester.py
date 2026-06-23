"""Backtester: runs end-to-end, respects no-lookahead at the trade level, and
never marks returns past the available price calendar."""
from __future__ import annotations

import pandas as pd

from reddit_hype.backtester import run_backtest, run_strategy_grid


def test_backtest_runs_and_reports(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "TopHypeLongOnly", 3, settings)
    assert isinstance(res.daily_returns, pd.Series)
    assert "sharpe" in res.stats
    assert len(res.trades) > 0


def test_trades_have_no_lookahead(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "TopHypeLongOnly", 5, settings)
    t = res.trades
    assert (pd.to_datetime(t["entry_date"]) > pd.to_datetime(t["signal_date"])).all()
    assert (pd.to_datetime(t["exit_date"]) >= pd.to_datetime(t["entry_date"])).all()


def test_returns_within_price_calendar(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "TopHypeLongOnly", 5, settings)
    if not res.daily_returns.empty:
        assert res.daily_returns.index.max() <= pd.to_datetime(prices["date"]).max()


def test_costs_reduce_returns(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "TopHypeLongOnly", 5, settings)
    # net return stream is gross minus a non-negative cost drag
    assert res.gross_returns.sum() >= res.daily_returns.sum() - 1e-9


def test_strategy_grid_scoreboard(labeled_panel, prices, settings):
    scoreboard, results = run_strategy_grid(labeled_panel, prices, settings)
    assert not scoreboard.empty
    assert {"strategy", "holding_days", "sharpe", "research_only"}.issubset(scoreboard.columns)
    # the research-only fade strategy, if present, is flagged
    if (scoreboard["strategy"] == "HypeExhaustionFade").any():
        assert scoreboard.loc[scoreboard["strategy"] == "HypeExhaustionFade", "research_only"].all()
