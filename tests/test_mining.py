"""TopMentioned strategy, most-talked-about report, and the in-sample/OOS miner."""
from __future__ import annotations

import pandas as pd

from reddit_hype.backtester import run_backtest
from reddit_hype.mining import mine_mention_strategies
from reddit_hype.reporting import most_talked_about


def test_top_mentioned_strategy_runs(labeled_panel, prices, settings):
    res = run_backtest(labeled_panel, prices, "TopMentioned", 5, settings,
                       cfg_override={"rank_by": "total_mentions", "top_n": 3, "weighting": "equal"})
    assert isinstance(res.daily_returns, pd.Series)
    assert len(res.trades) > 0


def test_most_talked_about_ranks_by_mentions(mentions, prices, universe, settings):
    tbl = most_talked_about(mentions, prices, universe, settings, top_n=5, save_name="")
    assert not tbl.empty
    assert tbl["rank"].tolist() == sorted(tbl["rank"].tolist())
    # sorted descending by mentions
    assert tbl["total_mentions"].is_monotonic_decreasing
    assert "period_return" in tbl.columns


def test_mining_reports_in_and_out_of_sample(labeled_panel, prices, settings):
    grid = {"rank_by": ["total_mentions", "unique_authors"], "top_n": [3],
            "weighting": ["equal"], "min_dollar_volume": [0.0], "holding_days": [5]}
    out = mine_mention_strategies(labeled_panel, prices, settings, grid=grid, top_k=2)
    assert not out["in_sample_board"].empty
    assert "oos_sharpe" in out["oos_check"].columns
    assert "in_sample_sharpe" in out["oos_check"].columns
    assert isinstance(out["verdict"], str)
