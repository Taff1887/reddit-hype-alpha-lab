#!/usr/bin/env python
"""Run all enabled strategy backtests and write the scoreboard + scorecard."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_backtest

if __name__ == "__main__":
    sb = step_backtest()
    if sb.empty:
        print("No backtest results. Run build-features (with prices) first.")
    else:
        cols = ["strategy", "holding_days", "sharpe", "ann_return", "max_drawdown",
                "hit_rate", "turnover", "n_trades", "research_only"]
        print(sb[cols].to_string(index=False))
        print("\nScorecard + equity curves written to reports/.")
