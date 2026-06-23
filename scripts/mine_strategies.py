#!/usr/bin/env python
"""Data-mine mention-based strategies IN-SAMPLE, then check the winners
OUT-OF-SAMPLE. The in-sample/out-of-sample gap is the real result."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_mine

if __name__ == "__main__":
    out = step_mine()
    print("\n=== IN-SAMPLE leaderboard (top 12) — EXPECT overfitting ===")
    cols = ["rank_by", "top_n", "weighting", "min_dollar_volume", "holding_days",
            "sharpe", "ann_return", "max_drawdown", "n_trades"]
    print(out["in_sample_board"][cols].head(12).to_string(index=False))
    print("\n=== OUT-OF-SAMPLE check of the in-sample winners ===")
    print(out["oos_check"].to_string(index=False))
    print("\nVERDICT:", out["verdict"])
