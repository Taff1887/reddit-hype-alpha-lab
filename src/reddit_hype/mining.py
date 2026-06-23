"""Honest strategy mining.

Sweeps a grid of mention-based long-only strategies (rank metric x top-N x
holding x weighting x liquidity floor), but does it the only defensible way:

  1. Split the panel in time (in-sample / out-of-sample).
  2. Backtest every combo on the IN-SAMPLE half, rank by net Sharpe.
  3. Take the in-sample winners and re-run them on the OUT-OF-SAMPLE half.

The gap between in-sample and out-of-sample Sharpe IS the result — it shows how
much of the "best" strategy was overfitting. Mining without this step is how you
fool yourself; the report makes the multiple-testing explicit.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .backtester import run_backtest
from .config import Settings, load_settings
from .utils import get_logger

log = get_logger(__name__)

DEFAULT_GRID = {
    "rank_by": ["total_mentions", "mentions_7d", "unique_authors",
                "weighted_attention", "hype_velocity_score", "final_hype_alpha_score"],
    "top_n": [25, 50, 100],
    "weighting": ["equal", "score"],
    "min_dollar_volume": [0.0, 1e7],     # all names vs liquid (>$10M/day)
    "holding_days": [5, 20],
}


def _split(panel: pd.DataFrame, frac: float = 0.5):
    dates = np.array(sorted(panel["date"].unique()))
    if len(dates) < 6:
        return panel, panel.iloc[0:0]
    cut = dates[int(len(dates) * frac)]
    return panel[panel["date"] <= cut], panel[panel["date"] > cut]


def _run(panel, prices, settings, combo) -> dict:
    cfg = {"rank_by": combo["rank_by"], "top_n": combo["top_n"],
           "weighting": combo["weighting"], "min_dollar_volume": combo["min_dollar_volume"],
           "max_weight": 0.05 if combo["top_n"] >= 25 else 0.2}
    res = run_backtest(panel, prices, "TopMentioned", combo["holding_days"], settings, cfg_override=cfg)
    return {**combo, "sharpe": res.stats.get("sharpe"), "ann_return": res.stats.get("ann_return"),
            "max_drawdown": res.stats.get("max_drawdown"), "n_trades": len(res.trades),
            "turnover": res.turnover}


def mine_mention_strategies(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings | None = None,
    grid: dict | None = None,
    oos_frac: float = 0.5,
    top_k: int = 8,
) -> dict:
    settings = settings or load_settings()
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    log.warning("DATA MINING %d strategy combos IN-SAMPLE — expect overfitting; OOS check follows.",
                len(combos))

    insample, oos = _split(panel, 1 - oos_frac)
    is_rows = [_run(insample, prices, settings, c) for c in combos]
    is_board = pd.DataFrame(is_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # take in-sample winners, re-run on OOS — the honesty check
    oos_rows = []
    for _, r in is_board.head(top_k).iterrows():
        combo = {k: r[k] for k in keys}
        is_sharpe = r["sharpe"]
        o = _run(oos, prices, settings, combo) if not oos.empty else {"sharpe": np.nan}
        oos_rows.append({**combo, "in_sample_sharpe": is_sharpe, "oos_sharpe": o["sharpe"],
                         "oos_ann_return": o.get("ann_return")})
    oos_check = pd.DataFrame(oos_rows)

    n = len(is_board)
    best_is = is_board.iloc[0]["sharpe"] if n else np.nan
    surviving = int((oos_check["oos_sharpe"] > 0.5).sum()) if not oos_check.empty else 0
    verdict = (
        f"Mined {n} combos. Best IN-SAMPLE Sharpe={best_is:.2f}. "
        f"Of the top {len(oos_check)} in-sample winners, {surviving} kept OOS Sharpe>0.5. "
        + ("Some signal MAY survive — confirm on more periods + costs."
           if surviving else "NONE survived out-of-sample — the in-sample winners were overfit noise.")
    )
    log.info(verdict)
    return {"in_sample_board": is_board, "oos_check": oos_check, "verdict": verdict,
            "n_combos": n}
