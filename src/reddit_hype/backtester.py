"""Backtest engine.

Uses the standard overlapping-cohort method (Jegadeesh-Titman style) so that a
"buy top-N, hold H days" rule with *daily* signals is simulated honestly:

* On each signal date D the strategy forms a cohort, entered at the **next open**
  (or next close) — never at or before D (no lookahead; entry dates come from
  :mod:`labels`).
* Each cohort holds for H trading days; it earns open-to-close on its entry day
  and close-to-close thereafter.
* The book is an equal blend of the up-to-H overlapping cohorts (each ~1/H of
  capital), so it is fully invested in steady state and naturally under-invested
  during ramp-up.
* Costs (commission + half-spread + slippage) are charged on each cohort's entry
  and exit sleeve. This ignores name-overlap netting between cohorts, so it is a
  *conservative* (slightly high) cost estimate.

Capacity (participation limits) is reported separately in :mod:`diagnostics`;
the core return stream is capital-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .costs import CostModel
from .metrics import portfolio_stats
from .portfolio import RESEARCH_ONLY, select_for_date
from .utils import annualization_factor, get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    strategy: str
    holding_days: int
    daily_returns: pd.Series           # net of costs
    gross_returns: pd.Series
    trades: pd.DataFrame               # one row per cohort
    stats: dict
    turnover: float
    research_only: bool = False
    meta: dict = field(default_factory=dict)


def _price_matrices(prices: pd.DataFrame):
    p = prices.sort_values(["ticker", "date"]).copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    close = p.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    open_ = p.pivot_table(index="date", columns="ticker", values="open").sort_index()
    open_ = open_.reindex(close.index)
    cc = close.pct_change()                 # close-to-close
    oc = close / open_ - 1.0                # open-to-close (entry day)
    return close.index, cc, oc


def run_backtest(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    strategy: str,
    holding_days: int,
    settings: Settings | None = None,
    cfg_override: dict | None = None,
) -> BacktestResult:
    settings = settings or load_settings()
    costs = CostModel.from_settings(settings)
    one_way = costs.one_way_rate
    H = int(holding_days)

    if panel.empty or prices.empty:
        empty = pd.Series(dtype="float64")
        return BacktestResult(strategy, H, empty, empty, pd.DataFrame(), portfolio_stats(empty), 0.0)

    calendar, cc, oc = _price_matrices(prices)
    cal_pos = {d: i for i, d in enumerate(calendar)}
    n = len(calendar)

    num = np.zeros(n)         # gross daily return contributions (already 1/H scaled)
    cost = np.zeros(n)        # daily cost drag
    traded = np.zeros(n)      # daily traded notional (for turnover)
    trades = []

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    for D, day_df in panel.groupby("date", sort=True):
        sel = select_for_date(day_df, strategy, settings, cfg_override=cfg_override)
        if sel.empty:
            continue
        # all selected share the same next-trading-day entry
        entry_dt = pd.to_datetime(sel["entry_date"].dropna().iloc[0]) if sel["entry_date"].notna().any() else None
        if entry_dt is None or entry_dt not in cal_pos:
            continue
        e = cal_pos[entry_dt]
        tickers = sel["ticker"].tolist()
        w = sel["weight"].to_numpy(dtype="float64")
        invested = float(w.sum())
        holding = min(H, n - e)
        if holding <= 0:
            continue

        compounded = 1.0
        for k in range(holding):
            gidx = e + k
            row = (oc if k == 0 else cc).iloc[gidx]
            r = row.reindex(tickers).to_numpy(dtype="float64")
            r = np.nan_to_num(r, nan=0.0)
            cohort_ret = float(np.dot(w, r))
            num[gidx] += cohort_ret / H
            compounded *= (1 + cohort_ret)
        # entry + exit costs on this cohort's 1/H sleeve
        cost[e] += one_way * invested / H
        traded[e] += invested / H
        exit_idx = e + holding - 1
        cost[exit_idx] += one_way * invested / H
        traded[exit_idx] += invested / H

        trades.append(
            {
                "signal_date": D,
                "entry_date": entry_dt,
                "exit_date": calendar[exit_idx],
                "holding_days": holding,
                "n_names": len(tickers),
                "gross_return": compounded - 1.0,
                "net_return": compounded - 1.0 - 2 * one_way * invested,
                "tickers": ",".join(tickers[:15]),
            }
        )

    gross = pd.Series(num, index=calendar)
    net = gross - pd.Series(cost, index=calendar)
    # trim to the active span
    active = (gross != 0) | (pd.Series(cost, index=calendar) != 0)
    if active.any():
        first, last = active.idxmax(), active[::-1].idxmax()
        gross, net = gross.loc[first:last], net.loc[first:last]

    ann = annualization_factor(settings.strat("portfolio", "rebalance", default="daily"))
    stats = portfolio_stats(net, periods_per_year=ann)
    stats["gross_sharpe"] = portfolio_stats(gross, periods_per_year=ann)["sharpe"]
    turnover = float(np.sum(traded))
    trades_df = pd.DataFrame(trades)
    research = strategy in RESEARCH_ONLY
    log.info(
        "Backtest %s (hold=%dd): net Sharpe=%.2f ann_ret=%.1f%% maxDD=%.1f%% trades=%d%s",
        strategy, H, stats["sharpe"], 100 * stats["ann_return"],
        100 * stats["max_drawdown"], len(trades_df),
        "  [RESEARCH ONLY]" if research else "",
    )
    return BacktestResult(strategy, H, net, gross, trades_df, stats, turnover, research)


def run_strategy_grid(
    panel: pd.DataFrame, prices: pd.DataFrame, settings: Settings | None = None
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Run every enabled strategy across its configured holding periods."""
    settings = settings or load_settings()
    strat_cfg = settings.strategy_params.get("strategies", {})
    default_holds = settings.strat("strategies", "TopHypeLongOnly", "holding_days", default=[1, 3, 5, 10, 20])
    rows, results = [], {}
    for name, cfg in strat_cfg.items():
        if not cfg.get("enabled", False):
            continue
        holds = cfg.get("holding_days", default_holds)
        if isinstance(holds, int):
            holds = [holds]
        for H in holds:
            res = run_backtest(panel, prices, name, H, settings)
            key = f"{name}__{H}d"
            results[key] = res
            row = {"strategy": name, "holding_days": H, "research_only": res.research_only,
                   "turnover": res.turnover, "n_trades": len(res.trades), **res.stats}
            rows.append(row)
    scoreboard = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return scoreboard, results
