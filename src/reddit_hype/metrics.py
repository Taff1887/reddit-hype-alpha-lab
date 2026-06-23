"""Performance + signal-quality metrics.

Two groups:
* **Portfolio metrics** from a daily return series (Sharpe, vol, max drawdown,
  CAGR, hit rate, turnover, ...).
* **Signal metrics** from ranked predictions vs realised forward returns
  (rank IC, precision@k, top-minus-bottom spread).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def newey_west_tstat(x, lag: int) -> float:
    """HAC (Newey-West) t-stat for the mean of a (possibly serially correlated)
    series — e.g. overlapping h-day forward returns. Treating overlapping
    observations as independent badly overstates significance; the Bartlett-
    weighted long-run variance corrects for autocorrelation up to ``lag``.
    """
    s = pd.Series(x).dropna().to_numpy(dtype="float64")
    n = len(s)
    if n < 3:
        return float("nan")
    mu = s.mean()
    e = s - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    L = int(min(max(lag, 1), n - 1))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        cov = float(e[l:] @ e[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = (var / n) ** 0.5
    return float(mu / se) if se > 0 else float("nan")


def portfolio_stats(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> dict:
    """Summary stats from a series of periodic (e.g. daily) net returns."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {k: np.nan for k in
                ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
                 "calmar", "hit_rate", "avg_win", "avg_loss", "n_periods", "cum_return"]}
    cum = float((1 + r).prod() - 1)
    n = len(r)
    ann_return = (1 + r).prod() ** (periods_per_year / n) - 1 if n > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = (r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year)) if r.std(ddof=0) > 0 else np.nan
    downside = r[r < 0].std(ddof=0)
    sortino = (r.mean() / downside * np.sqrt(periods_per_year)) if downside and downside > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    calmar = (ann_return / abs(dd)) if dd < 0 else np.nan
    wins = r[r > 0]
    losses = r[r < 0]
    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "calmar": float(calmar) if calmar == calmar else np.nan,
        "hit_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "n_periods": int(n),
        "cum_return": cum,
    }


def equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).fillna(0).astype(float)
    return (1 + r).cumprod()


def rank_ic(df: pd.DataFrame, score_col: str, ret_col: str, by: str = "date") -> dict:
    """Cross-sectional Spearman rank IC of a score vs forward return."""
    ics = []
    for _, g in df.dropna(subset=[score_col, ret_col]).groupby(by):
        if g[score_col].nunique() > 2 and len(g) >= 5:
            ics.append(g[score_col].corr(g[ret_col], method="spearman"))
    ics = pd.Series(ics).dropna()
    if ics.empty:
        return {"mean_ic": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "n_periods": 0, "hit": np.nan}
    return {
        "mean_ic": float(ics.mean()),
        "ic_std": float(ics.std(ddof=0)),
        "ic_ir": float(ics.mean() / ics.std(ddof=0)) if ics.std(ddof=0) > 0 else np.nan,
        "n_periods": int(len(ics)),
        "hit": float((ics > 0).mean()),
    }


def precision_at_k(df: pd.DataFrame, score_col: str, ret_col: str, k: int, by: str = "date") -> float:
    """Fraction of top-k names (by score, each period) with positive forward return."""
    hits, total = 0, 0
    for _, g in df.dropna(subset=[score_col, ret_col]).groupby(by):
        top = g.nlargest(k, score_col)
        hits += int((top[ret_col] > 0).sum())
        total += len(top)
    return hits / total if total else np.nan


def top_minus_bottom(df: pd.DataFrame, score_col: str, ret_col: str, k: int, by: str = "date") -> dict:
    """Average forward return of the top-k minus the bottom-k each period."""
    tops, bots = [], []
    for _, g in df.dropna(subset=[score_col, ret_col]).groupby(by):
        if len(g) < 2 * k:
            continue
        tops.append(g.nlargest(k, score_col)[ret_col].mean())
        bots.append(g.nsmallest(k, score_col)[ret_col].mean())
    if not tops:
        return {"top_mean": np.nan, "bottom_mean": np.nan, "spread": np.nan, "n_periods": 0}
    t, b = float(np.mean(tops)), float(np.mean(bots))
    return {"top_mean": t, "bottom_mean": b, "spread": t - b, "n_periods": len(tops)}
