"""Forward-return labels with STRICT no-lookahead.

The contract, for a signal accumulated over calendar date ``D``:

* The signal is considered known only *after the close of D*.
* ``asof_date``  = last trading day <= D   (the prices used by features).
* ``entry_date`` = first trading day **strictly after** D.
* Entry executes at the next open (default) or next close.
* ``fwd_ret_{h}d`` measures entry-price -> close ~h trading days later.

So nothing in a label can be observed at or before the signal timestamp. This
module is covered by ``tests/test_no_lookahead.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import get_logger

log = get_logger(__name__)


class _TickerSeries:
    """Positional access to one ticker's trading-day price arrays."""

    __slots__ = ("dates", "open", "close", "adj")

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("date")
        self.dates = df["date"].values.astype("datetime64[ns]")
        self.open = df["open"].to_numpy(dtype="float64")
        self.close = df["close"].to_numpy(dtype="float64")
        self.adj = df["adj_close"].to_numpy(dtype="float64")

    def entry_index(self, signal_date) -> int:
        # first trading day strictly AFTER the signal date
        return int(np.searchsorted(self.dates, np.datetime64(signal_date, "ns"), side="right"))


def _bench_forward(prices: pd.DataFrame, benchmark: str, horizons, execution: str) -> dict:
    """Benchmark forward returns keyed by (entry_date, horizon)."""
    b = prices[prices["ticker"] == benchmark]
    if b.empty:
        return {}
    s = _TickerSeries(b)
    out: dict[tuple, float] = {}
    for i, dt in enumerate(s.dates):
        for h in horizons:
            if execution == "next_close":
                entry = s.close[i]
                j = i + h
            else:
                entry = s.open[i]
                j = i + h - 1
            if j < len(s.close) and entry and not np.isnan(entry):
                out[(pd.Timestamp(dt), h)] = s.close[j] / entry - 1.0
    return out


def build_labels(
    panel: pd.DataFrame, prices: pd.DataFrame, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or load_settings()
    horizons = list(settings.strat("labels", "horizons", default=[1, 3, 5, 10, 20]))
    execution = settings.strat("labels", "execution", default="next_open")
    benchmark = settings.strat("labels", "market_benchmark", default="SPY")
    primary = int(settings.strat("labels", "primary_horizon", default=5))

    if panel.empty or prices.empty:
        log.warning("Empty panel or prices — no labels computed.")
        return panel

    series = {t: _TickerSeries(g) for t, g in prices.groupby("ticker")}
    bench = _bench_forward(prices, benchmark, horizons, execution)

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    entry_dates, entry_prices, gap = [], [], []
    fwd = {h: [] for h in horizons}
    mdd = []
    oc1 = []  # next-day open-to-close

    for row in df.itertuples(index=False):
        s = series.get(row.ticker)
        if s is None or len(s.dates) == 0:
            entry_dates.append(pd.NaT); entry_prices.append(np.nan); gap.append(np.nan)
            mdd.append(np.nan); oc1.append(np.nan)
            for h in horizons:
                fwd[h].append(np.nan)
            continue
        idx = s.entry_index(row.date)
        if idx >= len(s.dates):
            entry_dates.append(pd.NaT); entry_prices.append(np.nan); gap.append(np.nan)
            mdd.append(np.nan); oc1.append(np.nan)
            for h in horizons:
                fwd[h].append(np.nan)
            continue

        entry_dt = pd.Timestamp(s.dates[idx])
        entry_px = s.close[idx] if execution == "next_close" else s.open[idx]
        entry_dates.append(entry_dt)
        entry_prices.append(entry_px)

        # gap from prior close (the asof close) into the entry open
        prior_close = s.close[idx - 1] if idx >= 1 else np.nan
        gap.append((s.open[idx] / prior_close - 1.0) if prior_close and not np.isnan(prior_close) else np.nan)
        # next-day open-to-close (always open-based, independent of execution)
        oc1.append(s.close[idx] / s.open[idx] - 1.0 if s.open[idx] else np.nan)

        # forward returns per horizon
        worst = np.nan
        for h in horizons:
            j = idx + h if execution == "next_close" else idx + h - 1
            if j < len(s.close) and entry_px and not np.isnan(entry_px):
                fwd[h].append(s.close[j] / entry_px - 1.0)
            else:
                fwd[h].append(np.nan)
        # max drawdown over the primary holding window (close path)
        jp = idx + primary if execution == "next_close" else idx + primary - 1
        if jp < len(s.adj):
            path = s.adj[idx : jp + 1]
            if len(path) > 1 and entry_px:
                run_max = np.maximum.accumulate(path)
                worst = float(np.min(path / run_max - 1.0))
        mdd.append(worst)

    df["asof_date"] = pd.to_datetime(df.get("date"))
    df["entry_date"] = entry_dates
    df["entry_price"] = entry_prices
    df["gap_at_entry"] = gap
    df["gapped_up"] = (df["gap_at_entry"] > 0.03).astype("Int64")
    df["fwd_oc_1d"] = oc1
    for h in horizons:
        df[f"fwd_ret_{h}d"] = fwd[h]
    df[f"max_drawdown_{primary}d"] = mdd

    # market-adjusted (subtract benchmark over the same window/horizon)
    if bench:
        for h in horizons:
            df[f"bench_ret_{h}d"] = [
                bench.get((ed, h), np.nan) if pd.notna(ed) else np.nan for ed in df["entry_date"]
            ]
            df[f"mkt_adj_ret_{h}d"] = df[f"fwd_ret_{h}d"] - df[f"bench_ret_{h}d"]

    # sector-adjusted (cross-sectional sector mean on the signal date) for primary
    pcol = f"fwd_ret_{primary}d"
    if "sector" in df.columns and pcol in df.columns:
        sector_mean = df.groupby(["date", "sector"])[pcol].transform("mean")
        df[f"sector_adj_ret_{primary}d"] = df[pcol] - sector_mean

    # vol-adjusted (per sqrt-time) for primary horizon
    if "volatility_20d" in df.columns:
        denom = (df["volatility_20d"] * np.sqrt(primary)).replace(0, np.nan)
        df[f"vol_adj_ret_{primary}d"] = df[pcol] / denom

    n_lab = df[pcol].notna().sum()
    log.info("Labels built: %d rows, %d with a valid %d-day forward return (exec=%s)",
             len(df), n_lab, primary, execution)
    return df


def validate_no_lookahead(labeled: pd.DataFrame) -> None:
    """Raise if any label's entry happens at or before its signal date."""
    if labeled.empty or "entry_date" not in labeled:
        return
    valid = labeled.dropna(subset=["entry_date"])
    bad = valid[pd.to_datetime(valid["entry_date"]) <= pd.to_datetime(valid["date"])]
    if len(bad):
        raise AssertionError(
            f"No-lookahead violation: {len(bad)} rows have entry_date <= signal date. "
            f"Example: {bad[['ticker', 'date', 'entry_date']].head(3).to_dict('records')}"
        )
