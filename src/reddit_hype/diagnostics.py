"""Diagnostics: event study around hype spikes, performance by bucket, capacity
analysis, and the "are we just chasing already-ran stocks?" check.

These are the honesty checks. The point is not to find a number that looks good;
it is to find the conditions under which the signal does and does not work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .config import Settings, load_settings
from .costs import CostModel
from .utils import get_logger

log = get_logger(__name__)


def add_analysis_buckets(panel: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    df = panel.copy()
    if "market_cap" in df:
        df["mcap_bucket"] = pd.cut(
            df["market_cap"],
            bins=[-np.inf, 3e8, 2e9, 1e10, np.inf],
            labels=["micro (<300M)", "small (300M-2B)", "mid (2-10B)", "large (>10B)"],
        ).astype(str)
    if "net_bullish_sentiment" in df:
        df["sentiment_bucket"] = pd.cut(
            df["net_bullish_sentiment"], bins=[-1.01, -0.2, 0.2, 1.01],
            labels=["bearish", "neutral", "bullish"],
        ).astype(str)
    if "final_hype_alpha_score" in df:
        try:
            df["hype_bucket"] = df.groupby("date")["final_hype_alpha_score"].transform(
                lambda s: pd.qcut(s, 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
            ).astype(str)
        except Exception:
            df["hype_bucket"] = "all"
    return df


def event_study(
    panel: pd.DataFrame, settings: Settings | None = None, event_col: str = "sudden_spike_flag"
) -> pd.DataFrame:
    """Average forward (and trailing) returns around abnormal-attention events."""
    settings = settings or load_settings()
    horizons = settings.strat("labels", "horizons", default=[1, 3, 5, 10, 20])
    primary = int(settings.strat("labels", "primary_horizon", default=5))
    events = panel[panel.get(event_col, 0) == 1].copy()
    rows = []
    # pre-event (trailing) returns
    for col, label in [("price_return_5d", "pre_-5d"), ("price_return_1d", "pre_-1d")]:
        if col in events:
            s = events[col].dropna()
            rows.append(_event_row(label, s))
    # post-event forward returns (raw + market-adjusted)
    for h in horizons:
        for prefix, tag in [("fwd_ret", "raw"), ("mkt_adj_ret", "mkt_adj")]:
            col = f"{prefix}_{h}d"
            if col in events:
                rows.append(_event_row(f"fwd_+{h}d_{tag}", events[col].dropna()))
    out = pd.DataFrame(rows)
    n_events = len(events)
    log.info("Event study: %d abnormal-attention events (col=%s, primary=%dd)",
             n_events, event_col, primary)
    out.attrs["n_events"] = n_events
    return out


def _event_row(label: str, s: pd.Series) -> dict:
    n = len(s)
    mean = float(s.mean()) if n else np.nan
    std = float(s.std(ddof=0)) if n else np.nan
    tstat = (mean / (std / np.sqrt(n))) if (n > 1 and std and std > 0) else np.nan
    return {
        # t_stat_naive treats overlapping events as independent and so OVERSTATES
        # significance — use hype_decay_study's HAC (Newey-West) t-stats for inference.
        "window": label, "n": n, "mean_return": mean, "median_return": float(s.median()) if n else np.nan,
        "std": std, "t_stat_naive": tstat, "hit_rate": float((s > 0).mean()) if n else np.nan,
    }


def bucket_performance(
    panel: pd.DataFrame, by: str, ret_col: str | None = None, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or load_settings()
    primary = int(settings.strat("labels", "primary_horizon", default=5))
    ret_col = ret_col or f"fwd_ret_{primary}d"
    if by not in panel.columns or ret_col not in panel.columns:
        return pd.DataFrame()
    g = panel.dropna(subset=[ret_col]).groupby(by)[ret_col]
    out = g.agg(
        n="count", mean_return="mean", median_return="median",
        hit_rate=lambda s: float((s > 0).mean()), std="std",
    ).reset_index()
    out["t_stat"] = out["mean_return"] / (out["std"] / np.sqrt(out["n"].clip(lower=1)))
    return out.sort_values("mean_return", ascending=False).reset_index(drop=True)


def already_ran_check(panel: pd.DataFrame, settings: Settings | None = None) -> dict:
    """Is the signal merely chasing stocks that already ran? Correlate abnormal
    attention with *trailing* return, and compare forward returns of names that
    already ran vs names that did not."""
    settings = settings or load_settings()
    primary = int(settings.strat("labels", "primary_horizon", default=5))
    thr = float(settings.strat("scoring", "already_ran_return_threshold", default=0.15))
    fwd = f"fwd_ret_{primary}d"
    df = panel.dropna(subset=["attention_zscore", "price_return_5d"]).copy()
    corr = df["attention_zscore"].corr(df["price_return_5d"]) if len(df) > 5 else np.nan
    res = {"attn_vs_trailing_return_corr": float(corr) if corr == corr else np.nan}
    if fwd in df.columns:
        ran = df[df["price_return_5d"] >= thr][fwd].dropna()
        not_ran = df[df["price_return_5d"].abs() < thr][fwd].dropna()
        res["already_ran_fwd_mean"] = float(ran.mean()) if len(ran) else np.nan
        res["not_ran_fwd_mean"] = float(not_ran.mean()) if len(not_ran) else np.nan
        res["n_already_ran"] = int(len(ran))
        res["n_not_ran"] = int(len(not_ran))
    log.info("Already-ran check: corr(attn_z, trailing 5d ret)=%.3f", res["attn_vs_trailing_return_corr"])
    return res


def hype_decay_study(
    panel: pd.DataFrame,
    settings: Settings | None = None,
    rank_col: str = "attention_zscore",
    q: float = 0.8,
    min_dollar_volume: float = 0.0,
) -> pd.DataFrame:
    """Forward-return *path* of the top vs bottom attention names, by horizon.

    This is the test for the most influential hypotheses in this space:
    does abnormal attention produce a short pop that then **reverses** (the
    attention-induced-reversal / Barber-Odean story), or genuine continuation?
    Reads the sign and decay of (top-decile − bottom-decile) forward returns
    across horizons. Market-adjusted columns are the honest measure.
    """
    settings = settings or load_settings()
    horizons = settings.strat("labels", "horizons", default=[1, 3, 5, 10, 20])
    if rank_col not in panel.columns:
        return pd.DataFrame()
    df = panel.dropna(subset=[rank_col]).copy()
    if min_dollar_volume > 0 and "dollar_volume_20d" in df.columns:
        before = len(df)
        df = df[df["dollar_volume_20d"].fillna(0) >= min_dollar_volume]
        log.info("Decay study: liquidity filter $%.0f kept %d/%d ticker-days",
                 min_dollar_volume, len(df), before)
    df["_rk"] = df.groupby("date")[rank_col].rank(pct=True)
    top = df["_rk"] >= q
    bot = df["_rk"] <= (1 - q)
    rows = []
    for h in horizons:
        raw, madj = f"fwd_ret_{h}d", f"mkt_adj_ret_{h}d"
        if raw not in df.columns:
            continue
        # Aggregate to ONE observation per date (top-decile cross-sectional mean),
        # then use a Newey-West t-stat with lag=h so overlapping h-day forward
        # returns don't masquerade as independent draws. This is what keeps the
        # tool from "discovering" reversals in pure noise.
        top_by_date = df[top].groupby("date")[raw].mean()
        bot_by_date = df[bot].groupby("date")[raw].mean()
        row = {
            "horizon_days": h,
            "n_dates": int(top_by_date.notna().sum()),
            "n_obs_top": int(df.loc[top, raw].notna().sum()),
            "top_mean_raw": float(top_by_date.mean()) if len(top_by_date) else np.nan,
            "bottom_mean_raw": float(bot_by_date.mean()) if len(bot_by_date) else np.nan,
            "spread_raw": float((top_by_date - bot_by_date).mean())
            if len(top_by_date) and len(bot_by_date) else np.nan,
            "top_hit_rate_by_date": float((top_by_date > 0).mean()) if len(top_by_date) else np.nan,
        }
        if madj in df.columns:
            madj_by_date = df[top].groupby("date")[madj].mean().sort_index()
            row["top_mean_mktadj"] = float(madj_by_date.mean()) if len(madj_by_date) else np.nan
            row["top_mktadj_tstat_nw"] = metrics.newey_west_tstat(madj_by_date.values, lag=h)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["rank_col"] = rank_col
    out.attrs["verdict"] = _decay_verdict(out)
    log.info("Hype-decay study (%s): %s", rank_col, out.attrs["verdict"])
    return out


def _decay_verdict(decay: pd.DataFrame, t_threshold: float = 2.0) -> str:
    """Honest classification of the top-decile market-adjusted path.

    A pattern is only named when the relevant horizons are INDIVIDUALLY HAC-
    significant. A significant short-horizon pop with insignificant later horizons
    is reported as a pop ONLY (the fade is unconfirmed) — we never upgrade noise to
    a "reversal". A multiple-testing reminder is appended."""
    col = "top_mean_mktadj" if "top_mean_mktadj" in decay.columns else "top_mean_raw"
    tcol = "top_mktadj_tstat_nw"
    s = decay.dropna(subset=[col])
    if len(s) < 2 or tcol not in s.columns:
        return "insufficient data"

    n_tests = len(s)
    sig = s[s[tcol].abs() >= t_threshold]
    mx = s[tcol].abs().max()
    if sig.empty:
        return f"NO statistically significant signal (max HAC |t| = {mx:.2f} < {t_threshold})"

    short, long = s.iloc[0], s.iloc[-1]
    short_pop = abs(short[tcol]) >= t_threshold and short[col] > 0
    long_neg = abs(long[tcol]) >= t_threshold and long[col] < 0
    long_pos = abs(long[tcol]) >= t_threshold and long[col] > 0
    sig_horizons = ", ".join(f"{int(r.horizon_days)}d(t={r[tcol]:.1f})" for _, r in sig.iterrows())
    note = (f"  [significant: {sig_horizons}; {n_tests} horizons tested — "
            "treat a lone marginal t as multiple-testing-prone; confirm out-of-sample]")

    if short_pop and long_neg:
        verdict = "POP-AND-FADE: significant short-horizon pop AND significant long-horizon reversal"
    elif short_pop and long_pos:
        verdict = "CONTINUATION: significant positive drift that persists across horizons"
    elif short_pop:
        verdict = (f"SHORT-HORIZON POP ONLY: significant +{short[col]*100:.2f}% at "
                   f"{int(short.horizon_days)}d, but later horizons are NOT significant "
                   "(the fade is unconfirmed — not a reversal)")
    elif long_neg:
        verdict = "LONG-HORIZON UNDERPERFORMANCE: significant negative drift at the long horizon"
    else:
        verdict = "SIGNIFICANT BUT MIXED across horizons"
    return verdict + note


def conditional_study(
    panel: pd.DataFrame,
    mask_fn,
    label: str,
    settings: Settings | None = None,
    min_dollar_volume: float = 1e7,
) -> pd.DataFrame:
    """Test a CONDITION (e.g. accelerating attention, high-DD-quality) rather than
    raw attention. Among liquid names, measures the forward market-adjusted return
    of the conditioned group (per-date mean -> Newey-West t-stat) and the spread vs
    the rest. This is where conditional alpha would show up if it exists."""
    settings = settings or load_settings()
    horizons = settings.strat("labels", "horizons", default=[1, 3, 5, 10, 20])
    df = panel.copy()
    if min_dollar_volume > 0 and "dollar_volume_20d" in df.columns:
        df = df[df["dollar_volume_20d"].fillna(0) >= min_dollar_volume]
    m = mask_fn(df).fillna(False)
    cond, rest = df[m], df[~m]
    rows = []
    for h in horizons:
        madj = f"mkt_adj_ret_{h}d"
        if madj not in df.columns:
            continue
        c = cond.groupby("date")[madj].mean().sort_index()
        r = rest.groupby("date")[madj].mean().sort_index()
        spread = (c - r).dropna()
        rows.append({
            "horizon_days": h,
            "n_obs": int(cond[madj].notna().sum()),
            "n_dates": int(c.notna().sum()),
            "cond_mean_mktadj": float(c.mean()) if len(c) else float("nan"),
            "cond_tstat_nw": metrics.newey_west_tstat(c.values, lag=h),
            "spread_vs_rest": float(spread.mean()) if len(spread) else float("nan"),
            "spread_tstat_nw": metrics.newey_west_tstat(spread.values, lag=h),
        })
    out = pd.DataFrame(rows)
    out.attrs["label"] = label
    out.attrs["verdict"] = _conditional_verdict(out, label)
    log.info("Conditional study [%s]: %s", label, out.attrs["verdict"])
    return out


def _conditional_verdict(tbl: pd.DataFrame, label: str, t_threshold: float = 2.0) -> str:
    s = tbl.dropna(subset=["cond_tstat_nw"])
    if s.empty or s["n_obs"].max() < 30:
        return f"[{label}] too few observations for a reliable test (n_obs<30)"
    sig = s[s["cond_tstat_nw"].abs() >= t_threshold]
    mx = s["cond_tstat_nw"].abs().max()
    if sig.empty:
        return f"[{label}] NO significant predictive signal in liquid names (max HAC |t|={mx:.2f})"
    hor = ", ".join(f"{int(r.horizon_days)}d({r.cond_mean_mktadj*100:+.2f}%,t={r.cond_tstat_nw:.1f})"
                    for _, r in sig.iterrows())
    return f"[{label}] SIGNIFICANT at {hor} — candidate conditional signal; confirm out-of-sample"


def conditional_battery(
    panel: pd.DataFrame, settings: Settings | None = None, min_dollar_volume: float = 1e7
) -> pd.DataFrame:
    """Run the standard conditional hypotheses (acceleration, spike, DD quality,
    and acceleration+DD) on liquid names, with data-adaptive top-quartile cutoffs.
    Returns a tidy table with a ``condition`` column and per-condition verdict."""
    settings = settings or load_settings()
    liq = panel[panel.get("dollar_volume_20d", pd.Series(0, index=panel.index)).fillna(0) >= min_dollar_volume]
    vq = liq["hype_velocity_score"].quantile(0.75) if len(liq) else 1.0
    qq = liq["quality_dd_score"].quantile(0.75) if len(liq) else 1.0
    conds = {
        "acceleration_top_quartile": lambda d: d["hype_velocity_score"] >= vq,
        "sudden_spike": lambda d: d["sudden_spike_flag"] == 1,
        "high_dd_quality": lambda d: d["quality_dd_score"] >= qq,
        "acceleration_and_dd": lambda d: (d["hype_velocity_score"] >= vq) & (d["quality_dd_score"] >= qq),
    }
    frames = []
    for name, fn in conds.items():
        t = conditional_study(panel, fn, name, settings, min_dollar_volume)
        if t.empty:
            continue
        t.insert(0, "condition", name)
        t["verdict"] = t.attrs.get("verdict", "")
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def capacity_report(
    panel: pd.DataFrame, settings: Settings | None = None, top_n: int | None = None
) -> pd.DataFrame:
    """Daily tradable capacity at the participation cap for the top-N names."""
    settings = settings or load_settings()
    costs = CostModel.from_settings(settings)
    top_n = top_n or int(settings.strat("portfolio", "top_n", default=10))
    if "dollar_volume_20d" not in panel.columns:
        return pd.DataFrame()
    rows = []
    for D, g in panel.groupby("date"):
        top = g.nlargest(top_n, "final_hype_alpha_score")
        cap = (costs.max_participation * top["dollar_volume_20d"].fillna(0)).sum()
        rows.append({"date": D, "daily_capacity_usd": float(cap),
                     "min_name_capacity_usd": float((costs.max_participation *
                                                     top["dollar_volume_20d"].fillna(0)).min())})
    out = pd.DataFrame(rows)
    if not out.empty:
        log.info("Capacity: median daily book ~$%.1fM at %.0f%% participation",
                 out["daily_capacity_usd"].median() / 1e6, 100 * costs.max_participation)
    return out
