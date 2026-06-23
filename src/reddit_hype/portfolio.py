"""Portfolio construction: universe filtering, the nine strategy selectors, and
weighting schemes (equal / score / vol-scaled / liquidity-adjusted) with a
per-name cap.

Each strategy is a pure function ``(day_df, cfg, settings) -> ranked subset`` so
the backtester can apply any of them, one signal date at a time, with no
lookahead (it only ever sees rows for that date).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------- universe filters
def apply_universe_filters(
    day_df: pd.DataFrame, settings: Settings, overrides: dict | None = None
) -> pd.DataFrame:
    t = dict(settings.strategy_params.get("tradability", {}))
    t.update(overrides or {})
    df = day_df.copy()
    if "dollar_volume_20d" in df:
        df = df[df["dollar_volume_20d"].fillna(0) >= float(t.get("min_dollar_volume", 0))]
    if "market_cap" in df:
        # Only drop on market cap when it is KNOWN and below the floor — keyless
        # (SEC) universes have NaN market cap and shouldn't be filtered to nothing.
        mcap = df["market_cap"]
        df = df[mcap.isna() | (mcap >= float(t.get("min_market_cap", 0)))]
        max_cap = t.get("max_market_cap")
        if max_cap:
            df = df[mcap.isna() | (mcap <= float(max_cap))]
    if t.get("exclude_otc", True) and "is_otc" in df:
        df = df[~df["is_otc"].fillna(False)]
    if t.get("exclude_penny", True) and "close" in df:
        df = df[df["close"].fillna(0) >= float(t.get("penny_price", 1.0))]
    return df


# --------------------------------------------------------------- strategies
def _top(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    d = df.dropna(subset=[col]).copy()
    d["rank_score"] = d[col]
    return d.nlargest(n, "rank_score")


def strat_top_hype(day_df, cfg, settings):
    return _top(day_df, cfg.get("rank_by", "final_hype_alpha_score"), int(cfg.get("top_n", 10)))


def strat_hype_acceleration(day_df, cfg, settings):
    d = day_df.copy()
    if cfg.get("require_spike", True) and "sudden_spike_flag" in d:
        d = d[d["sudden_spike_flag"] == 1]
    maxr = float(cfg.get("max_prior_5d_return", 0.05))
    if "price_return_5d" in d:
        d = d[d["price_return_5d"].abs().fillna(0) <= maxr]
    return _top(d, "hype_velocity_score", int(cfg.get("top_n", settings.strat("portfolio", "top_n", default=10))))


def strat_conviction_only(day_df, cfg, settings):
    d = day_df.copy()
    d = d[d["conviction_score"].fillna(0) >= float(cfg.get("min_conviction", 0.5))]
    d = d[d["unique_authors"].fillna(0) >= int(cfg.get("min_unique_authors", 5))]
    return _top(d, "conviction_score", int(cfg.get("top_n", 10)))


def strat_quality_dd(day_df, cfg, settings):
    d = day_df.copy()
    d = d[d["quality_dd_score"].fillna(0) >= float(cfg.get("min_quality_dd", 0.6))]
    if "post_count" in d:
        d = d[d["post_count"].fillna(0) <= int(cfg.get("max_post_count", 20))]
    return _top(d, "quality_dd_score", int(cfg.get("top_n", 10)))


def strat_underreacted(day_df, cfg, settings):
    d = day_df.copy()
    d = d[d["attention_zscore"].fillna(0) >= float(cfg.get("min_attention_zscore", 1.5))]
    if "price_return_5d" in d:
        d = d[d["price_return_5d"].abs().fillna(0) <= float(cfg.get("max_abs_5d_return", 0.05))]
    return _top(d, "underreaction_score", int(cfg.get("top_n", 10)))


def strat_exhaustion_fade(day_df, cfg, settings):
    # RESEARCH ONLY. We surface the most overheated names; we do NOT assume a
    # tradable short. The backtester refuses to go short for this strategy.
    d = day_df.copy()
    d = d[d["attention_zscore"].fillna(0) >= float(cfg.get("min_attention_zscore", 3.0))]
    return _top(d, "pump_risk_score", int(cfg.get("top_n", 10)))


def strat_small_cap(day_df, cfg, settings):
    d = apply_universe_filters(
        day_df, settings,
        overrides={"min_market_cap": cfg.get("min_market_cap"),
                   "max_market_cap": cfg.get("max_market_cap")},
    )
    return _top(d, "final_hype_alpha_score", int(cfg.get("top_n", 10)))


def strat_sector_rotation(day_df, cfg, settings):
    d = day_df.dropna(subset=["sector"]).copy()
    if d.empty:
        return d.assign(rank_score=[])
    sector_strength = d.groupby("sector")["final_hype_alpha_score"].mean()
    top_sectors = sector_strength.nlargest(int(cfg.get("sector_top_k", 3))).index
    d = d[d["sector"].isin(top_sectors)]
    return _top(d, "final_hype_alpha_score", int(cfg.get("top_n", 10)))


def strat_cross_subreddit_breakout(day_df, cfg, settings):
    d = day_df.copy()
    if "breadth_change" in d:
        d = d[d["breadth_change"].fillna(0) >= float(cfg.get("min_breadth_increase", 2))]
    return _top(d, "cross_subreddit_breadth", int(cfg.get("top_n", 10)))


def strat_squeeze_setup(day_df, cfg, settings):
    # High short interest + accelerating attention. Requires real short-interest
    # data; if absent, has_short_data is False everywhere and nothing is selected.
    d = day_df.copy()
    if "has_short_data" in d:
        d = d[d["has_short_data"].fillna(False)]
    if "short_pct_float" in d:
        d = d[d["short_pct_float"].fillna(0) >= float(cfg.get("min_short_pct_float", 10.0))]
    if cfg.get("require_spike", True) and "sudden_spike_flag" in d:
        d = d[d["sudden_spike_flag"] == 1]
    return _top(d, "squeeze_setup_score", int(cfg.get("top_n", 10)))


def strat_top_mentioned(day_df, cfg, settings):
    # "Trade the N most talked-about stocks." rank_by can be any mention metric
    # (total_mentions, mentions_7d, unique_authors, weighted_attention, ...).
    col = cfg.get("rank_by", "total_mentions")
    if col not in day_df.columns:
        col = "total_mentions"
    return _top(day_df, col, int(cfg.get("top_n", 100)))


STRATEGIES = {
    "TopHypeLongOnly": strat_top_hype,
    "TopMentioned": strat_top_mentioned,
    "HypeAcceleration": strat_hype_acceleration,
    "ConvictionOnly": strat_conviction_only,
    "QualityDDOnly": strat_quality_dd,
    "UnderreactedHype": strat_underreacted,
    "HypeExhaustionFade": strat_exhaustion_fade,
    "SmallCapHype": strat_small_cap,
    "SectorHypeRotation": strat_sector_rotation,
    "CrossSubredditBreakout": strat_cross_subreddit_breakout,
    "SqueezeSetup": strat_squeeze_setup,
}
RESEARCH_ONLY = {"HypeExhaustionFade"}


# --------------------------------------------------------------- weighting
def build_weights(
    selected: pd.DataFrame, weighting: str, max_weight: float, settings: Settings
) -> pd.DataFrame:
    d = selected.copy()
    n = len(d)
    if n == 0:
        d["weight"] = []
        return d
    if weighting == "equal":
        w = np.ones(n)
    elif weighting == "score":
        s = d["rank_score"].to_numpy(dtype="float64")
        s = s - np.nanmin(s) + 1e-9 if np.nanmin(s) < 0 else s + 1e-9
        w = np.nan_to_num(s, nan=0.0)
    elif weighting == "vol_scaled":
        vol = d.get("volatility_20d", pd.Series(np.nan, index=d.index)).to_numpy(dtype="float64")
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nanmedian(vol[vol > 0]) if np.any(vol > 0) else 0.02)
        w = 1.0 / vol
    elif weighting == "liquidity_adjusted":
        ddv = d.get("dollar_volume_20d", pd.Series(1.0, index=d.index)).to_numpy(dtype="float64")
        w = np.sqrt(np.maximum(ddv, 0.0)) + 1e-9
    else:
        w = np.ones(n)

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() == 0:
        w = np.ones(n)
    w = w / w.sum()
    w = _cap_weights(w, max_weight)
    d["weight"] = w
    return d


def _cap_weights(w: np.ndarray, cap: float, iters: int = 100) -> np.ndarray:
    """Iteratively cap weights at ``cap`` (a hard ceiling), redistributing the
    excess proportionally onto the uncapped names. If ``cap * n < 1`` the book
    cannot be fully invested under the cap and a little cash is left over."""
    w = w.astype("float64").copy()
    if cap <= 0 or cap >= 1:
        return w / w.sum()
    w = w / w.sum()
    for _ in range(iters):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap
        if not under.any():
            break
        w[under] += excess * (w[under] / w[under].sum())
    return np.minimum(w, cap)  # hard ceiling; never renormalise back above cap


def select_for_date(
    day_df: pd.DataFrame,
    strategy: str,
    settings: Settings | None = None,
    cfg_override: dict | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    cfg = dict(settings.strategy_params.get("strategies", {}).get(strategy, {}))
    if cfg_override:
        cfg.update(cfg_override)        # lets the mining sweep vary params per run
    pf = settings.strategy_params.get("portfolio", {})
    # cfg_override may also carry tradability overrides (e.g. min_dollar_volume)
    filtered = apply_universe_filters(day_df, settings, overrides=cfg_override)
    fn = STRATEGIES.get(strategy)
    if fn is None:
        raise KeyError(f"Unknown strategy '{strategy}'. Choices: {list(STRATEGIES)}")
    selected = fn(filtered, cfg, settings)
    if selected.empty:
        return selected.assign(weight=[])
    weighting = cfg.get("weighting", pf.get("weighting", "score"))
    max_w = float(cfg.get("max_weight", pf.get("max_weight", 0.2)))
    return build_weights(selected, weighting, max_w, settings)
