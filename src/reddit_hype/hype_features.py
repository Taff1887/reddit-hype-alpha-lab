"""Build the ticker-day feature panel from the mention table + market data.

Three families of features:

* **Attention** — how much is being said (counts, unique authors/threads, breadth).
* **Velocity** — is attention *accelerating* vs its own trailing baseline
  (computed on a gap-filled daily grid so silent days correctly count as zero).
* **Sentiment / conviction / quality** — engagement-weighted language signals.

plus **market features** joined as-of the signal date (trailing returns, volume
z-score, market cap, sector, volatility, liquidity) — strictly point-in-time:
only data available at the close of the signal date is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import get_logger, safe_div, zscore

log = get_logger(__name__)


# ----------------------------------------------------------------- attention
def aggregate_attention(mentions: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    sub_meta = {s["name"]: s for s in settings.subreddit_list()}
    weight_map = {n: float(m.get("weight", 1.0)) for n, m in sub_meta.items()}
    dd_flairs = set(settings.subreddits.get("dd_flairs", []))
    low_flairs = set(settings.subreddits.get("low_effort_flairs", []))

    m = mentions.copy()
    m["sub_weight"] = m["subreddit"].map(weight_map).fillna(1.0)
    m["is_post"] = m["kind"].eq("post")
    m["is_comment"] = m["kind"].eq("comment")
    m["is_dd_flair"] = m["flair"].isin(dd_flairs)
    m["is_low_flair"] = m["flair"].isin(low_flairs)
    m["post_score"] = np.where(m["is_post"], m["score"], np.nan)
    m["comment_score"] = np.where(m["is_comment"], m["score"], np.nan)
    m["weighted_mention"] = m["sub_weight"] * m["confidence"]

    g = m.groupby(["ticker", "date"], sort=True)
    agg = g.agg(
        total_mentions=("id", "count"),
        post_count=("is_post", "sum"),
        comment_count=("is_comment", "sum"),
        unique_authors=("author_hash", "nunique"),
        unique_threads=("link_id", "nunique"),
        subreddit_count=("subreddit", "nunique"),
        weighted_attention=("weighted_mention", "sum"),
        total_post_score=("post_score", "sum"),
        total_comment_score=("comment_score", "sum"),
        avg_post_score=("post_score", "mean"),
        avg_upvote_ratio=("upvote_ratio", "mean"),
        mean_confidence=("confidence", "mean"),
        cashtag_frac=("has_cashtag", "mean"),
        # sentiment / conviction (engagement-naive means + sums)
        net_sentiment=("net_sentiment", "mean"),
        sentiment_dispersion=("net_sentiment", "std"),
        sum_pos=("n_pos_terms", "sum"),
        sum_neg=("n_neg_terms", "sum"),
        sum_squeeze=("n_squeeze_terms", "sum"),
        sum_commitment=("n_commitment", "sum"),
        sum_bear_commitment=("n_bearish_commitment", "sum"),
        sum_hype=("n_hype_terms", "sum"),
        sum_pump=("n_pump_terms", "sum"),
        sum_panic=("n_panic_terms", "sum"),
        sum_financial=("n_financial_terms", "sum"),
        sum_links=("n_links", "sum"),
        # quality
        avg_word_len=("word_len", "mean"),
        avg_financial_terms=("n_financial_terms", "mean"),
        dd_flair_count=("is_dd_flair", "sum"),
        low_flair_count=("is_low_flair", "sum"),
        low_effort_frac=("low_effort", "mean"),
        pumpy_frac=("pumpy", "mean"),
        bot_spam_mean=("bot_spam_likelihood", "mean"),
        spam_phrase_total=("n_spam_phrases", "sum"),
        any_synthetic=("synthetic", "max"),
    ).reset_index()

    agg["comments_per_post"] = safe_div(agg["comment_count"], agg["post_count"])
    agg["cross_subreddit_breadth"] = agg["subreddit_count"]
    # bounded, mention-normalised language intensities
    agg["bullish_intensity"] = safe_div(agg["sum_pos"], agg["total_mentions"])
    agg["bearish_intensity"] = safe_div(agg["sum_neg"], agg["total_mentions"])
    agg["hype_language_score"] = safe_div(agg["sum_hype"], agg["total_mentions"])
    agg["panic_language_score"] = safe_div(agg["sum_panic"], agg["total_mentions"])
    agg["squeeze_language_score"] = safe_div(agg["sum_squeeze"], agg["total_mentions"])
    agg["pump_language_score"] = safe_div(agg["sum_pump"], agg["total_mentions"])
    net_commit = agg["sum_commitment"] - agg["sum_bear_commitment"]
    agg["conviction_language_score"] = np.tanh(safe_div(net_commit, agg["total_mentions"]) * 2.0)
    agg["sentiment_dispersion"] = agg["sentiment_dispersion"].fillna(0.0)
    return agg


# ------------------------------------------------------------------- velocity
def add_velocity(panel: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Attention velocity/acceleration on a gap-filled daily grid (no lookahead)."""
    baseline_days = int(settings.strat("scoring", "attention_baseline_days", default=30))
    short_w = int(settings.strat("scoring", "velocity_short_window", default=1))
    long_w = int(settings.strat("scoring", "velocity_long_window", default=7))
    spike_z = float(settings.strat("scoring", "spike_zscore_threshold", default=2.0))

    out = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        grp = grp.sort_values("date").set_index("date")
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="D")
        daily = grp["total_mentions"].reindex(full_idx, fill_value=0)

        # trailing baselines use ONLY past days (shifted) to avoid lookahead
        short = daily.rolling(short_w, min_periods=1).sum()
        long_avg = daily.shift(1).rolling(long_w, min_periods=2).mean()
        base_avg = daily.shift(1).rolling(baseline_days, min_periods=5).mean()
        z = zscore(daily, baseline_days, min_periods=5)

        grp["mentions_short"] = short.reindex(grp.index).values
        grp["mentions_3d"] = daily.rolling(3, min_periods=1).sum().reindex(grp.index).values
        grp["mentions_7d"] = daily.rolling(7, min_periods=1).sum().reindex(grp.index).values
        grp["change_vs_7d_avg"] = safe_div(
            daily.reindex(grp.index).values - long_avg.reindex(grp.index).values,
            long_avg.reindex(grp.index).values,
        )
        grp["change_vs_30d_avg"] = safe_div(
            daily.reindex(grp.index).values - base_avg.reindex(grp.index).values,
            base_avg.reindex(grp.index).values,
        )
        grp["zscore_vs_30d_baseline"] = z.reindex(grp.index).values
        grp["acceleration_24h_vs_7d"] = safe_div(
            daily.reindex(grp.index).values, long_avg.reindex(grp.index).values, default=1.0
        ) - 1.0
        grp["sudden_spike_flag"] = (grp["zscore_vs_30d_baseline"] >= spike_z).astype(int)
        # sentiment + breadth change vs prior active day (for cross-subreddit breakout)
        grp["sentiment_change"] = grp["net_sentiment"].diff()
        grp["breadth_change"] = grp["subreddit_count"].diff()
        out.append(grp.reset_index())
    res = pd.concat(out, ignore_index=True)
    res["sentiment_change"] = res["sentiment_change"].fillna(0.0)
    res["breadth_change"] = res["breadth_change"].fillna(0.0)
    return res


# --------------------------------------------------------------- market join
def _price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Trailing price/volume features per ticker (point-in-time at each close)."""
    p = prices.sort_values(["ticker", "date"]).copy()
    grp = p.groupby("ticker", sort=False)
    p["price_return_1d"] = grp["adj_close"].pct_change(1)
    p["price_return_5d"] = grp["adj_close"].pct_change(5)
    p["price_return_20d"] = grp["adj_close"].pct_change(20)
    ret = grp["adj_close"].pct_change(1)
    p["volatility_20d"] = ret.groupby(p["ticker"]).transform(
        lambda s: s.rolling(20, min_periods=5).std()
    )
    p["dollar_volume_20d"] = grp["dollar_volume"].transform(
        lambda s: s.rolling(20, min_periods=5).median()
    )
    vol_mean = grp["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    vol_std = grp["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    p["volume_zscore"] = safe_div(p["volume"] - vol_mean, vol_std)
    # gap risk proxy: 20d high/low range relative to price
    p["gap_risk"] = grp["adj_close"].transform(
        lambda s: s.rolling(20, min_periods=5).std() / s.rolling(20, min_periods=5).mean()
    )
    return p


def add_market_features(
    panel: pd.DataFrame, prices: pd.DataFrame, universe: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """As-of join of trailing market features + static universe metadata."""
    if prices.empty:
        log.warning("No prices supplied; market features will be NaN.")
        for c in ["price_return_1d", "price_return_5d", "price_return_20d",
                  "volatility_20d", "dollar_volume_20d", "volume_zscore", "gap_risk"]:
            panel[c] = np.nan
    else:
        pf = _price_features(prices)
        pf["date"] = pd.to_datetime(pf["date"]).dt.normalize()
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        merged = []
        cols = ["date", "close", "adj_close", "price_return_1d", "price_return_5d",
                "price_return_20d", "volatility_20d", "dollar_volume_20d",
                "volume_zscore", "gap_risk", "dollar_volume"]
        for ticker, grp in panel.groupby("ticker", sort=False):
            pp = pf[pf["ticker"] == ticker][cols].sort_values("date")
            grp = grp.sort_values("date")
            if pp.empty:
                for c in cols[1:]:
                    grp[c] = np.nan
                merged.append(grp)
                continue
            # as-of: latest trading day <= signal date
            j = pd.merge_asof(grp, pp, on="date", direction="backward")
            merged.append(j)
        panel = pd.concat(merged, ignore_index=True)

    # static universe metadata
    meta_cols = ["ticker", "company_name", "exchange", "region", "sector",
                 "industry", "market_cap", "beta", "is_otc", "liquidity_bucket",
                 "short_pct_float", "days_to_cover", "shares_float"]
    have = [c for c in meta_cols if c in universe.columns]
    panel = panel.merge(universe[have], on="ticker", how="left")
    return panel


# --------------------------------------------------------------------- driver
def build_feature_panel(
    mentions: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    settings: Settings | None = None,
) -> pd.DataFrame:
    settings = settings or load_settings()
    if mentions.empty:
        log.warning("No mentions — empty feature panel.")
        return pd.DataFrame()
    attention = aggregate_attention(mentions, settings)
    paneled = add_velocity(attention, settings)
    paneled = add_market_features(paneled, prices, universe, settings)
    paneled = paneled.sort_values(["date", "ticker"]).reset_index(drop=True)
    log.info("Built feature panel: %d ticker-days, %d tickers, %s..%s",
             len(paneled), paneled["ticker"].nunique(),
             paneled["date"].min().date(), paneled["date"].max().date())
    return paneled
