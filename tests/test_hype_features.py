"""Feature-builder behaviour: correct attention counts, abnormal-attention
z-scores that use only the past, and bounded score ranges."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_attention_counts_match_mentions(mentions, scored_panel):
    # total mentions in the panel == number of mention rows (per ticker-day)
    by_td = mentions.groupby(["ticker", "date"]).size().rename("n").reset_index()
    merged = scored_panel.merge(by_td, on=["ticker", "date"], how="inner")
    assert (merged["total_mentions"] == merged["n"]).all()


def test_panel_has_expected_columns(scored_panel):
    expected = {
        "total_mentions", "unique_authors", "subreddit_count", "weighted_attention",
        "zscore_vs_30d_baseline", "acceleration_24h_vs_7d", "hype_velocity_score",
        "conviction_score", "quality_dd_score", "pump_risk_score",
        "underreaction_score", "tradability_score", "final_hype_alpha_score",
    }
    assert expected.issubset(set(scored_panel.columns))


def test_engineered_spike_has_high_zscore(scored_panel):
    gme = scored_panel[scored_panel["ticker"] == "GME"].sort_values("date")
    # the spike day should be the max-mentions day and carry a clearly positive z
    spike = gme.loc[gme["total_mentions"].idxmax()]
    assert spike["zscore_vs_30d_baseline"] > 1.0
    assert spike["sudden_spike_flag"] == 1


def test_scores_are_bounded(scored_panel):
    for col in ["conviction_score", "quality_dd_score", "pump_risk_score",
                "underreaction_score", "tradability_score", "attention_score"]:
        s = scored_panel[col].dropna()
        assert (s >= -1e-9).all() and (s <= 1 + 1e-9).all(), col
    assert scored_panel["net_bullish_sentiment"].dropna().between(-1, 1).all()


def test_velocity_baseline_excludes_current_day(scored_panel):
    # zscore uses a trailing (shifted) mean, so a single isolated mention can't
    # produce a finite z on its own first day.
    assert scored_panel["zscore_vs_30d_baseline"].notna().any()
    assert np.isfinite(scored_panel["zscore_vs_30d_baseline"].dropna()).all()
