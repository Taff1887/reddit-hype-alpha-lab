#!/usr/bin/env python
"""Aggregate mentions + market data into the scored ticker-day feature panel,
then build no-lookahead forward-return labels."""
import _bootstrap  # noqa: F401

from reddit_hype.pipeline import step_build_features

if __name__ == "__main__":
    scored = step_build_features()
    if scored.empty:
        print("Empty panel. Run extract-mentions (and fetch-prices) first.")
    else:
        print(f"Feature panel: {len(scored)} ticker-days, {scored['ticker'].nunique()} tickers.")
        cols = ["date", "ticker", "final_hype_alpha_score", "attention_zscore",
                "hype_velocity_score", "pump_risk_score"]
        have = [c for c in cols if c in scored.columns]
        print(scored.sort_values("final_hype_alpha_score", ascending=False)[have].head(10).to_string(index=False))
