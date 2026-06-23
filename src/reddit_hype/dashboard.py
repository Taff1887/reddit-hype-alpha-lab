"""Streamlit research dashboard.

Run with:  ``streamlit run src/reddit_hype/dashboard.py``  (or ``make dashboard``).

Reads the processed panel + latest watchlist + mentions written by the pipeline.
Importing this module never requires Streamlit (so the test suite stays light);
the UI code only executes when run under ``streamlit``.
"""
from __future__ import annotations

import pandas as pd

from .config import load_settings
from .utils import read_parquet_or_empty


def _load_panel():
    s = load_settings()
    panel = read_parquet_or_empty(s.path("panel"))
    if panel.empty:
        panel = read_parquet_or_empty(s.path("features"))
    mentions = read_parquet_or_empty(s.path("mentions"))
    return s, panel, mentions


def run_dashboard() -> None:  # pragma: no cover - requires streamlit runtime
    import streamlit as st

    st.set_page_config(page_title="Reddit Hype Alpha Lab", layout="wide")
    s, panel, mentions = _load_panel()

    st.title("📈 Reddit Hype Alpha Lab")
    if panel.empty:
        st.warning("No processed panel found. Run `make build-features` (and `make watchlist`) first.")
        return
    if bool(panel.get("any_synthetic", pd.Series([False])).max()):
        st.error("⚠️ This panel contains SYNTHETIC mock data — not real signals. "
                 "Set REDDIT_* and FMP_API_KEY and rebuild for real analysis.")

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    dates = sorted(panel["date"].unique())
    sel_date = st.sidebar.selectbox("Signal date", dates, index=len(dates) - 1)
    day = panel[panel["date"] == sel_date].copy()

    def show(title, df, cols):
        st.subheader(title)
        have = [c for c in cols if c in df.columns]
        st.dataframe(df[have].reset_index(drop=True), use_container_width=True)

    base_cols = ["ticker", "company_name", "sector", "final_hype_alpha_score",
                 "attention_zscore", "hype_velocity_score", "conviction_score",
                 "quality_dd_score", "underreaction_score", "pump_risk_score",
                 "total_mentions", "unique_authors", "subreddit_count",
                 "price_return_5d"]

    tab1, tab2, tab3 = st.tabs(["Leaderboards", "Sector hype", "Ticker detail"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            show("🔥 Top current hype", day.nlargest(15, "final_hype_alpha_score"), base_cols)
            show("🚀 Biggest mention spikes", day.nlargest(15, "zscore_vs_30d_baseline"), base_cols)
            show("💎 Highest conviction", day.nlargest(15, "conviction_score"), base_cols)
        with c2:
            show("🐢 Most underreacted", day.nlargest(15, "underreaction_score"), base_cols)
            show("⚠️ Most overheated (pump risk)", day.nlargest(15, "pump_risk_score"), base_cols)
            if "breadth_change" in day:
                show("🌐 Cross-subreddit breakout", day.nlargest(15, "breadth_change"), base_cols)

    with tab2:
        if "sector" in day:
            sector = day.groupby("sector").agg(
                hype=("final_hype_alpha_score", "mean"),
                mentions=("total_mentions", "sum"),
                names=("ticker", "nunique"),
            ).sort_values("hype", ascending=False)
            st.subheader("Sector hype dashboard")
            st.bar_chart(sector["hype"])
            st.dataframe(sector, use_container_width=True)

    with tab3:
        ticker = st.selectbox("Ticker", sorted(panel["ticker"].unique()))
        tdf = panel[panel["ticker"] == ticker].sort_values("date")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(tdf.set_index("date")[["total_mentions"]])
            st.line_chart(tdf.set_index("date")[["net_sentiment"]] if "net_sentiment" in tdf else tdf.set_index("date")[["final_hype_alpha_score"]])
        with c2:
            if "close" in tdf:
                st.line_chart(tdf.set_index("date")[["close"]])
            st.line_chart(tdf.set_index("date")[["final_hype_alpha_score"]])
        if not mentions.empty:
            recent = mentions[mentions["ticker"] == ticker].sort_values("score", ascending=False).head(15)
            st.subheader("Recent top posts/comments")
            st.dataframe(recent[["date", "subreddit", "score", "permalink"]].reset_index(drop=True),
                         use_container_width=True)


# When launched via `streamlit run`, __name__ == "__main__".
if __name__ == "__main__":  # pragma: no cover
    run_dashboard()
