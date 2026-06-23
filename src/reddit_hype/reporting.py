"""Reporting: the daily ranked watchlist, the strategy scoreboard, diagnostic
tables, and figures. Also assigns a human-readable suggested action + reason +
risk summary to each watchlist name.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import ensure_dir, get_logger

log = get_logger(__name__)

WATCHLIST_COLUMNS = [
    "date", "ticker", "company_name", "exchange", "sector", "market_cap",
    "dollar_volume", "final_hype_alpha_score", "attention_score", "velocity_score",
    "sentiment_score", "conviction_score", "quality_dd_score", "underreaction_score",
    "tradability_score", "pump_risk_score", "total_mentions", "post_count",
    "comment_count", "unique_authors", "subreddit_count", "top_subreddits",
    "top_reddit_threads", "price_return_1d", "price_return_5d", "volume_zscore",
    "suggested_action", "reason_summary", "risk_summary",
]


def _top_subreddits(mentions: pd.DataFrame, ticker: str, date) -> str:
    m = mentions[(mentions["ticker"] == ticker) & (mentions["date"] == date)]
    if m.empty:
        return ""
    vc = m["subreddit"].value_counts().head(3)
    return "; ".join(f"r/{s}({n})" for s, n in vc.items())


def _top_threads(mentions: pd.DataFrame, ticker: str, date, k: int = 3) -> str:
    m = mentions[(mentions["ticker"] == ticker) & (mentions["date"] == date)]
    if m.empty or "permalink" not in m:
        return ""
    top = m.sort_values("score", ascending=False).drop_duplicates("permalink").head(k)
    return " | ".join(top["permalink"].fillna("").tolist())


def _action_and_reasons(row: pd.Series, settings: Settings) -> tuple[str, str, str]:
    sc = settings.strategy_params.get("scoring", {})
    min_authors = int(sc.get("min_unique_authors", 3))
    pump = row.get("pump_risk_score", 0) or 0
    already_ran = row.get("already_ran_penalty", 0) or 0
    tradable = bool(row.get("meets_liquidity", False))
    final = row.get("final_hype_alpha_score", 0) or 0
    final_pct = row.get("final_score_pct", 0) or 0
    authors = row.get("unique_authors", 0) or 0
    underreac = row.get("underreaction_score", 0) or 0
    synthetic = bool(row.get("any_synthetic", False))

    # reasons (positive drivers)
    reasons = []
    if (row.get("attention_zscore", 0) or 0) >= 1.5:
        reasons.append(f"abnormal attention (z={row.get('attention_zscore', 0):.1f})")
    if (row.get("hype_velocity_score", 0) or 0) >= 0.3:
        reasons.append("attention accelerating")
    if (row.get("conviction_score", 0) or 0) >= 0.55:
        reasons.append(f"high conviction ({int(authors)} unique authors)")
    if (row.get("quality_dd_score", 0) or 0) >= 0.55:
        reasons.append("substantive DD")
    if underreac >= 0.4:
        reasons.append("price has not moved yet (underreaction)")
    if (row.get("cross_subreddit_breadth", 0) or 0) >= 3:
        reasons.append("broad cross-subreddit interest")
    reason_summary = "; ".join(reasons) or "elevated Reddit attention"

    # risks
    risks = []
    if pump >= 0.5:
        risks.append(f"high pump/spam risk ({pump:.2f})")
    if already_ran >= 0.5:
        risks.append("price may have already run")
    if authors < min_authors:
        risks.append(f"thin breadth (<{min_authors} authors)")
    if not tradable:
        risks.append("below liquidity/market-cap floor")
    if synthetic:
        risks.append("SYNTHETIC/mock data - not a real signal")
    risk_summary = "; ".join(risks) or "no major flags"

    # action
    if pump >= 0.6 or already_ran >= 0.7:
        action = "AVOID_OVERHEATED"
    elif synthetic or not tradable or authors < min_authors:
        action = "RESEARCH_ONLY"
    elif final_pct >= 0.8 and underreac >= 0.3:
        action = "BUY_CANDIDATE"
    else:
        action = "WATCH"
    return action, reason_summary, risk_summary


def generate_watchlist(
    scored_panel: pd.DataFrame,
    mentions: pd.DataFrame,
    settings: Settings | None = None,
    date=None,
    top_n: int = 50,
    save: bool = True,
) -> pd.DataFrame:
    settings = settings or load_settings()
    if scored_panel.empty:
        log.warning("Empty scored panel — no watchlist.")
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    panel = scored_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    if date is None:
        date = panel["date"].max()
    else:
        date = pd.to_datetime(date).normalize()
    day = panel[panel["date"] == date].copy()
    if day.empty:
        log.warning("No rows for %s — latest available is %s", date.date(), panel["date"].max().date())
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    day["velocity_score"] = ((day["hype_velocity_score"].fillna(0) + 1) / 2).clip(0, 1)
    day = day.sort_values("final_hype_alpha_score", ascending=False).head(top_n)

    records = []
    for _, row in day.iterrows():
        action, reason, risk = _action_and_reasons(row, settings)
        records.append(
            {
                "date": date.date().isoformat(),
                "ticker": row["ticker"],
                "company_name": row.get("company_name"),
                "exchange": row.get("exchange"),
                "sector": row.get("sector"),
                "market_cap": row.get("market_cap"),
                "dollar_volume": row.get("dollar_volume_20d"),
                "final_hype_alpha_score": round(float(row.get("final_hype_alpha_score", 0)), 4),
                "attention_score": round(float(row.get("attention_score", 0)), 4),
                "velocity_score": round(float(row.get("velocity_score", 0)), 4),
                "sentiment_score": round(float(row.get("sentiment_score", 0)), 4),
                "conviction_score": round(float(row.get("conviction_score", 0)), 4),
                "quality_dd_score": round(float(row.get("quality_dd_score", 0)), 4),
                "underreaction_score": round(float(row.get("underreaction_score", 0)), 4),
                "tradability_score": round(float(row.get("tradability_score", 0)), 4),
                "pump_risk_score": round(float(row.get("pump_risk_score", 0)), 4),
                "total_mentions": int(row.get("total_mentions", 0) or 0),
                "post_count": int(row.get("post_count", 0) or 0),
                "comment_count": int(row.get("comment_count", 0) or 0),
                "unique_authors": int(row.get("unique_authors", 0) or 0),
                "subreddit_count": int(row.get("subreddit_count", 0) or 0),
                "top_subreddits": _top_subreddits(mentions, row["ticker"], date) if not mentions.empty else "",
                "top_reddit_threads": _top_threads(mentions, row["ticker"], date) if not mentions.empty else "",
                "price_return_1d": _r(row.get("price_return_1d")),
                "price_return_5d": _r(row.get("price_return_5d")),
                "volume_zscore": _r(row.get("volume_zscore")),
                "suggested_action": action,
                "reason_summary": reason,
                "risk_summary": risk,
            }
        )
    wl = pd.DataFrame(records, columns=WATCHLIST_COLUMNS)
    if save:
        out_dir = ensure_dir(settings.path("watchlists"))
        path = out_dir / f"{date.date().isoformat()}_reddit_hype_watchlist.csv"
        wl.to_csv(path, index=False)
        log.info("Wrote watchlist (%d names) -> %s", len(wl), path)
    return wl


def most_talked_about(
    mentions: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    settings: Settings | None = None,
    top_n: int = 100,
    save_name: str = "most_talked_about",
) -> pd.DataFrame:
    """Rank tickers by how much Reddit talked about them over the whole window,
    with descriptive stats and the realised period return (NOT a forward bet —
    purely 'who got attention and how did they do')."""
    settings = settings or load_settings()
    if mentions.empty:
        return pd.DataFrame()
    agg = mentions.groupby("ticker").agg(
        total_mentions=("id", "count"),
        unique_authors=("author_hash", "nunique"),
        days_mentioned=("date", "nunique"),
        post_count=("kind", lambda s: int((s == "post").sum())),
        avg_score=("score", "mean"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    ).reset_index()

    if prices is not None and not prices.empty:
        rets = []
        p = prices.sort_values("date")
        for tk in agg["ticker"]:
            g = p[p["ticker"] == tk]
            rets.append(g["adj_close"].iloc[-1] / g["adj_close"].iloc[0] - 1 if len(g) > 1 else np.nan)
        agg["period_return"] = rets
    if universe is not None and "company_name" in getattr(universe, "columns", []):
        agg = agg.merge(universe[["ticker", "company_name"]], on="ticker", how="left")

    agg = agg.sort_values("total_mentions", ascending=False).head(top_n).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    if save_name:
        save_table(agg, save_name, settings)
    return agg


def _r(x, nd: int = 4):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return np.nan


# ----------------------------------------------------------------- scoreboard
def save_scoreboard(scoreboard: pd.DataFrame, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    out_dir = ensure_dir(settings.path("scorecards"))
    path = out_dir / "strategy_scoreboard.csv"
    scoreboard.to_csv(path, index=False)
    log.info("Wrote strategy scoreboard -> %s", path)
    return str(path)


def save_table(df: pd.DataFrame, name: str, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    out_dir = ensure_dir(settings.path("tables"))
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------- figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_equity_curves(results: dict, settings: Settings | None = None, fname="equity_curves.png") -> str | None:
    settings = settings or load_settings()
    try:
        plt = _mpl()
    except Exception as exc:  # pragma: no cover
        log.warning("matplotlib unavailable: %s", exc)
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, res in sorted(results.items(), key=lambda kv: -kv[1].stats.get("sharpe", -9)):
        if res.daily_returns.empty:
            continue
        eq = (1 + res.daily_returns.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, label=f"{key} (Sh {res.stats.get('sharpe', float('nan')):.2f})")
    ax.set_title("Strategy equity curves (net of costs)")
    ax.set_ylabel("Growth of $1"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    out = ensure_dir(settings.path("figures")) / fname
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    log.info("Wrote figure -> %s", out)
    return str(out)


def plot_event_study(event_df: pd.DataFrame, settings: Settings | None = None, fname="event_study.png") -> str | None:
    settings = settings or load_settings()
    if event_df.empty:
        return None
    try:
        plt = _mpl()
    except Exception:  # pragma: no cover
        return None
    raw = event_df[event_df["window"].str.contains("fwd_") & event_df["window"].str.endswith("raw")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(raw["window"], raw["mean_return"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Forward returns after abnormal-attention events (n={event_df.attrs.get('n_events', '?')})")
    ax.set_ylabel("Mean forward return"); plt.xticks(rotation=45, ha="right"); ax.grid(alpha=0.3)
    out = ensure_dir(settings.path("figures")) / fname
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return str(out)


def write_scorecard_md(
    scoreboard: pd.DataFrame, diagnostics: dict, settings: Settings | None = None
) -> str:
    """Write a markdown scorecard with an explicit, honest alpha verdict."""
    settings = settings or load_settings()
    out_dir = ensure_dir(settings.path("scorecards"))
    path = out_dir / "scorecard.md"
    acc = settings.strategy_params.get("research_loop", {}).get("alpha_acceptance", {})
    best = scoreboard[~scoreboard["research_only"]].head(1) if not scoreboard.empty else pd.DataFrame()
    lines = [
        "# Reddit Hype Alpha — Scorecard",
        f"_Generated {datetime.utcnow().isoformat(timespec='seconds')}Z_",
        "",
        "## Alpha acceptance gate",
        f"- min net Sharpe: {acc.get('min_sharpe_net')}",
        f"- min precision@10: {acc.get('min_precision_at_10')}",
        "",
        "## Top strategies (net of costs)",
        scoreboard.head(10).to_markdown(index=False) if not scoreboard.empty else "_no results_",
        "",
        "## Diagnostics",
    ]
    for k, v in diagnostics.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Verdict", _verdict(best, diagnostics, acc, settings)]
    path.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    log.info("Wrote scorecard -> %s", path)
    return str(path)


def _verdict(best: pd.DataFrame, diagnostics: dict, acc: dict, settings: Settings) -> str:
    if best.empty:
        return "No tradable strategy produced results."
    row = best.iloc[0]
    if bool(diagnostics.get("synthetic_data", False)):
        return (
            "⚠️ Results computed on SYNTHETIC mock data (random-walk prices with no hype "
            "linkage). They demonstrate the pipeline ONLY and must not be read as alpha. "
            "Provide REDDIT_* and FMP_API_KEY and re-run on real data before any conclusion."
        )
    sharpe_ok = row["sharpe"] >= float(acc.get("min_sharpe_net", 0.5))
    verdict = "PASSES" if sharpe_ok else "DOES NOT pass"
    return (
        f"Best net strategy: **{row['strategy']} (hold {int(row['holding_days'])}d)**, "
        f"net Sharpe {row['sharpe']:.2f}, ann. return {100*row['ann_return']:.1f}%, "
        f"maxDD {100*row['max_drawdown']:.1f}%. This {verdict} the net-Sharpe gate. "
        "Do not claim alpha until it also survives out-of-sample periods, is not driven by "
        "one meme event, and has a plausible behavioural explanation."
    )
