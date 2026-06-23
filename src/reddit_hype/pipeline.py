"""End-to-end pipeline orchestration. Each ``step_*`` reads its inputs from the
configured paths, runs one stage, and writes its output — so the scripts in
``scripts/`` and the console entry points are thin wrappers around these.

Data flow:
    universe -> reddit -> mentions -> prices -> features(+scores+labels)
             -> watchlist / event-study / backtest / research-loop
"""
from __future__ import annotations

import pandas as pd

from . import diagnostics as dg
from . import reporting as rp
from .backtester import run_strategy_grid
from .config import Settings, credential_report, load_settings
from .fmp_client import get_fmp_client
from .hype_features import build_feature_panel
from .labels import build_labels, validate_no_lookahead
from .models import compute_scores
from .reddit_client import get_reddit_client
from .research_loop import run_research_loop
from .ticker_extractor import TickerExtractor, build_mention_table
from .ticker_universe import build_universe, load_universe, save_universe
from .utils import get_logger, read_parquet, read_parquet_or_empty, write_parquet

log = get_logger(__name__)


def _merge_dedup(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new
    if new.empty:
        return existing
    combined = pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------- steps
def step_build_universe(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    uni = build_universe(settings)
    save_universe(uni, settings)
    return uni


def step_fetch_reddit(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    client = get_reddit_client(settings)
    new = client.fetch()
    path = settings.path("reddit_items")
    existing = read_parquet_or_empty(path)
    merged = _merge_dedup(existing, new, keys=["id"])
    # keep within the configured research window
    start = pd.Timestamp(settings.get("data_start", default="2000-01-01"), tz="UTC")
    merged = merged[pd.to_datetime(merged["created_dt"], utc=True) >= start]
    write_parquet(merged, path)
    log.info("Reddit items: %d total after merge -> %s", len(merged), path)
    return merged


def step_load_history(settings: Settings | None = None, **kwargs) -> pd.DataFrame:
    """Backfill reddit_items from historical dump files in data/raw/reddit/dumps."""
    from .reddit_history import load_dumps

    settings = settings or load_settings()
    return load_dumps(settings, **kwargs)


def step_extract_mentions(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    items = read_parquet(settings.path("reddit_items"))
    universe = read_parquet_or_empty(settings.path("ticker_universe"))
    extractor = TickerExtractor.from_settings(settings, universe if not universe.empty else None)
    mentions = build_mention_table(items, extractor, settings)
    write_parquet(mentions, settings.path("mentions"))
    return mentions


def step_fetch_prices(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    client = get_fmp_client(settings)
    benchmark = settings.strat("labels", "market_benchmark", default="SPY")

    mentions = read_parquet_or_empty(settings.path("mentions"))
    if not mentions.empty:
        tickers = sorted(set(mentions["ticker"].unique()) | {benchmark})
    else:
        uni = read_parquet_or_empty(settings.path("ticker_universe"))
        tickers = sorted(set(uni["ticker"].tolist()) | {benchmark}) if not uni.empty else [benchmark]
        log.warning("No mentions yet — fetching prices for the whole universe (%d).", len(tickers))

    start = settings.get("data_start", default="2023-01-01")
    end = pd.Timestamp.utcnow().date().isoformat()
    new = client.prices(tickers, start, end)
    path = settings.path("prices")
    existing = read_parquet_or_empty(path)
    merged = _merge_dedup(existing, new, keys=["ticker", "date"])
    write_parquet(merged, path)
    log.info("Prices: %d rows, %d tickers -> %s", len(merged), merged["ticker"].nunique() if not merged.empty else 0, path)
    return merged


def step_build_features(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    mentions = read_parquet(settings.path("mentions"))
    prices = read_parquet_or_empty(settings.path("prices"))
    universe = read_parquet_or_empty(settings.path("ticker_universe"))

    panel = build_feature_panel(mentions, prices, universe, settings)
    scored = compute_scores(panel, settings)
    write_parquet(scored, settings.path("features"))

    if not prices.empty:
        labeled = build_labels(scored, prices, settings)
        validate_no_lookahead(labeled)
        write_parquet(labeled, settings.path("panel"))
        label_cols = ["ticker", "date", "entry_date"] + [c for c in labeled.columns if c.startswith(("fwd_", "mkt_adj", "sector_adj", "vol_adj", "max_drawdown", "bench_"))]
        write_parquet(labeled[label_cols], settings.path("labels"))
    else:
        log.warning("No prices — skipping labels/panel (watchlist still works).")
    return scored


def step_watchlist(settings: Settings | None = None, date=None) -> pd.DataFrame:
    settings = settings or load_settings()
    scored = read_parquet(settings.path("features"))
    mentions = read_parquet_or_empty(settings.path("mentions"))
    return rp.generate_watchlist(scored, mentions, settings, date=date)


def _is_synthetic(panel: pd.DataFrame, settings: Settings) -> bool:
    # The data flag is definitive: real (keyless) data is NOT synthetic even when
    # no API keys are configured. Only fall back to mode when the flag is absent.
    if "any_synthetic" in panel.columns:
        return bool(panel["any_synthetic"].max())
    return settings.reddit_mode == "mock" or settings.fmp_mode == "mock"


def step_event_study(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    panel = read_parquet(settings.path("panel"))
    panel = dg.add_analysis_buckets(panel, settings)
    es = dg.event_study(panel, settings)
    rp.save_table(es, "event_study", settings)
    rp.plot_event_study(es, settings)
    # hype-decay / reversal study — the influential-finding tool
    decay_attn = dg.hype_decay_study(panel, settings, rank_col="attention_zscore")
    decay_score = dg.hype_decay_study(panel, settings, rank_col="final_hype_alpha_score")
    if not decay_attn.empty:
        rp.save_table(decay_attn, "hype_decay_attention", settings)
    if not decay_score.empty:
        rp.save_table(decay_score, "hype_decay_finalscore", settings)
    # conditional hypotheses (acceleration / DD quality / spike) on liquid names
    conditional = dg.conditional_battery(panel, settings)
    if not conditional.empty:
        rp.save_table(conditional, "conditional_signals", settings)
    already = dg.already_ran_check(panel, settings)
    for by in ["mcap_bucket", "sector", "sentiment_bucket", "hype_bucket"]:
        bt = dg.bucket_performance(panel, by, settings=settings)
        if not bt.empty:
            rp.save_table(bt, f"bucket_{by}", settings)
    cap = dg.capacity_report(panel, settings)
    rp.save_table(cap, "capacity", settings)
    log.info("Event study complete. Already-ran check: %s", already)
    return {
        "event_study": es,
        "already_ran": already,
        "decay_attention": decay_attn,
        "decay_finalscore": decay_score,
        "conditional": conditional,
    }


def step_backtest(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    panel = read_parquet(settings.path("panel"))
    prices = read_parquet(settings.path("prices"))
    scoreboard, results = run_strategy_grid(panel, prices, settings)
    rp.save_scoreboard(scoreboard, settings)
    rp.plot_equity_curves(results, settings)

    decay_attn = dg.hype_decay_study(panel, settings, rank_col="attention_zscore")
    decay_score = dg.hype_decay_study(panel, settings, rank_col="final_hype_alpha_score")
    diagnostics = {
        "synthetic_data": _is_synthetic(panel, settings),
        "attention_decay_verdict": decay_attn.attrs.get("verdict") if not decay_attn.empty else "n/a",
        "final_score_decay_verdict": decay_score.attrs.get("verdict") if not decay_score.empty else "n/a",
        **dg.already_ran_check(panel, settings),
    }
    rp.write_scorecard_md(scoreboard, diagnostics, settings)
    return scoreboard


def step_most_talked_about(settings: Settings | None = None, top_n: int = 100) -> pd.DataFrame:
    settings = settings or load_settings()
    mentions = read_parquet(settings.path("mentions"))
    prices = read_parquet_or_empty(settings.path("prices"))
    universe = read_parquet_or_empty(settings.path("ticker_universe"))
    return rp.most_talked_about(mentions, prices, universe, settings, top_n=top_n)


def step_mine(settings: Settings | None = None) -> dict:
    from .mining import mine_mention_strategies

    settings = settings or load_settings()
    panel = read_parquet(settings.path("panel"))
    prices = read_parquet(settings.path("prices"))
    out = mine_mention_strategies(panel, prices, settings)
    rp.save_table(out["in_sample_board"], "mining_in_sample", settings)
    rp.save_table(out["oos_check"], "mining_oos_check", settings)
    return out


def step_research(settings: Settings | None = None) -> dict:
    settings = settings or load_settings()
    panel = read_parquet(settings.path("panel"))
    prices = read_parquet(settings.path("prices"))
    out = run_research_loop(panel, prices, settings)
    if not out["iterations"].empty:
        rp.save_table(out["iterations"], "research_iterations", settings)
    if not out["hypotheses"].empty:
        rp.save_table(out["hypotheses"], "research_hypotheses", settings)
    if "ic_table" in out and not out["ic_table"].empty:
        rp.save_table(out["ic_table"], "component_ic", settings)
    log.info("Research loop done. Final OOS Sharpe=%.3f | weights=%s",
             out.get("final_oos_sharpe", float("nan")), out.get("final_weights"))
    return out


def doctor(settings: Settings | None = None) -> bool:
    """Pre-flight check: report credentials and actually ping each live API with a
    single minimal request, so a bad key is caught in seconds rather than deep in
    a long pipeline run. Returns True if no live check failed."""
    settings = settings or load_settings()
    print(credential_report())
    print("-" * 60)
    ok = True

    if settings.reddit_mode == "live":
        try:
            from .reddit_client import PrawRedditClient

            client = PrawRedditClient(settings)
            post = next(iter(client.reddit.subreddit("stocks").new(limit=1)))
            print(f"  Reddit  LIVE: OK - sample r/stocks post: {post.title[:70]!r}")
        except Exception as exc:
            ok = False
            print(f"  Reddit  LIVE: FAILED - {exc}\n"
                  "    Check REDDIT_CLIENT_ID/SECRET/USER_AGENT and that the app type is 'script'.")
    else:
        print("  Reddit  : MOCK mode (no keys) - skipping live check.")

    if settings.fmp_mode == "live":
        try:
            from .fmp_client import LiveFmpClient

            prof = LiveFmpClient(settings).profiles(["AAPL"])
            mc = prof.iloc[0].get("mktCap") if not prof.empty else None
            print(f"  FMP     LIVE: OK - AAPL market cap = {mc}")
        except Exception as exc:
            ok = False
            print(f"  FMP     LIVE: FAILED - {exc}\n"
                  "    Check FMP_API_KEY and that your plan permits the /profile and "
                  "/historical-price-full endpoints.")
    else:
        print("  FMP     : MOCK mode (no keys) - skipping live check.")

    print("-" * 60)
    print("Doctor:", "ALL CHECKS PASSED" if ok else "ONE OR MORE LIVE CHECKS FAILED")
    return ok


def backfill_keyless(
    settings: Settings | None = None,
    since: str = "2023-01-01",
    until: str | None = None,
    subreddits=None,
    max_records_per_kind: int = 20000,
    include_comments: bool = False,
    fresh: bool = True,
    min_ticker_mentions: int = 5,
    chunk_days: int | None = None,
) -> dict:
    """Fully KEYLESS real-data backfill: arctic_shift (Reddit history) + SEC
    universe + Yahoo prices -> mentions -> features -> ready for event-study.
    No API keys, no manual downloads."""
    from . import freedata
    from .reddit_history import fetch_arctic_shift
    from .ticker_universe import save_universe

    settings = settings or load_settings()
    if fresh:
        # start clean so leftover MOCK data can't contaminate the real study
        for key in ("reddit_items", "mentions", "prices", "features", "panel", "labels"):
            try:
                settings.path(key).unlink(missing_ok=True)
            except Exception:
                pass
        log.info("Cleared prior data files for a fresh real backfill.")

    kinds = ("posts", "comments") if include_comments else ("posts",)
    log.info("Fetching Reddit history via arctic_shift (keyless) | subs=%s window=%s..%s kinds=%s",
             subreddits or settings.subreddit_names(), since, until, kinds)
    items = fetch_arctic_shift(settings, subreddits=subreddits, since=since, until=until,
                               kinds=kinds, max_records_per_kind=max_records_per_kind,
                               chunk_days=chunk_days)
    if items.empty:
        return {"items": 0, "mentions": 0, "tickers": 0}

    save_universe(freedata.sec_company_universe(settings), settings)
    mentions = step_extract_mentions(settings)
    if mentions.empty:
        return {"items": len(items), "mentions": 0, "tickers": 0}

    benchmark = settings.strat("labels", "market_benchmark", default="SPY")
    # only price tickers mentioned enough to matter — keeps Yahoo fast and the
    # study focused (rarely-mentioned tickers are noise and would be label-NaN anyway)
    counts = mentions["ticker"].value_counts()
    frequent = counts[counts >= min_ticker_mentions].index.tolist()
    tickers = sorted(set(frequent) | {benchmark})
    log.info("Pricing %d tickers (>=%d mentions) of %d mentioned.",
             len(tickers), min_ticker_mentions, mentions["ticker"].nunique())
    end = pd.Timestamp.utcnow().date().isoformat()  # always pull through today for forward returns
    log.info("Fetching Yahoo prices (keyless) for %d tickers...", len(tickers))
    prices = freedata.yahoo_prices(tickers, since, end, settings=settings)
    write_parquet(prices, settings.path("prices"))

    scored = step_build_features(settings)
    return {
        "items": len(items),
        "mentions": len(mentions),
        "tickers": int(mentions["ticker"].nunique()),
        "priced_tickers": int(prices["ticker"].nunique()) if not prices.empty else 0,
        "panel_rows": len(scored),
    }


def full_pipeline(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    print(credential_report())
    step_build_universe(settings)
    step_fetch_reddit(settings)
    step_extract_mentions(settings)
    step_fetch_prices(settings)
    step_build_features(settings)
    step_watchlist(settings)
    log.info("Full pipeline complete.")
