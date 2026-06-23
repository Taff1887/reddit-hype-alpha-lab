"""Build and load the tradable ticker universe used to validate ticker mentions
and to attach sector / market-cap / liquidity metadata.

Live mode pulls the FMP listing and enriches with company profiles; mock mode
returns the curated synthetic universe. Either way the output schema is the
``UNIVERSE_COLUMNS`` contract from :mod:`fmp_client`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .fmp_client import UNIVERSE_COLUMNS, FmpClient, LiveFmpClient, get_fmp_client
from .utils import get_logger, read_parquet, write_parquet

log = get_logger(__name__)

_EXCHANGE_REGION = {
    "NASDAQ": "US", "NYSE": "US", "AMEX": "US", "NYSEARCA": "US", "BATS": "US",
    "ASX": "AU",
}
_OTC_EXCHANGES = {"OTC", "PNK", "OTCMKTS", "PINK", "GREY"}


def _region_for(exchange: str | None) -> str:
    return _EXCHANGE_REGION.get(str(exchange).upper(), "GLOBAL")


def _enrich_live(client: LiveFmpClient, settings: Settings) -> pd.DataFrame:
    listing = client.universe()  # ticker, company_name, exchange
    symbols = listing["ticker"].dropna().unique().tolist()
    log.info("Enriching %d symbols with FMP profiles (cached, may take a while)...", len(symbols))
    profiles = client.profiles(symbols)
    if profiles.empty:
        df = listing.copy()
        for col in UNIVERSE_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df["synthetic"] = False
        return df[UNIVERSE_COLUMNS]

    p = profiles.rename(
        columns={
            "symbol": "ticker",
            "companyName": "company_name",
            "exchangeShortName": "exchange",
            "mktCap": "market_cap",
            "volAvg": "avg_volume",
        }
    )
    keep = ["ticker", "company_name", "exchange", "sector", "industry",
            "market_cap", "price", "avg_volume", "beta", "isEtf", "isFund"]
    for c in keep:
        if c not in p.columns:
            p[c] = np.nan
    p = p[keep].copy()
    p["region"] = p["exchange"].map(_region_for)
    p["is_otc"] = p["exchange"].str.upper().isin(_OTC_EXCHANGES)
    p["synthetic"] = False
    # Short interest / float is NOT on FMP's standard tier. Leave NaN (the
    # squeeze features degrade gracefully) rather than fabricate it. Plug in a
    # FINRA / commercial short-interest source here to enable squeeze signals.
    for col in ("short_pct_float", "days_to_cover", "shares_float"):
        if col not in p.columns:
            p[col] = np.nan
    if p["short_pct_float"].isna().all():
        log.warning(
            "No short-interest data in LIVE mode — squeeze features will be inactive. "
            "Wire a FINRA/commercial short-interest source into ticker_universe to enable them."
        )
    return p[UNIVERSE_COLUMNS]


def build_universe(
    settings: Settings | None = None, client: FmpClient | None = None
) -> pd.DataFrame:
    settings = settings or load_settings()
    client = client or get_fmp_client(settings)

    if isinstance(client, LiveFmpClient):
        df = _enrich_live(client, settings)
    else:
        df = client.universe()

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    # Liquidity proxy + bucket
    df["avg_dollar_volume"] = df["avg_volume"].fillna(0) * df["price"].fillna(0)
    edges = settings.strat("tradability", "liquidity_buckets", default=[1e6, 1e7, 1e8])
    df["liquidity_bucket"] = pd.cut(
        df["avg_dollar_volume"],
        bins=[-np.inf, *edges, np.inf],
        labels=[f"<= {e:.0e}" for e in edges] + [f"> {edges[-1]:.0e}"],
    ).astype(str)

    # Apply a soft market-cap floor but always keep the benchmark.
    min_cap = settings.get("fmp", "min_market_cap", default=0) or 0
    benchmark = settings.strat("labels", "market_benchmark", default="SPY")
    before = len(df)
    df = df[(df["market_cap"].fillna(0) >= min_cap) | (df["ticker"] == benchmark)]
    log.info("Universe: %d tickers (filtered from %d on market-cap floor %s)",
             len(df), before, f"{min_cap:,.0f}")
    return df.reset_index(drop=True)


def save_universe(df: pd.DataFrame, settings: Settings | None = None) -> str:
    settings = settings or load_settings()
    path = settings.path("ticker_universe")
    write_parquet(df, path)
    log.info("Saved universe -> %s", path)
    return str(path)


def load_universe(settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    return read_parquet(settings.path("ticker_universe"))
