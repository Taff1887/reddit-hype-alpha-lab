"""Keyless real-data sources, so the whole pipeline can run on REAL data with no
API keys at all:

* **SEC company tickers** (``company_tickers.json``) -> a broad ticker+name
  universe for extraction validation (keyless).
* **Yahoo Finance chart API** -> real daily OHLCV/adj-close (keyless).

These are the "I cbf to get keys" fallbacks. FMP remains the higher-quality
source (market cap, sector, validated universe) when a key is present.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .fmp_client import PRICE_COLUMNS, UNIVERSE_COLUMNS
from .utils import get_logger

log = get_logger(__name__)

_UA = "reddit-hype-alpha-lab/0.1 (quant research; contact: research@example.com)"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------- SEC universe
def sec_company_universe(settings: Settings | None = None) -> pd.DataFrame:
    """Broad keyless ticker->name universe from the SEC. Numeric fields (market
    cap, sector, short interest) are unknown here and left NaN — extraction and
    the event/decay study work fine; only liquidity-gated portfolio filters are
    weaker than with FMP."""
    settings = settings or load_settings()
    try:
        raw = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    except Exception as exc:
        log.warning("SEC universe fetch failed (%s) — falling back to seed tickers.", exc)
        from .ticker_extractor import _SEED_TICKERS

        return pd.DataFrame(
            {"ticker": sorted(_SEED_TICKERS)}
        ).assign(company_name=None, exchange="US", region="US", sector=None, industry=None,
                 market_cap=np.nan, price=np.nan, avg_volume=np.nan, beta=np.nan,
                 short_pct_float=np.nan, days_to_cover=np.nan, shares_float=np.nan,
                 is_otc=False, synthetic=False)[UNIVERSE_COLUMNS]

    rows = []
    for v in raw.values():
        tk = str(v.get("ticker", "")).upper().strip()
        if tk:
            rows.append({"ticker": tk, "company_name": v.get("title")})
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
    for col in UNIVERSE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["exchange"] = "US"
    df["region"] = "US"
    df["is_otc"] = False
    df["synthetic"] = False
    log.info("SEC universe: %d tickers (keyless; market cap/sector unknown).", len(df))
    return df[UNIVERSE_COLUMNS]


# --------------------------------------------------------------- Yahoo prices
def _parse_yahoo_chart(payload: dict, ticker: str) -> pd.DataFrame:
    """Pure parser for one Yahoo chart response -> PRICE_COLUMNS rows."""
    try:
        res = payload["chart"]["result"][0]
    except (KeyError, IndexError, TypeError):
        return pd.DataFrame(columns=PRICE_COLUMNS)
    ts = res.get("timestamp") or []
    quote = (res.get("indicators", {}).get("quote") or [{}])[0]
    adj = (res.get("indicators", {}).get("adjclose") or [{}])
    adjclose = adj[0].get("adjclose") if adj and isinstance(adj[0], dict) else None
    o, h = quote.get("open") or [], quote.get("high") or []
    lo, c = quote.get("low") or [], quote.get("close") or []
    v = quote.get("volume") or []
    n = len(ts)
    if n == 0:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    dates = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    rows = []
    for i in range(n):
        close = c[i] if i < len(c) else None
        if close is None:
            continue
        vol = v[i] if i < len(v) and v[i] is not None else 0
        ac = adjclose[i] if adjclose and i < len(adjclose) and adjclose[i] is not None else close
        rows.append({
            "ticker": ticker, "date": dates[i],
            "open": o[i] if i < len(o) else close,
            "high": h[i] if i < len(h) else close,
            "low": lo[i] if i < len(lo) else close,
            "close": close, "adj_close": ac,
            "volume": vol, "dollar_volume": (close or 0) * (vol or 0),
        })
    return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def yahoo_prices(
    tickers, start: str, end: str, sleep: float = 0.25, settings: Settings | None = None
) -> pd.DataFrame:
    """Keyless daily prices from Yahoo for a list of tickers over [start, end]."""
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86400
    frames, failed = [], 0
    tickers = list(dict.fromkeys(tickers))
    for i, tk in enumerate(tickers):
        ytk = tk.replace(".AX", ".AX")  # Yahoo uses .AX for ASX too
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
               f"{urllib.parse.quote(ytk)}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits")
        try:
            df = _parse_yahoo_chart(json.loads(_get(url)), tk)
            if not df.empty:
                frames.append(df)
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            log.debug("yahoo price fail %s: %s", tk, exc)
        if (i + 1) % 50 == 0:
            log.info("  yahoo prices: %d/%d tickers...", i + 1, len(tickers))
        time.sleep(sleep)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
    log.info("Yahoo prices: %d rows for %d/%d tickers (%d failed/empty).",
             len(out), len(tickers) - failed, len(tickers), failed)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True) if not out.empty else out
