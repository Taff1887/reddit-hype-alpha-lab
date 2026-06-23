"""Financial Modeling Prep (FMP) access: tradable universe, company profiles,
and daily prices/volume.

* :class:`LiveFmpClient` calls the FMP REST API (requires ``FMP_API_KEY``) with a
  small on-disk cache to respect rate limits.
* :class:`MockFmpClient` produces deterministic synthetic data. **Mock prices are
  a pure random walk with NO engineered relationship to Reddit hype** — so a mock
  backtest honestly shows ~no alpha. Synthetic data is for plumbing/tests only.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Settings, load_settings
from .utils import ensure_dir, get_logger

log = get_logger(__name__)

PRICE_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close",
                 "volume", "dollar_volume"]
UNIVERSE_COLUMNS = ["ticker", "company_name", "exchange", "region", "sector",
                    "industry", "market_cap", "price", "avg_volume", "beta",
                    "short_pct_float", "days_to_cover", "shares_float",
                    "is_otc", "synthetic"]

# Thematic metadata for the synthetic universe (structure only, not returns).
_MOCK_META = {
    "NVDA": ("NVIDIA", "NASDAQ", "US", "Technology", "Semiconductors", 2.8e12, 1.6),
    "AMD": ("Advanced Micro Devices", "NASDAQ", "US", "Technology", "Semiconductors", 2.6e11, 1.7),
    "AVGO": ("Broadcom", "NASDAQ", "US", "Technology", "Semiconductors", 7.0e11, 1.1),
    "SMCI": ("Super Micro Computer", "NASDAQ", "US", "Technology", "Computer Hardware", 2.5e10, 1.9),
    "ARM": ("Arm Holdings", "NASDAQ", "US", "Technology", "Semiconductors", 1.4e11, 1.5),
    "NBIS": ("Nebius Group", "NASDAQ", "US", "Technology", "Cloud Infrastructure", 8.0e9, 2.1),
    "VRT": ("Vertiv Holdings", "NYSE", "US", "Industrials", "Electrical Equipment", 4.0e10, 1.4),
    "PLTR": ("Palantir", "NASDAQ", "US", "Technology", "Software", 3.0e11, 1.8),
    "BE": ("Bloom Energy", "NYSE", "US", "Industrials", "Electrical Equipment", 5.0e9, 2.3),
    "AI": ("C3.ai", "NYSE", "US", "Technology", "Software", 3.5e9, 1.9),
    "IREN": ("IREN", "NASDAQ", "US", "Technology", "Data Center / Mining", 3.0e9, 2.5),
    "CCJ": ("Cameco", "NYSE", "US", "Energy", "Uranium", 2.5e10, 1.0),
    "OKLO": ("Oklo", "NYSE", "US", "Utilities", "Nuclear", 6.0e9, 2.8),
    "SMR": ("NuScale Power", "NYSE", "US", "Utilities", "Nuclear", 4.0e9, 2.6),
    "LEU": ("Centrus Energy", "AMEX", "US", "Energy", "Uranium", 1.5e9, 2.2),
    "UEC": ("Uranium Energy", "AMEX", "US", "Energy", "Uranium", 3.0e9, 1.8),
    "UUUU": ("Energy Fuels", "AMEX", "US", "Energy", "Uranium", 1.2e9, 1.7),
    "DNN": ("Denison Mines", "AMEX", "US", "Energy", "Uranium", 1.5e9, 1.9),
    "ALB": ("Albemarle", "NYSE", "US", "Materials", "Lithium", 1.2e10, 1.3),
    "LAC": ("Lithium Americas", "NYSE", "US", "Materials", "Lithium", 2.0e9, 1.9),
    "MARA": ("MARA Holdings", "NASDAQ", "US", "Financials", "Bitcoin Mining", 6.0e9, 3.2),
    "RIOT": ("Riot Platforms", "NASDAQ", "US", "Financials", "Bitcoin Mining", 4.0e9, 3.1),
    "CLSK": ("CleanSpark", "NASDAQ", "US", "Financials", "Bitcoin Mining", 3.0e9, 3.3),
    "CORZ": ("Core Scientific", "NASDAQ", "US", "Financials", "Bitcoin Mining", 4.5e9, 2.9),
    "MSTR": ("MicroStrategy", "NASDAQ", "US", "Technology", "Software / BTC", 9.0e10, 3.0),
    "COIN": ("Coinbase", "NASDAQ", "US", "Financials", "Crypto Exchange", 7.0e10, 2.7),
    "GME": ("GameStop", "NYSE", "US", "Consumer Cyclical", "Specialty Retail", 1.2e10, 1.6),
    "AMC": ("AMC Entertainment", "NYSE", "US", "Communication Services", "Entertainment", 1.5e9, 1.8),
    "HOOD": ("Robinhood", "NASDAQ", "US", "Financials", "Capital Markets", 5.0e10, 2.0),
    "RKLB": ("Rocket Lab", "NASDAQ", "US", "Industrials", "Aerospace & Defense", 1.5e10, 2.2),
    "LCID": ("Lucid Group", "NASDAQ", "US", "Consumer Cyclical", "Auto Manufacturers", 6.0e9, 2.1),
    "RIVN": ("Rivian", "NASDAQ", "US", "Consumer Cyclical", "Auto Manufacturers", 1.4e10, 2.0),
    "TSLA": ("Tesla", "NASDAQ", "US", "Consumer Cyclical", "Auto Manufacturers", 8.0e11, 2.0),
    "AAPL": ("Apple", "NASDAQ", "US", "Technology", "Consumer Electronics", 3.4e12, 1.2),
    "MSFT": ("Microsoft", "NASDAQ", "US", "Technology", "Software", 3.3e12, 0.9),
    "AMZN": ("Amazon", "NASDAQ", "US", "Consumer Cyclical", "Internet Retail", 2.1e12, 1.1),
    "META": ("Meta Platforms", "NASDAQ", "US", "Communication Services", "Internet", 1.5e12, 1.2),
    "GOOGL": ("Alphabet", "NASDAQ", "US", "Communication Services", "Internet", 2.1e12, 1.0),
    "SPY": ("SPDR S&P 500 ETF", "NYSE", "US", "Index", "ETF", 5.0e11, 1.0),
    "PDN.AX": ("Paladin Energy", "ASX", "AU", "Energy", "Uranium", 3.0e9, 1.9),
    "BOE.AX": ("Boss Energy", "ASX", "AU", "Energy", "Uranium", 1.5e9, 2.0),
    "DYL.AX": ("Deep Yellow", "ASX", "AU", "Energy", "Uranium", 1.0e9, 2.1),
    "PLS.AX": ("Pilbara Minerals", "ASX", "AU", "Materials", "Lithium", 8.0e9, 1.8),
    "LTR.AX": ("Liontown Resources", "ASX", "AU", "Materials", "Lithium", 2.0e9, 2.0),
    "FMG.AX": ("Fortescue", "ASX", "AU", "Materials", "Iron Ore", 6.0e10, 1.3),
    "BHP.AX": ("BHP Group", "ASX", "AU", "Materials", "Diversified Mining", 1.5e11, 1.0),
    "RIO.AX": ("Rio Tinto", "ASX", "AU", "Materials", "Diversified Mining", 1.3e11, 0.9),
    "CBA.AX": ("Commonwealth Bank", "ASX", "AU", "Financials", "Banks", 1.7e11, 0.8),
    "WTC.AX": ("WiseTech Global", "ASX", "AU", "Technology", "Software", 3.0e10, 1.4),
    "ZIP.AX": ("Zip Co", "ASX", "AU", "Financials", "Consumer Finance", 3.0e9, 2.4),
}


class FmpClient:
    def universe(self) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- live
class LiveFmpClient(FmpClient):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.key = settings.credentials.fmp_api_key
        self.base = settings.get("fmp", "base_url", default="https://financialmodelingprep.com/api/v3")
        self.sleep = float(settings.get("fmp", "request_sleep_seconds", default=0.25))
        self.cache_days = float(settings.get("fmp", "cache_days", default=1))
        self.cache_dir = ensure_dir(settings.path("raw_fundamentals") / "_cache")

    def _get(self, endpoint: str, params: dict | None = None) -> list | dict:
        params = dict(params or {})
        params["apikey"] = self.key
        url = f"{self.base}/{endpoint}"
        cache_key = endpoint.replace("/", "_") + "_" + "_".join(
            f"{k}{v}" for k, v in sorted(params.items()) if k != "apikey"
        )
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            age_days = (time.time() - cache_file.stat().st_mtime) / 86400
            if age_days <= self.cache_days:
                return json.loads(cache_file.read_text())
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        time.sleep(self.sleep)
        return data

    def universe(self) -> pd.DataFrame:
        exchanges = set(self.settings.get("fmp", "exchanges", default=["NASDAQ", "NYSE", "AMEX"]))
        listed = self._get("stock/list")
        rows = []
        for r in listed:
            ex = r.get("exchangeShortName")
            if ex not in exchanges:
                continue
            rows.append({"ticker": r.get("symbol"), "company_name": r.get("name"), "exchange": ex})
        df = pd.DataFrame(rows).dropna(subset=["ticker"])
        log.info("FMP returned %d listed symbols on target exchanges", len(df))
        return df

    def profiles(self, tickers: list[str]) -> pd.DataFrame:
        rows = []
        for i in range(0, len(tickers), 50):
            chunk = ",".join(tickers[i : i + 50])
            for p in self._get(f"profile/{chunk}"):
                rows.append(p)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        frames = []
        for t in tickers:
            try:
                data = self._get(
                    f"historical-price-full/{t}", {"from": start, "to": end}
                )
            except Exception as exc:
                log.warning("price fetch failed for %s: %s", t, exc)
                continue
            hist = data.get("historical", []) if isinstance(data, dict) else []
            for h in hist:
                close = h.get("close")
                vol = h.get("volume") or 0
                frames.append(
                    {
                        "ticker": t,
                        "date": h.get("date"),
                        "open": h.get("open"),
                        "high": h.get("high"),
                        "low": h.get("low"),
                        "close": close,
                        "adj_close": h.get("adjClose", close),
                        "volume": vol,
                        "dollar_volume": (close or 0) * vol,
                    }
                )
        df = pd.DataFrame(frames, columns=PRICE_COLUMNS)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- mock
class MockFmpClient(FmpClient):
    """Deterministic synthetic universe + random-walk prices (no hype linkage)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        m = settings.get("mock", default={}) or {}
        self.seed = int(m.get("seed", 1887))
        self.n_days = int(m.get("n_days", 180))

    def universe(self) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 7)
        rows = []
        for ticker, meta in _MOCK_META.items():
            name, exch, region, sector, industry, mcap, beta = meta
            price = float(rng.uniform(5, 400))
            avg_volume = float(rng.uniform(5e5, 5e7))
            # Synthetic short interest: smaller-cap + higher-beta names carry more,
            # squeeze-prone names most. Structure only — never tied to returns.
            cap_boost = 18.0 if mcap < 5e9 else (7.0 if mcap < 2e10 else 1.0)
            short_pct = float(np.clip((beta - 0.8) * 7 + cap_boost + rng.normal(0, 3), 0.3, 45.0))
            shares_float = float(mcap / max(price, 1) * 0.8)
            short_shares = short_pct / 100.0 * shares_float
            days_to_cover = float(np.clip(short_shares / max(avg_volume, 1), 0.2, 15.0))
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": name,
                    "exchange": exch,
                    "region": region,
                    "sector": sector,
                    "industry": industry,
                    "market_cap": float(mcap),
                    "price": price,
                    "avg_volume": avg_volume,
                    "beta": float(beta),
                    "short_pct_float": short_pct,
                    "days_to_cover": days_to_cover,
                    "shares_float": shares_float,
                    "is_otc": False,
                    "synthetic": True,
                }
            )
        df = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
        log.warning("Generated SYNTHETIC universe (%d tickers, MOCK MODE).", len(df))
        return df

    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        rng = np.random.RandomState(self.seed + 99)
        # business-day calendar between start/end (approx, no holidays)
        end_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        dates = pd.bdate_range(end=end_dt.date(), periods=self.n_days)
        frames = []
        for t in tickers:
            meta = _MOCK_META.get(t)
            beta = meta[6] if meta else 1.5
            start_price = meta[5] if meta else 1e9
            start_price = float(rng.uniform(8, 250))
            # daily vol scaled loosely by beta. Log-drift = -0.5*var so prices are
            # a true martingale (no convexity drift) -> a clean "no alpha" null.
            daily_vol = 0.012 * beta
            rets = rng.normal(-0.5 * daily_vol**2, daily_vol, size=len(dates))
            close = start_price * np.exp(np.cumsum(rets))
            open_ = close * (1 + rng.normal(0, daily_vol / 2, size=len(dates)))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, daily_vol / 2, len(dates))))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, daily_vol / 2, len(dates))))
            base_vol = (meta[5] / start_price / 50) if meta else 1e6
            volume = np.abs(rng.normal(base_vol, base_vol * 0.4, size=len(dates))).astype(float)
            frames.append(
                pd.DataFrame(
                    {
                        "ticker": t,
                        "date": dates.normalize(),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "adj_close": close,
                        "volume": volume,
                        "dollar_volume": close * volume,
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLUMNS)
        log.warning("Generated SYNTHETIC random-walk prices (MOCK MODE, no hype linkage).")
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_fmp_client(settings: Settings | None = None) -> FmpClient:
    settings = settings or load_settings()
    if settings.fmp_mode == "live":
        log.info("Using LIVE FMP client.")
        return LiveFmpClient(settings)
    log.warning("Using MOCK FMP client — synthetic development data only.")
    return MockFmpClient(settings)
