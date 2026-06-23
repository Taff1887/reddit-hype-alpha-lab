"""Shared utilities: logging, hashing, IO helpers, and trading-calendar math.

Deliberately dependency-light so every other module can import this without
creating cycles.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger configured once with a sane default handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("RHAL_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed, returning it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_username(username: str | None, salt: str = "rhal") -> str | None:
    """Hash a Reddit username so we never persist personal identifiers.

    Returns a short stable token. ``None``/deleted authors pass through as None.
    """
    if username is None:
        return None
    uname = str(username).strip()
    if uname == "" or uname.lower() in {"[deleted]", "none", "automoderator"}:
        return None
    digest = hashlib.sha256((salt + ":" + uname.lower()).encode("utf-8")).hexdigest()
    return digest[:16]


# --------------------------------------------------------------------------- IO
def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to parquet, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected parquet at {p}. Run the upstream step first.")
    return pd.read_parquet(p)


def read_parquet_or_empty(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read parquet if present, else an empty frame (optionally with columns)."""
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame(columns=list(columns) if columns else None)


# ------------------------------------------------------------------ numeric math
def safe_div(numerator, denominator, default: float = 0.0):
    """Elementwise/scalar division that returns ``default`` where denom == 0."""
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    out = np.full(np.broadcast(num, den).shape, float(default), dtype="float64")
    mask = den != 0
    np.divide(num, den, out=out, where=mask)
    if out.ndim == 0:
        return float(out)
    return out


def zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing z-score using only past observations (shifted to avoid lookahead)."""
    min_periods = min_periods or max(2, window // 3)
    past = series.shift(1)
    mean = past.rolling(window, min_periods=min_periods).mean()
    std = past.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def minmax_clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip then rescale a series into [0, 1] given hard bounds."""
    clipped = series.clip(lo, hi)
    span = hi - lo
    if span == 0:
        return pd.Series(0.0, index=series.index)
    return (clipped - lo) / span


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1]; NaNs stay NaN."""
    return series.rank(pct=True)


# ----------------------------------------------------------- trading calendar
def to_trading_days(dates: pd.Series) -> pd.Series:
    """Coerce to tz-naive normalised dates (UTC date floor)."""
    s = pd.to_datetime(dates, utc=True).dt.tz_convert(None).dt.normalize()
    return s


def business_day_offset(dates: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Shift a DatetimeIndex by ``n`` US business days (approx. trading calendar).

    A genuine exchange-holiday calendar would use pandas_market_calendars; we
    use business days as a dependency-free approximation and document it.
    """
    return dates + pd.offsets.BDay(n)


def annualization_factor(rebalance: str) -> float:
    """Periods per year for Sharpe annualisation."""
    return {"daily": 252.0, "weekly": 52.0, "monthly": 12.0}.get(rebalance, 252.0)
