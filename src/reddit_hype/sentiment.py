"""Lexicon-based finance sentiment + conviction / hype / panic / squeeze scoring.

Deliberately rule-based and transparent: a finance-tuned lexicon plus the
commitment/hype/panic pattern counts from :mod:`text_cleaning`. If
``vaderSentiment`` is installed we blend in its general-purpose compound score,
but the lab never *requires* it.

Per-item scores are designed to *aggregate* cleanly (counts sum; net_sentiment
is averaged weighted by engagement in :mod:`hype_features`).
"""
from __future__ import annotations

from functools import lru_cache

from . import text_cleaning as tc
from .utils import safe_div

# Finance-tuned polarity lexicons (kept compact and auditable).
POSITIVE_TERMS = {
    "bullish", "buy", "buying", "long", "calls", "undervalued", "cheap",
    "breakout", "rally", "rallying", "strong", "beat", "beats", "upside",
    "growth", "accumulate", "oversold", "bottom", "rebound", "moon", "rocket",
    "green", "winner", "outperform", "upgrade", "raised", "record", "surge",
    "soaring", "gains", "printing", "tendies", "squeeze",
}
NEGATIVE_TERMS = {
    "bearish", "sell", "selling", "short", "puts", "overvalued", "expensive",
    "breakdown", "dump", "dumping", "weak", "miss", "missed", "downside",
    "decline", "crash", "crashing", "red", "loser", "underperform", "downgrade",
    "cut", "guidance cut", "bankrupt", "bankruptcy", "fraud", "dilution",
    "bagholder", "rugpull", "rug pull", "halt", "halted", "delisted", "scam",
}
SQUEEZE_TERMS = {
    "short squeeze", "gamma squeeze", "squeeze", "short interest", "ftd",
    "failure to deliver", "shorts trapped", "short ladder", "borrow rate",
    "days to cover",
}


def score_text(text: str) -> dict:
    """Return per-item sentiment / conviction / hype features for one post/comment."""
    stats = tc.analyze(text)
    lower = stats.lower_text

    pos = sum(lower.count(t) for t in POSITIVE_TERMS)
    neg = sum(lower.count(t) for t in NEGATIVE_TERMS)
    squeeze = sum(lower.count(t) for t in SQUEEZE_TERMS)

    # Bounded item-level net sentiment in [-1, 1].
    net = safe_div(pos - neg, pos + neg, default=0.0)
    vader = _vader_compound(stats.clean_text)
    if vader is not None:
        net = 0.6 * net + 0.4 * vader  # blend lexicon with VADER if available

    return {
        "n_pos_terms": int(pos),
        "n_neg_terms": int(neg),
        "n_squeeze_terms": int(squeeze),
        "n_commitment": int(stats.n_commitment),
        "n_bearish_commitment": int(stats.n_bearish_commitment),
        "n_hype_terms": int(stats.n_hype_terms),
        "n_pump_terms": int(stats.n_pump_terms),
        "n_panic_terms": int(stats.n_panic_terms),
        "n_financial_terms": int(stats.n_financial_terms),
        "n_links": int(stats.n_links),
        "char_len": int(stats.char_len),
        "word_len": int(stats.word_len),
        "net_sentiment": float(net),
        "vader_compound": float(vader) if vader is not None else float("nan"),
    }


@lru_cache(maxsize=1)
def _vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _vader_compound(text: str) -> float | None:
    analyzer = _vader()
    if analyzer is None or not text:
        return None
    try:
        return float(analyzer.polarity_scores(text)["compound"])
    except Exception:  # pragma: no cover
        return None
