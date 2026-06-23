"""Bot / spam / low-quality detection for Reddit items.

These signals (a) down-weight obvious astroturf so it doesn't inflate attention,
and (b) feed the pump-risk score. Everything is heuristic and transparent — the
goal is to *discount* suspicious activity, not to make hard moderation calls.
"""
from __future__ import annotations

import pandas as pd

from . import text_cleaning as tc
from .utils import get_logger

log = get_logger(__name__)

# Phrases strongly associated with low-effort pump / spam.
SPAM_PHRASES = [
    "join our discord", "join my discord", "free signals", "telegram", "dm me",
    "100% guaranteed", "guaranteed profit", "not financial advice but", "pump",
    "to the moon trust me", "buy now before", "last chance", "easy 10x",
]


def item_flags(text: str, score: int | float | None = None) -> dict:
    """Per-item spam/quality flags from text + engagement."""
    stats = tc.analyze(text)
    lower = stats.lower_text
    n_spam_phrases = sum(lower.count(p) for p in SPAM_PHRASES)

    low_effort = stats.word_len < 6 and stats.n_financial_terms == 0
    pumpy = (stats.n_pump_terms + n_spam_phrases) >= 2
    all_caps = bool(stats.clean_text) and stats.clean_text.isupper() and stats.word_len >= 3

    # A crude per-item bot/spam likelihood in [0, 1].
    score_val = 0.0
    score_val += 0.30 * min(1.0, n_spam_phrases)
    score_val += 0.25 * min(1.0, stats.n_pump_terms / 2)
    score_val += 0.20 * float(all_caps)
    score_val += 0.15 * float(low_effort)
    score_val += 0.10 * float((score or 0) <= 0)  # ignored/zero-score chatter
    bot_spam_likelihood = min(1.0, score_val)

    return {
        "n_spam_phrases": int(n_spam_phrases),
        "low_effort": bool(low_effort),
        "pumpy": bool(pumpy),
        "all_caps": bool(all_caps),
        "bot_spam_likelihood": float(bot_spam_likelihood),
    }


def flag_copypaste(df: pd.DataFrame, text_col: str = "body", by: str = "author_hash") -> pd.Series:
    """Flag near-duplicate posting (same author repeating near-identical text).

    Returns a boolean Series aligned to ``df.index``.
    """
    flags = pd.Series(False, index=df.index)
    if text_col not in df.columns or by not in df.columns:
        return flags
    for _, grp in df.groupby(by):
        texts = grp[text_col].fillna("").tolist()
        idx = grp.index.tolist()
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if tc.looks_copy_paste(texts[i], texts[j]):
                    flags.loc[idx[i]] = True
                    flags.loc[idx[j]] = True
    return flags
