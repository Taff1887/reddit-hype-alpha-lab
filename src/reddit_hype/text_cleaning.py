"""Text normalisation and lightweight linguistic feature extraction for Reddit
posts/comments. No heavy NLP deps — regex + curated lexicons keep this fast and
deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
WHITESPACE_RE = re.compile(r"\s+")
TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b")     # NVDA, PDN.AX
CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,3})?)\b")  # $TSLA, $PDN.AX

# Financial vocabulary used for the "substance" / DD-quality signal.
FINANCIAL_TERMS = {
    "revenue", "earnings", "eps", "guidance", "margin", "margins", "ebitda",
    "valuation", "p/e", "pe ratio", "free cash flow", "fcf", "balance sheet",
    "debt", "dilution", "catalyst", "moat", "tam", "addressable market",
    "backlog", "order book", "contract", "fda", "approval", "drilling",
    "resource", "grade", "feasibility", "offtake", "production", "capex",
    "dividend", "buyback", "short interest", "float", "insider", "10-k",
    "10-q", "8-k", "annual report", "analyst", "price target", "downgrade",
    "upgrade", "fundamentals", "cash flow", "book value", "yield",
}

# Phrases that signal real conviction / a stated position (not just chatter).
COMMITMENT_PATTERNS = [
    r"\bi (?:just )?(?:bought|bot|grabbed|picked up|added)\b",
    r"\bi'?m (?:in|long|holding|all in|loading)\b",
    r"\b(?:holding|hodl|diamond hands)\b",
    r"\b(?:calls?|leaps?)\b",
    r"\b\d+\s*shares?\b",
    r"\bmy position\b",
    r"\badding (?:more|to)\b",
    r"\bloading up\b",
    r"\ball in\b",
    r"\byolo(?:ing|ed)?\b",
    r"\baveraged? (?:down|up)\b",
    r"\bcost basis\b",
]

# Bearish / exit commitment.
BEARISH_COMMITMENT_PATTERNS = [
    r"\b(?:i (?:just )?(?:sold|dumped|exited))\b",
    r"\bputs?\b",
    r"\b(?:i'?m )?short(?:ing|ed)?\b",
    r"\bbag ?holder\b",
    r"\btaking (?:the )?loss\b",
    r"\bcutting (?:my )?losses\b",
]

# Hype / pump / squeeze register.
HYPE_TERMS = {
    "moon", "mooning", "rocket", "🚀", "to the moon", "tendies", "squeeze",
    "short squeeze", "gamma squeeze", "diamond hands", "ape", "apes",
    "free money", "easy money", "cant go tits up", "can't go tits up",
    "10x", "100x", "lambo", "printing", "money printer", "fomo",
}
PUMP_TERMS = {
    "pump", "pumping", "next gme", "next amc", "next nvda", "guaranteed",
    "easy 10x", "buy now", "dont miss", "don't miss", "last chance",
    "load up", "back up the truck", "all in", "yolo", "sure thing",
}
PANIC_TERMS = {
    "crash", "crashing", "dump", "dumping", "rugpull", "rug pull", "bagholder",
    "dead", "bankrupt", "bankruptcy", "halt", "halted", "delisted", "fraud",
}


@dataclass
class TextStats:
    clean_text: str
    lower_text: str
    char_len: int
    word_len: int
    n_links: int
    n_financial_terms: int
    n_commitment: int
    n_bearish_commitment: int
    n_hype_terms: int
    n_pump_terms: int
    n_panic_terms: int


def strip_urls(text: str) -> tuple[str, int]:
    """Remove URLs and markdown links; return (clean_text, link_count)."""
    if not text:
        return "", 0
    md_links = MD_LINK_RE.findall(text)
    text2 = MD_LINK_RE.sub(r"\1", text)        # keep anchor text, drop the URL
    bare = URL_RE.findall(text2)
    text3 = URL_RE.sub(" ", text2)
    return text3, len(md_links) + len(bare)


def normalize(text: str) -> str:
    text, _ = strip_urls(text or "")
    text = text.replace("’", "'")  # curly apostrophe
    return WHITESPACE_RE.sub(" ", text).strip()


def _count_terms(lower: str, terms: set[str]) -> int:
    return sum(lower.count(t) for t in terms)


def _count_patterns(lower: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, lower)) for p in patterns)


def analyze(text: str) -> TextStats:
    """Compute the full bundle of lightweight text features for one item."""
    clean, n_links = strip_urls(text or "")
    clean = WHITESPACE_RE.sub(" ", clean).strip()
    lower = clean.lower()
    words = lower.split()
    return TextStats(
        clean_text=clean,
        lower_text=lower,
        char_len=len(clean),
        word_len=len(words),
        n_links=n_links,
        n_financial_terms=_count_terms(lower, FINANCIAL_TERMS),
        n_commitment=_count_patterns(lower, COMMITMENT_PATTERNS),
        n_bearish_commitment=_count_patterns(lower, BEARISH_COMMITMENT_PATTERNS),
        n_hype_terms=_count_terms(lower, HYPE_TERMS),
        n_pump_terms=_count_terms(lower, PUMP_TERMS),
        n_panic_terms=_count_terms(lower, PANIC_TERMS),
    )


def looks_copy_paste(text_a: str, text_b: str, threshold: float = 0.92) -> bool:
    """Cheap near-duplicate check via Jaccard over word shingles (bot/spam cue)."""
    a = set(normalize(text_a).lower().split())
    b = set(normalize(text_b).lower().split())
    if not a or not b:
        return False
    inter = len(a & b)
    union = len(a | b)
    return union > 0 and inter / union >= threshold
