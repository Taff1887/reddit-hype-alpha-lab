"""Ticker extraction — the single most correctness-critical component.

Bad extraction silently poisons every downstream feature, score and backtest, so
this module is conservative by construction:

* Cashtags (``$NVDA``) are strong evidence.
* Bare uppercase tokens (``NVDA``) need to match the *valid listed universe* and
  carry finance context to clear the confidence threshold.
* English-word / acronym tickers (``BE``, ``AI``, ``ON``, ``IT``, ``DD`` ...) are
  blocklisted or require a cashtag.
* Company names / curated aliases (``Bloom Energy`` -> ``BE``,
  ``Nebius`` -> ``NBIS``) are matched explicitly.

Every mention carries a transparent confidence breakdown so the extractor's
decisions are auditable rather than a black box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .config import Settings, load_settings
from .text_cleaning import (
    CASHTAG_RE,
    TICKER_TOKEN_RE,
    normalize,
)
from .utils import get_logger

log = get_logger(__name__)

# A small fallback set of well-known tickers so the extractor is useful even
# before a universe file is built (dev/mock). Real runs use the FMP universe.
_SEED_TICKERS = {
    "NVDA", "AMD", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "PLTR",
    "SMCI", "AVGO", "ARM", "NBIS", "BE", "AI", "VRT", "IREN", "MARA", "RIOT",
    "CLSK", "CORZ", "MSTR", "COIN", "CCJ", "OKLO", "LEU", "SMR", "UEC", "UUUU",
    "DNN", "ALB", "LAC", "GME", "AMC", "HOOD", "RKLB", "LCID", "RIVN", "SPY",
    "PDN.AX", "BOE.AX", "DYL.AX", "PLS.AX", "LTR.AX", "FMG.AX", "BHP.AX",
    "RIO.AX", "CBA.AX", "WTC.AX", "ZIP.AX",
}


@dataclass
class TickerMention:
    ticker: str
    method: str                 # cashtag | bare | company_name | alias
    confidence: float
    n_occurrences: int
    has_cashtag: bool
    company_name_nearby: bool
    finance_context: bool
    valid_exchange: bool
    is_common_word: bool
    is_ambiguous: bool
    components: dict[str, float] = field(default_factory=dict)


class TickerExtractor:
    def __init__(
        self,
        valid_tickers: Iterable[str],
        company_names: dict[str, str] | None = None,
        aliases: dict[str, str] | None = None,
        filters: dict | None = None,
    ):
        self.valid = {t.upper() for t in valid_tickers}
        # ticker -> lowercased company name (for the "name nearby" bonus)
        self.company_names = {k.upper(): v.lower() for k, v in (company_names or {}).items()}
        # lowercased alias phrase -> ticker
        self.aliases = {k.lower(): v.upper() for k, v in (aliases or {}).items()}
        f = filters or {}
        # str() coercion: YAML 1.1 parses bare ON/NO/YES/OFF as booleans, so any
        # such ticker token arrives as a Python bool — normalise back to text.
        self.blocklist = {str(t).upper() for t in f.get("blocklist", [])}
        self.require_cashtag = {str(t).upper() for t in f.get("require_cashtag", [])}
        self.common_words = {str(t).upper() for t in f.get("common_english_words", [])}
        self.finance_terms = [str(t).lower() for t in f.get("finance_context_terms", [])]
        self.weights = f.get("confidence_weights", {})
        self.min_confidence = float(f.get("min_confidence", 0.55))
        self.asx_suffix = f.get("asx_suffix", ".AX")
        # Pre-sort alias phrases longest-first so multi-word names win.
        self._alias_phrases = sorted(self.aliases.keys(), key=len, reverse=True)

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, universe: pd.DataFrame | None = None
    ) -> "TickerExtractor":
        settings = settings or load_settings()
        valid: set[str] = set(_SEED_TICKERS)
        names: dict[str, str] = {}

        if universe is None:
            upath = settings.path("ticker_universe")
            if upath.exists():
                universe = pd.read_parquet(upath)
        if universe is not None and not universe.empty:
            valid = set(universe["ticker"].str.upper())
            if "company_name" in universe.columns:
                names = dict(
                    zip(universe["ticker"].str.upper(), universe["company_name"].fillna(""))
                )

        aliases = _load_aliases(settings.root)
        # Alias targets are also valid tickers.
        valid |= set(aliases.values())
        return cls(valid, names, aliases, settings.ticker_filters)

    # --------------------------------------------------------------- extraction
    def _finance_context(self, lower_text: str) -> bool:
        return any(term in lower_text for term in self.finance_terms)

    def _normalize_ticker(self, token: str) -> str:
        t = token.upper()
        if t in self.valid:
            return t
        # ASX bare ticker fallback (BHP -> BHP.AX) if the suffixed form is listed.
        if (t + self.asx_suffix) in self.valid:
            return t + self.asx_suffix
        return t

    def _score(
        self,
        ticker: str,
        *,
        has_cashtag: bool,
        method: str,
        company_name_nearby: bool,
        finance_context: bool,
    ) -> tuple[float, dict[str, float], dict[str, bool]]:
        w = self.weights
        valid = ticker in self.valid
        bare_token = ticker.split(".")[0]
        is_common = bare_token in self.common_words
        is_ambiguous = bare_token in self.require_cashtag

        comp: dict[str, float] = {}
        score = 0.0
        if has_cashtag:
            comp["cashtag_bonus"] = w.get("cashtag_bonus", 0.5)
        if company_name_nearby or method in {"company_name", "alias"}:
            comp["company_name_nearby_bonus"] = w.get("company_name_nearby_bonus", 0.35)
        if valid:
            comp["valid_exchange_bonus"] = w.get("valid_exchange_bonus", 0.40)
        if finance_context:
            comp["finance_context_bonus"] = w.get("finance_context_bonus", 0.20)

        # Penalties only apply to bare-token evidence; a cashtag or an explicit
        # company-name match disambiguates and disarms them.
        token_evidence = method == "bare" and not has_cashtag
        if token_evidence:
            if is_common:
                comp["common_word_penalty"] = -w.get("common_word_penalty", 0.50)
            if not finance_context:
                comp["low_context_penalty"] = -w.get("low_context_penalty", 0.20)
            if is_ambiguous:
                comp["ambiguous_no_cashtag_penalty"] = -w.get(
                    "ambiguous_no_cashtag_penalty", 0.60
                )

        score = sum(comp.values())
        score = max(0.0, min(1.0, score))
        flags = {
            "valid_exchange": valid,
            "is_common_word": is_common,
            "is_ambiguous": is_ambiguous,
        }
        return score, comp, flags

    def extract(self, text: str, region: str | None = None) -> list[TickerMention]:
        """Return one :class:`TickerMention` per distinct ticker found in ``text``."""
        clean = normalize(text or "")
        if not clean:
            return []
        lower = clean.lower()
        finance_ctx = self._finance_context(lower)

        # candidate -> dict of evidence accumulators
        cand: dict[str, dict] = {}

        def _bump(ticker: str, *, cashtag: bool, method: str, name_near: bool):
            d = cand.setdefault(
                ticker,
                {"n": 0, "cashtag": False, "method": method, "name_near": name_near},
            )
            d["n"] += 1
            d["cashtag"] = d["cashtag"] or cashtag
            d["name_near"] = d["name_near"] or name_near
            # cashtag/company evidence upgrades the recorded method
            if cashtag:
                d["method"] = "cashtag"
            elif method in {"company_name", "alias"} and d["method"] == "bare":
                d["method"] = method

        # 1) Cashtags ($NVDA, $PDN.AX) -------------------------------------
        for m in CASHTAG_RE.finditer(clean):
            ticker = self._normalize_ticker(m.group(1))
            if ticker.split(".")[0] in self.blocklist:
                continue
            _bump(ticker, cashtag=True, method="cashtag", name_near=False)

        # 2) Curated company-name aliases (longest phrase first) -----------
        consumed = lower
        for phrase in self._alias_phrases:
            if phrase in consumed:
                ticker = self.aliases[phrase]
                if ticker.split(".")[0] in self.blocklist:
                    continue
                _bump(ticker, cashtag=False, method="alias", name_near=True)

        # 3) Bare uppercase tokens (NVDA) ----------------------------------
        for m in TICKER_TOKEN_RE.finditer(clean):
            token = m.group(0)
            if token != token.upper():  # require all-caps
                continue
            base = token.split(".")[0]
            if base in self.blocklist:
                continue
            ticker = self._normalize_ticker(token)
            # company name present anywhere -> nearby bonus
            name = self.company_names.get(ticker, "")
            name_near = bool(name) and len(name) >= 3 and name in lower
            _bump(ticker, cashtag=False, method="bare", name_near=name_near)

        # Score + threshold ------------------------------------------------
        out: list[TickerMention] = []
        for ticker, d in cand.items():
            score, comp, flags = self._score(
                ticker,
                has_cashtag=d["cashtag"],
                method=d["method"],
                company_name_nearby=d["name_near"],
                finance_context=finance_ctx,
            )
            if score < self.min_confidence:
                continue
            out.append(
                TickerMention(
                    ticker=ticker,
                    method=d["method"],
                    confidence=round(score, 4),
                    n_occurrences=d["n"],
                    has_cashtag=d["cashtag"],
                    company_name_nearby=d["name_near"],
                    finance_context=finance_ctx,
                    valid_exchange=flags["valid_exchange"],
                    is_common_word=flags["is_common_word"],
                    is_ambiguous=flags["is_ambiguous"],
                    components=comp,
                )
            )
        return out


def build_mention_table(
    items: pd.DataFrame,
    extractor: "TickerExtractor",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Explode a Reddit-item table into one row per (item, ticker) mention,
    enriched with per-item sentiment and spam features.

    The market-clock ``date`` assigned here is what the no-lookahead rule keys
    off downstream: an item posted on date D contributes to the signal computed
    after the close of D, tradable at the open of D+1.
    """
    from . import bot_spam_filters as bsf
    from . import sentiment as sent

    settings = settings or load_settings()
    tz = settings.get("timezone", default="America/New_York")
    if items.empty:
        return pd.DataFrame()

    items = items.copy()
    created = pd.to_datetime(items["created_dt"], utc=True)
    items["date"] = created.dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)

    rows: list[dict] = []
    for item in items.itertuples(index=False):
        text = " ".join(filter(None, [getattr(item, "title", None), getattr(item, "body", None)]))
        mentions = extractor.extract(text)
        if not mentions:
            continue
        st = sent.score_text(text)
        sp = bsf.item_flags(text, getattr(item, "score", 0))
        for mn in mentions:
            rows.append(
                {
                    "id": item.id,
                    "kind": item.kind,
                    "subreddit": item.subreddit,
                    "author_hash": item.author_hash,
                    "created_dt": item.created_dt,
                    "date": item.date,
                    "score": getattr(item, "score", 0),
                    "num_comments": getattr(item, "num_comments", None),
                    "upvote_ratio": getattr(item, "upvote_ratio", float("nan")),
                    "flair": getattr(item, "flair", None),
                    "link_id": getattr(item, "link_id", None),
                    "permalink": getattr(item, "permalink", None),
                    "synthetic": getattr(item, "synthetic", False),
                    # mention evidence
                    "ticker": mn.ticker,
                    "method": mn.method,
                    "confidence": mn.confidence,
                    "n_occurrences": mn.n_occurrences,
                    "has_cashtag": mn.has_cashtag,
                    "finance_context": mn.finance_context,
                    "valid_exchange": mn.valid_exchange,
                    "is_common_word": mn.is_common_word,
                    "is_ambiguous": mn.is_ambiguous,
                    # text / sentiment
                    **st,
                    # spam / quality
                    **sp,
                }
            )
    out = pd.DataFrame(rows)
    log.info("Extracted %d mentions from %d items (%d distinct tickers)",
             len(out), len(items), out["ticker"].nunique() if not out.empty else 0)
    return out


def _load_aliases(root: Path) -> dict[str, str]:
    path = root / "data" / "manual" / "ticker_aliases" / "aliases.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, str] = {}
    for k, v in (raw.get("aliases") or {}).items():
        out[str(k).lower()] = str(v).upper()
    # misspellings may point at an alias key or a direct ticker
    for k, v in (raw.get("misspellings") or {}).items():
        target = str(v)
        out[str(k).lower()] = out.get(target.lower(), target.upper())
    return out
