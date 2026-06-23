"""Ticker-extraction correctness: cashtags accepted, common words / blocklist
rejected, ambiguous names require a cashtag, company aliases resolved."""
from __future__ import annotations

import pytest

from reddit_hype.ticker_extractor import TickerExtractor


@pytest.fixture(scope="module")
def extractor(settings):
    # built from seed tickers + curated aliases + config filters (no universe file dependency)
    return TickerExtractor.from_settings(settings)


def tickers(extractor, text):
    return {m.ticker for m in extractor.extract(text)}


def test_cashtag_accepted(extractor):
    assert "NVDA" in tickers(extractor, "loaded up on $NVDA calls, printing today")


def test_bare_valid_ticker_with_context_accepted(extractor):
    assert "NVDA" in tickers(extractor, "I bought NVDA shares today, holding long")


def test_bare_ambiguous_word_rejected_without_cashtag(extractor):
    # "BE" is a real ticker (Bloom Energy) but an English word — reject bare.
    assert "BE" not in tickers(extractor, "Just BE patient and hold your positions")


def test_cashtag_rescues_ambiguous(extractor):
    assert "BE" in tickers(extractor, "$BE is my favourite hydrogen play, buying shares")


def test_blocklist_hard_drop_even_with_cashtag(extractor):
    # "DD" is blocklisted: never a ticker, even as a cashtag.
    assert "DD" not in tickers(extractor, "great $DD on this name, calls incoming")


def test_common_non_ticker_word_not_extracted(extractor):
    out = tickers(extractor, "this is going to MOON, easy money, to the moon")
    assert "MOON" not in out


def test_company_alias_resolved(extractor):
    out = tickers(extractor, "Bloom Energy looks strong, buying calls and holding shares")
    assert "BE" in out


def test_alias_for_nebius(extractor):
    out = tickers(extractor, "Nebius is the cloud GPU play, I bought shares")
    assert "NBIS" in out
    assert "AI" not in out  # bare ambiguous, no cashtag


def test_confidence_cashtag_ge_bare(extractor):
    cash = {m.ticker: m.confidence for m in extractor.extract("$NVDA calls")}
    bare = {m.ticker: m.confidence for m in extractor.extract("I bought NVDA shares")}
    assert cash["NVDA"] >= bare["NVDA"]


def test_single_letter_bare_token_rejected(extractor):
    # "C" (Citigroup), "P" (from P/E), "S" (from S&P) are valid 1-letter tickers
    # but must NOT be extracted from bare prose — only as cashtags.
    assert "C" not in tickers(extractor, "Citigroup C looks cheap, buying shares")
    assert "P" not in tickers(extractor, "the P/E ratio and S&P 500 both matter for stocks")
    assert "S" not in tickers(extractor, "the S&P 500 is making new highs, calls")


def test_single_letter_cashtag_still_works(extractor):
    assert "C" in tickers(extractor, "$C earnings beat, buying calls")


def test_confidence_in_unit_interval(extractor):
    for m in extractor.extract("$NVDA $AMD calls, bought CCJ shares, Bloom Energy DD"):
        assert 0.0 <= m.confidence <= 1.0
