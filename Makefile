# reddit-hype-alpha-lab — task runner
#
# On Windows without GNU make, run the underlying commands directly, e.g.:
#   python scripts/generate_watchlist.py
# Every target below is a thin wrapper around a script in scripts/.

PY ?= python
PIP ?= pip

.DEFAULT_GOAL := help

.PHONY: help install dev-install doctor fetch-reddit load-history backfill most-talked mine \
        fetch-prices build-universe extract-mentions build-features watchlist event-study backtest \
        research-loop dashboard test lint format pipeline clean

help:  ## Show this help
	@echo "reddit-hype-alpha-lab targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

doctor:  ## Pre-flight: verify Reddit + FMP credentials with a live ping
	$(PY) scripts/doctor.py

install:  ## Install the package (runtime deps)
	$(PIP) install -e .

dev-install:  ## Install with all optional + dev dependencies
	$(PIP) install -e ".[all,dev]"

fetch-reddit:  ## Pull recent Reddit posts/comments for configured subreddits
	$(PY) scripts/fetch_reddit.py

load-history:  ## Backfill reddit_items from historical dumps in data/raw/reddit/dumps
	$(PY) scripts/load_reddit_history.py

backfill:  ## KEYLESS real-data study: arctic_shift Reddit + SEC universe + Yahoo prices + event study
	$(PY) scripts/backfill.py

most-talked:  ## List the most talked-about stocks over the loaded window
	$(PY) scripts/most_talked_about.py

mine:  ## Data-mine mention strategies in-sample, then check winners out-of-sample
	$(PY) scripts/mine_strategies.py

fetch-prices:  ## Pull prices/volume for the ticker universe from FMP
	$(PY) scripts/fetch_prices.py

build-universe:  ## Build the tradable ticker universe from FMP
	$(PY) scripts/build_ticker_universe.py

extract-mentions:  ## Extract ticker mentions from raw Reddit data
	$(PY) scripts/extract_mentions.py

build-features:  ## Aggregate mentions + market data into the ticker-day feature panel
	$(PY) scripts/build_features.py

watchlist:  ## Generate today's ranked Reddit hype watchlist
	$(PY) scripts/generate_watchlist.py

event-study:  ## Run the forward-return event study around hype spikes
	$(PY) scripts/run_event_study.py

backtest:  ## Run the TopHypeLongOnly (and other) strategy backtests
	$(PY) scripts/run_backtest.py

research-loop:  ## Run the bounded research/self-improvement loop
	$(PY) scripts/run_research_loop.py

dashboard:  ## Launch the Streamlit research dashboard
	streamlit run src/reddit_hype/dashboard.py

pipeline: build-universe fetch-reddit extract-mentions fetch-prices build-features watchlist  ## Run the full daily pipeline end-to-end

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff
	ruff check src tests scripts

format:  ## Format with black + ruff
	black src tests scripts
	ruff check --fix src tests scripts

clean:  ## Remove caches and generated interim files
	rm -rf .pytest_cache **/__pycache__ data/interim/* data/processed/*
