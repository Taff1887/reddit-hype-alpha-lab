# reddit-hype-alpha-lab

A serious research engine that scans Reddit for stock hype, measures
**sentiment / attention / velocity**, and rigorously backtests whether that hype
predicts forward returns — then produces a ranked, risk-aware watchlist.

This is **not** a meme toy. Raw mention counts are noise. The lab is built around
one question:

> **Does abnormal Reddit attention contain *tradable* alpha — after costs,
> realistic execution, liquidity limits, and out-of-sample validation — and is
> there a plausible behavioural reason it should?**

It looks hardest where retail attention can actually move price: small/mid caps,
AI-infrastructure names, uranium/mining, battery metals, bitcoin miners, short-
squeeze candidates, under-the-radar tickers starting to get unusual attention,
and ASX names on r/ASX_Bets / r/AusFinance / r/ausstocks.

> ⚠️ **Honesty first.** With no API keys the lab runs in **MOCK mode** on
> deterministic *synthetic* data (random-walk prices with **no** engineered link
> to hype). Mock outputs prove the pipeline works; they are **never** evidence of
> alpha. Every synthetic artifact is flagged `synthetic=True` and the scorecard
> refuses to claim alpha on it. Provide real credentials before drawing any
> conclusion.

---

## What it produces

1. **Daily ranked hype watchlist** — `reports/daily_watchlists/YYYY-MM-DD_reddit_hype_watchlist.csv`
   with a `suggested_action` (WATCH / BUY_CANDIDATE / AVOID_OVERHEATED / RESEARCH_ONLY),
   a `reason_summary`, and a `risk_summary` per name.
2. **Highest-conviction list** — high composite score **and** high data quality,
   broad unique-user breadth, tradable liquidity, low pump/spam risk.
3. **Backtested portfolios** — nine strategies × multiple holding periods, net of costs.
4. **Strategy scoreboard** — `reports/scorecards/` (what worked, what failed, honest verdict).
5. **Research dashboard** — Streamlit app for hype-over-time, sentiment, price-vs-hype, subreddit breakdown.

---

## Quickstart (works with zero API keys, in MOCK mode)

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows; use bin/activate on *nix
pip install -e ".[dev]"          # core + dev. Add ".[all]" for praw/streamlit/lightgbm/vader

make pipeline                    # universe -> reddit -> mentions -> prices -> features -> watchlist
make event-study                 # forward-return event study around attention spikes
make backtest                    # strategy scoreboard + equity curves + scorecard
make research-loop               # bounded hypothesis-testing / re-weighting loop
make dashboard                   # Streamlit UI
make test                        # 28 tests incl. extractor + no-lookahead
```

On Windows without GNU `make`, run the script directly, e.g. `python scripts/generate_watchlist.py`.

---

## Getting real data

Copy `.env.example` to `.env` and fill in:

```
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=reddit-hype-alpha-lab/0.1 by your_username
FMP_API_KEY=
```

**Reddit API credentials** (free): log in → <https://www.reddit.com/prefs/apps> →
*create another app…* → type **script** → redirect uri `http://localhost:8080`.
The string under the app name is your `client_id`; the `secret` is shown beside it.
Install the optional extra: `pip install -e ".[reddit]"`.

**FMP key**: <https://site.financialmodelingprep.com/>. Used for the tradable
universe, prices, market cap, sector/industry, volume, exchange, company names,
and ticker validation.

The lab auto-detects keys: present → **LIVE**, absent → **MOCK** (`mode: auto` in
`configs/config.yaml`; force with `mode: live|mock` or `RHAL_FORCE_MOCK=1`).

### Ethics & compliance (built in)
Reddit is used **only** as a passive public-data source via the official API/PRAW.
The lab **never posts, votes, comments, or messages** (`reddit.read_only = True`),
respects configurable rate limits, stores no unnecessary personal data, and
**hashes usernames** (SHA-256, truncated) before anything is persisted.

---

## How it works (the pipeline)

```
universe ─┐
          ├─ mentions ──┐
reddit ───┘             ├─ features (+ scores + no-lookahead labels) ──┬─ watchlist
prices ─────────────────┘                                              ├─ event study
                                                                       ├─ backtest / scoreboard
                                                                       └─ research loop
```

### 1 · Ticker extraction (the most correctness-critical step)
`src/reddit_hype/ticker_extractor.py`. Bad extraction silently poisons everything,
so it is conservative and **auditable** — each mention carries a confidence
breakdown:

```
confidence = cashtag_bonus + company_name_nearby_bonus + valid_exchange_bonus
           + finance_context_bonus
           − common_word_penalty − low_context_penalty − ambiguous_no_cashtag_penalty
```

- **Cashtags** (`$NVDA`, `$PDN.AX`) are strong evidence.
- **Bare tokens** (`NVDA`) must match the valid listed universe **and** carry
  finance context to clear `min_confidence` (default 0.55).
- **English-word / acronym tickers** (`BE`, `AI`, `ON`, `IT`, `DD`, `CEO`, `YOLO`,
  `CPI`, …) are blocklisted or **require a cashtag** (`configs/ticker_filters.yaml`).
- **Company names / aliases** (`Bloom Energy`→BE, `Nebius`→NBIS, misspellings) are
  matched from `data/manual/ticker_aliases/aliases.yaml`.

### 2 · Features (`hype_features.py`)
Per ticker-day: **attention** (mentions, unique authors/threads, breadth, scores),
**velocity** (vs 7d/30d baselines, z-score, acceleration, spike flag — computed on
a gap-filled grid so silent days count as zero, using only the past), **sentiment
/ conviction / quality** (engagement-weighted lexicon language, commitment phrases
like "I bought"/"calls"/"holding", DD substance, pump/spam), and **market** features
(trailing returns, volume z-score, market cap, sector, volatility, liquidity)
joined strictly **as-of** the signal date.

### 3 · Scores (`models.py`)
Seven bounded, auditable components — **Attention, Hype-Velocity, Conviction,
Quality-DD, Underreaction, Tradability, Pump-Risk** — combined into a
`final_hype_alpha_score` via configurable weights (`configs/strategy_params.yaml`,
`score_weights`). Nothing is a magic number in code.

### 4 · Labels with **strict no-lookahead** (`labels.py`)
A signal accumulated over calendar date *D* is known only after *D*'s close.
Entry is the **next trading open** (`asof_date` ≤ D < `entry_date`). Forward
returns over 1/3/5/10/20 days, plus market/sector/vol-adjusted variants, max
drawdown, and gap-at-entry. Enforced by `tests/test_no_lookahead.py` and a runtime
`validate_no_lookahead` assert.

### 5 · Backtest (`backtester.py`, `costs.py`)
Overlapping-cohort (Jegadeesh-Titman) daily simulation: each day forms a cohort
entered next-open and held H days; the book is a 1/H blend of overlapping cohorts.
Costs = commission + half-spread + slippage on each cohort's entry/exit sleeve
(conservative — ignores name-overlap netting). Capacity/participation reported
separately in `diagnostics.py`.

### 6 · ML (`models.walk_forward`)
Logistic regression + random forest (+ optional LightGBM) predicting P(positive
forward return), validated **walk-forward** (time-ordered, never a random split).
Reports AUC, feature importance, and stability across folds.

### 7 · Research loop (`research_loop.py`)
Bounded self-improvement: scores components by out-of-sample rank IC, tests the
standing hypotheses (acceleration vs raw mentions; unique authors vs total
mentions; DD quality vs meme hype; does underreaction matter; are we just chasing
already-ran names), proposes a re-weighting, and **keeps it only if OOS net Sharpe
improves**. Logs every iteration, including failures.

---

## Strategies (`configs/strategy_params.yaml → strategies`)

| Strategy | Idea |
|---|---|
| `TopHypeLongOnly` | Buy top-N by `final_hype_alpha_score`; hold 1/3/5/10/20d |
| `HypeAcceleration` | Mentions spike above baseline **and** price hasn't moved yet |
| `ConvictionOnly` | High commitment language + broad unique-author base |
| `QualityDDOnly` | Fewer posts but high DD substance |
| `UnderreactedHype` | Attention surging, price still muted |
| `HypeExhaustionFade` | **Research only** — do overhyped names fade? (no short assumed) |
| `SmallCapHype` | Liquid small/mid caps where retail can move price |
| `SectorHypeRotation` | Sector-level hype (uranium, AI power, miners, …) |
| `CrossSubredditBreakout` | Niche → large-subreddit spread |
| `SqueezeSetup` | High short interest + low float + accelerating attention (the GME mechanism). Auto-inactive without a short-interest data source |

---

## Interpreting the hype scores

- **High `final_hype_alpha_score`** = abnormal, accelerating, bullish, conviction-
  backed, broad, substantive, *and not yet priced* attention — minus pump/spam and
  already-ran penalties.
- **"Highest conviction" ≠ guaranteed winner.** It means high score **and** high
  data quality **and** broad unique-user breadth **and** tradable liquidity **and**
  positive historical signal performance **and** low pump/spam risk.
- **`suggested_action`** is a triage label, not advice. `AVOID_OVERHEATED` flags
  pump-risk/already-ran; `RESEARCH_ONLY` flags thin/illiquid/synthetic; `BUY_CANDIDATE`
  is a top-decile, underreacted, tradable name.

## Avoiding false positives
Cashtag-or-evidence requirement, blocklist + ambiguous (require-cashtag) lists,
finance-context gating, unique-author breadth floors, bot/spam down-weighting,
near-duplicate (copy-paste) detection, and a per-mention confidence threshold. See
`configs/ticker_filters.yaml` and notebook `02_ticker_extraction.ipynb`.

## Hunting for the influential finding

The two most influential, defensible hypotheses in this space — and the tools
built to test them:

- **Attention-induced reversal** (Barber-Odean / Da-Engelberg-Gao applied to
  Reddit): abnormal attention drives a short pop that then *reverses*. Tested by
  `diagnostics.hype_decay_study`, which reports the top-decile market-adjusted
  forward-return **path** across horizons. Crucially it aggregates to one
  observation per date and uses **Newey-West (HAC) t-stats** with `lag = horizon`,
  so overlapping forward returns can't masquerade as independent draws. The
  verdict only fires (`POP-AND-FADE` / `CONTINUATION` / `REVERSAL`) when the
  signal is HAC-significant — on noise it correctly says *"no significant signal"*.
- **Short-squeeze setup** (the GME mechanism): in high-short-interest, low-float
  names, accelerating retail attention can force continuation. `squeeze_setup_score`
  + the `SqueezeSetup` strategy combine short interest, days-to-cover, attention
  acceleration and conviction. **Short interest is not on FMP's standard tier**, so
  this dimension is *inactive by default* in LIVE mode and never fabricated — wire
  a FINRA/commercial source into `ticker_universe.py` to switch it on.

### Two real data paths (this matters)
- **Reddit API (PRAW) returns only recent posts** — you can run a live daily
  watchlist and *accumulate* real data going forward, but you **cannot** backtest
  the Jan-2021 GME era from it.
- **Retrospective study of the famous episodes needs a historical Reddit corpus**
  (Pushshift / Academic Torrents monthly dumps). The `reddit_history` loader
  streams `.zst`/`.gz`/`.jsonl`/`.csv` dumps, filters to your subreddits + date
  window while reading, hashes usernames, and writes the standard schema:

  ```bash
  pip install -e ".[history]"                          # zstandard + orjson
  # drop RS_2020-06.zst, RC_2020-06.zst, ... into data/raw/reddit/dumps/
  python scripts/load_reddit_history.py --since 2020-06-01 --until 2021-06-30
  make extract-mentions && make fetch-prices && make build-features
  make event-study     # HAC decay/reversal verdict on the real meme-stock era
  make backtest
  ```

  Sources: Academic Torrents "Reddit comments/submissions" monthly dumps
  (`RC_YYYY-MM.zst` / `RS_YYYY-MM.zst`), per-subreddit archive torrents
  (`wallstreetbets_submissions.zst`), or arctic_shift mirrors. See
  `data/raw/reddit/dumps/README.md`.

## Findings on real data (keyless: arctic_shift + SEC + Yahoo)

Run across multiple independent windows (2022 bear, 2023, 2024) on 4 US equity
subreddits. All returns are market-adjusted, net of costs where stated, with
HAC (Newey-West) t-stats and liquidity stratification.

1. **The naive "attention pop" is a liquidity artifact.** A raw +1.25% next-day
   pop after attention spikes (autumn 2023) **vanished** once restricted to liquid
   names (>$10M/day: HAC t≈−1.4). It was penny-stock microstructure, not return.

2. **"Fade the acceleration" replicates (the one robust effect).** In liquid
   names, the top quartile by *attention acceleration* had **negative** next-day
   market-adjusted returns in **4/4 windows** (HAC t up to −2.8). Chasing the
   loudest, fastest-rising Reddit hype is a consistent small *loser*.

3. **Trading the 100 most talked-about stocks doesn't work** (Jan–Jun 2024,
   equal-weight, net of costs):

   | Universe | 5-day hold | 20-day hold |
   |---|---|---|
   | All names | Sharpe −3.8 (−52%/yr) | Sharpe +0.2 |
   | Liquid >$10M/day | Sharpe −5.0 | Sharpe +0.6 (+6.6%, **below buy-and-hold SPY**) |

   Short rebalancing is destroyed by turnover/costs; the least-bad version barely
   matches the index in a bull market. Not alpha.

4. **Data mining is a mirage — demonstrated, not hidden.** Sweeping 24 mention
   strategies, the best **in-sample** Sharpe was **1.07**; re-run **out-of-sample**,
   all top winners collapsed to **−0.71**. *Nothing survived.* This is exactly why
   the lab splits in-sample/out-of-sample and reports the gap.

**Bottom line:** no robust, tradable alpha in raw or most-talked Reddit attention
among liquid names over these windows. The only effect that replicates is that
*chasing* acceleration loses slightly. That is a real, defensible conclusion —
and the kind this lab is built to reach honestly rather than curve-fit around.

## We do **not** claim alpha unless it
survives transaction costs **and** realistic execution, is not driven by one meme
event, works out-of-sample across multiple periods, is liquid enough to trade, and
has a plausible behavioural explanation. The scorecard states this verdict explicitly.

---

## Repo layout
```
configs/        config, subreddits, ticker_filters, strategy_params (all behaviour lives here)
data/           raw/ interim/ processed/ (gitignored) + manual/ (curated, versioned)
notebooks/      01 audit · 02 extraction · 03 scores · 04 event study · 05 backtest · 06 dashboard
src/reddit_hype/ the engine (clients, extractor, features, models, portfolio, backtester, …)
scripts/        thin CLI wrappers (one per make target)
tests/          extractor + no-lookahead + features + backtester + portfolio (28 tests)
reports/        watchlists, figures, tables, scorecards (gitignored)
```

## Known limitations
- **Daily granularity** by default; intraday (1h/6h) velocity needs an hourly pull.
- **Universe survivorship**: the live FMP universe is current-membership; delisted
  names are absent (matters for squeeze/penny studies).
- **No real exchange-holiday calendar** (business-day approximation); single trading
  calendar assumed when mixing US/ASX.
- **Sentiment is lexicon-based** (optional VADER blend) — not a fine-tuned model.
- **Reddit API only returns recent listings**; deep history needs incremental daily
  pulls accumulated over time (the lab merges/dedupes each run).
- **Mock prices are pure random walks** — mock backtests *should* show ~no alpha.

---
*Built for systematic-equities research. Not investment advice.*
