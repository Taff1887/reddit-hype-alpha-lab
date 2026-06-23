"""Reddit data access.

Two interchangeable clients return the SAME tidy schema (one row per post or
comment):

* :class:`PrawRedditClient` — live, read-only public data via PRAW. Never posts,
  votes, comments, or messages. Usernames are hashed immediately.
* :class:`MockRedditClient` — deterministic SYNTHETIC data for offline
  development and tests. **Synthetic rows are flagged ``synthetic=True`` and must
  never be used to make research claims.**

Unified columns:
    id, kind, subreddit, author_hash, created_utc (epoch s), created_dt (UTC),
    score, title, body, num_comments, upvote_ratio, flair, permalink, link_id,
    synthetic
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import get_logger, hash_username

log = get_logger(__name__)

REDDIT_ITEM_COLUMNS = [
    "id", "kind", "subreddit", "author_hash", "created_utc", "created_dt",
    "score", "title", "body", "num_comments", "upvote_ratio", "flair",
    "permalink", "link_id", "synthetic",
]


class RedditClient:
    """Interface contract."""

    def fetch(self, subreddits: list[str] | None = None) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- live
class PrawRedditClient(RedditClient):
    def __init__(self, settings: Settings):
        import praw  # imported lazily so the package works without praw installed

        self.settings = settings
        c = settings.credentials
        self.reddit = praw.Reddit(
            client_id=c.reddit_client_id,
            client_secret=c.reddit_client_secret,
            user_agent=c.reddit_user_agent,
            check_for_async=False,
        )
        self.reddit.read_only = True  # belt-and-braces: never write
        self.posts_per_sub = int(settings.get("reddit", "posts_per_subreddit", default=200))
        self.comments_per_post = int(settings.get("reddit", "comments_per_post", default=100))
        self.listing = settings.get("reddit", "listing", default="new")
        self.sleep = float(settings.get("reddit", "request_sleep_seconds", default=1.0))

    def _listing(self, subreddit):
        limit = self.posts_per_sub
        return {
            "new": subreddit.new,
            "hot": subreddit.hot,
            "top": subreddit.top,
        }.get(self.listing, subreddit.new)(limit=limit)

    def fetch(self, subreddits: list[str] | None = None) -> pd.DataFrame:
        subs = subreddits or self.settings.subreddit_names()
        rows: list[dict] = []
        for name in subs:
            try:
                subreddit = self.reddit.subreddit(name)
                for sub in self._listing(subreddit):
                    rows.append(self._submission_row(sub, name))
                    rows.extend(self._comment_rows(sub, name))
                    time.sleep(self.sleep)
            except Exception as exc:  # keep going if one sub fails
                log.warning("Failed to fetch r/%s: %s", name, exc)
        df = pd.DataFrame(rows, columns=REDDIT_ITEM_COLUMNS)
        log.info("Fetched %d Reddit items across %d subreddits (LIVE)", len(df), len(subs))
        return df

    def _submission_row(self, sub, name: str) -> dict:
        created = float(sub.created_utc)
        return {
            "id": f"t3_{sub.id}",
            "kind": "post",
            "subreddit": name,
            "author_hash": hash_username(getattr(sub.author, "name", None)),
            "created_utc": created,
            "created_dt": datetime.fromtimestamp(created, tz=timezone.utc),
            "score": int(sub.score),
            "title": sub.title or "",
            "body": sub.selftext or "",
            "num_comments": int(sub.num_comments),
            "upvote_ratio": float(getattr(sub, "upvote_ratio", float("nan"))),
            "flair": getattr(sub, "link_flair_text", None),
            "permalink": f"https://reddit.com{sub.permalink}",
            "link_id": f"t3_{sub.id}",
            "synthetic": False,
        }

    def _comment_rows(self, sub, name: str) -> list[dict]:
        rows: list[dict] = []
        try:
            sub.comments.replace_more(limit=0)
            comments = sub.comments.list()[: self.comments_per_post]
        except Exception as exc:
            log.debug("comment fetch failed for %s: %s", sub.id, exc)
            return rows
        for c in comments:
            created = float(c.created_utc)
            rows.append(
                {
                    "id": f"t1_{c.id}",
                    "kind": "comment",
                    "subreddit": name,
                    "author_hash": hash_username(getattr(c.author, "name", None)),
                    "created_utc": created,
                    "created_dt": datetime.fromtimestamp(created, tz=timezone.utc),
                    "score": int(c.score),
                    "title": None,
                    "body": c.body or "",
                    "num_comments": None,
                    "upvote_ratio": float("nan"),
                    "flair": None,
                    "permalink": f"https://reddit.com{c.permalink}",
                    "link_id": f"t3_{sub.id}",
                    "synthetic": False,
                }
            )
        return rows


# --------------------------------------------------------------------------- mock
# Templated text fragments — each contains finance context + a cashtag slot so the
# (conservative) extractor can recover the ticker. Sentiment registers vary.
_BULLISH = [
    "Just added more ${t}, loading up before earnings. calls printing.",
    "I bought ${t} shares today, this is my biggest position. holding long.",
    "${t} squeeze incoming, shorts are trapped. diamond hands 🚀",
    "DD on ${t}: revenue growth + margin expansion, price target raised.",
    "all in ${t} calls, easy money. to the moon",
]
_DD = [
    "Deep dive on ${t}: balance sheet is clean, FCF positive, backlog growing. "
    "Valuation looks cheap vs peers given the catalyst in Q3 guidance.",
    "${t} fundamentals thread — revenue up, dilution risk low, insider buying. "
    "Resource grade and offtake contracts support the production ramp.",
]
_BEARISH = [
    "Sold my ${t} position, dilution risk too high. buying puts.",
    "${t} is a pump, dont miss the exit. looks like a rugpull to me",
    "bagholding ${t}, this is dumping hard. taking the loss",
]
_MEME = [
    "${t} 🚀🚀 to the moon tendies incoming",
    "${t} yolo lambo soon trust me bro",
]
_PUMP = [
    "BUY ${t} NOW guaranteed 10x dont miss last chance back up the truck",
    "${t} next GME, sure thing, load up before it moons",
]


class MockRedditClient(RedditClient):
    """Deterministic synthetic generator. DEV/TEST ONLY — not real data."""

    def __init__(self, settings: Settings):
        self.settings = settings
        m = settings.get("mock", default={}) or {}
        self.seed = int(m.get("seed", 1887))
        self.n_days = int(m.get("n_days", 180))
        self.n_tickers = int(m.get("n_tickers", 40))

    def fetch(self, subreddits: list[str] | None = None) -> pd.DataFrame:
        from .ticker_extractor import _SEED_TICKERS  # reuse the curated seed list

        rng = np.random.RandomState(self.seed)
        sub_meta = self.settings.subreddit_list() or [
            {"name": "wallstreetbets", "tier": "large"},
            {"name": "stocks", "tier": "large"},
        ]
        subs = subreddits or [s["name"] for s in sub_meta]
        tier_map = {s["name"]: s.get("tier", "mid") for s in sub_meta}

        tickers = sorted(_SEED_TICKERS)
        rng.shuffle(tickers)
        tickers = tickers[: self.n_tickers]

        # Baseline daily mention propensity + a few engineered hype waves.
        base_rate = rng.uniform(0.2, 2.0, size=len(tickers))
        wave_centers = rng.randint(0, self.n_days, size=len(tickers))
        wave_amp = rng.choice([0, 0, 0, 4, 8, 15], size=len(tickers))  # most have no wave
        wave_width = rng.uniform(2, 6, size=len(tickers))

        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        rows: list[dict] = []
        uid = 0
        for d in range(self.n_days):
            day = end - timedelta(days=self.n_days - 1 - d)
            for ti, ticker in enumerate(tickers):
                wave = wave_amp[ti] * np.exp(-((d - wave_centers[ti]) ** 2) / (2 * wave_width[ti] ** 2))
                lam = base_rate[ti] + wave
                n_items = rng.poisson(lam)
                for _ in range(int(n_items)):
                    uid += 1
                    sub = subs[rng.randint(len(subs))]
                    bare = ticker.split(".")[0]
                    register = rng.choice(
                        ["bullish", "dd", "bearish", "meme", "pump"],
                        p=[0.45, 0.15, 0.2, 0.12, 0.08],
                    )
                    template = {
                        "bullish": _BULLISH, "dd": _DD, "bearish": _BEARISH,
                        "meme": _MEME, "pump": _PUMP,
                    }[register]
                    text = template[rng.randint(len(template))].replace("${t}", bare)
                    is_post = rng.rand() < 0.35
                    secs = int(rng.randint(0, 86400))
                    created = day + timedelta(seconds=secs)
                    flair = "DD" if register == "dd" else ("Meme" if register == "meme" else None)
                    rows.append(
                        {
                            "id": f"mock_{uid}",
                            "kind": "post" if is_post else "comment",
                            "subreddit": sub,
                            "author_hash": hash_username(f"user_{rng.randint(0, 5000)}"),
                            "created_utc": created.timestamp(),
                            "created_dt": created,
                            "score": int(max(0, rng.normal(20 if is_post else 5, 30))),
                            "title": (text[:120] if is_post else None),
                            "body": text,
                            "num_comments": int(abs(rng.normal(15, 25))) if is_post else None,
                            "upvote_ratio": round(float(rng.uniform(0.5, 0.99)), 2) if is_post else float("nan"),
                            "flair": flair,
                            "permalink": f"https://reddit.com/r/{sub}/comments/mock_{uid}",
                            "link_id": f"mock_thread_{uid // 5}",
                            "synthetic": True,
                        }
                    )
        df = pd.DataFrame(rows, columns=REDDIT_ITEM_COLUMNS)
        log.warning(
            "Generated %d SYNTHETIC Reddit items (MOCK MODE). Do NOT use for research claims.",
            len(df),
        )
        return df


def get_reddit_client(settings: Settings | None = None) -> RedditClient:
    settings = settings or load_settings()
    if settings.reddit_mode == "live":
        log.info("Using LIVE Reddit client (PRAW, read-only).")
        return PrawRedditClient(settings)
    log.warning("Using MOCK Reddit client — synthetic development data only.")
    return MockRedditClient(settings)
