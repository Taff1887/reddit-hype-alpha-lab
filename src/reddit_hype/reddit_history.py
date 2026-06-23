"""Historical Reddit backfill loader.

Ingests monthly Reddit dump files (Pushshift / Academic Torrents / arctic_shift)
into the SAME tidy schema the live client emits (``REDDIT_ITEM_COLUMNS``), so the
rest of the pipeline (extract-mentions -> features -> event-study -> backtest)
runs on real history — enabling a retrospective study of e.g. the 2021
meme-stock era that the live API can't reach.

Supported inputs (auto-detected by extension), streamed line-by-line so multi-GB
months don't blow memory:
  * ``.zst``  — canonical Pushshift NDJSON (needs ``pip install -e ".[history]"``)
  * ``.gz``   — gzipped NDJSON
  * ``.jsonl`` / ``.ndjson`` / ``.json`` / ``.txt`` — plain NDJSON
  * ``.csv``  — column-mapped best-effort

Posts vs comments are inferred from the filename (``RS_*`` / ``*submission*`` ->
post, ``RC_*`` / ``*comment*`` -> comment) and falls back to record fields.
Records are filtered to the configured subreddits and date window *while
streaming*, and usernames are hashed immediately.
"""
from __future__ import annotations

import csv as _csv
import gzip
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from .config import Settings, load_settings
from .reddit_client import REDDIT_ITEM_COLUMNS
from .utils import ensure_dir, get_logger, hash_username, read_parquet_or_empty, write_parquet

log = get_logger(__name__)

_DUMP_EXTS = {".zst", ".gz", ".jsonl", ".ndjson", ".json", ".txt", ".csv"}

try:  # fast JSON if available, else stdlib
    import orjson

    def _loads(line: str | bytes):
        return orjson.loads(line)
except Exception:  # pragma: no cover
    def _loads(line: str | bytes):
        return json.loads(line)


# ------------------------------------------------------------------ streaming
def _open_lines(path: Path) -> Iterator[str]:
    """Yield decoded text lines from a dump file, streaming."""
    suffix = path.suffix.lower()
    if suffix == ".zst":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"{path.name} is .zst but zstandard is not installed. "
                'Run:  pip install -e ".[history]"'
            ) from exc
        with open(path, "rb") as fh:
            # Reddit dumps use large windows -> raise the cap to avoid decode errors.
            dctx = zstd.ZstdDecompressor(max_window_size=2**31)
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
                yield from text
    elif suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            yield from fh
    else:  # plain text NDJSON
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            yield from fh


def _iter_records(path: Path) -> Iterator[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            for row in _csv.DictReader(fh):
                yield row
        return
    for line in _open_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            yield _loads(line)
        except Exception:
            continue  # skip malformed line, keep streaming


# ------------------------------------------------------------------ mapping
def _kind_for(path: Path, rec: dict) -> str:
    name = path.name.lower()
    if name.startswith("rs_") or "submission" in name:
        return "post"
    if name.startswith("rc_") or "comment" in name:
        return "comment"
    # fall back to record shape
    if "body" in rec and "title" not in rec:
        return "comment"
    return "post"


def _to_epoch(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(value).timestamp()
        except Exception:
            return None


def _permalink(rec: dict, kind: str) -> str:
    pl = rec.get("permalink")
    if pl:
        return pl if str(pl).startswith("http") else f"https://reddit.com{pl}"
    sub, _id = rec.get("subreddit", ""), rec.get("id", "")
    return f"https://reddit.com/r/{sub}/comments/{_id}"


def _normalize(rec: dict, kind: str, salt: str) -> dict | None:
    rid = rec.get("id")
    if not rid:
        return None
    created = _to_epoch(rec.get("created_utc"))
    if created is None:
        return None
    is_post = kind == "post"
    body = rec.get("selftext") if is_post else rec.get("body")
    return {
        "id": ("t3_" if is_post else "t1_") + str(rid),
        "kind": kind,
        "subreddit": rec.get("subreddit"),
        "author_hash": hash_username(rec.get("author"), salt=salt),
        "created_utc": created,
        "created_dt": datetime.fromtimestamp(created, tz=timezone.utc),
        "score": int(rec.get("score") or 0),
        "title": rec.get("title") if is_post else None,
        "body": body or "",
        "num_comments": int(rec.get("num_comments") or 0) if is_post else None,
        "upvote_ratio": float(rec["upvote_ratio"]) if (is_post and rec.get("upvote_ratio") not in (None, "")) else float("nan"),
        "flair": rec.get("link_flair_text") if is_post else None,
        "permalink": _permalink(rec, kind),
        "link_id": rec.get("link_id") or (("t3_" + str(rid)) if is_post else None),
        "synthetic": False,
    }


# ------------------------------------------------------------------ driver
def discover_dumps(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in _DUMP_EXTS)


def load_dumps(
    settings: Settings | None = None,
    input_dir: str | Path | None = None,
    since: str | None = None,
    until: str | None = None,
    subreddits: Iterable[str] | None = None,
    max_records: int | None = None,
    merge: bool = True,
    write: bool = True,
) -> pd.DataFrame:
    """Stream dump files, filter to target subreddits + date window, map to the
    standard schema, and write/merge into the reddit_items parquet."""
    settings = settings or load_settings()
    input_dir = Path(input_dir) if input_dir else (settings.path("raw_reddit") / "dumps")
    files = discover_dumps(Path(input_dir))
    if not files:
        log.warning("No dump files found in %s (looked for %s). See the README in that dir.",
                    input_dir, sorted(_DUMP_EXTS))
        return read_parquet_or_empty(settings.path("reddit_items"))

    sub_set = {s.lower() for s in (subreddits or settings.subreddit_names())}
    salt = "rhal"
    since_ts = pd.Timestamp(since or settings.get("data_start", default="2000-01-01"), tz="UTC").timestamp()
    until_ts = pd.Timestamp(until, tz="UTC").timestamp() if until else None

    rows: list[dict] = []
    scanned = kept = 0
    for path in files:
        kind = _kind_for(path, {})
        file_kept = 0
        for rec in _iter_records(path):
            scanned += 1
            if scanned % 1_000_000 == 0:
                log.info("  scanned %s lines, kept %s...", f"{scanned:,}", f"{kept:,}")
            sub = str(rec.get("subreddit", "")).lower()
            if sub_set and sub not in sub_set:
                continue
            created = _to_epoch(rec.get("created_utc"))
            if created is None or created < since_ts or (until_ts and created > until_ts):
                continue
            k = _kind_for(path, rec)
            row = _normalize(rec, k, salt)
            if row is None:
                continue
            rows.append(row)
            kept += 1
            file_kept += 1
            if max_records and kept >= max_records:
                log.warning("Hit max_records=%s — stopping early.", max_records)
                break
        log.info("%s -> kept %s rows (kind hint=%s)", path.name, f"{file_kept:,}", kind)
        if max_records and kept >= max_records:
            break

    new = pd.DataFrame(rows, columns=REDDIT_ITEM_COLUMNS)
    log.info("Loaded %s historical items from %d file(s) (scanned %s lines).",
             f"{len(new):,}", len(files), f"{scanned:,}")
    if new.empty:
        return new

    path_out = settings.path("reddit_items")
    if merge:
        existing = read_parquet_or_empty(path_out)
        if not existing.empty:
            new = pd.concat([existing, new], ignore_index=True)
    new = new.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    if write:
        write_parquet(new, path_out)
        log.info("reddit_items now %s rows (%s..%s) -> %s",
                 f"{len(new):,}",
                 pd.to_datetime(new["created_dt"]).min(),
                 pd.to_datetime(new["created_dt"]).max(),
                 path_out)
    return new


def _arctic_window(base, kind_path, kind, sub, after_ts, before_ts, cap, ua, sleep) -> list[dict]:
    """Paginate one (kind, subreddit, [after,before)) window up to ``cap`` rows."""
    import urllib.request

    rows, after, kept = [], after_ts, 0
    while kept < cap:
        url = (f"{base}/{kind_path}/search?subreddit={sub}"
               f"&after={after}&before={before_ts}&limit=100&sort=asc")
        try:
            req = urllib.request.Request(url, headers=ua)
            with urllib.request.urlopen(req, timeout=40) as r:
                batch = (_loads(r.read()) or {}).get("data", [])
        except Exception as exc:
            log.warning("arctic_shift %s r/%s failed at after=%s: %s", kind, sub, after, exc)
            break
        if not batch:
            break
        last_ts = after
        for rec in batch:
            row = _normalize(rec, kind, "rhal")
            if row is not None:
                rows.append(row)
                kept += 1
            last_ts = max(last_ts, int(_to_epoch(rec.get("created_utc")) or last_ts))
        time.sleep(sleep)
        if len(batch) < 100:
            break
        after = last_ts + 1 if last_ts <= after else last_ts
    return rows


def fetch_arctic_shift(
    settings: Settings | None = None,
    subreddits: Iterable[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    kinds: Iterable[str] = ("posts", "comments"),
    max_records_per_kind: int = 20000,
    sleep: float = 0.4,
    write: bool = True,
    chunk_days: int | None = None,
) -> pd.DataFrame:
    """Pull historical Reddit data from the keyless arctic_shift API (Pushshift
    successor) — no torrents, no key. Paginates by ``created_utc`` into the
    standard schema. Set ``chunk_days`` (e.g. 30) to spread the per-kind cap
    EVENLY across the window (so high-volume subs don't front-load coverage) —
    essential for credible 'most talked about over time' history."""
    settings = settings or load_settings()
    subs = list(subreddits or settings.subreddit_names())
    since_ts = int(pd.Timestamp(since or settings.get("data_start", default="2020-01-01"), tz="UTC").timestamp())
    until_ts = int(pd.Timestamp(until, tz="UTC").timestamp()) if until else int(pd.Timestamp.utcnow().timestamp())
    base = "https://arctic-shift.photon-reddit.com/api"
    ua = {"User-Agent": settings.credentials.reddit_user_agent or "reddit-hype-alpha-lab/0.1"}

    # build [a,b) chunk boundaries
    if chunk_days and chunk_days > 0:
        step = chunk_days * 86400
        bounds = list(range(since_ts, until_ts, step)) + [until_ts]
        chunks = list(zip(bounds[:-1], bounds[1:]))
        cap_per = max(100, max_records_per_kind // max(1, len(chunks)))
    else:
        chunks = [(since_ts, until_ts)]
        cap_per = max_records_per_kind

    rows: list[dict] = []
    for kind_path in kinds:
        kind = "post" if kind_path.startswith("post") else "comment"
        for sub in subs:
            kept0 = len(rows)
            for a, b in chunks:
                rows.extend(_arctic_window(base, kind_path, kind, sub, a, b, cap_per, ua, sleep))
            log.info("arctic_shift %s r/%s: kept %s (chunks=%d)", kind, sub,
                     f"{len(rows) - kept0:,}", len(chunks))

    new = pd.DataFrame(rows, columns=REDDIT_ITEM_COLUMNS)
    if new.empty:
        log.warning("arctic_shift returned no rows for subs=%s window=%s..%s", subs, since, until)
        return new
    path_out = settings.path("reddit_items")
    existing = read_parquet_or_empty(path_out)
    if not existing.empty:
        new = pd.concat([existing, new], ignore_index=True)
    new = new.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    if write:
        write_parquet(new, path_out)
        log.info("reddit_items now %s rows -> %s", f"{len(new):,}", path_out)
    return new


def write_dumps_readme(settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    d = ensure_dir(settings.path("raw_reddit") / "dumps")
    readme = d / "README.md"
    readme.write_text(
        "# Reddit history dumps\n\n"
        "Drop monthly Reddit dump files here, then run:\n\n"
        "    python scripts/load_reddit_history.py --since 2020-06-01 --until 2021-06-30\n\n"
        "## Where to get them\n"
        "- Academic Torrents 'Reddit comments/submissions' monthly dumps (`RC_YYYY-MM.zst`, `RS_YYYY-MM.zst`).\n"
        "- Per-subreddit dumps (e.g. `wallstreetbets_submissions.zst`) from the subreddit-archive torrents.\n"
        "- The arctic_shift / pushshift mirrors.\n\n"
        "## Accepted formats\n"
        "`.zst`, `.gz`, `.jsonl`/`.ndjson`/`.json`, `.csv`. Posts vs comments are inferred from the\n"
        "filename (`RS_`/submission -> post, `RC_`/comment -> comment).\n\n"
        "## Tips\n"
        "- These files are huge. The loader streams and filters to the subreddits in\n"
        "  `configs/subreddits.yaml` and the `--since/--until` window while reading.\n"
        "- `.zst` needs `pip install -e \".[history]\"` (zstandard). `orjson` speeds parsing.\n"
        "- Narrow the date window and subreddit list to keep the extraction step fast.\n",
        encoding="utf-8",
    )
    return readme
