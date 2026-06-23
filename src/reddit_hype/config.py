"""Configuration loading: resolves the repo root, reads the YAML configs, loads
``.env`` credentials, and decides whether we run live or in mock mode.

Everything downstream takes a :class:`Settings` object so there are no hidden
globals and tests can build a settings object pointing at a temp dir.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # optional, but recommended
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .utils import get_logger

log = get_logger(__name__)


def repo_root() -> Path:
    """Locate the repository root (parent of ``src/``), allowing an env override."""
    env = os.environ.get("RHAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at src/reddit_hype/config.py -> root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("Config file missing: %s (using empty config)", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Credentials:
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str | None = None
    fmp_api_key: str | None = None

    @property
    def has_reddit(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret and self.reddit_user_agent)

    @property
    def has_fmp(self) -> bool:
        return bool(self.fmp_api_key)


@dataclass
class Settings:
    root: Path
    config: dict[str, Any]
    subreddits: dict[str, Any]
    ticker_filters: dict[str, Any]
    strategy_params: dict[str, Any]
    credentials: Credentials
    force_mock: bool = False
    _path_cache: dict[str, Path] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- modes
    @property
    def reddit_mode(self) -> str:
        return self._resolve_mode(self.credentials.has_reddit)

    @property
    def fmp_mode(self) -> str:
        return self._resolve_mode(self.credentials.has_fmp)

    def _resolve_mode(self, has_keys: bool) -> str:
        configured = str(self.config.get("mode", "auto")).lower()
        if self.force_mock or configured == "mock":
            return "mock"
        if configured == "live":
            return "live"
        return "live" if has_keys else "mock"

    # ---------------------------------------------------------------- paths
    def path(self, key: str) -> Path:
        """Resolve a configured path/file key to an absolute Path.

        Looks first in ``config['files']`` then ``config['paths']``.
        """
        if key in self._path_cache:
            return self._path_cache[key]
        files = self.config.get("files", {})
        paths = self.config.get("paths", {})
        rel = files.get(key) or paths.get(key)
        if rel is None:
            raise KeyError(f"Unknown path key '{key}'. Define it in configs/config.yaml.")
        p = (self.root / rel).resolve()
        self._path_cache[key] = p
        return p

    def subreddit_list(self) -> list[dict[str, Any]]:
        return list(self.subreddits.get("subreddits", []))

    def subreddit_names(self) -> list[str]:
        return [s["name"] for s in self.subreddit_list()]

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested getter into the main config, e.g. settings.get('reddit', 'listing')."""
        node: Any = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def strat(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.strategy_params
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    root = repo_root()
    if load_dotenv is not None:
        load_dotenv(root / ".env")
    configs = root / "configs"

    creds = Credentials(
        reddit_client_id=os.environ.get("REDDIT_CLIENT_ID") or None,
        reddit_client_secret=os.environ.get("REDDIT_CLIENT_SECRET") or None,
        reddit_user_agent=os.environ.get("REDDIT_USER_AGENT") or None,
        fmp_api_key=os.environ.get("FMP_API_KEY") or None,
    )
    force_mock = os.environ.get("RHAL_FORCE_MOCK", "0") not in {"0", "", "false", "False"}

    settings = Settings(
        root=root,
        config=_load_yaml(configs / "config.yaml"),
        subreddits=_load_yaml(configs / "subreddits.yaml"),
        ticker_filters=_load_yaml(configs / "ticker_filters.yaml"),
        strategy_params=_load_yaml(configs / "strategy_params.yaml"),
        credentials=creds,
        force_mock=force_mock,
    )
    log.debug(
        "Loaded settings | reddit_mode=%s fmp_mode=%s root=%s",
        settings.reddit_mode,
        settings.fmp_mode,
        settings.root,
    )
    return settings


def credential_report() -> str:
    """Human-readable summary of which keys are present — surfaced by scripts."""
    s = load_settings()
    c = s.credentials
    lines = [
        "Credential status:",
        f"  REDDIT_CLIENT_ID     : {'set' if c.reddit_client_id else 'MISSING'}",
        f"  REDDIT_CLIENT_SECRET : {'set' if c.reddit_client_secret else 'MISSING'}",
        f"  REDDIT_USER_AGENT    : {'set' if c.reddit_user_agent else 'MISSING'}",
        f"  FMP_API_KEY          : {'set' if c.fmp_api_key else 'MISSING'}",
        f"  -> Reddit runs in    : {s.reddit_mode.upper()} mode",
        f"  -> FMP runs in       : {s.fmp_mode.upper()} mode",
    ]
    if s.reddit_mode == "mock" or s.fmp_mode == "mock":
        lines.append(
            "  NOTE: MOCK mode uses SYNTHETIC development data. Never treat mock"
            " outputs as research results. See README + .env.example."
        )
    return "\n".join(lines)
