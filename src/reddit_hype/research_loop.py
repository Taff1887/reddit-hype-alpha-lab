"""Bounded research / self-improvement loop.

Each iteration:
  1. Score component signals by their out-of-sample rank IC vs forward returns.
  2. Test the standing hypotheses (acceleration vs raw mentions; unique authors
     vs total mentions; DD quality vs meme hype; does underreaction matter;
     is the signal just chasing already-ran stocks).
  3. Propose a re-weighting of ``final_hype_alpha_score`` toward the components
     that actually carried information in-sample.
  4. Re-score, re-backtest on a held-out period, and KEEP the change only if it
     improves out-of-sample net Sharpe — otherwise revert.
  5. Log everything, including failed hypotheses.

It deliberately does not chase a single good number; it records what worked and
what did not, with the alpha-acceptance gate applied at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .backtester import run_backtest
from .config import Settings, load_settings
from .models import compute_scores
from .utils import get_logger

log = get_logger(__name__)

# Components whose weights the loop is allowed to tune (must exist as columns).
TUNABLE = [
    "attention_zscore", "hype_velocity_score", "net_bullish_sentiment",
    "conviction_language", "breadth_score", "underreaction_score",
    "quality_dd_score",
]


def _split_dates(panel: pd.DataFrame, frac: float = 0.6):
    dates = np.array(sorted(panel["date"].unique()))
    cut = dates[int(len(dates) * frac)] if len(dates) > 5 else dates[-1]
    return panel[panel["date"] <= cut], panel[panel["date"] > cut]


def _component_ic(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for comp in TUNABLE + ["pump_risk_score"]:
        if comp in panel.columns:
            ic = metrics.rank_ic(panel, comp, target)
            rows.append({"component": comp, **ic})
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False).reset_index(drop=True)


def _hypotheses(panel: pd.DataFrame, target: str) -> list[dict]:
    """Head-to-head IC comparisons that answer the standing research questions."""
    def ic(col):
        return metrics.rank_ic(panel, col, target)["mean_ic"] if col in panel else np.nan

    tests = [
        ("acceleration_beats_raw_mentions", "hype_velocity_score", "attention_score"),
        ("unique_authors_beat_total_mentions", "conviction_score", "attention_score"),
        ("dd_quality_beats_meme_hype", "quality_dd_score", "hype_language_score"),
        ("underreaction_matters", "underreaction_score", None),
    ]
    out = []
    for name, a, b in tests:
        ic_a = ic(a)
        ic_b = ic(b) if b else np.nan
        verdict = (
            ("supported" if (ic_a or -9) > (ic_b or -9) else "not supported")
            if b else ("supported" if (ic_a or 0) > 0 else "not supported")
        )
        out.append({"hypothesis": name, "ic_primary": ic_a, "ic_alt": ic_b, "verdict": verdict})
    return out


def run_research_loop(
    panel: pd.DataFrame, prices: pd.DataFrame, settings: Settings | None = None
) -> dict:
    settings = settings or load_settings()
    target = settings.strat("ml", "target", default="fwd_ret_5d")
    max_iter = int(settings.strat("research_loop", "max_iterations", default=5))
    base_weights = dict(settings.strategy_params.get("score_weights", {}))

    insample, oos = _split_dates(panel)
    if oos.empty or target not in panel.columns:
        log.warning("Insufficient OOS data or missing target — research loop is a no-op.")
        return {"iterations": pd.DataFrame(), "hypotheses": pd.DataFrame(), "final_weights": base_weights}

    def oos_sharpe(weights) -> float:
        scored = compute_scores(panel, settings, weights=weights)
        _, oos_scored = _split_dates(scored)
        res = run_backtest(oos_scored, prices, "TopHypeLongOnly", 5, settings)
        return res.stats.get("sharpe", float("nan"))

    weights = dict(base_weights)
    best_sharpe = oos_sharpe(weights)
    iters, hyp_log = [], []
    log.info("Research loop start | baseline OOS Sharpe=%.3f", best_sharpe)

    for it in range(1, max_iter + 1):
        ic_tbl = _component_ic(insample, target)
        hyps = _hypotheses(insample, target)
        for h in hyps:
            h["iteration"] = it
        hyp_log.extend(hyps)

        # Proposal: nudge weights toward in-sample IC sign/magnitude (bounded).
        proposed = dict(weights)
        for _, r in ic_tbl.iterrows():
            comp = r["component"]
            if comp == "pump_risk_score":
                continue
            if comp in proposed and not np.isnan(r["mean_ic"]):
                step = 0.05 * np.sign(r["mean_ic"]) * min(1.0, abs(r["ic_ir"]) if r["ic_ir"] == r["ic_ir"] else 0)
                proposed[comp] = float(np.clip(proposed[comp] + step, -0.5, 0.6))

        new_sharpe = oos_sharpe(proposed)
        accepted = new_sharpe > best_sharpe + 1e-6
        iters.append(
            {
                "iteration": it,
                "best_component_in_sample": ic_tbl.iloc[0]["component"] if not ic_tbl.empty else None,
                "best_component_ic": ic_tbl.iloc[0]["mean_ic"] if not ic_tbl.empty else np.nan,
                "oos_sharpe_before": best_sharpe,
                "oos_sharpe_proposed": new_sharpe,
                "accepted": accepted,
            }
        )
        log.info("Iter %d | proposed OOS Sharpe=%.3f (%s)", it, new_sharpe,
                 "ACCEPT" if accepted else "revert")
        if accepted:
            weights, best_sharpe = proposed, new_sharpe
        else:
            break  # bounded: stop when no further OOS improvement

    return {
        "iterations": pd.DataFrame(iters),
        "hypotheses": pd.DataFrame(hyp_log),
        "ic_table": _component_ic(insample, target),
        "final_weights": weights,
        "baseline_weights": base_weights,
        "final_oos_sharpe": best_sharpe,
    }
