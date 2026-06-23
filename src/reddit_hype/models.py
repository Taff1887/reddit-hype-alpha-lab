"""Composite hype scores + ML models.

Part 1 — the seven component scores and the configurable
``final_hype_alpha_score``. Each component is a bounded, auditable transform of
the raw features from :mod:`hype_features`; the final score is a weighted sum
whose weights live in ``configs/strategy_params.yaml`` (no magic numbers).

Part 2 — walk-forward ML (logistic regression, random forest, optional
LightGBM) predicting the forward return / its sign. Validation is strictly
time-ordered: never a random row split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Settings, load_settings
from .utils import get_logger, safe_div

log = get_logger(__name__)


# ============================================================ component scores
def _sigmoid(x: np.ndarray | pd.Series) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def _rank01(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    return r.fillna(0.5)


def compute_scores(
    panel: pd.DataFrame, settings: Settings | None = None, weights: dict | None = None
) -> pd.DataFrame:
    """Add the seven component scores + final_hype_alpha_score to the panel.

    Component scores are computed within each date (cross-sectional) where rank
    normalisation is used, so they answer "relative to the other names getting
    attention today". ``weights`` overrides ``score_weights`` from config (used by
    the research loop to test re-weightings).
    """
    settings = settings or load_settings()
    df = panel.copy()
    if df.empty:
        return df

    sc = settings.strategy_params.get("scoring", {})
    underreac_cap = float(sc.get("underreaction_return_cap", 0.05))
    already_ran_thr = float(sc.get("already_ran_return_threshold", 0.15))
    min_authors = int(sc.get("min_unique_authors", 3))
    min_ddv = float(settings.strat("tradability", "min_dollar_volume", default=1e6))
    min_mcap = float(settings.strat("tradability", "min_market_cap", default=1e8))

    df["abs_ret_5d"] = df["price_return_5d"].abs()

    # 1) Attention --------------------------------------------------------
    df["attention_zscore"] = df["zscore_vs_30d_baseline"].clip(-3, 3).fillna(0.0)
    df["attention_score"] = df.groupby("date")["weighted_attention"].transform(_rank01)

    # 2) Hype velocity ----------------------------------------------------
    accel = df["acceleration_24h_vs_7d"].clip(-1, 5).fillna(0.0)
    chg = df["change_vs_7d_avg"].clip(-1, 5).fillna(0.0)
    df["hype_velocity_score"] = np.tanh(0.5 * accel + 0.5 * chg)

    # 3) Sentiment / conviction ------------------------------------------
    df["net_bullish_sentiment"] = (df["bullish_intensity"] - df["bearish_intensity"]).clip(-1, 1)
    if "net_sentiment" in df:
        df["net_bullish_sentiment"] = (
            0.5 * df["net_bullish_sentiment"] + 0.5 * df["net_sentiment"].fillna(0.0)
        ).clip(-1, 1)
    df["sentiment_score"] = ((df["net_bullish_sentiment"] + 1) / 2).clip(0, 1)
    # conviction blends stated commitment language with breadth of unique authors
    author_breadth = np.tanh((df["unique_authors"].fillna(0) - min_authors) / 5.0).clip(-1, 1)
    df["conviction_language"] = df["conviction_language_score"].fillna(0.0)
    df["conviction_score"] = (
        0.6 * ((df["conviction_language"] + 1) / 2) + 0.4 * ((author_breadth + 1) / 2)
    ).clip(0, 1)

    # 4) Quality DD -------------------------------------------------------
    fin = _rank01(df["avg_financial_terms"].fillna(0))
    wlen = _rank01(df["avg_word_len"].fillna(0))
    dd_flair = _rank01(df["dd_flair_count"].fillna(0))
    quality = 0.45 * fin + 0.25 * wlen + 0.30 * dd_flair
    quality = quality - 0.5 * df["low_effort_frac"].fillna(0) - 0.5 * df["pumpy_frac"].fillna(0)
    df["quality_dd_score"] = quality.clip(0, 1)

    # 5) Underreaction (attention up, price hasn't moved) -----------------
    attn_pos = df["attention_zscore"].clip(lower=0) / 3.0
    price_quiet = (1.0 - (df["abs_ret_5d"].fillna(0) / underreac_cap)).clip(0, 1)
    df["underreaction_score"] = (attn_pos * price_quiet).clip(0, 1)

    # 6) Tradability ------------------------------------------------------
    ddv = df["dollar_volume_20d"].fillna(0)
    mcap = df["market_cap"].fillna(0)
    liq = _sigmoid(np.log10(np.maximum(ddv, 1) / max(min_ddv, 1)))
    capz = _sigmoid(np.log10(np.maximum(mcap, 1) / max(min_mcap, 1)))
    df["tradability_score"] = (0.6 * liq + 0.4 * capz).clip(0, 1)
    df["meets_liquidity"] = (ddv >= min_ddv) & (mcap >= min_mcap)

    # 7) Pump risk --------------------------------------------------------
    pump = (
        0.30 * _rank01(df["pump_language_score"].fillna(0))
        + 0.25 * df["bot_spam_mean"].fillna(0)
        + 0.20 * df["low_effort_frac"].fillna(0)
        + 0.15 * _rank01(df["spam_phrase_total"].fillna(0))
        + 0.10 * _rank01(df["hype_language_score"].fillna(0))
    )
    df["pump_risk_score"] = pump.clip(0, 1)

    # normalised cross-subreddit breadth (raw count kept for display/strategy)
    df["breadth_score"] = np.tanh((df["cross_subreddit_breadth"].fillna(0) - 1) / 3.0)

    # 8) Squeeze setup — the GME-style mechanism: high short interest + high
    # days-to-cover + accelerating retail attention + conviction. Degrades to 0
    # (and has_short_data=False) when no short-interest data is available.
    si = df.get("short_pct_float")
    if si is None:
        si = pd.Series(np.nan, index=df.index)
    dtc = df.get("days_to_cover", pd.Series(np.nan, index=df.index))
    df["has_short_data"] = si.notna()
    si_comp = _sigmoid((si.fillna(0) - 12.0) / 6.0)            # ramps past ~12% SI
    dtc_comp = (dtc.fillna(0) / 7.0).clip(0, 1)
    accel_pos = df["hype_velocity_score"].clip(lower=0)
    df["squeeze_setup_score"] = np.where(
        df["has_short_data"],
        (0.40 * si_comp + 0.20 * dtc_comp + 0.25 * accel_pos
         + 0.15 * df["conviction_score"].fillna(0)).clip(0, 1),
        0.0,
    )

    # already-ran penalty (price already ran before the hype) -------------
    df["already_ran_penalty"] = (
        (df["price_return_5d"].fillna(0) / already_ran_thr).clip(0, 1)
    )

    # ---- final weighted composite --------------------------------------
    weights = weights if weights is not None else settings.strategy_params.get("score_weights", {})
    df["final_hype_alpha_score"] = 0.0
    used = []
    for key, w in weights.items():
        if key in df.columns:
            df["final_hype_alpha_score"] = df["final_hype_alpha_score"] + float(w) * df[key].fillna(0.0)
            used.append(key)
        else:
            log.warning("score_weights key '%s' has no matching column — skipped", key)
    # cross-sectional percentile of the composite, for display/ranking
    df["final_score_pct"] = df.groupby("date")["final_hype_alpha_score"].transform(_rank01)
    log.info("Computed scores using weighted components: %s", used)
    return df


# ===================================================================== ML part
LEAKAGE_PREFIXES = ("fwd_ret", "label", "fwd_", "future_")
ID_COLS = {"ticker", "date", "company_name", "exchange", "region", "sector",
           "industry", "liquidity_bucket", "top_subreddits", "synthetic",
           "any_synthetic", "entry_date", "asof_date"}


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Numeric, non-leaking feature columns for ML."""
    cols = []
    for c in panel.columns:
        if c in ID_COLS or c.startswith(LEAKAGE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(panel[c]) and not pd.api.types.is_bool_dtype(panel[c]):
            cols.append(c)
    return cols


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame                       # date,ticker,y_true,pred_*,proba_*
    fold_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    features_used: list[str] = field(default_factory=list)


def _make_models(settings: Settings):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rs = int(settings.strat("ml", "random_state", default=1887))
    requested = settings.strat("ml", "models", default=["logreg", "random_forest"])
    models = {}
    if "logreg" in requested:
        models["logreg"] = Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=rs))]
        )
    if "random_forest" in requested:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=rs, n_jobs=-1
        )
    if "lightgbm" in requested:
        try:
            from lightgbm import LGBMClassifier

            models["lightgbm"] = LGBMClassifier(
                n_estimators=400, max_depth=-1, num_leaves=31, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=rs, verbose=-1,
            )
        except Exception:
            log.warning("lightgbm requested but not installed — skipping.")
    return models


def walk_forward(
    panel: pd.DataFrame, settings: Settings | None = None, target: str | None = None
) -> WalkForwardResult:
    """Time-ordered walk-forward classification of P(positive forward return)."""
    settings = settings or load_settings()
    target = target or settings.strat("ml", "target", default="fwd_ret_5d")
    thr = float(settings.strat("ml", "classification_target_threshold", default=0.0))
    wf = settings.strategy_params.get("ml", {}).get("walk_forward", {})
    train_days = int(wf.get("train_days", 252))
    test_days = int(wf.get("test_days", 21))
    step_days = int(wf.get("step_days", 21))
    min_rows = int(wf.get("min_train_rows", 200))

    if target not in panel.columns:
        raise KeyError(f"Target '{target}' not in panel — run labels first.")

    data = panel.dropna(subset=[target]).copy()
    feats = feature_columns(data)
    data = data.dropna(subset=feats, how="all")
    data["y"] = (data[target] > thr).astype(int)
    data = data.sort_values("date")
    dates = np.array(sorted(data["date"].unique()))
    if len(dates) < (train_days + test_days) // 5:
        log.warning("Not enough distinct dates (%d) for robust walk-forward.", len(dates))

    models_proto = _make_models(settings)
    preds, fold_rows, importances = [], [], []

    start = 0
    fold = 0
    while True:
        train_end = start + train_days
        test_end = train_end + test_days
        if train_end >= len(dates) or test_end > len(dates) + test_days:
            break
        train_dates = set(dates[start:train_end])
        test_dates = set(dates[train_end:min(test_end, len(dates))])
        if not test_dates:
            break
        tr = data[data["date"].isin(train_dates)]
        te = data[data["date"].isin(test_dates)]
        start += step_days
        if len(tr) < min_rows or te.empty or tr["y"].nunique() < 2:
            continue
        fold += 1
        Xtr = tr[feats].fillna(0.0).values
        ytr = tr["y"].values
        Xte = te[feats].fillna(0.0).values

        fold_pred = te[["date", "ticker", target, "y"]].copy()
        for name, proto in models_proto.items():
            from sklearn.base import clone

            mdl = clone(proto)
            mdl.fit(Xtr, ytr)
            proba = mdl.predict_proba(Xte)[:, 1]
            fold_pred[f"proba_{name}"] = proba
            # feature importance per fold
            imp = _importance(mdl, feats)
            if imp is not None:
                importances.append(imp.assign(model=name, fold=fold))
            # fold metric
            from sklearn.metrics import roc_auc_score

            try:
                auc = roc_auc_score(te["y"], proba) if te["y"].nunique() > 1 else np.nan
            except Exception:
                auc = np.nan
            fold_rows.append(
                {"fold": fold, "model": name, "n_train": len(tr), "n_test": len(te),
                 "test_start": min(test_dates), "test_end": max(test_dates), "auc": auc,
                 "base_rate": te["y"].mean()}
            )
        preds.append(fold_pred)

    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    fold_metrics = pd.DataFrame(fold_rows)
    fi = (
        pd.concat(importances, ignore_index=True).groupby("feature")["importance"].mean()
        .sort_values(ascending=False).reset_index()
        if importances else pd.DataFrame(columns=["feature", "importance"])
    )
    log.info("Walk-forward: %d folds, %d OOS predictions across %d models",
             fold, len(predictions), len(models_proto))
    return WalkForwardResult(predictions, fold_metrics, fi, feats)


def _importance(model, feats: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": feats, "importance": model.feature_importances_})
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            return pd.DataFrame({"feature": feats, "importance": np.abs(clf.coef_[0])})
    if hasattr(model, "coef_"):
        return pd.DataFrame({"feature": feats, "importance": np.abs(model.coef_[0])})
    return None
