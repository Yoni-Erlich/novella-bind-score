"""
Bind-score model: pooled regularized logistic regression over the (submission, t) panel.

- Temporal split (train = earlier createdDate, test = later) — fair for repeat-client history.
- agent_bind_rate is recomputed with the TRAIN base rate as the smoothing prior (strict leakage-clean).
- CV-within-train (grouped by submissionId) for feature-set / log1p selection.
- Exposes bind_score(submission_id, t).
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from src.features import (
    load_clean,
    build_panel,
    feature,
    smoothed_agent_rate,
    MODEL_FEATURES,
)

SPLIT_Q = 0.70  # 70% earliest-created submissions -> train


def temporal_split(panel: pd.DataFrame, q: float = SPLIT_Q):
    thr = panel.createdDate.quantile(q)
    return (
        panel[panel.createdDate <= thr].copy(),
        panel[panel.createdDate > thr].copy(),
        thr,
    )


def recompute_agent_rate(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Smooth agent_bind_rate toward the TRAIN-only base rate (no test leakage). Mutates in place."""
    g = float(train.drop_duplicates("submissionId").label.mean())
    for df in (train, test):
        df["agent_bind_rate"] = smoothed_agent_rate(
            df.agent_prior_binds, df.agent_prior_n, g
        )
    return g


def make_model(C: float = 1.0):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=C),
    )


def cv_auc(train: pd.DataFrame, feats: list[str]) -> float:
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    return cross_val_score(
        make_model(),
        train[feats],
        train.label,
        cv=cv,
        groups=train.submissionId,
        scoring="roc_auc",
    ).mean()


def select_features(train: pd.DataFrame) -> dict:
    """CV-within-train comparison: does agent_bind_rate help? does log1p(outbound_chars) help?"""
    train = train.copy()
    train["outbound_chars_log"] = np.log1p(train.outbound_chars_by_t)
    local = ["outbound_chars_by_t", "has_quote_by_t", "n_inbound_by_t", "t"]
    full = ["agent_bind_rate"] + local
    full_log = [
        "agent_bind_rate",
        "outbound_chars_log",
        "has_quote_by_t",
        "n_inbound_by_t",
        "t",
    ]
    return {
        "submission-local only": cv_auc(train, local),
        "+ agent_bind_rate (full)": cv_auc(train, full),
        "full, log1p(outbound_chars)": cv_auc(train, full_log),
    }


def train_bind_model(
    subs: pd.DataFrame, events: pd.DataFrame, q: float = SPLIT_Q
) -> dict:
    panel = build_panel(subs, events)
    train, test, thr = temporal_split(panel, q)
    g = recompute_agent_rate(train, test)
    model = make_model().fit(train[MODEL_FEATURES], train.label)
    return {
        "model": model,
        "train_g": g,
        "thr": thr,
        "panel": panel,
        "train": train,
        "test": test,
        "selection": select_features(train),
    }


def bind_score(
    submission_id: int, t: int, subs: pd.DataFrame, events: pd.DataFrame, fitted: dict
) -> float:
    """Score one (submission_id, t). Uses the train base rate for the agent prior. NaN if not open at t."""
    row = feature(submission_id, t, subs, events, global_rate=fitted["train_g"])
    if row[MODEL_FEATURES].isna().any():
        return float("nan")
    X = row[MODEL_FEATURES].to_frame().T
    return float(fitted["model"].predict_proba(X)[0, 1])


def standardized_coefficients(fitted: dict) -> pd.Series:
    """Task-2 ranking: |standardized logistic coefficient| per feature (contribution given the others)."""
    lr = fitted["model"].named_steps["logisticregression"]
    return pd.Series(np.abs(lr.coef_[0]), index=MODEL_FEATURES).sort_values(
        ascending=False
    )


if __name__ == "__main__":
    R = Path(__file__).resolve().parents[1]
    subs, events = load_clean(R / "data")
    fit = train_bind_model(subs, events)
    print(
        f"split: train<= {fit['thr'].date()} | train {len(fit['train'])} rows / test {len(fit['test'])} rows "
        f"| train base rate g={fit['train_g']:.3f}"
    )
    print("\nfeature selection (CV ROC-AUC on train):")
    for k, v in fit["selection"].items():
        print(f"  {k:32s} {v:.3f}")
    print("\nTask-2 ranking (|standardized coef|):")
    print(standardized_coefficients(fit).round(3).to_string())
    print(
        "\nbind_score(1, 0) =",
        round(bind_score(1, 0, subs, events, fit), 3),
        "| bind_score(1, 7) =",
        round(bind_score(1, 7, subs, events, fit), 3),
    )
