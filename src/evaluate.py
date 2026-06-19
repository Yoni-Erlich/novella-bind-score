"""
Evaluation for the bind-score: ranking metrics (ROC-AUC, PR-AUC) + prioritization metrics
(precision@top-20%, lift), reported PER t, on the held-out TEMPORAL test set.
Compares the logistic model against baselines (no-skill floor, single features, naive heuristic)
and XGBoost. Also produces the Task-2 feature ranking (univariate AUC + standardized coefficients).
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from src.features import load_clean, MODEL_FEATURES
from src.model import train_bind_model, standardized_coefficients

T_VALUES = (0, 7, 30)
K_FRAC = 0.20


def precision_at_k(y, s, k_frac=K_FRAC):
    n = max(1, int(np.ceil(k_frac * len(y))))
    top = np.argsort(-s, kind="stable")[:n]
    return float(np.mean(y[top]))


def per_t(test: pd.DataFrame, scores: np.ndarray) -> dict:
    res = {}
    for t in T_VALUES:
        m = (test.t == t).values
        y, s = test.label.values[m], scores[m]
        if 0 < y.sum() < len(y):
            base = y.mean()
            p = precision_at_k(y, s)
            res[t] = {
                "auc": roc_auc_score(y, s),
                "p@20": p,
                "lift": p / base,
                "base": base,
                "n": int(m.sum()),
            }
    return res


def overall(test, scores):
    return {
        "auc": roc_auc_score(test.label, scores),
        "pr_auc": average_precision_score(test.label, scores),
    }


def get_test_scores():
    """Train the model + baselines and return (fit, test, scores_dict) — no printing.
    `scores` maps model name -> per-test-row score array. Used by the notebook to plot comparisons."""
    R = Path(__file__).resolve().parents[1]
    subs, events = load_clean(R / "data")
    fit = train_bind_model(subs, events)
    train, test = fit["train"], fit["test"]
    scores = {}
    scores["logistic (final)"] = fit["model"].predict_proba(test[MODEL_FEATURES])[:, 1]
    scores["agent_bind_rate only"] = test.agent_bind_rate.values
    scores["outbound_chars only"] = test.outbound_chars_log.values
    scores["naive: quote>effort"] = (
        test.has_quote_by_t * 1e6 + test.outbound_chars_by_t
    ).values
    xgbm = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=150,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=5,
        scale_pos_weight=(train.label == 0).sum() / max((train.label == 1).sum(), 1),
        eval_metric="auc",
        random_state=0,
    )
    xgbm.fit(train[MODEL_FEATURES], train.label)
    scores["xgboost"] = xgbm.predict_proba(test[MODEL_FEATURES])[:, 1]
    return fit, test, scores


def run():
    fit, test, scores = get_test_scores()
    y_test = test.label.values

    print(
        f"TEST set: {len(test)} rows, {int(y_test.sum())} positive ({y_test.mean():.1%}) | "
        f"per t: {dict(test.t.value_counts().sort_index())}\n"
    )

    print("=== Ranking quality (overall, pooled test) ===")
    print(f"{'model':22s} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print(f"{'no-skill floor':22s} {0.5:>8.3f} {y_test.mean():>8.3f}")
    for name, s in scores.items():
        o = overall(test, s)
        print(f"{name:22s} {o['auc']:>8.3f} {o['pr_auc']:>8.3f}")

    print("\n=== Per-t ROC-AUC ===")
    print(f"{'model':22s} " + " ".join(f"t{t:>2}" for t in T_VALUES))
    for name, s in scores.items():
        pt = per_t(test, s)
        print(
            f"{name:22s} "
            + " ".join(f"{pt[t]['auc']:.3f}" if t in pt else "  -  " for t in T_VALUES)
        )

    print(
        "\n=== Prioritization: precision@top-20% (lift x) — logistic vs no-skill floor ==="
    )
    pt = per_t(test, scores["logistic (final)"])
    for t in T_VALUES:
        if t in pt:
            d = pt[t]
            print(
                f"  t={t:2d}: precision@20% {d['p@20']:.3f} (lift {d['lift']:.2f}x) vs floor {d['base']:.3f}  (n={d['n']})"
            )

    print("\n=== Task-2 feature ranking ===")
    uni = {f: roc_auc_score(test.label, test[f]) for f in MODEL_FEATURES}
    uni = pd.Series(uni).sort_values(ascending=False)
    coef = standardized_coefficients(fit)
    rank = pd.DataFrame({"univariate_test_AUC": uni, "abs_std_coef": coef}).sort_values(
        "univariate_test_AUC", ascending=False
    )
    print(rank.round(3).to_string())
    return fit, scores


if __name__ == "__main__":
    run()
