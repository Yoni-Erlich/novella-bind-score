"""
Feature significance via XGBoost + SHAP (Task 2).

Why: correlation only sees linear/monotone marginal signal. A gradient-boosted tree
captures nonlinearities + interactions; out-of-fold SHAP gives a consistent, signed
attribution that ranks features by their actual contribution to the model.

Guardrails:
  - leakage-safe per-t features (only events with d <= t); never resolvedDate/label
  - open-at-t censoring (resolvedDate > createdDate + t)
  - group-aware CV by submissionId (a sub's 3 rows never split across folds)
  - SHAP computed OUT-OF-FOLD (generalization, not memorization)
  - XGB shallow + regularized + scale_pos_weight (small, imbalanced data)
  - compared against a logistic baseline; if XGB doesn't beat it, there's little
    nonlinear/interaction signal to find.
We run TWO feature sets: ALL candidates, and LEGIT (drop calendar/noise) — to show the
high-cardinality SHAP-overfit artifact in ALL, and a trustworthy ranking in LEGIT.
Caveat: with collinear volume features SHAP splits credit among the correlated group.
"""

import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/Users/yonierlich/repos/challenge_1")
FIG = REPO / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

s = pd.read_csv(
    REPO / "data/features_submissions.csv", parse_dates=["createdDate", "resolvedDate"]
)
e = pd.read_csv(REPO / "data/features_events.csv", parse_dates=["event_date"])
e = e.merge(s[["submissionId", "createdDate"]], on="submissionId")
e["d"] = (e.event_date - e.createdDate).dt.total_seconds() / 86400
s["resolution_days"] = (s.resolvedDate - s.createdDate).dt.total_seconds() / 86400


def features_at_t(t):
    vis = e[e.d <= t]
    g = vis.groupby("submissionId")
    out = vis[vis.event_type == "EMAIL_OUTBOUND"].groupby("submissionId")
    inb = vis[vis.event_type == "EMAIL_INBOUND"].groupby("submissionId")
    df = pd.DataFrame(index=s.submissionId)
    df["n_outbound"] = out.size()
    df["outbound_chars"] = out.email_char_count.sum()
    df["n_inbound"] = inb.size()
    df["inbound_chars"] = inb.email_char_count.sum()
    df["n_quote"] = (
        vis[vis.event_type == "QUOTE_RECEIVED"].groupby("submissionId").size()
    )
    df["has_quote"] = (df["n_quote"].fillna(0) > 0).astype(int)
    df["recency"] = t - g.d.max()
    df["events_last7"] = vis[vis.d > t - 7].groupby("submissionId").size()
    df["outbound_last7"] = (
        vis[(vis.d > t - 7) & (vis.event_type == "EMAIL_OUTBOUND")]
        .groupby("submissionId")
        .size()
    )
    rec = vis[vis.d > t - 7].groupby("submissionId").size()
    pri = vis[(vis.d > t - 14) & (vis.d <= t - 7)].groupby("submissionId").size()
    df["accel"] = (rec / pri).replace([np.inf, -np.inf], np.nan)
    ow = vis[vis.event_type == "EMAIL_OUTBOUND"].copy()
    ow["wknd"] = ow.event_date.dt.weekday >= 5
    df["weekend_out_share"] = ow.groupby("submissionId").wknd.mean()
    df["created_dow"] = s.set_index("submissionId").createdDate.dt.weekday
    df["is_weekend"] = (df["created_dow"] >= 5).astype(int)
    df["created_month"] = s.set_index("submissionId").createdDate.dt.month
    qstart = s.set_index("submissionId").createdDate.dt.to_period("Q").dt.start_time
    df["day_in_quarter"] = (s.set_index("submissionId").createdDate - qstart).dt.days
    df["t"] = t
    df = df.join(s.set_index("submissionId")[["label", "resolution_days"]])
    df["submissionId"] = df.index
    for c in [
        "n_outbound",
        "outbound_chars",
        "n_inbound",
        "inbound_chars",
        "n_quote",
        "events_last7",
        "outbound_last7",
    ]:
        df[c] = df[c].fillna(0)
    return df[df.resolution_days > t]


P = pd.concat([features_at_t(t) for t in (0, 7, 30)], ignore_index=True)
y, grp, tcol = P.label.values, P.submissionId.values, P.t.values
print(
    f"panel: {len(P)} rows | positives {int(y.sum())} ({y.mean():.1%}) | "
    f"t0={int((tcol==0).sum())} t7={int((tcol==7).sum())} t30={int((tcol==30).sum())}\n"
)

FEATURES_ALL = [
    "n_outbound",
    "outbound_chars",
    "n_inbound",
    "inbound_chars",
    "n_quote",
    "has_quote",
    "recency",
    "events_last7",
    "outbound_last7",
    "accel",
    "weekend_out_share",
    "created_dow",
    "is_weekend",
    "created_month",
    "day_in_quarter",
    "t",
]
FEATURES_LEGIT = [
    "n_outbound",
    "outbound_chars",
    "n_inbound",
    "inbound_chars",
    "n_quote",
    "has_quote",
    "recency",
    "t",
]


def run(features, tag):
    X = P[features]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    oof_xgb = np.zeros(len(P))
    oof_log = np.zeros(len(P))
    oof_shap = np.zeros((len(P), len(features)))
    for tr, va in cv.split(X, y, groups=grp):
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=150,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=5,
            scale_pos_weight=spw,
            eval_metric="auc",
            random_state=0,
            n_jobs=4,
        )
        m.fit(X.iloc[tr], y[tr])
        oof_xgb[va] = m.predict_proba(X.iloc[va])[:, 1]
        oof_shap[va] = shap.TreeExplainer(m).shap_values(X.iloc[va])
        lr = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        lr.fit(X.iloc[tr], y[tr])
        oof_log[va] = lr.predict_proba(X.iloc[va])[:, 1]
    print(f"===== [{tag}]  {len(features)} features =====")
    print(
        f"  OOF AUC: XGBoost {roc_auc_score(y, oof_xgb):.3f} | logistic {roc_auc_score(y, oof_log):.3f}"
    )
    for t in (0, 7, 30):
        mk = tcol == t
        if mk.sum() and 0 < y[mk].sum() < mk.sum():
            print(
                f"     t={t:2d}: XGB {roc_auc_score(y[mk],oof_xgb[mk]):.3f} | LOG {roc_auc_score(y[mk],oof_log[mk]):.3f}"
            )
    shap_imp = pd.Series(np.abs(oof_shap).mean(0), index=features).sort_values(
        ascending=False
    )
    corr_imp = pd.Series(
        {c: abs(np.corrcoef(P[c].fillna(P[c].median()), y)[0, 1]) for c in features}
    )
    rank = pd.DataFrame(
        {
            "mean_abs_shap": shap_imp,
            "shap_rank": shap_imp.rank(ascending=False).astype(int),
            "abs_corr": corr_imp,
            "corr_rank": corr_imp.rank(ascending=False).astype(int),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    print(rank.round(4).to_string(), "\n")
    shap.summary_plot(
        oof_shap, X, feature_names=features, show=False, max_display=len(features)
    )
    plt.tight_layout()
    plt.savefig(FIG / f"shap_beeswarm_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()
    rank.round(4).to_csv(REPO / "reports" / f"feature_significance_shap_{tag}.csv")
    return rank


run(FEATURES_ALL, "all")
run(FEATURES_LEGIT, "legit")
print("saved beeswarm + csv for both feature sets under reports/")
