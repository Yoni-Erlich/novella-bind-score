"""
Experiment: does DISCRETIZING outbound volume into quantile bins add signal vs the continuous log1p?
Leakage-safe: quantile bin edges are fit on TRAIN only, then applied to test.
Compare on held-out temporal test (overall + per t):
  - BASE                       : current 5 feats (uses continuous outbound_chars_log)
  - REPLACE log → ordinal qbin : quartiles / quintiles / deciles (0..k-1)
  - REPLACE log → one-hot q4   : lets the model fit an arbitrary per-bin shape (the real test of discretization)
  - ADD ordinal q4 on top of log
"""

import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
import numpy as np
from sklearn.metrics import roc_auc_score
from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate, make_model

subs, events = load_clean(R / "data")
panel = build_panel(subs, events)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)
y = test.label.values
ts = (0, 7, 30)
LOCAL = [
    "agent_bind_rate",
    "has_quote_by_t",
    "n_inbound_by_t",
    "t",
]  # base minus outbound
COL = "outbound_chars_by_t"


def interior_edges(q):
    e = np.quantile(train[COL].values, np.linspace(0, 1, q + 1))
    return np.unique(e)[1:-1]  # train-fit cut points (no leakage)


def qbin(q):
    e = interior_edges(q)
    return (
        np.digitize(train[COL].values, e),
        np.digitize(test[COL].values, e),
        len(e) + 1,
    )


def auc(trX, teX):
    m = make_model().fit(trX, train.label)
    p = m.predict_proba(teX)[:, 1]
    out = {"overall": roc_auc_score(y, p)}
    for t in ts:
        mk = (test.t == t).values
        out[t] = roc_auc_score(y[mk], p[mk])
    return out


def row(name, out):
    base = REF["overall"]
    d = out["overall"] - base
    flag = "  <-- helps" if d > 0.003 else ("" if d > -0.003 else "  (worse)")
    print(
        f"{name:34s} {out['overall']:>7.3f} {out[0]:>7.3f} {out[7]:>7.3f} {out[30]:>7.3f}  Δ={d:+.3f}{flag}"
    )


REF = auc(train[MODEL_FEATURES], test[MODEL_FEATURES])
print(f"{'feature set':34s} {'overall':>7} {'t=0':>7} {'t=7':>7} {'t=30':>7}")
row("BASE (continuous log1p)", REF)

for q, label in [(4, "quartile"), (5, "quintile"), (10, "decile")]:
    trb, teb, k = qbin(q)
    trX = train[LOCAL].assign(ob_bin=trb)
    teX = test[LOCAL].assign(ob_bin=teb)
    row(f"REPLACE log → ordinal {label} (k={k})", auc(trX, teX))

# one-hot quartiles (drop bin 0 as reference) — lets the model fit any per-bin pattern
trb, teb, k = qbin(4)
trX, teX = train[LOCAL].copy(), test[LOCAL].copy()
for b in range(1, k):
    trX[f"ob_q{b}"] = (trb == b).astype(int)
    teX[f"ob_q{b}"] = (teb == b).astype(int)
row(f"REPLACE log → one-hot quartiles (k={k})", auc(trX, teX))

# add ordinal quartile ON TOP of the continuous log
trb, teb, k = qbin(4)
row(
    "ADD ordinal quartile on top of log",
    auc(
        train[MODEL_FEATURES].assign(ob_bin=trb),
        test[MODEL_FEATURES].assign(ob_bin=teb),
    ),
)

# univariate: discretized vs continuous, alone
trb, teb, _ = qbin(4)
print(
    f"\nunivariate test-AUC alone:  outbound_chars_log {roc_auc_score(y, test.outbound_chars_log.values):.3f}"
    f"  |  outbound quartile-bin {roc_auc_score(y, teb):.3f}"
)

# ROBUSTNESS: does binning also win grouped CV-within-train, or only this one holdout split?
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)


def cvauc(df):
    return cross_val_score(
        make_model(),
        df,
        train.label,
        cv=cv,
        groups=train.submissionId,
        scoring="roc_auc",
    ).mean()


trb5, _, _ = qbin(5)
print("\ngrouped CV-within-train (5-fold) — the decision check:")
print(f"  BASE (continuous log) {cvauc(train[MODEL_FEATURES]):.3f}")
print(f"  REPLACE → quantile    {cvauc(train[LOCAL].assign(ob_bin=trb5)):.3f}")
