"""
Compare ONE pooled model (t as a feature) vs THREE per-t models, same temporal split + features.
Pooled shares data across t but forces shared feature effects; per-t fits t-specific effects but
on 1/3 the data. Which wins on the held-out test?
"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "/Users/yonierlich/repos/challenge_1")
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate, make_model

R = Path("/Users/yonierlich/repos/challenge_1")
subs, events = load_clean(R / "data")
panel = build_panel(subs, events)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)
y_test = test.label.values


def p_at_20(y, s):
    n = max(1, int(np.ceil(0.2 * len(y))))
    return float(np.mean(y[np.argsort(-s, kind="stable")[:n]]))


print(
    "train rows/pos per t:",
    {
        t: (int((train.t == t).sum()), int(train.label[train.t == t].sum()))
        for t in (0, 7, 30)
    },
)
print(
    "test  rows/pos per t:",
    {
        t: (int((test.t == t).sum()), int(test.label[test.t == t].sum()))
        for t in (0, 7, 30)
    },
    "\n",
)

# pooled (t as feature)
pooled = make_model().fit(train[MODEL_FEATURES], train.label)
pooled_pred = pooled.predict_proba(test[MODEL_FEATURES])[:, 1]

# per-t (drop t; train one model per t)
feats_t = [f for f in MODEL_FEATURES if f != "t"]
per_t_pred = np.zeros(len(test))
for t in (0, 7, 30):
    tr_m, te_m = (train.t == t).values, (test.t == t).values
    m = make_model().fit(train[tr_m][feats_t], train.label[tr_m])
    per_t_pred[te_m] = m.predict_proba(test[te_m][feats_t])[:, 1]

print(f"{'':6} {'pooled':>16} {'per-t models':>16}")
print(f"{'':6} {'AUC   p@20':>16} {'AUC   p@20':>16}")
for t in (0, 7, 30):
    m = (test.t == t).values
    y = y_test[m]
    if 0 < y.sum() < len(y):
        a_p, a_t = roc_auc_score(y, pooled_pred[m]), roc_auc_score(y, per_t_pred[m])
        p_p, p_t = p_at_20(y, pooled_pred[m]), p_at_20(y, per_t_pred[m])
        print(f"t={t:<4} {a_p:>6.3f} {p_p:>6.3f}   {a_t:>6.3f} {p_t:>6.3f}")
print(
    f"\noverall AUC:  pooled {roc_auc_score(y_test, pooled_pred):.3f} | per-t {roc_auc_score(y_test, per_t_pred):.3f}"
)
