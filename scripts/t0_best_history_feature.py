"""
Give the customer-history idea its best shot: take the ONE new feature with real signal
(cust_hist_avg_outbound, univ AUC 0.713 on returning) and test it in the PRODUCTION POOLED model
(the architecture that actually works), not the losing dedicated-t=0 model.

Compare on the held-out temporal test, overall and at t=0:
  pooled base       = MODEL_FEATURES
  pooled + avg_out  = MODEL_FEATURES + cust_hist_avg_outbound
"""

import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate, make_model

subs, events = load_clean(R / "data")

# cust_hist_avg_outbound = avg #OUTBOUND emails per the customer's prior-RESOLVED submission
out = events[events.event_type == "EMAIL_OUTBOUND"]
n_out = out.groupby("submissionId").size().reindex(subs.submissionId).fillna(0.0)
S = subs.assign(n_out=n_out.values)
rows = []
for _, grp in S.groupby("agentEmail"):
    for _, r in grp.iterrows():
        prior = grp[grp.resolvedDate < r.createdDate]
        rows.append((r.submissionId, prior.n_out.mean() if len(prior) else np.nan))
hist = pd.DataFrame(rows, columns=["submissionId", "cust_hist_avg_outbound"])

panel = build_panel(subs, events)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)
train = train.merge(hist, on="submissionId")
test = test.merge(hist, on="submissionId")
y = test.label.values
t0 = (test.t == 0).values


def auc(mask, p):
    yy = y[mask]
    return roc_auc_score(yy, p[mask]) if 0 < yy.sum() < len(yy) else float("nan")


print(f"{'feature set':22s} {'overall':>8} {'t=0 only':>9}")
for name, cols in (
    ("pooled base", MODEL_FEATURES),
    ("pooled + avg_outbound", MODEL_FEATURES + ["cust_hist_avg_outbound"]),
):
    m = make_model().fit(train[cols], train.label)
    p = m.predict_proba(test[cols])[:, 1]
    print(f"{name:22s} {auc(np.ones(len(y), bool), p):>8.3f} {auc(t0, p):>9.3f}")
