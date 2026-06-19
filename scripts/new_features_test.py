"""
Test the 3 panel survivors on top of the current 5-feature model (leakage-safe, as-of createdDate):
  is_cold_start_flag          = 1 if agent has no prior-resolved history
  agent_prior_quote_propensity= smoothed frac of agent's prior-resolved subs that EVER got a quote
  system_bind_rate_trailing   = smoothed bind-rate over ALL subs resolved before this createdDate
Incremental CV-AUC on train + held-out test AUC.
"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "/Users/yonierlich/repos/challenge_1")
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from src.features import load_clean, build_panel, smoothed_agent_rate, MODEL_FEATURES
from src.model import temporal_split, make_model

R = Path("/Users/yonierlich/repos/challenge_1")
subs, events = load_clean(R / "data")
subs["resolution_days"] = (
    subs.resolvedDate - subs.createdDate
).dt.total_seconds() / 86400
subs["quoted_ever"] = (
    subs.submissionId.map(
        events.groupby("submissionId").event_type.apply(
            lambda x: (x == "QUOTE_RECEIVED").any()
        )
    )
    .fillna(False)
    .astype(int)
)

# as-of createdDate raw counts (leakage-safe: only OTHER subs resolved strictly before)
rec = []
for _, r in subs.iterrows():
    prior_all = subs[
        (subs.resolvedDate < r.createdDate) & (subs.submissionId != r.submissionId)
    ]
    prior_ag = prior_all[prior_all.agentEmail == r.agentEmail]
    rec.append(
        (
            r.submissionId,
            len(prior_all),
            int(prior_all.label.sum()),
            len(prior_ag),
            int(prior_ag.quoted_ever.sum()),
        )
    )
A = pd.DataFrame(
    rec, columns=["submissionId", "sys_n", "sys_binds", "ag_n", "ag_quoted"]
)

panel = build_panel(subs, events).merge(A, on="submissionId")
train, test, _ = temporal_split(panel)

# smoothing priors from TRAIN only (leakage-clean)
uniq = train.drop_duplicates("submissionId")
g_bind = float(uniq.label.mean())
q0 = float(subs.set_index("submissionId").loc[uniq.submissionId, "quoted_ever"].mean())
ALPHA = 5.0
for df in (train, test):
    df["agent_bind_rate"] = smoothed_agent_rate(
        df.agent_prior_binds, df.agent_prior_n, g_bind
    )
    df["is_cold_start_flag"] = (df.agent_prior_n == 0).astype(int)
    df["agent_prior_quote_propensity"] = (df.ag_quoted + ALPHA * q0) / (df.ag_n + ALPHA)
    df["system_bind_rate_trailing"] = (df.sys_binds + ALPHA * g_bind) / (
        df.sys_n + ALPHA
    )

cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)


def cvauc(feats):
    return cross_val_score(
        make_model(),
        train[feats],
        train.label,
        cv=cv,
        groups=train.submissionId,
        scoring="roc_auc",
    ).mean()


def testauc(feats):
    m = make_model().fit(train[feats], train.label)
    return roc_auc_score(test.label, m.predict_proba(test[feats])[:, 1])


base = list(MODEL_FEATURES)
NEW = [
    "is_cold_start_flag",
    "agent_prior_quote_propensity",
    "system_bind_rate_trailing",
]
b_cv, b_te = cvauc(base), testauc(base)
print(f"BASE (current 5):  train CV-AUC {b_cv:.4f} | held-out test AUC {b_te:.4f}\n")
print("incremental (each added alone):   dCV       dTest")
for f in NEW:
    print(f"  + {f:30s} {cvauc(base+[f])-b_cv:+.4f}   {testauc(base+[f])-b_te:+.4f}")
print(
    f"\n  + ALL THREE                      {cvauc(base+NEW)-b_cv:+.4f}   {testauc(base+NEW)-b_te:+.4f}"
)
print(f"\nfull model with all 3: test AUC {testauc(base+NEW):.4f} (base {b_te:.4f})")

print("\ncollinearity / coverage notes:")
print(
    f"  corr(agent_bind_rate, quote_propensity) = {train.agent_bind_rate.corr(train.agent_prior_quote_propensity):+.3f}"
)
print(
    f"  corr(agent_bind_rate, system_trailing)  = {train.agent_bind_rate.corr(train.system_bind_rate_trailing):+.3f}"
)
print(f"  cold_start flag share (test) = {test.is_cold_start_flag.mean():.1%}")
