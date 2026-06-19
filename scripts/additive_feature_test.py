"""
(1) How many submissions come from one-time agents?
(2) Forward test: does ANY brainstormed feature add value ON TOP of the current 5-feature model
    (which now includes agent_bind_rate)? Incremental CV-AUC on train (the selection-safe way).
"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "/Users/yonierlich/repos/challenge_1")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from src.features import load_clean, smoothed_agent_rate
from src.model import temporal_split, make_model

R = Path("/Users/yonierlich/repos/challenge_1")
subs, events = load_clean(R / "data")
subs["resolution_days"] = (
    subs.resolvedDate - subs.createdDate
).dt.total_seconds() / 86400

# ---------- (1) one-time agents ----------
vc = subs.agentEmail.value_counts()
one_time_subs = int((vc == 1).sum())  # agents with exactly 1 sub -> 1 sub each
print("=== one-time agents ===")
print(f"one-time agents: {(vc==1).sum()} of {vc.size} agents")
print(
    f"submissions from one-time agents: {one_time_subs}/{len(subs)} = {one_time_subs/len(subs):.1%}"
)
# contrast: cold-start at creation (no prior-resolved history) is broader (incl. first sub of repeat agents)
ev0 = events.merge(subs[["submissionId", "createdDate"]], on="submissionId")
priors = []
for _, g in subs.groupby("agentEmail"):
    for _, r in g.iterrows():
        priors.append(
            (
                r.submissionId,
                int((g.resolvedDate < r.createdDate).sum()),
                int(g.label[g.resolvedDate < r.createdDate].sum()),
                int((g.createdDate < r.createdDate).sum()),
                int(
                    (
                        (g.createdDate < r.createdDate)
                        & (g.resolvedDate > r.createdDate)
                    ).sum()
                ),
            )
        )
P0 = pd.DataFrame(
    priors,
    columns=[
        "submissionId",
        "agent_prior_n",
        "agent_prior_binds",
        "agent_prior_subs",
        "agent_open_now",
    ],
)
cold = (P0.agent_prior_n == 0).mean()
print(
    f"cold-start at creation (no prior-resolved history): {cold:.1%}  (broader: includes a repeat agent's 1st sub)\n"
)

# ---------- (2) additive feature test ----------
ev0["d"] = (ev0.event_date - ev0.createdDate).dt.total_seconds() / 86400


def per_t(t):
    vis = ev0[ev0.d <= t]
    g = vis.groupby("submissionId")
    ob = vis[vis.event_type == "EMAIL_OUTBOUND"].groupby("submissionId")
    ib = vis[vis.event_type == "EMAIL_INBOUND"].groupby("submissionId")
    f = pd.DataFrame(index=subs.submissionId)
    f["outbound_chars_by_t"] = ob.email_char_count.sum()
    f["n_outbound_by_t"] = ob.size()
    f["n_inbound_by_t"] = ib.size()
    f["inbound_chars_by_t"] = ib.email_char_count.sum()
    f["n_quote_by_t"] = (
        vis[vis.event_type == "QUOTE_RECEIVED"].groupby("submissionId").size()
    )
    f["has_quote_by_t"] = (
        vis[vis.event_type == "QUOTE_RECEIVED"]
        .groupby("submissionId")
        .size()
        .reindex(subs.submissionId)
        .fillna(0)
        .gt(0)
        .astype(int)
        .values
    )
    f["recency_by_t"] = t - g.d.max()
    f["events_last7_by_t"] = vis[vis.d > t - 7].groupby("submissionId").size()
    f["outbound_last7_by_t"] = (
        vis[(vis.d > t - 7) & (vis.event_type == "EMAIL_OUTBOUND")]
        .groupby("submissionId")
        .size()
    )
    rec = vis[vis.d > t - 7].groupby("submissionId").size()
    pri = vis[(vis.d > t - 14) & (vis.d <= t - 7)].groupby("submissionId").size()
    f["accel_by_t"] = (rec / pri).replace([np.inf, -np.inf], np.nan)
    ow = vis[vis.event_type == "EMAIL_OUTBOUND"].copy()
    ow["w"] = ow.event_date.dt.weekday >= 5
    f["weekend_out_share_by_t"] = ow.groupby("submissionId").w.mean()
    cnt = [
        "outbound_chars_by_t",
        "n_outbound_by_t",
        "n_inbound_by_t",
        "inbound_chars_by_t",
        "n_quote_by_t",
        "events_last7_by_t",
        "outbound_last7_by_t",
    ]
    f[cnt] = f[cnt].fillna(0)
    f = f.reset_index()
    f["t"] = t
    return f


panel = pd.concat([per_t(t) for t in (0, 7, 30)], ignore_index=True)
cal = subs[
    ["submissionId", "createdDate", "resolution_days", "label", "agentEmail"]
].copy()
cal["created_dow"] = cal.createdDate.dt.weekday
cal["is_weekend"] = (cal.created_dow >= 5).astype(int)
cal["created_month"] = cal.createdDate.dt.month
cal["day_in_quarter"] = (
    cal.createdDate - cal.createdDate.dt.to_period("Q").dt.start_time
).dt.days
panel = panel.merge(cal, on="submissionId").merge(P0, on="submissionId")
panel = panel[panel.resolution_days > panel.t]
panel["outbound_chars_log"] = np.log1p(panel.outbound_chars_by_t)
panel["inbound_chars_log"] = np.log1p(panel.inbound_chars_by_t)

train, test, _ = temporal_split(panel)
g_train = float(train.drop_duplicates("submissionId").label.mean())
for df in (train, test):
    df["agent_bind_rate"] = smoothed_agent_rate(
        df.agent_prior_binds, df.agent_prior_n, g_train
    )

BASE = [
    "agent_bind_rate",
    "outbound_chars_log",
    "has_quote_by_t",
    "n_inbound_by_t",
    "t",
]
CANDIDATES = [
    "n_outbound_by_t",
    "inbound_chars_log",
    "n_quote_by_t",
    "recency_by_t",
    "events_last7_by_t",
    "outbound_last7_by_t",
    "accel_by_t",
    "weekend_out_share_by_t",
    "agent_prior_subs",
    "agent_open_now",
    "created_dow",
    "is_weekend",
    "created_month",
    "day_in_quarter",
]
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


base = cvauc(BASE)
print(
    f"=== additive test (train CV-AUC; base = current 5-feature model = {base:.4f}) ==="
)
deltas = sorted(
    ((c, cvauc(BASE + [c]) - base) for c in CANDIDATES), key=lambda x: -x[1]
)
for c, d in deltas:
    flag = " <-- helps?" if d > 0.005 else ""
    print(f"  + {c:24s}  {d:+.4f}{flag}")
