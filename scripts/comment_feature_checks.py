"""
Test the candidate features from comments.md (TODO checks), leakage-safe, per (submission, t):
  1. attach_total_log   : log1p(Σ attachments by t)             ("attachment total")
  2. early_attach       : attachment count of the EARLIEST visible email by t  ("attachment early / at interaction start")
  3. inbound_chars_log  : log1p(Σ inbound chars by t)           ("inbound chars log")
  4. total_chars_log    : log1p(Σ inbound+outbound chars by t)  ("combine outbound + inbound")
  5. n_email_by_t       : inbound + outbound count by t          (combined volume)
Decision rule (our standing rule): incremental ROC-AUC on the HELD-OUT TEMPORAL TEST vs the 5-feature base (0.744).
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
ev = events.merge(subs[["submissionId", "createdDate"]], on="submissionId")
ev["d"] = (ev.event_date - ev.createdDate).dt.total_seconds() / 86400
em = ev[ev.event_type.isin(["EMAIL_INBOUND", "EMAIL_OUTBOUND"])]

# candidate features per (submission, t)
rows = []
for t in (0, 7, 30):
    vis = em[em.d <= t]
    g = vis.groupby("submissionId")
    f = pd.DataFrame(index=subs.submissionId)
    f["attach_total"] = g.email_attachment_count.sum()
    f["inbound_chars"] = (
        vis[vis.event_type == "EMAIL_INBOUND"]
        .groupby("submissionId")
        .email_char_count.sum()
    )
    f["total_chars"] = g.email_char_count.sum()
    f["n_email_by_t"] = g.size()
    # earliest visible email's attachment count
    first_idx = (
        vis.sort_values("d").groupby("submissionId").head(1).set_index("submissionId")
    )
    f["early_attach"] = first_idx.email_attachment_count
    f = f.fillna(0.0)
    f["t"] = t
    f["submissionId"] = f.index
    rows.append(f)
cand = pd.concat(rows, ignore_index=True)
cand["attach_total_log"] = np.log1p(cand.attach_total)
cand["inbound_chars_log"] = np.log1p(cand.inbound_chars)
cand["total_chars_log"] = np.log1p(cand.total_chars)

panel = build_panel(subs, events).merge(
    cand[
        [
            "submissionId",
            "t",
            "attach_total_log",
            "early_attach",
            "inbound_chars_log",
            "total_chars_log",
            "n_email_by_t",
        ]
    ],
    on=["submissionId", "t"],
)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)
y = test.label.values
ts = (0, 7, 30)


def auc_set(cols):
    m = make_model().fit(train[cols], train.label)
    p = m.predict_proba(test[cols])[:, 1]
    out = {"overall": roc_auc_score(y, p)}
    for t in ts:
        mk = (test.t == t).values
        out[t] = roc_auc_score(y[mk], p[mk])
    return out


base = auc_set(MODEL_FEATURES)
print(f"{'feature set':28s} {'overall':>8} {'t=0':>7} {'t=7':>7} {'t=30':>7}")
print(
    f"{'BASE (5 features)':28s} {base['overall']:>8.3f} {base[0]:>7.3f} {base[7]:>7.3f} {base[30]:>7.3f}"
)
for c in [
    "attach_total_log",
    "early_attach",
    "inbound_chars_log",
    "total_chars_log",
    "n_email_by_t",
]:
    r = auc_set(MODEL_FEATURES + [c])
    d = r["overall"] - base["overall"]
    flag = "  <-- helps" if d > 0.003 else ("" if d > -0.003 else "  (worse)")
    print(
        f"{'+ ' + c:28s} {r['overall']:>8.3f} {r[0]:>7.3f} {r[7]:>7.3f} {r[30]:>7.3f}   Δ={d:+.3f}{flag}"
    )

# univariate correlation of each candidate (sanity)
print("\nunivariate test-AUC of each candidate alone:")
for c in [
    "attach_total_log",
    "early_attach",
    "inbound_chars_log",
    "total_chars_log",
    "n_email_by_t",
]:
    print(f"  {c:20s} {roc_auc_score(y, test[c].values):.3f}")
