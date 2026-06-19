"""
Experiment: a DEDICATED t=0 model with extra CUSTOMER-history features (agentEmail = retail agent).
At t=0 the current submission has almost no event signal, so lean on the customer's PAST interactions.

New leakage-safe features (as-of createdDate, from the customer's submissions RESOLVED before createdDate,
using those submissions' FULL event history since they're entirely in the past):
  cust_hist_avg_outbound   : avg # OUTBOUND emails per past submission
  cust_hist_avg_inbound    : avg # INBOUND  emails per past submission
  cust_hist_out_chars/email: avg chars per OUTBOUND email across past interactions
  cust_hist_in_chars/email : avg chars per INBOUND  email across past interactions
  cust_hist_quote_rate     : fraction of past submissions that got a QUOTE
(All undefined for cold-start customers -> imputed; so they can only help the RETURNING segment.)

Compare, on the held-out temporal test, t=0 only:
  base    = current MODEL_FEATURES (minus t, constant at t=0)
  +hist   = base + the 5 customer-history features
Report AUC overall and split by returning / cold-start.
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
from src.features import load_clean, build_panel
from src.model import temporal_split, recompute_agent_rate, make_model

subs, events = load_clean(R / "data")

# ---- per-submission event aggregates (full lifecycle) ----
ev = events.copy()
out = ev[ev.event_type == "EMAIL_OUTBOUND"]
inb = ev[ev.event_type == "EMAIL_INBOUND"]
qt = ev[ev.event_type == "QUOTE_RECEIVED"]
per_sub = pd.DataFrame({"submissionId": subs.submissionId}).set_index("submissionId")
per_sub["n_out"] = out.groupby("submissionId").size()
per_sub["n_in"] = inb.groupby("submissionId").size()
per_sub["c_out"] = out.groupby("submissionId").email_char_count.sum()
per_sub["c_in"] = inb.groupby("submissionId").email_char_count.sum()
per_sub["has_q"] = qt.groupby("submissionId").size().reindex(subs.submissionId).gt(0)
per_sub = per_sub.fillna(0.0)
S = subs.merge(per_sub, on="submissionId")

# ---- customer-history features as-of createdDate (prior-RESOLVED subs only) ----
rows = []
for _, grp in S.groupby("agentEmail"):
    for _, r in grp.iterrows():
        prior = grp[grp.resolvedDate < r.createdDate]  # outcome fully known before now
        n = len(prior)
        if n == 0:
            rows.append((r.submissionId, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        tot_out, tot_in = prior.n_out.sum(), prior.n_in.sum()
        rows.append(
            (
                r.submissionId,
                tot_out / n,
                tot_in / n,
                prior.c_out.sum() / tot_out if tot_out else np.nan,
                prior.c_in.sum() / tot_in if tot_in else np.nan,
                prior.has_q.mean(),
            )
        )
HIST = [
    "cust_hist_avg_outbound",
    "cust_hist_avg_inbound",
    "cust_hist_out_chars_per_email",
    "cust_hist_in_chars_per_email",
    "cust_hist_quote_rate",
]
hist = pd.DataFrame(rows, columns=["submissionId"] + HIST)

# ---- panel, temporal split (same threshold/recompute as production), then t=0 only ----
panel = build_panel(subs, events)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)  # agent_bind_rate -> train base rate prior
train = train[train.t == 0].merge(hist, on="submissionId")
test = test[test.t == 0].merge(hist, on="submissionId")

BASE = ["agent_bind_rate", "outbound_chars_log", "has_quote_by_t", "n_inbound_by_t"]
cold = (test.agent_prior_n == 0).values


def seg_auc(cols):
    m = make_model().fit(train[cols], train.label)
    p = m.predict_proba(test[cols])[:, 1]
    y = test.label.values

    def a(mask):
        yy = y[mask]
        return roc_auc_score(yy, p[mask]) if 0 < yy.sum() < len(yy) else float("nan")

    return a(np.ones(len(y), bool)), a(~cold), a(cold)


print(f"t=0 held-out test: {len(test)} rows, {int(test.label.sum())} sold")
print(f"  returning {int((~cold).sum())} | cold-start {int(cold.sum())}")
print(
    f"  customer-history coverage (non-null) in test: "
    f"{test[HIST].notna().any(axis=1).mean():.0%}  (== returning share)\n"
)

print(f"{'feature set':16s} {'overall':>8} {'returning':>10} {'cold-start':>11}")
for name, cols in (("base (t=0)", BASE), ("base + hist", BASE + HIST)):
    o, r_, c = seg_auc(cols)
    print(f"{name:16s} {o:>8.3f} {r_:>10.3f} {c:>11.3f}")

# univariate signal of each new feature on RETURNING t=0 rows (where it's defined)
print("\nunivariate AUC of each customer-history feature (returning t=0 rows only):")
yr = test.label.values[~cold]
for f in HIST:
    v = test[f].values[~cold]
    ok = ~np.isnan(v)
    if 0 < yr[ok].sum() < ok.sum():
        print(f"  {f:32s} {roc_auc_score(yr[ok], v[ok]):.3f}   (n={ok.sum()})")
