"""
Test agent-level (repeat-client) features, leakage-safe, computed as-of createdDate:
  agent_bind_rate  : smoothed close-rate from the agent's PRIOR-RESOLVED subs (5 pseudo-deals @ global rate)
  agent_prior_subs : # of the agent's submissions created before this one (experience)
  agent_open_now   : # of the agent's other submissions open at this createdDate (parallel queue)
Evaluation: temporal split (train=earlier-created, test=later) — the right setup for repeat clients.
Compare submission-local features vs + agent features. Pooled panel, report per-t test AUC.
"""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "/Users/yonierlich/repos/challenge_1")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from src.preprocessing import clean

R = "/Users/yonierlich/repos/challenge_1"
s = pd.read_csv(
    f"{R}/data/features_submissions.csv", parse_dates=["createdDate", "resolvedDate"]
)
e = pd.read_csv(f"{R}/data/features_events.csv", parse_dates=["event_date"])
e, _ = clean(e, s, verbose=False)
e = e.merge(s[["submissionId", "createdDate"]], on="submissionId")
e["d"] = (e.event_date - e.createdDate).dt.total_seconds() / 86400
s["resolution_days"] = (s.resolvedDate - s.createdDate).dt.total_seconds() / 86400
G = s.label.mean()
ALPHA = 5

# ---- agent features as-of createdDate (submission level) ----
rows = []
for a, grp in s.groupby("agentEmail"):
    for _, r in grp.iterrows():
        c = r.createdDate
        prior_res = grp[grp.resolvedDate < c]  # outcome known before this sub created
        prior_created = grp[grp.createdDate < c]  # any earlier request
        open_now = grp[(grp.createdDate < c) & (grp.resolvedDate > c)]
        rate = (prior_res.label.sum() + ALPHA * G) / (len(prior_res) + ALPHA)
        rows.append(
            (r.submissionId, rate, len(prior_created), len(open_now), len(prior_res))
        )
af = pd.DataFrame(
    rows,
    columns=[
        "submissionId",
        "agent_bind_rate",
        "agent_prior_subs",
        "agent_open_now",
        "prior_res_n",
    ],
)
A = s.merge(af, on="submissionId")

print("=== agent-feature signal (submission level, n=881) ===")
for c in ["agent_bind_rate", "agent_prior_subs", "agent_open_now"]:
    print(f"  corr(label, {c}) = {A[c].corr(A.label):+.3f}")
print(f"  coverage with real history (prior_res_n>=1): {(A.prior_res_n>=1).mean():.0%}")


# ---- per-t panel with submission-local features ----
def panel():
    out = []
    for t in (0, 7, 30):
        vis = e[e.d <= t]
        o = vis[vis.event_type == "EMAIL_OUTBOUND"].groupby("submissionId")
        df = pd.DataFrame(index=s.submissionId)
        df["outbound_chars_by_t"] = o.email_char_count.sum()
        df["n_inbound_by_t"] = (
            vis[vis.event_type == "EMAIL_INBOUND"].groupby("submissionId").size()
        )
        df["has_quote_by_t"] = (
            vis[vis.event_type == "QUOTE_RECEIVED"]
            .groupby("submissionId")
            .size()
            .reindex(s.submissionId)
            .fillna(0)
            .gt(0)
            .astype(int)
            .values
        )
        df = df.fillna(0)
        df["t"] = t
        df = df.join(
            s.set_index("submissionId")[["label", "resolution_days", "createdDate"]]
        )
        df["submissionId"] = df.index
        out.append(df[df.resolution_days > t])
    P = pd.concat(out, ignore_index=True).merge(
        af.drop(columns="prior_res_n"), on="submissionId"
    )
    return P


P = panel()
thr = s.createdDate.quantile(0.70)
tr, te = P[P.createdDate <= thr], P[P.createdDate > thr]
base = ["outbound_chars_by_t", "has_quote_by_t", "n_inbound_by_t", "t"]
agent = ["agent_bind_rate", "agent_prior_subs", "agent_open_now"]


def fit_auc(cols):
    m = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    m.fit(tr[cols], tr.label)
    p = m.predict_proba(te[cols])[:, 1]
    res = {"overall": roc_auc_score(te.label, p)}
    for t in (0, 7, 30):
        mk = te.t == t
        if 0 < te.label[mk].sum() < mk.sum():
            res[f"t{t}"] = roc_auc_score(te.label[mk], p[mk])
    return res


print(
    f"\n=== TEMPORAL split (train<= {thr.date()}, n_test={len(te)}, test pos={int(te.label.sum())}) ==="
)
print("submission-local only :", {k: round(v, 3) for k, v in fit_auc(base).items()})
print(
    "+ agent features      :",
    {k: round(v, 3) for k, v in fit_auc(base + agent).items()},
)
print("agent features ONLY   :", {k: round(v, 3) for k, v in fit_auc(agent).items()})
