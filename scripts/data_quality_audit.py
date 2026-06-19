"""
Data-quality audit (read-only) — run BEFORE deciding any cleaning.
Quantifies temporal integrity, validity/garbage, outliers, and causal-ordering impossibilities.
Changes NOTHING; just reports. Repo: /Users/yonierlich/repos/challenge_1.
"""

import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
R = Path("/Users/yonierlich/repos/challenge_1")
s = pd.read_csv(
    R / "data/features_submissions.csv", parse_dates=["createdDate", "resolvedDate"]
)
e = pd.read_csv(R / "data/features_events.csv", parse_dates=["event_date"])
e = e.merge(
    s[["submissionId", "createdDate", "resolvedDate", "label"]],
    on="submissionId",
    how="left",
)
e["d"] = (e.event_date - e.createdDate).dt.total_seconds() / 86400
s["resolution_days"] = (s.resolvedDate - s.createdDate).dt.total_seconds() / 86400


def H(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


H("1. TEMPORAL INTEGRITY")
neg = e[e.d < 0]
print(f"events before createdDate (d<0): {len(neg)} ({len(neg)/len(e):.1%})")
if len(neg):
    print(
        "  d (days) percentiles:",
        neg.d.quantile([0, 0.5, 0.9, 1]).round(4).tolist(),
        f"| within 15min: {(neg.d > -0.0104).mean():.0%}",
    )
post = e[e.event_date > e.resolvedDate]
print(
    f"events AFTER resolvedDate: {len(post)} ({len(post)/len(e):.1%})  by type:",
    dict(post.event_type.value_counts()),
)
dup = e.duplicated(subset=["submissionId", "event_date", "event_type"], keep=False)
print(
    f"duplicate events (sub+timestamp+type): {dup.sum()}  | exact full-row dups: {e.drop(columns=['createdDate','resolvedDate','label','d']).duplicated().sum()}"
)
print(
    f"resolvedDate <= createdDate: {(s.resolution_days<=0).sum()}  | resolvedDate < createdDate: {(s.resolution_days<0).sum()}"
)

H("2. VALIDITY / GARBAGE")
print(
    "missing per events col:\n",
    e[["event_type", "event_date", "email_char_count", "email_attachment_count"]]
    .isna()
    .sum()
    .to_dict(),
)
em = e[e.event_type.isin(["EMAIL_INBOUND", "EMAIL_OUTBOUND"])]
print(
    f"EMAIL char_count: nulls={em.email_char_count.isna().sum()}, <=0={ (em.email_char_count<=0).sum() }, "
    f"max={em.email_char_count.max():.0f}, p99={em.email_char_count.quantile(.99):.0f}"
)
print(
    f"EMAIL attach: nulls={em.email_attachment_count.isna().sum()}, <0={(em.email_attachment_count<0).sum()}, "
    f"max={em.email_attachment_count.max():.0f}, p99={em.email_attachment_count.quantile(.99):.0f}"
)
qn = e[e.event_type == "QUOTE_RECEIVED"]
print(
    f"QUOTE rows with non-null char/attach (should be 0): {qn.email_char_count.notna().sum()} / {qn.email_attachment_count.notna().sum()}"
)
print(
    f"agentEmail: nulls={s.agentEmail.isna().sum()}, unique={s.agentEmail.nunique()}, "
    f"non-'N@gmail.com' format={(~s.agentEmail.astype(str).str.match(r'^\\d+@gmail\\.com$')).sum()}"
)
print(
    f"label values: {sorted(s.label.unique())} | event_type values: {sorted(e.event_type.unique())}"
)
print(
    f"instant resolution (<15min): {(s.resolution_days < 0.0104).sum()}  (<1h): {(s.resolution_days<1/24).sum()}  "
    f"by label among <15min:",
    dict(s[s.resolution_days < 0.0104].label.value_counts()),
)
noev = set(s.submissionId) - set(e.submissionId)
print(
    f"submissions with NO events: {len(noev)} -> labels:",
    dict(s[s.submissionId.isin(noev)].label.value_counts()),
)

H("3. OUTLIERS / DISTRIBUTIONS (per submission)")
g = e.groupby("submissionId")
agg = pd.DataFrame(
    {
        "n_events": g.size(),
        "n_outbound": g.event_type.apply(lambda x: (x == "EMAIL_OUTBOUND").sum()),
        "n_inbound": g.event_type.apply(lambda x: (x == "EMAIL_INBOUND").sum()),
        "n_quote": g.event_type.apply(lambda x: (x == "QUOTE_RECEIVED").sum()),
        "outbound_chars": e[e.event_type == "EMAIL_OUTBOUND"]
        .groupby("submissionId")
        .email_char_count.sum(),
    }
)
print(
    agg.describe(percentiles=[0.5, 0.9, 0.99]).round(1).T[["50%", "90%", "99%", "max"]]
)
print("\ntop 5 most-active submissions (n_events):")
print(
    agg.sort_values("n_events", ascending=False).head(5)[
        ["n_events", "n_outbound", "n_inbound", "n_quote"]
    ]
)

H("4. CAUSAL / ORDERING IMPOSSIBILITIES")
ev = e.sort_values(["submissionId", "event_date"])
first = ev.groupby("submissionId").first()
print("first-event type per submission:\n", first.event_type.value_counts().to_dict())


def first_d(tp):
    return ev[ev.event_type == tp].groupby("submissionId").d.min()


fq, fi, fo = (
    first_d("QUOTE_RECEIVED"),
    first_d("EMAIL_INBOUND"),
    first_d("EMAIL_OUTBOUND"),
)
fany = ev.groupby("submissionId").d.min()
quoted = fq.index
print(f"quoted submissions: {len(quoted)}")
print(
    f"  quote before ANY event (first event is a quote): {(fq <= fany.reindex(quoted)).sum()}"
)
print(
    f"  quote before first INBOUND (or no inbound):       {(fq < fi.reindex(quoted).fillna(np.inf)).sum()}"
)
print(
    f"  quote before first OUTBOUND (or no outbound):     {(fq < fo.reindex(quoted).fillna(np.inf)).sum()}"
)
fi_all = fi.reindex(s.submissionId)
fo_all = fo.reindex(s.submissionId)
print(
    f"OUTBOUND before any INBOUND (we replied first / no inbound): {(fo_all < fi_all.fillna(np.inf)).sum()}"
)
only_q = ev.groupby("submissionId").event_type.apply(
    lambda x: (x == "QUOTE_RECEIVED").all()
)
print(f"quote-only submissions (no emails at all): {only_q.sum()}")

H("5. SYNTHETIC SANITY")
print(
    f"createdDate at exactly midnight: {(s.createdDate.dt.time.astype(str)=='00:00:00').sum()} / {len(s)}"
)
print(
    f"createdDate distinct times-of-day: {s.createdDate.dt.time.nunique()}  | date range {s.createdDate.min().date()}..{s.createdDate.max().date()}"
)
print("done — nothing modified.")
