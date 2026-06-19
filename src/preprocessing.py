"""
Data validation + cleaning for the Novella bind-score challenge.

Design choices:
  - validate() runs the FULL battery of integrity checks and reports each one's count
    EVEN WHEN IT IS CLEAN (count 0) — the report documents what we look for, not just
    what we found.
  - clean() applies only the decided policy: drop exact-duplicate events, drop
    pre-creation (impossible) events. It does NOT touch char outliers — those are real
    rows; their skew is handled later by a log1p transform in the FEATURE pipeline.
  - Everything here is leakage-safe: it only removes invalid rows; it never uses the
    label or resolvedDate to build a feature.

Domain note (NOT errors, kept by design):
  - OUTBOUND or QUOTE before the first INBOUND: the request enters via `createdDate`
    (intake channel, not necessarily an email), and carrier comms are off-channel, so
    Novella can email out / receive a quote before the agent's first inbound email.
  - These respect time (events after createdDate). Contrast pre-creation events, which
    are dated before the submission exists — those violate causality and are dropped.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

ALLOWED_EVENTS = {"EMAIL_INBOUND", "EMAIL_OUTBOUND", "QUOTE_RECEIVED"}
EMAIL_TYPES = {"EMAIL_INBOUND", "EMAIL_OUTBOUND"}
PRE_CREATION_DROP_DAYS = (
    -1.0
)  # drop events more than 1 day before createdDate (impossible)
CHAR_OUTLIER_FLAG = 10_000  # flagged for the log1p transform downstream (NOT dropped)


def _prep(events: pd.DataFrame, subs: pd.DataFrame) -> pd.DataFrame:
    e = events.merge(
        subs[["submissionId", "createdDate", "resolvedDate", "label"]],
        on="submissionId",
        how="left",
    )
    e["d_days"] = (e["event_date"] - e["createdDate"]).dt.total_seconds() / 86400
    return e


def validate(events: pd.DataFrame, subs: pd.DataFrame) -> pd.DataFrame:
    """Run every integrity check; return a table of (check, count, severity, action).
    Checks are reported even when count==0 so the report shows the full battery."""
    e = _prep(events, subs)
    em = e[e.event_type.isin(EMAIL_TYPES)]
    dup_key = [
        "submissionId",
        "event_date",
        "event_type",
        "email_char_count",
        "email_attachment_count",
    ]
    ev = e.sort_values(["submissionId", "event_date"])
    first_d = lambda tp: ev[ev.event_type == tp].groupby("submissionId").d_days.min()
    fq, fi, fo = (
        first_d("QUOTE_RECEIVED"),
        first_d("EMAIL_INBOUND"),
        first_d("EMAIL_OUTBOUND"),
    )
    fany = ev.groupby("submissionId").d_days.min()
    quoted = fq.index
    orphan_events = set(events.submissionId) - set(subs.submissionId)
    no_event_subs = set(subs.submissionId) - set(events.submissionId)

    checks = [
        # --- temporal integrity ---
        (
            "events before createdDate (any, d<0)",
            int((e.d_days < 0).sum()),
            "info",
            "see buckets",
        ),
        (
            "  ...impossible: d < -1 day",
            int((e.d_days < PRE_CREATION_DROP_DAYS).sum()),
            "garbage",
            "DROP",
        ),
        (
            "  ...clock-skew: -15min <= d < 0",
            int(((e.d_days >= -0.0104) & (e.d_days < 0)).sum()),
            "ok",
            "keep",
        ),
        (
            "events after resolvedDate",
            int((e.event_date > e.resolvedDate).sum()),
            "error",
            "drop if any",
        ),
        (
            "resolvedDate <= createdDate",
            int((subs.resolvedDate <= subs.createdDate).sum()),
            "error",
            "review if any",
        ),
        # --- duplicates ---
        (
            "exact-duplicate event rows (removable)",
            int(e.duplicated(subset=dup_key, keep="first").sum()),
            "garbage",
            "DROP",
        ),
        # --- validity / garbage ---
        (
            "missing event_type",
            int(events.event_type.isna().sum()),
            "error",
            "review if any",
        ),
        (
            "missing event_date",
            int(events.event_date.isna().sum()),
            "error",
            "review if any",
        ),
        (
            "EMAIL with null char_count",
            int(em.email_char_count.isna().sum()),
            "error",
            "review if any",
        ),
        (
            "EMAIL with char_count <= 0",
            int((em.email_char_count <= 0).sum()),
            "error",
            "review if any",
        ),
        (
            "EMAIL with attachment_count < 0",
            int((em.email_attachment_count < 0).sum()),
            "error",
            "review if any",
        ),
        (
            "QUOTE with non-null char/attach",
            int(
                e[e.event_type == "QUOTE_RECEIVED"][
                    ["email_char_count", "email_attachment_count"]
                ]
                .notna()
                .any(axis=1)
                .sum()
            ),
            "info",
            "expected null",
        ),
        (
            "label not in {0,1}",
            int((~subs.label.isin([0, 1])).sum()),
            "error",
            "review if any",
        ),
        (
            "event_type not in allowed set",
            int((~events.event_type.isin(ALLOWED_EVENTS)).sum()),
            "error",
            "review if any",
        ),
        (
            "agentEmail nulls",
            int(subs.agentEmail.isna().sum()),
            "error",
            "review if any",
        ),
        (
            "agentEmail not 'N@gmail.com' format",
            int((~subs.agentEmail.astype(str).str.match(r"^\d+@gmail\.com$")).sum()),
            "info",
            "format check",
        ),
        (
            "orphan events (sub not in submissions)",
            len(orphan_events),
            "error",
            "review if any",
        ),
        (
            "submissions with NO events",
            len(no_event_subs),
            "info",
            "keep (zero activity informative)",
        ),
        # --- outliers (NOT dropped; handled by log1p in features) ---
        (
            f"EMAIL char_count > {CHAR_OUTLIER_FLAG}",
            int((em.email_char_count > CHAR_OUTLIER_FLAG).sum()),
            "outlier",
            "log1p in features",
        ),
        (
            "instant resolution (<15min)",
            int(
                (
                    ((subs.resolvedDate - subs.createdDate).dt.total_seconds() / 86400)
                    < 0.0104
                ).sum()
            ),
            "info",
            "keep",
        ),
        # --- causal / ordering ---
        (
            "quote before ANY event (first event=quote)",
            int((fq <= fany.reindex(quoted)).sum()),
            "flag",
            "keep (low impact)",
        ),
        (
            "quote before first INBOUND (or no inbound)",
            int((fq < fi.reindex(quoted).fillna(np.inf)).sum()),
            "ok",
            "keep (domain norm)",
        ),
        (
            "quote before first OUTBOUND (or no outbound)",
            int((fq < fo.reindex(quoted).fillna(np.inf)).sum()),
            "flag",
            "keep (carrier off-channel)",
        ),
        (
            "OUTBOUND before any INBOUND (or no inbound)",
            int(
                (
                    fo.reindex(subs.submissionId)
                    < fi.reindex(subs.submissionId).fillna(np.inf)
                ).sum()
            ),
            "ok",
            "keep (intake via createdDate)",
        ),
        (
            "quote-only submissions (no emails)",
            int(
                ev.groupby("submissionId")
                .event_type.apply(lambda x: (x == "QUOTE_RECEIVED").all())
                .sum()
            ),
            "flag",
            "review if any",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "count", "severity", "action"])


def clean(
    events: pd.DataFrame, subs: pd.DataFrame, verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Apply the decided policy: drop exact-duplicate events + pre-creation (impossible) events.
    Returns (cleaned_events, log). Char outliers are intentionally NOT touched here."""
    e = _prep(events, subs)
    n0 = len(e)
    dup_key = [
        "submissionId",
        "event_date",
        "event_type",
        "email_char_count",
        "email_attachment_count",
    ]
    dup_mask = e.duplicated(subset=dup_key, keep="first")
    pre_mask = e.d_days < PRE_CREATION_DROP_DAYS
    cleaned = (
        e[~(dup_mask | pre_mask)]
        .drop(columns=["createdDate", "resolvedDate", "label", "d_days"])
        .reset_index(drop=True)
    )
    log = {
        "rows_in": n0,
        "dropped_duplicates": int(dup_mask.sum()),
        "dropped_pre_creation": int(pre_mask.sum()),
        "subs_affected": int(e[dup_mask | pre_mask].submissionId.nunique()),
        "rows_out": len(cleaned),
    }
    if verbose:
        print("cleaning:", log)
    return cleaned, log


if __name__ == "__main__":
    from pathlib import Path

    R = Path(__file__).resolve().parents[1]
    subs = pd.read_csv(
        R / "data/features_submissions.csv", parse_dates=["createdDate", "resolvedDate"]
    )
    events = pd.read_csv(R / "data/features_events.csv", parse_dates=["event_date"])
    rep = validate(events, subs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print("=== VALIDATION (full battery; clean checks shown too) ===")
    print(rep.to_string(index=False))
    print()
    clean(events, subs)
