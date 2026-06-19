"""
Feature construction for the Novella bind-score.

Public API:
  - load_clean(data_dir)            -> (subs, events_clean)
  - build_panel(subs, events, ...)  -> one row per (submission, t), open-at-t, leakage-safe
  - feature(submission_id, t, ...)  -> the feature row for a single (submission, t)  [challenge contract]

Final feature set (5):
  agent_bind_rate     : repeat-customer track record (smoothed; see note), as-of createdDate
  outbound_chars_by_t : Novella (broker) effort = sum of OUTBOUND email chars by t
  has_quote_by_t      : 1 if a QUOTE_RECEIVED occurred by t
  n_inbound_by_t      : customer (agent) replies by t
  t                   : evaluation time (0/7/30) — lets one pooled model adapt across snapshots

LEAKAGE CONTRACT:
  - per-t features use ONLY events with d = (event_date - createdDate) <= t.
  - agent_bind_rate uses ONLY the agent's submissions RESOLVED BEFORE this submission's createdDate
    (raw counts agent_prior_n / agent_prior_binds are leakage-safe by construction).
  - resolvedDate / label are NEVER used as features.

SMOOTHING NOTE: agent_bind_rate = (agent_prior_binds + alpha*g) / (agent_prior_n + alpha).
  `g` is a prior (global base rate). For a strictly leakage-clean fit, model.py recomputes the rate
  with g = TRAIN base rate. build_panel/feature default g to the dataset base rate for standalone use
  and always expose the raw counts so the rate can be recomputed per-fold.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.preprocessing import clean

T_VALUES = (0, 7, 30)
SMOOTH_ALPHA = 5.0


def load_clean(data_dir: str | Path):
    data_dir = Path(data_dir)
    subs = pd.read_csv(
        data_dir / "features_submissions.csv",
        parse_dates=["createdDate", "resolvedDate"],
    )
    events = pd.read_csv(data_dir / "features_events.csv", parse_dates=["event_date"])
    events = clean(events, subs, verbose=False)[0]
    return subs, events


def smoothed_agent_rate(prior_binds, prior_n, global_rate, alpha=SMOOTH_ALPHA):
    """Empirical-Bayes shrink of an agent's close-rate toward `global_rate`.
    prior_n == 0 (cold start) -> returns exactly global_rate."""
    return (prior_binds + alpha * global_rate) / (prior_n + alpha)


def _agent_prior_counts(subs: pd.DataFrame) -> pd.DataFrame:
    """For each submission: how many of the SAME agent's submissions had resolved (outcome known)
    BEFORE this submission was created, and how many of those bound. Leakage-safe (as-of createdDate)."""
    rows = []
    for _, grp in subs.groupby("agentEmail"):
        for _, r in grp.iterrows():
            prior = grp[grp.resolvedDate < r.createdDate]
            rows.append((r.submissionId, len(prior), int(prior.label.sum())))
    return pd.DataFrame(
        rows, columns=["submissionId", "agent_prior_n", "agent_prior_binds"]
    )


def build_panel(
    subs: pd.DataFrame,
    events: pd.DataFrame,
    t_values=T_VALUES,
    global_rate: float | None = None,
) -> pd.DataFrame:
    """One row per (submission, t) for submissions still OPEN at t, with leakage-safe features."""
    subs = subs.copy()
    subs["resolution_days"] = (
        subs.resolvedDate - subs.createdDate
    ).dt.total_seconds() / 86400
    if global_rate is None:
        global_rate = float(subs.label.mean())

    ev = events.merge(
        subs[["submissionId", "createdDate"]], on="submissionId", how="left"
    )
    ev["d"] = (ev.event_date - ev.createdDate).dt.total_seconds() / 86400

    priors = _agent_prior_counts(subs)
    meta = subs[
        ["submissionId", "agentEmail", "createdDate", "resolution_days", "label"]
    ].merge(priors, on="submissionId")

    out = []
    for t in t_values:
        vis = ev[ev.d <= t]
        ob = vis[vis.event_type == "EMAIL_OUTBOUND"].groupby("submissionId")
        f = pd.DataFrame(index=subs.submissionId)
        f["outbound_chars_by_t"] = ob.email_char_count.sum()
        f["n_inbound_by_t"] = (
            vis[vis.event_type == "EMAIL_INBOUND"].groupby("submissionId").size()
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
        f = f.fillna(0.0).reset_index()
        f["t"] = t
        f = f.merge(meta, on="submissionId")
        f = f[f.resolution_days > t]  # open-at-t censoring
        out.append(f)

    panel = pd.concat(out, ignore_index=True)
    panel["agent_bind_rate"] = smoothed_agent_rate(
        panel.agent_prior_binds, panel.agent_prior_n, global_rate
    )
    # log1p(outbound_chars): chosen by the train CV A/B — tames the 43k-char outlier leverage on the linear model
    panel["outbound_chars_log"] = np.log1p(panel["outbound_chars_by_t"])
    cols = [
        "submissionId",
        "agentEmail",
        "createdDate",
        "t",
        "label",
        "resolution_days",
        "agent_bind_rate",
        "agent_prior_n",
        "agent_prior_binds",
        "outbound_chars_by_t",
        "outbound_chars_log",
        "has_quote_by_t",
        "n_inbound_by_t",
    ]
    return (
        panel[cols]
        .sort_values(["createdDate", "submissionId", "t"])
        .reset_index(drop=True)
    )


MODEL_FEATURES = [
    "agent_bind_rate",
    "outbound_chars_log",
    "has_quote_by_t",
    "n_inbound_by_t",
    "t",
]


def feature(
    submission_id: int,
    t: int,
    subs: pd.DataFrame,
    events: pd.DataFrame,
    global_rate: float | None = None,
) -> pd.Series:
    """Challenge contract: features for one (submission_id, t). Returns NaN row if not open at t.
    Note: agent_bind_rate needs the agent's history across all submissions, so we build the full
    panel at this single t and select the row (not a per-submission slice)."""
    panel = build_panel(subs, events, t_values=(t,), global_rate=global_rate)
    row = panel[(panel.submissionId == submission_id) & (panel.t == t)]
    if row.empty:
        return pd.Series({c: np.nan for c in MODEL_FEATURES}, name=(submission_id, t))
    return row.iloc[0][MODEL_FEATURES].rename((submission_id, t))


if __name__ == "__main__":
    R = Path(__file__).resolve().parents[1]
    subs, events = load_clean(R / "data")
    panel = build_panel(subs, events)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(
        f"panel rows: {len(panel)} | per t:",
        dict(panel.t.value_counts().sort_index()),
        f"| positives {int(panel.label.sum())} ({panel.label.mean():.1%})",
    )
    print("\nhead:")
    print(panel.head(6).to_string(index=False))
    print("\nfeature(submission_id=1, t=7):")
    print(feature(1, 7, subs, events).to_string())
    # leakage sanity: no feature column should reference resolvedDate/label
    print(
        "\nleakage check — agent_bind_rate range:",
        round(panel.agent_bind_rate.min(), 3),
        "..",
        round(panel.agent_bind_rate.max(), 3),
        "| cold-start value ~",
        round(float(subs.label.mean()), 3),
    )
