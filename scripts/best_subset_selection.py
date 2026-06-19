"""
Best-subset (exhaustive) feature selection for the bind-score, vs greedy forward selection.

p=4 real predictors (t is the structural snapshot index, always included) -> 2^4-1 = 15 subsets,
so exhaustive search is trivially affordable and is the optimal version of what forward selection
approximates greedily.

Selection protocol (no test leakage):
  - build panel, temporal split, recompute agent_bind_rate with TRAIN base rate (same as train_bind_model)
  - score every subset by grouped (submissionId) StratifiedGroupKFold CV, averaged over several seeds
  - rank by CV precision@20 (the deployment metric); also report CV ROC-AUC
  - confirm the winner once on the held-out temporal test
"""

from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features import load_clean, build_panel
from src.model import make_model, temporal_split, recompute_agent_rate
from src.evaluate import precision_at_k

CANDIDATES = [
    "agent_bind_rate",
    "outbound_chars_log",
    "has_quote_by_t",
    "n_inbound_by_t",
]
SEEDS = (0, 1, 2, 3, 4)
SHORT = {
    "agent_bind_rate": "agent",
    "outbound_chars_log": "outbound",
    "has_quote_by_t": "quote",
    "n_inbound_by_t": "inbound",
    "t": "t",
}


def cv_scores(train: pd.DataFrame, feats: list[str]) -> tuple[float, float]:
    """Mean grouped-CV (precision@20, ROC-AUC) over SEEDS."""
    from sklearn.metrics import roc_auc_score

    p_runs, a_runs = [], []
    for seed in SEEDS:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        ps, aucs = [], []
        for tr, va in cv.split(train[feats], train.label, groups=train.submissionId):
            m = make_model().fit(train[feats].iloc[tr], train.label.iloc[tr])
            s = m.predict_proba(train[feats].iloc[va])[:, 1]
            y = train.label.iloc[va].values
            ps.append(precision_at_k(y, s))
            aucs.append(roc_auc_score(y, s))
        p_runs.append(np.mean(ps))
        a_runs.append(np.mean(aucs))
    return float(np.mean(p_runs)), float(np.mean(a_runs))


def main():
    R = Path(__file__).resolve().parents[1]
    subs, events = load_clean(R / "data")
    panel = build_panel(subs, events)
    train, test, thr = temporal_split(panel)
    recompute_agent_rate(train, test)
    print(
        f"train {len(train)} rows / test {len(test)} | "
        f"train base rate {train.drop_duplicates('submissionId').label.mean():.3f}\n"
    )

    # ---- exhaustive best-subset over the 4 predictors (t always in) ----
    rows = []
    for k in range(1, len(CANDIDATES) + 1):
        for combo in combinations(CANDIDATES, k):
            feats = list(combo) + ["t"]
            p, a = cv_scores(train, feats)
            rows.append(
                {
                    "k": k,
                    "features": "+".join(SHORT[f] for f in combo),
                    "has_outbound": "outbound_chars_log" in combo,
                    "cv_p@20": p,
                    "cv_auc": a,
                }
            )
    tab = (
        pd.DataFrame(rows)
        .sort_values("cv_p@20", ascending=False)
        .reset_index(drop=True)
    )
    pd.set_option("display.width", 200)
    print(
        "=== EXHAUSTIVE best-subset (all 15 combos, +t), ranked by CV precision@20 ==="
    )
    print(tab.round(4).to_string(index=False))

    best = tab.iloc[0]
    full = tab[tab.features.str.count(r"\+") == len(CANDIDATES) - 1].iloc[0]
    print(
        f"\nwinner (CV p@20): {best.features}+t   p@20={best['cv_p@20']:.4f}  auc={best['cv_auc']:.4f}"
    )
    print(
        f"full 4+t:        {full.features}+t   p@20={full['cv_p@20']:.4f}  auc={full['cv_auc']:.4f}"
    )

    # outbound's incremental effect: each subset with outbound vs the same subset without it
    print("\n=== marginal effect of ADDING outbound to each subset (CV p@20) ===")
    base_map = {
        r.features.replace("+outbound", "")
        .replace("outbound+", "")
        .replace("outbound", ""): r
        for _, r in tab.iterrows()
        if not r.has_outbound
    }
    for _, r in tab[tab.has_outbound].iterrows():
        without = (
            r.features.replace("+outbound", "")
            .replace("outbound+", "")
            .replace("outbound", "")
            .strip("+")
        )
        match = tab[
            (~tab.has_outbound) & (tab.features == (without if without else ""))
        ]
        if not match.empty:
            base = match.iloc[0]
            print(
                f"  {without or '(none)':28s}+t  {base['cv_p@20']:.4f}  ->  +outbound {r['cv_p@20']:.4f}  (Δ {r['cv_p@20']-base['cv_p@20']:+.4f})"
            )

    # ---- greedy forward selection (seed = highest |label corr|) for comparison ----
    print("\n=== greedy forward selection (seed = highest |label corr|), CV p@20 ===")
    corr = (
        train[CANDIDATES]
        .apply(lambda c: np.corrcoef(c, train.label)[0, 1])
        .abs()
        .sort_values(ascending=False)
    )
    print("  |label corr|:", ", ".join(f"{SHORT[f]} {v:.3f}" for f, v in corr.items()))
    selected = [corr.index[0]]
    p0, _ = cv_scores(train, selected + ["t"])
    print(f"  seed [{SHORT[selected[0]]}]  CV p@20 = {p0:.4f}")
    remaining = [f for f in CANDIDATES if f not in selected]
    while remaining:
        gains = {f: cv_scores(train, selected + [f] + ["t"])[0] for f in remaining}
        bf = max(gains, key=gains.get)
        if gains[bf] > p0 + 1e-9:
            print(f"  + {SHORT[bf]:8s} -> {gains[bf]:.4f}  (added)")
            selected.append(bf)
            p0 = gains[bf]
            remaining.remove(bf)
        else:
            print(
                f"  + {SHORT[bf]:8s} -> {gains[bf]:.4f}  ✗ no gain over {p0:.4f} -> STOP"
            )
            break
    print(f"  greedy selected: {[SHORT[f] for f in selected]}+t")

    # ---- confirm on held-out test ----
    from sklearn.metrics import roc_auc_score

    print("\n=== held-out TEST confirmation ===")
    for label, combo in [
        ("winner (" + best.features + ")", best.features.split("+")),
        (
            "full (agent+outbound+quote+inbound)",
            ["agent", "outbound", "quote", "inbound"],
        ),
    ]:
        feats = [k for k, v in SHORT.items() if v in combo] + ["t"]
        feats = list(dict.fromkeys(feats))
        m = make_model().fit(train[feats], train.label)
        s = m.predict_proba(test[feats])[:, 1]
        print(
            f"  {label:42s}  test AUC {roc_auc_score(test.label, s):.4f}  test p@20 {precision_at_k(test.label.values, s):.4f}"
        )


if __name__ == "__main__":
    main()
