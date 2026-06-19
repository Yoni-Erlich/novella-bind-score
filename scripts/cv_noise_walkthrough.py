"""Trace every number in the 'why CV precision@20 is noisy' argument, from raw data to the SE.
Run: poetry run python scripts/cv_noise_walkthrough.py
Prints each intermediate value so the markdown writeup can cite real, reproduced figures."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate, make_model
from src.evaluate import precision_at_k

R = Path(__file__).resolve().parents[1]
subs, events = load_clean(R / "data")
panel = build_panel(subs, events)
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)

print("=" * 70)
print("STEP 1 — data shapes")
print("=" * 70)
print(f"panel rows (submission,t)         : {len(panel)}")
print(f"train rows                        : {len(train)}")
print(f"test rows                         : {len(test)}")
print(f"distinct train submissions        : {train.submissionId.nunique()}")
print(f"train positive ROWS               : {int(train.label.sum())}")
print(f"train base rate (rows)            : {train.label.mean():.4f}")

print("\n" + "=" * 70)
print("STEP 2 — one CV pass (seed=0), full model: per-fold bucket + binders")
print("=" * 70)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
fold_prec, fold_k, fold_hits, fold_nval = [], [], [], []
m = make_model()
for i, (tr, va) in enumerate(
    cv.split(train[MODEL_FEATURES], train.label, groups=train.submissionId)
):
    mm = make_model().fit(train[MODEL_FEATURES].iloc[tr], train.label.iloc[tr])
    sc = mm.predict_proba(train[MODEL_FEATURES].iloc[va])[:, 1]
    yv = train.label.iloc[va].values
    k = max(1, int(np.ceil(0.20 * len(yv))))
    topidx = np.argsort(-sc, kind="stable")[:k]
    hits = int(yv[topidx].sum())
    prec = hits / k
    fold_nval.append(len(yv))
    fold_k.append(k)
    fold_hits.append(hits)
    fold_prec.append(prec)
    print(
        f"  fold {i}: n_val={len(yv):3d}  k=ceil(.2*n)={k:2d}  binders_in_top_k={hits:2d}  "
        f"precision@20={prec:.3f}  (fold base rate {yv.mean():.3f})"
    )

p_bar = float(np.mean(fold_prec))
k_bar = float(np.mean(fold_k))
print(
    f"\n  mean over 5 folds: k≈{k_bar:.0f}  precision@20={p_bar:.3f}  "
    f"binders≈{np.mean(fold_hits):.1f}"
)

print("\n" + "=" * 70)
print("STEP 3 — random vs model-enriched binder count in the top bucket")
print("=" * 70)
base = train.label.mean()
print(
    f"if the top-{int(k_bar)} bucket were RANDOM : {base:.3f} * {k_bar:.0f} = {base*k_bar:.1f} binders"
)
print(
    f"model-enriched (precision {p_bar:.3f})    : {p_bar:.3f} * {k_bar:.0f} = {p_bar*k_bar:.1f} binders"
)
print(f"-> the gap IS the model's lift            : {p_bar/base:.2f}x")

print("\n" + "=" * 70)
print("STEP 4 — binomial standard error of ONE fold's precision")
print("=" * 70)
k = k_bar
se_rate = np.sqrt(p_bar * (1 - p_bar) / k)
se_count = np.sqrt(k * p_bar * (1 - p_bar))
print(f"p (the measured precision)        : {p_bar:.3f}")
print(f"k (top-bucket size)               : {k:.0f}")
print(f"SE(rate)  = sqrt(p(1-p)/k)        : {se_rate:.4f}   (±{se_rate:.3f})")
print(
    f"SE(count) = sqrt(k*p(1-p))        : {se_count:.3f}    (±{se_count:.1f} binders)"
)
print(f"one binder in/out = 1/k           : {1/k:.4f}")
print(f"plausible 1-SE range (rate)       : {p_bar-se_rate:.3f} .. {p_bar+se_rate:.3f}")
print(
    f"plausible 1-SE range (count)      : {p_bar*k-se_count:.1f} .. {p_bar*k+se_count:.1f} of {k:.0f}"
)

print("\n" + "=" * 70)
print("STEP 5 — EMPIRICAL noise: fold-to-fold and seed-to-seed")
print("=" * 70)
print(
    f"observed precision across the 5 folds (seed 0): "
    f"{[round(p,3) for p in fold_prec]}"
)
print(f"  -> std across folds              : {np.std(fold_prec, ddof=1):.4f}")

seed_means = []
for sd in range(5):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=sd)
    ps = []
    for tr, va in cv.split(
        train[MODEL_FEATURES], train.label, groups=train.submissionId
    ):
        mm = make_model().fit(train[MODEL_FEATURES].iloc[tr], train.label.iloc[tr])
        sc = mm.predict_proba(train[MODEL_FEATURES].iloc[va])[:, 1]
        ps.append(precision_at_k(train.label.iloc[va].values, sc))
    seed_means.append(float(np.mean(ps)))
print(
    f"\nmean precision@20 per seed (avg of its 5 folds): {[round(p,3) for p in seed_means]}"
)
print(f"  -> std across seeds              : {np.std(seed_means, ddof=1):.4f}")
print(
    f"  -> range across seeds            : {min(seed_means):.3f} .. {max(seed_means):.3f}"
)

print("\n" + "=" * 70)
print("STEP 6 — the signal we are trying to resolve")
print("=" * 70)
print("top-4 subset spread (from best_subset_selection.py): 0.251 - 0.246 = 0.005")
print(
    f"seed-to-seed noise on a SINGLE subset             : ±{np.std(seed_means, ddof=1):.3f}"
)
print(
    "-> the spread (0.005) is the same size as / smaller than the noise -> unresolvable"
)
