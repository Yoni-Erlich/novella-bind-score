"""
Idea (from the negative outbound coef): make the implicit "heavy broker writing WITHOUT a quote = struggling deal"
signal EXPLICIT via an interaction feature, instead of the model reconstructing it from −outbound + +quote.

New leakage-safe features (products of two existing leakage-safe features):
  outbound_no_quote   = outbound_chars_log * (1 - has_quote_by_t)   # writing while still unquoted
  outbound_with_quote = outbound_chars_log * has_quote_by_t          # writing once a quote exists

Judge on BOTH: held-out temporal test AND grouped CV-within-train (robustness — a holdout-only win is noise).
"""

import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate, make_model

subs, events = load_clean(R / "data")
panel = build_panel(subs, events)
for df in (panel,):
    df["outbound_no_quote"] = df.outbound_chars_log * (1 - df.has_quote_by_t)
    df["outbound_with_quote"] = df.outbound_chars_log * df.has_quote_by_t
train, test, thr = temporal_split(panel)
recompute_agent_rate(train, test)
y = test.label.values
ts = (0, 7, 30)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)


def evalset(cols):
    m = make_model().fit(train[cols], train.label)
    p = m.predict_proba(test[cols])[:, 1]
    cvm = cross_val_score(
        make_model(),
        train[cols],
        train.label,
        cv=cv,
        groups=train.submissionId,
        scoring="roc_auc",
    ).mean()
    out = {"cv": cvm, "test": roc_auc_score(y, p)}
    for t in ts:
        mk = (test.t == t).values
        out[t] = roc_auc_score(y[mk], p[mk])
    return out


LOCAL = [
    "agent_bind_rate",
    "has_quote_by_t",
    "n_inbound_by_t",
    "t",
]  # base minus outbound
SETS = {
    "BASE (5 feat)": MODEL_FEATURES,
    "+ outbound_no_quote": MODEL_FEATURES + ["outbound_no_quote"],
    "REPLACE outbound → no_quote only": LOCAL + ["outbound_no_quote"],
    "REPLACE outbound → no_quote + with_quote": LOCAL
    + ["outbound_no_quote", "outbound_with_quote"],
    "+ both interactions (keep outbound)": MODEL_FEATURES
    + ["outbound_no_quote", "outbound_with_quote"],
}

ref = evalset(MODEL_FEATURES)
print(
    f"{'feature set':42s} {'CV-train':>8} {'test':>7} {'t=0':>6} {'t=7':>6} {'t=30':>6}  Δtest  Δcv"
)
for name, cols in SETS.items():
    r = evalset(cols)
    dt, dc = r["test"] - ref["test"], r["cv"] - ref["cv"]
    print(
        f"{name:42s} {r['cv']:>8.3f} {r['test']:>7.3f} {r[0]:>6.3f} {r[7]:>6.3f} {r[30]:>6.3f}  "
        f"{dt:+.3f} {dc:+.3f}"
    )

# direction of the interaction in the full model
m = make_model().fit(train[MODEL_FEATURES + ["outbound_no_quote"]], train.label)
import pandas as pd

coefs = pd.Series(
    m.named_steps["logisticregression"].coef_[0],
    index=MODEL_FEATURES + ["outbound_no_quote"],
)
print("\nsigned coefs with outbound_no_quote added:")
print(coefs.round(3).to_string())
print(
    f"\nunivariate test-AUC: outbound_no_quote alone = {roc_auc_score(y, test.outbound_no_quote.values):.3f}"
)
