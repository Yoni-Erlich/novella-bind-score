"""
Refine the agent-alone proof: break the held-out test down by BOTH t AND segment.
Key question: at t=0 the per-t event features (has_quote/outbound/inbound) are mostly
empty -> do the 'other features' actually rescue the COLD-START segment at t=0, or only
at t=7/t=30? Also: how much non-zero per-t signal even exists at each t on cold-start.
"""

import warnings

warnings.filterwarnings("ignore")
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

import numpy as np
from sklearn.metrics import roc_auc_score
from src.evaluate import get_test_scores


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return roc_auc_score(y, s)


fit, test, scores = get_test_scores()
full = scores["logistic (final)"]
agent_only = scores["agent_bind_rate only"]
y = test.label.values
cold = (test.agent_prior_n == 0).values

print("AUC by (segment x t):  agent_bind_rate ONLY  /  FULL model   (n, #sold)\n")
print(f"{'t':>4} {'segment':12} {'agent-only':>11} {'full':>8}   {'n':>4} {'sold':>4}")
for t in (0, 7, 30):
    tm = (test.t == t).values
    for name, seg in (("returning", ~cold & tm), ("cold-start", cold & tm)):
        a, f = auc(y[seg], agent_only[seg]), auc(y[seg], full[seg])
        a = f"{a:.3f}" if a is not None else "  -  "
        f = f"{f:.3f}" if f is not None else "  -  "
        print(
            f"{t:>4} {name:12} {a:>11} {f:>8}   {seg.sum():>4} {int(y[seg].sum()):>4}"
        )
    print()

# how much per-t event signal even exists on the COLD-START rows at each t?
print("per-t event-feature presence on COLD-START rows (the only signal they have):")
print(f"{'t':>4} {'has_quote':>10} {'outbound>0':>11} {'inbound>0':>10} {'n':>5}")
for t in (0, 7, 30):
    m = cold & (test.t == t).values
    hq = test.has_quote_by_t.values[m].mean()
    ob = (test.outbound_chars_by_t.values[m] > 0).mean()
    ib = (test.n_inbound_by_t.values[m] > 0).mean()
    print(f"{t:>4} {hq:>10.0%} {ob:>11.0%} {ib:>10.0%} {m.sum():>5}")
