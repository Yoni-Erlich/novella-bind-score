"""
Question: why not use agent_bind_rate ALONE? Do we have proof more features help?

Proof strategy: overall AUC hides the answer (agent-alone ~= full model overall).
The truth shows up when you SEGMENT the held-out temporal test by whether the
customer has prior history at score time:
  - returning  (agent_prior_n >= 1): agent_bind_rate is informative
  - cold-start (agent_prior_n == 0): agent_bind_rate is a CONSTANT (= train base rate)
                                     -> it ranks these at random (AUC 0.50)
The full model still ranks the cold-start segment, which is the majority of rows.
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
    y = np.asarray(y)
    s = np.asarray(s)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")  # AUC undefined with one class
    return roc_auc_score(y, s)


fit, test, scores = get_test_scores()
full = scores["logistic (final)"]
agent_only = scores["agent_bind_rate only"]
y = test.label.values

cold = (test.agent_prior_n == 0).values  # no resolved history at createdDate
ret = ~cold

print(f"held-out temporal test: {len(test)} rows, {int(y.sum())} sold ({y.mean():.1%})")
print(
    f"  cold-start rows : {cold.sum():4d} ({cold.mean():.0%})  sold={int(y[cold].sum())}"
)
print(
    f"  returning rows  : {ret.sum():4d} ({ret.mean():.0%})  sold={int(y[ret].sum())}\n"
)

print(f"{'segment':16s} {'n':>5} {'agent_bind_rate ONLY':>22} {'FULL model':>12}")
print(
    f"{'overall':16s} {len(test):>5d} {auc(y, agent_only):>22.3f} {auc(y, full):>12.3f}"
)
print(
    f"{'returning only':16s} {ret.sum():>5d} {auc(y[ret], agent_only[ret]):>22.3f} {auc(y[ret], full[ret]):>12.3f}"
)
print(
    f"{'cold-start only':16s} {cold.sum():>5d} {auc(y[cold], agent_only[cold]):>22.3f} {auc(y[cold], full[cold]):>12.3f}"
)

# show agent_bind_rate is literally constant on cold-start (hence AUC 0.50)
u = np.unique(np.round(agent_only[cold], 6))
print(
    f"\nagent_bind_rate on cold-start rows: {len(u)} distinct value(s) = {u}  "
    f"(constant -> cannot rank -> AUC 0.50)"
)
