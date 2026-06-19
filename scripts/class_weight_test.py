"""Does the positive-class weight matter for our RANKING score? Compare none / balanced / custom ratios."""

import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, "/Users/yonierlich/repos/challenge_1")
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score
from src.features import load_clean, build_panel, MODEL_FEATURES
from src.model import temporal_split, recompute_agent_rate

R = Path("/Users/yonierlich/repos/challenge_1")
subs, events = load_clean(R / "data")
panel = build_panel(subs, events)
train, test, _ = temporal_split(panel)
recompute_agent_rate(train, test)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)


def pipe(w):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight=w),
    )


weights = {
    "none": None,
    "balanced (~5.7x)": "balanced",
    "pos x3": {0: 1, 1: 3},
    "pos x6": {0: 1, 1: 6},
    "pos x10": {0: 1, 1: 10},
}
print(f"{'class_weight':18} {'train CV-AUC':>12} {'test AUC':>9} {'test PR-AUC':>12}")
for name, w in weights.items():
    cvauc = cross_val_score(
        pipe(w),
        train[MODEL_FEATURES],
        train.label,
        cv=cv,
        groups=train.submissionId,
        scoring="roc_auc",
    ).mean()
    m = pipe(w).fit(train[MODEL_FEATURES], train.label)
    p = m.predict_proba(test[MODEL_FEATURES])[:, 1]
    print(
        f"{name:18} {cvauc:>12.3f} {roc_auc_score(test.label, p):>9.3f} {average_precision_score(test.label, p):>12.3f}"
    )
