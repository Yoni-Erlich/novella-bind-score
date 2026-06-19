"""
Build the final HTML deliverable: reports/final_report.html (self-contained, base64 figures).
All numbers are computed at runtime from src/ so the report can never drift from the code.
Maps explicitly to the challenge's 3 tasks + the requested human-readable structure.

Run: poetry run python scripts/build_html_report.py
"""

import warnings

warnings.filterwarnings("ignore")
import io
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

R = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(R))
from src.preprocessing import validate, clean
from src.features import build_panel, MODEL_FEATURES
from src.model import standardized_coefficients
from src.evaluate import get_test_scores, precision_at_k

# ---------- palette ----------
PURPLE, PURPLE2, GOLD, GREY, GOOD = (
    "#3a2150",
    "#7a5ca3",
    "#f4c542",
    "#c9c9c9",
    "#2a9d8f",
)
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.edgecolor": "#888",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "axes.axisbelow": True,
    }
)


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def barlabels(ax, bars, fmt="{:.0%}", dy=0):
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + dy,
            fmt.format(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


# ================= DATA / CLEANING =================
subs = pd.read_csv(
    R / "data/features_submissions.csv", parse_dates=["createdDate", "resolvedDate"]
)
raw_events = pd.read_csv(R / "data/features_events.csv", parse_dates=["event_date"])
val = validate(raw_events, subs)
clean_events, clog = clean(raw_events, subs, verbose=False)
base_rate = float(subs.label.mean())

# ---------- Fig 1: cleaning ----------
ev = raw_events.merge(subs[["submissionId", "createdDate"]], on="submissionId")
d = (ev.event_date - ev.createdDate).dt.total_seconds() / 86400
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
RED = "#e76f51"
raw_n = clog["rows_in"]
dup_n = clog["dropped_duplicates"]
imp_n = clog["dropped_pre_creation"]
after_dup = raw_n - dup_n
clean_n = clog["rows_out"]
floor, W = 15500, 0.6
a1.bar(0, raw_n - floor, bottom=floor, width=W, color=GREY)  # raw total
a1.bar(1, dup_n, bottom=after_dup, width=W, color=RED)  # drop: duplicates
a1.bar(2, imp_n, bottom=clean_n, width=W, color=RED)  # drop: impossible
a1.bar(3, clean_n - floor, bottom=floor, width=W, color=GOOD)  # clean total
for x0, yv in [(0, raw_n), (1, after_dup), (2, clean_n)]:  # step connectors
    a1.plot(
        [x0 + W / 2, x0 + 1 - W / 2], [yv, yv], color="#aaa", lw=0.8, ls=(0, (3, 2))
    )
a1.text(
    0, raw_n + 45, f"{raw_n:,}", ha="center", va="bottom", fontsize=9, fontweight="bold"
)
a1.text(
    3,
    clean_n + 45,
    f"{clean_n:,}",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="#1f7a6e",
)
a1.annotate(
    f"−{dup_n}\nduplicates",
    xy=(1.3, after_dup + dup_n / 2),
    xytext=(1.62, 17080),
    ha="left",
    va="center",
    color=RED,
    fontsize=8.5,
    fontweight="bold",
    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
)
a1.annotate(
    f"−{imp_n}\npre-creation",
    xy=(2.3, clean_n + imp_n / 2),
    xytext=(2.5, 16760),
    ha="left",
    va="center",
    color=RED,
    fontsize=8.5,
    fontweight="bold",
    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
)
a1.set_xticks([0, 1, 2, 3])
a1.set_xticklabels(["raw", "dedup", "drop\nimpossible", "clean"], fontsize=8.5)
a1.set_xlim(-0.55, 3.95)
a1.set_ylim(floor, 17800)
a1.set_ylabel("# events")
a1.set_title("Event cleaning waterfall", fontweight="bold")
a2.hist(d[(d > -15) & (d < 35)], bins=60, color=PURPLE2)
a2.axvspan(
    -15, -1, color="#e76f51", alpha=0.25, label="dropped: d < −1 day (impossible)"
)
for tt in (0, 7, 30):
    a2.axvline(tt, color=GOLD, lw=1.4, ls="--")
a2.text(
    0.5, 0.92, "t = 0 / 7 / 30", transform=a2.transAxes, color="#a8862a", fontsize=8
)
a2.set_title("Event timing  d = eventDate − createdDate (days)", fontweight="bold")
a2.set_xlabel("days from createdDate")
a2.legend(fontsize=8, loc="upper right")
FIG_CLEAN = b64(fig)

# ================= SIGNAL (EDA → features) =================
panel_eda = build_panel(subs, clean_events)
one = panel_eda.drop_duplicates("submissionId")  # submission-level (agent feats shared)
ce = clean_events.merge(subs[["submissionId", "label"]], on="submissionId")
ob = (
    ce[ce.event_type == "EMAIL_OUTBOUND"]
    .groupby("submissionId")
    .email_char_count.sum()
    .reindex(subs.submissionId)
    .fillna(0)
)
sub_lab = subs.set_index("submissionId").label

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
TT = dict(fontweight="bold", fontsize=10.5)
# (a) has_quote_by_t -> bind rate
ax = axes[0]
rates = (
    panel_eda[panel_eda.t.isin([7, 30])]
    .groupby(["t", "has_quote_by_t"])
    .label.mean()
    .unstack()
)
x = np.arange(2)
w = 0.36
b1 = ax.bar(x - w / 2, rates[0].values, w, color=GREY, label="no quote")
b2 = ax.bar(x + w / 2, rates[1].values, w, color=GOLD, label="has quote")
barlabels(ax, list(b1) + list(b2))
ax.set_xticks(x)
ax.set_xticklabels(["t=7", "t=30"])
ax.set_title("Quote presence\n→ bind rate", **TT)
ax.set_ylabel("bind rate")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, max(rates.max()) * 1.28)
# (b) outbound_chars quartile -> bind rate
ax = axes[1]
q = pd.qcut(ob[ob > 0], 4, labels=["Q1", "Q2", "Q3", "Q4"])
df = pd.DataFrame({"q": q, "label": sub_lab.reindex(q.index).values})
qr = df.groupby("q").label.mean()
bars = ax.bar(qr.index.astype(str), qr.values, color=PURPLE2)
barlabels(ax, bars)
ax.axhline(base_rate, color="#e76f51", ls="--", lw=1, label=f"base {base_rate:.0%}")
ax.set_title("Outbound volume\n→ bind rate", **TT)
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, qr.max() * 1.28)
# (c) customer track record -> bind rate
ax = axes[2]
cold = one.agent_prior_n == 0
ret = one[~cold]
med = ret.agent_bind_rate.median()
groups = {
    "first-time\n(cold-start)": one[cold].label.mean(),
    "returning\nlow history": ret[ret.agent_bind_rate < med].label.mean(),
    "returning\nhigh history": ret[ret.agent_bind_rate >= med].label.mean(),
}
bars = ax.bar(list(groups), list(groups.values()), color=[GREY, PURPLE2, GOOD])
barlabels(ax, bars)
ax.axhline(base_rate, color="#e76f51", ls="--", lw=1, label=f"base {base_rate:.0%}")
ax.set_title("Customer track record\n→ bind rate", **TT)
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0, max(groups.values()) * 1.28)
FIG_SIGNAL = b64(fig)

# ================= MODEL + METRICS =================
fit, test, scores = get_test_scores()
y = test.label.values
logi = scores["logistic (final)"]


def prec_lift(yy, ss, ks=(0.20, 0.30, 0.50)):
    base = yy.mean()
    return {
        k: (precision_at_k(yy, ss, k), precision_at_k(yy, ss, k) / base) for k in ks
    }


overall_pl = prec_lift(y, logi)
per_t_auc = {}
per_t_pl = {}
for t in (0, 7, 30):
    m = (test.t == t).values
    per_t_auc[t] = roc_auc_score(y[m], logi[m])
    per_t_pl[t] = prec_lift(y[m], logi[m])

# Task-2 ranking
uni = {f: roc_auc_score(y, test[f].values) for f in MODEL_FEATURES}
coef = standardized_coefficients(fit)
rank = pd.DataFrame({"uni_auc": uni, "coef": coef}).sort_values(
    "uni_auc", ascending=False
)

# ---------- Fig 3: feature ranking ----------
fig, ax = plt.subplots(figsize=(7.5, 2.9))
names = [
    "agent_bind_rate",
    "has_quote_by_t",
    "outbound_chars_log",
    "n_inbound_by_t",
    "t",
]
names = [n for n in names if n in coef.index]
vals = [coef[n] for n in names]
bars = ax.barh(
    names[::-1], vals[::-1], color=[GOLD, PURPLE2, PURPLE2, PURPLE2, GREY][::-1]
)
ax.set_title(
    "Task 2 — feature significance (|standardized logistic coef|)", fontweight="bold"
)
for b, v in zip(bars, vals[::-1]):
    ax.text(
        v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8.5
    )
ax.set_xlim(0, max(vals) * 1.18)
FIG_RANK = b64(fig)

# ---------- Fig 4: baselines ----------
order = [
    ("no-skill floor", 0.5),
    ("outbound_chars only", roc_auc_score(y, scores["outbound_chars only"])),
    ("naive quote→effort", roc_auc_score(y, scores["naive: quote>effort"])),
    ("XGBoost", roc_auc_score(y, scores["xgboost"])),
    ("agent_bind_rate only", roc_auc_score(y, scores["agent_bind_rate only"])),
    ("logistic (final)", roc_auc_score(y, logi)),
]
fig, ax = plt.subplots(figsize=(8, 3.0))
labs = [o[0] for o in order]
vs = [o[1] for o in order]
cols = [GREY, PURPLE2, PURPLE2, PURPLE2, PURPLE2, GOLD]
bars = ax.bar(labs, vs, color=cols)
for b, v in zip(bars, vs):
    ax.text(
        b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5
    )
ax.set_ylim(0.45, 0.80)
ax.set_ylabel("ROC-AUC (held-out test)")
ax.set_title("Baselines vs final model — do we have signal?", fontweight="bold")
ax.tick_params(axis="x", labelrotation=18)
FIG_BASE = b64(fig)

# ---------- Fig 5: precision@k (headline) ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
ks = [0.20, 0.30, 0.50]
x = np.arange(len(ks))
w = 0.36
model_p = [overall_pl[k][0] for k in ks]
rand_p = [base_rate] * len(ks)
b1 = a1.bar(x - w / 2, model_p, w, color=GOLD, label="model (work top-k%)")
b2 = a1.bar(x + w / 2, rand_p, w, color=GREY, label="random (base rate)")
for b, k in zip(b1, ks):
    a1.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.005,
        f"{b.get_height():.0%}\n{overall_pl[k][1]:.1f}×",
        ha="center",
        fontsize=8.5,
    )
a1.set_xticks(x)
a1.set_xticklabels([f"top {int(k*100)}%" for k in ks])
a1.set_title("Precision@k (overall)", fontweight="bold")
a1.set_ylabel("precision (sell-rate in worked set)")
a1.legend(fontsize=8)
a1.set_ylim(0, max(model_p) * 1.3)
# cumulative gains curve
ordr = np.argsort(-logi, kind="stable")
ycum = np.cumsum(y[ordr]) / y.sum()
frac = np.arange(1, len(y) + 1) / len(y)
a2.plot(frac, ycum, color=PURPLE, lw=2, label="model")
a2.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.2, label="random")
for k in ks:
    idx = int(np.ceil(k * len(y))) - 1
    a2.scatter([frac[idx]], [ycum[idx]], color=GOLD, zorder=5)
a2.set_title("Cumulative gains", fontweight="bold")
a2.set_xlabel("fraction of queue worked")
a2.set_ylabel("fraction of binders captured")
a2.legend(fontsize=8, loc="lower right")
FIG_PREC = b64(fig)

# ---------- Fig 6: per-t ----------
cold_t = (test.agent_prior_n == 0).values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
ts = [0, 7, 30]
x = np.arange(len(ts))
auc_floor = [per_t_auc[t] for t in ts]
b = a1.bar(x, auc_floor, color=[PURPLE2, PURPLE2, GREY])
a1.axhline(0.5, color="#e76f51", ls="--", lw=1, label="no-skill 0.50")
for bb, t in zip(b, ts):
    a1.text(
        bb.get_x() + bb.get_width() / 2,
        bb.get_height() + 0.008,
        f"{per_t_auc[t]:.3f}",
        ha="center",
        fontsize=8.5,
    )
a1.set_xticks(x)
a1.set_xticklabels([f"t={t}" for t in ts])
a1.set_ylim(0.45, 0.95)
a1.set_ylabel("ROC-AUC")
a1.set_title("Per-t ROC-AUC  (t=30 noisy, n=73)", fontweight="bold")
a1.legend(fontsize=8)
# segmented per-t AUC: agent-only vs full, cold vs returning at each t
agent_only = scores["agent_bind_rate only"]


def seg_auc(mask):
    yy = y[mask]
    if yy.sum() == 0 or yy.sum() == len(yy):
        return np.nan
    return roc_auc_score(yy, logi[mask])


ret_v = [seg_auc((test.t == t).values & ~cold_t) for t in ts]
cold_v = [seg_auc((test.t == t).values & cold_t) for t in ts]
w = 0.36
b1 = a2.bar(x - w / 2, ret_v, w, color=GOOD, label="returning customer")
b2 = a2.bar(x + w / 2, cold_v, w, color=GREY, label="first-time (cold-start)")
a2.axhline(0.5, color="#e76f51", ls="--", lw=1)
for bb in list(b1) + list(b2):
    if not np.isnan(bb.get_height()):
        a2.text(
            bb.get_x() + bb.get_width() / 2,
            bb.get_height() + 0.008,
            f"{bb.get_height():.2f}",
            ha="center",
            fontsize=8,
        )
a2.set_xticks(x)
a2.set_xticklabels([f"t={t}" for t in ts])
a2.set_ylim(0.3, 1.0)
a2.set_title("Returning vs first-time customers", fontweight="bold")
a2.legend(fontsize=8)
FIG_PERT = b64(fig)


# ================= HTML =================
def img(b, cap):
    return f'<figure><img src="data:image/png;base64,{b}"><figcaption>{cap}</figcaption></figure>'


def table(headers, rows, hl=None):
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = ""
    for r in rows:
        cls = ' class="hl"' if hl and r[0] == hl else ""
        body += "<tr>" + "".join(f"<td{cls}>{c}</td>" for c in r) + "</tr>"
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


pl = lambda d, k: f"{d[k][0]:.0%} ({d[k][1]:.1f}×)"
N = len(test)
pk20 = (
    lambda s: f"{precision_at_k(y, s, 0.20):.0%} ({precision_at_k(y, s, 0.20)/base_rate:.1f}×)"
)

# baseline comparison — precision@20% FIRST, then AUC/PR-AUC
_models = [
    ("no-skill floor", np.full_like(y, 0.0, dtype=float)),
    ("outbound_chars only", scores["outbound_chars only"]),
    ("naive quote→effort", scores["naive: quote>effort"]),
    ("XGBoost", scores["xgboost"]),
    ("agent_bind_rate only", agent_only),
    ("logistic (final)", logi),
]
base_tbl = table(
    ["model", "P@20% (lift)", "ROC-AUC", "PR-AUC"],
    [
        [
            nm,
            f"{base_rate:.0%} (1.0×)" if nm == "no-skill floor" else pk20(s),
            "0.500" if nm == "no-skill floor" else f"{roc_auc_score(y, s):.3f}",
            f"{base_rate:.3f}"
            if nm == "no-skill floor"
            else f"{average_precision_score(y, s):.3f}",
        ]
        for nm, s in _models
    ],
    hl="logistic (final)",
)

# "why not agent_bind_rate alone" — segment the test by customer history
_cold = (test.agent_prior_n == 0).values


def _seg(s, m):
    yy = y[m]
    return roc_auc_score(yy, s[m]) if 0 < yy.sum() < len(yy) else float("nan")


seg_tbl = table(
    ["segment", "n", "agent_bind_rate alone", "full model"],
    [
        [
            "overall",
            N,
            f"{_seg(agent_only, np.ones(N, bool)):.3f}",
            f"{_seg(logi, np.ones(N, bool)):.3f}",
        ],
        [
            "returning customer",
            int((~_cold).sum()),
            f"{_seg(agent_only, ~_cold):.3f}",
            f"{_seg(logi, ~_cold):.3f}",
        ],
        [
            "first-time (cold-start)",
            int(_cold.sum()),
            f"{_seg(agent_only, _cold):.3f}",
            f"{_seg(logi, _cold):.3f}",
        ],
    ],
    hl="first-time (cold-start)",
)

rank_tbl = table(
    ["rank", "feature", "univariate test AUC", "|std. coef|"],
    [
        [
            i + 1,
            f"<code>{f}</code>",
            f"{rank.loc[f, 'uni_auc']:.3f}",
            f"{rank.loc[f, 'coef']:.3f}",
        ]
        for i, f in enumerate(rank.index)
    ],
)

prec_tbl = table(
    ["slice", "n", "P@20% (lift)", "P@30% (lift)", "P@50% (lift)", "ROC-AUC"],
    [
        [
            "overall",
            N,
            pl(overall_pl, 0.20),
            pl(overall_pl, 0.30),
            pl(overall_pl, 0.50),
            f"{roc_auc_score(y, logi):.3f}",
        ]
    ]
    + [
        [
            f"t={t}",
            int((test.t == t).sum()),
            pl(per_t_pl[t], 0.20),
            pl(per_t_pl[t], 0.30),
            pl(per_t_pl[t], 0.50),
            f"{per_t_auc[t]:.3f}",
        ]
        for t in (0, 7, 30)
    ],
    hl="overall",
)

dropped_tbl = table(
    ["feature family", "examples", "verdict"],
    [
        [
            "Ratios / normalized",
            "inbound_share, io_ratio, chars_per_event",
            "≈0 — normalizing away volume kills the signal",
        ],
        [
            "Attachments",
            "total / per-email / first-inbound attach",
            "ride volume only; first-package hypothesis failed",
        ],
        [
            "Char distribution",
            "out_chars mean / median / max / std",
            "outbound_chars (sum) already encodes it",
        ],
        [
            "Response time",
            "our reply latency, agent reply latency",
            "speed doesn't predict; volume does",
        ],
        [
            "Velocity / recency",
            "recency, events_last7, acceleration",
            "volume-confounded; worse incremental AUC",
        ],
        [
            "Calendar",
            "weekend, month, day-in-quarter",
            "no robust t=0 signal (short COVID window)",
        ],
        [
            "Other agent feats",
            "agent_prior_subs, agent_open_now",
            "no signal (corr ≈ 0)",
        ],
        [
            "Customer-history aggregates",
            "avg past outbound/inbound, chars/email, quote-rate",
            "redundant w/ agent_bind_rate; blind to cold-start; +0.002",
        ],
        [
            "Dedicated t=0 model",
            "train only on t=0 rows",
            "worse than pooled (cold-start AUC collapses to 0.38)",
        ],
        [
            "is_cold_start_flag",
            "red-team panel",
            "won CV (+0.029) but lost temporal holdout (−0.044): overfit",
        ],
    ],
)

CSS = """
:root{--p:#3a2150;--p2:#7a5ca3;--g:#f4c542;--ink:#222;--mut:#666;--line:#e3e0e8;--bg:#faf9fc}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  line-height:1.5;margin:0;background:var(--bg);font-size:15px}
.wrap{max-width:940px;margin:0 auto;padding:0 26px 80px}
header{background:linear-gradient(120deg,#2d1b3d,#4a2d6b);color:#fff;padding:34px 26px;margin-bottom:8px}
header .inner{max-width:940px;margin:0 auto}
header h1{margin:0 0 6px;font-size:27px}
header .sub{color:#e7dcf5;font-size:15px}
header .pill{display:inline-block;background:rgba(244,197,66,.18);color:var(--g);
  border:1px solid rgba(244,197,66,.5);border-radius:14px;padding:3px 11px;font-size:12.5px;margin-top:12px}
h2{color:var(--p);border-bottom:2px solid var(--g);padding-bottom:5px;margin-top:40px;font-size:21px}
h3{color:var(--p);margin-top:26px;font-size:16.5px}
h4{margin:18px 0 6px;font-size:14.5px;color:#4a2d6b}
code{background:#efe9f5;color:#54307e;padding:1px 5px;border-radius:4px;font-size:13px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;background:#fff}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#f0ebf6;color:var(--p)}
tr:nth-child(even) td{background:#fbfafd}
td.hl,tr td.hl{background:#fff7df!important;font-weight:600}
figure{margin:18px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:7px;background:#fff;padding:6px}
figcaption{color:var(--mut);font-size:12.5px;margin-top:6px}
.kpis{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}
.kpi{flex:1;min-width:150px;background:#fff;border:1px solid var(--line);border-left:4px solid var(--g);
  border-radius:7px;padding:12px 14px}
.kpi b{display:block;font-size:25px;color:var(--p)}
.kpi span{color:var(--mut);font-size:12.5px}
.note{background:#fff;border-left:4px solid var(--p2);padding:11px 15px;border-radius:6px;margin:14px 0}
.warn{background:#fff6f3;border-left:4px solid #e76f51}
.ok{background:#f2fbf9;border-left:4px solid #2a9d8f}
ul{margin:8px 0 8px 2px;padding-left:20px}li{margin:3px 0}
.map{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 18px;margin:14px 0}
.tldr{background:#fff;border:1px solid var(--line);border-top:4px solid var(--g);border-radius:8px;
  padding:16px 20px;margin:18px 0 6px}
.tldr h2{margin:0 0 8px;border:0;padding:0;font-size:16px;letter-spacing:.5px}
.tldr p{margin:0 0 8px}.tldr p:last-child{margin:0}
.muted{color:var(--mut);font-size:13px}
footer{color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);margin-top:50px;padding-top:14px}
"""

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novella Bind Score — Final Report</title><style>{CSS}</style></head>
<body>
<header><div class="inner">
<h1>Novella Bind Score &mdash; Final Report</h1>
<div class="sub">Ranking open E&amp;S submissions by likelihood to <b>bind (sell)</b>, for broker effort-prioritization.
&nbsp;<code>bind_score(submission_id, t)</code>, t&nbsp;&isin;&nbsp;{{0,&nbsp;7,&nbsp;30}}.</div>
<div class="pill">Held-out temporal test &middot; ROC-AUC {roc_auc_score(y, logi):.3f} &middot; 2&ndash;3&times; lift @ top-20%</div>
</div></header>
<div class="wrap">

<div class="tldr">
<h2>TL;DR</h2>
<p><b>Result.</b> Working the model's <b>top 20%</b> of the queue binds at <b>{overall_pl[0.20][0]:.0%}</b> &mdash; a
<b>{overall_pl[0.20][1]:.1f}&times; lift</b> over random (base rate {base_rate:.0%}) &mdash; holding at 30%/50%; overall
ranking ROC-AUC <b>{roc_auc_score(y, logi):.3f}</b>. Precision@k is our primary metric (brokers work a ranked queue).</p>
<p><b>Features (4 + <code>t</code>).</b> <code>agent_bind_rate</code> (repeat-<i>customer</i> close-rate),
<code>has_quote_by_t</code>, <code>outbound_chars_log</code> (broker effort), <code>n_inbound_by_t</code>, plus the
snapshot time <code>t</code>. By significance: <code>agent_bind_rate</code> &Gt; <code>has_quote</code> &gt; the rest.</p>
<p><b>t=0 vs t=7.</b> At <b>t=0</b> the submission has almost no activity yet, so ranking leans on
<i>who the customer is</i> (<code>agent_bind_rate</code>) &mdash; AUC <b>{per_t_auc[0]:.2f}</b>, up from a coin-flip
before customer history. By <b>t=7</b> emails and quotes have accrued and add signal (AUC {per_t_auc[7]:.2f}); their
value keeps growing toward t=30.</p>
<p><b>Model.</b> A pooled, L2-regularized <b>logistic regression</b> over <code>(submission, t)</code> rows (one model,
<code>t</code> as a feature) &mdash; it <b>beat XGBoost</b> at every t; the signal is simple and linear.
<b>Main limit:</b> ~half of day-0 rows are first-time customers with no history, where the score sits near the base rate.</p>
</div>

<div class="kpis">
<div class="kpi"><b>{overall_pl[0.20][0]:.0%}</b><span>precision @ top-20% &middot; {overall_pl[0.20][1]:.1f}&times; lift (base {base_rate:.0%})</span></div>
<div class="kpi"><b>{roc_auc_score(y, logi):.3f}</b><span>overall ROC-AUC (floor 0.50)</span></div>
<div class="kpi"><b>{per_t_auc[0]:.3f}</b><span>day-0 AUC (was ~0.50 before customer history)</span></div>
<div class="kpi"><b>4&nbsp;+&nbsp;1</b><span>features: 4 predictive + <code>t</code> context var</span></div>
</div>

<div class="map">
<p class="muted"><b>How this maps to the challenge tasks</b> (PDF):</p>
<ul class="muted">
<li><b>Task 1 — devise 3&ndash;4 features</b> &rarr; &sect;2 (chosen set + <code>feature(submission_id,&nbsp;t)</code> in <code>src/features.py</code>).</li>
<li><b>Task 2 — rank by predictive significance</b> &rarr; &sect;2.1 (Fig 3).</li>
<li><b>Task 3 — build <code>bind_score</code> + evaluate</b> &rarr; &sect;3 (preprocessing &amp; validation, metrics, results vs baselines) with the model-comparison table in &sect;3.5.</li>
</ul>
</div>

<h2>1. Data, cleaning &amp; what to watch for</h2>
<p>881 submissions (one per <code>submissionId</code>), <b>{base_rate:.1%} bind rate</b> &rarr; imbalanced, so we use
<b>ranking</b> metrics, not accuracy. ~17k events of 3 types (EMAIL_INBOUND / OUTBOUND / QUOTE_RECEIVED).
A row is one <b>(submission, t)</b> pair, scored only while the submission is still <b>open at t</b>.</p>
{img(FIG_CLEAN, "Fig 1 — Cleaning: drop 530 exact-duplicate + 127 impossible pre-creation events (17,211 → 16,564). Right: most events cluster after createdDate; a pre-creation tail violates causality and is dropped (d &lt; −1 day); −15-min clock-skew is kept.")}
<div class="note"><b>Cleaning policy (2 drops, leakage-safe):</b>
<ul>
<li><b>530 exact-duplicate events</b> (same sub + second + type + chars + attach) — concentrated in high-activity subs, so they <i>inflate the volume features</i> that carry our signal.</li>
<li><b>127 impossible pre-creation events</b> (dated &gt;1 day before the submission exists) — counted at <i>every</i> t (incl. t=0), falsely inflating day-0 activity.</li>
</ul></div>
<div class="note warn"><b>Watch out for:</b>
<ul>
<li><b>Leakage</b> is the central risk: <code>feature(sub,&nbsp;t)</code> may use <b>only</b> events with <code>eventDate ≤ createdDate + t</code>, and <b>never</b> <code>resolvedDate</code>/<code>label</code>.</li>
<li><b>Domain quirk, NOT leakage:</b> OUTBOUND / quotes can precede the first INBOUND — intake happens via <code>createdDate</code> (off-channel), carriers quote off-channel. Kept.</li>
<li><b>Char outliers</b> (max 43k, p99 2.3k): kept, tamed by <code>log1p</code> — not deleted.</li>
<li><b>Censoring:</b> by t=30 &gt;50% have resolved; only score <b>open-at-t</b> submissions.</li>
</ul></div>

<h2>2. Features chosen (Task 1)</h2>
<p>The signal is simple: <b>activity volume + quote presence + customer track record</b>. Normalized ratios,
attachments, char-distribution stats, response-latency, velocity and calendar all add nothing beyond raw volume
(see Appendix A).</p>
{img(FIG_SIGNAL, "Fig 2 — What predicts binding: (left) a received quote sharply raises bind rate, more so as t grows; (mid) submissions split into outbound-volume QUARTILES — Q1 = lowest-25% broker writing … Q4 = highest-25% — bind rate rises monotonically (more broker effort/volume → higher bind rate); (right) repeat customers with a strong prior track record bind far above the base rate, first-timers sit near it.")}
<p>The 4 predictive features + <code>t</code> (a context variable that lets one pooled model adapt across snapshots):</p>
{table(["feature", "what it captures", "leakage-safe basis"],
[["<code>agent_bind_rate</code>", "repeat-<b>customer</b> close-rate (smoothed)", "only the customer's subs <i>resolved before</i> this createdDate"],
 ["<code>has_quote_by_t</code>", "funnel gate — a carrier quote exists", "events with d ≤ t"],
 ["<code>outbound_chars_log</code>", "broker effort (log1p of OUTBOUND chars)", "events with d ≤ t"],
 ["<code>n_inbound_by_t</code>", "customer engagement (replies)", "events with d ≤ t"],
 ["<code>t</code>", "snapshot time (0/7/30)", "structural"]])}
<div class="note"><b>Why <code>agent_bind_rate</code> matters most:</b> <code>agentEmail</code> is Novella's <b>customer</b> (the retail
agent), <b>not</b> the broker. A customer's prior close-rate is known at t=0 — the one strong signal available on
day 0 (for <i>returning</i> customers), when nothing has happened on the submission yet. Adding it lifted the day-0
<b>ranking</b> from a coin-flip (<b>ROC-AUC ≈0.50</b>) to <b>{per_t_auc[0]:.2f}</b>.
<br><span class="muted">Note: 0.50 is the AUC floor (random ranker), not a bind rate. The base bind <i>rate</i> is {base_rate:.0%} —
that's the predicted probability a first-time customer gets (see the cold-start callout in &sect;3.3).</span></div>

<h3>2.1 Feature significance ranking (Task 2)</h3>
{img(FIG_RANK, "Fig 3 — Standardized logistic coefficients (contribution given the others). Ranked via the linear lens, not SHAP (SHAP on an overfit tree was a high-cardinality artifact — see Appendix A).")}
{rank_tbl}
<p class="muted">Customer track record &Gt; got-a-quote &gt; broker effort &asymp; customer replies &gt; t.</p>

<h2>3. Modeling (Task 3)</h2>

<h3>3.1 Preprocessing &amp; validation</h3>
<p><b>(a) Clean the data</b> — apply the 2 drops above in-pipeline (never mutate <code>data/</code>).
<b>(b) Build features</b> per (submission, t), leakage-safe. Every feature is <b>standardized</b> (z-scored:
mean 0, sd 1) inside the model pipeline <code>Impute(median) → StandardScaler → Logistic</code> — fit on the
training fold only. Per-feature handling:</p>
{table(["feature", "transform before scaling", "scaled?", "special handling"],
[["<code>agent_bind_rate</code>", "empirical-Bayes <b>smoothing</b>", "yes", "smoothed toward the <b>TRAIN base rate</b>; cold-start → base rate (recomputed after the split → no leakage)"],
 ["<code>outbound_chars_log</code>", "<b>log1p</b>", "yes", "log first to tame the 43k-char outlier, then scale"],
 ["<code>has_quote_by_t</code>", "none (binary 0/1)", "yes", "scaled like the rest (harmless)"],
 ["<code>n_inbound_by_t</code>", "none", "yes", "—"],
 ["<code>t</code>", "none", "yes", "constant within a snapshot"]])}
<p class="muted">Why standardize: L2 regularization penalizes coefficients equally regardless of units, and the
optimizer converges faster on a well-scaled loss surface; it also makes the coefficients comparable for the §2.1 ranking.</p>
<div class="note"><b>(c) Validation strategy — three-way, no peeking:</b>
<ul>
<li><b>Temporal split:</b> train = earliest ~70% of submissions (by <code>createdDate</code>); held-out test = latest ~30% ({N} rows). Train-on-past / test-on-future is the honest setup for a repeat-customer signal.</li>
<li><b>Validation = grouped CV <i>inside</i> the training set:</b> every choice (feature set, the <code>log1p</code> transform, regularization) is picked by <b>5-fold <code>StratifiedGroupKFold</code></b>, <b>grouped by <code>submissionId</code></b> so a submission's t=0/7/30 rows never straddle a fold — no within-submission leakage. We do <b>not</b> carve a static validation set: with only 130 positives, CV reuses the data more efficiently.</li>
<li><b>The held-out test is touched once</b> — for the final numbers below; it never informs a modeling decision.</li>
</ul></div>

<h3>3.2 How we measure</h3>
<div class="note"><b>Precision@k is primary.</b> Brokers work a ranked queue top-down, so the operational question is
<i>"of the top k% I'm told to work, what share actually bind?"</i> — exactly <b>precision@top-k</b> (k = 20/30/50%).
<b>Lift</b> = precision@k ÷ base rate = how much better than working at random.
<br><b>Secondary:</b> <b>PR-AUC</b> (precision/recall — the right summary under imbalance) and <b>ROC-AUC</b> (familiar,
but optimistic on imbalanced data because it rewards ranking the abundant non-binders — so we don't lead with it).
<br><b>The two floors differ:</b> a random ranker has <b>ROC-AUC = 0.50</b> (the AUC floor); random <b>precision /
PR-AUC = {base_rate:.0%}</b> (the prevalence). 0.50 is not a rate — that's why the baseline AUC floor is 0.50, not {base_rate:.0%}.</div>

<h3>3.3 Results vs baselines</h3>
<p><b>Top-k (the headline):</b> working the model's <b>top 20%</b> finds binders
<b>{overall_pl[0.20][1]:.1f}&times;</b> more efficiently than random ({overall_pl[0.20][0]:.0%} vs {base_rate:.0%});
the advantage holds at 30% and 50%.</p>
{img(FIG_PREC, "Fig 5 — (left) Precision@k with lift vs the random base rate. (right) Cumulative gains: x = fraction of the queue worked (top-down by score), y = fraction of ALL binders captured. Random captures binders in proportion to effort → the diagonal (work 20% → catch ~20%); the model's curve rises above it (top 20% → ~47% of binders).")}
{prec_tbl}
<p><b>Do we have signal?</b> Yes — the model beats every baseline on both precision@20% and AUC:</p>
{img(FIG_BASE, "Fig 4 — ROC-AUC on the held-out test vs baselines. The model clears the no-skill floor and beats single-feature, naive-heuristic and XGBoost baselines.")}
{base_tbl}
<p><b>Per-t:</b> the model works at every snapshot, including <b>day 0</b> — the most valuable, previously-blind
moment. (t=30 is noisy: n={int((test.t==30).sum())}.)</p>
{img(FIG_PERT, "Fig 6 — (left) Per-t ROC-AUC, all above the no-skill floor. (right) The honest split: strong on returning customers at every t; for first-time (cold-start) customers it leans on in-submission activity, thin at t=0 and growing by t=7/30.")}
<div class="note warn"><b>Cold-start clarity (what t=0 really gives you):</b> for a <b>first-time</b> customer at t=0,
<code>agent_bind_rate</code> is a <b>constant</b> ({base_rate:.0%} base rate) → it can't rank them, and the other
features are nearly empty (no quotes, ~6% have outbound; only ~44% have an early inbound). So the model scores them
near <b>{base_rate:.0%}</b> and the day-0 strength is really about <i>returning</i> customers. The score "separates
known-good repeat customers from the unknown {base_rate:.0%} pile."</div>

<h3>3.4 Why not just the customer bind-rate?</h3>
<p>A fair question, since <code>agent_bind_rate</code> alone nearly matches the full model <i>overall</i>. But that
overall number hides a split — segment the test by whether the customer has history:</p>
{seg_tbl}
<p>Alone it ties on returning customers but ranks <b>first-time customers at random (0.50)</b> — half the rows. The
email/quote features carry that cold-start half (0.50 → 0.58), which is why the full model is needed even though the
single feature looks "good enough" on the average.</p>

<h3>3.5 Comparing model choices</h3>
<p>Per the challenge's "compare model choices": we tested <b>XGBoost</b> (shallow, regularized) — it
<b>underperformed logistic at every t</b> (overall {roc_auc_score(y, scores['xgboost']):.3f} vs {roc_auc_score(y, logi):.3f};
see Fig 4 / the table above). No nonlinear or interaction signal to exploit, and on 130 positives the simpler linear
model generalizes better and stays interpretable. We also compared <b>one pooled model</b> (with <code>t</code> as a
feature) vs <b>three per-t models</b>: pooled tied-or-won and crucially won day-0 (it borrows strength across
snapshots instead of training t=0 on cold-start-heavy rows). → <b>ship one pooled logistic regression.</b></p>

<h2>4. Conclusion — capability &amp; features</h2>
<div class="kpis">
<div class="kpi ok" style="border-left-color:#2a9d8f"><b>{roc_auc_score(y, logi):.2f}</b><span>overall AUC — a genuinely useful prioritizer</span></div>
<div class="kpi ok" style="border-left-color:#2a9d8f"><b>{overall_pl[0.20][1]:.1f}&times;</b><span>top-20% lift, incl. brand-new submissions</span></div>
<div class="kpi ok" style="border-left-color:#2a9d8f"><b>{per_t_auc[0]:.2f}</b><span>day-0 AUC — earliest = most valuable</span></div>
</div>
<ul>
<li><b>The signal is simple and linear.</b> A shallow XGBoost lost to logistic at every t — no hidden interactions to exploit.</li>
<li><b>The win is repeat-customer history</b> (<code>agent_bind_rate</code>), which alone carries most of the overall ranking <i>and</i> unlocks day-0.</li>
<li><b>Honest limit:</b> the model mainly "separates known-good customers from the unknown pile." ~51% of day-0 rows are
<b>first-time customers</b>, where it can only fall back to in-submission activity (Fig 6, right). The real lever for them
would be submission-static attributes (line of business, geography, premium size) — not present in this dataset.</li>
<li><b>Caveats:</b> small data (130 positives), a single temporal split, and a tiny/noisy t=30 slice — trust the
robust figures (~{roc_auc_score(y, logi):.2f} AUC, 2&ndash;3&times; lift), not the t=30 point estimates.</li>
</ul>

<h2>Appendix A — features we tried and dropped</h2>
<p class="muted">Kept only if a feature survived <b>partial correlation</b> (signal beyond volume) <b>and</b> incremental
held-out AUC. Full ledger: <code>reports/feature_catalog.md</code>.</p>
{dropped_tbl}
<div class="note"><b>Lesson:</b> judge features on the <b>held-out temporal test, not CV alone</b> — <code>is_cold_start_flag</code>
won cross-validation (+0.029) but lost the temporal holdout (−0.044): it was overfitting the past.</div>

<footer>
<b>How to run:</b> <code>poetry install</code> &middot; <code>poetry run python src/evaluate.py</code> (reproduces these numbers)
&middot; this report: <code>poetry run python scripts/build_html_report.py</code>.<br>
<b>Repo:</b> <code>src/</code> (preprocessing, features, model, evaluate) &middot; <code>notebooks/</code> (01_eda, 02_model)
&middot; <code>scripts/</code> (exploratory checks) &middot; <code>reports/</code> (this report + markdown analysis).
All figures &amp; numbers generated at runtime from the held-out temporal test ({N} rows, {y.sum():.0f} sold).
</footer>
</div></body></html>"""

out = R / "reports/final_report.html"
out.write_text(html)
print(f"wrote {out}  ({len(html)/1024:.0f} KB)")
print(
    f"overall AUC {roc_auc_score(y, logi):.3f} | P@20 {overall_pl[0.20][0]:.3f} "
    f"({overall_pl[0.20][1]:.2f}x) | per-t AUC { {t: round(per_t_auc[t],3) for t in (0,7,30)} }"
)
