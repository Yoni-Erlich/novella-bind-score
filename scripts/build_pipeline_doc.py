"""
Build reports/pipeline_and_model.html — a study & interview companion:
  (1) full pipeline raw-data -> model, stage by stage + the module responsible, with heuristics
  (2) deep-dive: why normalize? (optimization geometry — illustrated)
  (3) deep-dive: logistic regression loss function + interview prep

Self-contained (base64 figures). Run: poetry run python scripts/build_pipeline_doc.py
"""

import warnings

warnings.filterwarnings("ignore")
import io
import base64
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from src.preprocessing import clean
from src.features import load_clean, build_panel
from src.model import temporal_split

PURPLE, PURPLE2, GOLD, GREY, RED = "#3a2150", "#7a5ca3", "#f4c542", "#c9c9c9", "#e76f51"
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.color": "#ededed",
        "axes.axisbelow": True,
    }
)


def b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---- a few real numbers to ground the doc ----
subs, raw = load_clean(R / "data")  # raw here is already cleaned events
import pandas as pd

raw_events = pd.read_csv(R / "data/features_events.csv", parse_dates=["event_date"])
_, clog = clean(raw_events, subs, verbose=False)
panel = build_panel(subs, raw)
tr, te, thr = temporal_split(panel)
PC = {int(t): int((panel.t == t).sum()) for t in (0, 7, 30)}
N_PANEL, N_TR, N_TE = len(panel), len(tr), len(te)
BASE = float(subs.label.mean())


# ================= FIG A: optimization geometry =================
def loss_grid(h1, h2, lim=10):
    g = np.linspace(-lim, lim, 240)
    X, Y = np.meshgrid(g, g)
    return X, Y, 0.5 * (h1 * X**2 + h2 * Y**2)


def gd(h1, h2, w0, lr, n):
    w = np.array(w0, float)
    P = [w.copy()]
    for _ in range(n):
        w = w - lr * np.array([h1 * w[0], h2 * w[1]])
        P.append(w.copy())
    return np.array(P)


figA, (L, Rr) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
X, Y, Z = loss_grid(3, 4)
L.contour(X, Y, Z, levels=18, colors="#d8cbe8", linewidths=0.8)
P = gd(3, 4, (-8.5, 9), 0.16, 40)
L.plot(P[:, 0], P[:, 1], "-o", color=PURPLE, ms=3, lw=1.4)
L.scatter([0], [0], color=GOLD, s=160, marker="*", zorder=5, edgecolor="#a8862a")
L.set_title(
    "Normalized → round bowl\nκ≈1: straight, few steps", fontweight="bold", fontsize=11
)
L.set_xlabel("weight w₁"), L.set_ylabel("weight w₂")
X, Y, Z = loss_grid(1, 34)
Rr.contour(X, Y, Z, levels=18, colors="#d8cbe8", linewidths=0.8)
P = gd(1, 34, (-8.5, 8), 0.057, 40)
Rr.plot(P[:, 0], P[:, 1], "-o", color=RED, ms=3, lw=1.4)
Rr.scatter([0], [0], color=GOLD, s=160, marker="*", zorder=5, edgecolor="#a8862a")
Rr.set_title(
    "Un-normalized → elongated valley\nκ≈34: zig-zags, slow / may not converge",
    fontweight="bold",
    fontsize=11,
)
(
    Rr.set_xlabel("weight w₁ (large-scale feature)"),
    Rr.set_ylabel("weight w₂ (small-scale feature)"),
)
FIG_GEO = b64(figA)

# ================= FIG B: sigmoid saturation =================
z = np.linspace(-10, 10, 500)
s = 1 / (1 + np.exp(-z))
figB, ax = plt.subplots(figsize=(7.8, 3.3))
ax.plot(z, s, color=PURPLE, lw=2.2, label="σ(z) = 1/(1+e⁻ᶻ)")
ax.plot(
    z, 4 * s * (1 - s), color=GOLD, lw=2, ls="--", label="σ′(z)·4  (gradient signal)"
)
ax.axvspan(-10, -5, color=RED, alpha=0.12)
ax.axvspan(5, 10, color=RED, alpha=0.12, label="saturated: gradient ≈ 0")
(
    ax.set_xlabel("z = wᵀx + b"),
    ax.set_ylim(-0.05, 1.05),
    ax.legend(fontsize=8.5, loc="center left"),
)
ax.set_title(
    "Sigmoid saturation — a huge unscaled wᵀx lands here → no learning signal",
    fontweight="bold",
    fontsize=10.5,
)
FIG_SIG = b64(figB)

# ================= FIG C: log-loss =================
p = np.linspace(0.001, 0.999, 500)
figC, ax = plt.subplots(figsize=(7.8, 3.3))
ax.plot(p, -np.log(p), color=PURPLE, lw=2.2, label="true label y=1:  −log(p)")
ax.plot(p, -np.log(1 - p), color=GOLD, lw=2.2, label="true label y=0:  −log(1−p)")
(
    ax.set_ylim(0, 6),
    ax.set_xlabel("predicted P(bind) = p"),
    ax.set_ylabel("loss for one example"),
)
ax.legend(fontsize=9)
ax.set_title(
    "Log-loss (cross-entropy): confident-and-wrong is punished without bound",
    fontweight="bold",
    fontsize=10.5,
)
FIG_LOSS = b64(figC)


# ================= HTML =================
def img(b, cap):
    return f'<figure><img src="data:image/png;base64,{b}"><figcaption>{cap}</figcaption></figure>'


def flow(stages):
    out = '<div class="flow">'
    for i, (mod, title, desc) in enumerate(stages):
        out += (
            f'<div class="stage"><span class="mod">{mod}</span>'
            f"<b>{title}</b><p>{desc}</p></div>"
        )
        if i < len(stages) - 1:
            out += '<div class="arrow">▼</div>'
    return out + "</div>"


PIPE = flow(
    [
        (
            "data/*.csv",
            "Raw data",
            "<code>features_submissions.csv</code> (881 subs: createdDate, resolvedDate, agentEmail, label) + "
            "<code>features_events.csv</code> (~17k events: type, date, char_count, attach_count).",
        ),
        (
            "src/preprocessing.py",
            "Validate &amp; clean",
            f"<code>validate()</code> runs the full integrity battery; <code>clean()</code> drops "
            f"{clog['dropped_duplicates']} duplicate + {clog['dropped_pre_creation']} impossible pre-creation events "
            f"→ {clog['rows_out']:,}. Only removes invalid rows — never uses label/resolvedDate.",
        ),
        (
            "src/features.py",
            "Build the (submission, t) panel",
            f"<code>build_panel()</code> makes one row per submission × t∈[0,7,30] that is still "
            f"<i>open at t</i> ({N_PANEL:,} rows: {PC[0]}/{PC[7]}/{PC[30]} per t). Each row gets the 5 leakage-safe features. "
            f"<code>feature(submission_id, t)</code> is the single-row challenge contract.",
        ),
        (
            "src/model.py",
            "Split → validate → fit",
            f"<code>temporal_split()</code>: train = earliest 70% by createdDate (≤ {thr.date()}, {N_TR} rows), "
            f"test = latest 30% ({N_TE} rows). <code>StratifiedGroupKFold</code> CV <i>inside train</i> picks the feature "
            f"set &amp; log1p. <code>make_model()</code> = Impute → Scale → Logistic, fit on train.",
        ),
        (
            "src/model.py",
            "Score",
            "<code>bind_score(submission_id, t)</code> → builds that row's features → "
            "<code>predict_proba</code> → P(bind). Higher = work it first.",
        ),
        (
            "src/evaluate.py",
            "Evaluate vs baselines",
            "<code>get_test_scores()</code> scores the model + baselines once on the held-out test; "
            "<code>precision_at_k()</code> / <code>per_t()</code> produce the ranking &amp; prioritization metrics.",
        ),
    ]
)

CSS = """
:root{--p:#3a2150;--p2:#7a5ca3;--g:#f4c542;--ink:#222;--mut:#666;--line:#e3e0e8;--bg:#faf9fc}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  line-height:1.55;margin:0;background:var(--bg);font-size:15px}
.wrap{max-width:920px;margin:0 auto;padding:0 26px 80px}
header{background:linear-gradient(120deg,#2d1b3d,#4a2d6b);color:#fff;padding:34px 26px}
header .inner{max-width:920px;margin:0 auto}
header h1{margin:0 0 6px;font-size:26px}
header .sub{color:#e7dcf5;font-size:14.5px}
h2{color:var(--p);border-bottom:2px solid var(--g);padding-bottom:5px;margin-top:42px;font-size:21px}
h3{color:var(--p);margin-top:26px;font-size:16.5px}
h4{margin:16px 0 4px;color:#4a2d6b;font-size:14.5px}
code{background:#efe9f5;color:#54307e;padding:1px 5px;border-radius:4px;font-size:12.5px}
pre{background:#2d1b3d;color:#f3ecfb;padding:13px 15px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.45}
pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;background:#fff}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:#f0ebf6;color:var(--p)}
tr:nth-child(even) td{background:#fbfafd}
figure{margin:20px 0;text-align:center}
figure img{max-width:100%;border:1px solid var(--line);border-radius:7px;background:#fff;padding:6px}
figcaption{color:var(--mut);font-size:12.5px;margin-top:6px}
.heur{background:#fffaf0;border-left:4px solid var(--g);border-radius:6px;padding:10px 15px;margin:12px 0}
.heur b{color:#9a7a14}
.note{background:#fff;border-left:4px solid var(--p2);padding:10px 15px;border-radius:6px;margin:12px 0}
ul{margin:8px 0 8px 2px;padding-left:20px}li{margin:4px 0}
.flow{display:flex;flex-direction:column;align-items:center;margin:18px 0}
.stage{width:100%;max-width:680px;background:#fff;border:1px solid var(--line);border-left:4px solid var(--p2);
  border-radius:8px;padding:11px 16px}
.stage b{color:var(--p);font-size:15px}.stage p{margin:4px 0 0;font-size:13px;color:#444}
.stage .mod{float:right;background:#2d1b3d;color:var(--g);font-size:11px;padding:2px 8px;border-radius:10px;font-family:monospace}
.arrow{color:var(--p2);font-size:18px;line-height:1;margin:3px 0}
.q{font-weight:600;color:var(--p);margin-top:14px}
.a{margin:2px 0 0}
.muted{color:var(--mut);font-size:13px}
footer{color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);margin-top:50px;padding-top:14px}
"""

H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novella Bind Score — Pipeline &amp; Model Deep-Dive</title><style>{CSS}</style></head><body>
<header><div class="inner">
<h1>Pipeline &amp; Model Deep-Dive</h1>
<div class="sub">How raw data becomes a bind score — every stage, the module that owns it, and the ML underneath.
A study &amp; interview companion (paired with <code>final_report.html</code>).</div>
</div></header>
<div class="wrap">

<h2>1. The pipeline at a glance</h2>
<p>Six stages, four modules. Data flows top-to-bottom; the tag on each box is the file that owns that step.</p>
{PIPE}
<div class="heur"><b>Mental model:</b> think of it as an <b>assembly line</b> — <code>preprocessing</code> cleans the
parts, <code>features</code> stamps each submission into 3 dated snapshots, <code>model</code> learns the
pattern and scores, <code>evaluate</code> is QA at the end of the line. Each station only trusts what the
station before it produced.</p>

<h2>2. Stage by stage (what happens + why)</h2>

<h3>2.1 &nbsp;Ingest &amp; clean &nbsp;<span class="muted">— src/preprocessing.py</span></h3>
<p><code>validate()</code> runs every integrity check and reports its count <i>even when zero</i> (so the report
shows what we look for, not just what we found). <code>clean()</code> applies the decided policy: drop
<b>{clog['dropped_duplicates']} byte-identical duplicate events</b> and <b>{clog['dropped_pre_creation']} pre-creation
events</b> (dated &gt;1 day before the submission exists). Char outliers are <i>kept</i> (handled later by log1p).</p>
<div class="heur"><b>Heuristic:</b> "clean only what is <i>impossible</i> or <i>fake</i>, not what is merely
<i>extreme</i>." Duplicates &amp; time-travel events are garbage; a 43k-char email is real data — transform it, don't delete it.</div>

<h3>2.2 &nbsp;Feature construction &nbsp;<span class="muted">— src/features.py</span></h3>
<p>The key abstraction is the <b>panel</b>: one row per <b>(submission, t)</b> for t ∈ [0, 7, 30], kept only while
the submission is still <b>open at t</b>. So most submissions contribute up to 3 rows — three timed photographs.
Each row carries the 5 features; <code>agent_bind_rate</code> is smoothed (empirical-Bayes) and
<code>outbound_chars</code> is <code>log1p</code>-compressed.</p>
<div class="heur"><b>Heuristic — the leakage rule:</b> a feature at (sub, t) may use <b>only what a broker could have
actually seen at day t</b> — events with <code>eventDate ≤ createdDate + t</code>, and for customer history,
only that customer's submissions <i>already resolved</i> before this one was created. Never the future.</div>
<div class="heur"><b>Heuristic — agent_bind_rate smoothing:</b> "a new customer is assumed <i>average</i> until
proven otherwise." With 0 prior deals you get the base rate; each resolved deal pulls the estimate toward that
customer's true close-rate. (That's why first-timers all share one value — the cold-start problem.)</div>

<h3>2.3 &nbsp;Split &amp; validate &nbsp;<span class="muted">— src/model.py</span></h3>
<p><code>temporal_split()</code> trains on the earliest ~70% of submissions and tests on the latest ~30%.
Model choices are made by <code>StratifiedGroupKFold</code> cross-validation <b>inside the training set</b>,
<b>grouped by submissionId</b>. The test set is scored once, at the end.</p>
<div class="heur"><b>Heuristic — temporal split:</b> "train on the past, predict the future" — the only honest test
for a feature like customer history that accumulates over time. A random split would let the model peek at later
deals from the same era.</div>
<div class="heur"><b>Heuristic — grouped CV:</b> keep a submission's 3 snapshots (t=0/7/30) <b>together</b> in one fold.
Split them and you'd train on t=0 and validate on t=30 of the <i>same</i> submission — testing on a near-copy of
training, which flatters the score.</div>

<h3>2.4 &nbsp;The model pipeline &nbsp;<span class="muted">— src/model.py</span></h3>
<pre><code>make_pipeline(
    SimpleImputer(strategy="median"),   # fill gaps
    StandardScaler(),                    # normalize every feature -> mean 0, sd 1
    LogisticRegression(class_weight="balanced", C=1.0),  # L2-regularized
)</code></pre>
<p>One pooled logistic regression with <code>t</code> as a feature (not 3 separate models). Everything is fit
<b>inside train/the fold only</b>, so test statistics never leak into the scaler or the imputer.</p>
<div class="heur"><b>Heuristic — <code>class_weight="balanced"</code>:</b> only {BASE:.0%} of rows bind, so a lazy model
would predict "never binds" and be {1-BASE:.0%} accurate but useless. Balancing up-weights the rare "sold" class
in the loss so the model actually has to separate them. (§4 covers why this matters more than raw accuracy.)</div>

<h3>2.5 &nbsp;Score &amp; evaluate &nbsp;<span class="muted">— src/model.py + src/evaluate.py</span></h3>
<p><code>bind_score(submission_id, t)</code> rebuilds that one row's features and returns
<code>predict_proba</code> = P(bind) — the deliverable. <code>evaluate.py</code> scores the model and every baseline
once on the held-out test and computes precision@k / lift / per-t AUC.</p>

<h2>3. Deep-dive: why normalize features?</h2>
<p>Logistic regression's fit quality doesn't care about units in theory — but the <b>optimizer</b> that finds the
weights very much does. Feature scale sets the <b>shape of the loss surface</b> the optimizer must descend.</p>
{img(FIG_GEO, "Loss surface (contours) + gradient-descent path. Left: scaled features → near-circular bowl, the gradient points almost at the minimum (★), few steps. Right: one feature 30× larger → a long narrow valley; a single learning rate overshoots the steep axis and crawls along the flat one → zig-zag, many steps, can hit the iteration cap before converging.")}
<h4>The four ways unscaled features interfere</h4>
<ul>
<li><b>Ill-conditioned surface.</b> Different scales → very different curvature per axis → the Hessian's
<b>condition number</b> (κ = λ<sub>max</sub>/λ<sub>min</sub>) blows up. Gradient-descent convergence speed degrades directly with κ.</li>
<li><b>One learning rate can't serve all axes.</b> The step must be small enough to stay stable on the steep axis —
but then it's far too small on the flat axis. Result: zig-zag across the valley, crawl along it. With a fixed
<code>max_iter</code>, it can stop <i>before</i> converging (sklearn's ConvergenceWarning) and return an under-fit model.</li>
<li><b>Numerical saturation.</b> A raw value like <code>outbound_chars = 43,650</code> makes wᵀx enormous → the
sigmoid saturates (≈0 or 1) → its gradient ≈ 0 → no learning signal; you can also overflow <code>exp</code>.</li>
{img(FIG_SIG, "Sigmoid + its gradient. Past |z|≈5 the curve flattens and the gradient vanishes — an unscaled feature pushes z there and the weight stops updating.")}
<li><b>Regularization is distorted.</b> L2 penalizes Σwⱼ² <i>equally</i>, ignoring units. A large-scale feature
needs only a tiny weight (so L2 barely touches it); a small-scale feature needs a big weight (so L2 over-shrinks it).
Which features survive the penalty would depend on arbitrary unit choices — unless you standardize first.</li>
</ul>
<div class="heur"><b>One-line takeaway:</b> standardizing turns a long skewed valley into a round bowl → the
optimizer converges fast and reliably, <i>and</i> the L2 penalty treats every feature fairly. (Trees, e.g. our
XGBoost comparison, are scale-invariant and need none of this.) In our model <code>outbound_chars</code> also gets
<code>log1p</code> first, to pull in its long tail before scaling.</div>

<h2>4. Deep-dive: logistic regression (interview prep)</h2>

<h4>The model</h4>
<p>Logistic regression models the <b>log-odds</b> of the positive class as a linear function of the features:</p>
<pre><code>logit(p) = ln( p / (1−p) ) = wᵀx + b        ⟹        p = σ(wᵀx + b) = 1 / (1 + e^−(wᵀx+b))</code></pre>
<p>So it's a linear model wrapped in a sigmoid that squashes the score into a probability in (0, 1). The
<b>decision boundary is linear</b> in x (a hyperplane where wᵀx + b = 0, i.e. p = 0.5).</p>

<h4>The loss function — binary cross-entropy (log-loss)</h4>
<p>Derived from <b>maximum likelihood</b>: each label is Bernoulli(p<sub>i</sub>), so the likelihood is
∏ p<sub>i</sub><sup>y<sub>i</sub></sup>(1−p<sub>i</sub>)<sup>1−y<sub>i</sub></sup>. Maximizing it = minimizing the
negative log-likelihood, the <b>cross-entropy</b>:</p>
<pre><code>L(w) = − Σᵢ [ yᵢ·ln(pᵢ) + (1−yᵢ)·ln(1−pᵢ) ]   +   λ‖w‖²   (L2 term)</code></pre>
{img(FIG_LOSS, "Per-example log-loss. If the truth is y=1, loss = −log(p): near-0 when confident-right, exploding as p→0 (confident-wrong). It is unbounded above — that asymmetric pressure is what trains calibrated probabilities.")}
<div class="note"><b>Why log-loss and not MSE?</b> Paired with the sigmoid, squared error is <b>non-convex</b> in w
(local minima) and its gradient <b>vanishes</b> when the sigmoid saturates — so training stalls on confident-wrong
examples. Cross-entropy is <b>convex</b> (with L2, strictly convex → unique global optimum) and its gradient stays
strong exactly where the model is most wrong.</div>

<h4>How it's optimized</h4>
<p>The gradient has a famously clean form — <b>"prediction error times input"</b>:</p>
<pre><code>∇<sub>w</sub> L = Xᵀ (p − y)        (p − y is the residual; no closed-form solution → solved iteratively)</code></pre>
<p>sklearn's default solver (L-BFGS) is a quasi-Newton method that uses curvature to converge fast — but, per §3,
only when the features are scaled. <code>C</code> = 1/λ controls regularization strength (smaller C = stronger shrink).</p>

<h4>Interpreting it</h4>
<ul>
<li><b>Raw coefficient:</b> e<sup>w<sub>j</sub></sup> is the <b>odds ratio</b> — the multiplicative change in odds of
binding per one-unit increase in feature j.</li>
<li><b>Standardized coefficient:</b> after StandardScaler, |w<sub>j</sub>| is comparable across features → that's
our Task-2 importance ranking (effect per <b>1 standard deviation</b>).</li>
</ul>

<h4>Assumptions, strengths, weaknesses</h4>
<table>
<tr><th>Assumes</th><th>Strengths</th><th>Weaknesses</th></tr>
<tr>
<td>Linear in the log-odds · independent observations · limited multicollinearity · features scaled (for the solver/penalty)</td>
<td>Interpretable · well-calibrated probabilities · fast · few parameters · strong baseline · convex (no local minima)</td>
<td>Only a linear boundary (needs feature engineering for interactions) · sensitive to scaling/outliers · unstable coefficients under collinearity (L2 mitigates)</td>
</tr>
</table>

<h4>Why it fits <i>this</i> problem</h4>
<p>The signal here is simple and roughly linear (a shallow XGBoost lost at every t — no interactions to exploit),
the data is small (130 positives → a low-variance linear model resists overfitting), and brokers want an
interpretable, calibrated score. Logistic regression is the right tool; XGBoost was the comparison, not the ship.</p>

<h2>5. Interview cheat-sheet</h2>
<p class="q">Q. Why not just use accuracy?</p>
<p class="a">Only {BASE:.0%} bind — predicting "never" scores {1-BASE:.0%} accuracy yet ranks nothing. We use ranking
metrics (ROC/PR-AUC) and the operational one, <b>precision@top-k</b> (brokers work a ranked queue).</p>
<p class="q">Q. How do you handle the class imbalance?</p>
<p class="a"><code>class_weight="balanced"</code> up-weights the rare class in the loss; we judge on PR-AUC / precision@k,
not accuracy. (It nudges predicted probabilities up — fine, since we only need the <i>ranking</i>.)</p>
<p class="q">Q. How do you prevent leakage?</p>
<p class="a">Features obey <code>eventDate ≤ createdDate + t</code>; customer history uses only prior-resolved deals;
temporal train/past → test/future split; scaler/imputer fit inside the fold; grouped CV keeps a submission's rows together.</p>
<p class="q">Q. Why logistic over XGBoost here?</p>
<p class="a">Tested both — XGBoost lost at every t. No nonlinear/interaction signal to exploit, and on 130 positives the
simpler model generalizes better and stays interpretable.</p>
<p class="q">Q. What does standardizing change?</p>
<p class="a">Nothing about the ideal fit, everything about <i>reaching</i> it: it conditions the loss surface for the
optimizer and lets the L2 penalty treat features fairly (§3). It also makes coefficients comparable for ranking.</p>
<p class="q">Q. Biggest limitation?</p>
<p class="a">Cold-start: ~half of day-0 rows are first-time customers with no history, where the model falls back near
the base rate. The fix would be submission-static attributes (line of business, geography, premium) — not in this data.</p>

<footer>
Companion to <code>reports/final_report.html</code>. Generated from the live code via
<code>poetry run python scripts/build_pipeline_doc.py</code>. Grounded numbers: panel {N_PANEL:,} rows
({PC[0]}/{PC[7]}/{PC[30]} per t), train {N_TR} / test {N_TE}, base rate {BASE:.1%}.
</footer>
</div></body></html>"""

out = R / "reports/pipeline_and_model.html"
out.write_text(H)
print(f"wrote {out}  ({len(H)/1024:.0f} KB)")
