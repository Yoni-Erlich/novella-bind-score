# Novella Bind Score

A **bind score** for an E&S insurance broker: given `(submission_id, t)` — a submission and a time `t` days
after it was created — output a score so that **higher = more likely to bind (sell)**. Brokers use it to
**prioritize effort** (work the top of the ranked list first); earlier predictions are more valuable.

**Headline result (held-out temporal test):** ROC-AUC **0.744**, **2–3× lift** at the top-20% of the list,
and — crucially — **day-0 predictions work** (AUC 0.76, up from a coin-flip) thanks to repeat-customer history.

---

## Reports (start here)
Two **self-contained** HTML reports are committed under `reports/` — every chart is embedded, so they need
no data and no server. They don't render inline on GitHub; to read them:

> open the file on GitHub → **Download raw file** → open the downloaded `.html` in any browser.

| report | what it is |
|---|---|
| [`reports/final_report.html`](reports/final_report.html) | the **deliverable** — data story, the 3 tasks, results, limitations |
| [`reports/pipeline_and_model.html`](reports/pipeline_and_model.html) | study/interview companion — pipeline walkthrough + why-logistic |

Both are **generated from the live code** (numbers can't drift from the model). To rebuild them yourself, see
[Regenerate the reports](#regenerate-the-reports) below.

---

## Repository layout
```
challenge_1/
├── src/
│   ├── preprocessing.py      # validate() (full check battery) + clean() (dedupe + impossible-date removal)
│   ├── features.py           # feature(submission_id, t) + build_panel()            [Task 1]
│   ├── model.py              # train_bind_model() + bind_score(submission_id, t)    [Task 3]
│   └── evaluate.py           # metrics, baselines, XGBoost comparison
├── scripts/                  # exploratory checks + the two report builders
├── reports/
│   ├── final_report.html         # the deliverable (self-contained)
│   └── pipeline_and_model.html   # study/interview companion (self-contained)
├── data/                     # NOT shipped — drop the provided CSVs here (see "Run it")
└── pyproject.toml
```
> The raw `data/` CSVs, EDA notebooks, and figure sources are part of the full project but are **not** in this
> repo (it's the code + rendered reports). Everything below runs once you supply the two data files.

## Run it
```bash
# 0. Data — this repo ships without it. Put the two provided files here:
#      data/features_submissions.csv
#      data/features_events.csv

# 1. Install (dedicated in-project .venv, Python 3.12)
poetry install

# 2. Train + evaluate end-to-end (prints the full results tables)
poetry run python src/evaluate.py
```
Use the score in code:
```python
from src.features import load_clean
from src.model import train_bind_model, bind_score
subs, events = load_clean("data")
fit = train_bind_model(subs, events)
bind_score(submission_id=1, t=7, subs=subs, events=events, fitted=fit)   # -> P(bind)
```
> `bind_score`/`feature` honor the one-row challenge contract and rebuild the panel per call — fine for
> spot-checks, but use `build_panel(...)` + the fitted model directly for batch scoring.

### Regenerate the reports
With the data in place, rebuild either HTML from the live code:
```bash
poetry run python scripts/build_html_report.py     # -> reports/final_report.html
poetry run python scripts/build_pipeline_doc.py    # -> reports/pipeline_and_model.html
```

---

## Approach (shared across tasks)
- **One row = `(submission_id, t)`**, `t ∈ {0,7,30}`, scored only while the submission is still **open at `t`**
  (a resolved submission isn't a prioritization candidate). One **pooled** model with `t` as a feature.
- **Leakage-safe:** a feature at `(sub, t)` uses only events with `event_date ≤ createdDate + t`; customer-history
  features use only the agent's submissions **resolved before** this one was created; `resolvedDate`/`label` are never features.
- **Evaluation:** **temporal** train/test split (train on earlier-created submissions, test on later). Feature/transform
  choices made by cross-validation *inside* train; the test set is scored once.

---

## Task 1 — Features
Five features, all leakage-safe (`src/features.py`):

| feature | what it is |
|---|---|
| `agent_bind_rate` | the **customer's** smoothed past close-rate (as-of createdDate; cold-start → base rate) |
| `outbound_chars_log` | `log1p` of Novella's outbound email chars by `t` (broker **effort**) |
| `has_quote_by_t` | 1 if a carrier quote arrived by `t` (funnel gate) |
| `n_inbound_by_t` | customer replies by `t` (engagement) |
| `t` | the snapshot time (lets one pooled model adapt across 0/7/30) |

**Design note:** we explored ~40 candidates and found the signal is **activity volume + quote presence + customer
history**. Ratios, attachments, char-distribution stats, response-latency, velocity/recency/acceleration, and
calendar features were all tested and **dropped** (no signal beyond volume — verified by partial-correlation +
incremental CV-AUC). `log1p` on outbound chars tames extreme outliers (a 43k-char email) and won a train-CV A/B.

## Task 2 — Feature significance (ranking)
Ranked by the **linear lens** — univariate AUC (transform-stable) + standardized logistic coefficients:

| rank | feature | univariate test AUC |
|---|---|---|
| 1 | **agent_bind_rate** | 0.742 |
| 2 | has_quote_by_t | 0.567 |
| 3 | outbound_chars_log | 0.567 |
| 4 | n_inbound_by_t | 0.551 |
| 5 | t | 0.514 |

**Design note:** we also tried **XGBoost + SHAP**, but SHAP ranked a proven-noise calendar feature #1 and the
genuine top feature last — a high-cardinality artifact of an *overfit* tree (the tree lost to linear). So we rank
with the linear lens, not SHAP. (Univariate AUCs above are measured on the held-out test, for reporting only —
feature *selection* happens in train CV.)

## Task 3 — Model & evaluation
**Model:** pooled **regularized logistic regression** (`class_weight='balanced'`). **Metrics:** ROC-AUC + PR-AUC
(ranking) and **precision@top-20% + lift** (the prioritization payoff), reported **per `t`**.

**Held-out temporal test (578 rows, 14% sold):**

| `t` | ROC-AUC | precision@top-20% | lift vs random |
|---|---|---|---|
| 0 | 0.759 | 36.4% | **2.7×** |
| 7 | 0.722 | 29.8% | 2.2× |
| 30* | 0.877 | 53.3% | 3.2× |
| **overall** | **0.744** (PR-AUC 0.359 vs 0.14 floor) | | |

**Model comparison** (overall ROC-AUC, same split):

| model | ROC-AUC |
|---|---|
| **logistic regression (shipped)** | **0.744** |
| `agent_bind_rate` alone | 0.742 |
| XGBoost (shallow, regularized) | 0.696 |
| naive heuristic (quote → effort) | 0.567 |
| no-skill floor | 0.500 |

Also tested: **one pooled model vs three per-`t` models** → 0.744 vs 0.738 (pooled wins, esp. at day-0); class-weight
tuning → no effect on ranking.

> Full detail, charts, and the cautionary SHAP tale live in the two HTML reports above.

---

## What moved the needle
1. **Data cleaning** — removed 530 duplicate events + 127 impossible pre-creation events.
2. **Repeat-customer history** (`agent_bind_rate`) — the strongest feature; it also **rescues day-0**, where
   activity-based features are blind. `agent_bind_rate` alone nearly matches the full model on *returning* customers,
   but the full model is needed for the ~50% of submissions from **first-time customers** (where history is blank and
   it falls back on the live deal signal).

## Honest limitations
- Small data (130 positives); a **single** temporal split (one estimate); the `t=30` test slice is tiny (n=73) so its
  numbers (marked `*`) are noisy — trust the ~0.74 overall and the 2–3× lift.
- `agent_bind_rate` only helps the ~38% of submissions with prior customer history; the rest get the base-rate default.
- Even an adversarial feature panel found nothing that beats this model on the held-out test.
