# Novella Bind Score

A **bind score** for an E&S insurance broker: given `(submission_id, t)` — a submission and a time `t` days
after it was created — output a score so that **higher = more likely to bind (sell)**. Brokers use it to
**prioritize effort** (work the top of the ranked list first); earlier predictions are more valuable.

> ## 📄 Deliverable → **[`reports/final_report.html`](reports/final_report.html)** — start here
> This self-contained HTML report is the **primary output to review**: the full write-up — data story, the three
> tasks (features · significance · model+evaluation), results vs baselines, the overfit check, and honest
> limitations. Everything in `src/` exists to produce it. It's committed and embeds all its charts, so just open it
> in a browser. **▶ View it live, rendered: https://yoni-erlich.github.io/novella-bind-score/reports/final_report.html**
> — or download the file and open it locally (it won't render inline on the GitHub *file* view). The rest of this README
> is how to run and regenerate it.

**Headline result (held-out temporal test):** ROC-AUC **0.744**, **2–3× lift** at the top-20% of the list,
and — crucially — **day-0 predictions work** (AUC 0.76, up from a coin-flip) thanks to repeat-customer history.

---

## Quickstart (for reviewers)
The provided **data** (`data/`) and the EDA/model **notebooks** (`notebooks/`) are included, so it runs out of the box:
```bash
git clone <repo> && cd challenge_1
poetry install                                   # in-project .venv (Python 3.12)
poetry run python src/evaluate.py                # train + evaluate end-to-end — prints every headline number
poetry run python scripts/build_html_report.py   # regenerate reports/final_report.html from the live code
```
Then open `reports/final_report.html` in any browser (self-contained — no server). **Want to poke at it?** edit
anything in `src/`, rerun the two commands, and both the printed numbers and the report update — see
[How the report stays honest](#how-the-report-stays-honest). More detail in [Run it](#run-it) below.

---

## The report — how to open it
The **self-contained** HTML report `reports/final_report.html` is committed under `reports/` — every chart is
embedded, so it needs no data and no server. It doesn't render inline on GitHub; to read it:

> **Easiest — view it live, rendered:** https://yoni-erlich.github.io/novella-bind-score/reports/final_report.html
> Or on the GitHub *file* view → **Download raw file** → open the `.html` locally. (Or clone the repo and open it.)

[`reports/final_report.html`](reports/final_report.html) — the **deliverable**: data story, the 3 tasks,
results, limitations.

It's **generated from the live code, not hand-written** — see [How the report stays honest](#how-the-report-stays-honest).
To rebuild it yourself, see [Regenerate the report](#regenerate-the-report) below.

---

## Repository layout
```
challenge_1/
├── src/
│   ├── preprocessing.py      # validate() (full check battery) + clean() (dedupe + impossible-date removal)
│   ├── features.py           # feature(submission_id, t) + build_panel()            [Task 1]
│   ├── model.py              # train_bind_model() + bind_score(submission_id, t)    [Task 3]
│   └── evaluate.py           # metrics, baselines, XGBoost comparison, train-vs-test overfit check
├── scripts/                  # research / exploratory analyses + the report builder
├── notebooks/                # EDA + modeling notebooks (01_eda, 02_model, 03_outbound_coefficient)
├── reports/
│   └── final_report.html     # the deliverable (self-contained)
├── data/                     # the provided CSVs (features_submissions.csv, features_events.csv) — included
└── pyproject.toml
```
> `data/` (the provided CSVs) and `notebooks/` are **included**, so the repo runs out of the box.
>
> **`scripts/` is research, not the production path** — exploratory checks, feature searches, and one-off analyses
> that justify the choices in `src/` (the report builder also lives there). The shippable model is `src/`.

## Run it
```bash
# Data + notebooks are included — nothing to download.

# 1. Install (dedicated in-project .venv, Python 3.12)
poetry install

# 2. Train + evaluate end-to-end (prints the full results tables + the train-vs-test overfit check)
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

### Regenerate the report
Rebuild the HTML from the live code:
```bash
poetry run python scripts/build_html_report.py     # -> reports/final_report.html
```

### How the report stays honest
The report is **generated by running the code, not hand-written.** `scripts/build_html_report.py` imports the
functions in `src/`, runs them on the data, computes every number (and renders every figure) at build time, and
interpolates the results into the HTML. There are **no hard-typed numbers** — change anything in `src/`, rerun the
builder, and the report moves with it. You can verify the chain independently: `poetry run python src/evaluate.py`
reproduces the same headline numbers the report shows, because the report calls the same `src/` code path
(`data/` → `preprocessing` → `features` → `model` → `evaluate` → report).

---

## The thinking & analysis → in the report
The full design rationale and analysis — the leakage-safe design, one row per `(submission, t)`, the temporal split,
the five features (and why these), the significance ranking, the model, metrics, baselines and per-`t` results, what
moved the needle, and honest limitations — all live in **[`reports/final_report.html`](reports/final_report.html)**,
generated from the live code. **This README is for *running* the code; the report is the *thinking*.**
