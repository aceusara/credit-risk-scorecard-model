# Credit Risk PD Model

End-to-end **Probability of Default (PD)** modelling pipeline built on the
Give Me Some Credit dataset. Covers the full model lifecycle a senior ML
team would expect: data quality investigation → feature engineering →
scorecard → gradient boosting → champion/challenger → monitoring → governance.

Built as a portfolio project targeting senior credit risk ML roles.

---

## Business Context

A consumer lending team needs to predict the probability that a borrower
will experience 90+ days of financial distress within 2 years. This model
feeds directly into automated credit decisioning — influencing approval
rates, credit limits, and pricing tiers.

**Business objective:** Maximise approved volume while holding bad rate
within acceptable risk appetite.

**Dataset:** Give Me Some Credit (Kaggle) — 150,000 US consumer credit
bureau records, 6.68% default rate.

---

## Model Performance

| Metric | Scorecard (Challenger) | XGBoost (Champion) |
|---|---|---|
| AUC-ROC | 0.8619 | **0.8734** |
| Gini | 0.7237 | **0.7468** |
| KS Statistic | 0.5712 | **0.5896** |
| Role | Interpretability + adverse action | Automated decisioning |

Both models exceed the 0.70 Gini industry benchmark on first pass,
with zero hyperparameter tuning on the champion.

---

## Project Structure

```
credit-risk-scorecard-model/
├── notebooks/
│   ├── 01_eda.ipynb                  # Data quality investigation
│   ├── 02_feature_engineering.ipynb  # WoE binning, IV analysis
│   ├── 03_scorecard_baseline.ipynb   # Logistic regression scorecard
│   ├── 04_gradient_boosting.ipynb    # XGBoost champion + SHAP
│   ├── 05_champion_challenger.ipynb  # A/B framework (in progress)
│   ├── 06_fairness_audit.ipynb       # Bias and fairness (planned)
│   └── 07_monitoring.ipynb           # PSI/CSI drift detection (planned)
├── src/
│   ├── data/
│   │   └── cleaning.py               # Full cleaning pipeline
│   ├── features/
│   │   └── binning.py                # WoEBinner — fit, transform, save, load
│   ├── models/
│   │   ├── scorecard.py              # LR scorecard + score scaling
│   │   └── gradient_boosting.py     # XGBoost training + persistence
│   ├── evaluation/                   # Gini, KS, PSI metrics (planned)
│   ├── monitoring/                   # Drift detection engine (planned)
│   └── deployment/                   # FastAPI serving endpoint (planned)
├── models/
│   ├── champion_xgb.joblib           # Fitted XGBoost champion
│   ├── challenger_scorecard.joblib   # Fitted logistic regression challenger
│   └── woe_binner.json               # Fitted WoEBinner for scorecard
├── governance/
│   ├── model_card.md                 # Intended use, limitations, metrics
│   ├── assumptions_log.md            # Every modelling decision documented
│   ├── fairness_report.md            # Protected attribute audit (planned)
│   └── approval_trail.md            # Development → validation → sign-off
├── Makefile                          # make train | make test | make serve
└── pyproject.toml                    # Dependencies and tool config
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/aceusara/credit-risk-scorecard-model.git
cd credit-risk-scorecard-model

# 2. Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Add data (requires Kaggle account)
# Download Give Me Some Credit from:
# https://www.kaggle.com/competitions/GiveMeSomeCredit/data
# Place cs-training.csv and cs-test.csv in data/raw/

# 4. Run notebooks in order
jupyter lab
```

---

## Key Design Decisions

### Why a dual model strategy?
| | Scorecard (LR) | XGBoost |
|---|---|---|
| Role | Challenger | Champion |
| Gini | 0.7237 | 0.7468 |
| Interpretability | Full — every point traceable | SHAP explanations |
| Regulatory fit | SR 11-7 aligned | Requires additional documentation |
| Use case | Adverse action notices | Automated decisioning |

The scorecard is not a fallback — it is maintained in production as a
live challenger, used for adverse action notices, and acts as a sanity
check on champion stability.

### Why WoE binning over raw features for the scorecard?
Pearson correlation gave `revolving_util` — the strongest predictor in the
dataset — a score of **0.00** against the target. The relationship is
non-linear and completely invisible to linear methods. WoE binning captures
any shape of relationship and transforms all features to a common,
interpretable scale that logistic regression can use effectively.

XGBoost receives raw cleaned features — it finds its own optimal split
points and does not benefit from WoE pre-transformation.

### Why custom bins for DPD features?
`pd.qcut` collapsed `dpd_90_plus` (94.4% zeros) to a single bin and
reported IV = 0.00. Manual bins `[0, 1, 2, 3+]` revealed the true
IV of **0.88** — the second strongest predictor in the model. Naive
binning was hiding two of the three strongest features entirely.

### Why XGBoost Baseline over XGBoost Tuned?
Grid search (108 combinations × 5-fold CV = 540 fits) returned parameters
marginally worse than the baseline (Gini 0.7453 vs 0.7468). The baseline
hyperparameters were already close to optimal — tuning confirmed this
rather than improved on it. A common and valid outcome documented
explicitly rather than hidden.

### Data leakage prevention
All imputation parameters and WoE bin edges are fitted on training data
only and passed explicitly when transforming test or production data.
No test data information touches the training pipeline at any stage.

---

## What Was Found in EDA

| Finding | Impact |
|---|---|
| 1 record age=0, 13 records age>100 | Removed — data entry errors |
| `revolving_util` max = 50,708 | Capped at 5.0 — confirmed data errors |
| DPD columns contain sentinel values 96 and 98 | Capped at 10 — supplier codes for unknown |
| Sentinel records default at **54.65%** vs 6.60% baseline | Critical leakage risk if left uncapped |
| `monthly_income` 19.8% missing | Median imputation by age decile |
| Missing income records default at **5.61%** vs 6.95% | Missingness is not a high-risk signal |
| `revolving_util` Pearson r = **0.00** with target | WoE essential — linear methods miss this entirely |

---

## Feature Importance

| Feature | IV | XGB Gain rank | Notes |
|---|---|---|---|
| `revolving_util` | 1.11 | 1st | Non-linear — Pearson missed it completely |
| `dpd_90_plus` | 0.88 | 2nd | One event = 4.6% → 33.7% default rate |
| `dpd_30_59` | 0.76 | 3rd | Zero-inflated — custom bins required |
| `dpd_60_89` | 0.60 | 4th | Zero-inflated — custom bins required |
| `age` | 0.26 | 5th | Clean monotonic — younger = higher risk |
| `monthly_income` | 0.08 | 6th | Right-skewed, 20% missing |
| `debt_ratio` | 0.07 | 7th | Noisy — corrupted when income = 0 |
| `open_credit_lines` | 0.07 | 8th | Weak IV but XGB finds non-linear signal |
| `n_dependents` | 0.03 | 9th | Weak overall |
| `real_estate_loans` | 0.01 | — | Dropped — below useless threshold |

---

## Governance

This project follows SR 11-7 model risk management principles.

| Document | Contents |
|---|---|
| `governance/model_card.md` | Intended use, performance by segment, limitations |
| `governance/assumptions_log.md` | Rationale for every material modelling decision |
| `governance/fairness_report.md` | Demographic parity audit (planned) |
| `governance/approval_trail.md` | Development → validation → sign-off record |

---

## Tech Stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` ·
`SHAP` · `FastAPI` · `Streamlit` · `pytest` · `joblib`

---

## Status

| Component | Status |
|---|---|
| EDA | ✅ Complete |
| Feature engineering (WoE) | ✅ Complete |
| Logistic regression scorecard | ✅ Complete — Gini 0.7237 |
| XGBoost champion | ✅ Complete — Gini 0.7468 |
| `src/data/cleaning.py` | ✅ Complete |
| `src/features/binning.py` | ✅ Complete |
| `src/models/scorecard.py` | ✅ Complete |
| `src/models/gradient_boosting.py` | ✅ Complete |
| Champion/challenger framework | 🔄 In progress |
| Fairness audit | ⏳ Planned |
| Monitoring dashboard | ⏳ Planned |
| FastAPI serving endpoint | ⏳ Planned |
| Governance documents | ⏳ Planned |