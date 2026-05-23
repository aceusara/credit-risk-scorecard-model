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

| Metric | Scorecard (LR) | XGBoost (Champion) |
|---|---|---|
| AUC-ROC | 0.8619 | TBD |
| Gini | 0.7237 | TBD |
| KS Statistic | 0.5712 | TBD |

> XGBoost champion model in progress.

---

## Project Structure
credit-risk-scorecard-model/
├── notebooks/
│   ├── 01_eda.ipynb                  # Data quality investigation
│   ├── 02_feature_engineering.ipynb  # WoE binning, IV analysis
│   ├── 03_scorecard_baseline.ipynb   # Logistic regression scorecard
│   ├── 04_gradient_boosting.ipynb    # XGBoost champion (in progress)
│   ├── 05_champion_challenger.ipynb  # A/B framework
│   ├── 06_fairness_audit.ipynb       # Bias and fairness analysis
│   └── 07_monitoring.ipynb           # PSI/CSI drift detection
├── src/
│   ├── data/
│   │   └── cleaning.py               # Full cleaning pipeline
│   ├── features/
│   │   └── binning.py                # WoEBinner class
│   ├── models/                       # Scorecard, XGBoost, calibration
│   ├── evaluation/                   # Gini, KS, PSI metrics
│   ├── monitoring/                   # Drift detection engine
│   └── deployment/                   # FastAPI serving endpoint
├── governance/
│   ├── model_card.md                 # Intended use, limitations, metrics
│   ├── assumptions_log.md            # Every modelling decision documented
│   ├── fairness_report.md            # Protected attribute audit
│   └── approval_trail.md            # Development → validation → sign-off
├── Makefile                          # make train | make test | make serve
└── pyproject.toml                    # Dependencies and tool config

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

### Why WoE binning over raw features?
Pearson correlation gave `revolving_util` — the strongest predictor in the
dataset — a correlation of **0.00** with the target. The relationship is
non-linear and invisible to linear methods. WoE binning captures any shape
of relationship and transforms all features to a common, interpretable scale.

### Why a dual model strategy?
| | Scorecard (LR) | XGBoost |
|---|---|---|
| Role | Challenger | Champion |
| Gini | 0.7237 | TBD |
| Interpretability | Full — every point traceable | SHAP explanations |
| Regulatory fit | SR 11-7 aligned | Requires additional documentation |
| Use case | Adverse action notices | Automated decisioning |

### Why custom bins for DPD features?
`pd.qcut` collapsed `dpd_90_plus` (94.4% zeros) to a single bin, reporting
IV = 0.00. Manual bins `[0, 1, 2, 3+]` revealed the true IV of **0.88** —
the second strongest predictor in the model. Naive binning was hiding it
entirely.

### Data leakage prevention
All imputation parameters (income medians by age decile) and WoE bin edges
are fitted on training data only and passed explicitly to test/production
transforms. No test data information touches the training pipeline.

---

## What Was Found in EDA

| Finding | Impact |
|---|---|
| 1 record with age=0, 13 with age>100 | Removed — data entry errors |
| `revolving_util` max = 50,708 (should be a ratio) | Capped at 5.0 |
| DPD columns contain sentinel values 96 and 98 | Capped at 10 — supplier codes for "unknown" |
| Sentinel records default at **54.65%** vs 6.60% baseline | Critical leakage risk if uncapped |
| `monthly_income` 19.8% missing | Imputed via median by age decile |
| `revolving_util` Pearson r = 0.00 with target | WoE essential — linear methods miss this |

---

## Governance

This project follows SR 11-7 model risk management principles.

| Document | Contents |
|---|---|
| `governance/model_card.md` | Intended use, performance by segment, limitations |
| `governance/assumptions_log.md` | Rationale for every material modelling decision |
| `governance/fairness_report.md` | Demographic parity audit across age groups |
| `governance/approval_trail.md` | Development → validation → sign-off record |

---

## Tech Stack

`Python 3.11` · `pandas` · `scikit-learn` · `XGBoost` · `FastAPI` ·
`Streamlit` · `pytest` · `SHAP`

---

## Status

| Component | Status |
|---|---|
| EDA | ✅ Complete |
| Feature engineering (WoE) | ✅ Complete |
| Logistic regression scorecard | ✅ Complete — Gini 0.7237 |
| XGBoost champion | 🔄 In progress |
| Champion/challenger framework | ⏳ Planned |
| Fairness audit | ⏳ Planned |
| Monitoring dashboard | ⏳ Planned |
| FastAPI serving endpoint | ⏳ Planned |
| Governance documents | ⏳ Planned |