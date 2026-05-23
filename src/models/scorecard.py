"""
src/models/scorecard.py
───────────────────────
Logistic regression credit scorecard with industry-standard score scaling.

This is the CHALLENGER model — maintained alongside the XGBoost champion for:
- Regulatory interpretability (SR 11-7)
- Adverse action notice generation
- Sanity checking champion stability

Discovery and validation in: notebooks/03_scorecard_baseline.ipynb

Performance:
- AUC-ROC : 0.8619
- Gini     : 0.7237
- KS       : 0.5712
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features.binning import WoEBinner

logger = logging.getLogger(__name__)


# ── Score scaling constants ──

BASE_SCORE = 600   # score at which odds = BASE_ODDS
BASE_ODDS  = 50    # 50:1 good:bad ratio at base score
PDO        = 20    # points to double the odds

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


# ── Score scaling ──

def prob_to_score(prob: np.ndarray) -> np.ndarray:
    """
    Convert predicted default probability to integer credit score.

    Score = offset + factor * ln((1 - p) / p)

    Higher score = lower risk.
    Base score of 600 corresponds to 50:1 good:bad odds.
    Every 20 points doubles the odds.
    """
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    log_odds = np.log((1 - prob) / prob)
    return np.round(OFFSET + FACTOR * log_odds).astype(int)


def score_to_band(scores: np.ndarray) -> pd.Categorical:
    """Assign integer scores to risk bands A (lowest) through E (highest)."""
    return pd.cut(
        scores,
        bins=[-np.inf, 520, 550, 570, 590, np.inf],
        labels=["E — Very High", "D — High", "C — Medium", "B — Low", "A — Very Low"],
    )


# ── Scorecard table ──

def build_scorecard_table(
    model: LogisticRegression,
    feature_names: list,
    woe_tables: dict,
) -> pd.DataFrame:
    """
    Convert logistic regression coefficients + WoE values into a
    human-readable scorecard points table.

    Points per bin = -(coef * WoE + intercept/n) * factor + offset/n

    Parameters
    ----------
    model         : fitted LogisticRegression
    feature_names : list of WoE feature names (e.g. ['revolving_util_woe', ...])
    woe_tables    : dict of {feature_name: pd.DataFrame with bin/woe/event_rate}

    Returns
    -------
    pd.DataFrame with columns: feature, bin, event_rate, woe, points
    """
    coefficients = dict(zip(feature_names, model.coef_[0]))
    intercept    = model.intercept_[0]
    n_features   = len(feature_names)

    rows = []
    for feat_woe, coef in coefficients.items():
        feat = feat_woe.replace("_woe", "")
        if feat not in woe_tables:
            continue
        table = woe_tables[feat]
        for _, row in table.iterrows():
            points = (
                -(coef * row["woe"] + intercept / n_features)
                * FACTOR
                + OFFSET / n_features
            )
            rows.append({
                "feature":    feat,
                "bin":        str(row["bin"]),
                "event_rate": row["event_rate"],
                "woe":        round(row["woe"], 4),
                "points":     round(points, 1),
            })

    return pd.DataFrame(rows)


# ── Training ──

def train_scorecard(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[LogisticRegression, WoEBinner, dict]:
    """
    Full scorecard training pipeline.

    Steps:
    1. Train/test split (stratified)
    2. WoE binning (fitted on training data only)
    3. Logistic regression on WoE features
    4. Evaluate on held-out test set

    Parameters
    ----------
    X            : cleaned feature matrix (from cleaning.py)
    y            : binary target
    test_size    : fraction of data held out for evaluation
    C            : logistic regression regularisation strength
    random_state : random seed for reproducibility

    Returns
    -------
    model   : fitted LogisticRegression
    binner  : fitted WoEBinner
    metrics : dict of performance metrics
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # WoE transform — fit on training data only
    binner  = WoEBinner()
    X_train_woe = binner.fit_transform(X_train, y_train)
    X_test_woe  = binner.transform(X_test)

    # Fit logistic regression
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train_woe, y_train)

    # Evaluate
    y_test_prob  = model.predict_proba(X_test_woe)[:, 1]
    y_train_prob = model.predict_proba(X_train_woe)[:, 1]

    test_auc  = roc_auc_score(y_test,  y_test_prob)
    train_auc = roc_auc_score(y_train, y_train_prob)

    metrics = {
        "train_auc":  round(train_auc, 4),
        "test_auc":   round(test_auc, 4),
        "train_gini": round(2 * train_auc - 1, 4),
        "test_gini":  round(2 * test_auc - 1, 4),
    }

    logger.info(
        f"Scorecard — Train Gini: {metrics['train_gini']} | "
        f"Test Gini: {metrics['test_gini']}"
    )

    return model, binner, metrics


# ── Save / Load ──

def save_scorecard(
    model: LogisticRegression,
    binner: WoEBinner,
    model_dir: str = "models",
) -> None:
    """Save fitted scorecard and binner to disk."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "challenger_scorecard.joblib"))
    binner.save(os.path.join(model_dir, "woe_binner.json"))
    logger.info(f"Scorecard saved → {model_dir}/")


def load_scorecard(
    model_dir: str = "models",
) -> Tuple[LogisticRegression, WoEBinner]:
    """Load fitted scorecard and binner from disk."""
    model  = joblib.load(os.path.join(model_dir, "challenger_scorecard.joblib"))
    binner = WoEBinner.load(os.path.join(model_dir, "woe_binner.json"))
    logger.info(f"Scorecard loaded ← {model_dir}/")
    return model, binner