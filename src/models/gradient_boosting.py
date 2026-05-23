"""
src/models/gradient_boosting.py
────────────────────────────────
XGBoost champion model — training, evaluation, and persistence.

This is the CHAMPION model in the champion/challenger framework.

Discovery and validation in: notebooks/04_gradient_boosting.ipynb

Performance:
- AUC-ROC : 0.8734
- Gini     : 0.7468
- KS       : 0.5896

Design decisions:
- Raw cleaned features — WoE transformation not needed for tree models
- scale_pos_weight handles 13.9:1 class imbalance
- early_stopping_rounds prevents overfitting without manual n_estimators tuning
- Baseline hyperparameters outperformed grid search — documented in notebook
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ── Default hyperparameters ──
# These are the baseline parameters that outperformed grid search.
# Any change must be documented in governance/assumptions_log.md.

DEFAULT_PARAMS = {
    "n_estimators":      300,
    "learning_rate":     0.05,
    "max_depth":         4,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":       "auc",
    "early_stopping_rounds": 20,
    "random_state":      42,
    "n_jobs":            -1,
}


# ── Training ──

def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    params: Optional[Dict] = None,
    random_state: int = 42,
) -> Tuple[xgb.XGBClassifier, dict]:
    """
    Full XGBoost training pipeline.

    Steps:
    1. Train/test split (stratified)
    2. Compute class imbalance ratio for scale_pos_weight
    3. Fit XGBoost with early stopping
    4. Evaluate on held-out test set

    Parameters
    ----------
    X            : cleaned feature matrix (from cleaning.py)
                   Do NOT pass WoE-transformed features — use raw cleaned data
    y            : binary target
    test_size    : fraction held out for evaluation
    params       : override DEFAULT_PARAMS if provided
    random_state : random seed

    Returns
    -------
    model   : fitted XGBClassifier
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

    # Class imbalance ratio
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {imbalance_ratio:.2f}")

    # Merge params
    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model_params["scale_pos_weight"] = imbalance_ratio

    # Fit
    model = xgb.XGBClassifier(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_test_prob  = model.predict_proba(X_test)[:, 1]
    y_train_prob = model.predict_proba(X_train)[:, 1]

    test_auc  = roc_auc_score(y_test,  y_test_prob)
    train_auc = roc_auc_score(y_train, y_train_prob)

    metrics = {
        "train_auc":       round(train_auc, 4),
        "test_auc":        round(test_auc, 4),
        "train_gini":      round(2 * train_auc - 1, 4),
        "test_gini":       round(2 * test_auc - 1, 4),
        "best_iteration":  model.best_iteration,
        "scale_pos_weight": round(imbalance_ratio, 2),
    }

    logger.info(
        f"XGBoost — Train Gini: {metrics['train_gini']} | "
        f"Test Gini: {metrics['test_gini']} | "
        f"Best iteration: {metrics['best_iteration']}"
    )

    return model, metrics


# ── Save / Load ──

def save_champion(
    model: xgb.XGBClassifier,
    model_dir: str = "models",
) -> None:
    """Save fitted XGBoost champion to disk."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "champion_xgb.joblib")
    joblib.dump(model, path)
    logger.info(f"Champion saved → {path}")


def load_champion(
    model_dir: str = "models",
) -> xgb.XGBClassifier:
    """Load fitted XGBoost champion from disk."""
    path = os.path.join(model_dir, "champion_xgb.joblib")
    model = joblib.load(path)
    logger.info(f"Champion loaded ← {path}")
    return model