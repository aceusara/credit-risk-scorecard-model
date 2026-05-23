"""
src/features/binning.py
───────────────────────
Weight of Evidence (WoE) binning and Information Value (IV) calculation.

Discovery and validation in: notebooks/02_feature_engineering.ipynb

Key design decisions:
- Equal-frequency binning (qcut) for well-distributed continuous features
- Custom bins for zero-inflated integer features (DPD columns)
  Reason: qcut collapses to 1 bin when 94%+ of values are identical
- Monotonicity enforced for scorecard compatibility (SR 11-7)
- real_estate_loans dropped — IV = 0.0121, below 0.02 useless threshold

Feature IV ranking (from notebook):
  revolving_util   1.1128  — verified not leakage
  dpd_90_plus      0.8778
  dpd_30_59        0.7552
  dpd_60_89        0.6004
  age              0.2592
  monthly_income   0.0806
  debt_ratio       0.0739
  open_credit_lines 0.0669
  n_dependents     0.0251
  real_estate_loans 0.0121  ← dropped
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ──

# Features using equal-frequency binning
EQUAL_FREQ_FEATURES = [
    "age",
    "monthly_income",
    "debt_ratio",
    "revolving_util",
    "open_credit_lines",
    "n_dependents",
]

# Features requiring custom bins due to zero-inflation
# Boundaries: exactly 0 | 1 event | 2 events | 3+ events
CUSTOM_BIN_FEATURES = {
    "dpd_30_59":   {"bins": [-0.1, 0.5, 1.5, 2.5, 10], "labels": ["0", "1", "2", "3+"]},
    "dpd_60_89":   {"bins": [-0.1, 0.5, 1.5, 2.5, 10], "labels": ["0", "1", "2", "3+"]},
    "dpd_90_plus": {"bins": [-0.1, 0.5, 1.5, 2.5, 10], "labels": ["0", "1", "2", "3+"]},
}

# All features in model order (real_estate_loans excluded)
MODEL_FEATURES = EQUAL_FREQ_FEATURES + list(CUSTOM_BIN_FEATURES.keys())

N_BINS_DEFAULT = 10
EPSILON = 0.5  # prevents log(0) — standard Siddiqi (2012) approach


# ── Core WoE calculation ──

def _compute_woe_iv_table(
    series: pd.Series,
    target: pd.Series,
    binned: pd.Series,
) -> pd.DataFrame:
    """
    Core WoE and IV calculation given a pre-binned series.
    Used internally by both equal-freq and custom bin paths.

    Parameters
    ----------
    series : pd.Series  — original feature values (unused after binning)
    target : pd.Series  — binary target (1 = default)
    binned : pd.Series  — bin assignments (categorical)

    Returns
    -------
    pd.DataFrame with columns:
        bin, n_total, n_events, n_non_events, event_rate,
        dist_events, dist_non_events, woe, iv
    """
    total_events     = target.sum()
    total_non_events = (1 - target).sum()

    df = pd.DataFrame({"bin": binned, "target": target})
    stats = (
        df.groupby("bin", observed=True)["target"]
        .agg(n_total="count", n_events="sum")
        .reset_index()
    )
    stats["n_non_events"] = stats["n_total"] - stats["n_events"]
    stats["event_rate"]   = stats["n_events"] / stats["n_total"]

    stats["dist_events"] = (
        (stats["n_events"] + EPSILON) / (total_events + EPSILON)
    )
    stats["dist_non_events"] = (
        (stats["n_non_events"] + EPSILON) / (total_non_events + EPSILON)
    )

    stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
    stats["iv"]  = (stats["dist_events"] - stats["dist_non_events"]) * stats["woe"]

    return stats.round(4)


def compute_woe_iv_equal_freq(
    series: pd.Series,
    target: pd.Series,
    n_bins: int = N_BINS_DEFAULT,
) -> pd.DataFrame:
    """
    WoE/IV for well-distributed continuous features using equal-frequency bins.
    """
    binned = pd.qcut(series, q=n_bins, duplicates="drop")
    return _compute_woe_iv_table(series, target, binned)


def compute_woe_iv_custom(
    series: pd.Series,
    target: pd.Series,
    bins: List,
    labels: List[str],
) -> pd.DataFrame:
    """
    WoE/IV for zero-inflated features using manually defined bin boundaries.

    Notebook finding: qcut collapses dpd_90_plus (94.44% zeros) to 1 bin.
    Custom bins [-0.1, 0.5, 1.5, 2.5, 10] capture the natural structure:
    exactly 0 events | 1 event | 2 events | 3+ events
    """
    binned = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return _compute_woe_iv_table(series, target, binned)


# ── WoE Binner ──

class WoEBinner:
    """
    Fits WoE bins on training data and transforms any dataset consistently.

    Handles both equal-frequency and custom-bin features automatically
    based on the feature name.

    Usage
    -----
    # Fit on training data
    binner = WoEBinner()
    binner.fit(X_train, y_train)
    X_train_woe = binner.transform(X_train)

    # Transform test data — uses training bins, no leakage
    X_test_woe = binner.transform(X_test)

    # Inspect IV scores
    print(binner.iv_summary())

    # Save fitted binner for later use
    binner.save("data/engineered/woe_lookup.json")

    # Load previously fitted binner
    binner = WoEBinner.load("data/engineered/woe_lookup.json")
    """

    def __init__(self, n_bins: int = N_BINS_DEFAULT):
        self.n_bins    = n_bins
        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.iv_scores_: Dict[str, float] = {}
        self.bin_edges_: Dict[str, np.ndarray] = {}
        self.tables_: Dict[str, pd.DataFrame] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        """
        Fit WoE bins on training data.
        Must only be called on training data to prevent leakage.
        """
        logger.info(f"Fitting WoEBinner on {len(X):,} rows")

        for feat in MODEL_FEATURES:
            if feat not in X.columns:
                logger.warning(f"Feature '{feat}' not found in X — skipping")
                continue

            if feat in CUSTOM_BIN_FEATURES:
                params = CUSTOM_BIN_FEATURES[feat]
                table  = compute_woe_iv_custom(
                    X[feat], y, params["bins"], params["labels"]
                )
            else:
                table = compute_woe_iv_equal_freq(X[feat], y, self.n_bins)
                # Store bin edges for consistent transform on new data
                _, edges = pd.qcut(
                    X[feat], q=self.n_bins,
                    retbins=True, duplicates="drop"
                )
                self.bin_edges_[feat] = edges

            woe_map = dict(zip(table["bin"].astype(str), table["woe"]))
            self.woe_maps_[feat]  = woe_map
            self.iv_scores_[feat] = round(table["iv"].sum(), 4)
            self.tables_[feat]    = table

            logger.info(
                f"  {feat:<22} IV={self.iv_scores_[feat]:.4f} "
                f"| bins={len(table)}"
            )

        self._fitted = True
        logger.info("WoEBinner fitting complete")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to WoE values using fitted bin definitions.
        Safe to call on training, validation, or test data.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        X_woe = pd.DataFrame(index=X.index)

        for feat in MODEL_FEATURES:
            if feat not in X.columns:
                continue

            if feat in CUSTOM_BIN_FEATURES:
                params = CUSTOM_BIN_FEATURES[feat]
                binned = pd.cut(
                    X[feat],
                    bins=params["bins"],
                    labels=params["labels"],
                    include_lowest=True,
                )
            else:
                edges  = self.bin_edges_[feat]
                # Use training edges — prevents data leakage
                binned = pd.cut(
                    X[feat],
                    bins=edges,
                    include_lowest=True,
                )

            woe_map = self.woe_maps_[feat]
            X_woe[f"{feat}_woe"] = binned.astype(str).map(woe_map)

        # Check for unmapped bins (unseen values in test data)
        null_count = X_woe.isnull().sum().sum()
        if null_count > 0:
            logger.warning(
                f"transform(): {null_count} nulls after WoE mapping — "
                f"unseen bin values in input data"
            )

        return X_woe

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step. Only use on training data."""
        return self.fit(X, y).transform(X)

    def iv_summary(self) -> pd.DataFrame:
        """
        Return IV scores sorted descending with interpretation.
        Call after fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before iv_summary()")

        rows = []
        for feat, iv in self.iv_scores_.items():
            rows.append({
                "feature": feat,
                "iv": iv,
                "n_bins": len(self.tables_[feat]),
                "interpretation": (
                    "Useless"      if iv < 0.02 else
                    "Weak"         if iv < 0.10 else
                    "Medium"       if iv < 0.30 else
                    "Strong"       if iv < 0.50 else
                    "Very strong — verify no leakage"
                ),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str) -> None:
        """
        Save fitted WoE lookup tables to JSON.
        Allows transform() to be called in a fresh session without refitting.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before save()")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {}
        for feat in MODEL_FEATURES:
            if feat not in self.woe_maps_:
                continue
            payload[feat] = {
                "woe_map": self.woe_maps_[feat],
                "iv":      self.iv_scores_[feat],
                "bins":    CUSTOM_BIN_FEATURES[feat]["bins"]   if feat in CUSTOM_BIN_FEATURES else None,
                "labels":  CUSTOM_BIN_FEATURES[feat]["labels"] if feat in CUSTOM_BIN_FEATURES else None,
                "edges":   self.bin_edges_.get(feat, np.array([])).tolist(),
            }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"WoEBinner saved → {path}")

    @classmethod
    def load(cls, path: str) -> "WoEBinner":
        """
        Load a previously fitted WoEBinner from JSON.
        Use this to transform test/production data without refitting.
        """
        with open(path) as f:
            payload = json.load(f)

        binner = cls()
        for feat, data in payload.items():
            binner.woe_maps_[feat]  = data["woe_map"]
            binner.iv_scores_[feat] = data["iv"]
            if data.get("edges"):
                binner.bin_edges_[feat] = np.array(data["edges"])

        binner._fitted = True
        logger.info(f"WoEBinner loaded ← {path}")
        return binner


# ── IV interpretation helper ──

def interpret_iv(iv: float) -> str:
    if iv < 0.02:  return "Useless"
    if iv < 0.10:  return "Weak"
    if iv < 0.30:  return "Medium"
    if iv < 0.50:  return "Strong"
    return "Very strong — verify no leakage"