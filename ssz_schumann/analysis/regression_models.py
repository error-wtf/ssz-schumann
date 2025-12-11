#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Analysis

Statistical analysis and model comparison functions.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CrossValidationResult:
    """Result of cross-validation."""
    mean_score: float
    std_score: float
    scores: np.ndarray
    n_folds: int


def cross_validate_models(
    X: pd.DataFrame,
    y_classical_residuals: pd.Series,
    y_ssz_residuals: pd.Series,
    n_folds: int = 5,
) -> Dict[str, CrossValidationResult]:
    """
    Cross-validate classical and SSZ models.
    
    Args:
        X: Feature matrix
        y_classical_residuals: Residuals from classical model
        y_ssz_residuals: Residuals from SSZ model
        n_folds: Number of CV folds
    
    Returns:
        Dictionary with CV results for each model
    """
    results = {}
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Classical model: Can we predict residuals from features?
    # (If yes, classical model is missing something)
    model = LinearRegression()
    
    # Align data
    common_idx = X.index.intersection(y_classical_residuals.index)
    X_aligned = X.loc[common_idx]
    y_class = y_classical_residuals.loc[common_idx]
    
    valid = ~(X_aligned.isna().any(axis=1) | y_class.isna())
    X_clean = X_aligned[valid].values
    y_class_clean = y_class[valid].values
    
    if len(X_clean) > n_folds * 2:
        scores_class = cross_val_score(model, X_clean, y_class_clean, cv=kf, scoring='r2')
        results["classical_residuals"] = CrossValidationResult(
            mean_score=np.mean(scores_class),
            std_score=np.std(scores_class),
            scores=scores_class,
            n_folds=n_folds,
        )
    
    # SSZ model residuals
    common_idx = X.index.intersection(y_ssz_residuals.index)
    X_aligned = X.loc[common_idx]
    y_ssz = y_ssz_residuals.loc[common_idx]
    
    valid = ~(X_aligned.isna().any(axis=1) | y_ssz.isna())
    X_clean = X_aligned[valid].values
    y_ssz_clean = y_ssz[valid].values
    
    if len(X_clean) > n_folds * 2:
        scores_ssz = cross_val_score(model, X_clean, y_ssz_clean, cv=kf, scoring='r2')
        results["ssz_residuals"] = CrossValidationResult(
            mean_score=np.mean(scores_ssz),
            std_score=np.std(scores_ssz),
            scores=scores_ssz,
            n_folds=n_folds,
        )
    
    return results


def bootstrap_confidence_intervals(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for regression coefficients.
    
    Args:
        X: Feature matrix
        y: Target values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Dictionary with CI for each coefficient
    """
    n_samples = len(y)
    n_features = X.shape[1] if X.ndim > 1 else 1
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Storage for bootstrap estimates
    intercepts = []
    coefs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        y_boot = y[idx]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_boot, y_boot)
        
        intercepts.append(model.intercept_)
        coefs.append(model.coef_)
    
    intercepts = np.array(intercepts)
    coefs = np.array(coefs)
    
    # Compute confidence intervals
    alpha = 1 - confidence
    
    results = {
        "intercept": (
            np.percentile(intercepts, 100 * alpha / 2),
            np.percentile(intercepts, 100 * (1 - alpha / 2))
        )
    }
    
    for i in range(n_features):
        results[f"coef_{i}"] = (
            np.percentile(coefs[:, i], 100 * alpha / 2),
            np.percentile(coefs[:, i], 100 * (1 - alpha / 2))
        )
    
    return results


def hypothesis_test_ssz_signature(
    delta_seg_dict: Dict[int, pd.Series],
    significance: float = 0.05,
) -> Dict:
    """
    Statistical test for SSZ signature.
    
    Tests whether delta_seg is consistent across modes,
    which is the key SSZ prediction.
    
    Null hypothesis: delta_seg differs between modes
    Alternative: delta_seg is the same for all modes
    
    Args:
        delta_seg_dict: {mode: delta_seg_series}
        significance: Significance level
    
    Returns:
        Dictionary with test results
    """
    modes = sorted(delta_seg_dict.keys())
    
    if len(modes) < 2:
        return {
            "test": "insufficient_modes",
            "p_value": np.nan,
            "reject_null": False,
            "conclusion": "Need at least 2 modes for test",
        }
    
    # Align all series
    common_idx = delta_seg_dict[modes[0]].index
    for n in modes[1:]:
        common_idx = common_idx.intersection(delta_seg_dict[n].index)
    
    # Create aligned arrays
    arrays = []
    for n in modes:
        arr = delta_seg_dict[n].loc[common_idx].values
        arrays.append(arr)
    
    # Remove NaN
    valid = ~np.any([np.isnan(arr) for arr in arrays], axis=0)
    arrays = [arr[valid] for arr in arrays]
    
    n_points = len(arrays[0])
    
    if n_points < 30:
        return {
            "test": "insufficient_data",
            "p_value": np.nan,
            "reject_null": False,
            "conclusion": f"Only {n_points} valid points, need at least 30",
        }
    
    # Test 1: Paired t-tests between modes
    pairwise_tests = {}
    for i, n1 in enumerate(modes):
        for j, n2 in enumerate(modes):
            if j > i:
                t_stat, p_value = stats.ttest_rel(arrays[i], arrays[j])
                pairwise_tests[f"{n1}_vs_{n2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant_difference": p_value < significance,
                }
    
    # Test 2: Correlation test
    # High correlation supports SSZ (same signal in all modes)
    correlations = {}
    for i, n1 in enumerate(modes):
        for j, n2 in enumerate(modes):
            if j > i:
                r, p = stats.pearsonr(arrays[i], arrays[j])
                correlations[f"{n1}_vs_{n2}"] = {
                    "correlation": r,
                    "p_value": p,
                }
    
    # Test 3: Variance ratio test
    # SSZ predicts variance should be similar across modes
    variances = [np.var(arr) for arr in arrays]
    f_stat, p_levene = stats.levene(*arrays)
    
    # Overall conclusion
    # SSZ signature present if:
    # 1. No significant mean differences (pairwise t-tests)
    # 2. High correlations between modes
    # 3. Similar variances
    
    n_significant_diff = sum(
        1 for test in pairwise_tests.values()
        if test["significant_difference"]
    )
    
    mean_correlation = np.mean([
        c["correlation"] for c in correlations.values()
    ])
    
    ssz_signature_detected = (
        n_significant_diff == 0 and
        mean_correlation > 0.7 and
        p_levene > significance
    )
    
    result = {
        "test": "ssz_signature",
        "n_points": n_points,
        "n_modes": len(modes),
        "pairwise_tests": pairwise_tests,
        "correlations": correlations,
        "variance_test": {
            "f_statistic": f_stat,
            "p_value": p_levene,
            "equal_variances": p_levene > significance,
        },
        "mean_correlation": mean_correlation,
        "ssz_signature_detected": ssz_signature_detected,
        "conclusion": (
            "SSZ signature DETECTED: delta_seg consistent across modes"
            if ssz_signature_detected else
            "SSZ signature NOT detected: modes show different behavior"
        ),
    }
    
    logger.info(f"\nSSZ Signature Test:")
    logger.info(f"  N points: {n_points}")
    logger.info(f"  Mean correlation: {mean_correlation:.4f}")
    logger.info(f"  Significant differences: {n_significant_diff}/{len(pairwise_tests)}")
    logger.info(f"  Equal variances (Levene): p={p_levene:.4f}")
    logger.info(f"  Conclusion: {result['conclusion']}")
    
    return result


def compute_effect_size(
    classical_rmse: float,
    ssz_rmse: float,
    n_points: int,
) -> Dict:
    """
    Compute effect size of SSZ improvement.
    
    Args:
        classical_rmse: RMSE of classical model
        ssz_rmse: RMSE of SSZ model
        n_points: Number of data points
    
    Returns:
        Dictionary with effect size metrics
    """
    # Relative improvement
    relative_improvement = (classical_rmse - ssz_rmse) / classical_rmse
    
    # Cohen's d (standardized effect size)
    # Treating RMSE reduction as effect
    pooled_rmse = np.sqrt((classical_rmse**2 + ssz_rmse**2) / 2)
    cohens_d = (classical_rmse - ssz_rmse) / pooled_rmse
    
    # Interpretation
    if abs(cohens_d) < 0.2:
        interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        interpretation = "small"
    elif abs(cohens_d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        "relative_improvement": relative_improvement,
        "cohens_d": cohens_d,
        "interpretation": interpretation,
        "classical_rmse": classical_rmse,
        "ssz_rmse": ssz_rmse,
        "n_points": n_points,
    }
