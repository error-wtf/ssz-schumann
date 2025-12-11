#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Correction Model for Schumann Resonances

Implements the Segmented Spacetime (SSZ) correction to classical
Schumann resonance theory.

SSZ Theory Background:
    In SSZ, spacetime is discretized into segments, with segment
    density depending on local conditions. This introduces an
    additional "propagation delay" factor D_SSZ.
    
    For Schumann resonances:
        f_n_observed = f_n_classical / D_SSZ
    
    where:
        D_SSZ = 1 + delta_seg
    
    The key SSZ signature is that delta_seg is the SAME for all modes,
    producing a uniform relative frequency shift:
        
        Delta_f_n / f_n = -delta_seg  (for all n)
    
    This is distinct from classical dispersive effects, which can
    produce mode-dependent shifts.

Model:
    delta_seg(t) = beta_0 + beta_1 * F_iono(t) + beta_2 * F_thunder(t) + ...
    
    where F_iono is a normalized ionospheric proxy (e.g., F10.7, Kp)
    and F_thunder is a lightning activity proxy.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

from ..config import PHI

logger = logging.getLogger(__name__)

# Type alias
ArrayLike = Union[float, np.ndarray, pd.Series]


def D_SSZ(delta_seg: ArrayLike) -> ArrayLike:
    """
    Calculate SSZ correction factor from delta_seg.
    
    Formula:
        D_SSZ = 1 + delta_seg
    
    Args:
        delta_seg: Segmentation correction (small, typically |delta_seg| < 0.1)
    
    Returns:
        D_SSZ correction factor
    
    Physical Interpretation:
        D_SSZ > 1: Effective light speed reduced -> frequencies lower
        D_SSZ < 1: Effective light speed increased -> frequencies higher
        D_SSZ = 1: No SSZ correction (classical limit)
    
    Note:
        This is the simplest form. More sophisticated models could use:
        D_SSZ = 1 / (1 + Xi(r))  [from ssz-metric-pure]
        
        For Earth-scale effects, the linear approximation is sufficient.
    """
    return 1.0 + delta_seg


def f_n_ssz_model(
    f_n_classical: ArrayLike,
    delta_seg: ArrayLike,
) -> ArrayLike:
    """
    Apply SSZ correction to classical frequency.
    
    Formula:
        f_n_ssz = f_n_classical / D_SSZ
               = f_n_classical / (1 + delta_seg)
    
    For small delta_seg:
        f_n_ssz ≈ f_n_classical * (1 - delta_seg)
    
    Args:
        f_n_classical: Classical frequency (Hz)
        delta_seg: SSZ correction factor
    
    Returns:
        SSZ-corrected frequency (Hz)
    """
    d_ssz = D_SSZ(delta_seg)
    return f_n_classical / d_ssz


def delta_seg_from_observed(
    f_obs: ArrayLike,
    f_classical: ArrayLike,
) -> ArrayLike:
    """
    Extract delta_seg from observed and classical frequencies.
    
    Inverts the SSZ model:
        f_obs = f_classical / (1 + delta_seg)
        
        => delta_seg = f_classical / f_obs - 1
                     = (f_classical - f_obs) / f_obs
    
    For small deviations:
        delta_seg ≈ -(f_obs - f_classical) / f_classical
    
    Args:
        f_obs: Observed frequency (Hz)
        f_classical: Classical model frequency (Hz)
    
    Returns:
        Extracted delta_seg values
    
    Example:
        >>> f_obs = 7.80  # Hz
        >>> f_classical = 7.83  # Hz
        >>> delta_seg = delta_seg_from_observed(f_obs, f_classical)
        >>> print(f"delta_seg = {delta_seg:.4f}")
        delta_seg = 0.0038
    """
    # Exact formula
    delta_seg = f_classical / f_obs - 1.0
    
    # Equivalent: -(f_obs - f_classical) / f_classical for small deviations
    # delta_seg_approx = -(f_obs - f_classical) / f_classical
    
    return delta_seg


def compute_delta_seg_all_modes(
    f_obs_dict: Dict[int, ArrayLike],
    f_classical_dict: Dict[int, float],
) -> Dict[int, ArrayLike]:
    """
    Compute delta_seg for multiple modes.
    
    Args:
        f_obs_dict: {mode_n: observed_frequency}
        f_classical_dict: {mode_n: classical_frequency}
    
    Returns:
        {mode_n: delta_seg_n}
    
    Example:
        >>> f_obs = {1: f1_series, 2: f2_series, 3: f3_series}
        >>> f_class = {1: 7.83, 2: 14.1, 3: 20.3}
        >>> delta_seg = compute_delta_seg_all_modes(f_obs, f_class)
    """
    delta_seg_dict = {}
    
    for n in f_obs_dict:
        if n in f_classical_dict:
            delta_seg_dict[n] = delta_seg_from_observed(
                f_obs_dict[n],
                f_classical_dict[n]
            )
    
    return delta_seg_dict


def check_mode_consistency(
    delta_seg_dict: Dict[int, ArrayLike],
    tolerance: float = 0.01,
    corr_threshold: float = 0.7,
    score_threshold: float = 0.7,
    std_ref: float = 0.01,
) -> Dict[str, float]:
    """
    Check if delta_seg is consistent across modes (SSZ signature).
    
    The key SSZ prediction is that delta_seg should be the SAME
    for all modes. This function quantifies the consistency.
    
    SSZ Signature Criteria (from THEORY.md):
        1. High mode correlation (> corr_threshold, default 0.7)
        2. Low dispersion across modes (std_across_modes < std_ref)
        3. SSZ score > score_threshold
    
    SSZ Score Formula:
        ssz_score = mean_correlation * max(0, 1 - std_across_modes / std_ref)
    
    Args:
        delta_seg_dict: {mode_n: delta_seg_n} from compute_delta_seg_all_modes
        tolerance: Acceptable relative difference (legacy, for backward compat)
        corr_threshold: Minimum correlation for SSZ signature (default 0.7)
        score_threshold: Minimum SSZ score for consistency (default 0.7)
        std_ref: Reference std for normalization (default 0.01 = 1%)
    
    Returns:
        Dictionary with consistency metrics:
            - mean_delta_seg: Average across modes
            - std_delta_seg: Standard deviation across modes (mean over time)
            - std_across_modes: Same as std_delta_seg (explicit name)
            - max_diff: Maximum pairwise difference
            - correlations: Dict with r_12, r_13, r_23 pairwise correlations
            - mean_correlation: Average of pairwise correlations
            - ssz_score: 0-1 score (1 = perfect SSZ consistency)
            - is_consistent: Boolean (True if SSZ signature detected)
            - interpretation: String describing the result
    
    Example:
        >>> metrics = check_mode_consistency(delta_seg_dict)
        >>> print(f"SSZ Score: {metrics['ssz_score']:.4f}")
        >>> print(f"Mean Correlation: {metrics['mean_correlation']:.4f}")
        >>> print(metrics['interpretation'])
    """
    modes = sorted(delta_seg_dict.keys())
    n_modes = len(modes)
    
    if n_modes < 2:
        return {
            "mean_delta_seg": np.nan,
            "std_delta_seg": np.nan,
            "std_across_modes": np.nan,
            "max_diff": np.nan,
            "correlations": {},
            "mean_correlation": np.nan,
            "is_consistent": False,
            "ssz_score": 0.0,
            "interpretation": "Need at least 2 modes for consistency check",
            "n_modes": n_modes,
            "n_timepoints": 0,
        }
    
    # Convert to arrays for computation
    delta_arrays = {}
    for n in modes:
        ds = delta_seg_dict[n]
        if isinstance(ds, pd.Series):
            delta_arrays[n] = ds.values
        else:
            delta_arrays[n] = np.atleast_1d(ds)
    
    # Ensure same length
    min_len = min(len(arr) for arr in delta_arrays.values())
    for n in modes:
        delta_arrays[n] = delta_arrays[n][:min_len]
    
    # Stack into matrix (modes x time)
    delta_matrix = np.vstack([delta_arrays[n] for n in modes])
    
    # Mean delta_seg across modes at each time
    mean_delta_seg = np.nanmean(delta_matrix, axis=0)
    
    # Standard deviation across modes at each time
    std_across_modes = np.nanstd(delta_matrix, axis=0)
    
    # Overall statistics
    overall_mean = np.nanmean(mean_delta_seg)
    overall_std = np.nanmean(std_across_modes)
    
    # Pairwise differences
    max_diff = 0.0
    correlations = {}
    
    for i, n1 in enumerate(modes):
        for j, n2 in enumerate(modes):
            if j > i:
                diff = np.abs(delta_arrays[n1] - delta_arrays[n2])
                max_diff = max(max_diff, np.nanmax(diff))
                
                # Correlation
                valid = ~(np.isnan(delta_arrays[n1]) | np.isnan(delta_arrays[n2]))
                if np.sum(valid) > 10:
                    corr = np.corrcoef(
                        delta_arrays[n1][valid],
                        delta_arrays[n2][valid]
                    )[0, 1]
                else:
                    corr = np.nan
                
                correlations[f"{n1}_{n2}"] = corr
    
    # Mean correlation
    valid_corrs = [c for c in correlations.values() if not np.isnan(c)]
    mean_corr = np.mean(valid_corrs) if valid_corrs else np.nan
    
    # Rename correlations to explicit r_12, r_13, r_23 format
    correlations_explicit = {}
    for key, val in correlations.items():
        parts = key.split('_')
        if len(parts) == 2:
            correlations_explicit[f"r_{parts[0]}{parts[1]}"] = val
    
    # SSZ Score Calculation (from THEORY.md)
    # Formula: ssz_score = mean_correlation * max(0, 1 - std_across_modes / std_ref)
    #
    # This captures both requirements:
    # 1. High correlation between modes (same signal)
    # 2. Low dispersion across modes (std_across_modes small)
    
    if not np.isnan(mean_corr):
        # Penalize by std_across_modes relative to std_ref
        # If std_across_modes = 0, penalty = 0 -> full correlation score
        # If std_across_modes = std_ref, penalty = 1 -> score = 0
        std_penalty = min(overall_std / std_ref, 1.0) if std_ref > 0 else 0.0
        ssz_score = max(0.0, mean_corr * (1.0 - std_penalty))
    else:
        ssz_score = 0.0
    
    # Consistency check based on clear thresholds (from THEORY.md)
    # SSZ signature is present if:
    # 1. Mean correlation > corr_threshold (default 0.7)
    # 2. SSZ score > score_threshold (default 0.7)
    is_consistent = (
        not np.isnan(mean_corr) and
        mean_corr > corr_threshold and
        ssz_score > score_threshold
    )
    
    # Generate interpretation string
    if is_consistent:
        interpretation = "Strong SSZ signature detected"
    elif mean_corr > 0.5 or ssz_score > 0.3:
        interpretation = "Weak/partial SSZ signature"
    else:
        interpretation = "No SSZ signature detected"
    
    result = {
        "mean_delta_seg": overall_mean,
        "std_delta_seg": overall_std,
        "std_across_modes": overall_std,  # Explicit alias
        "max_diff": max_diff,
        "correlations": correlations_explicit,
        "mean_correlation": mean_corr,
        "is_consistent": is_consistent,
        "ssz_score": ssz_score,
        "interpretation": interpretation,
        "n_modes": n_modes,
        "n_timepoints": min_len,
        # Thresholds used (for transparency)
        "corr_threshold": corr_threshold,
        "score_threshold": score_threshold,
        "std_ref": std_ref,
    }
    
    # Log results
    logger.info(f"Mode consistency check:")
    logger.info(f"  Mean delta_seg: {overall_mean:.6f} ({overall_mean*100:.4f}%)")
    logger.info(f"  Std across modes: {overall_std:.6f} ({overall_std*100:.4f}%)")
    logger.info(f"  Mean correlation: {mean_corr:.4f}")
    logger.info(f"  SSZ score: {ssz_score:.4f}")
    logger.info(f"  Interpretation: {interpretation}")
    
    return result


@dataclass
class FitResult:
    """Result of delta_seg fitting."""
    beta_0: float
    beta_1: float
    beta_2: float
    r_squared: float
    rmse: float
    n_points: int
    feature_names: List[str]
    residuals: Optional[np.ndarray] = None


def fit_delta_seg_simple(
    features: pd.DataFrame,
    delta_seg_target: pd.Series,
    model_type: str = "linear",
) -> FitResult:
    """
    Fit delta_seg as function of ionospheric/lightning proxies.
    
    Model:
        delta_seg(t) = beta_0 + beta_1 * F_iono(t) + beta_2 * F_thunder(t) + ...
    
    Args:
        features: DataFrame with feature columns (e.g., f107_norm, kp_norm)
        delta_seg_target: Target delta_seg values
        model_type: "linear" (OLS) or "robust" (Huber)
    
    Returns:
        FitResult with coefficients and quality metrics
    
    Example:
        >>> features = df[["f107_norm", "kp_norm"]]
        >>> result = fit_delta_seg_simple(features, delta_seg_mean)
        >>> print(f"beta_0 = {result.beta_0:.6f}")
        >>> print(f"beta_1 (F10.7) = {result.beta_1:.6f}")
        >>> print(f"R² = {result.r_squared:.4f}")
    """
    from sklearn.linear_model import LinearRegression, HuberRegressor
    
    # Align indices
    common_idx = features.index.intersection(delta_seg_target.index)
    X = features.loc[common_idx].values
    y = delta_seg_target.loc[common_idx].values
    
    # Remove NaN
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]
    
    n_points = len(y)
    if n_points < 10:
        raise ValueError(f"Not enough valid data points: {n_points}")
    
    # Fit model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "robust":
        model = HuberRegressor()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.fit(X, y)
    
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Extract coefficients
    beta_0 = model.intercept_
    betas = model.coef_
    
    # Pad to at least 2 coefficients
    beta_1 = betas[0] if len(betas) > 0 else 0.0
    beta_2 = betas[1] if len(betas) > 1 else 0.0
    
    result = FitResult(
        beta_0=beta_0,
        beta_1=beta_1,
        beta_2=beta_2,
        r_squared=r_squared,
        rmse=rmse,
        n_points=n_points,
        feature_names=list(features.columns),
        residuals=residuals,
    )
    
    logger.info(f"Fit result (n={n_points}):")
    logger.info(f"  beta_0 = {beta_0:.6f}")
    for i, name in enumerate(features.columns):
        logger.info(f"  beta_{i+1} ({name}) = {betas[i]:.6f}")
    logger.info(f"  R² = {r_squared:.4f}")
    logger.info(f"  RMSE = {rmse:.6f}")
    
    return result


def predict_delta_seg(
    features: pd.DataFrame,
    fit_result: FitResult,
) -> pd.Series:
    """
    Predict delta_seg using fitted model.
    
    Args:
        features: Feature DataFrame
        fit_result: Result from fit_delta_seg_simple
    
    Returns:
        Predicted delta_seg series
    """
    X = features.values
    
    # Reconstruct coefficients
    betas = [fit_result.beta_1]
    if fit_result.beta_2 != 0:
        betas.append(fit_result.beta_2)
    betas = np.array(betas[:X.shape[1]])
    
    y_pred = fit_result.beta_0 + X @ betas
    
    return pd.Series(y_pred, index=features.index, name="delta_seg_predicted")


def ssz_model_frequencies(
    eta_0: float,
    delta_seg: ArrayLike,
    modes: List[int] = [1, 2, 3],
    R_earth: float = None,
    c: float = None,
) -> Dict[int, ArrayLike]:
    """
    Compute SSZ model frequencies for multiple modes.
    
    Args:
        eta_0: Baseline slowdown factor
        delta_seg: SSZ correction (scalar or time series)
        modes: List of mode numbers
        R_earth: Earth radius (uses default if None)
        c: Speed of light (uses default if None)
    
    Returns:
        {mode_n: f_n_ssz}
    """
    from .classical_schumann import f_n_classical
    from ..config import EARTH_RADIUS, C_LIGHT
    
    if R_earth is None:
        R_earth = EARTH_RADIUS
    if c is None:
        c = C_LIGHT
    
    result = {}
    for n in modes:
        f_class = f_n_classical(n, eta_0, R_earth, c)
        f_ssz = f_n_ssz_model(f_class, delta_seg)
        result[n] = f_ssz
    
    return result
