#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Fitting Wrappers

Provides high-level functions for fitting and comparing
classical and SSZ Schumann resonance models.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from .classical_schumann import (
    f_n_classical,
    compute_eta0_from_mean_f1,
    f_n_classical_timeseries,
)
from .ssz_correction import (
    delta_seg_from_observed,
    compute_delta_seg_all_modes,
    check_mode_consistency,
    fit_delta_seg_simple,
    f_n_ssz_model,
    FitResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """
    Complete result of model fitting.
    
    Contains all parameters, predictions, and quality metrics
    for either classical or SSZ model.
    """
    model_type: str  # "classical" or "ssz"
    
    # Parameters
    eta_0: float
    ssz_params: Optional[Dict[str, float]] = None
    
    # Predictions
    f_predicted: Optional[Dict[int, pd.Series]] = None
    delta_seg_predicted: Optional[pd.Series] = None
    
    # Residuals
    residuals: Optional[Dict[int, pd.Series]] = None
    
    # Quality metrics
    r_squared: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_params: int = 0
    n_points: int = 0
    
    # Mode consistency (SSZ only)
    mode_consistency: Optional[Dict] = None
    
    # Additional info
    info: Dict = field(default_factory=dict)


def fit_classical_model(
    ds: xr.Dataset,
    modes: List[int] = [1, 2, 3],
    features: Optional[pd.DataFrame] = None,
    fit_eta_variation: bool = False,
) -> ModelResult:
    """
    Fit classical Schumann model to data.
    
    Two variants:
    1. Simple: Constant eta_0 calibrated from mean f1
    2. Extended: eta(t) as function of features (if fit_eta_variation=True)
    
    Args:
        ds: Dataset with f1, f2, f3 observations
        modes: Mode numbers to fit
        features: Optional features for eta(t) fitting
        fit_eta_variation: Whether to fit time-varying eta
    
    Returns:
        ModelResult with fitted parameters and predictions
    """
    # Extract observed frequencies
    f_obs = {}
    for n in modes:
        if f"f{n}" in ds:
            f_obs[n] = ds[f"f{n}"].to_series()
    
    if 1 not in f_obs:
        raise ValueError("f1 required for calibration")
    
    # Calibrate eta_0 from mean f1
    eta_0 = compute_eta0_from_mean_f1(f_obs[1])
    
    # Compute classical predictions
    f_predicted = {}
    residuals = {}
    
    for n in modes:
        if n in f_obs:
            f_class = f_n_classical(n, eta_0)
            f_predicted[n] = pd.Series(
                f_class,
                index=f_obs[n].index,
                name=f"f{n}_classical"
            )
            residuals[n] = f_obs[n] - f_class
    
    # Compute overall metrics
    all_residuals = np.concatenate([r.dropna().values for r in residuals.values()])
    n_points = len(all_residuals)
    
    ss_res = np.sum(all_residuals**2)
    
    # Total sum of squares (pooled across modes)
    all_obs = np.concatenate([f_obs[n].dropna().values for n in modes if n in f_obs])
    ss_tot = np.sum((all_obs - np.mean(all_obs))**2)
    
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(all_residuals**2))
    
    # Information criteria
    n_params = 1  # Just eta_0
    aic = n_points * np.log(ss_res / n_points) + 2 * n_params
    bic = n_points * np.log(ss_res / n_points) + n_params * np.log(n_points)
    
    result = ModelResult(
        model_type="classical",
        eta_0=eta_0,
        f_predicted=f_predicted,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_params=n_params,
        n_points=n_points,
        info={
            "modes": modes,
            "fit_eta_variation": fit_eta_variation,
        }
    )
    
    logger.info(f"Classical model fit:")
    logger.info(f"  eta_0 = {eta_0:.6f}")
    logger.info(f"  R² = {r_squared:.4f}")
    logger.info(f"  RMSE = {rmse:.4f} Hz")
    logger.info(f"  AIC = {aic:.1f}, BIC = {bic:.1f}")
    
    return result


def fit_ssz_model(
    ds: xr.Dataset,
    features: pd.DataFrame,
    modes: List[int] = [1, 2, 3],
    feature_columns: Optional[List[str]] = None,
) -> ModelResult:
    """
    Fit SSZ Schumann model to data.
    
    Model:
        f_n(t) = f_n_classical / (1 + delta_seg(t))
        delta_seg(t) = beta_0 + beta_1 * F1(t) + beta_2 * F2(t) + ...
    
    Args:
        ds: Dataset with f1, f2, f3 observations
        features: DataFrame with feature columns
        modes: Mode numbers to fit
        feature_columns: Which columns to use (default: all)
    
    Returns:
        ModelResult with fitted parameters and predictions
    """
    # Extract observed frequencies
    f_obs = {}
    for n in modes:
        if f"f{n}" in ds:
            f_obs[n] = ds[f"f{n}"].to_series()
    
    if 1 not in f_obs:
        raise ValueError("f1 required for calibration")
    
    # Calibrate eta_0 from mean f1
    eta_0 = compute_eta0_from_mean_f1(f_obs[1])
    
    # Compute classical frequencies
    f_classical = {n: f_n_classical(n, eta_0) for n in modes}
    
    # Extract delta_seg from observations
    delta_seg_dict = compute_delta_seg_all_modes(f_obs, f_classical)
    
    # Check mode consistency (SSZ signature)
    consistency = check_mode_consistency(delta_seg_dict)
    
    # Use mean delta_seg across modes as target
    delta_seg_arrays = [
        delta_seg_dict[n].values if isinstance(delta_seg_dict[n], pd.Series)
        else np.full(len(f_obs[1]), delta_seg_dict[n])
        for n in modes if n in delta_seg_dict
    ]
    delta_seg_mean = pd.Series(
        np.nanmean(delta_seg_arrays, axis=0),
        index=f_obs[1].index,
        name="delta_seg_mean"
    )
    
    # Select features
    if feature_columns is None:
        feature_columns = [c for c in features.columns if "_norm" in c]
    
    if not feature_columns:
        feature_columns = list(features.columns)[:2]
    
    X = features[feature_columns]
    
    # Fit delta_seg model
    fit_result = fit_delta_seg_simple(X, delta_seg_mean)
    
    # Predict delta_seg
    from .ssz_correction import predict_delta_seg
    delta_seg_predicted = predict_delta_seg(X, fit_result)
    
    # Compute SSZ model predictions
    f_predicted = {}
    residuals = {}
    
    for n in modes:
        if n in f_obs:
            f_ssz = f_n_ssz_model(f_classical[n], delta_seg_predicted)
            f_predicted[n] = pd.Series(
                f_ssz.values,
                index=delta_seg_predicted.index,
                name=f"f{n}_ssz"
            )
            
            # Align for residual computation
            common_idx = f_obs[n].index.intersection(f_predicted[n].index)
            residuals[n] = f_obs[n].loc[common_idx] - f_predicted[n].loc[common_idx]
    
    # Compute overall metrics
    all_residuals = np.concatenate([r.dropna().values for r in residuals.values()])
    n_points = len(all_residuals)
    
    ss_res = np.sum(all_residuals**2)
    
    all_obs = np.concatenate([
        f_obs[n].loc[f_obs[n].index.intersection(f_predicted[n].index)].dropna().values
        for n in modes if n in f_obs
    ])
    ss_tot = np.sum((all_obs - np.mean(all_obs))**2)
    
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(all_residuals**2))
    
    # Information criteria
    n_params = 1 + len(feature_columns) + 1  # eta_0 + betas + intercept
    aic = n_points * np.log(ss_res / n_points) + 2 * n_params
    bic = n_points * np.log(ss_res / n_points) + n_params * np.log(n_points)
    
    # SSZ parameters
    ssz_params = {
        "beta_0": fit_result.beta_0,
        "beta_1": fit_result.beta_1,
        "beta_2": fit_result.beta_2,
    }
    for i, col in enumerate(feature_columns):
        ssz_params[f"feature_{i}"] = col
    
    result = ModelResult(
        model_type="ssz",
        eta_0=eta_0,
        ssz_params=ssz_params,
        f_predicted=f_predicted,
        delta_seg_predicted=delta_seg_predicted,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_params=n_params,
        n_points=n_points,
        mode_consistency=consistency,
        info={
            "modes": modes,
            "feature_columns": feature_columns,
            "fit_result": fit_result,
        }
    )
    
    logger.info(f"SSZ model fit:")
    logger.info(f"  eta_0 = {eta_0:.6f}")
    logger.info(f"  beta_0 = {fit_result.beta_0:.6f}")
    logger.info(f"  beta_1 = {fit_result.beta_1:.6f}")
    logger.info(f"  R² = {r_squared:.4f}")
    logger.info(f"  RMSE = {rmse:.4f} Hz")
    logger.info(f"  AIC = {aic:.1f}, BIC = {bic:.1f}")
    logger.info(f"  Mode consistency score: {consistency['ssz_score']:.4f}")
    
    return result


def compare_models(
    classical_result: ModelResult,
    ssz_result: ModelResult,
) -> Dict:
    """
    Compare classical and SSZ model fits.
    
    Args:
        classical_result: Result from fit_classical_model
        ssz_result: Result from fit_ssz_model
    
    Returns:
        Dictionary with comparison metrics:
            - delta_r_squared: R²(SSZ) - R²(classical)
            - delta_rmse: RMSE(SSZ) - RMSE(classical)
            - delta_aic: AIC(SSZ) - AIC(classical)
            - delta_bic: BIC(SSZ) - BIC(classical)
            - preferred_model: "ssz" or "classical"
            - ssz_improvement: Relative improvement in R²
    """
    comparison = {
        "classical": {
            "r_squared": classical_result.r_squared,
            "rmse": classical_result.rmse,
            "aic": classical_result.aic,
            "bic": classical_result.bic,
            "n_params": classical_result.n_params,
        },
        "ssz": {
            "r_squared": ssz_result.r_squared,
            "rmse": ssz_result.rmse,
            "aic": ssz_result.aic,
            "bic": ssz_result.bic,
            "n_params": ssz_result.n_params,
            "mode_consistency_score": ssz_result.mode_consistency.get("ssz_score", 0),
        },
        "delta_r_squared": ssz_result.r_squared - classical_result.r_squared,
        "delta_rmse": ssz_result.rmse - classical_result.rmse,
        "delta_aic": ssz_result.aic - classical_result.aic,
        "delta_bic": ssz_result.bic - classical_result.bic,
    }
    
    # Relative improvement
    if classical_result.r_squared > 0:
        comparison["ssz_improvement"] = (
            (ssz_result.r_squared - classical_result.r_squared) /
            classical_result.r_squared
        )
    else:
        comparison["ssz_improvement"] = np.inf if ssz_result.r_squared > 0 else 0
    
    # Model selection
    # Prefer SSZ if:
    # 1. Lower AIC/BIC (accounting for extra parameters)
    # 2. Good mode consistency (SSZ signature)
    
    ssz_preferred = (
        comparison["delta_aic"] < 0 and
        comparison["delta_bic"] < 0 and
        ssz_result.mode_consistency.get("ssz_score", 0) > 0.5
    )
    
    comparison["preferred_model"] = "ssz" if ssz_preferred else "classical"
    
    # Summary message
    if ssz_preferred:
        comparison["summary"] = (
            f"SSZ model preferred: "
            f"ΔR²={comparison['delta_r_squared']:+.4f}, "
            f"ΔAIC={comparison['delta_aic']:+.1f}, "
            f"SSZ score={ssz_result.mode_consistency.get('ssz_score', 0):.3f}"
        )
    else:
        comparison["summary"] = (
            f"Classical model preferred: "
            f"SSZ does not significantly improve fit or lacks mode consistency"
        )
    
    logger.info(f"\nModel comparison:")
    logger.info(f"  Classical: R²={classical_result.r_squared:.4f}, "
                f"RMSE={classical_result.rmse:.4f}, AIC={classical_result.aic:.1f}")
    logger.info(f"  SSZ:       R²={ssz_result.r_squared:.4f}, "
                f"RMSE={ssz_result.rmse:.4f}, AIC={ssz_result.aic:.1f}")
    logger.info(f"  {comparison['summary']}")
    
    return comparison
