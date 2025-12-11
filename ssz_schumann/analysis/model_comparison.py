#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Model Comparison for SSZ vs Classical Schumann Models

Implements model comparison metrics:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Bayes Factor approximation
- Cross-validation scores

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Result of a model fit."""
    name: str
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    rmse: float
    r_squared: float
    residuals: np.ndarray
    predictions: np.ndarray


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    preferred_model: str
    delta_aic: float
    delta_bic: float
    bayes_factor: float
    evidence_strength: str
    classical_result: ModelResult
    ssz_result: ModelResult


def log_likelihood_gaussian(
    residuals: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """
    Calculate log-likelihood assuming Gaussian errors.
    
    Args:
        residuals: Model residuals (observed - predicted)
        sigma: Error standard deviation (estimated from residuals if None)
    
    Returns:
        Log-likelihood value
    """
    n = len(residuals)
    
    if sigma is None:
        sigma = np.std(residuals, ddof=1)
    
    if sigma == 0:
        sigma = 1e-10  # Avoid division by zero
    
    # Gaussian log-likelihood
    ll = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
    
    return ll


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """
    Compute Akaike Information Criterion.
    
    AIC = 2k - 2*ln(L)
    
    Lower is better.
    
    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of free parameters
    
    Returns:
        AIC value
    """
    return 2 * n_params - 2 * log_likelihood


def compute_bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion.
    
    BIC = k*ln(n) - 2*ln(L)
    
    Lower is better. Penalizes complexity more than AIC.
    
    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of free parameters
        n_obs: Number of observations
    
    Returns:
        BIC value
    """
    return n_params * np.log(n_obs) - 2 * log_likelihood


def compute_bayes_factor(delta_bic: float) -> float:
    """
    Approximate Bayes Factor from BIC difference.
    
    BF ~ exp(-delta_BIC / 2)
    
    Args:
        delta_bic: BIC(model1) - BIC(model2)
    
    Returns:
        Approximate Bayes Factor (BF > 1 favors model2)
    """
    return np.exp(-delta_bic / 2)


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes Factor strength (Kass & Raftery 1995).
    
    Args:
        bf: Bayes Factor
    
    Returns:
        Interpretation string
    """
    if bf < 1:
        bf = 1 / bf
        direction = "against"
    else:
        direction = "for"
    
    if bf < 3:
        strength = "Not worth mentioning"
    elif bf < 20:
        strength = "Positive"
    elif bf < 150:
        strength = "Strong"
    else:
        strength = "Very strong"
    
    return f"{strength} evidence {direction} SSZ model (BF = {bf:.2f})"


def fit_classical_model(
    f_obs: Dict[int, np.ndarray],
    eta_0: Optional[float] = None,
) -> ModelResult:
    """
    Fit classical Schumann model.
    
    Model: f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    Free parameter: eta (or eta_0 if calibrated from f1)
    
    Args:
        f_obs: {mode: observed_frequencies}
        eta_0: Baseline eta (calibrated from f1 if None)
    
    Returns:
        ModelResult
    """
    from ..models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    
    # Calibrate eta_0 from f1 if not provided
    if eta_0 is None and 1 in f_obs:
        eta_0 = compute_eta0_from_mean_f1(f_obs[1])
    elif eta_0 is None:
        eta_0 = 0.74
    
    # Compute predictions and residuals
    all_residuals = []
    all_predictions = []
    all_observed = []
    
    for mode, f_observed in f_obs.items():
        f_pred = f_n_classical(mode, eta_0)
        residuals = f_observed - f_pred
        
        all_residuals.extend(residuals)
        all_predictions.extend([f_pred] * len(f_observed))
        all_observed.extend(f_observed)
    
    residuals = np.array(all_residuals)
    predictions = np.array(all_predictions)
    observed = np.array(all_observed)
    
    # Statistics
    n_obs = len(residuals)
    n_params = 1  # Only eta_0
    
    log_ll = log_likelihood_gaussian(residuals)
    aic = compute_aic(log_ll, n_params)
    bic = compute_bic(log_ll, n_params, n_obs)
    rmse = np.sqrt(np.mean(residuals**2))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return ModelResult(
        name="Classical",
        n_params=n_params,
        log_likelihood=log_ll,
        aic=aic,
        bic=bic,
        rmse=rmse,
        r_squared=r_squared,
        residuals=residuals,
        predictions=predictions,
    )


def fit_ssz_model(
    f_obs: Dict[int, np.ndarray],
    proxies: Optional[pd.DataFrame] = None,
    eta_0: Optional[float] = None,
) -> ModelResult:
    """
    Fit SSZ Schumann model.
    
    Model: f_n = f_n_classical / (1 + delta_seg)
           delta_seg = beta_0 + beta_1*F10.7 + beta_2*Kp
    
    Free parameters: eta_0, beta_0, beta_1, beta_2
    
    Args:
        f_obs: {mode: observed_frequencies}
        proxies: DataFrame with F10.7, Kp columns
        eta_0: Baseline eta (calibrated from f1 if None)
    
    Returns:
        ModelResult
    """
    from ..models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    from scipy.optimize import minimize
    
    # Calibrate eta_0
    if eta_0 is None and 1 in f_obs:
        eta_0 = compute_eta0_from_mean_f1(f_obs[1])
    elif eta_0 is None:
        eta_0 = 0.74
    
    # Get number of time points
    n_time = len(list(f_obs.values())[0])
    
    # Prepare proxy data
    if proxies is not None and len(proxies) >= n_time:
        f107 = proxies.iloc[:n_time]['f107_norm'].values if 'f107_norm' in proxies.columns else np.zeros(n_time)
        kp = proxies.iloc[:n_time]['kp_norm'].values if 'kp_norm' in proxies.columns else np.zeros(n_time)
    else:
        f107 = np.zeros(n_time)
        kp = np.zeros(n_time)
    
    # Objective function
    def objective(params):
        beta_0, beta_1, beta_2 = params
        
        # delta_seg time series
        delta_seg = beta_0 + beta_1 * f107 + beta_2 * kp
        
        total_error = 0
        for mode, f_observed in f_obs.items():
            f_class = f_n_classical(mode, eta_0)
            f_pred = f_class / (1 + delta_seg)
            total_error += np.sum((f_observed - f_pred)**2)
        
        return total_error
    
    # Optimize
    x0 = [0.0, 0.001, 0.001]
    bounds = [(-0.1, 0.1), (-0.01, 0.01), (-0.01, 0.01)]
    
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    beta_0, beta_1, beta_2 = result.x
    
    # Compute final predictions and residuals
    delta_seg = beta_0 + beta_1 * f107 + beta_2 * kp
    
    all_residuals = []
    all_predictions = []
    all_observed = []
    
    for mode, f_observed in f_obs.items():
        f_class = f_n_classical(mode, eta_0)
        f_pred = f_class / (1 + delta_seg)
        residuals = f_observed - f_pred
        
        all_residuals.extend(residuals)
        all_predictions.extend(f_pred)
        all_observed.extend(f_observed)
    
    residuals = np.array(all_residuals)
    predictions = np.array(all_predictions)
    observed = np.array(all_observed)
    
    # Statistics
    n_obs = len(residuals)
    n_params = 4  # eta_0, beta_0, beta_1, beta_2
    
    log_ll = log_likelihood_gaussian(residuals)
    aic = compute_aic(log_ll, n_params)
    bic = compute_bic(log_ll, n_params, n_obs)
    rmse = np.sqrt(np.mean(residuals**2))
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return ModelResult(
        name="SSZ",
        n_params=n_params,
        log_likelihood=log_ll,
        aic=aic,
        bic=bic,
        rmse=rmse,
        r_squared=r_squared,
        residuals=residuals,
        predictions=predictions,
    )


def compare_models(
    f_obs: Dict[int, np.ndarray],
    proxies: Optional[pd.DataFrame] = None,
    eta_0: Optional[float] = None,
) -> ComparisonResult:
    """
    Compare Classical vs SSZ models.
    
    Args:
        f_obs: {mode: observed_frequencies}
        proxies: DataFrame with F10.7, Kp columns
        eta_0: Baseline eta
    
    Returns:
        ComparisonResult with comparison metrics
    """
    # Fit both models
    classical = fit_classical_model(f_obs, eta_0)
    ssz = fit_ssz_model(f_obs, proxies, eta_0)
    
    # Compare
    delta_aic = classical.aic - ssz.aic  # Positive = SSZ better
    delta_bic = classical.bic - ssz.bic  # Positive = SSZ better
    
    # Bayes Factor (SSZ vs Classical)
    bayes_factor = compute_bayes_factor(delta_bic)
    evidence_strength = interpret_bayes_factor(bayes_factor)
    
    # Determine preferred model
    if delta_bic > 0:
        preferred = "SSZ"
    else:
        preferred = "Classical"
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON: Classical vs SSZ")
    logger.info("="*60)
    logger.info(f"\nClassical Model:")
    logger.info(f"  Parameters: {classical.n_params}")
    logger.info(f"  Log-Likelihood: {classical.log_likelihood:.2f}")
    logger.info(f"  AIC: {classical.aic:.2f}")
    logger.info(f"  BIC: {classical.bic:.2f}")
    logger.info(f"  RMSE: {classical.rmse:.4f} Hz")
    logger.info(f"  R^2: {classical.r_squared:.4f}")
    
    logger.info(f"\nSSZ Model:")
    logger.info(f"  Parameters: {ssz.n_params}")
    logger.info(f"  Log-Likelihood: {ssz.log_likelihood:.2f}")
    logger.info(f"  AIC: {ssz.aic:.2f}")
    logger.info(f"  BIC: {ssz.bic:.2f}")
    logger.info(f"  RMSE: {ssz.rmse:.4f} Hz")
    logger.info(f"  R^2: {ssz.r_squared:.4f}")
    
    logger.info(f"\nComparison:")
    logger.info(f"  Delta AIC: {delta_aic:.2f} (positive = SSZ better)")
    logger.info(f"  Delta BIC: {delta_bic:.2f} (positive = SSZ better)")
    logger.info(f"  Bayes Factor: {bayes_factor:.2f}")
    logger.info(f"  {evidence_strength}")
    logger.info(f"  Preferred Model: {preferred}")
    
    return ComparisonResult(
        preferred_model=preferred,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
        bayes_factor=bayes_factor,
        evidence_strength=evidence_strength,
        classical_result=classical,
        ssz_result=ssz,
    )


def cross_validate(
    f_obs: Dict[int, np.ndarray],
    proxies: Optional[pd.DataFrame] = None,
    n_folds: int = 5,
) -> Dict:
    """
    Cross-validate Classical vs SSZ models.
    
    Args:
        f_obs: {mode: observed_frequencies}
        proxies: DataFrame with F10.7, Kp columns
        n_folds: Number of cross-validation folds
    
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import KFold
    
    # Get time indices
    n_time = len(list(f_obs.values())[0])
    indices = np.arange(n_time)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    classical_scores = []
    ssz_scores = []
    
    for train_idx, test_idx in kf.split(indices):
        # Split data
        f_obs_train = {m: f[train_idx] for m, f in f_obs.items()}
        f_obs_test = {m: f[test_idx] for m, f in f_obs.items()}
        
        if proxies is not None:
            proxies_train = proxies.iloc[train_idx]
            proxies_test = proxies.iloc[test_idx]
        else:
            proxies_train = None
            proxies_test = None
        
        # Fit on train
        classical = fit_classical_model(f_obs_train)
        ssz = fit_ssz_model(f_obs_train, proxies_train)
        
        # Score on test
        classical_test = fit_classical_model(f_obs_test)
        ssz_test = fit_ssz_model(f_obs_test, proxies_test)
        
        classical_scores.append(classical_test.rmse)
        ssz_scores.append(ssz_test.rmse)
    
    results = {
        "classical_cv_rmse": np.mean(classical_scores),
        "classical_cv_std": np.std(classical_scores),
        "ssz_cv_rmse": np.mean(ssz_scores),
        "ssz_cv_std": np.std(ssz_scores),
        "cv_winner": "SSZ" if np.mean(ssz_scores) < np.mean(classical_scores) else "Classical",
    }
    
    logger.info(f"\nCross-Validation ({n_folds} folds):")
    logger.info(f"  Classical RMSE: {results['classical_cv_rmse']:.4f} +/- {results['classical_cv_std']:.4f}")
    logger.info(f"  SSZ RMSE: {results['ssz_cv_rmse']:.4f} +/- {results['ssz_cv_std']:.4f}")
    logger.info(f"  CV Winner: {results['cv_winner']}")
    
    return results


def print_comparison_summary(result: ComparisonResult):
    """Print formatted comparison summary."""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Classical':<15} {'SSZ':<15}")
    print("-"*55)
    print(f"{'Parameters':<25} {result.classical_result.n_params:<15} {result.ssz_result.n_params:<15}")
    print(f"{'Log-Likelihood':<25} {result.classical_result.log_likelihood:<15.2f} {result.ssz_result.log_likelihood:<15.2f}")
    print(f"{'AIC':<25} {result.classical_result.aic:<15.2f} {result.ssz_result.aic:<15.2f}")
    print(f"{'BIC':<25} {result.classical_result.bic:<15.2f} {result.ssz_result.bic:<15.2f}")
    print(f"{'RMSE (Hz)':<25} {result.classical_result.rmse:<15.4f} {result.ssz_result.rmse:<15.4f}")
    print(f"{'R^2':<25} {result.classical_result.r_squared:<15.4f} {result.ssz_result.r_squared:<15.4f}")
    
    print(f"\n{'Decision Metrics:'}")
    print("-"*55)
    print(f"Delta AIC: {result.delta_aic:+.2f} (positive favors SSZ)")
    print(f"Delta BIC: {result.delta_bic:+.2f} (positive favors SSZ)")
    print(f"Bayes Factor: {result.bayes_factor:.2f}")
    print(f"\n{result.evidence_strength}")
    print(f"\nPREFERRED MODEL: {result.preferred_model}")
    print("="*70)
