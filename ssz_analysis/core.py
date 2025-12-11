# -*- coding: utf-8 -*-
"""
SSZ Schumann Analysis - Core Functions

This module provides the core analysis functions for testing SSZ predictions
against real Schumann resonance data.

Key concepts:
- Classical Schumann: f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
- SSZ relationship: f_obs = f_classical / D_SSZ, where D_SSZ = 1 + delta_seg
- delta_seg_classical: offset vs classical frequencies
- delta_seg_anomaly: deviation from mode mean (for correlation analysis)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

# Physical constants
C = 299792458.0  # Speed of light [m/s]
R_EARTH = 6.371e6  # Earth radius [m]

# Mode numbers for Schumann resonances
MODE_NUMBERS = {'f1': 1, 'f2': 2, 'f3': 3, 'f4': 4}


@dataclass
class ClassicalReference:
    """Classical Schumann frequency reference."""
    eta: float
    f1: float
    f2: float
    f3: float
    f4: float
    
    def as_dict(self) -> Dict[str, float]:
        return {'f1': self.f1, 'f2': self.f2, 'f3': self.f3, 'f4': self.f4}


@dataclass
class SSZResult:
    """Result of SSZ hypothesis test."""
    delta_ssz_global: float
    delta_ssz_std: float
    chi_squared: float
    ndof: int
    chi_squared_reduced: float
    p_value: float
    is_consistent: bool
    upper_bound_95: float
    mode_deltas: Dict[str, float]
    mode_delta_errors: Dict[str, float]


def load_schumann_data(filepath: Path) -> pd.DataFrame:
    """
    Load processed Schumann data from CSV.
    
    Parameters
    ----------
    filepath : Path
        Path to the processed CSV file (e.g., schumann_1310_processed.csv)
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: timestamp, f1, f2, f3, f4, f1_amp, etc.
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    return df


def compute_classical_frequencies(n: int, eta: float = 0.74) -> float:
    """
    Compute classical Schumann frequency for mode n.
    
    f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    Parameters
    ----------
    n : int
        Mode number (1, 2, 3, 4, ...)
    eta : float
        Ionospheric correction factor (default 0.74)
    
    Returns
    -------
    f_n : float
        Classical frequency in Hz
    """
    return eta * C / (2 * np.pi * R_EARTH) * np.sqrt(n * (n + 1))


def get_classical_reference(eta: float = 0.74) -> ClassicalReference:
    """Get classical reference frequencies for all modes."""
    return ClassicalReference(
        eta=eta,
        f1=compute_classical_frequencies(1, eta),
        f2=compute_classical_frequencies(2, eta),
        f3=compute_classical_frequencies(3, eta),
        f4=compute_classical_frequencies(4, eta),
    )


def fit_classical_eta(
    f_obs: Dict[str, float],
    f_obs_err: Optional[Dict[str, float]] = None,
    method: str = 'least_squares'
) -> Tuple[float, float]:
    """
    Fit optimal eta to match observed frequencies.
    
    Parameters
    ----------
    f_obs : dict
        Observed mean frequencies {'f1': ..., 'f2': ..., 'f3': ..., 'f4': ...}
    f_obs_err : dict, optional
        Uncertainties on observed frequencies
    method : str
        'least_squares' - minimize sum of squared residuals
        'f1_match' - set eta so f1_classical = f1_observed
    
    Returns
    -------
    eta_best : float
        Best-fit eta
    eta_err : float
        Uncertainty on eta (0 if not computed)
    """
    modes = ['f1', 'f2', 'f3', 'f4']
    n_values = [1, 2, 3, 4]
    
    if method == 'f1_match':
        # Simple: eta such that f1_classical = f1_observed
        f1_ideal = C / (2 * np.pi * R_EARTH) * np.sqrt(2)  # eta=1
        eta_best = f_obs['f1'] / f1_ideal
        return eta_best, 0.0
    
    elif method == 'least_squares':
        # Minimize sum of squared residuals
        def objective(eta):
            residuals = 0.0
            for mode, n in zip(modes, n_values):
                if mode in f_obs:
                    f_theo = compute_classical_frequencies(n, eta)
                    weight = 1.0
                    if f_obs_err and mode in f_obs_err and f_obs_err[mode] > 0:
                        weight = 1.0 / f_obs_err[mode]**2
                    residuals += weight * (f_obs[mode] - f_theo)**2
            return residuals
        
        result = minimize_scalar(objective, bounds=(0.5, 1.0), method='bounded')
        eta_best = result.x
        
        # Estimate uncertainty via curvature
        h = 0.001
        d2f = (objective(eta_best + h) - 2*objective(eta_best) + objective(eta_best - h)) / h**2
        if d2f > 0:
            eta_err = np.sqrt(1.0 / d2f)
        else:
            eta_err = 0.0
        
        return eta_best, eta_err
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_delta_seg_classical(
    f_obs: float,
    f_classical: float
) -> float:
    """
    Compute SSZ segment density deviation vs classical prediction.
    
    delta_seg_classical = -(f_obs - f_classical) / f_classical
    
    Interpretation:
    - Positive delta_seg: f_obs < f_classical (slower effective c)
    - Negative delta_seg: f_obs > f_classical (faster effective c)
    
    Parameters
    ----------
    f_obs : float
        Observed frequency
    f_classical : float
        Classical reference frequency
    
    Returns
    -------
    delta_seg : float
        Segment density deviation (dimensionless)
    """
    return -(f_obs - f_classical) / f_classical


def compute_delta_seg_anomaly(
    f_obs: np.ndarray,
    f_mean: float
) -> np.ndarray:
    """
    Compute SSZ anomaly relative to mode mean.
    
    delta_seg_anomaly(t) = -(f_obs(t) - f_mean) / f_mean
    
    This is useful for correlation analysis where we want to remove
    the constant offset and focus on temporal variations.
    
    Parameters
    ----------
    f_obs : array
        Time series of observed frequencies
    f_mean : float
        Mean frequency for this mode
    
    Returns
    -------
    delta_anomaly : array
        Anomaly time series
    """
    return -(f_obs - f_mean) / f_mean


def estimate_delta_ssz_global(
    daily_data: pd.DataFrame,
    classical_ref: ClassicalReference,
    use_weights: bool = True
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """
    Estimate a single global delta_SSZ from all modes.
    
    SSZ predicts: f_n_obs = f_n_classical / (1 + delta_SSZ)
    => delta_SSZ should be the SAME for all modes.
    
    We estimate delta_SSZ as weighted average of per-mode deltas.
    
    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily aggregated data with f1_mean, f1_std, etc.
    classical_ref : ClassicalReference
        Classical reference frequencies
    use_weights : bool
        If True, weight by 1/sigma^2
    
    Returns
    -------
    delta_ssz : float
        Global delta_SSZ estimate
    delta_ssz_err : float
        Uncertainty on delta_SSZ
    mode_deltas : dict
        Per-mode delta values
    mode_delta_errors : dict
        Per-mode delta uncertainties
    """
    modes = ['f1', 'f2', 'f3', 'f4']
    classical_freqs = classical_ref.as_dict()
    
    mode_deltas = {}
    mode_delta_errors = {}
    
    for mode in modes:
        f_obs_mean = daily_data[f'{mode}_mean'].mean()
        f_obs_std = daily_data[f'{mode}_mean'].std()
        f_class = classical_freqs[mode]
        
        # delta = -(f_obs - f_class) / f_class
        delta = compute_delta_seg_classical(f_obs_mean, f_class)
        # Error propagation: sigma_delta = sigma_f / f_class
        delta_err = f_obs_std / f_class
        
        mode_deltas[mode] = delta
        mode_delta_errors[mode] = delta_err
    
    # Weighted average
    if use_weights:
        weights = []
        values = []
        for mode in modes:
            if mode_delta_errors[mode] > 0:
                w = 1.0 / mode_delta_errors[mode]**2
            else:
                w = 1.0
            weights.append(w)
            values.append(mode_deltas[mode])
        
        weights = np.array(weights)
        values = np.array(values)
        
        delta_ssz = np.sum(weights * values) / np.sum(weights)
        delta_ssz_err = np.sqrt(1.0 / np.sum(weights))
    else:
        values = np.array([mode_deltas[m] for m in modes])
        delta_ssz = np.mean(values)
        delta_ssz_err = np.std(values) / np.sqrt(len(values))
    
    return delta_ssz, delta_ssz_err, mode_deltas, mode_delta_errors


def compute_ssz_chi_squared(
    mode_deltas: Dict[str, float],
    mode_delta_errors: Dict[str, float],
    delta_ssz_global: float
) -> Tuple[float, int, float, float]:
    """
    Compute chi-squared test for SSZ consistency.
    
    If SSZ is correct, all mode deltas should equal delta_ssz_global.
    
    chi^2 = sum_n [(delta_n - delta_SSZ) / sigma_n]^2
    
    Parameters
    ----------
    mode_deltas : dict
        Per-mode delta values
    mode_delta_errors : dict
        Per-mode delta uncertainties
    delta_ssz_global : float
        Global delta_SSZ estimate
    
    Returns
    -------
    chi2 : float
        Chi-squared statistic
    ndof : int
        Degrees of freedom (n_modes - 1)
    chi2_reduced : float
        Reduced chi-squared (chi2 / ndof)
    p_value : float
        P-value for chi-squared test
    """
    modes = list(mode_deltas.keys())
    n_modes = len(modes)
    
    chi2 = 0.0
    for mode in modes:
        delta = mode_deltas[mode]
        sigma = mode_delta_errors[mode]
        if sigma > 0:
            chi2 += ((delta - delta_ssz_global) / sigma)**2
        else:
            # If no error, use a small default
            chi2 += ((delta - delta_ssz_global) / 0.001)**2
    
    ndof = n_modes - 1  # One parameter fitted (delta_ssz_global)
    chi2_reduced = chi2 / ndof if ndof > 0 else chi2
    
    # P-value: probability of getting chi2 this large or larger
    p_value = 1.0 - stats.chi2.cdf(chi2, ndof)
    
    return chi2, ndof, chi2_reduced, p_value


def compute_ssz_upper_bound(
    mode_deltas: Dict[str, float],
    mode_delta_errors: Dict[str, float],
    confidence_level: float = 0.95
) -> float:
    """
    Compute upper bound on |delta_SSZ| at given confidence level.
    
    Parameters
    ----------
    mode_deltas : dict
        Per-mode delta values
    mode_delta_errors : dict
        Per-mode delta uncertainties
    confidence_level : float
        Confidence level (default 0.95 for 95% CL)
    
    Returns
    -------
    upper_bound : float
        Upper bound on |delta_SSZ|
    """
    values = np.array(list(mode_deltas.values()))
    errors = np.array(list(mode_delta_errors.values()))
    
    # Weighted mean and error
    if np.all(errors > 0):
        weights = 1.0 / errors**2
        mean = np.sum(weights * values) / np.sum(weights)
        std_mean = np.sqrt(1.0 / np.sum(weights))
    else:
        mean = np.mean(values)
        std_mean = np.std(values) / np.sqrt(len(values))
    
    # For 95% CL, use 1.96 sigma (two-sided)
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Upper bound on |delta_SSZ|
    upper_bound = abs(mean) + z * std_mean
    
    return upper_bound


def run_ssz_hypothesis_test(
    daily_data: pd.DataFrame,
    classical_ref: ClassicalReference,
    consistency_threshold: float = 0.02,
    chi2_threshold: float = 3.0
) -> SSZResult:
    """
    Run complete SSZ hypothesis test.
    
    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily aggregated data
    classical_ref : ClassicalReference
        Classical reference frequencies
    consistency_threshold : float
        Maximum spread in mode deltas for SSZ consistency (default 2%)
    chi2_threshold : float
        Maximum reduced chi-squared for SSZ consistency
    
    Returns
    -------
    result : SSZResult
        Complete test results
    """
    # Estimate global delta_SSZ
    delta_ssz, delta_ssz_err, mode_deltas, mode_delta_errors = \
        estimate_delta_ssz_global(daily_data, classical_ref)
    
    # Chi-squared test
    chi2, ndof, chi2_red, p_value = compute_ssz_chi_squared(
        mode_deltas, mode_delta_errors, delta_ssz
    )
    
    # Upper bound
    upper_bound = compute_ssz_upper_bound(mode_deltas, mode_delta_errors)
    
    # Consistency check
    mode_spread = np.std(list(mode_deltas.values()))
    is_consistent = (mode_spread < consistency_threshold) and (chi2_red < chi2_threshold)
    
    return SSZResult(
        delta_ssz_global=delta_ssz,
        delta_ssz_std=delta_ssz_err,
        chi_squared=chi2,
        ndof=ndof,
        chi_squared_reduced=chi2_red,
        p_value=p_value,
        is_consistent=is_consistent,
        upper_bound_95=upper_bound,
        mode_deltas=mode_deltas,
        mode_delta_errors=mode_delta_errors,
    )


def analyze_correlations(
    daily_data: pd.DataFrame,
    delta_column: str = 'delta_ssz_anomaly',
    proxy_columns: List[str] = ['f107', 'kp']
) -> Dict[str, Dict[str, float]]:
    """
    Analyze correlations between delta_seg and space weather proxies.
    
    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily data with delta and proxy columns
    delta_column : str
        Name of delta column to correlate
    proxy_columns : list
        Names of proxy columns (e.g., 'f107', 'kp')
    
    Returns
    -------
    correlations : dict
        Dictionary with correlation results for each proxy
    """
    results = {}
    
    for proxy in proxy_columns:
        if proxy not in daily_data.columns:
            results[proxy] = {'r': np.nan, 'p': np.nan, 'n': 0}
            continue
        
        # Drop NaN values
        mask = daily_data[delta_column].notna() & daily_data[proxy].notna()
        x = daily_data.loc[mask, proxy].values
        y = daily_data.loc[mask, delta_column].values
        
        if len(x) < 3:
            results[proxy] = {'r': np.nan, 'p': np.nan, 'n': len(x)}
            continue
        
        r, p = stats.pearsonr(x, y)
        results[proxy] = {'r': r, 'p': p, 'n': len(x)}
    
    return results


def compute_mode_correlations(
    daily_data: pd.DataFrame,
    modes: List[str] = ['f1', 'f2', 'f3', 'f4']
) -> Tuple[float, np.ndarray]:
    """
    Compute mean correlation of delta_seg_anomaly between modes.
    
    SSZ predicts high correlation (all modes shift together).
    Classical dispersion predicts lower correlation.
    
    Parameters
    ----------
    daily_data : pd.DataFrame
        Daily data with delta_f1_anomaly, delta_f2_anomaly, etc.
    modes : list
        List of mode names
    
    Returns
    -------
    mean_corr : float
        Mean pairwise correlation
    corr_matrix : ndarray
        Full correlation matrix
    """
    delta_cols = [f'delta_{m}_anomaly' for m in modes]
    
    # Check which columns exist
    available_cols = [c for c in delta_cols if c in daily_data.columns]
    if len(available_cols) < 2:
        return np.nan, np.array([])
    
    # Compute correlation matrix
    corr_matrix = daily_data[available_cols].corr().values
    
    # Mean of off-diagonal elements
    n = len(available_cols)
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(corr_matrix[i, j]):
                off_diag.append(corr_matrix[i, j])
    
    mean_corr = np.mean(off_diag) if off_diag else np.nan
    
    return mean_corr, corr_matrix


def compute_ssz_score(
    mean_correlation: float,
    mode_spread: float,
    mean_delta: float
) -> float:
    """
    Compute SSZ consistency score.
    
    SSZ score = mean_correlation * (1 - mode_spread / |mean_delta|)
    
    High score (close to 1): consistent with SSZ
    Low score (close to 0 or negative): classical dispersion dominates
    
    Parameters
    ----------
    mean_correlation : float
        Mean correlation between mode anomalies
    mode_spread : float
        Standard deviation of mode deltas
    mean_delta : float
        Mean delta across modes
    
    Returns
    -------
    score : float
        SSZ consistency score
    """
    if np.isnan(mean_correlation) or abs(mean_delta) < 1e-10:
        return 0.0
    
    # Avoid division by zero
    ratio = mode_spread / abs(mean_delta) if abs(mean_delta) > 1e-10 else 1.0
    score = mean_correlation * max(0, 1 - ratio)
    
    return score
