#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Model Fitting Functions

Functions for fitting SSZ models to Schumann resonance data,
including calibration, mode consistency analysis, and parametric fits.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from sklearn.linear_model import LinearRegression
from scipy import stats

logger = logging.getLogger(__name__)

# Physical constants
C_LIGHT = 299792458.0  # m/s
EARTH_RADIUS = 6.371e6  # m


@dataclass
class FitResult:
    """Container for model fit results."""
    model_name: str
    parameters: Dict[str, float]
    r_squared: float
    rmse: float
    n_points: int
    p_values: Optional[Dict[str, float]] = None
    residuals: Optional[np.ndarray] = None


def calibrate_eta_from_data(
    df: pd.DataFrame,
    f1_column: str = 'f1_Hz',
    method: str = 'mean',
) -> float:
    """
    Calibrate the classical slowdown factor eta from observed f1 data.
    
    The classical Schumann formula is:
        f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    For n=1:
        f_1 = eta * c / (2*pi*R) * sqrt(2)
        
    Solving for eta:
        eta = f_1 * 2*pi*R / (c * sqrt(2))
    
    Args:
        df: DataFrame with Schumann data
        f1_column: Column name for f1 frequency
        method: 'mean' (use mean f1) or 'median' (use median f1)
    
    Returns:
        Calibrated eta value (typically ~0.74)
    """
    if f1_column not in df.columns:
        raise ValueError(f"Column {f1_column} not found in DataFrame")
    
    f1_data = df[f1_column].dropna()
    
    if len(f1_data) == 0:
        raise ValueError("No valid f1 data for calibration")
    
    if method == 'mean':
        f1_ref = f1_data.mean()
    elif method == 'median':
        f1_ref = f1_data.median()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # eta = f_1 * 2*pi*R / (c * sqrt(2))
    eta = f1_ref * 2 * np.pi * EARTH_RADIUS / (C_LIGHT * np.sqrt(2))
    
    logger.info(f"Calibrated eta = {eta:.6f} from f1 = {f1_ref:.4f} Hz ({method})")
    
    return eta


def compute_classical_frequencies(
    eta: float,
    modes: list = [1, 2, 3],
) -> Dict[int, float]:
    """
    Compute classical Schumann frequencies for given eta.
    
    Formula:
        f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    Args:
        eta: Slowdown factor
        modes: List of mode numbers
    
    Returns:
        Dictionary mapping mode number to frequency (Hz)
    """
    f_class = {}
    for n in modes:
        f_class[n] = eta * C_LIGHT / (2 * np.pi * EARTH_RADIUS) * np.sqrt(n * (n + 1))
    return f_class


def compute_delta_seg(
    df: pd.DataFrame,
    eta: float,
    modes: list = [1, 2, 3],
) -> pd.DataFrame:
    """
    Compute SSZ segmentation parameter delta_seg for each mode.
    
    For each timestamp and mode n:
        delta_f_rel_n(t) = (f_obs_n(t) - f_class_n) / f_class_n
        delta_seg_hat_n(t) = -delta_f_rel_n(t)
    
    Args:
        df: DataFrame with observed frequencies (f1_Hz, f2_Hz, f3_Hz)
        eta: Calibrated slowdown factor
        modes: List of mode numbers
    
    Returns:
        DataFrame with delta_seg columns for each mode
    """
    f_class = compute_classical_frequencies(eta, modes)
    
    result = pd.DataFrame(index=df.index)
    
    for n in modes:
        f_obs_col = f'f{n}_Hz'
        if f_obs_col not in df.columns:
            logger.warning(f"Column {f_obs_col} not found, skipping mode {n}")
            continue
        
        f_obs = df[f_obs_col]
        f_c = f_class[n]
        
        # Relative deviation
        delta_f_rel = (f_obs - f_c) / f_c
        
        # SSZ segmentation parameter (negative of relative deviation)
        delta_seg = -delta_f_rel
        
        result[f'delta_seg{n}'] = delta_seg
        result[f'delta_f_rel{n}'] = delta_f_rel
    
    # Compute mean delta_seg across modes
    delta_seg_cols = [f'delta_seg{n}' for n in modes if f'delta_seg{n}' in result.columns]
    if delta_seg_cols:
        result['delta_seg_mean'] = result[delta_seg_cols].mean(axis=1)
        result['delta_seg_std'] = result[delta_seg_cols].std(axis=1)
    
    return result


def compute_mode_consistency(
    delta_seg_df: pd.DataFrame,
    modes: list = [1, 2, 3],
) -> Dict:
    """
    Compute mode consistency metrics for SSZ signature detection.
    
    The key SSZ prediction is that delta_seg should be mode-independent,
    i.e., the same relative shift for all modes.
    
    Args:
        delta_seg_df: DataFrame with columns delta_seg1, delta_seg2, delta_seg3
        modes: List of mode numbers
    
    Returns:
        Dictionary with consistency metrics:
            - corr_12, corr_13, corr_23: Pairwise correlations
            - mean_correlation: Average of pairwise correlations
            - mean_abs_spread: Mean of (max - min) across modes per timestamp
            - ssz_score: 1 / (1 + mean_abs_spread * 100)
            - is_consistent: Boolean indicating strong SSZ signature
    """
    # Get delta_seg columns
    cols = [f'delta_seg{n}' for n in modes if f'delta_seg{n}' in delta_seg_df.columns]
    
    if len(cols) < 2:
        logger.warning("Need at least 2 modes for consistency check")
        return {
            'mean_correlation': np.nan,
            'ssz_score': 0.0,
            'is_consistent': False,
        }
    
    # Extract data
    data = delta_seg_df[cols].dropna()
    
    if len(data) < 10:
        logger.warning("Insufficient data for consistency check")
        return {
            'mean_correlation': np.nan,
            'ssz_score': 0.0,
            'is_consistent': False,
        }
    
    result = {}
    
    # Pairwise correlations
    correlations = []
    for i, col1 in enumerate(cols):
        for col2 in cols[i+1:]:
            corr = data[col1].corr(data[col2])
            mode1 = col1.replace('delta_seg', '')
            mode2 = col2.replace('delta_seg', '')
            result[f'corr_{mode1}{mode2}'] = corr
            if not np.isnan(corr):
                correlations.append(corr)
    
    result['mean_correlation'] = np.mean(correlations) if correlations else np.nan
    
    # Spread across modes (should be small for SSZ)
    spread = data[cols].max(axis=1) - data[cols].min(axis=1)
    result['mean_abs_spread'] = spread.abs().mean()
    result['std_abs_spread'] = spread.abs().std()
    
    # SSZ score: higher is better (more consistent)
    # Score = 1 / (1 + spread * 100) gives ~1 for spread=0, ~0.5 for spread=1%
    result['ssz_score'] = 1.0 / (1.0 + result['mean_abs_spread'] * 100)
    
    # Standard deviation across modes (should be small for SSZ)
    result['std_across_modes'] = data[cols].std(axis=1).mean()
    
    # Is consistent? (high correlation and low spread)
    result['is_consistent'] = (
        result['mean_correlation'] > 0.7 and 
        result['mean_abs_spread'] < 0.01  # Less than 1% spread
    )
    
    logger.info(f"Mode consistency: corr={result['mean_correlation']:.4f}, "
                f"spread={result['mean_abs_spread']:.6f}, "
                f"ssz_score={result['ssz_score']:.4f}")
    
    return result


def fit_global_ssz_model(
    df: pd.DataFrame,
    delta_seg_col: str = 'delta_seg_mean',
    proxy_cols: list = ['F107_norm', 'Kp_norm'],
) -> FitResult:
    """
    Fit a simple global SSZ model.
    
    Model:
        delta_seg_eff(t) = beta_0 + beta_1 * F107_norm + beta_2 * Kp_norm
    
    Args:
        df: DataFrame with delta_seg and proxy columns
        delta_seg_col: Column name for target variable
        proxy_cols: List of proxy column names
    
    Returns:
        FitResult with parameters, R², and statistics
    """
    # Check columns exist
    if delta_seg_col not in df.columns:
        raise ValueError(f"Column {delta_seg_col} not found")
    
    available_proxies = [col for col in proxy_cols if col in df.columns]
    
    if not available_proxies:
        logger.warning("No proxy columns found, fitting constant model")
        available_proxies = []
    
    # Prepare data
    data = df[[delta_seg_col] + available_proxies].dropna()
    
    if len(data) < 10:
        raise ValueError("Insufficient data for fitting")
    
    y = data[delta_seg_col].values
    
    if available_proxies:
        X = data[available_proxies].values
    else:
        X = np.ones((len(y), 1))
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Statistics
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    # Parameters
    parameters = {'beta_0': model.intercept_}
    for i, col in enumerate(available_proxies):
        parameters[f'beta_{i+1}'] = model.coef_[i]
        parameters[f'{col}_coef'] = model.coef_[i]
    
    # P-values (using scipy for t-test)
    p_values = {}
    n = len(y)
    p = len(available_proxies) + 1  # Including intercept
    
    if n > p:
        mse = ss_res / (n - p)
        
        # Standard errors (simplified)
        if available_proxies:
            X_with_intercept = np.column_stack([np.ones(n), X])
            try:
                var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
                se_beta = np.sqrt(var_beta)
                
                # t-statistics
                t_stats = np.concatenate([[model.intercept_], model.coef_]) / se_beta
                
                # p-values (two-tailed)
                p_values['beta_0'] = 2 * (1 - stats.t.cdf(abs(t_stats[0]), n - p))
                for i, col in enumerate(available_proxies):
                    p_values[f'beta_{i+1}'] = 2 * (1 - stats.t.cdf(abs(t_stats[i+1]), n - p))
            except np.linalg.LinAlgError:
                pass
    
    result = FitResult(
        model_name='global_ssz',
        parameters=parameters,
        r_squared=r_squared,
        rmse=rmse,
        n_points=len(data),
        p_values=p_values if p_values else None,
        residuals=residuals,
    )
    
    logger.info(f"Global SSZ fit: R²={r_squared:.4f}, RMSE={rmse:.6f}")
    
    return result


def fit_layered_ssz_model(
    df: pd.DataFrame,
    delta_seg_col: str = 'delta_seg_mean',
    proxy_cols: list = ['F107_norm', 'Kp_norm'],
    w_atm: float = 0.2,
    w_iono: float = 0.8,
) -> FitResult:
    """
    Fit the layered SSZ model.
    
    Model:
        D_SSZ = 1 + w_atm * sigma_atm + w_iono * sigma_iono
        sigma_iono(t) = beta_0 + beta_1 * F107_norm + beta_2 * Kp_norm
        sigma_atm = gamma_0 (constant, default 0)
    
    The effective delta_seg is:
        delta_seg_eff = w_atm * sigma_atm + w_iono * sigma_iono
    
    Args:
        df: DataFrame with delta_seg and proxy columns
        delta_seg_col: Column name for target variable
        proxy_cols: List of proxy column names
        w_atm: Atmosphere layer weight
        w_iono: Ionosphere layer weight
    
    Returns:
        FitResult with parameters including layer-specific values
    """
    # First fit the global model
    global_result = fit_global_ssz_model(df, delta_seg_col, proxy_cols)
    
    # Extract sigma_iono parameters from effective delta_seg
    # delta_seg_eff = w_iono * sigma_iono (assuming sigma_atm = 0)
    # sigma_iono = delta_seg_eff / w_iono
    
    parameters = global_result.parameters.copy()
    
    # Convert to sigma_iono parameters
    parameters['w_atm'] = w_atm
    parameters['w_iono'] = w_iono
    parameters['sigma_atm'] = 0.0  # Constant, can be extended later
    
    # sigma_iono = beta_0/w_iono + beta_1/w_iono * F107_norm + ...
    parameters['sigma_iono_beta_0'] = parameters['beta_0'] / w_iono
    
    for key in list(parameters.keys()):
        if key.startswith('beta_') and key != 'beta_0':
            idx = key.replace('beta_', '')
            parameters[f'sigma_iono_beta_{idx}'] = parameters[key] / w_iono
    
    result = FitResult(
        model_name='layered_ssz',
        parameters=parameters,
        r_squared=global_result.r_squared,
        rmse=global_result.rmse,
        n_points=global_result.n_points,
        p_values=global_result.p_values,
        residuals=global_result.residuals,
    )
    
    logger.info(f"Layered SSZ fit: sigma_iono_beta_0={parameters['sigma_iono_beta_0']:.6f}")
    
    return result


def compute_proxy_correlations(
    df: pd.DataFrame,
    delta_seg_col: str = 'delta_seg_mean',
    proxy_cols: list = ['F107_norm', 'Kp_norm', 'Ap_norm'],
) -> Dict[str, float]:
    """
    Compute correlations between delta_seg and various proxies.
    
    Args:
        df: DataFrame with delta_seg and proxy columns
        delta_seg_col: Column name for delta_seg
        proxy_cols: List of proxy column names
    
    Returns:
        Dictionary mapping proxy name to correlation coefficient
    """
    correlations = {}
    
    if delta_seg_col not in df.columns:
        return correlations
    
    for col in proxy_cols:
        if col in df.columns:
            corr = df[delta_seg_col].corr(df[col])
            correlations[col] = corr
            logger.info(f"Correlation {delta_seg_col} vs {col}: {corr:.4f}")
    
    return correlations


def generate_interpretation(
    mode_consistency: Dict,
    global_fit: FitResult,
    layered_fit: FitResult,
    correlations: Dict[str, float],
    delta_seg_stats: Dict,
) -> str:
    """
    Generate a text interpretation of the SSZ analysis results.
    
    Args:
        mode_consistency: Results from compute_mode_consistency()
        global_fit: Results from fit_global_ssz_model()
        layered_fit: Results from fit_layered_ssz_model()
        correlations: Proxy correlations
        delta_seg_stats: Statistics of delta_seg (mean, std)
    
    Returns:
        Multi-line interpretation text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SSZ ANALYSIS INTERPRETATION")
    lines.append("=" * 60)
    
    # Mode consistency
    lines.append("\n1. MODE CONSISTENCY (SSZ Signature Test)")
    lines.append("-" * 40)
    
    mean_corr = mode_consistency.get('mean_correlation', np.nan)
    ssz_score = mode_consistency.get('ssz_score', 0)
    spread = mode_consistency.get('mean_abs_spread', np.nan)
    
    if not np.isnan(mean_corr):
        if mean_corr > 0.8:
            lines.append(f"   Strong mode correlation: {mean_corr:.3f}")
            lines.append("   -> Consistent with SSZ prediction (mode-independent shifts)")
        elif mean_corr > 0.5:
            lines.append(f"   Moderate mode correlation: {mean_corr:.3f}")
            lines.append("   -> Partial SSZ signature detected")
        else:
            lines.append(f"   Weak mode correlation: {mean_corr:.3f}")
            lines.append("   -> No clear SSZ signature")
    
    lines.append(f"   SSZ Score: {ssz_score:.4f}")
    lines.append(f"   Mean spread across modes: {spread*100:.3f}%")
    
    # Delta_seg magnitude
    lines.append("\n2. SEGMENTATION MAGNITUDE")
    lines.append("-" * 40)
    
    mean_seg = delta_seg_stats.get('mean', 0)
    std_seg = delta_seg_stats.get('std', 0)
    
    lines.append(f"   Mean delta_seg: {mean_seg*100:.4f}%")
    lines.append(f"   Std delta_seg:  {std_seg*100:.4f}%")
    
    if abs(mean_seg) < 0.001:
        lines.append("   -> Segmentation effect near zero (< 0.1%)")
    elif abs(mean_seg) < 0.01:
        lines.append("   -> Small but measurable segmentation (0.1-1%)")
    else:
        lines.append("   -> Significant segmentation effect (> 1%)")
    
    # Proxy correlations
    lines.append("\n3. IONOSPHERIC PROXY CORRELATIONS")
    lines.append("-" * 40)
    
    for proxy, corr in correlations.items():
        if not np.isnan(corr):
            strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
            lines.append(f"   {proxy}: r = {corr:.4f} ({strength})")
    
    # Model fit quality
    lines.append("\n4. MODEL FIT QUALITY")
    lines.append("-" * 40)
    
    lines.append(f"   Global SSZ model R²: {global_fit.r_squared:.4f}")
    lines.append(f"   Layered SSZ model R²: {layered_fit.r_squared:.4f}")
    
    if global_fit.r_squared > 0.3:
        lines.append("   -> Proxies explain significant variance")
    elif global_fit.r_squared > 0.1:
        lines.append("   -> Proxies explain some variance")
    else:
        lines.append("   -> Proxies explain little variance")
    
    # Overall conclusion
    lines.append("\n5. CONCLUSION")
    lines.append("-" * 40)
    
    if mode_consistency.get('is_consistent', False) and global_fit.r_squared > 0.1:
        lines.append("   SSZ signature DETECTED with ionospheric coupling.")
        lines.append("   The mode-independent relative shifts are consistent")
        lines.append("   with SSZ theory predictions.")
    elif mode_consistency.get('is_consistent', False):
        lines.append("   SSZ signature DETECTED but weak proxy coupling.")
        lines.append("   Mode consistency suggests SSZ effect, but ionospheric")
        lines.append("   proxies do not fully explain the variation.")
    elif global_fit.r_squared > 0.1:
        lines.append("   Ionospheric coupling detected but NO clear SSZ signature.")
        lines.append("   Frequency variations correlate with proxies but are")
        lines.append("   not mode-independent as SSZ predicts.")
    else:
        lines.append("   Results CONSISTENT WITH ZERO SSZ within current sensitivity.")
        lines.append("   No significant mode-independent shifts or proxy correlations")
        lines.append("   detected at the current measurement precision.")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'f1_Hz': 7.83 + 0.05 * np.random.randn(n),
        'f2_Hz': 14.3 + 0.08 * np.random.randn(n),
        'f3_Hz': 20.8 + 0.12 * np.random.randn(n),
        'F107_norm': np.random.randn(n),
        'Kp_norm': np.random.randn(n),
    })
    
    # Test calibration
    eta = calibrate_eta_from_data(df)
    print(f"\nCalibrated eta: {eta:.6f}")
    
    # Test delta_seg computation
    delta_df = compute_delta_seg(df, eta)
    print(f"\nDelta_seg columns: {delta_df.columns.tolist()}")
    
    # Test mode consistency
    consistency = compute_mode_consistency(delta_df)
    print(f"\nMode consistency: {consistency}")
