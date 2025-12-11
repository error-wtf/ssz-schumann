#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eta Calibration Module - Breaking the eta_0 / delta_seg Degeneracy

This module implements different calibration strategies for eta_0:
1. Full fit: Calibrate eta_0 on full dataset (current behavior, has degeneracy)
2. Quiet interval: Calibrate only on quiet period with no SSZ signal
3. Fixed: Use theoretical/literature value
4. Joint fit: Simultaneously fit eta_0 and delta_seg parameters

The key insight is that any constant shift in frequency is absorbed into eta_0,
so the reconstructed delta_seg only captures time-varying components.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize, least_squares
import warnings

from ..config import (
    Config, EtaMode, SSZBasisFunction,
    C_LIGHT, EARTH_RADIUS, ETA_0_DEFAULT
)


@dataclass
class CalibrationResult:
    """Result of eta_0 calibration."""
    eta_0: float
    method: str
    residuals_mean: float
    residuals_std: float
    n_points: int
    metadata: Dict


@dataclass
class JointFitResult:
    """Result of joint eta_0 + delta_seg fit."""
    eta_0: float
    amplitude_A: float
    amplitude_B: float
    residuals: np.ndarray
    rmse: float
    r_squared: float
    converged: bool
    metadata: Dict


def f_n_classical(n: int, eta_0: float, R: float = EARTH_RADIUS, c: float = C_LIGHT) -> float:
    """
    Classical Schumann frequency for mode n.
    
    Formula:
        f_n = eta_0 * c / (2*pi*R) * sqrt(n*(n+1))
    
    Args:
        n: Mode number (1, 2, 3, ...)
        eta_0: Effective slowdown factor
        R: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        Frequency in Hz
    """
    return eta_0 * c / (2.0 * np.pi * R) * np.sqrt(n * (n + 1))


def f_n_ssz(n: int, eta_0: float, delta_seg: float, R: float = EARTH_RADIUS, c: float = C_LIGHT) -> float:
    """
    SSZ-corrected Schumann frequency for mode n.
    
    Formula:
        f_n^SSZ = f_n^classical / D_SSZ
        D_SSZ = 1 + delta_seg
    
    Args:
        n: Mode number
        eta_0: Effective slowdown factor
        delta_seg: Segmentation parameter
        R: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        Frequency in Hz
    """
    f_class = f_n_classical(n, eta_0, R, c)
    D_SSZ = 1.0 + delta_seg
    return f_class / D_SSZ


# =============================================================================
# CALIBRATION METHODS
# =============================================================================

def calibrate_eta_full_fit(
    f_obs: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> CalibrationResult:
    """
    Calibrate eta_0 using full dataset (current method).
    
    This method has a degeneracy: any constant SSZ shift is absorbed into eta_0.
    
    Args:
        f_obs: Dictionary of observed frequencies {mode: array}
        modes: List of modes to use
    
    Returns:
        CalibrationResult
    """
    # Use mode 1 mean to calibrate
    f1_mean = np.mean(f_obs[1])
    
    # eta_0 = f1_obs / f1_ideal_factor
    # f1_ideal_factor = c / (2*pi*R) * sqrt(2)
    f1_ideal_factor = C_LIGHT / (2.0 * np.pi * EARTH_RADIUS) * np.sqrt(2.0)
    eta_0 = f1_mean / f1_ideal_factor
    
    # Compute residuals
    residuals = []
    for n in modes:
        f_class = f_n_classical(n, eta_0)
        residuals.extend(f_obs[n] - f_class)
    
    residuals = np.array(residuals)
    
    return CalibrationResult(
        eta_0=eta_0,
        method="full_fit",
        residuals_mean=np.mean(residuals),
        residuals_std=np.std(residuals),
        n_points=len(residuals),
        metadata={"f1_mean": f1_mean}
    )


def calibrate_eta_quiet_interval(
    f_obs: Dict[int, np.ndarray],
    time_index: pd.DatetimeIndex,
    quiet_days: int = 14,
    modes: List[int] = [1, 2, 3],
) -> CalibrationResult:
    """
    Calibrate eta_0 using only a quiet interval (no SSZ signal).
    
    This breaks the degeneracy by using a period where delta_seg ≈ 0.
    
    Args:
        f_obs: Dictionary of observed frequencies {mode: array}
        time_index: Time index for the data
        quiet_days: Number of days from start to use as quiet interval
        modes: List of modes to use
    
    Returns:
        CalibrationResult
    """
    # Find quiet interval indices
    start_time = time_index[0]
    quiet_end = start_time + pd.Timedelta(days=quiet_days)
    quiet_mask = time_index < quiet_end
    n_quiet = np.sum(quiet_mask)
    
    if n_quiet < 10:
        warnings.warn(f"Quiet interval has only {n_quiet} points, using full fit")
        return calibrate_eta_full_fit(f_obs, modes)
    
    # Use mode 1 mean from quiet interval
    f1_quiet = f_obs[1][quiet_mask]
    f1_mean = np.mean(f1_quiet)
    
    # Calibrate eta_0
    f1_ideal_factor = C_LIGHT / (2.0 * np.pi * EARTH_RADIUS) * np.sqrt(2.0)
    eta_0 = f1_mean / f1_ideal_factor
    
    # Compute residuals on full dataset
    residuals = []
    for n in modes:
        f_class = f_n_classical(n, eta_0)
        residuals.extend(f_obs[n] - f_class)
    
    residuals = np.array(residuals)
    
    return CalibrationResult(
        eta_0=eta_0,
        method="quiet_interval",
        residuals_mean=np.mean(residuals),
        residuals_std=np.std(residuals),
        n_points=len(residuals),
        metadata={
            "f1_mean_quiet": f1_mean,
            "quiet_days": quiet_days,
            "n_quiet_points": n_quiet,
        }
    )


def calibrate_eta_fixed(
    f_obs: Dict[int, np.ndarray],
    eta_0_fixed: float = 0.74,
    modes: List[int] = [1, 2, 3],
) -> CalibrationResult:
    """
    Use a fixed eta_0 value from theory/literature.
    
    Args:
        f_obs: Dictionary of observed frequencies {mode: array}
        eta_0_fixed: Fixed eta_0 value
        modes: List of modes to use
    
    Returns:
        CalibrationResult
    """
    # Compute residuals
    residuals = []
    for n in modes:
        f_class = f_n_classical(n, eta_0_fixed)
        residuals.extend(f_obs[n] - f_class)
    
    residuals = np.array(residuals)
    
    return CalibrationResult(
        eta_0=eta_0_fixed,
        method="fixed",
        residuals_mean=np.mean(residuals),
        residuals_std=np.std(residuals),
        n_points=len(residuals),
        metadata={"eta_0_fixed": eta_0_fixed}
    )


# =============================================================================
# JOINT FIT MODEL
# =============================================================================

def create_basis_functions(
    time_index: pd.DatetimeIndex,
    f107: Optional[np.ndarray] = None,
    basis_type: SSZBasisFunction = SSZBasisFunction.SINUSOIDAL,
    period_days: float = 365.25,
    phase_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create basis functions for delta_seg(t) parameterization.
    
    Args:
        time_index: Time index
        f107: F10.7 solar flux (optional)
        basis_type: Type of basis function
        period_days: Period for sinusoidal basis
        phase_offset: Phase offset (radians)
    
    Returns:
        Tuple of (basis_1, basis_2) arrays
    """
    n = len(time_index)
    
    # Time in days from start
    t_days = (time_index - time_index[0]).total_seconds() / 86400.0
    
    if basis_type == SSZBasisFunction.CONSTANT:
        basis_1 = np.ones(n)
        basis_2 = np.zeros(n)
    
    elif basis_type == SSZBasisFunction.SINUSOIDAL:
        # Sinusoidal with given period
        basis_1 = np.sin(2.0 * np.pi * t_days / period_days + phase_offset)
        basis_2 = np.zeros(n)
    
    elif basis_type == SSZBasisFunction.F107_LINEAR:
        basis_1 = np.ones(n)
        if f107 is not None:
            # Normalize F10.7 around 100 SFU
            basis_2 = (f107 - 100.0) / 100.0
        else:
            basis_2 = np.zeros(n)
    
    elif basis_type == SSZBasisFunction.COMBINED:
        # Sinusoidal + F10.7
        basis_1 = np.sin(2.0 * np.pi * t_days / period_days + phase_offset)
        if f107 is not None:
            basis_2 = (f107 - 100.0) / 100.0
        else:
            basis_2 = np.zeros(n)
    
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
    
    return basis_1, basis_2


def joint_fit_eta_and_segmentation(
    f_obs: Dict[int, np.ndarray],
    time_index: pd.DatetimeIndex,
    f107: Optional[np.ndarray] = None,
    basis_type: SSZBasisFunction = SSZBasisFunction.SINUSOIDAL,
    period_days: float = 365.25,
    phase_offset: float = 0.0,
    modes: List[int] = [1, 2, 3],
    eta_0_init: float = ETA_0_DEFAULT,
    bounds: Optional[Dict] = None,
) -> JointFitResult:
    """
    Jointly fit eta_0 and delta_seg parameters.
    
    Model:
        f_n^obs(t) = f_n^classical(eta_0) / (1 + A*basis_1(t) + B*basis_2(t))
    
    Fit parameters: (eta_0, A, B)
    
    Args:
        f_obs: Dictionary of observed frequencies {mode: array}
        time_index: Time index
        f107: F10.7 solar flux (optional)
        basis_type: Type of basis function for delta_seg
        period_days: Period for sinusoidal basis
        phase_offset: Phase offset
        modes: List of modes to fit
        eta_0_init: Initial guess for eta_0
        bounds: Parameter bounds (optional)
    
    Returns:
        JointFitResult
    """
    n_times = len(time_index)
    
    # Create basis functions
    basis_1, basis_2 = create_basis_functions(
        time_index, f107, basis_type, period_days, phase_offset
    )
    
    # Stack observations
    f_obs_stack = np.concatenate([f_obs[n] for n in modes])
    mode_indices = np.concatenate([np.full(n_times, n) for n in modes])
    basis_1_stack = np.tile(basis_1, len(modes))
    basis_2_stack = np.tile(basis_2, len(modes))
    
    def model_func(params):
        """Model function: f_n^SSZ(t)."""
        eta_0, A, B = params
        delta_seg = A * basis_1_stack + B * basis_2_stack
        D_SSZ = 1.0 + delta_seg
        
        f_model = np.zeros_like(f_obs_stack)
        for i, n in enumerate(mode_indices):
            f_class = f_n_classical(int(n), eta_0)
            f_model[i] = f_class / D_SSZ[i]
        
        return f_model
    
    def residual_func(params):
        """Residual function for optimization."""
        f_model = model_func(params)
        return f_obs_stack - f_model
    
    def cost_func(params):
        """Cost function (sum of squared residuals)."""
        residuals = residual_func(params)
        return np.sum(residuals**2)
    
    # Initial guess
    x0 = [eta_0_init, 0.02, 0.0]
    
    # Bounds
    if bounds is None:
        bounds = {
            'eta_0': (0.6, 0.9),
            'A': (-0.2, 0.2),
            'B': (-0.1, 0.1),
        }
    
    lb = [bounds['eta_0'][0], bounds['A'][0], bounds['B'][0]]
    ub = [bounds['eta_0'][1], bounds['A'][1], bounds['B'][1]]
    
    # Optimize
    try:
        result = least_squares(
            residual_func,
            x0,
            bounds=(lb, ub),
            method='trf',
            verbose=0,
        )
        converged = result.success
        params_opt = result.x
    except Exception as e:
        warnings.warn(f"Joint fit failed: {e}")
        converged = False
        params_opt = x0
    
    eta_0_opt, A_opt, B_opt = params_opt
    
    # Compute final residuals and metrics
    residuals = residual_func(params_opt)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((f_obs_stack - np.mean(f_obs_stack))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    return JointFitResult(
        eta_0=eta_0_opt,
        amplitude_A=A_opt,
        amplitude_B=B_opt,
        residuals=residuals,
        rmse=rmse,
        r_squared=r_squared,
        converged=converged,
        metadata={
            "basis_type": basis_type.value,
            "period_days": period_days,
            "n_modes": len(modes),
            "n_times": n_times,
        }
    )


# =============================================================================
# RECONSTRUCTION COMPARISON
# =============================================================================

def reconstruct_delta_seg_residual(
    f_obs: Dict[int, np.ndarray],
    eta_0: float,
    modes: List[int] = [1, 2, 3],
) -> Dict[int, np.ndarray]:
    """
    Reconstruct delta_seg from residuals (current method).
    
    delta_seg_n = f_classical / f_obs - 1
    
    Args:
        f_obs: Observed frequencies
        eta_0: Calibrated eta_0
        modes: Modes to use
    
    Returns:
        Dictionary of delta_seg arrays per mode
    """
    delta_seg = {}
    for n in modes:
        f_class = f_n_classical(n, eta_0)
        delta_seg[n] = f_class / f_obs[n] - 1.0
    
    return delta_seg


def compare_reconstruction_methods(
    f_obs: Dict[int, np.ndarray],
    time_index: pd.DatetimeIndex,
    delta_seg_true: np.ndarray,
    f107: Optional[np.ndarray] = None,
    modes: List[int] = [1, 2, 3],
    quiet_days: int = 14,
) -> pd.DataFrame:
    """
    Compare different delta_seg reconstruction methods.
    
    Args:
        f_obs: Observed frequencies
        time_index: Time index
        delta_seg_true: True delta_seg (for synthetic data)
        f107: F10.7 data (optional)
        modes: Modes to analyze
        quiet_days: Days for quiet interval
    
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    # Method 1: Full fit (current)
    cal_full = calibrate_eta_full_fit(f_obs, modes)
    delta_seg_full = reconstruct_delta_seg_residual(f_obs, cal_full.eta_0, modes)
    delta_seg_mean_full = np.mean([delta_seg_full[n] for n in modes], axis=0)
    
    bias_full = np.mean(delta_seg_mean_full - delta_seg_true)
    rmse_full = np.sqrt(np.mean((delta_seg_mean_full - delta_seg_true)**2))
    corr_full = np.corrcoef(delta_seg_mean_full, delta_seg_true)[0, 1]
    
    results.append({
        'method': 'full_fit',
        'eta_0': cal_full.eta_0,
        'bias': bias_full,
        'rmse': rmse_full,
        'correlation': corr_full,
        'mean_delta_seg': np.mean(delta_seg_mean_full),
        'std_delta_seg': np.std(delta_seg_mean_full),
    })
    
    # Method 2: Quiet interval
    cal_quiet = calibrate_eta_quiet_interval(f_obs, time_index, quiet_days, modes)
    delta_seg_quiet = reconstruct_delta_seg_residual(f_obs, cal_quiet.eta_0, modes)
    delta_seg_mean_quiet = np.mean([delta_seg_quiet[n] for n in modes], axis=0)
    
    bias_quiet = np.mean(delta_seg_mean_quiet - delta_seg_true)
    rmse_quiet = np.sqrt(np.mean((delta_seg_mean_quiet - delta_seg_true)**2))
    corr_quiet = np.corrcoef(delta_seg_mean_quiet, delta_seg_true)[0, 1]
    
    results.append({
        'method': 'quiet_interval',
        'eta_0': cal_quiet.eta_0,
        'bias': bias_quiet,
        'rmse': rmse_quiet,
        'correlation': corr_quiet,
        'mean_delta_seg': np.mean(delta_seg_mean_quiet),
        'std_delta_seg': np.std(delta_seg_mean_quiet),
    })
    
    # Method 3: Joint fit
    joint_result = joint_fit_eta_and_segmentation(
        f_obs, time_index, f107,
        basis_type=SSZBasisFunction.SINUSOIDAL,
        modes=modes,
    )
    
    # Reconstruct delta_seg from joint fit
    basis_1, basis_2 = create_basis_functions(
        time_index, f107, SSZBasisFunction.SINUSOIDAL
    )
    delta_seg_joint = joint_result.amplitude_A * basis_1 + joint_result.amplitude_B * basis_2
    
    bias_joint = np.mean(delta_seg_joint - delta_seg_true)
    rmse_joint = np.sqrt(np.mean((delta_seg_joint - delta_seg_true)**2))
    corr_joint = np.corrcoef(delta_seg_joint, delta_seg_true)[0, 1]
    
    results.append({
        'method': 'joint_fit',
        'eta_0': joint_result.eta_0,
        'bias': bias_joint,
        'rmse': rmse_joint,
        'correlation': corr_joint,
        'mean_delta_seg': np.mean(delta_seg_joint),
        'std_delta_seg': np.std(delta_seg_joint),
        'amplitude_A': joint_result.amplitude_A,
        'amplitude_B': joint_result.amplitude_B,
    })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN CALIBRATION FUNCTION
# =============================================================================

def calibrate_eta(
    f_obs: Dict[int, np.ndarray],
    config: Config,
    time_index: Optional[pd.DatetimeIndex] = None,
) -> CalibrationResult:
    """
    Calibrate eta_0 using the method specified in config.
    
    Args:
        f_obs: Dictionary of observed frequencies {mode: array}
        config: Configuration object
        time_index: Time index (required for quiet_interval mode)
    
    Returns:
        CalibrationResult
    """
    eta_mode = config.classical.eta_mode
    modes = config.modes
    
    if eta_mode == EtaMode.FULL_FIT:
        return calibrate_eta_full_fit(f_obs, modes)
    
    elif eta_mode == EtaMode.QUIET_INTERVAL:
        if time_index is None:
            raise ValueError("time_index required for quiet_interval mode")
        return calibrate_eta_quiet_interval(
            f_obs, time_index,
            quiet_days=config.classical.quiet_interval_days,
            modes=modes
        )
    
    elif eta_mode == EtaMode.FIXED:
        return calibrate_eta_fixed(
            f_obs,
            eta_0_fixed=config.classical.eta_0_fixed,
            modes=modes
        )
    
    else:
        raise ValueError(f"Unknown eta_mode: {eta_mode}")
