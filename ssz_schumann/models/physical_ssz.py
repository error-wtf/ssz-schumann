#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical SSZ Model for Schumann Resonances

This module implements a physically-motivated SSZ model that connects
the segmentation parameter to ionospheric properties.

Theory:
    The SSZ hypothesis suggests that spacetime segmentation modifies
    the effective speed of light:
    
        c_eff = c / (1 + delta_seg)
    
    For Schumann resonances, this leads to:
    
        f_n_SSZ = f_n_classical / (1 + delta_seg)
    
    The key prediction is that delta_seg should be MODE-INDEPENDENT,
    meaning all modes should show the same relative frequency shift.

Physical Connection:
    We hypothesize that delta_seg is related to ionospheric properties:
    
        delta_seg = alpha * (n_e / n_e_ref) * (1 + beta * B / B_ref)
    
    where:
        - n_e: electron density in ionosphere
        - B: magnetic field strength
        - alpha, beta: coupling constants

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from ..config import PHI, C_LIGHT, EARTH_RADIUS, ETA_0_DEFAULT

logger = logging.getLogger(__name__)

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # H/m (vacuum permeability)
EPSILON_0 = 8.854187817e-12  # F/m (vacuum permittivity)
E_CHARGE = 1.602176634e-19  # C (electron charge)
M_ELECTRON = 9.1093837015e-31  # kg (electron mass)

# Reference values for ionosphere
N_E_REF = 1e11  # m^-3 (typical D-layer electron density)
B_REF = 5e-5  # T (typical Earth magnetic field)
H_IONO_REF = 85.0  # km (typical ionosphere height)


@dataclass
class IonosphereState:
    """State of the ionosphere at a given time."""
    n_e: float  # Electron density (m^-3)
    T_e: float  # Electron temperature (K)
    h_iono: float  # Ionosphere height (km)
    B_field: float  # Magnetic field (T)
    collision_freq: float  # Collision frequency (Hz)


@dataclass
class SSZPhysicalParams:
    """Physical parameters for SSZ model."""
    alpha: float = 1e-12  # Coupling constant for n_e
    beta: float = 0.1  # Coupling constant for B field
    gamma: float = 0.01  # Coupling constant for height
    n_e_ref: float = N_E_REF
    B_ref: float = B_REF
    h_ref: float = H_IONO_REF


def plasma_frequency(n_e: float) -> float:
    """
    Calculate plasma frequency.
    
    f_p = sqrt(n_e * e^2 / (epsilon_0 * m_e)) / (2*pi)
    
    Args:
        n_e: Electron density (m^-3)
    
    Returns:
        Plasma frequency (Hz)
    """
    omega_p = np.sqrt(n_e * E_CHARGE**2 / (EPSILON_0 * M_ELECTRON))
    return omega_p / (2 * np.pi)


def gyro_frequency(B: float) -> float:
    """
    Calculate electron gyro frequency.
    
    f_g = e * B / (2*pi*m_e)
    
    Args:
        B: Magnetic field strength (T)
    
    Returns:
        Gyro frequency (Hz)
    """
    return E_CHARGE * B / (2 * np.pi * M_ELECTRON)


def skin_depth(sigma: float, f: float) -> float:
    """
    Calculate electromagnetic skin depth.
    
    delta = sqrt(2 / (mu_0 * sigma * omega))
    
    Args:
        sigma: Conductivity (S/m)
        f: Frequency (Hz)
    
    Returns:
        Skin depth (m)
    """
    omega = 2 * np.pi * f
    return np.sqrt(2 / (MU_0 * sigma * omega))


def ionosphere_conductivity(n_e: float, collision_freq: float) -> float:
    """
    Calculate ionospheric conductivity (DC approximation).
    
    sigma = n_e * e^2 / (m_e * nu)
    
    Args:
        n_e: Electron density (m^-3)
        collision_freq: Collision frequency (Hz)
    
    Returns:
        Conductivity (S/m)
    """
    return n_e * E_CHARGE**2 / (M_ELECTRON * collision_freq)


def delta_seg_physical(
    state: IonosphereState,
    params: SSZPhysicalParams = None,
) -> float:
    """
    Calculate SSZ segmentation parameter from ionospheric state.
    
    Physical model:
        delta_seg = alpha * (n_e/n_e_ref - 1) 
                  + beta * (B/B_ref - 1)
                  + gamma * (h/h_ref - 1)
    
    Args:
        state: Ionospheric state
        params: Physical parameters
    
    Returns:
        SSZ segmentation parameter
    """
    if params is None:
        params = SSZPhysicalParams()
    
    # Normalized deviations from reference
    dn_e = (state.n_e / params.n_e_ref) - 1
    dB = (state.B_field / params.B_ref) - 1
    dh = (state.h_iono / params.h_ref) - 1
    
    # SSZ segmentation
    delta_seg = params.alpha * dn_e + params.beta * dB + params.gamma * dh
    
    return delta_seg


def delta_seg_from_proxies(
    f107: float,
    kp: float,
    h_iono: float = H_IONO_REF,
    params: SSZPhysicalParams = None,
) -> float:
    """
    Estimate SSZ segmentation from space weather proxies.
    
    Uses empirical relationships:
        - n_e ~ F10.7^0.5 (solar EUV ionization)
        - B perturbation ~ Kp (geomagnetic activity)
    
    Args:
        f107: F10.7 solar flux (SFU)
        kp: Kp geomagnetic index
        h_iono: Ionosphere height (km)
        params: Physical parameters
    
    Returns:
        Estimated delta_seg
    """
    if params is None:
        params = SSZPhysicalParams()
    
    # Empirical electron density from F10.7
    # n_e ~ 10^11 * (F10.7 / 100)^0.5 m^-3
    n_e = 1e11 * np.sqrt(f107 / 100)
    
    # Magnetic perturbation from Kp
    # dB/B ~ 0.01 * Kp
    B_eff = B_REF * (1 + 0.01 * kp)
    
    # Create state
    state = IonosphereState(
        n_e=n_e,
        T_e=1000,  # Typical value
        h_iono=h_iono,
        B_field=B_eff,
        collision_freq=1e3,  # Typical D-layer
    )
    
    return delta_seg_physical(state, params)


def f_n_ssz_physical(
    n: int,
    state: IonosphereState,
    eta_0: float = ETA_0_DEFAULT,
    params: SSZPhysicalParams = None,
) -> float:
    """
    Calculate SSZ-corrected Schumann frequency using physical model.
    
    Args:
        n: Mode number
        state: Ionospheric state
        eta_0: Baseline eta
        params: Physical parameters
    
    Returns:
        SSZ-corrected frequency (Hz)
    """
    from .classical_schumann import f_n_classical
    
    # Classical frequency
    f_class = f_n_classical(n, eta_0)
    
    # SSZ correction
    delta_seg = delta_seg_physical(state, params)
    
    # SSZ frequency
    f_ssz = f_class / (1 + delta_seg)
    
    return f_ssz


def fit_ssz_physical_params(
    f_obs: Dict[int, np.ndarray],
    f107: np.ndarray,
    kp: np.ndarray,
    h_iono: Optional[np.ndarray] = None,
) -> Tuple[SSZPhysicalParams, Dict]:
    """
    Fit physical SSZ parameters to observed data.
    
    Uses least squares to find optimal alpha, beta, gamma.
    
    Args:
        f_obs: {mode: observed_frequencies}
        f107: F10.7 time series
        kp: Kp time series
        h_iono: Ionosphere height time series (optional)
    
    Returns:
        Fitted parameters and diagnostics
    """
    from scipy.optimize import minimize
    from .classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    
    # Calibrate eta_0
    if 1 in f_obs:
        eta_0 = compute_eta0_from_mean_f1(f_obs[1])
    else:
        eta_0 = ETA_0_DEFAULT
    
    # Default h_iono
    if h_iono is None:
        h_iono = np.full_like(f107, H_IONO_REF)
    
    # Objective function
    def objective(x):
        alpha, beta, gamma = x
        params = SSZPhysicalParams(alpha=alpha, beta=beta, gamma=gamma)
        
        total_error = 0
        n_points = 0
        
        for mode, f_observed in f_obs.items():
            f_class = f_n_classical(mode, eta_0)
            
            # Predicted delta_seg
            delta_seg_pred = np.array([
                delta_seg_from_proxies(f, k, h, params)
                for f, k, h in zip(f107, kp, h_iono)
            ])
            
            # Predicted frequency
            f_pred = f_class / (1 + delta_seg_pred)
            
            # Error
            error = np.sum((f_observed - f_pred)**2)
            total_error += error
            n_points += len(f_observed)
        
        return total_error / n_points
    
    # Initial guess
    x0 = [1e-12, 0.1, 0.01]
    
    # Bounds
    bounds = [
        (-1e-10, 1e-10),  # alpha
        (-1, 1),  # beta
        (-0.1, 0.1),  # gamma
    ]
    
    # Optimize
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    # Extract parameters
    alpha_opt, beta_opt, gamma_opt = result.x
    
    fitted_params = SSZPhysicalParams(
        alpha=alpha_opt,
        beta=beta_opt,
        gamma=gamma_opt,
    )
    
    # Diagnostics
    diagnostics = {
        "success": result.success,
        "rmse": np.sqrt(result.fun),
        "n_iterations": result.nit,
        "alpha": alpha_opt,
        "beta": beta_opt,
        "gamma": gamma_opt,
    }
    
    logger.info(f"Physical SSZ fit:")
    logger.info(f"  alpha = {alpha_opt:.2e}")
    logger.info(f"  beta = {beta_opt:.4f}")
    logger.info(f"  gamma = {gamma_opt:.4f}")
    logger.info(f"  RMSE = {diagnostics['rmse']:.4f} Hz")
    
    return fitted_params, diagnostics


def predict_ssz_signature(
    f107_range: Tuple[float, float] = (70, 200),
    kp_range: Tuple[float, float] = (0, 9),
    params: SSZPhysicalParams = None,
) -> Dict:
    """
    Predict SSZ signature for different space weather conditions.
    
    Args:
        f107_range: Range of F10.7 values
        kp_range: Range of Kp values
        params: Physical parameters
    
    Returns:
        Predictions dictionary
    """
    if params is None:
        params = SSZPhysicalParams()
    
    # Grid
    f107_vals = np.linspace(f107_range[0], f107_range[1], 20)
    kp_vals = np.linspace(kp_range[0], kp_range[1], 10)
    
    # Calculate delta_seg grid
    delta_seg_grid = np.zeros((len(f107_vals), len(kp_vals)))
    
    for i, f107 in enumerate(f107_vals):
        for j, kp in enumerate(kp_vals):
            delta_seg_grid[i, j] = delta_seg_from_proxies(f107, kp, params=params)
    
    # Statistics
    predictions = {
        "f107_vals": f107_vals,
        "kp_vals": kp_vals,
        "delta_seg_grid": delta_seg_grid,
        "delta_seg_min": delta_seg_grid.min(),
        "delta_seg_max": delta_seg_grid.max(),
        "delta_seg_range": delta_seg_grid.max() - delta_seg_grid.min(),
        "f107_sensitivity": np.mean(np.diff(delta_seg_grid, axis=0)),
        "kp_sensitivity": np.mean(np.diff(delta_seg_grid, axis=1)),
    }
    
    logger.info(f"SSZ Signature Predictions:")
    logger.info(f"  delta_seg range: [{predictions['delta_seg_min']:.6f}, {predictions['delta_seg_max']:.6f}]")
    logger.info(f"  F10.7 sensitivity: {predictions['f107_sensitivity']:.2e} per SFU")
    logger.info(f"  Kp sensitivity: {predictions['kp_sensitivity']:.2e} per unit")
    
    return predictions


def print_physical_model_summary():
    """Print summary of physical SSZ model."""
    print("\n" + "="*70)
    print("PHYSICAL SSZ MODEL FOR SCHUMANN RESONANCES")
    print("="*70)
    
    print("\nTheoretical Framework:")
    print("-"*50)
    print("SSZ Hypothesis:")
    print("  c_eff = c / (1 + delta_seg)")
    print("  f_n_SSZ = f_n_classical / (1 + delta_seg)")
    print()
    print("Physical Model:")
    print("  delta_seg = alpha*(n_e/n_e_ref - 1)")
    print("            + beta*(B/B_ref - 1)")
    print("            + gamma*(h/h_ref - 1)")
    
    print("\nReference Values:")
    print("-"*50)
    print(f"  n_e_ref = {N_E_REF:.2e} m^-3 (D-layer)")
    print(f"  B_ref = {B_REF:.2e} T (Earth field)")
    print(f"  h_ref = {H_IONO_REF:.0f} km (ionosphere)")
    
    print("\nPlasma Parameters:")
    print("-"*50)
    f_p = plasma_frequency(N_E_REF)
    f_g = gyro_frequency(B_REF)
    print(f"  Plasma frequency: {f_p:.2f} Hz")
    print(f"  Gyro frequency: {f_g/1e6:.2f} MHz")
    print(f"  f_p / f_Schumann: {f_p/7.83:.2f}")
    
    print("\nKey Predictions:")
    print("-"*50)
    print("1. delta_seg should be MODE-INDEPENDENT")
    print("2. delta_seg correlates with F10.7 (solar activity)")
    print("3. delta_seg correlates with Kp (geomagnetic activity)")
    print("4. All modes show same relative shift: df/f = -delta_seg")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_physical_model_summary()
    
    # Test predictions
    predictions = predict_ssz_signature()
