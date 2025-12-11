#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Schumann Resonance Model

Implements the classical theory of Schumann resonances in the
Earth-ionosphere cavity.

Physical Background:
    The Earth and ionosphere form a spherical waveguide for ELF waves.
    The resonance frequencies are given by:
    
        f_n = c / (2*pi*R) * sqrt(n*(n+1))  [ideal]
    
    Real frequencies are lower due to:
    - Finite conductivity of Earth and ionosphere
    - Non-uniform ionosphere height
    - Day/night asymmetry
    
    We parameterize this as:
    
        f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    where eta < 1 is an effective "slowdown factor".

References:
    - Schumann, W.O. (1952) - Original theory
    - Nickolaenko & Hayakawa (2002) - Comprehensive review
    - Salinas et al. (2022) - Sierra Nevada measurements

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import logging

from ..config import C_LIGHT, EARTH_RADIUS, ETA_0_DEFAULT

logger = logging.getLogger(__name__)

# Type alias
ArrayLike = Union[float, np.ndarray, pd.Series]


def schumann_mode_factor(n: int) -> float:
    """
    Calculate the mode factor sqrt(n*(n+1)).
    
    This is the geometric factor that determines the frequency
    spacing of Schumann modes.
    
    Args:
        n: Mode number (1, 2, 3, ...)
    
    Returns:
        sqrt(n*(n+1))
    
    Examples:
        >>> schumann_mode_factor(1)
        1.4142135623730951  # sqrt(2)
        >>> schumann_mode_factor(2)
        2.449489742783178   # sqrt(6)
        >>> schumann_mode_factor(3)
        3.4641016151377544  # sqrt(12)
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    return np.sqrt(n * (n + 1))


def f_n_classical(
    n: int,
    eta: float = ETA_0_DEFAULT,
    R_earth: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> float:
    """
    Calculate classical Schumann resonance frequency for mode n.
    
    Formula:
        f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    Args:
        n: Mode number (1, 2, 3, ...)
        eta: Effective slowdown factor (0 < eta < 1)
            Default: ~0.74 (calibrated to observed f1 ~ 7.83 Hz)
        R_earth: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        Resonance frequency (Hz)
    
    Physical Interpretation:
        - eta = 1: Ideal conducting sphere (f1 ~ 10.6 Hz)
        - eta ~ 0.74: Real Earth-ionosphere (f1 ~ 7.83 Hz)
        - eta encapsulates all "classical" effects:
          * Finite conductivity
          * Ionosphere structure
          * Day/night asymmetry
    
    Examples:
        >>> f_n_classical(1)  # Fundamental mode
        7.83...
        >>> f_n_classical(2)  # Second mode
        14.1...
        >>> f_n_classical(3)  # Third mode
        20.3...
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    if not 0 < eta <= 1:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    
    # f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    mode_factor = schumann_mode_factor(n)
    f = eta * c / (2 * np.pi * R_earth) * mode_factor
    
    return f


def f_n_classical_timeseries(
    n: int,
    eta: ArrayLike,
    time_index: Optional[pd.DatetimeIndex] = None,
    R_earth: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> pd.Series:
    """
    Calculate classical Schumann frequency as time series.
    
    Allows eta to vary with time, enabling modeling of
    ionospheric variations.
    
    Args:
        n: Mode number
        eta: Slowdown factor (scalar or time series)
        time_index: Time index for output
            If None, uses index from eta (if Series)
        R_earth: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        pd.Series with frequency values
    
    Example:
        >>> # Constant eta
        >>> f1 = f_n_classical_timeseries(1, eta=0.74, time_index=time)
        
        >>> # Time-varying eta
        >>> eta_t = pd.Series(0.74 + 0.01*np.sin(2*np.pi*t/365), index=time)
        >>> f1 = f_n_classical_timeseries(1, eta=eta_t)
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    
    # Convert to array
    if isinstance(eta, pd.Series):
        if time_index is None:
            time_index = eta.index
        eta_values = eta.values
    elif isinstance(eta, np.ndarray):
        eta_values = eta
    else:
        # Scalar eta - broadcast to time_index length
        if time_index is not None:
            eta_values = np.full(len(time_index), float(eta))
        else:
            eta_values = np.atleast_1d(eta)
    
    # Validate eta
    if np.any(eta_values <= 0) or np.any(eta_values > 1):
        logger.warning("eta values outside (0, 1] detected")
    
    # Calculate frequency
    mode_factor = schumann_mode_factor(n)
    f_values = eta_values * c / (2 * np.pi * R_earth) * mode_factor
    
    # Create output series
    if time_index is not None:
        result = pd.Series(f_values, index=time_index, name=f"f{n}_classical")
    else:
        result = pd.Series(f_values, name=f"f{n}_classical")
    
    return result


def compute_eta0_from_mean_f1(
    f1_series: Union[pd.Series, np.ndarray, float],
    R_earth: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> float:
    """
    Compute baseline eta_0 from observed mean f1.
    
    Inverts the classical formula:
        eta_0 = f1_mean / [c / (2*pi*R) * sqrt(2)]
    
    This calibrates the model to match the observed
    fundamental frequency.
    
    Args:
        f1_series: Observed f1 values (Hz)
            Can be array (uses mean) or scalar
        R_earth: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        Calibrated eta_0 value
    
    Example:
        >>> f1_obs = pd.Series([7.82, 7.84, 7.83, 7.85])
        >>> eta_0 = compute_eta0_from_mean_f1(f1_obs)
        >>> print(f"eta_0 = {eta_0:.4f}")
        eta_0 = 0.7390
    """
    # Get mean f1
    if isinstance(f1_series, (pd.Series, np.ndarray)):
        f1_mean = np.nanmean(f1_series)
    else:
        f1_mean = float(f1_series)
    
    # Ideal f1 (eta = 1)
    f1_ideal = c / (2 * np.pi * R_earth) * np.sqrt(2)
    
    # Compute eta_0
    eta_0 = f1_mean / f1_ideal
    
    logger.info(f"Calibrated eta_0 = {eta_0:.6f} from f1_mean = {f1_mean:.4f} Hz")
    logger.info(f"  (Ideal f1 = {f1_ideal:.4f} Hz)")
    
    return eta_0


def compute_eta_timeseries(
    f1_obs: pd.Series,
    R_earth: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> pd.Series:
    """
    Compute time-varying eta from observed f1.
    
    This is the inverse operation: given observed f1(t),
    compute what eta(t) would be needed in the classical model.
    
    Args:
        f1_obs: Observed f1 time series (Hz)
        R_earth: Earth radius (m)
        c: Speed of light (m/s)
    
    Returns:
        eta(t) time series
    
    Note:
        This is useful for analyzing what "classical" variations
        would be needed to explain the data, before introducing SSZ.
    """
    f1_ideal = c / (2 * np.pi * R_earth) * np.sqrt(2)
    eta = f1_obs / f1_ideal
    eta.name = "eta"
    
    return eta


def predict_mode_ratios(eta: float = ETA_0_DEFAULT) -> dict:
    """
    Predict frequency ratios between modes.
    
    In the classical model, the ratios are independent of eta:
        f2/f1 = sqrt(6)/sqrt(2) = sqrt(3) ~ 1.732
        f3/f1 = sqrt(12)/sqrt(2) = sqrt(6) ~ 2.449
        f3/f2 = sqrt(12)/sqrt(6) = sqrt(2) ~ 1.414
    
    Args:
        eta: Slowdown factor (doesn't affect ratios)
    
    Returns:
        Dictionary with predicted ratios
    
    Note:
        Deviations from these ratios indicate non-classical effects
        (e.g., mode-dependent dispersion).
    """
    # Mode factors
    m1 = schumann_mode_factor(1)  # sqrt(2)
    m2 = schumann_mode_factor(2)  # sqrt(6)
    m3 = schumann_mode_factor(3)  # sqrt(12)
    
    ratios = {
        "f2_f1": m2 / m1,  # sqrt(3) ~ 1.732
        "f3_f1": m3 / m1,  # sqrt(6) ~ 2.449
        "f3_f2": m3 / m2,  # sqrt(2) ~ 1.414
    }
    
    # Also compute absolute frequencies for reference
    ratios["f1"] = f_n_classical(1, eta)
    ratios["f2"] = f_n_classical(2, eta)
    ratios["f3"] = f_n_classical(3, eta)
    
    return ratios


# =============================================================================
# T1: EXTENDED GEOMETRY-AWARE CLASSICAL MODEL
# =============================================================================

def f_n_classical_extended(
    n: int,
    eta: float = ETA_0_DEFAULT,
    R_ground: float = EARTH_RADIUS,
    h_iono: float = 80_000.0,
    c: float = C_LIGHT,
    include_height_correction: bool = True,
) -> float:
    """
    Extended classical Schumann frequency with explicit geometry.
    
    This function provides a more faithful classical model by explicitly
    accounting for ionospheric height, while keeping the SSZ correction
    as a separate, purely additional factor.
    
    Formula (SSZ_FORMULAS.md Eq. 3):
        f_n = eta * c / (2*pi*R_eff) * sqrt(n*(n+1))
    
    where R_eff depends on the geometry:
        - Simple: R_eff = R_ground
        - With height correction: R_eff = R_ground + h_iono/2
          (wave propagates at effective height between ground and ionosphere)
    
    Args:
        n: Mode number (1, 2, 3, ...)
        eta: Effective slowdown factor (0 < eta < 1)
            Absorbs finite conductivity, day/night asymmetry, etc.
        R_ground: Earth radius (m), default 6.371e6
        h_iono: Ionospheric height (m), default 80 km
            Typical range: 60-100 km (day/night variation)
        c: Speed of light (m/s)
        include_height_correction: If True, use R_eff = R_ground + h_iono/2
            If False, use R_eff = R_ground (backward compatible)
    
    Returns:
        Resonance frequency (Hz)
    
    Physical Interpretation:
        The ionospheric height h_iono affects the effective cavity size.
        During day: h_iono ~ 60-70 km (D-layer absorption)
        During night: h_iono ~ 80-90 km (E-layer reflection)
        
        This variation is CLASSICAL - it does NOT involve SSZ.
        The SSZ correction D_SSZ is applied separately on top.
    
    Relation to SSZ_FORMULAS.md:
        - This function implements the "classical part" of Eq. 5
        - The SSZ correction is: f_n_SSZ = f_n_classical / D_SSZ
        - D_SSZ = 1 + delta_seg (Eq. 6)
    
    Examples:
        >>> f_n_classical_extended(1)  # Default parameters
        7.83...
        
        >>> # Day vs night ionosphere
        >>> f_day = f_n_classical_extended(1, h_iono=65_000)
        >>> f_night = f_n_classical_extended(1, h_iono=85_000)
        >>> print(f"Day: {f_day:.3f} Hz, Night: {f_night:.3f} Hz")
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    if not 0 < eta <= 1:
        raise ValueError(f"eta must be in (0, 1], got {eta}")
    if h_iono < 0:
        raise ValueError(f"h_iono must be >= 0, got {h_iono}")
    
    # Effective radius
    if include_height_correction:
        R_eff = R_ground + h_iono / 2
    else:
        R_eff = R_ground
    
    # f_n = eta * c / (2*pi*R_eff) * sqrt(n*(n+1))
    mode_factor = schumann_mode_factor(n)
    f = eta * c / (2 * np.pi * R_eff) * mode_factor
    
    return f


def f_n_classical_with_latitude(
    n: int,
    eta: float = ETA_0_DEFAULT,
    R_ground: float = EARTH_RADIUS,
    h_iono_equator: float = 80_000.0,
    h_iono_pole: float = 100_000.0,
    latitude_deg: float = 0.0,
    c: float = C_LIGHT,
) -> float:
    """
    Classical Schumann frequency with latitude-dependent ionosphere.
    
    The ionospheric height varies with latitude due to:
    - Solar zenith angle effects
    - Magnetic field geometry
    - Particle precipitation at high latitudes
    
    Formula:
        h_iono(lat) = h_equator + (h_pole - h_equator) * sin^2(lat)
    
    Args:
        n: Mode number
        eta: Slowdown factor
        R_ground: Earth radius (m)
        h_iono_equator: Ionospheric height at equator (m)
        h_iono_pole: Ionospheric height at poles (m)
        latitude_deg: Geographic latitude (degrees)
        c: Speed of light (m/s)
    
    Returns:
        Resonance frequency (Hz)
    
    Note:
        This is a simplified model. Real ionospheric structure is
        much more complex (local time, season, solar activity, etc.).
        
        For SSZ analysis, this latitude dependence is CLASSICAL.
        The SSZ signature is the uniform relative shift across all modes.
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    
    # Latitude-dependent ionospheric height
    lat_rad = np.radians(latitude_deg)
    h_iono = h_iono_equator + (h_iono_pole - h_iono_equator) * np.sin(lat_rad)**2
    
    return f_n_classical_extended(n, eta, R_ground, h_iono, c)


def f_n_classical_diurnal(
    n: int,
    local_hour: float,
    eta: float = ETA_0_DEFAULT,
    R_ground: float = EARTH_RADIUS,
    h_iono_day: float = 65_000.0,
    h_iono_night: float = 85_000.0,
    c: float = C_LIGHT,
) -> float:
    """
    Classical Schumann frequency with diurnal (day/night) variation.
    
    The ionospheric height varies with local time:
    - Day (6-18h): Lower ionosphere due to D-layer
    - Night (18-6h): Higher ionosphere, E-layer reflection
    
    Formula:
        h_iono(t) = h_mean + delta_h * cos(2*pi*(t-12)/24)
    
    where:
        h_mean = (h_day + h_night) / 2
        delta_h = (h_night - h_day) / 2
    
    Args:
        n: Mode number
        local_hour: Local solar time (0-24 hours)
        eta: Slowdown factor
        R_ground: Earth radius (m)
        h_iono_day: Ionospheric height at local noon (m)
        h_iono_night: Ionospheric height at local midnight (m)
        c: Speed of light (m/s)
    
    Returns:
        Resonance frequency (Hz)
    
    Example:
        >>> # Frequency at noon vs midnight
        >>> f_noon = f_n_classical_diurnal(1, local_hour=12)
        >>> f_midnight = f_n_classical_diurnal(1, local_hour=0)
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    
    # Diurnal variation
    # At noon (hour=12): h_iono = h_day (lowest, D-layer) -> higher frequency
    # At midnight (hour=0 or 24): h_iono = h_night (highest, E-layer) -> lower frequency
    # 
    # We want: h(12) = h_day, h(0) = h_night
    # Using: h = h_mean + amplitude * cos(phase)
    # where phase = 0 at midnight, pi at noon
    # cos(0) = 1 -> h_night, cos(pi) = -1 -> h_day
    h_mean = (h_iono_day + h_iono_night) / 2
    amplitude = (h_iono_night - h_iono_day) / 2
    phase = np.pi * local_hour / 12  # 0 at hour=0, pi at hour=12
    h_iono = h_mean + amplitude * np.cos(phase)
    
    return f_n_classical_extended(n, eta, R_ground, h_iono, c)


def classical_model_residuals(
    f_obs: dict,
    eta: float,
    modes: list = [1, 2, 3],
) -> dict:
    """
    Compute residuals between observed and classical model.
    
    Args:
        f_obs: Dictionary {n: f_n_observed} or {n: pd.Series}
        eta: Slowdown factor
        modes: List of mode numbers to compute
    
    Returns:
        Dictionary with residuals for each mode:
            - absolute: f_obs - f_classical (Hz)
            - relative: (f_obs - f_classical) / f_classical
    """
    residuals = {}
    
    for n in modes:
        if n not in f_obs:
            continue
        
        f_class = f_n_classical(n, eta)
        f_observed = f_obs[n]
        
        if isinstance(f_observed, pd.Series):
            abs_resid = f_observed - f_class
            rel_resid = abs_resid / f_class
        else:
            abs_resid = float(f_observed) - f_class
            rel_resid = abs_resid / f_class
        
        residuals[n] = {
            "absolute": abs_resid,
            "relative": rel_resid,
            "f_classical": f_class,
        }
    
    return residuals
