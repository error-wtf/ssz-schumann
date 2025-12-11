#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layered SSZ Correction Model for Schumann Resonances

Implements a layer-based SSZ correction where each atmospheric layer
(ground, atmosphere, ionosphere) contributes to the total segmentation.

Physical Model:
    The Schumann resonance frequencies are modified by SSZ segmentation
    in each layer of the Earth-ionosphere waveguide:
    
    D_SSZ = 1 + Σ_j w_j · σ_j
    
    where:
        j ∈ {ground, atmosphere, ionosphere}
        w_j = weight of layer j (how much it affects wave propagation)
        σ_j = segmentation parameter of layer j
    
    The SSZ-corrected frequency is:
        f_n^(SSZ) = f_n^(classical) / D_SSZ
    
    Key SSZ signature: ALL modes shift by the same relative factor!
        Δf_n / f_n ≈ -Σ_j w_j · σ_j  (for all n)

Layer Model:
    - Ground (j=g): Hard boundary, typically w_g ≈ 0
    - Atmosphere (j=atm): Neutral layer, small contribution w_atm ≈ 0.2
    - Ionosphere (j=iono): Main waveguide boundary, w_iono ≈ 0.8

φ-Based Segmentation:
    σ_j can be expressed in SSZ φ-language:
        σ_j = λ_j · (φ_seg(r_j) / φ_seg(r_0) - 1)
    
    where φ_seg(r) is the segment density at radius r.

References:
    - Schumann, W.O. (1952) - Original theory
    - Casu & Wrede (2025) - SSZ framework
    - Nickolaenko & Hayakawa (2002) - Schumann resonance review

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, field
import logging

from ..config import C_LIGHT, EARTH_RADIUS, PHI

logger = logging.getLogger(__name__)

# Type alias
ArrayLike = Union[float, np.ndarray, pd.Series]


# =============================================================================
# Layer Configuration
# =============================================================================

@dataclass
class LayerConfig:
    """Configuration for a single atmospheric layer."""
    name: str
    weight: float  # w_j: contribution to wave propagation
    sigma: float = 0.0  # σ_j: segmentation parameter
    height_km: float = 0.0  # Characteristic height above ground
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            logger.warning(f"Layer {self.name}: weight {self.weight} outside [0,1]")


@dataclass 
class LayeredSSZConfig:
    """
    Configuration for layered SSZ model.
    
    Default weights based on Schumann waveguide physics:
    - Ground: ~0 (hard boundary, doesn't affect propagation)
    - Atmosphere: ~0.2 (neutral, small effect)
    - Ionosphere: ~0.8 (main waveguide boundary)
    """
    ground: LayerConfig = field(default_factory=lambda: LayerConfig(
        name="ground", weight=0.0, sigma=0.0, height_km=0
    ))
    atmosphere: LayerConfig = field(default_factory=lambda: LayerConfig(
        name="atmosphere", weight=0.2, sigma=0.0, height_km=50
    ))
    ionosphere: LayerConfig = field(default_factory=lambda: LayerConfig(
        name="ionosphere", weight=0.8, sigma=0.0, height_km=85
    ))
    
    # Classical reference
    eta_0: float = 0.74  # Classical slowdown factor
    f1_ref: float = 7.83  # Reference f1 in Hz
    
    @property
    def layers(self) -> List[LayerConfig]:
        return [self.ground, self.atmosphere, self.ionosphere]
    
    @property
    def total_weight(self) -> float:
        return sum(layer.weight for layer in self.layers)
    
    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = self.total_weight
        if total > 0:
            for layer in self.layers:
                layer.weight /= total


# =============================================================================
# Core SSZ Functions
# =============================================================================

def D_SSZ_layered(config: LayeredSSZConfig) -> float:
    """
    Calculate layered SSZ correction factor.
    
    D_SSZ = 1 + Σ_j w_j · σ_j
    
    Args:
        config: Layer configuration
    
    Returns:
        D_SSZ correction factor
    """
    delta_seg = sum(
        layer.weight * layer.sigma 
        for layer in config.layers
    )
    return 1.0 + delta_seg


def D_SSZ_from_sigmas(
    sigma_ground: float = 0.0,
    sigma_atm: float = 0.0,
    sigma_iono: float = 0.0,
    w_ground: float = 0.0,
    w_atm: float = 0.2,
    w_iono: float = 0.8,
) -> float:
    """
    Calculate D_SSZ from individual layer parameters.
    
    Args:
        sigma_ground: Ground segmentation
        sigma_atm: Atmosphere segmentation
        sigma_iono: Ionosphere segmentation
        w_ground: Ground weight
        w_atm: Atmosphere weight
        w_iono: Ionosphere weight
    
    Returns:
        D_SSZ = 1 + w_g·σ_g + w_atm·σ_atm + w_iono·σ_iono
    """
    delta_seg = (
        w_ground * sigma_ground +
        w_atm * sigma_atm +
        w_iono * sigma_iono
    )
    return 1.0 + delta_seg


def effective_delta_seg(config: LayeredSSZConfig) -> float:
    """
    Calculate effective delta_seg from layer configuration.
    
    δ_seg_eff = Σ_j w_j · σ_j
    
    Args:
        config: Layer configuration
    
    Returns:
        Effective delta_seg
    """
    return sum(layer.weight * layer.sigma for layer in config.layers)


# =============================================================================
# Frequency Calculations
# =============================================================================

def f_n_classical(
    n: int,
    f1_ref: float = 7.83,
) -> float:
    """
    Classical Schumann frequency for mode n.
    
    f_n = f1_ref · √(n(n+1)) / √2
    
    This formulation uses f1 as reference, so classical effects
    (η factor) are already absorbed.
    
    Args:
        n: Mode number (1, 2, 3, ...)
        f1_ref: Reference f1 frequency (Hz)
    
    Returns:
        Classical frequency (Hz)
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    
    mode_factor = np.sqrt(n * (n + 1)) / np.sqrt(2)
    return f1_ref * mode_factor


def f_n_ssz_layered(
    n: int,
    config: LayeredSSZConfig,
) -> float:
    """
    SSZ-corrected Schumann frequency with layered model.
    
    f_n^(SSZ) = f_n^(classical) / D_SSZ
    
    Args:
        n: Mode number
        config: Layer configuration
    
    Returns:
        SSZ-corrected frequency (Hz)
    """
    f_class = f_n_classical(n, config.f1_ref)
    d_ssz = D_SSZ_layered(config)
    return f_class / d_ssz


def compute_all_modes(
    config: LayeredSSZConfig,
    modes: List[int] = [1, 2, 3],
) -> Dict[int, Dict[str, float]]:
    """
    Compute classical and SSZ frequencies for all modes.
    
    Args:
        config: Layer configuration
        modes: List of mode numbers
    
    Returns:
        Dictionary with frequencies and shifts for each mode
    """
    d_ssz = D_SSZ_layered(config)
    delta_seg_eff = effective_delta_seg(config)
    
    results = {}
    for n in modes:
        f_class = f_n_classical(n, config.f1_ref)
        f_ssz = f_class / d_ssz
        
        results[n] = {
            "f_classical": f_class,
            "f_ssz": f_ssz,
            "delta_f": f_ssz - f_class,
            "relative_shift": (f_ssz - f_class) / f_class,
        }
    
    return results


# =============================================================================
# φ-Based Segmentation
# =============================================================================

def sigma_from_phi_ratio(
    phi_seg_layer: float,
    phi_seg_ref: float,
    lambda_coupling: float = 1.0,
) -> float:
    """
    Calculate σ from φ-based segment density ratio.
    
    σ = λ · (φ_seg(r) / φ_seg(r_0) - 1)
    
    Args:
        phi_seg_layer: Segment density at layer
        phi_seg_ref: Reference segment density
        lambda_coupling: Coupling strength
    
    Returns:
        Segmentation parameter σ
    """
    if phi_seg_ref == 0:
        return 0.0
    
    return lambda_coupling * (phi_seg_layer / phi_seg_ref - 1)


def Xi_ssz(
    r: ArrayLike,
    r_s: float,
    Xi_max: float = 1.0,
) -> ArrayLike:
    """
    SSZ Segment Density Field (CORRECT formula from ssz-metric-pure).
    
    Formula:
        Xi(r) = Xi_max * (1 - exp(-phi * r / r_s))
    
    Properties:
        - Xi(0) = 0 (no segments at center)
        - Xi(infinity) -> Xi_max (saturates)
        - phi in exponent ensures phi-based scaling
    
    Args:
        r: Radius (scalar or array)
        r_s: Schwarzschild radius (or reference scale)
        Xi_max: Maximum segment density (default 1.0)
    
    Returns:
        Segment density Xi(r) in [0, Xi_max]
    
    Reference:
        ssz-metric-pure/src/ssz_metric_pure/segmentation.py
    """
    r = np.asarray(r)
    return Xi_max * (1 - np.exp(-PHI * r / r_s))


def D_SSZ_from_Xi(
    Xi: ArrayLike,
) -> ArrayLike:
    """
    SSZ Time Dilation from Segment Density (CORRECT formula).
    
    Formula:
        D_SSZ = 1 / (1 + Xi)
    
    Properties:
        - D_SSZ(Xi=0) = 1.0 (no time dilation)
        - D_SSZ(Xi=1) = 0.5 (maximum time dilation)
        - 0 < D_SSZ <= 1 always (no singularity!)
    
    Args:
        Xi: Segment density
    
    Returns:
        Time dilation factor D_SSZ in (0, 1]
    
    Reference:
        ssz-metric-pure/src/ssz_core/segment_density.py
    """
    Xi = np.asarray(Xi)
    return 1.0 / (1.0 + Xi)


def phi_segment_density(
    r: float,
    r_s: float = 1.0,
    model: str = "ssz_core",
) -> float:
    """
    Calculate phi-based segment density.
    
    From SSZ core theory (CORRECT):
        Xi(r) = Xi_max * (1 - exp(-phi * r/r_s))
    
    Args:
        r: Radius
        r_s: Schwarzschild radius (or reference scale)
        model: "ssz_core" or "linear"
    
    Returns:
        Segment density
    
    Note:
        Prefer Xi_ssz() for the canonical implementation.
    """
    if model == "ssz_core":
        # From ssz-metric-pure: Xi(r) = 1 - exp(-phi * r/r_s)
        return 1 - np.exp(-PHI * r / r_s)
    elif model == "linear":
        # Simple linear model (approximation for small r)
        return PHI * r / r_s
    else:
        raise ValueError(f"Unknown model: {model}")


def create_phi_based_config(
    r_ground: float = EARTH_RADIUS,
    r_atm: float = EARTH_RADIUS + 50e3,
    r_iono: float = EARTH_RADIUS + 85e3,
    r_ref: float = EARTH_RADIUS,
    lambda_atm: float = 0.01,
    lambda_iono: float = 0.01,
    r_s: float = 1e9,  # Reference scale for segment density
) -> LayeredSSZConfig:
    """
    Create layer configuration from φ-based segment densities.
    
    Args:
        r_ground: Ground radius (m)
        r_atm: Atmosphere characteristic radius (m)
        r_iono: Ionosphere characteristic radius (m)
        r_ref: Reference radius for segment density
        lambda_atm: Atmosphere coupling strength
        lambda_iono: Ionosphere coupling strength
        r_s: Reference scale for segment density
    
    Returns:
        LayeredSSZConfig with φ-based σ values
    """
    # Calculate segment densities
    phi_ref = phi_segment_density(r_ref, r_s)
    phi_atm = phi_segment_density(r_atm, r_s)
    phi_iono = phi_segment_density(r_iono, r_s)
    
    # Calculate σ values
    sigma_atm = sigma_from_phi_ratio(phi_atm, phi_ref, lambda_atm)
    sigma_iono = sigma_from_phi_ratio(phi_iono, phi_ref, lambda_iono)
    
    config = LayeredSSZConfig()
    config.atmosphere.sigma = sigma_atm
    config.ionosphere.sigma = sigma_iono
    
    return config


# =============================================================================
# Time-Varying Model
# =============================================================================

def sigma_iono_from_proxy(
    F_iono: ArrayLike,
    beta_0: float = 0.0,
    beta_1: float = 0.01,
) -> ArrayLike:
    """
    Calculate ionosphere σ from ionospheric proxy.
    
    σ_iono(t) = β_0 + β_1 · F_iono(t)
    
    Args:
        F_iono: Ionospheric proxy (e.g., normalized F10.7, Kp)
        beta_0: Baseline segmentation
        beta_1: Coupling to proxy
    
    Returns:
        Time-varying σ_iono
    """
    return beta_0 + beta_1 * F_iono


def f_n_ssz_timeseries(
    n: int,
    sigma_iono_t: ArrayLike,
    sigma_atm: float = 0.0,
    w_atm: float = 0.2,
    w_iono: float = 0.8,
    f1_ref: float = 7.83,
) -> ArrayLike:
    """
    Calculate SSZ frequency time series.
    
    Args:
        n: Mode number
        sigma_iono_t: Time-varying ionosphere σ
        sigma_atm: Constant atmosphere σ
        w_atm: Atmosphere weight
        w_iono: Ionosphere weight
        f1_ref: Reference f1
    
    Returns:
        Time series of f_n^(SSZ)
    """
    # Classical frequency
    f_class = f_n_classical(n, f1_ref)
    
    # Time-varying D_SSZ
    delta_seg_t = w_atm * sigma_atm + w_iono * sigma_iono_t
    d_ssz_t = 1.0 + delta_seg_t
    
    return f_class / d_ssz_t


# =============================================================================
# Fitting Functions
# =============================================================================

@dataclass
class LayeredFitResult:
    """Result of layered SSZ fit."""
    beta_0: float
    beta_1: float
    w_atm: float
    w_iono: float
    r_squared: float
    rmse: float
    n_points: int
    residuals: Optional[np.ndarray] = None


def fit_layered_ssz(
    f_obs: Dict[int, pd.Series],
    F_iono: pd.Series,
    f1_ref: float = 7.83,
    w_atm: float = 0.2,
    w_iono: float = 0.8,
    sigma_atm: float = 0.0,
) -> LayeredFitResult:
    """
    Fit layered SSZ model to observed frequencies.
    
    Model:
        f_n^(obs) = f_n^(classical) / (1 + w_atm·σ_atm + w_iono·(β_0 + β_1·F_iono))
    
    Args:
        f_obs: {mode: observed_frequency_series}
        F_iono: Ionospheric proxy time series
        f1_ref: Reference f1
        w_atm: Atmosphere weight (fixed)
        w_iono: Ionosphere weight (fixed)
        sigma_atm: Atmosphere σ (fixed)
    
    Returns:
        LayeredFitResult with fitted β_0, β_1
    """
    from sklearn.linear_model import LinearRegression
    
    # Combine all modes
    all_delta_seg = []
    all_F_iono = []
    
    for n, f_series in f_obs.items():
        f_class = f_n_classical(n, f1_ref)
        
        # Extract delta_seg from observations
        # f_obs = f_class / (1 + delta_seg)
        # => delta_seg = f_class / f_obs - 1
        delta_seg = f_class / f_series - 1
        
        # Remove atmosphere contribution
        delta_seg_iono = (delta_seg - w_atm * sigma_atm) / w_iono
        
        # Align with F_iono
        common_idx = delta_seg.index.intersection(F_iono.index)
        
        all_delta_seg.extend(delta_seg_iono.loc[common_idx].values)
        all_F_iono.extend(F_iono.loc[common_idx].values)
    
    # Convert to arrays
    y = np.array(all_delta_seg)
    X = np.array(all_F_iono).reshape(-1, 1)
    
    # Remove NaN
    valid = ~(np.isnan(y) | np.isnan(X.flatten()))
    y = y[valid]
    X = X[valid]
    
    n_points = len(y)
    if n_points < 10:
        raise ValueError(f"Not enough valid points: {n_points}")
    
    # Fit: σ_iono = β_0 + β_1 · F_iono
    model = LinearRegression()
    model.fit(X, y)
    
    beta_0 = model.intercept_
    beta_1 = model.coef_[0]
    
    # Predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))
    
    result = LayeredFitResult(
        beta_0=beta_0,
        beta_1=beta_1,
        w_atm=w_atm,
        w_iono=w_iono,
        r_squared=r_squared,
        rmse=rmse,
        n_points=n_points,
        residuals=residuals,
    )
    
    logger.info(f"Layered SSZ fit (n={n_points}):")
    logger.info(f"  β_0 = {beta_0:.6f}")
    logger.info(f"  β_1 = {beta_1:.6f}")
    logger.info(f"  R² = {r_squared:.4f}")
    logger.info(f"  RMSE = {rmse:.6f}")
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def frequency_shift_estimate(
    delta_seg_eff: float,
    f_ref: float = 7.83,
) -> Dict[str, float]:
    """
    Estimate frequency shifts for a given effective δ_seg.
    
    Args:
        delta_seg_eff: Effective segmentation (Σ w_j·σ_j)
        f_ref: Reference f1 frequency
    
    Returns:
        Dictionary with shifts for modes 1, 2, 3
    """
    d_ssz = 1 + delta_seg_eff
    
    results = {}
    for n in [1, 2, 3]:
        f_class = f_n_classical(n, f_ref)
        f_ssz = f_class / d_ssz
        
        results[f"f{n}_classical"] = f_class
        results[f"f{n}_ssz"] = f_ssz
        results[f"delta_f{n}"] = f_ssz - f_class
        results[f"relative_shift_{n}"] = (f_ssz - f_class) / f_class
    
    results["delta_seg_eff"] = delta_seg_eff
    results["D_SSZ"] = d_ssz
    
    return results


def print_frequency_table(
    delta_seg_values: List[float] = [0.0, 0.005, 0.01, 0.02],
    f_ref: float = 7.83,
):
    """
    Print table of frequency shifts for different delta_seg values.
    
    Args:
        delta_seg_values: List of delta_seg to compute
        f_ref: Reference f1
    """
    print("\n" + "=" * 80)
    print("SSZ FREQUENCY SHIFT TABLE")
    print("=" * 80)
    print(f"Reference f1 = {f_ref} Hz")
    print("-" * 80)
    print(f"{'delta_seg':>10} | {'f1 (Hz)':>10} | {'f2 (Hz)':>10} | {'f3 (Hz)':>10} | {'Df1 (Hz)':>10}")
    print("-" * 80)
    
    for delta_seg in delta_seg_values:
        result = frequency_shift_estimate(delta_seg, f_ref)
        print(f"{delta_seg:>10.4f} | "
              f"{result['f1_ssz']:>10.4f} | "
              f"{result['f2_ssz']:>10.4f} | "
              f"{result['f3_ssz']:>10.4f} | "
              f"{result['delta_f1']:>+10.4f}")
    
    print("=" * 80)
    print("\nInterpretation:")
    print("  delta_seg = 0.01 (1%) -> Df1 ~ -0.08 Hz")
    print("  This is within typical observed variations (+/-0.1-0.2 Hz)")
    print("=" * 80)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: 1% ionosphere segmentation
    print("\n" + "=" * 80)
    print("LAYERED SSZ MODEL - EXAMPLE")
    print("=" * 80)
    
    # Create configuration
    config = LayeredSSZConfig()
    config.ionosphere.sigma = 0.01  # 1% segmentation
    
    print(f"\nConfiguration:")
    print(f"  Ground:     w={config.ground.weight:.2f}, sigma={config.ground.sigma:.4f}")
    print(f"  Atmosphere: w={config.atmosphere.weight:.2f}, sigma={config.atmosphere.sigma:.4f}")
    print(f"  Ionosphere: w={config.ionosphere.weight:.2f}, sigma={config.ionosphere.sigma:.4f}")
    
    # Calculate D_SSZ
    d_ssz = D_SSZ_layered(config)
    delta_seg_eff = effective_delta_seg(config)
    
    print(f"\nEffective delta_seg = {delta_seg_eff:.4f}")
    print(f"D_SSZ = {d_ssz:.6f}")
    
    # Calculate frequencies
    print(f"\nFrequencies:")
    results = compute_all_modes(config)
    for n, data in results.items():
        print(f"  Mode {n}: f_class={data['f_classical']:.2f} Hz -> "
              f"f_SSZ={data['f_ssz']:.2f} Hz (Df={data['delta_f']:+.3f} Hz)")
    
    # Print full table
    print_frequency_table()
