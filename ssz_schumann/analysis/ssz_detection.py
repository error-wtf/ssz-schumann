#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Detection Statistics Module

Implements rigorous statistical tests for SSZ signature detection:
1. T_SSZ test statistic with clear definition
2. Null hypothesis ensemble generation
3. P-value estimation
4. Sensitivity curves with detection probability

The SSZ signature is characterized by:
- High correlation between delta_seg from different modes
- Low relative scatter between modes
- Consistent temporal pattern across all modes

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings


@dataclass
class SSZTestResult:
    """Result of SSZ detection test."""
    T_SSZ: float                    # Test statistic
    p_value: float                  # P-value under null hypothesis
    is_significant: bool            # Whether p < alpha
    correlation_mean: float         # Mean mode-to-mode correlation
    scatter_ratio: float            # Relative scatter between modes
    n_modes: int
    n_times: int
    metadata: Dict


@dataclass
class NullDistribution:
    """Null distribution of T_SSZ."""
    T_values: np.ndarray            # T_SSZ values from null realizations
    mean: float
    std: float
    percentiles: Dict[float, float]  # {p: value} for key percentiles
    n_realizations: int


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""
    amplitudes: np.ndarray          # SSZ amplitudes tested
    detection_rates: np.ndarray     # P(p < alpha) for each amplitude
    mean_T_SSZ: np.ndarray          # Mean T_SSZ for each amplitude
    std_T_SSZ: np.ndarray           # Std T_SSZ for each amplitude
    detection_threshold: float      # Minimum detectable amplitude at 95% power
    noise_level: float
    n_realizations: int


# =============================================================================
# T_SSZ TEST STATISTIC
# =============================================================================

def compute_mode_correlations(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> Tuple[float, np.ndarray]:
    """
    Compute pairwise correlations between delta_seg from different modes.
    
    Args:
        delta_seg: Dictionary of delta_seg arrays per mode
        modes: List of modes
    
    Returns:
        Tuple of (mean_correlation, correlation_matrix)
    """
    n_modes = len(modes)
    corr_matrix = np.eye(n_modes)
    
    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            if i < j:
                # Handle NaN values
                valid = ~(np.isnan(delta_seg[m1]) | np.isnan(delta_seg[m2]))
                if np.sum(valid) > 2:
                    r = np.corrcoef(delta_seg[m1][valid], delta_seg[m2][valid])[0, 1]
                    if np.isnan(r):
                        r = 0.0
                else:
                    r = 0.0
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
    
    # Mean of off-diagonal elements
    mask = ~np.eye(n_modes, dtype=bool)
    mean_corr = np.mean(corr_matrix[mask])
    
    return mean_corr, corr_matrix


def compute_scatter_ratio(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> float:
    """
    Compute relative scatter between modes.
    
    Scatter ratio = std(between modes) / mean(within mode std)
    
    Low scatter ratio indicates consistent SSZ signal across modes.
    
    Args:
        delta_seg: Dictionary of delta_seg arrays per mode
        modes: List of modes
    
    Returns:
        Scatter ratio (lower is better for SSZ)
    """
    # Stack delta_seg from all modes
    stacked = np.array([delta_seg[n] for n in modes])
    
    # Between-mode scatter at each time point
    between_std = np.nanstd(stacked, axis=0)
    mean_between = np.nanmean(between_std)
    
    # Within-mode scatter (temporal std)
    within_stds = [np.nanstd(delta_seg[n]) for n in modes]
    mean_within = np.mean(within_stds)
    
    if mean_within > 0:
        scatter_ratio = mean_between / mean_within
    else:
        scatter_ratio = np.inf
    
    return scatter_ratio


def compute_T_SSZ(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
    w_corr: float = 0.7,
    w_scatter: float = 0.3,
) -> float:
    """
    Compute the SSZ test statistic T_SSZ.
    
    Formula:
        T_SSZ = w_corr * r_mean + w_scatter * (1 - scatter_ratio)
    
    where:
        r_mean = mean pairwise correlation between modes
        scatter_ratio = relative scatter between modes
    
    Higher T_SSZ indicates stronger SSZ signature.
    
    Args:
        delta_seg: Dictionary of delta_seg arrays per mode
        modes: List of modes
        w_corr: Weight for correlation component
        w_scatter: Weight for scatter component
    
    Returns:
        T_SSZ test statistic
    """
    # Correlation component
    mean_corr, _ = compute_mode_correlations(delta_seg, modes)
    
    # Scatter component
    scatter_ratio = compute_scatter_ratio(delta_seg, modes)
    
    # Clamp scatter ratio to [0, 2] for stability
    scatter_ratio = np.clip(scatter_ratio, 0, 2)
    
    # Combine: high correlation and low scatter = high T_SSZ
    T_SSZ = w_corr * mean_corr + w_scatter * (1.0 - scatter_ratio)
    
    return T_SSZ


# =============================================================================
# NULL HYPOTHESIS ENSEMBLE
# =============================================================================

def generate_null_realization_shuffle(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> Dict[int, np.ndarray]:
    """
    Generate null realization by shuffling time indices independently per mode.
    
    This destroys any true temporal correlation while preserving marginal distributions.
    
    Args:
        delta_seg: Original delta_seg arrays
        modes: List of modes
    
    Returns:
        Shuffled delta_seg dictionary
    """
    null_delta_seg = {}
    for n in modes:
        null_delta_seg[n] = np.random.permutation(delta_seg[n])
    
    return null_delta_seg


def generate_null_realization_phase_randomize(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> Dict[int, np.ndarray]:
    """
    Generate null realization by phase randomization in Fourier domain.
    
    This preserves the power spectrum but destroys phase coherence.
    
    Args:
        delta_seg: Original delta_seg arrays
        modes: List of modes
    
    Returns:
        Phase-randomized delta_seg dictionary
    """
    null_delta_seg = {}
    
    for n in modes:
        x = delta_seg[n]
        n_pts = len(x)
        
        # FFT
        X = np.fft.rfft(x)
        
        # Random phases (preserve DC and Nyquist)
        phases = np.random.uniform(0, 2*np.pi, len(X))
        phases[0] = 0  # DC component
        if n_pts % 2 == 0:
            phases[-1] = 0  # Nyquist
        
        # Apply random phases
        X_rand = np.abs(X) * np.exp(1j * phases)
        
        # Inverse FFT
        null_delta_seg[n] = np.fft.irfft(X_rand, n=n_pts)
    
    return null_delta_seg


def generate_null_realization_noise(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
) -> Dict[int, np.ndarray]:
    """
    Generate null realization with pure Gaussian noise.
    
    Noise has same mean and std as original data.
    
    Args:
        delta_seg: Original delta_seg arrays
        modes: List of modes
    
    Returns:
        Noise-only delta_seg dictionary
    """
    null_delta_seg = {}
    
    for n in modes:
        mean = np.nanmean(delta_seg[n])
        std = np.nanstd(delta_seg[n])
        null_delta_seg[n] = np.random.normal(mean, std, len(delta_seg[n]))
    
    return null_delta_seg


def compute_null_distribution(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
    n_realizations: int = 1000,
    method: str = "shuffle",
    w_corr: float = 0.7,
    w_scatter: float = 0.3,
) -> NullDistribution:
    """
    Compute null distribution of T_SSZ.
    
    Args:
        delta_seg: Original delta_seg arrays
        modes: List of modes
        n_realizations: Number of null realizations
        method: "shuffle", "phase_randomize", or "noise"
        w_corr: Weight for correlation in T_SSZ
        w_scatter: Weight for scatter in T_SSZ
    
    Returns:
        NullDistribution object
    """
    T_values = np.zeros(n_realizations)
    
    for i in range(n_realizations):
        # Generate null realization
        if method == "shuffle":
            null_delta_seg = generate_null_realization_shuffle(delta_seg, modes)
        elif method == "phase_randomize":
            null_delta_seg = generate_null_realization_phase_randomize(delta_seg, modes)
        elif method == "noise":
            null_delta_seg = generate_null_realization_noise(delta_seg, modes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute T_SSZ
        T_values[i] = compute_T_SSZ(null_delta_seg, modes, w_corr, w_scatter)
    
    # Compute statistics
    mean = np.mean(T_values)
    std = np.std(T_values)
    
    percentiles = {
        5: np.percentile(T_values, 5),
        25: np.percentile(T_values, 25),
        50: np.percentile(T_values, 50),
        75: np.percentile(T_values, 75),
        95: np.percentile(T_values, 95),
        99: np.percentile(T_values, 99),
    }
    
    return NullDistribution(
        T_values=T_values,
        mean=mean,
        std=std,
        percentiles=percentiles,
        n_realizations=n_realizations,
    )


def compute_p_value(
    T_observed: float,
    null_dist: NullDistribution,
) -> float:
    """
    Compute p-value for observed T_SSZ.
    
    P-value = P(T_null >= T_observed)
    
    Args:
        T_observed: Observed T_SSZ value
        null_dist: Null distribution
    
    Returns:
        P-value (one-sided, upper tail)
    """
    # Empirical p-value
    p_value = np.mean(null_dist.T_values >= T_observed)
    
    # Ensure p-value is not exactly 0 (use 1/n as minimum)
    if p_value == 0:
        p_value = 1.0 / (null_dist.n_realizations + 1)
    
    return p_value


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def test_ssz_signature(
    delta_seg: Dict[int, np.ndarray],
    modes: List[int] = [1, 2, 3],
    n_null_realizations: int = 1000,
    null_method: str = "shuffle",
    alpha: float = 0.05,
    w_corr: float = 0.7,
    w_scatter: float = 0.3,
) -> SSZTestResult:
    """
    Test for SSZ signature in delta_seg data.
    
    Args:
        delta_seg: Dictionary of delta_seg arrays per mode
        modes: List of modes
        n_null_realizations: Number of null realizations
        null_method: Method for null generation
        alpha: Significance level
        w_corr: Weight for correlation in T_SSZ
        w_scatter: Weight for scatter in T_SSZ
    
    Returns:
        SSZTestResult
    """
    # Compute observed T_SSZ
    T_observed = compute_T_SSZ(delta_seg, modes, w_corr, w_scatter)
    
    # Compute null distribution
    null_dist = compute_null_distribution(
        delta_seg, modes, n_null_realizations, null_method, w_corr, w_scatter
    )
    
    # Compute p-value
    p_value = compute_p_value(T_observed, null_dist)
    
    # Additional metrics
    mean_corr, _ = compute_mode_correlations(delta_seg, modes)
    scatter_ratio = compute_scatter_ratio(delta_seg, modes)
    
    return SSZTestResult(
        T_SSZ=T_observed,
        p_value=p_value,
        is_significant=p_value < alpha,
        correlation_mean=mean_corr,
        scatter_ratio=scatter_ratio,
        n_modes=len(modes),
        n_times=len(delta_seg[modes[0]]),
        metadata={
            "null_method": null_method,
            "n_null_realizations": n_null_realizations,
            "alpha": alpha,
            "null_mean": null_dist.mean,
            "null_std": null_dist.std,
            "null_95": null_dist.percentiles[95],
            "w_corr": w_corr,
            "w_scatter": w_scatter,
        }
    )


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(
    generate_data_func,
    amplitudes: np.ndarray,
    noise_level: float = 0.01,
    n_realizations: int = 100,
    n_null_realizations: int = 500,
    modes: List[int] = [1, 2, 3],
    alpha: float = 0.05,
) -> SensitivityResult:
    """
    Run sensitivity analysis to determine detection threshold.
    
    Args:
        generate_data_func: Function(amplitude, noise) -> (f_obs, delta_seg_true)
        amplitudes: Array of SSZ amplitudes to test
        noise_level: Noise level for data generation
        n_realizations: Number of realizations per amplitude
        n_null_realizations: Number of null realizations for p-value
        modes: List of modes
        alpha: Significance level
    
    Returns:
        SensitivityResult
    """
    n_amps = len(amplitudes)
    detection_rates = np.zeros(n_amps)
    mean_T_SSZ = np.zeros(n_amps)
    std_T_SSZ = np.zeros(n_amps)
    
    for i, amp in enumerate(amplitudes):
        T_values = []
        n_detected = 0
        
        for _ in range(n_realizations):
            # Generate data with this amplitude
            f_obs, delta_seg_true, delta_seg_reconstructed = generate_data_func(amp, noise_level)
            
            # Test for SSZ signature
            result = test_ssz_signature(
                delta_seg_reconstructed,
                modes=modes,
                n_null_realizations=n_null_realizations,
                alpha=alpha,
            )
            
            T_values.append(result.T_SSZ)
            if result.is_significant:
                n_detected += 1
        
        detection_rates[i] = n_detected / n_realizations
        mean_T_SSZ[i] = np.mean(T_values)
        std_T_SSZ[i] = np.std(T_values)
    
    # Find detection threshold (amplitude where detection rate >= 0.95)
    above_threshold = detection_rates >= 0.95
    if np.any(above_threshold):
        detection_threshold = amplitudes[np.argmax(above_threshold)]
    else:
        detection_threshold = np.inf
    
    return SensitivityResult(
        amplitudes=amplitudes,
        detection_rates=detection_rates,
        mean_T_SSZ=mean_T_SSZ,
        std_T_SSZ=std_T_SSZ,
        detection_threshold=detection_threshold,
        noise_level=noise_level,
        n_realizations=n_realizations,
    )


# =============================================================================
# REPORTING
# =============================================================================

def format_test_result(result: SSZTestResult) -> str:
    """Format SSZ test result as string."""
    lines = [
        "=" * 60,
        "SSZ DETECTION TEST RESULT",
        "=" * 60,
        f"Test statistic T_SSZ: {result.T_SSZ:.4f}",
        f"P-value: {result.p_value:.4f}",
        f"Significant at alpha={result.metadata['alpha']}: {result.is_significant}",
        "",
        "Components:",
        f"  Mean mode correlation: {result.correlation_mean:.4f}",
        f"  Scatter ratio: {result.scatter_ratio:.4f}",
        "",
        "Null distribution:",
        f"  Mean: {result.metadata['null_mean']:.4f}",
        f"  Std: {result.metadata['null_std']:.4f}",
        f"  95th percentile: {result.metadata['null_95']:.4f}",
        "",
        f"Data: {result.n_modes} modes, {result.n_times} time points",
        f"Null method: {result.metadata['null_method']}",
        f"Null realizations: {result.metadata['n_null_realizations']}",
        "=" * 60,
    ]
    
    return "\n".join(lines)


def format_sensitivity_result(result: SensitivityResult) -> str:
    """Format sensitivity analysis result as string."""
    lines = [
        "=" * 60,
        "SSZ SENSITIVITY ANALYSIS",
        "=" * 60,
        f"Noise level: {result.noise_level:.1%}",
        f"Detection threshold (95% power): {result.detection_threshold:.1%}",
        "",
        "Detection rates by amplitude:",
    ]
    
    for amp, rate in zip(result.amplitudes, result.detection_rates):
        lines.append(f"  {amp:.1%}: {rate:.1%} detected")
    
    lines.extend([
        "",
        f"Realizations per amplitude: {result.n_realizations}",
        "=" * 60,
    ])
    
    return "\n".join(lines)
