#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4: SSZ Signature Diagnostics

Helper functions for analyzing and visualizing SSZ signatures
in Schumann resonance data.

Key diagnostics:
1. Relative frequency shifts (should be mode-independent for SSZ)
2. Delta_seg time series with confidence bands
3. Mode consistency checks
4. Classical vs SSZ dispersion patterns

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
class RelativeShiftResult:
    """Result of relative frequency shift analysis."""
    mode: int
    mean_shift: float
    std_shift: float
    shift_series: pd.Series
    
    def __str__(self):
        return (
            f"Mode {self.mode}: "
            f"mean = {self.mean_shift*100:.4f}%, "
            f"std = {self.std_shift*100:.4f}%"
        )


@dataclass
class ModeIndependenceResult:
    """Result of mode independence check."""
    is_mode_independent: bool
    max_deviation: float
    mean_shifts: Dict[int, float]
    std_shifts: Dict[int, float]
    chi_squared: float
    p_value: float
    interpretation: str


def compute_relative_shifts(
    f_obs: Dict[int, pd.Series],
    f_classical: Dict[int, float],
) -> Dict[int, RelativeShiftResult]:
    """
    Compute relative frequency shifts for each mode.
    
    The relative shift is defined as:
        delta_f_n / f_n = (f_obs - f_classical) / f_classical
    
    For SSZ, this should be approximately:
        delta_f_n / f_n = -delta_seg  (SAME for all modes)
    
    Args:
        f_obs: Dictionary {mode: observed_frequency_series}
        f_classical: Dictionary {mode: classical_frequency}
    
    Returns:
        Dictionary {mode: RelativeShiftResult}
    
    Example:
        >>> shifts = compute_relative_shifts(
        ...     {1: f1_obs, 2: f2_obs, 3: f3_obs},
        ...     {1: 7.83, 2: 13.9, 3: 20.3}
        ... )
        >>> for mode, result in shifts.items():
        ...     print(result)
    """
    results = {}
    
    for mode in f_obs.keys():
        if mode not in f_classical:
            continue
        
        f_o = f_obs[mode]
        f_c = f_classical[mode]
        
        # Relative shift: (f_obs - f_classical) / f_classical
        shift = (f_o - f_c) / f_c
        
        results[mode] = RelativeShiftResult(
            mode=mode,
            mean_shift=float(np.nanmean(shift)),
            std_shift=float(np.nanstd(shift)),
            shift_series=shift,
        )
    
    return results


def check_mode_independence(
    shifts: Dict[int, RelativeShiftResult],
    tolerance: float = 0.01,
) -> ModeIndependenceResult:
    """
    Check if relative shifts are mode-independent (SSZ signature).
    
    The SSZ signature is a UNIFORM relative shift across all modes:
        delta_f_n / f_n = -delta_seg  (same for all n)
    
    Classical dispersive effects would show mode-dependent shifts.
    
    Args:
        shifts: Dictionary of RelativeShiftResult from compute_relative_shifts
        tolerance: Maximum allowed deviation between modes
    
    Returns:
        ModeIndependenceResult with interpretation
    
    Interpretation:
        - is_mode_independent = True: Consistent with SSZ
        - is_mode_independent = False: Suggests classical dispersion
    """
    modes = sorted(shifts.keys())
    
    if len(modes) < 2:
        return ModeIndependenceResult(
            is_mode_independent=True,
            max_deviation=0.0,
            mean_shifts={m: shifts[m].mean_shift for m in modes},
            std_shifts={m: shifts[m].std_shift for m in modes},
            chi_squared=0.0,
            p_value=1.0,
            interpretation="Insufficient modes for comparison",
        )
    
    # Extract mean shifts
    mean_shifts = {m: shifts[m].mean_shift for m in modes}
    std_shifts = {m: shifts[m].std_shift for m in modes}
    
    # Compute grand mean
    grand_mean = np.mean(list(mean_shifts.values()))
    
    # Maximum deviation from grand mean
    deviations = [abs(mean_shifts[m] - grand_mean) for m in modes]
    max_deviation = max(deviations)
    
    # Chi-squared test for consistency
    # H0: All modes have the same true relative shift
    chi_sq = 0.0
    for m in modes:
        if std_shifts[m] > 0:
            chi_sq += ((mean_shifts[m] - grand_mean) / std_shifts[m])**2
    
    # Degrees of freedom = number of modes - 1
    dof = len(modes) - 1
    
    # P-value (using scipy if available, otherwise approximate)
    try:
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi_sq, dof)
    except ImportError:
        # Rough approximation
        p_value = np.exp(-chi_sq / 2) if chi_sq > 0 else 1.0
    
    # Interpretation
    is_independent = max_deviation < tolerance
    
    if is_independent:
        interpretation = (
            f"Mode-independent shifts (max deviation {max_deviation*100:.3f}% < "
            f"{tolerance*100:.1f}% tolerance). CONSISTENT WITH SSZ SIGNATURE."
        )
    else:
        interpretation = (
            f"Mode-dependent shifts detected (max deviation {max_deviation*100:.3f}% > "
            f"{tolerance*100:.1f}% tolerance). Suggests CLASSICAL DISPERSION."
        )
    
    return ModeIndependenceResult(
        is_mode_independent=is_independent,
        max_deviation=max_deviation,
        mean_shifts=mean_shifts,
        std_shifts=std_shifts,
        chi_squared=chi_sq,
        p_value=p_value,
        interpretation=interpretation,
    )


def compute_delta_seg_with_confidence(
    f_obs: Dict[int, pd.Series],
    f_classical: Dict[int, float],
    confidence_level: float = 0.95,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute delta_seg time series with confidence bands.
    
    Combines delta_seg estimates from all modes and computes
    confidence intervals.
    
    Args:
        f_obs: Dictionary {mode: observed_frequency_series}
        f_classical: Dictionary {mode: classical_frequency}
        confidence_level: Confidence level for bands (default 95%)
    
    Returns:
        Tuple of (delta_seg_mean, delta_seg_lower, delta_seg_upper)
    
    Example:
        >>> mean, lower, upper = compute_delta_seg_with_confidence(
        ...     {1: f1_obs, 2: f2_obs, 3: f3_obs},
        ...     {1: 7.83, 2: 13.9, 3: 20.3}
        ... )
        >>> # Plot with confidence band
        >>> plt.fill_between(mean.index, lower, upper, alpha=0.3)
        >>> plt.plot(mean.index, mean)
    """
    # Compute delta_seg for each mode
    delta_seg_modes = []
    
    for mode in f_obs.keys():
        if mode not in f_classical:
            continue
        
        f_o = f_obs[mode]
        f_c = f_classical[mode]
        
        # delta_seg = f_classical / f_obs - 1
        delta_seg = f_c / f_o - 1
        delta_seg_modes.append(delta_seg)
    
    # Stack into DataFrame
    df = pd.concat(delta_seg_modes, axis=1)
    
    # Mean across modes
    delta_seg_mean = df.mean(axis=1)
    
    # Standard error
    n_modes = len(delta_seg_modes)
    std_across_modes = df.std(axis=1)
    se = std_across_modes / np.sqrt(n_modes)
    
    # Confidence interval
    try:
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
    except ImportError:
        # Approximate z for 95% CI
        z = 1.96 if confidence_level == 0.95 else 2.0
    
    delta_seg_lower = delta_seg_mean - z * se
    delta_seg_upper = delta_seg_mean + z * se
    
    return delta_seg_mean, delta_seg_lower, delta_seg_upper


def detect_dispersion_pattern(
    f_obs: Dict[int, pd.Series],
    f_classical: Dict[int, float],
) -> Dict[str, float]:
    """
    Detect classical dispersion patterns in frequency data.
    
    Classical dispersion would show:
    - Mode-dependent relative shifts
    - Systematic trend with mode number
    
    SSZ would show:
    - Mode-independent relative shifts
    - No systematic trend
    
    Args:
        f_obs: Dictionary {mode: observed_frequency_series}
        f_classical: Dictionary {mode: classical_frequency}
    
    Returns:
        Dictionary with dispersion metrics:
            - slope: Trend of relative shift with mode number
            - r_squared: How well the trend fits
            - is_dispersive: True if significant dispersion detected
    """
    modes = sorted(f_obs.keys())
    
    # Compute mean relative shifts
    mean_shifts = []
    for mode in modes:
        f_o = f_obs[mode]
        f_c = f_classical[mode]
        shift = float(np.nanmean((f_o - f_c) / f_c))
        mean_shifts.append(shift)
    
    # Linear regression: shift vs mode number
    x = np.array(modes)
    y = np.array(mean_shifts)
    
    # Fit line
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        
        # R-squared
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope = 0.0
        intercept = np.mean(y) if len(y) > 0 else 0.0
        r_squared = 0.0
    
    # Is dispersion significant?
    # Threshold: slope > 0.1% per mode AND R² > 0.5
    is_dispersive = abs(slope) > 0.001 and r_squared > 0.5
    
    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "is_dispersive": is_dispersive,
        "interpretation": (
            "Classical dispersion detected" if is_dispersive 
            else "No significant dispersion (consistent with SSZ)"
        ),
    }


def generate_diagnostic_report(
    f_obs: Dict[int, pd.Series],
    f_classical: Dict[int, float],
    delta_seg: Optional[pd.Series] = None,
) -> str:
    """
    Generate comprehensive diagnostic report.
    
    Args:
        f_obs: Dictionary {mode: observed_frequency_series}
        f_classical: Dictionary {mode: classical_frequency}
        delta_seg: Optional pre-computed delta_seg series
    
    Returns:
        Formatted diagnostic report string
    """
    lines = [
        "=" * 60,
        "SSZ SIGNATURE DIAGNOSTIC REPORT",
        "=" * 60,
        "",
    ]
    
    # 1. Relative shifts
    lines.append("1. RELATIVE FREQUENCY SHIFTS")
    lines.append("-" * 40)
    
    shifts = compute_relative_shifts(f_obs, f_classical)
    for mode, result in sorted(shifts.items()):
        lines.append(f"   {result}")
    
    # 2. Mode independence
    lines.append("")
    lines.append("2. MODE INDEPENDENCE CHECK")
    lines.append("-" * 40)
    
    independence = check_mode_independence(shifts)
    lines.append(f"   Max deviation: {independence.max_deviation*100:.4f}%")
    lines.append(f"   Chi-squared: {independence.chi_squared:.2f}")
    lines.append(f"   P-value: {independence.p_value:.4f}")
    lines.append(f"   Result: {independence.interpretation}")
    
    # 3. Dispersion analysis
    lines.append("")
    lines.append("3. DISPERSION ANALYSIS")
    lines.append("-" * 40)
    
    dispersion = detect_dispersion_pattern(f_obs, f_classical)
    lines.append(f"   Slope: {dispersion['slope']*100:.4f}% per mode")
    lines.append(f"   R-squared: {dispersion['r_squared']:.4f}")
    lines.append(f"   Result: {dispersion['interpretation']}")
    
    # 4. Delta_seg statistics
    if delta_seg is not None:
        lines.append("")
        lines.append("4. DELTA_SEG STATISTICS")
        lines.append("-" * 40)
        lines.append(f"   Mean: {np.nanmean(delta_seg)*100:.4f}%")
        lines.append(f"   Std: {np.nanstd(delta_seg)*100:.4f}%")
        lines.append(f"   Min: {np.nanmin(delta_seg)*100:.4f}%")
        lines.append(f"   Max: {np.nanmax(delta_seg)*100:.4f}%")
    
    # 5. Overall interpretation
    lines.append("")
    lines.append("5. OVERALL INTERPRETATION")
    lines.append("-" * 40)
    
    if independence.is_mode_independent and not dispersion["is_dispersive"]:
        lines.append("   CONSISTENT WITH SSZ SIGNATURE")
        lines.append("   - Relative shifts are mode-independent")
        lines.append("   - No classical dispersion detected")
    elif not independence.is_mode_independent:
        lines.append("   INCONSISTENT WITH SSZ")
        lines.append("   - Mode-dependent shifts detected")
        lines.append("   - Suggests classical effects dominate")
    else:
        lines.append("   AMBIGUOUS RESULT")
        lines.append("   - Further analysis needed")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("=== SSZ Diagnostics Test ===\n")
    
    # Generate test data
    np.random.seed(42)
    n_points = 100
    time = pd.date_range("2016-01-01", periods=n_points, freq="1h")
    
    # True delta_seg
    delta_seg_true = 0.01 * np.sin(2 * np.pi * np.arange(n_points) / 24)
    
    # Classical frequencies
    f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
    
    # Observed frequencies (with SSZ correction)
    f_obs = {}
    for mode, f_c in f_classical.items():
        noise = 0.01 * f_c * np.random.randn(n_points)
        f_obs[mode] = pd.Series(
            f_c / (1 + delta_seg_true) + noise,
            index=time
        )
    
    # Generate report
    report = generate_diagnostic_report(f_obs, f_classical)
    print(report)
