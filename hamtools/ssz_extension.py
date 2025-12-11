#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools SSZ Extension - Segmented Spacetime Expert Mode

Bridge between classical ham radio calculations and SSZ theory.

This module provides SSZ corrections for ham radio calculations,
using the same notation as SSZ_FORMULAS.md:

    D_SSZ = 1 + δ_seg
    c_eff = c / D_SSZ

The SSZ effect causes a slight modification to the effective
speed of light, which in turn affects wavelength calculations.

For typical δ_seg values (~1-3%), the effect is small but
potentially measurable with precise instrumentation.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import C_LIGHT, freq_to_lambda, freq_mhz_to_lambda


# =============================================================================
# SSZ CORE FUNCTIONS
# =============================================================================

def d_ssz_from_delta(delta_seg: float) -> float:
    """
    Calculate SSZ correction factor D_SSZ from segmentation parameter.
    
    Formula: D_SSZ = 1 + δ_seg
    
    This is consistent with SSZ_FORMULAS.md:
        "D_SSZ = 1 + delta_seg"
    
    Args:
        delta_seg: Segmentation parameter (dimensionless)
                   Typical values: 0.01 to 0.03 (1-3%)
    
    Returns:
        D_SSZ correction factor
    
    Example:
        >>> d_ssz_from_delta(0.01)  # 1% segmentation
        1.01
        >>> d_ssz_from_delta(0.02)  # 2% segmentation
        1.02
    
    Note:
        In SSZ theory, D_SSZ represents the time dilation factor
        due to spacetime segmentation. See SSZ_FORMULAS.md.
    """
    return 1.0 + delta_seg


def delta_seg_from_d_ssz(D_SSZ: float) -> float:
    """
    Calculate segmentation parameter from D_SSZ.
    
    Inverse of d_ssz_from_delta.
    
    Args:
        D_SSZ: SSZ correction factor
    
    Returns:
        Segmentation parameter δ_seg
    """
    return D_SSZ - 1.0


def effective_c_from_ssz(
    c_classical: float,
    delta_seg: float,
) -> float:
    """
    Calculate effective speed of light with SSZ correction.
    
    Formula: c_eff = c / D_SSZ = c / (1 + δ_seg)
    
    In SSZ theory, spacetime segmentation causes an effective
    reduction in the propagation speed of electromagnetic waves.
    
    Args:
        c_classical: Classical speed of light (m/s)
        delta_seg: Segmentation parameter
    
    Returns:
        Effective speed of light (m/s)
    
    Example:
        >>> effective_c_from_ssz(299792458, 0.01)
        296824215.8...  # ~1% slower
    
    Note:
        This is the same formula used in the Schumann resonance
        analysis. See ssz_schumann/models/ for full implementation.
    """
    D_SSZ = d_ssz_from_delta(delta_seg)
    return c_classical / D_SSZ


def ssz_corrected_lambda(
    f_hz: float,
    delta_seg: float,
) -> float:
    """
    Calculate SSZ-corrected wavelength.
    
    Formula: λ' = c_eff / f = c / (f × D_SSZ)
    
    The SSZ effect shortens the effective wavelength because
    the effective speed of light is reduced.
    
    Args:
        f_hz: Frequency in Hz
        delta_seg: Segmentation parameter
    
    Returns:
        SSZ-corrected wavelength in meters
    
    Example:
        >>> ssz_corrected_lambda(7.1e6, 0.01)  # 7.1 MHz, 1% SSZ
        41.80...  # vs 42.22 m classical
    """
    c_eff = effective_c_from_ssz(C_LIGHT, delta_seg)
    return c_eff / f_hz


def ssz_corrected_skip_distance(
    skip_distance_km: float,
    delta_seg: float,
) -> float:
    """
    Calculate SSZ-corrected skip distance.
    
    In SSZ theory, the effective path length is modified by
    the time dilation factor. This is a simplified model.
    
    Formula: d' = d × D_SSZ
    
    Interpretation: The signal travels through "more" spacetime
    segments, effectively increasing the path length.
    
    Args:
        skip_distance_km: Classical skip distance in km
        delta_seg: Segmentation parameter
    
    Returns:
        SSZ-corrected skip distance in km
    
    Note:
        This is a toy model for demonstration. The actual SSZ
        effect on ionospheric propagation would be more complex.
    """
    D_SSZ = d_ssz_from_delta(delta_seg)
    return skip_distance_km * D_SSZ


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

@dataclass
class SSZComparisonResult:
    """Result of SSZ vs classical comparison."""
    parameter_name: str
    classical_value: float
    ssz_value: float
    delta_seg: float
    difference_absolute: float
    difference_percent: float
    unit: str
    
    def __str__(self) -> str:
        sign = "+" if self.difference_percent >= 0 else ""
        return (
            f"SSZ Comparison: {self.parameter_name}\n"
            f"{'='*50}\n"
            f"δ_seg (segmentation): {self.delta_seg:.4f} ({self.delta_seg*100:.2f}%)\n"
            f"D_SSZ:                {1 + self.delta_seg:.6f}\n"
            f"\n"
            f"Classical {self.parameter_name}: {self.classical_value:.4f} {self.unit}\n"
            f"SSZ {self.parameter_name}:       {self.ssz_value:.4f} {self.unit}\n"
            f"Difference:           {sign}{self.difference_percent:.4f}%\n"
            f"\n"
            f"Interpretation:\n"
            f"  SSZ segmentation {'reduces' if self.difference_percent < 0 else 'increases'} "
            f"the effective {self.parameter_name.lower()}\n"
            f"  by {abs(self.difference_percent):.4f}% due to modified c_eff."
        )


@dataclass
class LambdaComparisonResult(SSZComparisonResult):
    """Specialized result for wavelength comparison."""
    frequency_mhz: float = 0.0
    c_classical: float = C_LIGHT
    c_eff: float = 0.0
    
    def __str__(self) -> str:
        base = super().__str__()
        return (
            f"{base}\n"
            f"\n"
            f"Details:\n"
            f"  Frequency:    {self.frequency_mhz:.4f} MHz\n"
            f"  c (vacuum):   {self.c_classical:.0f} m/s\n"
            f"  c_eff (SSZ):  {self.c_eff:.0f} m/s"
        )


def compare_lambda_with_ssz(
    f_mhz: float,
    delta_seg: float,
) -> LambdaComparisonResult:
    """
    Compare classical and SSZ-corrected wavelength.
    
    Args:
        f_mhz: Frequency in MHz
        delta_seg: Segmentation parameter
    
    Returns:
        LambdaComparisonResult with both values and comparison
    
    Example:
        >>> result = compare_lambda_with_ssz(7.1, 0.01)
        >>> print(result.classical_lambda_m)
        42.2242...
        >>> print(result.ssz_lambda_m)
        41.8061...
        >>> print(result.difference_percent)
        -0.99...
    """
    f_hz = f_mhz * 1e6
    
    # Classical
    lambda_classical = freq_to_lambda(f_hz)
    
    # SSZ
    lambda_ssz = ssz_corrected_lambda(f_hz, delta_seg)
    c_eff = effective_c_from_ssz(C_LIGHT, delta_seg)
    
    # Difference
    diff_abs = lambda_ssz - lambda_classical
    diff_pct = (diff_abs / lambda_classical) * 100
    
    return LambdaComparisonResult(
        parameter_name="Wavelength",
        classical_value=lambda_classical,
        ssz_value=lambda_ssz,
        delta_seg=delta_seg,
        difference_absolute=diff_abs,
        difference_percent=diff_pct,
        unit="m",
        frequency_mhz=f_mhz,
        c_classical=C_LIGHT,
        c_eff=c_eff,
    )


def compare_skip_with_ssz(
    skip_km: float,
    delta_seg: float,
) -> SSZComparisonResult:
    """
    Compare classical and SSZ-corrected skip distance.
    
    Args:
        skip_km: Classical skip distance in km
        delta_seg: Segmentation parameter
    
    Returns:
        SSZComparisonResult with both values
    """
    skip_ssz = ssz_corrected_skip_distance(skip_km, delta_seg)
    
    diff_abs = skip_ssz - skip_km
    diff_pct = (diff_abs / skip_km) * 100
    
    return SSZComparisonResult(
        parameter_name="Skip Distance",
        classical_value=skip_km,
        ssz_value=skip_ssz,
        delta_seg=delta_seg,
        difference_absolute=diff_abs,
        difference_percent=diff_pct,
        unit="km",
    )


def compare_antenna_length_with_ssz(
    f_mhz: float,
    delta_seg: float,
    antenna_type: str = "dipole",
    k: float = 0.95,
) -> SSZComparisonResult:
    """
    Compare classical and SSZ-corrected antenna length.
    
    Args:
        f_mhz: Frequency in MHz
        delta_seg: Segmentation parameter
        antenna_type: "dipole" (λ/2) or "vertical" (λ/4)
        k: Shortening factor
    
    Returns:
        SSZComparisonResult with both lengths
    """
    # Classical wavelength
    lambda_classical = freq_mhz_to_lambda(f_mhz)
    
    # SSZ wavelength
    lambda_ssz = ssz_corrected_lambda(f_mhz * 1e6, delta_seg)
    
    # Antenna lengths
    if antenna_type.lower() == "dipole":
        factor = 0.5
        name = "Dipole Length (λ/2)"
    elif antenna_type.lower() == "vertical":
        factor = 0.25
        name = "Vertical Length (λ/4)"
    else:
        factor = 0.5
        name = f"Antenna Length ({antenna_type})"
    
    len_classical = lambda_classical * factor * k
    len_ssz = lambda_ssz * factor * k
    
    diff_abs = len_ssz - len_classical
    diff_pct = (diff_abs / len_classical) * 100
    
    return SSZComparisonResult(
        parameter_name=name,
        classical_value=len_classical,
        ssz_value=len_ssz,
        delta_seg=delta_seg,
        difference_absolute=diff_abs,
        difference_percent=diff_pct,
        unit="m",
    )


# =============================================================================
# SSZ PARAMETER ESTIMATION
# =============================================================================

def estimate_delta_seg_from_schumann(
    f1_observed: float,
    f1_classical: float = 7.83,
) -> float:
    """
    Estimate δ_seg from Schumann resonance deviation.
    
    If the observed Schumann fundamental differs from classical,
    we can estimate the SSZ segmentation parameter.
    
    Formula: δ_seg ≈ (f_classical / f_observed) - 1
    
    Args:
        f1_observed: Observed Schumann f1 in Hz
        f1_classical: Classical f1 (default 7.83 Hz)
    
    Returns:
        Estimated δ_seg
    
    Note:
        This is a simplified estimate. Real Schumann variations
        have many classical causes (ionospheric height, etc.).
        See FINAL_REPORT.md for full analysis.
    """
    return (f1_classical / f1_observed) - 1


def typical_ssz_values() -> dict:
    """
    Return typical SSZ parameter values for reference.
    
    Returns:
        Dictionary of typical values and their interpretations
    """
    return {
        "minimal": {
            "delta_seg": 0.001,
            "description": "0.1% - Barely detectable",
        },
        "small": {
            "delta_seg": 0.01,
            "description": "1% - Small but measurable effect",
        },
        "moderate": {
            "delta_seg": 0.02,
            "description": "2% - Moderate SSZ signature",
        },
        "strong": {
            "delta_seg": 0.03,
            "description": "3% - Strong SSZ signature",
        },
        "schumann_typical": {
            "delta_seg": 0.02,
            "description": "~2% - Typical value from Schumann analysis",
        },
    }


# =============================================================================
# FORMATTED OUTPUT
# =============================================================================

def format_ssz_comparison(
    f_mhz: float,
    delta_seg: float,
) -> str:
    """
    Format a complete SSZ comparison for CLI output.
    
    Args:
        f_mhz: Frequency in MHz
        delta_seg: Segmentation parameter
    
    Returns:
        Formatted string for display
    """
    lambda_result = compare_lambda_with_ssz(f_mhz, delta_seg)
    
    lines = [
        "=" * 55,
        f"SSZ EXPERT MODE - Frequency Analysis",
        "=" * 55,
        f"",
        f"Input:",
        f"  Frequency:      {f_mhz:.4f} MHz",
        f"  δ_seg:          {delta_seg:.4f} ({delta_seg*100:.2f}%)",
        f"",
        f"SSZ Parameters:",
        f"  D_SSZ:          {1 + delta_seg:.6f}",
        f"  c (vacuum):     {C_LIGHT:,.0f} m/s",
        f"  c_eff (SSZ):    {lambda_result.c_eff:,.0f} m/s",
        f"",
        f"Wavelength Comparison:",
        f"  Classical λ:    {lambda_result.classical_value:.4f} m",
        f"  SSZ λ:          {lambda_result.ssz_value:.4f} m",
        f"  Difference:     {lambda_result.difference_percent:+.4f}%",
        f"",
        f"Interpretation:",
        f"  SSZ segmentation slightly reduces the effective wave speed,",
        f"  which shortens the apparent wavelength by ~{abs(lambda_result.difference_percent):.2f}%.",
        f"",
        f"  For antenna design, this means a {abs(lambda_result.difference_percent):.2f}% shorter",
        f"  element would be optimal in an SSZ-modified environment.",
        "=" * 55,
    ]
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("=== HamTools SSZ Extension Test ===\n")
    
    # Basic SSZ functions
    print(f"D_SSZ for δ_seg=0.01: {d_ssz_from_delta(0.01)}")
    print(f"c_eff for δ_seg=0.01: {effective_c_from_ssz(C_LIGHT, 0.01):,.0f} m/s")
    print()
    
    # Wavelength comparison
    result = compare_lambda_with_ssz(7.1, 0.01)
    print(result)
    print()
    
    # Formatted output
    print(format_ssz_comparison(14.2, 0.02))
