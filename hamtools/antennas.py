#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Antennas Module - Antenna Calculations

Calculations for common amateur radio antennas:
- Half-wave dipole (λ/2)
- Quarter-wave vertical (λ/4)
- Yagi-Uda antenna gain estimation

Based on concepts from Antennen_Basics.md and Glossar_Funktechnik.md.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Optional

from .core import C_LIGHT, freq_mhz_to_lambda


# =============================================================================
# ANTENNA CONSTANTS
# =============================================================================

# Typical velocity factors (shortening factors)
VELOCITY_FACTOR_WIRE = 0.95      # Bare wire in free space
VELOCITY_FACTOR_INSULATED = 0.93  # Insulated wire
VELOCITY_FACTOR_ALUMINUM = 0.97   # Aluminum tubing

# Dipole gain relative to isotropic
DIPOLE_GAIN_DBI = 2.15


# =============================================================================
# DIPOLE CALCULATIONS
# =============================================================================

def dipole_length_halfwave(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate half-wave dipole length.
    
    Formula: L = (λ/2) × k = (c / f) × 0.5 × k
    
    The shortening factor k accounts for:
    - End effects (capacitance at wire ends)
    - Wire diameter effects
    - Insulation (if present)
    
    Typical values:
    - Bare wire: k ≈ 0.95
    - Insulated wire: k ≈ 0.93
    - Aluminum tubing: k ≈ 0.97
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor (velocity factor), default 0.95
    
    Returns:
        Total dipole length in meters
    
    Example:
        >>> dipole_length_halfwave(7.1)  # 40m band
        20.06...
        >>> dipole_length_halfwave(14.2)  # 20m band
        10.03...
    
    Note:
        Each leg of the dipole is half this length.
        See Antennen_Basics.md for theory.
    """
    if f_mhz <= 0:
        raise ValueError(f"Frequency must be positive, got {f_mhz}")
    if not 0.5 <= k <= 1.0:
        raise ValueError(f"Shortening factor k should be 0.5-1.0, got {k}")
    
    wavelength = freq_mhz_to_lambda(f_mhz)
    return (wavelength / 2) * k


def dipole_length_each_leg(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate length of each dipole leg (half of total length).
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        Length of one dipole leg in meters
    """
    return dipole_length_halfwave(f_mhz, k) / 2


def dipole_length_feet(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate half-wave dipole length in feet.
    
    Common approximation: L(ft) ≈ 468 / f(MHz)
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        Total dipole length in feet
    """
    return dipole_length_halfwave(f_mhz, k) * 3.28084


# =============================================================================
# VERTICAL ANTENNA CALCULATIONS
# =============================================================================

def vertical_quarterwave(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate quarter-wave vertical antenna length.
    
    Formula: L = (λ/4) × k
    
    A quarter-wave vertical requires a ground plane (radials)
    for proper operation.
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        Vertical element length in meters
    
    Example:
        >>> vertical_quarterwave(7.1)  # 40m band
        10.03...
        >>> vertical_quarterwave(14.2)  # 20m band
        5.01...
    
    Note:
        Radials should typically be λ/4 or longer.
        See Antennen_Basics.md for ground plane requirements.
    """
    if f_mhz <= 0:
        raise ValueError(f"Frequency must be positive, got {f_mhz}")
    
    wavelength = freq_mhz_to_lambda(f_mhz)
    return (wavelength / 4) * k


def vertical_5_8_wave(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate 5/8-wave vertical antenna length.
    
    5/8λ verticals have ~3dB more gain than 1/4λ verticals
    but require a matching network.
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        Vertical element length in meters
    """
    wavelength = freq_mhz_to_lambda(f_mhz)
    return (wavelength * 5 / 8) * k


def radial_length(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> float:
    """
    Calculate recommended radial length for vertical antenna.
    
    Standard recommendation: λ/4 radials.
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        Radial length in meters
    """
    return vertical_quarterwave(f_mhz, k)


# =============================================================================
# YAGI-UDA ANTENNA CALCULATIONS
# =============================================================================

def estimate_yagi_gain(
    num_elements: int,
    boom_length_m: float,
    wavelength_m: Optional[float] = None,
) -> float:
    """
    Estimate Yagi-Uda antenna gain.
    
    This is a rough approximation based on empirical data.
    Actual gain depends on element spacing, lengths, and tuning.
    
    Approximation formula:
        G(dBd) ≈ 10 × log₁₀(n × L_boom/λ) + 2
    
    where:
        n = number of elements
        L_boom = boom length
        λ = wavelength
    
    Typical gains:
        - 3 elements: ~6 dBd
        - 5 elements: ~9 dBd
        - 7 elements: ~11 dBd
    
    Args:
        num_elements: Number of elements (driven + directors + reflector)
        boom_length_m: Boom length in meters
        wavelength_m: Wavelength in meters (optional, for better estimate)
    
    Returns:
        Estimated gain in dBd (relative to dipole)
    
    Example:
        >>> estimate_yagi_gain(5, 6.5)  # 5-element Yagi
        ~9 dBd
    
    Note:
        This is an approximation. For accurate gain, use antenna
        modeling software (e.g., EZNEC, 4NEC2).
    """
    if num_elements < 2:
        raise ValueError("Yagi needs at least 2 elements")
    if boom_length_m <= 0:
        raise ValueError("Boom length must be positive")
    
    # Simple empirical formula
    # Based on: gain increases ~3dB per doubling of boom length
    
    if wavelength_m is not None and wavelength_m > 0:
        # More accurate estimate using boom length in wavelengths
        boom_wavelengths = boom_length_m / wavelength_m
        gain_dbd = 10 * math.log10(num_elements * boom_wavelengths) + 2
    else:
        # Simpler estimate based on element count
        # Typical: 3-el ≈ 6dBd, 5-el ≈ 9dBd, 7-el ≈ 11dBd
        gain_dbd = 3 + 2.5 * math.log10(num_elements)
    
    return max(0, gain_dbd)  # Gain can't be negative


def yagi_element_spacing(
    f_mhz: float,
    element_type: str = "director",
) -> float:
    """
    Estimate typical Yagi element spacing.
    
    Typical spacings (as fraction of wavelength):
    - Reflector to driven: 0.15-0.25λ
    - Driven to first director: 0.1-0.15λ
    - Director to director: 0.2-0.35λ
    
    Args:
        f_mhz: Frequency in MHz
        element_type: "reflector", "director", or "driven"
    
    Returns:
        Typical spacing in meters
    """
    wavelength = freq_mhz_to_lambda(f_mhz)
    
    spacings = {
        "reflector": 0.2,   # Reflector to driven
        "driven": 0.12,     # Driven to first director
        "director": 0.25,   # Director to director
    }
    
    fraction = spacings.get(element_type.lower(), 0.2)
    return wavelength * fraction


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class DipoleResult:
    """Result of dipole calculation."""
    frequency_mhz: float
    total_length_m: float
    leg_length_m: float
    shortening_factor: float
    wavelength_m: float
    
    def __str__(self) -> str:
        return (
            f"Half-Wave Dipole for {self.frequency_mhz:.3f} MHz\n"
            f"{'='*40}\n"
            f"Wavelength:        {self.wavelength_m:.2f} m\n"
            f"Shortening factor: {self.shortening_factor:.2f}\n"
            f"Total length:      {self.total_length_m:.2f} m\n"
            f"Each leg:          {self.leg_length_m:.2f} m\n"
            f"Total (feet):      {self.total_length_m * 3.28084:.1f} ft"
        )


@dataclass
class VerticalResult:
    """Result of vertical antenna calculation."""
    frequency_mhz: float
    element_length_m: float
    radial_length_m: float
    shortening_factor: float
    wavelength_m: float
    
    def __str__(self) -> str:
        return (
            f"Quarter-Wave Vertical for {self.frequency_mhz:.3f} MHz\n"
            f"{'='*40}\n"
            f"Wavelength:        {self.wavelength_m:.2f} m\n"
            f"Shortening factor: {self.shortening_factor:.2f}\n"
            f"Element length:    {self.element_length_m:.2f} m\n"
            f"Radial length:     {self.radial_length_m:.2f} m\n"
            f"Element (feet):    {self.element_length_m * 3.28084:.1f} ft"
        )


@dataclass
class YagiResult:
    """Result of Yagi antenna calculation."""
    frequency_mhz: float
    num_elements: int
    boom_length_m: float
    estimated_gain_dbd: float
    estimated_gain_dbi: float
    wavelength_m: float
    
    def __str__(self) -> str:
        return (
            f"{self.num_elements}-Element Yagi for {self.frequency_mhz:.3f} MHz\n"
            f"{'='*40}\n"
            f"Wavelength:     {self.wavelength_m:.2f} m\n"
            f"Boom length:    {self.boom_length_m:.2f} m ({self.boom_length_m/self.wavelength_m:.2f}λ)\n"
            f"Est. gain:      {self.estimated_gain_dbd:.1f} dBd ({self.estimated_gain_dbi:.1f} dBi)\n"
            f"Note: Actual gain depends on design optimization."
        )


def calculate_dipole(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> DipoleResult:
    """
    Calculate dipole antenna dimensions.
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        DipoleResult with all dimensions
    """
    wavelength = freq_mhz_to_lambda(f_mhz)
    total = dipole_length_halfwave(f_mhz, k)
    
    return DipoleResult(
        frequency_mhz=f_mhz,
        total_length_m=total,
        leg_length_m=total / 2,
        shortening_factor=k,
        wavelength_m=wavelength,
    )


def calculate_vertical(
    f_mhz: float,
    k: float = VELOCITY_FACTOR_WIRE,
) -> VerticalResult:
    """
    Calculate vertical antenna dimensions.
    
    Args:
        f_mhz: Frequency in MHz
        k: Shortening factor
    
    Returns:
        VerticalResult with all dimensions
    """
    wavelength = freq_mhz_to_lambda(f_mhz)
    element = vertical_quarterwave(f_mhz, k)
    radial = radial_length(f_mhz, k)
    
    return VerticalResult(
        frequency_mhz=f_mhz,
        element_length_m=element,
        radial_length_m=radial,
        shortening_factor=k,
        wavelength_m=wavelength,
    )


def calculate_yagi(
    f_mhz: float,
    num_elements: int,
    boom_length_m: float,
) -> YagiResult:
    """
    Calculate Yagi antenna parameters.
    
    Args:
        f_mhz: Frequency in MHz
        num_elements: Number of elements
        boom_length_m: Boom length in meters
    
    Returns:
        YagiResult with estimated parameters
    """
    wavelength = freq_mhz_to_lambda(f_mhz)
    gain_dbd = estimate_yagi_gain(num_elements, boom_length_m, wavelength)
    
    return YagiResult(
        frequency_mhz=f_mhz,
        num_elements=num_elements,
        boom_length_m=boom_length_m,
        estimated_gain_dbd=gain_dbd,
        estimated_gain_dbi=gain_dbd + DIPOLE_GAIN_DBI,
        wavelength_m=wavelength,
    )


if __name__ == "__main__":
    # Quick test
    print("=== HamTools Antennas Test ===\n")
    
    # Dipole
    print(calculate_dipole(7.1))
    print()
    
    # Vertical
    print(calculate_vertical(14.2))
    print()
    
    # Yagi
    print(calculate_yagi(14.2, 5, 6.5))
