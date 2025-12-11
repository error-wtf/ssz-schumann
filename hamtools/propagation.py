#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Propagation Module - HF Propagation Calculations

Simple calculations for HF propagation:
- Critical frequency (foF2)
- Maximum Usable Frequency (MUF)
- Skip distance estimation

Based on concepts from Wellen_Ionosphaere.md.

Note: These are simplified models for educational purposes.
For accurate propagation prediction, use professional tools
like VOACAP, PropLab, or real-time ionospheric data.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Physical constants
EARTH_RADIUS_KM = 6371.0  # Mean Earth radius


# =============================================================================
# IONOSPHERIC CALCULATIONS
# =============================================================================

def critical_freq_fof2(N_max: float) -> float:
    """
    Calculate critical frequency (foF2) from peak electron density.
    
    The critical frequency is the highest frequency that will be
    reflected by the ionosphere at vertical incidence.
    
    Formula: foF2 = 9 × √(N_max)
    
    where N_max is in electrons/m³ and foF2 is in Hz.
    
    Args:
        N_max: Peak electron density in electrons/m³
               Typical values: 10^11 to 10^12 e/m³
    
    Returns:
        Critical frequency in MHz
    
    Example:
        >>> critical_freq_fof2(1e12)  # High solar activity
        9.0 MHz
        >>> critical_freq_fof2(2.5e11)  # Low solar activity
        4.5 MHz
    
    Note:
        See Wellen_Ionosphaere.md for ionospheric layer theory.
    """
    if N_max <= 0:
        raise ValueError(f"Electron density must be positive, got {N_max}")
    
    # Formula: f = 9 * sqrt(N) Hz, convert to MHz
    f_hz = 9.0 * math.sqrt(N_max)
    return f_hz / 1e6


def electron_density_from_fof2(fof2_mhz: float) -> float:
    """
    Calculate peak electron density from critical frequency.
    
    Inverse of critical_freq_fof2.
    
    Args:
        fof2_mhz: Critical frequency in MHz
    
    Returns:
        Peak electron density in electrons/m³
    """
    if fof2_mhz <= 0:
        raise ValueError(f"Frequency must be positive, got {fof2_mhz}")
    
    f_hz = fof2_mhz * 1e6
    return (f_hz / 9.0) ** 2


def muf_single_hop(
    fof2_mhz: float,
    path_distance_km: float,
    virtual_height_km: float = 300.0,
) -> float:
    """
    Calculate Maximum Usable Frequency (MUF) for single-hop propagation.
    
    Uses the secant law approximation:
        MUF = foF2 × sec(θ)
    
    where θ is the angle of incidence at the ionosphere.
    
    For a flat-Earth approximation:
        sec(θ) ≈ √(1 + (D / 2h)²)
    
    For curved Earth (more accurate):
        Uses geometric calculation with Earth radius.
    
    Args:
        fof2_mhz: Critical frequency in MHz
        path_distance_km: Ground distance in km
        virtual_height_km: Virtual reflection height (default 300 km for F2)
    
    Returns:
        MUF in MHz
    
    Example:
        >>> muf_single_hop(5.0, 1500, 300)
        ~12 MHz
        >>> muf_single_hop(5.0, 3000, 300)
        ~20 MHz
    
    Note:
        - Typical F2 layer height: 250-400 km
        - Typical E layer height: 100-120 km
        - MUF varies with time of day, season, and solar activity
    """
    if fof2_mhz <= 0:
        raise ValueError(f"foF2 must be positive, got {fof2_mhz}")
    if path_distance_km < 0:
        raise ValueError(f"Distance must be non-negative, got {path_distance_km}")
    if virtual_height_km <= 0:
        raise ValueError(f"Height must be positive, got {virtual_height_km}")
    
    # For very short distances, MUF ≈ foF2
    if path_distance_km < 100:
        return fof2_mhz
    
    # Curved Earth geometry
    R = EARTH_RADIUS_KM
    h = virtual_height_km
    D = path_distance_km
    
    # Half the ground distance
    d_half = D / 2
    
    # Angle at Earth center (radians)
    alpha = d_half / R
    
    # Using spherical geometry
    # sin(θ) = R × sin(α) / (R + h)
    sin_alpha = math.sin(alpha)
    sin_theta = R * sin_alpha / (R + h)
    
    # Clamp to valid range
    sin_theta = min(sin_theta, 0.999)
    
    # sec(θ) = 1 / cos(θ)
    cos_theta = math.sqrt(1 - sin_theta**2)
    sec_theta = 1.0 / cos_theta
    
    # MUF = foF2 × sec(θ)
    muf = fof2_mhz * sec_theta
    
    return muf


def muf_factor(
    path_distance_km: float,
    virtual_height_km: float = 300.0,
) -> float:
    """
    Calculate MUF factor (sec θ) for given path.
    
    MUF = foF2 × MUF_factor
    
    Args:
        path_distance_km: Ground distance in km
        virtual_height_km: Virtual reflection height
    
    Returns:
        MUF factor (dimensionless)
    """
    # Use muf_single_hop with foF2 = 1 to get factor
    return muf_single_hop(1.0, path_distance_km, virtual_height_km)


def skip_distance_km(
    freq_mhz: float,
    fof2_mhz: float,
    virtual_height_km: float = 300.0,
) -> float:
    """
    Calculate skip distance (dead zone) for given frequency.
    
    The skip distance is the minimum distance at which sky-wave
    signals can be received. Closer stations are in the "skip zone"
    where neither ground wave nor sky wave is usable.
    
    Args:
        freq_mhz: Operating frequency in MHz
        fof2_mhz: Critical frequency in MHz
        virtual_height_km: Virtual reflection height
    
    Returns:
        Skip distance in km, or 0 if frequency is below foF2
    
    Example:
        >>> skip_distance_km(14.0, 5.0, 300)
        ~1200 km
    """
    if freq_mhz <= fof2_mhz:
        # Frequency below critical - no skip zone
        return 0.0
    
    R = EARTH_RADIUS_KM
    h = virtual_height_km
    
    # sec(θ) = f / foF2
    sec_theta = freq_mhz / fof2_mhz
    
    if sec_theta <= 1:
        return 0.0
    
    # cos(θ) = foF2 / f
    cos_theta = fof2_mhz / freq_mhz
    sin_theta = math.sqrt(1 - cos_theta**2)
    
    # Reverse the geometry
    # sin(θ) = R × sin(α) / (R + h)
    # sin(α) = sin(θ) × (R + h) / R
    sin_alpha = sin_theta * (R + h) / R
    
    if sin_alpha > 1:
        # Maximum skip distance exceeded
        return 2 * math.pi * R / 4  # Quarter of Earth circumference
    
    alpha = math.asin(sin_alpha)
    
    # Ground distance = 2 × R × α
    skip_dist = 2 * R * alpha
    
    return skip_dist


def max_single_hop_distance(
    virtual_height_km: float = 300.0,
) -> float:
    """
    Calculate maximum single-hop distance.
    
    Limited by geometry - ray becomes tangent to Earth.
    
    Args:
        virtual_height_km: Virtual reflection height
    
    Returns:
        Maximum single-hop distance in km
    """
    R = EARTH_RADIUS_KM
    h = virtual_height_km
    
    # Maximum angle where ray is tangent to Earth
    # cos(α_max) = R / (R + h)
    cos_alpha_max = R / (R + h)
    alpha_max = math.acos(cos_alpha_max)
    
    # Maximum ground distance
    return 2 * R * alpha_max


def number_of_hops(
    path_distance_km: float,
    virtual_height_km: float = 300.0,
) -> int:
    """
    Estimate number of ionospheric hops needed for path.
    
    Args:
        path_distance_km: Total path distance in km
        virtual_height_km: Virtual reflection height
    
    Returns:
        Estimated number of hops
    """
    max_hop = max_single_hop_distance(virtual_height_km)
    return max(1, math.ceil(path_distance_km / max_hop))


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class MUFResult:
    """Result of MUF calculation."""
    fof2_mhz: float
    path_distance_km: float
    virtual_height_km: float
    muf_mhz: float
    muf_factor: float
    max_hop_distance_km: float
    estimated_hops: int
    
    def __str__(self) -> str:
        return (
            f"MUF Calculation\n"
            f"{'='*40}\n"
            f"foF2:            {self.fof2_mhz:.1f} MHz\n"
            f"Path distance:   {self.path_distance_km:.0f} km\n"
            f"Virtual height:  {self.virtual_height_km:.0f} km\n"
            f"MUF factor:      {self.muf_factor:.2f}\n"
            f"MUF:             {self.muf_mhz:.1f} MHz\n"
            f"Max hop dist:    {self.max_hop_distance_km:.0f} km\n"
            f"Est. hops:       {self.estimated_hops}\n"
            f"\n"
            f"Note: Use frequencies below MUF for reliable propagation.\n"
            f"      Optimal: 80-90% of MUF (FOT = Frequency of Optimum Traffic)"
        )


@dataclass
class SkipResult:
    """Result of skip distance calculation."""
    frequency_mhz: float
    fof2_mhz: float
    virtual_height_km: float
    skip_distance_km: float
    is_below_critical: bool
    
    def __str__(self) -> str:
        if self.is_below_critical:
            return (
                f"Skip Distance Calculation\n"
                f"{'='*40}\n"
                f"Frequency:       {self.frequency_mhz:.1f} MHz\n"
                f"foF2:            {self.fof2_mhz:.1f} MHz\n"
                f"\n"
                f"Frequency is BELOW critical frequency.\n"
                f"No skip zone - signals can propagate at all distances."
            )
        else:
            return (
                f"Skip Distance Calculation\n"
                f"{'='*40}\n"
                f"Frequency:       {self.frequency_mhz:.1f} MHz\n"
                f"foF2:            {self.fof2_mhz:.1f} MHz\n"
                f"Virtual height:  {self.virtual_height_km:.0f} km\n"
                f"Skip distance:   {self.skip_distance_km:.0f} km\n"
                f"\n"
                f"Stations closer than {self.skip_distance_km:.0f} km are in the skip zone."
            )


def calculate_muf(
    fof2_mhz: float,
    path_distance_km: float,
    virtual_height_km: float = 300.0,
) -> MUFResult:
    """
    Calculate MUF with full results.
    
    Args:
        fof2_mhz: Critical frequency in MHz
        path_distance_km: Path distance in km
        virtual_height_km: Virtual reflection height
    
    Returns:
        MUFResult with all values
    """
    muf = muf_single_hop(fof2_mhz, path_distance_km, virtual_height_km)
    factor = muf_factor(path_distance_km, virtual_height_km)
    max_hop = max_single_hop_distance(virtual_height_km)
    hops = number_of_hops(path_distance_km, virtual_height_km)
    
    return MUFResult(
        fof2_mhz=fof2_mhz,
        path_distance_km=path_distance_km,
        virtual_height_km=virtual_height_km,
        muf_mhz=muf,
        muf_factor=factor,
        max_hop_distance_km=max_hop,
        estimated_hops=hops,
    )


def calculate_skip(
    freq_mhz: float,
    fof2_mhz: float,
    virtual_height_km: float = 300.0,
) -> SkipResult:
    """
    Calculate skip distance with full results.
    
    Args:
        freq_mhz: Operating frequency in MHz
        fof2_mhz: Critical frequency in MHz
        virtual_height_km: Virtual reflection height
    
    Returns:
        SkipResult with all values
    """
    skip = skip_distance_km(freq_mhz, fof2_mhz, virtual_height_km)
    
    return SkipResult(
        frequency_mhz=freq_mhz,
        fof2_mhz=fof2_mhz,
        virtual_height_km=virtual_height_km,
        skip_distance_km=skip,
        is_below_critical=(freq_mhz <= fof2_mhz),
    )


if __name__ == "__main__":
    # Quick test
    print("=== HamTools Propagation Test ===\n")
    
    # MUF calculation
    result = calculate_muf(5.0, 3000, 300)
    print(result)
    print()
    
    # Skip distance
    skip = calculate_skip(14.0, 5.0, 300)
    print(skip)
