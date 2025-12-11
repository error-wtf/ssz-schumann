#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Field Strength Module - E-Field Calculations

Simple far-field electric field strength estimation.

WARNING: This is a simplified model for educational purposes.
It does NOT account for:
- Ground reflections
- Terrain effects
- Atmospheric absorption
- Near-field effects
- Regulatory compliance

For legal compliance, use professional EMC measurement tools.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Optional

from .core import ratio_from_db


# Physical constants
C_LIGHT = 299_792_458.0  # m/s
Z0 = 376.73  # Free space impedance (Ohms)


# =============================================================================
# FIELD STRENGTH CALCULATIONS
# =============================================================================

def field_strength_v_per_m(
    Pt_watt: float,
    G_dBi: float,
    distance_km: float,
    freq_mhz: Optional[float] = None,
) -> float:
    """
    Calculate far-field electric field strength.
    
    Uses the free-space formula:
        E = √(30 × EIRP) / d
    
    where:
        E = electric field strength (V/m)
        EIRP = effective isotropic radiated power (W)
        d = distance (m)
    
    Derivation:
        Power density S = EIRP / (4πd²)
        E = √(S × Z₀) = √(EIRP × Z₀ / 4πd²)
        E = √(30 × EIRP) / d  (since Z₀/4π ≈ 30)
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        distance_km: Distance in kilometers
        freq_mhz: Frequency in MHz (optional, for info only)
    
    Returns:
        Electric field strength in V/m
    
    Example:
        >>> field_strength_v_per_m(100, 6, 1)  # 100W, 6dBi, 1km
        0.69 V/m
    
    Note:
        This is a FREE-SPACE model. Real-world values may differ
        significantly due to ground effects, obstacles, etc.
    """
    if Pt_watt <= 0:
        raise ValueError(f"Power must be positive, got {Pt_watt}")
    if distance_km <= 0:
        raise ValueError(f"Distance must be positive, got {distance_km}")
    
    # Convert gain to linear
    G_linear = ratio_from_db(G_dBi)
    
    # EIRP
    eirp = Pt_watt * G_linear
    
    # Distance in meters
    d_m = distance_km * 1000
    
    # E-field (V/m)
    E = math.sqrt(30 * eirp) / d_m
    
    return E


def field_strength_dbuv_per_m(
    Pt_watt: float,
    G_dBi: float,
    distance_km: float,
    freq_mhz: Optional[float] = None,
) -> float:
    """
    Calculate field strength in dBμV/m.
    
    dBμV/m = 20 × log₁₀(E × 10⁶)
    
    where E is in V/m.
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        distance_km: Distance in kilometers
        freq_mhz: Frequency in MHz (optional)
    
    Returns:
        Field strength in dBμV/m
    
    Example:
        >>> field_strength_dbuv_per_m(100, 6, 1)
        116.8 dBμV/m
    """
    E_v_per_m = field_strength_v_per_m(Pt_watt, G_dBi, distance_km, freq_mhz)
    E_uv_per_m = E_v_per_m * 1e6
    return 20 * math.log10(E_uv_per_m)


def power_density_w_per_m2(
    Pt_watt: float,
    G_dBi: float,
    distance_km: float,
) -> float:
    """
    Calculate power density in W/m².
    
    Formula: S = EIRP / (4πd²)
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        distance_km: Distance in kilometers
    
    Returns:
        Power density in W/m²
    """
    G_linear = ratio_from_db(G_dBi)
    eirp = Pt_watt * G_linear
    d_m = distance_km * 1000
    
    return eirp / (4 * math.pi * d_m**2)


def distance_for_field_strength(
    Pt_watt: float,
    G_dBi: float,
    E_target_v_per_m: float,
) -> float:
    """
    Calculate distance for a target field strength.
    
    Inverse of field_strength_v_per_m.
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        E_target_v_per_m: Target field strength in V/m
    
    Returns:
        Distance in kilometers
    """
    G_linear = ratio_from_db(G_dBi)
    eirp = Pt_watt * G_linear
    
    # E = √(30 × EIRP) / d
    # d = √(30 × EIRP) / E
    d_m = math.sqrt(30 * eirp) / E_target_v_per_m
    
    return d_m / 1000


def safe_distance_icnirp(
    Pt_watt: float,
    G_dBi: float,
    freq_mhz: float,
) -> float:
    """
    Estimate safe distance based on ICNIRP guidelines.
    
    ICNIRP reference levels for general public (E-field):
    - 10-400 MHz: 28 V/m
    - 400-2000 MHz: 1.375 × √f V/m (f in MHz)
    - 2-300 GHz: 61 V/m
    
    WARNING: This is a simplified estimate. For compliance,
    consult official regulations and use proper measurement.
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        freq_mhz: Frequency in MHz
    
    Returns:
        Estimated safe distance in meters
    """
    # ICNIRP E-field limits (general public)
    if freq_mhz < 10:
        E_limit = 87  # V/m (simplified)
    elif freq_mhz <= 400:
        E_limit = 28  # V/m
    elif freq_mhz <= 2000:
        E_limit = 1.375 * math.sqrt(freq_mhz)  # V/m
    else:
        E_limit = 61  # V/m
    
    # Calculate distance for this limit
    d_km = distance_for_field_strength(Pt_watt, G_dBi, E_limit)
    
    return d_km * 1000  # Return in meters


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class FieldStrengthResult:
    """Result of field strength calculation."""
    Pt_watt: float
    G_dBi: float
    distance_km: float
    freq_mhz: Optional[float]
    E_v_per_m: float
    E_dbuv_per_m: float
    S_w_per_m2: float
    eirp_watt: float
    
    def __str__(self) -> str:
        lines = [
            f"Field Strength Calculation (Free Space)",
            f"{'='*45}",
            f"TX Power:        {self.Pt_watt:.1f} W",
            f"Antenna Gain:    {self.G_dBi:.1f} dBi",
            f"EIRP:            {self.eirp_watt:.1f} W",
            f"Distance:        {self.distance_km:.2f} km ({self.distance_km*1000:.0f} m)",
        ]
        
        if self.freq_mhz:
            lines.append(f"Frequency:       {self.freq_mhz:.3f} MHz")
        
        lines.extend([
            f"",
            f"Results:",
            f"  E-field:       {self.E_v_per_m:.4f} V/m",
            f"  E-field:       {self.E_dbuv_per_m:.1f} dBμV/m",
            f"  Power density: {self.S_w_per_m2:.2e} W/m²",
            f"",
            f"⚠️  WARNING: Free-space model only!",
            f"    Real values may differ significantly.",
        ])
        
        return "\n".join(lines)


@dataclass
class SafeDistanceResult:
    """Result of safe distance calculation."""
    Pt_watt: float
    G_dBi: float
    freq_mhz: float
    E_limit_v_per_m: float
    safe_distance_m: float
    
    def __str__(self) -> str:
        return (
            f"Safe Distance Estimate (ICNIRP)\n"
            f"{'='*45}\n"
            f"TX Power:        {self.Pt_watt:.1f} W\n"
            f"Antenna Gain:    {self.G_dBi:.1f} dBi\n"
            f"Frequency:       {self.freq_mhz:.3f} MHz\n"
            f"E-field limit:   {self.E_limit_v_per_m:.1f} V/m\n"
            f"Safe distance:   {self.safe_distance_m:.2f} m\n"
            f"\n"
            f"⚠️  WARNING: Simplified estimate only!\n"
            f"    For compliance, use proper EMC measurement."
        )


def calculate_field_strength(
    Pt_watt: float,
    G_dBi: float,
    distance_km: float,
    freq_mhz: Optional[float] = None,
) -> FieldStrengthResult:
    """
    Calculate field strength with full results.
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        distance_km: Distance in kilometers
        freq_mhz: Frequency in MHz (optional)
    
    Returns:
        FieldStrengthResult with all values
    """
    E = field_strength_v_per_m(Pt_watt, G_dBi, distance_km, freq_mhz)
    E_db = field_strength_dbuv_per_m(Pt_watt, G_dBi, distance_km, freq_mhz)
    S = power_density_w_per_m2(Pt_watt, G_dBi, distance_km)
    eirp = Pt_watt * ratio_from_db(G_dBi)
    
    return FieldStrengthResult(
        Pt_watt=Pt_watt,
        G_dBi=G_dBi,
        distance_km=distance_km,
        freq_mhz=freq_mhz,
        E_v_per_m=E,
        E_dbuv_per_m=E_db,
        S_w_per_m2=S,
        eirp_watt=eirp,
    )


def calculate_safe_distance(
    Pt_watt: float,
    G_dBi: float,
    freq_mhz: float,
) -> SafeDistanceResult:
    """
    Calculate safe distance estimate.
    
    Args:
        Pt_watt: Transmitter power in Watts
        G_dBi: Antenna gain in dBi
        freq_mhz: Frequency in MHz
    
    Returns:
        SafeDistanceResult with all values
    """
    # Get ICNIRP limit
    if freq_mhz < 10:
        E_limit = 87
    elif freq_mhz <= 400:
        E_limit = 28
    elif freq_mhz <= 2000:
        E_limit = 1.375 * math.sqrt(freq_mhz)
    else:
        E_limit = 61
    
    safe_dist = safe_distance_icnirp(Pt_watt, G_dBi, freq_mhz)
    
    return SafeDistanceResult(
        Pt_watt=Pt_watt,
        G_dBi=G_dBi,
        freq_mhz=freq_mhz,
        E_limit_v_per_m=E_limit,
        safe_distance_m=safe_dist,
    )


if __name__ == "__main__":
    # Quick test
    print("=== HamTools Field Strength Test ===\n")
    
    # Field strength at 1 km
    result = calculate_field_strength(100, 6, 1, 14.2)
    print(result)
    print()
    
    # Safe distance
    safe = calculate_safe_distance(100, 6, 14.2)
    print(safe)
