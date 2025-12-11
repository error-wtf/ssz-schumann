#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Feedline Module - Cable Attenuation Calculations

Calculations for coaxial cable losses:
- Attenuation per 100m for common cable types
- Total feedline loss
- Power at antenna

Based on concepts from Daempfung_dB_EMV.md.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .core import ratio_from_db


# =============================================================================
# CABLE DATABASE
# =============================================================================

# Attenuation data: dB per 100m at various frequencies
# Format: {cable_type: {freq_mhz: attenuation_db_per_100m}}
# Data from typical manufacturer specifications

CABLE_DATA: Dict[str, Dict[float, float]] = {
    "RG-58": {
        1.8: 5.5,
        3.5: 7.5,
        7.0: 10.5,
        14.0: 15.0,
        21.0: 18.5,
        28.0: 21.5,
        50.0: 29.0,
        144.0: 52.0,
        432.0: 95.0,
    },
    "RG-213": {
        1.8: 2.5,
        3.5: 3.5,
        7.0: 5.0,
        14.0: 7.0,
        21.0: 8.5,
        28.0: 10.0,
        50.0: 13.5,
        144.0: 24.0,
        432.0: 44.0,
    },
    "RG-8": {
        1.8: 2.5,
        3.5: 3.5,
        7.0: 5.0,
        14.0: 7.0,
        21.0: 8.5,
        28.0: 10.0,
        50.0: 13.5,
        144.0: 24.0,
        432.0: 44.0,
    },
    "AIRCELL-7": {
        1.8: 1.8,
        3.5: 2.5,
        7.0: 3.5,
        14.0: 5.0,
        21.0: 6.0,
        28.0: 7.0,
        50.0: 9.5,
        144.0: 16.5,
        432.0: 30.0,
    },
    "ECOFLEX-10": {
        1.8: 1.2,
        3.5: 1.7,
        7.0: 2.4,
        14.0: 3.4,
        21.0: 4.2,
        28.0: 4.9,
        50.0: 6.5,
        144.0: 11.5,
        432.0: 21.0,
    },
    "ECOFLEX-15": {
        1.8: 0.8,
        3.5: 1.1,
        7.0: 1.6,
        14.0: 2.3,
        21.0: 2.8,
        28.0: 3.3,
        50.0: 4.4,
        144.0: 7.8,
        432.0: 14.5,
    },
    "LMR-400": {
        1.8: 1.3,
        3.5: 1.8,
        7.0: 2.6,
        14.0: 3.7,
        21.0: 4.5,
        28.0: 5.2,
        50.0: 7.0,
        144.0: 12.2,
        432.0: 22.0,
    },
    "H-2000": {
        1.8: 1.0,
        3.5: 1.4,
        7.0: 2.0,
        14.0: 2.8,
        21.0: 3.5,
        28.0: 4.0,
        50.0: 5.4,
        144.0: 9.5,
        432.0: 17.5,
    },
}

# Cable characteristics
CABLE_INFO: Dict[str, Dict[str, any]] = {
    "RG-58": {
        "impedance_ohm": 50,
        "diameter_mm": 5.0,
        "velocity_factor": 0.66,
        "description": "Thin, flexible, high loss - short runs only",
    },
    "RG-213": {
        "impedance_ohm": 50,
        "diameter_mm": 10.3,
        "velocity_factor": 0.66,
        "description": "Standard HF coax, moderate loss",
    },
    "RG-8": {
        "impedance_ohm": 50,
        "diameter_mm": 10.3,
        "velocity_factor": 0.66,
        "description": "Similar to RG-213",
    },
    "AIRCELL-7": {
        "impedance_ohm": 50,
        "diameter_mm": 7.3,
        "velocity_factor": 0.83,
        "description": "Low loss, foam dielectric",
    },
    "ECOFLEX-10": {
        "impedance_ohm": 50,
        "diameter_mm": 10.2,
        "velocity_factor": 0.85,
        "description": "Very low loss, excellent for HF/VHF",
    },
    "ECOFLEX-15": {
        "impedance_ohm": 50,
        "diameter_mm": 14.6,
        "velocity_factor": 0.86,
        "description": "Ultra low loss, best for long runs",
    },
    "LMR-400": {
        "impedance_ohm": 50,
        "diameter_mm": 10.3,
        "velocity_factor": 0.85,
        "description": "Low loss, popular for VHF/UHF",
    },
    "H-2000": {
        "impedance_ohm": 50,
        "diameter_mm": 10.3,
        "velocity_factor": 0.83,
        "description": "Very low loss, German quality",
    },
}


# =============================================================================
# ATTENUATION CALCULATIONS
# =============================================================================

def get_available_cables() -> list:
    """
    Get list of available cable types.
    
    Returns:
        List of cable type names
    """
    return list(CABLE_DATA.keys())


def interpolate_attenuation(
    f_mhz: float,
    cable_data: Dict[float, float],
) -> float:
    """
    Interpolate attenuation for a given frequency.
    
    Cable attenuation roughly follows: α ∝ √f
    
    Args:
        f_mhz: Frequency in MHz
        cable_data: Dictionary of {freq: attenuation} pairs
    
    Returns:
        Interpolated attenuation in dB/100m
    """
    freqs = sorted(cable_data.keys())
    
    # If exact frequency exists
    if f_mhz in cable_data:
        return cable_data[f_mhz]
    
    # Find bracketing frequencies
    lower_freq = None
    upper_freq = None
    
    for freq in freqs:
        if freq <= f_mhz:
            lower_freq = freq
        if freq >= f_mhz and upper_freq is None:
            upper_freq = freq
    
    # Extrapolation cases
    if lower_freq is None:
        # Below lowest frequency - use sqrt scaling
        return cable_data[freqs[0]] * math.sqrt(f_mhz / freqs[0])
    
    if upper_freq is None:
        # Above highest frequency - use sqrt scaling
        return cable_data[freqs[-1]] * math.sqrt(f_mhz / freqs[-1])
    
    if lower_freq == upper_freq:
        return cable_data[lower_freq]
    
    # Linear interpolation in sqrt(f) domain
    # α ∝ √f, so we interpolate in sqrt space
    sqrt_f = math.sqrt(f_mhz)
    sqrt_lower = math.sqrt(lower_freq)
    sqrt_upper = math.sqrt(upper_freq)
    
    alpha_lower = cable_data[lower_freq]
    alpha_upper = cable_data[upper_freq]
    
    # Linear interpolation
    t = (sqrt_f - sqrt_lower) / (sqrt_upper - sqrt_lower)
    return alpha_lower + t * (alpha_upper - alpha_lower)


def attenuation_db_per_100m(
    f_mhz: float,
    cable_type: str,
) -> float:
    """
    Get cable attenuation in dB per 100 meters.
    
    Args:
        f_mhz: Frequency in MHz
        cable_type: Cable type (e.g., "RG-58", "ECOFLEX-10")
    
    Returns:
        Attenuation in dB per 100m
    
    Raises:
        ValueError: If cable type is unknown
    
    Example:
        >>> attenuation_db_per_100m(14.2, "RG-58")
        15.2...
        >>> attenuation_db_per_100m(14.2, "ECOFLEX-10")
        3.5...
    
    Note:
        See Daempfung_dB_EMV.md for dB calculation background.
    """
    cable_upper = cable_type.upper().replace(" ", "-")
    
    if cable_upper not in CABLE_DATA:
        available = ", ".join(get_available_cables())
        raise ValueError(
            f"Unknown cable type: {cable_type}\n"
            f"Available: {available}"
        )
    
    return interpolate_attenuation(f_mhz, CABLE_DATA[cable_upper])


def total_loss_db(
    f_mhz: float,
    cable_type: str,
    length_m: float,
) -> float:
    """
    Calculate total feedline loss.
    
    Formula: Loss = (attenuation_per_100m / 100) × length
    
    Args:
        f_mhz: Frequency in MHz
        cable_type: Cable type
        length_m: Cable length in meters
    
    Returns:
        Total loss in dB
    
    Example:
        >>> total_loss_db(14.2, "RG-58", 30)
        4.56...  # 30m of RG-58 at 14 MHz
    """
    if length_m < 0:
        raise ValueError(f"Length must be non-negative, got {length_m}")
    
    atten_per_100m = attenuation_db_per_100m(f_mhz, cable_type)
    return (atten_per_100m / 100.0) * length_m


def power_at_antenna(
    p_tx_watt: float,
    f_mhz: float,
    cable_type: str,
    length_m: float,
) -> Tuple[float, float]:
    """
    Calculate power reaching the antenna after feedline loss.
    
    Args:
        p_tx_watt: Transmitter power in Watts
        f_mhz: Frequency in MHz
        cable_type: Cable type
        length_m: Cable length in meters
    
    Returns:
        Tuple of (power_at_antenna_watt, loss_db)
    
    Example:
        >>> power_at_antenna(100, 14.2, "RG-58", 30)
        (35.0, 4.56)  # 35W reaches antenna, 4.56 dB lost
    """
    loss_db = total_loss_db(f_mhz, cable_type, length_m)
    loss_ratio = ratio_from_db(loss_db)
    power_out = p_tx_watt / loss_ratio
    return power_out, loss_db


def cable_comparison(
    f_mhz: float,
    length_m: float,
    p_tx_watt: float = 100.0,
) -> Dict[str, Dict]:
    """
    Compare all cables at given frequency and length.
    
    Args:
        f_mhz: Frequency in MHz
        length_m: Cable length in meters
        p_tx_watt: TX power for comparison
    
    Returns:
        Dictionary of cable comparisons
    """
    results = {}
    
    for cable in get_available_cables():
        loss = total_loss_db(f_mhz, cable, length_m)
        power_out, _ = power_at_antenna(p_tx_watt, f_mhz, cable, length_m)
        
        results[cable] = {
            "loss_db": loss,
            "power_out_watt": power_out,
            "efficiency_percent": (power_out / p_tx_watt) * 100,
        }
    
    return results


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class FeedlineResult:
    """Result of feedline loss calculation."""
    frequency_mhz: float
    cable_type: str
    length_m: float
    attenuation_per_100m: float
    total_loss_db: float
    p_tx_watt: Optional[float] = None
    p_antenna_watt: Optional[float] = None
    
    def __str__(self) -> str:
        lines = [
            f"Feedline Loss Calculation",
            f"{'='*40}",
            f"Cable:           {self.cable_type}",
            f"Frequency:       {self.frequency_mhz:.3f} MHz",
            f"Length:          {self.length_m:.1f} m",
            f"Atten/100m:      {self.attenuation_per_100m:.2f} dB",
            f"Total loss:      {self.total_loss_db:.2f} dB",
        ]
        
        if self.p_tx_watt is not None:
            efficiency = (self.p_antenna_watt / self.p_tx_watt) * 100
            lines.extend([
                f"TX power:        {self.p_tx_watt:.1f} W",
                f"Power at ant:    {self.p_antenna_watt:.1f} W",
                f"Efficiency:      {efficiency:.1f}%",
            ])
        
        return "\n".join(lines)


@dataclass
class CableComparisonResult:
    """Result of cable comparison."""
    frequency_mhz: float
    length_m: float
    p_tx_watt: float
    comparisons: Dict[str, Dict]
    
    def __str__(self) -> str:
        lines = [
            f"Cable Comparison at {self.frequency_mhz:.1f} MHz, {self.length_m:.0f}m",
            f"{'='*60}",
            f"{'Cable':<15} {'Loss (dB)':<12} {'Power Out':<12} {'Efficiency':<12}",
            f"{'-'*60}",
        ]
        
        # Sort by loss
        sorted_cables = sorted(
            self.comparisons.items(),
            key=lambda x: x[1]["loss_db"]
        )
        
        for cable, data in sorted_cables:
            lines.append(
                f"{cable:<15} {data['loss_db']:<12.2f} "
                f"{data['power_out_watt']:<12.1f} "
                f"{data['efficiency_percent']:<12.1f}%"
            )
        
        return "\n".join(lines)


def calculate_feedline_loss(
    f_mhz: float,
    cable_type: str,
    length_m: float,
    p_tx_watt: Optional[float] = None,
) -> FeedlineResult:
    """
    Calculate feedline loss with optional power calculation.
    
    Args:
        f_mhz: Frequency in MHz
        cable_type: Cable type
        length_m: Cable length in meters
        p_tx_watt: Optional TX power in Watts
    
    Returns:
        FeedlineResult with all values
    """
    atten = attenuation_db_per_100m(f_mhz, cable_type)
    loss = total_loss_db(f_mhz, cable_type, length_m)
    
    p_antenna = None
    if p_tx_watt is not None:
        p_antenna, _ = power_at_antenna(p_tx_watt, f_mhz, cable_type, length_m)
    
    return FeedlineResult(
        frequency_mhz=f_mhz,
        cable_type=cable_type.upper(),
        length_m=length_m,
        attenuation_per_100m=atten,
        total_loss_db=loss,
        p_tx_watt=p_tx_watt,
        p_antenna_watt=p_antenna,
    )


def compare_cables(
    f_mhz: float,
    length_m: float,
    p_tx_watt: float = 100.0,
) -> CableComparisonResult:
    """
    Compare all available cables.
    
    Args:
        f_mhz: Frequency in MHz
        length_m: Cable length in meters
        p_tx_watt: TX power for comparison
    
    Returns:
        CableComparisonResult
    """
    return CableComparisonResult(
        frequency_mhz=f_mhz,
        length_m=length_m,
        p_tx_watt=p_tx_watt,
        comparisons=cable_comparison(f_mhz, length_m, p_tx_watt),
    )


if __name__ == "__main__":
    # Quick test
    print("=== HamTools Feedline Test ===\n")
    
    # Single cable
    result = calculate_feedline_loss(14.2, "RG-58", 30, 100)
    print(result)
    print()
    
    # Cable comparison
    comparison = compare_cables(14.2, 30, 100)
    print(comparison)
