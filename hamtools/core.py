#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Core Module - Frequency, Wavelength, Period, dB Calculations

Fundamental calculations for amateur radio:
- Frequency ↔ Wavelength ↔ Period conversions
- dB calculations (power, voltage ratios)
- ERP/EIRP calculations

All calculations use c = 299,792,458 m/s (speed of light in vacuum).

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass
from typing import Tuple

# Physical constants
C_LIGHT = 299_792_458.0  # Speed of light in vacuum (m/s)


# =============================================================================
# FREQUENCY / WAVELENGTH / PERIOD CONVERSIONS
# =============================================================================

def freq_to_lambda(f_hz: float) -> float:
    """
    Convert frequency to wavelength.
    
    Formula: λ = c / f
    
    Args:
        f_hz: Frequency in Hertz (Hz)
    
    Returns:
        Wavelength in meters (m)
    
    Example:
        >>> freq_to_lambda(7_100_000)  # 7.1 MHz
        42.2242...
    """
    if f_hz <= 0:
        raise ValueError(f"Frequency must be positive, got {f_hz}")
    return C_LIGHT / f_hz


def lambda_to_freq(lambda_m: float) -> float:
    """
    Convert wavelength to frequency.
    
    Formula: f = c / λ
    
    Args:
        lambda_m: Wavelength in meters (m)
    
    Returns:
        Frequency in Hertz (Hz)
    
    Example:
        >>> lambda_to_freq(40)  # 40m band
        7494811.45
    """
    if lambda_m <= 0:
        raise ValueError(f"Wavelength must be positive, got {lambda_m}")
    return C_LIGHT / lambda_m


def freq_to_period(f_hz: float) -> float:
    """
    Convert frequency to period.
    
    Formula: T = 1 / f
    
    Args:
        f_hz: Frequency in Hertz (Hz)
    
    Returns:
        Period in seconds (s)
    
    Example:
        >>> freq_to_period(1000)  # 1 kHz
        0.001
    """
    if f_hz <= 0:
        raise ValueError(f"Frequency must be positive, got {f_hz}")
    return 1.0 / f_hz


def period_to_freq(T_s: float) -> float:
    """
    Convert period to frequency.
    
    Formula: f = 1 / T
    
    Args:
        T_s: Period in seconds (s)
    
    Returns:
        Frequency in Hertz (Hz)
    
    Example:
        >>> period_to_freq(0.001)  # 1 ms period
        1000.0
    """
    if T_s <= 0:
        raise ValueError(f"Period must be positive, got {T_s}")
    return 1.0 / T_s


# =============================================================================
# UNIT-SAFE WRAPPERS
# =============================================================================

def freq_mhz_to_lambda(f_mhz: float) -> float:
    """
    Convert frequency in MHz to wavelength in meters.
    
    Convenience wrapper for ham radio use.
    
    Args:
        f_mhz: Frequency in Megahertz (MHz)
    
    Returns:
        Wavelength in meters (m)
    
    Example:
        >>> freq_mhz_to_lambda(7.1)  # 40m band
        42.2242...
        >>> freq_mhz_to_lambda(14.2)  # 20m band
        21.1121...
    """
    return freq_to_lambda(f_mhz * 1e6)


def freq_khz_to_lambda(f_khz: float) -> float:
    """
    Convert frequency in kHz to wavelength in meters.
    
    Args:
        f_khz: Frequency in Kilohertz (kHz)
    
    Returns:
        Wavelength in meters (m)
    
    Example:
        >>> freq_khz_to_lambda(7100)  # 40m band
        42.2242...
    """
    return freq_to_lambda(f_khz * 1e3)


def lambda_to_freq_mhz(lambda_m: float) -> float:
    """
    Convert wavelength in meters to frequency in MHz.
    
    Args:
        lambda_m: Wavelength in meters (m)
    
    Returns:
        Frequency in Megahertz (MHz)
    
    Example:
        >>> lambda_to_freq_mhz(40)  # 40m band
        7.4948...
    """
    return lambda_to_freq(lambda_m) / 1e6


# =============================================================================
# dB CALCULATIONS
# =============================================================================

def db_from_ratio(ratio: float) -> float:
    """
    Convert power ratio to decibels.
    
    Formula: dB = 10 × log₁₀(ratio)
    
    Args:
        ratio: Power ratio (P2/P1)
    
    Returns:
        Value in decibels (dB)
    
    Example:
        >>> db_from_ratio(2)  # Double power
        3.0103...
        >>> db_from_ratio(10)  # 10× power
        10.0
    """
    if ratio <= 0:
        raise ValueError(f"Ratio must be positive, got {ratio}")
    return 10.0 * math.log10(ratio)


def ratio_from_db(db: float) -> float:
    """
    Convert decibels to power ratio.
    
    Formula: ratio = 10^(dB/10)
    
    Args:
        db: Value in decibels (dB)
    
    Returns:
        Power ratio
    
    Example:
        >>> ratio_from_db(3)  # 3 dB
        1.9953...
        >>> ratio_from_db(10)  # 10 dB
        10.0
    """
    return 10.0 ** (db / 10.0)


def db_from_power(P2: float, P1: float) -> float:
    """
    Calculate dB from two power values.
    
    Formula: dB = 10 × log₁₀(P2/P1)
    
    Args:
        P2: Power 2 (numerator)
        P1: Power 1 (denominator, reference)
    
    Returns:
        Power ratio in dB
    
    Example:
        >>> db_from_power(100, 50)  # 100W vs 50W
        3.0103...
    """
    if P1 <= 0 or P2 <= 0:
        raise ValueError("Powers must be positive")
    return db_from_ratio(P2 / P1)


def db_from_voltage(U2: float, U1: float) -> float:
    """
    Calculate dB from two voltage values.
    
    Formula: dB = 20 × log₁₀(U2/U1)
    
    Note: Voltage uses factor 20 (not 10) because P ∝ U²
    
    Args:
        U2: Voltage 2 (numerator)
        U1: Voltage 1 (denominator, reference)
    
    Returns:
        Voltage ratio in dB
    
    Example:
        >>> db_from_voltage(2, 1)  # Double voltage
        6.0206...
    """
    if U1 <= 0 or U2 <= 0:
        raise ValueError("Voltages must be positive")
    return 20.0 * math.log10(U2 / U1)


def voltage_ratio_from_db(db: float) -> float:
    """
    Convert dB to voltage ratio.
    
    Formula: ratio = 10^(dB/20)
    
    Args:
        db: Value in decibels (dB)
    
    Returns:
        Voltage ratio
    """
    return 10.0 ** (db / 20.0)


# =============================================================================
# ERP / EIRP CALCULATIONS
# =============================================================================

def erp_watt(P_tx_watt: float, gain_dBd: float, loss_db: float = 0.0) -> float:
    """
    Calculate Effective Radiated Power (ERP).
    
    ERP is referenced to a half-wave dipole.
    
    Formula: ERP = P_tx × G_d / L
    
    where:
        G_d = 10^(gain_dBd/10) (linear gain relative to dipole)
        L = 10^(loss_dB/10) (linear loss factor)
    
    Args:
        P_tx_watt: Transmitter power in Watts
        gain_dBd: Antenna gain in dBd (relative to dipole)
        loss_db: Total system losses in dB (feedline, connectors, etc.)
    
    Returns:
        ERP in Watts
    
    Example:
        >>> erp_watt(100, 3, 2)  # 100W, 3dBd gain, 2dB loss
        125.89...
    """
    gain_linear = ratio_from_db(gain_dBd)
    loss_linear = ratio_from_db(loss_db)
    return P_tx_watt * gain_linear / loss_linear


def eirp_watt(P_tx_watt: float, gain_dBi: float, loss_db: float = 0.0) -> float:
    """
    Calculate Effective Isotropic Radiated Power (EIRP).
    
    EIRP is referenced to an isotropic radiator.
    
    Formula: EIRP = P_tx × G_i / L
    
    Note: dBi = dBd + 2.15 (dipole has 2.15 dBi gain)
    
    Args:
        P_tx_watt: Transmitter power in Watts
        gain_dBi: Antenna gain in dBi (relative to isotropic)
        loss_db: Total system losses in dB
    
    Returns:
        EIRP in Watts
    
    Example:
        >>> eirp_watt(100, 5.15, 2)  # 100W, 5.15dBi (≈3dBd), 2dB loss
        206.54...
    """
    gain_linear = ratio_from_db(gain_dBi)
    loss_linear = ratio_from_db(loss_db)
    return P_tx_watt * gain_linear / loss_linear


def dbd_to_dbi(gain_dBd: float) -> float:
    """
    Convert antenna gain from dBd to dBi.
    
    Formula: dBi = dBd + 2.15
    
    Args:
        gain_dBd: Gain relative to dipole
    
    Returns:
        Gain relative to isotropic
    """
    return gain_dBd + 2.15


def dbi_to_dbd(gain_dBi: float) -> float:
    """
    Convert antenna gain from dBi to dBd.
    
    Formula: dBd = dBi - 2.15
    
    Args:
        gain_dBi: Gain relative to isotropic
    
    Returns:
        Gain relative to dipole
    """
    return gain_dBi - 2.15


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class FrequencyResult:
    """Result of frequency/wavelength calculation."""
    frequency_hz: float
    frequency_mhz: float
    wavelength_m: float
    period_s: float
    
    def __str__(self) -> str:
        return (
            f"Frequency: {self.frequency_mhz:.4f} MHz ({self.frequency_hz:.0f} Hz)\n"
            f"Wavelength: {self.wavelength_m:.4f} m\n"
            f"Period: {self.period_s:.6e} s"
        )


@dataclass
class ERPResult:
    """Result of ERP/EIRP calculation."""
    p_tx_watt: float
    gain_db: float
    loss_db: float
    erp_watt: float
    eirp_watt: float
    
    def __str__(self) -> str:
        return (
            f"TX Power: {self.p_tx_watt:.1f} W\n"
            f"Gain: {self.gain_db:.1f} dBd ({self.gain_db + 2.15:.1f} dBi)\n"
            f"Loss: {self.loss_db:.1f} dB\n"
            f"ERP: {self.erp_watt:.1f} W\n"
            f"EIRP: {self.eirp_watt:.1f} W"
        )


def calculate_frequency_info(f_mhz: float) -> FrequencyResult:
    """
    Calculate all frequency-related values.
    
    Args:
        f_mhz: Frequency in MHz
    
    Returns:
        FrequencyResult with all values
    """
    f_hz = f_mhz * 1e6
    return FrequencyResult(
        frequency_hz=f_hz,
        frequency_mhz=f_mhz,
        wavelength_m=freq_to_lambda(f_hz),
        period_s=freq_to_period(f_hz),
    )


def calculate_erp_info(
    p_tx_watt: float,
    gain_dBd: float,
    loss_db: float = 0.0,
) -> ERPResult:
    """
    Calculate ERP and EIRP.
    
    Args:
        p_tx_watt: Transmitter power in Watts
        gain_dBd: Antenna gain in dBd
        loss_db: System losses in dB
    
    Returns:
        ERPResult with all values
    """
    return ERPResult(
        p_tx_watt=p_tx_watt,
        gain_db=gain_dBd,
        loss_db=loss_db,
        erp_watt=erp_watt(p_tx_watt, gain_dBd, loss_db),
        eirp_watt=eirp_watt(p_tx_watt, dbd_to_dbi(gain_dBd), loss_db),
    )


if __name__ == "__main__":
    # Quick test
    print("=== HamTools Core Test ===\n")
    
    # Frequency/Wavelength
    print("7.1 MHz:")
    result = calculate_frequency_info(7.1)
    print(result)
    
    print("\n14.2 MHz:")
    result = calculate_frequency_info(14.2)
    print(result)
    
    # dB calculations
    print("\n=== dB Calculations ===")
    print(f"2× power = {db_from_ratio(2):.2f} dB")
    print(f"10× power = {db_from_ratio(10):.2f} dB")
    print(f"3 dB = {ratio_from_db(3):.2f}× power")
    
    # ERP
    print("\n=== ERP/EIRP ===")
    erp_result = calculate_erp_info(100, 3, 2)
    print(erp_result)
