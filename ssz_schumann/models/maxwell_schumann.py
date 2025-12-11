#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maxwell-basierte Schumann-Resonanz-Theorie

Erweiterte Frequenzformeln basierend auf den Maxwell-Gleichungen
für die Erde-Ionosphäre-Kavität.

Theorie:
    Die Schumann-Resonanzen entstehen aus den Maxwell-Gleichungen
    in einer sphärischen Kavität zwischen Erdoberfläche und Ionosphäre.
    
    Ideale Frequenz (perfekter Leiter):
        f_n = (c / 2πR) × √(n(n+1))
    
    Reale Frequenz (mit Dämpfung):
        f_n = η × (c / 2πR) × √(n(n+1))
    
    wobei η ≈ 0.74 die Dämpfung durch endliche Leitfähigkeit beschreibt.

Referenzen:
    - Schumann, W.O. (1952). Z. Naturforsch. 7a, 149-154.
    - Balser & Wagner (1960). Nature 188, 638-641.
    - Nickolaenko & Hayakawa (2002). "Resonances in the Earth-Ionosphere Cavity"

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Physikalische Konstanten
C_LIGHT = 299792458.0  # m/s
MU_0 = 4 * np.pi * 1e-7  # H/m (Vakuum-Permeabilität)
EPSILON_0 = 8.854187817e-12  # F/m (Vakuum-Permittivität)
EARTH_RADIUS = 6.371e6  # m
PHI = (1 + np.sqrt(5)) / 2  # Goldener Schnitt

# Beobachtete Schumann-Frequenzen (Referenzwerte)
OBSERVED_FREQUENCIES = {
    1: 7.83,   # Hz
    2: 14.3,   # Hz
    3: 20.8,   # Hz
    4: 27.3,   # Hz
    5: 33.8,   # Hz
    6: 39.0,   # Hz
    7: 45.0,   # Hz
}


@dataclass
class SchumannMode:
    """Eigenschaften einer Schumann-Resonanz-Mode."""
    n: int                    # Mode-Nummer
    f_ideal: float           # Ideale Frequenz (Hz)
    f_observed: float        # Beobachtete Frequenz (Hz)
    eta: float               # Dämpfungsfaktor
    Q_factor: float          # Gütefaktor
    wavelength: float        # Wellenlänge (km)
    bandwidth: float         # Bandbreite (Hz)


def f_n_ideal(n: int, R: float = EARTH_RADIUS, c: float = C_LIGHT) -> float:
    """
    Ideale Schumann-Frequenz (perfekter Leiter).
    
    Formel aus Maxwell-Gleichungen für TM-Moden in sphärischer Kavität:
        f_n = (c / 2πR) × √(n(n+1))
    
    Args:
        n: Mode-Nummer (1, 2, 3, ...)
        R: Erdradius (m)
        c: Lichtgeschwindigkeit (m/s)
    
    Returns:
        Ideale Frequenz in Hz
    
    Example:
        >>> f_n_ideal(1)
        10.591...
        >>> f_n_ideal(2)
        18.340...
    """
    if n < 1:
        raise ValueError(f"Mode number must be >= 1, got {n}")
    
    return c / (2 * np.pi * R) * np.sqrt(n * (n + 1))


def f_n_damped(
    n: int,
    eta: float = 0.74,
    R: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> float:
    """
    Gedämpfte Schumann-Frequenz (reale Ionosphäre).
    
    Die endliche Leitfähigkeit der Ionosphäre verlangsamt die
    Wellenausbreitung, was zu niedrigeren Frequenzen führt.
    
    Formel:
        f_n = η × (c / 2πR) × √(n(n+1))
    
    Args:
        n: Mode-Nummer
        eta: Dämpfungsfaktor (typisch 0.74)
        R: Erdradius (m)
        c: Lichtgeschwindigkeit (m/s)
    
    Returns:
        Gedämpfte Frequenz in Hz
    """
    return eta * f_n_ideal(n, R, c)


def f_n_extended(
    n: int,
    eta: float = 0.74,
    h_iono: float = 85.0,
    sigma_iono: Optional[float] = None,
    R: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> float:
    """
    Erweiterte Schumann-Frequenz mit Ionosphärenhöhe.
    
    Berücksichtigt:
        1. Effektiven Radius R_eff = R + h_iono
        2. Optional: Leitfähigkeits-Korrektur
    
    Args:
        n: Mode-Nummer
        eta: Basis-Dämpfungsfaktor
        h_iono: Ionosphärenhöhe (km)
        sigma_iono: Ionosphärische Leitfähigkeit (S/m), optional
        R: Erdradius (m)
        c: Lichtgeschwindigkeit (m/s)
    
    Returns:
        Frequenz in Hz
    
    Note:
        Die Ionosphärenhöhe hat einen kleinen aber messbaren Effekt:
        Δf/f ≈ -Δh/(2R) ≈ -0.007% pro km Höhenänderung
    """
    # Effektiver Radius
    R_eff = R + h_iono * 1e3
    
    # Basis-Frequenz
    f_base = c / (2 * np.pi * R_eff) * np.sqrt(n * (n + 1))
    
    # Leitfähigkeits-Korrektur (optional)
    if sigma_iono is not None and sigma_iono > 0:
        # Skin-Depth in der Ionosphäre
        f_approx = eta * f_base
        skin_depth = np.sqrt(2 / (MU_0 * sigma_iono * 2 * np.pi * f_approx))
        
        # Korrektur basierend auf Verhältnis h/skin_depth
        # Wenn h << skin_depth: wenig Dämpfung
        # Wenn h >> skin_depth: starke Dämpfung
        eta_cond = 1 - 0.1 * np.tanh(h_iono * 1e3 / skin_depth)
        eta = eta * eta_cond
    
    return eta * f_base


def compute_eta_from_observed(
    f_obs: float,
    n: int,
    R: float = EARTH_RADIUS,
    c: float = C_LIGHT,
) -> float:
    """
    Berechne Dämpfungsfaktor aus beobachteter Frequenz.
    
    Args:
        f_obs: Beobachtete Frequenz (Hz)
        n: Mode-Nummer
        R: Erdradius (m)
        c: Lichtgeschwindigkeit (m/s)
    
    Returns:
        Dämpfungsfaktor eta
    """
    f_ideal = f_n_ideal(n, R, c)
    return f_obs / f_ideal


def compute_Q_factor(
    f_center: float,
    bandwidth: float,
) -> float:
    """
    Berechne Gütefaktor Q aus Frequenz und Bandbreite.
    
    Q = f_center / Δf
    
    Typische Werte für Schumann-Resonanzen: Q ≈ 4-6
    
    Args:
        f_center: Zentralfrequenz (Hz)
        bandwidth: Bandbreite bei -3dB (Hz)
    
    Returns:
        Gütefaktor Q
    """
    if bandwidth <= 0:
        return np.inf
    return f_center / bandwidth


def get_schumann_mode(n: int) -> SchumannMode:
    """
    Hole alle Eigenschaften einer Schumann-Mode.
    
    Args:
        n: Mode-Nummer (1-7)
    
    Returns:
        SchumannMode mit allen Eigenschaften
    """
    if n not in OBSERVED_FREQUENCIES:
        raise ValueError(f"Mode {n} not in database (1-7 available)")
    
    f_obs = OBSERVED_FREQUENCIES[n]
    f_ideal = f_n_ideal(n)
    eta = compute_eta_from_observed(f_obs, n)
    
    # Typische Bandbreiten (ca. 20% der Frequenz)
    bandwidth = 0.2 * f_obs
    Q = compute_Q_factor(f_obs, bandwidth)
    
    # Wellenlänge
    wavelength = C_LIGHT / f_obs / 1e3  # km
    
    return SchumannMode(
        n=n,
        f_ideal=f_ideal,
        f_observed=f_obs,
        eta=eta,
        Q_factor=Q,
        wavelength=wavelength,
        bandwidth=bandwidth,
    )


def frequency_shift_from_height_change(
    delta_h: float,
    f_ref: float,
    R: float = EARTH_RADIUS,
) -> float:
    """
    Frequenzverschiebung durch Ionosphärenhöhen-Änderung.
    
    Näherung für kleine Änderungen:
        Δf/f ≈ -Δh / (2R)
    
    Args:
        delta_h: Höhenänderung (km)
        f_ref: Referenzfrequenz (Hz)
        R: Erdradius (m)
    
    Returns:
        Frequenzverschiebung Δf (Hz)
    
    Example:
        >>> frequency_shift_from_height_change(10, 7.83)  # 10 km Erhöhung
        -0.0061...  # ca. -6 mHz
    """
    return -f_ref * delta_h * 1e3 / (2 * R)


def frequency_shift_from_ssz(
    delta_seg: float,
    f_ref: float,
) -> float:
    """
    Frequenzverschiebung durch SSZ-Segmentierung.
    
    SSZ-Formel:
        f_SSZ = f_classical / (1 + δ_seg)
        Δf = f_SSZ - f_classical = -f_classical × δ_seg / (1 + δ_seg)
    
    Für kleine δ_seg:
        Δf ≈ -f_classical × δ_seg
    
    Args:
        delta_seg: SSZ-Segmentierungsparameter
        f_ref: Referenzfrequenz (Hz)
    
    Returns:
        Frequenzverschiebung Δf (Hz)
    """
    return -f_ref * delta_seg / (1 + delta_seg)


def relative_shift_ssz(delta_seg: float) -> float:
    """
    Relative Frequenzverschiebung durch SSZ.
    
    Δf/f = -δ_seg / (1 + δ_seg)
    
    Schlüssel-Vorhersage: Diese ist MODE-UNABHÄNGIG!
    
    Args:
        delta_seg: SSZ-Segmentierungsparameter
    
    Returns:
        Relative Verschiebung Δf/f
    """
    return -delta_seg / (1 + delta_seg)


def compute_mode_ratios() -> Dict[str, float]:
    """
    Berechne theoretische Mode-Verhältnisse.
    
    Die Verhältnisse f_n/f_1 sind unabhängig von η:
        f_2/f_1 = √(6/2) = √3 ≈ 1.732
        f_3/f_1 = √(12/2) = √6 ≈ 2.449
        f_4/f_1 = √(20/2) = √10 ≈ 3.162
    
    Returns:
        Dictionary mit theoretischen Verhältnissen
    """
    ratios = {}
    
    for n in range(2, 8):
        ratio_theory = np.sqrt(n * (n + 1) / 2)
        ratio_obs = OBSERVED_FREQUENCIES.get(n, np.nan) / OBSERVED_FREQUENCIES[1]
        
        ratios[f"f{n}/f1_theory"] = ratio_theory
        ratios[f"f{n}/f1_observed"] = ratio_obs
        ratios[f"f{n}/f1_deviation"] = (ratio_obs - ratio_theory) / ratio_theory
    
    return ratios


def print_schumann_summary():
    """Drucke Zusammenfassung der Schumann-Resonanzen."""
    print("\n" + "=" * 70)
    print("SCHUMANN RESONANCES - MAXWELL THEORY SUMMARY")
    print("=" * 70)
    
    print("\nIdeal vs Observed Frequencies:")
    print("-" * 50)
    print(f"{'Mode':>5} | {'f_ideal':>10} | {'f_obs':>10} | {'eta':>8} | {'Q':>6}")
    print("-" * 50)
    
    for n in range(1, 8):
        mode = get_schumann_mode(n)
        print(f"{n:>5} | {mode.f_ideal:>10.3f} | {mode.f_observed:>10.2f} | "
              f"{mode.eta:>8.4f} | {mode.Q_factor:>6.1f}")
    
    print("\nMode Ratios (f_n / f_1):")
    print("-" * 50)
    ratios = compute_mode_ratios()
    for n in range(2, 6):
        theory = ratios[f"f{n}/f1_theory"]
        obs = ratios[f"f{n}/f1_observed"]
        dev = ratios[f"f{n}/f1_deviation"] * 100
        print(f"f{n}/f1: Theory = {theory:.4f}, Observed = {obs:.4f}, "
              f"Deviation = {dev:+.2f}%")
    
    print("\nKey Physical Parameters:")
    print("-" * 50)
    print(f"Earth Radius: {EARTH_RADIUS:.3e} m ({EARTH_RADIUS/1e3:.0f} km)")
    print(f"Speed of Light: {C_LIGHT:.0f} m/s")
    print(f"Typical eta: 0.74 (range 0.72-0.76)")
    print(f"Typical Q-factor: 4-6")
    print(f"Ionosphere Height: 60-90 km (D-layer)")
    
    print("\nSSZ Prediction:")
    print("-" * 50)
    print("If SSZ effects exist, the relative frequency shift")
    print("df/f = -delta_seg/(1+delta_seg) should be IDENTICAL for all modes.")
    print("This is the key testable prediction!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_schumann_summary()
