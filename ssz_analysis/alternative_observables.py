# -*- coding: utf-8 -*-
"""
SSZ Alternative Observables Framework

Documents and provides analysis templates for alternative observables
that could test the SSZ (Segmented Spacetime) hypothesis.

The SSZ hypothesis predicts a universal modification to the effective
speed of light: c_eff = c / D_SSZ, where D_SSZ = 1 + delta_seg.

This should affect ANY measurement that depends on c:
1. Electromagnetic wave propagation (Schumann, GPS, VLBI)
2. Gravitational wave speed
3. Atomic clocks (fine structure constant)
4. Particle physics (muon lifetime)
5. Cosmological observations (CMB, redshift)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Observable:
    """An observable that could test SSZ."""
    name: str
    category: str
    sensitivity: str  # Expected sensitivity to delta_seg
    current_precision: str
    data_sources: List[str]
    ssz_prediction: str
    classical_background: str
    feasibility: str  # Easy, Medium, Hard
    notes: str = ""


# Catalog of alternative observables
ALTERNATIVE_OBSERVABLES = {
    # Electromagnetic propagation
    'schumann': Observable(
        name="Schumann Resonances",
        category="EM Propagation",
        sensitivity="delta_f/f = delta_seg",
        current_precision="~0.1% (hourly)",
        data_sources=["Zenodo Sierra Nevada", "Nagycenk", "Mitzpe Ramon"],
        ssz_prediction="All modes shift by same relative amount",
        classical_background="Ionospheric dispersion (mode-dependent)",
        feasibility="Easy",
        notes="Current analysis shows classical dispersion dominates"
    ),
    
    'gps_timing': Observable(
        name="GPS Timing Residuals",
        category="EM Propagation",
        sensitivity="delta_t = L/c * delta_seg",
        current_precision="~1 ns",
        data_sources=["IGS (International GNSS Service)", "JPL"],
        ssz_prediction="Systematic timing offset correlated with path length",
        classical_background="Ionospheric/tropospheric delays, clock drift",
        feasibility="Medium",
        notes="Requires careful separation from atmospheric effects"
    ),
    
    'vlbi': Observable(
        name="VLBI Baseline Measurements",
        category="EM Propagation",
        sensitivity="delta_L/L = delta_seg",
        current_precision="~1 mm over 1000 km",
        data_sources=["IVS (International VLBI Service)"],
        ssz_prediction="Apparent baseline changes",
        classical_background="Tectonic motion, atmospheric delays",
        feasibility="Hard",
        notes="Very high precision but complex systematics"
    ),
    
    # Gravitational waves
    'ligo_virgo': Observable(
        name="Gravitational Wave Speed",
        category="Gravitational",
        sensitivity="delta_c_gw/c = delta_seg",
        current_precision="~10^-15 (from GW170817)",
        data_sources=["LIGO", "Virgo", "KAGRA"],
        ssz_prediction="GW speed differs from EM by delta_seg",
        classical_background="None expected in GR",
        feasibility="Hard",
        notes="GW170817 already constrains |c_gw - c|/c < 10^-15"
    ),
    
    # Atomic physics
    'atomic_clocks': Observable(
        name="Atomic Clock Comparisons",
        category="Atomic Physics",
        sensitivity="delta_alpha/alpha ~ delta_seg",
        current_precision="~10^-18",
        data_sources=["NIST", "PTB", "NPL", "BIPM"],
        ssz_prediction="Fine structure constant variation",
        classical_background="Gravitational redshift, Doppler",
        feasibility="Medium",
        notes="Optical clocks now reach 10^-18 stability"
    ),
    
    'muon_lifetime': Observable(
        name="Muon Lifetime",
        category="Particle Physics",
        sensitivity="delta_tau/tau ~ delta_seg",
        current_precision="~10^-6",
        data_sources=["Fermilab Muon g-2", "J-PARC"],
        ssz_prediction="Lifetime variation with local delta_seg",
        classical_background="Time dilation (well understood)",
        feasibility="Hard",
        notes="Requires high-statistics muon experiments"
    ),
    
    # Cosmological
    'cmb_temperature': Observable(
        name="CMB Temperature Fluctuations",
        category="Cosmological",
        sensitivity="Complex (affects recombination)",
        current_precision="~10^-5 (Planck)",
        data_sources=["Planck", "WMAP", "ACT", "SPT"],
        ssz_prediction="Modified acoustic peaks",
        classical_background="Standard cosmology",
        feasibility="Hard",
        notes="Requires full cosmological modeling"
    ),
    
    'bao': Observable(
        name="Baryon Acoustic Oscillations",
        category="Cosmological",
        sensitivity="delta_r_s/r_s ~ delta_seg",
        current_precision="~1%",
        data_sources=["SDSS", "DESI", "Euclid"],
        ssz_prediction="Modified sound horizon",
        classical_background="Standard cosmology",
        feasibility="Hard",
        notes="Sensitive to early universe physics"
    ),
    
    # Laboratory
    'cavity_resonators': Observable(
        name="Microwave Cavity Resonators",
        category="Laboratory",
        sensitivity="delta_f/f = delta_seg",
        current_precision="~10^-15",
        data_sources=["Various metrology labs"],
        ssz_prediction="Frequency drift correlated with delta_seg",
        classical_background="Thermal expansion, material aging",
        feasibility="Medium",
        notes="Similar to Schumann but in controlled environment"
    ),
    
    'interferometers': Observable(
        name="Optical Interferometers",
        category="Laboratory",
        sensitivity="delta_L/L = delta_seg",
        current_precision="~10^-12",
        data_sources=["LIGO (as null test)", "Various labs"],
        ssz_prediction="Path length changes",
        classical_background="Thermal, seismic, quantum noise",
        feasibility="Medium",
        notes="LIGO-type interferometers could detect ~10^-20 effects"
    ),
}


def get_observable_by_feasibility(feasibility: str) -> List[Observable]:
    """Get observables filtered by feasibility."""
    return [obs for obs in ALTERNATIVE_OBSERVABLES.values() 
            if obs.feasibility.lower() == feasibility.lower()]


def get_observable_by_category(category: str) -> List[Observable]:
    """Get observables filtered by category."""
    return [obs for obs in ALTERNATIVE_OBSERVABLES.values()
            if category.lower() in obs.category.lower()]


def estimate_ssz_sensitivity(observable_name: str, delta_seg: float) -> Dict[str, float]:
    """
    Estimate the expected signal for a given delta_seg.
    
    Parameters
    ----------
    observable_name : str
        Name of the observable
    delta_seg : float
        SSZ segment density deviation
    
    Returns
    -------
    estimates : dict
        Expected signal magnitudes
    """
    estimates = {}
    
    if observable_name == 'schumann':
        # Schumann: delta_f/f = delta_seg
        f1 = 7.83  # Hz
        estimates['delta_f1'] = f1 * delta_seg
        estimates['delta_f1_percent'] = delta_seg * 100
        
    elif observable_name == 'gps_timing':
        # GPS: delta_t = L/c * delta_seg
        L = 20200e3  # GPS orbit altitude in meters
        c = 3e8
        estimates['delta_t_ns'] = (L / c) * delta_seg * 1e9
        
    elif observable_name == 'atomic_clocks':
        # Atomic clocks: delta_alpha/alpha ~ delta_seg
        estimates['delta_alpha_over_alpha'] = delta_seg
        
    elif observable_name == 'ligo_virgo':
        # GW speed: |c_gw - c|/c = delta_seg
        estimates['delta_c_gw_over_c'] = delta_seg
        
    elif observable_name == 'cavity_resonators':
        # Cavity: delta_f/f = delta_seg
        f_cavity = 10e9  # 10 GHz typical
        estimates['delta_f_Hz'] = f_cavity * delta_seg
        
    return estimates


def print_observable_catalog():
    """Print the catalog of alternative observables."""
    print("=" * 80)
    print("ALTERNATIVE OBSERVABLES FOR SSZ TESTING")
    print("=" * 80)
    print()
    
    categories = set(obs.category for obs in ALTERNATIVE_OBSERVABLES.values())
    
    for category in sorted(categories):
        print(f"\n### {category} ###\n")
        
        for name, obs in ALTERNATIVE_OBSERVABLES.items():
            if obs.category != category:
                continue
            
            print(f"**{obs.name}** [{obs.feasibility}]")
            print(f"  Sensitivity: {obs.sensitivity}")
            print(f"  Precision: {obs.current_precision}")
            print(f"  SSZ prediction: {obs.ssz_prediction}")
            print(f"  Background: {obs.classical_background}")
            print(f"  Data: {', '.join(obs.data_sources)}")
            if obs.notes:
                print(f"  Notes: {obs.notes}")
            print()


def recommend_next_tests(current_upper_bound: float) -> List[str]:
    """
    Recommend next observables to test based on current upper bound.
    
    Parameters
    ----------
    current_upper_bound : float
        Current 95% CL upper bound on |delta_seg|
    
    Returns
    -------
    recommendations : list
        List of recommended observables
    """
    recommendations = []
    
    # If upper bound is > 1%, Schumann is still useful
    if current_upper_bound > 0.01:
        recommendations.append("schumann: Continue with more stations/longer time series")
    
    # If upper bound is > 10^-6, GPS timing could help
    if current_upper_bound > 1e-6:
        recommendations.append("gps_timing: Analyze IGS timing residuals")
    
    # If upper bound is > 10^-15, atomic clocks are relevant
    if current_upper_bound > 1e-15:
        recommendations.append("atomic_clocks: Compare optical clock networks")
    
    # GW constraint is already very tight
    if current_upper_bound > 1e-15:
        recommendations.append("ligo_virgo: Already constrains to 10^-15 from GW170817")
    
    return recommendations


def generate_ssz_test_report(delta_seg_bound: float) -> str:
    """
    Generate a report on SSZ testability.
    
    Parameters
    ----------
    delta_seg_bound : float
        Current upper bound on |delta_seg|
    
    Returns
    -------
    report : str
        Formatted report
    """
    report = []
    report.append("=" * 70)
    report.append("SSZ TESTABILITY REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Current upper bound: |delta_seg| < {delta_seg_bound:.2e}")
    report.append("")
    
    report.append("Observable sensitivities:")
    report.append("-" * 40)
    
    for name, obs in ALTERNATIVE_OBSERVABLES.items():
        estimates = estimate_ssz_sensitivity(name, delta_seg_bound)
        if estimates:
            report.append(f"\n{obs.name}:")
            for key, value in estimates.items():
                report.append(f"  {key}: {value:.2e}")
            report.append(f"  Current precision: {obs.current_precision}")
            report.append(f"  Detectable: {'Yes' if obs.feasibility != 'Hard' else 'Challenging'}")
    
    report.append("")
    report.append("Recommendations:")
    report.append("-" * 40)
    for rec in recommend_next_tests(delta_seg_bound):
        report.append(f"  - {rec}")
    
    return "\n".join(report)
