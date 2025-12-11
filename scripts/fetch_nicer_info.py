#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICER Neutron Star Data Information for SSZ Testing

Provides information about NICER observations of millisecond pulsars
that can be used for SSZ strong-field tests.

Note: Actual data download requires HEASARC account and specific tools.
This script provides metadata and SSZ predictions.

Usage:
    python scripts/fetch_nicer_info.py
    python scripts/fetch_nicer_info.py --pulsar J0030+0451

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# Output directory
OUTPUT_DIR = Path("data/neutron_stars")


# Key pulsars for SSZ testing (from NICER M-R measurements)
SSZ_RELEVANT_PULSARS = {
    'J0030+0451': {
        'name': 'PSR J0030+0451',
        'type': 'Isolated MSP',
        'period_ms': 4.87,
        'distance_kpc': 0.325,
        'mass_msun': 1.44,  # +0.15/-0.14
        'mass_err': (0.15, 0.14),
        'radius_km': 13.02,  # +1.24/-1.06
        'radius_err': (1.24, 1.06),
        'reference': 'Miller+2019 (ApJL 887, L24)',
        'arxiv': '1912.05705',
        'nicer_exposure_ks': 1800,
        'ssz_relevance': 'Best M-R constraint, clean isolated pulsar',
    },
    'J0740+6620': {
        'name': 'PSR J0740+6620',
        'type': 'MSP in binary',
        'period_ms': 2.89,
        'distance_kpc': 1.14,
        'mass_msun': 2.08,  # +0.07/-0.07 (Shapiro delay)
        'mass_err': (0.07, 0.07),
        'radius_km': 12.39,  # +1.30/-0.98
        'radius_err': (1.30, 0.98),
        'reference': 'Riley+2021 (ApJL 918, L27)',
        'arxiv': '2105.06980',
        'nicer_exposure_ks': 1100,
        'ssz_relevance': 'Highest precisely measured NS mass',
    },
    'J1614-2230': {
        'name': 'PSR J1614-2230',
        'type': 'MSP in binary',
        'period_ms': 3.15,
        'distance_kpc': 1.3,
        'mass_msun': 1.97,  # +0.04/-0.04 (Shapiro delay)
        'mass_err': (0.04, 0.04),
        'radius_km': None,  # Not yet measured by NICER
        'reference': 'Demorest+2010 (Nature 467, 1081)',
        'arxiv': '1010.5788',
        'ssz_relevance': 'Very precise mass from Shapiro delay',
    },
    'J0437-4715': {
        'name': 'PSR J0437-4715',
        'type': 'MSP in binary',
        'period_ms': 5.76,
        'distance_kpc': 0.156,
        'mass_msun': 1.44,
        'mass_err': (0.07, 0.07),
        'radius_km': None,
        'reference': 'Reardon+2024',
        'ssz_relevance': 'Closest MSP, excellent timing',
    },
}


def compute_ns_ssz_quantities(mass_msun: float, radius_km: float) -> dict:
    """
    Compute SSZ-relevant quantities for a neutron star.
    
    Parameters
    ----------
    mass_msun : float
        Mass in solar masses
    radius_km : float
        Radius in kilometers
    
    Returns
    -------
    dict
        SSZ-relevant quantities
    """
    # Constants
    G = 6.674e-11  # m^3 kg^-1 s^-2
    c = 2.998e8    # m/s
    M_sun = 1.989e30  # kg
    
    M_kg = mass_msun * M_sun
    R_m = radius_km * 1000
    
    results = {}
    
    # Schwarzschild radius
    r_s = 2 * G * M_kg / c**2
    results['r_schwarzschild_km'] = r_s / 1000
    
    # Compactness parameter
    compactness = G * M_kg / (R_m * c**2)
    results['compactness'] = compactness
    results['gm_rc2'] = compactness
    
    # Gravitational redshift at surface
    # z = (1 - 2GM/Rc²)^(-1/2) - 1
    z_grav = (1 - 2 * compactness)**(-0.5) - 1
    results['z_surface'] = z_grav
    
    # Time dilation factor at surface
    # sqrt(1 - 2GM/Rc²)
    tau_factor = np.sqrt(1 - 2 * compactness)
    results['time_dilation_factor'] = tau_factor
    
    # Surface gravity
    g_surface = G * M_kg / R_m**2
    results['g_surface_m_s2'] = g_surface
    results['g_surface_earth'] = g_surface / 9.81
    
    # Escape velocity
    v_esc = np.sqrt(2 * G * M_kg / R_m)
    results['v_escape_c'] = v_esc / c
    
    # SSZ predictions
    results['ssz_test'] = {
        'description': 'Test SSZ via pulse profile modeling',
        'gr_prediction': {
            'z_surface': z_grav,
            'time_dilation': tau_factor,
        },
        'ssz_model': {
            'z_SSZ': 'z_GR × (1 + δ_seg)',
            'tau_SSZ': 'τ_GR × (1 + δ_seg)',
        },
        'observable': 'Pulse profile shape depends on z and light bending',
        'sensitivity': f'δ_seg detectable if > {0.01/z_grav:.1%} of z_surface',
    }
    
    return results


def print_pulsar_info(name: str, info: dict):
    """Print formatted pulsar information."""
    print(f"\n{'='*60}")
    print(f"  {info['name']}")
    print(f"{'='*60}")
    print(f"  Type: {info['type']}")
    print(f"  Period: {info['period_ms']:.2f} ms")
    print(f"  Distance: {info['distance_kpc']:.3f} kpc")
    print()
    
    # Mass
    m = info['mass_msun']
    m_err = info.get('mass_err', (0, 0))
    print(f"  Mass: {m:.2f} (+{m_err[0]:.2f}/-{m_err[1]:.2f}) M_sun")
    
    # Radius
    r = info.get('radius_km')
    if r:
        r_err = info.get('radius_err', (0, 0))
        print(f"  Radius: {r:.2f} (+{r_err[0]:.2f}/-{r_err[1]:.2f}) km")
    else:
        print(f"  Radius: Not measured")
    
    print()
    print(f"  Reference: {info['reference']}")
    if 'arxiv' in info:
        print(f"  arXiv: {info['arxiv']}")
    print(f"  SSZ relevance: {info['ssz_relevance']}")
    
    # Compute SSZ quantities if we have M and R
    if r:
        ssz = compute_ns_ssz_quantities(m, r)
        print()
        print(f"  SSZ-relevant quantities:")
        print(f"    Compactness GM/(Rc²): {ssz['compactness']:.4f}")
        print(f"    Surface redshift z: {ssz['z_surface']:.4f}")
        print(f"    Time dilation: {ssz['time_dilation_factor']:.4f}")
        print(f"    Escape velocity: {ssz['v_escape_c']:.3f} c")
        print(f"    Surface gravity: {ssz['g_surface_earth']:.2e} g_Earth")
        
        return ssz
    
    return None


def create_nicer_summary(output_dir: Path):
    """Create summary of NICER pulsars for SSZ testing."""
    summary = {
        'created': datetime.now().isoformat(),
        'purpose': 'SSZ Strong-Field Test - Neutron Star Data',
        'data_source': 'NICER (Neutron Star Interior Composition Explorer)',
        'archive': 'https://heasarc.gsfc.nasa.gov/docs/nicer/nicer_archive.html',
        'pulsars': {}
    }
    
    print("=" * 60)
    print("SSZ NEUTRON STAR DATA SUMMARY")
    print("=" * 60)
    print()
    print("NICER provides the best mass-radius measurements for neutron stars,")
    print("enabling strong-field tests of SSZ at GM/(Rc^2) ~ 0.1-0.3")
    print()
    print("Comparison with Schumann experiment:")
    print(f"  Earth surface: GM/(Rc^2) ~ 7e-10")
    print(f"  Neutron star:  GM/(Rc^2) ~ 0.1-0.3  (10^8-10^9 x stronger!)")
    
    for name, info in SSZ_RELEVANT_PULSARS.items():
        ssz = print_pulsar_info(name, info)
        
        summary['pulsars'][name] = {
            'info': info,
            'ssz_quantities': ssz,
        }
    
    # Save summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ssz_nicer_pulsars_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved summary: {summary_path}")
    
    # Print data access instructions
    print()
    print("=" * 60)
    print("HOW TO ACCESS NICER DATA")
    print("=" * 60)
    print()
    print("1. HEASARC Browse:")
    print("   https://heasarc.gsfc.nasa.gov/cgi-bin/W3Browse/w3browse.pl")
    print("   - Mission: NICER")
    print("   - Target: e.g., 'PSR J0030+0451'")
    print()
    print("2. Direct archive:")
    print("   https://heasarc.gsfc.nasa.gov/FTP/nicer/data/obs/")
    print()
    print("3. Required software:")
    print("   - HEASoft (NICER tools)")
    print("   - PINT (pulsar timing)")
    print("   - XSPEC (spectral fitting)")
    print()
    print("4. Key data products:")
    print("   - Event files (.evt)")
    print("   - Cleaned event lists")
    print("   - Pulse profiles")
    print("   - Spectra")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="NICER neutron star data for SSZ testing"
    )
    parser.add_argument(
        '--pulsar',
        type=str,
        default=None,
        help='Specific pulsar (e.g., J0030+0451)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/neutron_stars',
        help='Output directory'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.pulsar:
        # Show specific pulsar
        key = args.pulsar.replace('PSR ', '').replace('PSR', '')
        if key in SSZ_RELEVANT_PULSARS:
            print_pulsar_info(key, SSZ_RELEVANT_PULSARS[key])
        else:
            print(f"Unknown pulsar: {args.pulsar}")
            print(f"Available: {list(SSZ_RELEVANT_PULSARS.keys())}")
    else:
        # Show all
        create_nicer_summary(output_dir)


if __name__ == "__main__":
    main()
