#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Gravitational Wave Data for SSZ Testing

Downloads strain data and parameter estimates from GWOSC for
strong-field SSZ tests.

Usage:
    python scripts/fetch_gw_data.py --event GW150914
    python scripts/fetch_gw_data.py --event GW170817 --download-strain

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Check for required packages
try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

try:
    from gwosc.datasets import event_gps, run_segment
    from gwosc import datasets
    GWOSC_AVAILABLE = True
except ImportError:
    print("Note: gwosc package not installed. Install with: pip install gwosc")
    GWOSC_AVAILABLE = False

import numpy as np

# Output directory
OUTPUT_DIR = Path("data/gravitational_waves")


# Key events for SSZ testing
SSZ_RELEVANT_EVENTS = {
    'GW150914': {
        'type': 'BBH',
        'description': 'First GW detection, BBH merger',
        'm1': 35.6,  # M_sun
        'm2': 30.6,
        'm_final': 63.1,
        'spin_final': 0.69,
        'distance_mpc': 410,
        'ssz_relevance': 'Ringdown frequency test',
    },
    'GW170817': {
        'type': 'BNS',
        'description': 'Binary neutron star with EM counterpart',
        'm1': 1.46,
        'm2': 1.27,
        'distance_mpc': 40,
        'ssz_relevance': 'c_gw vs c_em comparison (already |Δc/c| < 10^-15)',
    },
    'GW190521': {
        'type': 'BBH',
        'description': 'Highest mass BBH, intermediate mass BH',
        'm1': 85,
        'm2': 66,
        'm_final': 142,
        'distance_mpc': 5300,
        'ssz_relevance': 'Strongest field, highest mass ringdown',
    },
    'GW190814': {
        'type': 'NSBH?',
        'description': 'Mass gap object (2.6 M_sun secondary)',
        'm1': 23,
        'm2': 2.6,
        'distance_mpc': 241,
        'ssz_relevance': 'Unusual mass ratio, potential NS',
    },
}


def get_event_info(event_name: str) -> dict:
    """Get event information from GWOSC API."""
    url = f"https://gwosc.org/eventapi/json/GWTC/{event_name}/"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Warning: Could not fetch {event_name} from GWOSC API")
            return None
    except Exception as e:
        print(f"Error fetching event info: {e}")
        return None


def get_parameter_estimates(event_name: str) -> dict:
    """Get parameter estimates for an event."""
    # Try GWOSC API first
    event_info = get_event_info(event_name)
    
    if event_info and 'events' in event_info:
        params = event_info['events'].get(event_name, {})
        return {
            'source': 'GWOSC API',
            'gps_time': params.get('GPS'),
            'mass_1': params.get('mass_1_source'),
            'mass_2': params.get('mass_2_source'),
            'chirp_mass': params.get('chirp_mass_source'),
            'total_mass': params.get('total_mass_source'),
            'final_mass': params.get('final_mass_source'),
            'final_spin': params.get('final_spin'),
            'luminosity_distance': params.get('luminosity_distance'),
            'redshift': params.get('redshift'),
            'snr': params.get('network_matched_filter_snr'),
        }
    
    # Fallback to hardcoded values
    if event_name in SSZ_RELEVANT_EVENTS:
        info = SSZ_RELEVANT_EVENTS[event_name]
        return {
            'source': 'Hardcoded (GWTC-3)',
            'mass_1': info.get('m1'),
            'mass_2': info.get('m2'),
            'final_mass': info.get('m_final'),
            'final_spin': info.get('spin_final'),
            'luminosity_distance': info.get('distance_mpc'),
        }
    
    return None


def compute_ssz_predictions(params: dict) -> dict:
    """Compute SSZ-relevant quantities from GW parameters."""
    if not params:
        return None
    
    # Constants
    G = 6.674e-11  # m^3 kg^-1 s^-2
    c = 2.998e8    # m/s
    M_sun = 1.989e30  # kg
    
    results = {}
    
    # Final mass and spin
    M_final = params.get('final_mass')
    a_final = params.get('final_spin')
    
    if M_final:
        M_kg = M_final * M_sun
        
        # Schwarzschild radius
        r_s = 2 * G * M_kg / c**2
        results['r_schwarzschild_km'] = r_s / 1000
        
        # GM/(Rc^2) at horizon
        results['gm_rc2_horizon'] = 0.5  # By definition for Schwarzschild
        
        # QNM frequency (Schwarzschild approximation)
        # f_QNM ≈ c^3 / (2π G M) × 0.0889 (for l=m=2, n=0)
        f_qnm_schwarz = c**3 / (2 * np.pi * G * M_kg) * 0.0889
        results['f_qnm_schwarzschild_hz'] = f_qnm_schwarz
        
        if a_final:
            # Kerr correction (approximate)
            # f_QNM increases with spin
            f_qnm_kerr = f_qnm_schwarz * (1 + 0.63 * (1 - a_final)**0.3)
            results['f_qnm_kerr_hz'] = f_qnm_kerr
            
            # ISCO radius (Kerr)
            # r_ISCO = r_s × Z(a) where Z ranges from 3 (a=0) to 0.5 (a=1)
            z1 = 1 + (1 - a_final**2)**(1/3) * ((1 + a_final)**(1/3) + (1 - a_final)**(1/3))
            z2 = np.sqrt(3 * a_final**2 + z1**2)
            r_isco_rg = 3 + z2 - np.sign(a_final) * np.sqrt((3 - z1) * (3 + z1 + 2*z2))
            results['r_isco_rg'] = r_isco_rg
            results['gm_rc2_isco'] = 1 / (2 * r_isco_rg)
    
    # SSZ test quantities
    results['ssz_test'] = {
        'description': 'Compare observed f_QNM with GR prediction',
        'gr_prediction': results.get('f_qnm_kerr_hz', results.get('f_qnm_schwarzschild_hz')),
        'ssz_model': 'f_QNM_SSZ = f_QNM_GR × (1 + δ_seg)',
        'expected_delta_seg': '< 1% if SSZ is small correction',
    }
    
    return results


def download_strain_data(event_name: str, output_dir: Path) -> bool:
    """Download strain data for an event."""
    if not GWOSC_AVAILABLE:
        print("Cannot download strain: gwosc package not installed")
        return False
    
    try:
        # Get GPS time
        gps = event_gps(event_name)
        print(f"Event GPS time: {gps}")
        
        # Get available data
        from gwosc.locate import get_event_urls
        urls = get_event_urls(event_name)
        
        print(f"Available data files: {len(urls)}")
        
        # Download first available file
        if urls:
            url = urls[0]
            filename = url.split('/')[-1]
            filepath = output_dir / filename
            
            print(f"Downloading: {filename}")
            response = requests.get(url, stream=True)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Saved: {filepath}")
            return True
        
    except Exception as e:
        print(f"Error downloading strain: {e}")
    
    return False


def create_ssz_gw_summary(events: list, output_dir: Path):
    """Create summary of GW events for SSZ testing."""
    summary = {
        'created': datetime.now().isoformat(),
        'purpose': 'SSZ Strong-Field Test Data',
        'events': {}
    }
    
    for event_name in events:
        print(f"\n{'='*60}")
        print(f"Processing: {event_name}")
        print('='*60)
        
        # Get parameters
        params = get_parameter_estimates(event_name)
        if params:
            print(f"Source: {params.get('source', 'Unknown')}")
            print(f"M1: {params.get('mass_1')} M_sun")
            print(f"M2: {params.get('mass_2')} M_sun")
            print(f"M_final: {params.get('final_mass')} M_sun")
            print(f"a_final: {params.get('final_spin')}")
            print(f"Distance: {params.get('luminosity_distance')} Mpc")
        
        # Compute SSZ predictions
        ssz = compute_ssz_predictions(params)
        if ssz:
            print(f"\nSSZ-relevant quantities:")
            r_s = ssz.get('r_schwarzschild_km')
            if r_s is not None:
                print(f"  r_s: {r_s:.1f} km")
            gm_rc2 = ssz.get('gm_rc2_horizon')
            if gm_rc2 is not None:
                print(f"  GM/(Rc^2) at horizon: {gm_rc2}")
            f_qnm = ssz.get('f_qnm_kerr_hz') or ssz.get('f_qnm_schwarzschild_hz')
            if f_qnm is not None:
                print(f"  f_QNM (GR): {f_qnm:.1f} Hz")
            if 'r_isco_rg' in ssz:
                print(f"  r_ISCO: {ssz['r_isco_rg']:.2f} r_g")
                print(f"  GM/(Rc^2) at ISCO: {ssz['gm_rc2_isco']:.3f}")
        
        # Store in summary
        summary['events'][event_name] = {
            'parameters': params,
            'ssz_predictions': ssz,
            'hardcoded_info': SSZ_RELEVANT_EVENTS.get(event_name, {}),
        }
    
    # Save summary
    summary_path = output_dir / "ssz_gw_events_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[OK] Saved summary: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GW data for SSZ testing"
    )
    parser.add_argument(
        '--event',
        type=str,
        default=None,
        help='Specific event to fetch (e.g., GW150914)'
    )
    parser.add_argument(
        '--all-ssz-events',
        action='store_true',
        help='Fetch all SSZ-relevant events'
    )
    parser.add_argument(
        '--download-strain',
        action='store_true',
        help='Download strain data (requires gwosc package)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/gravitational_waves',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SSZ GRAVITATIONAL WAVE DATA FETCHER")
    print("=" * 60)
    print()
    
    # Determine which events to process
    if args.event:
        events = [args.event]
    elif args.all_ssz_events:
        events = list(SSZ_RELEVANT_EVENTS.keys())
    else:
        # Default: show available events
        print("SSZ-relevant GW events:")
        print()
        for name, info in SSZ_RELEVANT_EVENTS.items():
            print(f"  {name}: {info['description']}")
            print(f"    Type: {info['type']}")
            print(f"    SSZ relevance: {info['ssz_relevance']}")
            print()
        
        print("Use --event <name> or --all-ssz-events to fetch data")
        return
    
    # Create summary
    summary = create_ssz_gw_summary(events, output_dir)
    
    # Optionally download strain
    if args.download_strain:
        for event_name in events:
            print(f"\nDownloading strain for {event_name}...")
            download_strain_data(event_name, output_dir)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review: {output_dir}/ssz_gw_events_summary.json")
    print(f"2. For ringdown analysis, install: pip install pycbc")
    print(f"3. Run SSZ ringdown test (to be implemented)")


if __name__ == "__main__":
    main()
