#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Real Data Analysis - End-to-End Test

This script performs a complete SSZ vs Classical hypothesis test using
REAL Schumann resonance data. No synthetic data, no hallucinated values.

Usage:
    python scripts/run_ssz_schumann_realdata.py --station sierra_nevada --year 2013
    python scripts/run_ssz_schumann_realdata.py --station sierra_nevada --year 2013 --fit-eta

Data Sources:
    - Schumann: Zenodo Sierra Nevada ELF Station (doi:10.5281/zenodo.7761644)
    - F10.7/Kp: Reconstructed from NOAA/GFZ monthly averages (clearly marked)

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
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_analysis.core import (
    C, R_EARTH,
    compute_classical_frequencies,
    get_classical_reference,
    fit_classical_eta,
    compute_delta_seg_classical,
    estimate_delta_ssz_global,
    compute_ssz_chi_squared,
    compute_ssz_upper_bound,
    run_ssz_hypothesis_test,
    analyze_correlations,
    compute_mode_correlations,
    compute_ssz_score,
    ClassicalReference,
)

# Configuration
STATIONS = {
    'sierra_nevada': {
        'name': 'Sierra Nevada ELF Station',
        'code': 'SNV',
        'lat': 37.0,
        'lon': -3.4,
        'data_source': 'Zenodo (doi:10.5281/zenodo.7761644)',
        'data_format': 'Processed CSV from raw int16 binary',
    }
}

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


def load_schumann_data(station: str, year: int, month: int = None) -> pd.DataFrame:
    """
    Load processed Schumann data for a station/year.
    
    Parameters
    ----------
    station : str
        Station identifier (e.g., 'sierra_nevada')
    year : int
        Year (e.g., 2013)
    month : int, optional
        Specific month (1-12), or None for all available
    
    Returns
    -------
    df : pd.DataFrame
        Schumann data with timestamp, f1, f2, f3, f4
    """
    processed_dir = DATA_DIR / "schumann" / "real" / "processed"
    
    if month:
        # Specific month file
        pattern = f"schumann_{year % 100:02d}{month:02d}_processed.csv"
        filepath = processed_dir / pattern
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        df = pd.read_csv(filepath)
    else:
        # Try to load all months for the year
        all_dfs = []
        for m in range(1, 13):
            pattern = f"schumann_{year % 100:02d}{m:02d}_processed.csv"
            filepath = processed_dir / pattern
            if filepath.exists():
                df = pd.read_csv(filepath)
                all_dfs.append(df)
        
        if not all_dfs:
            # Try combined file
            combined = processed_dir / "schumann_all_months_processed.csv"
            if combined.exists():
                df = pd.read_csv(combined)
                # Filter by year if timestamp available
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['timestamp'].dt.year == year]
            else:
                raise FileNotFoundError(f"No Schumann data found for {station}/{year}")
        else:
            df = pd.concat(all_dfs, ignore_index=True)
    
    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
    
    return df


def load_space_weather(year: int, month: int = None) -> tuple:
    """
    Load space weather proxy data (F10.7, Kp).
    
    Returns (f107_df, kp_df) or (None, None) if not available.
    Data source is clearly marked in the returned DataFrames.
    """
    solar_dir = DATA_DIR / "solar"
    geomag_dir = DATA_DIR / "geomag"
    
    f107_df = None
    kp_df = None
    
    # Try to load F10.7
    if month:
        f107_file = solar_dir / f"f107_{year}{month:02d}_daily.csv"
    else:
        f107_file = solar_dir / f"f107_{year}_2017_daily.csv"
        if not f107_file.exists():
            f107_file = solar_dir / f"f107_{year % 100:02d}10_daily.csv"  # Fallback to Oct
    
    if f107_file.exists():
        f107_df = pd.read_csv(f107_file)
        f107_df['date'] = pd.to_datetime(f107_df['date']).dt.date
        f107_df['data_source'] = 'RECONSTRUCTED from NOAA monthly averages'
    
    # Try to load Kp
    if month:
        kp_file = geomag_dir / f"kp_{year}{month:02d}_daily.csv"
    else:
        kp_file = geomag_dir / f"kp_{year}_2017_daily.csv"
        if not kp_file.exists():
            kp_file = geomag_dir / f"kp_{year % 100:02d}10_daily.csv"
    
    if kp_file.exists():
        kp_df = pd.read_csv(kp_file)
        kp_df['date'] = pd.to_datetime(kp_df['date']).dt.date
        kp_df['data_source'] = 'RECONSTRUCTED from GFZ monthly averages'
    
    return f107_df, kp_df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to daily means and stds."""
    if 'date' not in df.columns:
        raise ValueError("DataFrame must have 'date' column")
    
    modes = ['f1', 'f2', 'f3', 'f4']
    
    agg_dict = {}
    for mode in modes:
        if mode in df.columns:
            agg_dict[f'{mode}_mean'] = (mode, 'mean')
            agg_dict[f'{mode}_std'] = (mode, 'std')
            agg_dict[f'{mode}_count'] = (mode, 'count')
    
    daily = df.groupby('date').agg(**agg_dict).reset_index()
    return daily


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate Schumann data against expected ranges.
    
    Returns validation report.
    """
    # Expected ranges (literature values +/- 20%)
    expected = {
        'f1': (6.5, 9.5),
        'f2': (12.0, 17.0),
        'f3': (17.0, 24.0),
        'f4': (23.0, 32.0),
    }
    
    report = {'valid': True, 'issues': [], 'stats': {}}
    
    for mode, (f_min, f_max) in expected.items():
        if mode not in df.columns:
            report['issues'].append(f"Missing column: {mode}")
            report['valid'] = False
            continue
        
        values = df[mode].dropna()
        if len(values) == 0:
            report['issues'].append(f"No valid data for {mode}")
            report['valid'] = False
            continue
        
        mean = values.mean()
        std = values.std()
        
        report['stats'][mode] = {
            'mean': float(mean),
            'std': float(std),
            'min': float(values.min()),
            'max': float(values.max()),
            'count': int(len(values)),
        }
        
        if mean < f_min or mean > f_max:
            report['issues'].append(
                f"{mode}: mean={mean:.3f} Hz outside expected range [{f_min}, {f_max}]"
            )
            # Don't mark as invalid, just warn
    
    return report


def run_analysis(
    station: str,
    year: int,
    month: int = None,
    fit_eta: bool = False,
    eta_fixed: float = 0.74,
) -> dict:
    """
    Run complete SSZ vs Classical analysis.
    
    Returns dictionary with all results (no hallucinated values).
    """
    print("=" * 70)
    print("SSZ SCHUMANN REAL DATA ANALYSIS")
    print("=" * 70)
    print()
    
    results = {
        'metadata': {
            'station': station,
            'station_info': STATIONS.get(station, {}),
            'year': year,
            'month': month,
            'analysis_timestamp': datetime.now().isoformat(),
            'fit_eta': fit_eta,
        },
        'data': {},
        'classical': {},
        'ssz': {},
        'correlations': {},
        'conclusion': {},
    }
    
    # 1. Load Schumann data
    print("[1] Loading Schumann data...")
    try:
        schumann_df = load_schumann_data(station, year, month)
        n_records = len(schumann_df)
        print(f"    Loaded: {n_records} records")
        
        if 'timestamp' in schumann_df.columns:
            t_min = schumann_df['timestamp'].min()
            t_max = schumann_df['timestamp'].max()
            print(f"    Time range: {t_min} to {t_max}")
        
        results['data']['n_records'] = n_records
        results['data']['source'] = STATIONS.get(station, {}).get('data_source', 'Unknown')
        
    except FileNotFoundError as e:
        print(f"    [ERROR] {e}")
        results['data']['error'] = str(e)
        return results
    
    # 2. Validate data
    print("\n[2] Validating data...")
    validation = validate_data(schumann_df)
    results['data']['validation'] = validation
    
    if not validation['valid']:
        print(f"    [WARNING] Validation issues: {validation['issues']}")
    else:
        print("    [OK] Data passes validation")
    
    for mode, stats in validation['stats'].items():
        print(f"    {mode}: {stats['mean']:.3f} +/- {stats['std']:.3f} Hz (n={stats['count']})")
    
    # 3. Aggregate to daily
    print("\n[3] Aggregating to daily means...")
    daily_df = aggregate_daily(schumann_df)
    n_days = len(daily_df)
    print(f"    Daily records: {n_days}")
    results['data']['n_days'] = n_days
    
    # 4. Load space weather (optional)
    print("\n[4] Loading space weather proxies...")
    f107_df, kp_df = load_space_weather(year, month)
    
    if f107_df is not None:
        print(f"    F10.7: {len(f107_df)} days")
        print(f"    Source: {f107_df['data_source'].iloc[0]}")
        results['data']['f107_source'] = f107_df['data_source'].iloc[0]
        
        # Merge with daily data
        daily_df = daily_df.merge(
            f107_df[['date', 'f107']], 
            on='date', 
            how='left'
        )
    else:
        print("    F10.7: NOT AVAILABLE")
        results['data']['f107_source'] = 'NOT AVAILABLE'
    
    if kp_df is not None:
        print(f"    Kp: {len(kp_df)} days")
        print(f"    Source: {kp_df['data_source'].iloc[0]}")
        results['data']['kp_source'] = kp_df['data_source'].iloc[0]
        
        daily_df = daily_df.merge(
            kp_df[['date', 'kp']], 
            on='date', 
            how='left'
        )
    else:
        print("    Kp: NOT AVAILABLE")
        results['data']['kp_source'] = 'NOT AVAILABLE'
    
    # 5. Classical reference
    print("\n[5] Computing classical reference...")
    
    # Get observed means for eta fitting
    f_obs = {
        'f1': daily_df['f1_mean'].mean(),
        'f2': daily_df['f2_mean'].mean(),
        'f3': daily_df['f3_mean'].mean(),
        'f4': daily_df['f4_mean'].mean(),
    }
    f_obs_err = {
        'f1': daily_df['f1_mean'].std(),
        'f2': daily_df['f2_mean'].std(),
        'f3': daily_df['f3_mean'].std(),
        'f4': daily_df['f4_mean'].std(),
    }
    
    if fit_eta:
        eta, eta_err = fit_classical_eta(f_obs, f_obs_err, method='least_squares')
        print(f"    Fitted eta = {eta:.4f} +/- {eta_err:.4f}")
    else:
        eta = eta_fixed
        eta_err = 0.0
        print(f"    Fixed eta = {eta:.4f}")
    
    classical_ref = get_classical_reference(eta)
    
    results['classical']['eta'] = float(eta)
    results['classical']['eta_err'] = float(eta_err)
    results['classical']['eta_method'] = 'fit' if fit_eta else 'fixed'
    results['classical']['frequencies'] = {
        'f1': float(classical_ref.f1),
        'f2': float(classical_ref.f2),
        'f3': float(classical_ref.f3),
        'f4': float(classical_ref.f4),
    }
    
    print(f"    f1_classical = {classical_ref.f1:.3f} Hz")
    print(f"    f2_classical = {classical_ref.f2:.3f} Hz")
    print(f"    f3_classical = {classical_ref.f3:.3f} Hz")
    print(f"    f4_classical = {classical_ref.f4:.3f} Hz")
    
    # 6. Per-mode deltas
    print("\n[6] Computing per-mode delta_seg_classical...")
    
    results['classical']['observed'] = {}
    results['classical']['delta_seg'] = {}
    
    for mode in ['f1', 'f2', 'f3', 'f4']:
        f_obs_val = f_obs[mode]
        f_class = getattr(classical_ref, mode)
        delta = compute_delta_seg_classical(f_obs_val, f_class)
        delta_err = f_obs_err[mode] / f_class
        
        results['classical']['observed'][mode] = float(f_obs_val)
        results['classical']['delta_seg'][mode] = {
            'value': float(delta),
            'error': float(delta_err),
            'percent': float(delta * 100),
            'percent_err': float(delta_err * 100),
        }
        
        print(f"    {mode}: delta = {delta*100:.2f}% +/- {delta_err*100:.2f}%")
    
    # 7. SSZ hypothesis test
    print("\n[7] SSZ hypothesis test...")
    
    ssz_result = run_ssz_hypothesis_test(daily_df, classical_ref)
    
    results['ssz'] = {
        'delta_ssz_global': float(ssz_result.delta_ssz_global),
        'delta_ssz_global_percent': float(ssz_result.delta_ssz_global * 100),
        'delta_ssz_std': float(ssz_result.delta_ssz_std),
        'delta_ssz_std_percent': float(ssz_result.delta_ssz_std * 100),
        'chi_squared': float(ssz_result.chi_squared),
        'ndof': int(ssz_result.ndof),
        'chi_squared_reduced': float(ssz_result.chi_squared_reduced),
        'p_value': float(ssz_result.p_value),
        'is_consistent': bool(ssz_result.is_consistent),
        'upper_bound_95': float(ssz_result.upper_bound_95),
        'upper_bound_95_percent': float(ssz_result.upper_bound_95 * 100),
    }
    
    print(f"    delta_SSZ_global = {ssz_result.delta_ssz_global*100:.3f}% +/- {ssz_result.delta_ssz_std*100:.3f}%")
    print(f"    chi^2 = {ssz_result.chi_squared:.2f}")
    print(f"    ndof = {ssz_result.ndof}")
    print(f"    chi^2/ndof = {ssz_result.chi_squared_reduced:.2f}")
    print(f"    p-value = {ssz_result.p_value:.2e}")
    print(f"    SSZ consistent: {ssz_result.is_consistent}")
    
    # 8. Mode correlations
    print("\n[8] Computing mode correlations...")
    
    # Add anomaly columns
    for mode in ['f1', 'f2', 'f3', 'f4']:
        mean_col = f'{mode}_mean'
        if mean_col in daily_df.columns:
            mode_mean = daily_df[mean_col].mean()
            daily_df[f'delta_{mode}_anomaly'] = -(daily_df[mean_col] - mode_mean) / mode_mean
    
    mean_corr, corr_matrix = compute_mode_correlations(daily_df)
    
    results['ssz']['mean_mode_correlation'] = float(mean_corr) if not np.isnan(mean_corr) else None
    print(f"    Mean inter-mode correlation: {mean_corr:.3f}")
    
    # 9. Space weather correlations (if available)
    print("\n[9] Space weather correlations...")
    
    if 'f107' in daily_df.columns or 'kp' in daily_df.columns:
        # Compute mean delta anomaly
        anomaly_cols = [c for c in daily_df.columns if 'delta_' in c and '_anomaly' in c]
        if anomaly_cols:
            daily_df['delta_ssz_anomaly'] = daily_df[anomaly_cols].mean(axis=1)
            
            corr_results = analyze_correlations(daily_df, 'delta_ssz_anomaly', ['f107', 'kp'])
            results['correlations'] = {}
            
            for proxy, corr in corr_results.items():
                results['correlations'][proxy] = {
                    'r': float(corr['r']) if not np.isnan(corr['r']) else None,
                    'p': float(corr['p']) if not np.isnan(corr['p']) else None,
                    'n': int(corr['n']),
                }
                if not np.isnan(corr['r']):
                    print(f"    Corr(delta_SSZ, {proxy}): r={corr['r']:.3f}, p={corr['p']:.3f}")
                else:
                    print(f"    Corr(delta_SSZ, {proxy}): NOT AVAILABLE")
    else:
        print("    No space weather data available for correlation analysis")
    
    # 10. Conclusion
    print("\n[10] Conclusion...")
    
    mode_spread = np.std([
        results['classical']['delta_seg']['f1']['value'],
        results['classical']['delta_seg']['f2']['value'],
        results['classical']['delta_seg']['f3']['value'],
        results['classical']['delta_seg']['f4']['value'],
    ])
    
    results['conclusion'] = {
        'mode_spread_percent': float(mode_spread * 100),
        'ssz_minimalmodel_rejected': ssz_result.chi_squared_reduced > 3.0,
        'classical_dispersion_dominates': mode_spread > 0.02,
        'practical_upper_bound_percent': min(1.0, float(ssz_result.delta_ssz_std * 100 * 2)),
    }
    
    if ssz_result.chi_squared_reduced > 3.0:
        print("    --> SSZ MINIMALMODEL REJECTED (chi^2/ndof >> 1)")
        print("    --> Classical ionospheric dispersion dominates")
    else:
        print("    --> SSZ minimalmodel not rejected (chi^2/ndof ~ 1)")
    
    print(f"    Mode spread: {mode_spread*100:.2f}%")
    print(f"    Practical upper bound on |delta_SSZ|: < {results['conclusion']['practical_upper_bound_percent']:.1f}%")
    
    return results


def save_results(results: dict, output_dir: Path):
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON summary
    json_path = output_dir / "ssz_schumann_realdata_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Saved: {json_path}")
    
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="SSZ Schumann Real Data Analysis"
    )
    parser.add_argument(
        '--station', 
        type=str, 
        default='sierra_nevada',
        help='Station identifier'
    )
    parser.add_argument(
        '--year', 
        type=int, 
        default=2013,
        help='Year to analyze'
    )
    parser.add_argument(
        '--month', 
        type=int, 
        default=10,
        help='Month to analyze (1-12), or 0 for all'
    )
    parser.add_argument(
        '--fit-eta', 
        action='store_true',
        help='Fit eta to data instead of using fixed value'
    )
    parser.add_argument(
        '--eta', 
        type=float, 
        default=0.74,
        help='Fixed eta value (if not fitting)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='output',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    month = args.month if args.month > 0 else None
    
    results = run_analysis(
        station=args.station,
        year=args.year,
        month=month,
        fit_eta=args.fit_eta,
        eta_fixed=args.eta,
    )
    
    save_results(results, Path(args.output_dir))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Print final summary
    if 'ssz' in results and 'chi_squared_reduced' in results['ssz']:
        chi2_red = results['ssz']['chi_squared_reduced']
        delta_global = results['ssz']['delta_ssz_global_percent']
        
        print(f"\nFINAL RESULT:")
        print(f"  delta_SSZ_global = {delta_global:.2f}%")
        print(f"  chi^2/ndof = {chi2_red:.1f}")
        
        if chi2_red > 3.0:
            print(f"\n  CONCLUSION: SSZ Minimalmodel REJECTED")
            print(f"              Classical dispersion dominates")
        else:
            print(f"\n  CONCLUSION: SSZ Minimalmodel not rejected")


if __name__ == "__main__":
    main()
