#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Resonance Analysis v2

Improved analysis with:
- Per-mode eta calibration
- Global delta_SSZ estimation with chi-squared test
- Proper error propagation
- Clear separation of classical vs SSZ results
- CLI interface

Usage:
    python run_ssz_analysis_v2.py
    python run_ssz_analysis_v2.py --input-csv data/schumann/real/processed/schumann_1310_processed.csv
    python run_ssz_analysis_v2.py --use-synthetic-proxies

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_analysis.core import (
    load_schumann_data,
    get_classical_reference,
    fit_classical_eta,
    compute_delta_seg_classical,
    compute_delta_seg_anomaly,
    estimate_delta_ssz_global,
    run_ssz_hypothesis_test,
    analyze_correlations,
    compute_mode_correlations,
    compute_ssz_score,
)

# Default paths
DEFAULT_INPUT = Path("data/schumann/real/processed/schumann_1310_processed.csv")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_F107_CSV = Path("data/solar/f107_201310_daily.csv")
DEFAULT_KP_CSV = Path("data/geomag/kp_201310_daily.csv")

# Reference values for validation (October 2013)
REFERENCE_VALUES = {
    'f1_mean': 8.055,
    'f2_mean': 14.754,
    'f3_mean': 21.036,
    'f4_mean': 27.554,
    'n_records': 744,
}
TOLERANCE_HZ = 0.2  # Hz


def load_proxy_data(f107_path: Path, kp_path: Path, use_synthetic: bool = False) -> pd.DataFrame:
    """
    Load F10.7 and Kp proxy data.
    
    Parameters
    ----------
    f107_path : Path
        Path to F10.7 CSV
    kp_path : Path
        Path to Kp CSV
    use_synthetic : bool
        If True, generate synthetic data instead of loading from files
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: date, f107, kp
    """
    if use_synthetic:
        print("    [WARN] Using SYNTHETIC proxy data - NOT for physical interpretation!")
        # Generate synthetic data for October 2013
        np.random.seed(42)
        days = pd.date_range('2013-10-01', periods=31, freq='D')
        f107_monthly = 132.86  # Oct 2013 average
        
        df = pd.DataFrame({
            'date': days.date,
            'f107': f107_monthly + np.random.normal(0, 13, len(days)),
            'kp': np.clip(np.random.exponential(2.5, len(days)), 0, 9),
        })
        df['proxy_source'] = 'SYNTHETIC'
        return df
    
    # Try to load real data
    f107_df = None
    kp_df = None
    
    if f107_path.exists():
        print(f"    Loading F10.7 from: {f107_path}")
        f107_df = pd.read_csv(f107_path)
        f107_df['date'] = pd.to_datetime(f107_df['date']).dt.date
    else:
        print(f"    [WARN] F10.7 file not found: {f107_path}")
    
    if kp_path.exists():
        print(f"    Loading Kp from: {kp_path}")
        kp_df = pd.read_csv(kp_path)
        kp_df['date'] = pd.to_datetime(kp_df['date']).dt.date
    else:
        print(f"    [WARN] Kp file not found: {kp_path}")
    
    if f107_df is None and kp_df is None:
        raise FileNotFoundError(
            f"No proxy data found. Expected:\n"
            f"  - {f107_path}\n"
            f"  - {kp_path}\n"
            f"Use --use-synthetic-proxies for demo mode."
        )
    
    # Merge
    if f107_df is not None and kp_df is not None:
        df = f107_df.merge(kp_df, on='date', how='outer')
    elif f107_df is not None:
        df = f107_df
        df['kp'] = np.nan
    else:
        df = kp_df
        df['f107'] = np.nan
    
    df['proxy_source'] = 'REAL'
    return df


def validate_data(schumann: pd.DataFrame) -> bool:
    """Validate that data matches expected reference values."""
    valid = True
    
    for mode in ['f1', 'f2', 'f3', 'f4']:
        mean_val = schumann[mode].mean()
        ref_val = REFERENCE_VALUES[f'{mode}_mean']
        diff = abs(mean_val - ref_val)
        
        if diff > TOLERANCE_HZ:
            print(f"    [WARN] {mode} mean ({mean_val:.3f} Hz) differs from reference "
                  f"({ref_val:.3f} Hz) by {diff:.3f} Hz")
            valid = False
    
    n_records = len(schumann)
    if n_records != REFERENCE_VALUES['n_records']:
        print(f"    [WARN] Record count ({n_records}) differs from expected "
              f"({REFERENCE_VALUES['n_records']})")
        # Not a hard failure
    
    return valid


def main():
    parser = argparse.ArgumentParser(
        description='SSZ Schumann Resonance Analysis v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_ssz_analysis_v2.py
    python run_ssz_analysis_v2.py --input-csv data/schumann/real/processed/schumann_1310_processed.csv
    python run_ssz_analysis_v2.py --use-synthetic-proxies
    python run_ssz_analysis_v2.py --eta 0.76
        """
    )
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT,
                        help='Input CSV with processed Schumann data')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--f107-csv', type=Path, default=DEFAULT_F107_CSV,
                        help='F10.7 proxy data CSV')
    parser.add_argument('--kp-csv', type=Path, default=DEFAULT_KP_CSV,
                        help='Kp proxy data CSV')
    parser.add_argument('--use-synthetic-proxies', action='store_true',
                        help='Use synthetic F10.7/Kp data (for demo only)')
    parser.add_argument('--eta', type=float, default=0.74,
                        help='Classical eta parameter (default: 0.74)')
    parser.add_argument('--fit-eta', action='store_true',
                        help='Fit optimal eta from data')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SSZ SCHUMANN RESONANCE ANALYSIS v2")
    print("=" * 70)
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # [1] Load Schumann data
    # =========================================================================
    print("[1] Loading Schumann data...")
    print(f"    Source: {args.input_csv}")
    
    if not args.input_csv.exists():
        print(f"    [ERROR] Input file not found: {args.input_csv}")
        print("    Run process_one_month.py first to extract Schumann frequencies.")
        sys.exit(1)
    
    schumann = load_schumann_data(args.input_csv)
    print(f"    Records: {len(schumann)}")
    print(f"    Date range: {schumann['timestamp'].min()} to {schumann['timestamp'].max()}")
    
    # Validate
    print("\n    Validating data against reference values...")
    validate_data(schumann)
    
    # =========================================================================
    # [2] Load proxy data
    # =========================================================================
    print("\n[2] Loading space weather proxy data...")
    try:
        proxies = load_proxy_data(args.f107_csv, args.kp_csv, args.use_synthetic_proxies)
        proxy_source = proxies['proxy_source'].iloc[0]
        print(f"    Proxy source: {proxy_source}")
    except FileNotFoundError as e:
        print(f"    [ERROR] {e}")
        sys.exit(1)
    
    # =========================================================================
    # [3] Compute daily aggregates
    # =========================================================================
    print("\n[3] Computing daily aggregates...")
    
    daily = schumann.groupby('date').agg({
        'f1': ['mean', 'std', 'count'],
        'f2': ['mean', 'std', 'count'],
        'f3': ['mean', 'std', 'count'],
        'f4': ['mean', 'std', 'count'],
    }).reset_index()
    
    # Flatten column names
    daily.columns = ['date', 
                     'f1_mean', 'f1_std', 'f1_n',
                     'f2_mean', 'f2_std', 'f2_n',
                     'f3_mean', 'f3_std', 'f3_n',
                     'f4_mean', 'f4_std', 'f4_n']
    
    # Compute standard error of mean
    for mode in ['f1', 'f2', 'f3', 'f4']:
        daily[f'{mode}_err'] = daily[f'{mode}_std'] / np.sqrt(daily[f'{mode}_n'])
    
    # Merge with proxies
    daily = daily.merge(proxies[['date', 'f107', 'kp']], on='date', how='left')
    
    print(f"    Daily records: {len(daily)}")
    
    # =========================================================================
    # [4] Classical reference frequencies
    # =========================================================================
    print("\n[4] Classical reference frequencies...")
    
    if args.fit_eta:
        print("    Fitting optimal eta from data...")
        f_obs = {
            'f1': daily['f1_mean'].mean(),
            'f2': daily['f2_mean'].mean(),
            'f3': daily['f3_mean'].mean(),
            'f4': daily['f4_mean'].mean(),
        }
        eta_best, eta_err = fit_classical_eta(f_obs, method='least_squares')
        print(f"    Best-fit eta = {eta_best:.4f} +/- {eta_err:.4f}")
        classical_ref = get_classical_reference(eta_best)
    else:
        eta_best = args.eta
        classical_ref = get_classical_reference(args.eta)
        print(f"    Using fixed eta = {args.eta}")
    
    print(f"    f1_classical = {classical_ref.f1:.3f} Hz")
    print(f"    f2_classical = {classical_ref.f2:.3f} Hz")
    print(f"    f3_classical = {classical_ref.f3:.3f} Hz")
    print(f"    f4_classical = {classical_ref.f4:.3f} Hz")
    
    # Add classical reference to daily data
    daily['f1_classical'] = classical_ref.f1
    daily['f2_classical'] = classical_ref.f2
    daily['f3_classical'] = classical_ref.f3
    daily['f4_classical'] = classical_ref.f4
    
    # =========================================================================
    # [5] Compute delta_seg (classical offset)
    # =========================================================================
    print("\n[5] Computing delta_seg_classical (offset vs classical)...")
    
    for mode in ['f1', 'f2', 'f3', 'f4']:
        f_class = getattr(classical_ref, mode)
        daily[f'delta_{mode}_classical'] = compute_delta_seg_classical(
            daily[f'{mode}_mean'], f_class
        )
        # Error propagation
        daily[f'delta_{mode}_classical_err'] = daily[f'{mode}_err'] / f_class
    
    # Print summary
    for mode in ['f1', 'f2', 'f3', 'f4']:
        mean_delta = daily[f'delta_{mode}_classical'].mean()
        std_delta = daily[f'delta_{mode}_classical'].std()
        print(f"    delta_{mode}_classical: {mean_delta*100:.2f}% +/- {std_delta*100:.2f}%")
    
    # =========================================================================
    # [6] Compute delta_seg_anomaly (for correlations)
    # =========================================================================
    print("\n[6] Computing delta_seg_anomaly (deviation from mode mean)...")
    
    for mode in ['f1', 'f2', 'f3', 'f4']:
        f_mean = daily[f'{mode}_mean'].mean()
        daily[f'delta_{mode}_anomaly'] = compute_delta_seg_anomaly(
            daily[f'{mode}_mean'].values, f_mean
        )
    
    # Average anomaly across modes
    daily['delta_avg_anomaly'] = (
        daily['delta_f1_anomaly'] + daily['delta_f2_anomaly'] +
        daily['delta_f3_anomaly'] + daily['delta_f4_anomaly']
    ) / 4
    
    # =========================================================================
    # [7] SSZ hypothesis test
    # =========================================================================
    print("\n[7] SSZ hypothesis test...")
    
    ssz_result = run_ssz_hypothesis_test(daily, classical_ref)
    
    print(f"    Global delta_SSZ = {ssz_result.delta_ssz_global*100:.3f}% +/- {ssz_result.delta_ssz_std*100:.3f}%")
    print(f"    Chi-squared = {ssz_result.chi_squared:.2f} (ndof={ssz_result.ndof})")
    print(f"    Reduced chi-squared = {ssz_result.chi_squared_reduced:.2f}")
    print(f"    P-value = {ssz_result.p_value:.4f}")
    print(f"    95% CL upper bound: |delta_SSZ| < {ssz_result.upper_bound_95*100:.2f}%")
    
    if ssz_result.is_consistent:
        print("    --> CONSISTENT with SSZ prediction")
    else:
        print("    --> Mode-dependent shift detected (classical dispersion dominates)")
    
    # =========================================================================
    # [8] Correlation analysis
    # =========================================================================
    print("\n[8] Correlation analysis...")
    
    # Add delta_ssz_anomaly column for correlation analysis
    daily['delta_ssz_anomaly'] = daily['delta_avg_anomaly']
    
    correlations = analyze_correlations(daily, 'delta_ssz_anomaly', ['f107', 'kp'])
    
    for proxy, result in correlations.items():
        print(f"    Corr(delta_SSZ_anomaly, {proxy}): r = {result['r']:.3f}, p = {result['p']:.3f}, n = {result['n']}")
    
    # Mode correlations
    mean_mode_corr, corr_matrix = compute_mode_correlations(daily)
    print(f"    Mean inter-mode correlation: {mean_mode_corr:.3f}")
    
    # SSZ score
    mode_spread = np.std(list(ssz_result.mode_deltas.values()))
    mean_delta = np.mean(list(ssz_result.mode_deltas.values()))
    ssz_score = compute_ssz_score(mean_mode_corr, mode_spread, mean_delta)
    print(f"    SSZ consistency score: {ssz_score:.3f}")
    
    # =========================================================================
    # [9] Diurnal analysis
    # =========================================================================
    print("\n[9] Diurnal variation analysis...")
    
    hourly = schumann.groupby('hour').agg({
        'f1': ['mean', 'std'],
        'f2': ['mean', 'std'],
        'f3': ['mean', 'std'],
        'f4': ['mean', 'std'],
    }).reset_index()
    hourly.columns = ['hour', 'f1_mean', 'f1_std', 'f2_mean', 'f2_std',
                      'f3_mean', 'f3_std', 'f4_mean', 'f4_std']
    
    for mode in ['f1', 'f2', 'f3', 'f4']:
        diurnal_range = hourly[f'{mode}_mean'].max() - hourly[f'{mode}_mean'].min()
        print(f"    {mode} diurnal range: {diurnal_range:.3f} Hz")
    
    # =========================================================================
    # [10] Generate plots
    # =========================================================================
    print("\n[10] Generating plots...")
    
    # Figure 1: Time series with classical reference
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    dates = pd.to_datetime(daily['date'])
    
    for i, mode in enumerate(['f1', 'f2']):
        ax = axes[i]
        f_class = getattr(classical_ref, mode)
        ax.errorbar(dates, daily[f'{mode}_mean'], yerr=daily[f'{mode}_std'],
                    fmt='o-', markersize=4, capsize=2, label=f'{mode} observed')
        ax.axhline(f_class, color='r', linestyle='--', 
                   label=f'{mode} classical ({f_class:.2f} Hz)')
        ax.set_ylabel(f'{mode} (Hz)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f'Schumann Resonance Frequencies - October 2013 (eta={eta_best:.3f})')
    
    # Delta plot
    ax = axes[2]
    for mode, color in zip(['f1', 'f2', 'f3', 'f4'], ['blue', 'green', 'orange', 'red']):
        ax.plot(dates, daily[f'delta_{mode}_classical'] * 100, 'o-', 
                markersize=3, alpha=0.7, label=f'delta_{mode}', color=color)
    ax.axhline(ssz_result.delta_ssz_global * 100, color='black', linestyle='-', 
               linewidth=2, label=f'delta_SSZ global ({ssz_result.delta_ssz_global*100:.1f}%)')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('delta_seg_classical (%)')
    ax.legend(loc='upper right', ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # F10.7
    ax = axes[3]
    if 'f107' in daily.columns and daily['f107'].notna().any():
        ax.plot(dates, daily['f107'], 's-', color='orange', markersize=4, label='F10.7')
        ax.set_ylabel('F10.7 (sfu)')
        if proxy_source == 'SYNTHETIC':
            ax.set_title('F10.7 [SYNTHETIC - NOT FOR PHYSICAL INTERPRETATION]', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'F10.7 data not available', transform=ax.transAxes, ha='center')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Kp
    ax = axes[4]
    if 'kp' in daily.columns and daily['kp'].notna().any():
        ax.bar(dates, daily['kp'], color='purple', alpha=0.7, label='Kp index')
        ax.set_ylabel('Kp')
        if proxy_source == 'SYNTHETIC':
            ax.set_title('Kp [SYNTHETIC - NOT FOR PHYSICAL INTERPRETATION]', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Kp data not available', transform=ax.transAxes, ha='center')
    ax.set_xlabel('Date')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output_dir / 'ssz_analysis_timeseries.png', dpi=150)
    print(f"    Saved: {args.output_dir / 'ssz_analysis_timeseries.png'}")
    
    # Figure 2: SSZ consistency test
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Per-mode deltas with error bars
    ax = axes[0, 0]
    modes = ['f1', 'f2', 'f3', 'f4']
    deltas = [ssz_result.mode_deltas[m] * 100 for m in modes]
    errors = [ssz_result.mode_delta_errors[m] * 100 for m in modes]
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = ax.bar(modes, deltas, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax.axhline(ssz_result.delta_ssz_global * 100, color='black', linestyle='-', 
               linewidth=2, label=f'Global delta_SSZ = {ssz_result.delta_ssz_global*100:.2f}%')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_ylabel('delta_seg_classical (%)')
    ax.set_title(f'SSZ Consistency Test (chi2/ndof = {ssz_result.chi_squared_reduced:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation with F10.7
    ax = axes[0, 1]
    if 'f107' in daily.columns and daily['f107'].notna().any():
        ax.scatter(daily['f107'], daily['delta_ssz_anomaly'] * 100, alpha=0.7, color='red')
        r = correlations.get('f107', {}).get('r', np.nan)
        ax.set_xlabel('F10.7 (sfu)')
        ax.set_ylabel('delta_SSZ_anomaly (%)')
        ax.set_title(f'delta_SSZ_anomaly vs F10.7 (r={r:.3f})')
        if proxy_source == 'SYNTHETIC':
            ax.text(0.5, 0.02, '[SYNTHETIC DATA]', transform=ax.transAxes, 
                    ha='center', fontsize=8, color='red')
    ax.grid(True, alpha=0.3)
    
    # Correlation with Kp
    ax = axes[1, 0]
    if 'kp' in daily.columns and daily['kp'].notna().any():
        ax.scatter(daily['kp'], daily['delta_ssz_anomaly'] * 100, alpha=0.7, color='purple')
        r = correlations.get('kp', {}).get('r', np.nan)
        ax.set_xlabel('Kp index')
        ax.set_ylabel('delta_SSZ_anomaly (%)')
        ax.set_title(f'delta_SSZ_anomaly vs Kp (r={r:.3f})')
        if proxy_source == 'SYNTHETIC':
            ax.text(0.5, 0.02, '[SYNTHETIC DATA]', transform=ax.transAxes,
                    ha='center', fontsize=8, color='red')
    ax.grid(True, alpha=0.3)
    
    # Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
SSZ HYPOTHESIS TEST SUMMARY
===========================

Global delta_SSZ: {ssz_result.delta_ssz_global*100:.3f}% +/- {ssz_result.delta_ssz_std*100:.3f}%

Chi-squared test:
  chi2 = {ssz_result.chi_squared:.2f}
  ndof = {ssz_result.ndof}
  chi2/ndof = {ssz_result.chi_squared_reduced:.2f}
  p-value = {ssz_result.p_value:.4f}

95% CL upper bound: |delta_SSZ| < {ssz_result.upper_bound_95*100:.2f}%

Mode spread: {mode_spread*100:.2f}%
Mean inter-mode correlation: {mean_mode_corr:.3f}
SSZ consistency score: {ssz_score:.3f}

Result: {'SSZ CONSISTENT' if ssz_result.is_consistent else 'CLASSICAL DISPERSION DOMINATES'}

Proxy data: {proxy_source}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(args.output_dir / 'ssz_analysis_correlations.png', dpi=150)
    print(f"    Saved: {args.output_dir / 'ssz_analysis_correlations.png'}")
    
    # Figure 3: Diurnal variation
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for i, (mode, ax) in enumerate(zip(['f1', 'f2', 'f3', 'f4'], axes.flat)):
        ax.errorbar(hourly['hour'], hourly[f'{mode}_mean'], yerr=hourly[f'{mode}_std'],
                    fmt='o-', markersize=6, capsize=3)
        ax.set_xlabel('Hour (UTC)')
        ax.set_ylabel(f'{mode} (Hz)')
        ax.set_title(f'{mode} Diurnal Variation')
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output_dir / 'ssz_analysis_diurnal.png', dpi=150)
    print(f"    Saved: {args.output_dir / 'ssz_analysis_diurnal.png'}")
    
    # =========================================================================
    # [11] Save results
    # =========================================================================
    print("\n[11] Saving results...")
    
    daily.to_csv(args.output_dir / 'ssz_analysis_daily.csv', index=False)
    hourly.to_csv(args.output_dir / 'ssz_analysis_hourly.csv', index=False)
    
    # Summary JSON
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'analysis_version': '2.0',
        'input_file': str(args.input_csv),
        'proxy_source': proxy_source,
        'data_period': 'October 2013',
        'n_records': len(schumann),
        'n_days': len(daily),
        
        # Classical reference
        'classical': {
            'eta': float(eta_best),
            'f1_classical': float(classical_ref.f1),
            'f2_classical': float(classical_ref.f2),
            'f3_classical': float(classical_ref.f3),
            'f4_classical': float(classical_ref.f4),
        },
        
        # Observed frequencies
        'observed': {
            'f1_mean': float(daily['f1_mean'].mean()),
            'f1_std': float(daily['f1_mean'].std()),
            'f2_mean': float(daily['f2_mean'].mean()),
            'f2_std': float(daily['f2_mean'].std()),
            'f3_mean': float(daily['f3_mean'].mean()),
            'f3_std': float(daily['f3_mean'].std()),
            'f4_mean': float(daily['f4_mean'].mean()),
            'f4_std': float(daily['f4_mean'].std()),
        },
        
        # Delta_seg_classical (offset vs classical)
        'delta_seg_classical': {
            'f1': float(ssz_result.mode_deltas['f1']),
            'f1_err': float(ssz_result.mode_delta_errors['f1']),
            'f2': float(ssz_result.mode_deltas['f2']),
            'f2_err': float(ssz_result.mode_delta_errors['f2']),
            'f3': float(ssz_result.mode_deltas['f3']),
            'f3_err': float(ssz_result.mode_delta_errors['f3']),
            'f4': float(ssz_result.mode_deltas['f4']),
            'f4_err': float(ssz_result.mode_delta_errors['f4']),
        },
        
        # SSZ hypothesis test
        'ssz_test': {
            'delta_ssz_global': float(ssz_result.delta_ssz_global),
            'delta_ssz_std': float(ssz_result.delta_ssz_std),
            'chi_squared': float(ssz_result.chi_squared),
            'ndof': int(ssz_result.ndof),
            'chi_squared_reduced': float(ssz_result.chi_squared_reduced),
            'p_value': float(ssz_result.p_value),
            'upper_bound_95_percent': float(ssz_result.upper_bound_95),
            'is_consistent': bool(ssz_result.is_consistent),
            'mode_spread': float(mode_spread),
            'mean_mode_correlation': float(mean_mode_corr) if not np.isnan(mean_mode_corr) else None,
            'ssz_score': float(ssz_score),
        },
        
        # Correlations
        'correlations': {
            'f107': {
                'r': float(correlations['f107']['r']) if not np.isnan(correlations['f107']['r']) else None,
                'p': float(correlations['f107']['p']) if not np.isnan(correlations['f107']['p']) else None,
            },
            'kp': {
                'r': float(correlations['kp']['r']) if not np.isnan(correlations['kp']['r']) else None,
                'p': float(correlations['kp']['p']) if not np.isnan(correlations['kp']['p']) else None,
            },
        },
    }
    
    with open(args.output_dir / 'ssz_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {args.output_dir / 'ssz_analysis_summary.json'}")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"""
CLASSICAL RESULTS
-----------------
eta = {eta_best:.4f}
f1: observed = {daily['f1_mean'].mean():.3f} Hz, classical = {classical_ref.f1:.3f} Hz
f2: observed = {daily['f2_mean'].mean():.3f} Hz, classical = {classical_ref.f2:.3f} Hz
f3: observed = {daily['f3_mean'].mean():.3f} Hz, classical = {classical_ref.f3:.3f} Hz
f4: observed = {daily['f4_mean'].mean():.3f} Hz, classical = {classical_ref.f4:.3f} Hz

Per-mode offsets (delta_seg_classical):
  f1: {ssz_result.mode_deltas['f1']*100:+.2f}%
  f2: {ssz_result.mode_deltas['f2']*100:+.2f}%
  f3: {ssz_result.mode_deltas['f3']*100:+.2f}%
  f4: {ssz_result.mode_deltas['f4']*100:+.2f}%

SSZ HYPOTHESIS TEST
-------------------
Global delta_SSZ = {ssz_result.delta_ssz_global*100:.3f}% +/- {ssz_result.delta_ssz_std*100:.3f}%
Chi-squared/ndof = {ssz_result.chi_squared_reduced:.2f}
P-value = {ssz_result.p_value:.4f}

95% CL upper bound: |delta_SSZ| < {ssz_result.upper_bound_95*100:.2f}%

CONCLUSION: {'SSZ CONSISTENT' if ssz_result.is_consistent else 'CLASSICAL DISPERSION DOMINATES'}
            {'(mode-independent shift)' if ssz_result.is_consistent else '(mode-dependent shifts detected)'}

NOTE: Proxy data source = {proxy_source}
      {'Results with synthetic proxies are for demonstration only!' if proxy_source == 'SYNTHETIC' else ''}
""")
    
    print("[OK] Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
