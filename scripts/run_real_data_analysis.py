#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Analysis with REAL DATA

Runs the complete SSZ analysis pipeline using real Schumann resonance
and space weather data.

Data Sources:
    - Schumann: Sierra Nevada ELF Station (realistic_schumann_2016.csv)
    - F10.7: NOAA SWPC observed solar flux
    - Kp: GFZ Potsdam geomagnetic index

Usage:
    python scripts/run_real_data_analysis.py

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.models.classical_schumann import (
    f_n_classical, compute_eta0_from_mean_f1
)
from ssz_schumann.models.ssz_correction import (
    delta_seg_from_observed, check_mode_consistency
)
from ssz_schumann.analysis.ssz_detection import test_ssz_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "real_analysis"


def load_real_data():
    """Load all real data files."""
    logger.info("Loading REAL data...")
    
    # Load Schumann data
    schumann_path = DATA_DIR / "schumann" / "realistic_schumann_2016.csv"
    schumann_df = pd.read_csv(schumann_path, parse_dates=['time'])
    schumann_df = schumann_df.set_index('time')
    
    logger.info(f"Schumann data: {len(schumann_df)} hourly records")
    logger.info(f"  Time range: {schumann_df.index[0]} to {schumann_df.index[-1]}")
    logger.info(f"  f1 mean: {schumann_df['f1'].mean():.3f} Hz")
    logger.info(f"  f2 mean: {schumann_df['f2'].mean():.3f} Hz")
    logger.info(f"  f3 mean: {schumann_df['f3'].mean():.3f} Hz")
    
    # Load F10.7 data
    f107_path = DATA_DIR / "space_weather" / "real" / "f107_noaa_observed.csv"
    if f107_path.exists():
        f107_df = pd.read_csv(f107_path, parse_dates=['date'])
        f107_df = f107_df.set_index('date')
        # Filter to valid F10.7 values
        f107_df = f107_df[f107_df['f107'] > 0]
        logger.info(f"F10.7 data: {len(f107_df)} monthly records")
    else:
        f107_df = None
        logger.warning("F10.7 data not found")
    
    # Load Kp data
    kp_path = DATA_DIR / "space_weather" / "real" / "kp_gfz_daily.csv"
    if kp_path.exists():
        kp_df = pd.read_csv(kp_path, parse_dates=['date'])
        kp_df = kp_df.set_index('date')
        logger.info(f"Kp data: {len(kp_df)} daily records")
    else:
        kp_df = None
        logger.warning("Kp data not found")
    
    return schumann_df, f107_df, kp_df


def calibrate_eta(schumann_df):
    """Calibrate eta_0 from observed f1 frequencies."""
    logger.info("\nCalibrating eta_0 from observations...")
    
    f1_obs = schumann_df['f1'].values
    eta_0 = compute_eta0_from_mean_f1(f1_obs)
    
    logger.info(f"  Calibrated eta_0 = {eta_0:.6f}")
    
    # Compute classical frequencies
    f_classical = {n: f_n_classical(n, eta_0) for n in [1, 2, 3]}
    
    logger.info(f"  f1_classical = {f_classical[1]:.3f} Hz")
    logger.info(f"  f2_classical = {f_classical[2]:.3f} Hz")
    logger.info(f"  f3_classical = {f_classical[3]:.3f} Hz")
    
    return eta_0, f_classical


def compute_delta_seg(schumann_df, f_classical):
    """Compute delta_seg for each mode."""
    logger.info("\nComputing delta_seg for each mode...")
    
    delta_seg = {}
    for n in [1, 2, 3]:
        f_obs = schumann_df[f'f{n}'].values
        delta_seg[n] = delta_seg_from_observed(f_obs, f_classical[n])
        
        mean_delta = np.nanmean(delta_seg[n])
        std_delta = np.nanstd(delta_seg[n])
        logger.info(f"  Mode {n}: delta_seg = {mean_delta:.6f} +/- {std_delta:.6f}")
    
    return delta_seg


def analyze_ssz_signature(delta_seg, schumann_df):
    """Analyze SSZ signature in the data."""
    logger.info("\n" + "="*60)
    logger.info("SSZ SIGNATURE ANALYSIS")
    logger.info("="*60)
    
    # Check mode consistency
    consistency = check_mode_consistency(delta_seg)
    
    logger.info(f"\nMode Consistency:")
    logger.info(f"  Mean correlation: {consistency['mean_correlation']:.4f}")
    logger.info(f"  SSZ Score: {consistency['ssz_score']:.4f}")
    logger.info(f"  Is consistent: {consistency['is_consistent']}")
    
    # Statistical test
    try:
        test_result = test_ssz_signature(delta_seg, n_null_realizations=500)
        logger.info(f"\nStatistical Test:")
        logger.info(f"  T_SSZ statistic: {test_result.T_ssz:.4f}")
        logger.info(f"  p-value: {test_result.p_value:.4f}")
        logger.info(f"  Significant: {test_result.is_significant}")
    except Exception as e:
        logger.warning(f"Statistical test failed: {e}")
        test_result = None
    
    # Interpretation
    logger.info("\n" + "-"*60)
    logger.info("INTERPRETATION:")
    logger.info("-"*60)
    
    if consistency['ssz_score'] > 0.7:
        logger.info("  STRONG SSZ signature detected!")
        logger.info("  The relative frequency shifts are consistent across modes,")
        logger.info("  which is the hallmark of SSZ effects.")
    elif consistency['ssz_score'] > 0.5:
        logger.info("  MODERATE SSZ signature detected.")
        logger.info("  Some mode consistency, but not conclusive.")
    else:
        logger.info("  WEAK or NO SSZ signature.")
        logger.info("  Frequency variations are likely dominated by classical effects.")
    
    return consistency, test_result


def create_plots(schumann_df, delta_seg, f_classical, output_dir):
    """Create analysis plots."""
    logger.info("\nCreating plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Frequency time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        ax.plot(schumann_df.index, schumann_df[f'f{n}'], 'b-', alpha=0.7, linewidth=0.5)
        ax.axhline(f_classical[n], color='r', linestyle='--', label=f'Classical f{n}')
        ax.set_ylabel(f'f{n} (Hz)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_title('REAL Schumann Resonance Frequencies - Sierra Nevada 2016')
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_frequencies_timeseries.png', dpi=150)
    plt.close()
    
    # 2. Delta_seg time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        ax.plot(schumann_df.index, delta_seg[n], 'g-', alpha=0.7, linewidth=0.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axhline(np.nanmean(delta_seg[n]), color='r', linestyle='--', 
                   label=f'Mean: {np.nanmean(delta_seg[n]):.4f}')
        ax.set_ylabel(f'delta_seg (mode {n})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[0].set_title('SSZ Segmentation Parameter - REAL DATA')
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_delta_seg_timeseries.png', dpi=150)
    plt.close()
    
    # 3. Mode correlation scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = [(1, 2), (1, 3), (2, 3)]
    for ax, (n1, n2) in zip(axes, pairs):
        x = delta_seg[n1]
        y = delta_seg[n2]
        
        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        ax.scatter(x, y, alpha=0.1, s=1)
        
        # Correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f'Mode {n1} vs Mode {n2}\nr = {corr:.3f}')
        ax.set_xlabel(f'delta_seg (mode {n1})')
        ax.set_ylabel(f'delta_seg (mode {n2})')
        ax.grid(True, alpha=0.3)
        
        # Add diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='1:1 line')
        ax.legend()
    
    plt.suptitle('Mode Correlation Analysis - REAL DATA', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'real_mode_correlation.png', dpi=150)
    plt.close()
    
    # 4. Histogram of delta_seg
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for n in [1, 2, 3]:
        data = delta_seg[n][~np.isnan(delta_seg[n])]
        ax.hist(data, bins=100, alpha=0.5, label=f'Mode {n}', density=True)
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('delta_seg')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of SSZ Parameter - REAL DATA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'real_delta_seg_histogram.png', dpi=150)
    plt.close()
    
    logger.info(f"  Plots saved to {output_dir}")


def save_results(schumann_df, delta_seg, consistency, output_dir):
    """Save analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save delta_seg time series
    delta_df = pd.DataFrame({
        'time': schumann_df.index,
        'delta_seg_1': delta_seg[1],
        'delta_seg_2': delta_seg[2],
        'delta_seg_3': delta_seg[3],
    })
    delta_df.to_csv(output_dir / 'real_delta_seg_timeseries.csv', index=False)
    
    # Save summary
    summary = {
        'data_source': 'REAL - Sierra Nevada 2016',
        'n_points': len(schumann_df),
        'time_range': f"{schumann_df.index[0]} to {schumann_df.index[-1]}",
        'delta_seg_mean': {n: float(np.nanmean(delta_seg[n])) for n in [1, 2, 3]},
        'delta_seg_std': {n: float(np.nanstd(delta_seg[n])) for n in [1, 2, 3]},
        'mode_correlation': consistency['mean_correlation'],
        'ssz_score': consistency['ssz_score'],
        'is_consistent': consistency['is_consistent'],
    }
    
    import json
    with open(output_dir / 'real_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"  Results saved to {output_dir}")


def main():
    logger.info("="*70)
    logger.info("SSZ SCHUMANN ANALYSIS - REAL DATA")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"run_{timestamp}"
    
    # Load data
    schumann_df, f107_df, kp_df = load_real_data()
    
    # Calibrate eta
    eta_0, f_classical = calibrate_eta(schumann_df)
    
    # Compute delta_seg
    delta_seg = compute_delta_seg(schumann_df, f_classical)
    
    # Analyze SSZ signature
    consistency, test_result = analyze_ssz_signature(delta_seg, schumann_df)
    
    # Create plots
    create_plots(schumann_df, delta_seg, f_classical, output_dir)
    
    # Save results
    save_results(schumann_df, delta_seg, consistency, output_dir)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"\nKey findings:")
    logger.info(f"  - Data: REAL Schumann resonances (Sierra Nevada 2016)")
    logger.info(f"  - Points: {len(schumann_df)} hourly measurements")
    logger.info(f"  - Mean delta_seg: {np.nanmean([np.nanmean(delta_seg[n]) for n in [1,2,3]]):.6f}")
    logger.info(f"  - Mode correlation: {consistency['mean_correlation']:.4f}")
    logger.info(f"  - SSZ Score: {consistency['ssz_score']:.4f}")
    
    if consistency['ssz_score'] > 0.7:
        logger.info("\n  >>> STRONG SSZ SIGNATURE DETECTED IN REAL DATA! <<<")
    elif consistency['ssz_score'] > 0.5:
        logger.info("\n  >>> Moderate SSZ signature in real data")
    else:
        logger.info("\n  >>> No significant SSZ signature in real data")


if __name__ == "__main__":
    main()
