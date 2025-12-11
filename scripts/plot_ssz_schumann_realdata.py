#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Real Data Plots

Generates publication-quality plots from the SSZ analysis results.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_analysis.core import (
    compute_classical_frequencies,
    get_classical_reference,
    compute_delta_seg_classical,
)

OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")


def load_data(station: str, year: int, month: int) -> pd.DataFrame:
    """Load processed Schumann data."""
    processed_dir = DATA_DIR / "schumann" / "real" / "processed"
    pattern = f"schumann_{year % 100:02d}{month:02d}_processed.csv"
    filepath = processed_dir / pattern
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    return df


def plot_timeseries(df: pd.DataFrame, eta: float, output_path: Path):
    """Plot frequency time series with classical reference."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    classical_ref = get_classical_reference(eta)
    modes = ['f1', 'f2', 'f3', 'f4']
    classical_freqs = [classical_ref.f1, classical_ref.f2, classical_ref.f3, classical_ref.f4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (mode, f_class, color) in enumerate(zip(modes, classical_freqs, colors)):
        ax = axes[i]
        
        # Plot observed data
        ax.plot(df['timestamp'], df[mode], '.', color=color, alpha=0.3, markersize=2, label='Observed')
        
        # Daily mean
        daily = df.groupby('date')[mode].mean()
        daily_dates = pd.to_datetime(daily.index)
        ax.plot(daily_dates, daily.values, '-', color=color, linewidth=2, label='Daily mean')
        
        # Classical reference
        ax.axhline(f_class, color='black', linestyle='--', linewidth=1, label=f'Classical (eta={eta})')
        
        ax.set_ylabel(f'{mode} (Hz)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Annotate offset
        f_obs_mean = df[mode].mean()
        delta = compute_delta_seg_classical(f_obs_mean, f_class)
        ax.text(0.02, 0.95, f'delta = {delta*100:.1f}%', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top')
    
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    
    fig.suptitle('Schumann Resonance Frequencies - October 2013\nSierra Nevada ELF Station (Real Data)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_delta_comparison(df: pd.DataFrame, eta: float, output_path: Path):
    """Plot per-mode delta comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    classical_ref = get_classical_reference(eta)
    modes = ['f1', 'f2', 'f3', 'f4']
    
    # Calculate deltas
    deltas = []
    delta_errs = []
    for mode in modes:
        f_obs = df[mode].mean()
        f_std = df[mode].std()
        f_class = getattr(classical_ref, mode)
        delta = compute_delta_seg_classical(f_obs, f_class)
        delta_err = f_std / f_class
        deltas.append(delta * 100)
        delta_errs.append(delta_err * 100)
    
    # Bar plot
    ax1 = axes[0]
    x = np.arange(len(modes))
    bars = ax1.bar(x, deltas, yerr=delta_errs, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylabel('delta_seg_classical (%)')
    ax1.set_title('Per-Mode Offsets vs Classical Model')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (d, e) in enumerate(zip(deltas, delta_errs)):
        ax1.text(i, d + e + 0.3, f'{d:.1f}%', ha='center', fontsize=9)
    
    # Weighted mean line
    weights = 1.0 / np.array(delta_errs)**2
    delta_global = np.sum(weights * np.array(deltas)) / np.sum(weights)
    ax1.axhline(delta_global, color='red', linestyle='--', linewidth=2, label=f'Weighted mean: {delta_global:.1f}%')
    ax1.legend()
    
    # Mode number vs delta
    ax2 = axes[1]
    n_values = [1, 2, 3, 4]
    ax2.errorbar(n_values, deltas, yerr=delta_errs, fmt='o-', capsize=5, markersize=10, linewidth=2)
    ax2.set_xlabel('Mode number n')
    ax2.set_ylabel('delta_seg_classical (%)')
    ax2.set_title('Mode-Dependent Dispersion')
    ax2.set_xticks(n_values)
    ax2.grid(True, alpha=0.3)
    
    # Fit linear trend
    z = np.polyfit(n_values, deltas, 1)
    p = np.poly1d(z)
    ax2.plot(n_values, p(n_values), '--', color='gray', label=f'Linear fit: {z[0]:.1f}%/mode')
    ax2.legend()
    
    fig.suptitle(f'SSZ Hypothesis Test - eta = {eta}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_diurnal(df: pd.DataFrame, output_path: Path):
    """Plot diurnal variation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    modes = ['f1', 'f2', 'f3', 'f4']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (mode, color) in enumerate(zip(modes, colors)):
        ax = axes[i // 2, i % 2]
        
        # Hourly mean and std
        hourly = df.groupby('hour')[mode].agg(['mean', 'std'])
        
        ax.errorbar(hourly.index, hourly['mean'], yerr=hourly['std'], 
                   fmt='o-', capsize=3, color=color, markersize=6)
        ax.set_xlabel('Hour (UTC)')
        ax.set_ylabel(f'{mode} (Hz)')
        ax.set_title(f'{mode} Diurnal Variation')
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3)
        
        # Annotate range
        f_range = hourly['mean'].max() - hourly['mean'].min()
        ax.text(0.95, 0.95, f'Range: {f_range:.2f} Hz', 
                transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', horizontalalignment='right')
    
    fig.suptitle('Diurnal Variation of Schumann Resonances\nSierra Nevada, October 2013', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def plot_ssz_summary(results_path: Path, output_path: Path):
    """Plot SSZ hypothesis test summary."""
    with open(results_path) as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # 1. Per-mode deltas
    ax1 = axes[0]
    modes = ['f1', 'f2', 'f3', 'f4']
    deltas = [results['classical']['delta_seg'][m]['percent'] for m in modes]
    errors = [results['classical']['delta_seg'][m]['percent_err'] for m in modes]
    
    x = np.arange(len(modes))
    bars = ax1.bar(x, deltas, yerr=errors, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    
    # Global delta line
    delta_global = results['ssz']['delta_ssz_global_percent']
    ax1.axhline(delta_global, color='red', linestyle='--', linewidth=2, 
                label=f'Global: {delta_global:.1f}%')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylabel('delta_seg (%)')
    ax1.set_title('Per-Mode Offsets')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Chi-squared visualization
    ax2 = axes[1]
    chi2_red = results['ssz']['chi_squared_reduced']
    
    # Expected distribution
    x_chi2 = np.linspace(0, 10, 100)
    from scipy import stats
    y_chi2 = stats.chi2.pdf(x_chi2 * 3, 3)  # ndof=3
    ax2.fill_between(x_chi2, y_chi2, alpha=0.3, color='blue', label='Expected (ndof=3)')
    ax2.axvline(chi2_red, color='red', linewidth=3, label=f'Observed: {chi2_red:.1f}')
    ax2.axvline(3, color='green', linestyle='--', label='Threshold: 3')
    
    ax2.set_xlabel('chi^2 / ndof')
    ax2.set_ylabel('Probability density')
    ax2.set_title('Chi-Squared Test')
    ax2.set_xlim(0, 60)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Conclusion box
    ax3 = axes[2]
    ax3.axis('off')
    
    conclusion_text = f"""
SSZ HYPOTHESIS TEST RESULTS
===========================

Data: 744 hours (October 2013)
Station: Sierra Nevada ELF

CLASSICAL REFERENCE
eta = {results['classical']['eta']:.4f}

SSZ TEST
delta_SSZ_global = {results['ssz']['delta_ssz_global_percent']:.2f}% +/- {results['ssz']['delta_ssz_std_percent']:.2f}%
chi^2 = {results['ssz']['chi_squared']:.1f}
ndof = {results['ssz']['ndof']}
chi^2/ndof = {results['ssz']['chi_squared_reduced']:.1f}
p-value < 10^-4

CONCLUSION
SSZ Minimalmodel: REJECTED
Classical dispersion: DOMINATES
Upper bound: |delta_SSZ| < {results['conclusion']['practical_upper_bound_percent']:.1f}%
"""
    
    ax3.text(0.1, 0.9, conclusion_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('SSZ Schumann Real Data Analysis Summary', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SSZ Schumann plots")
    parser.add_argument('--station', type=str, default='sierra_nevada')
    parser.add_argument('--year', type=int, default=2013)
    parser.add_argument('--month', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.74)
    parser.add_argument('--output-dir', type=str, default='output')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SSZ SCHUMANN REAL DATA PLOTS")
    print("=" * 60)
    print()
    
    # Load data
    print("[1] Loading data...")
    df = load_data(args.station, args.year, args.month)
    print(f"    Loaded {len(df)} records")
    
    # Generate plots
    print("\n[2] Generating plots...")
    
    plot_timeseries(df, args.eta, output_dir / "ssz_realdata_timeseries.png")
    plot_delta_comparison(df, args.eta, output_dir / "ssz_realdata_deltas.png")
    plot_diurnal(df, output_dir / "ssz_realdata_diurnal.png")
    
    # Summary plot (if results exist)
    results_path = output_dir / "ssz_schumann_realdata_results.json"
    if results_path.exists():
        plot_ssz_summary(results_path, output_dir / "ssz_realdata_summary.png")
    
    print("\n[OK] All plots generated!")


if __name__ == "__main__":
    main()
