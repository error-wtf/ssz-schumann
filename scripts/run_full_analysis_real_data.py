#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data SSZ Schumann Resonance Analysis

This script runs the complete SSZ analysis on real Schumann resonance data
from the Sierra Nevada ELF station (Salinas et al., 2013-2017).

Usage:
    python run_full_analysis_real_data.py
    python run_full_analysis_real_data.py --schumann-path data/schumann/
    python run_full_analysis_real_data.py --use-sample  # Use sample data for testing

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.data_io.schumann_real import (
    load_schumann_real_data,
    create_sample_real_data,
    get_data_summary,
)
from ssz_schumann.data_io.space_weather_noaa import (
    create_synthetic_space_weather,
    normalize_index as normalize_series,
)
from ssz_schumann.analysis.model_fits import (
    calibrate_eta_from_data,
    compute_classical_frequencies,
    compute_delta_seg,
    compute_mode_consistency,
    fit_global_ssz_model,
    fit_layered_ssz_model,
    compute_proxy_correlations,
    generate_interpretation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_output_directories(base_dir: Path) -> dict:
    """Create output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"realdata_run_{timestamp}"
    
    dirs = {
        'run': run_dir,
        'plots': run_dir / 'plots',
        'reports': run_dir / 'reports',
        'data': run_dir / 'data',
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    return dirs


def load_and_prepare_data(args) -> pd.DataFrame:
    """Load and prepare all data sources."""
    
    # Load Schumann data
    if args.use_sample:
        logger.info("Using sample data for testing...")
        sample_path = Path("data/schumann/sample_real_2016.csv")
        
        if not sample_path.exists():
            logger.info("Creating sample data...")
            schumann_df = create_sample_real_data(sample_path, n_days=90)
        else:
            schumann_df = load_schumann_real_data(sample_path)
    else:
        schumann_path = Path(args.schumann_path)
        if not schumann_path.exists():
            raise FileNotFoundError(f"Schumann data not found: {schumann_path}")
        schumann_df = load_schumann_real_data(schumann_path)
    
    logger.info(f"Loaded {len(schumann_df)} Schumann data points")
    
    # Get date range
    start_date = schumann_df.index.min().strftime("%Y-%m-%d")
    end_date = schumann_df.index.max().strftime("%Y-%m-%d")
    
    # Load or create space weather data
    logger.info("Loading space weather data...")
    f107_series, kp_series = create_synthetic_space_weather(
        start=start_date,
        end=end_date,
    )
    
    # Normalize proxies
    f107_norm = normalize_series(f107_series, method='zscore')
    kp_norm = normalize_series(kp_series, method='zscore')
    
    # Merge data
    logger.info("Merging data sources...")
    
    # Resample Schumann to hourly for merging
    schumann_hourly = schumann_df.resample('1h').mean()
    
    # Remove timezone for compatibility
    if schumann_hourly.index.tzinfo is not None:
        schumann_hourly.index = schumann_hourly.index.tz_localize(None)
    
    # Create merged DataFrame
    merged = schumann_hourly.copy()
    
    # Remove timezone from space weather series
    if f107_series.index.tzinfo is not None:
        f107_series.index = f107_series.index.tz_localize(None)
        f107_norm.index = f107_norm.index.tz_localize(None)
    if kp_series.index.tzinfo is not None:
        kp_series.index = kp_series.index.tz_localize(None)
        kp_norm.index = kp_norm.index.tz_localize(None)
    
    # Add space weather (forward fill daily values)
    merged['F107'] = f107_series.reindex(merged.index, method='ffill')
    merged['F107_norm'] = f107_norm.reindex(merged.index, method='ffill')
    merged['Kp'] = kp_series.reindex(merged.index, method='ffill')
    merged['Kp_norm'] = kp_norm.reindex(merged.index, method='ffill')
    
    # Drop rows with missing data
    merged = merged.dropna(subset=['f1_Hz', 'f2_Hz', 'f3_Hz'])
    
    logger.info(f"Merged dataset: {len(merged)} points")
    
    return merged


def create_real_data_plots(
    df: pd.DataFrame,
    delta_seg_df: pd.DataFrame,
    eta: float,
    mode_consistency: dict,
    global_fit,
    output_dir: Path,
):
    """Create all diagnostic plots for real data analysis."""
    
    f_class = compute_classical_frequencies(eta)
    
    # 1. Time series plot
    logger.info("Creating time series plot...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        f_obs_col = f'f{n}_Hz'
        
        if f_obs_col in df.columns:
            ax.plot(df.index, df[f_obs_col], 'b-', alpha=0.6, linewidth=0.5, label='Observed')
            ax.axhline(f_class[n], color='r', linestyle='--', label=f'Classical ({f_class[n]:.2f} Hz)')
            ax.set_ylabel(f'f{n} (Hz)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (UTC)')
    fig.suptitle('Real Schumann Resonance Data: Observed vs Classical')
    plt.tight_layout()
    
    plot_path = output_dir / 'fig_real_timeseries.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {plot_path}")
    
    # 2. Delta_seg time series
    logger.info("Creating delta_seg time series plot...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        col = f'delta_seg{n}'
        if col in delta_seg_df.columns:
            ax.plot(delta_seg_df.index, delta_seg_df[col] * 100, 
                   color=colors[i], alpha=0.6, linewidth=0.5)
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.axhline(1, color='r', linestyle='--', alpha=0.3, label='+1%')
            ax.axhline(-1, color='r', linestyle='--', alpha=0.3, label='-1%')
            ax.set_ylabel(f'delta_seg{n} (%)')
            ax.set_ylim(-3, 3)
            ax.grid(True, alpha=0.3)
    
    # Mean delta_seg
    ax = axes[3]
    if 'delta_seg_mean' in delta_seg_df.columns:
        ax.plot(delta_seg_df.index, delta_seg_df['delta_seg_mean'] * 100,
               'k-', alpha=0.6, linewidth=0.5)
        ax.fill_between(delta_seg_df.index, -1, 1, alpha=0.2, color='green', label='SSZ band')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('Mean delta_seg (%)')
        ax.set_ylim(-3, 3)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (UTC)')
    fig.suptitle('SSZ Segmentation Parameter Time Series')
    plt.tight_layout()
    
    plot_path = output_dir / 'fig_real_delta_seg_vs_mode.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {plot_path}")
    
    # 3. Mode consistency scatter
    logger.info("Creating mode consistency plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if 'delta_seg1' in delta_seg_df.columns and 'delta_seg2' in delta_seg_df.columns:
        ax = axes[0]
        ax.scatter(delta_seg_df['delta_seg1'] * 100, delta_seg_df['delta_seg2'] * 100,
                  alpha=0.3, s=5)
        
        # 1:1 line
        lims = [-3, 3]
        ax.plot(lims, lims, 'r--', label='1:1 line')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('delta_seg1 (%)')
        ax.set_ylabel('delta_seg2 (%)')
        
        corr = mode_consistency.get('corr_12', np.nan)
        ax.set_title(f'Mode 1 vs Mode 2 (r = {corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    if 'delta_seg1' in delta_seg_df.columns and 'delta_seg3' in delta_seg_df.columns:
        ax = axes[1]
        ax.scatter(delta_seg_df['delta_seg1'] * 100, delta_seg_df['delta_seg3'] * 100,
                  alpha=0.3, s=5)
        
        lims = [-3, 3]
        ax.plot(lims, lims, 'r--', label='1:1 line')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('delta_seg1 (%)')
        ax.set_ylabel('delta_seg3 (%)')
        
        corr = mode_consistency.get('corr_13', np.nan)
        ax.set_title(f'Mode 1 vs Mode 3 (r = {corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    fig.suptitle('Mode Consistency Check (SSZ Signature)')
    plt.tight_layout()
    
    plot_path = output_dir / 'fig_real_mode_consistency.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {plot_path}")
    
    # 4. Delta_seg vs proxies
    logger.info("Creating proxy correlation plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    merged = pd.concat([delta_seg_df, df[['F107_norm', 'Kp_norm']]], axis=1)
    
    if 'delta_seg_mean' in merged.columns and 'F107_norm' in merged.columns:
        ax = axes[0]
        ax.scatter(merged['F107_norm'], merged['delta_seg_mean'] * 100,
                  alpha=0.3, s=5)
        
        # Regression line
        mask = ~(merged['F107_norm'].isna() | merged['delta_seg_mean'].isna())
        if mask.sum() > 10:
            x = merged.loc[mask, 'F107_norm']
            y = merged.loc[mask, 'delta_seg_mean'] * 100
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
            
            corr = x.corr(y)
            ax.set_title(f'delta_seg vs F10.7 (r = {corr:.3f})')
        
        ax.set_xlabel('F10.7 (normalized)')
        ax.set_ylabel('Mean delta_seg (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if 'delta_seg_mean' in merged.columns and 'Kp_norm' in merged.columns:
        ax = axes[1]
        ax.scatter(merged['Kp_norm'], merged['delta_seg_mean'] * 100,
                  alpha=0.3, s=5)
        
        mask = ~(merged['Kp_norm'].isna() | merged['delta_seg_mean'].isna())
        if mask.sum() > 10:
            x = merged.loc[mask, 'Kp_norm']
            y = merged.loc[mask, 'delta_seg_mean'] * 100
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Fit: y = {z[0]:.3f}x + {z[1]:.3f}')
            
            corr = x.corr(y)
            ax.set_title(f'delta_seg vs Kp (r = {corr:.3f})')
        
        ax.set_xlabel('Kp (normalized)')
        ax.set_ylabel('Mean delta_seg (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('SSZ Segmentation vs Ionospheric Proxies')
    plt.tight_layout()
    
    plot_path = output_dir / 'fig_real_delta_vs_f107_kp.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {plot_path}")
    
    # 5. Summary plot (2x2 grid)
    logger.info("Creating summary plot...")
    fig = plt.figure(figsize=(14, 12))
    
    # Panel 1: Delta_seg time series
    ax1 = fig.add_subplot(2, 2, 1)
    if 'delta_seg_mean' in delta_seg_df.columns:
        ax1.plot(delta_seg_df.index, delta_seg_df['delta_seg_mean'] * 100,
                'k-', alpha=0.6, linewidth=0.5)
        ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Mean delta_seg (%)')
        ax1.set_title('SSZ Segmentation Time Series')
        ax1.grid(True, alpha=0.3)
    
    # Panel 2: Mode consistency
    ax2 = fig.add_subplot(2, 2, 2)
    if 'delta_seg1' in delta_seg_df.columns and 'delta_seg2' in delta_seg_df.columns:
        ax2.scatter(delta_seg_df['delta_seg1'] * 100, delta_seg_df['delta_seg2'] * 100,
                   alpha=0.3, s=5)
        ax2.plot([-3, 3], [-3, 3], 'r--')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        ax2.set_xlabel('delta_seg1 (%)')
        ax2.set_ylabel('delta_seg2 (%)')
        ax2.set_title(f'Mode Consistency (r = {mode_consistency.get("corr_12", 0):.3f})')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
    
    # Panel 3: Delta_seg vs F10.7
    ax3 = fig.add_subplot(2, 2, 3)
    merged = pd.concat([delta_seg_df, df[['F107_norm']]], axis=1)
    if 'delta_seg_mean' in merged.columns and 'F107_norm' in merged.columns:
        ax3.scatter(merged['F107_norm'], merged['delta_seg_mean'] * 100,
                   alpha=0.3, s=5)
        ax3.set_xlabel('F10.7 (normalized)')
        ax3.set_ylabel('Mean delta_seg (%)')
        ax3.set_title('SSZ vs Solar Activity')
        ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    summary_text = f"""
REAL DATA SSZ ANALYSIS SUMMARY
==============================

Data Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}
Data Points: {len(df):,}

Classical eta_0: {eta:.6f}

SSZ Score: {mode_consistency.get('ssz_score', 0):.4f}
Mode Correlation: {mode_consistency.get('mean_correlation', 0):.4f}
Mean Spread: {mode_consistency.get('mean_abs_spread', 0)*100:.4f}%

Global Model R^2: {global_fit.r_squared:.4f}
beta_0: {global_fit.parameters.get('beta_0', 0):.6f}
beta_1 (F10.7): {global_fit.parameters.get('beta_1', 0):.6f}

Mean delta_seg: {delta_seg_df['delta_seg_mean'].mean()*100:.4f}%
Std delta_seg: {delta_seg_df['delta_seg_mean'].std()*100:.4f}%
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plot_path = output_dir / 'fig_real_summary.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved: {plot_path}")


def write_reports(
    output_dir: Path,
    eta: float,
    mode_consistency: dict,
    global_fit,
    layered_fit,
    correlations: dict,
    delta_seg_stats: dict,
    interpretation: str,
    df: pd.DataFrame,
):
    """Write analysis reports."""
    
    # Text report
    report_path = output_dir / 'realdata_analysis_results.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REAL DATA SSZ SCHUMANN ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Period: {df.index.min()} to {df.index.max()}\n")
        f.write(f"Data Points: {len(df)}\n\n")
        
        f.write("CLASSICAL MODEL\n")
        f.write("-" * 30 + "\n")
        f.write(f"Classical eta_0: {eta:.6f}\n\n")
        
        f.write("SSZ SIGNATURE\n")
        f.write("-" * 30 + "\n")
        f.write(f"SSZ score: {mode_consistency.get('ssz_score', 0):.6f}\n")
        f.write(f"Mode correlation (mean): {mode_consistency.get('mean_correlation', 0):.6f}\n")
        f.write(f"Mode correlation 1-2: {mode_consistency.get('corr_12', 0):.6f}\n")
        f.write(f"Mode correlation 1-3: {mode_consistency.get('corr_13', 0):.6f}\n")
        f.write(f"Mode correlation 2-3: {mode_consistency.get('corr_23', 0):.6f}\n")
        f.write(f"Mean spread: {mode_consistency.get('mean_abs_spread', 0):.6f}\n")
        f.write(f"Is consistent: {mode_consistency.get('is_consistent', False)}\n\n")
        
        f.write("GLOBAL SSZ MODEL\n")
        f.write("-" * 30 + "\n")
        f.write(f"R^2: {global_fit.r_squared:.6f}\n")
        f.write(f"RMSE: {global_fit.rmse:.6f}\n")
        for key, val in global_fit.parameters.items():
            f.write(f"{key}: {val:.6f}\n")
        f.write("\n")
        
        f.write("LAYERED SSZ MODEL\n")
        f.write("-" * 30 + "\n")
        f.write(f"R^2: {layered_fit.r_squared:.6f}\n")
        for key, val in layered_fit.parameters.items():
            f.write(f"{key}: {val:.6f}\n")
        f.write("\n")
        
        f.write("PROXY CORRELATIONS\n")
        f.write("-" * 30 + "\n")
        for proxy, corr in correlations.items():
            f.write(f"{proxy}: {corr:.6f}\n")
        f.write("\n")
        
        f.write("DELTA_SEG STATISTICS\n")
        f.write("-" * 30 + "\n")
        for key, val in delta_seg_stats.items():
            f.write(f"{key}: {val:.6f}\n")
        f.write("\n")
        
        f.write("INTERPRETATION\n")
        f.write("-" * 30 + "\n")
        f.write(interpretation)
    
    logger.info(f"Saved: {report_path}")
    
    # Markdown report
    md_path = output_dir / 'realdata_summary.md'
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Real Data SSZ Schumann Analysis\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- **Period:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Data Points:** {len(df):,}\n")
        f.write(f"- **Calibrated eta_0:** {eta:.6f}\n\n")
        
        f.write("## SSZ Signature Analysis\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| SSZ Score | {mode_consistency.get('ssz_score', 0):.4f} |\n")
        f.write(f"| Mode Correlation | {mode_consistency.get('mean_correlation', 0):.4f} |\n")
        f.write(f"| Mean Spread | {mode_consistency.get('mean_abs_spread', 0)*100:.4f}% |\n")
        f.write(f"| Consistent | {'Yes' if mode_consistency.get('is_consistent', False) else 'No'} |\n\n")
        
        f.write("## Model Fit Results\n\n")
        f.write("### Global SSZ Model\n\n")
        f.write(f"- **R²:** {global_fit.r_squared:.4f}\n")
        f.write(f"- **RMSE:** {global_fit.rmse:.6f}\n")
        f.write(f"- **beta_0:** {global_fit.parameters.get('beta_0', 0):.6f}\n")
        f.write(f"- **beta_1 (F10.7):** {global_fit.parameters.get('beta_1', 0):.6f}\n\n")
        
        f.write("### Proxy Correlations\n\n")
        f.write("| Proxy | Correlation |\n")
        f.write("|-------|-------------|\n")
        for proxy, corr in correlations.items():
            f.write(f"| {proxy} | {corr:.4f} |\n")
        f.write("\n")
        
        f.write("## Interpretation\n\n")
        f.write("```\n")
        f.write(interpretation)
        f.write("\n```\n")
        
        f.write("\n## Figures\n\n")
        f.write("- `fig_real_timeseries.png` - Observed vs classical frequencies\n")
        f.write("- `fig_real_delta_seg_vs_mode.png` - Delta_seg time series per mode\n")
        f.write("- `fig_real_mode_consistency.png` - Mode consistency scatter plots\n")
        f.write("- `fig_real_delta_vs_f107_kp.png` - Delta_seg vs ionospheric proxies\n")
        f.write("- `fig_real_summary.png` - Summary panel\n")
    
    logger.info(f"Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Real Data SSZ Schumann Analysis"
    )
    parser.add_argument(
        "--schumann-path",
        type=str,
        default="data/schumann/",
        help="Path to Schumann data file or directory"
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample data for testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/realdata",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("REAL DATA SSZ SCHUMANN ANALYSIS")
    logger.info("=" * 60)
    
    # Setup output directories
    dirs = setup_output_directories(Path(args.output_dir))
    logger.info(f"Output directory: {dirs['run']}")
    
    # Load and prepare data
    logger.info("\n[1/6] Loading data...")
    df = load_and_prepare_data(args)
    
    # Calibrate eta
    logger.info("\n[2/6] Calibrating classical model...")
    eta = calibrate_eta_from_data(df)
    
    # Compute delta_seg
    logger.info("\n[3/6] Computing SSZ segmentation parameters...")
    delta_seg_df = compute_delta_seg(df, eta)
    
    # Merge for analysis
    analysis_df = pd.concat([df, delta_seg_df], axis=1)
    
    # Mode consistency
    logger.info("\n[4/6] Analyzing mode consistency...")
    mode_consistency = compute_mode_consistency(delta_seg_df)
    
    # Fit models
    logger.info("\n[5/6] Fitting SSZ models...")
    global_fit = fit_global_ssz_model(analysis_df)
    layered_fit = fit_layered_ssz_model(analysis_df)
    
    # Proxy correlations
    correlations = compute_proxy_correlations(analysis_df)
    
    # Delta_seg statistics
    delta_seg_stats = {
        'mean': delta_seg_df['delta_seg_mean'].mean(),
        'std': delta_seg_df['delta_seg_mean'].std(),
        'min': delta_seg_df['delta_seg_mean'].min(),
        'max': delta_seg_df['delta_seg_mean'].max(),
    }
    
    # Generate interpretation
    interpretation = generate_interpretation(
        mode_consistency, global_fit, layered_fit, correlations, delta_seg_stats
    )
    
    # Create plots
    logger.info("\n[6/6] Creating plots...")
    create_real_data_plots(
        df, delta_seg_df, eta, mode_consistency, global_fit, dirs['plots']
    )
    
    # Write reports
    write_reports(
        dirs['reports'], eta, mode_consistency, global_fit, layered_fit,
        correlations, delta_seg_stats, interpretation, df
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {dirs['run']}")
    logger.info(f"Plots: {dirs['plots']}")
    logger.info(f"Reports: {dirs['reports']}")
    logger.info("")
    logger.info("Key Results:")
    logger.info(f"  eta_0 = {eta:.6f}")
    logger.info(f"  SSZ Score = {mode_consistency.get('ssz_score', 0):.4f}")
    logger.info(f"  Mode Correlation = {mode_consistency.get('mean_correlation', 0):.4f}")
    logger.info(f"  Global R^2 = {global_fit.r_squared:.4f}")
    logger.info(f"  Mean delta_seg = {delta_seg_stats['mean']*100:.4f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
