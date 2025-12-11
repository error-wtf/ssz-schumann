#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Sensitivity Scan

Runs a sensitivity study to determine how well the SSZ analysis pipeline
can detect SSZ signatures of different amplitudes.

This script:
1. Generates synthetic Schumann datasets with varying delta_seg amplitudes
2. Runs the SSZ analysis on each dataset
3. Measures how mean_correlation, ssz_score, and R² depend on amplitude
4. Produces a detection curve table and optional plot

Usage:
    python run_sensitivity_scan.py
    python run_sensitivity_scan.py --amplitudes 0.0 0.005 0.01 0.02 0.03 0.05
    python run_sensitivity_scan.py --plot --output-dir output/sensitivity_scan

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
from ssz_schumann.data_io.merge import merge_all
from ssz_schumann.analysis.compute_deltas import compute_all_deltas
from ssz_schumann.models.ssz_correction import check_mode_consistency
from ssz_schumann.models.fit_wrappers import fit_ssz_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_single_analysis(
    delta_seg_amplitude: float,
    noise_level: float = 0.01,
    random_seed: int = 42,
) -> Dict:
    """
    Run SSZ analysis for a single delta_seg amplitude.
    
    Args:
        delta_seg_amplitude: Amplitude of synthetic delta_seg signal
        noise_level: Noise level for synthetic data
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with analysis results
    """
    # Generate synthetic data
    np.random.seed(random_seed)
    
    schumann_ds = create_synthetic_schumann_data(
        start="2016-01-01",
        end="2016-03-31",  # 3 months
        freq="1h",
        eta_0=0.74,
        delta_seg_amplitude=delta_seg_amplitude,
        noise_level=noise_level,
    )
    
    f107, kp = create_synthetic_space_weather(
        start="2016-01-01",
        end="2016-03-31",
    )
    
    # Merge data
    merged = merge_all(schumann_ds, f107, kp, time_resolution="1h")
    
    # Compute deltas
    merged = compute_all_deltas(merged)
    
    # Extract delta_seg for each mode
    delta_seg_dict = {}
    for n in [1, 2, 3]:
        col = f"delta_seg_{n}"
        if col in merged:
            delta_seg_dict[n] = merged[col].values
    
    # Check mode consistency
    consistency = check_mode_consistency(delta_seg_dict)
    
    # Fit SSZ model (if features available)
    r_squared = np.nan
    try:
        features = pd.DataFrame({
            "f107_norm": merged["f107_norm"].values,
            "kp_norm": merged["kp_norm"].values,
        }, index=pd.DatetimeIndex(merged.time.values))
        
        ssz_result = fit_ssz_model(merged, features)
        r_squared = ssz_result.r_squared
    except Exception as e:
        logger.warning(f"Could not fit SSZ model: {e}")
    
    return {
        "delta_seg_amplitude": delta_seg_amplitude,
        "noise_level": noise_level,
        "mean_correlation": consistency["mean_correlation"],
        "ssz_score": consistency["ssz_score"],
        "std_across_modes": consistency["std_across_modes"],
        "is_consistent": consistency["is_consistent"],
        "interpretation": consistency["interpretation"],
        "r_squared": r_squared,
        "n_timepoints": consistency["n_timepoints"],
    }


def run_sensitivity_scan(
    amplitudes: List[float],
    noise_level: float = 0.01,
    n_runs: int = 1,
) -> pd.DataFrame:
    """
    Run sensitivity scan over multiple delta_seg amplitudes.
    
    Args:
        amplitudes: List of delta_seg amplitudes to test
        noise_level: Noise level for synthetic data
        n_runs: Number of runs per amplitude (for averaging)
    
    Returns:
        DataFrame with results for each amplitude
    """
    results = []
    
    for amp in amplitudes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing delta_seg_amplitude = {amp:.4f} ({amp*100:.2f}%)")
        logger.info(f"{'='*60}")
        
        if n_runs == 1:
            result = run_single_analysis(amp, noise_level)
            results.append(result)
        else:
            # Multiple runs for averaging
            run_results = []
            for i in range(n_runs):
                result = run_single_analysis(amp, noise_level, random_seed=42+i)
                run_results.append(result)
            
            # Average results
            avg_result = {
                "delta_seg_amplitude": amp,
                "noise_level": noise_level,
                "mean_correlation": np.mean([r["mean_correlation"] for r in run_results]),
                "ssz_score": np.mean([r["ssz_score"] for r in run_results]),
                "std_across_modes": np.mean([r["std_across_modes"] for r in run_results]),
                "is_consistent": any(r["is_consistent"] for r in run_results),
                "interpretation": run_results[0]["interpretation"],
                "r_squared": np.mean([r["r_squared"] for r in run_results if not np.isnan(r["r_squared"])]),
                "n_timepoints": run_results[0]["n_timepoints"],
            }
            results.append(avg_result)
    
    return pd.DataFrame(results)


def print_results_table(df: pd.DataFrame):
    """Print results as ASCII table."""
    print("\n" + "=" * 80)
    print("SSZ SENSITIVITY SCAN RESULTS")
    print("=" * 80)
    print()
    
    # Header
    header = f"{'delta_seg_amp':>14} | {'mean_corr':>10} | {'ssz_score':>10} | {'R2_layered':>10} | {'Detected':>10}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for _, row in df.iterrows():
        amp_str = f"{row['delta_seg_amplitude']:.4f}"
        corr_str = f"{row['mean_correlation']:.4f}" if not np.isnan(row['mean_correlation']) else "N/A"
        score_str = f"{row['ssz_score']:.4f}"
        r2_str = f"{row['r_squared']:.4f}" if not np.isnan(row['r_squared']) else "N/A"
        detected = "YES" if row['is_consistent'] else "no"
        
        print(f"{amp_str:>14} | {corr_str:>10} | {score_str:>10} | {r2_str:>10} | {detected:>10}")
    
    print()
    print("=" * 80)
    print()
    
    # Summary
    print("INTERPRETATION:")
    print("-" * 40)
    
    # Find detection threshold
    detected_amps = df[df['is_consistent']]['delta_seg_amplitude'].values
    if len(detected_amps) > 0:
        min_detected = detected_amps.min()
        print(f"  Minimum detectable amplitude: {min_detected:.4f} ({min_detected*100:.2f}%)")
    else:
        print("  No SSZ signature detected at any amplitude")
    
    # Correlation trend
    if len(df) > 1:
        corr_trend = np.corrcoef(df['delta_seg_amplitude'], df['mean_correlation'])[0, 1]
        print(f"  Correlation vs amplitude trend: r = {corr_trend:.4f}")
    
    print()


def create_detection_plot(df: pd.DataFrame, output_path: Path):
    """Create detection curve plot."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean correlation and SSZ score vs amplitude
    ax1 = axes[0]
    ax1.plot(df['delta_seg_amplitude'] * 100, df['mean_correlation'], 
             'b-o', linewidth=2, markersize=8, label='Mean Correlation')
    ax1.plot(df['delta_seg_amplitude'] * 100, df['ssz_score'], 
             'r-s', linewidth=2, markersize=8, label='SSZ Score')
    
    # Threshold lines
    ax1.axhline(0.7, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    
    ax1.set_xlabel('delta_seg Amplitude (%)')
    ax1.set_ylabel('Score')
    ax1.set_title('SSZ Detection vs Segmentation Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: R² vs amplitude
    ax2 = axes[1]
    valid_r2 = df[~df['r_squared'].isna()]
    if len(valid_r2) > 0:
        ax2.plot(valid_r2['delta_seg_amplitude'] * 100, valid_r2['r_squared'], 
                 'g-^', linewidth=2, markersize=8, label='R² (Layered Model)')
    
    ax2.set_xlabel('delta_seg Amplitude (%)')
    ax2.set_ylabel('R²')
    ax2.set_title('Model Fit Quality vs Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SSZ Sensitivity Scan"
    )
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
        help="List of delta_seg amplitudes to test"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.01,
        help="Noise level for synthetic data"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="Number of runs per amplitude (for averaging)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate detection curve plot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/sensitivity_scan",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SSZ SENSITIVITY SCAN")
    logger.info("=" * 60)
    logger.info(f"Amplitudes: {args.amplitudes}")
    logger.info(f"Noise level: {args.noise_level}")
    logger.info(f"Runs per amplitude: {args.n_runs}")
    
    # Run scan
    results_df = run_sensitivity_scan(
        amplitudes=args.amplitudes,
        noise_level=args.noise_level,
        n_runs=args.n_runs,
    )
    
    # Print results
    print_results_table(results_df)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "sensitivity_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    
    # Create plot if requested
    if args.plot:
        plot_path = output_dir / "detection_curve.png"
        create_detection_plot(results_df, plot_path)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SCAN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
