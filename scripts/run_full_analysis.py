#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full SSZ Schumann Analysis Pipeline

Runs both the original and layered SSZ analysis on available data.

Usage:
    python scripts/run_full_analysis.py --synthetic
    python scripts/run_full_analysis.py --data-dir data/

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
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.config import PHI, C_LIGHT, EARTH_RADIUS
from ssz_schumann.models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
from ssz_schumann.models.ssz_correction import (
    D_SSZ, delta_seg_from_observed, check_mode_consistency
)
from ssz_schumann.models.layered_ssz import (
    LayeredSSZConfig, D_SSZ_layered, f_n_ssz_layered,
    compute_all_modes, frequency_shift_estimate, print_frequency_table
)
from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
from ssz_schumann.data_io.merge import merge_all
from ssz_schumann.analysis.compute_deltas import compute_all_deltas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_synthetic_data():
    """Load synthetic data from files or generate new."""
    data_dir = Path(__file__).parent.parent / "data"
    
    schumann_path = data_dir / "schumann" / "synthetic_2016.nc"
    f107_path = data_dir / "space_weather" / "synthetic_f107.csv"
    kp_path = data_dir / "space_weather" / "synthetic_kp.csv"
    
    if schumann_path.exists():
        logger.info(f"Loading Schumann data from {schumann_path}")
        schumann_ds = xr.open_dataset(schumann_path)
    else:
        logger.info("Generating synthetic Schumann data...")
        schumann_ds = create_synthetic_schumann_data(
            start="2016-01-01", end="2016-12-31", freq="1h",
            delta_seg_amplitude=0.02, noise_level=0.01
        )
    
    if f107_path.exists():
        logger.info(f"Loading F10.7 data from {f107_path}")
        f107 = pd.read_csv(f107_path, index_col=0, parse_dates=True).squeeze()
    else:
        logger.info("Generating synthetic F10.7 data...")
        f107, _ = create_synthetic_space_weather("2016-01-01", "2016-12-31")
    
    if kp_path.exists():
        logger.info(f"Loading Kp data from {kp_path}")
        kp = pd.read_csv(kp_path, index_col=0, parse_dates=True).squeeze()
    else:
        logger.info("Generating synthetic Kp data...")
        _, kp = create_synthetic_space_weather("2016-01-01", "2016-12-31")
    
    return schumann_ds, f107, kp


def analyze_classical_model(merged_ds):
    """Analyze data with classical Schumann model."""
    logger.info("\n" + "=" * 70)
    logger.info("CLASSICAL MODEL ANALYSIS")
    logger.info("=" * 70)
    
    # Extract frequencies
    f1_obs = merged_ds["f1"].values
    f2_obs = merged_ds["f2"].values
    f3_obs = merged_ds["f3"].values
    
    # Compute eta from mean f1
    mean_f1 = np.nanmean(f1_obs)
    eta_0 = compute_eta0_from_mean_f1(mean_f1)
    
    logger.info(f"\nObserved frequencies:")
    logger.info(f"  f1: mean={np.nanmean(f1_obs):.3f} Hz, std={np.nanstd(f1_obs):.3f} Hz")
    logger.info(f"  f2: mean={np.nanmean(f2_obs):.3f} Hz, std={np.nanstd(f2_obs):.3f} Hz")
    logger.info(f"  f3: mean={np.nanmean(f3_obs):.3f} Hz, std={np.nanstd(f3_obs):.3f} Hz")
    logger.info(f"\nDerived eta_0: {eta_0:.4f}")
    
    # Classical predictions
    f1_class = f_n_classical(1, eta_0)
    f2_class = f_n_classical(2, eta_0)
    f3_class = f_n_classical(3, eta_0)
    
    logger.info(f"\nClassical predictions (eta={eta_0:.4f}):")
    logger.info(f"  f1_class = {f1_class:.3f} Hz")
    logger.info(f"  f2_class = {f2_class:.3f} Hz")
    logger.info(f"  f3_class = {f3_class:.3f} Hz")
    
    # Residuals
    res1 = f1_obs - f1_class
    res2 = f2_obs - f2_class
    res3 = f3_obs - f3_class
    
    logger.info(f"\nResiduals (obs - classical):")
    logger.info(f"  f1: mean={np.nanmean(res1):.4f} Hz, std={np.nanstd(res1):.4f} Hz")
    logger.info(f"  f2: mean={np.nanmean(res2):.4f} Hz, std={np.nanstd(res2):.4f} Hz")
    logger.info(f"  f3: mean={np.nanmean(res3):.4f} Hz, std={np.nanstd(res3):.4f} Hz")
    
    return {
        "eta_0": eta_0,
        "f_class": [f1_class, f2_class, f3_class],
        "residuals": [res1, res2, res3],
    }


def analyze_ssz_model(merged_ds, classical_result):
    """Analyze data with SSZ model."""
    logger.info("\n" + "=" * 70)
    logger.info("SSZ MODEL ANALYSIS")
    logger.info("=" * 70)
    
    time_idx = pd.DatetimeIndex(merged_ds.time.values)
    eta_0 = classical_result["eta_0"]
    
    # Extract delta_seg for each mode
    delta_seg_dict = {}
    for n in [1, 2, 3]:
        f_obs = pd.Series(merged_ds[f"f{n}"].values, index=time_idx)
        f_class = classical_result["f_class"][n-1]
        delta_seg = delta_seg_from_observed(f_obs, f_class)
        delta_seg_dict[n] = delta_seg
        
        logger.info(f"\nMode {n} delta_seg:")
        logger.info(f"  mean = {delta_seg.mean():.6f}")
        logger.info(f"  std  = {delta_seg.std():.6f}")
        logger.info(f"  min  = {delta_seg.min():.6f}")
        logger.info(f"  max  = {delta_seg.max():.6f}")
    
    # Check mode consistency (SSZ signature)
    consistency = check_mode_consistency(delta_seg_dict)
    
    logger.info(f"\nMode Consistency Check (SSZ Signature):")
    logger.info(f"  Mean correlation: {consistency['mean_correlation']:.4f}")
    logger.info(f"  Std across modes: {consistency['std_delta_seg']:.6f}")
    logger.info(f"  SSZ score: {consistency['ssz_score']:.4f}")
    logger.info(f"  Is consistent: {consistency['is_consistent']}")
    
    if consistency['ssz_score'] > 0.7:
        logger.info("  -> STRONG SSZ signature detected!")
    elif consistency['ssz_score'] > 0.5:
        logger.info("  -> Moderate SSZ signature")
    else:
        logger.info("  -> Weak/no SSZ signature")
    
    return {
        "delta_seg": delta_seg_dict,
        "consistency": consistency,
    }


def analyze_layered_model(merged_ds, classical_result):
    """Analyze data with layered SSZ model."""
    logger.info("\n" + "=" * 70)
    logger.info("LAYERED SSZ MODEL ANALYSIS")
    logger.info("=" * 70)
    
    time_idx = pd.DatetimeIndex(merged_ds.time.values)
    
    # Configuration
    config = LayeredSSZConfig()
    logger.info(f"\nLayer Configuration:")
    logger.info(f"  Ground:     w = {config.ground.weight:.2f}")
    logger.info(f"  Atmosphere: w = {config.atmosphere.weight:.2f}")
    logger.info(f"  Ionosphere: w = {config.ionosphere.weight:.2f}")
    
    # Extract sigma_iono from observations
    f1_obs = merged_ds["f1"].values
    f1_class = classical_result["f_class"][0]
    
    # delta_seg = f_class / f_obs - 1
    # sigma_iono = delta_seg / w_iono
    delta_seg = f1_class / f1_obs - 1
    sigma_iono = delta_seg / config.ionosphere.weight
    
    logger.info(f"\nExtracted sigma_iono (from f1):")
    logger.info(f"  mean = {np.nanmean(sigma_iono):.6f}")
    logger.info(f"  std  = {np.nanstd(sigma_iono):.6f}")
    
    # Correlation with F10.7
    if "f107_norm" in merged_ds:
        f107_norm = merged_ds["f107_norm"].values
        
        valid = ~(np.isnan(sigma_iono) | np.isnan(f107_norm))
        if np.sum(valid) > 100:
            corr = np.corrcoef(sigma_iono[valid], f107_norm[valid])[0, 1]
            logger.info(f"\nCorrelation sigma_iono vs F10.7: {corr:.4f}")
            
            # Linear fit
            from sklearn.linear_model import LinearRegression
            X = f107_norm[valid].reshape(-1, 1)
            y = sigma_iono[valid]
            
            model = LinearRegression()
            model.fit(X, y)
            
            beta_0 = model.intercept_
            beta_1 = model.coef_[0]
            
            logger.info(f"\nLinear fit: sigma_iono = beta_0 + beta_1 * F10.7")
            logger.info(f"  beta_0 = {beta_0:.6f} (baseline)")
            logger.info(f"  beta_1 = {beta_1:.6f} (coupling)")
            
            # R-squared
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res / ss_tot
            logger.info(f"  R^2 = {r_squared:.4f}")
        else:
            beta_0, beta_1, r_squared = 0, 0, 0
            corr = np.nan
    else:
        beta_0, beta_1, r_squared = 0, 0, 0
        corr = np.nan
    
    # Frequency shift at mean conditions
    mean_sigma = np.nanmean(sigma_iono)
    delta_seg_eff = config.ionosphere.weight * mean_sigma
    
    logger.info(f"\nFrequency Shifts at Mean Conditions:")
    logger.info(f"  Effective delta_seg = {delta_seg_eff:.4f} ({delta_seg_eff*100:.2f}%)")
    
    result = frequency_shift_estimate(delta_seg_eff, f_ref=7.83)
    for n in [1, 2, 3]:
        logger.info(f"  f{n}: {result[f'f{n}_classical']:.2f} Hz -> "
                   f"{result[f'f{n}_ssz']:.2f} Hz (Df = {result[f'delta_f{n}']:+.3f} Hz)")
    
    return {
        "sigma_iono": sigma_iono,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "r_squared": r_squared,
        "f107_corr": corr,
    }


def create_summary_plots(merged_ds, classical_result, ssz_result, layered_result, output_dir):
    """Create comprehensive summary plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time = pd.DatetimeIndex(merged_ds.time.values)
    
    # Figure 1: Frequency time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        f_obs = merged_ds[f"f{n}"].values
        f_class = classical_result["f_class"][i]
        
        ax.plot(time, f_obs, 'b-', alpha=0.5, linewidth=0.5, label='Observed')
        ax.axhline(f_class, color='r', linestyle='--', alpha=0.7, label='Classical')
        
        ax.set_ylabel(f'f{n} (Hz)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Schumann Resonance Frequencies: Observed vs Classical')
    plt.tight_layout()
    plt.savefig(output_dir / 'frequencies_timeseries.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'frequencies_timeseries.png'}")
    
    # Figure 2: Delta_seg comparison across modes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time series of delta_seg
    ax = axes[0, 0]
    for n in [1, 2, 3]:
        delta_seg = ssz_result["delta_seg"][n]
        ax.plot(time, delta_seg.values * 100, alpha=0.5, linewidth=0.5, label=f'Mode {n}')
    ax.set_xlabel('Time')
    ax.set_ylabel('delta_seg (%)')
    ax.set_title('SSZ Segmentation Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mode correlation scatter
    ax = axes[0, 1]
    d1 = ssz_result["delta_seg"][1].values
    d2 = ssz_result["delta_seg"][2].values
    valid = ~(np.isnan(d1) | np.isnan(d2))
    ax.scatter(d1[valid]*100, d2[valid]*100, alpha=0.2, s=3)
    lims = [min(np.nanmin(d1), np.nanmin(d2))*100, max(np.nanmax(d1), np.nanmax(d2))*100]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='1:1')
    ax.set_xlabel('delta_seg Mode 1 (%)')
    ax.set_ylabel('delta_seg Mode 2 (%)')
    ax.set_title(f'Mode Consistency (r={ssz_result["consistency"]["mean_correlation"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sigma_iono vs F10.7
    ax = axes[1, 0]
    if "f107_norm" in merged_ds:
        f107 = merged_ds["f107_norm"].values
        sigma = layered_result["sigma_iono"]
        valid = ~(np.isnan(sigma) | np.isnan(f107))
        ax.scatter(f107[valid], sigma[valid]*100, alpha=0.2, s=3)
        
        if layered_result["beta_1"] != 0:
            f_range = np.linspace(np.nanmin(f107), np.nanmax(f107), 100)
            sigma_fit = layered_result["beta_0"] + layered_result["beta_1"] * f_range
            ax.plot(f_range, sigma_fit*100, 'r-', linewidth=2, label='Fit')
        
        ax.set_xlabel('F10.7 (normalized)')
        ax.set_ylabel('sigma_iono (%)')
        ax.set_title(f'Ionosphere Segmentation vs Solar Flux (r={layered_result["f107_corr"]:.3f})')
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SSZ Score summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    SSZ ANALYSIS SUMMARY
    ====================
    
    Classical Model:
      eta_0 = {classical_result['eta_0']:.4f}
    
    SSZ Model:
      Mode correlation = {ssz_result['consistency']['mean_correlation']:.4f}
      SSZ score = {ssz_result['consistency']['ssz_score']:.4f}
      Is consistent = {ssz_result['consistency']['is_consistent']}
    
    Layered Model:
      beta_0 = {layered_result['beta_0']:.6f}
      beta_1 = {layered_result['beta_1']:.6f}
      R^2 = {layered_result['r_squared']:.4f}
      F10.7 correlation = {layered_result['f107_corr']:.4f}
    
    Interpretation:
      {'Strong SSZ signature!' if ssz_result['consistency']['ssz_score'] > 0.7 else 
       'Moderate SSZ signature' if ssz_result['consistency']['ssz_score'] > 0.5 else
       'Weak/no SSZ signature'}
    """
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ssz_analysis_summary.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'ssz_analysis_summary.png'}")
    
    # Figure 3: Frequency shift diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_seg_range = np.linspace(0, 0.05, 100)
    
    for n in [1, 2, 3]:
        f_class = classical_result["f_class"][n-1]
        d_ssz = 1 + delta_seg_range
        f_ssz = f_class / d_ssz
        delta_f = f_ssz - f_class
        ax.plot(delta_seg_range * 100, delta_f, label=f'Mode {n}', linewidth=2)
    
    # Mark observed mean
    mean_delta_seg = np.nanmean([ssz_result["delta_seg"][n].mean() for n in [1, 2, 3]])
    ax.axvline(mean_delta_seg * 100, color='r', linestyle='--', alpha=0.7, 
               label=f'Observed mean ({mean_delta_seg*100:.2f}%)')
    
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('delta_seg (%)')
    ax.set_ylabel('Frequency Shift (Hz)')
    ax.set_title('SSZ Frequency Shift vs Segmentation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_shift_diagram.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'frequency_shift_diagram.png'}")


def print_final_summary(classical_result, ssz_result, layered_result):
    """Print final analysis summary."""
    print("\n" + "=" * 70)
    print("SSZ SCHUMANN RESONANCE ANALYSIS - FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nPhysical Constants:")
    print(f"  Golden Ratio (phi) = {PHI:.6f}")
    print(f"  Speed of Light = {C_LIGHT:.0f} m/s")
    print(f"  Earth Radius = {EARTH_RADIUS:.3e} m ({EARTH_RADIUS/1e3:.0f} km)")
    
    print(f"\nClassical Model:")
    print(f"  Effective eta_0 = {classical_result['eta_0']:.4f}")
    print(f"  f1_class = {classical_result['f_class'][0]:.3f} Hz")
    print(f"  f2_class = {classical_result['f_class'][1]:.3f} Hz")
    print(f"  f3_class = {classical_result['f_class'][2]:.3f} Hz")
    
    print(f"\nSSZ Model (Mode Consistency):")
    print(f"  Mean mode correlation = {ssz_result['consistency']['mean_correlation']:.4f}")
    print(f"  SSZ score = {ssz_result['consistency']['ssz_score']:.4f}")
    
    print(f"\nLayered SSZ Model:")
    print(f"  Baseline sigma_iono (beta_0) = {layered_result['beta_0']:.6f}")
    print(f"  F10.7 coupling (beta_1) = {layered_result['beta_1']:.6f}")
    print(f"  Model R^2 = {layered_result['r_squared']:.4f}")
    
    print(f"\nInterpretation:")
    if ssz_result['consistency']['ssz_score'] > 0.7:
        print("  -> STRONG SSZ signature detected!")
        print("     All modes show consistent relative frequency shifts.")
    elif ssz_result['consistency']['ssz_score'] > 0.5:
        print("  -> Moderate SSZ signature")
        print("     Some mode consistency observed.")
    else:
        print("  -> Weak/no SSZ signature")
        print("     Mode shifts are not consistent with SSZ prediction.")
    
    if abs(layered_result['beta_1']) > 0.001:
        if layered_result['beta_1'] > 0:
            print("  -> Higher solar activity INCREASES segmentation")
        else:
            print("  -> Higher solar activity DECREASES segmentation")
    
    print("\n" + "=" * 70)
    
    # Print frequency table
    print_frequency_table()


def main():
    parser = argparse.ArgumentParser(description="Full SSZ Schumann Analysis")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--output-dir", type=str, default="output/full_analysis",
                       help="Output directory")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    
    logger.info("=" * 70)
    logger.info("SSZ SCHUMANN RESONANCE - FULL ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    
    # Load data
    logger.info("\n[1/5] Loading data...")
    schumann_ds, f107, kp = load_synthetic_data()
    
    # Merge data
    logger.info("\n[2/5] Merging data...")
    merged = merge_all(schumann_ds, f107, kp, time_resolution="1h")
    logger.info(f"  Merged dataset: {len(merged.time)} time points")
    
    # Classical analysis
    logger.info("\n[3/5] Classical model analysis...")
    classical_result = analyze_classical_model(merged)
    
    # SSZ analysis
    logger.info("\n[4/5] SSZ model analysis...")
    ssz_result = analyze_ssz_model(merged, classical_result)
    
    # Layered analysis
    logger.info("\n[5/5] Layered SSZ model analysis...")
    layered_result = analyze_layered_model(merged, classical_result)
    
    # Create plots
    logger.info("\nCreating summary plots...")
    create_summary_plots(merged, classical_result, ssz_result, layered_result, output_dir / "plots")
    
    # Final summary
    print_final_summary(classical_result, ssz_result, layered_result)
    
    # Save results
    results_file = output_dir / "analysis_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("SSZ SCHUMANN ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Classical eta_0: {classical_result['eta_0']:.6f}\n")
        f.write(f"SSZ score: {ssz_result['consistency']['ssz_score']:.6f}\n")
        f.write(f"Mode correlation: {ssz_result['consistency']['mean_correlation']:.6f}\n")
        f.write(f"Layered beta_0: {layered_result['beta_0']:.6f}\n")
        f.write(f"Layered beta_1: {layered_result['beta_1']:.6f}\n")
        f.write(f"Layered R^2: {layered_result['r_squared']:.6f}\n")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
