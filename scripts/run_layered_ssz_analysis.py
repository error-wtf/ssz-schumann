#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layered SSZ Schumann Resonance Analysis

This script runs the layered SSZ analysis on Schumann resonance data,
fitting the layer-based segmentation model to observed frequencies.

Usage:
    python run_layered_ssz_analysis.py --synthetic
    python run_layered_ssz_analysis.py --schumann-path data/schumann.csv --f107-path data/f107.csv

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.models.layered_ssz import (
    LayeredSSZConfig,
    D_SSZ_layered,
    f_n_classical,
    f_n_ssz_layered,
    compute_all_modes,
    sigma_iono_from_proxy,
    f_n_ssz_timeseries,
    frequency_shift_estimate,
    print_frequency_table,
)
from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
from ssz_schumann.data_io.merge import merge_all

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def fit_layered_model(
    f_obs: dict,
    F_iono: pd.Series,
    f1_ref: float = 7.83,
    w_atm: float = 0.2,
    w_iono: float = 0.8,
) -> dict:
    """
    Fit layered SSZ model to observed frequencies.
    
    Model: sigma_iono(t) = beta_0 + beta_1 * F_iono(t)
    
    Args:
        f_obs: {mode: frequency_series}
        F_iono: Ionospheric proxy (normalized)
        f1_ref: Reference f1
        w_atm: Atmosphere weight
        w_iono: Ionosphere weight
    
    Returns:
        Fit results dictionary
    """
    from sklearn.linear_model import LinearRegression
    
    # Extract delta_seg from each mode
    all_delta_seg = []
    all_F_iono = []
    mode_delta_segs = {}
    
    for n, f_series in f_obs.items():
        f_class = f_n_classical(n, f1_ref)
        
        # delta_seg = f_class / f_obs - 1
        delta_seg = f_class / f_series - 1
        
        # Convert to effective ionosphere sigma
        # delta_seg_eff = w_iono * sigma_iono (assuming w_atm * sigma_atm ~ 0)
        sigma_iono = delta_seg / w_iono
        
        mode_delta_segs[n] = sigma_iono
        
        # Align with F_iono
        common_idx = sigma_iono.index.intersection(F_iono.index)
        if len(common_idx) > 0:
            all_delta_seg.extend(sigma_iono.loc[common_idx].values)
            all_F_iono.extend(F_iono.loc[common_idx].values)
    
    # Convert to arrays
    y = np.array(all_delta_seg)
    X = np.array(all_F_iono).reshape(-1, 1)
    
    # Remove NaN
    valid = ~(np.isnan(y) | np.isnan(X.flatten()))
    y = y[valid]
    X = X[valid]
    
    n_points = len(y)
    logger.info(f"Fitting layered model with {n_points} points")
    
    if n_points < 10:
        logger.warning("Not enough points for reliable fit")
        return {"error": "insufficient_data", "n_points": n_points}
    
    # Fit: sigma_iono = beta_0 + beta_1 * F_iono
    model = LinearRegression()
    model.fit(X, y)
    
    beta_0 = model.intercept_
    beta_1 = model.coef_[0]
    
    # Predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Metrics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Mode consistency check
    if len(mode_delta_segs) >= 2:
        modes = list(mode_delta_segs.keys())
        correlations = []
        for i, n1 in enumerate(modes):
            for n2 in modes[i+1:]:
                s1 = mode_delta_segs[n1]
                s2 = mode_delta_segs[n2]
                common = s1.index.intersection(s2.index)
                if len(common) > 10:
                    corr = np.corrcoef(
                        s1.loc[common].values,
                        s2.loc[common].values
                    )[0, 1]
                    correlations.append(corr)
        mean_corr = np.mean(correlations) if correlations else np.nan
    else:
        mean_corr = np.nan
    
    result = {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "r_squared": r_squared,
        "rmse": rmse,
        "n_points": n_points,
        "w_atm": w_atm,
        "w_iono": w_iono,
        "mode_correlation": mean_corr,
        "residuals": residuals,
    }
    
    logger.info(f"Fit results:")
    logger.info(f"  beta_0 = {beta_0:.6f} (baseline sigma_iono)")
    logger.info(f"  beta_1 = {beta_1:.6f} (coupling to F_iono)")
    logger.info(f"  R^2 = {r_squared:.4f}")
    logger.info(f"  RMSE = {rmse:.6f}")
    logger.info(f"  Mode correlation = {mean_corr:.4f}")
    
    return result


def create_layered_plots(
    merged_ds,
    fit_result: dict,
    output_dir: Path,
):
    """Create visualization plots for layered SSZ analysis."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    time = pd.DatetimeIndex(merged_ds.time.values)
    
    # Plot 1: Observed vs Model frequencies
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for i, n in enumerate([1, 2, 3]):
        ax = axes[i]
        
        f_obs = merged_ds[f"f{n}"].values
        f_class = f_n_classical(n, f1_ref=7.83)
        
        # Model prediction
        if "beta_0" in fit_result and "beta_1" in fit_result:
            F_iono = merged_ds["f107_norm"].values
            sigma_iono = fit_result["beta_0"] + fit_result["beta_1"] * F_iono
            d_ssz = 1 + fit_result["w_iono"] * sigma_iono
            f_model = f_class / d_ssz
        else:
            f_model = np.full_like(f_obs, f_class)
        
        ax.plot(time, f_obs, 'b-', alpha=0.5, label='Observed', linewidth=0.5)
        ax.plot(time, f_model, 'r-', alpha=0.7, label='SSZ Model', linewidth=1)
        ax.axhline(f_class, color='g', linestyle='--', alpha=0.5, label='Classical')
        
        ax.set_ylabel(f'f{n} (Hz)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Layered SSZ Model: Observed vs Predicted Frequencies')
    plt.tight_layout()
    plt.savefig(output_dir / 'layered_frequencies.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'layered_frequencies.png'}")
    
    # Plot 2: Extracted sigma_iono vs F_iono
    fig, ax = plt.subplots(figsize=(10, 6))
    
    f1_obs = merged_ds["f1"].values
    f1_class = f_n_classical(1, f1_ref=7.83)
    delta_seg = f1_class / f1_obs - 1
    sigma_iono = delta_seg / fit_result.get("w_iono", 0.8)
    
    F_iono = merged_ds["f107_norm"].values
    
    ax.scatter(F_iono, sigma_iono, alpha=0.3, s=5, label='Extracted from f1')
    
    if "beta_0" in fit_result and "beta_1" in fit_result:
        F_range = np.linspace(np.nanmin(F_iono), np.nanmax(F_iono), 100)
        sigma_fit = fit_result["beta_0"] + fit_result["beta_1"] * F_range
        ax.plot(F_range, sigma_fit, 'r-', linewidth=2, 
                label=f'Fit: sigma = {fit_result["beta_0"]:.4f} + {fit_result["beta_1"]:.4f}*F')
    
    ax.set_xlabel('F10.7 (normalized)')
    ax.set_ylabel('sigma_iono (extracted)')
    ax.set_title('Ionosphere Segmentation vs Solar Flux')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sigma_vs_f107.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'sigma_vs_f107.png'}")
    
    # Plot 3: Mode consistency
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    modes = [1, 2, 3]
    sigma_modes = {}
    
    for n in modes:
        f_obs = merged_ds[f"f{n}"].values
        f_class = f_n_classical(n, f1_ref=7.83)
        delta_seg = f_class / f_obs - 1
        sigma_modes[n] = delta_seg / fit_result.get("w_iono", 0.8)
    
    pairs = [(1, 2), (1, 3), (2, 3)]
    for ax, (n1, n2) in zip(axes, pairs):
        ax.scatter(sigma_modes[n1], sigma_modes[n2], alpha=0.2, s=3)
        
        # Add 1:1 line
        lims = [
            min(np.nanmin(sigma_modes[n1]), np.nanmin(sigma_modes[n2])),
            max(np.nanmax(sigma_modes[n1]), np.nanmax(sigma_modes[n2]))
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='1:1')
        
        # Correlation
        valid = ~(np.isnan(sigma_modes[n1]) | np.isnan(sigma_modes[n2]))
        if np.sum(valid) > 10:
            corr = np.corrcoef(sigma_modes[n1][valid], sigma_modes[n2][valid])[0, 1]
            ax.set_title(f'Mode {n1} vs {n2} (r={corr:.3f})')
        else:
            ax.set_title(f'Mode {n1} vs {n2}')
        
        ax.set_xlabel(f'sigma from f{n1}')
        ax.set_ylabel(f'sigma from f{n2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Mode Consistency Check (SSZ Signature)')
    plt.tight_layout()
    plt.savefig(output_dir / 'mode_consistency_layered.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'mode_consistency_layered.png'}")
    
    # Plot 4: Frequency shift table visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    delta_seg_values = np.linspace(0, 0.03, 100)
    
    for n in [1, 2, 3]:
        f_class = f_n_classical(n, f1_ref=7.83)
        d_ssz = 1 + delta_seg_values
        f_ssz = f_class / d_ssz
        delta_f = f_ssz - f_class
        
        ax.plot(delta_seg_values * 100, delta_f, label=f'Mode {n}')
    
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(1, color='r', linestyle='--', alpha=0.5, label='1% segmentation')
    
    ax.set_xlabel('delta_seg (%)')
    ax.set_ylabel('Frequency shift (Hz)')
    ax.set_title('SSZ Frequency Shift vs Segmentation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'shift_vs_segmentation.png', dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'shift_vs_segmentation.png'}")


def print_summary(fit_result: dict, config: LayeredSSZConfig):
    """Print analysis summary."""
    
    print("\n" + "=" * 70)
    print("LAYERED SSZ SCHUMANN ANALYSIS - SUMMARY")
    print("=" * 70)
    
    print("\nLayer Configuration:")
    print(f"  Ground:     w = {config.ground.weight:.2f}")
    print(f"  Atmosphere: w = {config.atmosphere.weight:.2f}")
    print(f"  Ionosphere: w = {config.ionosphere.weight:.2f}")
    
    if "error" in fit_result:
        print(f"\nFit Error: {fit_result['error']}")
        return
    
    print("\nFit Results:")
    print(f"  beta_0 = {fit_result['beta_0']:.6f} (baseline ionosphere sigma)")
    print(f"  beta_1 = {fit_result['beta_1']:.6f} (coupling to F10.7)")
    print(f"  R^2 = {fit_result['r_squared']:.4f}")
    print(f"  RMSE = {fit_result['rmse']:.6f}")
    print(f"  N points = {fit_result['n_points']}")
    
    print("\nSSZ Signature Check:")
    print(f"  Mode correlation = {fit_result['mode_correlation']:.4f}")
    if fit_result['mode_correlation'] > 0.7:
        print("  -> Strong SSZ signature detected!")
    elif fit_result['mode_correlation'] > 0.5:
        print("  -> Moderate SSZ signature")
    else:
        print("  -> Weak/no SSZ signature")
    
    # Frequency shift at mean sigma
    mean_sigma = fit_result['beta_0']
    delta_seg_eff = fit_result['w_iono'] * mean_sigma
    
    print("\nFrequency Shifts at Mean Conditions:")
    print(f"  Effective delta_seg = {delta_seg_eff:.4f} ({delta_seg_eff*100:.2f}%)")
    
    for n in [1, 2, 3]:
        f_class = f_n_classical(n, f1_ref=7.83)
        f_ssz = f_class / (1 + delta_seg_eff)
        delta_f = f_ssz - f_class
        print(f"  f{n}: {f_class:.2f} Hz -> {f_ssz:.2f} Hz (Df = {delta_f:+.3f} Hz)")
    
    print("\nInterpretation:")
    if abs(fit_result['beta_1']) > 0.001:
        if fit_result['beta_1'] > 0:
            print("  -> Higher solar activity increases segmentation")
            print("     (frequencies decrease with F10.7)")
        else:
            print("  -> Higher solar activity decreases segmentation")
            print("     (frequencies increase with F10.7)")
    else:
        print("  -> No significant coupling to solar activity")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Layered SSZ Schumann Resonance Analysis"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for demonstration"
    )
    parser.add_argument(
        "--schumann-path", type=str,
        help="Path to Schumann data file"
    )
    parser.add_argument(
        "--f107-path", type=str,
        help="Path to F10.7 data file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/layered",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("LAYERED SSZ SCHUMANN ANALYSIS")
    logger.info("=" * 70)
    
    # Load or create data
    if args.synthetic:
        logger.info("\nGenerating synthetic data...")
        
        schumann_ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-06-30",
            freq="1h",
            delta_seg_amplitude=0.015,
            noise_level=0.008,
        )
        
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-06-30",
        )
        
        logger.info(f"  Schumann: {len(schumann_ds.time)} points")
        logger.info(f"  F10.7: {len(f107)} points")
        
    else:
        logger.error("Real data loading not implemented yet.")
        logger.error("Use --synthetic for demonstration.")
        return 1
    
    # Merge data
    logger.info("\nMerging data...")
    merged = merge_all(schumann_ds, f107, kp, time_resolution="1h")
    logger.info(f"  Merged: {len(merged.time)} points")
    
    # Prepare frequency observations
    time_idx = pd.DatetimeIndex(merged.time.values)
    f_obs = {
        n: pd.Series(merged[f"f{n}"].values, index=time_idx)
        for n in [1, 2, 3]
    }
    
    F_iono = pd.Series(merged["f107_norm"].values, index=time_idx)
    
    # Fit layered model
    logger.info("\nFitting layered SSZ model...")
    config = LayeredSSZConfig()
    
    fit_result = fit_layered_model(
        f_obs=f_obs,
        F_iono=F_iono,
        f1_ref=7.83,
        w_atm=config.atmosphere.weight,
        w_iono=config.ionosphere.weight,
    )
    
    # Create plots
    logger.info("\nCreating plots...")
    create_layered_plots(merged, fit_result, output_dir / "plots")
    
    # Print summary
    print_summary(fit_result, config)
    
    # Print frequency table
    print_frequency_table()
    
    # Save results
    results_file = output_dir / "fit_results.txt"
    with open(results_file, "w") as f:
        f.write("LAYERED SSZ FIT RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for key, value in fit_result.items():
            if key != "residuals":
                f.write(f"{key}: {value}\n")
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
