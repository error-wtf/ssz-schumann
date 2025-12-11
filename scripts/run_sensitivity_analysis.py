#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Sensitivity Analysis with P-Values

Determines the detection threshold for SSZ signals as a function of:
- SSZ amplitude
- Noise level
- Number of data points

Generates:
- Sensitivity curves (detection probability vs amplitude)
- Detection threshold at 95% power
- P-value distributions

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.config import Config, EtaMode, SSZBasisFunction, ClassicalParams, SSZParams
from ssz_schumann.models.classical_schumann import f_n_classical
from ssz_schumann.models.eta_calibration import (
    calibrate_eta_quiet_interval, reconstruct_delta_seg_residual, create_basis_functions
)
from ssz_schumann.analysis.ssz_detection import (
    test_ssz_signature, compute_T_SSZ, compute_null_distribution
)


def generate_synthetic_data_for_sensitivity(
    amplitude: float,
    noise_level: float,
    n_days: int = 365,
    quiet_days: int = 14,
    eta_0: float = 0.74,
    modes: List[int] = [1, 2, 3],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, pd.DatetimeIndex]:
    """
    Generate synthetic data for sensitivity analysis.
    
    Args:
        amplitude: SSZ amplitude (e.g., 0.03 for 3%)
        noise_level: Relative noise level (e.g., 0.01 for 1%)
        n_days: Number of days
        quiet_days: Days with no SSZ signal at start
        eta_0: Classical slowdown factor
        modes: Modes to generate
    
    Returns:
        Tuple of (f_obs, delta_seg_true, time_index)
    """
    # Time index
    start = pd.Timestamp("2016-01-01")
    time_index = pd.date_range(start=start, periods=n_days*24, freq="h")
    n = len(time_index)
    
    # Time in days
    t_days = np.array((time_index - time_index[0]).total_seconds() / 86400.0)
    
    # Generate delta_seg_true (sinusoidal)
    delta_seg_true = amplitude * np.sin(2.0 * np.pi * t_days / 365.25)
    
    # Quiet interval
    quiet_mask = t_days < quiet_days
    delta_seg_true[quiet_mask] = 0.0
    
    # Generate frequencies
    f_obs = {}
    for n_mode in modes:
        f_class = f_n_classical(n_mode, eta_0)
        D_SSZ = 1.0 + delta_seg_true
        f_ssz = f_class / D_SSZ
        noise = noise_level * f_class * np.random.randn(n)
        f_obs[n_mode] = f_ssz + noise
    
    return f_obs, delta_seg_true, time_index


def run_single_realization(
    amplitude: float,
    noise_level: float,
    n_null: int = 100,
    alpha: float = 0.05,
) -> Dict:
    """
    Run a single realization and return detection result.
    """
    # Generate data
    f_obs, delta_seg_true, time_index = generate_synthetic_data_for_sensitivity(
        amplitude, noise_level
    )
    
    # Calibrate eta_0 on quiet interval
    cal_result = calibrate_eta_quiet_interval(f_obs, time_index, quiet_days=14)
    
    # Reconstruct delta_seg
    delta_seg = reconstruct_delta_seg_residual(f_obs, cal_result.eta_0)
    
    # Test for SSZ signature
    test_result = test_ssz_signature(
        delta_seg,
        n_null_realizations=n_null,
        alpha=alpha,
    )
    
    return {
        'T_SSZ': test_result.T_SSZ,
        'p_value': test_result.p_value,
        'is_significant': test_result.is_significant,
        'correlation': test_result.correlation_mean,
    }


def run_sensitivity_scan(
    amplitudes: np.ndarray,
    noise_level: float = 0.01,
    n_realizations: int = 50,
    n_null: int = 100,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run sensitivity scan over multiple amplitudes.
    
    Args:
        amplitudes: Array of SSZ amplitudes to test
        noise_level: Noise level
        n_realizations: Number of realizations per amplitude
        n_null: Number of null realizations for p-value
        alpha: Significance level
    
    Returns:
        DataFrame with results
    """
    results = []
    
    for amp in amplitudes:
        print(f"  Amplitude {amp:.1%}...", end=" ", flush=True)
        
        T_values = []
        p_values = []
        n_detected = 0
        
        for _ in range(n_realizations):
            result = run_single_realization(amp, noise_level, n_null, alpha)
            T_values.append(result['T_SSZ'])
            p_values.append(result['p_value'])
            if result['is_significant']:
                n_detected += 1
        
        detection_rate = n_detected / n_realizations
        print(f"Detection rate: {detection_rate:.1%}")
        
        results.append({
            'amplitude': amp,
            'noise_level': noise_level,
            'detection_rate': detection_rate,
            'mean_T_SSZ': np.mean(T_values),
            'std_T_SSZ': np.std(T_values),
            'mean_p_value': np.mean(p_values),
            'median_p_value': np.median(p_values),
        })
    
    return pd.DataFrame(results)


def find_detection_threshold(df: pd.DataFrame, target_power: float = 0.95) -> float:
    """
    Find the minimum amplitude for target detection power.
    """
    above_threshold = df['detection_rate'] >= target_power
    if np.any(above_threshold):
        idx = np.argmax(above_threshold)
        return df.iloc[idx]['amplitude']
    else:
        return np.inf


def plot_sensitivity_curves(
    results_dict: Dict[float, pd.DataFrame],
    output_dir: Path,
):
    """
    Plot sensitivity curves for different noise levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Detection rate vs amplitude
    ax1 = axes[0]
    for noise_level, df in results_dict.items():
        ax1.plot(df['amplitude'] * 100, df['detection_rate'] * 100, 
                 'o-', label=f'Noise {noise_level:.1%}', linewidth=2, markersize=6)
    
    ax1.axhline(y=95, color='r', linestyle='--', alpha=0.7, label='95% power')
    ax1.axhline(y=5, color='gray', linestyle=':', alpha=0.7, label='Type I error')
    ax1.set_xlabel('SSZ Amplitude (%)', fontsize=12)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_title('SSZ Detection Power vs Amplitude', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Mean T_SSZ vs amplitude
    ax2 = axes[1]
    for noise_level, df in results_dict.items():
        ax2.errorbar(df['amplitude'] * 100, df['mean_T_SSZ'], 
                     yerr=df['std_T_SSZ'], fmt='o-', 
                     label=f'Noise {noise_level:.1%}', linewidth=2, markersize=6, capsize=3)
    
    ax2.axhline(y=0.7, color='g', linestyle='--', alpha=0.7, label='Strong SSZ threshold')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate SSZ threshold')
    ax2.set_xlabel('SSZ Amplitude (%)', fontsize=12)
    ax2.set_ylabel('T_SSZ', fontsize=12)
    ax2.set_title('SSZ Test Statistic vs Amplitude', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'sensitivity_curves.png'}")


def plot_pvalue_distribution(
    amplitudes: List[float],
    noise_level: float,
    n_realizations: int,
    output_dir: Path,
):
    """
    Plot p-value distributions for different amplitudes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, amp in enumerate(amplitudes[:6]):
        ax = axes[i]
        
        p_values = []
        for _ in range(n_realizations):
            result = run_single_realization(amp, noise_level, n_null=100)
            p_values.append(result['p_value'])
        
        ax.hist(p_values, bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(x=0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
        ax.set_xlabel('P-value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'Amplitude = {amp:.1%}', fontsize=12)
        ax.legend()
        ax.set_xlim(0, 1)
    
    plt.suptitle(f'P-value Distributions (Noise = {noise_level:.1%})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'pvalue_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir / 'pvalue_distributions.png'}")


def generate_sensitivity_report(
    results_dict: Dict[float, pd.DataFrame],
    output_dir: Path,
):
    """
    Generate markdown report for sensitivity analysis.
    """
    report = f"""# SSZ Sensitivity Analysis Report

**Generated:** {datetime.now().isoformat()}

## Overview

This report presents the sensitivity analysis for SSZ detection using the T_SSZ test statistic.
The analysis determines the minimum detectable SSZ amplitude at different noise levels.

## Detection Thresholds

| Noise Level | Detection Threshold (95% power) |
|-------------|--------------------------------|
"""
    
    for noise_level, df in results_dict.items():
        threshold = find_detection_threshold(df)
        if threshold == np.inf:
            threshold_str = "> max tested"
        else:
            threshold_str = f"{threshold:.1%}"
        report += f"| {noise_level:.1%} | {threshold_str} |\n"
    
    report += """

## Key Findings

"""
    
    # Find typical threshold
    typical_noise = 0.01
    if typical_noise in results_dict:
        df = results_dict[typical_noise]
        threshold = find_detection_threshold(df)
        
        report += f"""### Detection Statement

**With the current method and 1-year time series, SSZ amplitudes below {threshold:.1%} are 
statistically undetectable at 95% confidence (noise level = {typical_noise:.1%}).**

"""
    
    report += """## Detailed Results

"""
    
    for noise_level, df in results_dict.items():
        report += f"""### Noise Level = {noise_level:.1%}

| Amplitude | Detection Rate | Mean T_SSZ | Std T_SSZ | Mean P-value |
|-----------|----------------|------------|-----------|--------------|
"""
        for _, row in df.iterrows():
            report += f"| {row['amplitude']:.1%} | {row['detection_rate']:.1%} | {row['mean_T_SSZ']:.4f} | {row['std_T_SSZ']:.4f} | {row['mean_p_value']:.4f} |\n"
        
        report += "\n"
    
    report += """## Methodology

1. **Data Generation:** Synthetic Schumann frequencies with sinusoidal SSZ signal
2. **Quiet Interval:** First 14 days with δ_seg = 0 for η₀ calibration
3. **Test Statistic:** T_SSZ = 0.7 × r_mean + 0.3 × (1 - scatter_ratio)
4. **Null Distribution:** 100 shuffle realizations per test
5. **Significance Level:** α = 0.05

## Interpretation

- **Detection Rate > 95%:** SSZ signal reliably detectable
- **Detection Rate ~ 5%:** At Type I error rate (no real signal)
- **T_SSZ > 0.7:** Strong SSZ signature
- **T_SSZ 0.5-0.7:** Moderate SSZ signature

---

© 2025 Carmen Wrede & Lino Casu
"""
    
    with open(output_dir / 'SENSITIVITY_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved: {output_dir / 'SENSITIVITY_ANALYSIS.md'}")


def main():
    print("="*60)
    print("SSZ SENSITIVITY ANALYSIS")
    print("="*60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Define amplitudes to test
    amplitudes = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05])
    
    # Define noise levels
    noise_levels = [0.005, 0.01, 0.02]
    
    # Number of realizations (reduce for faster testing)
    n_realizations = 30
    
    results_dict = {}
    
    for noise_level in noise_levels:
        print(f"\n{'='*40}")
        print(f"Noise Level: {noise_level:.1%}")
        print("="*40)
        
        df = run_sensitivity_scan(
            amplitudes,
            noise_level=noise_level,
            n_realizations=n_realizations,
            n_null=100,
        )
        
        results_dict[noise_level] = df
        
        # Find detection threshold
        threshold = find_detection_threshold(df)
        print(f"\nDetection threshold (95% power): {threshold:.1%}")
    
    # Generate plots
    print("\n" + "="*40)
    print("Generating plots...")
    print("="*40)
    
    plot_sensitivity_curves(results_dict, output_dir)
    
    # Generate report
    print("\n" + "="*40)
    print("Generating report...")
    print("="*40)
    
    generate_sensitivity_report(results_dict, output_dir)
    
    # Save raw results
    for noise_level, df in results_dict.items():
        df.to_csv(output_dir / f'sensitivity_noise_{noise_level:.3f}.csv', index=False)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nDetection Thresholds (95% power):")
    for noise_level, df in results_dict.items():
        threshold = find_detection_threshold(df)
        print(f"  Noise {noise_level:.1%}: {threshold:.1%}")
    
    print(f"\nOutput saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
