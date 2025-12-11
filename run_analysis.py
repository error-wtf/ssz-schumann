#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Analysis - Main Entry Point

This script runs the complete SSZ analysis pipeline with configurable options.

Usage:
    python run_analysis.py --mode synthetic --eta-mode full_fit
    python run_analysis.py --mode synthetic --eta-mode quiet_interval --ssz-amplitude 0.03
    python run_analysis.py --mode synthetic --run-null-test --n-null 1000
    python run_analysis.py --mode synthetic --run-sensitivity

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import warnings

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ssz_schumann.config import (
    Config, EtaMode, DataMode, SSZBasisFunction,
    ClassicalParams, SSZParams, DEFAULT_CONFIG
)
from ssz_schumann.models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
from ssz_schumann.models.ssz_correction import delta_seg_from_observed, check_mode_consistency
from ssz_schumann.models.eta_calibration import (
    calibrate_eta, calibrate_eta_full_fit, calibrate_eta_quiet_interval,
    joint_fit_eta_and_segmentation, compare_reconstruction_methods,
    create_basis_functions, reconstruct_delta_seg_residual
)
from ssz_schumann.analysis.ssz_detection import (
    test_ssz_signature, compute_null_distribution, run_sensitivity_analysis,
    format_test_result, format_sensitivity_result, compute_T_SSZ
)
from ssz_schumann.analysis.model_comparison import compare_models, print_comparison_summary


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(
    config: Config,
    n_days: int = 365,
    freq: str = "1H",
) -> pd.DataFrame:
    """
    Generate synthetic Schumann resonance data with SSZ signal.
    
    Args:
        config: Configuration object
        n_days: Number of days
        freq: Time frequency
    
    Returns:
        DataFrame with time, f1, f2, f3, delta_seg_true, f107
    """
    # Time index
    start = pd.Timestamp("2016-01-01")
    time_index = pd.date_range(start=start, periods=n_days*24, freq=freq)
    n = len(time_index)
    
    # Time in days from start
    t_days = np.array((time_index - time_index[0]).total_seconds() / 86400.0)
    
    # Generate delta_seg_true based on basis function
    ssz = config.ssz
    basis_1, basis_2 = create_basis_functions(
        time_index,
        f107=None,
        basis_type=ssz.basis_function,
        period_days=ssz.period_days,
        phase_offset=ssz.phase_offset,
    )
    
    delta_seg_true = ssz.amplitude_A * basis_1 + ssz.amplitude_B * basis_2
    
    # Quiet interval: set delta_seg to 0 for first N days
    quiet_days = config.classical.quiet_interval_days
    quiet_mask = t_days < quiet_days
    delta_seg_true = np.array(delta_seg_true)  # Ensure it's a mutable numpy array
    delta_seg_true[quiet_mask] = 0.0
    
    # Generate classical frequencies
    eta_0 = config.classical.eta_0
    
    f_classical = {}
    f_obs = {}
    
    for n_mode in config.modes:
        f_class = f_n_classical(n_mode, eta_0)
        f_classical[n_mode] = f_class
        
        # Apply SSZ correction: f_obs = f_class / (1 + delta_seg)
        D_SSZ = 1.0 + delta_seg_true
        f_ssz = f_class / D_SSZ
        
        # Add noise
        noise = ssz.noise_level * f_class * np.random.randn(len(time_index))
        f_obs[n_mode] = f_ssz + noise
    
    # Synthetic F10.7 (solar cycle approximation)
    f107 = 100 + 50 * np.sin(2 * np.pi * t_days / 365.25)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_index,
        'f1': f_obs[1],
        'f2': f_obs[2],
        'f3': f_obs[3],
        'delta_seg_true': delta_seg_true,
        'f107': f107,
    })
    
    return df


def load_real_data(config: Config) -> pd.DataFrame:
    """
    Load real Schumann resonance data.
    
    Args:
        config: Configuration object
    
    Returns:
        DataFrame with time, f1, f2, f3, f107, kp
    """
    data_path = config.data_dir / "schumann" / "realistic_schumann_2016.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['time'] = pd.to_datetime(df['time'])
    
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def run_ssz_analysis(
    df: pd.DataFrame,
    config: Config,
) -> dict:
    """
    Run complete SSZ analysis.
    
    Args:
        df: DataFrame with frequency data
        config: Configuration object
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Extract data
    time_index = pd.DatetimeIndex(df['time'])
    f_obs = {n: df[f'f{n}'].values for n in config.modes}
    f107 = df['f107'].values if 'f107' in df.columns else None
    delta_seg_true = df['delta_seg_true'].values if 'delta_seg_true' in df.columns else None
    
    results['n_times'] = len(time_index)
    results['modes'] = config.modes
    
    # ==========================================================================
    # 1. Calibrate eta_0
    # ==========================================================================
    print("\n" + "="*60)
    print("1. ETA_0 CALIBRATION")
    print("="*60)
    
    cal_result = calibrate_eta(f_obs, config, time_index)
    eta_0 = cal_result.eta_0
    
    print(f"Method: {cal_result.method}")
    print(f"eta_0 = {eta_0:.6f}")
    print(f"Residuals: mean={cal_result.residuals_mean:.4f}, std={cal_result.residuals_std:.4f}")
    
    results['calibration'] = {
        'method': cal_result.method,
        'eta_0': eta_0,
        'residuals_mean': cal_result.residuals_mean,
        'residuals_std': cal_result.residuals_std,
    }
    
    # ==========================================================================
    # 2. Reconstruct delta_seg
    # ==========================================================================
    print("\n" + "="*60)
    print("2. DELTA_SEG RECONSTRUCTION")
    print("="*60)
    
    delta_seg = reconstruct_delta_seg_residual(f_obs, eta_0, config.modes)
    
    for n in config.modes:
        mean_ds = np.mean(delta_seg[n])
        std_ds = np.std(delta_seg[n])
        print(f"Mode {n}: delta_seg = {mean_ds:.6f} +/- {std_ds:.6f}")
    
    results['delta_seg'] = {n: {'mean': np.mean(delta_seg[n]), 'std': np.std(delta_seg[n])} 
                           for n in config.modes}
    
    # ==========================================================================
    # 3. SSZ Detection Test
    # ==========================================================================
    print("\n" + "="*60)
    print("3. SSZ DETECTION TEST")
    print("="*60)
    
    test_result = test_ssz_signature(
        delta_seg,
        modes=config.modes,
        n_null_realizations=config.n_null_realizations,
        alpha=config.significance_level,
    )
    
    print(format_test_result(test_result))
    
    results['ssz_test'] = {
        'T_SSZ': test_result.T_SSZ,
        'p_value': test_result.p_value,
        'is_significant': test_result.is_significant,
        'correlation_mean': test_result.correlation_mean,
        'scatter_ratio': test_result.scatter_ratio,
    }
    
    # ==========================================================================
    # 4. Model Comparison
    # ==========================================================================
    print("\n" + "="*60)
    print("4. MODEL COMPARISON")
    print("="*60)
    
    comparison = compare_models(f_obs)
    print_comparison_summary(comparison)
    
    results['model_comparison'] = {
        'preferred_model': comparison.preferred_model,
        'delta_aic': comparison.delta_aic,
        'delta_bic': comparison.delta_bic,
        'classical_rmse': comparison.classical_result.rmse,
        'ssz_rmse': comparison.ssz_result.rmse,
    }
    
    # ==========================================================================
    # 5. Reconstruction Comparison (if true delta_seg available)
    # ==========================================================================
    if delta_seg_true is not None:
        print("\n" + "="*60)
        print("5. RECONSTRUCTION METHOD COMPARISON")
        print("="*60)
        
        comparison_df = compare_reconstruction_methods(
            f_obs, time_index, delta_seg_true, f107, config.modes,
            quiet_days=config.classical.quiet_interval_days
        )
        
        print(comparison_df.to_string(index=False))
        
        results['reconstruction_comparison'] = comparison_df.to_dict('records')
    
    # ==========================================================================
    # 6. Joint Fit
    # ==========================================================================
    print("\n" + "="*60)
    print("6. JOINT FIT (eta_0 + delta_seg)")
    print("="*60)
    
    joint_result = joint_fit_eta_and_segmentation(
        f_obs, time_index, f107,
        basis_type=config.ssz.basis_function,
        period_days=config.ssz.period_days,
        modes=config.modes,
    )
    
    print(f"eta_0 (joint): {joint_result.eta_0:.6f}")
    print(f"Amplitude A: {joint_result.amplitude_A:.6f}")
    print(f"Amplitude B: {joint_result.amplitude_B:.6f}")
    print(f"RMSE: {joint_result.rmse:.4f} Hz")
    print(f"R²: {joint_result.r_squared:.4f}")
    print(f"Converged: {joint_result.converged}")
    
    results['joint_fit'] = {
        'eta_0': joint_result.eta_0,
        'amplitude_A': joint_result.amplitude_A,
        'amplitude_B': joint_result.amplitude_B,
        'rmse': joint_result.rmse,
        'r_squared': joint_result.r_squared,
        'converged': joint_result.converged,
    }
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_reports(results: dict, config: Config, output_dir: Path):
    """Generate markdown reports."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # SSZ Detection Null Test Report
    # ==========================================================================
    null_report = f"""# SSZ Detection Null Test Report

**Generated:** {datetime.now().isoformat()}

## Test Statistic Definition

The SSZ test statistic T_SSZ is defined as:

```
T_SSZ = w_corr * r_mean + w_scatter * (1 - scatter_ratio)
```

where:
- `r_mean` = mean pairwise correlation between delta_seg from different modes
- `scatter_ratio` = std(between modes) / mean(within mode std)
- `w_corr = 0.7`, `w_scatter = 0.3` (default weights)

Higher T_SSZ indicates stronger SSZ signature.

## Null Hypothesis

H0: No SSZ signal present (delta_seg variations are independent noise)

The null distribution is generated by shuffling time indices independently per mode,
which destroys any true temporal correlation while preserving marginal distributions.

## Results

| Metric | Value |
|--------|-------|
| T_SSZ (observed) | {results['ssz_test']['T_SSZ']:.4f} |
| P-value | {results['ssz_test']['p_value']:.4f} |
| Significant (α=0.05) | {results['ssz_test']['is_significant']} |
| Mean mode correlation | {results['ssz_test']['correlation_mean']:.4f} |
| Scatter ratio | {results['ssz_test']['scatter_ratio']:.4f} |

## Interpretation

{"**SSZ signature detected** at the 95% confidence level." if results['ssz_test']['is_significant'] else "**No significant SSZ signature detected** at the 95% confidence level."}

The observed T_SSZ of {results['ssz_test']['T_SSZ']:.4f} {"exceeds" if results['ssz_test']['is_significant'] else "does not exceed"} 
the critical value from the null distribution (p = {results['ssz_test']['p_value']:.4f}).

---

© 2025 Carmen Wrede & Lino Casu
"""
    
    with open(output_dir / "SSZ_DETECTION_NULLTEST.md", 'w', encoding='utf-8') as f:
        f.write(null_report)
    
    # ==========================================================================
    # Reconstruction Comparison Report
    # ==========================================================================
    if 'reconstruction_comparison' in results:
        recon_data = results['reconstruction_comparison']
        
        recon_report = f"""# Segmentation Reconstruction Comparison

**Generated:** {datetime.now().isoformat()}

## Methods Compared

1. **Full Fit**: Calibrate eta_0 on full dataset (has degeneracy)
2. **Quiet Interval**: Calibrate eta_0 only on quiet period (first {config.classical.quiet_interval_days} days)
3. **Joint Fit**: Simultaneously fit eta_0 and delta_seg parameters

## Results

| Method | eta_0 | Bias | RMSE | Correlation | Mean δ_seg | Std δ_seg |
|--------|-------|------|------|-------------|------------|-----------|
"""
        
        for row in recon_data:
            recon_report += f"| {row['method']} | {row['eta_0']:.6f} | {row['bias']:.6f} | {row['rmse']:.6f} | {row['correlation']:.4f} | {row['mean_delta_seg']:.6f} | {row['std_delta_seg']:.6f} |\n"
        
        recon_report += f"""

## Key Findings

1. **Degeneracy Problem**: The full_fit method absorbs constant SSZ shifts into eta_0,
   resulting in near-zero mean reconstructed delta_seg.

2. **Quiet Interval Method**: By calibrating eta_0 only on the quiet period,
   we can recover the true mean delta_seg level.

3. **Joint Fit Method**: Simultaneously fitting eta_0 and delta_seg parameters
   provides the best reconstruction with lowest bias and RMSE.

## Recommendation

For SSZ detection, use either:
- **Quiet Interval** method if a known quiet period exists
- **Joint Fit** method for general cases

---

© 2025 Carmen Wrede & Lino Casu
"""
        
        with open(output_dir / "SEGMENTATION_RECONSTRUCTION_COMPARISON.md", 'w', encoding='utf-8') as f:
            f.write(recon_report)
    
    # ==========================================================================
    # Save results as JSON
    # ==========================================================================
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_dir / "analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nReports saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSZ Schumann Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --mode synthetic
  python run_analysis.py --mode synthetic --eta-mode quiet_interval
  python run_analysis.py --mode synthetic --ssz-amplitude 0.03 --noise 0.01
  python run_analysis.py --mode real
        """
    )
    
    # Data mode
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic',
                        help='Data mode (default: synthetic)')
    
    # Eta calibration
    parser.add_argument('--eta-mode', choices=['full_fit', 'quiet_interval', 'fixed'],
                        default='full_fit', help='Eta calibration mode')
    parser.add_argument('--eta-fixed', type=float, default=0.74,
                        help='Fixed eta_0 value (for --eta-mode fixed)')
    parser.add_argument('--quiet-days', type=int, default=14,
                        help='Quiet interval days (for --eta-mode quiet_interval)')
    
    # SSZ parameters
    parser.add_argument('--ssz-amplitude', type=float, default=0.02,
                        help='SSZ amplitude (default: 0.02 = 2%%)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level (default: 0.01 = 1%%)')
    
    # Statistical tests
    parser.add_argument('--n-null', type=int, default=1000,
                        help='Number of null realizations')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SSZ SCHUMANN ANALYSIS")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Eta calibration: {args.eta_mode}")
    print(f"SSZ amplitude: {args.ssz_amplitude:.1%}")
    print(f"Noise level: {args.noise:.1%}")
    
    # Create configuration
    config = Config(
        data_mode=DataMode(args.mode),
        output_dir=Path(args.output_dir),
        classical=ClassicalParams(
            eta_mode=EtaMode(args.eta_mode),
            eta_0_fixed=args.eta_fixed,
            quiet_interval_days=args.quiet_days,
        ),
        ssz=SSZParams(
            amplitude_A=args.ssz_amplitude,
            noise_level=args.noise,
        ),
        n_null_realizations=args.n_null,
        significance_level=args.alpha,
    )
    
    # Load or generate data
    if config.data_mode == DataMode.SYNTHETIC:
        print("\nGenerating synthetic data...")
        df = generate_synthetic_data(config)
    else:
        print("\nLoading real data...")
        df = load_real_data(config)
    
    print(f"Data points: {len(df)}")
    
    # Run analysis
    results = run_ssz_analysis(df, config)
    
    # Generate reports
    generate_reports(results, config, config.output_dir)
    
    # Save config
    config.save()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"""
Data: {args.mode} ({len(df)} points)
Eta calibration: {args.eta_mode} (eta_0 = {results['calibration']['eta_0']:.6f})

SSZ Detection:
  T_SSZ = {results['ssz_test']['T_SSZ']:.4f}
  P-value = {results['ssz_test']['p_value']:.4f}
  Significant: {results['ssz_test']['is_significant']}

Model Comparison:
  Preferred: {results['model_comparison']['preferred_model']}
  Delta BIC: {results['model_comparison']['delta_bic']:.1f}

Joint Fit:
  eta_0 = {results['joint_fit']['eta_0']:.6f}
  Amplitude A = {results['joint_fit']['amplitude_A']:.6f}
  R² = {results['joint_fit']['r_squared']:.4f}

Reports saved to: {config.output_dir}
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
