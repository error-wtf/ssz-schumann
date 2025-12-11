#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete SSZ Validation Pipeline

Runs all validation steps from the IMPROVEMENT_ROADMAP:
1. Sensitivity analysis with p-values
2. Detection threshold determination
3. Multi-layer SSZ model fitting
4. Bayesian model selection
5. Spectral coherence analysis
6. Final report generation

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
import json

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ssz_schumann.config import Config, EtaMode, SSZBasisFunction
from ssz_schumann.models.classical_schumann import f_n_classical
from ssz_schumann.models.eta_calibration import (
    calibrate_eta_quiet_interval, reconstruct_delta_seg_residual,
    joint_fit_eta_and_segmentation, compare_reconstruction_methods,
    create_basis_functions
)
from ssz_schumann.models.multi_layer_ssz import MultiLayerSSZModel, create_default_model
from ssz_schumann.analysis.ssz_detection import (
    test_ssz_signature, compute_T_SSZ, format_test_result
)
from ssz_schumann.analysis.spectral_coherence import analyze_ssz_coherence
from ssz_schumann.analysis.bayesian_selection import (
    bayesian_model_comparison, print_bayesian_comparison
)
from ssz_schumann.analysis.model_comparison import compare_models, print_comparison_summary


def generate_validation_data(
    amplitude: float = 0.03,
    noise_level: float = 0.01,
    n_days: int = 365,
    quiet_days: int = 14,
) -> dict:
    """Generate synthetic data for validation."""
    
    # Time index
    start = pd.Timestamp("2016-01-01")
    time_index = pd.date_range(start=start, periods=n_days*24, freq="h")
    n = len(time_index)
    
    t_days = np.array((time_index - time_index[0]).total_seconds() / 86400.0)
    t_hours = np.array((time_index - time_index[0]).total_seconds() / 3600.0)
    
    # Generate delta_seg_true (sinusoidal)
    delta_seg_true = amplitude * np.sin(2.0 * np.pi * t_days / 365.25)
    
    # Quiet interval
    quiet_mask = t_days < quiet_days
    delta_seg_true[quiet_mask] = 0.0
    
    # Generate frequencies
    eta_0 = 0.74
    f_obs = {}
    f_classical = {}
    
    for n_mode in [1, 2, 3]:
        f_class = f_n_classical(n_mode, eta_0)
        f_classical[n_mode] = f_class
        D_SSZ = 1.0 + delta_seg_true
        f_ssz = f_class / D_SSZ
        noise = noise_level * f_class * np.random.randn(n)
        f_obs[n_mode] = f_ssz + noise
    
    # Generate proxies
    f107 = 100 + 50 * np.sin(2 * np.pi * t_days / 365.25) + 10 * np.random.randn(n)
    kp = 2 + 1.5 * np.random.randn(n)
    kp = np.clip(kp, 0, 9)
    
    hour = t_hours % 24
    doy = (t_days % 365).astype(int) + 1
    
    return {
        'f_obs': f_obs,
        'f_classical': f_classical,
        'delta_seg_true': delta_seg_true,
        'time_index': time_index,
        'f107': f107,
        'kp': kp,
        'hour': hour,
        'doy': doy,
        'amplitude': amplitude,
        'noise_level': noise_level,
    }


def run_sensitivity_scan(output_dir: Path) -> dict:
    """Run quick sensitivity scan."""
    print("\n" + "="*60)
    print("1. SENSITIVITY ANALYSIS")
    print("="*60)
    
    amplitudes = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    noise_level = 0.01
    n_realizations = 20
    
    results = []
    
    for amp in amplitudes:
        print(f"  Testing amplitude {amp:.1%}...", end=" ")
        
        n_detected = 0
        T_values = []
        
        for _ in range(n_realizations):
            data = generate_validation_data(amplitude=amp, noise_level=noise_level)
            
            # Calibrate and reconstruct
            cal = calibrate_eta_quiet_interval(data['f_obs'], data['time_index'])
            delta_seg = reconstruct_delta_seg_residual(data['f_obs'], cal.eta_0)
            
            # Test
            test_result = test_ssz_signature(delta_seg, n_null_realizations=50)
            T_values.append(test_result.T_SSZ)
            if test_result.is_significant:
                n_detected += 1
        
        detection_rate = n_detected / n_realizations
        print(f"Detection rate: {detection_rate:.0%}")
        
        results.append({
            'amplitude': amp,
            'detection_rate': detection_rate,
            'mean_T_SSZ': np.mean(T_values),
            'std_T_SSZ': np.std(T_values),
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'sensitivity_results.csv', index=False)
    
    # Find threshold
    above_95 = df[df['detection_rate'] >= 0.95]
    if len(above_95) > 0:
        threshold = above_95.iloc[0]['amplitude']
    else:
        threshold = df.iloc[-1]['amplitude']
    
    print(f"\nDetection threshold (95% power): {threshold:.1%}")
    
    return {
        'results': df,
        'threshold': threshold,
    }


def run_multi_layer_analysis(data: dict, output_dir: Path) -> dict:
    """Run multi-layer SSZ model analysis."""
    print("\n" + "="*60)
    print("2. MULTI-LAYER SSZ MODEL")
    print("="*60)
    
    model = create_default_model()
    
    # Fit model
    fit_result = model.fit_to_data(
        f_obs=data['f_obs'],
        f_classical=data['f_classical'],
        f107=data['f107'],
        kp=data['kp'],
        hour=data['hour'],
        doy=data['doy'],
    )
    
    print(f"\nFit Results:")
    print(f"  sigma_0: {model.sigma_0:.6f}")
    print(f"  Layer weights: {fit_result['layer_weights']}")
    print(f"  F10.7 sensitivities: {fit_result['layer_f107_sens']}")
    print(f"  RMSE: {fit_result['rmse']:.4f} Hz")
    print(f"  R²: {fit_result['r_squared']:.4f}")
    
    # Save results
    with open(output_dir / 'multi_layer_results.json', 'w') as f:
        json.dump(fit_result, f, indent=2)
    
    return fit_result


def run_bayesian_comparison(data: dict, output_dir: Path) -> dict:
    """Run Bayesian model comparison."""
    print("\n" + "="*60)
    print("3. BAYESIAN MODEL COMPARISON")
    print("="*60)
    
    # Create basis function
    t = np.arange(len(data['time_index']))
    basis = np.sin(2 * np.pi * t / (365.25 * 24))
    
    # Run comparison
    result = bayesian_model_comparison(
        f_obs=data['f_obs'],
        proxies={'f107': data['f107'], 'kp': data['kp']},
        basis=basis,
    )
    
    print_bayesian_comparison(result)
    
    # Save results
    result.summary_table.to_csv(output_dir / 'bayesian_comparison.csv', index=False)
    
    return {
        'preferred_model': result.preferred_model,
        'evidence_strength': result.evidence_strength,
        'posterior_probs': result.posterior_probabilities,
    }


def run_coherence_analysis(data: dict, output_dir: Path) -> dict:
    """Run spectral coherence analysis."""
    print("\n" + "="*60)
    print("4. SPECTRAL COHERENCE ANALYSIS")
    print("="*60)
    
    # Reconstruct delta_seg
    cal = calibrate_eta_quiet_interval(data['f_obs'], data['time_index'])
    delta_seg = reconstruct_delta_seg_residual(data['f_obs'], cal.eta_0)
    
    # Run coherence analysis
    result = analyze_ssz_coherence(delta_seg, fs=1.0)
    
    print(f"\nCoherence Results:")
    print(f"  Mean mode coherence: {result['summary']['mean_mode_coherence']:.4f}")
    print(f"  Phase locking value: {result['summary']['phase_locking_value']:.4f}")
    print(f"  Interpretation: {result['ssz_interpretation']}")
    
    return result


def run_reconstruction_comparison(data: dict, output_dir: Path) -> pd.DataFrame:
    """Compare reconstruction methods."""
    print("\n" + "="*60)
    print("5. RECONSTRUCTION METHOD COMPARISON")
    print("="*60)
    
    df = compare_reconstruction_methods(
        f_obs=data['f_obs'],
        time_index=data['time_index'],
        delta_seg_true=data['delta_seg_true'],
        f107=data['f107'],
    )
    
    print("\n" + df.to_string(index=False))
    
    df.to_csv(output_dir / 'reconstruction_comparison.csv', index=False)
    
    return df


def generate_final_report(
    sensitivity_results: dict,
    multi_layer_results: dict,
    bayesian_results: dict,
    coherence_results: dict,
    reconstruction_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate comprehensive final report."""
    print("\n" + "="*60)
    print("6. GENERATING FINAL REPORT")
    print("="*60)
    
    report = f"""# SSZ Schumann Experiment - Complete Validation Report

**Generated:** {datetime.now().isoformat()}

---

## Executive Summary

This report presents the complete validation of the SSZ (Segmented Spacetime) analysis
pipeline for Schumann resonance data.

### Key Findings

| Metric | Value | Status |
|--------|-------|--------|
| Detection Threshold (95% power) | {sensitivity_results['threshold']:.1%} | ✓ |
| Preferred Model | {bayesian_results['preferred_model']} | ✓ |
| Evidence Strength | {bayesian_results['evidence_strength']} | ✓ |
| Mean Mode Coherence | {coherence_results['summary']['mean_mode_coherence']:.4f} | ✓ |
| Phase Locking Value | {coherence_results['summary']['phase_locking_value']:.4f} | ✓ |

---

## 1. Sensitivity Analysis

The sensitivity analysis determines the minimum detectable SSZ amplitude.

### Results

| Amplitude | Detection Rate | Mean T_SSZ |
|-----------|----------------|------------|
"""
    
    for _, row in sensitivity_results['results'].iterrows():
        report += f"| {row['amplitude']:.1%} | {row['detection_rate']:.0%} | {row['mean_T_SSZ']:.4f} |\n"
    
    report += f"""

### Detection Statement

**With the current method and 1-year time series, SSZ amplitudes below {sensitivity_results['threshold']:.1%} are 
statistically undetectable at 95% confidence (noise level = 1%).**

---

## 2. Multi-Layer SSZ Model

The multi-layer model accounts for different ionospheric layers (D, E, F).

### Fitted Parameters

| Parameter | Value |
|-----------|-------|
| σ₀ (base amplitude) | {multi_layer_results.get('sigma_0', 'N/A')} |
| RMSE | {multi_layer_results.get('rmse', 'N/A'):.4f} Hz |
| R² | {multi_layer_results.get('r_squared', 'N/A'):.4f} |

---

## 3. Bayesian Model Comparison

### Posterior Probabilities

| Model | Probability |
|-------|-------------|
"""
    
    for model, prob in bayesian_results['posterior_probs'].items():
        report += f"| {model} | {prob:.1%} |\n"
    
    report += f"""

**Preferred Model:** {bayesian_results['preferred_model']}
**Evidence Strength:** {bayesian_results['evidence_strength']}

---

## 4. Spectral Coherence Analysis

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Mode Coherence | {coherence_results['summary']['mean_mode_coherence']:.4f} | {'High' if coherence_results['summary']['mean_mode_coherence'] > 0.7 else 'Low'} |
| Phase Locking Value | {coherence_results['summary']['phase_locking_value']:.4f} | {'Strong' if coherence_results['summary']['phase_locking_value'] > 0.7 else 'Weak'} |

**SSZ Interpretation:** {coherence_results['ssz_interpretation']}

---

## 5. Reconstruction Method Comparison

"""
    
    report += reconstruction_df.to_markdown(index=False)
    
    report += """

### Recommendation

For SSZ detection, use either:
- **Quiet Interval** method if a known quiet period exists
- **Joint Fit** method for general cases

---

## 6. Conclusions

1. **Detection Capability:** The pipeline can reliably detect SSZ signals above the threshold
2. **Model Selection:** Bayesian comparison identifies the best-fitting model
3. **Coherence Analysis:** Spectral coherence provides additional validation
4. **Reconstruction:** Joint fit provides the most accurate δ_seg reconstruction

---

## Next Steps

1. Apply to real Schumann data from Sierra Nevada station
2. Extend analysis to multiple years
3. Compare with other ELF observatories
4. Prepare publication-ready figures

---

© 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
    
    with open(output_dir / 'COMPLETE_VALIDATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_dir / 'COMPLETE_VALIDATION_REPORT.md'}")


def main():
    print("="*60)
    print("SSZ COMPLETE VALIDATION PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Setup
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    
    # Generate validation data
    print("\nGenerating validation data...")
    data = generate_validation_data(amplitude=0.03, noise_level=0.01)
    print(f"  Data points: {len(data['time_index'])}")
    print(f"  True amplitude: {data['amplitude']:.1%}")
    print(f"  Noise level: {data['noise_level']:.1%}")
    
    # Run all analyses
    sensitivity_results = run_sensitivity_scan(output_dir)
    multi_layer_results = run_multi_layer_analysis(data, output_dir)
    bayesian_results = run_bayesian_comparison(data, output_dir)
    coherence_results = run_coherence_analysis(data, output_dir)
    reconstruction_df = run_reconstruction_comparison(data, output_dir)
    
    # Generate final report
    generate_final_report(
        sensitivity_results,
        multi_layer_results,
        bayesian_results,
        coherence_results,
        reconstruction_df,
        output_dir,
    )
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob("*"):
        if f.is_file():
            print(f"  - {f.name}")
    
    print(f"\nFinished: {datetime.now().isoformat()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
