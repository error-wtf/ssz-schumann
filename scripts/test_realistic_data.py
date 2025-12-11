#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test SSZ Analysis with Realistic Schumann Data

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
from ssz_schumann.models.ssz_correction import delta_seg_from_observed, check_mode_consistency
from ssz_schumann.analysis.model_comparison import compare_models, print_comparison_summary


def main():
    print("="*60)
    print("SSZ ANALYSIS WITH REALISTIC SCHUMANN DATA")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "schumann" / "realistic_schumann_2016.csv"
    print(f"\nLoading: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Frequency statistics
    print("\n" + "-"*40)
    print("FREQUENCY STATISTICS")
    print("-"*40)
    print(f"  f1: {df['f1'].mean():.3f} +/- {df['f1'].std():.3f} Hz")
    print(f"  f2: {df['f2'].mean():.3f} +/- {df['f2'].std():.3f} Hz")
    print(f"  f3: {df['f3'].mean():.3f} +/- {df['f3'].std():.3f} Hz")
    print(f"  F10.7: {df['f107'].mean():.1f} +/- {df['f107'].std():.1f} SFU")
    print(f"  Kp: {df['kp'].mean():.2f} +/- {df['kp'].std():.2f}")
    
    # Classical calibration
    print("\n" + "-"*40)
    print("CLASSICAL CALIBRATION")
    print("-"*40)
    
    eta_0 = compute_eta0_from_mean_f1(df['f1'].values)
    print(f"Calibrated eta_0 = {eta_0:.6f}")
    
    for n in [1, 2, 3]:
        f_class = f_n_classical(n, eta_0)
        f_obs_mean = df[f'f{n}'].mean()
        residual = f_obs_mean - f_class
        print(f"  Mode {n}: f_class = {f_class:.3f} Hz, f_obs = {f_obs_mean:.3f} Hz, residual = {residual:.4f} Hz")
    
    # SSZ Analysis
    print("\n" + "-"*40)
    print("SSZ ANALYSIS")
    print("-"*40)
    
    delta_seg_dict = {}
    for n in [1, 2, 3]:
        f_obs = df[f'f{n}'].values
        f_class = f_n_classical(n, eta_0)
        delta_seg_dict[n] = delta_seg_from_observed(f_obs, f_class)
        mean_ds = np.mean(delta_seg_dict[n])
        std_ds = np.std(delta_seg_dict[n])
        print(f"  Mode {n}: delta_seg = {mean_ds:.6f} +/- {std_ds:.6f}")
    
    # Mode consistency check
    print("\n" + "-"*40)
    print("MODE CONSISTENCY CHECK (SSZ Signature)")
    print("-"*40)
    
    consistency = check_mode_consistency(delta_seg_dict)
    print(f"  Mean correlation: {consistency['mean_correlation']:.4f}")
    print(f"  SSZ score: {consistency['ssz_score']:.4f}")
    print(f"  Is consistent: {consistency['is_consistent']}")
    print(f"  Interpretation: {consistency['interpretation']}")
    
    # Correlation with space weather
    print("\n" + "-"*40)
    print("CORRELATION WITH SPACE WEATHER")
    print("-"*40)
    
    # Add delta_seg to dataframe
    df['delta_seg_1'] = delta_seg_dict[1]
    df['delta_seg_2'] = delta_seg_dict[2]
    df['delta_seg_3'] = delta_seg_dict[3]
    df['delta_seg_mean'] = (delta_seg_dict[1] + delta_seg_dict[2] + delta_seg_dict[3]) / 3
    
    corr_f107 = df['f107_norm'].corr(df['delta_seg_mean'])
    corr_kp = df['kp_norm'].corr(df['delta_seg_mean'])
    
    print(f"  delta_seg vs F10.7: r = {corr_f107:.4f}")
    print(f"  delta_seg vs Kp: r = {corr_kp:.4f}")
    
    # Model comparison
    print("\n" + "-"*40)
    print("MODEL COMPARISON (Classical vs SSZ)")
    print("-"*40)
    
    f_obs = {
        1: df['f1'].values,
        2: df['f2'].values,
        3: df['f3'].values,
    }
    
    result = compare_models(f_obs)
    print_comparison_summary(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"""
Data: Realistic Schumann 2016 (with real F10.7/Kp)
Records: {len(df)}

Frequencies:
  f1 = {df['f1'].mean():.3f} +/- {df['f1'].std():.3f} Hz
  f2 = {df['f2'].mean():.3f} +/- {df['f2'].std():.3f} Hz
  f3 = {df['f3'].mean():.3f} +/- {df['f3'].std():.3f} Hz

SSZ Analysis:
  Mode correlation: {consistency['mean_correlation']:.4f}
  SSZ score: {consistency['ssz_score']:.4f}
  Interpretation: {consistency['interpretation']}

Space Weather Correlation:
  delta_seg vs F10.7: r = {corr_f107:.4f}
  delta_seg vs Kp: r = {corr_kp:.4f}

Model Comparison:
  Classical RMSE: {result.classical_result.rmse:.4f} Hz
  SSZ RMSE: {result.ssz_result.rmse:.4f} Hz
  Delta BIC: {result.delta_bic:.1f}
  Preferred: {result.preferred_model}
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
