#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Complete Analysis Pipeline

Vollständige Analyse-Pipeline die alle Komponenten integriert:
1. Daten laden (echt oder synthetisch)
2. Klassische Kalibrierung
3. SSZ-Analyse (delta_seg, Mode-Konsistenz)
4. Spektrale Kohärenz-Analyse
5. Layered Model Fit
6. Plots und Reports generieren

Usage:
    python run_complete_analysis.py --synthetic
    python run_complete_analysis.py --real --schumann-path data/schumann/
    python run_complete_analysis.py --use-sample

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# UTF-8 für Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.config import PHI, C_LIGHT, EARTH_RADIUS, ETA_0_DEFAULT
from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
from ssz_schumann.data_io.merge import merge_all
from ssz_schumann.analysis.compute_deltas import compute_all_deltas
from ssz_schumann.models.ssz_correction import check_mode_consistency
from ssz_schumann.models.fit_wrappers import fit_ssz_model, fit_classical_model
from ssz_schumann.models.maxwell_schumann import (
    f_n_ideal, f_n_damped, compute_eta_from_observed,
    get_schumann_mode, compute_mode_ratios, print_schumann_summary
)
from ssz_schumann.analysis.spectral_coherence import (
    analyze_ssz_coherence, compute_mode_coherence_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_data(args) -> pd.DataFrame:
    """Lade Daten basierend auf Argumenten."""
    
    if args.use_sample:
        logger.info("Lade Sample-Daten...")
        from ssz_schumann.data_io.schumann_real import create_sample_real_data
        schumann_df = create_sample_real_data(
            output_path=Path("data/schumann/sample_complete.csv"),
            n_days=90,
            start_date="2016-01-01",
        )
    elif args.real and args.schumann_path:
        logger.info(f"Lade echte Daten von {args.schumann_path}...")
        from ssz_schumann.data_io.schumann_real import load_schumann_real_data
        schumann_df = load_schumann_real_data(args.schumann_path)
    else:
        logger.info("Generiere synthetische Daten...")
        schumann_df = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-06-30",
            freq="1h",
            eta_0=0.74,
            delta_seg_amplitude=args.delta_seg_amp,
            noise_level=args.noise_level,
        )
    
    # Convert xarray to DataFrame if needed
    import xarray as xr
    if isinstance(schumann_df, xr.Dataset):
        logger.info("Converting xarray Dataset to DataFrame...")
        schumann_df = schumann_df.to_dataframe().reset_index()
        if 'time' in schumann_df.columns:
            schumann_df = schumann_df.set_index('time')
    
    # Space Weather
    logger.info("Lade Space Weather Daten...")
    
    # Get time range from schumann_df
    if hasattr(schumann_df, 'index') and isinstance(schumann_df.index, pd.DatetimeIndex):
        start_time = schumann_df.index.min()
        end_time = schumann_df.index.max()
    elif hasattr(schumann_df, 'columns') and 'time' in schumann_df.columns:
        start_time = pd.to_datetime(schumann_df['time']).min()
        end_time = pd.to_datetime(schumann_df['time']).max()
    else:
        start_time = pd.Timestamp("2016-01-01")
        end_time = pd.Timestamp("2016-06-30")
    
    f107, kp = create_synthetic_space_weather(
        start=start_time.strftime("%Y-%m-%d"),
        end=end_time.strftime("%Y-%m-%d"),
    )
    
    # Convert schumann_df to xarray-like format for merge_all
    logger.info("Merge Daten...")
    
    # Ensure schumann_df has proper datetime index
    if not isinstance(schumann_df.index, pd.DatetimeIndex):
        if 'time' in schumann_df.columns:
            schumann_df = schumann_df.set_index('time')
        elif 'datetime' in schumann_df.columns:
            schumann_df = schumann_df.set_index('datetime')
    
    # Remove timezone if present
    if schumann_df.index.tzinfo is not None:
        schumann_df.index = schumann_df.index.tz_localize(None)
    
    # Resample to hourly
    schumann_hourly = schumann_df.resample('1h').mean()
    
    # Merge manually (simpler approach)
    merged = schumann_hourly.copy()
    
    # Add F10.7
    if f107.index.tzinfo is not None:
        f107.index = f107.index.tz_localize(None)
    merged['F107'] = f107.reindex(merged.index, method='ffill')
    merged['f107_norm'] = (merged['F107'] - merged['F107'].mean()) / merged['F107'].std()
    
    # Add Kp
    if kp.index.tzinfo is not None:
        kp.index = kp.index.tz_localize(None)
    merged['Kp'] = kp.reindex(merged.index, method='ffill')
    merged['kp_norm'] = (merged['Kp'] - merged['Kp'].mean()) / merged['Kp'].std()
    
    # Normalize column names (f1 vs f1_Hz)
    if 'f1' in merged.columns and 'f1_Hz' not in merged.columns:
        merged['f1_Hz'] = merged['f1']
        merged['f2_Hz'] = merged['f2']
        merged['f3_Hz'] = merged['f3']
    
    # Drop NaN
    merged = merged.dropna(subset=['f1_Hz', 'f2_Hz', 'f3_Hz'])
    
    # Add time column for compatibility
    merged['time'] = merged.index
    
    # Compute deltas manually (compute_all_deltas is for xarray)
    logger.info("Berechne Deltas...")
    
    from ssz_schumann.models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    from ssz_schumann.models.ssz_correction import delta_seg_from_observed
    
    # Calibrate eta
    f1_mean = merged['f1_Hz'].mean()
    eta_0 = compute_eta0_from_mean_f1(f1_mean)
    logger.info(f"Calibrated eta_0 = {eta_0:.6f} from f1_mean = {f1_mean:.4f} Hz")
    
    # Classical frequencies
    for n in [1, 2, 3]:
        f_class = f_n_classical(n, eta_0)
        merged[f'f{n}_classical'] = f_class
        logger.info(f"f{n}_classical = {f_class:.4f} Hz")
    
    # Delta_seg for each mode
    for n in [1, 2, 3]:
        f_obs = merged[f'f{n}_Hz'].values
        f_class = merged[f'f{n}_classical'].values[0]  # scalar
        delta_seg = delta_seg_from_observed(f_obs, f_class)
        merged[f'delta_seg_{n}'] = delta_seg
        logger.info(f"delta_seg_{n}: mean = {np.mean(delta_seg):.6f}")
    
    # Mean delta_seg
    merged['delta_seg_mean'] = (merged['delta_seg_1'] + merged['delta_seg_2'] + merged['delta_seg_3']) / 3
    
    # Store eta_0
    merged.attrs = {'eta_0': eta_0}
    
    return merged


def run_classical_analysis(df: pd.DataFrame) -> Dict:
    """Klassische Schumann-Analyse."""
    logger.info("\n" + "="*60)
    logger.info("KLASSISCHE SCHUMANN-ANALYSE")
    logger.info("="*60)
    
    # Mittlere Frequenzen
    f1_mean = df['f1_Hz'].mean()
    f2_mean = df['f2_Hz'].mean()
    f3_mean = df['f3_Hz'].mean()
    
    # Eta kalibrieren
    eta_1 = compute_eta_from_observed(f1_mean, 1)
    eta_2 = compute_eta_from_observed(f2_mean, 2)
    eta_3 = compute_eta_from_observed(f3_mean, 3)
    
    logger.info(f"Beobachtete Frequenzen:")
    logger.info(f"  f1 = {f1_mean:.4f} Hz (eta = {eta_1:.4f})")
    logger.info(f"  f2 = {f2_mean:.4f} Hz (eta = {eta_2:.4f})")
    logger.info(f"  f3 = {f3_mean:.4f} Hz (eta = {eta_3:.4f})")
    
    # Ideale Frequenzen
    logger.info(f"\nIdeale Frequenzen (ohne Daempfung):")
    for n in [1, 2, 3]:
        f_id = f_n_ideal(n)
        logger.info(f"  f{n}_ideal = {f_id:.4f} Hz")
    
    # Mode-Verhältnisse
    ratios = compute_mode_ratios()
    logger.info(f"\nMode-Verhaeltnisse:")
    logger.info(f"  f2/f1: Theorie = {ratios['f2/f1_theory']:.4f}, "
                f"Beobachtet = {f2_mean/f1_mean:.4f}")
    logger.info(f"  f3/f1: Theorie = {ratios['f3/f1_theory']:.4f}, "
                f"Beobachtet = {f3_mean/f1_mean:.4f}")
    
    return {
        "f1_mean": f1_mean,
        "f2_mean": f2_mean,
        "f3_mean": f3_mean,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "eta_3": eta_3,
        "eta_mean": np.mean([eta_1, eta_2, eta_3]),
    }


def run_ssz_analysis(df: pd.DataFrame) -> Dict:
    """SSZ-Analyse mit Mode-Konsistenz."""
    logger.info("\n" + "="*60)
    logger.info("SSZ-ANALYSE")
    logger.info("="*60)
    
    # Delta_seg pro Mode
    delta_seg_dict = {}
    for n in [1, 2, 3]:
        col = f"delta_seg_{n}"
        if col in df.columns:
            delta_seg_dict[n] = df[col].values
            mean_ds = df[col].mean()
            std_ds = df[col].std()
            logger.info(f"delta_seg_{n}: mean = {mean_ds:.6f} ({mean_ds*100:.4f}%), "
                       f"std = {std_ds:.6f}")
    
    # Mode-Konsistenz
    logger.info("\nMode-Konsistenz-Analyse:")
    consistency = check_mode_consistency(delta_seg_dict)
    
    logger.info(f"  Mean Correlation: {consistency['mean_correlation']:.4f}")
    logger.info(f"  SSZ Score: {consistency['ssz_score']:.4f}")
    logger.info(f"  Std across modes: {consistency['std_across_modes']:.6f}")
    logger.info(f"  Interpretation: {consistency['interpretation']}")
    
    return {
        "delta_seg_dict": delta_seg_dict,
        "consistency": consistency,
    }


def run_spectral_analysis(df: pd.DataFrame, delta_seg_dict: Dict) -> Dict:
    """Spektrale Kohärenz-Analyse."""
    logger.info("\n" + "="*60)
    logger.info("SPEKTRALE KOHAERENZ-ANALYSE")
    logger.info("="*60)
    
    # Proxies vorbereiten
    proxies = None
    if 'f107_norm' in df.columns and 'kp_norm' in df.columns:
        proxies = df[['f107_norm', 'kp_norm']].copy()
    
    # Kohärenz-Analyse
    coherence_results = analyze_ssz_coherence(
        delta_seg_dict,
        proxies=proxies,
        fs=1.0,  # 1 sample/hour
    )
    
    logger.info(f"\nErgebnisse:")
    logger.info(f"  Mean Mode Coherence: {coherence_results['summary']['mean_mode_coherence']:.4f}")
    logger.info(f"  Phase Locking Value: {coherence_results['summary']['phase_locking_value']:.4f}")
    logger.info(f"  SSZ Interpretation: {coherence_results['ssz_interpretation']}")
    
    # Granger-Kausalität
    if 'granger_causality' in coherence_results:
        logger.info(f"\nGranger-Kausalitaet:")
        for key, val in coherence_results['granger_causality'].items():
            if 'p_value' in val:
                sig = "*" if val['p_value'] < 0.05 else ""
                logger.info(f"  {key}: p = {val['p_value']:.4f} {sig}")
    
    return coherence_results


def run_model_fit(df: pd.DataFrame) -> Dict:
    """Modell-Fit mit Proxies."""
    logger.info("\n" + "="*60)
    logger.info("MODELL-FIT")
    logger.info("="*60)
    
    # Ensure f1, f2, f3 columns exist
    if 'f1' not in df.columns and 'f1_Hz' in df.columns:
        df = df.copy()
        df['f1'] = df['f1_Hz']
        df['f2'] = df['f2_Hz']
        df['f3'] = df['f3_Hz']
    
    # Features
    features = pd.DataFrame({
        'f107_norm': df['f107_norm'].values,
        'kp_norm': df['kp_norm'].values,
    }, index=df.index if hasattr(df, 'index') else None)
    
    # Classical fit
    try:
        classical_result = fit_classical_model(df)
        logger.info(f"\nKlassisches Modell:")
        logger.info(f"  eta_0 = {classical_result.eta_0:.6f}")
        logger.info(f"  RMSE = {classical_result.rmse:.4f} Hz")
    except Exception as e:
        logger.warning(f"Classical fit failed: {e}")
        classical_result = None
    
    # SSZ fit
    try:
        ssz_result = fit_ssz_model(df, features)
        logger.info(f"\nSSZ-Modell:")
        logger.info(f"  eta_0 = {ssz_result.eta_0:.6f}")
        logger.info(f"  beta_0 = {ssz_result.beta_0:.6f}")
        logger.info(f"  beta_1 (F10.7) = {ssz_result.beta_1:.6f}")
        logger.info(f"  R^2 = {ssz_result.r_squared:.4f}")
        logger.info(f"  RMSE = {ssz_result.rmse:.4f} Hz")
        logger.info(f"  Mode Consistency Score = {ssz_result.consistency_score:.4f}")
    except Exception as e:
        logger.warning(f"SSZ fit failed: {e}")
        ssz_result = None
    
    return {
        "classical": classical_result,
        "ssz": ssz_result,
    }


def create_summary_plots(
    df: pd.DataFrame,
    classical_result: Dict,
    ssz_result: Dict,
    coherence_result: Dict,
    output_dir: Path,
):
    """Erstelle Zusammenfassungs-Plots."""
    logger.info("\n" + "="*60)
    logger.info("ERSTELLE PLOTS")
    logger.info("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Frequenz-Zeitreihen
    ax = axes[0, 0]
    if 'time' in df.columns:
        x = pd.to_datetime(df['time'])
    else:
        x = np.arange(len(df))
    
    ax.plot(x[::24], df['f1_Hz'].values[::24], 'b-', alpha=0.7, label='f1')
    ax.plot(x[::24], df['f2_Hz'].values[::24], 'g-', alpha=0.7, label='f2')
    ax.plot(x[::24], df['f3_Hz'].values[::24], 'r-', alpha=0.7, label='f3')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Schumann Frequencies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Delta_seg pro Mode
    ax = axes[0, 1]
    for n in [1, 2, 3]:
        col = f"delta_seg_{n}"
        if col in df.columns:
            ax.plot(x[::24], df[col].values[::24] * 100, alpha=0.7, label=f'Mode {n}')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel('delta_seg (%)')
    ax.set_title('SSZ Segmentation Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Mode-Korrelation Scatter
    ax = axes[0, 2]
    if 'delta_seg_1' in df.columns and 'delta_seg_2' in df.columns:
        ax.scatter(df['delta_seg_1']*100, df['delta_seg_2']*100, 
                  alpha=0.3, s=5, c='blue')
        ax.plot([-5, 5], [-5, 5], 'r--', label='1:1 line')
        ax.set_xlabel('delta_seg_1 (%)')
        ax.set_ylabel('delta_seg_2 (%)')
        ax.set_title(f"Mode Correlation (r={ssz_result['consistency']['mean_correlation']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. F10.7 Korrelation
    ax = axes[1, 0]
    if 'delta_seg_mean' in df.columns and 'f107_norm' in df.columns:
        ax.scatter(df['f107_norm'], df['delta_seg_mean']*100, 
                  alpha=0.3, s=5, c='orange')
        # Fit-Linie
        z = np.polyfit(df['f107_norm'].dropna(), 
                      df['delta_seg_mean'].dropna()*100, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(df['f107_norm'].min(), df['f107_norm'].max(), 100)
        ax.plot(x_fit, p(x_fit), 'r-', linewidth=2)
        ax.set_xlabel('F10.7 (normalized)')
        ax.set_ylabel('delta_seg_mean (%)')
        ax.set_title('SSZ vs Solar Flux')
        ax.grid(True, alpha=0.3)
    
    # 5. SSZ Score Gauge
    ax = axes[1, 1]
    ssz_score = ssz_result['consistency']['ssz_score']
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Einfacher Balken
    ax.barh([0], [ssz_score], color='steelblue', height=0.5)
    ax.axvline(0.7, color='green', linestyle='--', label='Threshold (0.7)')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('SSZ Score')
    ax.set_title(f'SSZ Score: {ssz_score:.3f}')
    ax.legend()
    
    # 6. Zusammenfassung Text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
ANALYSIS SUMMARY
================

Data Points: {len(df)}
Time Range: {df.index.min()} to {df.index.max()}

Classical Model:
  eta_0 = {classical_result['eta_mean']:.4f}
  f1_mean = {classical_result['f1_mean']:.3f} Hz

SSZ Analysis:
  Mean Correlation = {ssz_result['consistency']['mean_correlation']:.4f}
  SSZ Score = {ssz_result['consistency']['ssz_score']:.4f}
  Interpretation: {ssz_result['consistency']['interpretation']}

Spectral Coherence:
  Mean Mode Coherence = {coherence_result['summary']['mean_mode_coherence']:.4f}
  Phase Locking Value = {coherence_result['summary']['phase_locking_value']:.4f}

Conclusion:
  {coherence_result['ssz_interpretation']}
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Speichern
    plot_path = output_dir / "complete_analysis_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {plot_path}")
    
    return plot_path


def save_results(
    results: Dict,
    output_dir: Path,
):
    """Speichere Ergebnisse als JSON und Markdown."""
    
    # JSON (nur serialisierbare Daten)
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "classical": {
            "f1_mean": results["classical"]["f1_mean"],
            "f2_mean": results["classical"]["f2_mean"],
            "f3_mean": results["classical"]["f3_mean"],
            "eta_mean": results["classical"]["eta_mean"],
        },
        "ssz": {
            "mean_correlation": results["ssz"]["consistency"]["mean_correlation"],
            "ssz_score": results["ssz"]["consistency"]["ssz_score"],
            "std_across_modes": results["ssz"]["consistency"]["std_across_modes"],
            "interpretation": results["ssz"]["consistency"]["interpretation"],
        },
        "coherence": {
            "mean_mode_coherence": results["coherence"]["summary"]["mean_mode_coherence"],
            "phase_locking_value": results["coherence"]["summary"]["phase_locking_value"],
            "ssz_interpretation": results["coherence"]["ssz_interpretation"],
        },
    }
    
    json_path = output_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved: {json_path}")
    
    # Markdown Report
    md_content = f"""# SSZ Schumann Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Classical Analysis

| Parameter | Value |
|-----------|-------|
| f1_mean | {results["classical"]["f1_mean"]:.4f} Hz |
| f2_mean | {results["classical"]["f2_mean"]:.4f} Hz |
| f3_mean | {results["classical"]["f3_mean"]:.4f} Hz |
| eta_mean | {results["classical"]["eta_mean"]:.4f} |

## SSZ Analysis

| Metric | Value |
|--------|-------|
| Mean Correlation | {results["ssz"]["consistency"]["mean_correlation"]:.4f} |
| SSZ Score | {results["ssz"]["consistency"]["ssz_score"]:.4f} |
| Std across modes | {results["ssz"]["consistency"]["std_across_modes"]:.6f} |
| **Interpretation** | {results["ssz"]["consistency"]["interpretation"]} |

## Spectral Coherence

| Metric | Value |
|--------|-------|
| Mean Mode Coherence | {results["coherence"]["summary"]["mean_mode_coherence"]:.4f} |
| Phase Locking Value | {results["coherence"]["summary"]["phase_locking_value"]:.4f} |
| **Interpretation** | {results["coherence"]["ssz_interpretation"]} |

## Conclusion

Based on the analysis:
- SSZ Score: {results["ssz"]["consistency"]["ssz_score"]:.4f} (threshold: 0.7)
- Mode Correlation: {results["ssz"]["consistency"]["mean_correlation"]:.4f} (threshold: 0.7)

**{results["coherence"]["ssz_interpretation"]}**

---
(c) 2025 Carmen Wrede & Lino Casu
"""
    
    md_path = output_dir / "report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    logger.info(f"Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SSZ Schumann Complete Analysis"
    )
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data")
    parser.add_argument("--real", action="store_true",
                       help="Use real data")
    parser.add_argument("--use-sample", action="store_true",
                       help="Use sample real-like data")
    parser.add_argument("--schumann-path", type=str,
                       help="Path to Schumann data")
    parser.add_argument("--delta-seg-amp", type=float, default=0.02,
                       help="Delta_seg amplitude for synthetic data")
    parser.add_argument("--noise-level", type=float, default=0.01,
                       help="Noise level for synthetic data")
    parser.add_argument("--output-dir", type=str,
                       default="output/complete_analysis",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Default to synthetic if nothing specified
    if not args.real and not args.use_sample:
        args.synthetic = True
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("SSZ SCHUMANN COMPLETE ANALYSIS")
    logger.info("="*70)
    logger.info(f"Output: {output_dir}")
    
    # 1. Daten laden
    logger.info("\n[1/5] Lade Daten...")
    df = load_data(args)
    logger.info(f"Geladen: {len(df)} Datenpunkte")
    
    # 2. Klassische Analyse
    logger.info("\n[2/5] Klassische Analyse...")
    classical_result = run_classical_analysis(df)
    
    # 3. SSZ-Analyse
    logger.info("\n[3/5] SSZ-Analyse...")
    ssz_result = run_ssz_analysis(df)
    
    # 4. Spektrale Kohärenz
    logger.info("\n[4/5] Spektrale Kohaerenz...")
    coherence_result = run_spectral_analysis(df, ssz_result["delta_seg_dict"])
    
    # 5. Modell-Fit
    logger.info("\n[5/5] Modell-Fit...")
    model_result = run_model_fit(df)
    
    # Ergebnisse zusammenfassen
    results = {
        "classical": classical_result,
        "ssz": ssz_result,
        "coherence": coherence_result,
        "model": model_result,
    }
    
    # Plots erstellen
    create_summary_plots(df, classical_result, ssz_result, coherence_result, output_dir)
    
    # Ergebnisse speichern
    save_results(results, output_dir)
    
    # Finale Zusammenfassung
    logger.info("\n" + "="*70)
    logger.info("ANALYSE ABGESCHLOSSEN")
    logger.info("="*70)
    logger.info(f"Output: {output_dir}")
    logger.info(f"\nKey Results:")
    logger.info(f"  SSZ Score: {ssz_result['consistency']['ssz_score']:.4f}")
    logger.info(f"  Mode Correlation: {ssz_result['consistency']['mean_correlation']:.4f}")
    logger.info(f"  Spectral Coherence: {coherence_result['summary']['mean_mode_coherence']:.4f}")
    logger.info(f"  Interpretation: {coherence_result['ssz_interpretation']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
