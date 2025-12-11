#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Analysis Script for SSZ Schumann Experiment

Runs the complete SSZ Schumann resonance analysis pipeline.

Usage:
    python scripts/run_schumann_ssz_analysis.py --synthetic
    python scripts/run_schumann_ssz_analysis.py --data-dir data/schumann/2016

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# UTF-8 for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'


def run_synthetic_analysis(output_dir: Path) -> dict:
    """
    Run analysis on synthetic data.
    
    This demonstrates the pipeline and validates that
    the SSZ signature can be recovered from synthetic data.
    """
    import numpy as np
    import pandas as pd
    
    from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
    from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
    from ssz_schumann.data_io.merge import merge_all, compute_derived_variables
    from ssz_schumann.analysis.compute_deltas import run_analysis_pipeline
    from ssz_schumann.analysis.correlation_plots import (
        plot_timeseries,
        plot_scatter_delta_vs_feature,
        plot_mode_consistency,
        create_summary_figure,
    )
    
    logger.info("=" * 70)
    logger.info("SSZ SCHUMANN RESONANCE ANALYSIS - SYNTHETIC DATA")
    logger.info("=" * 70)
    logger.info("")
    
    # Parameters
    eta_0 = 0.74
    delta_seg_amplitude = 0.02
    noise_level = 0.01
    
    logger.info("Synthetic data parameters:")
    logger.info(f"  eta_0 = {eta_0}")
    logger.info(f"  delta_seg_amplitude = {delta_seg_amplitude}")
    logger.info(f"  noise_level = {noise_level}")
    logger.info("")
    
    # Create synthetic data
    logger.info("Creating synthetic Schumann data...")
    ds = create_synthetic_schumann_data(
        start="2016-01-01",
        end="2016-06-30",
        freq="1h",
        eta_0=eta_0,
        delta_seg_amplitude=delta_seg_amplitude,
        noise_level=noise_level,
    )
    logger.info(f"  Created {len(ds.time)} time points")
    
    logger.info("Creating synthetic space weather data...")
    f107, kp = create_synthetic_space_weather(
        start="2016-01-01",
        end="2016-06-30",
    )
    logger.info(f"  F10.7: {len(f107)} points, mean={f107.mean():.1f}")
    logger.info(f"  Kp: {len(kp)} points, mean={kp.mean():.2f}")
    
    # Merge data
    logger.info("\nMerging data...")
    merged = merge_all(ds, f107, kp, time_resolution="1H")
    merged = compute_derived_variables(merged)
    logger.info(f"  Merged dataset: {len(merged.time)} time points")
    
    # Create features DataFrame
    features = pd.DataFrame({
        "f107_norm": merged["f107_norm"].values,
        "kp_norm": merged["kp_norm"].values,
    }, index=pd.DatetimeIndex(merged.time.values))
    
    # Run analysis pipeline
    logger.info("\nRunning analysis pipeline...")
    results = run_analysis_pipeline(
        merged,
        features,
        output_dir=output_dir,
        modes=[1, 2, 3],
    )
    
    # Create plots
    logger.info("\nCreating plots...")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Frequency time series
    plot_timeseries(
        merged,
        variables=["f1", "f2", "f3"],
        title="Schumann Resonance Frequencies (Synthetic)",
        save_path=plots_dir / "timeseries_frequencies.png"
    )
    
    # Delta_seg time series
    from ssz_schumann.analysis.compute_deltas import compute_all_deltas
    merged_with_deltas = compute_all_deltas(merged)
    
    plot_timeseries(
        merged_with_deltas,
        variables=["delta_seg_1", "delta_seg_2", "delta_seg_3"],
        title="SSZ Correction Factor by Mode",
        save_path=plots_dir / "timeseries_delta_seg.png"
    )
    
    # Scatter plot
    delta_seg_mean = pd.Series(
        merged_with_deltas["delta_seg_mean"].values,
        index=pd.DatetimeIndex(merged_with_deltas.time.values)
    )
    f107_series = pd.Series(
        merged["f107"].values,
        index=pd.DatetimeIndex(merged.time.values)
    )
    
    plot_scatter_delta_vs_feature(
        delta_seg_mean,
        f107_series,
        feature_name="F10.7 (sfu)",
        save_path=plots_dir / "scatter_delta_vs_f107.png"
    )
    
    # Mode consistency
    delta_seg_dict = {
        n: pd.Series(
            merged_with_deltas[f"delta_seg_{n}"].values,
            index=pd.DatetimeIndex(merged_with_deltas.time.values)
        )
        for n in [1, 2, 3]
    }
    
    plot_mode_consistency(
        delta_seg_dict,
        save_path=plots_dir / "mode_consistency.png"
    )
    
    # Summary figure
    create_summary_figure(
        merged_with_deltas,
        results,
        save_path=plots_dir / "summary.png"
    )
    
    logger.info(f"\nPlots saved to: {plots_dir}")
    
    return results


def run_real_data_analysis(
    schumann_path: Path,
    f107_path: Path = None,
    kp_path: Path = None,
    output_dir: Path = None,
) -> dict:
    """
    Run analysis on real data.
    """
    import pandas as pd
    import xarray as xr
    
    from ssz_schumann.data_io.schumann_sierra_nevada import load_schumann_sierra_nevada
    from ssz_schumann.data_io.space_weather_noaa import load_f107, load_kp
    from ssz_schumann.data_io.merge import merge_all, compute_derived_variables
    from ssz_schumann.analysis.compute_deltas import run_analysis_pipeline
    
    logger.info("=" * 70)
    logger.info("SSZ SCHUMANN RESONANCE ANALYSIS - REAL DATA")
    logger.info("=" * 70)
    logger.info("")
    
    # Load Schumann data
    logger.info(f"Loading Schumann data from: {schumann_path}")
    ds = load_schumann_sierra_nevada(schumann_path)
    logger.info(f"  Loaded {len(ds.time)} time points")
    
    # Load space weather data
    f107 = None
    kp = None
    
    if f107_path and f107_path.exists():
        logger.info(f"Loading F10.7 data from: {f107_path}")
        f107 = load_f107(f107_path)
        logger.info(f"  Loaded {len(f107)} points")
    
    if kp_path and kp_path.exists():
        logger.info(f"Loading Kp data from: {kp_path}")
        kp = load_kp(kp_path)
        logger.info(f"  Loaded {len(kp)} points")
    
    # Merge data
    logger.info("\nMerging data...")
    merged = merge_all(ds, f107, kp, time_resolution="1H")
    merged = compute_derived_variables(merged)
    
    # Create features DataFrame
    feature_cols = []
    if "f107_norm" in merged:
        feature_cols.append("f107_norm")
    if "kp_norm" in merged:
        feature_cols.append("kp_norm")
    
    if feature_cols:
        features = pd.DataFrame({
            col: merged[col].values for col in feature_cols
        }, index=pd.DatetimeIndex(merged.time.values))
    else:
        features = None
    
    # Run analysis
    results = run_analysis_pipeline(
        merged,
        features,
        output_dir=output_dir,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SSZ Schumann Resonance Analysis"
    )
    
    parser.add_argument("--synthetic", action="store_true",
                       help="Run analysis on synthetic data")
    parser.add_argument("--schumann-path", type=Path, default=None,
                       help="Path to Schumann data file")
    parser.add_argument("--f107-path", type=Path, default=None,
                       help="Path to F10.7 data file")
    parser.add_argument("--kp-path", type=Path, default=None,
                       help="Path to Kp data file")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("output") / f"run_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    log_file = args.output_dir / "logs" / "analysis.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Run analysis
    if args.synthetic or args.schumann_path is None:
        results = run_synthetic_analysis(args.output_dir)
    else:
        results = run_real_data_analysis(
            args.schumann_path,
            args.f107_path,
            args.kp_path,
            args.output_dir,
        )
    
    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("")
    
    # Print key findings
    if results.get("model_comparison"):
        comp = results["model_comparison"]
        logger.info("Key Findings:")
        logger.info(f"  Preferred model: {comp.get('preferred_model', 'N/A')}")
        logger.info(f"  SSZ improvement (Delta R^2): {comp.get('delta_r_squared', 0):+.4f}")
        
        mc = results.get("mode_consistency", {})
        logger.info(f"  Mode consistency score: {mc.get('ssz_score', 0):.4f}")
        
        if mc.get("is_consistent"):
            logger.info("")
            logger.info("  >>> SSZ SIGNATURE DETECTED <<<")
            logger.info("  delta_seg is consistent across modes!")
    
    return results


if __name__ == "__main__":
    main()
