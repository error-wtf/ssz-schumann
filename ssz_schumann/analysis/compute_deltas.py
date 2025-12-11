#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta Computation Pipeline

Main analysis pipeline for computing SSZ correction factors
from Schumann resonance data.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from ..models.classical_schumann import (
    f_n_classical,
    compute_eta0_from_mean_f1,
)
from ..models.ssz_correction import (
    delta_seg_from_observed,
    compute_delta_seg_all_modes,
    check_mode_consistency,
    fit_delta_seg_simple,
)
from ..models.fit_wrappers import (
    fit_classical_model,
    fit_ssz_model,
    compare_models,
)

logger = logging.getLogger(__name__)


def compute_all_deltas(
    ds: xr.Dataset,
    modes: List[int] = [1, 2, 3],
    eta_0: Optional[float] = None,
) -> xr.Dataset:
    """
    Compute all delta_seg values from Schumann data.
    
    Pipeline:
    1. Calibrate eta_0 from mean f1 (if not provided)
    2. Compute classical frequencies for each mode
    3. Compute delta_seg for each mode
    4. Compute mean delta_seg across modes
    
    Args:
        ds: Input dataset with f1, f2, f3
        modes: Mode numbers to process
        eta_0: Baseline eta (computed from data if None)
    
    Returns:
        Dataset with additional variables:
            - f1_classical, f2_classical, f3_classical
            - delta_seg_1, delta_seg_2, delta_seg_3
            - delta_seg_mean
    """
    ds = ds.copy()
    
    # Step 1: Calibrate eta_0
    if eta_0 is None:
        if "f1" not in ds:
            raise ValueError("f1 required for eta_0 calibration")
        eta_0 = compute_eta0_from_mean_f1(ds["f1"].values)
    
    ds.attrs["eta_0"] = eta_0
    logger.info(f"Using eta_0 = {eta_0:.6f}")
    
    # Step 2: Compute classical frequencies
    for n in modes:
        f_class = f_n_classical(n, eta_0)
        ds[f"f{n}_classical"] = f_class
        ds[f"f{n}_classical"].attrs = {
            "units": "Hz",
            "long_name": f"Classical Schumann frequency mode {n}",
            "eta_0": eta_0,
        }
        logger.info(f"f{n}_classical = {f_class:.4f} Hz")
    
    # Step 3: Compute delta_seg for each mode
    delta_arrays = []
    for n in modes:
        if f"f{n}" in ds:
            f_obs = ds[f"f{n}"].values
            f_class = ds[f"f{n}_classical"].values
            
            delta_seg = delta_seg_from_observed(f_obs, f_class)
            ds[f"delta_seg_{n}"] = (["time"], delta_seg)
            ds[f"delta_seg_{n}"].attrs = {
                "units": "dimensionless",
                "long_name": f"SSZ correction factor from mode {n}",
            }
            
            delta_arrays.append(delta_seg)
            
            # Statistics
            valid = ~np.isnan(delta_seg)
            if np.any(valid):
                logger.info(f"delta_seg_{n}: mean={np.nanmean(delta_seg):.6f}, "
                           f"std={np.nanstd(delta_seg):.6f}")
    
    # Step 4: Mean delta_seg across modes
    if delta_arrays:
        delta_matrix = np.vstack(delta_arrays)
        delta_seg_mean = np.nanmean(delta_matrix, axis=0)
        delta_seg_std = np.nanstd(delta_matrix, axis=0)
        
        ds["delta_seg_mean"] = (["time"], delta_seg_mean)
        ds["delta_seg_mean"].attrs = {
            "units": "dimensionless",
            "long_name": "Mean SSZ correction factor across modes",
        }
        
        ds["delta_seg_std"] = (["time"], delta_seg_std)
        ds["delta_seg_std"].attrs = {
            "units": "dimensionless",
            "long_name": "Std of SSZ correction factor across modes",
        }
        
        logger.info(f"delta_seg_mean: mean={np.nanmean(delta_seg_mean):.6f}, "
                   f"std={np.nanstd(delta_seg_mean):.6f}")
    
    return ds


def run_analysis_pipeline(
    ds: xr.Dataset,
    features: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    modes: List[int] = [1, 2, 3],
    feature_columns: Optional[List[str]] = None,
) -> Dict:
    """
    Run complete SSZ Schumann analysis pipeline.
    
    Steps:
    1. Compute all delta_seg values
    2. Check mode consistency (SSZ signature)
    3. Fit classical model
    4. Fit SSZ model (if features provided)
    5. Compare models
    6. Generate summary
    
    Args:
        ds: Input dataset
        features: Feature DataFrame for SSZ fitting
        output_dir: Directory for output files
        modes: Mode numbers to analyze
        feature_columns: Which features to use
    
    Returns:
        Dictionary with all results
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_timepoints": len(ds.time),
        "modes": modes,
    }
    
    # Step 1: Compute deltas
    logger.info("=" * 60)
    logger.info("STEP 1: Computing delta_seg values")
    logger.info("=" * 60)
    
    ds = compute_all_deltas(ds, modes=modes)
    results["eta_0"] = ds.attrs.get("eta_0")
    
    # Step 2: Mode consistency
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Checking mode consistency (SSZ signature)")
    logger.info("=" * 60)
    
    delta_seg_dict = {}
    for n in modes:
        if f"delta_seg_{n}" in ds:
            delta_seg_dict[n] = pd.Series(
                ds[f"delta_seg_{n}"].values,
                index=pd.DatetimeIndex(ds.time.values)
            )
    
    consistency = check_mode_consistency(delta_seg_dict)
    results["mode_consistency"] = consistency
    
    # Step 3: Fit classical model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Fitting classical model")
    logger.info("=" * 60)
    
    classical_result = fit_classical_model(ds, modes=modes)
    results["classical_model"] = {
        "eta_0": classical_result.eta_0,
        "r_squared": classical_result.r_squared,
        "rmse": classical_result.rmse,
        "aic": classical_result.aic,
        "bic": classical_result.bic,
    }
    
    # Step 4: Fit SSZ model (if features available)
    if features is not None and len(features) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Fitting SSZ model")
        logger.info("=" * 60)
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [c for c in features.columns if "_norm" in c]
            if not feature_columns:
                feature_columns = list(features.columns)[:2]
        
        ssz_result = fit_ssz_model(
            ds, features,
            modes=modes,
            feature_columns=feature_columns
        )
        
        results["ssz_model"] = {
            "eta_0": ssz_result.eta_0,
            "ssz_params": ssz_result.ssz_params,
            "r_squared": ssz_result.r_squared,
            "rmse": ssz_result.rmse,
            "aic": ssz_result.aic,
            "bic": ssz_result.bic,
            "mode_consistency_score": consistency.get("ssz_score", 0),
        }
        
        # Step 5: Compare models
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Comparing models")
        logger.info("=" * 60)
        
        comparison = compare_models(classical_result, ssz_result)
        results["model_comparison"] = comparison
        
        # Store predictions
        results["predictions"] = {
            "delta_seg_predicted": ssz_result.delta_seg_predicted,
            "f_predicted_ssz": ssz_result.f_predicted,
            "f_predicted_classical": classical_result.f_predicted,
        }
    else:
        logger.info("\nNo features provided - skipping SSZ model fitting")
        results["ssz_model"] = None
        results["model_comparison"] = None
    
    # Step 6: Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    summary = generate_summary(results)
    results["summary"] = summary
    
    logger.info(summary)
    
    # Save results if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary (excluding non-serializable objects)
        summary_dict = {k: v for k, v in results.items()
                       if k not in ["predictions"]}
        
        # Convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        summary_dict = convert_numpy(summary_dict)
        
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {output_dir}")
    
    return results


def generate_summary(results: Dict) -> str:
    """Generate human-readable summary of analysis results."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("SSZ SCHUMANN ANALYSIS SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\nData: {results['n_timepoints']} time points")
    lines.append(f"Modes analyzed: {results['modes']}")
    lines.append(f"Calibrated eta_0: {results['eta_0']:.6f}")
    
    # Mode consistency
    mc = results.get("mode_consistency", {})
    lines.append(f"\nMode Consistency (SSZ Signature):")
    lines.append(f"  Mean delta_seg: {mc.get('mean_delta_seg', 'N/A'):.6f}" 
                if isinstance(mc.get('mean_delta_seg'), float) else "  Mean delta_seg: N/A")
    lines.append(f"  Std across modes: {mc.get('std_delta_seg', 'N/A'):.6f}"
                if isinstance(mc.get('std_delta_seg'), float) else "  Std across modes: N/A")
    lines.append(f"  SSZ score: {mc.get('ssz_score', 'N/A'):.4f}"
                if isinstance(mc.get('ssz_score'), float) else "  SSZ score: N/A")
    lines.append(f"  Consistent: {mc.get('is_consistent', 'N/A')}")
    
    # Classical model
    cm = results.get("classical_model", {})
    lines.append(f"\nClassical Model:")
    lines.append(f"  R²: {cm.get('r_squared', 'N/A'):.4f}"
                if isinstance(cm.get('r_squared'), float) else "  R²: N/A")
    lines.append(f"  RMSE: {cm.get('rmse', 'N/A'):.4f} Hz"
                if isinstance(cm.get('rmse'), float) else "  RMSE: N/A")
    lines.append(f"  AIC: {cm.get('aic', 'N/A'):.1f}"
                if isinstance(cm.get('aic'), float) else "  AIC: N/A")
    
    # SSZ model
    sm = results.get("ssz_model")
    if sm is not None:
        lines.append(f"\nSSZ Model:")
        lines.append(f"  R²: {sm.get('r_squared', 'N/A'):.4f}"
                    if isinstance(sm.get('r_squared'), float) else "  R²: N/A")
        lines.append(f"  RMSE: {sm.get('rmse', 'N/A'):.4f} Hz"
                    if isinstance(sm.get('rmse'), float) else "  RMSE: N/A")
        lines.append(f"  AIC: {sm.get('aic', 'N/A'):.1f}"
                    if isinstance(sm.get('aic'), float) else "  AIC: N/A")
        
        params = sm.get("ssz_params", {})
        lines.append(f"  beta_0: {params.get('beta_0', 'N/A'):.6f}"
                    if isinstance(params.get('beta_0'), float) else "  beta_0: N/A")
        lines.append(f"  beta_1: {params.get('beta_1', 'N/A'):.6f}"
                    if isinstance(params.get('beta_1'), float) else "  beta_1: N/A")
    
    # Model comparison
    comp = results.get("model_comparison")
    if comp is not None:
        lines.append(f"\nModel Comparison:")
        lines.append(f"  Preferred: {comp.get('preferred_model', 'N/A')}")
        lines.append(f"  Delta R²: {comp.get('delta_r_squared', 'N/A'):+.4f}"
                    if isinstance(comp.get('delta_r_squared'), float) else "  Delta R²: N/A")
        lines.append(f"  Delta AIC: {comp.get('delta_aic', 'N/A'):+.1f}"
                    if isinstance(comp.get('delta_aic'), float) else "  Delta AIC: N/A")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
