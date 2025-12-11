#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2: Unified Analysis Pipeline for SSZ Schumann Experiment

Provides a single entry point for running the complete analysis:
1. Load data (synthetic or real)
2. Fit classical model (eta_0)
3. Estimate delta_seg (SSZ correction)
4. Compute diagnostics and statistics
5. Generate figures and reports

This is the function used by the CLI and tests.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the analysis pipeline.
    
    All parameters needed to run the full analysis.
    """
    # Data source
    data_source: str = "synthetic"
    data_path: Optional[str] = None
    
    # Synthetic data parameters
    start_date: str = "2016-01-01"
    end_date: str = "2016-12-31"
    time_freq: str = "1h"
    delta_seg_amplitude: float = 0.02
    noise_level: float = 0.01
    seed: int = 42
    
    # Classical model parameters
    eta_mode: str = "quiet_interval"  # "full_fit", "quiet_interval", "fixed", "joint_fit"
    eta_fixed: float = 0.74
    quiet_days: int = 14
    
    # SSZ analysis parameters
    ssz_basis: str = "sinusoidal"  # "constant", "sinusoidal", "f107_linear"
    
    # Output options
    output_dir: str = "output"
    save_figures: bool = True
    save_csv: bool = True
    verbose: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineResult:
    """
    Results from the analysis pipeline.
    
    Contains all outputs from the analysis.
    """
    # Configuration used
    config: PipelineConfig
    
    # Data
    data: pd.DataFrame = None
    
    # Classical model results
    eta_0: float = 0.0
    f_classical: Dict[int, float] = field(default_factory=dict)
    
    # SSZ results
    delta_seg: pd.Series = None
    delta_seg_mean: float = 0.0
    delta_seg_std: float = 0.0
    
    # Mode consistency (SSZ signature)
    mode_correlation: float = 0.0
    ssz_score: float = 0.0
    is_ssz_significant: bool = False
    p_value: float = 1.0
    
    # Residuals
    residuals: Dict[int, pd.Series] = field(default_factory=dict)
    
    # Fit quality
    rmse: float = 0.0
    r_squared: float = 0.0
    
    # Timing
    runtime_seconds: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "config": self.config.to_dict(),
            "eta_0": self.eta_0,
            "f_classical": self.f_classical,
            "delta_seg_mean": self.delta_seg_mean,
            "delta_seg_std": self.delta_seg_std,
            "mode_correlation": self.mode_correlation,
            "ssz_score": self.ssz_score,
            "is_ssz_significant": self.is_ssz_significant,
            "p_value": self.p_value,
            "rmse": self.rmse,
            "r_squared": self.r_squared,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "SSZ SCHUMANN ANALYSIS RESULTS",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Runtime: {self.runtime_seconds:.2f} s",
            "",
            "CLASSICAL MODEL",
            "-" * 40,
            f"eta_0: {self.eta_0:.6f}",
            f"f1_classical: {self.f_classical.get(1, 0):.4f} Hz",
            f"f2_classical: {self.f_classical.get(2, 0):.4f} Hz",
            f"f3_classical: {self.f_classical.get(3, 0):.4f} Hz",
            "",
            "SSZ ANALYSIS",
            "-" * 40,
            f"delta_seg mean: {self.delta_seg_mean:.6f} ({self.delta_seg_mean*100:.3f}%)",
            f"delta_seg std: {self.delta_seg_std:.6f}",
            f"Mode correlation: {self.mode_correlation:.4f}",
            f"SSZ score: {self.ssz_score:.4f}",
            f"P-value: {self.p_value:.4f}",
            f"SSZ significant: {self.is_ssz_significant}",
            "",
            "FIT QUALITY",
            "-" * 40,
            f"RMSE: {self.rmse:.4f} Hz",
            f"R-squared: {self.r_squared:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_full_pipeline(
    config: Optional[PipelineConfig] = None,
    **kwargs,
) -> PipelineResult:
    """
    Run the complete SSZ Schumann analysis pipeline.
    
    This is the main entry point for the analysis. It:
    1. Loads data (synthetic or real)
    2. Calibrates the classical model (eta_0)
    3. Estimates delta_seg (SSZ correction)
    4. Computes diagnostics and statistics
    5. Optionally saves figures and CSV files
    
    Args:
        config: PipelineConfig object (or None to use defaults)
        **kwargs: Override config parameters
    
    Returns:
        PipelineResult with all analysis outputs
    
    Example:
        >>> # Default synthetic analysis
        >>> result = run_full_pipeline()
        >>> print(result.summary())
        
        >>> # Custom configuration
        >>> config = PipelineConfig(
        ...     delta_seg_amplitude=0.03,
        ...     noise_level=0.02,
        ...     eta_mode="joint_fit"
        ... )
        >>> result = run_full_pipeline(config)
    """
    import time
    start_time = time.time()
    
    # Setup configuration
    if config is None:
        config = PipelineConfig()
    
    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if config.verbose:
        logger.info("Starting SSZ Schumann analysis pipeline")
        logger.info(f"Data source: {config.data_source}")
    
    # Initialize result
    result = PipelineResult(config=config)
    result.timestamp = datetime.now().isoformat()
    
    # Step 1: Load data
    from ..data_io.data_loader import load_schumann_timeseries
    
    if config.data_source == "synthetic":
        data = load_schumann_timeseries(
            "synthetic",
            start=config.start_date,
            end=config.end_date,
            freq=config.time_freq,
            delta_seg_amplitude=config.delta_seg_amplitude,
            noise_level=config.noise_level,
            seed=config.seed,
        )
    else:
        data = load_schumann_timeseries(config.data_path or config.data_source)
    
    result.data = data
    
    if config.verbose:
        logger.info(f"Loaded {len(data)} data points")
    
    # Step 2: Calibrate classical model
    from ..models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    from ..models.eta_calibration import (
        calibrate_eta_quiet_interval,
        joint_fit_eta_and_segmentation,
    )
    
    if config.eta_mode == "fixed":
        eta_0 = config.eta_fixed
    elif config.eta_mode == "full_fit":
        eta_0 = compute_eta0_from_mean_f1(data["f1_obs"])
    elif config.eta_mode == "quiet_interval":
        # Use first quiet_days as calibration period
        f_obs = {
            1: data["f1_obs"],
            2: data["f2_obs"],
            3: data["f3_obs"],
        }
        cal_result = calibrate_eta_quiet_interval(
            f_obs, data.index, quiet_days=config.quiet_days
        )
        eta_0 = cal_result.eta_0
    elif config.eta_mode == "joint_fit":
        f_obs = {
            1: data["f1_obs"],
            2: data["f2_obs"],
            3: data["f3_obs"],
        }
        fit_result = joint_fit_eta_and_segmentation(f_obs, data.index)
        eta_0 = fit_result["eta_0"]
    else:
        raise ValueError(f"Unknown eta_mode: {config.eta_mode}")
    
    result.eta_0 = eta_0
    result.f_classical = {
        1: f_n_classical(1, eta_0),
        2: f_n_classical(2, eta_0),
        3: f_n_classical(3, eta_0),
    }
    
    if config.verbose:
        logger.info(f"Calibrated eta_0 = {eta_0:.6f}")
    
    # Step 3: Estimate delta_seg
    from ..models.ssz_correction import delta_seg_from_observed, check_mode_consistency
    
    delta_seg_dict = {}
    for n in [1, 2, 3]:
        f_obs = data[f"f{n}_obs"]
        f_class = result.f_classical[n]
        delta_seg_dict[n] = delta_seg_from_observed(f_obs, f_class)
    
    # Combined delta_seg (mean across modes)
    delta_seg_combined = (
        delta_seg_dict[1] + delta_seg_dict[2] + delta_seg_dict[3]
    ) / 3
    
    result.delta_seg = delta_seg_combined
    result.delta_seg_mean = float(np.nanmean(delta_seg_combined))
    result.delta_seg_std = float(np.nanstd(delta_seg_combined))
    
    # Step 4: Check mode consistency (SSZ signature)
    consistency = check_mode_consistency(delta_seg_dict)
    result.mode_correlation = consistency.get("mean_correlation", 0.0)
    result.ssz_score = consistency.get("ssz_score", 0.0)
    
    # Step 5: Statistical test
    from ..analysis.ssz_detection import test_ssz_signature
    
    # test_ssz_signature expects a dictionary {mode: series}
    try:
        test_result = test_ssz_signature(delta_seg_dict, n_null_realizations=100)
        result.is_ssz_significant = test_result.is_significant
        result.p_value = test_result.p_value
    except Exception as e:
        logger.warning(f"SSZ significance test failed: {e}")
        result.is_ssz_significant = False
        result.p_value = 1.0
    
    # Step 6: Compute residuals and fit quality
    residuals = {}
    ss_res = 0.0
    ss_tot = 0.0
    
    for n in [1, 2, 3]:
        f_obs = data[f"f{n}_obs"]
        f_pred = result.f_classical[n] / (1 + delta_seg_combined)
        resid = f_obs - f_pred
        residuals[n] = resid
        
        ss_res += np.sum(resid**2)
        ss_tot += np.sum((f_obs - np.mean(f_obs))**2)
    
    result.residuals = residuals
    result.rmse = float(np.sqrt(ss_res / (3 * len(data))))
    result.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Step 7: Save outputs
    if config.save_csv or config.save_figures:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if config.save_csv:
            # Save delta_seg time series
            delta_seg_df = pd.DataFrame({
                "delta_seg_1": delta_seg_dict[1],
                "delta_seg_2": delta_seg_dict[2],
                "delta_seg_3": delta_seg_dict[3],
                "delta_seg_combined": delta_seg_combined,
            })
            delta_seg_df.to_csv(output_dir / "delta_seg_timeseries.csv")
            
            # Save summary
            with open(output_dir / "pipeline_results.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
    
    # Finalize
    result.runtime_seconds = time.time() - start_time
    
    if config.verbose:
        print(result.summary())
    
    return result


def run_quick_analysis(
    data: pd.DataFrame,
    eta_0: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run quick analysis on pre-loaded data.
    
    Simplified version of run_full_pipeline for interactive use.
    
    Args:
        data: DataFrame with f1_obs, f2_obs, f3_obs columns
        eta_0: Pre-calibrated eta_0 (or None to compute)
    
    Returns:
        Dictionary with key results
    """
    from ..models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
    from ..models.ssz_correction import delta_seg_from_observed, check_mode_consistency
    
    # Calibrate eta_0 if not provided
    if eta_0 is None:
        eta_0 = compute_eta0_from_mean_f1(data["f1_obs"])
    
    # Compute classical frequencies
    f_classical = {n: f_n_classical(n, eta_0) for n in [1, 2, 3]}
    
    # Extract delta_seg for each mode
    delta_seg = {}
    for n in [1, 2, 3]:
        delta_seg[n] = delta_seg_from_observed(data[f"f{n}_obs"], f_classical[n])
    
    # Check consistency
    consistency = check_mode_consistency(delta_seg)
    
    # Combined delta_seg
    delta_seg_combined = (delta_seg[1] + delta_seg[2] + delta_seg[3]) / 3
    
    return {
        "eta_0": eta_0,
        "f_classical": f_classical,
        "delta_seg": delta_seg,
        "delta_seg_combined": delta_seg_combined,
        "delta_seg_mean": float(np.nanmean(delta_seg_combined)),
        "delta_seg_std": float(np.nanstd(delta_seg_combined)),
        "mode_correlation": consistency.get("mean_correlation", 0.0),
        "ssz_score": consistency.get("ssz_score", 0.0),
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("=== Pipeline Test ===\n")
    
    # Run with default config
    result = run_full_pipeline(
        PipelineConfig(
            delta_seg_amplitude=0.02,
            noise_level=0.01,
            verbose=True,
        )
    )
