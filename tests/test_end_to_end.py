#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Tests for SSZ Schumann Analysis

Tests the complete analysis pipeline with synthetic data.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.config import C_LIGHT, EARTH_RADIUS
from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
from ssz_schumann.data_io.merge import merge_all, compute_derived_variables
from ssz_schumann.models.classical_schumann import f_n_classical, compute_eta0_from_mean_f1
from ssz_schumann.models.ssz_correction import (
    delta_seg_from_observed,
    fit_delta_seg_simple,
    check_mode_consistency,
)
from ssz_schumann.models.fit_wrappers import (
    fit_classical_model,
    fit_ssz_model,
    compare_models,
)
from ssz_schumann.analysis.compute_deltas import compute_all_deltas, run_analysis_pipeline


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_create_synthetic_schumann(self):
        """Test synthetic Schumann data creation."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
            eta_0=0.74,
            delta_seg_amplitude=0.02,
            noise_level=0.01,
        )
        
        assert "f1" in ds
        assert "f2" in ds
        assert "f3" in ds
        assert "delta_seg_true" in ds
        
        # Check frequency ranges
        assert 6 < ds["f1"].mean() < 10
        assert 12 < ds["f2"].mean() < 18
        assert 18 < ds["f3"].mean() < 24
    
    def test_create_synthetic_space_weather(self):
        """Test synthetic space weather data creation."""
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-01-31",
        )
        
        assert len(f107) > 0
        assert len(kp) > 0
        
        # Check ranges
        assert 70 < f107.mean() < 200
        assert 0 < kp.mean() < 9


class TestDataMerging:
    """Tests for data merging functionality."""
    
    def test_merge_all(self):
        """Test merging Schumann and space weather data."""
        # Create synthetic data
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-01-31",
        )
        
        # Merge
        merged = merge_all(ds, f107, kp, time_resolution="1h")
        
        assert "f1" in merged
        assert "f107" in merged
        assert "f107_norm" in merged
        assert "kp" in merged
        assert "kp_norm" in merged
    
    def test_compute_derived_variables(self):
        """Test derived variable computation."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        ds = compute_derived_variables(ds)
        
        assert "f_mean" in ds
        assert "f_ratio_21" in ds
        assert "hour_utc" in ds
        assert "day_of_year" in ds


class TestDeltaComputation:
    """Tests for delta_seg computation."""
    
    def test_compute_all_deltas(self):
        """Test delta computation pipeline."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
            delta_seg_amplitude=0.02,
        )
        
        ds = compute_all_deltas(ds)
        
        assert "delta_seg_1" in ds
        assert "delta_seg_2" in ds
        assert "delta_seg_3" in ds
        assert "delta_seg_mean" in ds
        assert "eta_0" in ds.attrs
    
    def test_delta_recovery(self):
        """Test that we can recover the true delta_seg."""
        # Create data with known delta_seg
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
            eta_0=0.74,
            delta_seg_amplitude=0.02,
            noise_level=0.001,  # Very low noise
        )
        
        ds = compute_all_deltas(ds)
        
        # Compare extracted delta_seg with true
        delta_true = ds["delta_seg_true"].values
        delta_extracted = ds["delta_seg_mean"].values
        
        # Should be correlated
        valid = ~(np.isnan(delta_true) | np.isnan(delta_extracted))
        correlation = np.corrcoef(delta_true[valid], delta_extracted[valid])[0, 1]
        
        assert correlation > 0.9, f"Correlation = {correlation}, expected > 0.9"


class TestModelFitting:
    """Tests for model fitting."""
    
    def test_fit_classical_model(self):
        """Test classical model fitting."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        result = fit_classical_model(ds)
        
        assert result.model_type == "classical"
        assert 0.7 < result.eta_0 < 0.8
        assert result.r_squared >= 0  # Can be low for synthetic data
        assert result.rmse > 0
    
    def test_fit_ssz_model(self):
        """Test SSZ model fitting."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-01-31",
        )
        
        merged = merge_all(ds, f107, kp, time_resolution="1h")
        
        features = pd.DataFrame({
            "f107_norm": merged["f107_norm"].values,
            "kp_norm": merged["kp_norm"].values,
        }, index=pd.DatetimeIndex(merged.time.values))
        
        result = fit_ssz_model(merged, features)
        
        assert result.model_type == "ssz"
        assert result.ssz_params is not None
        assert "beta_0" in result.ssz_params
        assert "beta_1" in result.ssz_params
    
    def test_compare_models(self):
        """Test model comparison."""
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-01-31",
        )
        
        merged = merge_all(ds, f107, kp, time_resolution="1h")
        
        features = pd.DataFrame({
            "f107_norm": merged["f107_norm"].values,
            "kp_norm": merged["kp_norm"].values,
        }, index=pd.DatetimeIndex(merged.time.values))
        
        classical_result = fit_classical_model(merged)
        ssz_result = fit_ssz_model(merged, features)
        
        comparison = compare_models(classical_result, ssz_result)
        
        assert "delta_r_squared" in comparison
        assert "delta_aic" in comparison
        assert "preferred_model" in comparison


class TestModeConsistency:
    """Tests for mode consistency (SSZ signature)."""
    
    def test_ssz_signature_detection(self):
        """Test that SSZ signature is detected in synthetic data."""
        # Create data with SSZ signature (same delta_seg for all modes)
        # Use lower noise to get cleaner SSZ signature
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-03-31",
            freq="1h",
            delta_seg_amplitude=0.02,
            noise_level=0.002,  # Lower noise for clearer signal
        )
        
        ds = compute_all_deltas(ds)
        
        delta_seg_dict = {
            n: pd.Series(ds[f"delta_seg_{n}"].values,
                        index=pd.DatetimeIndex(ds.time.values))
            for n in [1, 2, 3]
        }
        
        result = check_mode_consistency(delta_seg_dict)
        
        # Should detect SSZ signature
        # Key indicator is high correlation between modes
        assert result["mean_correlation"] > 0.7, f"Mean correlation = {result['mean_correlation']}"
        # SSZ score should be reasonable (depends on noise level)
        assert result["ssz_score"] > 0.3, f"SSZ score = {result['ssz_score']}"


class TestFullPipeline:
    """Tests for complete analysis pipeline."""
    
    def test_run_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        # Create synthetic data
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-01-31",
            freq="1h",
        )
        
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-01-31",
        )
        
        merged = merge_all(ds, f107, kp, time_resolution="1h")
        
        features = pd.DataFrame({
            "f107_norm": merged["f107_norm"].values,
            "kp_norm": merged["kp_norm"].values,
        }, index=pd.DatetimeIndex(merged.time.values))
        
        # Run pipeline
        results = run_analysis_pipeline(merged, features)
        
        assert "eta_0" in results
        assert "mode_consistency" in results
        assert "classical_model" in results
        assert "ssz_model" in results
        assert "model_comparison" in results
        assert "summary" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
