#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for T1-T4 Implementation

Tests for:
- T1: Extended classical frequency formula
- T2: Standardized data structure and pipeline
- T3: Real data hooks
- T4: SSZ diagnostics

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# T1: EXTENDED CLASSICAL FREQUENCY TESTS
# =============================================================================

class TestT1ExtendedClassical:
    """Tests for T1: Extended geometry-aware classical model."""
    
    def test_f_n_classical_extended_default(self):
        """Test extended formula with default parameters."""
        from ssz_schumann.models.classical_schumann import (
            f_n_classical,
            f_n_classical_extended,
        )
        
        # With include_height_correction=False, should match original
        f1_original = f_n_classical(1)
        f1_extended = f_n_classical_extended(1, include_height_correction=False)
        
        assert abs(f1_extended - f1_original) < 0.01
    
    def test_f_n_classical_extended_height_effect(self):
        """Test that ionospheric height affects frequency."""
        from ssz_schumann.models.classical_schumann import f_n_classical_extended
        
        # Higher ionosphere -> larger effective radius -> lower frequency
        f1_low = f_n_classical_extended(1, h_iono=60_000)
        f1_high = f_n_classical_extended(1, h_iono=100_000)
        
        assert f1_low > f1_high, "Higher ionosphere should give lower frequency"
    
    def test_f_n_classical_with_latitude(self):
        """Test latitude-dependent ionosphere."""
        from ssz_schumann.models.classical_schumann import f_n_classical_with_latitude
        
        # Equator vs pole
        f1_equator = f_n_classical_with_latitude(1, latitude_deg=0)
        f1_pole = f_n_classical_with_latitude(1, latitude_deg=90)
        
        # Pole has higher ionosphere -> lower frequency
        assert f1_equator > f1_pole
    
    def test_f_n_classical_diurnal(self):
        """Test diurnal (day/night) variation."""
        from ssz_schumann.models.classical_schumann import f_n_classical_diurnal
        
        # Noon vs midnight
        f1_noon = f_n_classical_diurnal(1, local_hour=12)
        f1_midnight = f_n_classical_diurnal(1, local_hour=0)
        
        # Day has lower ionosphere -> higher frequency
        assert f1_noon > f1_midnight
    
    def test_extended_mode_ratios(self):
        """Test that mode ratios are preserved."""
        from ssz_schumann.models.classical_schumann import f_n_classical_extended
        
        f1 = f_n_classical_extended(1)
        f2 = f_n_classical_extended(2)
        f3 = f_n_classical_extended(3)
        
        # Classical ratios
        expected_f2_f1 = np.sqrt(3)  # sqrt(6)/sqrt(2)
        expected_f3_f1 = np.sqrt(6)  # sqrt(12)/sqrt(2)
        
        assert abs(f2/f1 - expected_f2_f1) < 0.01
        assert abs(f3/f1 - expected_f3_f1) < 0.01
    
    def test_invalid_parameters(self):
        """Test validation of invalid parameters."""
        from ssz_schumann.models.classical_schumann import f_n_classical_extended
        
        with pytest.raises(ValueError):
            f_n_classical_extended(0)  # Invalid mode
        
        with pytest.raises(ValueError):
            f_n_classical_extended(1, eta=1.5)  # eta > 1
        
        with pytest.raises(ValueError):
            f_n_classical_extended(1, h_iono=-1000)  # Negative height


# =============================================================================
# T2: DATA STRUCTURE AND PIPELINE TESTS
# =============================================================================

class TestT2DataLoader:
    """Tests for T2: Standardized data loader."""
    
    def test_load_synthetic_data(self):
        """Test synthetic data generation."""
        from ssz_schumann.data_io.data_loader import load_schumann_timeseries
        
        df = load_schumann_timeseries("synthetic", seed=42)
        
        # Check required columns
        assert "f1_obs" in df.columns
        assert "f2_obs" in df.columns
        assert "f3_obs" in df.columns
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_schema_validation(self):
        """Test schema validation."""
        from ssz_schumann.data_io.data_loader import SchumannDataSchema
        
        # Valid DataFrame
        df = pd.DataFrame({
            "f1_obs": [7.8, 7.9],
            "f2_obs": [13.9, 14.0],
            "f3_obs": [20.2, 20.3],
        }, index=pd.date_range("2016-01-01", periods=2, freq="1h"))
        
        assert SchumannDataSchema.validate(df)
        
        # Invalid DataFrame (missing column)
        df_invalid = pd.DataFrame({
            "f1_obs": [7.8, 7.9],
            "f2_obs": [13.9, 14.0],
        }, index=pd.date_range("2016-01-01", periods=2, freq="1h"))
        
        with pytest.raises(ValueError):
            SchumannDataSchema.validate(df_invalid)
    
    def test_synthetic_data_has_true_delta_seg(self):
        """Test that synthetic data includes true delta_seg."""
        from ssz_schumann.data_io.data_loader import load_synthetic_schumann_data
        
        df = load_synthetic_schumann_data(delta_seg_amplitude=0.02)
        
        assert "delta_seg_true" in df.columns
        assert abs(df["delta_seg_true"].std()) > 0  # Not constant
    
    def test_get_frequency_dict(self):
        """Test frequency dictionary extraction."""
        from ssz_schumann.data_io.data_loader import (
            load_schumann_timeseries,
            get_frequency_dict,
        )
        
        df = load_schumann_timeseries("synthetic")
        f_dict = get_frequency_dict(df)
        
        assert 1 in f_dict
        assert 2 in f_dict
        assert 3 in f_dict
        assert isinstance(f_dict[1], pd.Series)


class TestT2Pipeline:
    """Tests for T2: Unified analysis pipeline."""
    
    def test_pipeline_default_config(self):
        """Test pipeline with default configuration."""
        from ssz_schumann.analysis.pipeline import run_full_pipeline, PipelineConfig
        
        config = PipelineConfig(
            start_date="2016-01-01",
            end_date="2016-01-31",  # Short period for testing
            verbose=False,
            save_csv=False,
            save_figures=False,
        )
        
        result = run_full_pipeline(config)
        
        assert result.eta_0 > 0
        assert result.eta_0 < 1
        assert 1 in result.f_classical
        assert result.delta_seg is not None
    
    def test_pipeline_result_summary(self):
        """Test pipeline result summary generation."""
        from ssz_schumann.analysis.pipeline import run_full_pipeline, PipelineConfig
        
        config = PipelineConfig(
            start_date="2016-01-01",
            end_date="2016-01-07",
            verbose=False,
            save_csv=False,
        )
        
        result = run_full_pipeline(config)
        summary = result.summary()
        
        assert "SSZ SCHUMANN ANALYSIS RESULTS" in summary
        assert "eta_0" in summary
        assert "delta_seg" in summary
    
    def test_quick_analysis(self):
        """Test quick analysis function."""
        from ssz_schumann.data_io.data_loader import load_schumann_timeseries
        from ssz_schumann.analysis.pipeline import run_quick_analysis
        
        df = load_schumann_timeseries("synthetic", seed=42)
        result = run_quick_analysis(df)
        
        assert "eta_0" in result
        assert "f_classical" in result
        assert "delta_seg_mean" in result


# =============================================================================
# T3: REAL DATA HOOKS TESTS
# =============================================================================

class TestT3RealDataHooks:
    """Tests for T3: Real data hooks."""
    
    def test_real_data_loader_not_implemented(self):
        """Test that real data loader raises NotImplementedError."""
        from ssz_schumann.data_io.data_loader import load_real_schumann_data
        
        with pytest.raises(NotImplementedError) as exc_info:
            load_real_schumann_data("dummy_path.csv")
        
        # Check that error message is informative
        assert "Expected CSV format" in str(exc_info.value)
    
    def test_load_from_csv_path(self):
        """Test loading from CSV file (when it exists)."""
        from ssz_schumann.data_io.data_loader import load_schumann_timeseries
        import tempfile
        import os
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("time,f1_obs,f2_obs,f3_obs\n")
            f.write("2016-01-01 00:00:00,7.82,13.9,20.1\n")
            f.write("2016-01-01 01:00:00,7.84,13.95,20.15\n")
            temp_path = f.name
        
        try:
            df = load_schumann_timeseries(temp_path)
            assert len(df) == 2
            assert "f1_obs" in df.columns
        finally:
            os.unlink(temp_path)


# =============================================================================
# T4: SSZ DIAGNOSTICS TESTS
# =============================================================================

class TestT4Diagnostics:
    """Tests for T4: SSZ signature diagnostics."""
    
    def test_compute_relative_shifts(self):
        """Test relative shift computation."""
        from ssz_schumann.analysis.ssz_diagnostics import compute_relative_shifts
        
        # Create test data
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        
        # SSZ-like data (uniform relative shift)
        delta_seg = 0.01
        f_obs = {
            n: pd.Series(f_c / (1 + delta_seg), index=time)
            for n, f_c in f_classical.items()
        }
        
        shifts = compute_relative_shifts(f_obs, f_classical)
        
        # All modes should have same relative shift
        for mode in [1, 2, 3]:
            assert abs(shifts[mode].mean_shift + delta_seg) < 0.001
    
    def test_check_mode_independence_ssz(self):
        """Test mode independence check with SSZ-like data."""
        from ssz_schumann.analysis.ssz_diagnostics import (
            compute_relative_shifts,
            check_mode_independence,
        )
        
        # Create SSZ-like data (uniform shift)
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        delta_seg = 0.01
        
        f_obs = {
            n: pd.Series(f_c / (1 + delta_seg), index=time)
            for n, f_c in f_classical.items()
        }
        
        shifts = compute_relative_shifts(f_obs, f_classical)
        result = check_mode_independence(shifts)
        
        assert result.is_mode_independent
        assert "CONSISTENT WITH SSZ" in result.interpretation
    
    def test_check_mode_independence_dispersive(self):
        """Test mode independence check with dispersive data."""
        from ssz_schumann.analysis.ssz_diagnostics import (
            compute_relative_shifts,
            check_mode_independence,
        )
        
        # Create dispersive data (mode-dependent shift)
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        
        # Different shift for each mode
        f_obs = {
            1: pd.Series(f_classical[1] * 0.99, index=time),   # -1%
            2: pd.Series(f_classical[2] * 0.97, index=time),   # -3%
            3: pd.Series(f_classical[3] * 0.95, index=time),   # -5%
        }
        
        shifts = compute_relative_shifts(f_obs, f_classical)
        result = check_mode_independence(shifts)
        
        assert not result.is_mode_independent
        assert "CLASSICAL DISPERSION" in result.interpretation
    
    def test_delta_seg_with_confidence(self):
        """Test delta_seg computation with confidence bands."""
        from ssz_schumann.analysis.ssz_diagnostics import (
            compute_delta_seg_with_confidence,
        )
        
        # Create test data
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        delta_seg_true = 0.01
        
        f_obs = {
            n: pd.Series(f_c / (1 + delta_seg_true), index=time)
            for n, f_c in f_classical.items()
        }
        
        mean, lower, upper = compute_delta_seg_with_confidence(f_obs, f_classical)
        
        # Mean should be close to true value
        assert abs(mean.mean() - delta_seg_true) < 0.001
        
        # Confidence band should contain mean
        assert (lower <= mean).all()
        assert (mean <= upper).all()
    
    def test_detect_dispersion_pattern(self):
        """Test dispersion pattern detection."""
        from ssz_schumann.analysis.ssz_diagnostics import detect_dispersion_pattern
        
        # Create test data with dispersion
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        
        # Mode-dependent shift (dispersive)
        f_obs = {
            1: pd.Series(f_classical[1] * 0.99, index=time),
            2: pd.Series(f_classical[2] * 0.98, index=time),
            3: pd.Series(f_classical[3] * 0.97, index=time),
        }
        
        result = detect_dispersion_pattern(f_obs, f_classical)
        
        assert result["is_dispersive"]
        assert abs(result["slope"]) > 0.001
    
    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        from ssz_schumann.analysis.ssz_diagnostics import generate_diagnostic_report
        
        # Create test data
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        f_classical = {1: 7.83, 2: 13.9, 3: 20.3}
        
        f_obs = {
            n: pd.Series(f_c * 0.99, index=time)
            for n, f_c in f_classical.items()
        }
        
        report = generate_diagnostic_report(f_obs, f_classical)
        
        assert "SSZ SIGNATURE DIAGNOSTIC REPORT" in report
        assert "RELATIVE FREQUENCY SHIFTS" in report
        assert "MODE INDEPENDENCE CHECK" in report
        assert "DISPERSION ANALYSIS" in report


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for T1-T4."""
    
    def test_full_workflow(self):
        """Test complete workflow from data to diagnostics."""
        from ssz_schumann.data_io.data_loader import load_schumann_timeseries
        from ssz_schumann.models.classical_schumann import f_n_classical
        from ssz_schumann.analysis.ssz_diagnostics import generate_diagnostic_report
        
        # Load data
        df = load_schumann_timeseries("synthetic", delta_seg_amplitude=0.02, seed=42)
        
        # Compute classical frequencies
        eta_0 = 0.74
        f_classical = {n: f_n_classical(n, eta_0) for n in [1, 2, 3]}
        
        # Extract observed frequencies
        f_obs = {
            1: df["f1_obs"],
            2: df["f2_obs"],
            3: df["f3_obs"],
        }
        
        # Generate diagnostic report
        report = generate_diagnostic_report(f_obs, f_classical)
        
        assert len(report) > 0
        assert "SSZ" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
