#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SSZ Schumann Models

Tests classical Schumann model and SSZ correction functions.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ssz_schumann.config import C_LIGHT, EARTH_RADIUS, ETA_0_DEFAULT, PHI
from ssz_schumann.models.classical_schumann import (
    f_n_classical,
    schumann_mode_factor,
    compute_eta0_from_mean_f1,
    f_n_classical_timeseries,
)
from ssz_schumann.models.ssz_correction import (
    D_SSZ,
    delta_seg_from_observed,
    f_n_ssz_model,
    check_mode_consistency,
)


class TestClassicalSchumann:
    """Tests for classical Schumann model."""
    
    def test_mode_factor(self):
        """Test mode factor calculation."""
        # sqrt(n*(n+1))
        assert np.isclose(schumann_mode_factor(1), np.sqrt(2))
        assert np.isclose(schumann_mode_factor(2), np.sqrt(6))
        assert np.isclose(schumann_mode_factor(3), np.sqrt(12))
    
    def test_f_n_classical_values(self):
        """Test that classical frequencies are in expected range."""
        # With default eta_0 ~ 0.74, expect:
        # f1 ~ 7.83 Hz
        # f2 ~ 14.1 Hz
        # f3 ~ 20.3 Hz
        
        f1 = f_n_classical(1)
        f2 = f_n_classical(2)
        f3 = f_n_classical(3)
        
        assert 7.0 < f1 < 9.0, f"f1 = {f1} out of range"
        assert 13.0 < f2 < 16.0, f"f2 = {f2} out of range"
        assert 19.0 < f3 < 22.0, f"f3 = {f3} out of range"
    
    def test_f_n_classical_eta_1(self):
        """Test ideal frequencies with eta = 1."""
        # Ideal: f_n = c / (2*pi*R) * sqrt(n*(n+1))
        f1_ideal = f_n_classical(1, eta=1.0)
        f1_expected = C_LIGHT / (2 * np.pi * EARTH_RADIUS) * np.sqrt(2)
        
        assert np.isclose(f1_ideal, f1_expected)
        assert f1_ideal > 10.0  # Ideal f1 ~ 10.6 Hz
    
    def test_f_n_classical_scaling(self):
        """Test that frequencies scale correctly with eta."""
        eta1 = 0.7
        eta2 = 0.8
        
        f1_eta1 = f_n_classical(1, eta=eta1)
        f1_eta2 = f_n_classical(1, eta=eta2)
        
        # f should scale linearly with eta
        assert np.isclose(f1_eta2 / f1_eta1, eta2 / eta1)
    
    def test_compute_eta0_from_mean_f1(self):
        """Test eta_0 calibration from observed f1."""
        # If we observe f1 = 7.83 Hz, eta_0 should be ~ 0.74
        f1_obs = 7.83
        eta_0 = compute_eta0_from_mean_f1(f1_obs)
        
        assert 0.7 < eta_0 < 0.8
        
        # Verify: f_n_classical(1, eta_0) should give back f1_obs
        f1_reconstructed = f_n_classical(1, eta=eta_0)
        assert np.isclose(f1_reconstructed, f1_obs, rtol=1e-6)
    
    def test_f_n_classical_timeseries(self):
        """Test time series generation."""
        time = pd.date_range("2016-01-01", periods=100, freq="1h")
        eta = 0.74
        
        f1_series = f_n_classical_timeseries(1, eta, time_index=time)
        
        assert len(f1_series) == len(time)
        assert f1_series.index.equals(time)
        assert np.all(f1_series == f_n_classical(1, eta))


class TestSSZCorrection:
    """Tests for SSZ correction model."""
    
    def test_D_SSZ_basic(self):
        """Test D_SSZ calculation."""
        # D_SSZ = 1 + delta_seg
        assert D_SSZ(0.0) == 1.0
        assert D_SSZ(0.01) == 1.01
        assert D_SSZ(-0.01) == 0.99
    
    def test_D_SSZ_array(self):
        """Test D_SSZ with array input."""
        delta_seg = np.array([0.0, 0.01, 0.02, -0.01])
        d_ssz = D_SSZ(delta_seg)
        
        expected = np.array([1.0, 1.01, 1.02, 0.99])
        assert np.allclose(d_ssz, expected)
    
    def test_f_n_ssz_model(self):
        """Test SSZ frequency modification."""
        f_classical = 7.83
        delta_seg = 0.01
        
        f_ssz = f_n_ssz_model(f_classical, delta_seg)
        
        # f_ssz = f_classical / (1 + delta_seg)
        expected = f_classical / 1.01
        assert np.isclose(f_ssz, expected)
        
        # SSZ should lower frequency when delta_seg > 0
        assert f_ssz < f_classical
    
    def test_delta_seg_from_observed(self):
        """Test delta_seg extraction."""
        f_classical = 7.83
        delta_seg_true = 0.02
        
        # Generate observed frequency
        f_obs = f_classical / (1 + delta_seg_true)
        
        # Extract delta_seg
        delta_seg_extracted = delta_seg_from_observed(f_obs, f_classical)
        
        assert np.isclose(delta_seg_extracted, delta_seg_true, rtol=1e-6)
    
    def test_delta_seg_roundtrip(self):
        """Test that delta_seg extraction is inverse of SSZ model."""
        f_classical = 7.83
        
        for delta_seg_true in [0.0, 0.01, 0.02, -0.01, 0.05]:
            f_obs = f_n_ssz_model(f_classical, delta_seg_true)
            delta_seg_recovered = delta_seg_from_observed(f_obs, f_classical)
            
            assert np.isclose(delta_seg_recovered, delta_seg_true, rtol=1e-6), \
                f"Failed for delta_seg = {delta_seg_true}"
    
    def test_mode_consistency_perfect(self):
        """Test mode consistency with perfectly consistent data."""
        # Create synthetic data where all modes have same delta_seg
        time = pd.date_range("2016-01-01", periods=1000, freq="1h")
        delta_seg_true = 0.01 + 0.005 * np.sin(2 * np.pi * np.arange(1000) / 24)
        
        delta_seg_dict = {
            1: pd.Series(delta_seg_true, index=time),
            2: pd.Series(delta_seg_true, index=time),
            3: pd.Series(delta_seg_true, index=time),
        }
        
        result = check_mode_consistency(delta_seg_dict)
        
        # Should show perfect consistency
        assert result["ssz_score"] > 0.9
        assert result["is_consistent"]
        assert result["std_delta_seg"] < 1e-10
    
    def test_mode_consistency_inconsistent(self):
        """Test mode consistency with inconsistent data."""
        time = pd.date_range("2016-01-01", periods=1000, freq="1h")
        
        # Different delta_seg for each mode
        delta_seg_dict = {
            1: pd.Series(0.01 * np.ones(1000), index=time),
            2: pd.Series(0.05 * np.ones(1000), index=time),  # Very different!
            3: pd.Series(-0.02 * np.ones(1000), index=time),  # Very different!
        }
        
        result = check_mode_consistency(delta_seg_dict)
        
        # Should show inconsistency
        assert result["ssz_score"] < 0.5
        assert not result["is_consistent"]


class TestPhysicalConsistency:
    """Tests for physical consistency of the model."""
    
    def test_frequency_ratios(self):
        """Test that frequency ratios match theoretical predictions."""
        eta = 0.74
        
        f1 = f_n_classical(1, eta)
        f2 = f_n_classical(2, eta)
        f3 = f_n_classical(3, eta)
        
        # Theoretical ratios (independent of eta)
        # f2/f1 = sqrt(6)/sqrt(2) = sqrt(3)
        # f3/f1 = sqrt(12)/sqrt(2) = sqrt(6)
        
        assert np.isclose(f2 / f1, np.sqrt(3), rtol=1e-6)
        assert np.isclose(f3 / f1, np.sqrt(6), rtol=1e-6)
    
    def test_ssz_preserves_ratios(self):
        """Test that SSZ correction preserves frequency ratios."""
        eta = 0.74
        delta_seg = 0.02
        
        f1_class = f_n_classical(1, eta)
        f2_class = f_n_classical(2, eta)
        f3_class = f_n_classical(3, eta)
        
        f1_ssz = f_n_ssz_model(f1_class, delta_seg)
        f2_ssz = f_n_ssz_model(f2_class, delta_seg)
        f3_ssz = f_n_ssz_model(f3_class, delta_seg)
        
        # Ratios should be preserved (key SSZ prediction!)
        assert np.isclose(f2_ssz / f1_ssz, f2_class / f1_class, rtol=1e-6)
        assert np.isclose(f3_ssz / f1_ssz, f3_class / f1_class, rtol=1e-6)
    
    def test_relative_shift_uniform(self):
        """Test that relative frequency shift is uniform across modes."""
        eta = 0.74
        delta_seg = 0.02
        
        for n in [1, 2, 3]:
            f_class = f_n_classical(n, eta)
            f_ssz = f_n_ssz_model(f_class, delta_seg)
            
            # Relative shift: (f_ssz - f_class) / f_class
            relative_shift = (f_ssz - f_class) / f_class
            
            # For small delta_seg: relative_shift ≈ -delta_seg
            expected_shift = -delta_seg / (1 + delta_seg)
            
            assert np.isclose(relative_shift, expected_shift, rtol=1e-6), \
                f"Mode {n}: relative_shift = {relative_shift}, expected = {expected_shift}"


class TestSSZSignatureDetection:
    """Tests for SSZ signature detection with synthetic data."""
    
    def test_strong_ssz_detection(self):
        """Test that strong SSZ signal is detected correctly."""
        # Create synthetic data with strong, correlated delta_seg
        np.random.seed(42)
        n_points = 500
        
        # Common signal (SSZ-like: same for all modes)
        common_signal = 0.02 * np.sin(np.linspace(0, 4*np.pi, n_points))
        
        # Small noise
        noise_level = 0.002
        
        delta_seg_dict = {
            1: common_signal + noise_level * np.random.randn(n_points),
            2: common_signal + noise_level * np.random.randn(n_points),
            3: common_signal + noise_level * np.random.randn(n_points),
        }
        
        result = check_mode_consistency(delta_seg_dict)
        
        # Should detect strong SSZ signature
        assert result["mean_correlation"] > 0.7, \
            f"Expected high correlation, got {result['mean_correlation']:.4f}"
        assert result["ssz_score"] > 0.5, \
            f"Expected high SSZ score, got {result['ssz_score']:.4f}"
    
    def test_null_ssz_detection(self):
        """Test that no SSZ signal gives low scores."""
        # Create synthetic data with NO common signal (just noise)
        np.random.seed(42)
        n_points = 500
        noise_level = 0.01
        
        delta_seg_dict = {
            1: noise_level * np.random.randn(n_points),
            2: noise_level * np.random.randn(n_points),
            3: noise_level * np.random.randn(n_points),
        }
        
        result = check_mode_consistency(delta_seg_dict)
        
        # Should NOT detect SSZ signature
        assert result["is_consistent"] == False, \
            "Should not detect SSZ in pure noise"
        assert abs(result["mean_correlation"]) < 0.5, \
            f"Expected low correlation for noise, got {result['mean_correlation']:.4f}"
    
    def test_ssz_score_formula(self):
        """Test that SSZ score follows the documented formula."""
        np.random.seed(42)
        n_points = 500
        
        # Perfect correlation, zero std across modes
        signal = np.linspace(0, 0.02, n_points)
        delta_seg_dict = {
            1: signal.copy(),
            2: signal.copy(),
            3: signal.copy(),
        }
        
        result = check_mode_consistency(delta_seg_dict, std_ref=0.01)
        
        # With perfect correlation and zero std, score should equal correlation
        assert result["mean_correlation"] > 0.99, "Should have perfect correlation"
        assert result["ssz_score"] > 0.99, "Score should be high with zero std"
    
    def test_interpretation_strings(self):
        """Test that interpretation strings are generated correctly."""
        np.random.seed(42)
        n_points = 100
        
        # Strong signal
        signal = 0.02 * np.sin(np.linspace(0, 4*np.pi, n_points))
        delta_seg_strong = {
            1: signal + 0.001 * np.random.randn(n_points),
            2: signal + 0.001 * np.random.randn(n_points),
            3: signal + 0.001 * np.random.randn(n_points),
        }
        
        result_strong = check_mode_consistency(delta_seg_strong)
        assert "interpretation" in result_strong
        
        # Noise only
        delta_seg_noise = {
            1: 0.01 * np.random.randn(n_points),
            2: 0.01 * np.random.randn(n_points),
            3: 0.01 * np.random.randn(n_points),
        }
        
        result_noise = check_mode_consistency(delta_seg_noise)
        assert "interpretation" in result_noise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
