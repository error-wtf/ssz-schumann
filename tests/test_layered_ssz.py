#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Layered SSZ Model

Comprehensive tests for the layered SSZ correction model.
Includes tests for core SSZ formulas from ssz-metric-pure.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import pytest

from ssz_schumann.models.layered_ssz import (
    LayerConfig,
    LayeredSSZConfig,
    D_SSZ_layered,
    D_SSZ_from_sigmas,
    effective_delta_seg,
    f_n_classical,
    f_n_ssz_layered,
    compute_all_modes,
    sigma_from_phi_ratio,
    phi_segment_density,
    create_phi_based_config,
    sigma_iono_from_proxy,
    f_n_ssz_timeseries,
    frequency_shift_estimate,
)
from ssz_schumann.config import PHI


class TestLayerConfig:
    """Tests for LayerConfig dataclass."""
    
    def test_layer_config_creation(self):
        """Test basic layer config creation."""
        layer = LayerConfig(name="test", weight=0.5, sigma=0.01)
        assert layer.name == "test"
        assert layer.weight == 0.5
        assert layer.sigma == 0.01
    
    def test_layer_config_defaults(self):
        """Test default values."""
        layer = LayerConfig(name="test", weight=0.5)
        assert layer.sigma == 0.0
        assert layer.height_km == 0.0


class TestLayeredSSZConfig:
    """Tests for LayeredSSZConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LayeredSSZConfig()
        
        assert config.ground.weight == 0.0
        assert config.atmosphere.weight == 0.2
        assert config.ionosphere.weight == 0.8
        
        assert config.eta_0 == 0.74
        assert config.f1_ref == 7.83
    
    def test_layers_property(self):
        """Test layers list property."""
        config = LayeredSSZConfig()
        layers = config.layers
        
        assert len(layers) == 3
        assert layers[0].name == "ground"
        assert layers[1].name == "atmosphere"
        assert layers[2].name == "ionosphere"
    
    def test_total_weight(self):
        """Test total weight calculation."""
        config = LayeredSSZConfig()
        # Default: 0 + 0.2 + 0.8 = 1.0
        assert config.total_weight == pytest.approx(1.0)
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        config = LayeredSSZConfig()
        config.atmosphere.weight = 0.4
        config.ionosphere.weight = 1.6
        # Total = 2.0
        
        config.normalize_weights()
        
        assert config.total_weight == pytest.approx(1.0)
        assert config.atmosphere.weight == pytest.approx(0.2)
        assert config.ionosphere.weight == pytest.approx(0.8)


class TestDSSZCalculations:
    """Tests for D_SSZ calculations."""
    
    def test_D_SSZ_no_segmentation(self):
        """Test D_SSZ with zero segmentation."""
        config = LayeredSSZConfig()
        # All sigma = 0
        
        d_ssz = D_SSZ_layered(config)
        assert d_ssz == pytest.approx(1.0)
    
    def test_D_SSZ_ionosphere_only(self):
        """Test D_SSZ with ionosphere segmentation only."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.01  # 1%
        
        d_ssz = D_SSZ_layered(config)
        # D_SSZ = 1 + 0.8 * 0.01 = 1.008
        assert d_ssz == pytest.approx(1.008)
    
    def test_D_SSZ_all_layers(self):
        """Test D_SSZ with all layers contributing."""
        config = LayeredSSZConfig()
        config.ground.sigma = 0.005
        config.atmosphere.sigma = 0.01
        config.ionosphere.sigma = 0.02
        
        d_ssz = D_SSZ_layered(config)
        # D_SSZ = 1 + 0*0.005 + 0.2*0.01 + 0.8*0.02
        #       = 1 + 0 + 0.002 + 0.016 = 1.018
        assert d_ssz == pytest.approx(1.018)
    
    def test_D_SSZ_from_sigmas_function(self):
        """Test D_SSZ_from_sigmas convenience function."""
        d_ssz = D_SSZ_from_sigmas(
            sigma_iono=0.01,
            w_iono=0.8
        )
        assert d_ssz == pytest.approx(1.008)
    
    def test_effective_delta_seg(self):
        """Test effective delta_seg calculation."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.01
        
        delta_seg = effective_delta_seg(config)
        assert delta_seg == pytest.approx(0.008)


class TestFrequencyCalculations:
    """Tests for frequency calculations."""
    
    def test_f_n_classical_mode1(self):
        """Test classical f1."""
        f1 = f_n_classical(1, f1_ref=7.83)
        # f1 = 7.83 * sqrt(2) / sqrt(2) = 7.83
        assert f1 == pytest.approx(7.83)
    
    def test_f_n_classical_mode2(self):
        """Test classical f2."""
        f2 = f_n_classical(2, f1_ref=7.83)
        # f2 = 7.83 * sqrt(6) / sqrt(2) = 7.83 * sqrt(3)
        assert f2 == pytest.approx(7.83 * np.sqrt(3))
    
    def test_f_n_classical_mode3(self):
        """Test classical f3."""
        f3 = f_n_classical(3, f1_ref=7.83)
        # f3 = 7.83 * sqrt(12) / sqrt(2) = 7.83 * sqrt(6)
        assert f3 == pytest.approx(7.83 * np.sqrt(6))
    
    def test_f_n_classical_invalid_mode(self):
        """Test error for invalid mode."""
        with pytest.raises(ValueError):
            f_n_classical(0)
    
    def test_f_n_ssz_layered_no_correction(self):
        """Test SSZ frequency with no correction."""
        config = LayeredSSZConfig()
        
        f1_ssz = f_n_ssz_layered(1, config)
        f1_class = f_n_classical(1, config.f1_ref)
        
        assert f1_ssz == pytest.approx(f1_class)
    
    def test_f_n_ssz_layered_with_correction(self):
        """Test SSZ frequency with correction."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.01
        
        f1_ssz = f_n_ssz_layered(1, config)
        f1_class = f_n_classical(1, config.f1_ref)
        
        # f1_ssz = f1_class / 1.008
        expected = f1_class / 1.008
        assert f1_ssz == pytest.approx(expected)
    
    def test_compute_all_modes(self):
        """Test computing all modes at once."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.01
        
        results = compute_all_modes(config)
        
        assert 1 in results
        assert 2 in results
        assert 3 in results
        
        for n in [1, 2, 3]:
            assert "f_classical" in results[n]
            assert "f_ssz" in results[n]
            assert "delta_f" in results[n]
            assert "relative_shift" in results[n]
    
    def test_relative_shift_uniform(self):
        """Test that relative shift is uniform across modes (SSZ signature)."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.01
        
        results = compute_all_modes(config)
        
        shifts = [results[n]["relative_shift"] for n in [1, 2, 3]]
        
        # All shifts should be equal (SSZ signature!)
        assert shifts[0] == pytest.approx(shifts[1], rel=1e-10)
        assert shifts[1] == pytest.approx(shifts[2], rel=1e-10)


class TestPhiBasedSegmentation:
    """Tests for phi-based segmentation functions."""
    
    def test_phi_segment_density_ssz_core(self):
        """Test SSZ core segment density formula."""
        # Xi(r) = 1 - exp(-phi * r / r_s)
        r = 1.0
        r_s = 1.0
        
        xi = phi_segment_density(r, r_s, model="ssz_core")
        expected = 1 - np.exp(-PHI)
        
        assert xi == pytest.approx(expected)
    
    def test_phi_segment_density_linear(self):
        """Test linear segment density model."""
        r = 1.0
        r_s = 1.0
        
        xi = phi_segment_density(r, r_s, model="linear")
        expected = PHI
        
        assert xi == pytest.approx(expected)
    
    def test_sigma_from_phi_ratio_no_difference(self):
        """Test sigma when phi values are equal."""
        sigma = sigma_from_phi_ratio(1.0, 1.0, lambda_coupling=1.0)
        assert sigma == pytest.approx(0.0)
    
    def test_sigma_from_phi_ratio_positive(self):
        """Test sigma for increased phi."""
        sigma = sigma_from_phi_ratio(1.1, 1.0, lambda_coupling=1.0)
        # sigma = 1.0 * (1.1/1.0 - 1) = 0.1
        assert sigma == pytest.approx(0.1)
    
    def test_create_phi_based_config(self):
        """Test creating config from phi-based densities."""
        config = create_phi_based_config(
            lambda_atm=0.01,
            lambda_iono=0.01
        )
        
        assert isinstance(config, LayeredSSZConfig)
        # Sigma values should be set based on phi ratios
        assert config.atmosphere.sigma is not None
        assert config.ionosphere.sigma is not None


class TestTimeVaryingModel:
    """Tests for time-varying SSZ model."""
    
    def test_sigma_iono_from_proxy_constant(self):
        """Test sigma from proxy with constant input."""
        F_iono = np.ones(100)
        sigma = sigma_iono_from_proxy(F_iono, beta_0=0.005, beta_1=0.01)
        
        # sigma = 0.005 + 0.01 * 1 = 0.015
        assert np.all(sigma == pytest.approx(0.015))
    
    def test_sigma_iono_from_proxy_varying(self):
        """Test sigma from proxy with varying input."""
        F_iono = np.array([0.0, 0.5, 1.0])
        sigma = sigma_iono_from_proxy(F_iono, beta_0=0.0, beta_1=0.01)
        
        expected = np.array([0.0, 0.005, 0.01])
        np.testing.assert_array_almost_equal(sigma, expected)
    
    def test_f_n_ssz_timeseries(self):
        """Test frequency time series calculation."""
        sigma_iono_t = np.array([0.0, 0.01, 0.02])
        
        f1_t = f_n_ssz_timeseries(
            n=1,
            sigma_iono_t=sigma_iono_t,
            f1_ref=7.83
        )
        
        assert len(f1_t) == 3
        # Higher sigma -> lower frequency
        assert f1_t[0] > f1_t[1] > f1_t[2]
    
    def test_f_n_ssz_timeseries_pandas(self):
        """Test with pandas Series input."""
        sigma_iono_t = pd.Series([0.0, 0.01, 0.02])
        
        f1_t = f_n_ssz_timeseries(
            n=1,
            sigma_iono_t=sigma_iono_t,
            f1_ref=7.83
        )
        
        assert len(f1_t) == 3


class TestFrequencyShiftEstimate:
    """Tests for frequency shift estimation."""
    
    def test_zero_segmentation(self):
        """Test with zero segmentation."""
        result = frequency_shift_estimate(0.0)
        
        assert result["D_SSZ"] == 1.0
        assert result["delta_f1"] == 0.0
        assert result["f1_ssz"] == result["f1_classical"]
    
    def test_one_percent_segmentation(self):
        """Test with 1% segmentation."""
        result = frequency_shift_estimate(0.01, f_ref=7.83)
        
        # D_SSZ = 1.01
        assert result["D_SSZ"] == pytest.approx(1.01)
        
        # f1_ssz = 7.83 / 1.01 ~ 7.75
        assert result["f1_ssz"] == pytest.approx(7.83 / 1.01)
        
        # delta_f1 ~ -0.08 Hz
        assert result["delta_f1"] == pytest.approx(-0.0775, rel=0.01)
    
    def test_shift_proportional_to_frequency(self):
        """Test that absolute shift is proportional to frequency."""
        result = frequency_shift_estimate(0.01)
        
        # |delta_f2| / |delta_f1| should equal f2 / f1
        ratio_shift = abs(result["delta_f2"]) / abs(result["delta_f1"])
        ratio_freq = result["f2_classical"] / result["f1_classical"]
        
        assert ratio_shift == pytest.approx(ratio_freq, rel=1e-10)


class TestPhysicalConsistency:
    """Tests for physical consistency of the model."""
    
    def test_positive_segmentation_lowers_frequency(self):
        """Test that positive sigma lowers frequency."""
        config = LayeredSSZConfig()
        f1_no_seg = f_n_ssz_layered(1, config)
        
        config.ionosphere.sigma = 0.01
        f1_with_seg = f_n_ssz_layered(1, config)
        
        assert f1_with_seg < f1_no_seg
    
    def test_negative_segmentation_raises_frequency(self):
        """Test that negative sigma raises frequency."""
        config = LayeredSSZConfig()
        f1_no_seg = f_n_ssz_layered(1, config)
        
        config.ionosphere.sigma = -0.01
        f1_with_seg = f_n_ssz_layered(1, config)
        
        assert f1_with_seg > f1_no_seg
    
    def test_frequency_ratios_preserved(self):
        """Test that mode frequency ratios are preserved by SSZ."""
        config = LayeredSSZConfig()
        config.ionosphere.sigma = 0.02
        
        results = compute_all_modes(config)
        
        # Classical ratios
        ratio_21_class = results[2]["f_classical"] / results[1]["f_classical"]
        ratio_31_class = results[3]["f_classical"] / results[1]["f_classical"]
        
        # SSZ ratios
        ratio_21_ssz = results[2]["f_ssz"] / results[1]["f_ssz"]
        ratio_31_ssz = results[3]["f_ssz"] / results[1]["f_ssz"]
        
        # Ratios should be preserved (SSZ is a uniform scaling)
        assert ratio_21_ssz == pytest.approx(ratio_21_class, rel=1e-10)
        assert ratio_31_ssz == pytest.approx(ratio_31_class, rel=1e-10)
    
    def test_realistic_shift_magnitude(self):
        """Test that 1% segmentation gives realistic shift."""
        result = frequency_shift_estimate(0.01, f_ref=7.83)
        
        # Shift should be ~0.08 Hz (within observed variation range)
        assert abs(result["delta_f1"]) < 0.1
        assert abs(result["delta_f1"]) > 0.05
        
        # Relative shift should be ~0.8-1%
        rel_shift = abs(result["relative_shift_1"])
        assert 0.008 < rel_shift < 0.012


class TestCoreSSZFormulas:
    """Tests for core SSZ formulas from ssz-metric-pure."""
    
    def test_Xi_ssz_at_zero(self):
        """Test Xi(0) = 0."""
        from ssz_schumann.models.layered_ssz import Xi_ssz
        
        xi = Xi_ssz(0, r_s=1.0)
        assert xi == pytest.approx(0.0)
    
    def test_Xi_ssz_at_infinity(self):
        """Test Xi(infinity) -> Xi_max."""
        from ssz_schumann.models.layered_ssz import Xi_ssz
        
        xi = Xi_ssz(1e10, r_s=1.0, Xi_max=1.0)
        assert xi == pytest.approx(1.0, rel=1e-6)
    
    def test_Xi_ssz_at_r_s(self):
        """Test Xi(r_s) = Xi_max * (1 - exp(-phi))."""
        from ssz_schumann.models.layered_ssz import Xi_ssz
        
        xi = Xi_ssz(1.0, r_s=1.0, Xi_max=1.0)
        expected = 1.0 - np.exp(-PHI)
        assert xi == pytest.approx(expected)
    
    def test_Xi_ssz_array(self):
        """Test Xi with array input."""
        from ssz_schumann.models.layered_ssz import Xi_ssz
        
        r = np.array([0, 1, 2, 10])
        xi = Xi_ssz(r, r_s=1.0)
        
        assert len(xi) == 4
        assert xi[0] == pytest.approx(0.0)
        assert xi[-1] > xi[0]  # Monotonically increasing
    
    def test_D_SSZ_from_Xi_at_zero(self):
        """Test D_SSZ(Xi=0) = 1."""
        from ssz_schumann.models.layered_ssz import D_SSZ_from_Xi
        
        d = D_SSZ_from_Xi(0.0)
        assert d == pytest.approx(1.0)
    
    def test_D_SSZ_from_Xi_at_one(self):
        """Test D_SSZ(Xi=1) = 0.5."""
        from ssz_schumann.models.layered_ssz import D_SSZ_from_Xi
        
        d = D_SSZ_from_Xi(1.0)
        assert d == pytest.approx(0.5)
    
    def test_D_SSZ_from_Xi_range(self):
        """Test D_SSZ is always in (0, 1]."""
        from ssz_schumann.models.layered_ssz import D_SSZ_from_Xi
        
        xi_values = np.linspace(0, 10, 100)
        d_values = D_SSZ_from_Xi(xi_values)
        
        assert np.all(d_values > 0)
        assert np.all(d_values <= 1)
    
    def test_D_SSZ_no_singularity(self):
        """Test that D_SSZ never reaches zero (no singularity)."""
        from ssz_schumann.models.layered_ssz import Xi_ssz, D_SSZ_from_Xi
        
        # Even at very large Xi, D_SSZ > 0
        xi_large = Xi_ssz(1e20, r_s=1.0, Xi_max=1.0)
        d = D_SSZ_from_Xi(xi_large)
        
        assert d > 0
        assert d == pytest.approx(0.5, rel=1e-6)  # Saturates at 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
