#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools Test Suite

Tests for all hamtools modules:
- core: Frequency, wavelength, dB calculations
- antennas: Dipole, vertical, Yagi calculations
- feedline: Cable attenuation
- propagation: MUF, skip distance
- ssz_extension: SSZ corrections

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import pytest
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hamtools import core, antennas, feedline, propagation, field_strength, ssz_extension


# =============================================================================
# CORE MODULE TESTS
# =============================================================================

class TestCoreFrequency:
    """Tests for frequency/wavelength conversions."""
    
    def test_freq_to_lambda_7mhz(self):
        """Test 7 MHz → ~42.8m wavelength."""
        lambda_m = core.freq_to_lambda(7e6)
        assert 42 < lambda_m < 43
    
    def test_freq_to_lambda_14mhz(self):
        """Test 14 MHz → ~21.4m wavelength."""
        lambda_m = core.freq_to_lambda(14e6)
        assert 21 < lambda_m < 22
    
    def test_lambda_to_freq_roundtrip(self):
        """Test frequency → wavelength → frequency roundtrip."""
        f_original = 7.1e6
        lambda_m = core.freq_to_lambda(f_original)
        f_back = core.lambda_to_freq(lambda_m)
        assert abs(f_back - f_original) < 1  # Within 1 Hz
    
    def test_freq_mhz_to_lambda(self):
        """Test MHz convenience function."""
        lambda_m = core.freq_mhz_to_lambda(7.1)
        assert 42 < lambda_m < 43
    
    def test_freq_khz_to_lambda(self):
        """Test kHz convenience function."""
        lambda_m = core.freq_khz_to_lambda(7100)
        assert 42 < lambda_m < 43
    
    def test_period_roundtrip(self):
        """Test frequency → period → frequency roundtrip."""
        f_original = 1000.0
        T = core.freq_to_period(f_original)
        f_back = core.period_to_freq(T)
        assert abs(f_back - f_original) < 1e-10
    
    def test_negative_frequency_raises(self):
        """Test that negative frequency raises error."""
        with pytest.raises(ValueError):
            core.freq_to_lambda(-1e6)


class TestCoreDB:
    """Tests for dB calculations."""
    
    def test_db_from_ratio_double(self):
        """Test 2× power = ~3 dB."""
        db = core.db_from_ratio(2)
        assert abs(db - 3.0103) < 0.001
    
    def test_db_from_ratio_10x(self):
        """Test 10× power = 10 dB."""
        db = core.db_from_ratio(10)
        assert abs(db - 10.0) < 0.001
    
    def test_ratio_from_db_3db(self):
        """Test 3 dB ≈ 2× power."""
        ratio = core.ratio_from_db(3)
        assert abs(ratio - 2.0) < 0.01
    
    def test_db_roundtrip(self):
        """Test ratio → dB → ratio roundtrip."""
        ratio_original = 5.0
        db = core.db_from_ratio(ratio_original)
        ratio_back = core.ratio_from_db(db)
        assert abs(ratio_back - ratio_original) < 1e-10
    
    def test_voltage_db(self):
        """Test voltage dB (factor 20)."""
        db = core.db_from_voltage(2, 1)
        assert abs(db - 6.0206) < 0.001  # 2× voltage = 6 dB


class TestCoreERP:
    """Tests for ERP/EIRP calculations."""
    
    def test_erp_no_gain_no_loss(self):
        """Test ERP with 0 dB gain and 0 dB loss."""
        erp = core.erp_watt(100, 0, 0)
        assert abs(erp - 100) < 0.1
    
    def test_erp_with_gain(self):
        """Test ERP with 3 dBd gain."""
        erp = core.erp_watt(100, 3, 0)
        assert 195 < erp < 205  # ~2× power
    
    def test_erp_with_loss(self):
        """Test ERP with 3 dB loss."""
        erp = core.erp_watt(100, 0, 3)
        assert 49 < erp < 51  # ~0.5× power
    
    def test_dbd_to_dbi(self):
        """Test dBd to dBi conversion."""
        dbi = core.dbd_to_dbi(0)
        assert abs(dbi - 2.15) < 0.01


# =============================================================================
# ANTENNA MODULE TESTS
# =============================================================================

class TestAntennas:
    """Tests for antenna calculations."""
    
    def test_dipole_40m(self):
        """Test 40m dipole length."""
        length = antennas.dipole_length_halfwave(7.1)
        # Approximate: 300/f * 0.5 * 0.95 ≈ 20m
        assert 19 < length < 21
    
    def test_dipole_20m(self):
        """Test 20m dipole length."""
        length = antennas.dipole_length_halfwave(14.2)
        assert 9 < length < 11
    
    def test_vertical_40m(self):
        """Test 40m vertical length."""
        length = antennas.vertical_quarterwave(7.1)
        # Should be half of dipole
        dipole = antennas.dipole_length_halfwave(7.1)
        assert abs(length - dipole/2) < 0.1
    
    def test_yagi_gain_positive(self):
        """Test Yagi gain is positive."""
        gain = antennas.estimate_yagi_gain(5, 6.5)
        assert gain > 0
    
    def test_yagi_gain_increases_with_elements(self):
        """Test Yagi gain increases with more elements."""
        gain_3 = antennas.estimate_yagi_gain(3, 3.0)
        gain_5 = antennas.estimate_yagi_gain(5, 6.0)
        gain_7 = antennas.estimate_yagi_gain(7, 9.0)
        assert gain_3 < gain_5 < gain_7
    
    def test_shortening_factor_effect(self):
        """Test shortening factor reduces length."""
        length_95 = antennas.dipole_length_halfwave(7.1, k=0.95)
        length_100 = antennas.dipole_length_halfwave(7.1, k=1.0)
        assert length_95 < length_100


# =============================================================================
# FEEDLINE MODULE TESTS
# =============================================================================

class TestFeedline:
    """Tests for feedline calculations."""
    
    def test_rg58_higher_loss_than_ecoflex(self):
        """Test RG-58 has higher loss than Ecoflex-10."""
        loss_rg58 = feedline.attenuation_db_per_100m(14.2, "RG-58")
        loss_ecoflex = feedline.attenuation_db_per_100m(14.2, "ECOFLEX-10")
        assert loss_rg58 > loss_ecoflex
    
    def test_loss_increases_with_frequency(self):
        """Test loss increases with frequency."""
        loss_7 = feedline.attenuation_db_per_100m(7, "RG-58")
        loss_14 = feedline.attenuation_db_per_100m(14, "RG-58")
        loss_28 = feedline.attenuation_db_per_100m(28, "RG-58")
        assert loss_7 < loss_14 < loss_28
    
    def test_total_loss_proportional_to_length(self):
        """Test total loss is proportional to length."""
        loss_30m = feedline.total_loss_db(14.2, "RG-58", 30)
        loss_60m = feedline.total_loss_db(14.2, "RG-58", 60)
        assert abs(loss_60m - 2 * loss_30m) < 0.01
    
    def test_power_at_antenna(self):
        """Test power at antenna calculation."""
        p_out, loss = feedline.power_at_antenna(100, 14.2, "RG-58", 30)
        assert p_out < 100  # Power should decrease
        assert loss > 0     # Loss should be positive
    
    def test_unknown_cable_raises(self):
        """Test unknown cable type raises error."""
        with pytest.raises(ValueError):
            feedline.attenuation_db_per_100m(14.2, "UNKNOWN-CABLE")


# =============================================================================
# PROPAGATION MODULE TESTS
# =============================================================================

class TestPropagation:
    """Tests for propagation calculations."""
    
    def test_critical_freq_formula(self):
        """Test critical frequency formula."""
        # f = 9 * sqrt(N) Hz
        # For N = 1e12: f = 9e6 Hz = 9 MHz
        fof2 = propagation.critical_freq_fof2(1e12)
        assert abs(fof2 - 9.0) < 0.1
    
    def test_muf_increases_with_distance(self):
        """Test MUF increases with path distance."""
        muf_1000 = propagation.muf_single_hop(5.0, 1000, 300)
        muf_2000 = propagation.muf_single_hop(5.0, 2000, 300)
        muf_3000 = propagation.muf_single_hop(5.0, 3000, 300)
        assert muf_1000 < muf_2000 < muf_3000
    
    def test_muf_at_zero_distance(self):
        """Test MUF at very short distance equals foF2."""
        muf = propagation.muf_single_hop(5.0, 50, 300)
        assert abs(muf - 5.0) < 0.1
    
    def test_skip_distance_below_critical(self):
        """Test skip distance is 0 below critical frequency."""
        skip = propagation.skip_distance_km(3.0, 5.0, 300)
        assert skip == 0.0
    
    def test_skip_distance_above_critical(self):
        """Test skip distance is positive above critical frequency."""
        skip = propagation.skip_distance_km(14.0, 5.0, 300)
        assert skip > 0


# =============================================================================
# SSZ EXTENSION TESTS
# =============================================================================

class TestSSZExtension:
    """Tests for SSZ extension module."""
    
    def test_d_ssz_from_delta(self):
        """Test D_SSZ = 1 + δ_seg."""
        D = ssz_extension.d_ssz_from_delta(0.01)
        assert abs(D - 1.01) < 1e-10
    
    def test_effective_c_reduced(self):
        """Test c_eff < c for positive δ_seg."""
        c_eff = ssz_extension.effective_c_from_ssz(core.C_LIGHT, 0.01)
        assert c_eff < core.C_LIGHT
    
    def test_ssz_lambda_shorter(self):
        """Test SSZ wavelength is shorter than classical."""
        lambda_classical = core.freq_to_lambda(7.1e6)
        lambda_ssz = ssz_extension.ssz_corrected_lambda(7.1e6, 0.01)
        assert lambda_ssz < lambda_classical
    
    def test_ssz_effect_proportional(self):
        """Test ~1% δ_seg gives ~1% wavelength change."""
        result = ssz_extension.compare_lambda_with_ssz(7.1, 0.01)
        # Difference should be approximately -1%
        assert -1.1 < result.difference_percent < -0.9
    
    def test_ssz_effect_scales(self):
        """Test SSZ effect scales with δ_seg."""
        result_1pct = ssz_extension.compare_lambda_with_ssz(7.1, 0.01)
        result_2pct = ssz_extension.compare_lambda_with_ssz(7.1, 0.02)
        # 2% should give roughly 2× the effect
        assert abs(result_2pct.difference_percent) > abs(result_1pct.difference_percent)
    
    def test_zero_delta_no_effect(self):
        """Test δ_seg = 0 gives no SSZ effect."""
        result = ssz_extension.compare_lambda_with_ssz(7.1, 0.0)
        assert abs(result.difference_percent) < 1e-10
    
    def test_ssz_skip_distance(self):
        """Test SSZ skip distance modification."""
        result = ssz_extension.compare_skip_with_ssz(1000, 0.01)
        # 1% δ_seg should give ~1% increase
        assert 0.9 < result.difference_percent < 1.1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests across modules."""
    
    def test_antenna_uses_correct_wavelength(self):
        """Test antenna calculation uses correct wavelength."""
        f_mhz = 7.1
        lambda_m = core.freq_mhz_to_lambda(f_mhz)
        dipole = antennas.dipole_length_halfwave(f_mhz, k=1.0)
        # Dipole should be λ/2
        assert abs(dipole - lambda_m/2) < 0.01
    
    def test_ssz_antenna_correction(self):
        """Test SSZ antenna length correction."""
        f_mhz = 7.1
        delta_seg = 0.02
        
        # Classical dipole
        dipole_classical = antennas.dipole_length_halfwave(f_mhz)
        
        # SSZ-corrected wavelength
        lambda_ssz = ssz_extension.ssz_corrected_lambda(f_mhz * 1e6, delta_seg)
        dipole_ssz = (lambda_ssz / 2) * 0.95
        
        # SSZ dipole should be shorter
        assert dipole_ssz < dipole_classical
        
        # Difference should be ~2%
        diff_pct = (dipole_ssz - dipole_classical) / dipole_classical * 100
        assert -2.5 < diff_pct < -1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
