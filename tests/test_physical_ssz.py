#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Physical SSZ Model

Tests the physical SSZ model that connects ionospheric properties
to the SSZ segmentation parameter.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import pytest
import numpy as np

from ssz_schumann.models.physical_ssz import (
    IonosphereState,
    SSZPhysicalParams,
    plasma_frequency,
    gyro_frequency,
    skin_depth,
    ionosphere_conductivity,
    delta_seg_physical,
    delta_seg_from_proxies,
    f_n_ssz_physical,
    predict_ssz_signature,
    N_E_REF,
    B_REF,
    H_IONO_REF,
)


class TestPlasmaParameters:
    """Test plasma physics calculations."""
    
    def test_plasma_frequency_typical(self):
        """Test plasma frequency for typical D-layer density."""
        f_p = plasma_frequency(1e11)  # 10^11 m^-3
        
        # Should be in MHz range
        assert 1e6 < f_p < 10e6, f"Plasma frequency {f_p} Hz out of range"
        
        # Known value: f_p ~ 2.84 MHz for n_e = 10^11 m^-3
        assert abs(f_p - 2.84e6) < 0.1e6, f"Expected ~2.84 MHz, got {f_p/1e6:.2f} MHz"
    
    def test_plasma_frequency_scaling(self):
        """Test that plasma frequency scales as sqrt(n_e)."""
        f_p_1 = plasma_frequency(1e11)
        f_p_4 = plasma_frequency(4e11)
        
        # f_p should double when n_e quadruples
        ratio = f_p_4 / f_p_1
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio 2.0, got {ratio}"
    
    def test_gyro_frequency_typical(self):
        """Test gyro frequency for typical Earth field."""
        f_g = gyro_frequency(5e-5)  # 50 uT
        
        # Should be in MHz range
        assert 1e6 < f_g < 2e6, f"Gyro frequency {f_g} Hz out of range"
        
        # Known value: f_g ~ 1.4 MHz for B = 50 uT
        assert abs(f_g - 1.4e6) < 0.1e6, f"Expected ~1.4 MHz, got {f_g/1e6:.2f} MHz"
    
    def test_gyro_frequency_linear(self):
        """Test that gyro frequency scales linearly with B."""
        f_g_1 = gyro_frequency(5e-5)
        f_g_2 = gyro_frequency(10e-5)
        
        ratio = f_g_2 / f_g_1
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio 2.0, got {ratio}"


class TestIonosphereState:
    """Test IonosphereState dataclass."""
    
    def test_create_state(self):
        """Test creating an ionosphere state."""
        state = IonosphereState(
            n_e=1e11,
            T_e=1000,
            h_iono=85,
            B_field=5e-5,
            collision_freq=1e3,
        )
        
        assert state.n_e == 1e11
        assert state.h_iono == 85
        assert state.B_field == 5e-5
    
    def test_reference_state(self):
        """Test reference ionosphere state."""
        state = IonosphereState(
            n_e=N_E_REF,
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=B_REF,
            collision_freq=1e3,
        )
        
        # At reference values, delta_seg should be ~0
        delta = delta_seg_physical(state)
        assert abs(delta) < 1e-10, f"Expected delta_seg ~0 at reference, got {delta}"


class TestDeltaSegPhysical:
    """Test physical delta_seg calculation."""
    
    def test_reference_gives_zero(self):
        """At reference values, delta_seg should be zero."""
        state = IonosphereState(
            n_e=N_E_REF,
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=B_REF,
            collision_freq=1e3,
        )
        
        delta = delta_seg_physical(state)
        assert abs(delta) < 1e-10
    
    def test_increased_density(self):
        """Higher electron density should change delta_seg."""
        state = IonosphereState(
            n_e=2 * N_E_REF,  # Double density
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=B_REF,
            collision_freq=1e3,
        )
        
        params = SSZPhysicalParams(alpha=1e-12, beta=0, gamma=0)
        delta = delta_seg_physical(state, params)
        
        # Should be positive (alpha * 1.0)
        expected = 1e-12 * 1.0  # (2*n_e_ref / n_e_ref - 1) = 1
        assert abs(delta - expected) < 1e-15
    
    def test_increased_b_field(self):
        """Higher B field should change delta_seg."""
        state = IonosphereState(
            n_e=N_E_REF,
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=2 * B_REF,  # Double B field
            collision_freq=1e3,
        )
        
        params = SSZPhysicalParams(alpha=0, beta=0.1, gamma=0)
        delta = delta_seg_physical(state, params)
        
        # Should be 0.1 * 1.0 = 0.1
        expected = 0.1 * 1.0
        assert abs(delta - expected) < 1e-10


class TestDeltaSegFromProxies:
    """Test delta_seg estimation from space weather proxies."""
    
    def test_typical_quiet_sun(self):
        """Test quiet Sun conditions (F10.7 ~ 70)."""
        delta = delta_seg_from_proxies(f107=70, kp=1)
        
        # Should be small
        assert abs(delta) < 0.1, f"Expected small delta_seg, got {delta}"
    
    def test_active_sun(self):
        """Test active Sun conditions (F10.7 ~ 200)."""
        delta_quiet = delta_seg_from_proxies(f107=70, kp=1)
        delta_active = delta_seg_from_proxies(f107=200, kp=1)
        
        # Active should be different from quiet
        # (direction depends on alpha sign)
        assert delta_quiet != delta_active
    
    def test_geomagnetic_storm(self):
        """Test geomagnetic storm conditions (Kp ~ 7)."""
        delta_quiet = delta_seg_from_proxies(f107=100, kp=1)
        delta_storm = delta_seg_from_proxies(f107=100, kp=7)
        
        # Storm should be different from quiet
        assert delta_quiet != delta_storm
    
    def test_height_variation(self):
        """Test ionosphere height variation."""
        delta_low = delta_seg_from_proxies(f107=100, kp=3, h_iono=70)
        delta_high = delta_seg_from_proxies(f107=100, kp=3, h_iono=100)
        
        # Different heights should give different delta_seg
        assert delta_low != delta_high


class TestSSZFrequency:
    """Test SSZ-corrected frequency calculation."""
    
    def test_reference_state_matches_classical(self):
        """At reference state, SSZ frequency should match classical."""
        state = IonosphereState(
            n_e=N_E_REF,
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=B_REF,
            collision_freq=1e3,
        )
        
        from ssz_schumann.models.classical_schumann import f_n_classical
        
        f_ssz = f_n_ssz_physical(1, state, eta_0=0.74)
        f_class = f_n_classical(1, eta=0.74)
        
        # Should be very close
        assert abs(f_ssz - f_class) < 1e-6, \
            f"Expected f_ssz ~ f_class, got {f_ssz} vs {f_class}"
    
    def test_mode_independence(self):
        """Test that relative shift is mode-independent."""
        state = IonosphereState(
            n_e=2 * N_E_REF,
            T_e=1000,
            h_iono=H_IONO_REF,
            B_field=B_REF,
            collision_freq=1e3,
        )
        
        from ssz_schumann.models.classical_schumann import f_n_classical
        
        # Calculate relative shifts for modes 1, 2, 3
        shifts = []
        for n in [1, 2, 3]:
            f_ssz = f_n_ssz_physical(n, state, eta_0=0.74)
            f_class = f_n_classical(n, eta=0.74)
            rel_shift = (f_ssz - f_class) / f_class
            shifts.append(rel_shift)
        
        # All relative shifts should be equal (mode-independent)
        assert abs(shifts[0] - shifts[1]) < 1e-10, \
            f"Mode 1 and 2 shifts differ: {shifts[0]} vs {shifts[1]}"
        assert abs(shifts[1] - shifts[2]) < 1e-10, \
            f"Mode 2 and 3 shifts differ: {shifts[1]} vs {shifts[2]}"


class TestPredictions:
    """Test SSZ signature predictions."""
    
    def test_predict_signature_returns_dict(self):
        """Test that predict_ssz_signature returns expected keys."""
        predictions = predict_ssz_signature()
        
        assert "f107_vals" in predictions
        assert "kp_vals" in predictions
        assert "delta_seg_grid" in predictions
        assert "delta_seg_min" in predictions
        assert "delta_seg_max" in predictions
    
    def test_grid_shape(self):
        """Test that prediction grid has correct shape."""
        predictions = predict_ssz_signature(
            f107_range=(70, 200),
            kp_range=(0, 9),
        )
        
        grid = predictions["delta_seg_grid"]
        f107_vals = predictions["f107_vals"]
        kp_vals = predictions["kp_vals"]
        
        assert grid.shape == (len(f107_vals), len(kp_vals))
    
    def test_range_is_finite(self):
        """Test that delta_seg range is finite and reasonable."""
        predictions = predict_ssz_signature()
        
        assert np.isfinite(predictions["delta_seg_min"])
        assert np.isfinite(predictions["delta_seg_max"])
        assert predictions["delta_seg_max"] >= predictions["delta_seg_min"]


class TestPhysicalParams:
    """Test SSZPhysicalParams dataclass."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = SSZPhysicalParams()
        
        assert params.alpha == 1e-12
        assert params.beta == 0.1
        assert params.gamma == 0.01
        assert params.n_e_ref == N_E_REF
    
    def test_custom_params(self):
        """Test custom parameter values."""
        params = SSZPhysicalParams(
            alpha=1e-10,
            beta=0.5,
            gamma=0.05,
        )
        
        assert params.alpha == 1e-10
        assert params.beta == 0.5
        assert params.gamma == 0.05
