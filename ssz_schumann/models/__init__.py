#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models Module for SSZ Schumann Analysis

Contains:
- Classical Schumann resonance model
- SSZ correction factor computation
- Fitting wrappers for model comparison

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

from .classical_schumann import (
    f_n_classical,
    f_n_classical_timeseries,
    compute_eta0_from_mean_f1,
    schumann_mode_factor,
)

from .ssz_correction import (
    D_SSZ,
    delta_seg_from_observed,
    fit_delta_seg_simple,
    check_mode_consistency,
    f_n_ssz_model,
)

from .fit_wrappers import (
    fit_classical_model,
    fit_ssz_model,
    compare_models,
    ModelResult,
)

from .maxwell_schumann import (
    f_n_ideal,
    f_n_damped,
    f_n_extended,
    compute_eta_from_observed,
    compute_Q_factor,
    get_schumann_mode,
    frequency_shift_from_height_change,
    frequency_shift_from_ssz,
    relative_shift_ssz,
    compute_mode_ratios,
    SchumannMode,
    OBSERVED_FREQUENCIES,
)

from .physical_ssz import (
    IonosphereState,
    SSZPhysicalParams,
    plasma_frequency,
    gyro_frequency,
    delta_seg_physical,
    delta_seg_from_proxies,
    f_n_ssz_physical,
    fit_ssz_physical_params,
    predict_ssz_signature,
)

from .layered_ssz import (
    LayerConfig,
    LayeredSSZConfig,
    D_SSZ_layered,
    D_SSZ_from_sigmas,
    f_n_ssz_layered,
    compute_all_modes,
    sigma_from_phi_ratio,
    phi_segment_density,
    create_phi_based_config,
    sigma_iono_from_proxy,
    f_n_ssz_timeseries,
    fit_layered_ssz,
    frequency_shift_estimate,
    print_frequency_table,
    # Core SSZ formulas (from ssz-metric-pure)
    Xi_ssz,
    D_SSZ_from_Xi,
)

__all__ = [
    # Classical
    "f_n_classical",
    "f_n_classical_timeseries",
    "compute_eta0_from_mean_f1",
    "schumann_mode_factor",
    # SSZ
    "D_SSZ",
    "delta_seg_from_observed",
    "fit_delta_seg_simple",
    "check_mode_consistency",
    "f_n_ssz_model",
    # Fitting
    "fit_classical_model",
    "fit_ssz_model",
    "compare_models",
    "ModelResult",
    # Layered SSZ
    "LayerConfig",
    "LayeredSSZConfig",
    "D_SSZ_layered",
    "D_SSZ_from_sigmas",
    "f_n_ssz_layered",
    "compute_all_modes",
    "sigma_from_phi_ratio",
    "phi_segment_density",
    "create_phi_based_config",
    "sigma_iono_from_proxy",
    "f_n_ssz_timeseries",
    "fit_layered_ssz",
    "frequency_shift_estimate",
    "print_frequency_table",
    # Core SSZ formulas
    "Xi_ssz",
    "D_SSZ_from_Xi",
    # Maxwell-Schumann
    "f_n_ideal",
    "f_n_damped",
    "f_n_extended",
    "compute_eta_from_observed",
    "compute_Q_factor",
    "get_schumann_mode",
    "frequency_shift_from_height_change",
    "frequency_shift_from_ssz",
    "relative_shift_ssz",
    "compute_mode_ratios",
    "SchumannMode",
    "OBSERVED_FREQUENCIES",
    # Physical SSZ
    "IonosphereState",
    "SSZPhysicalParams",
    "plasma_frequency",
    "gyro_frequency",
    "delta_seg_physical",
    "delta_seg_from_proxies",
    "f_n_ssz_physical",
    "fit_ssz_physical_params",
    "predict_ssz_signature",
]
