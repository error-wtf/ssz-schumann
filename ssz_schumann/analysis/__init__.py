#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Module for SSZ Schumann Experiment

Provides:
- Delta computation pipeline
- Correlation analysis
- Visualization functions
- Regression model comparison

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

from .compute_deltas import (
    compute_all_deltas,
    run_analysis_pipeline,
)

from .correlation_plots import (
    plot_timeseries,
    plot_scatter_delta_vs_feature,
    plot_mode_consistency,
    plot_diurnal_pattern,
    plot_seasonal_pattern,
    create_summary_figure,
)

from .regression_models import (
    cross_validate_models,
    bootstrap_confidence_intervals,
    hypothesis_test_ssz_signature,
)

from .model_comparison import (
    ModelResult,
    ComparisonResult,
    log_likelihood_gaussian,
    compute_aic,
    compute_bic,
    compute_bayes_factor,
    interpret_bayes_factor,
    compare_models,
    cross_validate,
    print_comparison_summary,
)

from .model_fits import (
    calibrate_eta_from_data,
    compute_classical_frequencies,
    compute_delta_seg,
    compute_mode_consistency,
    fit_global_ssz_model,
    fit_layered_ssz_model,
    compute_proxy_correlations,
    generate_interpretation,
    FitResult,
)

# T2: Unified pipeline
from .pipeline import (
    PipelineConfig,
    PipelineResult,
    run_full_pipeline,
    run_quick_analysis,
)

# T4: SSZ diagnostics
from .ssz_diagnostics import (
    compute_relative_shifts,
    check_mode_independence,
    compute_delta_seg_with_confidence,
    detect_dispersion_pattern,
    generate_diagnostic_report,
    RelativeShiftResult,
    ModeIndependenceResult,
)

__all__ = [
    # Compute
    "compute_all_deltas",
    "run_analysis_pipeline",
    # Plots
    "plot_timeseries",
    "plot_scatter_delta_vs_feature",
    "plot_mode_consistency",
    "plot_diurnal_pattern",
    "plot_seasonal_pattern",
    "create_summary_figure",
    # Regression
    "cross_validate_models",
    "bootstrap_confidence_intervals",
    "hypothesis_test_ssz_signature",
    # Model fits
    "calibrate_eta_from_data",
    "compute_classical_frequencies",
    "compute_delta_seg",
    "compute_mode_consistency",
    "fit_global_ssz_model",
    "fit_layered_ssz_model",
    "compute_proxy_correlations",
    "generate_interpretation",
    "FitResult",
    # Model comparison
    "ModelResult",
    "ComparisonResult",
    "log_likelihood_gaussian",
    "compute_aic",
    "compute_bic",
    "compute_bayes_factor",
    "interpret_bayes_factor",
    "compare_models",
    "cross_validate",
    "print_comparison_summary",
    # T2: Pipeline
    "PipelineConfig",
    "PipelineResult",
    "run_full_pipeline",
    "run_quick_analysis",
    # T4: Diagnostics
    "compute_relative_shifts",
    "check_mode_independence",
    "compute_delta_seg_with_confidence",
    "detect_dispersion_pattern",
    "generate_diagnostic_report",
    "RelativeShiftResult",
    "ModeIndependenceResult",
]
