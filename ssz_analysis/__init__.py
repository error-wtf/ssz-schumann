# -*- coding: utf-8 -*-
"""
SSZ Schumann Analysis Module

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

from .core import (
    load_schumann_data,
    compute_classical_frequencies,
    fit_classical_eta,
    compute_delta_seg_classical,
    compute_delta_seg_anomaly,
    estimate_delta_ssz_global,
    compute_ssz_chi_squared,
    analyze_correlations,
    compute_ssz_upper_bound,
)

__version__ = "1.0.0"
__all__ = [
    "load_schumann_data",
    "compute_classical_frequencies",
    "fit_classical_eta",
    "compute_delta_seg_classical",
    "compute_delta_seg_anomaly",
    "estimate_delta_ssz_global",
    "compute_ssz_chi_squared",
    "analyze_correlations",
    "compute_ssz_upper_bound",
]
