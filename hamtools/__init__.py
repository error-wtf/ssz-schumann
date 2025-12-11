#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HamTools - Amateur Radio Calculator with SSZ Extension

A practical toolkit for ham radio operators with an optional
Segmented Spacetime (SSZ) expert mode.

Modules:
    core        - Frequency, wavelength, period, dB calculations
    antennas    - Dipole, vertical, Yagi antenna calculations
    feedline    - Cable attenuation and loss calculations
    propagation - MUF, critical frequency, skip distance
    field_strength - Far-field E-field estimation
    ssz_extension - SSZ corrections for expert mode

Usage:
    # Python API
    from hamtools import core, antennas, feedline
    
    wavelength = core.freq_mhz_to_lambda(7.1)
    dipole_len = antennas.dipole_length_halfwave(7.1)
    loss = feedline.total_loss_db(14.2, 'RG-58', 30)
    
    # CLI
    $ hamtool freq --mhz 7.1
    $ hamtool antenna dipole --mhz 7.1
    $ hamtool feedline loss --mhz 14.2 --cable RG-58 --length 30

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

__version__ = "1.0.0"
__author__ = "Carmen Wrede & Lino Casu"

from . import core
from . import antennas
from . import feedline
from . import propagation
from . import field_strength
from . import ssz_extension

# Convenience imports
from .core import (
    freq_to_lambda,
    lambda_to_freq,
    freq_mhz_to_lambda,
    freq_khz_to_lambda,
    db_from_ratio,
    ratio_from_db,
)

from .antennas import (
    dipole_length_halfwave,
    vertical_quarterwave,
    estimate_yagi_gain,
)

from .feedline import (
    attenuation_db_per_100m,
    total_loss_db,
)

from .ssz_extension import (
    d_ssz_from_delta,
    effective_c_from_ssz,
    compare_lambda_with_ssz,
)

__all__ = [
    'core',
    'antennas',
    'feedline',
    'propagation',
    'field_strength',
    'ssz_extension',
    # Core functions
    'freq_to_lambda',
    'lambda_to_freq',
    'freq_mhz_to_lambda',
    'freq_khz_to_lambda',
    'db_from_ratio',
    'ratio_from_db',
    # Antenna functions
    'dipole_length_halfwave',
    'vertical_quarterwave',
    'estimate_yagi_gain',
    # Feedline functions
    'attenuation_db_per_100m',
    'total_loss_db',
    # SSZ functions
    'd_ssz_from_delta',
    'effective_c_from_ssz',
    'compare_lambda_with_ssz',
]
