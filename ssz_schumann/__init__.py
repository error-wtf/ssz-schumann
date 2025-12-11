#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSZ Schumann Resonance Analysis Package

A Python package for analyzing Schumann resonances in the context of
Segmented Spacetime (SSZ) theory.

The package provides:
- Data I/O for Schumann resonance measurements (Sierra Nevada ELF station)
- Space weather data integration (F10.7, Kp/Ap indices)
- Classical Schumann resonance models
- SSZ correction factor computation
- Statistical analysis and visualization

Physical Background:
    The Schumann resonances are electromagnetic resonances in the
    Earth-ionosphere cavity, with fundamental frequency ~7.83 Hz.
    
    Classical formula:
        f_n = eta * c / (2*pi*R) * sqrt(n*(n+1))
    
    SSZ modification:
        f_n_ssz = f_n_classical / D_SSZ(t)
        D_SSZ(t) = 1 + delta_seg(t)
    
    The SSZ signature is that delta_seg(t) produces the SAME relative
    frequency shift for ALL modes n, unlike classical dispersive effects.

Authors:
    Carmen Wrede & Lino Casu

License:
    Anti-Capitalist Software License v1.4

(c) 2025
"""

__version__ = "0.1.0"
__author__ = "Carmen Wrede & Lino Casu"
__license__ = "Anti-Capitalist Software License v1.4"

from .config import Config, EARTH_RADIUS, C_LIGHT, PHI

__all__ = [
    "Config",
    "EARTH_RADIUS",
    "C_LIGHT",
    "PHI",
    "__version__",
]
