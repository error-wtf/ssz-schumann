#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data I/O Module for SSZ Schumann Analysis

Provides functions to load and preprocess:
- Schumann resonance data (Sierra Nevada ELF station)
- Space weather indices (F10.7, Kp/Ap)
- Lightning activity data (WWLLN)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

from .schumann_sierra_nevada import (
    load_schumann_sierra_nevada,
    resample_schumann,
    apply_quality_filter,
)

from .space_weather_noaa import (
    load_f107,
    load_kp,
    to_time_resolution,
)

from .lightning_wwlln import (
    load_thunder_hours,
    thunder_index_for_station,
)

from .merge import merge_all

from .schumann_real import (
    load_schumann_real_data,
    create_sample_real_data,
    get_data_summary,
    download_zenodo_data,
)

# T2 & T3: Unified data loader
from .data_loader import (
    load_schumann_timeseries,
    load_synthetic_schumann_data,
    load_real_schumann_data,
    SchumannDataSchema,
    get_frequency_dict,
    split_train_test,
)

# Real data loaders
from .real_schumann_loader import (
    load_real_schumann_data as load_real_schumann_from_config,
    validate_schumann_data,
    RealSchumannConfig,
)

from .real_space_weather_loader import (
    load_f107,
    load_kp,
    load_space_weather_from_config,
)

# Unified loader
from .unified_loader import (
    load_all_data,
    load_all_data_from_config,
    UnifiedData,
    UnifiedDataConfig,
)

__all__ = [
    # Schumann
    "load_schumann_sierra_nevada",
    "resample_schumann",
    "apply_quality_filter",
    # Space weather
    "load_f107",
    "load_kp",
    "to_time_resolution",
    # Lightning
    "load_thunder_hours",
    "thunder_index_for_station",
    # Merge
    "merge_all",
    # Real data
    "load_schumann_real_data",
    "create_sample_real_data",
    "get_data_summary",
    "download_zenodo_data",
    # T2 & T3: Unified data loader
    "load_schumann_timeseries",
    "load_synthetic_schumann_data",
    "load_real_schumann_data",
    "SchumannDataSchema",
    "get_frequency_dict",
    "split_train_test",
    # Real data loaders
    "load_real_schumann_from_config",
    "validate_schumann_data",
    "load_f107",
    "load_kp",
    "load_space_weather_from_config",
    # Unified loader
    "load_all_data",
    "load_all_data_from_config",
    "UnifiedData",
    "UnifiedDataConfig",
]
