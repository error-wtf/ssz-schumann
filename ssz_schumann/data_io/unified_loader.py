#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Data Loader - Dispatcher for Synthetic and Real Data

Central entry point for loading Schumann and space weather data.
Automatically dispatches to appropriate loader based on configuration.

Usage:
    # From YAML config
    data = load_all_data_from_config("configs/real_example.yml")
    
    # Programmatic
    data = load_all_data(data_source_type="synthetic")
    data = load_all_data(data_source_type="real", config=real_config)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Union, Any
from dataclasses import dataclass, field
import logging
import yaml

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDataConfig:
    """
    Unified configuration for data loading.
    
    Supports both synthetic and real data sources.
    """
    # Data source type: "synthetic" or "real"
    data_source_type: str = "synthetic"
    
    # Schumann data configuration
    schumann_path: Optional[str] = None
    schumann_type: str = "netcdf"  # "netcdf" or "csv"
    schumann_time_column: str = "time"
    schumann_freq_columns: Dict[int, str] = field(default_factory=lambda: {1: "f1", 2: "f2", 3: "f3"})
    schumann_q_columns: Optional[Dict[int, str]] = None
    schumann_timezone: str = "UTC"
    
    # Space weather configuration
    f107_path: Optional[str] = None
    f107_time_column: str = "time"
    f107_value_column: str = "f107"
    f107_timezone: str = "UTC"
    
    kp_path: Optional[str] = None
    kp_time_column: str = "time"
    kp_value_column: str = "kp"
    kp_timezone: str = "UTC"
    
    # Synthetic data parameters
    synthetic_start: str = "2016-01-01"
    synthetic_end: str = "2016-12-31"
    synthetic_freq: str = "1h"
    synthetic_delta_seg_amplitude: float = 0.02
    synthetic_noise_level: float = 0.01
    synthetic_seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "UnifiedDataConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, d: dict) -> "UnifiedDataConfig":
        """Create configuration from dictionary."""
        data_source = d.get("data_source", {})
        
        config = cls(
            data_source_type=data_source.get("type", "synthetic"),
        )
        
        # Schumann configuration
        schumann = data_source.get("schumann", {})
        config.schumann_path = schumann.get("path")
        config.schumann_type = schumann.get("type", "netcdf")
        config.schumann_time_column = schumann.get("time_column", "time")
        config.schumann_freq_columns = schumann.get("freq_columns", {1: "f1", 2: "f2", 3: "f3"})
        config.schumann_q_columns = schumann.get("q_columns")
        config.schumann_timezone = schumann.get("timezone", "UTC")
        
        # Space weather configuration
        space_weather = data_source.get("space_weather", {})
        
        f107 = space_weather.get("f107", {})
        config.f107_path = f107.get("path")
        config.f107_time_column = f107.get("time_column", "time")
        config.f107_value_column = f107.get("value_column", "f107")
        config.f107_timezone = f107.get("timezone", "UTC")
        
        kp = space_weather.get("kp", {})
        config.kp_path = kp.get("path")
        config.kp_time_column = kp.get("time_column", "time")
        config.kp_value_column = kp.get("value_column", "kp")
        config.kp_timezone = kp.get("timezone", "UTC")
        
        return config


@dataclass
class UnifiedData:
    """
    Container for loaded data.
    
    Provides a unified interface regardless of data source.
    """
    # Schumann data as xarray Dataset
    schumann: xr.Dataset
    
    # Space weather as pandas Series
    f107: pd.Series
    kp: pd.Series
    
    # Metadata
    data_source_type: str
    config: Optional[UnifiedDataConfig] = None
    
    @property
    def time_index(self) -> pd.DatetimeIndex:
        """Get time index from Schumann data."""
        return pd.DatetimeIndex(self.schumann.time.values)
    
    @property
    def n_points(self) -> int:
        """Number of time points."""
        return len(self.schumann.time)
    
    @property
    def time_range(self) -> str:
        """Human-readable time range."""
        t0 = pd.Timestamp(self.schumann.time.values[0])
        t1 = pd.Timestamp(self.schumann.time.values[-1])
        return f"{t0.date()} to {t1.date()}"
    
    def get_frequencies(self, modes: list = [1, 2, 3]) -> Dict[int, pd.Series]:
        """
        Get frequency time series for specified modes.
        
        Returns dictionary {mode: frequency_series}.
        """
        result = {}
        for mode in modes:
            # Try different variable naming conventions
            for var_name in [f"f{mode}", f"f_mode{mode}", f"freq{mode}"]:
                if var_name in self.schumann:
                    result[mode] = pd.Series(
                        self.schumann[var_name].values,
                        index=self.time_index,
                        name=f"f{mode}",
                    )
                    break
        return result
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"UnifiedData ({self.data_source_type})",
            f"  Time range: {self.time_range}",
            f"  Points: {self.n_points}",
            f"  Schumann vars: {list(self.schumann.data_vars)}",
            f"  F10.7: {len(self.f107)} points, mean={self.f107.mean():.1f}",
            f"  Kp: {len(self.kp)} points, mean={self.kp.mean():.2f}",
        ]
        return "\n".join(lines)


def load_all_data(
    data_source_type: str = "synthetic",
    config: Optional[UnifiedDataConfig] = None,
    **kwargs,
) -> UnifiedData:
    """
    Load all data (Schumann + space weather) from specified source.
    
    Parameters
    ----------
    data_source_type : str
        "synthetic" or "real"
    config : UnifiedDataConfig, optional
        Configuration object. If None, uses defaults.
    **kwargs
        Override config parameters.
    
    Returns
    -------
    UnifiedData
        Container with all loaded data.
    
    Examples
    --------
    >>> # Synthetic data (default)
    >>> data = load_all_data("synthetic")
    
    >>> # Real data with config
    >>> config = UnifiedDataConfig.from_yaml("configs/real_example.yml")
    >>> data = load_all_data("real", config=config)
    """
    if config is None:
        config = UnifiedDataConfig(data_source_type=data_source_type)
    
    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.data_source_type = data_source_type
    
    if data_source_type == "synthetic":
        return _load_synthetic_data(config)
    elif data_source_type == "real":
        return _load_real_data(config)
    else:
        raise ValueError(f"Unknown data_source_type: {data_source_type}")


def load_all_data_from_config(
    config_path: Union[str, Path],
) -> UnifiedData:
    """
    Load all data using YAML configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.
    
    Returns
    -------
    UnifiedData
        Container with all loaded data.
    
    Examples
    --------
    >>> data = load_all_data_from_config("configs/real_example.yml")
    >>> print(data.summary())
    """
    config = UnifiedDataConfig.from_yaml(config_path)
    return load_all_data(config.data_source_type, config)


def _load_synthetic_data(config: UnifiedDataConfig) -> UnifiedData:
    """Load synthetic data."""
    from .schumann_sierra_nevada import create_synthetic_schumann_data
    from .space_weather_noaa import create_synthetic_space_weather
    
    logger.info("Loading synthetic data...")
    
    # Check if files exist, otherwise generate
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    schumann_path = data_dir / "schumann" / "synthetic_2016.nc"
    f107_path = data_dir / "space_weather" / "synthetic_f107.csv"
    kp_path = data_dir / "space_weather" / "synthetic_kp.csv"
    
    # Load or generate Schumann data
    if schumann_path.exists():
        logger.info(f"Loading Schumann from {schumann_path}")
        schumann = xr.open_dataset(schumann_path)
    else:
        logger.info("Generating synthetic Schumann data...")
        schumann = create_synthetic_schumann_data(
            start=config.synthetic_start,
            end=config.synthetic_end,
            freq=config.synthetic_freq,
            delta_seg_amplitude=config.synthetic_delta_seg_amplitude,
            noise_level=config.synthetic_noise_level,
            seed=config.synthetic_seed,
        )
    
    # Load or generate F10.7
    if f107_path.exists():
        logger.info(f"Loading F10.7 from {f107_path}")
        f107 = pd.read_csv(f107_path, index_col=0, parse_dates=True).squeeze()
    else:
        logger.info("Generating synthetic F10.7 data...")
        f107, _ = create_synthetic_space_weather(
            config.synthetic_start, config.synthetic_end
        )
    
    # Load or generate Kp
    if kp_path.exists():
        logger.info(f"Loading Kp from {kp_path}")
        kp = pd.read_csv(kp_path, index_col=0, parse_dates=True).squeeze()
    else:
        logger.info("Generating synthetic Kp data...")
        _, kp = create_synthetic_space_weather(
            config.synthetic_start, config.synthetic_end
        )
    
    return UnifiedData(
        schumann=schumann,
        f107=f107,
        kp=kp,
        data_source_type="synthetic",
        config=config,
    )


def _load_real_data(config: UnifiedDataConfig) -> UnifiedData:
    """Load real data."""
    from .real_schumann_loader import load_real_schumann_data, convert_to_standard_format
    from .real_space_weather_loader import load_f107, load_kp, resample_to_match
    
    logger.info("Loading real data...")
    
    # Load Schumann data
    if not config.schumann_path:
        raise ValueError(
            "schumann_path must be specified for real data.\n"
            "Set data_source.schumann.path in your config file."
        )
    
    schumann_raw = load_real_schumann_data(
        path=config.schumann_path,
        time_column=config.schumann_time_column,
        freq_columns=config.schumann_freq_columns,
        q_columns=config.schumann_q_columns,
        timezone=config.schumann_timezone,
    )
    
    # Convert to standard format
    schumann = convert_to_standard_format(schumann_raw)
    
    # Get time index for resampling space weather
    time_index = pd.DatetimeIndex(schumann.time.values)
    
    # Load F10.7
    if config.f107_path:
        f107_raw = load_f107(
            path=config.f107_path,
            time_column=config.f107_time_column,
            value_column=config.f107_value_column,
            timezone=config.f107_timezone,
        )
        f107 = resample_to_match(f107_raw, time_index, method="nearest")
    else:
        logger.warning("No F10.7 path specified, using NaN")
        f107 = pd.Series(np.nan, index=time_index, name="f107")
    
    # Load Kp
    if config.kp_path:
        kp_raw = load_kp(
            path=config.kp_path,
            time_column=config.kp_time_column,
            value_column=config.kp_value_column,
            timezone=config.kp_timezone,
        )
        kp = resample_to_match(kp_raw, time_index, method="nearest")
    else:
        logger.warning("No Kp path specified, using NaN")
        kp = pd.Series(np.nan, index=time_index, name="kp")
    
    return UnifiedData(
        schumann=schumann,
        f107=f107,
        kp=kp,
        data_source_type="real",
        config=config,
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("=== Unified Loader Test ===\n")
    
    # Test synthetic loading
    print("1. Loading synthetic data...")
    data = load_all_data("synthetic")
    print(data.summary())
    
    # Test frequency extraction
    print("\n2. Extracting frequencies...")
    freqs = data.get_frequencies()
    for mode, series in freqs.items():
        print(f"   Mode {mode}: mean={series.mean():.2f} Hz")
