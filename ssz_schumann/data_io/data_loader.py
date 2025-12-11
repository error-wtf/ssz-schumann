#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T2 & T3: Unified Data Loader for Schumann Resonance Data

Provides a standardized interface for loading both synthetic and real
Schumann resonance data.

Data Contract (API_REFERENCE.md):
    The returned DataFrame/Dataset must contain:
    
    Required columns:
        - time: DatetimeIndex (UTC)
        - f1_obs, f2_obs, f3_obs: Observed frequencies (Hz)
    
    Optional columns:
        - sigma_f1, sigma_f2, sigma_f3: Frequency uncertainties (Hz)
        - width1, width2, width3: Resonance widths (Hz)
        - amp1, amp2, amp3: Amplitudes
        - f107: F10.7 solar flux index
        - kp: Kp geomagnetic index
        - local_time: Local solar time (hours)
        - delta_seg_true: True delta_seg (synthetic data only)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SchumannDataSchema:
    """
    Schema definition for Schumann resonance data.
    
    This defines the expected structure of data returned by
    all data loaders in this module.
    """
    # Required frequency columns
    REQUIRED_FREQ = ["f1_obs", "f2_obs", "f3_obs"]
    
    # Optional uncertainty columns
    OPTIONAL_SIGMA = ["sigma_f1", "sigma_f2", "sigma_f3"]
    
    # Optional resonance parameters
    OPTIONAL_PARAMS = ["width1", "width2", "width3", "amp1", "amp2", "amp3"]
    
    # Optional covariates
    OPTIONAL_COVARIATES = ["f107", "kp", "local_time", "latitude", "longitude"]
    
    # Synthetic data marker
    SYNTHETIC_MARKER = "delta_seg_true"
    
    @classmethod
    def validate(cls, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame conforms to schema.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If required columns missing
        """
        missing = [col for col in cls.REQUIRED_FREQ if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Expected: {cls.REQUIRED_FREQ}"
            )
        
        # Check time index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Index must be DatetimeIndex, got {type(df.index)}"
            )
        
        return True


def load_schumann_timeseries(
    source: Union[str, Path, pd.DataFrame, xr.Dataset],
    dataset_type: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    """
    Load Schumann resonance data from various sources.
    
    This is the main entry point for loading data. It returns a
    standardized DataFrame conforming to SchumannDataSchema.
    
    Args:
        source: Data source, one of:
            - "synthetic": Generate synthetic data
            - Path to CSV/HDF5/NetCDF file
            - pandas DataFrame (passed through)
            - xarray Dataset (converted)
        dataset_type: Type hint for source:
            - "auto": Auto-detect from source
            - "synthetic": Generate synthetic data
            - "csv": Load from CSV
            - "real": Load real data (placeholder)
        **kwargs: Additional arguments passed to specific loader
    
    Returns:
        pd.DataFrame with standardized columns:
            - Index: DatetimeIndex (time)
            - f1_obs, f2_obs, f3_obs: Frequencies (Hz)
            - Optional: sigma_f*, width*, amp*, f107, kp, etc.
    
    Example:
        >>> # Synthetic data
        >>> df = load_schumann_timeseries("synthetic", n_days=365)
        
        >>> # From CSV
        >>> df = load_schumann_timeseries("data/schumann_2016.csv")
        
        >>> # From existing DataFrame
        >>> df = load_schumann_timeseries(my_dataframe)
    """
    # Handle different source types
    if isinstance(source, pd.DataFrame):
        df = _standardize_dataframe(source)
    
    elif isinstance(source, xr.Dataset):
        df = _dataset_to_dataframe(source)
    
    elif isinstance(source, str):
        if source.lower() == "synthetic" or dataset_type == "synthetic":
            df = load_synthetic_schumann_data(**kwargs)
        elif Path(source).exists():
            df = _load_from_file(Path(source), **kwargs)
        else:
            raise FileNotFoundError(f"Data source not found: {source}")
    
    elif isinstance(source, Path):
        df = _load_from_file(source, **kwargs)
    
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
    
    # Validate schema
    SchumannDataSchema.validate(df)
    
    logger.info(f"Loaded Schumann data: {len(df)} points, "
                f"columns: {list(df.columns)}")
    
    return df


def load_synthetic_schumann_data(
    start: str = "2016-01-01",
    end: str = "2016-12-31",
    freq: str = "1h",
    eta_0: float = 0.74,
    delta_seg_amplitude: float = 0.02,
    noise_level: float = 0.01,
    include_covariates: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic Schumann resonance data.
    
    Creates data with known SSZ correction for validation and testing.
    
    Args:
        start: Start date (ISO format)
        end: End date (ISO format)
        freq: Time frequency (pandas freq string)
        eta_0: Baseline slowdown factor
        delta_seg_amplitude: Amplitude of SSZ variation
        noise_level: Relative noise level (fraction of frequency)
        include_covariates: Include F10.7 and Kp proxies
        seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame with columns:
            - f1_obs, f2_obs, f3_obs: Observed frequencies
            - sigma_f1, sigma_f2, sigma_f3: Uncertainties
            - delta_seg_true: True delta_seg (for validation)
            - f107, kp: Covariates (if include_covariates=True)
    
    Example:
        >>> df = load_synthetic_schumann_data(
        ...     delta_seg_amplitude=0.02,
        ...     noise_level=0.01,
        ...     seed=42
        ... )
        >>> print(df.head())
    """
    from ..config import C_LIGHT, EARTH_RADIUS
    
    np.random.seed(seed)
    
    # Create time index
    time = pd.date_range(start=start, end=end, freq=freq)
    n_points = len(time)
    
    # Time in days from start
    t_days = (time - time[0]).total_seconds() / 86400.0
    
    # Generate true delta_seg with multiple components
    # Daily variation (solar heating effect)
    daily = 0.4 * np.sin(2 * np.pi * t_days)
    
    # Annual variation (solar cycle proxy)
    annual = 0.4 * np.sin(2 * np.pi * t_days / 365.25)
    
    # Random component
    random = 0.2 * np.random.randn(n_points)
    
    # Total delta_seg
    delta_seg_true = delta_seg_amplitude * (daily + annual + random)
    
    # Classical frequency function
    def f_classical(n):
        return eta_0 * C_LIGHT / (2 * np.pi * EARTH_RADIUS) * np.sqrt(n * (n + 1))
    
    # SSZ correction factor
    D_SSZ = 1 + delta_seg_true
    
    # Generate frequencies for each mode
    data = {"time": time}
    
    for n in [1, 2, 3]:
        f_class = f_classical(n)
        f_obs = f_class / D_SSZ
        
        # Add measurement noise
        sigma = noise_level * f_class
        noise = sigma * np.random.randn(n_points)
        f_obs = f_obs + noise
        
        data[f"f{n}_obs"] = f_obs
        data[f"sigma_f{n}"] = np.full(n_points, sigma)
    
    # Store true delta_seg for validation
    data["delta_seg_true"] = delta_seg_true
    
    # Add covariates if requested
    if include_covariates:
        # Synthetic F10.7 (solar flux)
        f107_base = 100 + 50 * np.sin(2 * np.pi * t_days / 365.25)
        f107_noise = 10 * np.random.randn(n_points)
        data["f107"] = f107_base + f107_noise
        
        # Synthetic Kp (geomagnetic index)
        kp_base = 2 + 1.5 * np.random.randn(n_points)
        data["kp"] = np.clip(kp_base, 0, 9)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index("time")
    
    # Add metadata as attributes (stored in attrs for xarray compatibility)
    df.attrs = {
        "source": "synthetic",
        "eta_0": eta_0,
        "delta_seg_amplitude": delta_seg_amplitude,
        "noise_level": noise_level,
        "seed": seed,
    }
    
    return df


def load_real_schumann_data(
    source: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load real Schumann resonance data.
    
    T3: Placeholder for real data loading. Currently raises
    NotImplementedError with instructions for expected format.
    
    Args:
        source: Path to data file or URL
        **kwargs: Additional arguments
    
    Returns:
        pd.DataFrame conforming to SchumannDataSchema
    
    Raises:
        NotImplementedError: With instructions for data format
    
    Expected CSV format (QUICKSTART.md):
        time,f1_obs,f2_obs,f3_obs,sigma_f1,sigma_f2,sigma_f3
        2016-01-01 00:00:00,7.82,13.9,20.1,0.05,0.08,0.12
        2016-01-01 01:00:00,7.84,13.95,20.15,0.05,0.08,0.12
        ...
    
    Notes:
        - time: UTC datetime
        - f*_obs: Frequencies in Hz
        - sigma_f*: Uncertainties in Hz (optional)
    """
    raise NotImplementedError(
        "Real data loading not yet implemented.\n\n"
        "Expected CSV format:\n"
        "  time,f1_obs,f2_obs,f3_obs[,sigma_f1,sigma_f2,sigma_f3]\n"
        "  2016-01-01 00:00:00,7.82,13.9,20.1[,0.05,0.08,0.12]\n\n"
        "To use your own data:\n"
        "1. Prepare CSV with columns: time, f1_obs, f2_obs, f3_obs\n"
        "2. Use: df = load_schumann_timeseries('path/to/data.csv')\n\n"
        "See QUICKSTART.md for detailed format specification."
    )


def _load_from_file(path: Path, **kwargs) -> pd.DataFrame:
    """Load data from file based on extension."""
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif suffix in [".h5", ".hdf5", ".hdf"]:
        df = pd.read_hdf(path, **kwargs)
    elif suffix in [".nc", ".netcdf"]:
        ds = xr.open_dataset(path, **kwargs)
        df = _dataset_to_dataframe(ds)
    elif suffix == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    else:
        # Try CSV as fallback
        df = pd.read_csv(path, **kwargs)
    
    return _standardize_dataframe(df)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names and index.
    
    Maps various column naming conventions to the standard schema.
    """
    df = df.copy()
    
    # Column name mapping
    rename_map = {
        # Frequency columns
        "f1": "f1_obs", "freq1": "f1_obs", "frequency1": "f1_obs",
        "f2": "f2_obs", "freq2": "f2_obs", "frequency2": "f2_obs",
        "f3": "f3_obs", "freq3": "f3_obs", "frequency3": "f3_obs",
        # Uncertainty columns
        "sigma1": "sigma_f1", "err_f1": "sigma_f1", "uncertainty1": "sigma_f1",
        "sigma2": "sigma_f2", "err_f2": "sigma_f2", "uncertainty2": "sigma_f2",
        "sigma3": "sigma_f3", "err_f3": "sigma_f3", "uncertainty3": "sigma_f3",
    }
    
    # Apply renaming (case-insensitive)
    for old_name in df.columns:
        lower_name = old_name.lower()
        if lower_name in rename_map:
            df = df.rename(columns={old_name: rename_map[lower_name]})
    
    # Handle time column/index
    if not isinstance(df.index, pd.DatetimeIndex):
        time_cols = ["time", "datetime", "timestamp", "date", "utc"]
        for col in time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    
    df.index.name = "time"
    
    return df


def _dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Convert xarray Dataset to standardized DataFrame."""
    df = ds.to_dataframe()
    return _standardize_dataframe(df)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_frequency_dict(df: pd.DataFrame) -> Dict[int, pd.Series]:
    """
    Extract frequency time series as dictionary.
    
    Args:
        df: DataFrame with f1_obs, f2_obs, f3_obs columns
    
    Returns:
        Dictionary {1: f1_series, 2: f2_series, 3: f3_series}
    """
    return {
        1: df["f1_obs"],
        2: df["f2_obs"],
        3: df["f3_obs"],
    }


def split_train_test(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    method: str = "end",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    Args:
        df: Input DataFrame
        test_fraction: Fraction of data for testing
        method: Split method:
            - "end": Last test_fraction for testing
            - "random": Random split
    
    Returns:
        Tuple of (train_df, test_df)
    """
    n = len(df)
    n_test = int(n * test_fraction)
    
    if method == "end":
        train_df = df.iloc[:-n_test]
        test_df = df.iloc[-n_test:]
    elif method == "random":
        idx = np.random.permutation(n)
        train_idx = idx[:-n_test]
        test_idx = idx[-n_test:]
        train_df = df.iloc[train_idx].sort_index()
        test_df = df.iloc[test_idx].sort_index()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return train_df, test_df


if __name__ == "__main__":
    # Quick test
    print("=== Data Loader Test ===\n")
    
    # Load synthetic data
    df = load_schumann_timeseries("synthetic", n_days=30)
    print(f"Loaded {len(df)} points")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Validate schema
    SchumannDataSchema.validate(df)
    print("\nSchema validation: PASSED")
