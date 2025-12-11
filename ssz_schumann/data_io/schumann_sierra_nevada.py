#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schumann Resonance Data I/O - Sierra Nevada ELF Station

Loads and processes Schumann resonance measurements from the
Sierra Nevada ELF station (Salinas et al., 2022).

Data source:
    Zenodo: https://doi.org/10.5281/zenodo.6348930
    Paper: "Schumann resonance data processing programs and four-year
           measurements from Sierra Nevada ELF station"
           Computers & Geosciences, 2022

Data format:
    - 10-minute intervals
    - Parameters: f1, f2, f3 (frequencies), width1-3, amp1-3
    - Time in UTC

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from datetime import datetime
import warnings
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# EXPECTED COLUMN NAMES (may vary by year/file)
# =============================================================================

# Mapping of possible column names to standardized names
COLUMN_MAPPING = {
    # Frequencies
    "f1": "f1", "freq1": "f1", "frequency1": "f1", "f_1": "f1",
    "f2": "f2", "freq2": "f2", "frequency2": "f2", "f_2": "f2",
    "f3": "f3", "freq3": "f3", "frequency3": "f3", "f_3": "f3",
    
    # Widths (FWHM or damping)
    "width1": "width1", "w1": "width1", "gamma1": "width1", "fwhm1": "width1",
    "width2": "width2", "w2": "width2", "gamma2": "width2", "fwhm2": "width2",
    "width3": "width3", "w3": "width3", "gamma3": "width3", "fwhm3": "width3",
    
    # Amplitudes
    "amp1": "amp1", "a1": "amp1", "amplitude1": "amp1", "ampl1": "amp1",
    "amp2": "amp2", "a2": "amp2", "amplitude2": "amp2", "ampl2": "amp2",
    "amp3": "amp3", "a3": "amp3", "amplitude3": "amp3", "ampl3": "amp3",
    
    # Time
    "time": "time", "datetime": "time", "timestamp": "time", "date": "time",
    "utc": "time", "utc_time": "time",
}

# Required columns after standardization
REQUIRED_COLUMNS = ["f1", "f2", "f3"]
OPTIONAL_COLUMNS = ["width1", "width2", "width3", "amp1", "amp2", "amp3"]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to expected format.
    
    Args:
        df: DataFrame with potentially non-standard column names
    
    Returns:
        DataFrame with standardized column names
    """
    # Convert to lowercase for matching
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in COLUMN_MAPPING:
            rename_map[col] = COLUMN_MAPPING[col_lower]
    
    return df.rename(columns=rename_map)


def _parse_time_column(
    df: pd.DataFrame,
    time_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse and set time column as index.
    
    Args:
        df: DataFrame
        time_col: Name of time column (auto-detected if None)
    
    Returns:
        DataFrame with DatetimeIndex
    """
    # Auto-detect time column
    if time_col is None:
        time_candidates = ["time", "datetime", "timestamp", "date", "utc"]
        for candidate in time_candidates:
            if candidate in df.columns:
                time_col = candidate
                break
    
    if time_col is None:
        # Try to use index if it looks like datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        raise ValueError("Could not identify time column. "
                        f"Available columns: {list(df.columns)}")
    
    # Parse time
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col)
    df.index.name = "time"
    
    return df


def load_schumann_sierra_nevada(
    path_or_url: Union[str, Path],
    year: Optional[int] = None,
    **kwargs
) -> xr.Dataset:
    """
    Load Schumann resonance data from Sierra Nevada ELF station.
    
    Args:
        path_or_url: Path to data file or URL
            Supported formats: CSV, HDF5, NetCDF
        year: Optional year filter (2013-2017)
        **kwargs: Additional arguments passed to pandas read function
    
    Returns:
        xarray.Dataset with dimensions:
            - time: DatetimeIndex (UTC)
        
        Data variables:
            - f1, f2, f3: Resonance frequencies (Hz)
            - width1, width2, width3: Resonance widths (Hz) [if available]
            - amp1, amp2, amp3: Amplitudes [if available]
        
        Attributes:
            - station: "sierra_nevada"
            - source: Data source path/URL
            - units: Variable units
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: If required columns missing
    
    Example:
        >>> ds = load_schumann_sierra_nevada("data/schumann/2016.csv")
        >>> print(ds.f1.mean())
    """
    path = Path(path_or_url) if not str(path_or_url).startswith("http") else path_or_url
    
    # Determine file format and load
    if isinstance(path, Path):
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == ".csv":
            df = pd.read_csv(path, **kwargs)
        elif suffix in [".h5", ".hdf5", ".hdf"]:
            df = pd.read_hdf(path, **kwargs)
        elif suffix in [".nc", ".netcdf"]:
            # Load directly as xarray
            ds = xr.open_dataset(path, **kwargs)
            return _postprocess_dataset(ds, str(path))
        else:
            # Try CSV as default
            df = pd.read_csv(path, **kwargs)
    else:
        # URL - try to load as CSV
        df = pd.read_csv(path_or_url, **kwargs)
    
    # Standardize columns
    df = _standardize_columns(df)
    
    # Parse time
    df = _parse_time_column(df)
    
    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                        f"Available: {list(df.columns)}")
    
    # Filter by year if specified
    if year is not None:
        df = df[df.index.year == year]
        if len(df) == 0:
            warnings.warn(f"No data found for year {year}")
    
    # Convert to xarray Dataset
    ds = df.to_xarray()
    
    return _postprocess_dataset(ds, str(path_or_url))


def _postprocess_dataset(ds: xr.Dataset, source: str) -> xr.Dataset:
    """Add metadata and ensure correct dtypes."""
    
    # Add attributes
    ds.attrs["station"] = "sierra_nevada"
    ds.attrs["source"] = source
    ds.attrs["created"] = datetime.utcnow().isoformat()
    
    # Variable attributes
    freq_attrs = {"units": "Hz", "long_name": "Schumann resonance frequency"}
    width_attrs = {"units": "Hz", "long_name": "Resonance width (FWHM)"}
    amp_attrs = {"units": "arbitrary", "long_name": "Resonance amplitude"}
    
    for n in [1, 2, 3]:
        if f"f{n}" in ds:
            ds[f"f{n}"].attrs = {**freq_attrs, "mode": n}
        if f"width{n}" in ds:
            ds[f"width{n}"].attrs = {**width_attrs, "mode": n}
        if f"amp{n}" in ds:
            ds[f"amp{n}"].attrs = {**amp_attrs, "mode": n}
    
    return ds


def resample_schumann(
    ds: xr.Dataset,
    freq: str = "1H",
    method: str = "mean"
) -> xr.Dataset:
    """
    Resample Schumann data to different time resolution.
    
    Args:
        ds: Input dataset
        freq: Target frequency (pandas frequency string)
            Examples: "1H" (hourly), "1D" (daily), "30T" (30 minutes)
        method: Aggregation method ("mean", "median", "std")
    
    Returns:
        Resampled dataset
    
    Example:
        >>> ds_hourly = resample_schumann(ds, freq="1h", method="mean")
    """
    # Resample using xarray
    resampler = ds.resample(time=freq)
    
    if method == "mean":
        ds_resampled = resampler.mean()
    elif method == "median":
        ds_resampled = resampler.median()
    elif method == "std":
        ds_resampled = resampler.std()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', or 'std'.")
    
    # Update attributes
    ds_resampled.attrs = ds.attrs.copy()
    ds_resampled.attrs["resampled_to"] = freq
    ds_resampled.attrs["resample_method"] = method
    
    return ds_resampled


def apply_quality_filter(
    ds: xr.Dataset,
    min_amplitude: Optional[float] = None,
    max_width: Optional[float] = None,
    f1_range: Optional[Tuple[float, float]] = None,
    f2_range: Optional[Tuple[float, float]] = None,
    f3_range: Optional[Tuple[float, float]] = None,
    mask_intervals: Optional[List[Tuple[str, str]]] = None,
) -> xr.Dataset:
    """
    Apply quality filters to Schumann data.
    
    Args:
        ds: Input dataset
        min_amplitude: Minimum amplitude threshold (mask if below)
        max_width: Maximum width threshold (mask if above)
        f1_range: Valid range for f1 (min, max) in Hz
        f2_range: Valid range for f2 (min, max) in Hz
        f3_range: Valid range for f3 (min, max) in Hz
        mask_intervals: List of (start, end) time intervals to mask
    
    Returns:
        Dataset with invalid data set to NaN
    
    Example:
        >>> ds_clean = apply_quality_filter(
        ...     ds,
        ...     min_amplitude=0.1,
        ...     max_width=5.0,
        ...     f1_range=(6.0, 10.0)
        ... )
    """
    ds = ds.copy()
    
    # Create quality mask (True = valid)
    mask = xr.ones_like(ds["f1"], dtype=bool)
    
    # Amplitude filter
    if min_amplitude is not None and "amp1" in ds:
        for n in [1, 2, 3]:
            if f"amp{n}" in ds:
                mask = mask & (ds[f"amp{n}"] >= min_amplitude)
    
    # Width filter
    if max_width is not None:
        for n in [1, 2, 3]:
            if f"width{n}" in ds:
                mask = mask & (ds[f"width{n}"] <= max_width)
    
    # Frequency range filters
    ranges = {1: f1_range, 2: f2_range, 3: f3_range}
    for n, frange in ranges.items():
        if frange is not None and f"f{n}" in ds:
            mask = mask & (ds[f"f{n}"] >= frange[0]) & (ds[f"f{n}"] <= frange[1])
    
    # Time interval masking
    if mask_intervals is not None:
        for start, end in mask_intervals:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            time_mask = (ds.time < start_dt) | (ds.time > end_dt)
            mask = mask & time_mask
    
    # Apply mask (set invalid to NaN)
    for var in ds.data_vars:
        ds[var] = ds[var].where(mask)
    
    # Record filtering in attributes
    ds.attrs["quality_filtered"] = True
    ds.attrs["filter_params"] = {
        "min_amplitude": min_amplitude,
        "max_width": max_width,
        "f1_range": f1_range,
        "f2_range": f2_range,
        "f3_range": f3_range,
        "mask_intervals": mask_intervals,
    }
    
    # Count valid points
    n_valid = int(mask.sum())
    n_total = int(mask.size)
    logger.info(f"Quality filter: {n_valid}/{n_total} points valid "
                f"({100*n_valid/n_total:.1f}%)")
    
    return ds


def create_synthetic_schumann_data(
    start: str = "2016-01-01",
    end: str = "2016-12-31",
    freq: str = "10T",
    eta_0: float = 0.74,
    delta_seg_amplitude: float = 0.02,
    noise_level: float = 0.05,
    seed: int = 42,
) -> xr.Dataset:
    """
    Create synthetic Schumann resonance data for testing.
    
    Generates data with known SSZ correction for validation.
    
    Args:
        start: Start date
        end: End date
        freq: Time frequency
        eta_0: Baseline slowdown factor
        delta_seg_amplitude: Amplitude of SSZ correction
        noise_level: Relative noise level
        seed: Random seed
    
    Returns:
        Synthetic dataset with known parameters
    
    Example:
        >>> ds_synth = create_synthetic_schumann_data(
        ...     delta_seg_amplitude=0.02,
        ...     noise_level=0.01
        ... )
    """
    from ..config import C_LIGHT, EARTH_RADIUS
    
    np.random.seed(seed)
    
    # Create time index (without timezone for xarray compatibility)
    time = pd.date_range(start=start, end=end, freq=freq)
    n_points = len(time)
    
    # Generate synthetic delta_seg (sinusoidal with daily and annual components)
    t_days = (time - time[0]).total_seconds() / 86400
    
    # Daily variation (solar heating)
    daily = 0.5 * np.sin(2 * np.pi * t_days)
    
    # Annual variation (solar cycle proxy)
    annual = 0.3 * np.sin(2 * np.pi * t_days / 365.25)
    
    # Random component
    random = 0.2 * np.random.randn(n_points)
    
    # Total delta_seg
    delta_seg = delta_seg_amplitude * (daily + annual + random)
    
    # Classical frequencies
    def f_classical(n):
        return eta_0 * C_LIGHT / (2 * np.pi * EARTH_RADIUS) * np.sqrt(n * (n + 1))
    
    # SSZ-modified frequencies
    D_SSZ = 1 + delta_seg
    
    data_vars = {}
    for n in [1, 2, 3]:
        f_class = f_classical(n)
        f_obs = f_class / D_SSZ
        
        # Add measurement noise
        noise = noise_level * f_class * np.random.randn(n_points)
        f_obs = f_obs + noise
        
        data_vars[f"f{n}"] = (["time"], f_obs)
        data_vars[f"width{n}"] = (["time"], 1.0 + 0.5 * np.random.randn(n_points))
        data_vars[f"amp{n}"] = (["time"], 1.0 + 0.2 * np.random.randn(n_points))
    
    # Store true delta_seg for validation
    data_vars["delta_seg_true"] = (["time"], delta_seg)
    
    ds = xr.Dataset(
        data_vars,
        coords={"time": time},
        attrs={
            "station": "synthetic",
            "eta_0": eta_0,
            "delta_seg_amplitude": delta_seg_amplitude,
            "noise_level": noise_level,
            "seed": seed,
            "synthetic": True,
        }
    )
    
    return ds
