#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Merge Module

Combines Schumann resonance data with space weather indices
and lightning activity on a common time grid.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def merge_all(
    schumann_ds: xr.Dataset,
    f107_series: Optional[pd.Series] = None,
    kp_series: Optional[pd.Series] = None,
    thunder_series: Optional[pd.Series] = None,
    time_resolution: str = "1h",
    interpolation_method: str = "linear",
) -> xr.Dataset:
    """
    Merge all data sources onto a common time grid.
    
    Combines:
    - Schumann resonance measurements (f1, f2, f3, widths, amplitudes)
    - F10.7 solar flux
    - Kp geomagnetic index
    - Lightning activity index
    
    All data is resampled/interpolated to the target time resolution.
    
    Args:
        schumann_ds: Schumann resonance dataset (xarray)
        f107_series: F10.7 solar flux (pandas Series, daily)
        kp_series: Kp index (pandas Series, 3-hourly or daily)
        thunder_series: Lightning index (pandas Series)
        time_resolution: Target time resolution (pandas freq string)
        interpolation_method: Method for interpolating indices
            - "linear": Linear interpolation
            - "ffill": Forward fill (step function)
            - "nearest": Nearest neighbor
    
    Returns:
        xr.Dataset with all variables on common time grid
        
        Variables:
            - f1, f2, f3: Schumann frequencies (Hz)
            - width1, width2, width3: Resonance widths (Hz)
            - amp1, amp2, amp3: Amplitudes
            - f107: Solar flux (sfu)
            - f107_norm: Normalized F10.7
            - kp: Geomagnetic index
            - kp_norm: Normalized Kp
            - thunder: Lightning index
            - thunder_norm: Normalized lightning
    
    Example:
        >>> merged = merge_all(
        ...     schumann_ds,
        ...     f107_series,
        ...     kp_series,
        ...     time_resolution="1H"
        ... )
    """
    # Get time range from Schumann data
    time_start = pd.Timestamp(schumann_ds.time.values[0])
    time_end = pd.Timestamp(schumann_ds.time.values[-1])
    
    logger.info(f"Merging data from {time_start} to {time_end}")
    logger.info(f"Target resolution: {time_resolution}")
    
    # Create target time grid (without timezone for xarray compatibility)
    target_time = pd.date_range(
        start=time_start,
        end=time_end,
        freq=time_resolution,
    )
    
    # Remove timezone if present (xarray interp has issues with tz-aware datetimes)
    if target_time.tzinfo is not None:
        target_time = target_time.tz_localize(None)
    
    # Resample Schumann data
    schumann_resampled = schumann_ds.resample(time=time_resolution).mean()
    
    # Interpolate to exact target grid
    schumann_interp = schumann_resampled.interp(
        time=target_time,
        method="linear"
    )
    
    # Start with Schumann data
    merged = schumann_interp.copy()
    
    # Add F10.7
    if f107_series is not None:
        f107_aligned = _align_series_to_time(
            f107_series,
            target_time,
            interpolation_method,
            "f107"
        )
        merged["f107"] = (["time"], f107_aligned.values)
        merged["f107"].attrs = {"units": "sfu", "long_name": "F10.7 solar flux"}
        
        # Normalized version
        f107_norm = (f107_aligned - f107_aligned.mean()) / f107_aligned.std()
        merged["f107_norm"] = (["time"], f107_norm.values)
        merged["f107_norm"].attrs = {"units": "dimensionless", "long_name": "Normalized F10.7"}
        
        logger.info(f"Added F10.7: mean={f107_aligned.mean():.1f}, std={f107_aligned.std():.1f}")
    
    # Add Kp
    if kp_series is not None:
        kp_aligned = _align_series_to_time(
            kp_series,
            target_time,
            interpolation_method,
            "kp"
        )
        merged["kp"] = (["time"], kp_aligned.values)
        merged["kp"].attrs = {"units": "dimensionless", "long_name": "Kp geomagnetic index"}
        
        # Normalized version
        kp_norm = (kp_aligned - kp_aligned.mean()) / kp_aligned.std()
        merged["kp_norm"] = (["time"], kp_norm.values)
        merged["kp_norm"].attrs = {"units": "dimensionless", "long_name": "Normalized Kp"}
        
        logger.info(f"Added Kp: mean={kp_aligned.mean():.2f}, std={kp_aligned.std():.2f}")
    
    # Add Lightning
    if thunder_series is not None:
        thunder_aligned = _align_series_to_time(
            thunder_series,
            target_time,
            interpolation_method,
            "thunder"
        )
        merged["thunder"] = (["time"], thunder_aligned.values)
        merged["thunder"].attrs = {"units": "index", "long_name": "Lightning activity index"}
        
        # Normalized version
        thunder_norm = (thunder_aligned - thunder_aligned.mean()) / thunder_aligned.std()
        merged["thunder_norm"] = (["time"], thunder_norm.values)
        merged["thunder_norm"].attrs = {"units": "dimensionless", "long_name": "Normalized lightning"}
        
        logger.info(f"Added Lightning: mean={thunder_aligned.mean():.2f}")
    
    # Update attributes
    merged.attrs["merged"] = True
    merged.attrs["time_resolution"] = time_resolution
    merged.attrs["interpolation_method"] = interpolation_method
    merged.attrs["n_timepoints"] = len(target_time)
    
    # Count valid data points
    n_valid = {}
    for var in merged.data_vars:
        n_valid[var] = int((~np.isnan(merged[var].values)).sum())
    merged.attrs["n_valid"] = n_valid
    
    logger.info(f"Merged dataset: {len(target_time)} time points, "
                f"{len(merged.data_vars)} variables")
    
    return merged


def _align_series_to_time(
    series: pd.Series,
    target_time: pd.DatetimeIndex,
    method: str,
    name: str,
) -> pd.Series:
    """
    Align a pandas Series to target time grid.
    
    Args:
        series: Input series with DatetimeIndex
        target_time: Target time index
        method: Interpolation method
        name: Variable name for logging
    
    Returns:
        Aligned series
    """
    # Ensure timezone consistency
    if series.index.tzinfo is None and target_time.tzinfo is not None:
        series = series.tz_localize("UTC")
    elif series.index.tzinfo is not None and target_time.tzinfo is None:
        series = series.tz_localize(None)
    
    # Reindex to target time
    if method == "ffill":
        # Forward fill: appropriate for daily indices
        aligned = series.reindex(target_time, method="ffill")
    elif method == "linear":
        # Linear interpolation
        # First, combine indices
        combined_index = series.index.union(target_time)
        series_extended = series.reindex(combined_index)
        series_interp = series_extended.interpolate(method="time")
        aligned = series_interp.reindex(target_time)
    elif method == "nearest":
        aligned = series.reindex(target_time, method="nearest")
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Fill remaining NaNs at edges
    aligned = aligned.ffill().bfill()
    
    # Log coverage
    original_range = (series.index.min(), series.index.max())
    target_range = (target_time.min(), target_time.max())
    
    if original_range[0] > target_range[0] or original_range[1] < target_range[1]:
        logger.warning(
            f"{name}: Original data range {original_range} does not fully "
            f"cover target range {target_range}. Edge values extrapolated."
        )
    
    return aligned


def compute_derived_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute derived variables useful for analysis.
    
    Adds:
        - f_mean: Mean frequency across modes
        - f_ratio_21: f2/f1 ratio
        - f_ratio_31: f3/f1 ratio
        - local_hour: Local solar time (approximate)
        - day_of_year: Day of year
        - is_daytime: Boolean daytime flag
    
    Args:
        ds: Merged dataset
    
    Returns:
        Dataset with additional variables
    """
    ds = ds.copy()
    
    # Mean frequency
    if all(f"f{n}" in ds for n in [1, 2, 3]):
        ds["f_mean"] = (ds["f1"] + ds["f2"] + ds["f3"]) / 3
        ds["f_mean"].attrs = {"units": "Hz", "long_name": "Mean Schumann frequency"}
    
    # Frequency ratios
    if "f1" in ds and "f2" in ds:
        ds["f_ratio_21"] = ds["f2"] / ds["f1"]
        ds["f_ratio_21"].attrs = {"units": "dimensionless", "long_name": "f2/f1 ratio"}
    
    if "f1" in ds and "f3" in ds:
        ds["f_ratio_31"] = ds["f3"] / ds["f1"]
        ds["f_ratio_31"].attrs = {"units": "dimensionless", "long_name": "f3/f1 ratio"}
    
    # Time-based variables
    time = pd.DatetimeIndex(ds.time.values)
    
    ds["hour_utc"] = (["time"], time.hour.values.astype(float))
    ds["hour_utc"].attrs = {"units": "hour", "long_name": "Hour of day (UTC)"}
    
    ds["day_of_year"] = (["time"], time.dayofyear.values.astype(float))
    ds["day_of_year"].attrs = {"units": "day", "long_name": "Day of year"}
    
    ds["month"] = (["time"], time.month.values.astype(float))
    ds["month"].attrs = {"units": "month", "long_name": "Month"}
    
    # Approximate local solar time for Sierra Nevada (lon ~ -3.4°)
    # Local time ≈ UTC + lon/15 hours
    station_lon = ds.attrs.get("station_lon", -3.4)
    local_hour_values = (time.hour.values + station_lon / 15) % 24
    ds["local_hour"] = (["time"], local_hour_values)
    ds["local_hour"].attrs = {"units": "hour", "long_name": "Local solar time (approx)"}
    
    # Daytime flag (6-18 local time)
    is_daytime = (local_hour_values >= 6) & (local_hour_values < 18)
    ds["is_daytime"] = (["time"], is_daytime)
    ds["is_daytime"].attrs = {"long_name": "Daytime flag"}
    
    return ds
