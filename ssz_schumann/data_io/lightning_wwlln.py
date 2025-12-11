#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Activity Data I/O - WWLLN

Loads and processes lightning activity data from the
World Wide Lightning Location Network (WWLLN).

Data source:
    http://wwlln.net/
    Thunder Hour dataset (gridded global lightning activity)

Note:
    WWLLN data requires registration/subscription for full access.
    This module provides the interface; actual data must be obtained
    separately.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Union, Optional, Tuple, List
import warnings
import logging

logger = logging.getLogger(__name__)


def load_thunder_hours(
    path_or_url: Union[str, Path],
    region: Optional[Tuple[float, float, float, float]] = None,
) -> xr.DataArray:
    """
    Load WWLLN Thunder Hour data.
    
    Thunder Hours represent the number of hours per month/day with
    detected lightning activity in each grid cell.
    
    Args:
        path_or_url: Path to data file (NetCDF, CSV, or HDF5)
        region: Optional bounding box (lat_min, lat_max, lon_min, lon_max)
            to subset the data
    
    Returns:
        xr.DataArray with dimensions (time, lat, lon)
        Values: Thunder hours count
    
    Example:
        >>> thunder = load_thunder_hours("data/lightning/wwlln_2016.nc")
        >>> thunder_europe = load_thunder_hours(
        ...     "data/lightning/wwlln_2016.nc",
        ...     region=(35, 70, -10, 40)
        ... )
    """
    path = Path(path_or_url)
    
    if not path.exists():
        raise FileNotFoundError(f"Lightning data file not found: {path}")
    
    suffix = path.suffix.lower()
    
    if suffix in [".nc", ".netcdf"]:
        ds = xr.open_dataset(path)
        # Find the main data variable
        var_candidates = ["thunder_hours", "lightning", "strokes", "flash_density"]
        var_name = None
        for var in var_candidates:
            if var in ds:
                var_name = var
                break
        if var_name is None:
            # Use first data variable
            var_name = list(ds.data_vars)[0]
        da = ds[var_name]
    
    elif suffix == ".csv":
        df = pd.read_csv(path)
        # Expect columns: time, lat, lon, value
        df.columns = df.columns.str.lower().str.strip()
        
        # Pivot to grid
        if "time" in df.columns and "lat" in df.columns and "lon" in df.columns:
            # Find value column
            val_col = [c for c in df.columns if c not in ["time", "lat", "lon"]][0]
            
            # Create multi-index and unstack
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index(["time", "lat", "lon"])
            da = df[val_col].to_xarray()
        else:
            raise ValueError(f"Expected columns: time, lat, lon, value. Got: {list(df.columns)}")
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Apply region filter
    if region is not None:
        lat_min, lat_max, lon_min, lon_max = region
        da = da.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
    
    da.name = "thunder_hours"
    da.attrs["units"] = "hours"
    da.attrs["long_name"] = "Thunder hours (lightning activity)"
    
    return da


def thunder_index_for_station(
    da: xr.DataArray,
    station_lat: float,
    station_lon: float,
    radius_deg: float = 5.0,
    method: str = "mean",
) -> pd.Series:
    """
    Calculate lightning activity index for a station location.
    
    Aggregates lightning data within a radius around the station.
    
    Args:
        da: Thunder hours DataArray (time, lat, lon)
        station_lat: Station latitude (degrees)
        station_lon: Station longitude (degrees)
        radius_deg: Radius in degrees for aggregation
        method: Aggregation method ("mean", "sum", "max")
    
    Returns:
        pd.Series with time index, lightning activity index
    
    Example:
        >>> thunder_idx = thunder_index_for_station(
        ...     thunder_da,
        ...     station_lat=37.07,
        ...     station_lon=-3.38,
        ...     radius_deg=5.0
        ... )
    """
    # Select region around station
    lat_min = station_lat - radius_deg
    lat_max = station_lat + radius_deg
    lon_min = station_lon - radius_deg
    lon_max = station_lon + radius_deg
    
    # Handle longitude wrapping
    if lon_min < -180:
        lon_min += 360
    if lon_max > 180:
        lon_max -= 360
    
    # Subset
    da_region = da.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    # Aggregate spatially
    if method == "mean":
        da_agg = da_region.mean(dim=["lat", "lon"])
    elif method == "sum":
        da_agg = da_region.sum(dim=["lat", "lon"])
    elif method == "max":
        da_agg = da_region.max(dim=["lat", "lon"])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to pandas Series
    series = da_agg.to_series()
    series.name = "thunder_index"
    
    return series


def create_synthetic_lightning(
    start: str = "2016-01-01",
    end: str = "2016-12-31",
    freq: str = "1D",
    lat_range: Tuple[float, float] = (30, 50),
    lon_range: Tuple[float, float] = (-20, 20),
    resolution: float = 1.0,
    seed: int = 42,
) -> xr.DataArray:
    """
    Create synthetic lightning activity data for testing.
    
    Generates realistic patterns:
    - Seasonal variation (summer maximum)
    - Diurnal variation (afternoon maximum)
    - Spatial variation (land vs ocean proxy)
    
    Args:
        start: Start date
        end: End date
        freq: Time frequency
        lat_range: Latitude range (min, max)
        lon_range: Longitude range (min, max)
        resolution: Grid resolution in degrees
        seed: Random seed
    
    Returns:
        xr.DataArray with synthetic thunder hours
    """
    np.random.seed(seed)
    
    # Create coordinates
    time = pd.date_range(start=start, end=end, freq=freq)
    lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    
    n_time = len(time)
    n_lat = len(lats)
    n_lon = len(lons)
    
    # Generate patterns
    # Seasonal: Summer maximum (Northern Hemisphere)
    day_of_year = time.dayofyear.values
    seasonal = 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 200) / 365))
    
    # Spatial: More activity at mid-latitudes, less over "ocean"
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    spatial = np.exp(-((lat_grid - 40)**2) / 200)
    
    # Combine
    data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        base = seasonal[t] * spatial
        noise = 0.3 * np.random.rand(n_lat, n_lon)
        data[t] = 10 * (base + noise)
    
    # Clip to realistic range
    data = np.clip(data, 0, 24)  # Max 24 hours per day
    
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": time,
            "lat": lats,
            "lon": lons,
        },
        name="thunder_hours",
        attrs={
            "units": "hours",
            "long_name": "Synthetic thunder hours",
            "synthetic": True,
        }
    )
    
    return da


def estimate_global_lightning_activity(
    time: pd.DatetimeIndex,
    method: str = "climatology",
) -> pd.Series:
    """
    Estimate global lightning activity without detailed data.
    
    Uses climatological patterns based on published research.
    
    Args:
        time: Time index
        method: Estimation method
            - "climatology": Seasonal + diurnal pattern
            - "constant": Constant value
    
    Returns:
        pd.Series with estimated global lightning index
    
    Reference:
        Christian et al. (2003) - Global frequency and distribution
        of lightning as observed from space
    """
    n = len(time)
    
    if method == "climatology":
        # Day of year (seasonal)
        doy = time.dayofyear.values
        
        # Global lightning peaks in Northern Hemisphere summer
        # (July-August) due to land mass distribution
        seasonal = 0.5 * (1 + np.cos(2 * np.pi * (doy - 200) / 365))
        
        # Hour of day (diurnal) - peak in afternoon UTC
        # Global average peaks around 14-18 UTC
        hour = time.hour.values
        diurnal = 0.5 * (1 + np.cos(2 * np.pi * (hour - 16) / 24))
        
        # Combine
        index = 50 * (0.7 * seasonal + 0.3 * diurnal)
        
        # Add noise
        index = index + 5 * np.random.randn(n)
        index = np.clip(index, 10, 100)
    
    elif method == "constant":
        # Global average: ~45 flashes/second
        index = np.full(n, 45.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    series = pd.Series(index, index=time, name="global_lightning_index")
    series.attrs = {"units": "flashes/second", "method": method}
    
    return series
