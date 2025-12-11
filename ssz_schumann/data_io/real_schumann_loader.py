#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Schumann Resonance Data Loader

Robust loader for real Schumann resonance measurements from various sources.
Supports CSV and NetCDF formats with configurable column mappings.

Data Sources:
    - Sierra Nevada ELF Station (Zenodo: doi:10.5281/zenodo.6348930)
    - Other observatories with similar data formats

Expected Data Format:
    - Time series of central frequencies f1, f2, f3 (Hz)
    - Optional: Q-factors, amplitudes, widths
    - Time resolution: typically 10 min to 1 hour

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RealSchumannConfig:
    """Configuration for loading real Schumann data."""
    path: str
    time_column: str
    freq_columns: Dict[int, str]  # {mode: column_name}
    q_columns: Optional[Dict[int, str]] = None
    amp_columns: Optional[Dict[int, str]] = None
    width_columns: Optional[Dict[int, str]] = None
    timezone: str = "UTC"
    file_type: str = "auto"  # "auto", "csv", "netcdf"
    
    @classmethod
    def from_dict(cls, d: dict) -> "RealSchumannConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            path=d.get("path", ""),
            time_column=d.get("time_column", "time"),
            freq_columns=d.get("freq_columns", {1: "f1", 2: "f2", 3: "f3"}),
            q_columns=d.get("q_columns"),
            amp_columns=d.get("amp_columns"),
            width_columns=d.get("width_columns"),
            timezone=d.get("timezone", "UTC"),
            file_type=d.get("type", "auto"),
        )


def load_real_schumann_data(
    path: Union[str, Path],
    time_column: str,
    freq_columns: Dict[int, str],
    q_columns: Optional[Dict[int, str]] = None,
    amp_columns: Optional[Dict[int, str]] = None,
    width_columns: Optional[Dict[int, str]] = None,
    timezone: str = "UTC",
    **kwargs,
) -> xr.Dataset:
    """
    Load real Schumann resonance data from CSV or NetCDF.

    Parameters
    ----------
    path : str or Path
        Path to the data file (CSV or NetCDF).
    time_column : str
        Name of the time column (for CSV) or coordinate (for NetCDF).
    freq_columns : dict
        Mapping {mode_index: column_name} for central frequencies (Hz).
        Example: {1: "f1", 2: "f2", 3: "f3"}
    q_columns : dict, optional
        Mapping {mode_index: column_name} for Q-factors.
    amp_columns : dict, optional
        Mapping {mode_index: column_name} for amplitudes.
    width_columns : dict, optional
        Mapping {mode_index: column_name} for resonance widths.
    timezone : str
        Timezone of the timestamps. Converted internally to UTC.
    **kwargs
        Additional arguments passed to pandas/xarray read functions.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ('time') and variables:
        - f_mode1, f_mode2, f_mode3, ... (Hz)
        - q_mode1, q_mode2, ... (dimensionless, if provided)
        - amp_mode1, amp_mode2, ... (if provided)
        - width_mode1, width_mode2, ... (if provided)

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If required columns are missing or data format is invalid.

    Examples
    --------
    >>> ds = load_real_schumann_data(
    ...     "data/schumann/real/sierra_nevada_2017.csv",
    ...     time_column="datetime",
    ...     freq_columns={1: "f1_hz", 2: "f2_hz", 3: "f3_hz"},
    ...     q_columns={1: "Q1", 2: "Q2", 3: "Q3"},
    ...     timezone="UTC"
    ... )
    >>> print(ds)
    """
    path = Path(path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Schumann data file not found: {path}\n"
            f"Please download real data and place it at this location.\n"
            f"See docs/QUICKSTART.md for data sources."
        )
    
    # Determine file type
    suffix = path.suffix.lower()
    
    if suffix in [".csv", ".txt", ".dat"]:
        ds = _load_csv_schumann(
            path, time_column, freq_columns, q_columns, 
            amp_columns, width_columns, timezone, **kwargs
        )
    elif suffix in [".nc", ".netcdf", ".nc4"]:
        ds = _load_netcdf_schumann(
            path, time_column, freq_columns, q_columns,
            amp_columns, width_columns, timezone, **kwargs
        )
    else:
        # Try CSV as fallback
        logger.warning(f"Unknown file extension '{suffix}', trying CSV loader")
        ds = _load_csv_schumann(
            path, time_column, freq_columns, q_columns,
            amp_columns, width_columns, timezone, **kwargs
        )
    
    # Add metadata
    ds.attrs["source_file"] = str(path)
    ds.attrs["data_type"] = "real"
    ds.attrs["original_timezone"] = timezone
    
    # Log summary
    n_points = len(ds.time)
    time_range = f"{ds.time.values[0]} to {ds.time.values[-1]}"
    logger.info(f"Loaded real Schumann data: {n_points} points, {time_range}")
    
    return ds


def _load_csv_schumann(
    path: Path,
    time_column: str,
    freq_columns: Dict[int, str],
    q_columns: Optional[Dict[int, str]],
    amp_columns: Optional[Dict[int, str]],
    width_columns: Optional[Dict[int, str]],
    timezone: str,
    **kwargs,
) -> xr.Dataset:
    """Load Schumann data from CSV file."""
    
    # Read CSV
    df = pd.read_csv(path, **kwargs)
    
    # Check time column exists
    if time_column not in df.columns:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Time column '{time_column}' not found in CSV.\n"
            f"Available columns: {available}...\n"
            f"Please check your configuration."
        )
    
    # Parse time
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert to UTC if needed
    if timezone != "UTC":
        try:
            df[time_column] = df[time_column].dt.tz_localize(timezone).dt.tz_convert("UTC")
        except Exception:
            # Already has timezone info
            df[time_column] = df[time_column].dt.tz_convert("UTC")
    
    # Remove timezone info for xarray compatibility
    if df[time_column].dt.tz is not None:
        df[time_column] = df[time_column].dt.tz_localize(None)
    
    df = df.set_index(time_column)
    df.index.name = "time"
    
    # Build dataset
    data_vars = {}
    
    # Frequency columns (required)
    for mode, col_name in freq_columns.items():
        if col_name not in df.columns:
            raise ValueError(
                f"Frequency column '{col_name}' for mode {mode} not found.\n"
                f"Available columns: {', '.join(df.columns[:10])}..."
            )
        data_vars[f"f_mode{mode}"] = (["time"], df[col_name].values)
    
    # Q-factor columns (optional)
    if q_columns:
        for mode, col_name in q_columns.items():
            if col_name in df.columns:
                data_vars[f"q_mode{mode}"] = (["time"], df[col_name].values)
            else:
                logger.warning(f"Q-factor column '{col_name}' not found, skipping")
    
    # Amplitude columns (optional)
    if amp_columns:
        for mode, col_name in amp_columns.items():
            if col_name in df.columns:
                data_vars[f"amp_mode{mode}"] = (["time"], df[col_name].values)
    
    # Width columns (optional)
    if width_columns:
        for mode, col_name in width_columns.items():
            if col_name in df.columns:
                data_vars[f"width_mode{mode}"] = (["time"], df[col_name].values)
    
    # Create dataset
    ds = xr.Dataset(
        data_vars,
        coords={"time": df.index.values},
    )
    
    # Add variable attributes
    for mode in freq_columns.keys():
        ds[f"f_mode{mode}"].attrs = {
            "units": "Hz",
            "long_name": f"Schumann resonance frequency mode {mode}",
            "mode": mode,
        }
    
    return ds


def _load_netcdf_schumann(
    path: Path,
    time_column: str,
    freq_columns: Dict[int, str],
    q_columns: Optional[Dict[int, str]],
    amp_columns: Optional[Dict[int, str]],
    width_columns: Optional[Dict[int, str]],
    timezone: str,
    **kwargs,
) -> xr.Dataset:
    """Load Schumann data from NetCDF file."""
    
    # Open dataset
    ds_raw = xr.open_dataset(path, **kwargs)
    
    # Rename time coordinate if needed
    if time_column != "time" and time_column in ds_raw.coords:
        ds_raw = ds_raw.rename({time_column: "time"})
    
    # Build new dataset with standardized variable names
    data_vars = {}
    
    # Frequency variables
    for mode, var_name in freq_columns.items():
        if var_name in ds_raw:
            data_vars[f"f_mode{mode}"] = ds_raw[var_name]
        else:
            raise ValueError(
                f"Frequency variable '{var_name}' for mode {mode} not found.\n"
                f"Available variables: {list(ds_raw.data_vars)}"
            )
    
    # Q-factor variables
    if q_columns:
        for mode, var_name in q_columns.items():
            if var_name in ds_raw:
                data_vars[f"q_mode{mode}"] = ds_raw[var_name]
    
    # Amplitude variables
    if amp_columns:
        for mode, var_name in amp_columns.items():
            if var_name in ds_raw:
                data_vars[f"amp_mode{mode}"] = ds_raw[var_name]
    
    # Width variables
    if width_columns:
        for mode, var_name in width_columns.items():
            if var_name in ds_raw:
                data_vars[f"width_mode{mode}"] = ds_raw[var_name]
    
    # Create standardized dataset
    ds = xr.Dataset(data_vars)
    
    # Handle timezone conversion
    if timezone != "UTC":
        # Convert time coordinate
        time_values = pd.to_datetime(ds.time.values)
        try:
            time_values = time_values.tz_localize(timezone).tz_convert("UTC").tz_localize(None)
        except Exception:
            time_values = time_values.tz_convert("UTC").tz_localize(None)
        ds = ds.assign_coords(time=time_values)
    
    ds_raw.close()
    
    return ds


def load_real_schumann_from_config(config: Union[dict, RealSchumannConfig]) -> xr.Dataset:
    """
    Load real Schumann data using configuration dictionary or object.
    
    Parameters
    ----------
    config : dict or RealSchumannConfig
        Configuration with path and column mappings.
    
    Returns
    -------
    xr.Dataset
        Standardized Schumann dataset.
    
    Examples
    --------
    >>> config = {
    ...     "path": "data/schumann/real/data.csv",
    ...     "time_column": "datetime",
    ...     "freq_columns": {1: "f1", 2: "f2", 3: "f3"},
    ... }
    >>> ds = load_real_schumann_from_config(config)
    """
    if isinstance(config, dict):
        config = RealSchumannConfig.from_dict(config)
    
    return load_real_schumann_data(
        path=config.path,
        time_column=config.time_column,
        freq_columns=config.freq_columns,
        q_columns=config.q_columns,
        amp_columns=config.amp_columns,
        width_columns=config.width_columns,
        timezone=config.timezone,
    )


def validate_schumann_data(ds: xr.Dataset, modes: List[int] = [1, 2, 3]) -> dict:
    """
    Validate loaded Schumann data for quality and completeness.
    
    Parameters
    ----------
    ds : xr.Dataset
        Loaded Schumann dataset.
    modes : list
        Mode numbers to validate.
    
    Returns
    -------
    dict
        Validation results with statistics and warnings.
    """
    results = {
        "valid": True,
        "warnings": [],
        "statistics": {},
    }
    
    for mode in modes:
        var_name = f"f_mode{mode}"
        
        if var_name not in ds:
            results["valid"] = False
            results["warnings"].append(f"Missing frequency data for mode {mode}")
            continue
        
        data = ds[var_name].values
        
        # Statistics
        stats = {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "min": float(np.nanmin(data)),
            "max": float(np.nanmax(data)),
            "nan_fraction": float(np.isnan(data).sum() / len(data)),
        }
        results["statistics"][f"mode{mode}"] = stats
        
        # Warnings
        if stats["nan_fraction"] > 0.1:
            results["warnings"].append(
                f"Mode {mode}: {stats['nan_fraction']*100:.1f}% missing values"
            )
        
        # Physical range checks
        expected_ranges = {
            1: (6.0, 10.0),
            2: (12.0, 16.0),
            3: (18.0, 24.0),
        }
        
        if mode in expected_ranges:
            low, high = expected_ranges[mode]
            if stats["mean"] < low or stats["mean"] > high:
                results["warnings"].append(
                    f"Mode {mode}: Mean {stats['mean']:.2f} Hz outside expected range [{low}, {high}] Hz"
                )
    
    return results


def convert_to_standard_format(
    ds: xr.Dataset,
    output_path: Optional[Union[str, Path]] = None,
) -> xr.Dataset:
    """
    Convert loaded data to standard SSZ Schumann format.
    
    Renames variables to match the synthetic data format:
    - f_mode1 -> f1
    - f_mode2 -> f2
    - f_mode3 -> f3
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with f_mode* variables.
    output_path : str or Path, optional
        If provided, save converted dataset to this path.
    
    Returns
    -------
    xr.Dataset
        Dataset with standard variable names.
    """
    rename_map = {}
    
    for var in ds.data_vars:
        if var.startswith("f_mode"):
            mode = var.replace("f_mode", "")
            rename_map[var] = f"f{mode}"
        elif var.startswith("q_mode"):
            mode = var.replace("q_mode", "")
            rename_map[var] = f"q{mode}"
        elif var.startswith("amp_mode"):
            mode = var.replace("amp_mode", "")
            rename_map[var] = f"amp{mode}"
        elif var.startswith("width_mode"):
            mode = var.replace("width_mode", "")
            rename_map[var] = f"width{mode}"
    
    ds_standard = ds.rename(rename_map)
    
    if output_path:
        ds_standard.to_netcdf(output_path)
        logger.info(f"Saved standardized data to {output_path}")
    
    return ds_standard


if __name__ == "__main__":
    # Quick test
    print("=== Real Schumann Loader Test ===\n")
    
    # Test with example configuration
    config = {
        "path": "data/schumann/realistic_schumann_2016.csv",
        "time_column": "time",
        "freq_columns": {1: "f1", 2: "f2", 3: "f3"},
    }
    
    try:
        ds = load_real_schumann_from_config(config)
        print(f"Loaded dataset:\n{ds}")
        
        # Validate
        validation = validate_schumann_data(ds)
        print(f"\nValidation: {validation}")
        
    except FileNotFoundError as e:
        print(f"Test file not found (expected): {e}")
