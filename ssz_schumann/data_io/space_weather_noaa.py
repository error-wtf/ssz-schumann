#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Space Weather Data I/O - NOAA/GFZ Sources

Loads and processes space weather indices:
- F10.7 Solar Radio Flux (NOAA/LASP)
- Kp/Ap Geomagnetic Indices (GFZ Potsdam)

Data sources:
    F10.7: https://lasp.colorado.edu/lisird/data/noaa_radio_flux/
           https://psl.noaa.gov/data/timeseries/month/SOLAR/
    Kp/Ap: https://kp.gfz-potsdam.de/en/data
           ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Tuple
from datetime import datetime
import warnings
import logging
import urllib.request
import io

logger = logging.getLogger(__name__)

# =============================================================================
# F10.7 SOLAR FLUX
# =============================================================================

# Known F10.7 data URLs
F107_URLS = {
    "swpc_json": "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json",
    "psl_csv": "https://psl.noaa.gov/data/correlation/solar.data",
}


def load_f107(
    path_or_url: Union[str, Path, None] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """
    Load F10.7 solar radio flux data.
    
    The F10.7 index measures solar radio emission at 10.7 cm wavelength,
    serving as a proxy for solar EUV radiation and ionospheric ionization.
    
    Args:
        path_or_url: Path to local file or URL
            If None, attempts to fetch from NOAA SWPC
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        pd.Series with DatetimeIndex (daily values)
        Values in Solar Flux Units (sfu), 1 sfu = 10^-22 W/m²/Hz
    
    Example:
        >>> f107 = load_f107()
        >>> f107_2016 = f107["2016"]
    """
    if path_or_url is None:
        # Try to fetch from SWPC JSON API
        return _fetch_f107_swpc(start_date, end_date)
    
    path = Path(path_or_url) if not str(path_or_url).startswith("http") else None
    
    if path is not None and path.exists():
        # Load from local file
        return _load_f107_local(path, start_date, end_date)
    else:
        # Try as URL
        return _fetch_f107_url(str(path_or_url), start_date, end_date)


def _fetch_f107_swpc(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Fetch F10.7 from NOAA SWPC JSON API."""
    import json
    
    url = F107_URLS["swpc_json"]
    logger.info(f"Fetching F10.7 from {url}")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        raise ConnectionError(f"Failed to fetch F10.7 data: {e}")
    
    # Parse JSON data
    # Format: [{"time-tag": "YYYY-MM", "f10.7": value, ...}, ...]
    records = []
    for entry in data:
        try:
            time_tag = entry.get("time-tag", "")
            f107_val = entry.get("f10.7")
            
            if time_tag and f107_val is not None:
                # Parse YYYY-MM format
                dt = pd.Timestamp(time_tag + "-01")
                records.append({"time": dt, "f107": float(f107_val)})
        except (ValueError, TypeError):
            continue
    
    if not records:
        raise ValueError("No valid F10.7 data found in response")
    
    df = pd.DataFrame(records)
    df = df.set_index("time")
    series = df["f107"]
    series.name = "f107"
    
    # Filter by date range
    if start_date:
        series = series[series.index >= start_date]
    if end_date:
        series = series[series.index <= end_date]
    
    logger.info(f"Loaded {len(series)} F10.7 values")
    return series


def _load_f107_local(
    path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Load F10.7 from local file."""
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        # Try CSV
        df = pd.read_csv(path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Find time and value columns
    time_cols = ["time", "date", "datetime", "time-tag", "time_tag"]
    val_cols = ["f10.7", "f107", "flux", "solar_flux", "value"]
    
    time_col = None
    for col in time_cols:
        if col in df.columns:
            time_col = col
            break
    
    val_col = None
    for col in val_cols:
        if col in df.columns:
            val_col = col
            break
    
    if time_col is None or val_col is None:
        raise ValueError(f"Could not identify columns. Available: {list(df.columns)}")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    series = df[val_col].astype(float)
    series.name = "f107"
    
    # Filter
    if start_date:
        series = series[series.index >= start_date]
    if end_date:
        series = series[series.index <= end_date]
    
    return series


def _fetch_f107_url(
    url: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Fetch F10.7 from arbitrary URL."""
    logger.info(f"Fetching F10.7 from {url}")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read().decode("utf-8")
    except Exception as e:
        raise ConnectionError(f"Failed to fetch F10.7 data: {e}")
    
    # Try to parse as CSV
    df = pd.read_csv(io.StringIO(content))
    
    # Use same logic as local file
    df.columns = df.columns.str.lower().str.strip()
    
    time_cols = ["time", "date", "datetime", "time-tag"]
    val_cols = ["f10.7", "f107", "flux", "value"]
    
    time_col = next((c for c in time_cols if c in df.columns), None)
    val_col = next((c for c in val_cols if c in df.columns), None)
    
    if time_col is None or val_col is None:
        raise ValueError(f"Could not identify columns. Available: {list(df.columns)}")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    series = df[val_col].astype(float)
    series.name = "f107"
    
    if start_date:
        series = series[series.index >= start_date]
    if end_date:
        series = series[series.index <= end_date]
    
    return series


# =============================================================================
# Kp/Ap GEOMAGNETIC INDICES
# =============================================================================

def load_kp(
    path_or_url: Union[str, Path, None] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_type: str = "Kp",
) -> pd.Series:
    """
    Load Kp or Ap geomagnetic index data.
    
    The Kp index measures geomagnetic activity on a quasi-logarithmic
    scale from 0 to 9. Ap is the daily average derived from Kp.
    
    Args:
        path_or_url: Path to local file or URL
            If None, attempts to load from default location
        start_date: Optional start date filter
        end_date: Optional end date filter
        index_type: "Kp" (3-hourly) or "Ap" (daily)
    
    Returns:
        pd.Series with DatetimeIndex
        Kp: 3-hourly values (0-9 scale)
        Ap: Daily values (nT equivalent)
    
    Example:
        >>> kp = load_kp(index_type="Kp")
        >>> ap = load_kp(index_type="Ap")
    """
    if path_or_url is None:
        # Try default data directory
        default_path = Path("data/space_weather/kp_ap.csv")
        if default_path.exists():
            path_or_url = default_path
        else:
            raise FileNotFoundError(
                "No Kp data file specified and default not found. "
                "Please download from https://kp.gfz-potsdam.de/en/data"
            )
    
    path = Path(path_or_url)
    
    if not path.exists():
        raise FileNotFoundError(f"Kp data file not found: {path}")
    
    return _load_kp_local(path, start_date, end_date, index_type)


def _load_kp_local(
    path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_type: str = "Kp",
) -> pd.Series:
    """Load Kp/Ap from local file."""
    suffix = path.suffix.lower()
    
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".txt":
        # GFZ format is fixed-width
        df = _parse_gfz_kp_format(path)
    else:
        df = pd.read_csv(path)
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Find columns
    time_cols = ["time", "date", "datetime", "timestamp"]
    kp_cols = ["kp", "kp_index", "kp-index"]
    ap_cols = ["ap", "ap_index", "ap-index"]
    
    time_col = next((c for c in time_cols if c in df.columns), None)
    
    if index_type.lower() == "kp":
        val_col = next((c for c in kp_cols if c in df.columns), None)
    else:
        val_col = next((c for c in ap_cols if c in df.columns), None)
    
    if time_col is None:
        # Try to construct from year/month/day columns
        if all(c in df.columns for c in ["year", "month", "day"]):
            df["time"] = pd.to_datetime(df[["year", "month", "day"]])
            time_col = "time"
        else:
            raise ValueError(f"Could not identify time column. Available: {list(df.columns)}")
    
    if val_col is None:
        raise ValueError(f"Could not find {index_type} column. Available: {list(df.columns)}")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    series = df[val_col].astype(float)
    series.name = index_type.lower()
    
    # Filter
    if start_date:
        series = series[series.index >= start_date]
    if end_date:
        series = series[series.index <= end_date]
    
    return series


def _parse_gfz_kp_format(path: Path) -> pd.DataFrame:
    """
    Parse GFZ Potsdam Kp/Ap format.
    
    Format description: https://kp.gfz-potsdam.de/en/data
    
    Columns (fixed-width):
        Year, Month, Day, Days, Days_m, Bsr, dB,
        Kp1, Kp2, ..., Kp8 (8 x 3-hourly Kp values),
        ap1, ap2, ..., ap8 (8 x 3-hourly ap values),
        Ap (daily average)
    """
    records = []
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            try:
                parts = line.split()
                if len(parts) < 20:
                    continue
                
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                
                # Daily Ap is typically the last value
                ap_daily = float(parts[-1]) if len(parts) > 19 else np.nan
                
                # 8 Kp values (indices 7-14 typically)
                # Kp is given as 0, 0+, 1-, 1, 1+, ... encoded as integers
                # For simplicity, take mean
                kp_values = []
                for i in range(7, 15):
                    if i < len(parts):
                        try:
                            kp_values.append(float(parts[i]))
                        except ValueError:
                            pass
                
                kp_daily = np.mean(kp_values) if kp_values else np.nan
                
                records.append({
                    "year": year,
                    "month": month,
                    "day": day,
                    "kp": kp_daily,
                    "ap": ap_daily,
                })
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df[["year", "month", "day"]])
    
    return df


# =============================================================================
# TIME RESOLUTION CONVERSION
# =============================================================================

def to_time_resolution(
    series: pd.Series,
    freq: str = "1H",
    method: str = "ffill",
) -> pd.Series:
    """
    Convert time series to target resolution.
    
    Args:
        series: Input time series
        freq: Target frequency (pandas frequency string)
        method: Interpolation method
            - "ffill": Forward fill (step function)
            - "linear": Linear interpolation
            - "nearest": Nearest neighbor
    
    Returns:
        Resampled series
    
    Example:
        >>> f107_hourly = to_time_resolution(f107_daily, freq="1h", method="ffill")
    """
    # First, resample to target frequency
    if method == "ffill":
        # Forward fill is appropriate for indices that represent
        # a period (e.g., daily F10.7 valid for that day)
        resampled = series.resample(freq).ffill()
    elif method == "linear":
        # Linear interpolation
        resampled = series.resample(freq).interpolate(method="linear")
    elif method == "nearest":
        resampled = series.resample(freq).nearest()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return resampled


def normalize_index(
    series: pd.Series,
    method: str = "zscore",
) -> pd.Series:
    """
    Normalize space weather index.
    
    Args:
        series: Input series
        method: Normalization method
            - "zscore": (x - mean) / std
            - "minmax": (x - min) / (max - min)
            - "log": log10(x)
    
    Returns:
        Normalized series
    
    Example:
        >>> f107_norm = normalize_index(f107, method="zscore")
    """
    if method == "zscore":
        return (series - series.mean()) / series.std()
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    elif method == "log":
        return np.log10(series.clip(lower=1e-10))
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# SYNTHETIC DATA FOR TESTING
# =============================================================================

def create_synthetic_space_weather(
    start: str = "2016-01-01",
    end: str = "2016-12-31",
    freq: str = "1D",
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Create synthetic F10.7 and Kp data for testing.
    
    Args:
        start: Start date
        end: End date
        freq: Time frequency
        seed: Random seed
    
    Returns:
        Tuple of (f107_series, kp_series)
    """
    np.random.seed(seed)
    
    time = pd.date_range(start=start, end=end, freq=freq)
    n = len(time)
    
    # F10.7: Solar cycle variation + noise
    # Typical range: 70-250 sfu
    t_days = np.arange(n)
    
    # 11-year cycle (scaled to data range)
    cycle = 50 * np.sin(2 * np.pi * t_days / (365.25 * 11))
    
    # 27-day rotation
    rotation = 20 * np.sin(2 * np.pi * t_days / 27)
    
    # Noise
    noise = 10 * np.random.randn(n)
    
    f107 = 120 + cycle + rotation + noise
    f107 = np.clip(f107, 70, 300)
    
    f107_series = pd.Series(f107, index=time, name="f107")
    
    # Kp: Geomagnetic activity (0-9 scale)
    # Correlated with F10.7 but with more variability
    kp_base = 2 + 0.02 * (f107 - 120)  # Weak correlation with F10.7
    kp_noise = 1.5 * np.random.randn(n)
    kp = kp_base + kp_noise
    kp = np.clip(kp, 0, 9)
    
    kp_series = pd.Series(kp, index=time, name="kp")
    
    return f107_series, kp_series
