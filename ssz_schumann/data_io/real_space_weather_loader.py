#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Space Weather Data Loader

Robust loaders for real space weather indices:
- F10.7: Solar radio flux at 10.7 cm (sfu)
- Kp: Planetary geomagnetic index (0-9)
- Dst: Disturbance storm time index (nT) [optional]
- AE: Auroral electrojet index (nT) [optional]

Data Sources:
    - F10.7: NOAA SWPC, NRCan
    - Kp: GFZ Potsdam, NOAA SWPC
    - Dst: Kyoto WDC
    - AE: Kyoto WDC

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class SpaceWeatherConfig:
    """Configuration for loading space weather data."""
    path: str
    time_column: str
    value_column: str
    timezone: str = "UTC"
    
    @classmethod
    def from_dict(cls, d: dict) -> "SpaceWeatherConfig":
        """Create config from dictionary."""
        return cls(
            path=d.get("path", ""),
            time_column=d.get("time_column", "time"),
            value_column=d.get("value_column", "value"),
            timezone=d.get("timezone", "UTC"),
        )


def load_f107(
    path: Union[str, Path],
    time_column: str,
    value_column: str,
    timezone: str = "UTC",
    **kwargs,
) -> pd.Series:
    """
    Load real F10.7 solar flux data.

    The F10.7 index is the solar radio flux at 10.7 cm wavelength,
    measured in solar flux units (sfu, 1 sfu = 10^-22 W/m^2/Hz).
    It's a proxy for solar EUV radiation affecting the ionosphere.

    Parameters
    ----------
    path : str or Path
        Path to the data file (CSV).
    time_column : str
        Name of the time/date column.
    value_column : str
        Name of the F10.7 value column.
    timezone : str
        Timezone of timestamps (converted to UTC).
    **kwargs
        Additional arguments passed to pd.read_csv().

    Returns
    -------
    pd.Series
        F10.7 time series with DatetimeIndex (UTC).

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If required columns are missing.

    Examples
    --------
    >>> f107 = load_f107(
    ...     "data/space_weather/real/f107_daily.csv",
    ...     time_column="date",
    ...     value_column="f107"
    ... )
    >>> print(f107.head())
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"F10.7 data file not found: {path}\n"
            f"Download from: https://www.swpc.noaa.gov/products/solar-cycle-progression\n"
            f"Or: ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/"
        )
    
    # Handle different file formats
    suffix = path.suffix.lower()
    
    if suffix == ".json":
        df = _load_f107_json(path, time_column, value_column)
    elif suffix in [".csv", ".txt", ".dat"]:
        df = pd.read_csv(path, **kwargs)
    else:
        # Try CSV as fallback
        df = pd.read_csv(path, **kwargs)
    
    return _process_space_weather_df(df, time_column, value_column, timezone, "f107")


def load_kp(
    path: Union[str, Path],
    time_column: str,
    value_column: str,
    timezone: str = "UTC",
    **kwargs,
) -> pd.Series:
    """
    Load real Kp geomagnetic index data.

    The Kp index is a planetary 3-hour geomagnetic activity index
    ranging from 0 (quiet) to 9 (extremely disturbed).

    Parameters
    ----------
    path : str or Path
        Path to the data file (CSV or GFZ format).
    time_column : str
        Name of the time column.
    value_column : str
        Name of the Kp value column.
    timezone : str
        Timezone of timestamps (converted to UTC).
    **kwargs
        Additional arguments passed to pd.read_csv().

    Returns
    -------
    pd.Series
        Kp time series with DatetimeIndex (UTC).

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.

    Examples
    --------
    >>> kp = load_kp(
    ...     "data/space_weather/real/kp_3hourly.csv",
    ...     time_column="datetime",
    ...     value_column="kp"
    ... )
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Kp data file not found: {path}\n"
            f"Download from: https://kp.gfz-potsdam.de/en/data\n"
            f"Or: https://www.swpc.noaa.gov/products/planetary-k-index"
        )
    
    # Check for GFZ format
    suffix = path.suffix.lower()
    
    if suffix == ".txt" and "gfz" in path.name.lower():
        df = _load_kp_gfz_format(path)
        return _process_space_weather_df(df, "time", "kp", "UTC", "kp")
    else:
        df = pd.read_csv(path, **kwargs)
        return _process_space_weather_df(df, time_column, value_column, timezone, "kp")


def load_dst(
    path: Union[str, Path],
    time_column: str,
    value_column: str,
    timezone: str = "UTC",
    **kwargs,
) -> pd.Series:
    """
    Load Dst (Disturbance Storm Time) index data.

    The Dst index measures the intensity of the ring current
    during geomagnetic storms, in nanoTesla (nT).
    Negative values indicate storm activity.

    Parameters
    ----------
    path : str or Path
        Path to the data file.
    time_column : str
        Name of the time column.
    value_column : str
        Name of the Dst value column.
    timezone : str
        Timezone of timestamps.
    **kwargs
        Additional arguments for pd.read_csv().

    Returns
    -------
    pd.Series
        Dst time series with DatetimeIndex (UTC).
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Dst data file not found: {path}\n"
            f"Download from: https://wdc.kugi.kyoto-u.ac.jp/dstdir/"
        )
    
    df = pd.read_csv(path, **kwargs)
    return _process_space_weather_df(df, time_column, value_column, timezone, "dst")


def _process_space_weather_df(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    timezone: str,
    name: str,
) -> pd.Series:
    """Process DataFrame to standardized Series."""
    
    # Check columns exist
    if time_column not in df.columns:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Time column '{time_column}' not found.\n"
            f"Available columns: {available}..."
        )
    
    if value_column not in df.columns:
        available = ", ".join(df.columns[:10])
        raise ValueError(
            f"Value column '{value_column}' not found.\n"
            f"Available columns: {available}..."
        )
    
    # Parse time
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert timezone to UTC
    if timezone != "UTC":
        try:
            df[time_column] = df[time_column].dt.tz_localize(timezone).dt.tz_convert("UTC")
        except Exception:
            df[time_column] = df[time_column].dt.tz_convert("UTC")
    
    # Remove timezone info for compatibility
    if df[time_column].dt.tz is not None:
        df[time_column] = df[time_column].dt.tz_localize(None)
    
    # Create series
    series = pd.Series(
        df[value_column].values,
        index=pd.DatetimeIndex(df[time_column]),
        name=name,
    )
    
    # Sort by time
    series = series.sort_index()
    
    # Remove duplicates (keep first)
    series = series[~series.index.duplicated(keep="first")]
    
    logger.info(f"Loaded {name}: {len(series)} points, "
                f"{series.index[0]} to {series.index[-1]}")
    
    return series


def _load_f107_json(path: Path, time_column: str, value_column: str) -> pd.DataFrame:
    """Load F10.7 from NOAA SWPC JSON format."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # SWPC format has nested structure
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and "data" in data:
        df = pd.DataFrame(data["data"])
    else:
        df = pd.DataFrame(data)
    
    return df


def _load_kp_gfz_format(path: Path) -> pd.DataFrame:
    """
    Load Kp data from GFZ Potsdam format.
    
    GFZ format has fixed-width columns with 8 Kp values per day.
    """
    records = []
    
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            try:
                # Parse date (first 10 characters typically)
                parts = line.split()
                if len(parts) < 9:
                    continue
                
                # Try to parse as: YYYY MM DD Kp1 Kp2 ... Kp8
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                
                # 8 Kp values for 3-hour intervals
                for i, kp_str in enumerate(parts[3:11]):
                    try:
                        # Handle fractional Kp (e.g., "3+" = 3.33, "3-" = 2.67)
                        kp = _parse_kp_value(kp_str)
                        hour = i * 3
                        dt = pd.Timestamp(year=year, month=month, day=day, hour=hour)
                        records.append({"time": dt, "kp": kp})
                    except (ValueError, IndexError):
                        continue
                        
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(records)


def _parse_kp_value(kp_str: str) -> float:
    """Parse Kp value including +/- notation."""
    kp_str = kp_str.strip()
    
    if kp_str.endswith("+"):
        return float(kp_str[:-1]) + 0.33
    elif kp_str.endswith("-"):
        return float(kp_str[:-1]) - 0.33
    elif kp_str.endswith("o"):
        return float(kp_str[:-1])
    else:
        return float(kp_str)


def load_space_weather_from_config(
    config: Dict[str, Any],
) -> Dict[str, pd.Series]:
    """
    Load all space weather data from configuration.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with f107 and kp sections.
    
    Returns
    -------
    dict
        Dictionary with "f107" and "kp" Series.
    
    Examples
    --------
    >>> config = {
    ...     "f107": {"path": "f107.csv", "time_column": "date", "value_column": "f107"},
    ...     "kp": {"path": "kp.csv", "time_column": "time", "value_column": "kp"},
    ... }
    >>> data = load_space_weather_from_config(config)
    """
    result = {}
    
    if "f107" in config:
        cfg = config["f107"]
        result["f107"] = load_f107(
            path=cfg["path"],
            time_column=cfg.get("time_column", "time"),
            value_column=cfg.get("value_column", "f107"),
            timezone=cfg.get("timezone", "UTC"),
        )
    
    if "kp" in config:
        cfg = config["kp"]
        result["kp"] = load_kp(
            path=cfg["path"],
            time_column=cfg.get("time_column", "time"),
            value_column=cfg.get("value_column", "kp"),
            timezone=cfg.get("timezone", "UTC"),
        )
    
    if "dst" in config:
        cfg = config["dst"]
        result["dst"] = load_dst(
            path=cfg["path"],
            time_column=cfg.get("time_column", "time"),
            value_column=cfg.get("value_column", "dst"),
            timezone=cfg.get("timezone", "UTC"),
        )
    
    return result


def resample_to_match(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    method: str = "nearest",
) -> pd.Series:
    """
    Resample space weather data to match target time index.
    
    Parameters
    ----------
    series : pd.Series
        Space weather time series.
    target_index : pd.DatetimeIndex
        Target time index to match.
    method : str
        Interpolation method: "nearest", "linear", "ffill".
    
    Returns
    -------
    pd.Series
        Resampled series matching target index.
    """
    if method == "nearest":
        # Use merge_asof for nearest-neighbor matching
        target_df = pd.DataFrame(index=target_index)
        source_df = series.to_frame()
        
        merged = pd.merge_asof(
            target_df.reset_index(),
            source_df.reset_index(),
            left_on="index",
            right_on=series.index.name or "index",
            direction="nearest",
        )
        
        result = pd.Series(
            merged[series.name].values,
            index=target_index,
            name=series.name,
        )
    
    elif method == "linear":
        # Reindex with linear interpolation
        combined_index = series.index.union(target_index)
        reindexed = series.reindex(combined_index).interpolate(method="time")
        result = reindexed.reindex(target_index)
    
    elif method == "ffill":
        # Forward fill
        combined_index = series.index.union(target_index)
        reindexed = series.reindex(combined_index).ffill()
        result = reindexed.reindex(target_index)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


if __name__ == "__main__":
    # Quick test
    print("=== Space Weather Loader Test ===\n")
    
    # Test with existing files
    data_dir = Path(__file__).parent.parent.parent / "data" / "space_weather"
    
    # Try loading existing F10.7 data
    f107_path = data_dir / "f107_daily.csv"
    if f107_path.exists():
        try:
            f107 = load_f107(f107_path, "date", "f107")
            print(f"F10.7: {len(f107)} points")
            print(f"  Range: {f107.min():.1f} - {f107.max():.1f} sfu")
            print(f"  Mean: {f107.mean():.1f} sfu")
        except Exception as e:
            print(f"F10.7 load error: {e}")
    else:
        print(f"F10.7 file not found: {f107_path}")
    
    # Try loading existing Kp data
    kp_path = data_dir / "kp_daily.csv"
    if kp_path.exists():
        try:
            kp = load_kp(kp_path, "date", "kp")
            print(f"\nKp: {len(kp)} points")
            print(f"  Range: {kp.min():.1f} - {kp.max():.1f}")
            print(f"  Mean: {kp.mean():.1f}")
        except Exception as e:
            print(f"Kp load error: {e}")
