# -*- coding: utf-8 -*-
"""
SSZ Multi-Station Analysis Framework

Provides tools for combining data from multiple Schumann resonance stations
to improve consistency checks and reduce systematic errors.

Known ELF/Schumann Stations:
- Sierra Nevada, Spain (Zenodo data)
- Mitzpe Ramon, Israel
- Nagycenk, Hungary
- Arrival Heights, Antarctica
- Syowa, Antarctica
- Onagawa, Japan

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StationInfo:
    """Information about an ELF/Schumann station."""
    name: str
    code: str
    latitude: float
    longitude: float
    altitude: float  # meters
    data_source: str
    data_format: str
    notes: str = ""


# Known Schumann resonance stations
KNOWN_STATIONS = {
    'sierra_nevada': StationInfo(
        name="Sierra Nevada ELF Station",
        code="SNV",
        latitude=37.0,
        longitude=-3.4,
        altitude=2500,
        data_source="Zenodo (doi:10.5281/zenodo.7761644)",
        data_format="int16 binary, 256 Hz",
        notes="Primary station for this analysis"
    ),
    'mitzpe_ramon': StationInfo(
        name="Mitzpe Ramon",
        code="MRM",
        latitude=30.6,
        longitude=34.8,
        altitude=900,
        data_source="Tel Aviv University",
        data_format="Various",
        notes="Middle East coverage"
    ),
    'nagycenk': StationInfo(
        name="Nagycenk Observatory",
        code="NCK",
        latitude=47.6,
        longitude=16.7,
        altitude=150,
        data_source="Hungarian Academy of Sciences",
        data_format="Various",
        notes="Long-term European record"
    ),
    'arrival_heights': StationInfo(
        name="Arrival Heights",
        code="AH",
        latitude=-77.8,
        longitude=166.7,
        altitude=200,
        data_source="Antarctica NZ",
        data_format="Various",
        notes="Antarctic coverage"
    ),
    'onagawa': StationInfo(
        name="Onagawa",
        code="ONG",
        latitude=38.4,
        longitude=141.5,
        altitude=50,
        data_source="Tohoku University",
        data_format="Various",
        notes="Japanese coverage"
    ),
}


@dataclass
class MultiStationResult:
    """Result of multi-station consistency analysis."""
    n_stations: int
    station_codes: List[str]
    delta_ssz_per_station: Dict[str, float]
    delta_ssz_errors: Dict[str, float]
    weighted_mean: float
    weighted_error: float
    chi_squared: float
    ndof: int
    is_consistent: bool
    consistency_p_value: float


def load_station_data(station_code: str, data_path: Path) -> Optional[pd.DataFrame]:
    """
    Load processed Schumann data for a station.
    
    Parameters
    ----------
    station_code : str
        Station code (e.g., 'SNV', 'NCK')
    data_path : Path
        Path to the processed CSV file
    
    Returns
    -------
    df : pd.DataFrame or None
        Loaded data or None if not found
    """
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    df['station'] = station_code
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def combine_station_data(station_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine data from multiple stations.
    
    Parameters
    ----------
    station_data : dict
        Dictionary mapping station codes to DataFrames
    
    Returns
    -------
    combined : pd.DataFrame
        Combined data with station column
    """
    dfs = []
    for code, df in station_data.items():
        df = df.copy()
        df['station'] = code
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def compute_station_delta_ssz(
    df: pd.DataFrame,
    classical_ref: Dict[str, float]
) -> Tuple[float, float]:
    """
    Compute delta_SSZ for a single station.
    
    Parameters
    ----------
    df : pd.DataFrame
        Station data with f1, f2, f3, f4 columns
    classical_ref : dict
        Classical reference frequencies
    
    Returns
    -------
    delta_ssz : float
        Weighted mean delta_SSZ
    delta_ssz_err : float
        Uncertainty
    """
    modes = ['f1', 'f2', 'f3', 'f4']
    deltas = []
    weights = []
    
    for mode in modes:
        if mode not in df.columns or mode not in classical_ref:
            continue
        
        f_obs = df[mode].mean()
        f_std = df[mode].std()
        f_class = classical_ref[mode]
        
        delta = -(f_obs - f_class) / f_class
        delta_err = f_std / f_class
        
        if delta_err > 0:
            deltas.append(delta)
            weights.append(1.0 / delta_err**2)
    
    if not deltas:
        return np.nan, np.nan
    
    weights = np.array(weights)
    deltas = np.array(deltas)
    
    delta_ssz = np.sum(weights * deltas) / np.sum(weights)
    delta_ssz_err = np.sqrt(1.0 / np.sum(weights))
    
    return delta_ssz, delta_ssz_err


def multi_station_consistency_test(
    station_deltas: Dict[str, float],
    station_errors: Dict[str, float]
) -> MultiStationResult:
    """
    Test consistency of delta_SSZ across multiple stations.
    
    If SSZ is a real global effect, all stations should measure
    the same delta_SSZ within uncertainties.
    
    Parameters
    ----------
    station_deltas : dict
        delta_SSZ per station
    station_errors : dict
        Uncertainties per station
    
    Returns
    -------
    result : MultiStationResult
        Consistency test results
    """
    from scipy import stats
    
    codes = list(station_deltas.keys())
    n_stations = len(codes)
    
    if n_stations < 2:
        return MultiStationResult(
            n_stations=n_stations,
            station_codes=codes,
            delta_ssz_per_station=station_deltas,
            delta_ssz_errors=station_errors,
            weighted_mean=list(station_deltas.values())[0] if station_deltas else np.nan,
            weighted_error=list(station_errors.values())[0] if station_errors else np.nan,
            chi_squared=0,
            ndof=0,
            is_consistent=True,
            consistency_p_value=1.0,
        )
    
    # Weighted mean
    weights = []
    values = []
    for code in codes:
        if station_errors[code] > 0:
            weights.append(1.0 / station_errors[code]**2)
            values.append(station_deltas[code])
    
    weights = np.array(weights)
    values = np.array(values)
    
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    weighted_error = np.sqrt(1.0 / np.sum(weights))
    
    # Chi-squared test
    chi2 = 0
    for code in codes:
        if station_errors[code] > 0:
            chi2 += ((station_deltas[code] - weighted_mean) / station_errors[code])**2
    
    ndof = n_stations - 1
    p_value = 1.0 - stats.chi2.cdf(chi2, ndof) if ndof > 0 else 1.0
    
    # Consistent if chi2/ndof < 2 and p > 0.05
    is_consistent = (chi2 / ndof < 2.0) and (p_value > 0.05) if ndof > 0 else True
    
    return MultiStationResult(
        n_stations=n_stations,
        station_codes=codes,
        delta_ssz_per_station=station_deltas,
        delta_ssz_errors=station_errors,
        weighted_mean=weighted_mean,
        weighted_error=weighted_error,
        chi_squared=chi2,
        ndof=ndof,
        is_consistent=is_consistent,
        consistency_p_value=p_value,
    )


def geographic_correlation_test(
    station_data: Dict[str, pd.DataFrame],
    station_info: Dict[str, StationInfo]
) -> Dict[str, float]:
    """
    Test for geographic correlations in delta_SSZ.
    
    If SSZ is global, there should be no correlation with latitude/longitude.
    Classical effects (ionospheric) may show geographic patterns.
    
    Parameters
    ----------
    station_data : dict
        Data per station
    station_info : dict
        Station metadata
    
    Returns
    -------
    correlations : dict
        Correlation coefficients with lat/lon
    """
    from scipy import stats
    
    lats = []
    lons = []
    deltas = []
    
    for code, df in station_data.items():
        if code not in station_info:
            continue
        
        info = station_info[code]
        delta = df['delta_ssz'].mean() if 'delta_ssz' in df.columns else np.nan
        
        if not np.isnan(delta):
            lats.append(info.latitude)
            lons.append(info.longitude)
            deltas.append(delta)
    
    if len(deltas) < 3:
        return {'lat_corr': np.nan, 'lon_corr': np.nan}
    
    lat_corr, lat_p = stats.pearsonr(lats, deltas)
    lon_corr, lon_p = stats.pearsonr(lons, deltas)
    
    return {
        'lat_corr': lat_corr,
        'lat_p': lat_p,
        'lon_corr': lon_corr,
        'lon_p': lon_p,
    }


def print_station_summary():
    """Print summary of known Schumann stations."""
    print("=" * 70)
    print("KNOWN SCHUMANN RESONANCE STATIONS")
    print("=" * 70)
    print()
    
    for code, info in KNOWN_STATIONS.items():
        print(f"{info.code}: {info.name}")
        print(f"    Location: {info.latitude:.1f}N, {info.longitude:.1f}E, {info.altitude}m")
        print(f"    Source: {info.data_source}")
        print(f"    Format: {info.data_format}")
        if info.notes:
            print(f"    Notes: {info.notes}")
        print()
