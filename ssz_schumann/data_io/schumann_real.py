#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Schumann Resonance Data Loader

Loads Schumann resonance data from the Sierra Nevada ELF station
(Salinas et al., 2013-2017) from Zenodo.

Data Source:
    DOI: 10.5281/zenodo.6348930
    Period: March 2013 - February 2017
    Resolution: 10-minute intervals
    Location: Sierra Nevada, Spain (37.05°N, 3.38°W)

Reference:
    Salinas, A. et al. (2022). Computers & Geosciences, 165, 105148.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging
import requests
import zipfile
import io

logger = logging.getLogger(__name__)

# Zenodo DOI and URLs
ZENODO_DOI = "10.5281/zenodo.6348930"
ZENODO_RECORD_ID = "6348930"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Expected column mappings (may need adjustment based on actual file format)
COLUMN_MAPPINGS = {
    # Possible column names in source -> standardized names
    'f1': 'f1_Hz',
    'f2': 'f2_Hz', 
    'f3': 'f3_Hz',
    'freq1': 'f1_Hz',
    'freq2': 'f2_Hz',
    'freq3': 'f3_Hz',
    'frequency1': 'f1_Hz',
    'frequency2': 'f2_Hz',
    'frequency3': 'f3_Hz',
    'width1': 'width1_Hz',
    'width2': 'width2_Hz',
    'width3': 'width3_Hz',
    'bandwidth1': 'width1_Hz',
    'bandwidth2': 'width2_Hz',
    'bandwidth3': 'width3_Hz',
    'amp1': 'amp1',
    'amp2': 'amp2',
    'amp3': 'amp3',
    'amplitude1': 'amp1',
    'amplitude2': 'amp2',
    'amplitude3': 'amp3',
}

# Physical constraints for quality filtering
QUALITY_BOUNDS = {
    'f1_Hz': (6.5, 9.0),      # Mode 1: typically 7.83 Hz ± ~1 Hz
    'f2_Hz': (12.0, 16.0),    # Mode 2: typically 14.3 Hz
    'f3_Hz': (18.0, 24.0),    # Mode 3: typically 20.8 Hz
    'width1_Hz': (0.1, 3.0),  # Reasonable bandwidth range
    'width2_Hz': (0.1, 4.0),
    'width3_Hz': (0.1, 5.0),
}


def download_zenodo_data(
    output_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Download Schumann data from Zenodo.
    
    Args:
        output_dir: Directory to save downloaded files
        force_download: If True, re-download even if files exist
    
    Returns:
        Path to the downloaded/extracted data directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = output_dir / "zenodo_schumann"
    
    if data_dir.exists() and not force_download:
        logger.info(f"Data already exists at {data_dir}")
        return data_dir
    
    logger.info(f"Fetching Zenodo record metadata from {ZENODO_BASE_URL}")
    
    try:
        response = requests.get(ZENODO_BASE_URL, timeout=30)
        response.raise_for_status()
        record = response.json()
        
        # Find downloadable files
        files = record.get('files', [])
        if not files:
            raise ValueError("No files found in Zenodo record")
        
        logger.info(f"Found {len(files)} files in Zenodo record")
        
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in files:
            filename = file_info.get('key', '')
            download_url = file_info.get('links', {}).get('self', '')
            
            if not download_url:
                continue
            
            logger.info(f"Downloading {filename}...")
            
            file_response = requests.get(download_url, timeout=300, stream=True)
            file_response.raise_for_status()
            
            file_path = data_dir / filename
            
            # Handle zip files
            if filename.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(file_response.content)) as zf:
                    zf.extractall(data_dir)
                logger.info(f"Extracted {filename}")
            else:
                with open(file_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Saved {filename}")
        
        return data_dir
        
    except requests.RequestException as e:
        logger.error(f"Failed to download from Zenodo: {e}")
        raise


def load_schumann_real_data(
    data_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    apply_quality_filter: bool = True,
) -> pd.DataFrame:
    """
    Load real Schumann resonance data from local files.
    
    Args:
        data_path: Path to data file or directory
        start_date: Start date filter (ISO format)
        end_date: End date filter (ISO format)
        apply_quality_filter: If True, filter out bad data points
    
    Returns:
        DataFrame with standardized columns:
            - time (index): UTC timestamps
            - f1_Hz, f2_Hz, f3_Hz: Resonance frequencies
            - width1_Hz, width2_Hz, width3_Hz: Bandwidths
            - amp1, amp2, amp3: Amplitudes
            - quality_flag: 1=good, 0=suspect
    """
    data_path = Path(data_path)
    
    if data_path.is_dir():
        # Find data files in directory
        data_files = list(data_path.glob("*.csv")) + \
                     list(data_path.glob("*.txt")) + \
                     list(data_path.glob("*.dat"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found in {data_path}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load and concatenate all files
        dfs = []
        for file in sorted(data_files):
            try:
                df = _load_single_file(file)
                if df is not None and len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
        
        if not dfs:
            raise ValueError("No valid data loaded from any file")
        
        df = pd.concat(dfs, ignore_index=False)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        
    else:
        df = _load_single_file(data_path)
    
    # Apply date filters
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]
    
    # Apply quality filter
    if apply_quality_filter:
        df = _apply_quality_filter(df)
    
    logger.info(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
    
    return df


def _load_single_file(file_path: Path) -> pd.DataFrame:
    """Load a single data file and standardize columns."""
    file_path = Path(file_path)
    
    # Try different delimiters and formats
    for sep in [',', '\t', ';', ' ']:
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                parse_dates=True,
                index_col=0,
                encoding='utf-8',
            )
            
            if len(df.columns) >= 3:  # At least f1, f2, f3
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {file_path}")
    
    # Standardize column names
    df = _standardize_columns(df)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            df.index = pd.to_datetime(df.index)
    
    # Localize to UTC if not already
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    df.index.name = 'time'
    
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to expected format."""
    # Create mapping for this dataframe
    rename_map = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in COLUMN_MAPPINGS:
            rename_map[col] = COLUMN_MAPPINGS[col_lower]
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    required = ['f1_Hz', 'f2_Hz', 'f3_Hz']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        # Try to infer from column positions if names don't match
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            logger.warning(f"Required columns {missing} not found, inferring from positions")
            for i, col in enumerate(required):
                if col not in df.columns and i < len(numeric_cols):
                    df = df.rename(columns={numeric_cols[i]: col})
    
    return df


def _apply_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filtering based on physical constraints."""
    n_original = len(df)
    
    # Initialize quality flag
    df['quality_flag'] = 1
    
    # Check bounds for each column
    for col, (low, high) in QUALITY_BOUNDS.items():
        if col in df.columns:
            mask = (df[col] < low) | (df[col] > high) | df[col].isna()
            df.loc[mask, 'quality_flag'] = 0
    
    # Check for NaN in required columns
    required = ['f1_Hz', 'f2_Hz', 'f3_Hz']
    for col in required:
        if col in df.columns:
            df.loc[df[col].isna(), 'quality_flag'] = 0
    
    # Filter to good data only
    df_filtered = df[df['quality_flag'] == 1].copy()
    
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    
    if n_removed > 0:
        logger.info(f"Quality filter removed {n_removed} points ({100*n_removed/n_original:.1f}%)")
    
    return df_filtered


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics for Schumann data.
    
    Args:
        df: Schumann data DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_points': len(df),
        'start_date': str(df.index.min()),
        'end_date': str(df.index.max()),
        'duration_days': (df.index.max() - df.index.min()).days,
    }
    
    for col in ['f1_Hz', 'f2_Hz', 'f3_Hz']:
        if col in df.columns:
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()
            summary[f'{col}_min'] = df[col].min()
            summary[f'{col}_max'] = df[col].max()
    
    return summary


def create_sample_real_data(
    output_path: Path,
    n_days: int = 30,
    start_date: str = "2016-01-01",
) -> pd.DataFrame:
    """
    Create sample data mimicking real Schumann measurements.
    
    This is for testing when real data is not available.
    Uses realistic noise and diurnal/seasonal patterns.
    
    Args:
        output_path: Path to save sample data
        n_days: Number of days to generate
        start_date: Start date
    
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)
    
    # 10-minute resolution
    n_points = n_days * 24 * 6
    time = pd.date_range(start=start_date, periods=n_points, freq='10min', tz='UTC')
    
    # Base frequencies (classical values)
    f1_base = 7.83
    f2_base = 14.3
    f3_base = 20.8
    
    # Time in hours for diurnal variation
    t_hours = np.arange(n_points) / 6.0
    
    # Diurnal variation (ionosphere height changes)
    diurnal = 0.05 * np.sin(2 * np.pi * t_hours / 24)
    
    # Seasonal variation (27-day solar rotation)
    seasonal = 0.03 * np.sin(2 * np.pi * t_hours / (24 * 27))
    
    # Random noise
    noise1 = 0.08 * np.random.randn(n_points)
    noise2 = 0.12 * np.random.randn(n_points)
    noise3 = 0.15 * np.random.randn(n_points)
    
    # Generate frequencies with correlated variations (SSZ-like)
    common_variation = diurnal + seasonal
    
    df = pd.DataFrame({
        'f1_Hz': f1_base + common_variation * f1_base / 100 + noise1,
        'f2_Hz': f2_base + common_variation * f2_base / 100 + noise2,
        'f3_Hz': f3_base + common_variation * f3_base / 100 + noise3,
        'width1_Hz': 0.8 + 0.2 * np.random.randn(n_points),
        'width2_Hz': 1.0 + 0.3 * np.random.randn(n_points),
        'width3_Hz': 1.2 + 0.4 * np.random.randn(n_points),
        'amp1': 1.0 + 0.3 * np.random.randn(n_points),
        'amp2': 0.5 + 0.15 * np.random.randn(n_points),
        'amp3': 0.3 + 0.1 * np.random.randn(n_points),
        'quality_flag': 1,
    }, index=time)
    
    df.index.name = 'time'
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    
    logger.info(f"Created sample real data: {output_path}")
    
    return df


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    sample_path = Path("data/schumann/sample_real_2016.csv")
    df = create_sample_real_data(sample_path, n_days=30)
    
    print("\nSample Data Summary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        print(f"  {key}: {value}")
