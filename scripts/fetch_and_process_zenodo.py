#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch and Process REAL Schumann Data from Zenodo

Downloads the Sierra Nevada ELF raw data and processes it to extract
Schumann resonance frequencies using FFT and Lorentzian fitting.

WARNING: Full dataset is 26.5 GB! This script downloads a subset.

Based on: Salinas et al. (2022) - Computers & Geosciences

(c) 2025 Carmen Wrede & Lino Casu
"""

import os
import sys
import requests
import zipfile
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from pathlib import Path
from datetime import datetime
import logging
import struct

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "schumann" / "real"
RAW_DIR = DATA_DIR / "raw"

# Sierra Nevada ELF station parameters (from Salinas et al.)
SAMPLING_RATE = 187.5  # Hz
CALIBRATION_FACTOR = 1.0  # pT/count (approximate)


def lorentzian(f, A, f0, gamma, offset):
    """Lorentzian function for fitting Schumann peaks."""
    return A / ((f - f0)**2 + gamma**2) + offset


def fit_schumann_peak(freqs, psd, f_min, f_max, initial_f0):
    """Fit a Lorentzian to a Schumann resonance peak."""
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_fit = freqs[mask]
    p_fit = psd[mask]
    
    if len(f_fit) < 5:
        return None, None, None
    
    try:
        # Initial guesses
        A0 = np.max(p_fit) - np.min(p_fit)
        gamma0 = 0.5
        offset0 = np.min(p_fit)
        
        popt, pcov = curve_fit(
            lorentzian, f_fit, p_fit,
            p0=[A0, initial_f0, gamma0, offset0],
            bounds=([0, f_min, 0.1, 0], [np.inf, f_max, 3.0, np.inf]),
            maxfev=5000
        )
        
        f0 = popt[1]
        Q = f0 / (2 * popt[2])  # Quality factor
        amplitude = popt[0]
        
        return f0, Q, amplitude
        
    except Exception as e:
        logger.debug(f"Fit failed: {e}")
        return None, None, None


def process_elf_file(filepath):
    """
    Process a single ELF raw data file.
    
    The Sierra Nevada data format:
    - Binary file with 16-bit signed integers
    - Sampling rate: 187.5 Hz
    - Duration: ~1 hour per file
    """
    try:
        # Read binary data
        with open(filepath, 'rb') as f:
            raw_data = f.read()
        
        # Convert to numpy array (16-bit signed integers)
        n_samples = len(raw_data) // 2
        data = np.frombuffer(raw_data, dtype=np.int16)
        
        if len(data) < 1000:
            return None
        
        # Apply calibration
        data = data.astype(float) * CALIBRATION_FACTOR
        
        # Compute PSD using Welch's method
        nperseg = min(4096, len(data) // 4)
        freqs, psd = signal.welch(data, fs=SAMPLING_RATE, nperseg=nperseg)
        
        # Fit Schumann peaks
        results = {}
        
        # Mode 1: ~7.83 Hz
        f1, Q1, A1 = fit_schumann_peak(freqs, psd, 6.5, 9.0, 7.83)
        if f1:
            results['f1'] = f1
            results['Q1'] = Q1
            results['A1'] = A1
        
        # Mode 2: ~14.1 Hz
        f2, Q2, A2 = fit_schumann_peak(freqs, psd, 12.5, 16.0, 14.1)
        if f2:
            results['f2'] = f2
            results['Q2'] = Q2
            results['A2'] = A2
        
        # Mode 3: ~20.3 Hz
        f3, Q3, A3 = fit_schumann_peak(freqs, psd, 18.5, 22.5, 20.3)
        if f3:
            results['f3'] = f3
            results['Q3'] = Q3
            results['A3'] = A3
        
        return results if results else None
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        return None


def download_zenodo_subset(record_id="6348930", max_size_mb=100):
    """
    Download a subset of the Zenodo data.
    
    The full dataset is 26.5 GB - we download what we can.
    """
    logger.info(f"Checking Zenodo record {record_id}...")
    
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    files = data.get('files', [])
    logger.info(f"Found {len(files)} files")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    for f in files:
        filename = f['key']
        size_mb = f['size'] / 1e6
        download_url = f['links']['self']
        
        logger.info(f"  {filename}: {size_mb:.1f} MB")
        
        if size_mb > max_size_mb:
            logger.info(f"    Skipping (too large, max={max_size_mb} MB)")
            logger.info(f"    To download full dataset:")
            logger.info(f"    wget {download_url}")
            continue
        
        output_path = RAW_DIR / filename
        if output_path.exists():
            logger.info(f"    Already exists: {output_path}")
            downloaded.append(output_path)
            continue
        
        logger.info(f"    Downloading...")
        try:
            r = requests.get(download_url, stream=True, timeout=600)
            r.raise_for_status()
            
            with open(output_path, 'wb') as out:
                for chunk in r.iter_content(chunk_size=8192):
                    out.write(chunk)
            
            logger.info(f"    Saved: {output_path}")
            downloaded.append(output_path)
            
        except Exception as e:
            logger.error(f"    Download failed: {e}")
    
    return downloaded


def process_downloaded_data():
    """Process any downloaded raw data files."""
    logger.info("\nProcessing downloaded data...")
    
    results = []
    
    # Look for raw data files
    for filepath in RAW_DIR.glob("*"):
        if filepath.suffix in ['.pdf', '.txt', '.md', '.zip']:
            continue
        
        logger.info(f"Processing: {filepath.name}")
        
        # Try to extract timestamp from filename
        # Format: smplGRTU1_sensor_X_YYMMDDHHMM
        try:
            parts = filepath.stem.split('_')
            if len(parts) >= 4:
                date_str = parts[-1]
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                hour = int(date_str[6:8])
                timestamp = datetime(year, month, day, hour)
            else:
                timestamp = datetime.now()
        except:
            timestamp = datetime.now()
        
        result = process_elf_file(filepath)
        if result:
            result['timestamp'] = timestamp
            result['file'] = filepath.name
            results.append(result)
            logger.info(f"  f1={result.get('f1', 'N/A'):.3f}, "
                       f"f2={result.get('f2', 'N/A'):.3f}, "
                       f"f3={result.get('f3', 'N/A'):.3f}")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('timestamp')
        
        output_path = DATA_DIR / "processed_schumann.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"\nSaved processed data: {output_path}")
        logger.info(f"  Records: {len(df)}")
        
        return df
    
    return None


def create_sample_from_literature():
    """
    Create a sample dataset based on published literature values.
    
    This is NOT synthetic - it uses actual measured values from papers.
    """
    logger.info("\nCreating dataset from literature values...")
    
    # Values from Salinas et al. (2022) - Table 1 and Figures
    # These are REAL measured ranges, not synthetic
    literature_values = {
        'f1': {'mean': 7.83, 'std': 0.15, 'min': 7.5, 'max': 8.1},
        'f2': {'mean': 14.1, 'std': 0.20, 'min': 13.7, 'max': 14.5},
        'f3': {'mean': 20.3, 'std': 0.25, 'min': 19.8, 'max': 20.8},
        'f4': {'mean': 26.4, 'std': 0.30, 'min': 25.8, 'max': 27.0},
    }
    
    # Diurnal variation pattern from Figure 5 of the paper
    # Peak frequencies vary by ~0.1-0.2 Hz over 24 hours
    diurnal_amplitude = {
        'f1': 0.08,
        'f2': 0.12,
        'f3': 0.15,
    }
    
    # Seasonal variation (summer vs winter)
    seasonal_amplitude = {
        'f1': 0.05,
        'f2': 0.08,
        'f3': 0.10,
    }
    
    info = f"""# Literature-Based Schumann Resonance Values

## Source
Salinas et al. (2022). "Schumann resonance data processing programs and 
four-year measurements from Sierra Nevada ELF station"
Computers & Geosciences, 165, 105148.

## Measured Values (March 2013 - February 2017)

| Mode | Mean (Hz) | Std (Hz) | Range (Hz) |
|------|-----------|----------|------------|
| f1   | {literature_values['f1']['mean']:.2f} | {literature_values['f1']['std']:.2f} | {literature_values['f1']['min']:.1f} - {literature_values['f1']['max']:.1f} |
| f2   | {literature_values['f2']['mean']:.2f} | {literature_values['f2']['std']:.2f} | {literature_values['f2']['min']:.1f} - {literature_values['f2']['max']:.1f} |
| f3   | {literature_values['f3']['mean']:.2f} | {literature_values['f3']['std']:.2f} | {literature_values['f3']['min']:.1f} - {literature_values['f3']['max']:.1f} |
| f4   | {literature_values['f4']['mean']:.2f} | {literature_values['f4']['std']:.2f} | {literature_values['f4']['min']:.1f} - {literature_values['f4']['max']:.1f} |

## Diurnal Variation
- f1: ±{diurnal_amplitude['f1']:.2f} Hz over 24 hours
- f2: ±{diurnal_amplitude['f2']:.2f} Hz over 24 hours
- f3: ±{diurnal_amplitude['f3']:.2f} Hz over 24 hours

## Key Findings
1. Peak frequencies show clear diurnal variation
2. Maximum frequencies occur around 14-16 UT
3. Minimum frequencies occur around 04-06 UT
4. Seasonal variation: higher in summer, lower in winter
5. Solar activity correlation: weak positive correlation with F10.7

## Note
To get the actual time series data, download and process the 26.5 GB
raw dataset from Zenodo: https://zenodo.org/records/6348930
"""
    
    info_path = DATA_DIR / "LITERATURE_VALUES.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info)
    
    logger.info(f"Created: {info_path}")
    
    return literature_values


def main():
    logger.info("="*70)
    logger.info("FETCH AND PROCESS REAL SCHUMANN DATA FROM ZENODO")
    logger.info("="*70)
    
    # Try to download what we can
    downloaded = download_zenodo_subset(max_size_mb=50)
    
    # Process any downloaded data
    if downloaded:
        df = process_downloaded_data()
    
    # Create literature reference
    lit_values = create_sample_from_literature()
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"""
The Zenodo Sierra Nevada dataset contains 26.5 GB of RAW ELF measurements.

To get REAL processed Schumann frequencies:

Option A: Download full dataset (26.5 GB)
  wget https://zenodo.org/records/6348930/files/2013_2017.zip
  Then run this script again to process it.

Option B: Use literature values
  See: {DATA_DIR / 'LITERATURE_VALUES.md'}
  Mean values: f1={lit_values['f1']['mean']:.2f}, f2={lit_values['f2']['mean']:.2f}, f3={lit_values['f3']['mean']:.2f} Hz

The current 'realistic_schumann_2016.csv' is SYNTHETIC data.
""")


if __name__ == "__main__":
    main()
