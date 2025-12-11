#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process One Month of Zenodo Schumann Data

Extracts Schumann resonance frequencies (f1, f2, f3) from raw ELF data.
Uses FFT + peak detection in the 5-35 Hz range.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import sys
import zipfile
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from datetime import datetime
import pandas as pd
from pathlib import Path

# Configuration
ZIP_PATH = Path("data/schumann/real/raw/2013_2017.zip")
OUTPUT_DIR = Path("data/schumann/real/processed")
MONTH = "1310"  # October 2013 (YYMM format)
FS = 256.0  # Sampling rate in Hz

# Schumann frequency ranges for peak detection
SCHUMANN_RANGES = {
    'f1': (6.5, 9.0),    # ~7.83 Hz
    'f2': (13.0, 16.0),  # ~14.1 Hz
    'f3': (19.0, 22.0),  # ~20.3 Hz
    'f4': (25.0, 28.0),  # ~26.4 Hz
}


def lorentzian(f, f0, gamma, A, offset):
    """Lorentzian peak function for fitting."""
    return A * (gamma**2) / ((f - f0)**2 + gamma**2) + offset


def find_schumann_peak(freqs, psd, f_min, f_max):
    """
    Find Schumann peak in given frequency range using Lorentzian fit.
    
    Returns: (f_center, amplitude, width) or (None, None, None) if no peak found
    """
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_range = freqs[mask]
    psd_range = psd[mask]
    
    if len(f_range) < 5:
        return None, None, None
    
    # Initial guess
    idx_max = np.argmax(psd_range)
    f0_guess = f_range[idx_max]
    A_guess = psd_range[idx_max] - psd_range.min()
    gamma_guess = 0.5
    offset_guess = psd_range.min()
    
    try:
        popt, _ = curve_fit(
            lorentzian, f_range, psd_range,
            p0=[f0_guess, gamma_guess, A_guess, offset_guess],
            bounds=([f_min, 0.1, 0, 0], [f_max, 3.0, np.inf, np.inf]),
            maxfev=1000
        )
        f_center, gamma, amplitude, offset = popt
        return f_center, amplitude, gamma
    except:
        # Fallback: simple peak detection
        return f0_guess, psd_range[idx_max], 0.5


def process_file(data, fs=256.0):
    """
    Process one hour of ELF data.
    
    Returns dict with f1, f2, f3, f4 and their amplitudes.
    """
    # Compute PSD using Welch method
    freqs, psd = signal.welch(data, fs=fs, nperseg=8192, noverlap=4096)
    
    results = {}
    for mode, (f_min, f_max) in SCHUMANN_RANGES.items():
        f_center, amplitude, width = find_schumann_peak(freqs, psd, f_min, f_max)
        results[mode] = f_center
        results[f'{mode}_amp'] = amplitude
        results[f'{mode}_width'] = width
    
    return results


def parse_timestamp(info_content):
    """Parse timestamp from info file content."""
    for line in info_content.split('\n'):
        if '1st sample timestamp' in line:
            # Format: "1st sample timestamp: 10-10-2013 15:02:49.233 UTC"
            parts = line.split(': ', 1)
            if len(parts) == 2:
                ts_str = parts[1].replace(' UTC', '').strip()
                try:
                    return datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S.%f')
                except:
                    try:
                        return datetime.strptime(ts_str, '%d-%m-%Y %H:%M:%S')
                    except:
                        pass
    return None


def main():
    print("=" * 70)
    print("SCHUMANN RESONANCE EXTRACTION - One Month Processing")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Open ZIP file
    if not ZIP_PATH.exists():
        print(f"[ERROR] ZIP file not found: {ZIP_PATH}")
        return
    
    print(f"Opening: {ZIP_PATH}")
    z = zipfile.ZipFile(ZIP_PATH, 'r')
    
    # Find all files for the target month
    all_files = z.namelist()
    month_files = [f for f in all_files if f'/{MONTH}/' in f and '_info.txt' not in f 
                   and not f.endswith('/') and 'sensor_0' in f and '.DS_Store' not in f]
    
    print(f"Found {len(month_files)} data files for month {MONTH}")
    
    if len(month_files) == 0:
        print("[ERROR] No files found for this month")
        # List available months
        months = set()
        for f in all_files:
            parts = f.split('/')
            if len(parts) >= 2 and len(parts[1]) == 4:
                months.add(parts[1])
        print(f"Available months: {sorted(months)[:20]}...")
        return
    
    # Process files
    results = []
    processed = 0
    errors = 0
    
    for i, filepath in enumerate(sorted(month_files)):  # Process ALL files
        try:
            # Read data file
            with z.open(filepath) as f:
                data = np.frombuffer(f.read(), dtype=np.int16)
            
            # Read corresponding info file
            info_path = filepath + '_info.txt'
            timestamp = None
            if info_path in all_files:
                with z.open(info_path) as f:
                    info_content = f.read().decode('utf-8', errors='replace')
                    timestamp = parse_timestamp(info_content)
            
            # Process
            result = process_file(data, FS)
            result['timestamp'] = timestamp
            result['file'] = filepath
            results.append(result)
            processed += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(month_files)} files...")
                
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [WARN] Error processing {filepath}: {e}")
    
    z.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Save results
    output_file = OUTPUT_DIR / f"schumann_{MONTH}_processed.csv"
    df.to_csv(output_file, index=False)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Processed: {processed} files")
    print(f"Errors: {errors} files")
    print(f"Output: {output_file}")
    print()
    
    # Statistics
    print("Schumann Frequency Statistics:")
    for mode in ['f1', 'f2', 'f3', 'f4']:
        if mode in df.columns:
            vals = df[mode].dropna()
            if len(vals) > 0:
                print(f"  {mode}: mean={vals.mean():.3f} Hz, std={vals.std():.3f} Hz, n={len(vals)}")
    
    print()
    print("[OK] Processing complete!")


if __name__ == "__main__":
    main()
