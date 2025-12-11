#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process All Available Months from Zenodo Schumann Data

Extracts Schumann resonance frequencies from all available months (2013-2017).

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

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Configuration
ZIP_PATH = Path("data/schumann/real/raw/2013_2017.zip")
OUTPUT_DIR = Path("data/schumann/real/processed")
FS = 256.0  # Sampling rate in Hz

# Schumann frequency ranges for peak detection
SCHUMANN_RANGES = {
    'f1': (6.5, 9.0),
    'f2': (13.0, 16.0),
    'f3': (19.0, 22.0),
    'f4': (25.0, 28.0),
}


def lorentzian(f, f0, gamma, A, offset):
    """Lorentzian peak function for fitting."""
    return A * (gamma**2) / ((f - f0)**2 + gamma**2) + offset


def find_schumann_peak(freqs, psd, f_min, f_max):
    """Find Schumann peak in given frequency range using Lorentzian fit."""
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_range = freqs[mask]
    psd_range = psd[mask]
    
    if len(f_range) < 5:
        return None, None, None
    
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
        return f0_guess, psd_range[idx_max], 0.5


def process_file(data, fs=256.0):
    """Process one hour of ELF data."""
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


def get_available_months(z):
    """Get list of available months in the ZIP file."""
    all_files = z.namelist()
    months = set()
    for f in all_files:
        parts = f.split('/')
        if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isdigit():
            months.add(parts[1])
    return sorted(months)


def process_month(z, month, all_files):
    """Process all files for a given month."""
    month_files = [f for f in all_files if f'/{month}/' in f and '_info.txt' not in f 
                   and not f.endswith('/') and 'sensor_0' in f and '.DS_Store' not in f]
    
    if len(month_files) == 0:
        return None, 0, 0
    
    results = []
    processed = 0
    errors = 0
    
    for filepath in sorted(month_files):
        try:
            with z.open(filepath) as f:
                data = np.frombuffer(f.read(), dtype=np.int16)
            
            info_path = filepath + '_info.txt'
            timestamp = None
            if info_path in all_files:
                with z.open(info_path) as f:
                    info_content = f.read().decode('utf-8', errors='replace')
                    timestamp = parse_timestamp(info_content)
            
            result = process_file(data, FS)
            result['timestamp'] = timestamp
            result['file'] = filepath
            result['month'] = month
            results.append(result)
            processed += 1
            
        except Exception as e:
            errors += 1
    
    if results:
        df = pd.DataFrame(results)
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        return df, processed, errors
    
    return None, processed, errors


def main():
    print("=" * 70)
    print("SCHUMANN RESONANCE EXTRACTION - All Months")
    print("=" * 70)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not ZIP_PATH.exists():
        print(f"[ERROR] ZIP file not found: {ZIP_PATH}")
        return
    
    print(f"Opening: {ZIP_PATH}")
    z = zipfile.ZipFile(ZIP_PATH, 'r')
    all_files = z.namelist()
    
    available_months = get_available_months(z)
    print(f"Available months: {len(available_months)}")
    print(f"  {available_months[:5]}...{available_months[-5:]}")
    print()
    
    all_results = []
    summary = []
    
    for i, month in enumerate(available_months):
        print(f"[{i+1}/{len(available_months)}] Processing month {month}...")
        
        df, processed, errors = process_month(z, month, all_files)
        
        if df is not None and len(df) > 0:
            # Save individual month file
            output_file = OUTPUT_DIR / f"schumann_{month}_processed.csv"
            df.to_csv(output_file, index=False)
            
            all_results.append(df)
            
            # Statistics
            f1_mean = df['f1'].mean()
            f1_std = df['f1'].std()
            
            summary.append({
                'month': month,
                'n_records': len(df),
                'f1_mean': f1_mean,
                'f1_std': f1_std,
                'f2_mean': df['f2'].mean(),
                'f3_mean': df['f3'].mean(),
                'f4_mean': df['f4'].mean(),
                'errors': errors,
            })
            
            print(f"    -> {processed} files, f1={f1_mean:.3f}+/-{f1_std:.3f} Hz")
        else:
            print(f"    -> No data")
    
    z.close()
    
    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "schumann_all_months_processed.csv", index=False)
        print(f"\n[OK] Saved combined file: {OUTPUT_DIR / 'schumann_all_months_processed.csv'}")
        print(f"     Total records: {len(combined)}")
    
    # Save summary
    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUTPUT_DIR / "schumann_monthly_summary.csv", index=False)
        print(f"\n[OK] Saved summary: {OUTPUT_DIR / 'schumann_monthly_summary.csv'}")
        
        print("\n" + "=" * 70)
        print("MONTHLY SUMMARY")
        print("=" * 70)
        print(f"{'Month':<8} {'Records':<10} {'f1 (Hz)':<15} {'f2 (Hz)':<15}")
        print("-" * 50)
        for row in summary:
            print(f"{row['month']:<8} {row['n_records']:<10} "
                  f"{row['f1_mean']:.3f}+/-{row['f1_std']:.3f}  "
                  f"{row['f2_mean']:.3f}")
    
    print("\n[OK] Processing complete!")


if __name__ == "__main__":
    main()
