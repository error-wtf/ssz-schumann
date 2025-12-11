#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and Process Zenodo Sierra Nevada Schumann Data

Downloads the 26.5 GB ZIP file and processes it to extract f1, f2, f3 frequencies.

Usage:
    # Full download and process (takes hours)
    python scripts/download_and_process_zenodo.py --download --process
    
    # Just download
    python scripts/download_and_process_zenodo.py --download
    
    # Just process (if ZIP already downloaded)
    python scripts/download_and_process_zenodo.py --process
    
    # Process specific year
    python scripts/download_and_process_zenodo.py --process --year 2016

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""
import os
import sys
import argparse
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.optimize import curve_fit
import logging
import struct
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "schumann"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Zenodo URLs
ZENODO_RECORDS = {
    "2013_2017": {
        "record_id": "6348930",
        "filename": "2013_2017.zip",
        "size_gb": 26.5,
    },
    "2016_only": {
        "record_id": "6348838",
        "filename": "2016.zip",
        "size_gb": 26.5,
    },
}


def download_with_progress(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: Download URL
        output_path: Local file path
        chunk_size: Download chunk size
    
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed / 1e6 if elapsed > 0 else 0
                    
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        remaining = (total_size - downloaded) / (speed * 1e6) if speed > 0 else 0
                        print(f"\r  Progress: {pct:.1f}% | {downloaded/1e9:.2f}/{total_size/1e9:.2f} GB | "
                              f"{speed:.1f} MB/s | ETA: {remaining/60:.0f} min", end='', flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded/1e9:.2f} GB | {speed:.1f} MB/s", end='', flush=True)
        
        print()  # Newline
        logger.info(f"Download complete: {output_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def get_zenodo_download_url(record_id: str, filename: str) -> str:
    """Get direct download URL from Zenodo."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        r = requests.get(api_url, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        for f in data.get('files', []):
            if f['key'] == filename:
                return f['links']['self']
        
        # If exact filename not found, return first file
        if data.get('files'):
            return data['files'][0]['links']['self']
            
    except Exception as e:
        logger.error(f"Failed to get download URL: {e}")
    
    return None


def download_zenodo_data(dataset: str = "2013_2017") -> Path:
    """
    Download Zenodo Schumann data.
    
    Args:
        dataset: Which dataset to download
    
    Returns:
        Path to downloaded ZIP file
    """
    if dataset not in ZENODO_RECORDS:
        logger.error(f"Unknown dataset: {dataset}")
        return None
    
    info = ZENODO_RECORDS[dataset]
    
    logger.info("="*60)
    logger.info(f"DOWNLOADING ZENODO SCHUMANN DATA")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Size: {info['size_gb']} GB")
    logger.info(f"Record: {info['record_id']}")
    logger.info("")
    logger.info("WARNING: This will take a LONG time (hours)!")
    logger.info("Make sure you have enough disk space.")
    logger.info("")
    
    # Get download URL
    download_url = get_zenodo_download_url(info['record_id'], info['filename'])
    
    if not download_url:
        logger.error("Could not get download URL")
        return None
    
    logger.info(f"Download URL: {download_url}")
    
    # Output path
    output_path = RAW_DIR / info['filename']
    
    # Check if already exists
    if output_path.exists():
        existing_size = output_path.stat().st_size / 1e9
        logger.info(f"File already exists: {output_path} ({existing_size:.2f} GB)")
        
        if existing_size > info['size_gb'] * 0.9:
            logger.info("File appears complete, skipping download")
            return output_path
        else:
            logger.info("File appears incomplete, re-downloading")
    
    # Download
    logger.info(f"Starting download to: {output_path}")
    
    if download_with_progress(download_url, output_path):
        return output_path
    
    return None


def lorentzian(f, f0, gamma, A, offset):
    """Lorentzian function for peak fitting."""
    return A * (gamma/2)**2 / ((f - f0)**2 + (gamma/2)**2) + offset


def extract_schumann_from_spectrum(freqs, psd, mode_ranges=None):
    """
    Extract Schumann frequencies from power spectrum.
    
    Args:
        freqs: Frequency array
        psd: Power spectral density
        mode_ranges: Dict of (f_min, f_max) for each mode
    
    Returns:
        Dict with frequencies, widths, amplitudes
    """
    if mode_ranges is None:
        mode_ranges = {
            1: (6.5, 9.0),
            2: (12.5, 15.5),
            3: (18.5, 22.0),
            4: (24.5, 28.5),
        }
    
    results = {}
    
    for mode, (f_min, f_max) in mode_ranges.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        f_range = freqs[mask]
        s_range = psd[mask]
        
        if len(f_range) < 5:
            results[f'f{mode}'] = np.nan
            results[f'width{mode}'] = np.nan
            results[f'amp{mode}'] = np.nan
            continue
        
        try:
            # Find peak
            peak_idx = np.argmax(s_range)
            f_peak = f_range[peak_idx]
            
            # Fit Lorentzian
            p0 = [f_peak, 1.0, s_range[peak_idx], np.min(s_range)]
            bounds = ([f_min, 0.1, 0, 0], [f_max, 5.0, np.inf, np.inf])
            
            popt, _ = curve_fit(lorentzian, f_range, s_range, p0=p0, bounds=bounds, maxfev=1000)
            
            results[f'f{mode}'] = popt[0]
            results[f'width{mode}'] = popt[1]
            results[f'amp{mode}'] = popt[2]
            
        except Exception:
            results[f'f{mode}'] = f_peak if 'f_peak' in dir() else np.nan
            results[f'width{mode}'] = np.nan
            results[f'amp{mode}'] = np.nan
    
    return results


def read_sierra_nevada_dat(filepath: Path, fs: float = 100.0):
    """
    Read Sierra Nevada .dat file.
    
    The format is binary with 2-byte integers (little-endian).
    
    Args:
        filepath: Path to .dat file
        fs: Sampling frequency (Hz)
    
    Returns:
        numpy array of samples
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Try different formats
        # Format 1: 16-bit signed integers
        try:
            samples = np.frombuffer(data, dtype='<i2')
            if len(samples) > 0:
                return samples.astype(float)
        except:
            pass
        
        # Format 2: 32-bit floats
        try:
            samples = np.frombuffer(data, dtype='<f4')
            if len(samples) > 0:
                return samples.astype(float)
        except:
            pass
        
        # Format 3: ASCII
        try:
            samples = np.loadtxt(filepath)
            return samples
        except:
            pass
        
        logger.warning(f"Could not read {filepath}")
        return None
        
    except Exception as e:
        logger.warning(f"Error reading {filepath}: {e}")
        return None


def process_dat_file(filepath: Path, fs: float = 100.0):
    """
    Process a single .dat file to extract Schumann frequencies.
    
    Args:
        filepath: Path to .dat file
        fs: Sampling frequency
    
    Returns:
        Dict with timestamp and frequencies
    """
    # Read data
    data = read_sierra_nevada_dat(filepath, fs)
    
    if data is None or len(data) < fs * 60:  # Need at least 1 minute
        return None
    
    # Compute power spectrum
    nperseg = min(len(data), int(fs * 60))
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    
    # Extract frequencies
    result = extract_schumann_from_spectrum(freqs, psd)
    
    # Add timestamp from filename
    # Typical format: YYYYMMDD_HHMMSS.dat
    try:
        stem = filepath.stem
        if len(stem) >= 15:
            year = int(stem[0:4])
            month = int(stem[4:6])
            day = int(stem[6:8])
            hour = int(stem[9:11])
            minute = int(stem[11:13])
            second = int(stem[13:15])
            result['time'] = datetime(year, month, day, hour, minute, second)
        else:
            result['time'] = None
    except:
        result['time'] = None
    
    result['file'] = filepath.name
    
    return result


def process_zenodo_data(zip_path: Path, year: int = None, max_files: int = None) -> pd.DataFrame:
    """
    Process Zenodo ZIP file to extract Schumann frequencies.
    
    Args:
        zip_path: Path to ZIP file
        year: Only process specific year (optional)
        max_files: Maximum number of files to process (for testing)
    
    Returns:
        DataFrame with processed data
    """
    logger.info("="*60)
    logger.info("PROCESSING ZENODO SCHUMANN DATA")
    logger.info("="*60)
    logger.info(f"ZIP file: {zip_path}")
    
    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        return None
    
    # Extract to temp directory
    extract_dir = RAW_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting to: {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List files
            all_files = zf.namelist()
            dat_files = [f for f in all_files if f.endswith('.dat')]
            
            logger.info(f"Total files in ZIP: {len(all_files)}")
            logger.info(f"DAT files: {len(dat_files)}")
            
            # Filter by year if specified
            if year:
                dat_files = [f for f in dat_files if str(year) in f]
                logger.info(f"Files for year {year}: {len(dat_files)}")
            
            # Limit for testing
            if max_files:
                dat_files = dat_files[:max_files]
                logger.info(f"Processing first {max_files} files")
            
            # Extract files
            logger.info("Extracting files...")
            for f in dat_files:
                zf.extract(f, extract_dir)
            
    except zipfile.BadZipFile as e:
        logger.error(f"Bad ZIP file: {e}")
        return None
    
    # Find extracted .dat files
    extracted_files = list(extract_dir.rglob("*.dat"))
    logger.info(f"Extracted {len(extracted_files)} .dat files")
    
    if not extracted_files:
        logger.error("No .dat files found after extraction")
        return None
    
    # Process files
    logger.info("Processing files...")
    results = []
    
    for i, filepath in enumerate(extracted_files):
        if i % 100 == 0:
            logger.info(f"  Processing file {i+1}/{len(extracted_files)}")
        
        result = process_dat_file(filepath)
        if result:
            results.append(result)
    
    logger.info(f"Successfully processed {len(results)} files")
    
    if not results:
        logger.error("No results obtained")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    
    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if year:
        output_path = PROCESSED_DIR / f"schumann_real_{year}.csv"
    else:
        output_path = PROCESSED_DIR / "schumann_real_all.csv"
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total records: {len(df)}")
    
    if 'f1' in df.columns:
        logger.info(f"f1 range: {df['f1'].min():.2f} - {df['f1'].max():.2f} Hz")
    if 'f2' in df.columns:
        logger.info(f"f2 range: {df['f2'].min():.2f} - {df['f2'].max():.2f} Hz")
    if 'f3' in df.columns:
        logger.info(f"f3 range: {df['f3'].min():.2f} - {df['f3'].max():.2f} Hz")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Zenodo Schumann data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download only
    python download_and_process_zenodo.py --download
    
    # Process existing ZIP
    python download_and_process_zenodo.py --process
    
    # Download and process
    python download_and_process_zenodo.py --download --process
    
    # Process specific year
    python download_and_process_zenodo.py --process --year 2016
    
    # Test with few files
    python download_and_process_zenodo.py --process --max-files 100
        """
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Download data from Zenodo')
    parser.add_argument('--process', action='store_true',
                       help='Process downloaded data')
    parser.add_argument('--dataset', default='2013_2017',
                       choices=['2013_2017', '2016_only'],
                       help='Which dataset to download')
    parser.add_argument('--year', type=int, default=None,
                       help='Only process specific year')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum files to process (for testing)')
    parser.add_argument('--zip-path', type=str, default=None,
                       help='Path to existing ZIP file')
    
    args = parser.parse_args()
    
    if not args.download and not args.process:
        parser.print_help()
        return 1
    
    zip_path = None
    
    # Download
    if args.download:
        zip_path = download_zenodo_data(args.dataset)
        if not zip_path:
            logger.error("Download failed")
            return 1
    
    # Process
    if args.process:
        if args.zip_path:
            zip_path = Path(args.zip_path)
        elif not zip_path:
            # Look for existing ZIP
            info = ZENODO_RECORDS[args.dataset]
            zip_path = RAW_DIR / info['filename']
        
        if not zip_path or not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            logger.error("Use --download first or specify --zip-path")
            return 1
        
        df = process_zenodo_data(zip_path, year=args.year, max_files=args.max_files)
        
        if df is None:
            logger.error("Processing failed")
            return 1
    
    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
