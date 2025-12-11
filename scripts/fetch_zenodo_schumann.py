#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zenodo Schumann Resonance Data Downloader

Downloads the Sierra Nevada ELF Station data from Zenodo.
Data covers March 2013 to February 2017.

Zenodo DOIs:
    - 10.5281/zenodo.6348838 (Year 2013)
    - 10.5281/zenodo.6348930 (Year 2014)
    - 10.5281/zenodo.6348958 (Year 2015)
    - 10.5281/zenodo.6348972 (Year 2016)

Reference:
    Salinas et al. (2022). "Schumann resonance data processing programs 
    and four-year measurements from Sierra Nevada ELF station"
    Computers & Geosciences, 165, 105148.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import Optional, List
import logging

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Zenodo record IDs for Sierra Nevada ELF data
ZENODO_RECORDS = {
    "2013": "6348838",
    "2014": "6348930",
    "2015": "6348958",
    "2016": "6348972",
}

# Alternative: Direct file URLs (if known)
# These are the processed CSV files with Schumann parameters
ZENODO_BASE_URL = "https://zenodo.org/api/records"


def get_zenodo_files(record_id: str) -> List[dict]:
    """
    Get list of files from a Zenodo record.
    
    Args:
        record_id: Zenodo record ID
    
    Returns:
        List of file info dicts with 'key', 'size', 'links'
    """
    url = f"{ZENODO_BASE_URL}/{record_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        files = data.get('files', [])
        logger.info(f"Found {len(files)} files in record {record_id}")
        
        return files
        
    except requests.RequestException as e:
        logger.error(f"Failed to get record {record_id}: {e}")
        return []


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: Download URL
        output_path: Local file path
        chunk_size: Download chunk size
    
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        pct = 100 * downloaded / total_size
                        print(f"\r  Downloading: {pct:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print()  # Newline after progress
        logger.info(f"Downloaded: {output_path}")
        return True
        
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def download_zenodo_record(
    record_id: str,
    output_dir: Path,
    file_pattern: Optional[str] = None,
) -> List[Path]:
    """
    Download all files from a Zenodo record.
    
    Args:
        record_id: Zenodo record ID
        output_dir: Output directory
        file_pattern: Optional pattern to filter files (e.g., '.csv')
    
    Returns:
        List of downloaded file paths
    """
    files = get_zenodo_files(record_id)
    downloaded = []
    
    for file_info in files:
        filename = file_info.get('key', '')
        
        # Filter by pattern if specified
        if file_pattern and file_pattern not in filename:
            continue
        
        # Get download URL
        download_url = file_info.get('links', {}).get('self')
        if not download_url:
            logger.warning(f"No download URL for {filename}")
            continue
        
        output_path = output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            logger.info(f"Already exists: {output_path}")
            downloaded.append(output_path)
            continue
        
        logger.info(f"Downloading {filename} ({file_info.get('size', 0)} bytes)...")
        
        if download_file(download_url, output_path):
            downloaded.append(output_path)
            
            # Extract if zip
            if filename.endswith('.zip'):
                extract_zip(output_path, output_dir)
    
    return downloaded


def extract_zip(zip_path: Path, output_dir: Path):
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
            logger.info(f"Extracted: {zip_path}")
    except zipfile.BadZipFile as e:
        logger.error(f"Failed to extract {zip_path}: {e}")


def download_all_years(
    output_dir: Path,
    years: Optional[List[str]] = None,
) -> dict:
    """
    Download Schumann data for all years.
    
    Args:
        output_dir: Base output directory
        years: List of years to download (default: all)
    
    Returns:
        Dict mapping year to list of downloaded files
    """
    if years is None:
        years = list(ZENODO_RECORDS.keys())
    
    results = {}
    
    for year in years:
        if year not in ZENODO_RECORDS:
            logger.warning(f"Unknown year: {year}")
            continue
        
        record_id = ZENODO_RECORDS[year]
        year_dir = output_dir / f"year_{year}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading Year {year} (Record: {record_id})")
        logger.info(f"{'='*60}")
        
        files = download_zenodo_record(record_id, year_dir)
        results[year] = files
    
    return results


def create_sample_from_zenodo(
    zenodo_dir: Path,
    output_path: Path,
    n_days: int = 30,
) -> Optional[Path]:
    """
    Create a sample dataset from downloaded Zenodo data.
    
    Args:
        zenodo_dir: Directory with Zenodo data
        output_path: Output CSV path
        n_days: Number of days to include
    
    Returns:
        Path to created sample file
    """
    import pandas as pd
    import glob
    
    # Find CSV files
    csv_files = list(zenodo_dir.glob("**/*.csv"))
    
    if not csv_files:
        logger.warning("No CSV files found in Zenodo directory")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Try to load and combine
    dfs = []
    for csv_file in csv_files[:5]:  # Limit to first 5 files
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows, columns: {list(df.columns)}")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
    
    if not dfs:
        logger.error("No data loaded")
        return None
    
    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    
    # Take sample
    if len(combined) > n_days * 24:
        combined = combined.head(n_days * 24)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Created sample: {output_path} ({len(combined)} rows)")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Schumann resonance data from Zenodo"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/zenodo_schumann",
        help="Output directory"
    )
    parser.add_argument(
        "--years", "-y",
        nargs="+",
        choices=["2013", "2014", "2015", "2016"],
        default=["2016"],
        help="Years to download"
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list available files, don't download"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset after download"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.list_only:
        # Just list files
        for year in args.years:
            record_id = ZENODO_RECORDS.get(year)
            if record_id:
                logger.info(f"\nYear {year} (Record {record_id}):")
                files = get_zenodo_files(record_id)
                for f in files:
                    size_mb = f.get('size', 0) / 1e6
                    logger.info(f"  - {f.get('key')} ({size_mb:.2f} MB)")
        return 0
    
    # Download
    logger.info("="*60)
    logger.info("ZENODO SCHUMANN DATA DOWNLOADER")
    logger.info("="*60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Years: {args.years}")
    
    results = download_all_years(output_dir, args.years)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    
    total_files = 0
    for year, files in results.items():
        logger.info(f"  Year {year}: {len(files)} files")
        total_files += len(files)
    
    logger.info(f"  Total: {total_files} files")
    
    # Create sample if requested
    if args.create_sample and total_files > 0:
        sample_path = output_dir / "sample_combined.csv"
        create_sample_from_zenodo(output_dir, sample_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
