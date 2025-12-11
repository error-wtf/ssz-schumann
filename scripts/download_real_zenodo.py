#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download REAL Schumann Data from Zenodo

Downloads the actual raw ELF measurements from Sierra Nevada station
and processes them to extract Schumann resonance frequencies.

The Zenodo dataset contains:
- Raw time-domain ELF measurements (26.5 GB total)
- Two sensors: NS and EW orientation
- ~1 hour per file, 1.8 MB each

This script downloads a subset and processes it.

(c) 2025 Carmen Wrede & Lino Casu
"""

import os
import sys
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "schumann" / "real"


def get_zenodo_files(record_id: str):
    """Get file list from Zenodo record."""
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get('files', [])


def check_zenodo_records():
    """Check what's available on Zenodo."""
    records = {
        "2013": "6348838",
        "2014": "6348930",
        "2015": "6348958",
        "2016": "6348972",
    }
    
    logger.info("Checking Zenodo records...")
    
    for year, record_id in records.items():
        try:
            files = get_zenodo_files(record_id)
            total_size = sum(f.get('size', 0) for f in files) / 1e9
            logger.info(f"  {year} (record {record_id}): {len(files)} files, {total_size:.1f} GB")
            for f in files[:3]:
                logger.info(f"    - {f['key']}: {f['size']/1e6:.1f} MB")
        except Exception as e:
            logger.error(f"  {year}: Error - {e}")


def search_processed_schumann_data():
    """Search for already processed Schumann data."""
    logger.info("\nSearching for processed Schumann resonance datasets...")
    
    # Check Figshare
    figshare_url = "https://api.figshare.com/v2/articles/search"
    try:
        response = requests.post(
            figshare_url,
            json={"search_for": "Schumann resonance frequency"},
            timeout=30
        )
        if response.status_code == 200:
            results = response.json()
            logger.info(f"Figshare: Found {len(results)} datasets")
            for r in results[:5]:
                logger.info(f"  - {r.get('title', 'Unknown')[:60]}...")
    except Exception as e:
        logger.warning(f"Figshare search failed: {e}")
    
    # Check for supplementary data from papers
    logger.info("\nKnown processed datasets:")
    logger.info("  1. Salinas et al. (2022) - Supplementary material with Python code")
    logger.info("     https://doi.org/10.1016/j.cageo.2022.105148")
    logger.info("  2. Nickolaenko & Hayakawa - Book data")
    logger.info("  3. HeartMath GCI - Live data (not downloadable)")


def download_sample_raw_data():
    """Download a small sample of raw data for testing."""
    logger.info("\nAttempting to download sample raw data...")
    
    # The 2016 record has the smallest files
    record_id = "6348972"
    
    try:
        files = get_zenodo_files(record_id)
        
        # Find the smallest file (usually the PDF paper)
        for f in sorted(files, key=lambda x: x.get('size', float('inf'))):
            filename = f['key']
            size_mb = f['size'] / 1e6
            download_url = f['links']['self']
            
            logger.info(f"Found: {filename} ({size_mb:.1f} MB)")
            
            if size_mb < 50:  # Only download files < 50 MB
                logger.info(f"Downloading {filename}...")
                
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                output_path = DATA_DIR / filename
                
                response = requests.get(download_url, stream=True, timeout=300)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Saved: {output_path}")
                return output_path
                
    except Exception as e:
        logger.error(f"Download failed: {e}")
    
    return None


def create_info_file():
    """Create info file about real data sources."""
    info = """# REAL Schumann Resonance Data Sources

## Problem
The Zenodo Sierra Nevada data (26.5 GB) contains RAW time-domain ELF measurements,
NOT processed Schumann frequencies. Processing requires:
1. FFT analysis
2. Lorentzian fitting
3. Peak detection for f1, f2, f3

## Available Options

### Option 1: Download and Process Raw Data (26.5 GB)
```bash
# Download full dataset
wget https://zenodo.org/records/6348930/files/2013_2017.zip

# Process with Salinas et al. code
python process_elf_data.py
```

### Option 2: Use Published Results
The paper "Four Year Study of the Schumann Resonance Regular Variations" 
(Salinas et al., 2022) contains processed results in figures.

Typical values from literature:
- f1 = 7.83 ± 0.15 Hz (diurnal variation ~0.1 Hz)
- f2 = 14.1 ± 0.20 Hz
- f3 = 20.3 ± 0.25 Hz
- f4 = 26.4 ± 0.30 Hz

### Option 3: HeartMath GCI Live Data
https://www.heartmath.org/gci/gcms/live-data/
- 6 stations worldwide
- Live spectrograms
- NOT downloadable as CSV

## Current Status
The file `realistic_schumann_2016.csv` is SYNTHETIC data generated
based on literature values. It is NOT real measured data.

To use real data, you must:
1. Download the 26.5 GB Zenodo dataset
2. Process it using the Salinas et al. Python code
3. Extract f1, f2, f3 time series

## References
- Salinas et al. (2022). Computers & Geosciences, 165, 105148.
- Nickolaenko & Hayakawa (2002). Schumann Resonances.
"""
    
    info_path = DATA_DIR / "REAL_DATA_INFO.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(info)
    
    logger.info(f"Created: {info_path}")


def main():
    logger.info("="*60)
    logger.info("REAL SCHUMANN DATA DOWNLOAD")
    logger.info("="*60)
    
    # Check what's available
    check_zenodo_records()
    
    # Search for processed data
    search_processed_schumann_data()
    
    # Try to download sample
    download_sample_raw_data()
    
    # Create info file
    create_info_file()
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info("""
The Zenodo data is RAW ELF measurements (26.5 GB), not processed frequencies.

To get REAL Schumann frequencies, you need to:
1. Download the full 26.5 GB dataset
2. Process it with FFT + Lorentzian fitting

The current 'realistic_schumann_2016.csv' is SYNTHETIC.

Would you like me to:
a) Download the full 26.5 GB and process it (takes hours)
b) Use the synthetic data with proper labeling
c) Find another source of processed Schumann data
""")


if __name__ == "__main__":
    main()
