#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Fetching Script for SSZ Schumann Experiment

Downloads required data from public sources:
- Schumann resonance data from Zenodo (Sierra Nevada ELF station)
- F10.7 solar flux from NOAA
- Kp/Ap indices from GFZ Potsdam

Usage:
    python scripts/fetch_data.py --all
    python scripts/fetch_data.py --schumann
    python scripts/fetch_data.py --f107
    python scripts/fetch_data.py --kp

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path(__file__).parent.parent / "data"
SCHUMANN_DIR = DATA_DIR / "schumann"
SPACE_WEATHER_DIR = DATA_DIR / "space_weather"
LIGHTNING_DIR = DATA_DIR / "lightning"

# URLs
ZENODO_BASE = "https://zenodo.org/records"
ZENODO_SCHUMANN_IDS = {
    "2013": "6348930",
    "2014": "6348866",
    "2015": "6348852",
    "2016": "6348838",
    "2017": "6348824",
}

F107_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
KP_GFZ_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: Source URL
        dest_path: Destination path
        description: Description for logging
    
    Returns:
        True if successful
    """
    logger.info(f"Downloading {description or url}")
    logger.info(f"  -> {dest_path}")
    
    try:
        # Create directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print()  # New line after progress
        
        logger.info(f"  Downloaded: {dest_path.stat().st_size / 1024:.1f} KB")
        return True
        
    except Exception as e:
        logger.error(f"  Failed: {e}")
        return False


def fetch_schumann_data(years: list = None) -> bool:
    """
    Fetch Schumann resonance data from Zenodo.
    
    The Sierra Nevada ELF station data is available at:
    https://doi.org/10.5281/zenodo.6348930
    
    Args:
        years: List of years to fetch (default: all 2013-2017)
    
    Returns:
        True if at least one year was fetched successfully
    """
    if years is None:
        years = list(ZENODO_SCHUMANN_IDS.keys())
    
    logger.info("=" * 60)
    logger.info("Fetching Schumann Resonance Data (Sierra Nevada ELF Station)")
    logger.info("=" * 60)
    logger.info("Source: Salinas et al. (2022)")
    logger.info("DOI: 10.5281/zenodo.6348930")
    logger.info("")
    
    success_count = 0
    
    for year in years:
        if year not in ZENODO_SCHUMANN_IDS:
            logger.warning(f"Unknown year: {year}")
            continue
        
        zenodo_id = ZENODO_SCHUMANN_IDS[year]
        
        # Zenodo download URL format
        # Note: Actual file names may vary - this is a placeholder
        # The real data needs to be downloaded manually from Zenodo
        
        dest_dir = SCHUMANN_DIR / year
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        info_file = dest_dir / "README.txt"
        
        # Write info file with download instructions
        info_content = f"""Sierra Nevada ELF Station - Schumann Resonance Data {year}

Source: Zenodo
Record ID: {zenodo_id}
DOI: 10.5281/zenodo.{zenodo_id}
URL: https://zenodo.org/records/{zenodo_id}

To download the data:
1. Visit https://zenodo.org/records/{zenodo_id}
2. Download the data files
3. Extract to this directory

Reference:
Salinas, A., Rodriguez-Camacho, J., Porti, J., Carrion, M.C., 
Fornieles-Callejon, J., & Toledo-Redondo, S. (2022).
Schumann resonance data processing programs and four-year 
measurements from Sierra Nevada ELF station.
Computers & Geosciences, 165, 105148.
https://doi.org/10.1016/j.cageo.2022.105148

Data format:
- 10-minute intervals
- Parameters: f1, f2, f3 (frequencies), widths, amplitudes
- Time in UTC

Downloaded: {datetime.now().isoformat()}
"""
        
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(info_content)
        
        logger.info(f"Created info file for {year}: {info_file}")
        success_count += 1
    
    logger.info("")
    logger.info("NOTE: Schumann data must be downloaded manually from Zenodo.")
    logger.info("Visit the URLs in the README files to download the actual data.")
    
    return success_count > 0


def fetch_f107_data() -> bool:
    """
    Fetch F10.7 solar flux data from NOAA.
    
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("Fetching F10.7 Solar Flux Data")
    logger.info("=" * 60)
    logger.info(f"Source: NOAA SWPC")
    logger.info(f"URL: {F107_URL}")
    logger.info("")
    
    dest_path = SPACE_WEATHER_DIR / "f107_noaa.json"
    
    if download_file(F107_URL, dest_path, "F10.7 data"):
        # Also create a CSV version
        try:
            with open(dest_path, "r") as f:
                data = json.load(f)
            
            # Convert to CSV
            csv_path = SPACE_WEATHER_DIR / "f107_noaa.csv"
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("time,f107,ssn\n")
                for entry in data:
                    time_tag = entry.get("time-tag", "")
                    f107 = entry.get("f10.7", "")
                    ssn = entry.get("ssn", "")
                    if time_tag and f107:
                        f.write(f"{time_tag},{f107},{ssn}\n")
            
            logger.info(f"Created CSV: {csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create CSV: {e}")
            return True  # JSON was still downloaded
    
    return False


def fetch_kp_data() -> bool:
    """
    Fetch Kp/Ap index data from GFZ Potsdam.
    
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("Fetching Kp/Ap Geomagnetic Index Data")
    logger.info("=" * 60)
    logger.info("Source: GFZ Potsdam")
    logger.info("")
    
    # GFZ provides data via their website
    # Direct download may not work - provide instructions
    
    dest_dir = SPACE_WEATHER_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    info_file = dest_dir / "kp_README.txt"
    
    info_content = f"""Kp/Ap Geomagnetic Index Data

Source: GFZ Helmholtz Centre Potsdam
Website: https://kp.gfz-potsdam.de/en/data

To download the data:
1. Visit https://kp.gfz-potsdam.de/en/data
2. Select the desired time range
3. Download in your preferred format (ASCII, JSON, etc.)
4. Save to this directory as 'kp_ap.csv' or 'kp_ap.txt'

Alternative FTP access:
ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/

Data format:
- Kp: 3-hourly values (0-9 scale)
- Ap: Daily average
- Available since 1932

Reference:
Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O., & Morschhauser, A. (2021).
The geomagnetic Kp index and derived indices of geomagnetic activity.
Space Weather, 19, e2020SW002641.
https://doi.org/10.1029/2020SW002641

Downloaded: {datetime.now().isoformat()}
"""
    
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(info_content)
    
    logger.info(f"Created info file: {info_file}")
    logger.info("")
    logger.info("NOTE: Kp data must be downloaded manually from GFZ Potsdam.")
    logger.info("Visit https://kp.gfz-potsdam.de/en/data")
    
    return True


def create_sample_data() -> bool:
    """
    Create sample/synthetic data for testing.
    
    Returns:
        True if successful
    """
    logger.info("=" * 60)
    logger.info("Creating Sample Data for Testing")
    logger.info("=" * 60)
    
    # Add parent to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from ssz_schumann.data_io.schumann_sierra_nevada import create_synthetic_schumann_data
        from ssz_schumann.data_io.space_weather_noaa import create_synthetic_space_weather
        from ssz_schumann.data_io.lightning_wwlln import create_synthetic_lightning
        
        # Create synthetic Schumann data
        logger.info("Creating synthetic Schumann data...")
        ds = create_synthetic_schumann_data(
            start="2016-01-01",
            end="2016-12-31",
            freq="1h",
            eta_0=0.74,
            delta_seg_amplitude=0.02,
            noise_level=0.01,
        )
        
        # Save as NetCDF
        schumann_path = SCHUMANN_DIR / "synthetic_2016.nc"
        schumann_path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(schumann_path)
        logger.info(f"  Saved: {schumann_path}")
        
        # Create synthetic space weather
        logger.info("Creating synthetic space weather data...")
        f107, kp = create_synthetic_space_weather(
            start="2016-01-01",
            end="2016-12-31",
        )
        
        # Save as CSV
        f107_path = SPACE_WEATHER_DIR / "synthetic_f107.csv"
        f107_path.parent.mkdir(parents=True, exist_ok=True)
        f107.to_csv(f107_path, header=True)
        logger.info(f"  Saved: {f107_path}")
        
        kp_path = SPACE_WEATHER_DIR / "synthetic_kp.csv"
        kp.to_csv(kp_path, header=True)
        logger.info(f"  Saved: {kp_path}")
        
        # Create synthetic lightning
        logger.info("Creating synthetic lightning data...")
        thunder = create_synthetic_lightning(
            start="2016-01-01",
            end="2016-12-31",
            freq="1D",
        )
        
        thunder_path = LIGHTNING_DIR / "synthetic_thunder.nc"
        thunder_path.parent.mkdir(parents=True, exist_ok=True)
        thunder.to_netcdf(thunder_path)
        logger.info(f"  Saved: {thunder_path}")
        
        logger.info("")
        logger.info("Sample data created successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create sample data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fetch data for SSZ Schumann experiment"
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Fetch all data sources")
    parser.add_argument("--schumann", action="store_true",
                       help="Fetch Schumann resonance data info")
    parser.add_argument("--f107", action="store_true",
                       help="Fetch F10.7 solar flux data")
    parser.add_argument("--kp", action="store_true",
                       help="Fetch Kp/Ap index data info")
    parser.add_argument("--sample", action="store_true",
                       help="Create synthetic sample data for testing")
    parser.add_argument("--years", nargs="+", default=None,
                       help="Years to fetch for Schumann data (default: all)")
    
    args = parser.parse_args()
    
    # Default to --all if no specific option given
    if not any([args.all, args.schumann, args.f107, args.kp, args.sample]):
        args.all = True
    
    logger.info("SSZ Schumann Experiment - Data Fetcher")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("")
    
    success = True
    
    if args.all or args.schumann:
        success = fetch_schumann_data(args.years) and success
        logger.info("")
    
    if args.all or args.f107:
        success = fetch_f107_data() and success
        logger.info("")
    
    if args.all or args.kp:
        success = fetch_kp_data() and success
        logger.info("")
    
    if args.sample:
        success = create_sample_data() and success
        logger.info("")
    
    if success:
        logger.info("=" * 60)
        logger.info("Data fetching completed!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Download Schumann data manually from Zenodo")
        logger.info("2. Download Kp data manually from GFZ Potsdam")
        logger.info("3. Run: python scripts/run_schumann_ssz_analysis.py")
    else:
        logger.error("Some data fetching failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
