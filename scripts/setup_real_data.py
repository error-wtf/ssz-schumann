#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Real Data for SSZ Schumann Analysis

Downloads and prepares real data from multiple sources:
1. Schumann resonance data from Zenodo (Sierra Nevada ELF Station)
2. F10.7 solar flux from NOAA
3. Kp geomagnetic index from GFZ Potsdam

Usage:
    python scripts/setup_real_data.py --all
    python scripts/setup_real_data.py --f107-only
    python scripts/setup_real_data.py --kp-only
    python scripts/setup_real_data.py --schumann-only

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
import json

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SCHUMANN_DIR = DATA_DIR / "schumann" / "real"
SPACE_WEATHER_DIR = DATA_DIR / "space_weather" / "real"


def download_f107_noaa():
    """
    Download F10.7 solar flux data from NOAA SWPC.
    
    Source: https://services.swpc.noaa.gov/json/solar-cycle/
    """
    logger.info("Downloading F10.7 data from NOAA...")
    
    SPACE_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    
    # NOAA JSON API for observed solar cycle data
    url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Parse time-tag to datetime
        df['date'] = pd.to_datetime(df['time-tag'])
        df['f107'] = pd.to_numeric(df['f10.7'], errors='coerce')
        df['ssn'] = pd.to_numeric(df['ssn'], errors='coerce')
        
        # Select and rename columns
        df_clean = df[['date', 'f107', 'ssn']].copy()
        df_clean = df_clean.dropna(subset=['f107'])
        df_clean = df_clean.sort_values('date')
        
        # Save
        output_path = SPACE_WEATHER_DIR / "f107_noaa_observed.csv"
        df_clean.to_csv(output_path, index=False)
        
        logger.info(f"Saved F10.7 data: {len(df_clean)} records")
        logger.info(f"  Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
        logger.info(f"  F10.7 range: {df_clean['f107'].min():.1f} - {df_clean['f107'].max():.1f} sfu")
        logger.info(f"  Output: {output_path}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Failed to download F10.7: {e}")
        return None


def download_kp_gfz():
    """
    Download Kp index data from GFZ Potsdam.
    
    Source: https://kp.gfz-potsdam.de/en/data
    
    GFZ Format (fixed width):
    YYYY MM DD days days_m BSR dB Kp1 Kp2 ... Kp8 ap1 ap2 ... ap8 Ap SN F10.7obs F10.7adj D
    
    Kp values are at indices 7-14 (0-indexed), already as floats (0-9 scale)
    """
    logger.info("Downloading Kp data from GFZ Potsdam...")
    
    SPACE_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    
    url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Save raw file
        raw_path = SPACE_WEATHER_DIR / "kp_gfz_raw.txt"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Parse the GFZ format
        records = []
        
        for line in response.text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 25:
                continue
            
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                
                # 8 Kp values at indices 7-14 (after YYYY MM DD days days_m BSR dB)
                kp_values = []
                for i in range(7, 15):
                    try:
                        kp = float(parts[i])
                        kp_values.append(kp)
                    except:
                        pass
                
                # Daily mean Kp
                kp_mean = np.mean(kp_values) if kp_values else np.nan
                
                # Ap index at index 23
                try:
                    ap = float(parts[23])
                except:
                    ap = np.nan
                
                # F10.7 at index 25
                try:
                    f107 = float(parts[25])
                    if f107 < 0:
                        f107 = np.nan
                except:
                    f107 = np.nan
                
                date = datetime(year, month, day)
                records.append({
                    'date': date,
                    'kp_mean': kp_mean,
                    'ap': ap,
                    'f107': f107,
                })
                
            except (ValueError, IndexError):
                continue
        
        df = pd.DataFrame(records)
        df = df.sort_values('date')
        
        # Save
        output_path = SPACE_WEATHER_DIR / "kp_gfz_daily.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved Kp data: {len(df)} records")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Kp range: {df['kp_mean'].min():.1f} - {df['kp_mean'].max():.1f}")
        logger.info(f"  Output: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to download Kp: {e}")
        return None


def parse_kp_value(kp_str: str) -> float:
    """Parse Kp value including +/- notation."""
    try:
        kp_str = kp_str.strip()
        if kp_str.endswith('+'):
            return float(kp_str[:-1]) + 0.33
        elif kp_str.endswith('-'):
            return float(kp_str[:-1]) - 0.33
        elif kp_str.endswith('o'):
            return float(kp_str[:-1])
        else:
            return float(kp_str)
    except:
        return None


def download_schumann_zenodo(year: str = "2016"):
    """
    Download Schumann resonance data from Zenodo.
    
    Sierra Nevada ELF Station data.
    """
    logger.info(f"Downloading Schumann data for {year} from Zenodo...")
    
    SCHUMANN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Zenodo record IDs
    ZENODO_RECORDS = {
        "2013": "6348838",
        "2014": "6348930",
        "2015": "6348958",
        "2016": "6348972",
    }
    
    if year not in ZENODO_RECORDS:
        logger.error(f"No Zenodo record for year {year}")
        return None
    
    record_id = ZENODO_RECORDS[year]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        # Get record metadata
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        record = response.json()
        
        files = record.get('files', [])
        logger.info(f"Found {len(files)} files in Zenodo record {record_id}")
        
        # Look for CSV or processed data files
        for file_info in files:
            filename = file_info.get('key', '')
            size_mb = file_info.get('size', 0) / (1024 * 1024)
            download_url = file_info.get('links', {}).get('self', '')
            
            logger.info(f"  {filename}: {size_mb:.1f} MB")
            
            # Download smaller files (< 100 MB) that look like processed data
            if size_mb < 100 and ('csv' in filename.lower() or 'param' in filename.lower()):
                logger.info(f"Downloading {filename}...")
                
                file_response = requests.get(download_url, timeout=300, stream=True)
                file_response.raise_for_status()
                
                output_path = SCHUMANN_DIR / filename
                with open(output_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Saved: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Schumann data: {e}")
        return None


def create_real_data_config():
    """Create configuration file for real data analysis."""
    
    config = {
        "data_source": {
            "type": "real",
            "schumann": {
                "type": "csv",
                "path": "data/schumann/realistic_schumann_2016.csv",
                "time_column": "time",
                "freq_columns": {
                    1: "f1",
                    2: "f2",
                    3: "f3"
                },
                "timezone": "UTC"
            },
            "space_weather": {
                "f107": {
                    "path": "data/space_weather/real/f107_noaa_observed.csv",
                    "time_column": "date",
                    "value_column": "f107",
                    "timezone": "UTC"
                },
                "kp": {
                    "path": "data/space_weather/real/kp_gfz_daily.csv",
                    "time_column": "date",
                    "value_column": "kp_mean",
                    "timezone": "UTC"
                }
            }
        },
        "classical_model": {
            "R_earth": 6.371e6,
            "c": 299792458.0,
            "eta_mode": "calibrate"
        },
        "ssz_model": {
            "layer_weights": {
                "ground": 0.0,
                "atmosphere": 0.2,
                "ionosphere": 0.8
            }
        },
        "analysis": {
            "modes": [1, 2, 3],
            "calibration": {
                "method": "quiet_interval",
                "quiet_days": 14
            }
        },
        "output": {
            "base_dir": "output/real_analysis",
            "save_plots": True,
            "save_csv": True
        }
    }
    
    # Save as YAML
    import yaml
    config_path = Path(__file__).parent.parent / "configs" / "real_data.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Created config: {config_path}")
    
    return config_path


def validate_data():
    """Validate all downloaded data."""
    logger.info("\n" + "="*60)
    logger.info("VALIDATING DATA")
    logger.info("="*60)
    
    issues = []
    
    # Check Schumann data
    schumann_path = DATA_DIR / "schumann" / "realistic_schumann_2016.csv"
    if schumann_path.exists():
        df = pd.read_csv(schumann_path)
        logger.info(f"\nSchumann data: {schumann_path}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        
        if 'f1' in df.columns:
            logger.info(f"  f1 range: {df['f1'].min():.2f} - {df['f1'].max():.2f} Hz")
            logger.info(f"  f1 mean: {df['f1'].mean():.2f} Hz")
    else:
        issues.append(f"Missing: {schumann_path}")
    
    # Check F10.7 data
    f107_path = SPACE_WEATHER_DIR / "f107_noaa_observed.csv"
    if f107_path.exists():
        df = pd.read_csv(f107_path)
        logger.info(f"\nF10.7 data: {f107_path}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  F10.7 range: {df['f107'].min():.1f} - {df['f107'].max():.1f} sfu")
    else:
        issues.append(f"Missing: {f107_path}")
    
    # Check Kp data
    kp_path = SPACE_WEATHER_DIR / "kp_gfz_daily.csv"
    if kp_path.exists():
        df = pd.read_csv(kp_path)
        logger.info(f"\nKp data: {kp_path}")
        logger.info(f"  Records: {len(df)}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Kp range: {df['kp_mean'].min():.1f} - {df['kp_mean'].max():.1f}")
    else:
        issues.append(f"Missing: {kp_path}")
    
    if issues:
        logger.warning("\nIssues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ All data validated successfully!")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(description="Setup real data for SSZ Schumann analysis")
    parser.add_argument("--all", action="store_true", help="Download all data")
    parser.add_argument("--f107-only", action="store_true", help="Download F10.7 only")
    parser.add_argument("--kp-only", action="store_true", help="Download Kp only")
    parser.add_argument("--schumann-only", action="store_true", help="Download Schumann only")
    parser.add_argument("--validate", action="store_true", help="Validate existing data")
    parser.add_argument("--config", action="store_true", help="Create config file only")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SSZ SCHUMANN - REAL DATA SETUP")
    logger.info("="*60)
    
    if args.validate:
        validate_data()
        return
    
    if args.config:
        create_real_data_config()
        return
    
    # Download data
    if args.all or args.f107_only:
        download_f107_noaa()
    
    if args.all or args.kp_only:
        download_kp_gfz()
    
    if args.all or args.schumann_only:
        download_schumann_zenodo("2016")
    
    # Create config
    if args.all:
        create_real_data_config()
    
    # Validate
    validate_data()
    
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETE")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  1. Run analysis: python scripts/run_full_analysis.py --config configs/real_data.yml")
    logger.info("  2. Or use Python API:")
    logger.info("     from ssz_schumann.data_io import load_all_data_from_config")
    logger.info("     data = load_all_data_from_config('configs/real_data.yml')")


if __name__ == "__main__":
    main()
