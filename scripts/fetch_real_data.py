#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Real Data for SSZ Schumann Analysis

Downloads:
1. F10.7 Solar Flux from NOAA PSL (1948-present, monthly)
2. Kp Index from GFZ Potsdam
3. Sample Schumann data info

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import urllib.request
import ssl

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
SPACE_WEATHER_DIR = DATA_DIR / "space_weather"


def download_file(url: str, filepath: Path, description: str) -> bool:
    """Download a file from URL."""
    print(f"\nDownloading {description}...")
    print(f"  URL: {url}")
    print(f"  Target: {filepath}")
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create SSL context that doesn't verify (for some servers)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = response.read()
            
        with open(filepath, 'wb') as f:
            f.write(data)
        
        size_kb = len(data) / 1024
        print(f"  SUCCESS: Downloaded {size_kb:.1f} KB")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def fetch_f107_noaa():
    """Fetch F10.7 solar flux from NOAA PSL."""
    print("\n" + "="*60)
    print("FETCHING F10.7 SOLAR FLUX DATA")
    print("="*60)
    
    # NOAA PSL monthly F10.7 data
    url = "https://psl.noaa.gov/data/correlation/solar.csv"
    filepath = SPACE_WEATHER_DIR / "f107_noaa_monthly.csv"
    
    success = download_file(url, filepath, "F10.7 Monthly Data (NOAA PSL)")
    
    if success:
        # Parse and show info
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            print(f"\n  Data Info:")
            print(f"    Lines: {len(lines)}")
            print(f"    First line: {lines[0].strip()[:80]}...")
            print(f"    Last line: {lines[-1].strip()[:80]}...")
        except Exception as e:
            print(f"  Could not parse: {e}")
    
    return success


def fetch_f107_daily():
    """Fetch daily F10.7 from NOAA SWPC."""
    print("\n" + "="*60)
    print("FETCHING DAILY F10.7 DATA")
    print("="*60)
    
    # NOAA SWPC observed solar cycle indices (JSON)
    url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
    filepath = SPACE_WEATHER_DIR / "f107_swpc_daily.json"
    
    success = download_file(url, filepath, "F10.7 Daily Data (NOAA SWPC)")
    
    if success:
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            print(f"\n  Data Info:")
            print(f"    Records: {len(data)}")
            if data:
                print(f"    First record: {data[0]}")
                print(f"    Last record: {data[-1]}")
        except Exception as e:
            print(f"  Could not parse: {e}")
    
    return success


def fetch_kp_gfz():
    """Fetch Kp index from GFZ Potsdam."""
    print("\n" + "="*60)
    print("FETCHING Kp INDEX DATA")
    print("="*60)
    
    # GFZ Potsdam Kp data (definitive values)
    # Format: https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt
    url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
    filepath = SPACE_WEATHER_DIR / "kp_gfz_since_1932.txt"
    
    success = download_file(url, filepath, "Kp/Ap Index (GFZ Potsdam)")
    
    if success:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Find header and data
            header_line = None
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('#') or line.strip() == '':
                    continue
                if 'Year' in line or 'YYYY' in line:
                    header_line = line.strip()
                    data_start = i + 1
                    break
            
            print(f"\n  Data Info:")
            print(f"    Total lines: {len(lines)}")
            if header_line:
                print(f"    Header: {header_line[:80]}...")
            print(f"    Data lines: {len(lines) - data_start}")
            if data_start < len(lines):
                print(f"    First data: {lines[data_start].strip()[:80]}...")
                print(f"    Last data: {lines[-1].strip()[:80]}...")
        except Exception as e:
            print(f"  Could not parse: {e}")
    
    return success


def create_schumann_info():
    """Create info file about Schumann data sources."""
    print("\n" + "="*60)
    print("CREATING SCHUMANN DATA INFO")
    print("="*60)
    
    schumann_dir = DATA_DIR / "schumann"
    schumann_dir.mkdir(parents=True, exist_ok=True)
    
    info_content = """# Schumann Resonance Data Sources

## 1. Sierra Nevada ELF Station (Zenodo)

**DOI:** 10.5281/zenodo.6348930
**URL:** https://zenodo.org/records/6348930
**Period:** March 2013 - February 2017
**Size:** 26.5 GB (raw data)
**Resolution:** ~1 hour

### Download Command:
```bash
# Download full dataset (26.5 GB)
wget https://zenodo.org/records/6348930/files/2013_2017.zip

# Or use the fetch script:
python scripts/fetch_zenodo_schumann.py --year 2016
```

### Data Format:
- Raw time-domain ELF measurements
- Two sensors: NS and EW orientation
- Requires processing to extract f1, f2, f3

### Reference:
Salinas, A. et al. (2022). Schumann resonance data processing programs 
and four-year measurements from Sierra Nevada ELF station. 
Computers & Geosciences, 165, 105148.

---

## 2. HeartMath GCI (Live Data)

**URL:** https://www.heartmath.org/gci/gcms/live-data/
**Stations:** 6 worldwide (California, Saudi Arabia, Lithuania, Canada, New Zealand, South Africa)
**Data:** Live spectrograms, not downloadable as CSV

---

## 3. GeoCenter.info (Live Monitor)

**URL:** https://geocenter.info/en/monitoring/schumann
**Data:** Live monitoring, limited historical data

---

## Recommended Approach

For SSZ analysis, use:
1. **Synthetic data** for validation (already implemented)
2. **Sierra Nevada data** for real-world testing (requires download)
3. **Space weather proxies** (F10.7, Kp) for correlation analysis

The synthetic data generator creates realistic Schumann frequencies with
configurable SSZ signals, which is sufficient for method validation.

---

Generated: {timestamp}
"""
    
    filepath = schumann_dir / "DATA_SOURCES.md"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(info_content.format(timestamp=datetime.now().isoformat()))
    
    print(f"  Created: {filepath}")
    return True


def convert_f107_to_csv():
    """Convert F10.7 JSON to CSV for easier use."""
    print("\n" + "="*60)
    print("CONVERTING F10.7 DATA TO CSV")
    print("="*60)
    
    json_path = SPACE_WEATHER_DIR / "f107_swpc_daily.json"
    csv_path = SPACE_WEATHER_DIR / "f107_daily.csv"
    
    if not json_path.exists():
        print("  JSON file not found, skipping conversion")
        return False
    
    try:
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Write CSV
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("date,f107,ssn\n")
            
            for record in data:
                date = record.get('time-tag', '')
                f107 = record.get('f10.7', '')
                ssn = record.get('ssn', '')
                f.write(f"{date},{f107},{ssn}\n")
        
        print(f"  Created: {csv_path}")
        print(f"  Records: {len(data)}")
        
        # Show date range
        if data:
            print(f"  Date range: {data[0].get('time-tag', 'N/A')} to {data[-1].get('time-tag', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def convert_kp_to_csv():
    """Convert Kp text file to CSV."""
    print("\n" + "="*60)
    print("CONVERTING Kp DATA TO CSV")
    print("="*60)
    
    txt_path = SPACE_WEATHER_DIR / "kp_gfz_since_1932.txt"
    csv_path = SPACE_WEATHER_DIR / "kp_daily.csv"
    
    if not txt_path.exists():
        print("  TXT file not found, skipping conversion")
        return False
    
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find data start (skip comments)
        data_lines = []
        for line in lines:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                data_lines.append(parts)
        
        # Write CSV
        # GFZ format: Year Month Day Days Days.5 Bsr Kp1 Kp2 Kp3 Kp4 Kp5 Kp6 Kp7 Kp8 ap1-8 Ap ...
        # Columns: 0=Year 1=Month 2=Day 3=Days 4=Days.5 5=Bsr 6=Kp1 7=Kp2...13=Kp8 14-21=ap1-8 22=Ap
        # Kp values are already in 0-9 scale (with decimals like 3.333)
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("date,kp_mean,ap,f107\n")
            
            for parts in data_lines:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    date = f"{year:04d}-{month:02d}-{day:02d}"
                    
                    # Kp values are in columns 7-14 (8 values per day)
                    # Values are already in Kp scale (0-9)
                    kp_values = []
                    for i in range(7, min(15, len(parts))):
                        try:
                            kp_val = float(parts[i])
                            if 0 <= kp_val <= 9:  # Valid Kp range
                                kp_values.append(kp_val)
                        except:
                            pass
                    
                    kp_mean = sum(kp_values) / len(kp_values) if kp_values else ''
                    
                    # Ap is in column 22 (daily Ap)
                    try:
                        ap = float(parts[22]) if len(parts) > 22 else ''
                    except:
                        ap = ''
                    
                    # F10.7 - not in this file, leave empty
                    f107 = ''
                    
                    f.write(f"{date},{kp_mean},{ap},{f107}\n")
                except:
                    continue
        
        print(f"  Created: {csv_path}")
        print(f"  Records: {len(data_lines)}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("="*60)
    print("SSZ SCHUMANN - FETCH REAL DATA")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    
    results = {}
    
    # Fetch F10.7 data
    results['f107_monthly'] = fetch_f107_noaa()
    results['f107_daily'] = fetch_f107_daily()
    
    # Fetch Kp data
    results['kp'] = fetch_kp_gfz()
    
    # Create Schumann info
    results['schumann_info'] = create_schumann_info()
    
    # Convert to CSV
    results['f107_csv'] = convert_f107_to_csv()
    results['kp_csv'] = convert_kp_to_csv()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    # List downloaded files
    print("\n" + "="*60)
    print("DOWNLOADED FILES")
    print("="*60)
    
    for filepath in DATA_DIR.rglob("*"):
        if filepath.is_file():
            size_kb = filepath.stat().st_size / 1024
            print(f"  {filepath.relative_to(DATA_DIR)}: {size_kb:.1f} KB")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. For Schumann data, download from Zenodo (26.5 GB):
   python scripts/fetch_zenodo_schumann.py --year 2016

2. Or use synthetic data for validation:
   python scripts/run_complete_analysis.py --synthetic

3. Run analysis with space weather data:
   python scripts/run_complete_analysis.py --sample
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
