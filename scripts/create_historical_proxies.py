#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Historical Space Weather Proxy Data

Uses published monthly averages and known solar cycle behavior to create
realistic daily proxy data for 2013-2017.

Data sources:
- F10.7: NOAA Solar Cycle 24 monthly averages
- Kp: GFZ Potsdam monthly statistics

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Output directories
SOLAR_DIR = Path("data/solar")
GEOMAG_DIR = Path("data/geomag")

# Historical F10.7 monthly averages (Solar Cycle 24)
# Source: NOAA SWPC archives
F107_MONTHLY = {
    # 2013 - Solar maximum
    (2013, 1): 130.5, (2013, 2): 107.5, (2013, 3): 110.4, (2013, 4): 120.7,
    (2013, 5): 132.9, (2013, 6): 113.4, (2013, 7): 118.8, (2013, 8): 115.8,
    (2013, 9): 108.5, (2013, 10): 132.9, (2013, 11): 143.2, (2013, 12): 147.1,
    # 2014 - Post-maximum
    (2014, 1): 157.4, (2014, 2): 170.5, (2014, 3): 152.4, (2014, 4): 138.3,
    (2014, 5): 119.9, (2014, 6): 114.7, (2014, 7): 130.5, (2014, 8): 127.0,
    (2014, 9): 145.8, (2014, 10): 161.9, (2014, 11): 150.8, (2014, 12): 155.3,
    # 2015 - Declining phase
    (2015, 1): 133.7, (2015, 2): 121.2, (2015, 3): 126.4, (2015, 4): 120.1,
    (2015, 5): 117.8, (2015, 6): 124.4, (2015, 7): 113.4, (2015, 8): 102.5,
    (2015, 9): 93.3, (2015, 10): 103.4, (2015, 11): 107.0, (2015, 12): 113.0,
    # 2016 - Minimum approach
    (2016, 1): 106.4, (2016, 2): 97.6, (2016, 3): 89.4, (2016, 4): 94.6,
    (2016, 5): 89.0, (2016, 6): 89.0, (2016, 7): 83.6, (2016, 8): 82.0,
    (2016, 9): 87.3, (2016, 10): 79.3, (2016, 11): 78.6, (2016, 12): 75.4,
    # 2017 - Near minimum
    (2017, 1): 77.4, (2017, 2): 80.0, (2017, 3): 77.0, (2017, 4): 73.5,
    (2017, 5): 75.0, (2017, 6): 72.5,
}

# Historical Kp monthly averages
# Source: GFZ Potsdam archives
KP_MONTHLY = {
    # 2013
    (2013, 1): 1.5, (2013, 2): 1.8, (2013, 3): 2.4, (2013, 4): 1.7,
    (2013, 5): 2.1, (2013, 6): 2.3, (2013, 7): 1.6, (2013, 8): 1.4,
    (2013, 9): 1.6, (2013, 10): 2.0, (2013, 11): 1.8, (2013, 12): 1.5,
    # 2014
    (2014, 1): 2.0, (2014, 2): 2.5, (2014, 3): 1.9, (2014, 4): 2.2,
    (2014, 5): 1.8, (2014, 6): 1.6, (2014, 7): 1.4, (2014, 8): 2.1,
    (2014, 9): 2.8, (2014, 10): 2.0, (2014, 11): 1.9, (2014, 12): 2.3,
    # 2015
    (2015, 1): 1.7, (2015, 2): 1.5, (2015, 3): 2.6, (2015, 4): 1.8,
    (2015, 5): 1.6, (2015, 6): 2.4, (2015, 7): 1.9, (2015, 8): 2.1,
    (2015, 9): 2.0, (2015, 10): 2.2, (2015, 11): 1.8, (2015, 12): 2.5,
    # 2016
    (2016, 1): 2.0, (2016, 2): 1.6, (2016, 3): 1.4, (2016, 4): 1.9,
    (2016, 5): 1.7, (2016, 6): 1.5, (2016, 7): 1.3, (2016, 8): 1.6,
    (2016, 9): 2.1, (2016, 10): 2.4, (2016, 11): 1.8, (2016, 12): 1.5,
    # 2017
    (2017, 1): 1.4, (2017, 2): 1.6, (2017, 3): 1.8, (2017, 4): 1.5,
    (2017, 5): 2.0, (2017, 6): 1.7,
}


def generate_daily_f107(year: int, month: int) -> pd.DataFrame:
    """
    Generate realistic daily F10.7 from monthly average.
    
    Includes:
    - 27-day solar rotation modulation
    - Random day-to-day variation
    """
    import calendar
    
    if (year, month) not in F107_MONTHLY:
        return None
    
    monthly_avg = F107_MONTHLY[(year, month)]
    n_days = calendar.monthrange(year, month)[1]
    
    # Reproducible random seed
    np.random.seed(year * 100 + month)
    
    days = pd.date_range(f'{year}-{month:02d}-01', periods=n_days, freq='D')
    t = np.arange(n_days)
    
    # 27-day solar rotation modulation (amplitude ~10%)
    rotation_phase = 2 * np.pi * (t + (year - 2013) * 365 + (month - 1) * 30) / 27.0
    modulation = 0.10 * monthly_avg * np.sin(rotation_phase)
    
    # Random day-to-day variation (std ~5%)
    noise = np.random.normal(0, 0.05 * monthly_avg, n_days)
    
    f107_daily = monthly_avg + modulation + noise
    
    return pd.DataFrame({
        'date': days.date,
        'f107': np.round(f107_daily, 1),
        'source': 'HISTORICAL_RECONSTRUCTED'
    })


def generate_daily_kp(year: int, month: int) -> pd.DataFrame:
    """
    Generate realistic daily Kp from monthly average.
    
    Kp follows a gamma distribution with occasional storms.
    """
    import calendar
    
    if (year, month) not in KP_MONTHLY:
        return None
    
    monthly_avg = KP_MONTHLY[(year, month)]
    n_days = calendar.monthrange(year, month)[1]
    
    # Reproducible random seed
    np.random.seed(year * 100 + month + 1000)
    
    days = pd.date_range(f'{year}-{month:02d}-01', periods=n_days, freq='D')
    
    # Gamma distribution with shape parameter to match monthly mean
    # Mean of gamma = shape * scale, we want mean = monthly_avg
    shape = 2.0
    scale = monthly_avg / shape
    kp_daily = np.random.gamma(shape, scale, n_days)
    
    # Add occasional storms (5% chance of Kp > 5)
    storm_mask = np.random.random(n_days) < 0.05
    kp_daily[storm_mask] = np.random.uniform(5, 7, storm_mask.sum())
    
    # Clip to valid range
    kp_daily = np.clip(kp_daily, 0, 9)
    
    return pd.DataFrame({
        'date': days.date,
        'kp': np.round(kp_daily, 1),
        'source': 'HISTORICAL_RECONSTRUCTED'
    })


def main():
    print("=" * 70)
    print("HISTORICAL SPACE WEATHER DATA GENERATOR")
    print("=" * 70)
    print()
    
    SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    GEOMAG_DIR.mkdir(parents=True, exist_ok=True)
    
    all_f107 = []
    all_kp = []
    
    # Generate for all available months
    for (year, month) in sorted(F107_MONTHLY.keys()):
        print(f"Generating {year}-{month:02d}...")
        
        f107_df = generate_daily_f107(year, month)
        kp_df = generate_daily_kp(year, month)
        
        if f107_df is not None:
            all_f107.append(f107_df)
        if kp_df is not None:
            all_kp.append(kp_df)
    
    # Combine and save
    combined_f107 = pd.concat(all_f107, ignore_index=True)
    combined_kp = pd.concat(all_kp, ignore_index=True)
    
    # Save combined files
    combined_f107.to_csv(SOLAR_DIR / "f107_2013_2017_daily.csv", index=False)
    combined_kp.to_csv(GEOMAG_DIR / "kp_2013_2017_daily.csv", index=False)
    
    print()
    print(f"[OK] Saved F10.7: {SOLAR_DIR / 'f107_2013_2017_daily.csv'}")
    print(f"     Records: {len(combined_f107)}")
    print(f"     Date range: {combined_f107['date'].min()} to {combined_f107['date'].max()}")
    print(f"     Mean: {combined_f107['f107'].mean():.1f} sfu")
    
    print()
    print(f"[OK] Saved Kp: {GEOMAG_DIR / 'kp_2013_2017_daily.csv'}")
    print(f"     Records: {len(combined_kp)}")
    print(f"     Date range: {combined_kp['date'].min()} to {combined_kp['date'].max()}")
    print(f"     Mean: {combined_kp['kp'].mean():.1f}")
    
    # Also save individual month files for October 2013
    oct2013_f107 = combined_f107[
        (pd.to_datetime(combined_f107['date']).dt.year == 2013) &
        (pd.to_datetime(combined_f107['date']).dt.month == 10)
    ]
    oct2013_kp = combined_kp[
        (pd.to_datetime(combined_kp['date']).dt.year == 2013) &
        (pd.to_datetime(combined_kp['date']).dt.month == 10)
    ]
    
    oct2013_f107.to_csv(SOLAR_DIR / "f107_201310_daily.csv", index=False)
    oct2013_kp.to_csv(GEOMAG_DIR / "kp_201310_daily.csv", index=False)
    
    print()
    print(f"[OK] Saved October 2013 files:")
    print(f"     {SOLAR_DIR / 'f107_201310_daily.csv'} ({len(oct2013_f107)} records)")
    print(f"     {GEOMAG_DIR / 'kp_201310_daily.csv'} ({len(oct2013_kp)} records)")
    
    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print()
    print("NOTE: This data is RECONSTRUCTED from published monthly averages.")
    print("      Daily values include realistic solar rotation and random variation.")
    print("      For definitive results, use original daily data from:")
    print("      - F10.7: ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/")
    print("      - Kp: ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/")


if __name__ == "__main__":
    main()
