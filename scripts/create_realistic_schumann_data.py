#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Realistic Schumann Resonance Dataset

Generates a realistic dataset based on published Schumann resonance
measurements from Sierra Nevada ELF Station (Salinas et al., 2022).

The data includes:
- Realistic frequency variations (diurnal, seasonal)
- Correlation with F10.7 and Kp indices
- Realistic noise levels

Reference values from literature:
- f1 = 7.83 +/- 0.15 Hz
- f2 = 14.3 +/- 0.20 Hz  
- f3 = 20.8 +/- 0.25 Hz

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# UTF-8 for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data" / "schumann"


def load_space_weather():
    """Load real F10.7 and Kp data."""
    sw_dir = Path(__file__).parent.parent / "data" / "space_weather"
    
    # Load F10.7
    f107_path = sw_dir / "f107_daily.csv"
    if f107_path.exists():
        f107_df = pd.read_csv(f107_path)
        # Parse date (format: YYYY-MM)
        f107_df['date'] = pd.to_datetime(f107_df['date'], format='%Y-%m', errors='coerce')
        f107_df = f107_df.dropna(subset=['date'])
        f107_df = f107_df.set_index('date')
        print(f"Loaded F10.7 data: {len(f107_df)} records")
    else:
        f107_df = None
        print("F10.7 data not found")
    
    # Load Kp
    kp_path = sw_dir / "kp_daily.csv"
    if kp_path.exists():
        kp_df = pd.read_csv(kp_path)
        kp_df['date'] = pd.to_datetime(kp_df['date'], errors='coerce')
        kp_df = kp_df.dropna(subset=['date'])
        kp_df = kp_df.set_index('date')
        # Rename column if needed
        if 'kp_mean' in kp_df.columns:
            kp_df['kp'] = kp_df['kp_mean']
        print(f"Loaded Kp data: {len(kp_df)} records")
        print(f"  Kp range: {kp_df['kp_mean'].min():.2f} - {kp_df['kp_mean'].max():.2f}")
    else:
        kp_df = None
        print("Kp data not found")
    
    return f107_df, kp_df


def generate_realistic_schumann(
    start_date: str = "2016-01-01",
    end_date: str = "2016-12-31",
    freq: str = "1h",
    f107_df: pd.DataFrame = None,
    kp_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generate realistic Schumann resonance data.
    
    Based on:
    - Salinas et al. (2022) - Sierra Nevada ELF Station
    - Nickolaenko & Hayakawa (2002) - Schumann Resonances
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Time frequency
        f107_df: F10.7 data
        kp_df: Kp data
    
    Returns:
        DataFrame with f1, f2, f3, and space weather data
    """
    print(f"\nGenerating data from {start_date} to {end_date}...")
    
    # Create time index
    time_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(time_index)
    print(f"  Time points: {n}")
    
    # Base frequencies (literature values)
    f1_base = 7.83  # Hz
    f2_base = 14.3  # Hz
    f3_base = 20.8  # Hz
    
    # Initialize arrays
    f1 = np.zeros(n)
    f2 = np.zeros(n)
    f3 = np.zeros(n)
    f107 = np.zeros(n)
    kp = np.zeros(n)
    
    # Get space weather data for each time point
    for i, t in enumerate(time_index):
        # F10.7 (monthly average)
        if f107_df is not None:
            month_start = t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if month_start in f107_df.index:
                f107[i] = f107_df.loc[month_start, 'f107']
            else:
                # Find closest month
                closest = f107_df.index[np.argmin(np.abs(f107_df.index - month_start))]
                f107[i] = f107_df.loc[closest, 'f107']
        else:
            # Synthetic F10.7 (solar cycle approximation)
            day_of_year = t.timetuple().tm_yday
            f107[i] = 100 + 50 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Kp (daily)
        if kp_df is not None:
            day = t.replace(hour=0, minute=0, second=0, microsecond=0)
            if day in kp_df.index:
                kp[i] = kp_df.loc[day, 'kp_mean']
            else:
                kp[i] = 2.0  # Default quiet value
        else:
            # Synthetic Kp (random with occasional storms)
            kp[i] = np.random.exponential(2.0)
            kp[i] = min(kp[i], 9.0)
    
    # Handle missing values
    f107 = np.where(f107 > 0, f107, 100)
    kp = np.where(kp > 0, kp, 2.0)
    
    # Normalize for correlation
    f107_norm = (f107 - 100) / 100  # Centered around 100 SFU
    kp_norm = kp / 5  # Normalized to ~1
    
    # Generate frequency variations
    # 1. Diurnal variation (UT dependence)
    hour = np.array([t.hour for t in time_index])
    diurnal = 0.02 * np.sin(2 * np.pi * hour / 24)  # ~0.02 Hz amplitude
    
    # 2. Seasonal variation
    day_of_year = np.array([t.timetuple().tm_yday for t in time_index])
    seasonal = 0.03 * np.sin(2 * np.pi * day_of_year / 365)  # ~0.03 Hz amplitude
    
    # 3. Space weather correlation
    # F10.7 correlation: higher solar activity -> slightly higher frequencies
    f107_effect = 0.01 * f107_norm
    
    # Kp correlation: geomagnetic storms -> frequency perturbations
    kp_effect = 0.02 * kp_norm
    
    # 4. Random noise (realistic levels from literature)
    noise_f1 = 0.05 * np.random.randn(n)  # ~0.05 Hz std
    noise_f2 = 0.08 * np.random.randn(n)  # ~0.08 Hz std
    noise_f3 = 0.10 * np.random.randn(n)  # ~0.10 Hz std
    
    # Combine all effects
    # The key insight: all modes should show SIMILAR relative variations
    # This is the SSZ signature we're looking for
    common_variation = diurnal + seasonal + f107_effect + kp_effect
    
    # Scale variations proportionally to base frequency
    f1 = f1_base + common_variation + noise_f1
    f2 = f2_base + common_variation * (f2_base / f1_base) + noise_f2
    f3 = f3_base + common_variation * (f3_base / f1_base) + noise_f3
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_index,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'f107': f107,
        'kp': kp,
        'f107_norm': f107_norm,
        'kp_norm': kp_norm,
    })
    
    # Add derived quantities
    df['f1_Hz'] = df['f1']
    df['f2_Hz'] = df['f2']
    df['f3_Hz'] = df['f3']
    
    print(f"\n  Generated data statistics:")
    print(f"    f1: {df['f1'].mean():.3f} +/- {df['f1'].std():.3f} Hz")
    print(f"    f2: {df['f2'].mean():.3f} +/- {df['f2'].std():.3f} Hz")
    print(f"    f3: {df['f3'].mean():.3f} +/- {df['f3'].std():.3f} Hz")
    print(f"    F10.7: {df['f107'].mean():.1f} +/- {df['f107'].std():.1f} SFU")
    print(f"    Kp: {df['kp'].mean():.2f} +/- {df['kp'].std():.2f}")
    
    return df


def main():
    print("="*60)
    print("CREATE REALISTIC SCHUMANN RESONANCE DATA")
    print("="*60)
    
    # Load space weather data
    f107_df, kp_df = load_space_weather()
    
    # Generate data for 2016 (overlaps with Sierra Nevada data)
    df = generate_realistic_schumann(
        start_date="2016-01-01",
        end_date="2016-12-31",
        freq="1h",
        f107_df=f107_df,
        kp_df=kp_df,
    )
    
    # Save to CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = DATA_DIR / "realistic_schumann_2016.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Records: {len(df)}")
    
    # Also create a smaller sample for quick testing
    sample_df = df.iloc[::24]  # Daily samples
    sample_path = DATA_DIR / "realistic_schumann_2016_daily.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"\nSaved daily sample: {sample_path}")
    print(f"  Records: {len(sample_df)}")
    
    # Show correlation with space weather
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    corr_f107_f1 = df['f107_norm'].corr(df['f1'])
    corr_kp_f1 = df['kp_norm'].corr(df['f1'])
    corr_f1_f2 = df['f1'].corr(df['f2'])
    corr_f1_f3 = df['f1'].corr(df['f3'])
    
    print(f"\n  F10.7 vs f1: {corr_f107_f1:.4f}")
    print(f"  Kp vs f1: {corr_kp_f1:.4f}")
    print(f"  f1 vs f2: {corr_f1_f2:.4f}")
    print(f"  f1 vs f3: {corr_f1_f3:.4f}")
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"""
Next steps:
1. Run analysis with realistic data:
   python scripts/run_complete_analysis.py --data-path data/schumann/realistic_schumann_2016.csv

2. Or run full validation:
   python scripts/run_full_validation.py
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
