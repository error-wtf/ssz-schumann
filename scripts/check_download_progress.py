#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Zenodo Download Progress

Run this to see how much has been downloaded.

Usage:
    python scripts/check_download_progress.py
    
    # Watch mode (updates every 30 seconds)
    python scripts/check_download_progress.py --watch

(c) 2025 Carmen Wrede & Lino Casu
"""
import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
RAW_DIR = PROJECT_DIR / "data" / "schumann" / "raw"

EXPECTED_SIZE_GB = 26.5


def check_progress():
    """Check download progress."""
    zip_path = RAW_DIR / "2013_2017.zip"
    
    if not zip_path.exists():
        print("Download not started yet (file not found)")
        return 0
    
    size_bytes = zip_path.stat().st_size
    size_mb = size_bytes / 1e6
    size_gb = size_bytes / 1e9
    pct = size_gb / EXPECTED_SIZE_GB * 100
    
    print(f"Downloaded: {size_mb:.1f} MB ({size_gb:.2f} GB)")
    print(f"Progress: {pct:.1f}% of {EXPECTED_SIZE_GB} GB")
    print(f"Remaining: {EXPECTED_SIZE_GB - size_gb:.2f} GB")
    
    if pct >= 99.5:
        print("\n✅ Download appears complete!")
        print("Run: python scripts/download_and_process_zenodo.py --process")
    
    return pct


def watch_progress(interval: int = 30):
    """Watch download progress."""
    print(f"Watching download progress (Ctrl+C to stop)")
    print(f"Updating every {interval} seconds\n")
    
    last_size = 0
    last_time = time.time()
    
    try:
        while True:
            zip_path = RAW_DIR / "2013_2017.zip"
            
            if zip_path.exists():
                size_bytes = zip_path.stat().st_size
                size_gb = size_bytes / 1e9
                pct = size_gb / EXPECTED_SIZE_GB * 100
                
                # Calculate speed
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed > 0 and last_size > 0:
                    speed_mbps = (size_bytes - last_size) / elapsed / 1e6
                    remaining_gb = EXPECTED_SIZE_GB - size_gb
                    eta_hours = remaining_gb * 1e9 / (speed_mbps * 1e6) / 3600 if speed_mbps > 0 else 0
                else:
                    speed_mbps = 0
                    eta_hours = 0
                
                print(f"\r[{time.strftime('%H:%M:%S')}] {size_gb:.2f}/{EXPECTED_SIZE_GB} GB ({pct:.1f}%) | "
                      f"Speed: {speed_mbps:.1f} MB/s | ETA: {eta_hours:.1f}h", end='', flush=True)
                
                last_size = size_bytes
                last_time = current_time
                
                if pct >= 99.5:
                    print("\n\n✅ Download complete!")
                    break
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for download to start...", end='', flush=True)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Check Zenodo download progress")
    parser.add_argument('--watch', '-w', action='store_true',
                       help='Watch mode (continuous updates)')
    parser.add_argument('--interval', '-i', type=int, default=30,
                       help='Update interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if args.watch:
        watch_progress(args.interval)
    else:
        check_progress()


if __name__ == "__main__":
    main()
