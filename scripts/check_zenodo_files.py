#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check Zenodo files for Schumann data."""
import requests

RECORDS = {
    "2013": "6348838",
    "2014": "6348930", 
    "2015": "6348958",
    "2016": "6348972",
}

for year, record_id in RECORDS.items():
    try:
        r = requests.get(f'https://zenodo.org/api/records/{record_id}', timeout=30)
        data = r.json()
        files = data.get('files', [])
        total_gb = sum(f['size'] for f in files) / 1e9
        print(f"\n=== Year {year} (Record {record_id}) - Total: {total_gb:.2f} GB ===")
        for f in sorted(files, key=lambda x: x['size']):
            size_mb = f['size'] / 1e6
            print(f"  {f['key']}: {size_mb:.1f} MB")
    except Exception as e:
        print(f"Error for {year}: {e}")
