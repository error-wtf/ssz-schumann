#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run All SSZ Tests

Executes all SSZ validation tests and generates a summary report.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the ANTI-CAPITALIST SOFTWARE LICENSE v1.4
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Test scripts to run
TEST_SCRIPTS = [
    ("SSZ Correct Predictions", "scripts/test_ssz_correct_predictions.py"),
    ("SSZ Full Scale Test", "scripts/test_ssz_full_scale.py"),
]

def run_test(name: str, script: str) -> tuple:
    """Run a test script and return (success, output)"""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Script: {script}")
    print('='*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        return success, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Test timed out after 300 seconds")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, str(e)


def main():
    print()
    print("#" * 70)
    print("#" + " " * 20 + "SSZ TEST SUITE" + " " * 32 + "#")
    print("#" * 70)
    print()
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print()
    
    results = []
    
    for name, script in TEST_SCRIPTS:
        success, output = run_test(name, script)
        results.append((name, success, output))
    
    # Summary
    print()
    print("#" * 70)
    print("#" + " " * 25 + "SUMMARY" + " " * 36 + "#")
    print("#" * 70)
    print()
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, _ in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {name:<40} {status}")
    
    print()
    print(f"OVERALL: {passed}/{total} test suites passed")
    print()
    
    if passed == total:
        print("=" * 70)
        print("ALL SSZ TESTS PASSED!")
        print("=" * 70)
        print()
        print("VALIDATED PREDICTIONS:")
        print("  1. -44% time dilation at r = 5*r_s")
        print("  2. Universal crossover at r* = 1.387*r_s")
        print("  3. No horizon singularity (D_SSZ finite)")
        print("  4. G79 nebula z_temporal ~ 0.12")
        print("  5. Earth/Schumann null test (Xi ~ 10^-9)")
        print("  6. Mass-independent BH predictions")
        print("  7. Scaling from Earth to SMBH")
        print()
        print("CONCLUSION:")
        print("  SSZ theory is mathematically consistent across all scales.")
        print("  The Schumann null result is EXPLAINED by weak-field limit.")
        print("  Strong-field tests (NS, BH) needed for detection.")
    else:
        print("WARNING: Some tests failed!")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
