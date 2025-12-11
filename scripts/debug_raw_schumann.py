#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG: Raw Schumann Data Analysis

Brutaler Reality-Check OHNE Fits, OHNE Normalisierung.

Prüft:
1. Rohdaten - erste 10 Zeilen
2. Einfache delta_seg mit FESTEN klassischen Referenzen
3. Korrelationsmatrix der delta_seg Zeitreihen
4. Scatter-Plots

(c) 2025 Carmen Wrede & Lino Casu
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "debug"


def main():
    print("="*70)
    print("DEBUG: RAW SCHUMANN DATA ANALYSIS")
    print("="*70)
    print("\nKEINE Fits, KEINE Normalisierung, KEINE Eta-Kalibrierung!")
    print()
    
    # =========================================================================
    # 1. ROHDATEN LADEN UND ANZEIGEN
    # =========================================================================
    print("="*70)
    print("1. ROHDATEN")
    print("="*70)
    
    schumann_path = DATA_DIR / "schumann" / "realistic_schumann_2016.csv"
    print(f"\nDatei: {schumann_path}")
    
    df = pd.read_csv(schumann_path)
    
    print(f"\nErste 10 Zeilen (RAW):")
    print("-"*70)
    print(df[['time', 'f1', 'f2', 'f3']].head(10).to_string())
    print("-"*70)
    
    print(f"\nSpalten: {df.columns.tolist()}")
    print(f"Anzahl Datenpunkte: {len(df)}")
    
    # =========================================================================
    # 2. STATISTIK DER ROHDATEN
    # =========================================================================
    print("\n" + "="*70)
    print("2. STATISTIK DER ROHDATEN (in Hz)")
    print("="*70)
    
    for mode in ['f1', 'f2', 'f3']:
        data = df[mode].values
        print(f"\n{mode}:")
        print(f"  Mittelwert: {np.mean(data):.4f} Hz")
        print(f"  Std:        {np.std(data):.4f} Hz")
        print(f"  Min:        {np.min(data):.4f} Hz")
        print(f"  Max:        {np.max(data):.4f} Hz")
        print(f"  Bereich:    {np.max(data) - np.min(data):.4f} Hz")
    
    # Erwartete Werte aus Literatur
    print("\n" + "-"*70)
    print("Erwartete Werte (Literatur):")
    print("  f1: 7.83 ± 0.15 Hz")
    print("  f2: 14.1 ± 0.20 Hz")
    print("  f3: 20.3 ± 0.25 Hz")
    print("-"*70)
    
    # =========================================================================
    # 3. DELTA_SEG MIT FESTEN KLASSISCHEN REFERENZEN
    # =========================================================================
    print("\n" + "="*70)
    print("3. DELTA_SEG MIT FESTEN KLASSISCHEN REFERENZEN")
    print("="*70)
    
    # Feste klassische Frequenzen (KEINE Kalibrierung!)
    # f_n = 7.83 * sqrt(n*(n+1)) / sqrt(2)
    f1_klass = 7.83  # Hz (Grundmode)
    f2_klass = 7.83 * np.sqrt(6) / np.sqrt(2)   # = 7.83 * 1.732 = 13.56 Hz
    f3_klass = 7.83 * np.sqrt(12) / np.sqrt(2)  # = 7.83 * 2.449 = 19.18 Hz
    
    print(f"\nFeste klassische Referenzen (OHNE Fit):")
    print(f"  f1_klass = {f1_klass:.4f} Hz")
    print(f"  f2_klass = {f2_klass:.4f} Hz")
    print(f"  f3_klass = {f3_klass:.4f} Hz")
    
    # Delta_seg berechnen: (f_klass - f_obs) / f_klass
    # KEINE Normalisierung!
    delta_seg_1 = (f1_klass - df['f1'].values) / f1_klass
    delta_seg_2 = (f2_klass - df['f2'].values) / f2_klass
    delta_seg_3 = (f3_klass - df['f3'].values) / f3_klass
    
    print(f"\nFormel: delta_seg_n = (f_n_klass - f_n_obs) / f_n_klass")
    print(f"        (positiv = Frequenz niedriger als klassisch)")
    
    print(f"\nDelta_seg Statistik:")
    print("-"*70)
    for i, (name, ds) in enumerate([('delta_seg_1', delta_seg_1), 
                                      ('delta_seg_2', delta_seg_2), 
                                      ('delta_seg_3', delta_seg_3)], 1):
        print(f"\n{name}:")
        print(f"  Mittelwert: {np.mean(ds):.6f} ({np.mean(ds)*100:.4f}%)")
        print(f"  Std:        {np.std(ds):.6f} ({np.std(ds)*100:.4f}%)")
        print(f"  Min:        {np.min(ds):.6f} ({np.min(ds)*100:.4f}%)")
        print(f"  Max:        {np.max(ds):.6f} ({np.max(ds)*100:.4f}%)")
    
    # =========================================================================
    # 4. KORRELATIONSMATRIX
    # =========================================================================
    print("\n" + "="*70)
    print("4. KORRELATIONSMATRIX DER DELTA_SEG ZEITREIHEN")
    print("="*70)
    
    # Korrelationen berechnen
    corr_12 = np.corrcoef(delta_seg_1, delta_seg_2)[0, 1]
    corr_13 = np.corrcoef(delta_seg_1, delta_seg_3)[0, 1]
    corr_23 = np.corrcoef(delta_seg_2, delta_seg_3)[0, 1]
    
    print(f"\nKorrelationsmatrix:")
    print("-"*40)
    print(f"              delta_seg_1  delta_seg_2  delta_seg_3")
    print(f"delta_seg_1      1.0000      {corr_12:+.4f}      {corr_13:+.4f}")
    print(f"delta_seg_2     {corr_12:+.4f}       1.0000      {corr_23:+.4f}")
    print(f"delta_seg_3     {corr_13:+.4f}      {corr_23:+.4f}       1.0000")
    print("-"*40)
    
    mean_corr = (corr_12 + corr_13 + corr_23) / 3
    print(f"\nMittlere Korrelation: {mean_corr:.4f}")
    
    # SSZ-Erwartung
    print("\n" + "-"*70)
    print("SSZ-ERWARTUNG:")
    print("  Wenn SSZ-Effekt vorhanden: Korrelationen sollten HOCH sein (>0.7)")
    print("  weil alle Moden GLEICH skaliert werden durch D_SSZ")
    print("-"*70)
    
    # =========================================================================
    # 5. CHECK: SIND DIE DATEN SYNTHETISCH?
    # =========================================================================
    print("\n" + "="*70)
    print("5. CHECK: SYNTHETISCHE DATEN?")
    print("="*70)
    
    # Prüfe auf verdächtige Muster
    f1_diff = np.diff(df['f1'].values)
    f2_diff = np.diff(df['f2'].values)
    f3_diff = np.diff(df['f3'].values)
    
    # Korrelation der Differenzen (sollte bei echten Daten niedriger sein)
    diff_corr_12 = np.corrcoef(f1_diff, f2_diff)[0, 1]
    diff_corr_13 = np.corrcoef(f1_diff, f3_diff)[0, 1]
    
    print(f"\nKorrelation der DIFFERENZEN (df/dt):")
    print(f"  corr(df1/dt, df2/dt) = {diff_corr_12:.4f}")
    print(f"  corr(df1/dt, df3/dt) = {diff_corr_13:.4f}")
    
    # Bei synthetischen Daten mit gemeinsamer Variation: hohe Korrelation
    # Bei echten Daten: niedrigere Korrelation wegen unabhängigem Rauschen
    
    if diff_corr_12 > 0.8 and diff_corr_13 > 0.8:
        print("\n  [!] WARNUNG: Sehr hohe Korrelation der Differenzen!")
        print("      Das deutet auf SYNTHETISCHE Daten mit gemeinsamer Variation hin!")
    elif diff_corr_12 > 0.5:
        print("\n  [!] Moderate Korrelation - koennte synthetisch sein")
    else:
        print("\n  [OK] Niedrige Korrelation - konsistent mit echten Daten")
    
    # =========================================================================
    # 6. PLOTS ERSTELLEN
    # =========================================================================
    print("\n" + "="*70)
    print("6. PLOTS ERSTELLEN")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Scatter delta_seg_1 vs delta_seg_2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.scatter(delta_seg_1, delta_seg_2, alpha=0.1, s=1)
    ax.set_xlabel('delta_seg_1')
    ax.set_ylabel('delta_seg_2')
    ax.set_title(f'Mode 1 vs Mode 2\nr = {corr_12:.4f}')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    # Diagonale für perfekte Korrelation
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]), 
              abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='1:1')
    ax.legend()
    
    ax = axes[1]
    ax.scatter(delta_seg_1, delta_seg_3, alpha=0.1, s=1)
    ax.set_xlabel('delta_seg_1')
    ax.set_ylabel('delta_seg_3')
    ax.set_title(f'Mode 1 vs Mode 3\nr = {corr_13:.4f}')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='1:1')
    ax.legend()
    
    ax = axes[2]
    ax.scatter(delta_seg_2, delta_seg_3, alpha=0.1, s=1)
    ax.set_xlabel('delta_seg_2')
    ax.set_ylabel('delta_seg_3')
    ax.set_title(f'Mode 2 vs Mode 3\nr = {corr_23:.4f}')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='1:1')
    ax.legend()
    
    plt.suptitle('Delta_seg Korrelationen (OHNE Fits, OHNE Normalisierung)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'debug_delta_seg_scatter.png', dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'debug_delta_seg_scatter.png'}")
    
    # Plot 2: Histogramme
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, ds) in zip(axes, [('delta_seg_1', delta_seg_1), 
                                      ('delta_seg_2', delta_seg_2), 
                                      ('delta_seg_3', delta_seg_3)]):
        ax.hist(ds, bins=100, alpha=0.7, density=True)
        ax.axvline(np.mean(ds), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(ds):.4f}')
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nMean={np.mean(ds):.4f}, Std={np.std(ds):.4f}')
        ax.legend()
    
    plt.suptitle('Delta_seg Histogramme (OHNE Fits)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'debug_delta_seg_histograms.png', dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'debug_delta_seg_histograms.png'}")
    
    # Plot 3: Zeitreihen
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time_idx = range(len(delta_seg_1))
    
    for ax, (name, ds) in zip(axes, [('delta_seg_1', delta_seg_1), 
                                      ('delta_seg_2', delta_seg_2), 
                                      ('delta_seg_3', delta_seg_3)]):
        ax.plot(time_idx, ds, 'b-', alpha=0.5, linewidth=0.3)
        ax.axhline(np.mean(ds), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(ds):.4f}')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time index (hours)')
    plt.suptitle('Delta_seg Zeitreihen (OHNE Fits)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'debug_delta_seg_timeseries.png', dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'debug_delta_seg_timeseries.png'}")
    
    # =========================================================================
    # 7. FAZIT
    # =========================================================================
    print("\n" + "="*70)
    print("7. FAZIT")
    print("="*70)
    
    print(f"""
ROHDATEN-CHECK:
  f1: {np.mean(df['f1']):.3f} ± {np.std(df['f1']):.3f} Hz (erwartet: 7.83 ± 0.15)
  f2: {np.mean(df['f2']):.3f} ± {np.std(df['f2']):.3f} Hz (erwartet: 14.1 ± 0.20)
  f3: {np.mean(df['f3']):.3f} ± {np.std(df['f3']):.3f} Hz (erwartet: 20.3 ± 0.25)

DELTA_SEG (mit festen klassischen Referenzen):
  delta_seg_1: {np.mean(delta_seg_1)*100:+.3f}% ± {np.std(delta_seg_1)*100:.3f}%
  delta_seg_2: {np.mean(delta_seg_2)*100:+.3f}% ± {np.std(delta_seg_2)*100:.3f}%
  delta_seg_3: {np.mean(delta_seg_3)*100:+.3f}% ± {np.std(delta_seg_3)*100:.3f}%

KORRELATIONEN:
  corr(1,2) = {corr_12:.4f}
  corr(1,3) = {corr_13:.4f}
  corr(2,3) = {corr_23:.4f}
  Mittel    = {mean_corr:.4f}

INTERPRETATION:
""")
    
    if mean_corr > 0.7:
        print("  ✓ HOHE Korrelation → SSZ-Signatur möglich!")
    elif mean_corr > 0.4:
        print("  ~ MODERATE Korrelation → Schwaches SSZ-Signal oder Mischeffekte")
    else:
        print("  ✗ NIEDRIGE Korrelation → Kein klares SSZ-Signal")
        print("    Die Moden variieren UNABHÄNGIG voneinander (dispersiv)")
    
    # Check auf systematische Offsets
    if abs(np.mean(delta_seg_2)) > 0.03 or abs(np.mean(delta_seg_3)) > 0.05:
        print("\n  ⚠️  SYSTEMATISCHE OFFSETS in delta_seg_2 und delta_seg_3!")
        print("      Die klassischen Referenzen passen nicht zu den Daten.")
        print("      Das könnte bedeuten:")
        print("      a) Die Daten sind synthetisch mit anderen Parametern")
        print("      b) Die klassische Formel braucht Korrekturen")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
