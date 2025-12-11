#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layered SSZ Schumann Resonance - Interactive Demo

This script demonstrates the layered SSZ model for Schumann resonances.
Can be run as a standalone script or converted to Jupyter notebook.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

# %% [markdown]
# # Layered SSZ Model for Schumann Resonances
#
# This notebook demonstrates the application of Segmented Spacetime (SSZ) theory
# to Schumann resonance frequencies using a layer-based approach.
#
# ## Physical Model
#
# The Schumann resonance frequencies are modified by SSZ segmentation in each
# atmospheric layer:
#
# $$D_{SSZ} = 1 + \sum_j w_j \cdot \sigma_j$$
#
# where:
# - $j \in \{ground, atmosphere, ionosphere\}$
# - $w_j$ = weight of layer $j$
# - $\sigma_j$ = segmentation parameter of layer $j$
#
# The SSZ-corrected frequency is:
# $$f_n^{(SSZ)} = \frac{f_n^{(classical)}}{D_{SSZ}}$$

# %% Setup
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from ssz_schumann.models.layered_ssz import (
    LayeredSSZConfig,
    D_SSZ_layered,
    f_n_classical,
    f_n_ssz_layered,
    compute_all_modes,
    frequency_shift_estimate,
    print_frequency_table,
)
from ssz_schumann.config import PHI

# %% [markdown]
# ## 1. Layer Configuration
#
# Default weights based on Schumann waveguide physics:
# - Ground: $w_g \approx 0$ (hard boundary)
# - Atmosphere: $w_{atm} \approx 0.2$ (neutral layer)
# - Ionosphere: $w_{iono} \approx 0.8$ (main waveguide boundary)

# %% Layer Configuration
config = LayeredSSZConfig()

print("Layer Configuration:")
print(f"  Ground:     w = {config.ground.weight:.2f}, sigma = {config.ground.sigma:.4f}")
print(f"  Atmosphere: w = {config.atmosphere.weight:.2f}, sigma = {config.atmosphere.sigma:.4f}")
print(f"  Ionosphere: w = {config.ionosphere.weight:.2f}, sigma = {config.ionosphere.sigma:.4f}")
print(f"\nTotal weight: {config.total_weight:.2f}")

# %% [markdown]
# ## 2. Classical Schumann Frequencies
#
# The classical formula (calibrated to f1 = 7.83 Hz):
# $$f_n = f_1 \cdot \frac{\sqrt{n(n+1)}}{\sqrt{2}}$$

# %% Classical Frequencies
print("\nClassical Schumann Frequencies:")
for n in [1, 2, 3, 4, 5]:
    f = f_n_classical(n, f1_ref=7.83)
    print(f"  Mode {n}: f{n} = {f:.2f} Hz")

# %% [markdown]
# ## 3. SSZ Correction with Ionosphere Segmentation
#
# Let's apply a 1% segmentation in the ionosphere:

# %% SSZ Correction
config.ionosphere.sigma = 0.01  # 1% segmentation

d_ssz = D_SSZ_layered(config)
print(f"\nWith sigma_iono = {config.ionosphere.sigma}:")
print(f"  D_SSZ = {d_ssz:.6f}")

print("\nSSZ-Corrected Frequencies:")
results = compute_all_modes(config)
for n, data in results.items():
    print(f"  Mode {n}: {data['f_classical']:.2f} Hz -> {data['f_ssz']:.2f} Hz "
          f"(Df = {data['delta_f']:+.3f} Hz, {data['relative_shift']*100:+.2f}%)")

# %% [markdown]
# ## 4. Frequency Shift Table
#
# How frequency shifts depend on segmentation:

# %% Frequency Shift Table
print_frequency_table(delta_seg_values=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03])

# %% [markdown]
# ## 5. Visualization: Frequency Shift vs Segmentation

# %% Plot: Shift vs Segmentation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Absolute shift
ax1 = axes[0]
delta_seg_range = np.linspace(0, 0.05, 100)

for n in [1, 2, 3]:
    f_class = f_n_classical(n, f1_ref=7.83)
    d_ssz = 1 + delta_seg_range
    f_ssz = f_class / d_ssz
    delta_f = f_ssz - f_class
    
    ax1.plot(delta_seg_range * 100, delta_f, label=f'Mode {n}', linewidth=2)

ax1.axhline(0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(1, color='r', linestyle='--', alpha=0.5, label='1% segmentation')
ax1.set_xlabel('delta_seg (%)')
ax1.set_ylabel('Frequency Shift (Hz)')
ax1.set_title('Absolute Frequency Shift')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Relative shift (should be identical for all modes - SSZ signature!)
ax2 = axes[1]

for n in [1, 2, 3]:
    f_class = f_n_classical(n, f1_ref=7.83)
    d_ssz = 1 + delta_seg_range
    f_ssz = f_class / d_ssz
    rel_shift = (f_ssz - f_class) / f_class * 100
    
    ax2.plot(delta_seg_range * 100, rel_shift, label=f'Mode {n}', linewidth=2)

ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
ax2.axvline(1, color='r', linestyle='--', alpha=0.5, label='1% segmentation')
ax2.set_xlabel('delta_seg (%)')
ax2.set_ylabel('Relative Shift (%)')
ax2.set_title('Relative Frequency Shift (SSZ Signature: All Modes Equal!)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('layered_ssz_shifts.png', dpi=150)
plt.close()
print("Plot saved: layered_ssz_shifts.png")

print("\nKey SSZ Signature: All modes have the SAME relative shift!")
print("This distinguishes SSZ from classical dispersive effects.")

# %% [markdown]
# ## 6. Layer Weight Sensitivity
#
# How do different layer weights affect the result?

# %% Layer Weight Sensitivity
fig, ax = plt.subplots(figsize=(10, 6))

sigma_iono = 0.01  # Fixed 1% ionosphere segmentation
w_iono_range = np.linspace(0.5, 1.0, 50)

for n in [1, 2, 3]:
    f_class = f_n_classical(n, f1_ref=7.83)
    delta_f_list = []
    
    for w_iono in w_iono_range:
        delta_seg_eff = w_iono * sigma_iono
        d_ssz = 1 + delta_seg_eff
        f_ssz = f_class / d_ssz
        delta_f_list.append(f_ssz - f_class)
    
    ax.plot(w_iono_range, delta_f_list, label=f'Mode {n}', linewidth=2)

ax.axvline(0.8, color='r', linestyle='--', alpha=0.5, label='Default w_iono=0.8')
ax.set_xlabel('Ionosphere Weight (w_iono)')
ax.set_ylabel('Frequency Shift (Hz)')
ax.set_title(f'Sensitivity to Ionosphere Weight (sigma_iono = {sigma_iono})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('layer_weight_sensitivity.png', dpi=150)
plt.close()
print("Plot saved: layer_weight_sensitivity.png")

# %% [markdown]
# ## 7. Time-Varying Model
#
# Simulate how frequencies change with ionospheric conditions:

# %% Time-Varying Simulation
from ssz_schumann.models.layered_ssz import sigma_iono_from_proxy, f_n_ssz_timeseries

# Simulate F10.7 proxy (normalized)
np.random.seed(42)
n_days = 180
time = pd.date_range('2016-01-01', periods=n_days*24, freq='1h')

# Daily variation + random noise
t_hours = np.arange(len(time))
F_iono = (
    0.5 * np.sin(2 * np.pi * t_hours / (24 * 27))  # ~27 day solar rotation
    + 0.3 * np.sin(2 * np.pi * t_hours / 24)  # Daily variation
    + 0.2 * np.random.randn(len(time))  # Noise
)
F_iono = (F_iono - F_iono.mean()) / F_iono.std()  # Normalize

# Calculate sigma_iono from proxy
beta_0 = 0.005  # Baseline
beta_1 = 0.003  # Coupling
sigma_iono_t = sigma_iono_from_proxy(F_iono, beta_0, beta_1)

# Calculate frequencies
f1_t = f_n_ssz_timeseries(1, sigma_iono_t, f1_ref=7.83)
f2_t = f_n_ssz_timeseries(2, sigma_iono_t, f1_ref=7.83)
f3_t = f_n_ssz_timeseries(3, sigma_iono_t, f1_ref=7.83)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

ax1 = axes[0]
ax1.plot(time, F_iono, 'b-', alpha=0.7, linewidth=0.5)
ax1.set_ylabel('F_iono (norm)')
ax1.set_title('Ionospheric Proxy')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(time, sigma_iono_t * 100, 'g-', alpha=0.7, linewidth=0.5)
ax2.axhline(beta_0 * 100, color='r', linestyle='--', alpha=0.5, label=f'beta_0 = {beta_0}')
ax2.set_ylabel('sigma_iono (%)')
ax2.set_title('Ionosphere Segmentation')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(time, f1_t, 'b-', alpha=0.7, linewidth=0.5, label='f1')
ax3.axhline(f_n_classical(1), color='r', linestyle='--', alpha=0.5, label='Classical')
ax3.set_ylabel('f1 (Hz)')
ax3.set_title('Mode 1 Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[3]
# Relative shift (should be same for all modes)
rel_shift_1 = (f1_t - f_n_classical(1)) / f_n_classical(1) * 100
rel_shift_2 = (f2_t - f_n_classical(2)) / f_n_classical(2) * 100
rel_shift_3 = (f3_t - f_n_classical(3)) / f_n_classical(3) * 100

ax4.plot(time, rel_shift_1, 'b-', alpha=0.5, linewidth=0.5, label='Mode 1')
ax4.plot(time, rel_shift_2, 'g-', alpha=0.5, linewidth=0.5, label='Mode 2')
ax4.plot(time, rel_shift_3, 'r-', alpha=0.5, linewidth=0.5, label='Mode 3')
ax4.set_ylabel('Relative Shift (%)')
ax4.set_xlabel('Time')
ax4.set_title('Relative Frequency Shift (All Modes - SSZ Signature)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_varying_ssz.png', dpi=150)
plt.close()
print("Plot saved: time_varying_ssz.png")

print("\nNote: All three modes show IDENTICAL relative shifts!")
print("This is the key SSZ signature that distinguishes it from classical effects.")

# %% [markdown]
# ## 8. Summary
#
# ### Key Results:
#
# 1. **Layered Model**: SSZ segmentation can be modeled per atmospheric layer
# 2. **Default Weights**: Ground (0), Atmosphere (0.2), Ionosphere (0.8)
# 3. **1% Segmentation**: Shifts f1 by ~0.08 Hz (within observed variations)
# 4. **SSZ Signature**: All modes shift by the SAME relative amount
#
# ### Testable Prediction:
#
# If SSZ effects are present, we should observe:
# - Uniform relative frequency shifts across all modes
# - Correlation with ionospheric proxies (F10.7, Kp)
# - Systematic deviations from classical dispersive behavior

# %% Final Summary
print("\n" + "=" * 70)
print("LAYERED SSZ MODEL - SUMMARY")
print("=" * 70)
print(f"\nGolden Ratio (phi): {PHI:.6f}")
print(f"Reference f1: 7.83 Hz")
print(f"\nDefault Layer Weights:")
print(f"  Ground:     0.0 (hard boundary)")
print(f"  Atmosphere: 0.2 (neutral layer)")
print(f"  Ionosphere: 0.8 (main waveguide)")
print(f"\nKey Prediction:")
print(f"  1% ionosphere segmentation -> ~0.08 Hz shift in f1")
print(f"  All modes shift by SAME relative amount (SSZ signature)")
print("=" * 70)
