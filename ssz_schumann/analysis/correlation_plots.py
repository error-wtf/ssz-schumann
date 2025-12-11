#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation and Visualization Functions

Plotting functions for SSZ Schumann analysis.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    1: '#1f77b4',  # Blue
    2: '#ff7f0e',  # Orange
    3: '#2ca02c',  # Green
    'classical': '#7f7f7f',  # Gray
    'ssz': '#d62728',  # Red
    'f107': '#9467bd',  # Purple
    'kp': '#8c564b',  # Brown
}


def plot_timeseries(
    ds: xr.Dataset,
    variables: List[str] = ["f1", "f2", "f3"],
    title: str = "Schumann Resonance Frequencies",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot time series of Schumann frequencies.
    
    Args:
        ds: Dataset with time series
        variables: Variables to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, sharex=True)
    
    if n_vars == 1:
        axes = [axes]
    
    time = pd.DatetimeIndex(ds.time.values)
    
    for i, var in enumerate(variables):
        ax = axes[i]
        
        if var in ds:
            data = ds[var].values
            
            # Determine color
            if var.startswith("f"):
                mode = int(var[1]) if var[1].isdigit() else 1
                color = COLORS.get(mode, 'black')
            elif "delta" in var:
                color = COLORS['ssz']
            else:
                color = 'black'
            
            ax.plot(time, data, color=color, linewidth=0.5, alpha=0.7)
            
            # Add rolling mean
            if len(data) > 24:
                rolling = pd.Series(data).rolling(24, center=True).mean()
                ax.plot(time, rolling, color=color, linewidth=2, label=f"{var} (24h mean)")
            
            ax.set_ylabel(ds[var].attrs.get("units", ""))
            ax.set_title(ds[var].attrs.get("long_name", var))
            ax.legend(loc="upper right")
    
    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_scatter_delta_vs_feature(
    delta_seg: pd.Series,
    feature: pd.Series,
    feature_name: str = "F10.7",
    fit_line: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter plot of delta_seg vs. ionospheric feature.
    
    Args:
        delta_seg: SSZ correction factor
        feature: Feature values (e.g., F10.7)
        feature_name: Name for axis label
        fit_line: Whether to add regression line
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Align data
    common_idx = delta_seg.index.intersection(feature.index)
    x = feature.loc[common_idx].values
    y = delta_seg.loc[common_idx].values
    
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.3, s=10, color=COLORS['ssz'])
    
    # Fit line
    if fit_line and len(x) > 10:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        
        ax.plot(x_line, y_line, 'k-', linewidth=2,
               label=f"y = {slope:.4f}x + {intercept:.4f}\nR² = {r_value**2:.4f}, p = {p_value:.2e}")
        ax.legend(loc="best")
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel("delta_seg")
    ax.set_title(f"SSZ Correction vs. {feature_name}")
    
    # Add zero line
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_mode_consistency(
    delta_seg_dict: Dict[int, pd.Series],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot mode consistency analysis.
    
    Shows:
    - Time series of delta_seg for each mode
    - Pairwise scatter plots
    - Correlation matrix
    
    Args:
        delta_seg_dict: {mode: delta_seg_series}
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    modes = sorted(delta_seg_dict.keys())
    n_modes = len(modes)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # Top: Time series comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    for n in modes:
        ds = delta_seg_dict[n]
        if isinstance(ds, pd.Series):
            # Rolling mean for clarity
            rolling = ds.rolling(24, center=True).mean()
            ax1.plot(ds.index, rolling, color=COLORS.get(n, 'black'),
                    label=f"Mode {n}", linewidth=1.5)
    
    ax1.set_xlabel("Time")
    ax1.set_ylabel("delta_seg")
    ax1.set_title("SSZ Correction Factor by Mode (24h rolling mean)")
    ax1.legend()
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Bottom left: Pairwise scatter (mode 1 vs mode 2)
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 1 in delta_seg_dict and 2 in delta_seg_dict:
        ds1 = delta_seg_dict[1]
        ds2 = delta_seg_dict[2]
        
        common_idx = ds1.index.intersection(ds2.index)
        x = ds1.loc[common_idx].values
        y = ds2.loc[common_idx].values
        
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        ax2.scatter(x, y, alpha=0.2, s=5)
        
        # 1:1 line
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax2.plot(lims, lims, 'r--', label="1:1 line")
        
        # Correlation
        if len(x) > 10:
            corr = np.corrcoef(x, y)[0, 1]
            ax2.set_title(f"Mode 1 vs Mode 2 (r = {corr:.3f})")
        
        ax2.set_xlabel("delta_seg (mode 1)")
        ax2.set_ylabel("delta_seg (mode 2)")
        ax2.legend()
    
    # Bottom right: Histogram of differences
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 1 in delta_seg_dict and 2 in delta_seg_dict:
        diff = ds1.loc[common_idx].values - ds2.loc[common_idx].values
        diff = diff[~np.isnan(diff)]
        
        ax3.hist(diff, bins=50, alpha=0.7, color=COLORS['ssz'])
        ax3.axvline(0, color='black', linestyle='--')
        ax3.axvline(np.mean(diff), color='red', linestyle='-',
                   label=f"Mean = {np.mean(diff):.6f}")
        ax3.axvline(np.mean(diff) + np.std(diff), color='red', linestyle=':')
        ax3.axvline(np.mean(diff) - np.std(diff), color='red', linestyle=':')
        
        ax3.set_xlabel("delta_seg(mode 1) - delta_seg(mode 2)")
        ax3.set_ylabel("Count")
        ax3.set_title("Mode Difference Distribution")
        ax3.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_diurnal_pattern(
    ds: xr.Dataset,
    variable: str = "delta_seg_mean",
    hour_var: str = "local_hour",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot diurnal (daily) pattern of variable.
    
    Args:
        ds: Dataset
        variable: Variable to analyze
        hour_var: Hour variable name
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if variable not in ds or hour_var not in ds:
        logger.warning(f"Variables {variable} or {hour_var} not in dataset")
        return fig
    
    # Group by hour
    df = pd.DataFrame({
        "hour": ds[hour_var].values,
        "value": ds[variable].values,
    })
    
    hourly = df.groupby(df["hour"].round())["value"].agg(["mean", "std", "count"])
    
    # Plot
    ax.errorbar(hourly.index, hourly["mean"], yerr=hourly["std"],
               fmt='o-', capsize=3, color=COLORS['ssz'])
    
    ax.set_xlabel("Local Hour")
    ax.set_ylabel(variable)
    ax.set_title(f"Diurnal Pattern of {variable}")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    
    # Add day/night shading
    ax.axvspan(6, 18, alpha=0.1, color='yellow', label='Daytime')
    ax.axvspan(0, 6, alpha=0.1, color='blue')
    ax.axvspan(18, 24, alpha=0.1, color='blue', label='Nighttime')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_seasonal_pattern(
    ds: xr.Dataset,
    variable: str = "delta_seg_mean",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot seasonal pattern of variable.
    
    Args:
        ds: Dataset
        variable: Variable to analyze
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if variable not in ds:
        logger.warning(f"Variable {variable} not in dataset")
        return fig
    
    # Get month
    time = pd.DatetimeIndex(ds.time.values)
    
    df = pd.DataFrame({
        "month": time.month,
        "value": ds[variable].values,
    })
    
    monthly = df.groupby("month")["value"].agg(["mean", "std", "count"])
    
    # Plot
    ax.errorbar(monthly.index, monthly["mean"], yerr=monthly["std"],
               fmt='o-', capsize=3, color=COLORS['ssz'], markersize=8)
    
    ax.set_xlabel("Month")
    ax.set_ylabel(variable)
    ax.set_title(f"Seasonal Pattern of {variable}")
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def create_summary_figure(
    ds: xr.Dataset,
    results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create comprehensive summary figure.
    
    Includes:
    - Frequency time series
    - delta_seg time series
    - Mode consistency
    - Model comparison
    
    Args:
        ds: Dataset
        results: Analysis results dictionary
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    time = pd.DatetimeIndex(ds.time.values)
    
    # Top left: f1 time series
    ax1 = fig.add_subplot(gs[0, 0])
    if "f1" in ds:
        ax1.plot(time, ds["f1"].values, alpha=0.3, linewidth=0.5, color=COLORS[1])
        if len(ds["f1"]) > 24:
            rolling = pd.Series(ds["f1"].values).rolling(24*7, center=True).mean()
            ax1.plot(time, rolling, color=COLORS[1], linewidth=2, label="f1 (weekly mean)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_title("Schumann f1")
        ax1.legend()
    
    # Top right: delta_seg time series
    ax2 = fig.add_subplot(gs[0, 1])
    if "delta_seg_mean" in ds:
        ax2.plot(time, ds["delta_seg_mean"].values, alpha=0.3, linewidth=0.5, color=COLORS['ssz'])
        if len(ds["delta_seg_mean"]) > 24:
            rolling = pd.Series(ds["delta_seg_mean"].values).rolling(24*7, center=True).mean()
            ax2.plot(time, rolling, color=COLORS['ssz'], linewidth=2, label="delta_seg (weekly mean)")
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel("delta_seg")
        ax2.set_title("SSZ Correction Factor")
        ax2.legend()
    
    # Middle left: Mode comparison
    ax3 = fig.add_subplot(gs[1, 0])
    for n in [1, 2, 3]:
        if f"delta_seg_{n}" in ds:
            rolling = pd.Series(ds[f"delta_seg_{n}"].values).rolling(24*7, center=True).mean()
            ax3.plot(time, rolling, color=COLORS.get(n, 'black'), label=f"Mode {n}")
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel("delta_seg")
    ax3.set_title("Mode Comparison (weekly mean)")
    ax3.legend()
    
    # Middle right: delta_seg vs F10.7
    ax4 = fig.add_subplot(gs[1, 1])
    if "delta_seg_mean" in ds and "f107_norm" in ds:
        x = ds["f107_norm"].values
        y = ds["delta_seg_mean"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        ax4.scatter(x[valid], y[valid], alpha=0.1, s=5, color=COLORS['ssz'])
        ax4.set_xlabel("F10.7 (normalized)")
        ax4.set_ylabel("delta_seg")
        ax4.set_title("SSZ Correction vs. Solar Activity")
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Bottom: Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = []
    summary_text.append(f"Data points: {results.get('n_timepoints', 'N/A')}")
    summary_text.append(f"eta_0: {results.get('eta_0', 'N/A'):.6f}" if results.get('eta_0') else "eta_0: N/A")
    
    mc = results.get("mode_consistency", {})
    summary_text.append(f"SSZ Score: {mc.get('ssz_score', 'N/A'):.4f}" if mc.get('ssz_score') else "SSZ Score: N/A")
    
    cm = results.get("classical_model", {})
    summary_text.append(f"Classical R²: {cm.get('r_squared', 'N/A'):.4f}" if cm.get('r_squared') else "Classical R²: N/A")
    
    sm = results.get("ssz_model", {})
    if sm:
        summary_text.append(f"SSZ R²: {sm.get('r_squared', 'N/A'):.4f}" if sm.get('r_squared') else "SSZ R²: N/A")
    
    comp = results.get("model_comparison", {})
    if comp:
        summary_text.append(f"Preferred: {comp.get('preferred_model', 'N/A')}")
    
    ax5.text(0.5, 0.5, "\n".join(summary_text),
            transform=ax5.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')
    
    fig.suptitle("SSZ Schumann Analysis Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig
