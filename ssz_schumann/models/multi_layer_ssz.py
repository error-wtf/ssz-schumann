#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Layer SSZ Model

Extended layered model with physical parameters for each atmospheric layer.

Layers:
    1. D-Layer (60-90 km): Daytime ionization by X-rays and Lyman-alpha
    2. E-Layer (90-150 km): Daytime ionization by UV
    3. F-Layer (150-500 km): Main ionospheric layer, EUV ionization

Each layer contributes to the effective SSZ segmentation:
    D_SSZ = 1 + Σ_j w_j · σ_j(t)

where σ_j depends on:
    - Solar flux (F10.7)
    - Geomagnetic activity (Kp)
    - Local time (day/night)
    - Season

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize


@dataclass
class LayerConfig:
    """Configuration for a single atmospheric layer."""
    name: str
    h_min: float  # Minimum height (km)
    h_max: float  # Maximum height (km)
    weight: float  # Layer weight in D_SSZ
    
    # Proxy sensitivities
    f107_sensitivity: float = 0.0  # Sensitivity to F10.7
    kp_sensitivity: float = 0.0    # Sensitivity to Kp
    diurnal_amplitude: float = 0.0  # Day/night variation amplitude
    seasonal_amplitude: float = 0.0  # Seasonal variation amplitude
    
    # Physical parameters
    base_electron_density: float = 1e10  # m^-3
    scale_height: float = 10.0  # km
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'h_min': self.h_min,
            'h_max': self.h_max,
            'weight': self.weight,
            'f107_sensitivity': self.f107_sensitivity,
            'kp_sensitivity': self.kp_sensitivity,
            'diurnal_amplitude': self.diurnal_amplitude,
            'seasonal_amplitude': self.seasonal_amplitude,
        }


@dataclass
class MultiLayerSSZModel:
    """
    Multi-layer SSZ model with physical parameterization.
    
    Model:
        D_SSZ(t) = 1 + Σ_j w_j · σ_j(t)
        
        σ_j(t) = σ_j0 · [1 + a_F · (F10.7 - 100)/100 
                          + a_K · Kp/5 
                          + a_D · cos(2π·hour/24)
                          + a_S · cos(2π·doy/365)]
    """
    
    layers: List[LayerConfig] = field(default_factory=list)
    
    # Global parameters
    sigma_0: float = 0.01  # Base segmentation amplitude
    
    def __post_init__(self):
        if not self.layers:
            # Default ionospheric layers
            self.layers = [
                LayerConfig(
                    name='D',
                    h_min=60, h_max=90,
                    weight=0.15,
                    f107_sensitivity=0.3,
                    kp_sensitivity=0.1,
                    diurnal_amplitude=0.8,  # Strong day/night
                    seasonal_amplitude=0.2,
                    base_electron_density=1e8,
                ),
                LayerConfig(
                    name='E',
                    h_min=90, h_max=150,
                    weight=0.35,
                    f107_sensitivity=0.5,
                    kp_sensitivity=0.2,
                    diurnal_amplitude=0.6,
                    seasonal_amplitude=0.3,
                    base_electron_density=1e11,
                ),
                LayerConfig(
                    name='F',
                    h_min=150, h_max=500,
                    weight=0.50,
                    f107_sensitivity=0.8,
                    kp_sensitivity=0.4,
                    diurnal_amplitude=0.3,
                    seasonal_amplitude=0.4,
                    base_electron_density=1e12,
                ),
            ]
    
    @property
    def total_weight(self) -> float:
        return sum(layer.weight for layer in self.layers)
    
    def compute_layer_sigma(
        self,
        layer: LayerConfig,
        f107: np.ndarray,
        kp: np.ndarray,
        hour: np.ndarray,
        doy: np.ndarray,
    ) -> np.ndarray:
        """
        Compute segmentation parameter for a single layer.
        
        Args:
            layer: Layer configuration
            f107: F10.7 solar flux (SFU)
            kp: Kp index (0-9)
            hour: Hour of day (0-24)
            doy: Day of year (1-366)
        
        Returns:
            Layer segmentation parameter σ_j(t)
        """
        # Normalize inputs
        f107_norm = (f107 - 100) / 100  # Centered around 100 SFU
        kp_norm = kp / 5  # Normalized to ~1 at moderate activity
        
        # Diurnal variation (maximum at local noon)
        diurnal = np.cos(2 * np.pi * (hour - 12) / 24)
        
        # Seasonal variation (maximum at summer solstice)
        seasonal = np.cos(2 * np.pi * (doy - 172) / 365)
        
        # Compute sigma
        sigma = self.sigma_0 * (
            1.0
            + layer.f107_sensitivity * f107_norm
            + layer.kp_sensitivity * kp_norm
            + layer.diurnal_amplitude * diurnal
            + layer.seasonal_amplitude * seasonal
        )
        
        return sigma
    
    def compute_D_SSZ(
        self,
        f107: np.ndarray,
        kp: np.ndarray,
        hour: np.ndarray,
        doy: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute total SSZ correction factor.
        
        Args:
            f107: F10.7 solar flux
            kp: Kp index
            hour: Hour of day
            doy: Day of year
        
        Returns:
            Tuple of (D_SSZ, layer_contributions)
        """
        layer_sigmas = {}
        total_sigma = np.zeros_like(f107, dtype=float)
        
        for layer in self.layers:
            sigma = self.compute_layer_sigma(layer, f107, kp, hour, doy)
            layer_sigmas[layer.name] = sigma
            total_sigma += layer.weight * sigma
        
        D_SSZ = 1.0 + total_sigma
        
        return D_SSZ, layer_sigmas
    
    def compute_frequencies(
        self,
        f_classical: Dict[int, float],
        f107: np.ndarray,
        kp: np.ndarray,
        hour: np.ndarray,
        doy: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Compute SSZ-corrected frequencies for all modes.
        
        Args:
            f_classical: Classical frequencies per mode
            f107, kp, hour, doy: Time-varying parameters
        
        Returns:
            Dictionary of frequency arrays per mode
        """
        D_SSZ, _ = self.compute_D_SSZ(f107, kp, hour, doy)
        
        f_ssz = {}
        for n, f_class in f_classical.items():
            f_ssz[n] = f_class / D_SSZ
        
        return f_ssz
    
    def fit_to_data(
        self,
        f_obs: Dict[int, np.ndarray],
        f_classical: Dict[int, float],
        f107: np.ndarray,
        kp: np.ndarray,
        hour: np.ndarray,
        doy: np.ndarray,
        fit_weights: bool = True,
        fit_sensitivities: bool = True,
    ) -> Dict:
        """
        Fit model parameters to observed data.
        
        Args:
            f_obs: Observed frequencies per mode
            f_classical: Classical frequencies per mode
            f107, kp, hour, doy: Proxy data
            fit_weights: Whether to fit layer weights
            fit_sensitivities: Whether to fit proxy sensitivities
        
        Returns:
            Dictionary with fit results
        """
        modes = list(f_obs.keys())
        n_times = len(f107)
        
        # Build parameter vector
        params_init = [self.sigma_0]
        param_names = ['sigma_0']
        
        if fit_weights:
            for layer in self.layers:
                params_init.append(layer.weight)
                param_names.append(f'w_{layer.name}')
        
        if fit_sensitivities:
            for layer in self.layers:
                params_init.append(layer.f107_sensitivity)
                param_names.append(f'a_F_{layer.name}')
                params_init.append(layer.kp_sensitivity)
                param_names.append(f'a_K_{layer.name}')
        
        def objective(params):
            # Unpack parameters
            idx = 0
            self.sigma_0 = params[idx]
            idx += 1
            
            if fit_weights:
                for layer in self.layers:
                    layer.weight = params[idx]
                    idx += 1
            
            if fit_sensitivities:
                for layer in self.layers:
                    layer.f107_sensitivity = params[idx]
                    idx += 1
                    layer.kp_sensitivity = params[idx]
                    idx += 1
            
            # Compute model frequencies
            f_model = self.compute_frequencies(f_classical, f107, kp, hour, doy)
            
            # Compute residuals
            residuals = []
            for n in modes:
                residuals.extend(f_obs[n] - f_model[n])
            
            return np.sum(np.array(residuals)**2)
        
        # Bounds
        bounds = [(0.001, 0.1)]  # sigma_0
        if fit_weights:
            bounds.extend([(0.0, 1.0)] * len(self.layers))
        if fit_sensitivities:
            bounds.extend([(-1.0, 1.0)] * (2 * len(self.layers)))
        
        # Optimize
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            bounds=bounds,
        )
        
        # Compute final metrics
        f_model = self.compute_frequencies(f_classical, f107, kp, hour, doy)
        
        residuals = []
        for n in modes:
            residuals.extend(f_obs[n] - f_model[n])
        residuals = np.array(residuals)
        
        rmse = np.sqrt(np.mean(residuals**2))
        
        # R-squared
        f_obs_flat = np.concatenate([f_obs[n] for n in modes])
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((f_obs_flat - np.mean(f_obs_flat))**2)
        r_squared = 1.0 - ss_res / ss_tot
        
        return {
            'success': result.success,
            'sigma_0': self.sigma_0,
            'layer_weights': {layer.name: layer.weight for layer in self.layers},
            'layer_f107_sens': {layer.name: layer.f107_sensitivity for layer in self.layers},
            'layer_kp_sens': {layer.name: layer.kp_sensitivity for layer in self.layers},
            'rmse': rmse,
            'r_squared': r_squared,
            'n_params': len(params_init),
        }
    
    def to_dict(self) -> Dict:
        return {
            'sigma_0': self.sigma_0,
            'layers': [layer.to_dict() for layer in self.layers],
            'total_weight': self.total_weight,
        }


def create_default_model() -> MultiLayerSSZModel:
    """Create model with default ionospheric layers."""
    return MultiLayerSSZModel()


def create_simple_model() -> MultiLayerSSZModel:
    """Create simplified 2-layer model."""
    return MultiLayerSSZModel(
        layers=[
            LayerConfig(
                name='lower',
                h_min=60, h_max=120,
                weight=0.3,
                f107_sensitivity=0.4,
                kp_sensitivity=0.2,
            ),
            LayerConfig(
                name='upper',
                h_min=120, h_max=500,
                weight=0.7,
                f107_sensitivity=0.7,
                kp_sensitivity=0.3,
            ),
        ]
    )
