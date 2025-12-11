#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration and Physical Constants for SSZ Schumann Analysis

Contains all physical constants, default parameters, and configuration
settings used throughout the package.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import json

# =============================================================================
# PHYSICAL CONSTANTS (SI units)
# =============================================================================

# Golden Ratio (fundamental to SSZ theory)
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # ≈ 1.618033988749

# Speed of light in vacuum (m/s)
C_LIGHT = 299792458.0

# Earth radius (m) - mean radius
EARTH_RADIUS = 6.371e6  # 6371 km

# Ionosphere effective height (m) - approximate
IONOSPHERE_HEIGHT = 85.0e3  # 85 km (D-region lower boundary)

# Ideal Schumann fundamental frequency (Hz)
# f1_ideal = c / (2*pi*R) * sqrt(2) ≈ 10.6 Hz
F1_IDEAL = C_LIGHT / (2.0 * math.pi * EARTH_RADIUS) * math.sqrt(2.0)

# Observed Schumann fundamental frequency (Hz)
F1_OBSERVED = 7.83  # Typical observed value

# Effective slowdown factor eta_0 (dimensionless)
# eta_0 = f1_observed / f1_ideal ≈ 0.74
ETA_0_DEFAULT = F1_OBSERVED / F1_IDEAL

# =============================================================================
# SCHUMANN MODE FREQUENCIES (ideal, for reference)
# =============================================================================

def schumann_ideal(n: int) -> float:
    """
    Calculate ideal Schumann resonance frequency for mode n.
    
    Formula:
        f_n = c / (2*pi*R) * sqrt(n*(n+1))
    
    Args:
        n: Mode number (1, 2, 3, ...)
    
    Returns:
        Ideal frequency in Hz
    """
    return C_LIGHT / (2.0 * math.pi * EARTH_RADIUS) * math.sqrt(n * (n + 1))


# Precomputed ideal frequencies for modes 1-7
SCHUMANN_IDEAL = {
    1: schumann_ideal(1),   # ~10.6 Hz
    2: schumann_ideal(2),   # ~18.4 Hz
    3: schumann_ideal(3),   # ~26.0 Hz
    4: schumann_ideal(4),   # ~33.5 Hz
    5: schumann_ideal(5),   # ~41.0 Hz
    6: schumann_ideal(6),   # ~48.4 Hz
    7: schumann_ideal(7),   # ~55.8 Hz
}

# Typical observed frequencies (approximate)
SCHUMANN_OBSERVED_TYPICAL = {
    1: 7.83,
    2: 14.1,
    3: 20.3,
    4: 26.4,
    5: 32.5,
    6: 38.6,
    7: 44.7,
}

# =============================================================================
# STATION METADATA
# =============================================================================

@dataclass
class StationInfo:
    """Metadata for an ELF measurement station."""
    name: str
    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters
    timezone: str  # e.g., "Europe/Madrid"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "timezone": self.timezone,
            "description": self.description,
        }


# Known stations
STATIONS = {
    "sierra_nevada": StationInfo(
        name="Sierra Nevada ELF Station",
        latitude=37.0667,  # ~37°04'N
        longitude=-3.3833,  # ~3°23'W
        altitude=2500.0,  # ~2500 m
        timezone="Europe/Madrid",
        description="University of Granada ELF station, Spain. "
                    "Data from Salinas et al. (2022), 2013-2017."
    ),
}

# =============================================================================
# DATA SOURCE URLS
# =============================================================================

DATA_SOURCES = {
    # Schumann resonance data (Zenodo)
    "schumann_zenodo_base": "https://zenodo.org/records/",
    "schumann_zenodo_2013": "6348930",  # DOI: 10.5281/zenodo.6348930
    "schumann_zenodo_2014": "6348866",
    "schumann_zenodo_2015": "6348852",
    "schumann_zenodo_2016": "6348838",
    "schumann_zenodo_2017": "6348824",
    
    # F10.7 Solar Flux
    "f107_noaa_psl": "https://psl.noaa.gov/data/timeseries/month/SOLAR/",
    "f107_lisird": "https://lasp.colorado.edu/lisird/data/noaa_radio_flux/",
    "f107_swpc": "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json",
    
    # Kp/Ap Index
    "kp_gfz_ftp": "ftp://ftp.gfz-potsdam.de/pub/home/obs/kp-ap/",
    "kp_gfz_web": "https://kp.gfz-potsdam.de/en/data",
    
    # Lightning (WWLLN)
    "wwlln_info": "http://wwlln.net/",
}

# =============================================================================
# ENUMS FOR CONFIGURATION OPTIONS
# =============================================================================

class EtaMode(str, Enum):
    """Mode for eta_0 calibration."""
    FULL_FIT = "full_fit"           # Fit eta_0 on full dataset (current behavior)
    QUIET_INTERVAL = "quiet_interval"  # Fit only on quiet interval (no SSZ)
    FIXED = "fixed"                 # Use fixed value from theory


class DataMode(str, Enum):
    """Mode for data source."""
    SYNTHETIC = "synthetic"         # Use synthetic data
    REAL = "real"                   # Use real Schumann data


class SSZBasisFunction(str, Enum):
    """Basis functions for delta_seg(t) parameterization."""
    CONSTANT = "constant"           # delta_seg = A
    SINUSOIDAL = "sinusoidal"       # delta_seg = A * sin(2*pi*t/T)
    F107_LINEAR = "f107_linear"     # delta_seg = A + B * F10.7
    COMBINED = "combined"           # delta_seg = A * sin(...) + B * F10.7


# =============================================================================
# CLASSICAL PARAMETERS (separated from SSZ)
# =============================================================================

@dataclass
class ClassicalParams:
    """
    Classical Schumann resonance model parameters.
    
    These parameters describe the Earth-ionosphere cavity without SSZ effects.
    """
    # Fundamental constants
    c_light: float = C_LIGHT                    # Speed of light (m/s)
    earth_radius: float = EARTH_RADIUS          # Earth radius (m)
    ionosphere_height: float = IONOSPHERE_HEIGHT  # Ionosphere height (m)
    
    # Effective slowdown factor
    eta_0: float = ETA_0_DEFAULT                # Dimensionless, ~0.74
    
    # Calibration mode
    eta_mode: EtaMode = EtaMode.FULL_FIT
    
    # Quiet interval for two-stage calibration (days from start)
    quiet_interval_days: int = 14
    
    # Fixed eta_0 value (used when eta_mode == FIXED)
    eta_0_fixed: float = 0.74
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "c_light": self.c_light,
            "earth_radius": self.earth_radius,
            "ionosphere_height": self.ionosphere_height,
            "eta_0": self.eta_0,
            "eta_mode": self.eta_mode.value,
            "quiet_interval_days": self.quiet_interval_days,
            "eta_0_fixed": self.eta_0_fixed,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassicalParams":
        return cls(
            c_light=d.get("c_light", C_LIGHT),
            earth_radius=d.get("earth_radius", EARTH_RADIUS),
            ionosphere_height=d.get("ionosphere_height", IONOSPHERE_HEIGHT),
            eta_0=d.get("eta_0", ETA_0_DEFAULT),
            eta_mode=EtaMode(d.get("eta_mode", "full_fit")),
            quiet_interval_days=d.get("quiet_interval_days", 14),
            eta_0_fixed=d.get("eta_0_fixed", 0.74),
        )


# =============================================================================
# SSZ PARAMETERS (separated from classical)
# =============================================================================

@dataclass
class SSZParams:
    """
    SSZ (Segmented Spacetime) model parameters.
    
    These parameters describe the spacetime segmentation effects on Schumann resonances.
    
    Model:
        D_SSZ = 1 + sum_j(w_j * sigma_j)
        f_n^SSZ = f_n^classical / D_SSZ
        
    Parameterization of delta_seg(t):
        delta_seg(t) = A * basis_1(t) + B * basis_2(t) + ...
    """
    
    # Layer weights (must sum to <= 1)
    w_ground: float = 0.0       # Ground layer weight
    w_atmosphere: float = 0.2   # Atmosphere layer weight
    w_ionosphere: float = 0.8   # Ionosphere layer weight
    
    # Basis function for delta_seg(t)
    basis_function: SSZBasisFunction = SSZBasisFunction.SINUSOIDAL
    
    # Amplitude parameters
    amplitude_A: float = 0.02   # Primary amplitude (e.g., sinusoidal)
    amplitude_B: float = 0.0    # Secondary amplitude (e.g., F10.7 coupling)
    
    # Period for sinusoidal basis (days)
    period_days: float = 365.25
    
    # Phase offset (radians)
    phase_offset: float = 0.0
    
    # Noise level for synthetic data
    noise_level: float = 0.01   # Relative noise on frequencies
    
    # Detection thresholds
    ssz_score_threshold: float = 0.5    # Minimum SSZ score for detection
    correlation_threshold: float = 0.7  # Minimum mode correlation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_ground": self.w_ground,
            "w_atmosphere": self.w_atmosphere,
            "w_ionosphere": self.w_ionosphere,
            "basis_function": self.basis_function.value,
            "amplitude_A": self.amplitude_A,
            "amplitude_B": self.amplitude_B,
            "period_days": self.period_days,
            "phase_offset": self.phase_offset,
            "noise_level": self.noise_level,
            "ssz_score_threshold": self.ssz_score_threshold,
            "correlation_threshold": self.correlation_threshold,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SSZParams":
        return cls(
            w_ground=d.get("w_ground", 0.0),
            w_atmosphere=d.get("w_atmosphere", 0.2),
            w_ionosphere=d.get("w_ionosphere", 0.8),
            basis_function=SSZBasisFunction(d.get("basis_function", "sinusoidal")),
            amplitude_A=d.get("amplitude_A", 0.02),
            amplitude_B=d.get("amplitude_B", 0.0),
            period_days=d.get("period_days", 365.25),
            phase_offset=d.get("phase_offset", 0.0),
            noise_level=d.get("noise_level", 0.01),
            ssz_score_threshold=d.get("ssz_score_threshold", 0.5),
            correlation_threshold=d.get("correlation_threshold", 0.7),
        )
    
    @property
    def total_weight(self) -> float:
        """Total layer weight (should be <= 1)."""
        return self.w_ground + self.w_atmosphere + self.w_ionosphere


# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """
    Configuration for SSZ Schumann analysis.
    
    This is the main configuration object that combines classical and SSZ parameters.
    
    Attributes:
        data_dir: Base directory for data files
        output_dir: Directory for output files
        data_mode: Whether to use synthetic or real data
        classical: Classical model parameters
        ssz: SSZ model parameters
        station: Station identifier
        time_resolution: Target time resolution for analysis
        quality_thresholds: Thresholds for data quality filtering
        modes: List of Schumann modes to analyze
        n_null_realizations: Number of null realizations for statistical tests
    """
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Data mode
    data_mode: DataMode = DataMode.SYNTHETIC
    
    # Classical model parameters (separated)
    classical: ClassicalParams = field(default_factory=ClassicalParams)
    
    # SSZ model parameters (separated)
    ssz: SSZParams = field(default_factory=SSZParams)
    
    # Station
    station: str = "sierra_nevada"
    
    # Time resolution
    time_resolution: str = "1H"  # pandas frequency string
    
    # Modes to analyze
    modes: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_amplitude": 0.1,  # Minimum amplitude (arbitrary units)
        "max_width": 5.0,  # Maximum resonance width (Hz)
        "min_f1": 6.0,  # Minimum f1 (Hz)
        "max_f1": 10.0,  # Maximum f1 (Hz)
    })
    
    # Statistical test parameters
    n_null_realizations: int = 1000  # Number of null realizations for p-value estimation
    significance_level: float = 0.05  # Alpha for hypothesis testing
    
    # Legacy compatibility (deprecated, use classical.eta_0)
    eta_0: float = field(default=ETA_0_DEFAULT, repr=False)
    earth_radius: float = field(default=EARTH_RADIUS, repr=False)
    c_light: float = field(default=C_LIGHT, repr=False)
    ssz_params: Dict[str, float] = field(default_factory=lambda: {
        "beta_0": 0.0,
        "beta_1": 0.0,
        "beta_2": 0.0,
    }, repr=False)
    
    def __post_init__(self):
        """Convert paths to Path objects if needed."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to JSON file."""
        if path is None:
            path = self.output_dir / "config.json"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "data_mode": self.data_mode.value,
            "classical": self.classical.to_dict(),
            "ssz": self.ssz.to_dict(),
            "station": self.station,
            "time_resolution": self.time_resolution,
            "modes": self.modes,
            "quality_thresholds": self.quality_thresholds,
            "n_null_realizations": self.n_null_realizations,
            "significance_level": self.significance_level,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # Parse nested objects
        classical_dict = config_dict.get("classical", {})
        ssz_dict = config_dict.get("ssz", {})
        
        return cls(
            data_dir=Path(config_dict.get("data_dir", "data")),
            output_dir=Path(config_dict.get("output_dir", "output")),
            data_mode=DataMode(config_dict.get("data_mode", "synthetic")),
            classical=ClassicalParams.from_dict(classical_dict),
            ssz=SSZParams.from_dict(ssz_dict),
            station=config_dict.get("station", "sierra_nevada"),
            time_resolution=config_dict.get("time_resolution", "1H"),
            modes=config_dict.get("modes", [1, 2, 3]),
            quality_thresholds=config_dict.get("quality_thresholds", {}),
            n_null_realizations=config_dict.get("n_null_realizations", 1000),
            significance_level=config_dict.get("significance_level", 0.05),
        )
    
    def get_station_info(self) -> StationInfo:
        """Get station metadata."""
        return STATIONS.get(self.station, STATIONS["sierra_nevada"])
    
    def get_eta_0(self) -> float:
        """Get effective eta_0 based on calibration mode."""
        if self.classical.eta_mode == EtaMode.FIXED:
            return self.classical.eta_0_fixed
        return self.classical.eta_0


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = Config()
