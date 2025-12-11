#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Real Data Loaders

Tests the real Schumann and space weather data loaders using
mini-fixtures that simulate real data format.

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Fixture directory
FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestRealSchumannLoader:
    """Tests for real Schumann data loader."""
    
    def test_load_csv_schumann(self):
        """Test loading Schumann data from CSV."""
        from ssz_schumann.data_io.real_schumann_loader import load_real_schumann_data
        
        ds = load_real_schumann_data(
            path=FIXTURE_DIR / "sample_schumann.csv",
            time_column="datetime",
            freq_columns={1: "f1_hz", 2: "f2_hz", 3: "f3_hz"},
            q_columns={1: "Q1", 2: "Q2", 3: "Q3"},
        )
        
        # Check structure
        assert "f_mode1" in ds.data_vars
        assert "f_mode2" in ds.data_vars
        assert "f_mode3" in ds.data_vars
        assert "q_mode1" in ds.data_vars
        
        # Check time dimension
        assert "time" in ds.dims
        assert len(ds.time) == 20
        
        # Check values are in expected range
        assert 7.0 < ds.f_mode1.mean() < 9.0
        assert 13.0 < ds.f_mode2.mean() < 15.0
        assert 19.0 < ds.f_mode3.mean() < 22.0
    
    def test_validate_schumann_data(self):
        """Test Schumann data validation."""
        from ssz_schumann.data_io.real_schumann_loader import (
            load_real_schumann_data,
            validate_schumann_data,
        )
        
        ds = load_real_schumann_data(
            path=FIXTURE_DIR / "sample_schumann.csv",
            time_column="datetime",
            freq_columns={1: "f1_hz", 2: "f2_hz", 3: "f3_hz"},
        )
        
        result = validate_schumann_data(ds)
        
        assert result["valid"]
        assert "mode1" in result["statistics"]
        assert result["statistics"]["mode1"]["mean"] > 7.0
    
    def test_convert_to_standard_format(self):
        """Test conversion to standard format."""
        from ssz_schumann.data_io.real_schumann_loader import (
            load_real_schumann_data,
            convert_to_standard_format,
        )
        
        ds_raw = load_real_schumann_data(
            path=FIXTURE_DIR / "sample_schumann.csv",
            time_column="datetime",
            freq_columns={1: "f1_hz", 2: "f2_hz", 3: "f3_hz"},
        )
        
        ds_std = convert_to_standard_format(ds_raw)
        
        # Standard format uses f1, f2, f3
        assert "f1" in ds_std.data_vars
        assert "f2" in ds_std.data_vars
        assert "f3" in ds_std.data_vars
    
    def test_missing_file_error(self):
        """Test error handling for missing file."""
        from ssz_schumann.data_io.real_schumann_loader import load_real_schumann_data
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_real_schumann_data(
                path="nonexistent_file.csv",
                time_column="time",
                freq_columns={1: "f1"},
            )
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_missing_column_error(self):
        """Test error handling for missing column."""
        from ssz_schumann.data_io.real_schumann_loader import load_real_schumann_data
        
        with pytest.raises(ValueError) as exc_info:
            load_real_schumann_data(
                path=FIXTURE_DIR / "sample_schumann.csv",
                time_column="datetime",
                freq_columns={1: "nonexistent_column"},
            )
        
        assert "not found" in str(exc_info.value).lower()


class TestRealSpaceWeatherLoader:
    """Tests for real space weather data loaders."""
    
    def test_load_f107(self):
        """Test loading F10.7 data."""
        from ssz_schumann.data_io.real_space_weather_loader import load_f107
        
        f107 = load_f107(
            path=FIXTURE_DIR / "sample_f107.csv",
            time_column="date",
            value_column="f107",
        )
        
        assert isinstance(f107, pd.Series)
        assert len(f107) == 10
        assert f107.name == "f107"
        assert isinstance(f107.index, pd.DatetimeIndex)
        
        # Check values are in expected range
        assert 100 < f107.mean() < 120
    
    def test_load_kp(self):
        """Test loading Kp data."""
        from ssz_schumann.data_io.real_space_weather_loader import load_kp
        
        kp = load_kp(
            path=FIXTURE_DIR / "sample_kp.csv",
            time_column="datetime",
            value_column="kp",
        )
        
        assert isinstance(kp, pd.Series)
        assert len(kp) == 10
        assert kp.name == "kp"
        
        # Check values are in expected range (0-9)
        assert 0 <= kp.min() <= 9
        assert 0 <= kp.max() <= 9
    
    def test_resample_to_match(self):
        """Test resampling to match target index."""
        from ssz_schumann.data_io.real_space_weather_loader import (
            load_f107,
            resample_to_match,
        )
        
        f107 = load_f107(
            path=FIXTURE_DIR / "sample_f107.csv",
            time_column="date",
            value_column="f107",
        )
        
        # Create hourly target index
        target_index = pd.date_range("2016-01-01", "2016-01-02", freq="1h")
        
        resampled = resample_to_match(f107, target_index, method="nearest")
        
        assert len(resampled) == len(target_index)
        assert not resampled.isna().all()
    
    def test_load_space_weather_from_config(self):
        """Test loading from config dictionary."""
        from ssz_schumann.data_io.real_space_weather_loader import (
            load_space_weather_from_config,
        )
        
        config = {
            "f107": {
                "path": str(FIXTURE_DIR / "sample_f107.csv"),
                "time_column": "date",
                "value_column": "f107",
            },
            "kp": {
                "path": str(FIXTURE_DIR / "sample_kp.csv"),
                "time_column": "datetime",
                "value_column": "kp",
            },
        }
        
        data = load_space_weather_from_config(config)
        
        assert "f107" in data
        assert "kp" in data
        assert len(data["f107"]) == 10
        assert len(data["kp"]) == 10


class TestUnifiedLoader:
    """Tests for unified data loader."""
    
    def test_load_synthetic_data(self):
        """Test loading synthetic data."""
        from ssz_schumann.data_io.unified_loader import load_all_data
        
        data = load_all_data("synthetic")
        
        assert data.data_source_type == "synthetic"
        assert data.n_points > 0
        assert len(data.f107) > 0
        assert len(data.kp) > 0
    
    def test_unified_data_get_frequencies(self):
        """Test frequency extraction from UnifiedData."""
        from ssz_schumann.data_io.unified_loader import load_all_data
        
        data = load_all_data("synthetic")
        freqs = data.get_frequencies()
        
        assert 1 in freqs
        assert 2 in freqs
        assert 3 in freqs
        
        # Check frequency values are reasonable
        assert 7.0 < freqs[1].mean() < 9.0
        assert 13.0 < freqs[2].mean() < 15.0
    
    def test_unified_data_summary(self):
        """Test summary generation."""
        from ssz_schumann.data_io.unified_loader import load_all_data
        
        data = load_all_data("synthetic")
        summary = data.summary()
        
        assert "synthetic" in summary.lower()
        assert "points" in summary.lower()
    
    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        from ssz_schumann.data_io.unified_loader import UnifiedDataConfig
        
        config_dict = {
            "data_source": {
                "type": "synthetic",
                "schumann": {
                    "path": "test.csv",
                    "freq_columns": {1: "f1", 2: "f2", 3: "f3"},
                },
            },
        }
        
        config = UnifiedDataConfig.from_dict(config_dict)
        
        assert config.data_source_type == "synthetic"
        assert config.schumann_path == "test.csv"


class TestIntegrationRealPipeline:
    """Integration tests for real data pipeline."""
    
    def test_real_pipeline_smoke(self):
        """Smoke test: real data pipeline runs without error."""
        from ssz_schumann.data_io.real_schumann_loader import (
            load_real_schumann_data,
            convert_to_standard_format,
        )
        from ssz_schumann.models.classical_schumann import (
            f_n_classical,
            compute_eta0_from_mean_f1,
        )
        from ssz_schumann.models.ssz_correction import (
            delta_seg_from_observed,
            check_mode_consistency,
        )
        
        # Load fixture data
        ds = load_real_schumann_data(
            path=FIXTURE_DIR / "sample_schumann.csv",
            time_column="datetime",
            freq_columns={1: "f1_hz", 2: "f2_hz", 3: "f3_hz"},
        )
        ds = convert_to_standard_format(ds)
        
        # Calibrate eta
        f1_obs = ds.f1.values
        eta_0 = compute_eta0_from_mean_f1(f1_obs)
        
        assert 0.7 < eta_0 < 0.8
        
        # Compute classical frequencies
        f_classical = {n: f_n_classical(n, eta_0) for n in [1, 2, 3]}
        
        # Extract delta_seg
        delta_seg = {}
        for n in [1, 2, 3]:
            f_obs = ds[f"f{n}"].values
            delta_seg[n] = delta_seg_from_observed(f_obs, f_classical[n])
        
        # Check mode consistency
        consistency = check_mode_consistency(delta_seg)
        
        assert "mean_correlation" in consistency
        assert "ssz_score" in consistency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
