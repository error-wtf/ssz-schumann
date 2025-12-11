#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for SSZ Schumann Experiment

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="ssz-schumann",
    version="0.1.0",
    author="Carmen Wrede & Lino Casu",
    author_email="",
    description="SSZ Analysis of Schumann Resonances",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/error-wtf/ssz-schuhman-experiment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "full": [
            "seaborn>=0.11.0",
            "statsmodels>=0.13.0",
            "jupyter>=1.0.0",
            "netCDF4>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ssz-schumann=scripts.run_schumann_ssz_analysis:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="schumann resonance, ssz, segmented spacetime, elf, ionosphere",
)
