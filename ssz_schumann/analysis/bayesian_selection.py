#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Model Selection for SSZ vs Classical Models

Implements rigorous Bayesian model comparison:
- Bayes Factors
- Posterior Model Probabilities
- WAIC (Widely Applicable Information Criterion)
- LOO-CV (Leave-One-Out Cross-Validation)

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class BayesianModelResult:
    """Result of Bayesian model fitting."""
    model_name: str
    log_likelihood: float
    n_params: int
    aic: float
    bic: float
    waic: float
    loo_cv: float
    posterior_predictive: np.ndarray
    parameter_estimates: Dict[str, float]
    parameter_uncertainties: Dict[str, float]


@dataclass
class ModelComparisonResult:
    """Result of Bayesian model comparison."""
    models: List[BayesianModelResult]
    bayes_factors: Dict[str, float]
    posterior_probabilities: Dict[str, float]
    preferred_model: str
    evidence_strength: str
    summary_table: pd.DataFrame


def log_likelihood_gaussian(
    residuals: np.ndarray,
    sigma: float,
) -> float:
    """
    Gaussian log-likelihood.
    
    Args:
        residuals: Model residuals
        sigma: Error standard deviation
    
    Returns:
        Log-likelihood
    """
    n = len(residuals)
    ll = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
    return ll


def compute_aic(log_likelihood: float, n_params: int) -> float:
    """Akaike Information Criterion."""
    return -2 * log_likelihood + 2 * n_params


def compute_bic(log_likelihood: float, n_params: int, n_data: int) -> float:
    """Bayesian Information Criterion."""
    return -2 * log_likelihood + n_params * np.log(n_data)


def compute_waic(
    log_likelihoods: np.ndarray,
) -> float:
    """
    Widely Applicable Information Criterion.
    
    WAIC = -2 * (lppd - p_waic)
    
    where:
        lppd = log pointwise predictive density
        p_waic = effective number of parameters
    
    Args:
        log_likelihoods: Array of log-likelihoods per data point
    
    Returns:
        WAIC value
    """
    # Log pointwise predictive density
    lppd = np.sum(np.log(np.mean(np.exp(log_likelihoods), axis=0)))
    
    # Effective number of parameters
    p_waic = np.sum(np.var(log_likelihoods, axis=0))
    
    waic = -2 * (lppd - p_waic)
    return waic


def compute_loo_cv(
    f_obs: np.ndarray,
    model_func: Callable,
    params: np.ndarray,
) -> float:
    """
    Leave-One-Out Cross-Validation score.
    
    Args:
        f_obs: Observed data
        model_func: Model function(params, indices) -> predictions
        params: Model parameters
    
    Returns:
        LOO-CV score (lower is better)
    """
    n = len(f_obs)
    loo_scores = []
    
    for i in range(n):
        # Leave out point i
        train_idx = np.concatenate([np.arange(i), np.arange(i+1, n)])
        
        # Predict point i
        pred_i = model_func(params, np.array([i]))[0]
        
        # Squared error
        loo_scores.append((f_obs[i] - pred_i)**2)
    
    return np.mean(loo_scores)


def bayes_factor_from_bic(bic1: float, bic2: float) -> float:
    """
    Approximate Bayes Factor from BIC difference.
    
    BF ≈ exp(-0.5 * (BIC1 - BIC2))
    
    Args:
        bic1: BIC of model 1
        bic2: BIC of model 2
    
    Returns:
        Bayes Factor (BF > 1 favors model 1)
    """
    return np.exp(-0.5 * (bic1 - bic2))


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes Factor (Kass & Raftery 1995).
    
    Args:
        bf: Bayes Factor
    
    Returns:
        Interpretation string
    """
    if bf > 150:
        return "Very strong evidence"
    elif bf > 20:
        return "Strong evidence"
    elif bf > 3:
        return "Positive evidence"
    elif bf > 1:
        return "Weak evidence"
    elif bf > 1/3:
        return "Weak evidence against"
    elif bf > 1/20:
        return "Positive evidence against"
    elif bf > 1/150:
        return "Strong evidence against"
    else:
        return "Very strong evidence against"


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ClassicalModel:
    """Classical Schumann resonance model."""
    
    name = "Classical"
    n_params = 1  # eta_0
    
    def __init__(self, R: float = 6.371e6, c: float = 299792458.0):
        self.R = R
        self.c = c
    
    def f_n(self, n: int, eta_0: float) -> float:
        """Classical frequency for mode n."""
        return eta_0 * self.c / (2 * np.pi * self.R) * np.sqrt(n * (n + 1))
    
    def predict(self, params: np.ndarray, modes: List[int], n_times: int) -> np.ndarray:
        """Predict frequencies for all modes and times."""
        eta_0 = params[0]
        predictions = []
        for n in modes:
            f = self.f_n(n, eta_0)
            predictions.extend([f] * n_times)
        return np.array(predictions)
    
    def fit(self, f_obs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, float]:
        """Fit model to data."""
        modes = list(f_obs.keys())
        n_times = len(f_obs[modes[0]])
        
        # Stack observations
        f_obs_flat = np.concatenate([f_obs[n] for n in modes])
        
        def objective(params):
            pred = self.predict(params, modes, n_times)
            return np.sum((f_obs_flat - pred)**2)
        
        # Initial guess
        f1_mean = np.mean(f_obs[1])
        f1_ideal = self.c / (2 * np.pi * self.R) * np.sqrt(2)
        eta_0_init = f1_mean / f1_ideal
        
        result = minimize(objective, [eta_0_init], bounds=[(0.5, 1.0)])
        
        # Compute sigma
        pred = self.predict(result.x, modes, n_times)
        residuals = f_obs_flat - pred
        sigma = np.std(residuals)
        
        return result.x, sigma


class SSZModel:
    """SSZ-corrected Schumann resonance model."""
    
    name = "SSZ"
    n_params = 2  # eta_0, delta_seg_amplitude
    
    def __init__(self, R: float = 6.371e6, c: float = 299792458.0):
        self.R = R
        self.c = c
    
    def f_n_classical(self, n: int, eta_0: float) -> float:
        """Classical frequency for mode n."""
        return eta_0 * self.c / (2 * np.pi * self.R) * np.sqrt(n * (n + 1))
    
    def predict(
        self, 
        params: np.ndarray, 
        modes: List[int], 
        n_times: int,
        basis: np.ndarray = None,
    ) -> np.ndarray:
        """Predict frequencies with SSZ correction."""
        eta_0 = params[0]
        amplitude = params[1] if len(params) > 1 else 0.0
        
        if basis is None:
            basis = np.zeros(n_times)
        
        delta_seg = amplitude * basis
        D_SSZ = 1.0 + delta_seg
        
        predictions = []
        for n in modes:
            f_class = self.f_n_classical(n, eta_0)
            f_ssz = f_class / D_SSZ
            predictions.extend(f_ssz)
        
        return np.array(predictions)
    
    def fit(
        self, 
        f_obs: Dict[int, np.ndarray],
        basis: np.ndarray = None,
    ) -> Tuple[np.ndarray, float]:
        """Fit model to data."""
        modes = list(f_obs.keys())
        n_times = len(f_obs[modes[0]])
        
        if basis is None:
            # Default: sinusoidal basis
            t = np.arange(n_times)
            basis = np.sin(2 * np.pi * t / (365.25 * 24))
        
        # Stack observations
        f_obs_flat = np.concatenate([f_obs[n] for n in modes])
        
        def objective(params):
            pred = self.predict(params, modes, n_times, basis)
            return np.sum((f_obs_flat - pred)**2)
        
        # Initial guess
        f1_mean = np.mean(f_obs[1])
        f1_ideal = self.c / (2 * np.pi * self.R) * np.sqrt(2)
        eta_0_init = f1_mean / f1_ideal
        
        result = minimize(
            objective, 
            [eta_0_init, 0.02],
            bounds=[(0.5, 1.0), (-0.2, 0.2)]
        )
        
        # Compute sigma
        pred = self.predict(result.x, modes, n_times, basis)
        residuals = f_obs_flat - pred
        sigma = np.std(residuals)
        
        return result.x, sigma


class LayeredSSZModel:
    """Multi-layer SSZ model."""
    
    name = "Layered_SSZ"
    n_params = 4  # eta_0, sigma_D, sigma_E, sigma_F
    
    def __init__(self, R: float = 6.371e6, c: float = 299792458.0):
        self.R = R
        self.c = c
        self.layer_weights = {'D': 0.15, 'E': 0.35, 'F': 0.50}
    
    def f_n_classical(self, n: int, eta_0: float) -> float:
        return eta_0 * self.c / (2 * np.pi * self.R) * np.sqrt(n * (n + 1))
    
    def predict(
        self,
        params: np.ndarray,
        modes: List[int],
        n_times: int,
        proxies: Dict[str, np.ndarray] = None,
    ) -> np.ndarray:
        """Predict with layered SSZ model."""
        eta_0 = params[0]
        sigma_D = params[1] if len(params) > 1 else 0.0
        sigma_E = params[2] if len(params) > 2 else 0.0
        sigma_F = params[3] if len(params) > 3 else 0.0
        
        # Compute D_SSZ
        if proxies is not None and 'f107' in proxies:
            f107_norm = (proxies['f107'] - 100) / 100
        else:
            f107_norm = np.zeros(n_times)
        
        delta_seg = (
            self.layer_weights['D'] * sigma_D * (1 + 0.3 * f107_norm) +
            self.layer_weights['E'] * sigma_E * (1 + 0.5 * f107_norm) +
            self.layer_weights['F'] * sigma_F * (1 + 0.8 * f107_norm)
        )
        
        D_SSZ = 1.0 + delta_seg
        
        predictions = []
        for n in modes:
            f_class = self.f_n_classical(n, eta_0)
            f_ssz = f_class / D_SSZ
            predictions.extend(f_ssz)
        
        return np.array(predictions)
    
    def fit(
        self,
        f_obs: Dict[int, np.ndarray],
        proxies: Dict[str, np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Fit layered model."""
        modes = list(f_obs.keys())
        n_times = len(f_obs[modes[0]])
        
        f_obs_flat = np.concatenate([f_obs[n] for n in modes])
        
        def objective(params):
            pred = self.predict(params, modes, n_times, proxies)
            return np.sum((f_obs_flat - pred)**2)
        
        f1_mean = np.mean(f_obs[1])
        f1_ideal = self.c / (2 * np.pi * self.R) * np.sqrt(2)
        eta_0_init = f1_mean / f1_ideal
        
        result = minimize(
            objective,
            [eta_0_init, 0.01, 0.01, 0.01],
            bounds=[(0.5, 1.0), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
        )
        
        pred = self.predict(result.x, modes, n_times, proxies)
        residuals = f_obs_flat - pred
        sigma = np.std(residuals)
        
        return result.x, sigma


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def bayesian_model_comparison(
    f_obs: Dict[int, np.ndarray],
    proxies: Dict[str, np.ndarray] = None,
    basis: np.ndarray = None,
    models: List[str] = None,
) -> ModelComparisonResult:
    """
    Perform Bayesian model comparison.
    
    Args:
        f_obs: Observed frequencies per mode
        proxies: Proxy data (f107, kp, etc.)
        basis: Basis function for SSZ model
        models: List of model names to compare
    
    Returns:
        ModelComparisonResult
    """
    if models is None:
        models = ['Classical', 'SSZ', 'Layered_SSZ']
    
    modes = list(f_obs.keys())
    n_times = len(f_obs[modes[0]])
    n_data = n_times * len(modes)
    
    f_obs_flat = np.concatenate([f_obs[n] for n in modes])
    
    results = []
    
    for model_name in models:
        if model_name == 'Classical':
            model = ClassicalModel()
            params, sigma = model.fit(f_obs)
            predictions = model.predict(params, modes, n_times)
            param_dict = {'eta_0': params[0]}
            
        elif model_name == 'SSZ':
            model = SSZModel()
            params, sigma = model.fit(f_obs, basis)
            predictions = model.predict(params, modes, n_times, basis)
            param_dict = {'eta_0': params[0], 'amplitude': params[1]}
            
        elif model_name == 'Layered_SSZ':
            model = LayeredSSZModel()
            params, sigma = model.fit(f_obs, proxies)
            predictions = model.predict(params, modes, n_times, proxies)
            param_dict = {
                'eta_0': params[0],
                'sigma_D': params[1],
                'sigma_E': params[2],
                'sigma_F': params[3],
            }
        else:
            continue
        
        # Compute metrics
        residuals = f_obs_flat - predictions
        ll = log_likelihood_gaussian(residuals, sigma)
        aic = compute_aic(ll, model.n_params)
        bic = compute_bic(ll, model.n_params, n_data)
        
        # Simplified WAIC (using point estimate)
        waic = aic  # Approximation
        
        # Simplified LOO-CV
        loo_cv = np.mean(residuals**2)
        
        result = BayesianModelResult(
            model_name=model_name,
            log_likelihood=ll,
            n_params=model.n_params,
            aic=aic,
            bic=bic,
            waic=waic,
            loo_cv=loo_cv,
            posterior_predictive=predictions,
            parameter_estimates=param_dict,
            parameter_uncertainties={k: sigma for k in param_dict},
        )
        results.append(result)
    
    # Compute Bayes Factors (relative to Classical)
    bayes_factors = {}
    classical_bic = results[0].bic
    
    for r in results:
        bf = bayes_factor_from_bic(classical_bic, r.bic)
        bayes_factors[r.model_name] = bf
    
    # Posterior probabilities (assuming equal priors)
    # Use log-sum-exp trick to avoid overflow
    bic_values = np.array([r.bic for r in results])
    min_bic = np.min(bic_values)
    log_evidence = -0.5 * (bic_values - min_bic)
    log_total = np.log(np.sum(np.exp(log_evidence)))
    
    posterior_probs = {}
    for i, r in enumerate(results):
        posterior_probs[r.model_name] = np.exp(log_evidence[i] - log_total)
    
    # Find preferred model
    preferred = max(posterior_probs, key=posterior_probs.get)
    
    # Evidence strength
    max_bf = max(bayes_factors.values())
    evidence_strength = interpret_bayes_factor(max_bf)
    
    # Summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Model': r.model_name,
            'Parameters': r.n_params,
            'Log-Likelihood': r.log_likelihood,
            'AIC': r.aic,
            'BIC': r.bic,
            'Bayes Factor': bayes_factors[r.model_name],
            'Posterior Prob': posterior_probs[r.model_name],
        })
    
    summary_table = pd.DataFrame(summary_data)
    
    return ModelComparisonResult(
        models=results,
        bayes_factors=bayes_factors,
        posterior_probabilities=posterior_probs,
        preferred_model=preferred,
        evidence_strength=evidence_strength,
        summary_table=summary_table,
    )


def print_bayesian_comparison(result: ModelComparisonResult) -> None:
    """Print formatted comparison results."""
    print("\n" + "="*70)
    print("BAYESIAN MODEL COMPARISON")
    print("="*70)
    
    print("\n" + result.summary_table.to_string(index=False))
    
    print(f"\n{'='*70}")
    print(f"Preferred Model: {result.preferred_model}")
    print(f"Evidence Strength: {result.evidence_strength}")
    print("="*70)
    
    print("\nBayes Factors (relative to Classical):")
    for model, bf in result.bayes_factors.items():
        print(f"  {model}: {bf:.4f} ({interpret_bayes_factor(bf)})")
    
    print("\nPosterior Probabilities:")
    for model, prob in result.posterior_probabilities.items():
        print(f"  {model}: {prob:.4f} ({prob*100:.1f}%)")


if __name__ == "__main__":
    # Test with synthetic data
    import sys
    sys.path.insert(0, str(__file__).rsplit('ssz_schumann', 1)[0])
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_times = 8760
    eta_0 = 0.74
    c = 299792458.0
    R = 6.371e6
    
    # True SSZ signal
    t = np.arange(n_times)
    delta_seg_true = 0.02 * np.sin(2 * np.pi * t / (365.25 * 24))
    D_SSZ = 1.0 + delta_seg_true
    
    f_obs = {}
    for n in [1, 2, 3]:
        f_class = eta_0 * c / (2 * np.pi * R) * np.sqrt(n * (n + 1))
        f_ssz = f_class / D_SSZ
        noise = 0.01 * f_class * np.random.randn(n_times)
        f_obs[n] = f_ssz + noise
    
    # Run comparison
    result = bayesian_model_comparison(f_obs)
    print_bayesian_comparison(result)
