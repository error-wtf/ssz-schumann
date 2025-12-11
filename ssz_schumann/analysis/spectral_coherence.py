#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spektrale Kohärenz-Analyse für SSZ-Schumann-Experiment

Erweiterte Analyse-Methoden zur Detektion von SSZ-Signaturen:
- Spektrale Kohärenz zwischen Moden
- Wavelet-Kohärenz (zeitaufgelöst)
- Phasen-Kohärenz
- Granger-Kausalität

(c) 2025 Carmen Wrede & Lino Casu
Licensed under the Anti-Capitalist Software License v1.4
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from scipy import signal
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_spectral_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechne spektrale Kohärenz zwischen zwei Signalen.
    
    Die Kohärenz misst die lineare Korrelation zwischen zwei Signalen
    als Funktion der Frequenz.
    
    Cxy(f) = |Pxy(f)|² / (Pxx(f) × Pyy(f))
    
    Args:
        x: Erstes Signal
        y: Zweites Signal
        fs: Abtastfrequenz (Hz)
        nperseg: Segmentlänge für Welch-Methode
    
    Returns:
        frequencies: Frequenz-Array
        coherence: Kohärenz-Array (0-1)
    """
    frequencies, coherence = signal.coherence(
        x, y, fs=fs, nperseg=nperseg
    )
    return frequencies, coherence


def compute_cross_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Berechne Kreuzspektrum und Phase zwischen zwei Signalen.
    
    Args:
        x: Erstes Signal
        y: Zweites Signal
        fs: Abtastfrequenz
        nperseg: Segmentlänge
    
    Returns:
        frequencies: Frequenz-Array
        magnitude: Kreuzspektrum-Magnitude
        phase: Phase (Radiant)
    """
    frequencies, Pxy = signal.csd(x, y, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Pxy)
    phase = np.angle(Pxy)
    return frequencies, magnitude, phase


def compute_phase_coherence(
    phases: List[np.ndarray],
) -> float:
    """
    Berechne Phasen-Kohärenz (Phase Locking Value).
    
    PLV = |⟨exp(i×Δφ)⟩|
    
    PLV = 1: Perfekte Phasen-Synchronisation
    PLV = 0: Keine Phasen-Beziehung
    
    Args:
        phases: Liste von Phasen-Arrays
    
    Returns:
        Phase Locking Value (0-1)
    """
    if len(phases) < 2:
        return np.nan
    
    # Phasen-Differenzen
    phase_diffs = []
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            diff = phases[i] - phases[j]
            phase_diffs.append(diff)
    
    # PLV für jedes Paar
    plvs = []
    for diff in phase_diffs:
        plv = np.abs(np.mean(np.exp(1j * diff)))
        plvs.append(plv)
    
    return np.mean(plvs)


def compute_instantaneous_phase(
    x: np.ndarray,
    fs: float = 1.0,
    freq_band: Tuple[float, float] = None,
) -> np.ndarray:
    """
    Berechne instantane Phase via Hilbert-Transformation.
    
    Args:
        x: Signal
        fs: Abtastfrequenz
        freq_band: Optional Bandpass-Filter (f_low, f_high)
    
    Returns:
        Instantane Phase (Radiant)
    """
    # Optional: Bandpass-Filter
    if freq_band is not None:
        nyq = fs / 2
        low = freq_band[0] / nyq
        high = freq_band[1] / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        x = signal.filtfilt(b, a, x)
    
    # Hilbert-Transformation
    analytic = signal.hilbert(x)
    phase = np.angle(analytic)
    
    return phase


def compute_mode_coherence_matrix(
    delta_seg_dict: Dict[int, np.ndarray],
    fs: float = 1.0,
    nperseg: int = 256,
) -> Dict:
    """
    Berechne Kohärenz-Matrix zwischen allen Moden.
    
    Args:
        delta_seg_dict: {mode_n: delta_seg_array}
        fs: Abtastfrequenz
        nperseg: Segmentlänge
    
    Returns:
        Dictionary mit:
            - coherence_matrix: Paarweise Kohärenz
            - mean_coherence: Mittlere Kohärenz
            - frequencies: Frequenz-Array
            - coherence_spectra: Kohärenz vs Frequenz für jedes Paar
    """
    modes = sorted(delta_seg_dict.keys())
    n_modes = len(modes)
    
    if n_modes < 2:
        return {"mean_coherence": np.nan}
    
    # Kohärenz-Matrix
    coherence_matrix = np.zeros((n_modes, n_modes))
    coherence_spectra = {}
    
    for i, m1 in enumerate(modes):
        for j, m2 in enumerate(modes):
            if i == j:
                coherence_matrix[i, j] = 1.0
            elif j > i:
                x = delta_seg_dict[m1]
                y = delta_seg_dict[m2]
                
                # Gleiche Länge sicherstellen
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Spektrale Kohärenz
                freqs, coh = compute_spectral_coherence(x, y, fs, nperseg)
                
                # Mittlere Kohärenz über alle Frequenzen
                mean_coh = np.mean(coh)
                coherence_matrix[i, j] = mean_coh
                coherence_matrix[j, i] = mean_coh
                
                coherence_spectra[f"{m1}_{m2}"] = {
                    "frequencies": freqs,
                    "coherence": coh,
                }
    
    # Mittlere Kohärenz (nur obere Dreiecksmatrix)
    upper_tri = coherence_matrix[np.triu_indices(n_modes, k=1)]
    mean_coherence = np.mean(upper_tri)
    
    return {
        "coherence_matrix": coherence_matrix,
        "mean_coherence": mean_coherence,
        "modes": modes,
        "coherence_spectra": coherence_spectra,
    }


def compute_wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    wavelet: str = 'morl',
    scales: np.ndarray = None,
) -> Dict:
    """
    Berechne Wavelet-Kohärenz (zeitaufgelöst).
    
    Die Wavelet-Kohärenz zeigt, wie die Korrelation zwischen
    zwei Signalen über Zeit und Frequenz variiert.
    
    Args:
        x: Erstes Signal
        y: Zweites Signal
        fs: Abtastfrequenz
        wavelet: Wavelet-Typ ('morl', 'cmor', etc.)
        scales: Wavelet-Skalen
    
    Returns:
        Dictionary mit:
            - coherence: 2D Kohärenz-Array (Zeit × Frequenz)
            - frequencies: Frequenz-Array
            - times: Zeit-Array
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets nicht installiert, verwende vereinfachte Methode")
        return {"coherence": None, "error": "PyWavelets not installed"}
    
    n = len(x)
    
    if scales is None:
        # Standard-Skalen
        scales = np.arange(1, min(128, n // 4))
    
    # Wavelet-Transformation
    coef_x, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)
    coef_y, _ = pywt.cwt(y, scales, wavelet, sampling_period=1/fs)
    
    # Kreuzspektrum
    Wxy = coef_x * np.conj(coef_y)
    
    # Glätten (über Zeit)
    window_size = max(1, n // 20)
    kernel = np.ones(window_size) / window_size
    
    Wxy_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), 
        axis=1, arr=np.abs(Wxy)
    )
    
    Wxx_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=1, arr=np.abs(coef_x)**2
    )
    
    Wyy_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'),
        axis=1, arr=np.abs(coef_y)**2
    )
    
    # Kohärenz
    coherence = Wxy_smooth**2 / (Wxx_smooth * Wyy_smooth + 1e-10)
    
    times = np.arange(n) / fs
    
    return {
        "coherence": coherence,
        "frequencies": freqs,
        "times": times,
        "scales": scales,
    }


def granger_causality_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
) -> Dict:
    """
    Granger-Kausalitäts-Test.
    
    Testet, ob x Granger-kausal für y ist, d.h. ob vergangene
    Werte von x helfen, y vorherzusagen.
    
    Args:
        x: Potentielle Ursache
        y: Potentielle Wirkung
        max_lag: Maximale Verzögerung
    
    Returns:
        Dictionary mit:
            - f_statistic: F-Statistik
            - p_value: p-Wert
            - optimal_lag: Optimale Verzögerung
            - is_causal: Boolean (p < 0.05)
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        
        # Daten vorbereiten
        data = np.column_stack([y, x])
        
        # Test durchführen
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Beste Verzögerung finden (niedrigster p-Wert)
        best_lag = 1
        best_p = 1.0
        best_f = 0.0
        
        for lag in range(1, max_lag + 1):
            if lag in results:
                # F-Test Ergebnisse
                f_stat = results[lag][0]['ssr_ftest'][0]
                p_val = results[lag][0]['ssr_ftest'][1]
                
                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
                    best_lag = lag
        
        return {
            "f_statistic": best_f,
            "p_value": best_p,
            "optimal_lag": best_lag,
            "is_causal": best_p < 0.05,
        }
        
    except ImportError:
        logger.warning("statsmodels nicht installiert")
        return {"error": "statsmodels not installed"}
    except Exception as e:
        logger.warning(f"Granger-Test fehlgeschlagen: {e}")
        return {"error": str(e)}


def compute_transfer_entropy(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 1,
    l: int = 1,
    bins: int = 10,
) -> float:
    """
    Berechne Transfer-Entropie von x nach y.
    
    TE(X→Y) misst die Informationsübertragung von X nach Y.
    
    Args:
        x: Quell-Signal
        y: Ziel-Signal
        k: Einbettungsdimension für y
        l: Einbettungsdimension für x
        bins: Anzahl Bins für Diskretisierung
    
    Returns:
        Transfer-Entropie (bits)
    """
    n = len(x)
    
    if n < k + l + 1:
        return np.nan
    
    # Diskretisieren
    x_disc = np.digitize(x, np.linspace(x.min(), x.max(), bins))
    y_disc = np.digitize(y, np.linspace(y.min(), y.max(), bins))
    
    # Verzögerte Versionen
    y_future = y_disc[k+l:]
    y_past = np.column_stack([y_disc[k+l-i-1:-i-1] for i in range(k)])
    x_past = np.column_stack([x_disc[l-i-1:-(k+i+1)] for i in range(l)])
    
    # Joint und marginale Entropien
    def entropy(data):
        _, counts = np.unique(data, axis=0, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    # H(Y_future | Y_past)
    joint_y = np.column_stack([y_future, y_past])
    h_y_given_ypast = entropy(joint_y) - entropy(y_past)
    
    # H(Y_future | Y_past, X_past)
    joint_all = np.column_stack([y_future, y_past, x_past])
    joint_yx = np.column_stack([y_past, x_past])
    h_y_given_all = entropy(joint_all) - entropy(joint_yx)
    
    # Transfer-Entropie
    te = h_y_given_ypast - h_y_given_all
    
    return max(0, te)  # TE sollte nicht negativ sein


def analyze_ssz_coherence(
    delta_seg_dict: Dict[int, np.ndarray],
    proxies: Optional[pd.DataFrame] = None,
    fs: float = 1.0,
) -> Dict:
    """
    Umfassende Kohärenz-Analyse für SSZ-Detektion.
    
    Args:
        delta_seg_dict: {mode_n: delta_seg_array}
        proxies: DataFrame mit Proxy-Zeitreihen
        fs: Abtastfrequenz
    
    Returns:
        Dictionary mit allen Kohärenz-Metriken
    """
    results = {}
    
    # 1. Mode-Kohärenz-Matrix
    logger.info("Computing mode coherence matrix...")
    mode_coh = compute_mode_coherence_matrix(delta_seg_dict, fs)
    results["mode_coherence"] = mode_coh
    
    # 2. Phasen-Kohärenz
    logger.info("Computing phase coherence...")
    phases = []
    for mode, data in delta_seg_dict.items():
        phase = compute_instantaneous_phase(data, fs)
        phases.append(phase)
    
    plv = compute_phase_coherence(phases)
    results["phase_locking_value"] = plv
    
    # 3. Granger-Kausalität zwischen Moden
    logger.info("Computing Granger causality...")
    modes = sorted(delta_seg_dict.keys())
    granger_results = {}
    
    for i, m1 in enumerate(modes):
        for m2 in modes[i+1:]:
            x = delta_seg_dict[m1]
            y = delta_seg_dict[m2]
            
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            
            # Beide Richtungen
            gc_xy = granger_causality_test(x, y)
            gc_yx = granger_causality_test(y, x)
            
            granger_results[f"{m1}->{m2}"] = gc_xy
            granger_results[f"{m2}->{m1}"] = gc_yx
    
    results["granger_causality"] = granger_results
    
    # 4. Proxy-Kohärenz (falls vorhanden)
    if proxies is not None:
        logger.info("Computing proxy coherence...")
        proxy_coh = {}
        
        # Mittlere delta_seg
        mean_delta = np.mean([delta_seg_dict[m] for m in modes], axis=0)
        
        for col in proxies.columns:
            proxy_data = proxies[col].values
            min_len = min(len(mean_delta), len(proxy_data))
            
            freqs, coh = compute_spectral_coherence(
                mean_delta[:min_len],
                proxy_data[:min_len],
                fs
            )
            
            proxy_coh[col] = {
                "mean_coherence": np.mean(coh),
                "max_coherence": np.max(coh),
                "freq_at_max": freqs[np.argmax(coh)],
            }
        
        results["proxy_coherence"] = proxy_coh
    
    # 5. Zusammenfassung
    results["summary"] = {
        "mean_mode_coherence": mode_coh["mean_coherence"],
        "phase_locking_value": plv,
        "n_modes": len(modes),
    }
    
    # SSZ-Interpretation
    if mode_coh["mean_coherence"] > 0.7 and plv > 0.7:
        results["ssz_interpretation"] = "Strong SSZ signature (high coherence)"
    elif mode_coh["mean_coherence"] > 0.5 or plv > 0.5:
        results["ssz_interpretation"] = "Weak/partial SSZ signature"
    else:
        results["ssz_interpretation"] = "No SSZ signature detected"
    
    logger.info(f"SSZ Coherence Analysis: {results['ssz_interpretation']}")
    
    return results


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    n = 1000
    
    # Synthetische Daten mit gemeinsamer Komponente
    common = 0.02 * np.sin(np.linspace(0, 10*np.pi, n))
    noise = 0.005
    
    delta_seg_dict = {
        1: common + noise * np.random.randn(n),
        2: common + noise * np.random.randn(n),
        3: common + noise * np.random.randn(n),
    }
    
    results = analyze_ssz_coherence(delta_seg_dict)
    
    print("\nSSZ Coherence Analysis Results:")
    print(f"  Mean Mode Coherence: {results['summary']['mean_mode_coherence']:.4f}")
    print(f"  Phase Locking Value: {results['summary']['phase_locking_value']:.4f}")
    print(f"  Interpretation: {results['ssz_interpretation']}")
