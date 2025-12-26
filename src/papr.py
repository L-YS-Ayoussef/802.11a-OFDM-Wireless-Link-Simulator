"""
papr.py
PAPR metrics + CCDF + clipping + filtering (frequency-domain mask).
"""
from __future__ import annotations

import numpy as np
from . import params

def papr_db(x: np.ndarray) -> float:
    """PAPR in dB for a complex sequence."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    p = np.abs(x)**2
    if np.mean(p) == 0:
        return float("-inf")
    return 10*np.log10(np.max(p) / np.mean(p))

def papr_per_symbol(wave: np.ndarray, os_factor: int = 1, has_cp: bool = True) -> np.ndarray:
    """
    Compute PAPR per OFDM symbol over the useful part (without CP by default).
    """
    wave = np.asarray(wave, dtype=np.complex128).reshape(-1)
    N = params.N_FFT * os_factor
    cp = params.N_CP * os_factor if has_cp else 0
    sym_len = N + cp
    num_sym = wave.size // sym_len
    paprs = np.zeros(num_sym, dtype=float)
    for n in range(num_sym):
        seg = wave[n*sym_len:(n+1)*sym_len]
        if has_cp:
            seg = seg[cp:]
        paprs[n] = papr_db(seg)
    return paprs

def ccdf(papr_db_values: np.ndarray, thresholds_db: np.ndarray) -> np.ndarray:
    """Compute CCDF: Pr(PAPR > threshold)."""
    papr_db_values = np.asarray(papr_db_values, dtype=float).reshape(-1)
    thresholds_db = np.asarray(thresholds_db, dtype=float).reshape(-1)
    return np.array([(papr_db_values > th).mean() for th in thresholds_db], dtype=float)
