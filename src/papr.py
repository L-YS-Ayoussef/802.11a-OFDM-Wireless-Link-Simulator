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

def clip_time(x: np.ndarray, clipping_ratio: float) -> np.ndarray:
    """
    Hard clipping by magnitude. clipping_ratio = A / RMS, where A is clip level.
    """
    x = np.asarray(x, dtype=np.complex128)
    rms = np.sqrt(np.mean(np.abs(x)**2))
    if rms == 0:
        return x.copy()
    A = clipping_ratio * rms
    mag = np.abs(x)
    y = x.copy()
    over = mag > A
    y[over] = A * x[over] / mag[over]
    return y

def bandlimit_filter(x: np.ndarray, os_factor: int = 1) -> np.ndarray:
    """
    Ideal bandlimiting in frequency:
    - Keep only bins corresponding to ACTIVE_SUBCARRIERS (and DC=0 stays 0).
    - Zero out everything else to remove out-of-band clipping regrowth.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    N = params.N_FFT * os_factor
    if x.size != N:
        raise ValueError(f"Expected symbol length {N} (no CP).")

    X = np.fft.fft(x, norm="ortho")
    mask = np.zeros(N, dtype=bool)
    active_bins = params.bins_from_subcarriers(params.ACTIVE_SUBCARRIERS, N)
    mask[active_bins] = True
    # keep DC as is (usually zero)
    mask[params.subcarrier_to_bin(0, N)] = True

    Xf = np.zeros_like(X)
    Xf[mask] = X[mask]
    xf = np.fft.ifft(Xf, norm="ortho")
    return xf

def clip_and_filter_symbol(x: np.ndarray, os_factor: int, clipping_ratio: float, iters: int = 1) -> np.ndarray:
    """
    Iterative clipping and filtering on ONE symbol (useful part only, no CP).
    """
    y = np.asarray(x, dtype=np.complex128).copy()
    for _ in range(int(iters)):
        y = clip_time(y, clipping_ratio)
        y = bandlimit_filter(y, os_factor=os_factor)
    return y
