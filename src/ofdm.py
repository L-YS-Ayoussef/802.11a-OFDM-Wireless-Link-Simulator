"""
ofdm.py
Subcarrier allocation, pilot insertion, IFFT/FFT, CP handling, oversampling helpers.
"""
from __future__ import annotations

import numpy as np
from . import params

def build_ofdm_symbol(data_symbols: np.ndarray, symbol_index: int, n_fft: int = params.N_FFT) -> np.ndarray:
    """
    Build one frequency-domain OFDM symbol of length n_fft bins:
    - Place 48 data symbols on DATA_SUBCARRIERS (excluding pilots).
    - Insert 4 pilots at -21,-7,+7,+21 with sign p[symbol_index] * [1,1,1,-1].

    Returns: X (complex) length n_fft in numpy FFT-bin order.
    """
    data_symbols = np.asarray(data_symbols, dtype=np.complex128).reshape(-1)
    if data_symbols.size != len(params.DATA_SUBCARRIERS):
        raise ValueError(f"Expected {len(params.DATA_SUBCARRIERS)} data symbols, got {data_symbols.size}.")

    X = np.zeros(n_fft, dtype=np.complex128)

    # Data
    data_bins = params.bins_from_subcarriers(params.DATA_SUBCARRIERS, n_fft)
    X[data_bins] = data_symbols

    # Pilots
    pilot_bins = params.bins_from_subcarriers(params.PILOT_SUBCARRIERS, n_fft)
    p = params.pilot_polarity(symbol_index)  # +/-1
    pilots = p * params.PILOT_BASE.astype(np.complex128)
    X[pilot_bins] = pilots

    # DC and unused bins remain 0.
    return X

def ifft_symbol(X: np.ndarray) -> np.ndarray:
    """Orthogonal IFFT (energy-preserving)."""
    return np.fft.ifft(X, norm="ortho")

def fft_symbol(x: np.ndarray) -> np.ndarray:
    """Orthogonal FFT (energy-preserving)."""
    return np.fft.fft(x, norm="ortho")

def add_cp(x: np.ndarray, n_cp: int = params.N_CP) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if n_cp <= 0:
        return x
    return np.concatenate([x[-n_cp:], x])

def remove_cp(xcp: np.ndarray, n_cp: int = params.N_CP) -> np.ndarray:
    xcp = np.asarray(xcp, dtype=np.complex128).reshape(-1)
    if n_cp <= 0:
        return xcp
    return xcp[n_cp:]

def oversample_freq(X64: np.ndarray, os_factor: int) -> np.ndarray:
    """
    Oversample by zero-padding in frequency.
    Input X64 is length 64 in FFT-bin order. Output is length 64*os_factor.
    We copy the occupied subcarriers (indices -26..+26 excluding DC) into the larger FFT grid.
    """
    X64 = np.asarray(X64, dtype=np.complex128).reshape(-1)
    if X64.size != params.N_FFT:
        raise ValueError("Expected 64-point spectrum for oversampling.")
    if os_factor == 1:
        return X64.copy()

    N = params.N_FFT * os_factor
    Xos = np.zeros(N, dtype=np.complex128)

    # Copy bins corresponding to the same logical subcarrier indices k
    # We recreate by reading the logical subcarrier values from X64 using k indices.
    for k in params.ACTIVE_SUBCARRIERS.tolist() + [0]:
        b64 = params.subcarrier_to_bin(k, params.N_FFT)
        bos = params.subcarrier_to_bin(k, N)
        Xos[bos] = X64[b64]
    return Xos

def ofdm_modulate_frame(data_symbols_matrix: np.ndarray, os_factor: int = 1, add_cyclic_prefix: bool = True) -> np.ndarray:
    """
    Modulate multiple OFDM symbols.
    data_symbols_matrix shape: (num_symbols, 48) complex (data only).
    Returns concatenated time-domain waveform (with CP if add_cyclic_prefix).
    If os_factor>1, uses N=64*os_factor IFFT.
    """
    data_symbols_matrix = np.asarray(data_symbols_matrix, dtype=np.complex128)
    if data_symbols_matrix.ndim != 2 or data_symbols_matrix.shape[1] != 48:
        raise ValueError("data_symbols_matrix must have shape (num_symbols, 48).")
    num_sym = data_symbols_matrix.shape[0]

    wave = []
    for n in range(num_sym):
        X64 = build_ofdm_symbol(data_symbols_matrix[n], symbol_index=n, n_fft=params.N_FFT)
        X = oversample_freq(X64, os_factor)
        x = np.fft.ifft(X, norm="ortho")
        if add_cyclic_prefix:
            cp = params.N_CP * os_factor
            x = np.concatenate([x[-cp:], x])
        wave.append(x)
    return np.concatenate(wave)

def ofdm_demodulate_frame(wave: np.ndarray, num_symbols: int, os_factor: int = 1, has_cp: bool = True):
    """
    Demodulate a concatenated OFDM waveform back to frequency domain.
    Returns:
      X_hat: (num_symbols, N) full spectrum in FFT bins
      data_hat: (num_symbols, 48) extracted data subcarriers
      pilot_hat: (num_symbols, 4) extracted pilots
    """
    wave = np.asarray(wave, dtype=np.complex128).reshape(-1)
    N = params.N_FFT * os_factor
    cp = params.N_CP * os_factor if has_cp else 0
    sym_len = N + cp
    if wave.size < num_symbols * sym_len:
        raise ValueError("Waveform too short for requested number of symbols.")
    wave = wave[: num_symbols * sym_len]

    data_bins = params.bins_from_subcarriers(params.DATA_SUBCARRIERS, N)
    pilot_bins = params.bins_from_subcarriers(params.PILOT_SUBCARRIERS, N)

    X_hat = np.zeros((num_symbols, N), dtype=np.complex128)
    data_hat = np.zeros((num_symbols, 48), dtype=np.complex128)
    pilot_hat = np.zeros((num_symbols, 4), dtype=np.complex128)

    for n in range(num_symbols):
        seg = wave[n*sym_len:(n+1)*sym_len]
        if has_cp:
            seg = seg[cp:]
        Xn = np.fft.fft(seg, norm="ortho")
        X_hat[n] = Xn
        data_hat[n] = Xn[data_bins]
        pilot_hat[n] = Xn[pilot_bins]
    return X_hat, data_hat, pilot_hat
