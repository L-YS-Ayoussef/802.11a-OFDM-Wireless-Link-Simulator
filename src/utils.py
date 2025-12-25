"""
utils.py
Plot helpers and convenience functions.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_ccdf(thresholds_db: np.ndarray, ccdf_vals: np.ndarray, title: str = "PAPR CCDF"):
    plt.figure()
    plt.semilogy(thresholds_db, ccdf_vals)
    plt.grid(True, which="both")
    plt.xlabel("PAPR threshold (dB)")
    plt.ylabel("Pr(PAPR > threshold)")
    plt.title(title)

def plot_psd(x: np.ndarray, fs_hz: float, title: str = "PSD (Welch)"):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    f, Pxx = welch(x, fs=fs_hz, nperseg=min(4096, len(x)), return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    plt.figure()
    plt.plot(f/1e6, 10*np.log10(Pxx + 1e-20))
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(title)
