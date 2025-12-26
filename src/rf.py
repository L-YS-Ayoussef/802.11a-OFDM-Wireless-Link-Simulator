# rf.py
from __future__ import annotations
import numpy as np


def _lowpass_fir(cutoff_hz: float, fs: float, num_taps: int = 129) -> np.ndarray:
    """
    Windowed-sinc lowpass FIR. cutoff_hz is one-sided cutoff (Hz).
    """
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2
    fc = cutoff_hz / fs  # normalized (cycles/sample)
    h = 2 * fc * np.sinc(2 * fc * n)  # ideal LPF impulse response
    w = np.hamming(num_taps)
    h = h * w
    h = h / np.sum(h)
    return h.astype(np.float64)


def iq_modulate(
    x_bb: np.ndarray, fs: float, fc: float, phase: float = 0.0
) -> np.ndarray:
    """
    Complex baseband -> real RF:
      s(t) = Re{ x_bb(t) * exp(j(2πf_c t + phase)) }
    """
    x_bb = np.asarray(x_bb, dtype=np.complex128).reshape(-1)
    n = np.arange(x_bb.size)
    t = n / fs
    carrier = np.exp(1j * (2.0 * np.pi * fc * t + phase))
    s_rf = np.real(x_bb * carrier)
    return s_rf.astype(np.float64)


def iq_demodulate(
    x_rf: np.ndarray,
    fs: float,
    fc: float,
    lpf_cutoff_hz: float,
    lpf_taps: int = 129,
    phase: float = 0.0,
) -> np.ndarray:
    """
    Real RF -> complex baseband:
      z(t) = 2 * x_rf(t) * exp(-j(2πf_c t + phase)) then lowpass.
    """
    x_rf = np.asarray(x_rf).reshape(-1)
    # If someone passed complex RF, keep only real part
    x_rf = np.real(x_rf).astype(np.float64)

    n = np.arange(x_rf.size)
    t = n / fs
    lo = np.exp(-1j * (2.0 * np.pi * fc * t + phase))
    z = 2.0 * x_rf * lo  # complex mixed-down

    h = _lowpass_fir(lpf_cutoff_hz, fs, num_taps=lpf_taps)
    # Filter I and Q separately
    zi = np.convolve(np.real(z), h, mode="same")
    zq = np.convolve(np.imag(z), h, mode="same")
    return (zi + 1j * zq).astype(np.complex128)
