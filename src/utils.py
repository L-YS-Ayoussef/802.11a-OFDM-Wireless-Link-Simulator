"""
utils.py
Plot helpers and convenience functions.
"""

from __future__ import annotations
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.signal import spectrogram
from .papr import papr_per_symbol, ccdf


def plot_papr_ccdf(
    wave_baseline: np.ndarray,
    wave_cf: np.ndarray | None = None,
    *,
    os_factor: int = 1,
    has_cp: bool = True,
    thresholds_db: np.ndarray | None = None,
    title: str = "PAPR CCDF",
    label_baseline: str = "baseline",
    label_cf: str = "clip+filter",
):
    wave_baseline = np.asarray(wave_baseline, dtype=np.complex128).reshape(-1)

    papr_base_db = papr_per_symbol(wave_baseline, os_factor=os_factor, has_cp=has_cp)
    if papr_base_db.size == 0:
        print(
            "[plot_papr_ccdf] No complete OFDM symbols in baseline waveform. Skipping CCDF plot."
        )
        return papr_base_db, None, None, None, None

    papr_cf_db = None
    if wave_cf is not None:
        wave_cf = np.asarray(wave_cf, dtype=np.complex128).reshape(-1)
        papr_cf_db = papr_per_symbol(wave_cf, os_factor=os_factor, has_cp=has_cp)
        if papr_cf_db.size == 0:
            papr_cf_db = None  # treat as missing

    # Threshold grid (if not provided)
    if thresholds_db is None:
        max_papr = float(papr_base_db.max())
        if papr_cf_db is not None:
            max_papr = max(max_papr, float(papr_cf_db.max()))
        hi = float(np.ceil(max_papr + 1.0))
        thresholds_db = np.arange(0.0, hi + 1e-9, 0.1)

    thresholds_db = np.asarray(thresholds_db, dtype=float).reshape(-1)

    ccdf_base = ccdf(papr_base_db, thresholds_db)
    ccdf_cf = ccdf(papr_cf_db, thresholds_db) if papr_cf_db is not None else None

    plt.figure()
    plt.semilogy(thresholds_db, ccdf_base, label=label_baseline)
    if ccdf_cf is not None:
        plt.semilogy(thresholds_db, ccdf_cf, label=label_cf)

    plt.grid(True, which="both")
    plt.xlabel("PAPR threshold (dB)")
    plt.ylabel("Pr(PAPR > threshold)")
    plt.title(title)
    plt.legend()

    return papr_base_db, papr_cf_db, thresholds_db, ccdf_base, ccdf_cf

def plot_psd(x: np.ndarray, fs_hz: float, title: str = "PSD (Welch)"):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    f, Pxx = welch(x, fs=fs_hz, nperseg=min(4096, len(x)), return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    plt.figure()
    plt.plot(f / 1e6, 10 * np.log10(Pxx + 1e-20))
    plt.grid(True)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(title)


def plot_constellation(sym: np.ndarray, title: str):
    sym = np.asarray(sym, dtype=np.complex128).reshape(-1)
    plt.figure()
    plt.scatter(sym.real, sym.imag, s=6, alpha=0.35)
    plt.grid(True)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("I")
    plt.ylabel("Q")


def plot_spectrogram(x: np.ndarray, fs: float, title: str):
    """
    Spectrogram for complex baseband:
    uses return_onesided=False and shifts frequency axis.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        nperseg=2048,
        noverlap=1536,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    plt.figure()
    plt.pcolormesh(t * 1e3, f / 1e6, 10 * np.log10(Sxx + 1e-20), shading="auto")
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.colorbar(label="PSD (dB/Hz)")


def set_average_power(x: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    p = np.mean(np.abs(x) ** 2)
    if p == 0:
        return x.copy()
    return x * np.sqrt(target_power / p)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    n = (b.size // 8) * 8
    b = b[:n].reshape(-1, 8)
    vals = np.packbits(b, axis=1, bitorder="big").reshape(-1)
    return vals.tobytes()


def evm_rms(ref: np.ndarray, est: np.ndarray) -> float:
    """
    RMS EVM: sqrt(E|e|^2 / E|ref|^2)
    """
    ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
    est = np.asarray(est, dtype=np.complex128).reshape(-1)
    e = est - ref
    num = np.mean(np.abs(e) ** 2)
    den = np.mean(np.abs(ref) ** 2)
    if den == 0:
        return float("nan")
    return float(np.sqrt(num / den))


# def cpe_correct(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
#     """
#     Optional: correct a single common phase error by rotating est to best match ref.
#     """
#     ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
#     est = np.asarray(est, dtype=np.complex128).reshape(-1)
#     # minimize ||ref - est*exp(jphi)|| => phi = angle(sum(ref*conj(est)))
#     phi = np.angle(np.sum(ref * np.conj(est)))
#     return est * np.exp(1j * phi)


def complex_gain_correct(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """
    Find complex scalar g minimizing ||ref - g*est||^2
    Then return g*est (aligns both amplitude and phase).
    """
    ref = ref.reshape(-1)
    est = est.reshape(-1)
    g = np.vdot(est, ref) / np.vdot(est, est)
    return g * est

def write_wav_from_pcm_bytes(
    out_path: str, pcm_bytes: bytes, fs: int, nch: int = 1, sampwidth: int = 2
):
    with wave.open(out_path, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sampwidth)
        w.setframerate(fs)
        w.writeframes(pcm_bytes)
