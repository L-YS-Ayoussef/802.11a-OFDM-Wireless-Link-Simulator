"""
main.py
Run an OFDM simulation with selectable modulation + coding rate.
Generates:
- baseline OFDM waveform
- PAPR CCDF
- clipping+filtering PAPR improvement
- HPA (Rapp) nonlinearity effect
- EVM + PSD plots
- Spectrogram + constellation (TX and after HPA)

Run:
  python main.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import spectrogram

from src import params
from src.coding import encode_with_rate, ndbps_per_symbol
from src.interleaver import interleave_symbol
from src.mapping import bits_to_symbols, NBPSC
from src.ofdm import ofdm_modulate_frame, ofdm_demodulate_frame
from src.papr import papr_per_symbol, ccdf, clip_and_filter_symbol
from src.hpa import (
    RappHPA,
    set_average_power,
    evm_rms,
    cpe_correct,
    complex_gain_correct,
)
from src.utils import plot_ccdf, plot_psd


def _prompt_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(prompt)
        for i, opt in enumerate(options, start=1):
            print(f"  {i}) {opt}")
        s = input("Enter number: ").strip()
        if s.isdigit():
            idx = int(s)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice, try again.\n")


def mp3_to_bits(path: str) -> np.ndarray:
    """
    Read an MP3 file, extract raw PCM samples, and convert to a bitstream (uint8 0/1).
    Requires: pydub + ffmpeg
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "Missing dependency: pydub. Install with: pip install pydub"
        ) from e

    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)  # mono
    raw = audio.raw_data  # bytes (PCM)
    b = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(b, bitorder="big").astype(np.uint8)
    return bits


def build_tx_chain(
    modulation: str,
    rate: str,
    num_symbols: int,
    seed: int = 1,
    info_bits_in: np.ndarray | None = None,
):
    """
    Builds the TX symbol stream (data subcarriers only).

    Returns:
      info_bits: (num_symbols*NDBPS) information bits (uint8)
      data_symbols: (num_symbols, 48) complex mapped data symbols
    """
    nbpsc = NBPSC[modulation]
    ncbps = 48 * nbpsc
    ndbps = ndbps_per_symbol(nbpsc, rate)

    # ----- Input bits -----
    if info_bits_in is None:
        rng = np.random.default_rng(seed)
        info_bits = rng.integers(0, 2, size=num_symbols * ndbps, dtype=np.uint8)
    else:
        info_bits_in = np.asarray(info_bits_in, dtype=np.uint8).reshape(-1)
        need_info = num_symbols * ndbps
        if info_bits_in.size < need_info:
            info_bits = np.concatenate(
                [info_bits_in, np.zeros(need_info - info_bits_in.size, dtype=np.uint8)]
            )
        else:
            info_bits = info_bits_in[:need_info]

    # ----- FEC: mother rate-1/2 then puncture to selected rate -----
    coded = encode_with_rate(info_bits, rate=rate, add_tail=False)

    # Ensure we have exactly num_symbols * NCBPS coded bits
    need = num_symbols * ncbps
    if coded.size < need:
        coded = np.concatenate([coded, np.zeros(need - coded.size, dtype=np.uint8)])
    coded = coded[:need]

    # ----- Interleave per OFDM symbol -----
    inter = np.zeros_like(coded)
    for n in range(num_symbols):
        block = coded[n * ncbps : (n + 1) * ncbps]
        inter[n * ncbps : (n + 1) * ncbps] = interleave_symbol(block, nbpsc=nbpsc)

    # ----- Map bits to constellation symbols -----
    data_symbols = np.zeros((num_symbols, 48), dtype=np.complex128)
    for n in range(num_symbols):
        b = inter[n * ncbps : (n + 1) * ncbps]
        data_symbols[n] = bits_to_symbols(b, modulation=modulation)

    return info_bits, data_symbols


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


def main():
    modulation = _prompt_choice(
        "Select modulation:", ["BPSK", "QPSK", "16QAM", "64QAM"]
    )
    rate = _prompt_choice("Select coding rate:", ["1/2", "2/3", "3/4"])

    num_symbols = int(input("Number of OFDM data symbols (e.g., 50): ").strip() or "50")
    os_factor = int(
        input("Oversampling factor for PAPR/PSD (1,2,4) [default 4]: ").strip() or "4"
    )
    clipping_ratio = float(
        input("Clipping ratio (A/RMS) [default 1.2]: ").strip() or "1.2"
    )
    clip_iters = int(input("Clip+filter iterations [default 2]: ").strip() or "2")
    ibo_db = float(input("HPA Input Backoff (dB) [default 6]: ").strip() or "6")
    rapp_p = float(input("Rapp smoothness p [default 2]: ").strip() or "2")

    use_audio = (
        input("Use MP3 file as input bits? (y/n) [default n]: ").strip().lower() == "y"
    )
    audio_bits = None
    if use_audio:
        path = input("Enter path to MP3 (e.g., test.mp3): ").strip()
        audio_bits = mp3_to_bits(path)

    # Results folder + filename tag
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    tag = f"{modulation}_{rate.replace('/','-')}_N{num_symbols}_OS{os_factor}"

    # ----- TX chain -----
    info_bits, data_syms = build_tx_chain(
        modulation, rate, num_symbols, seed=2, info_bits_in=audio_bits
    )

    # TX constellation (mapped data)
    plot_constellation(
        data_syms, title=f"TX constellation (mapped data) - {modulation} {rate}"
    )

    # ----- OFDM modulation -----
    tx_wave = ofdm_modulate_frame(
        data_syms, os_factor=os_factor, add_cyclic_prefix=True
    )
    tx_wave = set_average_power(tx_wave, 1.0)

    # Sample rate used for plots
    fs = 20e6 * os_factor

    # Spectrogram (TX waveform)
    plot_spectrogram(tx_wave, fs=fs, title=f"Spectrogram (TX waveform) - {tag}")

    # ----- Clipping + filtering -----
    N = params.N_FFT * os_factor
    cp = params.N_CP * os_factor
    sym_len = N + cp

    tx_cf = np.zeros_like(tx_wave)
    for n in range(num_symbols):
        seg = tx_wave[n * sym_len : (n + 1) * sym_len]
        x = seg[cp:]  # useful part only
        y = clip_and_filter_symbol(
            x, os_factor=os_factor, clipping_ratio=clipping_ratio, iters=clip_iters
        )
        tx_cf[n * sym_len : (n + 1) * sym_len] = np.concatenate([y[-cp:], y])

    tx_cf = set_average_power(tx_cf, 1.0)

    # ----- PAPR -----
    paprs = papr_per_symbol(tx_wave, os_factor=os_factor, has_cp=True)
    paprs_cf = papr_per_symbol(tx_cf, os_factor=os_factor, has_cp=True)

    thr_max = max(paprs.max(), paprs_cf.max()) + 1.0
    thr = np.linspace(0, thr_max, 400)

    cc = ccdf(paprs, thr)
    cc_cf = ccdf(paprs_cf, thr)

    plot_ccdf(thr, cc, title=f"PAPR CCDF (baseline) - {modulation} {rate}")
    plot_ccdf(
        thr,
        cc_cf,
        title=f"PAPR CCDF (clip+filter) CR={clipping_ratio}, iters={clip_iters}",
    )

    # ----- HPA model (Rapp) with input backoff -----
    A_sat = 1.0
    hpa = RappHPA(A_sat=A_sat, p=rapp_p)

    target_pin = (A_sat**2) / (10 ** (ibo_db / 10))
    tx_wave_ibo = set_average_power(tx_wave, target_pin)
    tx_cf_ibo = set_average_power(tx_cf, target_pin)

    y_base = hpa(tx_wave_ibo)
    y_cf = hpa(tx_cf_ibo)

    # PSD plots
    plot_psd(y_base, fs_hz=fs, title=f"PSD after HPA (baseline) IBO={ibo_db} dB")
    plot_psd(y_cf, fs_hz=fs, title=f"PSD after HPA (clip+filter) IBO={ibo_db} dB")

    # Spectrogram after HPA (baseline)
    plot_spectrogram(
        y_base, fs=fs, title=f"Spectrogram after HPA (baseline) - IBO={ibo_db} dB"
    )

    # ----- EVM + RX constellation after HPA -----
    _, data_hat_base, _ = ofdm_demodulate_frame(
        y_base, num_symbols=num_symbols, os_factor=os_factor, has_cp=True
    )
    _, data_hat_cf, _ = ofdm_demodulate_frame(
        y_cf, num_symbols=num_symbols, os_factor=os_factor, has_cp=True
    )

    ref = data_syms.reshape(-1)
    est_base = data_hat_base.reshape(-1)
    est_cf = data_hat_cf.reshape(-1)

    est_base_c = complex_gain_correct(ref, est_base)
    est_cf_c = complex_gain_correct(ref, est_cf)

    evm_base = evm_rms(ref, est_base_c)
    evm_cf = evm_rms(ref, est_cf_c)

    plot_constellation(
        est_base_c, title=f"RX constellation after HPA (baseline) - IBO={ibo_db} dB"
    )
    plot_constellation(
        est_cf_c, title=f"RX constellation after HPA (clip+filter) - IBO={ibo_db} dB"
    )

    print("\n--- Results ---")
    print(
        f"Modulation: {modulation}, Rate: {rate}, Symbols: {num_symbols}, OS: {os_factor}x"
    )
    print(
        f"Mean PAPR baseline: {paprs.mean():.2f} dB, 99.9%: {np.quantile(paprs, 0.999):.2f} dB"
    )
    print(
        f"Mean PAPR clip+filter: {paprs_cf.mean():.2f} dB, 99.9%: {np.quantile(paprs_cf, 0.999):.2f} dB"
    )
    print(f"EVM after HPA (baseline, CPE-corrected): {evm_base * 100:.2f}%")
    print(f"EVM after HPA (clip+filter, CPE-corrected): {evm_cf * 100:.2f}%")

    # Save all figures
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.savefig(
            results_dir / f"{tag}_fig{fig_num}.png", dpi=200, bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":
    main()
