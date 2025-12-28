"""
transmitter.py
TX chain:
  info bits -> FEC (conv + puncture) -> interleave -> map -> OFDM mod -> power normalize
Optionally:
  clip + (simple OOB filter by projecting back to active subcarriers)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.phy.coding import encode_with_rate, ndbps_per_symbol
from src.phy.interleaver import interleave_symbol
from src.phy.mapping import bits_to_symbols, NBPSC
from src.phy.ofdm import ofdm_modulate_frame
from src.metrics.utils import plot_constellation, set_average_power
from src.phy import params


@dataclass
class TxConfig:
    modulation: str
    rate: str
    num_symbols: int
    os_factor: int = 1
    seed: int = 1

    # waveform options
    add_cp: bool = True
    target_avg_power: float = 1.0  # normalize TX average power to this value

    # clip+filter options
    clip_ratio: float = 2.0
    clip_iters: int = 0  # 0 => disable clip+filter


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


def _clip_amplitude(x: np.ndarray, clip_ratio: float) -> np.ndarray:
    """
    Hard clip magnitude to A = clip_ratio * RMS(x).
    Keeps phase; clips only amplitude.
    """
    x = np.asarray(x, dtype=np.complex128)
    rms = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)
    A = clip_ratio * rms
    r = np.abs(x)
    scale = np.ones_like(r)
    mask = r > A
    scale[mask] = A / (r[mask] + 1e-20)
    return x * scale


def _active_bins(N: int) -> np.ndarray:
    """
    Active subcarrier bins in FFT-bin indexing for length N.
    Keep active subcarriers + DC (0) for convenience (DC is usually 0 anyway).
    """
    # active subcarriers (e.g. -26..-1, +1..+26) should be in params.ACTIVE_SUBCARRIERS
    bins = [
        params.subcarrier_to_bin(int(k), N) for k in params.ACTIVE_SUBCARRIERS.tolist()
    ]
    # include DC bin index (subcarrier 0)
    bins.append(params.subcarrier_to_bin(0, N))
    return np.array(sorted(set(bins)), dtype=int)


def _project_to_active_subcarriers(
    wave: np.ndarray, num_symbols: int, os_factor: int, has_cp: bool
) -> np.ndarray:
    """
    "Filtering" step: FFT each symbol, zero all bins except active (in-band) bins,
    then IFFT back. This removes OOB spectral regrowth from clipping.
    """
    wave = np.asarray(wave, dtype=np.complex128).reshape(-1)
    N = params.N_FFT * os_factor
    cp = (params.N_CP * os_factor) if has_cp else 0
    sym_len = N + cp

    if wave.size < num_symbols * sym_len:
        raise ValueError("Waveform too short for requested number of symbols.")

    wave = wave[: num_symbols * sym_len]
    keep = _active_bins(N)
    keep_mask = np.zeros(N, dtype=bool)
    keep_mask[keep] = True

    out = []
    for n in range(num_symbols):
        seg = wave[n * sym_len : (n + 1) * sym_len]
        if has_cp:
            seg_no_cp = seg[cp:]
        else:
            seg_no_cp = seg

        X = np.fft.fft(seg_no_cp, norm="ortho")
        X[~keep_mask] = 0.0
        x_filt = np.fft.ifft(X, norm="ortho")

        if has_cp:
            x_filt = np.concatenate([x_filt[-cp:], x_filt])

        out.append(x_filt)

    return np.concatenate(out)


def clip_and_filter_frame(
    wave: np.ndarray,
    num_symbols: int,
    os_factor: int,
    clip_ratio: float,
    iters: int,
    has_cp: bool = True,
) -> np.ndarray:
    """
    Iterative clip + project-to-active-subcarriers "filter".
    """
    y = np.asarray(wave, dtype=np.complex128).reshape(-1)
    for _ in range(int(iters)):
        y = _clip_amplitude(y, clip_ratio=clip_ratio)
        y = _project_to_active_subcarriers(
            y, num_symbols=num_symbols, os_factor=os_factor, has_cp=has_cp
        )
    return y


def build_tx_waveforms(cfg: TxConfig, info_bits_in: np.ndarray | None = None) -> dict:
    """
    Returns a dict with:
      info_bits
      data_symbols  (num_symbols,48)
      tx_wave       baseline time waveform
      tx_cf         clip+filter waveform (or None if disabled)
    """
    info_bits, data_symbols = build_tx_chain(
        modulation=cfg.modulation,
        rate=cfg.rate,
        num_symbols=cfg.num_symbols,
        seed=cfg.seed,
        info_bits_in=info_bits_in,
    )
    plot_constellation(data_symbols.reshape(-1), title="TX constellation symbols")

    tx_wave = ofdm_modulate_frame(
        data_symbols_matrix=data_symbols,
        os_factor=cfg.os_factor,
        add_cyclic_prefix=cfg.add_cp,
    )

    # normalize baseline average power (this is usually what you want before PAPR/HPA comparisons)
    tx_wave = set_average_power(tx_wave, cfg.target_avg_power)

    tx_cf = None
    if cfg.clip_iters and cfg.clip_iters > 0:
        tx_cf = clip_and_filter_frame(
            tx_wave,
            num_symbols=cfg.num_symbols,
            os_factor=cfg.os_factor,
            clip_ratio=cfg.clip_ratio,
            iters=cfg.clip_iters,
            has_cp=cfg.add_cp,
        )
        # re-normalize after clip+filter so comparisons are fair at equal average power
        tx_cf = set_average_power(tx_cf, cfg.target_avg_power)

    return {
        "info_bits": info_bits,
        "data_symbols": data_symbols,
        "tx_wave": tx_wave,
        "tx_cf": tx_cf,
    }
