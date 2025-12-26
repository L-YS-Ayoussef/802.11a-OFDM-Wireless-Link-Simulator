# receiver.py (NEW)
from __future__ import annotations
import numpy as np
from src.ofdm import ofdm_demodulate_frame
from src.rf import iq_demodulate
from src.utils import complex_gain_correct, evm_rms  
from src.mapping import symbols_to_bits_hard
from src.interleaver import deinterleave_symbol
from src.fec_decode import fec_decode

def rx_constellation_and_evm(
    rx_in: np.ndarray,
    ref_data_symbols: np.ndarray,
    num_symbols: int,
    os_factor: int,
    do_iq_demod: bool = False,
    fs: float | None = None,
    fc: float | None = None,
    lpf_cutoff_hz: float | None = None,
):
    """
    Returns:
      est_c: complex equalized constellation points (flattened)
      evm: RMS EVM (0..1)
      data_hat: (num_symbols,48) raw demod data subcarriers
    """
    y = rx_in
    if do_iq_demod:
        if fs is None or fc is None or lpf_cutoff_hz is None:
            raise ValueError("fs, fc, lpf_cutoff_hz required when do_iq_demod=True")
        y = iq_demodulate(y, fs=fs, fc=fc, lpf_cutoff_hz=lpf_cutoff_hz)

    _, data_hat, _ = ofdm_demodulate_frame(
        y, num_symbols=num_symbols, os_factor=os_factor, has_cp=True
    )

    ref = np.asarray(ref_data_symbols, dtype=np.complex128).reshape(-1)
    est = data_hat.reshape(-1)

    est_c = complex_gain_correct(ref, est)  # your existing correction
    evm = evm_rms(ref, est_c)
    return est_c, evm, data_hat

def rx_recover_bits(
    data_hat: np.ndarray,  # (num_symbols,48) complex
    modulation: str,
    rate: str,
) -> np.ndarray:
    """
    data_hat -> demap -> deinterleave per symbol -> FEC decode -> recovered info bits
    """
    num_symbols = data_hat.shape[0]
    nbpsc = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}[modulation]
    ncbps = 48 * nbpsc

    # Demap per symbol -> interleaved coded bits
    coded_interleaved = []
    for n in range(num_symbols):
        bits_n = symbols_to_bits_hard(data_hat[n], modulation)  # 48*nbpsc
        coded_interleaved.append(bits_n)
    coded_interleaved = np.concatenate(coded_interleaved).astype(np.uint8)

    # Deinterleave per symbol
    coded = np.zeros_like(coded_interleaved)
    for n in range(num_symbols):
        blk = coded_interleaved[n * ncbps : (n + 1) * ncbps]
        coded[n * ncbps : (n + 1) * ncbps] = deinterleave_symbol(blk, nbpsc=nbpsc)

    # FEC decode (Viterbi) to recover info bits
    info = fec_decode(coded, rate=rate, add_tail=True)
    return info
