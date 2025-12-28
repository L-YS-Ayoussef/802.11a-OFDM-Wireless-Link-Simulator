# receiver.py
from __future__ import annotations
import numpy as np
from src.phy.ofdm import ofdm_demodulate_frame
from src.rf.rf import iq_demodulate
from src.metrics.utils import complex_gain_correct, evm_rms, add_awgn
from src.phy.mapping import symbols_to_bits_hard
from src.phy.interleaver import deinterleave_symbol
from src.phy.fec_decode import fec_decode
from src.rf.equalizer import equalize_frame_from_pilots


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


def rx_one_snr_equalized(
    rx_clean: np.ndarray,
    ref_data_symbols: np.ndarray,  # flattened reference data constellation points
    tx_bits: np.ndarray,
    modulation: str,
    rate: str,
    num_symbols: int,
    os_factor: int,
    snr_db: float,
    eq_method: str,  # "zf" or "mmse"
    rng: np.random.Generator,
):
    """
    rx_clean -> AWGN -> OFDM demod -> equalize (ZF/MMSE) -> gain correct -> EVM
            -> demap/deint/FEC decode -> BER

    Returns:
      ber: float
      rx_bits: np.ndarray (decoded info bits)
      est_c: np.ndarray complex constellation points after EQ + gain correction (flattened)
      evm: float (0..1)
    """
    # Add noise
    rx_noisy, noise_var = add_awgn(rx_clean, snr_db=float(snr_db), rng=rng)

    # OFDM demod
    _, data_hat, pilot_hat = ofdm_demodulate_frame(
        rx_noisy, num_symbols=num_symbols, os_factor=os_factor, has_cp=True
    )

    # Equalize
    nv = noise_var if eq_method.lower() == "mmse" else None
    data_eq, _, _ = equalize_frame_from_pilots(
        data_hat=data_hat, pilot_hat=pilot_hat, method=eq_method, noise_var=nv
    )

    # Constellation alignment + EVM (after equalizer)
    ref = np.asarray(ref_data_symbols, dtype=np.complex128).reshape(-1)
    est = data_eq.reshape(-1)
    est_c = complex_gain_correct(ref, est)
    evm = evm_rms(ref, est_c)

    # Recover bits
    rx_bits = rx_recover_bits(data_eq, modulation=modulation, rate=rate)

    # BER
    tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    L = min(tx_bits.size, rx_bits.size)
    ber = float(np.mean(rx_bits[:L] != tx_bits[:L])) if L > 0 else float("nan")

    return ber, rx_bits, est_c, evm
