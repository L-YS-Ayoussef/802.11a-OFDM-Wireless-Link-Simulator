"""
params.py
802.11a-like OFDM parameters used in the assignment.
Focus: data portion (no preamble/SIGNAL PHY framing), but includes pilots and pilot polarity sequence.
"""
from __future__ import annotations

import numpy as np

FS = 20e6

# ----------------------------
# OFDM core parameters (802.11a)
# ----------------------------
N_FFT: int = 64
N_CP: int = 16

# Active (occupied) subcarriers are -26..-1 and +1..+26 (DC=0 unused).
ACTIVE_SUBCARRIERS = np.array(
    list(range(-26, 0)) + list(range(1, 27)),
    dtype=int,
)

# Pilot subcarriers (subset of active)
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)

# Data subcarriers = active minus pilots
DATA_SUBCARRIERS = np.array(
    [k for k in ACTIVE_SUBCARRIERS.tolist() if k not in set(PILOT_SUBCARRIERS.tolist())],
    dtype=int,
)
assert len(DATA_SUBCARRIERS) == 48, "Expected 48 data subcarriers"

# Pilot base signs for (k = -21, -7, +7, +21)
# From IEEE 802.11a OFDM pilot sequence P_{-26..26}, the effective per-symbol pilots are p[n] * [1, 1, 1, -1].
PILOT_BASE = np.array([1, 1, 1, -1], dtype=float)

def _pilot_polarity_sequence_127() -> np.ndarray:
    """
    Generate the 127-length pilot polarity sequence p[n] using the same LFSR as the IEEE 802.11a scrambler
    (polynomial x^7 + x^4 + 1), initialized to all ones.

    Standard text: generate with the scrambler (all-ones initial state), then replace 1->-1 and 0->+1.
    That mapping is p[n] = 1 if out_bit==0 else -1.
    """
    reg = [1]*7
    seq = np.empty(127, dtype=int)
    for i in range(127):
        out_bit = reg[6] ^ reg[3]   # taps at x^7 and x^4
        reg = [out_bit] + reg[:-1]  # shift
        seq[i] = 1 if out_bit == 0 else -1
    return seq

# Pilot polarity sequence p0..p126 (Eq. 25 in IEEE 802.11a-1999)
PILOT_POLARITY_127 = _pilot_polarity_sequence_127()
assert len(PILOT_POLARITY_127) == 127

def pilot_polarity(symbol_index: int) -> int:
    """
    Returns pilot polarity p for OFDM symbol index.

    IEEE uses p[0] for SIGNAL and then p[1],p[2],... for DATA symbols.
    In a DATA-only simulation, using p[symbol_index] is fine (still cyclic with period 127).
    """
    return int(PILOT_POLARITY_127[symbol_index % 127])

def subcarrier_to_bin(k: int, n_fft: int) -> int:
    """
    Map logical subcarrier index k in [-n_fft/2, ..., n_fft/2-1] to numpy FFT bin [0..n_fft-1].
    Example: k=-1 -> bin=n_fft-1, k=0 -> bin=0, k=1 -> bin=1.
    """
    return k % n_fft

def bins_from_subcarriers(subcarriers: np.ndarray, n_fft: int) -> np.ndarray:
    return np.array([subcarrier_to_bin(int(k), n_fft) for k in subcarriers], dtype=int)

# Precomputed bins for speed (64-pt)
ACTIVE_BINS_64 = bins_from_subcarriers(ACTIVE_SUBCARRIERS, N_FFT)
DATA_BINS_64 = bins_from_subcarriers(DATA_SUBCARRIERS, N_FFT)
PILOT_BINS_64 = bins_from_subcarriers(PILOT_SUBCARRIERS, N_FFT)
