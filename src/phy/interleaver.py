"""
interleaver.py
IEEE 802.11a-style bit interleaver and deinterleaver for one OFDM symbol.

Interleaver is a 2-step permutation:
i = (NCBPS/16) * (k mod 16) + floor(k/16)
j = s*floor(i/s) + (i + NCBPS - floor(16*i/NCBPS)) mod s
s = max(NBPSC/2, 1)
"""
from __future__ import annotations

import numpy as np

def interleave_symbol(bits: np.ndarray, nbpsc: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    ncbps = 48 * nbpsc
    if bits.size != ncbps:
        raise ValueError(f"Expected {ncbps} bits for one symbol, got {bits.size}.")
    k = np.arange(ncbps, dtype=int)
    i = (ncbps // 16) * (k % 16) + (k // 16)
    s = max(nbpsc // 2, 1)
    j = s * (i // s) + (i + ncbps - (16 * i) // ncbps) % s
    out = np.empty_like(bits)
    out[j] = bits[k]
    return out

def deinterleave_symbol(bits: np.ndarray, nbpsc: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    ncbps = 48 * nbpsc
    if bits.size != ncbps:
        raise ValueError(f"Expected {ncbps} bits for one symbol, got {bits.size}.")
    k = np.arange(ncbps, dtype=int)
    # Inverse permutations from IEEE 802.11a (Eq. 18-19):
    s = max(nbpsc // 2, 1)
    i = s * (k // s) + (k + (16 * k) // ncbps) % s
    j = 16 * i - (ncbps - 1) * ((16 * i) // ncbps)
    out = np.empty_like(bits)
    out[j] = bits[k]
    return out
