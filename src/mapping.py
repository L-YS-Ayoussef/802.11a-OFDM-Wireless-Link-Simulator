"""
mapping.py
Gray mapper for BPSK/QPSK/16QAM/64QAM + normalization factors (corrected).

Normalization (K_mod) so average constellation power is 1:
BPSK  : 1
QPSK  : 1/sqrt(2)
16QAM : 1/sqrt(10)
64QAM : 1/sqrt(42)
"""
from __future__ import annotations
import numpy as np

KMOD = {
    "BPSK": 1.0,
    "QPSK": 1/np.sqrt(2),
    "16QAM": 1/np.sqrt(10),
    "64QAM": 1/np.sqrt(42),
}

NBPSC = {
    "BPSK": 1,
    "QPSK": 2,
    "16QAM": 4,
    "64QAM": 6,
}

def _gray_to_binary(g: int) -> int:
    """Convert Gray-coded integer to binary integer."""
    b = 0
    while g:
        b ^= g
        g >>= 1
    return b

def _pam_level_from_gray_bits(gray_bits: np.ndarray) -> int:
    """
    For m bits (Gray), map to PAM levels:
    m=1 => {-1,+1}
    m=2 => {-3,-1,+1,+3}
    m=3 => {-7,-5,-3,-1,+1,+3,+5,+7}
    """
    m = gray_bits.size
    g = 0
    for bit in gray_bits.astype(int).tolist():
        g = (g << 1) | bit
    b = _gray_to_binary(g)
    M = 2**m
    level = 2*b - (M - 1)  # odd integer levels
    return int(level)

def bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """
    Map a 1D bit array into complex symbols for the chosen modulation.
    Output average power is 1 due to KMOD scaling.
    """
    if modulation not in NBPSC:
        raise ValueError(f"Unknown modulation '{modulation}'.")
    nbpsc = NBPSC[modulation]
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % nbpsc != 0:
        raise ValueError(f"Number of bits ({bits.size}) not multiple of NBPSC ({nbpsc}).")

    if modulation == "BPSK":
        b0 = bits.astype(int)
        I = 2*b0 - 1  # 0->-1, 1->+1
        Q = np.zeros_like(I)
        sym = (I + 1j*Q) * KMOD[modulation]
        return sym.astype(np.complex128)

    if modulation == "QPSK":
        pairs = bits.reshape(-1, 2)
        I = 2*pairs[:, 0].astype(int) - 1
        Q = 2*pairs[:, 1].astype(int) - 1
        sym = (I + 1j*Q) * KMOD[modulation]
        return sym.astype(np.complex128)

    # Square QAM: split bits into I and Q Gray-PAM of m bits each
    m = nbpsc // 2
    groups = bits.reshape(-1, nbpsc)
    I_levels = np.array([_pam_level_from_gray_bits(g[:m]) for g in groups], dtype=float)
    Q_levels = np.array([_pam_level_from_gray_bits(g[m:]) for g in groups], dtype=float)
    sym = (I_levels + 1j*Q_levels) * KMOD[modulation]
    return sym.astype(np.complex128)

def slicer(symbols: np.ndarray, modulation: str) -> np.ndarray:
    """
    Hard-decision slicer returning nearest constellation points (scaled).
    Useful for EVM computations with decision-directed reference if desired.
    """
    syms = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if modulation == "BPSK":
        I = np.sign(np.real(syms))
        I[I == 0] = 1
        return (I + 0j) * KMOD["BPSK"]

    if modulation == "QPSK":
        I = np.sign(np.real(syms)); I[I == 0] = 1
        Q = np.sign(np.imag(syms)); Q[Q == 0] = 1
        return (I + 1j*Q) * KMOD["QPSK"]

    # QAM
    nbpsc = NBPSC[modulation]
    m = nbpsc // 2
    M = 2**m
    # Levels are odd integers from -(M-1) to +(M-1)
    levels = np.arange(-(M-1), M, 2)
    # scale levels by KMOD
    levels_scaled = levels * KMOD[modulation]

    I = np.real(syms)
    Q = np.imag(syms)
    I_hat = levels_scaled[np.argmin(np.abs(I[:, None] - levels_scaled[None, :]), axis=1)]
    Q_hat = levels_scaled[np.argmin(np.abs(Q[:, None] - levels_scaled[None, :]), axis=1)]
    return (I_hat + 1j*Q_hat).astype(np.complex128)
