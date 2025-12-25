"""
coding.py
Convolutional encoder (K=7, rate 1/2 mother code) + puncturing for rates 2/3 and 3/4.

Mother code polynomials are the classic 802.11a: (133, 171) in octal.
We implement a simple bitwise encoder with tail bits (optional).

Puncturing is applied to the serialized stream: [A0, B0, A1, B1, ...].
Common puncture vectors:
- R=2/3: [1, 1, 1, 0] repeating  (drop every 4th bit)
- R=3/4: [1, 1, 0, 1, 1, 0] repeating (drop 3rd and 6th)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

def _octal_to_bits(oct_val: int, width: int) -> np.ndarray:
    """Convert octal integer like 0o133 to a binary tap vector (MSB..LSB)."""
    b = np.array([(oct_val >> i) & 1 for i in range(width-1, -1, -1)], dtype=np.uint8)
    return b

@dataclass(frozen=True)
class ConvCode:
    constraint_len: int = 7
    g0_octal: int = 0o133
    g1_octal: int = 0o171

    @property
    def g0(self) -> np.ndarray:
        return _octal_to_bits(self.g0_octal, self.constraint_len)

    @property
    def g1(self) -> np.ndarray:
        return _octal_to_bits(self.g1_octal, self.constraint_len)

def conv_encode_rate_half(bits: np.ndarray, code: ConvCode = ConvCode(), add_tail: bool = True) -> np.ndarray:
    """
    Convolutionally encode bits with rate 1/2, K=7.
    Output is serialized as [A0,B0,A1,B1,...] (dtype uint8 {0,1}).

    If add_tail=True, append (K-1) zeros to terminate to all-zero state.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if add_tail:
        bits = np.concatenate([bits, np.zeros(code.constraint_len - 1, dtype=np.uint8)])

    K = code.constraint_len
    g0, g1 = code.g0, code.g1
    state = np.zeros(K, dtype=np.uint8)  # shift register, state[0] is newest

    out = np.empty(bits.size * 2, dtype=np.uint8)
    oi = 0
    for b in bits:
        state[1:] = state[:-1]
        state[0] = b
        a = int(np.sum(state * g0) % 2)
        c = int(np.sum(state * g1) % 2)
        out[oi] = a
        out[oi + 1] = c
        oi += 2
    return out

_PUNCTURE = {
    "1/2": np.array([1, 1], dtype=np.uint8),                 # keep all
    "2/3": np.array([1, 1, 1, 0], dtype=np.uint8),           # keep 3 of 4
    "3/4": np.array([1, 1, 0, 1, 1, 0], dtype=np.uint8),     # keep 4 of 6
}

def puncture(encoded_bits: np.ndarray, rate: str) -> np.ndarray:
    """
    Apply puncturing to the serialized encoded stream.
    rate must be "1/2", "2/3", or "3/4".
    """
    if rate not in _PUNCTURE:
        raise ValueError(f"Unsupported rate '{rate}'. Choose from {list(_PUNCTURE.keys())}.")
    enc = np.asarray(encoded_bits, dtype=np.uint8).reshape(-1)
    pattern = _PUNCTURE[rate]
    mask = np.tile(pattern, int(np.ceil(enc.size / pattern.size)))[:enc.size].astype(bool)
    return enc[mask].astype(np.uint8)

def encode_with_rate(bits: np.ndarray, rate: str, add_tail: bool = True) -> np.ndarray:
    """
    Convenience wrapper: rate-1/2 conv encode then puncture to desired rate.
    """
    enc12 = conv_encode_rate_half(bits, add_tail=add_tail)
    return puncture(enc12, rate)

def ndbps_per_symbol(nbpsc: int, rate: str) -> int:
    """
    NDBPS = number of data (information) bits per OFDM symbol for a given modulation (NBPSC) and code rate.
    802.11a: NCBPS = 48*NBPSC. NDBPS = NCBPS * R.
    """
    ncbps = 48 * nbpsc
    num, den = rate.split("/")
    R = float(num) / float(den)
    return int(round(ncbps * R))
