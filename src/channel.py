# src/channel.py
"""
channel.py
Time-invariant complex FIR channel (multipath) implemented as linear convolution.

Given in your handout (delay spread 0.15 us at Fs=20 MHz => Ts=0.05 us):
h = [0.8208+0.2052j, 0.4104+0.1026j, 0.2052+0.2052j, 0.1026+0.1026j]^T

For OFDM, you typically want channel length <= CP length (in samples) to avoid ISI
after CP removal (i.e., preserve circular convolution per symbol).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# Handout channel taps (length=4 -> delay spread = (4-1)*Ts = 0.15 us at Fs=20 MHz)
DEFAULT_H = np.array(
    [0.8208 + 0.2052j, 0.4104 + 0.1026j, 0.2052 + 0.2052j, 0.1026 + 0.1026j],
    dtype=np.complex128,
)


@dataclass
class FIRChannel:
    """
    Memory channel: y[n] = sum_k h[k] x[n-k]   (linear convolution, causal taps starting at k=0)

    normalize=True scales h so that sum |h|^2 = 1 (keeps average power comparable).
    """

    h: np.ndarray = field(default_factory=lambda: DEFAULT_H.copy())
    normalize: bool = True

    def __post_init__(self):
        h = np.asarray(self.h, dtype=np.complex128).reshape(-1)
        if h.size < 1:
            raise ValueError("Channel taps h must be non-empty.")
        if self.normalize:
            p = np.sum(np.abs(h) ** 2)
            if p > 0:
                h = h / np.sqrt(p)
        object.__setattr__(self, "h", h)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply channel via linear convolution.
        Output length is len(x) + len(h) - 1 (so it's never 'too short' for the receiver).
        """
        x = np.asarray(x, dtype=np.complex128).reshape(-1)
        return np.convolve(x, self.h, mode="full").astype(np.complex128)


def apply_fir_channel(
    x: np.ndarray, h: np.ndarray = DEFAULT_H, normalize: bool = True
) -> np.ndarray:
    """Convenience functional wrapper."""
    ch = FIRChannel(h=h, normalize=normalize)
    return ch(x)
