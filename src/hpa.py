"""
hpa.py
HPA models + metrics helpers (EVM, PSD).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class RappHPA:
    """
    Rapp AM/AM model (memoryless), no AM/PM by default.

    y = x / (1 + (|x|/A_sat)^(2p))^(1/(2p))
    """
    A_sat: float = 1.0
    p: float = 2.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128)
        r = np.abs(x)
        denom = (1.0 + (r / self.A_sat)**(2*self.p))**(1.0/(2*self.p))
        # avoid division by 0
        denom = np.where(denom == 0, 1.0, denom)
        return x / denom

def set_average_power(x: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    p = np.mean(np.abs(x)**2)
    if p == 0:
        return x.copy()
    return x * np.sqrt(target_power / p)

def evm_rms(ref: np.ndarray, est: np.ndarray) -> float:
    """
    RMS EVM: sqrt(E|e|^2 / E|ref|^2)
    """
    ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
    est = np.asarray(est, dtype=np.complex128).reshape(-1)
    e = est - ref
    num = np.mean(np.abs(e)**2)
    den = np.mean(np.abs(ref)**2)
    if den == 0:
        return float("nan")
    return float(np.sqrt(num / den))

def cpe_correct(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """
    Optional: correct a single common phase error by rotating est to best match ref.
    """
    ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
    est = np.asarray(est, dtype=np.complex128).reshape(-1)
    # minimize ||ref - est*exp(jphi)|| => phi = angle(sum(ref*conj(est)))
    phi = np.angle(np.sum(ref * np.conj(est)))
    return est * np.exp(1j*phi)
