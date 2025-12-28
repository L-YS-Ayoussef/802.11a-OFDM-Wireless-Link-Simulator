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
        x_in = np.asarray(x)
        x = np.asarray(x, dtype=np.complex128)
        r = np.abs(x)
        denom = (1.0 + (r / self.A_sat)**(2*self.p))**(1.0/(2*self.p))
        denom = np.where(denom == 0, 1.0, denom)
        y = x / denom
        # If original input was real, return real output
        if np.isrealobj(x_in):
            return np.real(y).astype(np.float64)
        return y
