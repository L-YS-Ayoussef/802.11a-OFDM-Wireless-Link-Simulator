"""
equalizer.py
Pilot-aided 1-tap per-subcarrier equalization for OFDM (802.11a-style pilots).

We estimate the channel on the 4 pilot subcarriers:
    H_hat(k_p) = Y(k_p) / X(k_p)
Then interpolate H_hat onto data subcarriers, and equalize each subcarrier:

ZF:
    W_zf(k) = 1 / H_hat(k)
    X_hat(k) = W_zf(k) * Y(k)

MMSE (1-tap, assumes E[|X|^2]=1):
    W_mmse(k) = H_hat*(k) / (|H_hat(k)|^2 + sigma2)
    X_hat(k) = W_mmse(k) * Y(k)

sigma2 is complex noise power E[|N|^2] per subcarrier (same as per time sample for unitary FFT).
"""

from __future__ import annotations
import numpy as np
from src.phy import params


def _tx_pilots_for_symbol(symbol_index: int) -> np.ndarray:
    """
    Transmitted pilots for this OFDM symbol:
        pilots = p[n] * PILOT_BASE, where p[n] is +/-1 polarity sequence
    """
    p = params.pilot_polarity(symbol_index)  # +/-1
    return (p * params.PILOT_BASE).astype(np.complex128)  # shape (4,)


def estimate_channel_from_pilots(pilot_hat: np.ndarray) -> np.ndarray:
    """
    Estimate channel on pilot subcarriers for each OFDM symbol.

    pilot_hat: shape (num_symbols, 4) extracted from FFT bins at pilot subcarriers
    return: H_pilots shape (num_symbols, 4)
    """
    pilot_hat = np.asarray(pilot_hat, dtype=np.complex128)
    if pilot_hat.ndim != 2 or pilot_hat.shape[1] != len(params.PILOT_SUBCARRIERS):
        raise ValueError("pilot_hat must have shape (num_symbols, 4).")

    num_symbols = pilot_hat.shape[0]
    H_p = np.zeros_like(pilot_hat)

    for n in range(num_symbols):
        Xp = _tx_pilots_for_symbol(n)  # known transmitted pilot values
        # avoid divide-by-zero if Xp ever contains 0 (it shouldn't)
        denom = np.where(np.abs(Xp) == 0, 1.0 + 0j, Xp)
        H_p[n] = pilot_hat[n] / denom

    return H_p


def interpolate_channel_to_data(H_pilots: np.ndarray) -> np.ndarray:
    """
    Interpolate channel estimates from pilot subcarriers to data subcarriers.

    H_pilots: (num_symbols, 4) channel on pilot subcarriers (k = [-21,-7,+7,+21])
    returns: H_data (num_symbols, 48) channel on DATA_SUBCARRIERS
    """
    H_pilots = np.asarray(H_pilots, dtype=np.complex128)
    if H_pilots.ndim != 2 or H_pilots.shape[1] != 4:
        raise ValueError("H_pilots must have shape (num_symbols, 4).")

    k_p = np.array(params.PILOT_SUBCARRIERS, dtype=float)  # [-21,-7,7,21]
    k_d = np.array(params.DATA_SUBCARRIERS, dtype=float)  # 48 values

    # Ensure pilot order is increasing for np.interp
    order = np.argsort(k_p)
    k_p_sorted = k_p[order]

    num_symbols = H_pilots.shape[0]
    H_data = np.zeros((num_symbols, len(params.DATA_SUBCARRIERS)), dtype=np.complex128)

    for n in range(num_symbols):
        Hp = H_pilots[n][order]
        # Interpolate real and imag parts separately (simple + stable)
        Hr = np.interp(k_d, k_p_sorted, np.real(Hp))
        Hi = np.interp(k_d, k_p_sorted, np.imag(Hp))
        H_data[n] = Hr + 1j * Hi

    return H_data


def equalize_data(
    data_hat: np.ndarray,
    H_data: np.ndarray,
    method: str,
    noise_var: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply 1-tap equalization on data subcarriers.

    data_hat: (num_symbols, 48) received data subcarriers Y(k)
    H_data:   (num_symbols, 48) channel estimate H_hat(k)
    method:   "zf" or "mmse"
    noise_var: sigma^2 = E[|N|^2] (required for MMSE)
    """
    Y = np.asarray(data_hat, dtype=np.complex128)
    H = np.asarray(H_data, dtype=np.complex128)

    if Y.shape != H.shape:
        raise ValueError(f"Shape mismatch: data_hat {Y.shape} vs H_data {H.shape}")

    if method.lower() == "zf":
        denom = np.where(np.abs(H) < eps, eps + 0j, H)
        return Y / denom

    if method.lower() == "mmse":
        if noise_var is None:
            raise ValueError("noise_var is required for MMSE.")
        # W = H* / (|H|^2 + sigma^2)
        denom = (np.abs(H) ** 2) + float(noise_var)
        denom = np.where(denom < eps, eps, denom)
        W = np.conj(H) / denom
        return W * Y

    raise ValueError("method must be 'zf' or 'mmse'.")


def equalize_frame_from_pilots(
    data_hat: np.ndarray,
    pilot_hat: np.ndarray,
    method: str,
    noise_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper:
      pilot_hat -> H_pilots -> interpolate -> H_data -> equalize

    returns:
      data_eq  (num_symbols,48)
      H_data   (num_symbols,48)
      H_pilots (num_symbols,4)
    """
    H_p = estimate_channel_from_pilots(pilot_hat)
    H_d = interpolate_channel_to_data(H_p)
    data_eq = equalize_data(data_hat, H_d, method=method, noise_var=noise_var)
    return data_eq, H_d, H_p
