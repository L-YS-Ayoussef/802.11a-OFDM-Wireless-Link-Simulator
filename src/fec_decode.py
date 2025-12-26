# fec_decode.py (NEW)
from __future__ import annotations
import numpy as np
from src.coding import ConvCode, _PUNCTURE


def depuncture_to_rate_half(punctured: np.ndarray, rate: str) -> np.ndarray:
    """
    Expand punctured stream back to mother rate-1/2 stream length with erasures = 2.
    Output values: 0/1 for known bits, 2 for erased.
    """
    p = np.asarray(punctured, dtype=np.uint8).reshape(-1)
    pat = _PUNCTURE[rate]
    out = []
    pi = 0
    while pi < p.size:
        for keep in pat:
            if keep == 1:
                if pi < p.size:
                    out.append(int(p[pi]))
                    pi += 1
                else:
                    out.append(2)
            else:
                out.append(2)
    return np.array(out, dtype=np.uint8)


def viterbi_decode_hard(
    rate_half_stream: np.ndarray, code: ConvCode = ConvCode()
) -> np.ndarray:
    """
    Hard-decision Viterbi for K=7, rate-1/2 mother code.
    Input is serialized [A0,B0,A1,B1,...] with erasures=2 allowed.
    Returns decoded bits (includes tail bits if they were used).
    """
    y = np.asarray(rate_half_stream, dtype=np.uint8).reshape(-1)
    if y.size % 2 != 0:
        y = np.concatenate([y, np.array([2], dtype=np.uint8)])

    K = code.constraint_len
    n_states = 2 ** (K - 1)
    g0 = code.g0.astype(np.uint8)
    g1 = code.g1.astype(np.uint8)

    # Precompute trellis: for each state and input bit -> (next_state, outA, outB)
    next_state = np.zeros((n_states, 2), dtype=np.int32)
    outA = np.zeros((n_states, 2), dtype=np.uint8)
    outB = np.zeros((n_states, 2), dtype=np.uint8)

    for s in range(n_states):
        reg = np.array(
            [(s >> i) & 1 for i in range(K - 2, -1, -1)], dtype=np.uint8
        )  # length K-1
        for u in (0, 1):
            state = np.concatenate([[u], reg])  # length K, newest first
            a = int(np.sum(state * g0) % 2)
            b = int(np.sum(state * g1) % 2)
            s_next = (u << (K - 2)) | (s >> 1)
            next_state[s, u] = s_next
            outA[s, u] = a
            outB[s, u] = b

    # Viterbi
    n_steps = y.size // 2
    INF = 10**9
    metric = np.full(n_states, INF, dtype=np.int32)
    metric[0] = 0
    prev_state = np.zeros((n_steps, n_states), dtype=np.int32)
    prev_bit = np.zeros((n_steps, n_states), dtype=np.uint8)

    for t in range(n_steps):
        ya = y[2 * t]
        yb = y[2 * t + 1]
        new_metric = np.full(n_states, INF, dtype=np.int32)

        for s in range(n_states):
            if metric[s] >= INF:
                continue
            for u in (0, 1):
                sn = next_state[s, u]
                # Branch metric: Hamming distance, ignore erasures (value==2)
                bm = 0
                if ya != 2 and ya != outA[s, u]:
                    bm += 1
                if yb != 2 and yb != outB[s, u]:
                    bm += 1
                m = metric[s] + bm
                if m < new_metric[sn]:
                    new_metric[sn] = m
                    prev_state[t, sn] = s
                    prev_bit[t, sn] = u
        metric = new_metric

    # Traceback from best final state
    s = int(np.argmin(metric))
    uhat = np.zeros(n_steps, dtype=np.uint8)
    for t in range(n_steps - 1, -1, -1):
        uhat[t] = prev_bit[t, s]
        s = prev_state[t, s]
    return uhat


def fec_decode(
    punctured_bits: np.ndarray, rate: str, add_tail: bool = True
) -> np.ndarray:
    """
    Full decode: depuncture -> Viterbi -> remove tail if add_tail.
    """
    r12 = depuncture_to_rate_half(punctured_bits, rate=rate)
    decoded = viterbi_decode_hard(r12)
    if add_tail:
        # remove K-1 tail bits that were appended in encoder
        K = ConvCode().constraint_len
        if decoded.size >= (K - 1):
            decoded = decoded[: -(K - 1)]
    return decoded.astype(np.uint8)
