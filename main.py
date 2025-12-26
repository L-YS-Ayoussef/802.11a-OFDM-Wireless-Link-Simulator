"""
main.py
End-to-end OFDM TX/RX simulation:
- Source bits (random or MP3 bytes->bits)
- FEC encode + puncture (rate 1/2, 2/3, 3/4)
- Interleave per OFDM symbol (802.11a style)
- Constellation mapping (BPSK/QPSK/16QAM/64QAM)
- OFDM modulation (64-pt IFFT, CP, optional oversampling)
- (Optional) clip+filter PAPR reduction
- Average power normalization
- HPA (Rapp AM/AM) with specified IBO
- Receiver: OFDM demod + EVM/constellation
- (Optional) full bit recovery: demap -> deinterleave -> Viterbi decode -> BER
- Save plots and (if MP3 input) save recovered MP3

This file assumes you already created:
- transmitter.py (build_tx_waveforms)
- receiver.py (rx_constellation_and_evm, rx_recover_bits)
- rf.py (optional IQ mod/demod; not enabled by default here)
- hpa.py (RappHPA)
- utils.py (plot_* helpers, set_average_power, bits_to_bytes, etc.)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src import params
from src.coding import ndbps_per_symbol
from src.mapping import NBPSC
from src.hpa import RappHPA
from src.transmitter import TxConfig, build_tx_waveforms
from src.channel import FIRChannel, DEFAULT_H
from src.receiver import rx_constellation_and_evm, rx_recover_bits

from src.utils import (
    set_average_power,
    bits_to_bytes,
    plot_constellation,
    plot_spectrogram,
    plot_psd,
    plot_papr_ccdf,
)

# Optional RF chain (not enabled by default)
from src.rf import iq_modulate, iq_demodulate


def _ask_choice(prompt: str, options: list[str]) -> str:
    print(prompt)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        try:
            x = int(input("Enter number: ").strip())
            if 1 <= x <= len(options):
                return options[x - 1]
        except Exception:
            pass
        print("Invalid choice, try again.")


def _ask_int(prompt: str, default: int) -> int:
    s = input(f"{prompt} [default {default}]: ").strip()
    if s == "":
        return default
    return int(s)


def _ask_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} [default {default}]: ").strip()
    if s == "":
        return default
    return float(s)


def _ask_yesno(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    s = input(f"{prompt} (y/n) [default {d}]: ").strip().lower()
    if s == "":
        return default
    return s.startswith("y")


def _bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big")
    return bits.astype(np.uint8)

def _prepare_source_bits(
    modulation: str,
    rate: str,
    num_symbols: int,
    use_mp3: bool,
    mp3_path: str | None,
):
    """
    Returns:
      tx_bits: information bits padded to num_symbols_used*NDBPS
      raw_bytes: MP3 bytes (or None)
      orig_byte_len: original byte length (or 0)
      orig_bit_len: original bit length (or 0)
      num_symbols_used: possibly increased symbol count to fit the file
    """
    nbpsc = NBPSC[modulation]
    ndbps = ndbps_per_symbol(nbpsc, rate)

    if use_mp3:
        if mp3_path is None:
            raise ValueError("mp3_path required if use_mp3=True")

        raw = Path(mp3_path).read_bytes()
        orig_byte_len = len(raw)
        orig_bits = _bytes_to_bits(raw)
        orig_bit_len = orig_bits.size

        # compute how many symbols are needed to carry the whole file (no truncation)
        num_symbols_needed = int(np.ceil(orig_bit_len / ndbps))
        num_symbols_used = max(num_symbols, num_symbols_needed)

        n_info = num_symbols_used * ndbps
        tx_bits = np.concatenate(
            [orig_bits, np.zeros(n_info - orig_bit_len, dtype=np.uint8)]
        )

        return (
            tx_bits.astype(np.uint8),
            raw,
            orig_byte_len,
            orig_bit_len,
            num_symbols_used,
        )

    # random bits
    num_symbols_used = num_symbols
    n_info = num_symbols_used * ndbps
    tx_bits = np.random.randint(0, 2, size=n_info, dtype=np.uint8)
    return tx_bits, None, 0, 0, num_symbols_used

def main():
    # ---------------- User inputs ----------------
    modulation = _ask_choice("Select modulation:", ["BPSK", "QPSK", "16QAM", "64QAM"])
    rate = _ask_choice("Select coding rate:", ["1/2", "2/3", "3/4"])

    num_symbols = _ask_int("Number of OFDM data symbols (e.g., 50)", 200)
    os_factor = _ask_int("Oversampling factor for PAPR/PSD (1,2,4)", 4)

    clip_ratio = _ask_float("Clipping ratio (A/RMS)", 1.2)
    clip_iters = _ask_int("Clip+filter iterations", 2)

    ibo_db = _ask_float("HPA Input Backoff (dB)", 6.0)
    rapp_p = _ask_float("Rapp smoothness p", 2.0)

    use_mp3 = _ask_yesno("Use MP3 file as input bits?", default=False)
    mp3_path = None
    if use_mp3:
        mp3_path = input("Enter path to MP3 (e.g., test.mp3): ").strip()
        if mp3_path == "":
            mp3_path = "test.mp3"

    # Results directory
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{modulation}_{rate.replace('/','-')}_N{num_symbols}_OS{os_factor}"

    # Sampling rate for plotting (baseband). Prefer params.FS if you defined it.
    fs_base = getattr(params.FS, "FS", 20e6)
    fs = fs_base * os_factor

    # ---------------- Source bits ----------------
    tx_bits, src_bytes, orig_byte_len, orig_bit_len, num_symbols_used = _prepare_source_bits(
        modulation=modulation,
        rate=rate,
        num_symbols=num_symbols,
        use_mp3=use_mp3,
        mp3_path=mp3_path,
    )

    # IMPORTANT: use the updated symbol count downstream
    if num_symbols_used != num_symbols:
        print(f"Auto-updated num_symbols from {num_symbols} -> {num_symbols_used} to fit the MP3 bitstream.")
    num_symbols = num_symbols_used

    # ---------------- Transmitter ----------------
    cfg = TxConfig(
        modulation=modulation,
        rate=rate,
        num_symbols=num_symbols,
        os_factor=os_factor,
        seed=2,
        add_cp=True,
        target_avg_power=1.0,
        clip_ratio=clip_ratio,
        clip_iters=clip_iters,  
    )

    tx = build_tx_waveforms(cfg, info_bits_in=tx_bits)  
    tx_wave = tx["tx_wave"]
    tx_cf = tx["tx_cf"]
    data_syms = tx["data_symbols"]

    cf_enabled = (clip_iters > 0) and (tx_cf is not None)

    # Important normalization: define a consistent reference power level
    tx_wave = set_average_power(tx_wave, 1.0)
    if cf_enabled:
        tx_cf = set_average_power(tx_cf, 1.0)

    # Plots before HPA
    plot_spectrogram(tx_wave, fs=fs, title=f"Spectrogram (TX waveform) - {tag}")

    # Always plot baseline CCDF
    plot_papr_ccdf(
        wave_baseline=tx_wave,
        wave_cf=None,  # baseline only here
        os_factor=os_factor,
        has_cp=True,
        title=f"PAPR CCDF (baseline) - {modulation} {rate}",
    )

    # Plot clip+filter only if enabled and waveform exists
    if tx_cf is not None and clip_iters > 0:
        plot_papr_ccdf(
            wave_baseline=tx_wave,
            wave_cf=tx_cf,
            os_factor=os_factor,
            has_cp=True,
            title=f"PAPR CCDF (baseline vs clip+filter) CR={clip_ratio}, iters={clip_iters}",
            label_baseline="baseline",
            label_cf="clip+filter",
        )
    else:
        print("Clip+filter disabled -> skipping clip+filter CCDF plot.")

    # ---------------- HPA (Power Amplifier) ----------------
    A_sat = 1.0
    hpa = RappHPA(A_sat=A_sat, p=rapp_p)

    # Set input average power to meet IBO:
    # IBO(dB) = 10*log10( A_sat^2 / Pin_avg )
    target_pin = (A_sat**2) / (10 ** (ibo_db / 10))

    tx_wave_ibo = set_average_power(tx_wave, target_pin)
    y_base = hpa(tx_wave_ibo)

    y_cf = None
    if cf_enabled:
        tx_cf_ibo = set_average_power(tx_cf, target_pin)
        y_cf = hpa(tx_cf_ibo)

    # PSD / Spectrogram after HPA
    plot_psd(y_base, fs_hz=fs, title=f"PSD after HPA (baseline) IBO={ibo_db:.1f} dB")
    if cf_enabled:
        plot_psd(y_cf, fs_hz=fs, title=f"PSD after HPA (clip+filter) IBO={ibo_db:.1f} dB")

    plot_spectrogram(
        y_base, fs=fs, title=f"Spectrogram after HPA (baseline) - IBO={ibo_db:.1f} dB"
    )

    # ---------------- Time-invariant complex FIR filter wireless channel ----------------
    chan = FIRChannel(h=DEFAULT_H, normalize=True)
    r_base = chan(y_base)
    r_cf = chan(y_cf) if cf_enabled else None

    # ---------------- Receiver: EVM + constellation ----------------
    ref_syms = data_syms.reshape(-1)

    est_base_c, evm_base, data_hat_base = rx_constellation_and_evm(
        r_base,
        ref_data_symbols=ref_syms,
        num_symbols=num_symbols,
        os_factor=os_factor,
        do_iq_demod=False,  # set True only after you enable RF chain
    )
    est_cf_c = evm_cf = data_hat_cf = None
    if cf_enabled:
        est_cf_c, evm_cf, data_hat_cf = rx_constellation_and_evm(
            r_cf,
            ref_data_symbols=data_syms.reshape(-1),
            num_symbols=num_symbols,
            os_factor=os_factor,
            do_iq_demod=False,
        )

    plot_constellation(
        est_base_c, title=f"RX constellation after HPA (baseline) - IBO={ibo_db:.1f} dB"
    )
    if cf_enabled:
        plot_constellation(est_cf_c, title=f"RX constellation after HPA (clip+filter) - IBO={ibo_db:.1f} dB")
    else:
        print("Clip+filter disabled -> skipping CF RX constellation/EVM/BER.")

    # ---------------- Full bit recovery + BER ----------------
    rx_bits_base = rx_recover_bits(data_hat_base, modulation=modulation, rate=rate)
    L = min(tx_bits.size, rx_bits_base.size)
    ber_base = (
        float(np.mean(rx_bits_base[:L] != tx_bits[:L])) if L > 0 else float("nan")
    )

    ber_cf = None
    if cf_enabled:
        rx_bits_cf = rx_recover_bits(data_hat_cf, modulation=modulation, rate=rate)
        L2 = min(tx_bits.size, rx_bits_cf.size)
        ber_cf = float(np.mean(rx_bits_cf[:L2] != tx_bits[:L2])) if L2 > 0 else float("nan")

    # ---------------- Print summary ----------------
    print("\n--- Results ---")
    print(
        f"Modulation: {modulation}, Rate: {rate}, Symbols: {num_symbols}, OS: {os_factor}x"
    )
    print(f"EVM after HPA (baseline, CPE-corrected): {evm_base * 100:.2f}%")
    print(f"BER after HPA (baseline): {ber_base:.6e}")
    if cf_enabled:
        print(f"EVM after HPA (clip+filter): {evm_cf*100:.2f}%")
        print(f"BER after HPA (clip+filter): {ber_cf:.6e}")

    if use_mp3: 
        rb = rx_bits_base[:orig_bit_len]
        rec_base = bits_to_bytes(rb)[:orig_byte_len]
        out_base = results_dir / f"{tag}_recovered_baseline.mp3"
        out_base.write_bytes(rec_base)

        print(f"Saved recovered MP3 (baseline): {out_base}")
        if cf_enabled:
            rc = rx_bits_cf[:orig_bit_len]
            rec_cf = bits_to_bytes(rc)[:orig_byte_len]
            out_cf = results_dir / f"{tag}_recovered_clipfilter.mp3"
            out_cf.write_bytes(rec_cf)
            print(f"Saved recovered MP3 (clip+filter): {out_cf}")

    # ---------------- Save all figures ----------------
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.savefig(
            results_dir / f"{tag}_fig{fig_num}.png", dpi=200, bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":
    main()
