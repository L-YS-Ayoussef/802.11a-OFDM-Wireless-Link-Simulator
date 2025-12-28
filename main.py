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
- receiver.py (rx_one_snr_equalized)
- rf.py (optional IQ mod/demod; not enabled by default here)
- hpa.py (RappHPA)
- utils.py (plot_* helpers, set_average_power, bits_to_bytes, etc.)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.phy import params
from src.phy.coding import ndbps_per_symbol
from src.phy.mapping import NBPSC
from src.rf.hpa import RappHPA
from src.chains.transmitter import TxConfig, build_tx_waveforms
from src.rf.channel import FIRChannel, DEFAULT_H
from src.chains.receiver import rx_one_snr_equalized

from src.metrics.utils import (
    set_average_power,
    bits_to_bytes,
    plot_constellation,
    plot_spectrogram,
    plot_psd,
    plot_papr_ccdf,
    add_awgn,
)

# RF chain (not enabled by default)
from src.rf.rf import iq_modulate, iq_demodulate


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

    clip_iters = _ask_int("Clip+filter iterations", 2)
    if clip_iters > 0:
        clip_ratio = _ask_float("Clipping ratio (A/RMS)", 1.2)
    else:
        clip_ratio = None

    ibo_db = _ask_float("HPA Input Backoff (dB)", 6.0)
    rapp_p = _ask_float("Rapp smoothness p", 2.0)

    use_mp3 = _ask_yesno("Use MP3 file as input bits?", default=False)
    mp3_path = None
    if use_mp3:
        mp3_path = input("Enter path to MP3 (e.g., test.mp3): ").strip()
        if mp3_path == "":
            mp3_path = "test.mp3"

    # Results directory
    results_dir = Path("tests/BPSK_tc2")
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{modulation}_{rate.replace('/','-')}_N{num_symbols}_OS{os_factor}"

    # Sampling rate for plotting (baseband). Prefer params.FS if you defined it.
    fs_base = getattr(params.FS, "FS", 20e6)
    fs = fs_base * os_factor

    # ---------------- Source bits ----------------
    tx_bits, src_bytes, orig_byte_len, orig_bit_len, num_symbols_used = (
        _prepare_source_bits(
            modulation=modulation,
            rate=rate,
            num_symbols=num_symbols,
            use_mp3=use_mp3,
            mp3_path=mp3_path,
        )
    )

    # IMPORTANT: use the updated symbol count downstream
    if num_symbols_used != num_symbols:
        print(
            f"Auto-updated num_symbols from {num_symbols} -> {num_symbols_used} to fit the MP3 bitstream."
        )
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
        plot_psd(
            y_cf, fs_hz=fs, title=f"PSD after HPA (clip+filter) IBO={ibo_db:.1f} dB"
        )

    if modulation in ("16QAM", "64QAM"):
        pass
    else:
        # ---------------- Time-invariant complex FIR filter wireless channel ----------------
        chan = FIRChannel(h=DEFAULT_H, normalize=True)
        r_base = chan(y_base)
        r_cf = chan(y_cf) if cf_enabled else None

        # ---------------- SNR sweep: BER for ZF & MMSE ----------------
        snr_dbs = np.linspace(0, 10, 5)
        ber_base_zf = np.zeros_like(snr_dbs, dtype=float)
        ber_base_mmse = np.zeros_like(snr_dbs, dtype=float)
        ber_cf_zf = np.zeros_like(snr_dbs, dtype=float) if cf_enabled else None
        ber_cf_mmse = np.zeros_like(snr_dbs, dtype=float) if cf_enabled else None

        rng = np.random.default_rng(123)

        ref_syms = data_syms.reshape(-1)  # reference constellation points

        for idx, snr_db in enumerate(snr_dbs):
            print(idx)
            is_last = idx == len(snr_dbs) - 1

            # ----- Baseline -----
            b_zf, rx_bits_base_zf, est_base_zf_c, evm_base_zf = rx_one_snr_equalized(
                rx_clean=r_base,
                ref_data_symbols=ref_syms,
                tx_bits=tx_bits,
                modulation=modulation,
                rate=rate,
                num_symbols=num_symbols,
                os_factor=os_factor,
                snr_db=snr_db,
                eq_method="zf",
                rng=rng,
            )
            b_mm, rx_bits_base_mm, est_base_mm_c, evm_base_mm = rx_one_snr_equalized(
                rx_clean=r_base,
                ref_data_symbols=ref_syms,
                tx_bits=tx_bits,
                modulation=modulation,
                rate=rate,
                num_symbols=num_symbols,
                os_factor=os_factor,
                snr_db=snr_db,
                eq_method="mmse",
                rng=rng,
            )

            ber_base_zf[idx] = b_zf
            ber_base_mmse[idx] = b_mm

            # ----- Clip+filter -----
            if cf_enabled:
                c_zf, rx_bits_cf_zf, est_cf_zf_c, evm_cf_zf = rx_one_snr_equalized(
                    rx_clean=r_cf,
                    ref_data_symbols=ref_syms,
                    tx_bits=tx_bits,
                    modulation=modulation,
                    rate=rate,
                    num_symbols=num_symbols,
                    os_factor=os_factor,
                    snr_db=snr_db,
                    eq_method="zf",
                    rng=rng,
                )
                c_mm, rx_bits_cf_mm, est_cf_mm_c, evm_cf_mm = rx_one_snr_equalized(
                    rx_clean=r_cf,
                    ref_data_symbols=ref_syms,
                    tx_bits=tx_bits,
                    modulation=modulation,
                    rate=rate,
                    num_symbols=num_symbols,
                    os_factor=os_factor,
                    snr_db=snr_db,
                    eq_method="mmse",
                    rng=rng,
                )

                ber_cf_zf[idx] = c_zf
                ber_cf_mmse[idx] = c_mm

            # ----- Only at final SNR: plots + audio + summary -----
            if is_last:
                plot_constellation(
                    est_base_zf_c,
                    title=f"EQ(zf) constellation (baseline) @ SNR={snr_db:.1f} dB",
                )
                plot_constellation(
                    est_base_mm_c,
                    title=f"EQ(MMSE) constellation (baseline) @ SNR={snr_db:.1f} dB",
                )
                if cf_enabled:
                    plot_constellation(
                        est_cf_zf_c,
                        title=f"EQ(zf) constellation (clip+filter) @ SNR={snr_db:.1f} dB",
                    )
                    plot_constellation(
                        est_cf_mm_c,
                        title=f"EQ(MMSE) constellation (clip+filter) @ SNR={snr_db:.1f} dB",
                    )
                else:
                    print(
                        "Clip+filter disabled -> skipping CF constellation/audio/summary."
                    )

                print("\n--- Results (final SNR only) ---")
                print(
                    f"Modulation: {modulation}, Rate: {rate}, Symbols: {num_symbols}, OS: {os_factor}x"
                )
                print(f"Final SNR: {snr_db:.1f} dB")

                print(f"Baseline:  BER(ZF)={b_zf:.6e},  BER(MMSE)={b_mm:.6e}")
                print(
                    f"Baseline:  EVM(ZF)={evm_base_zf*100:.2f}%,  EVM(MMSE)={evm_base_mm*100:.2f}%"
                )

                if cf_enabled:
                    print(f"ClipFilt:  BER(ZF)={c_zf:.6e},  BER(MMSE)={c_mm:.6e}")
                    print(
                        f"ClipFilt:  EVM(ZF)={evm_cf_zf*100:.2f}%,  EVM(MMSE)={evm_cf_mm*100:.2f}%"
                    )

                # Save recovered MP3 only for final SNR (use MMSE bits)
                if use_mp3:
                    # baseline MMSE
                    rb = rx_bits_base_mm[:orig_bit_len]
                    rec_base = bits_to_bytes(rb)[:orig_byte_len]
                    out_base = (
                        results_dir / f"{tag}_recovered_baseline_SNR{snr_db:.1f}dB_mmse.mp3"
                    )
                    out_base.write_bytes(rec_base)
                    print(f"Saved recovered MP3 (baseline, MMSE): {out_base}")

                    if cf_enabled:
                        rc = rx_bits_cf_mm[:orig_bit_len]
                        rec_cf = bits_to_bytes(rc)[:orig_byte_len]
                        out_cf = (
                            results_dir
                            / f"{tag}_recovered_clipfilter_SNR{snr_db:.1f}dB_mmse.mp3"
                        )
                        out_cf.write_bytes(rec_cf)
                        print(f"Saved recovered MP3 (clip+filter, MMSE): {out_cf}")

        plt.figure()
        plt.semilogy(snr_dbs, ber_base_zf, marker="o", label="Baseline ZF")
        plt.semilogy(snr_dbs, ber_base_mmse, marker="o", label="Baseline MMSE")

        if cf_enabled:
            plt.semilogy(snr_dbs, ber_cf_zf, marker="s", label="Clip+Filter ZF")
            plt.semilogy(snr_dbs, ber_cf_mmse, marker="s", label="Clip+Filter MMSE")

        plt.grid(True, which="both")
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.title("BER vs SNR (ZF vs MMSE)")
        plt.legend()

    # ---------------- Save all figures ----------------
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        fig.savefig(
            results_dir / f"{tag}_fig{fig_num}.png", dpi=200, bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":
    main()
