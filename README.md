# 📡 OFDM Wireless Link Simulator (802.11a-Style)

A Python-based end-to-end OFDM communication chain inspired by IEEE 802.11a.  
Includes transmitter + receiver processing, PAPR reduction (clip & filter), nonlinear power amplifier modeling (Rapp), multipath FIR channel + AWGN, and equalizers Zero-Forcing (ZF) & Minimum Mean-Squared Error (MMSE), with metrics like BER/EVM and spectral plots.

---

## 🌍 Overview

This project simulates a **full OFDM wireless link** starting from input bits (random or MP3-derived) all the way to recovered bits/audio at the receiver.  
It is designed for learning and evaluation of:

- OFDM waveform generation (FFT/IFFT + cyclic prefix)
- Practical PHY steps (FEC, interleaving, mapping)
- PAPR behavior + reduction methods
- Nonlinear PA distortion using **Rapp HPA**
- Frequency-domain equalization (**ZF and MMSE**) under noise
- Quantitative and qualitative performance metrics (BER, EVM, constellation, PSD, CCDF)

---

## ✨ Features

✅ **Modulations:** BPSK, QPSK, 16QAM, 64QAM  
✅ **FEC:** Convolutional encoder (K=7, rate-1/2 mother code) + puncturing (2/3, 3/4)  
✅ **Interleaving:** IEEE 802.11a-style 2-step permutation per OFDM symbol  
✅ **OFDM:** Subcarrier allocation + pilots, IFFT/FFT, cyclic prefix, oversampling for PSD/PAPR  
✅ **PAPR Reduction:** Clip + Filter (iterative)  
✅ **Power Amplifier:** Memoryless **Rapp AM/AM** nonlinearity with configurable IBO and smoothness  
✅ **Channel:** Time-invariant complex FIR multipath channel
✅ **Noise:** AWGN injection with selectable SNR sweep
✅ **Equalizers:** **Zero Forcing (ZF)** and **MMSE**
✅ **MP3 input:** treat MP3 bytes as bitstream, transmit, recover, and re-save output file
✅ **Metrics & Plots:**
- **BER vs SNR:** measures bit reliability across noise conditions (core required plot).
- **EVM:** measures constellation distortion magnitude (sensitive to clipping + HPA).
- **Constellation plots:** visualize distortion/dispersion after equalization in addition to the TX symbols.
- **PAPR CCDF**
  - Shows how often the OFDM signal exceeds a given peak-to-average ratio threshold (the probability of “rare high peaks”). Useful to compare baseline vs clip+filter in terms of peak reduction.
- **PSD / Spectrogram (TX waveform)**
  - **PSD:** Frequency-domain power distribution (average spectrum) used to check bandwidth occupancy and out-of-band emissions (spectral regrowth after clipping/HPA).
  - **Spectrogram:** Time–frequency view showing how the signal’s spectrum evolves over time; useful for spotting transient bursts, leakage, or distortion that varies across the frame.

These metrics and plots are evaulated for the baseline and after applying clip+filter

---

## 🧩 Project Parts

Below is the TX→Channel→RX chain, broken into practical steps (matching the implementation structure).

### 🛰️ Transmitter Chain
1. **Source Bits (Random / MP3)**
  - Generates information bits either from a random generator (for controlled testing) or from a real MP3 file (to evaluate end-to-end reconstruction quality).
  - If MP3 is used, the system can auto-adjust `num_symbols` to avoid truncating the bitstream.

2. **FEC Encoding (Convolutional Coding + Puncturing)**
  - Adds redundancy to protect against channel noise/fading.
  - Puncturing adjusts the effective code rate (1/2, 2/3, 3/4) to trade reliability vs throughput.

3. **Interleaving (Per OFDM Symbol)**
  - 802.11a two-stage permutation 
  - Reorders coded bits inside each OFDM symbol to spread burst errors across time/frequency.
  - Improves Viterbi decoding performance when errors are clustered.

4. **Symbol Mapping (BPSK / QPSK / QAM)**
  - Converts groups of bits into complex constellation points (I/Q symbols).
  - Normalized average power  
  - Higher-order modulation increases spectral efficiency but is more sensitive to noise and nonlinear distortion.

5. **OFDM Modulation (Subcarrier Mapping + IFFT + CP)**
  - Maps data symbols onto active subcarriers (48 data tones) and inserts pilot tones (4 pilots).
  - Uses IFFT to generate the time-domain OFDM waveform.
  - Adds a cyclic prefix (CP) to mitigate inter-symbol interference (ISI) and enable simple frequency-domain equalization.

6. **PAPR Measurement + PAPR Reduction (Clip + Filter)**
  - OFDM has inherently high peak-to-average power ratio (PAPR), which stresses power amplifiers.
  - **Clip+Filter** reduces peaks, improving amplifier efficiency, but may introduce distortion that increases EVM/BER.

7. **HPA (High Power Amplifier) Nonlinearity Model**
  - Rapp HPA with IBO control (drives nonlinearity level)
  - Simulates realistic transmitter power amplifier effects (memoryless Rapp model).
  - Used to evaluate how waveform properties (especially PAPR) impact distortion, spectral regrowth, and receiver performance.

8. **(Optional) I/Q modulation**
   - RF upconversion with fc = 2.4/5/6 GHz
---

### 🌫️ Channel
1. **FIR Multipath Channel**
  - Models frequency-selective fading via a complex FIR filter (limited delay spread to stay within CP length).
  - Convolution with complex taps `h`. By Default: h = [0.8208 + 0.2052j, 0.4104 + 0.1026j, 0.2052 + 0.2052j, 0.1026 + 0.1026j]
  - Creates subcarrier-dependent attenuation/phase rotation (main OFDM challenge).

2. **AWGN Noise**
  - Adds controlled Gaussian noise at selected SNR points.
  - Enables BER vs SNR evaluation for different equalizers and waveform variants (baseline vs clip+filter).

---

### 📡 Receiver Chain
1. **(Optional) I/Q demodulation**
   - RF downconversion and low-pass filter

2. **OFDM Demodulation (Remove CP + FFT + Subcarrier Extraction)**
  - Removes CP, applies FFT, then extracts data and pilot subcarriers.
  - Produces received frequency-domain symbols per subcarrier.

3. **Channel Estimation**
  - Estimates channel frequency response (typically from known pilots).
  - Feeds the equalizer with the per-subcarrier channel estimate.

4. **Equalization (ZF and MMSE)**
  - **ZF (Zero-Forcing):** cancels channel distortion strongly, but can amplify noise at deep fades.
  - **MMSE:** balances channel inversion with noise suppression, usually more robust at low SNR.

5. **Symbol Demapping**
  - Converts equalized constellation points back into bits via hard decisions.

6. **Deinterleaving + FEC Decoding**
  - Reverses interleaving per OFDM symbol.
  - Uses Viterbi decoding to recover the original information bits.

---

## 📊 Output & Results

This section shows test cases for two modulation techniques (**BPSK** and **QPSK**).

---

### ✅ BPSK — Test Case 1

### 🎛️ Inputs
- Modulation: **BPSK**
- Code rate: **1/2**
- Initial OFDM symbols: **2000**
- Oversampling (PAPR/PSD): **4×**
- Clip+Filter iterations: **2**
- Clipping ratio (A/RMS): **2**
- HPA Input Backoff (IBO): **10 dB**
- Rapp smoothness: **p = 2**
- Input source: **MP3 (test.mp3)**
- Auto-adjust: `num_symbols` updated **2000 → 7394** to fit the MP3 bitstream
- Final SNR: **20 dB**

### 📌 Results (Final SNR = 20 dB)
- **Baseline**
  - BER(ZF): **0.000000e+00**
  - BER(MMSE): **0.000000e+00**
  - EVM(ZF): **8.79%**
  - EVM(MMSE): **8.78%**

- **Clip+Filter**
  - BER(ZF): **0.000000e+00**
  - BER(MMSE): **0.000000e+00**
  - EVM(ZF): **13.18%**
  - EVM(MMSE): **13.17%**

- **Figures**

| Constellation (TX) | Spectogram (TX) | PAPR CCDF | PSD (TX, Baseline) | PSD (TX, Clip+Filter) | Constellation (RX, Baseline, ZF) | Constellation (RX, Baseline, MMSE) | Constellation (RX, Clip+Filter, ZF) | (RX, Clip+Filter, MMSE) | BER Vs. SNR |
|---|---|---|---|---|---|---|---|---|---|
| ![fig1](tests/BPSK_tc1/fig1.png) |  ![fig2](tests/BPSK_tc1/fig2.png) | ![fig3](tests/BPSK_tc1/fig3.png) | ![fig4](tests/BPSK_tc1/fig4.png) | ![fig5](tests/BPSK_tc1/fig5.png) | ![fig6](tests/BPSK_tc1/fig6.png) | ![fig7](tests/BPSK_tc1/fig7.png) | ![fig8](tests/BPSK_tc1/fig8.png) | ![fig9](tests/BPSK_tc1/fig9.png) | ![fig10](tests/BPSK_tc1/fig10.png) |
---

### ✅ BPSK — Test Case 2

### 🎛️ Inputs
- Modulation: **BPSK**
- Code rate: **1/2**
- Initial OFDM symbols: **2000**
- Oversampling (PAPR/PSD): **4×**
- Clip+Filter iterations: **2**
- Clipping ratio (A/RMS): **1.2**
- HPA Input Backoff (IBO): **6 dB**
- Rapp smoothness: **p = 2**
- Input source: **MP3 (test.mp3)**
- Auto-adjust: `num_symbols` updated **2000 → 7394** to fit the MP3 bitstream
- Final SNR: **10 dB**

### 📌 Results (Final SNR = 10 dB)
- **Baseline**
  - BER(ZF): **0.000000e+00**
  - BER(MMSE): **0.000000e+00**
  - EVM(ZF): **21.88%**
  - EVM(MMSE): **21.66%**

- **Clip+Filter**
  - BER(ZF): **0.000000e+00**
  - BER(MMSE): **5.635390e-05**
  - EVM(ZF): **38.39%**
  - EVM(MMSE): **36.7%**

- **Figures**

| Constellation (TX) | Spectogram (TX) | PAPR CCDF | PSD (TX, Baseline) | PSD (TX, Clip+Filter) | Constellation (RX, Baseline, ZF) | Constellation (RX, Baseline, MMSE) | Constellation (RX, Clip+Filter, ZF) | (RX, Clip+Filter, MMSE) | BER Vs. SNR |
|---|---|---|---|---|---|---|---|---|---|
| ![fig1](tests/BPSK_tc2/fig1.png) |  ![fig2](tests/BPSK_tc2/fig2.png) | ![fig3](tests/BPSK_tc2/fig3.png) | ![fig4](tests/BPSK_tc2/fig4.png) | ![fig5](tests/BPSK_tc2/fig5.png) | ![fig6](tests/BPSK_tc2/fig6.png) | ![fig7](tests/BPSK_tc2/fig7.png) | ![fig8](tests/BPSK_tc2/fig8.png) | ![fig9](tests/BPSK_tc2/fig9.png) | ![fig10](tests/BPSK_tc2/fig10.png) |

---

### ✅ QPSK — Test Case 1

### 🎛️ Inputs
- Modulation: **QPSK**
- Code rate: **1/2**
- Initial OFDM symbols: **2000**
- Oversampling (PAPR/PSD): **4×**
- Clip+Filter iterations: **2**
- Clipping ratio (A/RMS): **2**
- HPA Input Backoff (IBO): **10 dB**
- Rapp smoothness: **p = 2**
- Input source: **MP3 (test.mp3)**
- Auto-adjust: `num_symbols` updated **2000 → 3697** to fit the MP3 bitstream
- Final SNR: **20 dB**

### 📌 Results (Final SNR = 20 dB)
- **Baseline**
  - BER(ZF): **0.000000e+00**
  - BER(MMSE): **0.000000e+00**
  - EVM(ZF): **8.79%**
  - EVM(MMSE): **8.77%**

- **Clip+Filter**
  - BER(ZF): **1.29614e-03**
  - BER(MMSE): **1.16656e-03**
  - EVM(ZF): **13.45%**
  - EVM(MMSE): **13.45%**

- **Figures**

| Constellation (TX) | Spectogram (TX) | PAPR CCDF | PSD (TX, Baseline) | PSD (TX, Clip+Filter) | Constellation (RX, Baseline, ZF) | Constellation (RX, Baseline, MMSE) | Constellation (RX, Clip+Filter, ZF) | (RX, Clip+Filter, MMSE) | BER Vs. SNR |
|---|---|---|---|---|---|---|---|---|---|
| ![fig1](tests/QPSK_tc1/fig1.png) |  ![fig2](tests/QPSK_tc1/fig2.png) | ![fig3](tests/QPSK_tc1/fig3.png) | ![fig4](tests/QPSK_tc1/fig4.png) | ![fig5](tests/QPSK_tc1/fig5.png) | ![fig6](tests/QPSK_tc1/fig6.png) | ![fig7](tests/QPSK_tc1/fig7.png) | ![fig8](tests/QPSK_tc1/fig8.png) | ![fig9](tests/QPSK_tc1/fig9.png) | ![fig10](tests/QPSK_tc1/fig10.png) |

---

### ✅ QPSK — Test Case 2

### 🎛️ Inputs
- Modulation: **BPSK**
- Code rate: **1/2**
- Initial OFDM symbols: **2000**
- Oversampling (PAPR/PSD): **4×**
- Clip+Filter iterations: **2**
- Clipping ratio (A/RMS): **1.2**
- HPA Input Backoff (IBO): **6 dB**
- Rapp smoothness: **p = 2**
- Input source: **MP3 (test.mp3)**
- Auto-adjust: `num_symbols` updated **2000 → 3697** to fit the MP3 bitstream
- Final SNR: **10 dB**

### 📌 Results (Final SNR = 10 dB)
- **Baseline**
  - BER(ZF): **3.81234e-04**
  - BER(MMSE): **2.761341e-04**
  - EVM(ZF): **21.87%**
  - EVM(MMSE): **21.76%**

- **Clip+Filter**
  - BER(ZF): **2.631727e-03**
  - BER(MMSE): **2.428853e-03**
  - EVM(ZF): **36.64%**
  - EVM(MMSE): **36.18%**

- **Figures**

| Constellation (TX) | Spectogram (TX) | PAPR CCDF | PSD (TX, Baseline) | PSD (TX, Clip+Filter) | Constellation (RX, Baseline, ZF) | Constellation (RX, Baseline, MMSE) | Constellation (RX, Clip+Filter, ZF) | (RX, Clip+Filter, MMSE) | BER Vs. SNR |
|---|---|---|---|---|---|---|---|---|---|
| ![fig1](tests/QPSK_tc2/fig1.png) |  ![fig2](tests/QPSK_tc2/fig2.png) | ![fig3](tests/QPSK_tc2/fig3.png) | ![fig4](tests/QPSK_tc2/fig4.png) | ![fig5](tests/QPSK_tc2/fig5.png) | ![fig6](tests/QPSK_tc2/fig6.png) | ![fig7](tests/QPSK_tc2/fig7.png) | ![fig8](tests/QPSK_tc2/fig8.png) | ![fig9](tests/QPSK_tc2/fig9.png) | ![fig10](tests/QPSK_tc2/fig10.png) |

---

## 📜 License

⚠️ **Important Notice**: This repository is publicly available for viewing only. 
Forking, cloning, or redistributing this project is NOT permitted without explicit permission.
