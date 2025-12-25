# Advanced Wireless Communication Systems – OFDM/PAPR/HPA Project (Python)

This project implements an **802.11a-like OFDM transmitter chain** and the distortion analysis blocks often required in PAPR / clipping / HPA assignments.

## What’s included
- Convolutional encoder (K=7, rate 1/2 mother code) + puncturing for 2/3 and 3/4
- **Interleaver** (802.11a 2-step permutation)
- Gray mapping for BPSK/QPSK/16QAM/64QAM with corrected normalization factors
- OFDM subcarrier mapping (48 data + 4 pilots), IFFT, cyclic prefix, oversampling
- PAPR per symbol + CCDF
- Clipping + ideal bandlimiting filter (iterative clip/filter)
- Rapp HPA model + PSD + EVM (with optional CPE correction)

## Run
From the project folder:
```bash
python main.py
```

You will be asked to select:
- Modulation type
- Code rate
- Number of OFDM symbols
- Oversampling factor
- Clipping ratio + iterations
- HPA backoff and Rapp parameter

## Dependencies
- numpy
- scipy
- matplotlib
