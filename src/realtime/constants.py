"""
src/realtime/constants.py
=========================
Global, immutable audio processing constants for AETHEL123.

The audio block size is fixed at 256 samples (16 ms at 16 kHz).
"""
from __future__ import annotations

# Global Immutable Processing Constants
SAMPLE_RATE: int = 16000
BLOCK_SIZE: int = 256
BLOCK_DURATION_MS: float = 16.0  # (256 / 16000) * 1000 = 16 ms
HOP_SIZE: int = 128              # 8 ms (STFT / overlap-add analysis)
N_FFT: int = 512                 # 512-point FFT for spectrogram
WINDOW_SIZE: int = 512
N_FREQ_BINS: int = 257           # N_FFT // 2 + 1

# Permanent Non-Editable Status String
AUDIO_CONFIG_INDICATOR: str = "256 samples | 16 ms | 16 kHz"
