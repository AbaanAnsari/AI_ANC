"""
evaluation/snr.py — SNR Computation Utilities
==============================================
Signal-to-Noise Ratio metrics for speech enhancement evaluation.
All computations use the actual clean target waveform.
"""
from __future__ import annotations

import numpy as np


def compute_snr(clean: np.ndarray, noisy_or_enhanced: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    SNR = 10 * log10( E[clean^2] / E[(clean - signal)^2] )

    Parameters
    ----------
    clean : np.ndarray
        Reference clean speech waveform (1-D float32).
    noisy_or_enhanced : np.ndarray
        Degraded or processed signal of the same shape.

    Returns
    -------
    float
        SNR in dB. Returns -inf if signal power is zero; +inf if residual is zero.
    """
    clean = np.asarray(clean, dtype=np.float64)
    sig = np.asarray(noisy_or_enhanced, dtype=np.float64)

    if clean.shape != sig.shape:
        # Align lengths by truncating to the shorter
        min_len = min(len(clean), len(sig))
        clean = clean[:min_len]
        sig = sig[:min_len]

    signal_power = np.mean(clean ** 2)
    if signal_power == 0.0:
        return float("-inf")

    residual_power = np.mean((clean - sig) ** 2)
    if residual_power == 0.0:
        return float("inf")

    return float(10.0 * np.log10(signal_power / residual_power))


def compute_snr_improvement(
    clean: np.ndarray,
    noisy: np.ndarray,
    enhanced: np.ndarray,
) -> dict:
    """
    Compute input SNR, output SNR, and SNR improvement.

    Parameters
    ----------
    clean : np.ndarray
        Reference clean speech.
    noisy : np.ndarray
        Noisy (input) waveform.
    enhanced : np.ndarray
        Enhanced (output) waveform.

    Returns
    -------
    dict with:
        input_snr_db    : float
        output_snr_db   : float
        snr_improvement : float  (output_snr - input_snr)
    """
    input_snr = compute_snr(clean, noisy)
    output_snr = compute_snr(clean, enhanced)
    improvement = output_snr - input_snr
    return {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement": improvement,
    }
