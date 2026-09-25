from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import lfilter


def apply_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    """
    Apply integer-sample delay to a 1-D waveform using zero-padding.
    Preserves exact input length.

    - Positive delay (>0): shifts signal right (lag), padding zeros at start.
    - Negative delay (<0): shifts signal left (lead), padding zeros at end.
    - Zero delay (=0): returns a copy of signal.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be a NumPy array, got {type(signal).__name__}")
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D array, got shape {signal.shape}")

    length = signal.shape[0]
    if delay_samples == 0:
        return signal.copy().astype(np.float32)

    output = np.zeros(length, dtype=np.float32)

    if delay_samples > 0:
        if delay_samples < length:
            output[delay_samples:] = signal[: length - delay_samples]
    else:
        abs_delay = abs(delay_samples)
        if abs_delay < length:
            output[: length - abs_delay] = signal[abs_delay:]

    return output


def apply_gain(signal: np.ndarray, gain: float) -> np.ndarray:
    """
    Apply gain scaling to a 1-D waveform.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be a NumPy array, got {type(signal).__name__}")
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D array, got shape {signal.shape}")

    return (signal * float(gain)).astype(np.float32)


def apply_path_filter(
    signal: np.ndarray,
    coefficients: Sequence[float] | None = (0.9, 0.1),
) -> np.ndarray:
    """
    Apply a mild FIR transfer filter to simulate acoustic path / mic response difference.
    Guaranteed stable FIR filtering.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"signal must be a NumPy array, got {type(signal).__name__}")
    if signal.ndim != 1:
        raise ValueError(f"signal must be 1-D array, got shape {signal.shape}")

    if coefficients is None or len(coefficients) == 0:
        return signal.copy().astype(np.float32)

    b = np.asarray(coefficients, dtype=np.float32)
    # Normalize b so gain is preserved
    b_sum = float(np.sum(b))
    if abs(b_sum) > 1e-6:
        b = b / b_sum

    filtered = lfilter(b, [1.0], signal).astype(np.float32)
    return filtered


def simulate_dual_mic(
    clean_speech: np.ndarray,
    scaled_noise: np.ndarray,
    relative_delay_samples: int = 5,
    gain_mismatch: float = 1.0,
    filter_coefficients: Sequence[float] | None = (0.9, 0.1),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate dual-microphone array acquisition (M1 primary mic, M2 reference mic).

    Model:
        Primary path noise N1 = scaled_noise
        Primary mic M1 = clean_speech + N1

        Reference path noise N2 = filter(gain * delay(scaled_noise))
        Reference mic M2 = N2

    Parameters
    ----------
    clean_speech : np.ndarray
        Clean speech 1-D float32 array.
    scaled_noise : np.ndarray
        Scaled noise 1-D float32 array (same length as clean_speech).
    relative_delay_samples : int
        Integer sample delay for reference channel (-20 to +20).
    gain_mismatch : float
        Gain mismatch factor for reference channel (0.8 to 1.2).
    filter_coefficients : Sequence[float] | None
        FIR filter coefficients for acoustic path difference.
    rng : np.random.Generator | None
        Optional random generator.

    Returns
    -------
    m1 : np.ndarray
        Primary microphone signal (speech + N1).
    m2 : np.ndarray
        Reference microphone signal (correlated N2).
    """
    if not isinstance(clean_speech, np.ndarray) or not isinstance(
        scaled_noise, np.ndarray
    ):
        raise TypeError("clean_speech and scaled_noise must be NumPy arrays")

    if clean_speech.ndim != 1 or scaled_noise.ndim != 1:
        raise ValueError("clean_speech and scaled_noise must be 1-D arrays")

    if clean_speech.shape[0] != scaled_noise.shape[0]:
        raise ValueError(
            f"Shape mismatch: clean {clean_speech.shape} vs noise {scaled_noise.shape}"
        )

    # 1. Primary path noise N1
    n1 = scaled_noise.astype(np.float32)
    m1 = (clean_speech + n1).astype(np.float32)

    # 2. Reference path noise N2 (delay -> gain -> filter)
    n2_delayed = apply_delay(n1, relative_delay_samples)
    n2_gained = apply_gain(n2_delayed, gain_mismatch)
    n2_filtered = apply_path_filter(n2_gained, filter_coefficients)

    m2 = n2_filtered.astype(np.float32)

    if not np.all(np.isfinite(m1)) or not np.all(np.isfinite(m2)):
        raise RuntimeError("Dual mic signals contain non-finite values")

    return m1, m2


def calculate_cross_correlation(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Calculate normalized cross-correlation peak between two signals.
    """
    if sig1.shape != sig2.shape:
        raise ValueError("Signals must have identical shapes")

    norm1 = np.linalg.norm(sig1)
    norm2 = np.linalg.norm(sig2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    correlation = np.correlate(sig1, sig2, mode="full")
    max_corr = float(np.max(np.abs(correlation)))
    return max_corr / float(norm1 * norm2)
