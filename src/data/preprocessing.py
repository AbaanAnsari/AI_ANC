from __future__ import annotations

import numpy as np


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC component from 1-D float32 audio waveform.
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError(f"audio must be a NumPy array, got {type(audio).__name__}")
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D array, got shape {audio.shape}")

    mean_val = float(np.mean(audio))
    return (audio - mean_val).astype(np.float32)
