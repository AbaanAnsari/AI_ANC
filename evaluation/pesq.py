"""
evaluation/pesq.py — PESQ Availability Wrapper
================================================
Wraps pesq with a clean availability check.

If the 'pesq' package is not installed, all functions return None and
log the dependency as UNAVAILABLE. No values are fabricated.
"""
from __future__ import annotations

from typing import Optional

try:
    from pesq import pesq as _pesq_fn
    from pesq import PesqError
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    PesqError = Exception

PESQ_UNAVAILABLE_MSG = (
    "PESQ is UNAVAILABLE: 'pesq' package is not installed. "
    "Install with: pip install pesq"
)


def compute_pesq(
    clean: "np.ndarray",
    degraded: "np.ndarray",
    sample_rate: int = 16000,
    mode: str = "wb",
) -> Optional[float]:
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ).

    Parameters
    ----------
    clean : np.ndarray
        Reference clean speech (1-D float32).
    degraded : np.ndarray
        Noisy or enhanced speech of the same length.
    sample_rate : int
        Audio sample rate. PESQ supports 8000 or 16000 Hz.
    mode : str
        'wb' = wideband (16 kHz), 'nb' = narrowband (8 kHz).

    Returns
    -------
    float or None
        PESQ MOS-LQO score if pesq is available, else None.
    """
    if not HAS_PESQ:
        return None

    import numpy as np
    clean_f = np.asarray(clean, dtype=np.float32)
    deg_f = np.asarray(degraded, dtype=np.float32)

    min_len = min(len(clean_f), len(deg_f))
    clean_f = clean_f[:min_len]
    deg_f = deg_f[:min_len]

    try:
        return float(_pesq_fn(sample_rate, clean_f, deg_f, mode))
    except PesqError:
        return None


def pesq_available() -> bool:
    """Return True if the pesq package is installed and usable."""
    return HAS_PESQ


def pesq_unavailable_message() -> str:
    return PESQ_UNAVAILABLE_MSG
