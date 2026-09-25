"""
evaluation/stoi.py — STOI Availability Wrapper
================================================
Wraps pystoi with a clean availability check.

If pystoi is not installed, all functions return None and log the
dependency as UNAVAILABLE. No values are fabricated.
"""
from __future__ import annotations

from typing import Optional

try:
    from pystoi import stoi as _pystoi_stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

STOI_UNAVAILABLE_MSG = (
    "STOI is UNAVAILABLE: 'pystoi' package is not installed. "
    "Install with: pip install pystoi"
)


def compute_stoi(
    clean: "np.ndarray",
    degraded: "np.ndarray",
    sample_rate: int = 16000,
    extended: bool = False,
) -> Optional[float]:
    """
    Compute Short-Time Objective Intelligibility (STOI).

    Parameters
    ----------
    clean : np.ndarray
        Reference clean speech (1-D float32).
    degraded : np.ndarray
        Noisy or enhanced speech of the same length.
    sample_rate : int
        Audio sample rate in Hz (default 16000).
    extended : bool
        Use Extended STOI (ESTOI) formulation if True.

    Returns
    -------
    float or None
        STOI score in [0, 1] if pystoi is available, else None.
    """
    if not HAS_STOI:
        return None

    import numpy as np
    clean_f = np.asarray(clean, dtype=np.float64)
    deg_f = np.asarray(degraded, dtype=np.float64)

    min_len = min(len(clean_f), len(deg_f))
    clean_f = clean_f[:min_len]
    deg_f = deg_f[:min_len]

    return float(_pystoi_stoi(clean_f, deg_f, sample_rate, extended=extended))


def stoi_available() -> bool:
    """Return True if pystoi is installed and usable."""
    return HAS_STOI


def stoi_unavailable_message() -> str:
    return STOI_UNAVAILABLE_MSG
