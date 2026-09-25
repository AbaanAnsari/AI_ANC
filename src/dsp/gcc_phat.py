from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def gcc_phat(
    m1: np.ndarray,
    m2: np.ndarray,
    sample_rate: int = 16000,
    max_tau: int | None = None,
    interp: int = 1,
) -> dict:
    """
    Compute Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    delay estimation between two microphone signals M1 and M2.

    Sign Convention:
    ----------------
    - Positive estimated delay (+tau > 0): M2 is delayed relative to M1
      (the signal arrives at M1 first, then at M2 after tau samples).
    - Negative estimated delay (-tau < 0): M1 is delayed relative to M2
      (the signal arrives at M2 first, then at M1 after tau samples).
    - Zero estimated delay (tau == 0): Signals arrive at M1 and M2 simultaneously.

    Parameters
    ----------
    m1 : np.ndarray
        Primary microphone signal (1-D float array).
    m2 : np.ndarray
        Reference microphone signal (1-D float array).
    sample_rate : int
        Audio sampling rate in Hz (default: 16000).
    max_tau : int | None
        Maximum search lag in samples. If specified, search is restricted
        to lags in [-max_tau, +max_tau].
    interp : int
        Interpolation factor for sub-sample accuracy (default: 1).

    Returns
    -------
    dict
        {
            "estimated_delay_samples": int,
            "estimated_delay_seconds": float,
            "gcc_cc": np.ndarray, # 1-D float32 array of correlation values
            "lags": np.ndarray,   # 1-D int array of signed sample lags
        }
    """
    if not isinstance(m1, np.ndarray) or not isinstance(m2, np.ndarray):
        raise TypeError("m1 and m2 must be NumPy arrays")

    if m1.ndim != 1 or m2.ndim != 1:
        raise ValueError(
            f"m1 and m2 must be 1-D arrays, got shapes {m1.shape} and {m2.shape}"
        )

    if not np.all(np.isfinite(m1)) or not np.all(np.isfinite(m2)):
        raise ValueError("Inputs contain NaN or Inf values")

    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

    # Work on copies/aligned float32 arrays
    s1 = m1.astype(np.float32)
    s2 = m2.astype(np.float32)

    # Handle zero / near-zero signal edge cases
    std1 = float(np.std(s1))
    std2 = float(np.std(s2))
    if std1 < EPSILON or std2 < EPSILON:
        zero_lags = (
            np.arange(-max_tau, max_tau + 1, dtype=int)
            if max_tau is not None
            else np.array([0], dtype=int)
        )
        return {
            "estimated_delay_samples": 0,
            "estimated_delay_seconds": 0.0,
            "gcc_cc": np.zeros(len(zero_lags), dtype=np.float32),
            "lags": zero_lags,
        }

    # Handle unequal input lengths by zero-padding shorter signal
    len1, len2 = s1.shape[0], s2.shape[0]
    if len1 != len2:
        max_len = max(len1, len2)
        if len1 < max_len:
            pad1 = np.zeros(max_len, dtype=np.float32)
            pad1[:len1] = s1
            s1 = pad1
        else:
            pad2 = np.zeros(max_len, dtype=np.float32)
            pad2[:len2] = s2
            s2 = pad2

    n_samples = s1.shape[0]

    # Use FFT size that prevents circular wrap-around (at least 2 * n_samples - 1)
    n_fft = interp * int(2 ** np.ceil(np.log2(2 * n_samples - 1)))

    # Compute FFT
    x1 = np.fft.fft(s1, n=n_fft)
    x2 = np.fft.fft(s2, n=n_fft)

    # Cross-spectrum R21 = conj(X1) * X2
    # So if M2[t] = M1[t - tau] (M2 delayed by +tau), peak occurs at +tau
    r12 = np.conj(x1) * x2

    # PHAT weighting Psi = R12 / (|R12| + epsilon)
    denom = np.abs(r12) + EPSILON
    phat = r12 / denom

    # Inverse FFT to get cross-correlation
    cc = np.real(np.fft.ifft(phat))

    # Construct signed lags array
    # Unshifted lags from np.fft.fftfreq
    raw_lags = (np.fft.fftfreq(n_fft, d=1.0) * n_fft / interp).astype(int)

    # Apply fftshift to center zero lag
    cc_shifted = np.fft.fftshift(cc).astype(np.float32)
    lags_shifted = np.fft.fftshift(raw_lags)

    # Restrict lag search window if max_tau is specified
    if max_tau is not None:
        mask = (lags_shifted >= -max_tau) & (lags_shifted <= max_tau)
        lags_out = lags_shifted[mask]
        gcc_cc_out = cc_shifted[mask]
    else:
        lags_out = lags_shifted
        gcc_cc_out = cc_shifted

    # Locate peak lag
    peak_idx = int(np.argmax(gcc_cc_out))
    estimated_delay_samples = int(lags_out[peak_idx])
    estimated_delay_seconds = float(estimated_delay_samples / sample_rate)

    return {
        "estimated_delay_samples": estimated_delay_samples,
        "estimated_delay_seconds": estimated_delay_seconds,
        "gcc_cc": gcc_cc_out,
        "lags": lags_out,
    }
