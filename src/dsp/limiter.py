"""
src/dsp/limiter.py
===================
Safety Limiter — Final output stage before audio delivery.

Architecture Role:
    Stage 7 — Safety Limiter (last stage before Enhanced Speech Output).

This is the ultimate safety net. No clipping or instability must pass through.

Requirements:
    - Configurable peak threshold
    - No clipping at output
    - Smooth gain control (attack/release)
    - Finite output guaranteed (NaN/Inf → silence)
    - Deterministic behavior
    - Works on CPU without GPU
    - Tested with extreme inputs (silence, clipping, +/-Inf, NaN)

Algorithm:
    Lookahead peak limiter:
    1. Detect peaks in the signal
    2. Compute required gain to bring peaks below threshold
    3. Smooth gain with attack/release time constants
    4. Apply smoothed gain to signal
    5. Hard-clip to [-1.0, 1.0] as final safety net

    gain[n] = min(1.0, threshold / (|signal[n]| + epsilon))
    smoothed_gain = attack/release smooth of gain[n]
    output[n] = smoothed_gain * signal[n]
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Safety Limiter
# ---------------------------------------------------------------------------

class SafetyLimiter:
    """
    Peak limiter with smooth attack/release gain control.

    Processes audio block-by-block. Stateful — maintains gain history
    across consecutive process() calls for seamless streaming.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz. Default 16000.
    threshold_db : float
        Peak threshold in dBFS. Must be <= 0.0. Default -1.0 dBFS.
        Signals above this level will be attenuated.
    attack_ms : float
        Attack time in milliseconds. How quickly the limiter responds
        to peaks. Default 5.0 ms.
    release_ms : float
        Release time in milliseconds. How quickly the gain recovers
        after a peak. Default 100.0 ms.
    epsilon : float
        Small value added to denominator to prevent division by zero.
        Default 1e-8.
    hard_clip : bool
        Apply hard clip as final safety net. Default True.
    hard_clip_level : float
        Hard clip level (absolute sample value). Default 0.9999.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_db: float = -1.0,
        attack_ms: float = 5.0,
        release_ms: float = 100.0,
        epsilon: float = 1e-8,
        hard_clip: bool = True,
        hard_clip_level: float = 0.9999,
        enabled: bool = True,
    ) -> None:
        if threshold_db > 0:
            raise ValueError(f"threshold_db must be <= 0, got {threshold_db}")
        if attack_ms <= 0:
            raise ValueError(f"attack_ms must be > 0, got {attack_ms}")
        if release_ms <= 0:
            raise ValueError(f"release_ms must be > 0, got {release_ms}")
        if hard_clip_level <= 0 or hard_clip_level > 1.0:
            raise ValueError(f"hard_clip_level must be in (0, 1], got {hard_clip_level}")

        self.sample_rate = int(sample_rate)
        self.threshold_linear = float(10 ** (threshold_db / 20.0))
        self.epsilon = float(epsilon)
        self.hard_clip = bool(hard_clip)
        self.hard_clip_level = float(hard_clip_level)
        self.enabled = bool(enabled)

        # Convert time constants to per-sample coefficients
        # alpha = exp(-1 / (time_constant_samples))
        attack_samples = max(1.0, attack_ms * sample_rate / 1000.0)
        release_samples = max(1.0, release_ms * sample_rate / 1000.0)
        self.attack_coeff = float(math.exp(-1.0 / attack_samples))
        self.release_coeff = float(math.exp(-1.0 / release_samples))

        # State
        self._gain: float = 1.0      # current gain (linear)
        self._peak: float = 0.0      # peak detector state
        self._clipping_count: int = 0
        self._n_samples_processed: int = 0
        self._nan_inf_count: int = 0

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the safety limiter to a signal block.
        If enabled is False (evaluation mode), bypasses compression while maintaining finite safety.
        """
        signal = np.asarray(signal, dtype=np.float64).ravel()

        # --- Stage 0: NaN/Inf guard ---
        non_finite_mask = ~np.isfinite(signal)
        if np.any(non_finite_mask):
            self._nan_inf_count += int(np.sum(non_finite_mask))
            signal = signal.copy()
            signal[non_finite_mask] = 0.0

        if not self.enabled:
            self._n_samples_processed += len(signal)
            return signal.astype(np.float32)

        N = len(signal)
        output = np.empty(N, dtype=np.float64)

        for i in range(N):
            sample = signal[i]
            abs_sample = abs(sample)

            # Peak detection (exponential decay)
            if abs_sample > self._peak:
                self._peak = abs_sample
            else:
                self._peak *= self.release_coeff

            # Compute required gain
            if self._peak > self.threshold_linear:
                target_gain = self.threshold_linear / (self._peak + self.epsilon)
            else:
                target_gain = 1.0

            target_gain = min(1.0, max(0.0, target_gain))

            # Smooth gain (attack faster than release)
            if target_gain < self._gain:
                # Gain must decrease: use attack coefficient (fast)
                self._gain = self.attack_coeff * self._gain + (1 - self.attack_coeff) * target_gain
            else:
                # Gain can increase: use release coefficient (slow)
                self._gain = self.release_coeff * self._gain + (1 - self.release_coeff) * target_gain

            output[i] = self._gain * sample
            self._n_samples_processed += 1

        # --- Stage 1: Hard clip safety net ---
        if self.hard_clip:
            clipped = (np.abs(output) > self.hard_clip_level)
            if np.any(clipped):
                self._clipping_count += int(np.sum(clipped))
                output = np.clip(output, -self.hard_clip_level, self.hard_clip_level)

        # Final finite check
        if not np.all(np.isfinite(output)):
            # Should never happen given the guards above, but be safe
            output = np.where(np.isfinite(output), output, 0.0)

        return output.astype(np.float32)

    def reset(self) -> None:
        """Reset all state variables to initial values."""
        self._gain = 1.0
        self._peak = 0.0
        self._clipping_count = 0
        self._n_samples_processed = 0
        self._nan_inf_count = 0

    @property
    def diagnostics(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "current_gain_linear": self._gain,
            "current_gain_db": float(20.0 * math.log10(max(self._gain, 1e-12))),
            "peak_detector_level": self._peak,
            "clipping_count": self._clipping_count,
            "n_samples_processed": self._n_samples_processed,
            "nan_inf_count": self._nan_inf_count,
            "threshold_linear": self.threshold_linear,
        }

    @property
    def is_clipping(self) -> bool:
        """True if limiter is actively attenuating or has clipped."""
        return bool(self._gain < 0.99 or self._clipping_count > 0)

    @property
    def clipping_count(self) -> int:
        return self._clipping_count

    @property
    def nan_inf_count(self) -> int:
        return self._nan_inf_count
