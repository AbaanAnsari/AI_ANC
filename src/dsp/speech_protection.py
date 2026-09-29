"""
src/dsp/speech_protection.py
==============================
Speech Protection module — prevents over-suppression of speech energy
in the AI+DSP noise cancellation pipeline.

Architecture Role:
    Stage 6 — Speech protection within Adaptive Fusion.

Mechanism:
    1. Estimate speech energy from VAD probability
    2. Compute the speech-preservation gain
    3. When speech is detected, limit suppression depth
    4. Smooth gain transitions to avoid musical noise / pumping

Design Goals:
    - Avoid suppressing speech: prefer under-suppression over over-suppression
    - Conservative: fail safely by preserving speech
    - Smooth: gain changes use attack/release time constants
    - Impulsive noise: protect speech onsets and transients

Speech Protection Model:
    If VAD detects speech (speech_prob >= threshold):
        max_suppression = max_suppression_dB  (limited, e.g. -6 dB)
    Else (silence/noise only):
        max_suppression = full_suppression_dB  (e.g. -40 dB)

    Smooth the gain across frames with attack/release.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Speech Protection Gain Smoother
# ---------------------------------------------------------------------------

class SpeechProtectionGainSmoother:
    """
    Smooth attack/release gain controller for speech protection.

    Implements:
        If new_gain > current_gain:  use attack_alpha (fast attack)
        If new_gain < current_gain:  use release_alpha (slow release)
        [Prevents sudden jumps in gain]

    Parameters
    ----------
    attack_alpha : float
        Smoothing coefficient for gain increases (0 < alpha < 1).
        alpha=0.9 = fast attack. Default 0.8.
    release_alpha : float
        Smoothing coefficient for gain decreases (0 < alpha < 1).
        alpha=0.1 = slow release. Default 0.05.
    initial_gain : float
        Initial gain value (linear). Default 1.0.
    """

    def __init__(
        self,
        attack_alpha: float = 0.8,
        release_alpha: float = 0.05,
        initial_gain: float = 1.0,
    ) -> None:
        if not (0 < attack_alpha < 1):
            raise ValueError(f"attack_alpha must be in (0,1), got {attack_alpha}")
        if not (0 < release_alpha < 1):
            raise ValueError(f"release_alpha must be in (0,1), got {release_alpha}")
        self.attack_alpha = float(attack_alpha)
        self.release_alpha = float(release_alpha)
        self._gain: float = float(initial_gain)

    def update(self, target_gain: float) -> float:
        """
        Update and return smoothed gain toward target_gain.

        Parameters
        ----------
        target_gain : float
            Desired linear gain value.

        Returns
        -------
        float
            Smoothed linear gain.
        """
        if target_gain > self._gain:
            # Gain increasing: use fast attack
            alpha = self.attack_alpha
        else:
            # Gain decreasing: use slow release
            alpha = self.release_alpha

        self._gain = alpha * target_gain + (1 - alpha) * self._gain
        return self._gain

    @property
    def current_gain(self) -> float:
        return self._gain

    def reset(self, gain: float = 1.0) -> None:
        self._gain = float(gain)


# ---------------------------------------------------------------------------
# Speech Protector
# ---------------------------------------------------------------------------

class SpeechProtector:
    """
    Speech energy protection for the fusion stage.

    Computes a speech-preservation gain mask that limits the maximum
    suppression applied to speech-dominant regions.

    Parameters
    ----------
    speech_threshold : float
        Speech probability threshold above which speech protection activates.
        Default 0.4 (conservative — protect speech more readily).
    max_suppression_speech_db : float
        Maximum suppression allowed when speech is detected, in dB.
        Negative value. Default -6.0 dB.
    max_suppression_silence_db : float
        Maximum suppression allowed during silence, in dB.
        Negative value. Default -40.0 dB.
    impulsive_protection_db : float
        Maximum suppression during impulsive noise detection, in dB.
        Negative value (conservative). Default -3.0 dB.
    attack_alpha : float
        Gain smoother attack coefficient. Default 0.7.
    release_alpha : float
        Gain smoother release coefficient. Default 0.05.
    enabled : bool
        Enable speech protection floor. Set to False for evaluation mode (floor OFF).
        Default True.
    """

    def __init__(
        self,
        speech_threshold: float = 0.4,
        max_suppression_speech_db: float = -24.0,
        max_suppression_silence_db: float = -40.0,
        impulsive_protection_db: float = -18.0,
        attack_alpha: float = 0.7,
        release_alpha: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.speech_threshold = float(speech_threshold)
        self.enabled = bool(enabled)
        # Convert dB limits to linear gain limits
        self.min_gain_speech = float(10 ** (max_suppression_speech_db / 20.0))
        self.min_gain_silence = float(10 ** (max_suppression_silence_db / 20.0))
        self.min_gain_impulsive = float(10 ** (impulsive_protection_db / 20.0))

        self._smoother = SpeechProtectionGainSmoother(
            attack_alpha=attack_alpha,
            release_alpha=release_alpha,
            initial_gain=0.0 if not enabled else self.min_gain_silence,
        )
        self._last_gain: float = 0.0
        self._last_protection_state: str = "disabled" if not enabled else "inactive"

    def compute_protection_gain(
        self,
        speech_probability: float,
        noise_class: int = 0,
        is_impulsive_transient: bool = False,
    ) -> float:
        """
        Compute the minimum gain (speech-protection floor) for the current frame.

        Parameters
        ----------
        speech_probability : float
            VAD speech probability in [0, 1].
        noise_class : int
            Current noise class (0=stationary, 1=non-stationary, 2=impulsive).
        is_impulsive_transient : bool
            True if an impulsive transient has been detected in this frame.

        Returns
        -------
        float
            Linear minimum gain in [min_gain_silence, 1.0] (or 0.0 if disabled).
        """
        if not self.enabled:
            self._last_gain = 0.0
            self._last_protection_state = "disabled"
            return 0.0

        # During impulsive transients: conservative protection
        if is_impulsive_transient or noise_class == 2:
            base_min_gain = self.min_gain_impulsive
            state = "active_impulsive"
        elif speech_probability >= self.speech_threshold:
            base_min_gain = self.min_gain_speech
            state = "active_speech"
        else:
            base_min_gain = self.min_gain_silence
            state = "inactive"

        # Interpolate smoothly based on speech probability
        speech_blend = min(1.0, max(0.0, (speech_probability - self.speech_threshold) / 
                           max(1.0 - self.speech_threshold, 1e-6)))
        
        target_gain = (
            self.min_gain_silence + 
            speech_blend * (base_min_gain - self.min_gain_silence)
        )

        smoothed_gain = self._smoother.update(target_gain)
        clipped_gain = float(np.clip(smoothed_gain, self.min_gain_silence, 1.0))
        self._last_gain = clipped_gain
        self._last_protection_state = state
        return clipped_gain

    def apply_protection(
        self,
        signal: np.ndarray,
        reference_signal: np.ndarray,
        speech_probability: float,
        noise_class: int = 0,
        is_impulsive_transient: bool = False,
    ) -> np.ndarray:
        """
        Apply speech protection to a signal block.
        """
        if not self.enabled:
            self._last_gain = 0.0
            self._last_protection_state = "disabled"
            return np.asarray(signal, dtype=np.float32).ravel().copy()

        protection_gain = self.compute_protection_gain(
            speech_probability=speech_probability,
            noise_class=noise_class,
            is_impulsive_transient=is_impulsive_transient,
        )

        signal = np.asarray(signal, dtype=np.float32).ravel()
        reference_signal = np.asarray(reference_signal, dtype=np.float32).ravel()

        protected = signal + protection_gain * (reference_signal - signal)
        return protected.astype(np.float32)

    @property
    def protection_state(self) -> str:
        return self._last_protection_state

    @property
    def applied_attenuation_floor_db(self) -> float:
        if self._last_gain <= 1e-6:
            return -100.0
        return float(20.0 * np.log10(self._last_gain))

    def reset(self) -> None:
        """Reset gain smoother state."""
        self._smoother.reset(gain=0.0 if not self.enabled else self.min_gain_silence)
        self._last_gain = 0.0
        self._last_protection_state = "disabled" if not self.enabled else "inactive"


# ---------------------------------------------------------------------------
# Impulsive Transient Detector
# ---------------------------------------------------------------------------

class ImpulsiveTransientDetector:
    """
    Detects impulsive noise transients by comparing instantaneous energy
    to a smoothed background estimate.

    Parameters
    ----------
    alpha_fast : float
        Fast smoothing coefficient for transient tracking. Default 0.5.
    alpha_slow : float
        Slow smoothing coefficient for background tracking. Default 0.01.
    threshold : float
        Ratio threshold: fast/slow ratio above this → transient. Default 6.0.
    """

    def __init__(
        self,
        alpha_fast: float = 0.5,
        alpha_slow: float = 0.01,
        threshold: float = 6.0,
    ) -> None:
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.threshold = float(threshold)
        self._fast_power: Optional[float] = None
        self._slow_power: Optional[float] = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Detect if the frame contains an impulsive transient.

        Parameters
        ----------
        frame : np.ndarray
            Audio samples for this frame.

        Returns
        -------
        bool
            True if transient detected.
        """
        frame = np.asarray(frame, dtype=np.float64).ravel()
        power = float(np.mean(frame ** 2)) if len(frame) > 0 else 0.0

        if self._fast_power is None or self._slow_power is None:
            self._fast_power = power
            self._slow_power = power
            return False

        self._fast_power = self.alpha_fast * power + (1 - self.alpha_fast) * self._fast_power
        self._slow_power = self.alpha_slow * power + (1 - self.alpha_slow) * self._slow_power

        ratio = self._fast_power / max(self._slow_power, 1e-12)
        return ratio > self.threshold

    def reset(self) -> None:
        self._fast_power = None
        self._slow_power = None

