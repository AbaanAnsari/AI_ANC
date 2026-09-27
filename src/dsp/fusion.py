"""
src/dsp/fusion.py
==================
AI + DSP Adaptive Fusion Stage.

Architecture Role:
    Stage 6 — Adaptive Fusion & Speech Protection

Inputs:
    - AI-enhanced signal (from CNN+GRU CRM model)
    - DSP-enhanced signal (from NLMS adaptive filter)
    - Noise class (from AI classifier: 0=stationary, 1=non-stationary, 2=impulsive)
    - Classifier confidence (softmax probability)
    - VAD speech probability
    - Speech protection gain

Output:
    - Fused enhanced signal

Fusion Strategy:
    weighted_output = w_ai * ai_signal + w_dsp * dsp_signal

    where:
        w_ai, w_dsp depend on:
            - noise_class (classifier-informed weighting)
            - confidence (how certain the classifier is)
            - speech_probability (VAD state)

    Constraints:
        - w_ai + w_dsp = 1.0 (normalized)
        - Weights bounded in [0, 1]
        - Smooth weight transitions (no abrupt switching)

    Design:
        Stationary noise:     ai and dsp roughly equal (DSP very effective)
        Non-stationary:       slightly more AI (better tracking)
        Impulsive:            more AI (transient-aware), less NLMS
        Low confidence:       fall back to more AI-only output
        High speech prob:     protect speech — reduce aggressiveness of both

    No abrupt weight switching — all weights are exponentially smoothed.
"""
from __future__ import annotations

import numpy as np

from src.dsp.speech_protection import SpeechProtector, ImpulsiveTransientDetector
from src.dsp.vad import VoiceActivityDetector


# ---------------------------------------------------------------------------
# Noise-class weight priors
# ---------------------------------------------------------------------------

# Base AI weight per noise class (DSP weight = 1 - AI weight)
# These are priors; actual weights are modulated by confidence and VAD
_BASE_AI_WEIGHTS = {
    0: 0.45,   # stationary: DSP (NLMS) is very effective
    1: 0.55,   # non-stationary: AI is better at tracking
    2: 0.70,   # impulsive: AI transient-aware, NLMS may ring
}

_DEFAULT_AI_WEIGHT = 0.50   # fallback if class unknown


# ---------------------------------------------------------------------------
# Confidence-modulated fusion
# ---------------------------------------------------------------------------

def compute_fusion_weights(
    noise_class: int,
    confidence: float,
    speech_probability: float,
    low_confidence_threshold: float = 0.5,
) -> dict:
    """
    Compute AI and DSP fusion weights based on noise class, confidence, and VAD.

    Parameters
    ----------
    noise_class : int
        0=stationary, 1=non-stationary, 2=impulsive.
    confidence : float
        Classifier confidence in [0, 1] (max softmax probability).
    speech_probability : float
        VAD speech probability in [0, 1].
    low_confidence_threshold : float
        Below this confidence, reduce DSP weight (less aggressive).

    Returns
    -------
    dict with:
        w_ai   : float — AI signal weight in [0, 1]
        w_dsp  : float — DSP signal weight in [0, 1]
        rationale : str
    """
    # Base weight from noise class
    base_ai = _BASE_AI_WEIGHTS.get(noise_class, _DEFAULT_AI_WEIGHT)
    base_dsp = 1.0 - base_ai

    # Confidence modulation: if classifier is uncertain, trust DSP less
    if confidence < low_confidence_threshold:
        # Blend toward 50/50 proportionally to uncertainty
        uncertainty_blend = 1.0 - (confidence / low_confidence_threshold)
        base_ai = base_ai + uncertainty_blend * (0.5 - base_ai) * 0.5
        base_dsp = 1.0 - base_ai

    # Speech protection: during strong speech, reduce total suppression
    # by blending both outputs closer to original (via speech protector)
    # Here we just normalize the weights — speech protection is applied separately
    w_ai = float(np.clip(base_ai, 0.0, 1.0))
    w_dsp = float(np.clip(base_dsp, 0.0, 1.0))

    # Normalize
    total = w_ai + w_dsp
    if total > 0:
        w_ai /= total
        w_dsp /= total
    else:
        w_ai, w_dsp = 0.5, 0.5

    rationale = (
        f"class={noise_class}, conf={confidence:.2f}, "
        f"speech_prob={speech_probability:.2f} → "
        f"w_ai={w_ai:.3f}, w_dsp={w_dsp:.3f}"
    )

    return {"w_ai": w_ai, "w_dsp": w_dsp, "rationale": rationale}


# ---------------------------------------------------------------------------
# Weight Smoother
# ---------------------------------------------------------------------------

class WeightSmoother:
    """
    Exponential smoothing for fusion weights to prevent abrupt switching.

    Parameters
    ----------
    alpha : float
        Smoothing coefficient in (0, 1). 1.0 = no smoothing (instant).
        Default 0.1 (slow adaptation).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._w_ai: float = 0.5
        self._w_dsp: float = 0.5

    def update(self, w_ai: float, w_dsp: float) -> tuple:
        """
        Update and return smoothed weights.

        Returns
        -------
        (smoothed_w_ai, smoothed_w_dsp)
        """
        self._w_ai = self.alpha * w_ai + (1 - self.alpha) * self._w_ai
        self._w_dsp = self.alpha * w_dsp + (1 - self.alpha) * self._w_dsp
        # Renormalize
        total = self._w_ai + self._w_dsp
        if total > 1e-12:
            self._w_ai /= total
            self._w_dsp /= total
        return self._w_ai, self._w_dsp

    def reset(self) -> None:
        self._w_ai = 0.5
        self._w_dsp = 0.5


# ---------------------------------------------------------------------------
# Adaptive Fusion Engine
# ---------------------------------------------------------------------------

class AdaptiveFusionEngine:
    """
    Adaptive AI + DSP fusion with confidence weighting and speech protection.

    Full processing pipeline:
        1. Compute fusion weights (class-dependent, confidence-modulated)
        2. Smooth weights
        3. Mix AI and DSP signals
        4. Apply speech protection
        5. Return fused output with diagnostics

    Parameters
    ----------
    sample_rate : int
        Audio sample rate. Default 16000.
    weight_smoothing_alpha : float
        Fusion weight smoothing coefficient. Default 0.1.
    speech_threshold : float
        Speech probability threshold for protection. Default 0.4.
    max_suppression_speech_db : float
        Max suppression when speech detected (dB). Default -6.0.
    max_suppression_silence_db : float
        Max suppression during silence (dB). Default -40.0.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        weight_smoothing_alpha: float = 0.1,
        speech_threshold: float = 0.4,
        max_suppression_speech_db: float = -6.0,
        max_suppression_silence_db: float = -40.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._weight_smoother = WeightSmoother(alpha=weight_smoothing_alpha)
        self._speech_protector = SpeechProtector(
            speech_threshold=speech_threshold,
            max_suppression_speech_db=max_suppression_speech_db,
            max_suppression_silence_db=max_suppression_silence_db,
        )
        self._transient_detector = ImpulsiveTransientDetector()

        # Diagnostics
        self._last_weights: dict = {"w_ai": 0.5, "w_dsp": 0.5}
        self._last_speech_prob: float = 0.0
        self._n_blocks_processed: int = 0

    def fuse(
        self,
        ai_enhanced: np.ndarray,
        dsp_enhanced: np.ndarray,
        original_primary: np.ndarray,
        noise_class: int = 0,
        confidence: float = 0.5,
        speech_probability: float = 0.0,
    ) -> dict:
        """
        Fuse AI and DSP enhanced signals.

        Parameters
        ----------
        ai_enhanced : np.ndarray, shape (N,)
            AI model enhanced signal.
        dsp_enhanced : np.ndarray, shape (N,)
            NLMS DSP enhanced signal.
        original_primary : np.ndarray, shape (N,)
            Original M1 primary microphone signal.
        noise_class : int
            Noise class from AI classifier.
        confidence : float
            Classifier max softmax probability.
        speech_probability : float
            VAD speech probability.

        Returns
        -------
        dict with:
            fused_signal : np.ndarray, shape (N,) float32
            w_ai         : float — smoothed AI weight
            w_dsp        : float — smoothed DSP weight
            protection_gain : float — speech protection gain applied
            is_transient : bool — transient detected
        """
        ai_enhanced = np.asarray(ai_enhanced, dtype=np.float32).ravel()
        dsp_enhanced = np.asarray(dsp_enhanced, dtype=np.float32).ravel()
        original_primary = np.asarray(original_primary, dtype=np.float32).ravel()

        # Ensure same length
        min_len = min(len(ai_enhanced), len(dsp_enhanced), len(original_primary))
        ai_enhanced = ai_enhanced[:min_len]
        dsp_enhanced = dsp_enhanced[:min_len]
        original_primary = original_primary[:min_len]

        # Guard NaN/Inf
        ai_enhanced = np.where(np.isfinite(ai_enhanced), ai_enhanced, 0.0)
        dsp_enhanced = np.where(np.isfinite(dsp_enhanced), dsp_enhanced, 0.0)
        original_primary = np.where(np.isfinite(original_primary), original_primary, 0.0)

        # 1. Detect transient
        is_transient = self._transient_detector.detect(original_primary)

        # 2. Compute fusion weights
        weight_dict = compute_fusion_weights(
            noise_class=noise_class,
            confidence=confidence,
            speech_probability=speech_probability,
        )

        # 3. Smooth weights
        w_ai, w_dsp = self._weight_smoother.update(weight_dict["w_ai"], weight_dict["w_dsp"])

        # 4. Mix AI and DSP signals
        mixed = w_ai * ai_enhanced + w_dsp * dsp_enhanced

        # 5. Speech protection
        protection_gain = self._speech_protector.compute_protection_gain(
            speech_probability=speech_probability,
            noise_class=noise_class,
            is_impulsive_transient=is_transient,
        )
        protected = self._speech_protector.apply_protection(
            signal=mixed,
            reference_signal=original_primary,
            speech_probability=speech_probability,
            noise_class=noise_class,
            is_impulsive_transient=is_transient,
        )

        self._last_weights = {"w_ai": w_ai, "w_dsp": w_dsp}
        self._last_speech_prob = speech_probability
        self._n_blocks_processed += 1

        return {
            "fused_signal": protected.astype(np.float32),
            "w_ai": w_ai,
            "w_dsp": w_dsp,
            "protection_gain": protection_gain,
            "is_transient": is_transient,
            "noise_class": noise_class,
            "confidence": confidence,
            "speech_probability": speech_probability,
        }

    def reset(self) -> None:
        """Reset all internal state."""
        self._weight_smoother.reset()
        self._speech_protector.reset()
        self._transient_detector.reset()
        self._n_blocks_processed = 0

    @property
    def diagnostics(self) -> dict:
        return {
            "n_blocks_processed": self._n_blocks_processed,
            "last_w_ai": self._last_weights.get("w_ai", 0.5),
            "last_w_dsp": self._last_weights.get("w_dsp", 0.5),
            "last_speech_prob": self._last_speech_prob,
        }
