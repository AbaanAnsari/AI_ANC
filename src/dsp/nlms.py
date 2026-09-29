"""
src/dsp/nlms.py
================
Normalized Least Mean Squares (NLMS) Adaptive Filter
for dual-microphone noise cancellation.

Architecture Role:
    Stage 5 — Conditional Noise Processing (after AI classification)

    MIC2 (reference noise) ──► NLMS ──► estimated noise
    MIC1 (speech + noise)  ──► subtract estimated noise ──► enhanced output

Noise-class-dependent NLMS configurations:
    Stationary:     stable/conservative — small step size, longer filter
    Non-stationary: adaptive — larger step size, shorter filter, faster tracking
    Impulsive:      conservative + transient gate — very small step size,
                    impulsive sample detection and bypass

Mathematical Formulation:
    Adaptive FIR filter h[k] of length L:
        y[n] = h[n]^T x[n]           (estimated noise)
        e[n] = d[n] - y[n]           (error = signal - estimated noise)
        h[n+1] = h[n] + mu * e[n] * x[n] / (||x[n]||^2 + epsilon)

    where:
        x[n]  = [x_ref[n], x_ref[n-1], ..., x_ref[n-L+1]] (reference vector)
        d[n]  = primary microphone sample (speech + noise)
        mu    = normalized step size (0 < mu < 2 for convergence)
        epsilon = regularization to avoid division by zero

Leakage:
    Optional leakage coefficient lambda (0 < lambda <= 1):
        h[n+1] = lambda * h[n] + mu * e[n] * x[n] / (||x[n]||^2 + epsilon)
    Leakage prevents coefficient drift in non-stationary conditions.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOISE_CLASS_STATIONARY    = 0
NOISE_CLASS_NON_STATIONARY = 1
NOISE_CLASS_IMPULSIVE     = 2

# Default NLMS configurations per noise class
_DEFAULT_CONFIGS = {
    NOISE_CLASS_STATIONARY: {
        "filter_length": 128,
        "step_size": 0.02,
        "leakage": 1.0,       # no leakage
        "epsilon": 1e-6,
        "impulsive_gate": False,
        "impulsive_threshold": None,
    },
    NOISE_CLASS_NON_STATIONARY: {
        "filter_length": 64,
        "step_size": 0.10,
        "leakage": 0.9999,
        "epsilon": 1e-6,
        "impulsive_gate": False,
        "impulsive_threshold": None,
    },
    NOISE_CLASS_IMPULSIVE: {
        "filter_length": 32,
        "step_size": 0.005,
        "leakage": 0.9995,
        "epsilon": 1e-6,
        "impulsive_gate": True,
        "impulsive_threshold": 5.0,   # samples > 5*RMS are gated
    },
}


# ---------------------------------------------------------------------------
# Core NLMS Filter
# ---------------------------------------------------------------------------

class NLMSFilter:
    """
    Normalized Least Mean Squares adaptive filter for noise cancellation.

    Processes sample-by-sample or block-by-block.

    Parameters
    ----------
    filter_length : int
        Number of adaptive FIR taps (filter order). Must be >= 1.
    step_size : float
        Normalized step size mu (0 < mu < 2). Typical range 0.01–0.5.
        Smaller = slower convergence but more stable.
    leakage : float
        Leakage coefficient (0 < leakage <= 1). Default 1.0 (no leakage).
        Values < 1 prevent coefficient drift in non-stationary conditions.
    epsilon : float
        Regularization constant to prevent division by zero. Default 1e-6.
    impulsive_gate : bool
        If True, bypass weight update for samples identified as impulsive.
    impulsive_threshold : float or None
        Threshold in multiples of local RMS for impulsive detection.
        Only used if impulsive_gate=True.
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.05,
        leakage: float = 1.0,
        epsilon: float = 1e-6,
        impulsive_gate: bool = False,
        impulsive_threshold: Optional[float] = 5.0,
        adaptation_enabled: bool = True,
        speech_freeze_threshold: float = 0.5,
        max_weight_norm: float = 5.0,
    ) -> None:
        if filter_length < 1:
            raise ValueError(f"filter_length must be >= 1, got {filter_length}")
        if not (0 < step_size < 2):
            raise ValueError(f"step_size must be in (0, 2), got {step_size}")
        if not (0 < leakage <= 1.0):
            raise ValueError(f"leakage must be in (0, 1], got {leakage}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if impulsive_gate and impulsive_threshold is not None and impulsive_threshold <= 0:
            raise ValueError(f"impulsive_threshold must be > 0, got {impulsive_threshold}")

        self.filter_length = int(filter_length)
        self.step_size = float(step_size)
        self.leakage = float(leakage)
        self.epsilon = float(epsilon)
        self.impulsive_gate = bool(impulsive_gate)
        self.impulsive_threshold = float(impulsive_threshold) if impulsive_threshold is not None else 5.0
        self.adaptation_enabled = bool(adaptation_enabled)
        self.speech_freeze_threshold = float(speech_freeze_threshold)
        self.max_weight_norm = float(max_weight_norm)

        # Adaptive weights — initialized to zero
        self._weights: np.ndarray = np.zeros(self.filter_length, dtype=np.float64)
        # Circular reference buffer
        self._ref_buffer: np.ndarray = np.zeros(self.filter_length, dtype=np.float64)
        self._buf_idx: int = 0

        # Diagnostics
        self._n_samples_processed: int = 0
        self._n_impulsive_gates: int = 0
        self._running_power: float = 0.0  # exponential moving average of |ref|^2

    def reset(self) -> None:
        """Reset all adaptive weights and buffer state to zero."""
        self._weights[:] = 0.0
        self._ref_buffer[:] = 0.0
        self._buf_idx = 0
        self._n_samples_processed = 0
        self._n_impulsive_gates = 0
        self._running_power = 0.0

    def process_sample(
        self,
        primary: float,
        reference: float,
    ) -> Tuple[float, float]:
        """
        Process one sample through the NLMS filter.

        Parameters
        ----------
        primary : float
            Primary microphone sample d[n] = speech + noise.
        reference : float
            Reference microphone sample x[n] = correlated noise.

        Returns
        -------
        (error, estimated_noise) : (float, float)
            error            = primary - estimated_noise (enhanced speech)
            estimated_noise  = h^T x (filter output)
        """
        if not math.isfinite(primary) or not math.isfinite(reference):
            return float(primary), 0.0

        # Update circular buffer with new reference sample
        self._ref_buffer[self._buf_idx] = reference
        self._buf_idx = (self._buf_idx + 1) % self.filter_length

        # Construct reference vector x (newest first)
        # Unroll from circular buffer
        idx_arr = (self._buf_idx - 1 - np.arange(self.filter_length)) % self.filter_length
        x_vec = self._ref_buffer[idx_arr]

        # Filter output: estimated noise
        estimated_noise = float(np.dot(self._weights, x_vec))

        # Error signal: enhanced (hopefully) primary
        error = float(primary) - estimated_noise

        # Normalized power of reference vector
        ref_power = float(np.dot(x_vec, x_vec)) + self.epsilon

        # Impulsive gate: skip weight update if sample is impulsive
        is_impulsive = False
        if self.impulsive_gate:
            # Exponential moving average of reference power for local RMS estimate
            alpha = 0.01  # slow tracking
            self._running_power = (1 - alpha) * self._running_power + alpha * float(reference ** 2)
            local_rms = math.sqrt(self._running_power + 1e-12)
            if abs(reference) > self.impulsive_threshold * local_rms:
                is_impulsive = True
                self._n_impulsive_gates += 1

        # Weight update (skip if gated)
        if not is_impulsive:
            # h[n+1] = leakage * h[n] + mu * e[n] * x[n] / ref_power
            update = (self.step_size * error / ref_power) * x_vec
            self._weights = self.leakage * self._weights + update

        self._n_samples_processed += 1

        return error, estimated_noise

    def process_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a block of samples through the NLMS filter.

        Parameters
        ----------
        primary : np.ndarray, shape (N,)
            Primary microphone block.
        reference : np.ndarray, shape (N,)
            Reference microphone block.

        Returns
        -------
        (error_block, estimated_noise_block) : (np.ndarray, np.ndarray)
            Both arrays of shape (N,) float32.
        """
        primary = np.asarray(primary, dtype=np.float64).ravel()
        reference = np.asarray(reference, dtype=np.float64).ravel()

        if primary.shape[0] != reference.shape[0]:
            raise ValueError(
                f"primary and reference must have same length, "
                f"got {primary.shape[0]} and {reference.shape[0]}"
            )

        N = primary.shape[0]
        error_block = np.empty(N, dtype=np.float32)
        estimated_noise_block = np.empty(N, dtype=np.float32)

        for i in range(N):
            e, n_hat = self.process_sample(float(primary[i]), float(reference[i]))
            error_block[i] = np.float32(e)
            estimated_noise_block[i] = np.float32(n_hat)

        return error_block, estimated_noise_block

    def process_hop(
        self,
        primary_hop: np.ndarray,
        reference_hop: np.ndarray,
        speech_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a 128-sample audio hop through the NLMS filter.

        Maintains persistent FIR reference history across consecutive hops.
        Each sample undergoes exactly one adaptive weight update.
        Freezes adaptation if speech_prob > speech_freeze_threshold or adaptation_enabled is False.
        """
        primary = np.asarray(primary_hop, dtype=np.float64).ravel()
        reference = np.asarray(reference_hop, dtype=np.float64).ravel()

        N = min(primary.shape[0], reference.shape[0])
        error_hop = np.empty(N, dtype=np.float32)
        estimated_noise_hop = np.empty(N, dtype=np.float32)

        freeze = (not self.adaptation_enabled) or (speech_prob > self.speech_freeze_threshold)

        for i in range(N):
            p = float(primary[i])
            r = float(reference[i])

            if not math.isfinite(p) or not math.isfinite(r):
                error_hop[i] = np.float32(p)
                estimated_noise_hop[i] = 0.0
                continue

            # Update circular reference buffer
            self._ref_buffer[self._buf_idx] = r
            self._buf_idx = (self._buf_idx + 1) % self.filter_length

            # Construct reference vector x (newest first)
            idx_arr = (self._buf_idx - 1 - np.arange(self.filter_length)) % self.filter_length
            x_vec = self._ref_buffer[idx_arr]

            # Filter output: estimated noise
            estimated_noise = float(np.dot(self._weights, x_vec))
            error = p - estimated_noise

            # Normalized power of reference vector
            ref_power = float(np.dot(x_vec, x_vec)) + self.epsilon

            # Impulsive detection
            is_impulsive = False
            if self.impulsive_gate:
                alpha = 0.01
                self._running_power = (1 - alpha) * self._running_power + alpha * (r ** 2)
                local_rms = math.sqrt(self._running_power + 1e-12)
                if abs(r) > self.impulsive_threshold * local_rms:
                    is_impulsive = True
                    self._n_impulsive_gates += 1

            # Weight update (skip if adaptation is frozen or sample is impulsive)
            if not freeze and not is_impulsive:
                update = (self.step_size * error / ref_power) * x_vec
                self._weights = self.leakage * self._weights + update

                # Divergence safeguard: bound weight vector norm
                w_norm = float(np.linalg.norm(self._weights))
                if w_norm > self.max_weight_norm:
                    self._weights *= (self.max_weight_norm / w_norm)

            self._n_samples_processed += 1
            error_hop[i] = np.float32(error)
            estimated_noise_hop[i] = np.float32(estimated_noise)

        return error_hop, estimated_noise_hop

    @property
    def weights(self) -> np.ndarray:
        """Return a copy of current filter weights."""
        return self._weights.copy()

    @property
    def diagnostics(self) -> dict:
        """Return diagnostic statistics."""
        return {
            "filter_length": self.filter_length,
            "step_size": self.step_size,
            "leakage": self.leakage,
            "n_samples_processed": self._n_samples_processed,
            "n_impulsive_gates": self._n_impulsive_gates,
            "weight_norm": float(np.linalg.norm(self._weights)),
            "weight_max_abs": float(np.max(np.abs(self._weights))),
            "running_power": self._running_power,
        }


# ---------------------------------------------------------------------------
# Noise-class-aware NLMS factory
# ---------------------------------------------------------------------------

def create_nlms_for_noise_class(
    noise_class: int,
    filter_length: Optional[int] = None,
    step_size: Optional[float] = None,
    leakage: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> NLMSFilter:
    """
    Create an NLMS filter configured for the specified noise class.

    Parameters
    ----------
    noise_class : int
        0 = stationary, 1 = non-stationary, 2 = impulsive.
    filter_length, step_size, leakage, epsilon : optional overrides
        Override the class-specific defaults if provided.

    Returns
    -------
    NLMSFilter configured for the specified noise class.

    Raises
    ------
    ValueError
        If noise_class is not 0, 1, or 2.
    """
    if noise_class not in _DEFAULT_CONFIGS:
        raise ValueError(
            f"noise_class must be 0, 1, or 2; got {noise_class}. "
            f"0=stationary, 1=non-stationary, 2=impulsive"
        )

    defaults = _DEFAULT_CONFIGS[noise_class].copy()

    # Apply overrides
    if filter_length is not None:
        defaults["filter_length"] = filter_length
    if step_size is not None:
        defaults["step_size"] = step_size
    if leakage is not None:
        defaults["leakage"] = leakage
    if epsilon is not None:
        defaults["epsilon"] = epsilon

    return NLMSFilter(
        filter_length=defaults["filter_length"],
        step_size=defaults["step_size"],
        leakage=defaults["leakage"],
        epsilon=defaults["epsilon"],
        impulsive_gate=defaults["impulsive_gate"],
        impulsive_threshold=defaults["impulsive_threshold"],
    )


# ---------------------------------------------------------------------------
# Adaptive NLMS (auto-tunes step size based on input statistics)
# ---------------------------------------------------------------------------

class AdaptiveNLMSController:
    """
    Wraps NLMSFilter and dynamically adjusts configuration based on
    noise class updates from the AI classifier.

    Use in the streaming pipeline where noise class may change frame-by-frame.
    """

    def __init__(self) -> None:
        self._current_class: int = NOISE_CLASS_STATIONARY
        self._filter = create_nlms_for_noise_class(NOISE_CLASS_STATIONARY)
        self._class_switch_count: int = 0
        self._last_primary_rms: float = 0.0
        self._last_reference_rms: float = 0.0
        self._last_output_rms: float = 0.0
        self._last_correlation: float = 0.0

    def update_noise_class(self, noise_class: int) -> None:
        """
        Update the active noise class and reconfigure the NLMS filter.

        NOTE: Preserves filter weights and reference buffer history.
        Does NOT reset internal delay buffers merely because of a class flip.

        Parameters
        ----------
        noise_class : int
            New noise class from AI classifier (0, 1, or 2).
        """
        if noise_class not in _DEFAULT_CONFIGS:
            return

        if noise_class == self._current_class:
            return

        cfg = _DEFAULT_CONFIGS[noise_class]

        # If filter length is the same, simply update the hyper-parameters in-place!
        if cfg["filter_length"] == self._filter.filter_length:
            self._filter.step_size = cfg["step_size"]
            self._filter.leakage = cfg["leakage"]
            self._filter.epsilon = cfg["epsilon"]
            self._filter.impulsive_gate = cfg["impulsive_gate"]
            self._filter.impulsive_threshold = cfg["impulsive_threshold"]
        else:
            old_weights = self._filter.weights
            old_ref = self._filter._ref_buffer
            old_idx = self._filter._buf_idx
            old_power = self._filter._running_power
            old_n_samples = self._filter._n_samples_processed

            new_filter = NLMSFilter(
                filter_length=cfg["filter_length"],
                step_size=cfg["step_size"],
                leakage=cfg["leakage"],
                epsilon=cfg["epsilon"],
                impulsive_gate=cfg["impulsive_gate"],
                impulsive_threshold=cfg["impulsive_threshold"],
            )
            # Transfer weights
            min_w_len = min(len(old_weights), cfg["filter_length"])
            new_filter._weights[:min_w_len] = old_weights[:min_w_len]

            # Transfer reference buffer history (unroll chronological order)
            unrolled_ref = np.roll(old_ref, -old_idx)
            copy_r_len = min(len(unrolled_ref), cfg["filter_length"])
            new_filter._ref_buffer[:copy_r_len] = unrolled_ref[-copy_r_len:]
            new_filter._buf_idx = copy_r_len % cfg["filter_length"]
            new_filter._running_power = old_power
            new_filter._n_samples_processed = old_n_samples

            self._filter = new_filter

        self._current_class = noise_class
        self._class_switch_count += 1

    def process_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a block through the current NLMS configuration."""
        return self._filter.process_block(primary, reference)

    def process_hop(
        self,
        primary_hop: np.ndarray,
        reference_hop: np.ndarray,
        speech_prob: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process an audio hop with persistent FIR history, speech freezing, and diagnostics."""
        p_rms = float(np.sqrt(np.mean(primary_hop.astype(np.float64) ** 2) + 1e-12))
        r_rms = float(np.sqrt(np.mean(reference_hop.astype(np.float64) ** 2) + 1e-12))
        self._last_primary_rms = p_rms
        self._last_reference_rms = r_rms

        if p_rms > 1e-6 and r_rms > 1e-6:
            min_l = min(len(primary_hop), len(reference_hop))
            corr = float(np.mean(primary_hop[:min_l] * reference_hop[:min_l]) / (p_rms * r_rms))
            self._last_correlation = float(np.clip(corr, -1.0, 1.0))
        else:
            self._last_correlation = 0.0

        error_hop, est_noise = self._filter.process_hop(primary_hop, reference_hop, speech_prob=speech_prob)
        self._last_output_rms = float(np.sqrt(np.mean(error_hop.astype(np.float64) ** 2) + 1e-12))
        return error_hop, est_noise

    def reset(self) -> None:
        """Reset the current NLMS filter state."""
        self._filter.reset()
        self._class_switch_count = 0
        self._last_primary_rms = 0.0
        self._last_reference_rms = 0.0
        self._last_output_rms = 0.0
        self._last_correlation = 0.0

    @property
    def step_size(self) -> float:
        return self._filter.step_size

    @step_size.setter
    def step_size(self, val: float) -> None:
        self._filter.step_size = float(val)

    @property
    def adaptation_enabled(self) -> bool:
        return self._filter.adaptation_enabled

    @adaptation_enabled.setter
    def adaptation_enabled(self, val: bool) -> None:
        self._filter.adaptation_enabled = bool(val)

    @property
    def speech_freeze_threshold(self) -> float:
        return self._filter.speech_freeze_threshold

    @speech_freeze_threshold.setter
    def speech_freeze_threshold(self, val: float) -> None:
        self._filter.speech_freeze_threshold = float(val)

    @property
    def current_noise_class(self) -> int:
        return self._current_class

    @property
    def class_switch_count(self) -> int:
        return self._class_switch_count

    @property
    def diagnostics(self) -> dict:
        return {
            "noise_class": self._current_class,
            "class_switch_count": self._class_switch_count,
            "primary_rms": self._last_primary_rms,
            "reference_rms": self._last_reference_rms,
            "m1_m2_correlation": self._last_correlation,
            "nlms_output_rms": self._last_output_rms,
            **self._filter.diagnostics,
        }
