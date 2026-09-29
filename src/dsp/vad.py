"""
src/dsp/vad.py
================
Voice Activity Detection (VAD) with temporal smoothing and hysteresis.

Architecture Role:
    Stage 6 — VAD / Speech Protection input to Fusion stage.

Design Principles:
    - Energy-based VAD on short frames
    - Temporal smoothing prevents single-frame spurious decisions
    - Hysteresis: higher threshold to enter "speech active" state,
      lower threshold to exit — avoids rapid on/off "pumping"
    - Conservative: when uncertain, prefer "speech active" (fail safely)
    - No decision from single frame — minimum hangover duration

Algorithm:
    1. Compute frame energy (RMS or log-energy)
    2. Apply adaptive noise floor estimation
    3. Classify frame using hysteresis thresholds
    4. Apply temporal smoothing (hangover)
    5. Output: speech_probability in [0, 1], speech_active (bool)
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Frame-level energy computation
# ---------------------------------------------------------------------------

def compute_frame_energy(
    frame: np.ndarray,
    use_log: bool = True,
    log_floor_db: float = -60.0,
) -> float:
    """
    Compute energy of a waveform frame.

    Parameters
    ----------
    frame : np.ndarray
        1-D float audio samples.
    use_log : bool
        Return energy in dB if True, linear RMS if False.
    log_floor_db : float
        Minimum dB value to prevent -inf log.

    Returns
    -------
    float
        Energy value (dB or linear).
    """
    frame = np.asarray(frame, dtype=np.float64).ravel()
    if len(frame) == 0:
        return log_floor_db if use_log else 0.0

    rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))

    if use_log:
        energy_db = float(20.0 * np.log10(max(rms, 10 ** (log_floor_db / 20.0))))
        return max(energy_db, log_floor_db)
    return rms


# ---------------------------------------------------------------------------
# Adaptive Noise Floor Estimator
# ---------------------------------------------------------------------------

class NoiseFloorEstimator:
    """
    Exponentially weighted minimum noise floor estimator.

    Tracks the background noise level by maintaining a slow-attack,
    fast-release estimate of the minimum observed energy.

    Parameters
    ----------
    attack_alpha : float
        Update rate when energy < current floor (decreasing floor). Default 0.05.
    release_alpha : float
        Update rate when energy > current floor (increasing floor). Default 0.002.
    initial_floor_db : float
        Initial noise floor estimate in dB. Default -50.0.
    """

    def __init__(
        self,
        attack_alpha: float = 0.08,
        release_alpha: float = 0.015,
        initial_floor_db: float = -50.0,
        warmup_frames: int = 15,
    ) -> None:
        self.attack_alpha = float(attack_alpha)
        self.release_alpha = float(release_alpha)
        self.default_initial_floor_db = float(initial_floor_db)
        self.warmup_frames = int(warmup_frames)
        self._floor_db: float = float(initial_floor_db)
        self._initialized: bool = False
        self._frame_count: int = 0
        self._speech_dwell: int = 0

    def update(
        self,
        energy_db: float,
        speech_active: bool = False,
        force: bool = False,
    ) -> float:
        """
        Update the noise floor estimate with a new energy observation.

        Returns the current noise floor estimate.
        """
        self._frame_count += 1

        if not self._initialized:
            self._floor_db = min(self.default_initial_floor_db, float(energy_db))
            self._initialized = True
            return self._floor_db

        # Warmup phase: lock directly to the lowest observed energy in the initial window
        if self._frame_count <= self.warmup_frames:
            self._floor_db = min(self._floor_db, float(energy_db))
            return self._floor_db

        if energy_db < self._floor_db:
            # Energy decreasing: floor tracks down rapidly
            alpha = self.attack_alpha
            self._speech_dwell = 0
        else:
            if speech_active:
                self._speech_dwell += 1
                # If speech is latched continuously for > 50 frames (400ms),
                # increase tracking rate so sustained stationary noise (fan, siren) cannot latch VAD permanently.
                if self._speech_dwell > 50:
                    alpha = self.release_alpha * 2.0
                else:
                    alpha = self.release_alpha * 0.2
            else:
                self._speech_dwell = 0
                alpha = self.release_alpha if not force else self.attack_alpha

        self._floor_db = (1.0 - alpha) * self._floor_db + alpha * energy_db
        return self._floor_db

    @property
    def floor_db(self) -> float:
        return self._floor_db

    def reset(self, initial_floor_db: float = -50.0) -> None:
        self._floor_db = float(initial_floor_db)
        self._initialized = False
        self._frame_count = 0
        self._speech_dwell = 0


# ---------------------------------------------------------------------------
# Main VAD
# ---------------------------------------------------------------------------

class VoiceActivityDetector:
    """
    Temporal smoothing VAD with hysteresis and hangover.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz. Default 16000.
    frame_size_ms : float
        Analysis frame size in milliseconds. Default 20.0.
    smoothing_frames : int
        Number of frames over which to smooth the speech probability.
        Default 5.
    onset_threshold_db : float
        dB above noise floor to trigger speech onset. Default 12.0.
    offset_threshold_db : float
        dB above noise floor to trigger speech offset. Default 6.0.
    hangover_frames : int
        Minimum number of speech-active frames before declaring silence.
        Prevents premature cutoff of trailing fricatives. Default 8.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size_ms: float = 20.0,
        smoothing_frames: int = 5,
        onset_threshold_db: float = 12.0,
        offset_threshold_db: float = 6.0,
        hangover_frames: int = 8,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_size_samples = int(round(sample_rate * frame_size_ms / 1000.0))
        self.smoothing_frames = int(smoothing_frames)
        self.onset_threshold_db = float(onset_threshold_db)
        self.offset_threshold_db = float(offset_threshold_db)
        self.hangover_frames = int(hangover_frames)

        self._noise_floor = NoiseFloorEstimator()
        self._energy_history: list = []     # recent energy values for smoothing
        self._speech_active: bool = False   # current VAD state
        self._hangover_count: int = 0       # frames remaining in hangover

        # Speech probability (smoothed), in [0, 1]
        self._speech_probability: float = 0.0

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one audio frame and update VAD state.

        Parameters
        ----------
        frame : np.ndarray
            1-D audio samples for this frame.

        Returns
        -------
        dict with:
            speech_active     : bool    — is speech currently active?
            speech_probability: float   — smooth probability in [0,1]
            frame_energy_db   : float   — raw frame energy in dB
            noise_floor_db    : float   — current noise floor estimate in dB
            snr_estimate_db   : float   — energy - noise floor in dB
        """
        frame = np.asarray(frame, dtype=np.float64).ravel()

        # 1. Frame energy
        energy_db = compute_frame_energy(frame, use_log=True)

        # 2. Update noise floor
        # Always update downward if energy < floor.
        # If speech is active, allow slow upward tracking so sustained continuous noise (fan, siren) cannot latch VAD.
        noise_floor_db = self._noise_floor.update(energy_db, speech_active=self._speech_active)

        # 3. Compute instantaneous SNR
        snr_db = energy_db - noise_floor_db

        # 4. Raw speech decision with hysteresis
        if not self._speech_active:
            # Use higher onset threshold to enter speech state
            raw_speech = snr_db >= self.onset_threshold_db
        else:
            # Use lower offset threshold to exit speech state
            raw_speech = snr_db >= self.offset_threshold_db

        raw_prob = 1.0 if raw_speech else 0.0

        # 5. Energy history for smoothing
        self._energy_history.append(float(snr_db))
        if len(self._energy_history) > self.smoothing_frames:
            self._energy_history.pop(0)

        # 6. Smooth probability: fraction of recent frames above onset threshold
        above_onset = sum(1 for e in self._energy_history if e >= self.onset_threshold_db)
        smooth_prob = float(above_onset) / max(len(self._energy_history), 1)

        # 7. Hangover: once speech detected, maintain for at least hangover_frames
        if raw_speech:
            self._speech_active = True
            self._hangover_count = self.hangover_frames
        elif self._hangover_count > 0:
            self._hangover_count -= 1
            self._speech_active = True  # still in hangover
        else:
            self._speech_active = False

        self._speech_probability = smooth_prob

        return {
            "speech_active": self._speech_active,
            "speech_probability": smooth_prob,
            "raw_vad_probability": raw_prob,
            "smoothed_vad_probability": smooth_prob,
            "frame_energy_db": energy_db,
            "noise_floor_db": noise_floor_db,
            "snr_estimate_db": snr_db,
        }

    def process_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """
        Process an entire waveform and return per-sample speech mask in [0,1].

        The output mask can be used as a speech protection weight.

        Parameters
        ----------
        waveform : np.ndarray, shape (N,)
            Audio samples.

        Returns
        -------
        np.ndarray, shape (N,)
            Speech probability mask per sample, in [0, 1].
        """
        waveform = np.asarray(waveform, dtype=np.float64).ravel()
        N = len(waveform)
        mask = np.zeros(N, dtype=np.float32)

        hop = self.frame_size_samples // 2  # 50% overlap
        pos = 0
        frame_probs = []

        while pos < N:
            end = min(pos + self.frame_size_samples, N)
            frame = waveform[pos:end]
            result = self.process_frame(frame)
            prob = result["speech_probability"]
            frame_probs.append((pos, end, prob))
            pos += hop

        # Interpolate probabilities back to sample rate
        for (start, end, prob) in frame_probs:
            mask[start:end] = np.maximum(mask[start:end], np.float32(prob))

        return mask

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def speech_probability(self) -> float:
        return self._speech_probability

    def reset(self) -> None:
        """Reset all state."""
        self._noise_floor.reset()
        self._energy_history.clear()
        self._speech_active = False
        self._hangover_count = 0
        self._speech_probability = 0.0
