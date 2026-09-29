"""
src/inference/streaming_pipeline.py
=====================================
True Streaming Inference Pipeline.

Architecture:
    Input: dual-channel audio (M1=primary, M2=reference)
    ↓
    Ring Buffer (M1, M2)
    ↓
    Frame extraction (hop_size samples)
    ↓
    DC removal + preprocessing
    ↓
    GCC-PHAT (relative delay estimation per frame)
    ↓
    Kalman filter (smooth delay estimate)
    ↓
    Exact STFT of M1 frame → complex features (2, 257, T)
        (Using 1.0 / win.sum() scaling matching scipy.signal.stft)
    ↓
    Rolling 7-frame feature buffer [t-3 ... t+3]
    ↓
    AI Model (CRM CNN+GRU) → enhanced STFT (center frame t) + noise class + mask
    ↓
    ISTFT (win.sum() unscaling + WOLA) → AI-enhanced waveform
    ↓
    NLMS (configured by smoothed noise class, state preserved) → DSP-enhanced waveform
    ↓
    VAD (speech probability with continuous tracking)
    ↓
    Adaptive Fusion (AI + DSP + configurable speech protection)
    ↓
    Safety Limiter (configurable / neutral in evaluation modes)
    ↓
    Output ring buffer

Pipeline Modes:
    - "ai_only": AI model ON, NLMS OFF, fusion OFF, VAD floor OFF, limiter OFF.
    - "ai_plus_nlms": AI ON, NLMS ON, configurable fusion weight, VAD floor OFF, limiter OFF.
    - "full_production": All production components active with corrected temporal context,
      exact STFT scaling, stabilized classifier, and continuous VAD.
    - "oracle": Clean speech bypasses AI model to isolate downstream DSP degradation.

STFT parameters (locked to project specification):
    sample_rate = 16000
    n_fft       = 512
    hop_size    = 128
    window_size = 512
    window      = Hann
    Output      = (257, T) complex

Algorithmic Latency:
    - WOLA synthesis delay = (n_fft - hop_size) = 384 samples (24.0 ms)
    - Future lookahead delay = 3 hops = 384 samples (24.0 ms)
    - Total AI streaming latency = 768 samples (48.0 ms = 6 hops)
    - DSP and primary mic delay FIFOs are primed with 768 samples for exact phase alignment.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch

from src.dsp.gcc_phat import gcc_phat
from src.dsp.kalman import DelayKalmanFilter
from src.dsp.fusion import AdaptiveFusionEngine
from src.dsp.limiter import SafetyLimiter
from src.dsp.nlms import AdaptiveNLMSController
from src.dsp.vad import VoiceActivityDetector
from src.features.stft import transform_frame, inverse_frame
from src.inference.ai_inference import AIInferenceWrapper
from src.realtime.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STFT parameters (locked)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
N_FFT = 512
HOP_SIZE = 128
WINDOW_SIZE = 512
N_FREQ_BINS = 257   # N_FFT//2 + 1
CONTEXT_FRAMES = 7  # [t-3, t-2, t-1, t, t+1, t+2, t+3]
LOOKAHEAD_HOPS = 3  # 3 future frames required for center frame t


# ---------------------------------------------------------------------------
# Streaming Pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """
    True streaming dual-microphone noise cancellation pipeline with
    temporally consistent rolling feature buffer and exact training-STFT scaling.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the AI model checkpoint.
    device : str
        Device for AI inference ('cpu', 'cuda', 'auto').
    pipeline_mode : str
        One of 'ai_only', 'ai_plus_nlms', 'full_production', 'oracle'.
    enable_ai : Optional[bool]
        Explicit override for AI enhancement toggle.
    enable_nlms : Optional[bool]
        Explicit override for NLMS DSP toggle.
    enable_vad : Optional[bool]
        Explicit override for VAD processing toggle.
    enable_fusion : Optional[bool]
        Explicit override for fusion engine toggle.
    enable_limiter : Optional[bool]
        Explicit override for safety limiter toggle.
    nlms_weight : Optional[float]
        Fixed NLMS weight for fusion (e.g. 0.0 for AI-only, 0.2 for AI+NLMS).
    speech_protection_enabled : Optional[bool]
        Enable speech protection attenuation floor in fusion.
    classifier_window_frames : int
        Rolling GRU frames for sequence mean-pooled classifier (default 15).
    classifier_smoothing_alpha : float
        Exponential smoothing factor for class probabilities (default 0.85).
    classifier_hysteresis_threshold : float
        Minimum probability threshold to switch noise class (default 0.60).
    classifier_min_dwell_frames : int
        Minimum consecutive frames required to commit a class switch (default 8).
    buffer_size_s : float
        Ring buffer size in seconds. Default 5.0.
    debug : bool
        If True, record per-hop diagnostics.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        pipeline_mode: str = "full_production",
        enable_ai: Optional[bool] = None,
        enable_nlms: Optional[bool] = None,
        enable_vad: Optional[bool] = None,
        enable_fusion: Optional[bool] = None,
        enable_limiter: Optional[bool] = None,
        nlms_weight: Optional[float] = None,
        speech_protection_enabled: Optional[bool] = None,
        classifier_window_frames: int = 15,
        classifier_smoothing_alpha: float = 0.85,
        classifier_hysteresis_threshold: float = 0.60,
        classifier_min_dwell_frames: int = 8,
        buffer_size_s: float = 5.0,
        debug: bool = False,
    ) -> None:
        self.pipeline_mode = pipeline_mode.lower()
        self.debug = debug

        # Apply pipeline mode defaults if explicit overrides not given
        if self.pipeline_mode == "ai_only":
            self.enable_ai = True if enable_ai is None else bool(enable_ai)
            self.enable_nlms = False if enable_nlms is None else bool(enable_nlms)
            self.enable_vad = False if enable_vad is None else bool(enable_vad)
            self.enable_fusion = False if enable_fusion is None else bool(enable_fusion)
            self.enable_limiter = False if enable_limiter is None else bool(enable_limiter)
            self.speech_protection_enabled = False if speech_protection_enabled is None else bool(speech_protection_enabled)
            self.nlms_weight = 0.0 if nlms_weight is None else float(nlms_weight)
        elif self.pipeline_mode == "ai_plus_nlms":
            self.enable_ai = True if enable_ai is None else bool(enable_ai)
            self.enable_nlms = True if enable_nlms is None else bool(enable_nlms)
            self.enable_vad = False if enable_vad is None else bool(enable_vad)
            self.enable_fusion = True if enable_fusion is None else bool(enable_fusion)
            self.enable_limiter = False if enable_limiter is None else bool(enable_limiter)
            self.speech_protection_enabled = False if speech_protection_enabled is None else bool(speech_protection_enabled)
            self.nlms_weight = 0.2 if nlms_weight is None else float(nlms_weight)
        elif self.pipeline_mode == "oracle":
            self.enable_ai = False  # Bypassed with oracle clean signal
            self.enable_nlms = True if enable_nlms is None else bool(enable_nlms)
            self.enable_vad = True if enable_vad is None else bool(enable_vad)
            self.enable_fusion = True if enable_fusion is None else bool(enable_fusion)
            self.enable_limiter = False if enable_limiter is None else bool(enable_limiter)
            self.speech_protection_enabled = False if speech_protection_enabled is None else bool(speech_protection_enabled)
            self.nlms_weight = nlms_weight
        else:  # "full_production"
            self.enable_ai = True if enable_ai is None else bool(enable_ai)
            self.enable_nlms = True if enable_nlms is None else bool(enable_nlms)
            self.enable_vad = True if enable_vad is None else bool(enable_vad)
            self.enable_fusion = True if enable_fusion is None else bool(enable_fusion)
            self.enable_limiter = True if enable_limiter is None else bool(enable_limiter)
            self.speech_protection_enabled = True if speech_protection_enabled is None else bool(speech_protection_enabled)
            self.nlms_weight = nlms_weight

        buffer_capacity = int(buffer_size_s * SAMPLE_RATE)

        # --- Input ring buffers ---
        self._m1_buffer = RingBuffer(buffer_capacity)
        self._m2_buffer = RingBuffer(buffer_capacity)
        self._oracle_clean_buffer = RingBuffer(buffer_capacity)

        # --- Output ring buffer ---
        self._output_buffer = RingBuffer(buffer_capacity)

        # --- Overlap-add buffers & Normalization (Stage 1 WOLA) ---
        self._ola_buffer = np.zeros(N_FFT, dtype=np.float32)
        self._m1_frame_buffer = np.zeros(N_FFT, dtype=np.float32)
        self._m2_frame_buffer = np.zeros(N_FFT, dtype=np.float32)

        # STFT analysis and synthesis windows (Hann)
        self._window = np.hanning(N_FFT).astype(np.float32)
        self._synthesis_window = np.hanning(N_FFT).astype(np.float32)
        self._win_sum = float(np.sum(self._window))
        self._win_scale = 1.0 / self._win_sum

        # Precompute exact point-by-point COLA normalization vector across the 128-sample hop
        self._ola_norm_hop = np.zeros(HOP_SIZE, dtype=np.float32)
        for i in range(HOP_SIZE):
            s = 0.0
            for k in range(N_FFT // HOP_SIZE):
                s += self._window[i + k * HOP_SIZE] * self._synthesis_window[i + k * HOP_SIZE]
            self._ola_norm_hop[i] = s

        # Algorithmic Latency calculation:
        # WOLA synthesis delay = N_FFT - HOP_SIZE = 384 samples (24 ms)
        # AI Lookahead delay = LOOKAHEAD_HOPS * HOP_SIZE = 384 samples (24 ms) when AI is enabled
        self._lookahead_hops = LOOKAHEAD_HOPS if (self.enable_ai or self.pipeline_mode == "oracle") else 0
        self._stft_group_delay = (N_FFT - HOP_SIZE) + (self._lookahead_hops * HOP_SIZE)

        # Delay FIFOs to phase-align DSP and primary signals with AI WOLA output
        self._dsp_delay_fifo = RingBuffer(4096)
        self._primary_delay_fifo = RingBuffer(4096)
        self._oracle_clean_fifo = RingBuffer(4096)
        self._dsp_delay_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))
        self._primary_delay_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))
        self._oracle_clean_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))

        # Reference microphone delay line (GCC-PHAT alignment)
        self._m2_delay_line = np.zeros(1024, dtype=np.float32)
        self._aligned_delay_samples = 0.0

        # Stateful recurrent inference mode
        self.stateful_ai: bool = True

        # --- DSP components ---
        self._kalman = DelayKalmanFilter(
            initial_delay=0.0,
            initial_covariance=1.0,
            process_noise=0.01,
            measurement_noise=1.0,
        )
        self._nlms = AdaptiveNLMSController()
        self._vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
        self._fusion = AdaptiveFusionEngine(
            sample_rate=SAMPLE_RATE,
            fixed_nlms_weight=self.nlms_weight,
            enable_speech_protection=self.speech_protection_enabled,
        )
        self._limiter = SafetyLimiter(sample_rate=SAMPLE_RATE, enabled=self.enable_limiter)

        # --- AI model wrapper ---
        if self.enable_ai:
            self._ai = AIInferenceWrapper.from_checkpoint(
                checkpoint_path,
                device=device,
                classifier_window_frames=classifier_window_frames,
                classifier_smoothing_alpha=classifier_smoothing_alpha,
                classifier_hysteresis_threshold=classifier_hysteresis_threshold,
                classifier_min_dwell_frames=classifier_min_dwell_frames,
            )
        else:
            self._ai = None

        # --- Rolling 7-frame feature buffer [t-3 ... t+3] ---
        # Prime with 3 zero frames at startup to represent t < 0 boundary condition
        self._feature_buffer: List[np.ndarray] = [
            np.zeros((2, N_FREQ_BINS), dtype=np.float32) for _ in range(LOOKAHEAD_HOPS)
        ]

        # --- State ---
        self._current_noise_class: int = 0
        self._current_confidence: float = 0.5
        self._current_probs: np.ndarray = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        self._current_smoothed_probs: np.ndarray = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        self._current_speech_prob: float = 0.0
        self._current_smoothed_speech_prob: float = 0.0
        self._current_delay_samples: float = 0.0
        self._last_ai_rms: float = 0.0
        self._last_nlms_rms: float = 0.0
        self._last_primary_rms: float = 0.0
        self._last_fused_rms: float = 0.0
        self._last_fusion_weights: Dict[str, float] = {"ai_weight": 1.0, "dsp_weight": 0.0}

        # --- Feature statistics tracking ---
        self._feat_sum: float = 0.0
        self._feat_sq_sum: float = 0.0
        self._feat_min: float = float("inf")
        self._feat_max: float = float("-inf")
        self._feat_count: int = 0

        # --- Performance metrics ---
        self._n_frames_processed: int = 0
        self._total_processing_time_s: float = 0.0
        self._total_ai_time_s: float = 0.0
        self._total_dsp_time_s: float = 0.0

        logger.info(
            "StreamingPipeline initialized: mode=%s, AI=%s, NLMS=%s, VAD=%s, fusion=%s, limiter=%s, latency=%d samples (%.1f ms)",
            self.pipeline_mode, self.enable_ai, self.enable_nlms, self.enable_vad,
            self.enable_fusion, self.enable_limiter, self._stft_group_delay,
            1000.0 * self._stft_group_delay / SAMPLE_RATE,
        )

    def push(
        self,
        m1_samples: np.ndarray,
        m2_samples: np.ndarray,
        clean_samples: Optional[np.ndarray] = None,
    ) -> None:
        """
        Push new audio samples into the pipeline input buffers.

        Parameters
        ----------
        m1_samples : np.ndarray
            Primary microphone samples (float32).
        m2_samples : np.ndarray
            Reference microphone samples (float32).
        clean_samples : np.ndarray, optional
            Clean speech samples for oracle mode or objective alignment.
        """
        m1 = np.asarray(m1_samples, dtype=np.float32).ravel()
        m2 = np.asarray(m2_samples, dtype=np.float32).ravel()
        self._m1_buffer.write(m1)
        self._m2_buffer.write(m2)
        if clean_samples is not None:
            c = np.asarray(clean_samples, dtype=np.float32).ravel()
            self._oracle_clean_buffer.write(c)

    def process_available(self) -> int:
        """
        Process all available complete frames from the input buffers.

        Returns the number of output samples produced.
        """
        n_produced = 0
        while self._m1_buffer.available >= HOP_SIZE and self._m2_buffer.available >= HOP_SIZE:
            n_produced += self._process_one_hop()
        return n_produced

    def read_output(self, n_samples: int) -> np.ndarray:
        """
        Read processed output samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to read.

        Returns
        -------
        np.ndarray, shape (k,) where k <= n_samples
        """
        return self._output_buffer.read(n_samples)

    def _process_one_hop(self) -> int:
        """
        Process one hop of audio through the full pipeline.

        Returns the number of output samples produced (= HOP_SIZE).
        """
        t_start = time.perf_counter()

        # 1. Read one hop from input buffers
        m1_hop = self._m1_buffer.read(HOP_SIZE)
        m2_hop = self._m2_buffer.read(HOP_SIZE)

        if len(m1_hop) < HOP_SIZE or len(m2_hop) < HOP_SIZE:
            return 0

        # Read optional clean speech for oracle bypass
        clean_hop = (
            self._oracle_clean_buffer.read(HOP_SIZE)
            if self._oracle_clean_buffer.available >= HOP_SIZE
            else None
        )

        self._last_primary_rms = float(np.sqrt(np.mean(m1_hop**2) + 1e-12))

        # 2. Update analysis frame buffers (shift + append new hop)
        self._m1_frame_buffer = np.roll(self._m1_frame_buffer, -HOP_SIZE)
        self._m1_frame_buffer[-HOP_SIZE:] = m1_hop

        self._m2_frame_buffer = np.roll(self._m2_frame_buffer, -HOP_SIZE)
        self._m2_frame_buffer[-HOP_SIZE:] = m2_hop

        # 3. DC removal on analysis frame
        m1_frame = self._m1_frame_buffer - np.mean(self._m1_frame_buffer)
        m2_frame = self._m2_frame_buffer - np.mean(self._m2_frame_buffer)

        # 4. GCC-PHAT delay estimation & Kalman smoothing
        gcc_result = gcc_phat(m1_frame, m2_frame, sample_rate=SAMPLE_RATE, max_tau=50)
        raw_delay = gcc_result["estimated_delay_samples"]
        smoothed_delay = self._kalman.update(raw_delay)
        self._current_delay_samples = smoothed_delay

        # Slew-rate limit reference delay to prevent clicks
        delay_target = float(np.clip(smoothed_delay, -40.0, 40.0))
        delta = float(np.clip(delay_target - self._aligned_delay_samples, -1.0, 1.0))
        self._aligned_delay_samples += delta
        int_delay = int(round(self._aligned_delay_samples))

        # 5. Reference mic delay alignment
        self._m2_delay_line = np.roll(self._m2_delay_line, -HOP_SIZE)
        self._m2_delay_line[-HOP_SIZE:] = m2_hop
        base_offset = 64
        read_idx = base_offset + int_delay
        start_idx = len(self._m2_delay_line) - HOP_SIZE - read_idx
        start_idx = max(0, min(start_idx, len(self._m2_delay_line) - HOP_SIZE))
        m2_hop_aligned = self._m2_delay_line[start_idx : start_idx + HOP_SIZE].copy()

        # 6. VAD (Voice Activity Detection on M1 analysis frame)
        if self.enable_vad:
            vad_result = self._vad.process_frame(m1_frame)
            speech_prob = vad_result["speech_probability"]
            smoothed_prob = vad_result.get("smoothed_vad_probability", speech_prob)
            self._current_speech_prob = speech_prob
            self._current_smoothed_speech_prob = smoothed_prob
        else:
            speech_prob = 0.0
            smoothed_prob = 0.0
            self._current_speech_prob = 0.0
            self._current_smoothed_speech_prob = 0.0

        # 7. NLMS DSP processing (Stage 3 — Hop-based adaptation with persistent state)
        t_dsp_start = time.perf_counter()
        if self.enable_nlms:
            self._nlms.update_noise_class(self._current_noise_class)
            try:
                dsp_error_hop, _ = self._nlms.process_hop(
                    m1_hop, m2_hop_aligned, speech_prob=self._current_smoothed_speech_prob
                )
                dsp_enhanced_hop = dsp_error_hop.astype(np.float32)
            except Exception as e:
                logger.warning("NLMS processing failed: %s — using pass-through", e)
                dsp_enhanced_hop = m1_hop.copy()
        else:
            dsp_enhanced_hop = m1_hop.copy()
        t_dsp_end = time.perf_counter()
        self._total_dsp_time_s += (t_dsp_end - t_dsp_start)
        self._last_nlms_rms = float(np.sqrt(np.mean(dsp_enhanced_hop**2) + 1e-12))

        # 8. Exact STFT Analysis of M1 frame (matches training scipy.signal.stft scaling)
        m1_windowed = m1_frame * self._window
        stft_complex = np.fft.rfft(m1_windowed, n=N_FFT) * self._win_scale  # (257,) scaled complex
        stft_real = stft_complex.real.astype(np.float32)
        stft_imag = stft_complex.imag.astype(np.float32)
        cur_feature = np.stack([stft_real, stft_imag], axis=0)  # (2, 257)

        # Update feature statistics
        feat_n = cur_feature.size
        self._feat_sum += float(np.sum(cur_feature))
        self._feat_sq_sum += float(np.sum(cur_feature**2))
        self._feat_min = min(self._feat_min, float(np.min(cur_feature)))
        self._feat_max = max(self._feat_max, float(np.max(cur_feature)))
        self._feat_count += feat_n

        # 9. AI Inference with Rolling 7-Frame Buffer [t-3 ... t+3]
        t_ai_start = time.perf_counter()
        if self.enable_ai and self._ai is not None:
            # Append current frame to rolling feature buffer
            self._feature_buffer.append(cur_feature)

            if len(self._feature_buffer) < CONTEXT_FRAMES:
                # Startup phase: accumulating future lookahead frames
                enhanced_complex = np.zeros(N_FREQ_BINS, dtype=np.complex64)
            else:
                # Normal steady-state: feed exactly 7 frames [t-3 ... t+3]
                context_features = np.stack(self._feature_buffer[:CONTEXT_FRAMES], axis=-1)  # (2, 257, 7)
                try:
                    enhanced_stft_np, logits, mask = self._ai.infer(
                        context_features, stateful=self.stateful_ai
                    )
                    # Smoothed classification with dwell hysteresis
                    noise_class, conf, probs, smoothed_probs, _ = self._ai.classify_smoothed(logits)
                    self._current_noise_class = noise_class
                    self._current_confidence = conf
                    self._current_probs = probs
                    self._current_smoothed_probs = smoothed_probs

                    # Extract enhanced complex STFT for center frame t
                    enhanced_complex = (
                        enhanced_stft_np[0, :, 0] + 1j * enhanced_stft_np[1, :, 0]
                    )
                except Exception as e:
                    logger.warning("AI inference failed: %s — using pass-through", e)
                    # Center frame noisy STFT
                    noisy_center = self._feature_buffer[LOOKAHEAD_HOPS]
                    enhanced_complex = noisy_center[0] + 1j * noisy_center[1]

                # Slide window by popping the oldest past frame
                self._feature_buffer.pop(0)

        elif self.pipeline_mode == "oracle" and clean_hop is not None:
            # Oracle Mode: bypass AI model directly with clean speech
            enhanced_complex = None
        else:
            # AI disabled: passthrough STFT
            enhanced_complex = stft_complex.copy()

        t_ai_end = time.perf_counter()
        self._total_ai_time_s += (t_ai_end - t_ai_start)

        # 10. iSTFT Synthesis + Mathematically Correct Overlap-Add (Stage 1 WOLA)
        if self.pipeline_mode == "oracle" and clean_hop is not None:
            # Write clean speech to oracle delay FIFO to align with DSP and primary
            self._oracle_clean_fifo.write(clean_hop)
            ai_enhanced_hop = self._oracle_clean_fifo.read(HOP_SIZE)
        else:
            # Unscale STFT by win_sum before inverse FFT
            raw_rec = np.fft.irfft(enhanced_complex * self._win_sum, n=N_FFT)[:N_FFT]
            synth_frame = (raw_rec * self._synthesis_window).astype(np.float32)
            self._ola_buffer += synth_frame
            ai_enhanced_hop = (self._ola_buffer[:HOP_SIZE] / self._ola_norm_hop).astype(np.float32)
            self._ola_buffer[:-HOP_SIZE] = self._ola_buffer[HOP_SIZE:]
            self._ola_buffer[-HOP_SIZE:] = 0.0

        self._last_ai_rms = float(np.sqrt(np.mean(ai_enhanced_hop**2) + 1e-12))

        # 11. Phase-Aligned Delay FIFO & Fusion (Stage 4)
        # Delay DSP and primary hops by exact algorithmic latency (768 samples) to match AI OLA phase
        self._dsp_delay_fifo.write(dsp_enhanced_hop)
        self._primary_delay_fifo.write(m1_hop)

        aligned_dsp_hop = self._dsp_delay_fifo.read(HOP_SIZE)
        aligned_primary_hop = self._primary_delay_fifo.read(HOP_SIZE)

        # 12. Fusion Routing
        if self.pipeline_mode == "ai_only" or not self.enable_fusion:
            output_hop = ai_enhanced_hop
            self._last_fusion_weights = {"ai_weight": 1.0, "dsp_weight": 0.0}
        else:
            try:
                fusion_result = self._fusion.fuse(
                    ai_enhanced=ai_enhanced_hop,
                    dsp_enhanced=aligned_dsp_hop,
                    original_primary=aligned_primary_hop,
                    noise_class=self._current_noise_class,
                    confidence=self._current_confidence,
                    speech_probability=self._current_smoothed_speech_prob,
                )
                output_hop = fusion_result["fused_signal"]
                self._last_fusion_weights = {
                    "ai_weight": float(fusion_result.get("ai_weight", 1.0 - (self.nlms_weight or 0.0))),
                    "dsp_weight": float(fusion_result.get("dsp_weight", self.nlms_weight or 0.0)),
                }
            except Exception as e:
                logger.warning("Fusion failed: %s — using AI output", e)
                output_hop = ai_enhanced_hop
                self._last_fusion_weights = {"ai_weight": 1.0, "dsp_weight": 0.0}

        self._last_fused_rms = float(np.sqrt(np.mean(output_hop**2) + 1e-12))

        # 13. Safety Limiter
        if self.enable_limiter:
            output_hop = self._limiter.process(output_hop)

        # 14. Write to output buffer
        self._output_buffer.write(output_hop)

        t_end = time.perf_counter()
        self._total_processing_time_s += (t_end - t_start)
        self._n_frames_processed += 1

        return HOP_SIZE

    def reset(self) -> None:
        """Reset all pipeline state."""
        self._m1_buffer.reset()
        self._m2_buffer.reset()
        self._oracle_clean_buffer.reset()
        self._output_buffer.reset()
        self._ola_buffer[:] = 0.0
        self._m1_frame_buffer[:] = 0.0
        self._m2_frame_buffer[:] = 0.0
        self._m2_delay_line[:] = 0.0
        self._aligned_delay_samples = 0.0
        self._dsp_delay_fifo.reset()
        self._primary_delay_fifo.reset()
        self._oracle_clean_fifo.reset()
        self._dsp_delay_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))
        self._primary_delay_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))
        self._oracle_clean_fifo.write(np.zeros(self._stft_group_delay, dtype=np.float32))

        # Reset rolling feature buffer to initial startup padding
        self._feature_buffer = [
            np.zeros((2, N_FREQ_BINS), dtype=np.float32) for _ in range(LOOKAHEAD_HOPS)
        ]

        if self._ai is not None:
            self._ai.reset_state()
        self._kalman.reset()
        self._nlms.reset()
        self._vad.reset()
        self._fusion.reset()
        self._limiter.reset()

        self._current_noise_class = 0
        self._current_confidence = 0.5
        self._current_probs = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        self._current_smoothed_probs = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        self._current_speech_prob = 0.0
        self._current_smoothed_speech_prob = 0.0
        self._current_delay_samples = 0.0
        self._last_ai_rms = 0.0
        self._last_nlms_rms = 0.0
        self._last_primary_rms = 0.0
        self._last_fused_rms = 0.0
        self._last_fusion_weights = {"ai_weight": 1.0, "dsp_weight": 0.0}

        self._feat_sum = 0.0
        self._feat_sq_sum = 0.0
        self._feat_min = float("inf")
        self._feat_max = float("-inf")
        self._feat_count = 0

        self._n_frames_processed = 0
        self._total_processing_time_s = 0.0
        self._total_ai_time_s = 0.0
        self._total_dsp_time_s = 0.0

    @property
    def algorithmic_delay_samples(self) -> int:
        """Return total algorithmic latency in samples."""
        return self._stft_group_delay

    @property
    def algorithmic_delay_ms(self) -> float:
        """Return total algorithmic latency in milliseconds."""
        return 1000.0 * self._stft_group_delay / SAMPLE_RATE

    @property
    def diagnostics(self) -> dict:
        """Return comprehensive pipeline performance and state diagnostics."""
        avg_frame_time_ms = (
            1000.0 * self._total_processing_time_s / self._n_frames_processed
            if self._n_frames_processed > 0 else 0.0
        )
        hop_duration_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE
        total_audio_time_s = self._n_frames_processed * HOP_SIZE / SAMPLE_RATE
        real_time_factor = (
            self._total_processing_time_s / total_audio_time_s
            if total_audio_time_s > 0 else 0.0
        )

        # Feature stats
        if self._feat_count > 0:
            feat_mean = self._feat_sum / self._feat_count
            feat_var = max(0.0, (self._feat_sq_sum / self._feat_count) - (feat_mean**2))
            feat_std = float(np.sqrt(feat_var))
            feat_min = self._feat_min
            feat_max = self._feat_max
        else:
            feat_mean = 0.0
            feat_std = 0.0
            feat_min = 0.0
            feat_max = 0.0

        class_switch_count = self._ai.switch_count if self._ai is not None else 0
        switch_freq_hz = (
            class_switch_count / total_audio_time_s if total_audio_time_s > 0 else 0.0
        )

        return {
            "pipeline_mode": self.pipeline_mode,
            "n_frames_processed": self._n_frames_processed,
            "frame_size_samples": N_FFT,
            "hop_size_samples": HOP_SIZE,
            "sample_rate": SAMPLE_RATE,
            "hop_duration_ms": hop_duration_ms,
            "avg_frame_processing_ms": round(avg_frame_time_ms, 2),
            "real_time_factor": round(real_time_factor, 3),
            "total_processing_time_s": round(self._total_processing_time_s, 4),
            "total_ai_time_s": round(self._total_ai_time_s, 4),
            "total_dsp_time_s": round(self._total_dsp_time_s, 4),
            "algorithmic_latency_samples": self._stft_group_delay,
            "algorithmic_latency_ms": round(self.algorithmic_delay_ms, 2),
            "temporal_context_shape": (2, N_FREQ_BINS, CONTEXT_FRAMES) if self.enable_ai else (2, N_FREQ_BINS, 1),
            "current_noise_class": self._current_noise_class,
            "current_confidence": round(self._current_confidence, 4),
            "classifier_probabilities": [round(float(p), 4) for p in self._current_probs],
            "smoothed_probabilities": [round(float(p), 4) for p in self._current_smoothed_probs],
            "class_switch_count": class_switch_count,
            "class_switch_frequency_hz": round(switch_freq_hz, 4),
            "current_speech_prob": round(self._current_speech_prob, 4),
            "smoothed_speech_prob": round(self._current_smoothed_speech_prob, 4),
            "speech_protection_enabled": self.speech_protection_enabled,
            "protection_floor_state": getattr(self._fusion, "protection_state", False),
            "applied_attenuation_floor_db": getattr(self._fusion, "applied_attenuation_floor_db", 0.0),
            "limiter_enabled": self.enable_limiter,
            "limiter_clipping_count": self._limiter.clipping_count if self.enable_limiter else 0,
            "current_delay_samples": round(self._current_delay_samples, 3),
            "feature_mean": round(feat_mean, 6),
            "feature_std": round(feat_std, 6),
            "feature_min": round(feat_min, 6),
            "feature_max": round(feat_max, 6),
            "primary_input_rms": round(self._last_primary_rms, 6),
            "ai_output_rms": round(self._last_ai_rms, 6),
            "nlms_output_rms": round(self._last_nlms_rms, 6),
            "fused_output_rms": round(self._last_fused_rms, 6),
            "fusion_weights": self._last_fusion_weights,
            "m1_buffer_available": self._m1_buffer.available,
            "m2_buffer_available": self._m2_buffer.available,
            "output_buffer_available": self._output_buffer.available,
        }


# ---------------------------------------------------------------------------
# Convenience: process a complete WAV through streaming pipeline
# ---------------------------------------------------------------------------

def process_file_streaming(
    checkpoint_path: str | Path,
    m1_waveform: np.ndarray,
    m2_waveform: np.ndarray,
    device: str = "cpu",
    pipeline_mode: str = "full_production",
    enable_ai: Optional[bool] = None,
    enable_nlms: Optional[bool] = None,
    enable_vad: Optional[bool] = None,
    enable_fusion: Optional[bool] = None,
    enable_limiter: Optional[bool] = None,
    nlms_weight: Optional[float] = None,
    speech_protection_enabled: Optional[bool] = None,
    clean_waveform: Optional[np.ndarray] = None,
    chunk_size: int = HOP_SIZE,
) -> Tuple[np.ndarray, dict]:
    """
    Process a complete dual-mic waveform through the streaming pipeline.

    Simulates real-time streaming by feeding audio in chunks.

    Parameters
    ----------
    checkpoint_path : str or Path
        AI model checkpoint.
    m1_waveform : np.ndarray, shape (N,)
        Primary mic waveform (float32).
    m2_waveform : np.ndarray, shape (N,)
        Reference mic waveform (float32).
    device : str
        Inference device.
    pipeline_mode : str
        'ai_only', 'ai_plus_nlms', 'full_production', or 'oracle'.
    clean_waveform : Optional[np.ndarray]
        Clean speech for oracle mode or ground truth objective SNR computation.
    chunk_size : int
        Chunk size to simulate streaming. Default = HOP_SIZE = 128 samples.

    Returns
    -------
    (output_waveform, diagnostics) : (np.ndarray, dict)
    """
    m1 = np.asarray(m1_waveform, dtype=np.float32).ravel()
    m2 = np.asarray(m2_waveform, dtype=np.float32).ravel()
    clean = np.asarray(clean_waveform, dtype=np.float32).ravel() if clean_waveform is not None else None

    pipeline = StreamingPipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        pipeline_mode=pipeline_mode,
        enable_ai=enable_ai,
        enable_nlms=enable_nlms,
        enable_vad=enable_vad,
        enable_fusion=enable_fusion,
        enable_limiter=enable_limiter,
        nlms_weight=nlms_weight,
        speech_protection_enabled=speech_protection_enabled,
        buffer_size_s=max(10.0, len(m1) / SAMPLE_RATE + 5.0),
    )

    n_samples = min(len(m1), len(m2))
    pos = 0
    while pos < n_samples:
        end = min(pos + chunk_size, n_samples)
        c_chunk = clean[pos:end] if clean is not None else None
        pipeline.push(m1[pos:end], m2[pos:end], clean_samples=c_chunk)
        pipeline.process_available()
        pos = end

    # Flush end-of-stream lookahead frames if AI mode is active
    if pipeline.enable_ai:
        silence = np.zeros(HOP_SIZE * LOOKAHEAD_HOPS, dtype=np.float32)
        pipeline.push(silence, silence, clean_samples=silence if clean is not None else None)
        pipeline.process_available()

    # Read output
    output = pipeline.read_output(pipeline._output_buffer.available)
    diagnostics = pipeline.diagnostics

    # If clean speech is provided, compute objective SNR before and after enhancement
    if clean is not None:
        delay = pipeline.algorithmic_delay_samples
        # Primary input SNR
        noise_in = m1[:len(clean)] - clean[:len(m1)]
        p_clean = np.mean(clean[:len(m1)]**2)
        p_noise_in = np.mean(noise_in**2) + 1e-15
        input_snr = 10.0 * np.log10(p_clean / p_noise_in)

        # Output SNR (accounting for algorithmic delay)
        if len(output) > delay:
            clean_aligned = clean[:len(output) - delay]
            out_aligned = output[delay : delay + len(clean_aligned)]
            noise_out = out_aligned - clean_aligned
            p_noise_out = np.mean(noise_out**2) + 1e-15
            p_sig = np.mean(clean_aligned**2)
            output_snr = 10.0 * np.log10(p_sig / p_noise_out)
            snr_improvement = output_snr - input_snr
        else:
            output_snr = 0.0
            snr_improvement = 0.0

        diagnostics["input_snr_db"] = round(float(input_snr), 2)
        diagnostics["output_snr_db"] = round(float(output_snr), 2)
        diagnostics["snr_improvement_db"] = round(float(snr_improvement), 2)

    return output, diagnostics
