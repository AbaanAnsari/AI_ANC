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
    STFT of M1 frame → complex features (2, 257, T)
    ↓
    AI Model (CRM CNN+GRU) → enhanced STFT + noise class + mask
    ↓
    ISTFT → AI-enhanced waveform
    ↓
    NLMS (configured by noise class) → DSP-enhanced waveform
    ↓
    VAD (speech probability from M1 frame)
    ↓
    Adaptive Fusion (AI + DSP + speech protection)
    ↓
    Safety Limiter
    ↓
    Output ring buffer

Processing is incremental — processes hop_size samples at a time with
overlap-add reconstruction.

STFT parameters (locked to project specification):
    sample_rate = 16000
    n_fft       = 512
    hop_size    = 128
    window_size = 512
    window      = Hann
    Output      = (257, T) complex

Streaming:
    - Frame size = n_fft = 512 samples
    - Hop = 128 samples
    - Process one hop at a time
    - Accumulate in output overlap-add buffer
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.dsp.gcc_phat import gcc_phat
from src.dsp.kalman import DelayKalmanFilter
from src.dsp.fusion import AdaptiveFusionEngine
from src.dsp.limiter import SafetyLimiter
from src.dsp.nlms import AdaptiveNLMSController
from src.dsp.vad import VoiceActivityDetector
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

# Minimum frames to accumulate before AI inference
# Model expects at least a few frames; we accumulate 1 second = 126 frames
FRAMES_PER_SEGMENT = 126  # (16000 - 512) / 128 + 1 ≈ 122, we use 126 for padding


# ---------------------------------------------------------------------------
# Streaming Pipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """
    True streaming dual-microphone noise cancellation pipeline.

    Processes audio incrementally via overlapping frames with overlap-add
    ISTFT reconstruction.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to the AI model checkpoint.
    device : str
        Device for AI inference ('cpu', 'cuda', 'auto').
    enable_ai : bool
        Enable AI enhancement. Default True.
    enable_nlms : bool
        Enable NLMS DSP processing. Default True.
    enable_vad : bool
        Enable VAD-based speech protection. Default True.
    enable_fusion : bool
        Enable AI+DSP adaptive fusion. Default True.
    enable_limiter : bool
        Enable safety limiter. Default True.
    buffer_size_s : float
        Ring buffer size in seconds. Default 5.0.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        enable_ai: bool = True,
        enable_nlms: bool = True,
        enable_vad: bool = True,
        enable_fusion: bool = True,
        enable_limiter: bool = True,
        buffer_size_s: float = 5.0,
    ) -> None:
        self.enable_ai = bool(enable_ai)
        self.enable_nlms = bool(enable_nlms)
        self.enable_vad = bool(enable_vad)
        self.enable_fusion = bool(enable_fusion)
        self.enable_limiter = bool(enable_limiter)

        buffer_capacity = int(buffer_size_s * SAMPLE_RATE)

        # --- Input ring buffers ---
        self._m1_buffer = RingBuffer(buffer_capacity)
        self._m2_buffer = RingBuffer(buffer_capacity)

        # --- Output ring buffer ---
        self._output_buffer = RingBuffer(buffer_capacity)

        # --- Overlap-add buffers ---
        self._ola_buffer = np.zeros(N_FFT, dtype=np.float32)
        self._m1_frame_buffer = np.zeros(N_FFT, dtype=np.float32)
        self._m2_frame_buffer = np.zeros(N_FFT, dtype=np.float32)

        # STFT window
        self._window = np.hanning(N_FFT).astype(np.float32)

        # --- DSP components ---
        self._kalman = DelayKalmanFilter(
            initial_delay=0.0,
            initial_covariance=1.0,
            process_noise=0.01,
            measurement_noise=1.0,
        )
        self._nlms = AdaptiveNLMSController()
        self._vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
        self._fusion = AdaptiveFusionEngine(sample_rate=SAMPLE_RATE)
        self._limiter = SafetyLimiter(sample_rate=SAMPLE_RATE)

        # --- AI model ---
        if enable_ai:
            self._ai = AIInferenceWrapper.from_checkpoint(checkpoint_path, device=device)
        else:
            self._ai = None

        # --- State ---
        self._current_noise_class: int = 0
        self._current_confidence: float = 0.5
        self._current_speech_prob: float = 0.0
        self._current_delay_samples: float = 0.0

        # --- STFT accumulator for AI (accumulate frames) ---
        self._stft_frames_real: list = []
        self._stft_frames_imag: list = []
        self._ai_enhanced_frames_real: list = []
        self._ai_enhanced_frames_imag: list = []

        # --- Performance metrics ---
        self._n_frames_processed: int = 0
        self._total_processing_time_s: float = 0.0
        self._total_ai_time_s: float = 0.0
        self._total_dsp_time_s: float = 0.0

        logger.info(
            "StreamingPipeline initialized: AI=%s, NLMS=%s, VAD=%s, fusion=%s, limiter=%s",
            enable_ai, enable_nlms, enable_vad, enable_fusion, enable_limiter,
        )

    def push(self, m1_samples: np.ndarray, m2_samples: np.ndarray) -> None:
        """
        Push new audio samples into the pipeline input buffers.

        Parameters
        ----------
        m1_samples : np.ndarray
            Primary microphone samples (float32).
        m2_samples : np.ndarray
            Reference microphone samples (float32).
        """
        m1 = np.asarray(m1_samples, dtype=np.float32).ravel()
        m2 = np.asarray(m2_samples, dtype=np.float32).ravel()
        self._m1_buffer.write(m1)
        self._m2_buffer.write(m2)

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

        # 2. Update frame buffers (shift + append new hop)
        self._m1_frame_buffer = np.roll(self._m1_frame_buffer, -HOP_SIZE)
        self._m1_frame_buffer[-HOP_SIZE:] = m1_hop

        self._m2_frame_buffer = np.roll(self._m2_frame_buffer, -HOP_SIZE)
        self._m2_frame_buffer[-HOP_SIZE:] = m2_hop

        # 3. DC removal
        m1_frame = self._m1_frame_buffer - np.mean(self._m1_frame_buffer)
        m2_frame = self._m2_frame_buffer - np.mean(self._m2_frame_buffer)

        # 4. GCC-PHAT delay estimation
        gcc_result = gcc_phat(m1_frame, m2_frame, sample_rate=SAMPLE_RATE, max_tau=50)
        raw_delay = gcc_result["estimated_delay_samples"]
        smoothed_delay = self._kalman.update(raw_delay)
        self._current_delay_samples = smoothed_delay

        # 5. STFT of M1 frame
        m1_windowed = m1_frame * self._window
        stft_complex = np.fft.rfft(m1_windowed, n=N_FFT)  # (257,) complex
        stft_real = stft_complex.real.astype(np.float32)
        stft_imag = stft_complex.imag.astype(np.float32)

        # 6. AI inference
        t_ai_start = time.perf_counter()
        ai_enhanced_frame = m1_frame.copy()  # fallback: pass-through

        if self.enable_ai and self._ai is not None:
            # Build feature tensor: (2, 257, 1) — single frame
            features = np.stack([stft_real, stft_imag], axis=0)[:, :, np.newaxis]  # (2,257,1)
            
            try:
                enhanced_stft_np, logits, mask = self._ai.infer(features)
                noise_class, confidence, probs = self._ai.classify(logits)
                self._current_noise_class = noise_class
                self._current_confidence = confidence

                # Reconstruct enhanced frame from enhanced STFT
                enhanced_complex = (
                    enhanced_stft_np[0, :, 0] + 1j * enhanced_stft_np[1, :, 0]
                )
                ai_enhanced_frame_raw = np.fft.irfft(enhanced_complex, n=N_FFT)[:N_FFT]
                ai_enhanced_frame = (ai_enhanced_frame_raw * self._window).astype(np.float32)
            except Exception as e:
                logger.warning("AI inference failed: %s — using pass-through", e)
                ai_enhanced_frame = m1_frame.copy()

        t_ai_end = time.perf_counter()
        self._total_ai_time_s += (t_ai_end - t_ai_start)

        # 7. NLMS DSP processing
        t_dsp_start = time.perf_counter()
        dsp_enhanced_frame = m1_frame.copy()

        if self.enable_nlms:
            self._nlms.update_noise_class(self._current_noise_class)
            try:
                dsp_error, _ = self._nlms.process_block(m1_frame, m2_frame)
                dsp_enhanced_frame = dsp_error.astype(np.float32)
            except Exception as e:
                logger.warning("NLMS processing failed: %s — using pass-through", e)

        t_dsp_end = time.perf_counter()
        self._total_dsp_time_s += (t_dsp_end - t_dsp_start)

        # 8. VAD
        if self.enable_vad:
            vad_result = self._vad.process_frame(m1_frame)
            speech_prob = vad_result["speech_probability"]
            self._current_speech_prob = speech_prob
        else:
            speech_prob = 0.0

        # 9. Fusion
        if self.enable_fusion:
            try:
                fusion_result = self._fusion.fuse(
                    ai_enhanced=ai_enhanced_frame[-HOP_SIZE:],
                    dsp_enhanced=dsp_enhanced_frame[-HOP_SIZE:],
                    original_primary=m1_hop,
                    noise_class=self._current_noise_class,
                    confidence=self._current_confidence,
                    speech_probability=speech_prob,
                )
                output_hop = fusion_result["fused_signal"]
            except Exception as e:
                logger.warning("Fusion failed: %s — using AI output", e)
                output_hop = ai_enhanced_frame[-HOP_SIZE:]
        else:
            output_hop = ai_enhanced_frame[-HOP_SIZE:]

        # 10. Safety limiter
        if self.enable_limiter:
            output_hop = self._limiter.process(output_hop)

        # 11. Write to output buffer
        self._output_buffer.write(output_hop)

        t_end = time.perf_counter()
        self._total_processing_time_s += (t_end - t_start)
        self._n_frames_processed += 1

        return HOP_SIZE

    def reset(self) -> None:
        """Reset all pipeline state."""
        self._m1_buffer.reset()
        self._m2_buffer.reset()
        self._output_buffer.reset()
        self._ola_buffer[:] = 0.0
        self._m1_frame_buffer[:] = 0.0
        self._m2_frame_buffer[:] = 0.0
        self._kalman.reset()
        self._nlms.reset()
        self._vad.reset()
        self._fusion.reset()
        self._limiter.reset()
        self._stft_frames_real.clear()
        self._stft_frames_imag.clear()
        self._current_noise_class = 0
        self._current_confidence = 0.5
        self._current_speech_prob = 0.0
        self._current_delay_samples = 0.0
        self._n_frames_processed = 0
        self._total_processing_time_s = 0.0
        self._total_ai_time_s = 0.0
        self._total_dsp_time_s = 0.0

    @property
    def diagnostics(self) -> dict:
        """Return pipeline performance diagnostics."""
        avg_frame_time_ms = (
            1000.0 * self._total_processing_time_s / self._n_frames_processed
            if self._n_frames_processed > 0 else 0.0
        )
        hop_duration_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE
        real_time_factor = (
            self._total_processing_time_s / 
            (self._n_frames_processed * HOP_SIZE / SAMPLE_RATE)
            if self._n_frames_processed > 0 else 0.0
        )
        return {
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
            "current_noise_class": self._current_noise_class,
            "current_confidence": round(self._current_confidence, 4),
            "current_speech_prob": round(self._current_speech_prob, 4),
            "current_delay_samples": round(self._current_delay_samples, 3),
            "m1_buffer_available": self._m1_buffer.available,
            "m2_buffer_available": self._m2_buffer.available,
            "output_buffer_available": self._output_buffer.available,
            "limiter_clipping_count": self._limiter.clipping_count if self.enable_limiter else 0,
        }


# ---------------------------------------------------------------------------
# Convenience: process a complete WAV through streaming pipeline
# ---------------------------------------------------------------------------

def process_file_streaming(
    checkpoint_path: str | Path,
    m1_waveform: np.ndarray,
    m2_waveform: np.ndarray,
    device: str = "cpu",
    enable_ai: bool = True,
    enable_nlms: bool = True,
    enable_vad: bool = True,
    enable_fusion: bool = True,
    enable_limiter: bool = True,
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
    enable_ai, enable_nlms, enable_vad, enable_fusion, enable_limiter : bool
        Pipeline stage toggles.
    chunk_size : int
        Chunk size to simulate streaming. Default = HOP_SIZE = 128 samples.

    Returns
    -------
    (output_waveform, diagnostics) : (np.ndarray, dict)
    """
    m1 = np.asarray(m1_waveform, dtype=np.float32).ravel()
    m2 = np.asarray(m2_waveform, dtype=np.float32).ravel()

    pipeline = StreamingPipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        enable_ai=enable_ai,
        enable_nlms=enable_nlms,
        enable_vad=enable_vad,
        enable_fusion=enable_fusion,
        enable_limiter=enable_limiter,
        buffer_size_s=max(10.0, len(m1) / SAMPLE_RATE + 5.0),
    )

    # Feed audio in chunks
    n_samples = min(len(m1), len(m2))
    pos = 0
    while pos < n_samples:
        end = min(pos + chunk_size, n_samples)
        pipeline.push(m1[pos:end], m2[pos:end])
        pipeline.process_available()
        pos = end

    # Drain remaining output
    output = pipeline.read_output(pipeline._output_buffer.available)
    diagnostics = pipeline.diagnostics

    return output, diagnostics
