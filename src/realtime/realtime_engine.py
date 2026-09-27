"""
src/realtime/realtime_engine.py
=================================
Real-time processing engine for the streaming simulator.

Wraps the StreamingPipeline with a simplified interface for:
    - WAV file simulation mode
    - State management
    - Configuration control
    - Metrics collection

NOTE: This is a SIMULATION/RESEARCH MODE engine.
      It does NOT perform true real-time audio I/O.
      Audio I/O would require a separate audio driver integration.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.inference.streaming_pipeline import StreamingPipeline, SAMPLE_RATE, HOP_SIZE

logger = logging.getLogger(__name__)


class RealtimeEngine:
    """
    Real-time simulation engine.

    Parameters
    ----------
    checkpoint_path : str or Path
        AI model checkpoint path.
    device : str
        Inference device. Default 'cpu'.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._pipeline: Optional[StreamingPipeline] = None
        self._running: bool = False

        # Configuration
        self.enable_ai = True
        self.enable_nlms = True
        self.enable_vad = True
        self.enable_fusion = True
        self.enable_limiter = True

        # State
        self._processed_waveform: Optional[np.ndarray] = None
        self._processing_time_s: float = 0.0
        self._status: str = "idle"

    def configure(
        self,
        enable_ai: bool = True,
        enable_nlms: bool = True,
        enable_vad: bool = True,
        enable_fusion: bool = True,
        enable_limiter: bool = True,
    ) -> None:
        """Configure which pipeline stages are active."""
        self.enable_ai = bool(enable_ai)
        self.enable_nlms = bool(enable_nlms)
        self.enable_vad = bool(enable_vad)
        self.enable_fusion = bool(enable_fusion)
        self.enable_limiter = bool(enable_limiter)

    def _build_pipeline(self, buffer_size_s: float = 10.0) -> StreamingPipeline:
        return StreamingPipeline(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            enable_ai=self.enable_ai,
            enable_nlms=self.enable_nlms,
            enable_vad=self.enable_vad,
            enable_fusion=self.enable_fusion,
            enable_limiter=self.enable_limiter,
            buffer_size_s=buffer_size_s,
        )

    def process_waveform(
        self,
        m1: np.ndarray,
        m2: np.ndarray,
        chunk_size: int = HOP_SIZE,
    ) -> dict:
        """
        Process dual-microphone waveforms through the pipeline.

        Parameters
        ----------
        m1 : np.ndarray, shape (N,)
            Primary microphone waveform (float32).
        m2 : np.ndarray, shape (N,)
            Reference microphone waveform (float32).
        chunk_size : int
            Streaming chunk size in samples.

        Returns
        -------
        dict with:
            enhanced_waveform : np.ndarray
            processing_time_s : float
            diagnostics       : dict
        """
        m1 = np.asarray(m1, dtype=np.float32).ravel()
        m2 = np.asarray(m2, dtype=np.float32).ravel()

        n_samples = min(len(m1), len(m2))
        duration_s = n_samples / SAMPLE_RATE
        buffer_size_s = max(duration_s + 5.0, 10.0)

        pipeline = self._build_pipeline(buffer_size_s=buffer_size_s)
        self._status = "processing"

        t_start = time.perf_counter()

        pos = 0
        while pos < n_samples:
            end = min(pos + chunk_size, n_samples)
            pipeline.push(m1[pos:end], m2[pos:end])
            pipeline.process_available()
            pos = end

        # Collect output
        output = pipeline.read_output(pipeline._output_buffer.available)
        self._processing_time_s = time.perf_counter() - t_start
        self._processed_waveform = output
        diagnostics = pipeline.diagnostics

        self._status = "idle"
        logger.info(
            "Processed %.2fs audio in %.3fs (RTF=%.2f)",
            duration_s, self._processing_time_s,
            self._processing_time_s / max(duration_s, 1e-6),
        )

        return {
            "enhanced_waveform": output,
            "processing_time_s": self._processing_time_s,
            "audio_duration_s": duration_s,
            "real_time_factor": self._processing_time_s / max(duration_s, 1e-6),
            "diagnostics": diagnostics,
        }

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_processing_time_s(self) -> float:
        return self._processing_time_s
