"""
simulator/backend/state.py
===========================
Global application state for the AETHEL123 Real-Time Audio Backend.

Unifies application state with LiveAudioEngine:
    - Authoritative model loading state
    - Hardware device discovery and streaming state
    - Current configuration
    - Processing telemetry and metrics
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE3_100EP_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase3_100ep" / "best_checkpoint.pt"
PHASE3_D_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
PHASE3_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase3_H" / "best_checkpoint.pt"
STEP6_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"

if PHASE3_100EP_CHECKPOINT.exists():
    DEFAULT_CHECKPOINT = PHASE3_100EP_CHECKPOINT
elif PHASE3_D_CHECKPOINT.exists():
    DEFAULT_CHECKPOINT = PHASE3_D_CHECKPOINT
elif PHASE3_CHECKPOINT.exists():
    DEFAULT_CHECKPOINT = PHASE3_CHECKPOINT
else:
    DEFAULT_CHECKPOINT = STEP6_CHECKPOINT


@dataclass
class SimulatorConfig:
    """Runtime configuration."""
    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    device: str = "cpu"
    enable_ai: bool = True
    enable_nlms: bool = True
    enable_vad: bool = True
    enable_fusion: bool = True
    enable_limiter: bool = True
    # Noise simulation / test parameters
    noise_class: int = 0
    snr_db: float = 10.0
    delay_samples: int = 5
    noise_gain: float = 1.0


@dataclass
class ProcessingMetrics:
    """Per-session processing metrics."""
    audio_duration_s: float = 0.0
    processing_time_s: float = 0.0
    real_time_factor: float = 0.0
    input_snr_db: Optional[float] = None
    output_snr_db: Optional[float] = None
    snr_improvement_db: Optional[float] = None
    noise_class_predicted: int = 0
    classifier_confidence: float = 0.0
    speech_probability: float = 0.0
    estimated_delay_samples: float = 0.0
    ai_weight: float = 0.5
    dsp_weight: float = 0.5
    limiter_clipping_count: int = 0
    rms_input: float = 0.0
    rms_output: float = 0.0
class CallableBool:
    """Helper enabling both method call state.is_model_loaded() and boolean evaluation."""
    def __init__(self, getter: Any) -> None:
        self._getter = getter
    def __call__(self) -> bool:
        return bool(self._getter())
    def __bool__(self) -> bool:
        return bool(self._getter())
    def __repr__(self) -> str:
        return str(bool(self))
    def __eq__(self, other: Any) -> bool:
        return bool(self) == bool(other)


class AppState:
    """
    Thread-safe global application state integrating the live real-time audio engine.
    """

    def __init__(self, live_engine: Optional[Any] = None) -> None:
        self._lock = threading.Lock()
        self.config = SimulatorConfig()
        self.metrics = ProcessingMetrics()
        self._live_engine = live_engine
        self._engine = None
        self._is_processing: bool = False
        self._session_count: int = 0
        self._error_log: List[str] = []

    def set_live_engine(self, engine: Any) -> None:
        """Bind the active LiveAudioEngine."""
        with self._lock:
            self._live_engine = engine

    @property
    def live_engine(self) -> Any:
        with self._lock:
            if self._live_engine is None:
                try:
                    from src.realtime.live_audio_engine import LiveAudioEngine
                    self._live_engine = LiveAudioEngine(
                        checkpoint_path=self.config.checkpoint_path,
                        device=self.config.device,
                    )
                except Exception as e:
                    logger.warning("Could not auto-instantiate LiveAudioEngine: %s", e)
            return self._live_engine
    @property
    def is_model_loaded(self) -> CallableBool:
        """Can be called as state.is_model_loaded() or checked as bool(state.is_model_loaded)."""
        return CallableBool(lambda: self.model_loaded)

    @property
    def model_loaded(self) -> bool:
        """True if the AI model is loaded and verified in memory."""
        try:
            if self.live_engine is not None:
                return bool(self.live_engine.is_model_loaded)
        except Exception as e:
            logger.warning("Error checking model_loaded: %s", e)
        return False

    @property
    def running(self) -> bool:
        """True if physical audio stream is running."""
        try:
            if self.live_engine is not None:
                return bool(self.live_engine.is_running)
        except Exception:
            pass
        return False

    @property
    def checkpoint_path(self) -> str:
        return str(self.config.checkpoint_path)

    def get_status(self) -> dict:
        """
        Authoritative status dictionary strictly matching the project specification.
        Never crashes; returns valid state under any engine condition.
        """
        try:
            if self.live_engine is not None:
                return self.live_engine.get_status_dict()
        except Exception as e:
            logger.error("Error retrieving status from live engine: %s", e)

        # Fallback safe status
        return {
            "running": False,
            "model_loaded": self.is_model_loaded,
            "input_device": None,
            "output_device": None,
            "input_channels": None,
            "output_channels": None,
            "sample_rate": 16000,
            "block_size": 256,
            "ai_enabled": self.config.enable_ai,
            "nlms_enabled": self.config.enable_nlms,
            "vad_enabled": self.config.enable_vad,
            "fusion_enabled": self.config.enable_fusion,
            "limiter_enabled": self.config.enable_limiter,
            "rtf": 0.0,
            "input_rms": None,
            "reference_rms": None,
            "output_rms": None,
            "input_underruns": 0,
            "output_underruns": 0,
            "dropped_blocks": 0,
            "error": None,
        }

    def get_config(self) -> dict:
        with self._lock:
            return {
                "checkpoint_path": self.config.checkpoint_path,
                "device": self.config.device,
                "enable_ai": self.config.enable_ai,
                "enable_nlms": self.config.enable_nlms,
                "enable_vad": self.config.enable_vad,
                "enable_fusion": self.config.enable_fusion,
                "enable_limiter": self.config.enable_limiter,
                "noise_class": self.config.noise_class,
                "snr_db": self.config.snr_db,
                "delay_samples": self.config.delay_samples,
                "noise_gain": self.config.noise_gain,
            }

    def update_config(self, updates: dict) -> None:
        with self._lock:
            for k, v in updates.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self._engine = None

    def get_metrics(self) -> dict:
        with self._lock:
            m = self.metrics
            return {
                "audio_duration_s": m.audio_duration_s,
                "processing_time_s": m.processing_time_s,
                "real_time_factor": m.real_time_factor,
                "input_snr_db": m.input_snr_db,
                "output_snr_db": m.output_snr_db,
                "snr_improvement_db": m.snr_improvement_db,
                "noise_class_predicted": m.noise_class_predicted,
                "classifier_confidence": m.classifier_confidence,
                "speech_probability": m.speech_probability,
                "estimated_delay_samples": m.estimated_delay_samples,
                "ai_weight": m.ai_weight,
                "dsp_weight": m.dsp_weight,
                "limiter_clipping_count": m.limiter_clipping_count,
                "rms_input": m.rms_input,
                "rms_output": m.rms_output,
                "session_count": self._session_count,
            }

    def set_processing(self, flag: bool) -> None:
        with self._lock:
            self._is_processing = flag

    def is_processing(self) -> bool:
        with self._lock:
            return self._is_processing

    def get_engine(self):
        """Get or create the offline/synthetic processing engine."""
        with self._lock:
            if self._engine is None:
                try:
                    from src.realtime.realtime_engine import RealtimeEngine
                    self._engine = RealtimeEngine(
                        checkpoint_path=self.config.checkpoint_path,
                        device=self.config.device,
                    )
                    self._engine.configure(
                        enable_ai=self.config.enable_ai,
                        enable_nlms=self.config.enable_nlms,
                        enable_vad=self.config.enable_vad,
                        enable_fusion=self.config.enable_fusion,
                        enable_limiter=self.config.enable_limiter,
                    )
                except Exception as e:
                    self._error_log.append(f"Engine init error: {e}")
                    raise
            return self._engine

    def update_metrics(self, result: dict, pipeline_diag: dict) -> None:
        with self._lock:
            m = self.metrics
            m.audio_duration_s = result.get("audio_duration_s", 0.0)
            m.processing_time_s = result.get("processing_time_s", 0.0)
            m.real_time_factor = result.get("real_time_factor", 0.0)
            m.noise_class_predicted = pipeline_diag.get("current_noise_class", 0)
            m.classifier_confidence = pipeline_diag.get("current_confidence", 0.0)
            m.speech_probability = pipeline_diag.get("current_speech_prob", 0.0)
            m.estimated_delay_samples = pipeline_diag.get("current_delay_samples", 0.0)
            m.ai_weight = 0.5
            m.dsp_weight = 0.5
            m.limiter_clipping_count = pipeline_diag.get("limiter_clipping_count", 0)
            self._session_count += 1


# Module-level singleton
state = AppState()
