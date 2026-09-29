"""
src/realtime/live_audio_engine.py
=================================
Authoritative True Real-Time Audio Engine.

Coordinates:
    - Physical device discovery (AudioDeviceManager)
    - Low-latency physical capture (AudioInputStream)
    - True streaming DSP + Phase-2 Step-6 AI pipeline (StreamingPipeline)
    - Low-latency physical playback (AudioOutputStream)
    - Decoupled worker thread with thread-safe ring buffers
    - Real-time telemetry, latency profiling, and RTF calculation
    - Acoustic feedback safety and master gain controls
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.realtime.constants import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    BLOCK_DURATION_MS,
    HOP_SIZE,
    N_FFT,
    AUDIO_CONFIG_INDICATOR,
)
from src.inference.streaming_pipeline import StreamingPipeline
from src.realtime.audio_input import AudioInputStream
from src.realtime.audio_output import AudioOutputStream
from src.realtime.device_manager import AudioDeviceManager, AudioDeviceInfo
from src.realtime.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)

_root = Path(__file__).resolve().parents[2]
_p3_100 = _root / "experiments" / "phase3_100ep" / "best_checkpoint.pt"
_p3_d = _root / "experiments" / "phase3_D" / "best_checkpoint.pt"
_p2_step6 = _root / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"

DEFAULT_CHECKPOINT = (
    _p3_100 if _p3_100.exists() else (_p3_d if _p3_d.exists() else _p2_step6)
)


@dataclass
class LiveTelemetry:
    """Snapshot of real-time engine telemetry."""
    status: str = "STOPPED"
    input_device: str = "None"
    output_device: str = "None"
    input_channels: int = 0
    output_channels: int = 0
    primary_channel: int = 0
    ref_channel: int = 1
    sample_rate: int = SAMPLE_RATE
    block_size: int = BLOCK_SIZE
    block_duration_ms: float = BLOCK_DURATION_MS
    is_dual_mic: bool = True
    warning: Optional[str] = None
    error: Optional[str] = None

    # Audio Signal Metrics
    input_rms_m1: float = 0.0
    input_rms_m2: float = 0.0
    output_rms: float = 0.0
    input_peak_m1: float = 0.0
    output_peak: float = 0.0

    # DSP & AI Metrics
    delay_samples_raw: float = 0.0
    delay_samples_kalman: float = 0.0
    noise_class: str = "stationary"
    noise_class_id: int = 0
    classification_confidence: float = 0.0
    is_speech: bool = False
    speech_probability: float = 0.0
    nlms_active: bool = True
    limiter_active: bool = False
    clipping_count: int = 0

    # Latency & Performance Metrics
    total_audio_duration_s: float = 0.0
    total_processing_time_s: float = 0.0
    real_time_factor: float = 0.0
    mean_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    last_block_processing_ms: float = 0.0
    block_duration_ms: float = 16.0  # 256 / 16000 = 16 ms

    # Latency Breakdown & Queue Health Metrics
    queue_latency_ms: float = 0.0
    input_buffer_latency_ms: float = 16.0
    stft_latency_ms: float = 16.0
    processing_latency_ms: float = 0.0
    output_buffer_latency_ms: float = 0.0
    total_end_to_end_latency_ms: float = 32.0
    latency_status: str = "GOOD"

    # Backlog Recovery & Drop Telemetry
    backlog_drop_events: int = 0
    dropped_samples_total: int = 0
    dropped_ms_total: float = 0.0
    last_drop_ms: float = 0.0

    # Buffer & Stream Health
    input_overflows: int = 0
    output_underflows: int = 0
    dropped_blocks: int = 0
    queue_depth_samples: int = 0

    # Diagnostic Internal Levels (Phase 9)
    raw_input_rms: float = 0.0
    ai_output_rms: float = 0.0
    nlms_output_rms: float = 0.0
    fused_output_rms: float = 0.0
    limited_output_rms: float = 0.0

    # Operational Diagnostic Modes (Phase 10)
    passthrough_mode: bool = False

    # Waveform Snippets (higher precision for live visualizer)
    m1_waveform: List[float] = field(default_factory=list)
    m2_waveform: List[float] = field(default_factory=list)
    output_waveform: List[float] = field(default_factory=list)

    # Live Spectrogram Slice (Phase 12: 64 normalized magnitude bins in dB)
    spectrogram_slice: List[float] = field(default_factory=list)


class LiveAudioEngine:
    """
    True Real-Time Live Audio Engine for AETHEL123.
    """

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.device_manager = AudioDeviceManager()

        # Stream components
        self._input_stream: Optional[AudioInputStream] = None
        self._output_stream: Optional[AudioOutputStream] = None
        self._pipeline: Optional[StreamingPipeline] = None

        # Buffers (bounded 1.0 second capacity for strict realtime operation)
        self.buffer_capacity = int(1.0 * SAMPLE_RATE)
        self._m1_buffer = RingBuffer(self.buffer_capacity)
        self._m2_buffer = RingBuffer(self.buffer_capacity)
        self._output_buffer = RingBuffer(self.buffer_capacity)
        self.diagnostic_delay_ms: float = 0.0


        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._is_running: bool = False
        self._is_paused: bool = False

        # Diagnostic Mode (Phase 10: Output Path Direct Mic Passthrough)
        self.passthrough_mode: bool = False

        # Configuration
        self.selected_input_id: Optional[int] = None
        self.selected_output_id: Optional[int] = None
        self.primary_channel: int = 0
        self.ref_channel: int = 1
        self.sample_rate: int = SAMPLE_RATE
        self.block_size: int = 256
        self.master_gain: float = 0.5

        # Telemetry & Profiling
        self._telemetry = LiveTelemetry()
        self._block_times_ms: List[float] = []
        self._max_history = 1000
        self._total_hops_processed: int = 0
        self._total_processing_time_s: float = 0.0
        self._total_audio_duration_s: float = 0.0
        self._processing_errors: int = 0
        self._last_proc_log_time: float = 0.0

        # Live Spectrogram Engine (Phase 12: n_fft=512, hop=128, Hann)
        self._spec_buffer: np.ndarray = np.zeros(N_FFT, dtype=np.float32)
        self._spec_window: np.ndarray = np.hanning(N_FFT).astype(np.float32)
        self._spectrogram_matrix: List[List[float]] = []
        self._spectrogram_lock = threading.Lock()

        # Pre-load AI Model
        self._ensure_pipeline_loaded()

    def _ensure_pipeline_loaded(self) -> None:
        """Pre-load AI model once at startup."""
        if self._pipeline is None:
            logger.info("Initializing StreamingPipeline with checkpoint %s", self.checkpoint_path)
            self._pipeline = StreamingPipeline(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                enable_ai=True,
                enable_nlms=True,
                enable_vad=True,
                enable_fusion=True,
                enable_limiter=True,
            )

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_model_loaded(self) -> bool:
        """Authoritative check if AI model is loaded and ready."""
        try:
            return (
                self._pipeline is not None
                and hasattr(self._pipeline, "_ai")
                and self._pipeline._ai is not None
                and hasattr(self._pipeline._ai, "model")
                and self._pipeline._ai.model is not None
            )
        except Exception:
            return False

    @property
    def model_loaded(self) -> bool:
        return self.is_model_loaded

    @property
    def pipeline(self) -> Optional[StreamingPipeline]:
        return self._pipeline

    def set_passthrough(self, enabled: bool) -> None:
        """Enable or disable direct microphone passthrough diagnostic mode (Phase 10)."""
        self.passthrough_mode = bool(enabled)
        self._telemetry.passthrough_mode = self.passthrough_mode
        logger.info("[INFO] LiveAudioEngine: Passthrough mode set to %s", self.passthrough_mode)

    def get_input_waveform(self, n_samples: int = 512) -> dict:
        """Return the most recent unrounded physical input audio samples (Phase 11)."""
        if self._input_stream:
            samples = self._input_stream.get_recent_samples(n_samples)
        else:
            samples = np.zeros(n_samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        return {
            "samples": [round(float(v), 5) for v in samples],
            "rms": rms,
            "peak": peak,
            "nonzero": bool(peak > 1e-4),
            "running": self._is_running,
        }

    def get_output_waveform(self, n_samples: int = 512) -> dict:
        """Return the most recent unrounded physical output audio samples (Phase 11)."""
        if self._output_stream:
            samples = self._output_stream.get_recent_samples(n_samples)
        else:
            samples = np.zeros(n_samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        return {
            "samples": [round(float(v), 5) for v in samples],
            "rms": rms,
            "peak": peak,
            "nonzero": bool(peak > 1e-4),
            "running": self._is_running,
        }

    def get_spectrogram(self) -> dict:
        """Return the rolling 2D spectrogram from real physical input M1 audio (Phase 12)."""
        with self._spectrogram_lock:
            mat = [list(row) for row in self._spectrogram_matrix]
        return {
            "time_frames": len(mat),
            "freq_bins": 64,
            "spectrogram": mat,
            "running": self._is_running,
        }

    def get_audio_health(self) -> dict:
        """Phase 18 strict health check endpoint contract."""
        in_open = bool(self._input_stream and self._input_stream.is_running)
        in_cb_active = bool(self._input_stream and self._input_stream.input_callback_count > 0)
        in_fr = self._input_stream.input_frames_received if self._input_stream else 0
        in_rms = float(self._input_stream.last_input_rms_m1) if self._input_stream else 0.0

        proc_active = bool(self._worker_thread and self._worker_thread.is_alive() and self._is_running)
        proc_blocks = int(self._total_hops_processed)

        out_open = bool(self._output_stream and self._output_stream.is_running)
        out_cb_active = bool(self._output_stream and self._output_stream.output_callback_count > 0)
        out_fr = self._output_stream.output_frames_sent if self._output_stream else 0
        out_rms = float(self._output_stream.last_output_rms) if self._output_stream else 0.0

        errors: List[str] = []
        if self._telemetry and self._telemetry.error:
            errors.append(str(self._telemetry.error))
        if self._processing_errors > 0:
            errors.append(f"{self._processing_errors} worker processing errors")
        if self._input_stream and self._input_stream.callback_errors > 0:
            errors.append(f"{self._input_stream.callback_errors} input callback errors")
        if self._output_stream and self._output_stream.callback_errors > 0:
            errors.append(f"{self._output_stream.callback_errors} output callback errors")

        return {
            "input_stream_open": in_open,
            "input_callback_active": in_cb_active,
            "input_frames_received": in_fr,
            "input_rms": in_rms,
            "processing_active": proc_active,
            "processing_blocks": proc_blocks,
            "output_stream_open": out_open,
            "output_callback_active": out_cb_active,
            "output_frames_sent": out_fr,
            "output_rms": out_rms,
            "errors": errors,
        }

    def get_status_dict(self) -> dict:
        """
        Return defensive, strictly typed status dictionary matching the required contract.
        Never raises exceptions, even when stopped or in error state.
        Includes Phase 2 strictly structured 'audio' dictionary.
        """
        try:
            t = self._telemetry
            pipeline = self._pipeline

            # Determine input device name
            in_name = None
            if t and t.input_device and t.input_device != "None":
                in_name = t.input_device
            elif self.selected_input_id is not None:
                d = self.device_manager.get_device_by_id(self.selected_input_id)
                in_name = d.name if d else None
            else:
                d = self.device_manager.get_default_input_device()
                in_name = d.name if d else None

            # Determine output device name
            out_name = None
            if t and t.output_device and t.output_device != "None":
                out_name = t.output_device
            elif self.selected_output_id is not None:
                d = self.device_manager.get_device_by_id(self.selected_output_id)
                out_name = d.name if d else None
            else:
                d = self.device_manager.get_default_output_device()
                out_name = d.name if d else None

            # Channels
            in_ch = None
            if t and t.input_channels > 0:
                in_ch = t.input_channels
            elif self.selected_input_id is not None:
                d = self.device_manager.get_device_by_id(self.selected_input_id)
                in_ch = d.max_input_channels if d else None
            else:
                d = self.device_manager.get_default_input_device()
                in_ch = d.max_input_channels if d else None

            out_ch = None
            if t and t.output_channels > 0:
                out_ch = t.output_channels
            elif self.selected_output_id is not None:
                d = self.device_manager.get_device_by_id(self.selected_output_id)
                out_ch = d.max_output_channels if d else None
            else:
                d = self.device_manager.get_default_output_device()
                out_ch = d.max_output_channels if d else None

            # Audio levels (float or None if unmeasured/silent)
            in_rms = float(t.input_rms_m1) if t else None
            ref_rms = float(t.input_rms_m2) if t else None
            out_rms = float(t.output_rms) if t else None

            # RTF
            rtf_val = round(float(t.real_time_factor), 3) if (t and t.real_time_factor > 0) else 0.0

            # Build Phase 2 audio diagnostics sub-dictionary
            in_cb = self._input_stream.input_callback_count if self._input_stream else 0
            in_fr = self._input_stream.input_frames_received if self._input_stream else 0
            in_peak = self._input_stream.last_input_peak_m1 if self._input_stream else 0.0
            in_nonzero = bool(in_peak > 1e-4)

            out_cb = self._output_stream.output_callback_count if self._output_stream else 0
            out_fr = self._output_stream.output_frames_sent if self._output_stream else 0
            out_peak = self._output_stream.last_output_peak if self._output_stream else 0.0
            out_nonzero = bool(out_peak > 1e-4)

            worker_alive = bool(self._worker_thread and self._worker_thread.is_alive())
            queue_depth = self._m1_buffer.available if self._m1_buffer else 0
            in_overflow = self._input_stream.overflow_count if self._input_stream else 0
            out_underflow = self._output_stream.underflow_count if self._output_stream else 0

            audio_dict = {
                "running": bool(self._is_running),
                "passthrough_mode": bool(self.passthrough_mode),
                "input": {
                    "device_id": self.selected_input_id,
                    "device_name": in_name,
                    "callback_count": in_cb,
                    "frames_received": in_fr,
                    "rms": in_rms,
                    "peak": in_peak,
                    "nonzero": in_nonzero,
                },
                "processing": {
                    "blocks": self._total_hops_processed,
                    "errors": self._processing_errors,
                    "thread_alive": worker_alive,
                    "queue_depth": queue_depth,
                    "queue_overruns": in_overflow,
                    "queue_underruns": out_underflow,
                    "queue_latency_ms": t.queue_latency_ms if t else 0.0,
                    "total_latency_ms": t.total_end_to_end_latency_ms if t else 32.0,
                    "latency_status": t.latency_status if t else "GOOD",
                    "backlog_drops": t.backlog_drop_events if t else 0,
                    "dropped_ms_total": round(t.dropped_ms_total, 1) if t else 0.0,
                },
                "latency": {
                    "queue_ms": t.queue_latency_ms if t else 0.0,
                    "input_buffer_ms": t.input_buffer_latency_ms if t else 16.0,
                    "stft_ms": t.stft_latency_ms if t else 16.0,
                    "processing_ms": t.processing_latency_ms if t else 0.0,
                    "output_buffer_ms": t.output_buffer_latency_ms if t else 0.0,
                    "total_ms": t.total_end_to_end_latency_ms if t else 32.0,
                    "status": t.latency_status if t else "GOOD",
                    "backlog_drops": t.backlog_drop_events if t else 0,
                    "dropped_ms_total": round(t.dropped_ms_total, 1) if t else 0.0,
                },
                "output": {
                    "device_id": self.selected_output_id,
                    "device_name": out_name,
                    "callback_count": out_cb,
                    "frames_sent": out_fr,
                    "rms": out_rms,
                    "peak": out_peak,
                    "nonzero": out_nonzero,
                },
                "underflows": out_underflow,
                "overflows": in_overflow,
                "last_input_timestamp": self._input_stream.last_input_timestamp if self._input_stream else 0.0,
                "last_output_timestamp": self._output_stream.last_output_timestamp if self._output_stream else 0.0,
            }

            return {
                "running": bool(self._is_running),
                "model_loaded": bool(self.is_model_loaded),
                "input_device": in_name,
                "output_device": out_name,
                "input_channels": in_ch,
                "output_channels": out_ch,
                "sample_rate": int(self.sample_rate),
                "block_size": int(self.block_size),
                "ai_enabled": bool(pipeline.enable_ai) if pipeline else True,
                "nlms_enabled": bool(pipeline.enable_nlms) if pipeline else True,
                "vad_enabled": bool(pipeline.enable_vad) if pipeline else True,
                "fusion_enabled": bool(pipeline.enable_fusion) if pipeline else True,
                "limiter_enabled": bool(pipeline.enable_limiter) if pipeline else True,
                "rtf": rtf_val,
                "latency_ms": t.total_end_to_end_latency_ms if t else 32.0,
                "queue_latency_ms": t.queue_latency_ms if t else 0.0,
                "latency_status": t.latency_status if t else "GOOD",
                "input_rms": in_rms,
                "reference_rms": ref_rms,
                "output_rms": out_rms,
                "input_underruns": int(t.input_overflows) if t else 0,
                "output_underruns": int(t.output_underflows) if t else 0,
                "dropped_blocks": int(t.dropped_blocks) if t else 0,
                "error": t.error if (t and t.error) else None,
                "audio": audio_dict,
            }
        except Exception as e:
            logger.error("Error generating status dict: %s", e)
            return {
                "running": False,
                "model_loaded": False,
                "input_device": None,
                "output_device": None,
                "input_channels": None,
                "output_channels": None,
                "sample_rate": 16000,
                "block_size": 256,
                "ai_enabled": True,
                "nlms_enabled": True,
                "vad_enabled": True,
                "fusion_enabled": True,
                "limiter_enabled": True,
                "rtf": None,
                "input_rms": None,
                "reference_rms": None,
                "output_rms": None,
                "input_underruns": 0,
                "output_underruns": 0,
                "dropped_blocks": 0,
                "error": str(e),
            }

    def get_devices(self) -> dict:
        """Return categorized device list."""
        all_devs = self.device_manager.get_all_devices()
        input_devs = [d.to_dict() for d in self.device_manager.get_input_devices()]
        output_devs = [d.to_dict() for d in self.device_manager.get_output_devices()]
        default_in = self.device_manager.get_default_input_device()
        default_out = self.device_manager.get_default_output_device()

        return {
            "all_devices": [d.to_dict() for d in all_devs],
            "input_devices": input_devs,
            "output_devices": output_devs,
            "default_input_id": default_in.id if default_in else None,
            "default_output_id": default_out.id if default_out else None,
        }

    def refresh_devices(self) -> dict:
        """Re-enumerate audio devices."""
        self.device_manager.refresh()
        return self.get_devices()

    def set_gain(self, gain: float) -> None:
        """Update master output gain."""
        self.master_gain = max(0.0, min(1.0, float(gain)))
        if self._output_stream:
            self._output_stream.set_gain(self.master_gain)

    def set_paused(self, paused: bool) -> None:
        """Pause or resume audio processing."""
        self._is_paused = bool(paused)
        if self._output_stream:
            self._output_stream.set_muted(self._is_paused)

    def start(
        self,
        input_device_id: Optional[int] = None,
        output_device_id: Optional[int] = None,
        primary_channel: int = 0,
        ref_channel: int = 1,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        master_gain: float = 0.5,
    ) -> Tuple[bool, str]:
        """
        Validate devices and start live audio streaming with fixed 256-sample (16 ms @ 16 kHz) processing.
        """
        with self._state_lock:
            if self._is_running:
                return True, "Stream is already running."

            # Enforce fixed immutable block size (256 samples / 16 ms @ 16 kHz)
            block_size = BLOCK_SIZE
            sample_rate = SAMPLE_RATE

            # Resolve default devices if None
            if input_device_id is None:
                def_in = self.device_manager.get_default_input_device()
                if def_in is None:
                    return False, "No audio input devices detected on the system."
                input_device_id = def_in.id

            if output_device_id is None:
                def_out = self.device_manager.get_default_output_device()
                if def_out is None:
                    return False, "No audio output devices detected on the system."
                output_device_id = def_out.id

            # 1. Validate Input Device & Channels
            val_in, msg_in, det_in = self.device_manager.validate_input_config(
                device_id=input_device_id,
                primary_channel=primary_channel,
                ref_channel=ref_channel,
                sample_rate=sample_rate,
            )
            if not val_in:
                return False, f"Input validation failed: {msg_in}"

            # 2. Validate Output Device
            val_out, msg_out, det_out = self.device_manager.validate_output_config(
                device_id=output_device_id,
                channels=2,
                sample_rate=sample_rate,
            )
            if not val_out:
                return False, f"Output validation failed: {msg_out}"

            # Store configuration (immutable 256 block size)
            self.selected_input_id = input_device_id
            self.selected_output_id = output_device_id
            self.primary_channel = primary_channel
            self.ref_channel = det_in["ref_channel"]
            self.sample_rate = sample_rate
            self.block_size = BLOCK_SIZE
            self.master_gain = master_gain

            # Reset buffers & telemetry
            self._m1_buffer.reset()
            self._m2_buffer.reset()
            self._output_buffer.reset()
            self._block_times_ms.clear()
            self._total_hops_processed = 0
            self._total_processing_time_s = 0.0
            self._total_audio_duration_s = 0.0

            # Ensure pipeline is ready
            self._ensure_pipeline_loaded()
            if self._pipeline:
                self._pipeline.reset()

            # Initialize streams
            try:
                self._input_stream = AudioInputStream(
                    device_id=input_device_id,
                    channels=det_in["channels_to_open"],
                    primary_channel=self.primary_channel,
                    ref_channel=self.ref_channel,
                    sample_rate=sample_rate,
                    block_size=block_size,
                    m1_buffer=self._m1_buffer,
                    m2_buffer=self._m2_buffer,
                )

                self._output_stream = AudioOutputStream(
                    device_id=output_device_id,
                    channels=det_out["channels"],
                    sample_rate=sample_rate,
                    block_size=block_size,
                    output_buffer=self._output_buffer,
                    master_gain=self.master_gain,
                )

                # Start worker thread
                self._stop_event.clear()
                self._worker_thread = threading.Thread(
                    target=self._processing_worker,
                    name="LiveAudioWorker",
                    daemon=True,
                )
                self._worker_thread.start()

                # Start physical streams (output first to avoid backlog, then input)
                self._output_stream.start()
                self._input_stream.start()

                self._is_running = True
                self._telemetry.status = "RUNNING"
                self._telemetry.input_device = det_in["device_name"]
                self._telemetry.output_device = det_out["device_name"]
                self._telemetry.input_channels = det_in["max_channels"]
                self._telemetry.output_channels = det_out["channels"]
                self._telemetry.primary_channel = self.primary_channel
                self._telemetry.ref_channel = self.ref_channel
                self._telemetry.sample_rate = sample_rate
                self._telemetry.block_size = block_size
                self._telemetry.is_dual_mic = det_in["is_dual_mic"]
                self._telemetry.warning = det_in.get("warning")
                self._telemetry.error = None

                logger.info("LiveAudioEngine started successfully: in=%s, out=%s", det_in['device_name'], det_out['device_name'])
                return True, f"Real-time processing active on '{det_in['device_name']}' -> '{det_out['device_name']}'."

            except Exception as e:
                logger.error("Failed to start LiveAudioEngine: %s", e)
                self._cleanup_streams()
                self._telemetry.status = "ERROR"
                self._telemetry.error = str(e)
                return False, f"Failed to open live streams: {e}"

    def stop(self) -> None:
        """Safely stop physical streams, terminate worker, and release devices."""
        with self._state_lock:
            if not self._is_running:
                return

            logger.info("Stopping LiveAudioEngine...")
            self._cleanup_streams()
            self._is_running = False
            self._telemetry.status = "STOPPED"
            logger.info("LiveAudioEngine stopped cleanly.")

    def _cleanup_streams(self) -> None:
        """Internal helper to stop streams and worker thread."""
        self._stop_event.set()

        if self._input_stream:
            try:
                self._input_stream.stop()
            except Exception as e:
                logger.warning("Error stopping input stream: %s", e)
            self._input_stream = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        if self._output_stream:
            try:
                self._output_stream.stop()
            except Exception as e:
                logger.warning("Error stopping output stream: %s", e)
            self._output_stream = None

        if self._pipeline:
            try:
                self._pipeline.reset()
            except Exception as e:
                logger.warning("Error resetting streaming pipeline: %s", e)

    def _processing_worker(self) -> None:
        """
        Dedicated worker thread that processes audio blocks from the input ring buffer
        through the StreamingPipeline and writes enhanced audio to the output ring buffer.
        Enforces strict real-time backpressure: latest audio wins, no multi-second queue backlog.
        """
        MAX_QUEUE_MS = 100.0
        TARGET_RECOVERY_MS = 16.0
        MAX_OUTPUT_QUEUE_MS = 100.0

        max_queue_samples = int(self.sample_rate * MAX_QUEUE_MS / 1000.0)
        target_recovery_samples = int(self.sample_rate * TARGET_RECOVERY_MS / 1000.0)
        max_output_queue_samples = int(self.sample_rate * MAX_OUTPUT_QUEUE_MS / 1000.0)

        block_duration_s = self.block_size / self.sample_rate  # 256 / 16000 = 0.016 s (16 ms)

        while not self._stop_event.is_set():
            avail_m1 = self._m1_buffer.available
            avail_m2 = self._m2_buffer.available
            queued_samples = min(avail_m1, avail_m2)

            if queued_samples < self.block_size:
                time.sleep(0.001)
                continue

            # Real-Time Backpressure Watchdog Policy ("LATEST AUDIO WINS")
            if queued_samples > max_queue_samples:
                drop_samples = queued_samples - target_recovery_samples
                d1 = self._m1_buffer.discard(drop_samples)
                d2 = self._m2_buffer.discard(drop_samples)
                actual_dropped = min(d1, d2)

                if actual_dropped > 0:
                    dropped_ms = (actual_dropped / self.sample_rate) * 1000.0
                    queue_before_ms = (queued_samples / self.sample_rate) * 1000.0
                    queue_after_ms = ((queued_samples - actual_dropped) / self.sample_rate) * 1000.0

                    self._telemetry.backlog_drop_events += 1
                    self._telemetry.dropped_samples_total += actual_dropped
                    self._telemetry.dropped_ms_total += dropped_ms
                    self._telemetry.last_drop_ms = round(dropped_ms, 1)

                    logger.warning(
                        "[WARNING] REALTIME BACKLOG DROP: dropped_samples=%d (%.1f ms) queue_before=%.1f ms queue_after=%.1f ms",
                        actual_dropped, dropped_ms, queue_before_ms, queue_after_ms
                    )

            # Check Output Queue Backlog
            if self._output_buffer.available > max_output_queue_samples:
                out_drop = self._output_buffer.available - (self.block_size * 2)
                if out_drop > 0:
                    self._output_buffer.discard(out_drop)
                    logger.warning("[WARNING] OUTPUT BACKLOG TRIM: trimmed %d samples", out_drop)

            # Optional dev diagnostic delay test mode
            if hasattr(self, "diagnostic_delay_ms") and self.diagnostic_delay_ms > 0:
                time.sleep(self.diagnostic_delay_ms / 1000.0)

            t_start = time.perf_counter()

            # Read exactly one 256-sample block from ring buffers
            m1_block = self._m1_buffer.read(self.block_size)
            m2_block = self._m2_buffer.read(self.block_size)

            if len(m1_block) < self.block_size or len(m2_block) < self.block_size:
                continue

            enhanced_hops = []
            spec_slice = [0.0] * 64

            # Process block via 128-sample STFT hops (2 hops per 256-sample block)
            for hop_idx in range(0, self.block_size, HOP_SIZE):
                m1_hop = m1_block[hop_idx : hop_idx + HOP_SIZE]
                m2_hop = m2_block[hop_idx : hop_idx + HOP_SIZE]

                # Real-time STFT Spectrogram computation from genuine physical M1 input
                try:
                    self._spec_buffer = np.roll(self._spec_buffer, -HOP_SIZE)
                    self._spec_buffer[-HOP_SIZE:] = m1_hop
                    spec_win = self._spec_buffer * self._spec_window
                    fft_mag = np.abs(np.fft.rfft(spec_win, n=N_FFT))  # (257,)
                    chunks = np.array_split(fft_mag[:256], 64)
                    band_mags = np.array([float(np.mean(c)) for c in chunks], dtype=np.float32)
                    band_db = 20.0 * np.log10(np.maximum(band_mags, 1e-6))
                    band_norm = np.clip((band_db + 80.0) / 80.0, 0.0, 1.0)
                    spec_slice = [round(float(v), 4) for v in band_norm]

                    with self._spectrogram_lock:
                        self._spectrogram_matrix.append(spec_slice)
                        if len(self._spectrogram_matrix) > 32:
                            self._spectrogram_matrix.pop(0)
                except Exception as e:
                    spec_slice = [0.0] * 64

                # Phase 10: "OUTPUT PATH TEST" — direct mic passthrough mode
                if self.passthrough_mode:
                    enhanced_hop = m1_hop.copy()
                elif self._pipeline is not None and not self._is_paused:
                    try:
                        self._pipeline.push(m1_hop, m2_hop)
                        self._pipeline.process_available()
                        enhanced_hop = self._pipeline.read_output(HOP_SIZE)

                        if len(enhanced_hop) < HOP_SIZE:
                            pad = m1_hop[len(enhanced_hop):]
                            enhanced_hop = np.concatenate([enhanced_hop, pad])

                        if not np.all(np.isfinite(enhanced_hop)):
                            logger.warning("[WARN] Non-finite output detected in pipeline! Clamping to 0.0")
                            enhanced_hop = np.nan_to_num(enhanced_hop, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception as e:
                        self._processing_errors += 1
                        logger.error("[ERR] Processing block error in pipeline: %s", e)
                        enhanced_hop = m1_hop * 0.5
                else:
                    enhanced_hop = m1_hop * 0.5

                enhanced_hops.append(enhanced_hop)

            enhanced_block = np.concatenate(enhanced_hops)

            t_proc = time.perf_counter() - t_start
            proc_ms = t_proc * 1000.0

            # Write full 256-sample enhanced block to output ring buffer
            w_out = self._output_buffer.write(enhanced_block)
            if w_out < len(enhanced_block):
                self._telemetry.dropped_blocks += 1

            # Update metrics & telemetry
            self._total_hops_processed += len(enhanced_hops)
            self._total_processing_time_s += t_proc
            self._total_audio_duration_s += block_duration_s

            self._block_times_ms.append(proc_ms)
            if len(self._block_times_ms) > self._max_history:
                self._block_times_ms.pop(0)

            # Rate-limited diagnostics logging (~once per second)
            now_sec = time.time()
            if now_sec - self._last_proc_log_time >= 1.0:
                self._last_proc_log_time = now_sec
                in_rms_log = float(np.sqrt(np.mean(m1_block ** 2)))
                out_rms_log = float(np.sqrt(np.mean(enhanced_block ** 2)))
                logger.info(
                    "[INFO] Processing block: block=%d (256 smp) input_rms=%.6f output_rms=%.6f errors=%d passthrough=%s",
                    self._total_hops_processed // 2, in_rms_log, out_rms_log, self._processing_errors, self.passthrough_mode,
                )

            # Update telemetry snapshot
            self._update_telemetry_snapshot(m1_block, m2_block, enhanced_block, proc_ms, spec_slice)

    def _update_telemetry_snapshot(
        self,
        m1_hop: np.ndarray,
        m2_hop: np.ndarray,
        out_hop: np.ndarray,
        proc_ms: float,
        spec_slice: List[float],
    ) -> None:
        """Update live telemetry fields."""
        t = self._telemetry

        # Signal levels (genuine float measurement)
        t.input_rms_m1 = float(np.sqrt(np.mean(m1_hop ** 2)))
        t.input_rms_m2 = float(np.sqrt(np.mean(m2_hop ** 2)))
        t.output_rms = float(np.sqrt(np.mean(out_hop ** 2)))
        t.input_peak_m1 = float(np.max(np.abs(m1_hop)))
        t.output_peak = float(np.max(np.abs(out_hop)))

        # Diagnostic levels (Phase 9)
        t.raw_input_rms = t.input_rms_m1
        t.limited_output_rms = t.output_rms
        t.passthrough_mode = self.passthrough_mode
        t.spectrogram_slice = spec_slice

        # Pipeline diagnostics
        if self._pipeline:
            t.delay_samples_kalman = float(self._pipeline._current_delay_samples)
            t.noise_class_id = int(self._pipeline._current_noise_class)
            class_map = {0: "Stationary", 1: "Non-Stationary", 2: "Impulsive"}
            t.noise_class = class_map.get(t.noise_class_id, "Unknown")
            t.classification_confidence = float(self._pipeline._current_confidence)
            t.speech_probability = float(self._pipeline._current_speech_prob)
            t.is_speech = t.speech_probability > 0.5
            t.nlms_active = self._pipeline.enable_nlms
            t.limiter_active = self._pipeline._limiter.is_clipping
            t.clipping_count = self._pipeline._limiter.clipping_count

        # Timing & RTF
        t.last_block_processing_ms = round(proc_ms, 2)
        t.total_audio_duration_s = round(self._total_audio_duration_s, 2)
        t.total_processing_time_s = round(self._total_processing_time_s, 2)

        if self._total_audio_duration_s > 0:
            t.real_time_factor = round(self._total_processing_time_s / self._total_audio_duration_s, 3)

        if self._block_times_ms:
            times = np.array(self._block_times_ms)
            t.mean_processing_time_ms = round(float(np.mean(times)), 2)
            t.p95_processing_time_ms = round(float(np.percentile(times, 95)), 2)
            t.p99_processing_time_ms = round(float(np.percentile(times, 99)), 2)
            t.max_processing_time_ms = round(float(np.max(times)), 2)

        # Buffer & worker health
        if self._input_stream:
            t.input_overflows = self._input_stream.overflow_count
        if self._output_stream:
            t.output_underflows = self._output_stream.underflow_count

        t.queue_depth_samples = self._m1_buffer.available

        # Latency Breakdown Calculations
        queue_samples = self._m1_buffer.available
        queue_lat_ms = (queue_samples / self.sample_rate) * 1000.0
        in_buf_lat_ms = (self.block_size / self.sample_rate) * 1000.0
        stft_lat_ms = (N_FFT / self.sample_rate) * 1000.0
        proc_lat_ms = proc_ms
        out_buf_samples = self._output_buffer.available
        out_buf_lat_ms = (out_buf_samples / self.sample_rate) * 1000.0
        total_lat_ms = queue_lat_ms + in_buf_lat_ms + proc_lat_ms + out_buf_lat_ms

        t.queue_latency_ms = round(queue_lat_ms, 1)
        t.input_buffer_latency_ms = round(in_buf_lat_ms, 1)
        t.stft_latency_ms = round(stft_lat_ms, 1)
        t.processing_latency_ms = round(proc_lat_ms, 2)
        t.output_buffer_latency_ms = round(out_buf_lat_ms, 1)
        t.total_end_to_end_latency_ms = round(total_lat_ms, 1)

        if total_lat_ms <= 30.0:
            t.latency_status = "GOOD"
        elif total_lat_ms <= 50.0:
            t.latency_status = "NORMAL"
        elif total_lat_ms <= 100.0:
            t.latency_status = "WARNING"
        else:
            t.latency_status = "CRITICAL"

        # Waveforms for live visualizer
        step = max(1, len(m1_hop) // 32)
        t.m1_waveform = [round(float(v), 5) for v in m1_hop[::step]]
        t.m2_waveform = [round(float(v), 5) for v in m2_hop[::step]]
        t.output_waveform = [round(float(v), 5) for v in out_hop[::step]]

    def get_telemetry(self) -> dict:
        """Return full telemetry dictionary for REST/WebSocket."""
        t = self._telemetry
        return {
            "status": t.status,
            "input_device": t.input_device,
            "output_device": t.output_device,
            "input_channels": t.input_channels,
            "output_channels": t.output_channels,
            "primary_channel": t.primary_channel,
            "ref_channel": t.ref_channel,
            "sample_rate": t.sample_rate,
            "block_size": t.block_size,
            "is_dual_mic": t.is_dual_mic,
            "warning": t.warning,
            "error": t.error,
            "passthrough_mode": self.passthrough_mode,
            "levels": {
                "input_rms_m1": t.input_rms_m1,
                "input_rms_m2": t.input_rms_m2,
                "output_rms": t.output_rms,
                "input_peak_m1": t.input_peak_m1,
                "output_peak": t.output_peak,
                "raw_input_rms": t.raw_input_rms,
                "limited_output_rms": t.limited_output_rms,
            },
            "dsp": {
                "kalman_delay": t.delay_samples_kalman,
                "noise_class": t.noise_class,
                "noise_class_id": t.noise_class_id,
                "confidence": t.classification_confidence,
                "is_speech": t.is_speech,
                "speech_probability": t.speech_probability,
                "nlms_active": t.nlms_active,
                "limiter_active": t.limiter_active,
                "clipping_count": t.clipping_count,
            },
            "performance": {
                "rtf": t.real_time_factor,
                "last_processing_ms": t.last_block_processing_ms,
                "mean_processing_ms": t.mean_processing_time_ms,
                "p95_processing_ms": t.p95_processing_time_ms,
                "p99_processing_ms": t.p99_processing_time_ms,
                "max_processing_ms": t.max_processing_time_ms,
                "total_audio_s": t.total_audio_duration_s,
                "total_processing_s": t.total_processing_time_s,
                "input_overflows": t.input_overflows,
                "output_underflows": t.output_underflows,
                "dropped_blocks": t.dropped_blocks,
                "queue_depth": t.queue_depth_samples,
                "processing_errors": self._processing_errors,
                "queue_latency_ms": t.queue_latency_ms,
                "input_buffer_latency_ms": t.input_buffer_latency_ms,
                "stft_latency_ms": t.stft_latency_ms,
                "processing_latency_ms": t.processing_latency_ms,
                "output_buffer_latency_ms": t.output_buffer_latency_ms,
                "total_end_to_end_latency_ms": t.total_end_to_end_latency_ms,
                "latency_status": t.latency_status,
                "backlog_drop_events": t.backlog_drop_events,
                "dropped_samples_total": t.dropped_samples_total,
                "dropped_ms_total": round(t.dropped_ms_total, 1),
                "last_drop_ms": t.last_drop_ms,
            },
            "spectrogram_slice": t.spectrogram_slice,
            "waveforms": {
                "m1": t.m1_waveform,
                "m2": t.m2_waveform,
                "output": t.output_waveform,
            },
        }

