"""
src/realtime/audio_input.py
===========================
Physical Audio Input Stream.

Manages low-latency audio capture from physical microphones using sounddevice.InputStream.
Directs incoming multichannel audio to primary (M1) and reference (M2) ring buffers.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Tuple

import numpy as np
import sounddevice as sd

from src.realtime.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioInputStream:
    """
    Physical audio input stream wrapper.

    Parameters
    ----------
    device_id : int
        PortAudio input device index.
    channels : int
        Number of channels to capture from hardware.
    primary_channel : int
        Channel index for primary mic (speech + noise). Default 0.
    ref_channel : int
        Channel index for reference mic (noise reference). Default 1 (-1 for single-mic).
    sample_rate : int
        Sampling rate in Hz. Default 16000.
    block_size : int
        PortAudio block size in frames. Default 256.
    m1_buffer : RingBuffer
        Ring buffer for primary mic samples.
    m2_buffer : RingBuffer
        Ring buffer for reference mic samples.
    """

    def __init__(
        self,
        device_id: int,
        channels: int,
        primary_channel: int,
        ref_channel: int,
        sample_rate: int,
        block_size: int,
        m1_buffer: RingBuffer,
        m2_buffer: RingBuffer,
    ) -> None:
        self.device_id = device_id
        self.channels = channels
        self.primary_channel = primary_channel
        self.ref_channel = ref_channel
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._m1_buffer = m1_buffer
        self._m2_buffer = m2_buffer

        self._stream: Optional[sd.InputStream] = None
        self._is_running: bool = False
        self._lock = threading.Lock()

        # Telemetry / counters (Phase 2 strictly required)
        self.input_callback_count: int = 0
        self.input_frames_received: int = 0
        self.overflow_count: int = 0
        self.underflow_count: int = 0
        self.callback_errors: int = 0
        self.input_nonzero_frames: int = 0
        self.last_status: Optional[str] = None
        self.last_input_rms_m1: float = 0.0
        self.last_input_rms_m2: float = 0.0
        self.last_input_peak_m1: float = 0.0
        self.last_input_timestamp: float = 0.0

        # Backward compatibility aliases
        self.frames_captured: int = 0

        # Bounded telemetry buffer for genuine physical input visualization (~250 ms at 16kHz)
        self._telemetry_capacity: int = 4000
        self._telemetry_buffer: np.ndarray = np.zeros(self._telemetry_capacity, dtype=np.float32)
        self._telemetry_lock = threading.Lock()
        self._last_log_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_recent_samples(self, n: int = 1024) -> np.ndarray:
        """Return the most recent unrounded physical input samples."""
        with self._telemetry_lock:
            n = min(n, self._telemetry_capacity)
            return self._telemetry_buffer[-n:].copy()

    def _callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Lightweight non-blocking audio capture callback."""
        import time as _time
        now = _time.time()
        self.input_callback_count += 1
        self.last_input_timestamp = now

        if status:
            self.last_status = str(status)
            self.callback_errors += 1
            if status.input_overflow:
                self.overflow_count += 1
            if status.input_underflow:
                self.underflow_count += 1

        self.input_frames_received += frames
        self.frames_captured = self.input_frames_received

        # Extract primary channel
        if self.primary_channel < indata.shape[1]:
            m1_chunk = indata[:, self.primary_channel]
        else:
            m1_chunk = indata[:, 0]

        # Extract reference channel (or zero if single-mic fallback)
        if 0 <= self.ref_channel < indata.shape[1]:
            m2_chunk = indata[:, self.ref_channel]
        else:
            # Single-mic fallback: zero reference
            m2_chunk = np.zeros(frames, dtype=np.float32)

        # Quick peak & RMS for monitoring
        peak1 = float(np.max(np.abs(m1_chunk))) if len(m1_chunk) > 0 else 0.0
        rms1 = float(np.sqrt(np.mean(m1_chunk ** 2))) if len(m1_chunk) > 0 else 0.0
        rms2 = float(np.sqrt(np.mean(m2_chunk ** 2))) if len(m2_chunk) > 0 else 0.0

        self.last_input_peak_m1 = peak1
        self.last_input_rms_m1 = rms1
        self.last_input_rms_m2 = rms2

        if peak1 > 1e-4:
            self.input_nonzero_frames += frames

        # Store in rolling telemetry buffer
        with self._telemetry_lock:
            if frames >= self._telemetry_capacity:
                self._telemetry_buffer[:] = m1_chunk[-self._telemetry_capacity:]
            else:
                self._telemetry_buffer = np.roll(self._telemetry_buffer, -frames)
                self._telemetry_buffer[-frames:] = m1_chunk

        # Push to ring buffers without blocking
        w1 = self._m1_buffer.write(m1_chunk)
        w2 = self._m2_buffer.write(m2_chunk)

        if w1 < frames or w2 < frames:
            self.overflow_count += 1

        # Rate-limited diagnostics logging (Phase 5: ~once per second)
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            logger.info(
                "[INFO] Input callback active: frames=%d rms=%.6f peak=%.6f nonzero=%s",
                frames, rms1, peak1, (peak1 > 1e-4),
            )

    def start(self) -> None:
        """Start capturing audio from physical hardware."""
        with self._lock:
            if self._is_running:
                return

            logger.info(
                "[INFO] RealtimeAudioEngine: AUDIO STREAM OPENED (Input) device=%d, channels=%d (M1=%d, M2=%d), sr=%d, blocksize=%d",
                self.device_id, self.channels, self.primary_channel, self.ref_channel,
                self.sample_rate, self.block_size,
            )

            try:
                self._stream = sd.InputStream(
                    device=self.device_id,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    dtype=np.float32,
                    callback=self._callback,
                )
                self._stream.start()
                self._is_running = True
                logger.info("[INFO] RealtimeAudioEngine: AUDIO INPUT CALLBACK ACTIVE")
            except Exception as e:
                self._is_running = False
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                raise RuntimeError(f"Failed to start AudioInputStream on device {self.device_id}: {e}") from e
            except Exception as e:
                self._is_running = False
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                raise RuntimeError(f"Failed to start AudioInputStream on device {self.device_id}: {e}") from e

    def stop(self) -> None:
        """Stop capturing audio and release device handle."""
        with self._lock:
            if not self._is_running and self._stream is None:
                return

            self._is_running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error stopping AudioInputStream: %s", e)
                finally:
                    self._stream = None
            logger.info("AudioInputStream stopped.")
