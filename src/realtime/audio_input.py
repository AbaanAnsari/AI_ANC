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
from src.realtime.constants import SAMPLE_RATE, BLOCK_SIZE

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
        Audio block size in frames. Fixed to 256.
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
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        m1_buffer: RingBuffer = None,
        m2_buffer: RingBuffer = None,
    ) -> None:
        self.device_id = device_id
        self.channels = channels
        self.primary_channel = primary_channel
        self.ref_channel = ref_channel
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._m1_buffer = m1_buffer
        self._m2_buffer = m2_buffer

        # Buffering adapter for hardware with variable/non-256 buffer sizes
        self._adapter_m1 = np.empty(0, dtype=np.float32)
        self._adapter_m2 = np.empty(0, dtype=np.float32)
        self._adapter_lock = threading.Lock()

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
        """Lightweight non-blocking audio capture callback with strict 256-sample block buffering adapter."""
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

        # Buffer incoming samples into adapter to guarantee fixed block_size (256) processing
        with self._adapter_lock:
            self._adapter_m1 = np.concatenate([self._adapter_m1, m1_chunk])
            self._adapter_m2 = np.concatenate([self._adapter_m2, m2_chunk])

            while len(self._adapter_m1) >= self.block_size:
                block_m1 = self._adapter_m1[:self.block_size]
                block_m2 = self._adapter_m2[:self.block_size]
                self._adapter_m1 = self._adapter_m1[self.block_size:]
                self._adapter_m2 = self._adapter_m2[self.block_size:]

                # Quick peak & RMS for monitoring on this exact 256-sample block
                peak1 = float(np.max(np.abs(block_m1))) if len(block_m1) > 0 else 0.0
                rms1 = float(np.sqrt(np.mean(block_m1 ** 2))) if len(block_m1) > 0 else 0.0
                rms2 = float(np.sqrt(np.mean(block_m2 ** 2))) if len(block_m2) > 0 else 0.0

                self.last_input_peak_m1 = peak1
                self.last_input_rms_m1 = rms1
                self.last_input_rms_m2 = rms2

                if peak1 > 1e-4:
                    self.input_nonzero_frames += len(block_m1)

                # Store in rolling telemetry buffer
                with self._telemetry_lock:
                    if len(block_m1) >= self._telemetry_capacity:
                        self._telemetry_buffer[:] = block_m1[-self._telemetry_capacity:]
                    else:
                        self._telemetry_buffer = np.roll(self._telemetry_buffer, -len(block_m1))
                        self._telemetry_buffer[-len(block_m1):] = block_m1

                # Push exact 256-sample block to ring buffers
                w1 = self._m1_buffer.write(block_m1)
                w2 = self._m2_buffer.write(block_m2)

                if w1 < len(block_m1) or w2 < len(block_m2):
                    self.overflow_count += 1

                # Rate-limited diagnostics logging (~once per second)
                if now - self._last_log_time >= 1.0:
                    self._last_log_time = now
                    logger.info(
                        "[INFO] Input callback active: block_size=%d rms=%.6f peak=%.6f nonzero=%s",
                        self.block_size, rms1, peak1, (peak1 > 1e-4),
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
