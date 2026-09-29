"""
src/realtime/audio_output.py
============================
Physical Audio Output Stream.

Manages low-latency audio playback to physical headphones/speakers using sounddevice.OutputStream.
Consumes processed audio from the output ring buffer with safe gain and limiter controls.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from src.realtime.ring_buffer import RingBuffer
from src.realtime.constants import SAMPLE_RATE, BLOCK_SIZE

logger = logging.getLogger(__name__)


class AudioOutputStream:
    """
    Physical audio output stream wrapper.

    Parameters
    ----------
    device_id : int
        PortAudio output device index.
    channels : int
        Number of output channels (typically 2 for stereo headphones/speakers).
    sample_rate : int
        Sampling rate in Hz. Default 16000.
    block_size : int
        PortAudio block size in frames. Fixed to 256.
    output_buffer : RingBuffer
        Ring buffer containing enhanced audio samples ready for playback.
    master_gain : float
        Conservative master output gain (0.0 to 1.0). Default 0.5.
    """

    def __init__(
        self,
        device_id: int,
        channels: int,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        output_buffer: RingBuffer = None,
        master_gain: float = 0.5,
    ) -> None:
        self.device_id = device_id
        self.channels = channels
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._output_buffer = output_buffer
        self.master_gain = max(0.0, min(1.0, float(master_gain)))
        self.is_muted: bool = False

        # Output adapter for hardware with variable/non-256 buffer sizes
        self._adapter_out = np.empty(0, dtype=np.float32)
        self._adapter_lock = threading.Lock()

        self._stream: Optional[sd.OutputStream] = None
        self._is_running: bool = False
        self._lock = threading.Lock()

        # Telemetry / counters (Phase 2 strictly required)
        self.output_callback_count: int = 0
        self.output_frames_sent: int = 0
        self.underflow_count: int = 0
        self.callback_errors: int = 0
        self.output_nonzero_frames: int = 0
        self.last_status: Optional[str] = None
        self.last_output_rms: float = 0.0
        self.last_output_peak: float = 0.0
        self.last_output_timestamp: float = 0.0

        # Backward compatibility alias
        self.frames_played: int = 0

        # Bounded telemetry buffer for genuine physical output visualization (~250 ms at 16kHz)
        self._telemetry_capacity: int = 4000
        self._telemetry_buffer: np.ndarray = np.zeros(self._telemetry_capacity, dtype=np.float32)
        self._telemetry_lock = threading.Lock()
        self._last_log_time: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_gain(self, gain: float) -> None:
        """Set output volume gain (0.0 to 1.0)."""
        self.master_gain = max(0.0, min(1.0, float(gain)))

    def set_muted(self, muted: bool) -> None:
        """Mute or unmute output stream."""
        self.is_muted = bool(muted)

    def get_recent_samples(self, n: int = 1024) -> np.ndarray:
        """Return the most recent unrounded physical output samples."""
        with self._telemetry_lock:
            n = min(n, self._telemetry_capacity)
            return self._telemetry_buffer[-n:].copy()

    def _callback(self, outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        """Lightweight non-blocking audio playback callback enforcing 256-sample block buffering adapter."""
        import time as _time
        now = _time.time()
        self.output_callback_count += 1
        self.last_output_timestamp = now

        if status:
            self.last_status = str(status)
            self.callback_errors += 1
            if status.output_underflow:
                self.underflow_count += 1

        self.output_frames_sent += frames
        self.frames_played = self.output_frames_sent

        # Read samples ensuring internal fixed block consumption
        if frames == self.block_size:
            samples = self._output_buffer.read(frames)
            n_read = len(samples)
            if n_read < frames:
                self.underflow_count += 1
                padded = np.zeros(frames, dtype=np.float32)
                if n_read > 0:
                    padded[:n_read] = samples
                samples = padded
        else:
            with self._adapter_lock:
                while len(self._adapter_out) < frames:
                    chunk = self._output_buffer.read(self.block_size)
                    n_read = len(chunk)
                    if n_read < self.block_size:
                        self.underflow_count += 1
                        pad = np.zeros(self.block_size - n_read, dtype=np.float32)
                        chunk = np.concatenate([chunk, pad]) if n_read > 0 else pad
                    self._adapter_out = np.concatenate([self._adapter_out, chunk])

                samples = self._adapter_out[:frames]
                self._adapter_out = self._adapter_out[frames:]

        # Apply gain and mute
        if self.is_muted:
            samples = np.zeros_like(samples)
        else:
            samples = samples * self.master_gain

        # Measure peak and RMS for telemetry
        peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
        self.last_output_peak = peak
        self.last_output_rms = rms

        if peak > 1e-4:
            self.output_nonzero_frames += frames

        # Store in rolling telemetry buffer
        with self._telemetry_lock:
            if frames >= self._telemetry_capacity:
                self._telemetry_buffer[:] = samples[-self._telemetry_capacity:]
            else:
                self._telemetry_buffer = np.roll(self._telemetry_buffer, -frames)
                self._telemetry_buffer[-frames:] = samples

        # Route to channels (e.g. mono enhanced signal copied to stereo L/R)
        if self.channels == 1:
            outdata[:, 0] = samples
        elif self.channels >= 2:
            outdata[:, 0] = samples
            outdata[:, 1] = samples
            if self.channels > 2:
                outdata[:, 2:] = 0.0

        # Rate-limited diagnostics logging (Phase 5: ~once per second)
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            logger.info(
                "[INFO] Output callback active: frames=%d rms=%.6f peak=%.6f nonzero=%s",
                frames, rms, peak, (peak > 1e-4),
            )

    def start(self) -> None:
        """Start streaming audio to physical output device."""
        with self._lock:
            if self._is_running:
                return

            logger.info(
                "[INFO] RealtimeAudioEngine: AUDIO STREAM OPENED (Output) device=%d, channels=%d, sr=%d, blocksize=%d, gain=%.2f",
                self.device_id, self.channels, self.sample_rate, self.block_size, self.master_gain,
            )

            # Pre-buffer 2 blocks of silence into output ring buffer to eliminate startup underflow
            initial_silence = np.zeros(self.block_size * 2, dtype=np.float32)
            self._output_buffer.write(initial_silence)

            try:
                self._stream = sd.OutputStream(
                    device=self.device_id,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    dtype=np.float32,
                    callback=self._callback,
                )
                self._stream.start()
                self._is_running = True
                logger.info("[INFO] RealtimeAudioEngine: AUDIO OUTPUT CALLBACK ACTIVE")
            except Exception as e:
                self._is_running = False
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                raise RuntimeError(f"Failed to start AudioOutputStream on device {self.device_id}: {e}") from e

    def stop(self) -> None:
        """Stop audio output and release device handle."""
        with self._lock:
            if not self._is_running and self._stream is None:
                return

            self._is_running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error stopping AudioOutputStream: %s", e)
                finally:
                    self._stream = None
            logger.info("AudioOutputStream stopped.")
