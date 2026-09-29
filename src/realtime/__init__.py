"""
src/realtime
============
Real-time audio I/O and processing engine for AETHEL123.
"""
from src.realtime.ring_buffer import RingBuffer
from src.realtime.audio_input import AudioInputStream
from src.realtime.audio_output import AudioOutputStream
from src.realtime.device_manager import AudioDeviceManager, AudioDeviceInfo
from src.realtime.constants import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    BLOCK_DURATION_MS,
    HOP_SIZE,
    N_FFT,
    AUDIO_CONFIG_INDICATOR,
)


def __getattr__(name: str):
    if name in ("LiveAudioEngine", "LiveTelemetry"):
        from src.realtime.live_audio_engine import LiveAudioEngine, LiveTelemetry
        if name == "LiveAudioEngine":
            return LiveAudioEngine
        return LiveTelemetry
    if name == "RealtimeEngine":
        from src.realtime.realtime_engine import RealtimeEngine
        return RealtimeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



__all__ = [
    "AudioDeviceManager",
    "AudioDeviceInfo",
    "AudioInputStream",
    "AudioOutputStream",
    "LiveAudioEngine",
    "LiveTelemetry",
    "RealtimeEngine",
    "RingBuffer",
    "SAMPLE_RATE",
    "BLOCK_SIZE",
    "BLOCK_DURATION_MS",
    "HOP_SIZE",
    "N_FFT",
    "AUDIO_CONFIG_INDICATOR",
]

