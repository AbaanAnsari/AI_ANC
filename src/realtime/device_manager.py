"""
src/realtime/device_manager.py
==============================
Physical Audio Device Discovery and Management.

Discovers and validates hardware input and output devices using sounddevice/PortAudio.
Supports:
    - Enumerating physical input and output devices
    - Inspecting device capabilities (host API, channels, sample rates)
    - Channel configuration and validation (primary vs reference mics)
    - Single-channel vs multi-channel detection
    - Device availability and refresh handling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioDeviceInfo:
    id: int
    name: str
    hostapi_id: int
    hostapi_name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_input: bool
    is_output: bool
    is_default_input: bool
    is_default_output: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["index"] = self.id
        d["host_api"] = self.hostapi_name
        d["default_sample_rate"] = self.default_samplerate
        return d


class AudioDeviceManager:
    """
    Manager for querying and validating physical audio devices on the system.
    """

    def __init__(self) -> None:
        self._devices: List[AudioDeviceInfo] = []
        self._hostapis: Dict[int, str] = {}
        self.refresh()

    def refresh(self) -> List[AudioDeviceInfo]:
        """
        Enumerate all physical audio devices and cache device metadata.
        """
        self._devices.clear()
        self._hostapis.clear()

        # Query Host APIs (e.g. MME, DirectSound, WASAPI, CoreAudio, ALSA)
        try:
            for idx, api in enumerate(sd.query_hostapis()):
                self._hostapis[idx] = api.get("name", f"API {idx}")
        except Exception as e:
            logger.warning("Failed to query host APIs: %s", e)

        # Query default devices
        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in, default_out = -1, -1

        # Query all audio devices
        try:
            raw_devices = sd.query_devices()
            if isinstance(raw_devices, dict):
                raw_devices = [raw_devices]

            for idx, dev in enumerate(raw_devices):
                api_id = dev.get("hostapi", 0)
                api_name = self._hostapis.get(api_id, "Unknown API")
                max_in = int(dev.get("max_input_channels", 0))
                max_out = int(dev.get("max_output_channels", 0))
                sr = float(dev.get("default_samplerate", 44100.0))

                info = AudioDeviceInfo(
                    id=idx,
                    name=dev.get("name", f"Device {idx}"),
                    hostapi_id=api_id,
                    hostapi_name=api_name,
                    max_input_channels=max_in,
                    max_output_channels=max_out,
                    default_samplerate=sr,
                    is_input=max_in > 0,
                    is_output=max_out > 0,
                    is_default_input=(idx == default_in),
                    is_default_output=(idx == default_out),
                )
                self._devices.append(info)
        except Exception as e:
            logger.error("Failed to query audio devices: %s", e)

        logger.info(
            "AudioDeviceManager refreshed: %d total devices (%d input, %d output)",
            len(self._devices),
            len(self.get_input_devices()),
            len(self.get_output_devices()),
        )
        return self._devices

    def get_all_devices(self) -> List[AudioDeviceInfo]:
        """Return all enumerated devices."""
        return list(self._devices)

    def get_input_devices(self) -> List[AudioDeviceInfo]:
        """Return all devices with at least 1 input channel."""
        return [d for d in self._devices if d.is_input]

    def get_output_devices(self) -> List[AudioDeviceInfo]:
        """Return all devices with at least 1 output channel."""
        return [d for d in self._devices if d.is_output]

    def get_device_by_id(self, device_id: int) -> Optional[AudioDeviceInfo]:
        """Get device info by device index."""
        for d in self._devices:
            if d.id == device_id:
                return d
        return None

    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        """Return default input device if available."""
        for d in self._devices:
            if d.is_default_input and d.is_input:
                return d
        inputs = self.get_input_devices()
        return inputs[0] if inputs else None

    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        """Return default output device if available."""
        for d in self._devices:
            if d.is_default_output and d.is_output:
                return d
        outputs = self.get_output_devices()
        return outputs[0] if outputs else None

    def validate_input_config(
        self,
        device_id: int,
        primary_channel: int = 0,
        ref_channel: int = 1,
        sample_rate: int = 16000,
    ) -> Tuple[bool, str, dict]:
        """
        Validate input device and channel configuration.

        Returns:
            (valid: bool, message: str, details: dict)
        """
        dev = self.get_device_by_id(device_id)
        if dev is None:
            return False, f"Input device ID {device_id} not found in system.", {}

        if not dev.is_input:
            return False, f"Device '{dev.name}' (ID {device_id}) has no input channels.", {}

        # Channel availability
        max_ch = dev.max_input_channels
        if primary_channel < 0 or primary_channel >= max_ch:
            return (
                False,
                f"Primary channel {primary_channel} out of range for device '{dev.name}' (max {max_ch} channels).",
                {},
            )

        # Check multi-channel dual mic
        is_dual_mic = (max_ch >= 2)
        single_mic_fallback = False
        warning = None

        if ref_channel < 0 or ref_channel >= max_ch or ref_channel == primary_channel:
            if not is_dual_mic:
                single_mic_fallback = True
                ref_channel = -1
                warning = (
                    f"Selected device '{dev.name}' has only {max_ch} input channel. "
                    "Dual-microphone ANC requires 2 physical channels. "
                    "Operating in single-mic test mode with reference synthesis disabled."
                )
            else:
                return (
                    False,
                    f"Reference channel {ref_channel} invalid (max {max_ch}, must differ from primary).",
                    {},
                )

        # Check PortAudio device settings
        channels_to_open = max_ch if max_ch <= 2 else max(primary_channel, ref_channel) + 1
        try:
            sd.check_input_settings(device=device_id, channels=channels_to_open, samplerate=sample_rate)
        except Exception as e:
            return (
                False,
                f"Device '{dev.name}' does not support {sample_rate} Hz with {channels_to_open} channels: {e}",
                {},
            )

        details = {
            "device_id": device_id,
            "device_name": dev.name,
            "hostapi": dev.hostapi_name,
            "max_channels": max_ch,
            "channels_to_open": channels_to_open,
            "primary_channel": primary_channel,
            "ref_channel": ref_channel,
            "is_dual_mic": is_dual_mic and not single_mic_fallback,
            "single_mic_fallback": single_mic_fallback,
            "warning": warning,
            "sample_rate": sample_rate,
        }
        msg = warning if warning else f"Input configuration valid ({dev.name}, {channels_to_open} channels)."
        return True, msg, details

    def validate_output_config(
        self,
        device_id: int,
        channels: int = 2,
        sample_rate: int = 16000,
    ) -> Tuple[bool, str, dict]:
        """
        Validate output device configuration.

        Returns:
            (valid: bool, message: str, details: dict)
        """
        dev = self.get_device_by_id(device_id)
        if dev is None:
            return False, f"Output device ID {device_id} not found in system.", {}

        if not dev.is_output:
            return False, f"Device '{dev.name}' (ID {device_id}) has no output channels.", {}

        max_out = dev.max_output_channels
        actual_channels = min(channels, max_out)
        if actual_channels < 1:
            return False, f"Device '{dev.name}' has 0 output channels.", {}

        try:
            sd.check_output_settings(device=device_id, channels=actual_channels, samplerate=sample_rate)
        except Exception as e:
            return (
                False,
                f"Device '{dev.name}' does not support {sample_rate} Hz with {actual_channels} channels: {e}",
                {},
            )

        details = {
            "device_id": device_id,
            "device_name": dev.name,
            "hostapi": dev.hostapi_name,
            "channels": actual_channels,
            "sample_rate": sample_rate,
        }
        return True, f"Output configuration valid ({dev.name}, {actual_channels} channels).", details
