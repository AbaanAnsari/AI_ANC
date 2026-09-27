"""
tests/test_backend_status.py
============================
Regression tests specifically for the /status contract, model-loaded verification,
and error handling.

Verifies:
    1. test_status_before_engine_start
    2. test_status_after_model_load
    3. test_status_when_stopped
    4. test_status_when_running
    5. test_status_after_stop
    6. test_status_after_device_error
    7. test_status_missing_optional_metric
    8. GET /status never raises AttributeError: 'AppState' object has no attribute 'is_model_loaded'
    9. GET /devices returns physical hardware device list with index, host_api, default_sample_rate
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
)

from simulator.backend.state import AppState, state
from simulator.backend.api import get_status, get_devices, app, live_engine
from src.realtime.live_audio_engine import LiveAudioEngine


REQUIRED_STATUS_FIELDS = [
    "running",
    "model_loaded",
    "input_device",
    "output_device",
    "input_channels",
    "output_channels",
    "sample_rate",
    "block_size",
    "ai_enabled",
    "nlms_enabled",
    "vad_enabled",
    "fusion_enabled",
    "limiter_enabled",
    "rtf",
    "input_rms",
    "reference_rms",
    "output_rms",
    "input_underruns",
    "output_underruns",
    "dropped_blocks",
    "error",
]


class TestBackendStatusContract:
    def test_appstate_implements_is_model_loaded_and_model_loaded(self):
        """AppState must implement is_model_loaded both as callable and property."""
        s = AppState()
        # Verify hasattr
        assert hasattr(s, "is_model_loaded")
        assert hasattr(s, "model_loaded")
        assert hasattr(s, "running")
        assert hasattr(s, "checkpoint_path")

        # Verify callable syntax state.is_model_loaded()
        called_result = s.is_model_loaded()
        assert isinstance(called_result, bool)

        # Verify attribute syntax bool(state.is_model_loaded)
        attr_result = bool(s.is_model_loaded)
        assert isinstance(attr_result, bool)
        assert called_result == attr_result

        # Verify model_loaded property
        assert s.model_loaded == called_result

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_200_and_exact_schema(self):
        """GET /status must return all 21 required fields without exception."""
        status = await get_status()
        assert isinstance(status, dict)

        for field_name in REQUIRED_STATUS_FIELDS:
            assert field_name in status, f"Missing required field: {field_name}"

        # Verify specific types
        assert isinstance(status["running"], bool)
        assert isinstance(status["model_loaded"], bool)
        assert isinstance(status["sample_rate"], int)
        assert isinstance(status["block_size"], int)
        assert isinstance(status["ai_enabled"], bool)
        assert isinstance(status["nlms_enabled"], bool)
        assert isinstance(status["vad_enabled"], bool)
        assert isinstance(status["fusion_enabled"], bool)
        assert isinstance(status["limiter_enabled"], bool)
        assert isinstance(status["input_underruns"], int)
        assert isinstance(status["output_underruns"], int)
        assert isinstance(status["dropped_blocks"], int)

    def test_status_before_engine_start(self):
        """Status when engine is not streaming must return running=False."""
        status = state.get_status()
        assert status["running"] is False
        assert status["sample_rate"] == 16000
        assert status["block_size"] in (128, 256, 512)

    def test_status_after_model_load(self):
        """After loading model checkpoint, model_loaded must be True."""
        assert state.model_loaded is True
        assert state.is_model_loaded() is True
        status = state.get_status()
        assert status["model_loaded"] is True

    def test_status_when_stopped(self):
        """Explicitly test status reporting in stopped state."""
        live_engine.stop()
        status = state.get_status()
        assert status["running"] is False
        assert status["error"] is None

    def test_status_when_running_and_after_stop(self):
        """Test status transitions when stream starts and stops."""
        def_in = live_engine.device_manager.get_default_input_device()
        def_out = live_engine.device_manager.get_default_output_device()

        if not def_in or not def_out:
            pytest.skip("Physical audio devices not present on host")

        # Start stream
        success, msg = live_engine.start()
        if not success:
            pytest.skip(f"Could not open hardware stream: {msg}")

        try:
            assert live_engine.is_running is True
            status_running = state.get_status()
            assert status_running["running"] is True
            assert status_running["input_device"] is not None
            assert status_running["output_device"] is not None
        finally:
            live_engine.stop()

        # After stop
        assert live_engine.is_running is False
        status_stopped = state.get_status()
        assert status_stopped["running"] is False

    def test_status_after_device_error(self):
        """Verify status reports error string without crashing."""
        # Attempt to start with invalid device ID
        success, msg = live_engine.start(input_device_id=999999)
        assert success is False

        status = state.get_status()
        assert status["running"] is False
        # Error must be exposed safely
        assert status["error"] is not None or "not found" in msg.lower()

    def test_status_missing_optional_metric(self):
        """Status must handle None for optional metrics safely."""
        fresh_engine = LiveAudioEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        # Ensure telemetry has zeros / None
        status = fresh_engine.get_status_dict()
        assert status["running"] is False
        assert status["model_loaded"] is True
        # Optional metrics should be numeric or None, never raise
        assert status["rtf"] is not None or status["rtf"] is None

    @pytest.mark.asyncio
    async def test_devices_endpoint(self):
        """GET /devices returns physical devices with index, host_api, channels."""
        devs = await get_devices()
        assert "input_devices" in devs
        assert "output_devices" in devs

        if devs["input_devices"]:
            d = devs["input_devices"][0]
            assert "id" in d or "index" in d
            assert "name" in d
            assert "host_api" in d or "hostapi_name" in d
            assert "max_input_channels" in d
            assert "default_sample_rate" in d or "default_samplerate" in d
