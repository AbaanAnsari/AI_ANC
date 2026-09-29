"""
simulator/backend/api.py
=========================
FastAPI REST & WebSocket API for the AETHEL123 Real-Time Audio Console.

Authoritative endpoints for:
    - /status: Defensive system state contract
    - /devices and /api/devices: Physical hardware device enumeration
    - /start and /api/stream/start: Start physical audio processing
    - /stop and /api/stream/stop: Stop physical audio processing
    - /pause and /gain: Processing controls
    - /ws/live: Live WebSocket telemetry and waveform stream
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("realtime.backend")

try:
    from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, field_validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("FastAPI not installed. Backend API unavailable.")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

from simulator.backend.state import state, SimulatorConfig
from src.realtime.constants import (
    SAMPLE_RATE,
    BLOCK_SIZE,
    BLOCK_DURATION_MS,
    HOP_SIZE,
    N_FFT,
    AUDIO_CONFIG_INDICATOR,
)
from src.realtime.live_audio_engine import LiveAudioEngine

FRONTEND_DIR = PROJECT_ROOT / "simulator" / "frontend"

# Authoritative engine instance
live_engine = LiveAudioEngine()
state.set_live_engine(live_engine)

# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

if HAS_FASTAPI:
    class StreamStartRequest(BaseModel):
        input_device_id: Optional[int] = None
        output_device_id: Optional[int] = None
        primary_channel: int = Field(default=0, ge=0)
        ref_channel: int = Field(default=1, ge=-1)
        sample_rate: int = Field(default=SAMPLE_RATE)
        block_size: int = Field(default=BLOCK_SIZE)
        master_gain: float = Field(default=0.5, ge=0.0, le=1.0)

        @field_validator("block_size")
        def validate_block_size(cls, v: int) -> int:
            if v != BLOCK_SIZE:
                logger.info("Enforcing immutable audio block size %d samples", BLOCK_SIZE)
            return BLOCK_SIZE

    class StreamGainRequest(BaseModel):
        gain: float = Field(..., ge=0.0, le=1.0)

    class StreamPauseRequest(BaseModel):
        paused: bool

    class StreamPassthroughRequest(BaseModel):
        enabled: bool

    class ConfigUpdateRequest(BaseModel):
        enable_ai: Optional[bool] = None
        enable_nlms: Optional[bool] = None
        enable_vad: Optional[bool] = None
        enable_fusion: Optional[bool] = None
        enable_limiter: Optional[bool] = None
        noise_class: Optional[int] = None
        snr_db: Optional[float] = None
        delay_samples: Optional[int] = None
        noise_gain: Optional[float] = None

    class SyntheticProcessRequest(BaseModel):
        duration_s: float = Field(default=1.0, ge=0.1, le=30.0)
        snr_db: float = Field(default=10.0, ge=-30.0, le=60.0)
        noise_class: int = Field(default=0, ge=0, le=2)
        delay_samples: int = Field(default=5, ge=0, le=200)
        noise_gain: float = Field(default=1.0, ge=0.1, le=5.0)
        seed: int = Field(default=42)


# ---------------------------------------------------------------------------
# Module-level Endpoint Functions (Directly Importable for Testing)
# ---------------------------------------------------------------------------

async def get_status() -> dict:
    """
    Defensive status endpoint returning the required hardware/model contract:
    running, model_loaded, input_device, output_device, input_channels, output_channels,
    sample_rate, block_size, ai_enabled, nlms_enabled, vad_enabled, fusion_enabled,
    limiter_enabled, rtf, input_rms, reference_rms, output_rms, input_underruns,
    output_underruns, dropped_blocks, error.
    """
    try:
        return state.get_status()
    except Exception as e:
        logger.error("Unexpected error in get_status: %s", e)
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
            "rtf": 0.0,
            "input_rms": None,
            "reference_rms": None,
            "output_rms": None,
            "input_underruns": 0,
            "output_underruns": 0,
            "dropped_blocks": 0,
            "error": str(e),
        }


async def get_devices() -> dict:
    """Enumerate physical audio input and output devices."""
    return live_engine.get_devices()


async def refresh_devices() -> dict:
    """Re-scan physical audio devices on host."""
    return live_engine.refresh_devices()


async def start_stream(req: Optional[StreamStartRequest] = None):
    """Open physical audio streams and begin processing with immutable 256-sample block."""
    if req is None:
        req = StreamStartRequest()

    success, msg = live_engine.start(
        input_device_id=req.input_device_id,
        output_device_id=req.output_device_id,
        primary_channel=req.primary_channel,
        ref_channel=req.ref_channel,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
        master_gain=req.master_gain,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg, "running": True, "state": state.get_status()}


async def stop_stream():
    """Stop physical audio streams and release hardware handles."""
    live_engine.stop()
    return {"status": "ok", "message": "Stream stopped successfully.", "running": False, "state": state.get_status()}


async def pause_stream(req: StreamPauseRequest):
    """Pause or mute audio processing."""
    live_engine.set_paused(req.paused)
    return {"status": "ok", "paused": req.paused}


async def set_gain(req: StreamGainRequest):
    """Update master output gain."""
    live_engine.set_gain(req.gain)
    return {"status": "ok", "gain": req.gain}


async def get_telemetry():
    """Get live telemetry snapshot."""
    return live_engine.get_telemetry()


async def get_audio_health():
    """Phase 18 real audio health check."""
    return live_engine.get_audio_health()


async def get_audio_input_waveform():
    """Phase 11 real audio input waveform."""
    return live_engine.get_input_waveform(n_samples=512)


async def get_audio_output_waveform():
    """Phase 11 real audio output waveform."""
    return live_engine.get_output_waveform(n_samples=512)


async def get_audio_spectrogram():
    """Phase 12 real audio spectrogram."""
    return live_engine.get_spectrogram()


async def set_passthrough(req: StreamPassthroughRequest):
    """Phase 10 direct mic passthrough diagnostic mode."""
    live_engine.set_passthrough(req.enabled)
    return {"status": "ok", "passthrough_mode": live_engine.passthrough_mode}


async def get_quality_metrics():
    """
    Expose latest validated held-out test evaluation metrics from aggregated_metrics.json.
    """
    metrics_path = PROJECT_ROOT / "experiments" / "final_project_validation" / "aggregated_metrics.json"
    if not metrics_path.exists():
        return {"available": False, "reason": "Evaluation results unavailable"}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "available": True,
            "source": "offline_evaluation",
            "label": "Latest Held-Out Test Evaluation",
            "timestamp": "2026-09-27",
            "samples": data.get("total_samples", 202),
            "stoi": {
                "noisy": round(float(data["stoi_noisy"]["mean"]), 4),
                "enhanced": round(float(data["stoi_enhanced"]["mean"]), 4),
                "delta": round(float(data["stoi_improvement"]["mean"]), 4),
                "target": 0.85,
                "target_met": bool(data["stoi_enhanced"]["mean"] >= 0.85),
            },
            "pesq": {
                "noisy": round(float(data["pesq_noisy"]["mean"]), 4),
                "enhanced": round(float(data["pesq_enhanced"]["mean"]), 4),
                "delta": round(float(data["pesq_improvement"]["mean"]), 4),
                "target": 2.5,
                "target_met": bool(data["pesq_enhanced"]["mean"] >= 2.5),
            },
            "snr_db": {
                "noisy": round(float(data["snr_noisy"]["mean"]), 4),
                "enhanced": round(float(data["snr_enhanced"]["mean"]), 4),
                "delta": round(float(data["snr_improvement"]["mean"]), 4),
                "target": 15.0,
                "target_met": bool(data["snr_enhanced"]["mean"] >= 15.0),
            },
            "classification": {
                "accuracy": round(float(data.get("classification_accuracy", 0.7624)), 4),
                "correct": data.get("classification_correct", 154),
                "total": data.get("classification_total", 202),
                "per_class_accuracy": data.get("per_class_accuracy", {}),
            },
            "model_info": {
                "parameters": 70789,
                "parameter_limit": 100000,
                "model_epoch": data.get("model_epoch", 29),
            }
        }
    except Exception as e:
        logger.error("Error reading aggregated metrics: %s", e)
        return {"available": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_app() -> Any:
    if not HAS_FASTAPI:
        return None

    app = FastAPI(
        title="AETHEL123 Real-Time Audio Console",
        description="True Real-Time Live Audio Processing and Hardware Interface.",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 1. Authoritative Status Endpoint
    app.get("/status")(get_status)

    # 2. Hardware Device Endpoints
    app.get("/devices")(get_devices)
    app.get("/api/devices")(get_devices)
    app.post("/devices/refresh")(refresh_devices)
    app.post("/api/devices/refresh")(refresh_devices)

    # 3. Stream Controls
    app.post("/start")(start_stream)
    app.post("/api/stream/start")(start_stream)
    app.post("/stop")(stop_stream)
    app.post("/api/stream/stop")(stop_stream)
    app.post("/pause")(pause_stream)
    app.post("/api/stream/pause")(pause_stream)
    app.post("/gain")(set_gain)
    app.post("/api/stream/gain")(set_gain)
    app.get("/telemetry")(get_telemetry)
    app.get("/api/stream/telemetry")(get_telemetry)

    # 4. Phase 18 Health Check
    app.get("/audio/health")(get_audio_health)
    app.get("/api/audio/health")(get_audio_health)

    # 5. Phase 11 Real-time Waveforms
    app.get("/audio/input/waveform")(get_audio_input_waveform)
    app.get("/api/stream/waveform/input")(get_audio_input_waveform)
    app.get("/audio/output/waveform")(get_audio_output_waveform)
    app.get("/api/stream/waveform/output")(get_audio_output_waveform)

    # 6. Phase 12 Real-time Spectrogram
    app.get("/audio/spectrogram")(get_audio_spectrogram)
    app.get("/api/stream/spectrogram")(get_audio_spectrogram)

    # 7. Phase 10 Passthrough Diagnostic Mode
    app.post("/audio/passthrough")(set_passthrough)
    app.post("/api/stream/passthrough")(set_passthrough)

    @app.get("/audio/passthrough")
    async def get_passthrough_status():
        return {"passthrough_mode": live_engine.passthrough_mode}

    # 8. Speech Quality Evaluation Endpoint
    app.get("/metrics/quality")(get_quality_metrics)
    app.get("/api/metrics/quality")(get_quality_metrics)

    # 8. WebSocket Live Stream
    @app.websocket("/ws/live")
    async def websocket_live_telemetry(websocket: WebSocket):
        """Broadcast live telemetry snapshots and waveforms at ~25 Hz."""
        await websocket.accept()
        try:
            while True:
                telemetry = live_engine.get_telemetry()
                await websocket.send_text(json.dumps(telemetry))
                await asyncio.sleep(0.04)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WebSocket client disconnected or error: %s", e)

    # 5. Configuration & Legacy Compatibility Endpoints
    @app.get("/config")
    async def config_endpoint():
        return state.get_config()

    @app.post("/config")
    async def update_config_endpoint(config_update: ConfigUpdateRequest):
        updates = {k: v for k, v in config_update.model_dump().items() if v is not None}
        return state.update_config(updates)

    @app.get("/metrics")
    async def metrics_endpoint():
        return live_engine.get_telemetry()

    @app.post("/reset")
    async def reset_endpoint():
        live_engine.stop()
        state.get_engine().configure()
        return {"status": "ok", "message": "Pipeline and streams reset"}

    @app.get("/model/info")
    async def model_info_endpoint():
        return {
            "architecture": "LightweightCNNGRUMaskModel",
            "parameter_count": 70789,
            "checkpoint": str(state.checkpoint_path),
            "model_loaded": state.is_model_loaded,
        }

    @app.get("/config/audio")
    async def audio_config_endpoint():
        return {
            "sample_rate": SAMPLE_RATE,
            "block_size": BLOCK_SIZE,
            "block_duration_ms": BLOCK_DURATION_MS,
            "hop_size": HOP_SIZE,
            "n_fft": N_FFT,
            "indicator": AUDIO_CONFIG_INDICATOR,
        }

    # 6. Static UI Serving
    if FRONTEND_DIR.exists():
        if (FRONTEND_DIR / "css").exists():
            app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
        if (FRONTEND_DIR / "js").exists():
            app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

        @app.get("/")
        @app.get("/overview")
        @app.get("/hardware")
        @app.get("/monitoring")
        @app.get("/performance")
        async def serve_index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


app = create_app()
