"""
src/realtime/live_audio_app.py
==============================
AETHEL123 Real-Time Live Audio Application Server.

FastAPI application providing:
    - Physical device enumeration and validation endpoints
    - Real-time physical audio stream start/stop/pause controls
    - WebSocket streaming of genuine live telemetry, latency, and waveforms
    - Static file serving for the Real-Time Audio Console UI
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime.live_audio_engine import LiveAudioEngine

logger = logging.getLogger("live_audio_app")

# Global singleton LiveAudioEngine
engine = LiveAudioEngine()

app = FastAPI(
    title="AETHEL123 Real-Time Audio Console",
    description="True real-time dual-microphone live audio processing and monitoring.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "simulator" / "frontend"


class StreamStartRequest(BaseModel):
    input_device_id: Optional[int] = None
    output_device_id: Optional[int] = None
    primary_channel: int = Field(default=0, ge=0)
    ref_channel: int = Field(default=1, ge=-1)
    sample_rate: int = Field(default=16000)
    block_size: int = Field(default=256, ge=64, le=2048)
    master_gain: float = Field(default=0.5, ge=0.0, le=1.0)


class StreamGainRequest(BaseModel):
    gain: float = Field(..., ge=0.0, le=1.0)


class StreamPauseRequest(BaseModel):
    paused: bool


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/devices")
async def get_devices():
    """Return physical audio devices detected on the host system."""
    return engine.get_devices()


@app.post("/api/devices/refresh")
async def refresh_devices():
    """Re-scan and enumerate hardware audio devices."""
    return engine.refresh_devices()


@app.post("/api/stream/start")
async def start_stream(req: StreamStartRequest):
    """Open physical audio streams and start real-time processing."""
    success, msg = engine.start(
        input_device_id=req.input_device_id,
        output_device_id=req.output_device_id,
        primary_channel=req.primary_channel,
        ref_channel=req.ref_channel,
        sample_rate=req.sample_rate,
        block_size=req.block_size,
        master_gain=req.master_gain,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "ok", "message": msg, "telemetry": engine.get_telemetry()}


@app.post("/api/stream/stop")
async def stop_stream():
    """Stop physical audio streams and release hardware handles."""
    engine.stop()
    return {"status": "ok", "message": "Stream stopped successfully.", "telemetry": engine.get_telemetry()}


@app.post("/api/stream/pause")
async def pause_stream(req: StreamPauseRequest):
    """Pause or resume audio processing."""
    engine.set_paused(req.paused)
    return {"status": "ok", "paused": req.paused}


@app.post("/api/stream/gain")
async def set_gain(req: StreamGainRequest):
    """Update master output gain."""
    engine.set_gain(req.gain)
    return {"status": "ok", "gain": req.gain}


@app.get("/api/stream/telemetry")
async def get_telemetry():
    """Get instantaneous telemetry snapshot."""
    return engine.get_telemetry()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "engine_running": engine.is_running,
        "checkpoint": str(engine.checkpoint_path),
        "device": engine.device,
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoint for Low-Latency Real-Time Telemetry & Waveforms
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_live_telemetry(websocket: WebSocket):
    """
    Broadcasts live telemetry snapshots (metrics, RTF, waveforms) at ~25 Hz.
    Only transmits while client is connected.
    """
    await websocket.accept()
    logger.info("WebSocket telemetry client connected.")
    try:
        while True:
            telemetry = engine.get_telemetry()
            await websocket.send_text(json.dumps(telemetry))
            # 25 updates per second = 40 ms interval
            await asyncio.sleep(0.04)
    except WebSocketDisconnect:
        logger.info("WebSocket telemetry client disconnected.")
    except Exception as e:
        logger.warning("WebSocket error: %s", e)


# ---------------------------------------------------------------------------
# Static frontend serving
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


def run_app(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run("src.realtime.live_audio_app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_app()
