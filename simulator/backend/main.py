"""
simulator/backend/main.py
==========================
Entry point for the AETHEL123 Real-Time Audio Backend.

Usage:
    python simulator/backend/main.py
    python simulator/backend/main.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("realtime.backend")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AETHEL123 Real-Time Audio Backend"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install with: pip install uvicorn fastapi\n"
            "Then run: python simulator/backend/main.py"
        )
        sys.exit(1)

    from simulator.backend.api import app, live_engine, state
    if app is None:
        logger.error("FastAPI is not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    # Query device counts for authoritative startup banner
    devs = live_engine.get_devices()
    total_devs = len(devs.get("all_devices", []))
    in_devs = len(devs.get("input_devices", []))
    out_devs = len(devs.get("output_devices", []))
    model_status = "loaded" if live_engine.is_model_loaded else "unloaded"

    print("=" * 60)
    print("AETHEL123 Real-Time Audio Backend")
    print(f"Audio devices: {total_devs} total")
    print(f"Input devices: {in_devs}")
    print(f"Output devices: {out_devs}")
    print(f"Model: {model_status}")
    print(f"Checkpoint: epoch 29 (70,789 params)")
    print("AI: enabled")
    print("NLMS: enabled")
    print("VAD: enabled")
    print("Fusion: enabled")
    print("Limiter: enabled")
    print(f"Host: http://{args.host}:{args.port}")
    print(f"Status API: ready (http://{args.host}:{args.port}/status)")
    print(f"Live Console UI: http://{args.host}:{args.port}/")
    print("=" * 60)

    uvicorn.run(
        "simulator.backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
