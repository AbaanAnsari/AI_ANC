"""
scripts/benchmark_live_stream.py
================================
Sustained Real-Time Hardware Audio Streaming Stress Test.

Measures genuine physical-hardware streaming performance over 60 seconds:
    - Audio capture through physical input device
    - Full streaming DSP + AI pipeline processing
    - Audio delivery through physical output device
    - Latency profiling (mean, p95, p99, max)
    - Real-Time Factor (RTF)
    - Buffer underruns, overruns, dropped blocks
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime.live_audio_engine import LiveAudioEngine, DEFAULT_CHECKPOINT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def run_benchmark(duration_s: float = 60.0, block_size: int = 256) -> dict:
    logger.info("=" * 70)
    logger.info("AETHEL123 SUSTAINED LIVE AUDIO STREAMING BENCHMARK")
    logger.info("=" * 70)
    logger.info("Duration target: %.1f seconds", duration_s)
    logger.info("Block size:      %d samples", block_size)
    logger.info("Checkpoint:      %s", DEFAULT_CHECKPOINT)

    # 1. Gather System Hardware Environment
    sys_info = {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "execution_mode": "CPU",
        "audio_backend": f"sounddevice {getattr(sd, '__version__', 'unknown')} / PortAudio {sd.get_portaudio_version()[1]}",
        "sample_rate": 16000,
        "block_size": block_size,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "model_parameters": 70789,
    }

    # 2. Initialize Engine
    engine = LiveAudioEngine(checkpoint_path=DEFAULT_CHECKPOINT, device="cpu")
    devices = engine.get_devices()

    def_in = engine.device_manager.get_default_input_device()
    def_out = engine.device_manager.get_default_output_device()

    if not def_in or not def_out:
        raise RuntimeError("Missing default audio input or output device on host.")

    sys_info["input_device_id"] = def_in.id
    sys_info["input_device_name"] = def_in.name
    sys_info["input_hostapi"] = def_in.hostapi_name
    sys_info["input_channels"] = def_in.max_input_channels
    sys_info["output_device_id"] = def_out.id
    sys_info["output_device_name"] = def_out.name
    sys_info["output_hostapi"] = def_out.hostapi_name
    sys_info["output_channels"] = def_out.max_output_channels

    # Determine channel configuration
    primary_ch = 0
    ref_ch = 1 if def_in.max_input_channels >= 2 else -1
    is_dual = def_in.max_input_channels >= 2

    sys_info["primary_channel"] = primary_ch
    sys_info["ref_channel"] = ref_ch
    sys_info["is_dual_mic"] = is_dual

    logger.info("Input Device:  [#%d] %s (%s, %d channels)", def_in.id, def_in.name, def_in.hostapi_name, def_in.max_input_channels)
    logger.info("Output Device: [#%d] %s (%s, %d channels)", def_out.id, def_out.name, def_out.hostapi_name, def_out.max_output_channels)
    logger.info("Channels:      M1=Ch %d, M2=Ch %d (Dual-mic: %s)", primary_ch, ref_ch, is_dual)

    # 3. Start Live Stream
    logger.info("Starting live hardware stream...")
    t_start_wall = time.perf_counter()
    success, msg = engine.start(
        input_device_id=def_in.id,
        output_device_id=def_out.id,
        primary_channel=primary_ch,
        ref_channel=ref_ch,
        sample_rate=16000,
        block_size=block_size,
        master_gain=0.4,
    )
    if not success:
        raise RuntimeError(f"Stream startup failed: {msg}")

    # 4. Sustained Monitoring Loop
    checkpoints = []
    cpu_measurements = []
    interval = 1.0
    elapsed = 0.0

    try:
        while elapsed < duration_s:
            time.sleep(interval)
            elapsed = time.perf_counter() - t_start_wall
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_measurements.append(cpu_pct)

            telem = engine.get_telemetry()
            checkpoints.append({
                "elapsed_s": round(elapsed, 1),
                "audio_s": telem["performance"]["total_audio_s"],
                "rtf": telem["performance"]["rtf"],
                "last_ms": telem["performance"]["last_processing_ms"],
                "mean_ms": telem["performance"]["mean_processing_ms"],
                "queue_depth": telem["performance"]["queue_depth"],
                "underflows": telem["performance"]["output_underflows"],
                "overflows": telem["performance"]["input_overflows"],
                "cpu_pct": cpu_pct,
            })

            if int(elapsed) % 10 == 0 or elapsed >= duration_s - 1:
                logger.info(
                    "[%4.1fs / %.1fs] Audio: %.1fs | RTF: %.3fx | Mean Proc: %.2f ms | Queue: %d smp | Underflows: %d | CPU: %.1f%%",
                    elapsed, duration_s,
                    telem["performance"]["total_audio_s"],
                    telem["performance"]["rtf"],
                    telem["performance"]["mean_processing_ms"],
                    telem["performance"]["queue_depth"],
                    telem["performance"]["output_underflows"],
                    cpu_pct,
                )
    finally:
        # 5. Stop Stream Cleanly
        logger.info("Stopping physical audio streams...")
        engine.stop()
        t_total_wall = time.perf_counter() - t_start_wall

    final_telem = engine.get_telemetry()

    # 6. Aggregate Results
    results = {
        "system_info": sys_info,
        "test_parameters": {
            "target_duration_s": duration_s,
            "actual_wall_time_s": round(t_total_wall, 2),
            "sample_rate": 16000,
            "block_size": block_size,
        },
        "performance_metrics": {
            "total_audio_duration_processed_s": final_telem["performance"]["total_audio_s"],
            "total_processing_time_s": final_telem["performance"]["total_processing_s"],
            "real_time_factor": final_telem["performance"]["rtf"],
            "real_time_capable": final_telem["performance"]["rtf"] < 1.0,
            "mean_processing_time_ms": final_telem["performance"]["mean_processing_ms"],
            "p95_processing_time_ms": final_telem["performance"]["p95_processing_ms"],
            "p99_processing_time_ms": final_telem["performance"]["p99_processing_ms"],
            "max_processing_time_ms": final_telem["performance"]["max_processing_ms"],
            "audio_hop_duration_ms": 1000.0 * 128 / 16000,  # 8.0 ms
            "mean_cpu_percent": round(float(np.mean(cpu_measurements)), 2) if cpu_measurements else 0.0,
            "max_cpu_percent": round(float(np.max(cpu_measurements)), 2) if cpu_measurements else 0.0,
        },
        "buffer_health": {
            "input_overflows": final_telem["performance"]["input_overflows"],
            "output_underflows": final_telem["performance"]["output_underflows"],
            "dropped_blocks": final_telem["performance"]["dropped_blocks"],
            "queue_depth_samples_at_exit": final_telem["performance"]["queue_depth"],
        },
        "signal_stats": {
            "input_rms_m1": final_telem["levels"]["input_rms_m1"],
            "input_rms_m2": final_telem["levels"]["input_rms_m2"],
            "output_rms": final_telem["levels"]["output_rms"],
            "last_noise_class": final_telem["dsp"]["noise_class"],
            "confidence": final_telem["dsp"]["confidence"],
            "clipping_events": final_telem["dsp"]["clipping_count"],
        },
    }

    # Save artifact
    out_dir = PROJECT_ROOT / "experiments" / "final_project_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sustained_streaming_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark report to %s", out_path)

    # Print summary
    print()
    print("=" * 70)
    print("SUSTAINED REAL-TIME AUDIO BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Wall-clock Duration:   {results['test_parameters']['actual_wall_time_s']} s")
    print(f"  Audio Processed:       {results['performance_metrics']['total_audio_duration_processed_s']} s")
    print(f"  Total Processing Time: {results['performance_metrics']['total_processing_time_s']} s")
    print(f"  Real-Time Factor:      {results['performance_metrics']['real_time_factor']}x")
    print(f"  RTF < 1.0 (Capable):   {'YES (REAL-TIME PASS)' if results['performance_metrics']['real_time_capable'] else 'NO (FAIL)'}")
    print(f"  Hop Processing Time:   {results['performance_metrics']['mean_processing_time_ms']} ms (Budget: 8.0 ms)")
    print(f"  p95 / p99 Latency:     {results['performance_metrics']['p95_processing_time_ms']} ms / {results['performance_metrics']['p99_processing_time_ms']} ms")
    print(f"  Max Block Latency:     {results['performance_metrics']['max_processing_time_ms']} ms")
    print(f"  Mean CPU Usage:        {results['performance_metrics']['mean_cpu_percent']}%")
    print(f"  Input Overflows:       {results['buffer_health']['input_overflows']}")
    print(f"  Output Underflows:     {results['buffer_health']['output_underflows']}")
    print(f"  Dropped Blocks:        {results['buffer_health']['dropped_blocks']}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    run_benchmark(duration_s=dur)
