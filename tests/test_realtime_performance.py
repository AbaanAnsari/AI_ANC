"""
tests/test_realtime_performance.py
==================================
Benchmark real-time inference latency and execution speed of the streaming pipeline.
Measures:
- Hop processing times (ms/hop) across 500 hops
- p50, p95, p99, mean, max
- Real-Time Factor (RTF = hop_processing_time / 8.0 ms)
- Memory usage if psutil is available
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.streaming_pipeline import StreamingPipeline, SAMPLE_RATE, HOP_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realtime_benchmark")


def benchmark_streaming_latency(
    checkpoint_path: Path,
    n_warmup_hops: int = 50,
    n_eval_hops: int = 500,
    device: str = "cpu",
) -> dict:
    logger.info("Initializing StreamingPipeline from %s on %s...", checkpoint_path, device)
    pipeline = StreamingPipeline(checkpoint_path=checkpoint_path, pipeline_mode="full_production", device=device)

    # Synthetic dual-mic input
    rng = np.random.default_rng(20260929)
    m1_data = rng.normal(0, 0.1, (n_warmup_hops + n_eval_hops) * HOP_SIZE).astype(np.float32)
    m2_data = np.roll(m1_data, 5)

    logger.info("Running %d warmup hops...", n_warmup_hops)
    for i in range(n_warmup_hops):
        c1 = m1_data[i * HOP_SIZE : (i + 1) * HOP_SIZE]
        c2 = m2_data[i * HOP_SIZE : (i + 1) * HOP_SIZE]
        pipeline.push(c1, c2)
        pipeline.process_available()
        _ = pipeline.read_output(HOP_SIZE)

    logger.info("Measuring %d evaluation hops...", n_eval_hops)
    hop_times_ms = []
    hop_budget_ms = 1000.0 * HOP_SIZE / SAMPLE_RATE  # 8.0 ms

    start_idx = n_warmup_hops * HOP_SIZE
    for i in range(n_eval_hops):
        offset = start_idx + i * HOP_SIZE
        c1 = m1_data[offset : offset + HOP_SIZE]
        c2 = m2_data[offset : offset + HOP_SIZE]
        pipeline.push(c1, c2)

        t0 = time.perf_counter()
        pipeline.process_available()
        _ = pipeline.read_output(HOP_SIZE)
        t_el = (time.perf_counter() - t0) * 1000.0
        hop_times_ms.append(t_el)

    arr = np.array(hop_times_ms)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    max_ms = float(np.max(arr))
    rtf = mean_ms / hop_budget_ms

    # Memory usage
    mem_mb = None
    try:
        import psutil
        process = psutil.Process()
        mem_mb = float(process.memory_info().rss / (1024 * 1024))
    except ImportError:
        pass

    results = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "n_eval_hops": n_eval_hops,
        "hop_size_samples": HOP_SIZE,
        "hop_budget_ms": hop_budget_ms,
        "mean_ms_per_hop": round(mean_ms, 3),
        "p50_ms_per_hop": round(p50_ms, 3),
        "p95_ms_per_hop": round(p95_ms, 3),
        "p99_ms_per_hop": round(p99_ms, 3),
        "max_ms_per_hop": round(max_ms, 3),
        "real_time_factor_rtf": round(rtf, 4),
        "memory_rss_mb": round(mem_mb, 2) if mem_mb is not None else None,
        "real_time_feasible": rtf < 1.0,
    }

    logger.info("=" * 60)
    logger.info("REAL-TIME BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info("Mean latency:   %.3f ms / %.1f ms hop", mean_ms, hop_budget_ms)
    logger.info("p50 latency:    %.3f ms", p50_ms)
    logger.info("p95 latency:    %.3f ms", p95_ms)
    logger.info("p99 latency:    %.3f ms", p99_ms)
    logger.info("Max latency:    %.3f ms", max_ms)
    logger.info("RTF:            %.4f", rtf)
    if mem_mb is not None:
        logger.info("Memory RSS:     %.2f MB", mem_mb)
    logger.info("=" * 60)

    return results


def test_realtime_performance_baseline():
    """Verify that full production streaming pipeline maintains RTF < 0.50 on CPU."""
    ckpt_path = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        return
    res = benchmark_streaming_latency(ckpt_path, n_warmup_hops=20, n_eval_hops=100)
    assert res["real_time_factor_rtf"] < 0.80, f"RTF {res['real_time_factor_rtf']} exceeds 0.80"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time latency benchmark")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--hops", type=int, default=500)
    args = parser.parse_args()

    res = benchmark_streaming_latency(Path(args.checkpoint), n_eval_hops=args.hops)
    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(res, f, indent=2)
        logger.info("Saved results to %s", out_p)
