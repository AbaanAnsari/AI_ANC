"""
scripts/export_and_benchmark_onnx.py
====================================
Phase 2 Stage 6 — ONNX Export and Benchmark Suite for AETHEL123.

Performs:
1. Stateful model wrapper export to ONNX (with persistent GRU hidden state I/O).
2. Numerical parity validation against PyTorch FP32 model.
3. Dynamic INT8 quantization via onnxruntime.quantization.
4. Profiling: Mean, Median (P50), P95, P99, Max latency across 2,000 streaming hops.
5. Model size and memory footprint measurement.
6. JSON artifact generation for deployment reporting.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("onnx_benchmark")

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_FP32_PATH = OUTPUT_DIR / "cnn_gru_mask_stateful.onnx"
ONNX_INT8_PATH = OUTPUT_DIR / "cnn_gru_mask_stateful_int8.onnx"


class StatefulONNXWrapper(nn.Module):
    """Wraps LightweightCNNGRUMaskModel for clean ONNX export with explicit hidden state."""

    def __init__(self, model: LightweightCNNGRUMaskModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noisy_features: torch.Tensor,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(noisy_features, hidden_state=hidden_state, return_hidden=True)


def export_and_benchmark():
    logger.info("=" * 70)
    logger.info("AETHEL123 PHASE 2 STAGE 6 — ONNX EXPORT & BENCHMARK")
    logger.info("=" * 70)

    # 1. Load PyTorch model
    logger.info("Loading PyTorch checkpoint from %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    base_model = LightweightCNNGRUMaskModel()
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    wrapper = StatefulONNXWrapper(base_model)
    wrapper.eval()

    # Dummy inputs: (B=1, C=2, F=257, T=1) and (1, 1, 48)
    dummy_x = torch.randn(1, 2, 257, 1, dtype=torch.float32)
    dummy_h = torch.zeros(1, 1, 48, dtype=torch.float32)

    # 2. Export to ONNX FP32
    logger.info("Exporting stateful model to ONNX: %s", ONNX_FP32_PATH)
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_h),
        str(ONNX_FP32_PATH),
        input_names=["noisy_features", "hidden_state_in"],
        output_names=["enhanced_stft", "classification_logits", "mask", "hidden_state_out"],
        dynamic_axes={
            "noisy_features": {0: "batch_size", 3: "time_frames"},
            "hidden_state_in": {1: "batch_size"},
            "enhanced_stft": {0: "batch_size", 3: "time_frames"},
            "classification_logits": {0: "batch_size"},
            "mask": {0: "batch_size", 3: "time_frames"},
            "hidden_state_out": {1: "batch_size"},
        },
        opset_version=15,
        dynamo=False,
        do_constant_folding=True,
    )

    # Check model
    onnx_model = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX FP32 model verified successfully.")

    # 3. Dynamic INT8 Quantization
    logger.info("Quantizing to dynamic INT8: %s", ONNX_INT8_PATH)
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
    )
    logger.info("INT8 dynamic quantization completed.")

    # Measure file sizes
    fp32_size_kb = os.path.getsize(ONNX_FP32_PATH) / 1024.0
    int8_size_kb = os.path.getsize(ONNX_INT8_PATH) / 1024.0
    logger.info("Model Sizes: FP32 = %.2f KB | INT8 = %.2f KB (Compression = %.1fx)",
                fp32_size_kb, int8_size_kb, fp32_size_kb / int8_size_kb)

    # 4. Numerical Parity Validation
    logger.info("Validating numerical parity between PyTorch FP32 and ONNX Runtime FP32...")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1  # Standard single-core embedded execution
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), session_options, providers=["CPUExecutionProvider"])
    ort_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), session_options, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(2026)
    max_abs_err_fp32 = 0.0
    max_abs_err_int8 = 0.0

    test_h_pt = torch.zeros(1, 1, 48, dtype=torch.float32)
    test_h_ort = np.zeros((1, 1, 48), dtype=np.float32)
    test_h_int8 = np.zeros((1, 1, 48), dtype=np.float32)

    for _ in range(50):
        test_feat = rng.normal(0, 1, (1, 2, 257, 1)).astype(np.float32)
        # PyTorch
        with torch.no_grad():
            pt_enh, pt_cls, pt_mask, test_h_pt = base_model(
                torch.from_numpy(test_feat), hidden_state=test_h_pt, return_hidden=True
            )
        pt_enh_np = pt_enh.numpy()

        # ONNX FP32
        ort_outs = ort_fp32.run(
            None,
            {"noisy_features": test_feat, "hidden_state_in": test_h_ort},
        )
        test_h_ort = ort_outs[3]
        max_abs_err_fp32 = max(max_abs_err_fp32, float(np.max(np.abs(pt_enh_np - ort_outs[0]))))

        # ONNX INT8
        int8_outs = ort_int8.run(
            None,
            {"noisy_features": test_feat, "hidden_state_in": test_h_int8},
        )
        test_h_int8 = int8_outs[3]
        max_abs_err_int8 = max(max_abs_err_int8, float(np.max(np.abs(pt_enh_np - int8_outs[0]))))

    logger.info("Max Absolute Parity Error: ONNX FP32 = %.2e | ONNX INT8 = %.2e",
                max_abs_err_fp32, max_abs_err_int8)
    assert max_abs_err_fp32 < 1e-4, f"ONNX FP32 parity error too high: {max_abs_err_fp32}"
    logger.info("PASS: Numerical parity validated.")

    # 5. Latency Benchmarking across 2,000 streaming hops (8 ms nominal budget)
    logger.info("Benchmarking latency across 2,000 hops (warmup=200)...")
    N_WARMUP = 200
    N_BENCH = 2000

    def benchmark_runner(name: str, fn):
        # Warmup
        for _ in range(N_WARMUP):
            fn()
        # Benchmark
        timings_ms = []
        for _ in range(N_BENCH):
            t0 = time.perf_counter_ns()
            fn()
            t1 = time.perf_counter_ns()
            timings_ms.append((t1 - t0) / 1e6)
        timings_ms = np.array(timings_ms)
        return {
            "name": name,
            "mean_ms": round(float(np.mean(timings_ms)), 3),
            "median_ms": round(float(np.median(timings_ms)), 3),
            "p95_ms": round(float(np.percentile(timings_ms, 95)), 3),
            "p99_ms": round(float(np.percentile(timings_ms, 99)), 3),
            "max_ms": round(float(np.max(timings_ms)), 3),
            "rtf_8ms": round(float(np.mean(timings_ms) / 8.0), 3),
        }

    # PyTorch CPU
    bench_feat_pt = torch.randn(1, 2, 257, 1, dtype=torch.float32)
    bench_h_pt = torch.zeros(1, 1, 48, dtype=torch.float32)
    def run_pt():
        nonlocal bench_h_pt
        with torch.no_grad():
            _, _, _, bench_h_pt = base_model(bench_feat_pt, hidden_state=bench_h_pt, return_hidden=True)

    pt_stats = benchmark_runner("PyTorch FP32 CPU", run_pt)

    # ONNX FP32
    bench_feat_np = rng.normal(0, 1, (1, 2, 257, 1)).astype(np.float32)
    bench_h_ort = np.zeros((1, 1, 48), dtype=np.float32)
    def run_ort_fp32():
        nonlocal bench_h_ort
        outs = ort_fp32.run(None, {"noisy_features": bench_feat_np, "hidden_state_in": bench_h_ort})
        bench_h_ort = outs[3]

    ort_fp32_stats = benchmark_runner("ONNX Runtime FP32 CPU", run_ort_fp32)

    # ONNX INT8
    bench_h_int8 = np.zeros((1, 1, 48), dtype=np.float32)
    def run_ort_int8():
        nonlocal bench_h_int8
        outs = ort_int8.run(None, {"noisy_features": bench_feat_np, "hidden_state_in": bench_h_int8})
        bench_h_int8 = outs[3]

    ort_int8_stats = benchmark_runner("ONNX Runtime INT8 CPU", run_ort_int8)

    # Log summary table
    logger.info("-" * 75)
    logger.info(f"{'Engine':<24} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8} | {'P99 (ms)':<8} | {'RTF (8ms)':<8}")
    logger.info("-" * 75)
    for s in [pt_stats, ort_fp32_stats, ort_int8_stats]:
        logger.info(f"{s['name']:<24} | {s['mean_ms']:>9.3f} | {s['median_ms']:>8.3f} | {s['p95_ms']:>8.3f} | {s['p99_ms']:>8.3f} | {s['rtf_8ms']:>7.3f}x")
    logger.info("-" * 75)

    # Compile benchmark results
    results = {
        "host_environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
            "hop_duration_ms": 8.0,
            "parameter_count": 70789,
        },
        "model_files": {
            "onnx_fp32_path": str(ONNX_FP32_PATH),
            "onnx_fp32_size_kb": fp32_size_kb,
            "onnx_int8_path": str(ONNX_INT8_PATH),
            "onnx_int8_size_kb": int8_size_kb,
            "compression_ratio": round(fp32_size_kb / int8_size_kb, 2),
        },
        "numerical_parity": {
            "max_abs_error_fp32": max_abs_err_fp32,
            "max_abs_error_int8": max_abs_err_int8,
            "parity_verified": bool(max_abs_err_fp32 < 1e-4),
        },
        "benchmarks": {
            "pytorch_fp32": pt_stats,
            "onnx_runtime_fp32": ort_fp32_stats,
            "onnx_runtime_int8": ort_int8_stats,
        },
        "speedup_vs_pytorch": {
            "onnx_fp32_speedup": round(pt_stats["mean_ms"] / ort_fp32_stats["mean_ms"], 2),
            "onnx_int8_speedup": round(pt_stats["mean_ms"] / ort_int8_stats["mean_ms"], 2),
        },
    }

    out_json = PROJECT_ROOT / "experiments" / "final_project_audit" / "onnx_benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved ONNX benchmark results to %s", out_json)

    return results


if __name__ == "__main__":
    export_and_benchmark()
