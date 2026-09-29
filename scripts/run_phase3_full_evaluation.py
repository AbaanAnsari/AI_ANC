"""
scripts/run_phase3_full_evaluation.py
======================================
Automated evaluation pipeline for Phase 3 v2:
1. Strict checkpoint integrity check
2. 12-Case Synthetic Benchmark (AI-only, AI+NLMS, Full Prod, Oracle)
3. Direct comparison against Checkpoint D baseline
4. Held-out test split evaluation (unseen speakers and noise)
5. Downstream DSP ablation matrix (Configs A through K)
6. Real-time RTF benchmark
7. Generation of the Section 44 master comparison table
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

from evaluation.benchmark_12cases import run_all_benchmarks
from evaluation.benchmark_downstream_ablation import run_full_ablation_suite
from evaluation.evaluate_phase3_v2_heldout import evaluate_checkpoint
from tests.test_realtime_performance import benchmark_streaming_latency
from src.project_paths import PHASE3_V2_CHECKPOINT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("full_evaluation")

CKPT_V2_PATH = PHASE3_V2_CHECKPOINT


def run_evaluation(ckpt_path: Path = CKPT_V2_PATH):
    output_dir = Path(ckpt_path).resolve().parent
    from scripts.audit_manifest_leakage import audit_manifests

    manifest_dir = PROJECT_ROOT / "data" / "manifests" / "phase3_v2"
    leakage_errors = audit_manifests(
        {
            "train": manifest_dir / "train_manifest.jsonl",
            "validation": manifest_dir / "val_manifest.jsonl",
            "test": manifest_dir / "test_manifest.jsonl",
        },
        PROJECT_ROOT / "data" / "raw" / "dataset",
    )
    if leakage_errors:
        raise RuntimeError(
            f"Full Phase 3 evaluation blocked by {len(leakage_errors)} manifest leakage/integrity finding(s). "
            f"Run scripts/audit_manifest_leakage.py --manifest-dir {manifest_dir} first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info("RUNNING FULL PHASE 3 V2 EVALUATION PIPELINE")
    logger.info("Evaluating: %s", ckpt_path)
    logger.info("=" * 80)

    # 1. 12-Case Synthetic Benchmark
    logger.info("\n--- 1. 12-CASE SYNTHETIC BENCHMARK ---")
    bench_results = run_all_benchmarks(checkpoint_path=ckpt_path)
    bench_out = output_dir / "benchmark_12cases_results.json"
    with open(bench_out, "w") as f:
        json.dump(bench_results, f, indent=2)
    logger.info("Saved 12-case benchmark to %s", bench_out)

    # 2. Downstream Ablation Matrix (A through K)
    logger.info("\n--- 2. DOWNSTREAM DSP ABLATION MATRIX ---")
    ablation_results = run_full_ablation_suite(checkpoint_path=ckpt_path)
    ablation_out = output_dir / "benchmark_downstream_ablation_results.json"
    with open(ablation_out, "w") as f:
        json.dump(ablation_results, f, indent=2)
    logger.info("Saved downstream ablation to %s", ablation_out)

    # 3. Held-out Test Set Evaluation
    logger.info("\n--- 3. HELD-OUT TEST SPLIT EVALUATION ---")
    heldout_results = evaluate_checkpoint(checkpoint_path=ckpt_path)
    heldout_out = output_dir / "heldout_test_results.json"
    with open(heldout_out, "w") as f:
        json.dump(heldout_results, f, indent=2)
    logger.info("Saved held-out test results to %s", heldout_out)

    # 4. Real-time RTF Benchmark
    logger.info("\n--- 4. REAL-TIME RTF BENCHMARK ---")
    rt_results = benchmark_streaming_latency(checkpoint_path=ckpt_path, n_eval_hops=500)
    rt_out = output_dir / "realtime_benchmark.json"
    with open(rt_out, "w") as f:
        json.dump(rt_results, f, indent=2)
    logger.info("Saved real-time benchmark to %s", rt_out)

    # 5. Summarize values measured by this run; historical values are not embedded here.
    print_summary(bench_results, ablation_results, heldout_results, rt_results)


def print_summary(bench, ablation, heldout, realtime):
    cases = bench["cases"]
    ai_only = np.asarray([case["C_corrected_ai_only"] for case in cases], dtype=float)
    full_production = np.asarray(
        [case["E_corrected_full_prod"] for case in cases], dtype=float
    )
    switches = np.asarray([case["class_switches"] for case in cases], dtype=float)

    print("\nPhase 3 v2 measured results")
    print(f"12-case synthetic AI-only mean delta SNR: {np.mean(ai_only):+.2f} dB")
    print(
        f"12-case synthetic full-production mean delta SNR: {np.mean(full_production):+.2f} dB"
    )
    print(
        f"12-case synthetic classifier switches: {np.mean(switches) / 2.0:.3f} per second"
    )
    print(f"Held-out samples: {heldout['total_samples']}")
    print(f"Held-out mean delta SNR: {heldout['mean_delta_snr_db']:+.2f} dB")
    print(
        f"Held-out classification accuracy: {heldout['classification_accuracy']:.2f}%"
    )
    print(f"Held-out macro-F1: {heldout['classification_macro_f1']:.4f}")
    print(f"Realtime mean time/hop: {realtime['mean_ms_per_hop']:.2f} ms")
    print(f"Realtime factor: {realtime['real_time_factor_rtf']:.3f}")

    ablation_cases = ablation.get("cases", [])
    if ablation_cases:
        print("Ablation mean delta SNR by configuration:")
        for name in ablation_cases[0].get("ablations", {}):
            values = [
                float(case["ablations"][name]["snr_improvement_db"])
                for case in ablation_cases
            ]
            print(f"  {name}: {np.mean(values):+.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_V2_PATH))
    args = parser.parse_args()
    run_evaluation(Path(args.checkpoint))
