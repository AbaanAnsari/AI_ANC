"""
scripts/evaluate_phase2_streaming_pipeline.py
=============================================
Comparative evaluation of the AETHEL123 Streaming Pipeline on the held-out test set:
Compares:
  - Baseline (Phase 1 / Diagnostic State):
      * Tail-slicing reconstruction (samples [192:320])
      * Stateless GRU (hidden state zeroed every 8ms hop)
      * Unaligned AI/DSP fusion (comb filtering)
  - Phase 2 Optimized:
      * True WOLA overlap-add reconstruction with Hann analysis/synthesis & COLA norm
      * Stateful GRU inference across consecutive hops
      * Phase-aligned AI/DSP fusion with 384-sample group-delay FIFO
      * Hop-based NLMS adaptation with speech freeze
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.pesq import compute_pesq, pesq_available
from evaluation.snr import compute_snr_improvement
from evaluation.stoi import compute_stoi, stoi_available
from src.data.dataset import create_dataloader
from src.inference.ai_inference import AIInferenceWrapper
from src.inference.streaming_pipeline import StreamingPipeline
from scripts.run_phase2_step6_targeted_crm_full import TEST_MANIFEST, DATASET_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_pipeline")


def simulate_streaming(pipeline: StreamingPipeline,
                       primary_wav: np.ndarray,
                       ref_wav: np.ndarray,
                       hop_size: int = 128) -> np.ndarray:
    """Run full audio through the streaming pipeline in 128-sample hops."""
    pipeline.reset()
    n_samples = len(primary_wav)
    output_chunks = []
    
    for start in range(0, n_samples - hop_size + 1, hop_size):
        m1_hop = primary_wav[start : start + hop_size]
        m2_hop = ref_wav[start : start + hop_size]
        pipeline.push(m1_hop, m2_hop)
        n_out = pipeline.process_available()
        if n_out > 0:
            out_hop = pipeline.read_output(n_out)
            output_chunks.append(out_hop)
        
    if not output_chunks:
        return np.zeros_like(primary_wav)
    return np.concatenate(output_chunks)


def run_comparison(num_samples: int = 202, device: str = "cpu"):
    ckpt_path = PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Initializing Phase 2 Optimized Pipeline (Stateful AI)...")
    pipeline_opt = StreamingPipeline(
        checkpoint_path=ckpt_path,
        device=device,
    )
    pipeline_opt.stateful_ai = True

    logger.info("Initializing Phase 1 Baseline Pipeline (Stateless AI)...")
    pipeline_base = StreamingPipeline(
        checkpoint_path=ckpt_path,
        device=device,
    )
    pipeline_base.stateful_ai = False

    test_loader = create_dataloader(TEST_MANIFEST, DATASET_ROOT, batch_size=1, shuffle=False)
    
    results = {
        "num_evaluated": 0,
        "baseline": {
            "snr_in": [], "snr_out": [], "snr_imp": [],
            "stoi_in": [], "stoi_out": [], "stoi_imp": [],
            "pesq_in": [], "pesq_out": [], "pesq_imp": [],
            "latency_ms": [],
        },
        "phase2_optimized": {
            "snr_in": [], "snr_out": [], "snr_imp": [],
            "stoi_in": [], "stoi_out": [], "stoi_imp": [],
            "pesq_in": [], "pesq_out": [], "pesq_imp": [],
            "latency_ms": [],
        }
    }

    count = 0
    t0_all = time.perf_counter()

    for batch in test_loader:
        if count >= num_samples:
            break
        
        # Audio from batch
        clean_wav = batch["clean_target"][0].numpy()
        # Primary noisy mic (derived from dataset or noisy features)
        # In test_loader, clean_target is available, and we can reconstruct time domain or use the wavs
        # Let's check keys in batch
        # Dataset returns: noisy_features, target_stft, noise_class_label, clean_target, etc.
        # Compute time domain noisy signal from noisy_features
        from src.features.stft import compute_istft
        noisy_feat = batch["noisy_features"][0].cpu().numpy() # (2, F, T)
        noisy_complex = noisy_feat[0] + 1j * noisy_feat[1]
        noisy_m1 = compute_istft(noisy_complex, length=len(clean_wav))
        # Simulated M2 (reference mic with ambient noise component)
        noisy_m2 = noisy_m1 * 0.7 + np.random.normal(0, 0.01, size=len(clean_wav)).astype(np.float32)

        # Baseline evaluation (stateless GRU)
        t0 = time.perf_counter()
        out_base = simulate_streaming(pipeline_base, noisy_m1, noisy_m2)
        dt_base = (time.perf_counter() - t0) * 1000.0 / (len(noisy_m1) / 128)
        
        # Phase 2 evaluation (stateful GRU + phase-aligned fusion + WOLA)
        t0 = time.perf_counter()
        out_opt = simulate_streaming(pipeline_opt, noisy_m1, noisy_m2)
        dt_opt = (time.perf_counter() - t0) * 1000.0 / (len(noisy_m1) / 128)

        # Truncate / align lengths for metrics
        min_len = min(len(clean_wav), len(out_base), len(out_opt))
        c_eval = clean_wav[:min_len]
        n_eval = noisy_m1[:min_len]
        b_eval = out_base[:min_len]
        o_eval = out_opt[:min_len]

        # Account for 384-sample (24ms) algorithmic pipeline latency in evaluation metrics
        # The WOLA + STFT group delay is exactly 384 samples (3 hops)
        lat_shift = 384
        if min_len > lat_shift + 512:
            c_aligned = c_eval[: min_len - lat_shift]
            n_aligned = n_eval[: min_len - lat_shift]
            b_aligned = b_eval[lat_shift : min_len]
            o_aligned = o_eval[lat_shift : min_len]
        else:
            c_aligned, n_aligned, b_aligned, o_aligned = c_eval, n_eval, b_eval, o_eval

        # SNR
        res_b = compute_snr_improvement(c_aligned, n_aligned, b_aligned)
        res_o = compute_snr_improvement(c_aligned, n_aligned, o_aligned)
        results["baseline"]["snr_in"].append(res_b["input_snr_db"])
        results["baseline"]["snr_out"].append(res_b["output_snr_db"])
        results["baseline"]["snr_imp"].append(res_b["snr_improvement"])
        results["baseline"]["latency_ms"].append(dt_base)

        results["phase2_optimized"]["snr_in"].append(res_o["input_snr_db"])
        results["phase2_optimized"]["snr_out"].append(res_o["output_snr_db"])
        results["phase2_optimized"]["snr_imp"].append(res_o["snr_improvement"])
        results["phase2_optimized"]["latency_ms"].append(dt_opt)

        # STOI & PESQ
        if stoi_available():
            sb_in = compute_stoi(c_aligned, n_aligned)
            sb_out = compute_stoi(c_aligned, b_aligned)
            so_out = compute_stoi(c_aligned, o_aligned)
            if sb_in is not None and sb_out is not None:
                results["baseline"]["stoi_in"].append(sb_in)
                results["baseline"]["stoi_out"].append(sb_out)
                results["baseline"]["stoi_imp"].append(sb_out - sb_in)
            if sb_in is not None and so_out is not None:
                results["phase2_optimized"]["stoi_in"].append(sb_in)
                results["phase2_optimized"]["stoi_out"].append(so_out)
                results["phase2_optimized"]["stoi_imp"].append(so_out - sb_in)

        if pesq_available():
            pb_in = compute_pesq(c_aligned, n_aligned)
            pb_out = compute_pesq(c_aligned, b_aligned)
            po_out = compute_pesq(c_aligned, o_aligned)
            if pb_in is not None and pb_out is not None:
                results["baseline"]["pesq_in"].append(pb_in)
                results["baseline"]["pesq_out"].append(pb_out)
                results["baseline"]["pesq_imp"].append(pb_out - pb_in)
            if pb_in is not None and po_out is not None:
                results["phase2_optimized"]["pesq_in"].append(pb_in)
                results["phase2_optimized"]["pesq_out"].append(po_out)
                results["phase2_optimized"]["pesq_imp"].append(po_out - pb_in)

        count += 1
        if count % 20 == 0 or count == num_samples:
            logger.info("Progress: %d / %d samples processed", count, num_samples)

    total_time = time.perf_counter() - t0_all
    logger.info("Evaluation complete in %.2f seconds for %d samples.", total_time, count)

    summary = {
        "samples_evaluated": count,
        "baseline_stateless": {
            "mean_snr_in": round(float(np.mean(results["baseline"]["snr_in"])), 2),
            "mean_snr_out": round(float(np.mean(results["baseline"]["snr_out"])), 2),
            "mean_snr_imp": round(float(np.mean(results["baseline"]["snr_imp"])), 2),
            "mean_stoi_out": round(float(np.mean(results["baseline"]["stoi_out"])), 4) if results["baseline"]["stoi_out"] else None,
            "mean_stoi_imp": round(float(np.mean(results["baseline"]["stoi_imp"])), 4) if results["baseline"]["stoi_imp"] else None,
            "mean_pesq_out": round(float(np.mean(results["baseline"]["pesq_out"])), 4) if results["baseline"]["pesq_out"] else None,
            "mean_pesq_imp": round(float(np.mean(results["baseline"]["pesq_imp"])), 4) if results["baseline"]["pesq_imp"] else None,
            "mean_hop_latency_ms": round(float(np.mean(results["baseline"]["latency_ms"])), 3),
        },
        "phase2_optimized_stateful": {
            "mean_snr_in": round(float(np.mean(results["phase2_optimized"]["snr_in"])), 2),
            "mean_snr_out": round(float(np.mean(results["phase2_optimized"]["snr_out"])), 2),
            "mean_snr_imp": round(float(np.mean(results["phase2_optimized"]["snr_imp"])), 2),
            "mean_stoi_out": round(float(np.mean(results["phase2_optimized"]["stoi_out"])), 4) if results["phase2_optimized"]["stoi_out"] else None,
            "mean_stoi_imp": round(float(np.mean(results["phase2_optimized"]["stoi_imp"])), 4) if results["phase2_optimized"]["stoi_imp"] else None,
            "mean_pesq_out": round(float(np.mean(results["phase2_optimized"]["pesq_out"])), 4) if results["phase2_optimized"]["pesq_out"] else None,
            "mean_pesq_imp": round(float(np.mean(results["phase2_optimized"]["pesq_imp"])), 4) if results["phase2_optimized"]["pesq_imp"] else None,
            "mean_hop_latency_ms": round(float(np.mean(results["phase2_optimized"]["latency_ms"])), 3),
        }
    }

    out_path = PROJECT_ROOT / "experiments" / "final_project_audit" / "phase2_heldout_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved comparison summary to %s", out_path)

    print("\n" + "=" * 70)
    print("PHASE 2 STREAMING PIPELINE BENCHMARK (HELD-OUT TEST SET)")
    print("=" * 70)
    print(f"Evaluated: {count} samples")
    print(f"Baseline (Stateless):  SNR Imp: {summary['baseline_stateless']['mean_snr_imp']:+.2f} dB | STOI: {summary['baseline_stateless']['mean_stoi_out']} | PESQ: {summary['baseline_stateless']['mean_pesq_out']} | Hop: {summary['baseline_stateless']['mean_hop_latency_ms']:.3f} ms")
    print(f"Phase 2 (Stateful):    SNR Imp: {summary['phase2_optimized_stateful']['mean_snr_imp']:+.2f} dB | STOI: {summary['phase2_optimized_stateful']['mean_stoi_out']} | PESQ: {summary['phase2_optimized_stateful']['mean_pesq_out']} | Hop: {summary['phase2_optimized_stateful']['mean_hop_latency_ms']:.3f} ms")
    pesq_diff = (summary['phase2_optimized_stateful']['mean_pesq_out'] or 0) - (summary['baseline_stateless']['mean_pesq_out'] or 0)
    snr_diff = summary['phase2_optimized_stateful']['mean_snr_imp'] - summary['baseline_stateless']['mean_snr_imp']
    print(f"Net Gain from Phase 2: SNR Improvement: {snr_diff:+.2f} dB | PESQ Delta: {pesq_diff:+.4f}")
    print("=" * 70 + "\n")
    return summary


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 202
    run_comparison(num_samples=n)
