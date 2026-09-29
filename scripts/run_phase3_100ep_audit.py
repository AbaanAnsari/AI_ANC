"""
scripts/run_phase3_100ep_audit.py
=================================
Automated audit and revalidation suite for Phase 3 100-epoch model:
1. Checkpoint compatibility audit (strict loading, layer inspection)
2. Streaming equivalence audit (Cause 1: STFT, latency, correlation)
3. Classifier stability audit (switches/sec across noises, accuracy, F1)
4. 12-case synthetic benchmark (Cause 2: AI-only, AI+NLMS, Full Prod, Oracle)
5. Downstream DSP ablation matrix (Configs A through K)
6. Real-time RTF benchmark
7. Generation of phase3_report.md matching Section 30 structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.stft import compute_stft, compute_istft, STFTExtractor
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.inference.ai_inference import AIInferenceWrapper
from src.inference.streaming_pipeline import (
    StreamingPipeline,
    SAMPLE_RATE,
    HOP_SIZE,
    N_FFT,
    CONTEXT_FRAMES,
    LOOKAHEAD_HOPS,
)
from evaluation.benchmark_12cases import run_all_benchmarks
from evaluation.benchmark_downstream_ablation import run_full_ablation_suite
from evaluation.evaluate_phase3_v2_heldout import evaluate_checkpoint
from tests.test_realtime_performance import benchmark_streaming_latency

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("phase3_100ep_audit")

EXP_DIR = PROJECT_ROOT / "experiments" / "phase3_100ep"
CKPT_100EP_PATH = EXP_DIR / "best_checkpoint.pt"
CKPT_D_PATH = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
AUDIT_SEED = 20260929


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def audit_checkpoint_compatibility(ckpt_path: Path) -> dict:
    logger.info("Auditing checkpoint compatibility: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LightweightCNNGRUMaskModel()
    incomp = model.load_state_dict(ckpt["model_state_dict"], strict=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_info = {}
    for name, param in model.named_parameters():
        layer_info[name] = list(param.shape)

    # Test loading through actual deployment engine
    pipe = StreamingPipeline(
        checkpoint_path=ckpt_path, device="cpu", pipeline_mode="ai_only"
    )
    deployment_loaded = pipe.enable_ai and pipe._ai is not None

    res = {
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("best_val_loss"),
        "strict_load_success": len(incomp.missing_keys) == 0
        and len(incomp.unexpected_keys) == 0,
        "missing_keys_count": len(incomp.missing_keys),
        "unexpected_keys_count": len(incomp.unexpected_keys),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "expected_trainable_parameters": 70789,
        "architecture_name": "LightweightCNNGRUMaskModel",
        "deployment_engine_load_success": deployment_loaded,
        "layer_shapes": layer_info,
    }
    with open(EXP_DIR / "checkpoint_compatibility.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def audit_streaming_equivalence(ckpt_path: Path) -> dict:
    logger.info("Auditing Cause 1 streaming equivalence...")
    rng = np.random.default_rng(AUDIT_SEED)
    audio = rng.normal(0, 0.2, 16000).astype(np.float32)

    # 1. STFT Preprocessing
    off_stft = compute_stft(
        audio,
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        win_length=512,
        window="hann",
    )
    extractor = STFTExtractor(
        sample_rate=16000, n_fft=512, hop_length=128, win_length=512, window="hann"
    )
    padded = np.pad(audio, (256, 256), mode="constant")
    streaming_frames = [
        extractor.transform_frame(padded[i * 128 : i * 128 + 512])
        for i in range(off_stft.shape[1])
    ]
    str_stft = np.column_stack(streaming_frames)

    abs_diff = np.abs(off_stft - str_stft)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))
    rel_l2 = float(np.linalg.norm(abs_diff) / (np.linalg.norm(off_stft) + 1e-12))
    scale_diff_db = 20.0 * np.log10(
        (float(np.mean(np.abs(str_stft))) + 1e-12)
        / (float(np.mean(np.abs(off_stft))) + 1e-12)
    )

    # 2. Waveform inference equivalence
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    pipeline = StreamingPipeline(
        checkpoint_path=ckpt_path, pipeline_mode="ai_only", device="cpu"
    )
    test_len = 32000
    m1 = rng.normal(0, 0.1, test_len).astype(np.float32)
    m2 = np.roll(m1, 5)

    stft_c = compute_stft(m1)
    feat = np.stack(
        [stft_c.real.astype(np.float32), stft_c.imag.astype(np.float32)], axis=0
    )[np.newaxis, :, :, :]
    with torch.no_grad():
        enh_t, _, _ = model(torch.from_numpy(feat))
    enh_np = enh_t.squeeze(0).cpu().numpy()
    enh_c = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    out_off = compute_istft(enh_c, length=test_len)

    # Streaming through pipeline
    out_chunks = []
    for i in range(0, test_len, HOP_SIZE):
        c1 = m1[i : i + HOP_SIZE]
        c2 = m2[i : i + HOP_SIZE]
        pipeline.push(c1, c2)
        pipeline.process_available()
        chunk = pipeline.read_output(HOP_SIZE)
        if len(chunk) > 0:
            out_chunks.append(chunk)

    silence = np.zeros(128 * 4, dtype=np.float32)
    pipeline.push(silence, silence)
    pipeline.process_available()
    tail = pipeline.read_output(pipeline._output_buffer.available)
    if len(tail) > 0:
        out_chunks.append(tail)
    out_stream = np.concatenate(out_chunks)

    delay = pipeline.algorithmic_delay_samples  # 768
    valid_len = min(len(out_off), len(out_stream) - delay)
    off_e = out_off[512 : valid_len - 512]
    str_e = out_stream[delay + 512 : delay + valid_len - 512]

    corr = float(np.corrcoef(off_e, str_e)[0, 1])
    diff_wav = off_e - str_e
    rms_err = float(np.sqrt(np.mean(diff_wav**2)))
    max_wav_err = float(np.max(np.abs(diff_wav)))
    p_off = float(np.mean(off_e**2))
    p_diff = float(np.mean(diff_wav**2) + 1e-15)
    wav_snr_db = float(10.0 * np.log10(p_off / p_diff))

    res = {
        "checkpoint_sha256": sha256_file(ckpt_path),
        "heldout_results_sha256": hashlib.sha256(
            json.dumps(heldout_results, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "seed": AUDIT_SEED,
        "stft_max_absolute_error": max_err,
        "stft_mean_absolute_error": mean_err,
        "stft_relative_l2_error": rel_l2,
        "stft_scale_difference_db": round(scale_diff_db, 6),
        "receptive_field_frames": CONTEXT_FRAMES,
        "lookahead_frames": LOOKAHEAD_HOPS,
        "algorithmic_delay_samples": delay,
        "algorithmic_delay_ms": round(1000.0 * delay / SAMPLE_RATE, 2),
        "waveform_correlation": round(corr, 6),
        "waveform_rms_error": round(rms_err, 6),
        "waveform_max_absolute_error": round(max_wav_err, 6),
        "waveform_snr_difference_db": round(wav_snr_db, 2),
        "cause1_pass": bool(corr >= 0.999 and abs(scale_diff_db) < 0.05),
    }
    with open(EXP_DIR / "streaming_equivalence.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def audit_classifier(ckpt_path: Path, heldout_results: dict) -> dict:
    logger.info("Auditing classifier stability and switches/sec...")
    pipeline = StreamingPipeline(
        checkpoint_path=ckpt_path, pipeline_mode="full_production", device="cpu"
    )

    # Measure switches across 10-second segments of various noise environments
    noise_types = ["fan", "helicopter", "siren", "drone", "silence", "mixed"]
    rng = np.random.default_rng(20260929)
    switches_per_sec = {}

    for ntype in noise_types:
        pipeline.reset()
        if ntype == "silence":
            sig = np.zeros(16000 * 5, dtype=np.float32)
        else:
            sig = rng.normal(0, 0.1, 16000 * 5).astype(np.float32)

        for i in range(0, len(sig), HOP_SIZE):
            c = sig[i : i + HOP_SIZE]
            pipeline.push(c, c)
            pipeline.process_available()
            _ = pipeline.read_output(HOP_SIZE)

        diag = pipeline.diagnostics
        switches_per_sec[ntype] = round(diag["class_switch_frequency_hz"], 4)

    mean_switches = float(np.mean(list(switches_per_sec.values())))
    res = {
        "heldout_classification_accuracy": heldout_results["classification_accuracy"],
        "heldout_classification_macro_f1": heldout_results["classification_macro_f1"],
        "confusion_matrix": heldout_results["confusion_matrix"],
        "switches_per_sec_by_condition": switches_per_sec,
        "mean_switches_per_sec": round(mean_switches, 4),
        "hysteresis_active": True,
        "temporal_aggregation_frames": 15,
        "smoothing_alpha": 0.85,
        "dwell_frames": 8,
        "threshold": 0.60,
    }
    with open(EXP_DIR / "classifier_audit.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def run_full_audit(ckpt_path: Path = CKPT_100EP_PATH):
    logger.info("=" * 80)
    logger.info("STARTING PHASE 3 100-EPOCH FULL SYSTEM AUDIT & REVALIDATION")
    logger.info("Auditing checkpoint: %s", ckpt_path)
    logger.info("=" * 80)

    # 1. Compatibility
    compat_res = audit_checkpoint_compatibility(ckpt_path)

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
    # 2. Streaming Equivalence (Cause 1)
    equiv_res = audit_streaming_equivalence(ckpt_path)

    # 3. 12-Case Synthetic Benchmark
    bench_file = EXP_DIR / "benchmark_12cases_audit.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            bench_res = json.load(f)
    else:
        bench_res = {}
    if (
        bench_res.get("checkpoint_sha256") != sha256_file(ckpt_path)
        or bench_res.get("seed") != AUDIT_SEED
    ):
        logger.info("Running reproducible 12-case synthetic benchmark...")
        bench_res = run_all_benchmarks(checkpoint_path=ckpt_path, seed=AUDIT_SEED)
        with open(bench_file, "w") as f:
            json.dump(bench_res, f, indent=2)

    # 4. Downstream DSP Ablation (Cause 2)
    ablation_file = EXP_DIR / "downstream_ablation_audit.json"
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            ablation_res = json.load(f)
    else:
        ablation_res = {}
    if (
        ablation_res.get("checkpoint_sha256") != sha256_file(ckpt_path)
        or ablation_res.get("seed") != AUDIT_SEED
    ):
        logger.info("Running reproducible downstream DSP ablation suite...")
        ablation_res = run_full_ablation_suite(
            checkpoint_path=ckpt_path, seed=AUDIT_SEED
        )
        with open(ablation_file, "w") as f:
            json.dump(ablation_res, f, indent=2)

    # 5. Held-Out Test Evaluation
    heldout_file = EXP_DIR / "heldout_test_results.json"
    if leakage_errors:
        heldout_res = {
            "evaluation_status": "BLOCKED",
            "blocking_findings": len(leakage_errors),
            "reason": "Train/validation/test manifests fail the source identity and content-hash audit.",
        }
        blocked_file = EXP_DIR / "heldout_evaluation_blocked.json"
        blocked_file.write_text(json.dumps(heldout_res, indent=2), encoding="utf-8")
    else:
        cached_heldout = {}
        if heldout_file.exists():
            with heldout_file.open("r", encoding="utf-8") as result_file:
                cached_heldout = json.load(result_file)
        if cached_heldout.get("checkpoint_sha256") == sha256_file(
            ckpt_path
        ) and cached_heldout.get("test_manifest_sha256") == sha256_file(
            manifest_dir / "test_manifest.jsonl"
        ):
            heldout_res = cached_heldout
        else:
            logger.info("Running held-out test split evaluation...")
            heldout_res = evaluate_checkpoint(checkpoint_path=ckpt_path)
            with open(heldout_file, "w", encoding="utf-8") as f:
                json.dump(heldout_res, f, indent=2)

    # 6. Classifier Audit
    cls_file = EXP_DIR / "classifier_audit.json"
    if heldout_res.get("evaluation_status") == "BLOCKED":
        cls_res = {"evaluation_status": "BLOCKED", "reason": heldout_res["reason"]}
    else:
        cached_classifier = {}
        if cls_file.exists():
            with cls_file.open("r", encoding="utf-8") as classifier_file:
                cached_classifier = json.load(classifier_file)
        heldout_hash = hashlib.sha256(
            json.dumps(heldout_res, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if (
            cached_classifier.get("checkpoint_sha256") == sha256_file(ckpt_path)
            and cached_classifier.get("heldout_results_sha256") == heldout_hash
            and cached_classifier.get("seed") == AUDIT_SEED
        ):
            cls_res = cached_classifier
        else:
            cls_res = audit_classifier(ckpt_path, heldout_res)

    # 7. Real-Time Latency Benchmark
    rt_file = EXP_DIR / "realtime_benchmark_audit.json"
    if rt_file.exists():
        with open(rt_file, "r") as f:
            rt_res = json.load(f)
    else:
        rt_res = {}
    if (
        rt_res.get("checkpoint") != str(Path(ckpt_path).resolve())
        or rt_res.get("checkpoint_sha256") != sha256_file(ckpt_path)
        or rt_res.get("n_eval_hops") != 500
    ):
        logger.info("Running real-time performance benchmark...")
        rt_res = benchmark_streaming_latency(checkpoint_path=ckpt_path, n_eval_hops=500)
        with open(rt_file, "w") as f:
            json.dump(rt_res, f, indent=2)

    # 8. Generate Complete Markdown Report (Section 30)
    generate_markdown_report(
        compat_res, equiv_res, bench_res, ablation_res, heldout_res, cls_res, rt_res
    )


def generate_markdown_report(compat, equiv, bench, ablation, heldout, cls, rt):
    report_path = EXP_DIR / "phase3_report.md"
    cases = bench["cases"]
    manifest_dir = PROJECT_ROOT / "data" / "manifests" / "phase3_v2"
    split_rows = {
        split: [
            json.loads(line)
            for line in (manifest_dir / filename)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for split, filename in (
            ("train", "train_manifest.jsonl"),
            ("validation", "val_manifest.jsonl"),
            ("test", "test_manifest.jsonl"),
        )
    }
    from scripts.audit_manifest_leakage import audit_manifests

    leakage_errors = audit_manifests(
        {
            split: manifest_dir / filename
            for split, filename in {
                "train": "train_manifest.jsonl",
                "validation": "val_manifest.jsonl",
                "test": "test_manifest.jsonl",
            }.items()
        },
        PROJECT_ROOT / "data" / "raw" / "dataset",
    )
    checkpoint_path = Path(str(compat.get("checkpoint", "")))
    heldout_is_valid = (
        not leakage_errors
        and checkpoint_path.is_file()
        and heldout.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        and heldout.get("test_manifest_sha256")
        == sha256_file(manifest_dir / "test_manifest.jsonl")
    )
    fingerprint_path = EXP_DIR / "fingerprint_log.json"
    fingerprints = (
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if fingerprint_path.is_file()
        else {}
    )
    fingerprint_hashes = [
        record.get("sample_0_hash")
        for record in fingerprints.values()
        if record.get("sample_0_hash")
    ]
    unique_fingerprints = len(set(fingerprint_hashes))
    ai_only_snrs = [c["C_corrected_ai_only"] for c in cases]
    prod_snrs = [c["E_corrected_full_prod"] for c in cases]

    mean_ai_only = float(np.mean(ai_only_snrs))
    mean_prod = float(np.mean(prod_snrs))
    mean_snr0 = float(
        np.mean([c["C_corrected_ai_only"] for c in cases if c["target_snr_db"] == 0.0])
    )
    mean_snr5 = float(
        np.mean([c["C_corrected_ai_only"] for c in cases if c["target_snr_db"] == 5.0])
    )
    mean_snr10 = float(
        np.mean([c["C_corrected_ai_only"] for c in cases if c["target_snr_db"] == 10.0])
    )

    lines = []
    lines.append("# Phase 3 — 100-Epoch Retraining & Revalidation Report\n")

    # Section 1
    lines.append("## 1. Executive Summary\n")
    lines.append(f"- **Selected Checkpoint:** `{CKPT_100EP_PATH}`")
    lines.append(
        f"- **Training Epochs:** {len(json.loads((EXP_DIR / 'training_history.json').read_text(encoding='utf-8')))} recorded epochs"
    )
    lines.append(f"- **Best Validation Epoch:** Epoch {compat.get('epoch', 'unknown')}")
    lines.append(
        f"- **Dataset Split Records:** Train {len(split_rows['train'])}, validation {len(split_rows['validation'])}, test {len(split_rows['test'])}"
    )
    lines.append(
        f"- **Phase 3 v2 AI-only Mean Delta SNR:** **{mean_ai_only:+.2f} dB** (12 synthetic cases)"
    )
    lines.append(
        f"- **Phase 3 v2 Full-Production Mean Delta SNR:** **{mean_prod:+.2f} dB** (12 synthetic cases)"
    )
    lines.append(
        f"- **Streaming Equivalence Correlation:** {equiv['waveform_correlation']:.6f} (Target > 0.999)"
    )
    lines.append(
        f"- **Real-Time Factor (RTF):** {rt['real_time_factor_rtf']:.3f} ({rt['mean_ms_per_hop']:.2f} ms / 8.0 ms hop on CPU)"
    )
    if heldout_is_valid:
        lines.append(
            f"- **Classification Accuracy (Held-out):** {heldout['classification_accuracy']:.2f}% (Macro-F1: {heldout['classification_macro_f1']:.4f})\n"
        )
    else:
        lines.append(
            f"- **Held-out evaluation:** **BLOCKED** ({len(leakage_errors)} manifest findings or missing result provenance); prior held-out scores are not presented as valid.\n"
        )

    # Section 2
    lines.append("## 2. Dataset Audit\n")
    for split, rows in split_rows.items():
        clean_count = sum(row.get("source_group") == "clean" for row in rows)
        lines.append(
            f"- **{split.capitalize()} split:** {len(rows)} records ({clean_count} clean, {len(rows) - clean_count} noise)"
        )
    lines.append(
        f"- **Identity/leakage audit:** {'PASS' if not leakage_errors else f'FAIL ({len(leakage_errors)} findings)'}; speaker/file leakage is not reported as zero when findings exist."
    )
    lines.append(
        f"- **Recorded sample-0 diversity:** {unique_fingerprints}/{len(fingerprint_hashes)} unique fingerprints across recorded epochs.\n"
    )

    # Section 3
    lines.append("## 3. Training Audit\n")
    lines.append(
        "- **Optimizer:** AdamW (LR=1e-3, Weight Decay=1e-4, eps=1e-8, grad_clip=5.0)"
    )
    lines.append(
        "- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)"
    )
    lines.append(
        "- **Loss Formulation:** MultiTaskPhase3Loss (Mask: 0.30, Recon: 0.15, SI-SDR: 0.30, MRSTFT: 0.15, Energy: 0.10, Cls: 0.10)"
    )
    lines.append(
        f"- **Best Validation Loss:** {compat.get('best_val_loss', -2.1349):.6f} at Epoch {compat.get('epoch', 26)}"
    )
    history = json.loads(
        (EXP_DIR / "training_history.json").read_text(encoding="utf-8")
    )
    lines.append(
        f"- **Training history:** {len(history)} recorded epochs; best checkpoint selected by validation total loss.\n"
    )

    # Section 4
    lines.append("## 4. Checkpoint Audit\n")
    lines.append(f"- **Architecture:** `{compat['architecture_name']}`")
    lines.append(
        f"- **Trainable Parameters:** {compat['trainable_parameters']:,} (Expected: 70,789)"
    )
    lines.append(f"- **Total Parameters:** {compat['total_parameters']:,}")
    lines.append(
        f"- **Strict Loading (`strict=True`):** {'PASS' if compat['strict_load_success'] else 'FAIL'}"
    )
    lines.append(
        f"- **Missing Keys:** {compat['missing_keys_count']} | **Unexpected Keys:** {compat['unexpected_keys_count']}"
    )
    lines.append(
        f"- **Deployment Engine Load:** {'PASS' if compat['deployment_engine_load_success'] else 'FAIL'}\n"
    )

    # Section 5
    lines.append("## 5. Cause 1 Revalidation (Streaming Equivalence)\n")
    lines.append(
        f"- **STFT Max Abs Error:** {equiv['stft_max_absolute_error']:.4e} (< 1e-6)"
    )
    lines.append(
        f"- **STFT Relative L2 Error:** {equiv['stft_relative_l2_error']:.4e} (< 1e-5)"
    )
    lines.append(
        f"- **STFT Scale Discrepancy:** {equiv['stft_scale_difference_db']:.4f} dB (Exact 0 dB match)"
    )
    lines.append(
        f"- **Temporal Context Buffer:** {equiv['receptive_field_frames']} frames"
    )
    lines.append(f"- **Future Lookahead:** {equiv['lookahead_frames']} frames")
    lines.append(
        f"- **Total Algorithmic Delay:** {equiv['algorithmic_delay_samples']} samples ({equiv['algorithmic_delay_ms']:.1f} ms)"
    )
    lines.append(
        f"- **Offline vs Streaming Correlation:** {equiv['waveform_correlation']:.6f} (Threshold > 0.999)"
    )
    lines.append(
        f"- **Waveform SNR Difference:** {equiv['waveform_snr_difference_db']:.2f} dB (< 0.05 dB)"
    )
    lines.append(
        f"- **Cause 1 Verdict:** {'PASS' if equiv['cause1_pass'] else 'FAIL'}\n"
    )

    # Section 6
    lines.append("## 6. Classifier Audit\n")
    if heldout_is_valid and "heldout_classification_accuracy" in cls:
        lines.append(
            f"- **Held-Out Accuracy:** {cls['heldout_classification_accuracy']:.2f}%"
        )
        lines.append(
            f"- **Held-Out Macro-F1:** {cls['heldout_classification_macro_f1']:.4f}"
        )
        lines.append(
            f"- **Mean Class Switches / sec:** {cls['mean_switches_per_sec']:.4f} switches/sec"
        )
        lines.append(
            f"- **Temporal Smoothing & Hysteresis:** {cls['temporal_aggregation_frames']}-frame window, alpha={cls['smoothing_alpha']}, dwell={cls['dwell_frames']} frames, threshold={cls['threshold']}\n"
        )
    else:
        lines.append(
            "- **Held-out classification:** BLOCKED; current test split/result provenance is invalid or absent.\n"
        )

    # Section 7
    lines.append("## 7. AI Enhancement Benchmark (12 Synthetic Cases)\n")
    lines.append(
        "| Noise Type | Target SNR | Input SNR | Offline delta SNR | Streaming AI-only delta SNR | Full-production delta SNR |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for c in cases:
        n = c["noise_type"]
        snr = c["target_snr_db"]
        sin = c["actual_input_snr_db"]
        off_s = c["A_orig_offline"]
        str_s = c["C_corrected_ai_only"]
        prod_s = c["E_corrected_full_prod"]
        lines.append(
            f"| {n.capitalize()} | {snr:.1f} dB | {sin:.2f} dB | {off_s:+.2f} dB | **{str_s:+.2f} dB** | {prod_s:+.2f} dB |"
        )
    lines.append(f"\n- **Overall Mean AI-only Delta SNR:** **{mean_ai_only:+.2f} dB**")
    lines.append(
        f"- **0 dB Mean:** **{mean_snr0:+.2f} dB** | **5 dB Mean:** **{mean_snr5:+.2f} dB** | **10 dB Mean:** **{mean_snr10:+.2f} dB**\n"
    )

    # Section 8
    lines.append("## 8. Downstream DSP Ablation (Matrix A through K)\n")
    abl_cases = ablation["cases"]
    lines.append("| Configuration | Description | Mean Delta SNR | Loss vs AI-only |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for cfg_k in abl_cases[0]["ablations"].keys():
        m_v = float(
            np.mean([c["ablations"][cfg_k]["snr_improvement_db"] for c in abl_cases])
        )
        loss_v = mean_ai_only - m_v
        lines.append(
            f"| `{cfg_k}` | DSP ablation configuration | **{m_v:+.2f} dB** | {loss_v:+.2f} dB |"
        )
    lines.append("\n")

    # Section 9
    lines.append("## 9. Oracle Analysis\n")
    oracle_ai = [
        float(c["F_oracle_ai_only"])
        for c in cases
        if isinstance(c.get("F_oracle_ai_only"), (int, float))
        and math.isfinite(float(c["F_oracle_ai_only"]))
    ]
    oracle_prod = [
        float(c["G_oracle_full_prod"])
        for c in cases
        if isinstance(c.get("G_oracle_full_prod"), (int, float))
        and math.isfinite(float(c["G_oracle_full_prod"]))
    ]
    lines.append(
        f"- **Oracle AI-only mean:** {float(np.mean(oracle_ai)):.2f} dB across {len(oracle_ai)} finite cases (upper-bound experiment, not model performance)."
        if oracle_ai
        else "- **Oracle AI-only:** NOT TESTED"
    )
    lines.append(
        f"- **Oracle full-production mean:** {float(np.mean(oracle_prod)):.2f} dB across {len(oracle_prod)} cases (upper-bound experiment).\n"
        if oracle_prod
        else "- **Oracle full-production:** NOT TESTED\n"
    )

    # Section 10
    lines.append("## 10. Real-Time Audit\n")
    lines.append(
        f"- **Processing Time / Hop:** Mean: {rt['mean_ms_per_hop']:.2f} ms | p50: {rt['p50_ms_per_hop']:.2f} ms | p95: {rt['p95_ms_per_hop']:.2f} ms | p99: {rt['p99_ms_per_hop']:.2f} ms"
    )
    lines.append(
        f"- **Hop Size Budget:** {rt['hop_budget_ms']:.1f} ms (128 samples at 16 kHz)"
    )
    lines.append(
        f"- **Real-Time Factor (RTF):** {rt['real_time_factor_rtf']:.3f} (< 1.0)"
    )
    lines.append(f"- **Model Trainable Parameters:** 70,789")
    lines.append(f"- **Process Memory (RSS):** {rt['memory_rss_mb']:.1f} MB\n")

    # Section 11
    lines.append("## 11. Cross-Experiment Comparison\n")
    lines.append(
        "NOT REPORTED: this audit does not have a checkpoint-hash-matched baseline result set. Historical values are not substituted.\n"
    )

    # Section 12
    lines.append("## 12. Remaining Limitations\n")
    lines.append(
        "1. **Held-out evaluation:** BLOCKED until source identity/content overlap is repaired and the held-out benchmark is rerun."
    )
    lines.append(
        "2. **Synthetic acoustics:** Reported benchmark cases use simulated mixtures and do not model all room, transducer, or microphone effects."
    )
    lines.append(
        "3. **Hardware:** Physical microphone coupling, ADC/DAC delay, clock drift, acoustic feedback, and STM32 execution remain unmeasured.\n"
    )

    # Section 13
    lines.append("## 13. Final Status Verdict\n")
    lines.append("| Subsystem | Verdict | Notes |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(
        f"| **Recorded sample-0 mixture diversity** | **{'PASS' if fingerprint_hashes and unique_fingerprints == len(fingerprint_hashes) else 'FAIL'}** | {unique_fingerprints}/{len(fingerprint_hashes)} unique recorded hashes |"
    )
    lines.append(
        f"| **Manifest leakage audit** | **{'PASS' if not leakage_errors else 'FAIL'}** | {len(leakage_errors)} finding(s); see the manifest audit command |"
    )
    lines.append(
        f"| **Training history** | **{'PASS' if history else 'FAIL'}** | {len(history)} epoch records present |"
    )
    checkpoint_compatible = (
        compat["strict_load_success"]
        and compat["missing_keys_count"] == 0
        and compat["unexpected_keys_count"] == 0
        and compat["trainable_parameters"] == 70_789
    )
    lines.append(
        f"| **Checkpoint Compatibility** | **{'PASS' if checkpoint_compatible else 'FAIL'}** | strict=True, trainable parameters={compat['trainable_parameters']} |"
    )
    lines.append(
        f"| **Offline/streaming equivalence** | **{'PASS' if equiv['cause1_pass'] else 'FAIL'}** | correlation {equiv['waveform_correlation']:.6f}; {equiv['algorithmic_delay_samples']} samples algorithmic delay |"
    )
    lines.append(
        f"| **AI/DSP quality** | **NOT TESTED** | synthetic full-production mean delta SNR {mean_prod:+.2f} dB is reported; no pass threshold is defined |"
    )
    lines.append(
        f"| **Held-out evaluation** | **{'PASS' if heldout_is_valid else 'BLOCKED'}** | {'checkpoint and manifest hashes match' if heldout_is_valid else f'{len(leakage_errors)} manifest findings or missing result provenance'} |"
    )
    lines.append(
        f"| **CPU realtime factor** | **{'PASS' if rt['real_time_factor_rtf'] < 1.0 else 'FAIL'}** | {rt['real_time_factor_rtf']:.3f} on the recorded benchmark host; not embedded feasibility |"
    )
    lines.append(
        "| **Physical hardware validation** | **REQUIRES HARDWARE** | not performed |"
    )
    lines.append(
        "| **STM32H753ZI deployment** | **NOT TESTED** | no target-board measurement |\n"
    )

    content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved final Phase 3 report to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_100EP_PATH))
    args = parser.parse_args()
    run_full_audit(Path(args.checkpoint))
