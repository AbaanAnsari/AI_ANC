"""
evaluation/evaluate_model.py
==============================
Comprehensive per-sample evaluation on the held-out TEST set.

Computes per-sample and aggregated:
    - SNR (noisy input vs clean, enhanced vs clean)
    - STOI (noisy input vs clean, enhanced vs clean)
    - PESQ (noisy input vs clean, enhanced vs clean)
    - Noise classification accuracy

Pipeline:
    For each test clean record:
        1. Load clean waveform
        2. Sample noise from same-split noise pool
        3. Mix at random SNR from [-5, 0, 5, 10, 15, 20]
        4. Simulate dual-mic (M1=speech+noise, M2=reference)
        5. Run M1 through the FROZEN AI checkpoint → enhanced
        6. Compute SNR(clean, noisy), SNR(clean, enhanced)
        7. Compute STOI(clean, noisy), STOI(clean, enhanced)
        8. Compute PESQ(clean, noisy), PESQ(clean, enhanced)

Rules:
    - TEST set is used ONCE, at the end
    - Checkpoint is FROZEN (Step6 best_checkpoint.pt)
    - No values are fabricated
    - All failures are recorded with reason
    - Resampling is documented
"""
from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
script_dir = str(Path(__file__).resolve().parent)
while script_dir in sys.path:
    sys.path.remove(script_dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
SEGMENT_SAMPLES = 16000          # 1-second segments
DEFAULT_SNR_SET = [-5, 0, 5, 10, 15, 20]
SEED = 20260925

PHASE3_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase3_H" / "best_checkpoint.pt"
STEP6_CHECKPOINT = PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
BEST_CHECKPOINT = PHASE3_CHECKPOINT if PHASE3_CHECKPOINT.exists() else STEP6_CHECKPOINT
TEST_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "test_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"

# Max samples to evaluate (None = all)
MAX_SAMPLES: Optional[int] = None

# ---------------------------------------------------------------------------
# Package availability
# ---------------------------------------------------------------------------

try:
    from pystoi import stoi as _pystoi_fn
    HAS_STOI = True
    import pystoi
    PYSTOI_VERSION = getattr(pystoi, "__version__", "0.4.1")
except ImportError:
    HAS_STOI = False
    PYSTOI_VERSION = "NOT INSTALLED"

try:
    from pesq import pesq as _pesq_fn, PesqError
    HAS_PESQ = True
    import pesq as _pesq_mod
    PESQ_VERSION = "0.0.4"  # pesq 0.0.4 has no __version__
except ImportError:
    HAS_PESQ = False
    PESQ_VERSION = "NOT INSTALLED"
    PesqError = Exception

import torch

# ---------------------------------------------------------------------------
# DSP and data imports
# ---------------------------------------------------------------------------

from src.data.audio_loader import load_audio
from src.data.mixer import mix_signals, calculate_snr_db, DEFAULT_SNR_SET
from src.data.dual_mic import simulate_dual_mic
from src.data.noise_selector import NoiseSelector, NOISE_CLASSES
from src.data.segmenter import segment_audio
from src.features.stft import compute_stft, compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.training_config import set_seed
from src.dsp.gcc_phat import gcc_phat
from src.dsp.kalman import DelayKalmanFilter
from evaluation.snr import compute_snr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def align_lengths(*arrays):
    """Truncate all arrays to the shortest length."""
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)


def _safe_stoi(clean: np.ndarray, degraded: np.ndarray, extended: bool = False) -> Optional[float]:
    """Compute STOI; return None on error."""
    if not HAS_STOI:
        return None
    try:
        c = np.asarray(clean, dtype=np.float64).ravel()
        d = np.asarray(degraded, dtype=np.float64).ravel()
        c, d = align_lengths(c, d)
        if len(c) < 256:
            return None
        # Finite check
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(d))):
            return None
        return float(_pystoi_fn(c, d, SAMPLE_RATE, extended=extended))
    except Exception as e:
        return None


def _safe_pesq(clean: np.ndarray, degraded: np.ndarray) -> tuple:
    """
    Compute PESQ in wideband mode (16 kHz).
    Returns (score_or_None, error_reason_or_None).
    """
    if not HAS_PESQ:
        return None, "pesq not installed"
    try:
        c = np.asarray(clean, dtype=np.float32).ravel()
        d = np.asarray(degraded, dtype=np.float32).ravel()
        c, d = align_lengths(c, d)
        if len(c) < 8000:   # PESQ needs at least ~0.5s
            return None, f"too short: {len(c)} samples"
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(d))):
            return None, "non-finite values"
        score = float(_pesq_fn(SAMPLE_RATE, c, d, "wb"))
        return score, None
    except PesqError as e:
        return None, f"PesqError: {e}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------

def load_frozen_model(checkpoint_path: Path, device: str = "cpu") -> LightweightCNNGRUMaskModel:
    """Load LightweightCNNGRUMaskModel from checkpoint. Freeze in eval mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(torch.device(device))
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Loaded model: %d trainable params (epoch=%s)", params, ckpt.get("epoch", "?"))
    return model, ckpt


def model_enhance(model: LightweightCNNGRUMaskModel, m1_waveform: np.ndarray, device: str) -> tuple:
    """
    Run M1 waveform through the AI model and return enhanced waveform.

    Process: waveform → STFT features → model → enhanced STFT → ISTFT.

    Returns
    -------
    (enhanced_waveform, noise_class_pred, confidence)
    """
    stft_complex = compute_stft(m1_waveform)   # (257, T) complex64
    stft_real = stft_complex.real.astype(np.float32)
    stft_imag = stft_complex.imag.astype(np.float32)
    features = np.stack([stft_real, stft_imag], axis=0)[np.newaxis, :, :, :]  # (1,2,257,T)

    x = torch.from_numpy(features).to(torch.device(device))
    with torch.no_grad():
        enhanced_stft_t, logits_t, mask_t = model(x)

    enhanced_stft_np = enhanced_stft_t.squeeze(0).cpu().numpy()  # (2,257,T)
    logits_np = logits_t.squeeze(0).cpu().numpy()                # (3,)

    # Softmax + argmax
    logits_shifted = logits_np - logits_np.max()
    probs = np.exp(logits_shifted) / (np.sum(np.exp(logits_shifted)) + 1e-12)
    noise_class_pred = int(np.argmax(probs))
    confidence = float(probs[noise_class_pred])

    # ISTFT: reconstruct waveform from enhanced complex STFT
    enhanced_complex = (enhanced_stft_np[0] + 1j * enhanced_stft_np[1]).astype(np.complex64)
    enhanced_waveform = compute_istft(enhanced_complex)  # (N,) float32

    return enhanced_waveform, noise_class_pred, confidence


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_one_sample(
    clean_record: dict,
    noise_selector: NoiseSelector,
    model: LightweightCNNGRUMaskModel,
    device: str,
    rng: np.random.Generator,
    sample_idx: int,
) -> dict:
    """
    Generate and evaluate a single test sample.

    Returns a dict with all metrics, or error information.
    """
    result = {
        "sample_idx": sample_idx,
        "clean_path": clean_record["source_path"],
        "noise_class_true": None,
        "noise_class_pred": None,
        "snr_db_noisy": None,
        "snr_db_enhanced": None,
        "snr_improvement_db": None,
        "stoi_noisy": None,
        "stoi_enhanced": None,
        "stoi_improvement": None,
        "pesq_noisy": None,
        "pesq_enhanced": None,
        "pesq_improvement": None,
        "pesq_noisy_error": None,
        "pesq_enhanced_error": None,
        "status": "ok",
        "error": None,
    }

    try:
        # 1. Load clean audio
        clean_path = DATASET_ROOT / clean_record["source_path"]
        clean_raw = load_audio(clean_path)

        # 2. Choose noise class and sample noise
        nc_idx = int(rng.integers(0, len(NOISE_CLASSES)))
        noise_class = NOISE_CLASSES[nc_idx]
        result["noise_class_true"] = noise_class

        noise_record = noise_selector.select(noise_class, rng=rng)
        noise_path = DATASET_ROOT / noise_record["source_path"]
        noise_raw = load_audio(noise_path)

        # 3. Segment to 1 second
        clean_seg = segment_audio(clean_raw, segment_samples=SEGMENT_SAMPLES, rng=rng)
        noise_seg = segment_audio(noise_raw, segment_samples=SEGMENT_SAMPLES, rng=rng)

        # 4. Mix at random SNR
        snr_idx = int(rng.integers(0, len(DEFAULT_SNR_SET)))
        target_snr = DEFAULT_SNR_SET[snr_idx]
        noisy_m1, scaled_noise = mix_signals(clean_seg, noise_seg, target_snr_db=target_snr, rng=rng)

        # 5. Dual-mic simulation
        delay = int(rng.integers(-20, 21))
        m1, m2 = simulate_dual_mic(
            clean_speech=clean_seg,
            scaled_noise=scaled_noise,
            relative_delay_samples=delay,
            gain_mismatch=1.0,
            filter_coefficients=(0.9, 0.1),
            rng=rng,
        )

        # 6. AI enhancement
        enhanced, noise_class_pred, confidence = model_enhance(model, m1, device)
        result["noise_class_pred"] = noise_class_pred

        # 7. Align lengths: clean, m1 (noisy input), enhanced
        clean_seg, m1_aligned, enhanced_aligned = align_lengths(
            clean_seg.astype(np.float64),
            m1.astype(np.float64),
            enhanced.astype(np.float64),
        )

        # 8. Sanity checks
        assert np.all(np.isfinite(clean_seg)), "clean_seg has non-finite values"
        assert np.all(np.isfinite(m1_aligned)), "noisy has non-finite values"
        assert np.all(np.isfinite(enhanced_aligned)), "enhanced has non-finite values"
        assert len(clean_seg) == len(m1_aligned) == len(enhanced_aligned), "length mismatch"

        # 9. SNR (using E[clean^2] / E[(clean - signal)^2])
        snr_noisy = compute_snr(clean_seg, m1_aligned)
        snr_enhanced = compute_snr(clean_seg, enhanced_aligned)
        result["snr_db_noisy"] = round(float(snr_noisy), 4)
        result["snr_db_enhanced"] = round(float(snr_enhanced), 4)
        result["snr_improvement_db"] = round(float(snr_enhanced - snr_noisy), 4)

        # 10. STOI (pystoi, 16 kHz, standard STOI not extended)
        stoi_noisy = _safe_stoi(clean_seg, m1_aligned, extended=False)
        stoi_enhanced = _safe_stoi(clean_seg, enhanced_aligned, extended=False)
        result["stoi_noisy"] = round(stoi_noisy, 4) if stoi_noisy is not None else None
        result["stoi_enhanced"] = round(stoi_enhanced, 4) if stoi_enhanced is not None else None
        if stoi_noisy is not None and stoi_enhanced is not None:
            result["stoi_improvement"] = round(stoi_enhanced - stoi_noisy, 4)

        # 11. PESQ (pesq 0.0.4, wideband mode 'wb', 16 kHz native — no resampling needed)
        pesq_noisy, pesq_noisy_err = _safe_pesq(clean_seg, m1_aligned)
        pesq_enhanced, pesq_enhanced_err = _safe_pesq(clean_seg, enhanced_aligned)
        result["pesq_noisy"] = round(pesq_noisy, 4) if pesq_noisy is not None else None
        result["pesq_enhanced"] = round(pesq_enhanced, 4) if pesq_enhanced is not None else None
        result["pesq_noisy_error"] = pesq_noisy_err
        result["pesq_enhanced_error"] = pesq_enhanced_err
        if pesq_noisy is not None and pesq_enhanced is not None:
            result["pesq_improvement"] = round(pesq_enhanced - pesq_noisy, 4)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.warning("Sample %d failed: %s", sample_idx, result["error"])

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(results: list) -> dict:
    """Aggregate per-sample results into mean/median/std/min/max."""

    def stats(vals):
        arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
        if len(arr) == 0:
            return {"mean": None, "median": None, "std": None, "min": None, "max": None, "count": 0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "count": len(arr),
        }

    ok_results = [r for r in results if r["status"] == "ok"]
    error_results = [r for r in results if r["status"] == "error"]

    # Classification accuracy
    correct = sum(
        1 for r in ok_results
        if r["noise_class_pred"] is not None and r["noise_class_true"] is not None
        and {0: "stationary", 1: "non-stationary", 2: "impulsive"}.get(r["noise_class_pred"]) == r["noise_class_true"]
    )
    total_classified = len([r for r in ok_results if r["noise_class_pred"] is not None])

    # Per-class accuracy
    class_correct = {"stationary": 0, "non-stationary": 0, "impulsive": 0}
    class_total = {"stationary": 0, "non-stationary": 0, "impulsive": 0}
    class_map = {0: "stationary", 1: "non-stationary", 2: "impulsive"}
    for r in ok_results:
        if r["noise_class_true"] in class_total:
            class_total[r["noise_class_true"]] += 1
            if r["noise_class_pred"] is not None and class_map.get(r["noise_class_pred"]) == r["noise_class_true"]:
                class_correct[r["noise_class_true"]] += 1

    per_class_acc = {
        nc: (round(class_correct[nc] / class_total[nc], 4) if class_total[nc] > 0 else None)
        for nc in class_total
    }

    # Count PESQ failures
    pesq_noisy_failures = sum(1 for r in ok_results if r["pesq_noisy"] is None)
    pesq_enhanced_failures = sum(1 for r in ok_results if r["pesq_enhanced"] is None)

    return {
        "total_samples": len(results),
        "ok_samples": len(ok_results),
        "error_samples": len(error_results),
        "snr_noisy": stats([r["snr_db_noisy"] for r in ok_results]),
        "snr_enhanced": stats([r["snr_db_enhanced"] for r in ok_results]),
        "snr_improvement": stats([r["snr_improvement_db"] for r in ok_results]),
        "stoi_noisy": stats([r["stoi_noisy"] for r in ok_results]),
        "stoi_enhanced": stats([r["stoi_enhanced"] for r in ok_results]),
        "stoi_improvement": stats([r["stoi_improvement"] for r in ok_results]),
        "pesq_noisy": stats([r["pesq_noisy"] for r in ok_results]),
        "pesq_enhanced": stats([r["pesq_enhanced"] for r in ok_results]),
        "pesq_improvement": stats([r["pesq_improvement"] for r in ok_results]),
        "pesq_noisy_failure_count": pesq_noisy_failures,
        "pesq_enhanced_failure_count": pesq_enhanced_failures,
        "classification_accuracy": round(correct / total_classified, 4) if total_classified > 0 else None,
        "classification_correct": correct,
        "classification_total": total_classified,
        "per_class_accuracy": per_class_acc,
        "per_class_total": dict(class_total),
        "stoi_extended": False,
        "stoi_sample_rate": SAMPLE_RATE,
        "pesq_mode": "wb",
        "pesq_sample_rate": SAMPLE_RATE,
        "pesq_resampling": "none (native 16 kHz)",
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    checkpoint_path: Path = BEST_CHECKPOINT,
    test_manifest: Path = TEST_MANIFEST,
    dataset_root: Path = DATASET_ROOT,
    max_samples: Optional[int] = MAX_SAMPLES,
    device: str = "cpu",
    seed: int = SEED,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run full per-sample evaluation on the held-out test set.

    Returns aggregated metrics dict.
    """
    t_start = time.perf_counter()

    set_seed(seed)
    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("AETHEL123 FINAL EVALUATION -- HELD-OUT TEST SET")
    logger.info("=" * 70)
    logger.info("Checkpoint:   %s", checkpoint_path)
    logger.info("Test manifest: %s", test_manifest)
    logger.info("Dataset root:  %s", dataset_root)
    logger.info("pystoi:        %s (STOI available: %s)", PYSTOI_VERSION, HAS_STOI)
    logger.info("pesq:          %s (PESQ available: %s)", PESQ_VERSION, HAS_PESQ)
    logger.info("Device:        %s", device)
    logger.info("Seed:          %d", seed)
    logger.info("-" * 70)

    if not HAS_STOI:
        raise RuntimeError("pystoi is not installed. Cannot proceed without STOI.")
    if not HAS_PESQ:
        raise RuntimeError("pesq is not installed. Cannot proceed without PESQ.")

    # Load model
    model, ckpt = load_frozen_model(checkpoint_path, device=device)
    model_epoch = ckpt.get("epoch", "unknown")

    # Load test manifest
    content = test_manifest.read_text(encoding="utf-8").strip()
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    clean_records = [r for r in records if r.get("source_group") == "clean"]

    logger.info("Clean test records: %d", len(clean_records))

    # Noise selector (test split)
    noise_selector = NoiseSelector(test_manifest, rng=np.random.default_rng(seed + 1))

    # Limit samples if requested
    if max_samples is not None:
        clean_records = clean_records[:max_samples]
        logger.info("Evaluating on %d samples (limited by max_samples)", len(clean_records))
    else:
        logger.info("Evaluating on ALL %d clean test samples", len(clean_records))

    # Per-sample evaluation
    all_results = []
    for i, clean_record in enumerate(clean_records):
        sample_rng = np.random.default_rng(seed + i + 100)
        r = evaluate_one_sample(
            clean_record=clean_record,
            noise_selector=noise_selector,
            model=model,
            device=device,
            rng=sample_rng,
            sample_idx=i,
        )
        all_results.append(r)

        if (i + 1) % 20 == 0 or (i + 1) == len(clean_records):
            ok_so_far = sum(1 for x in all_results if x["status"] == "ok")
            logger.info("Progress: %d/%d  (ok=%d)", i + 1, len(clean_records), ok_so_far)

    # Aggregate
    agg = aggregate_metrics(all_results)
    agg["model_epoch"] = model_epoch
    agg["checkpoint_path"] = str(checkpoint_path)
    agg["evaluation_duration_s"] = round(time.perf_counter() - t_start, 2)

    # Save results
    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / "final_project_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = output_dir / "per_sample_results.jsonl"
    with open(per_sample_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    logger.info("Per-sample results saved: %s", per_sample_path)

    agg_path = output_dir / "aggregated_metrics.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    logger.info("Aggregated metrics saved: %s", agg_path)

    return agg, all_results


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_final_report(agg: dict):
    """Print the mandatory final metric table to stdout."""
    s = agg.get

    def _fmt(d, key="mean"):
        if d is None:
            return "N/A"
        v = d.get(key)
        return f"{v:.4f}" if v is not None else "N/A"

    def _fmt_improvement(d, key="mean"):
        if d is None:
            return "N/A"
        v = d.get(key)
        if v is None:
            return "N/A"
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.4f}"

    print()
    print("=" * 75)
    print("AETHEL123 -- FINAL EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 75)
    print(f"  Checkpoint:          {agg.get('checkpoint_path', 'N/A')}")
    print(f"  Model epoch:         {agg.get('model_epoch', 'N/A')}")
    print(f"  Total samples:       {agg.get('total_samples', 0)}")
    print(f"  OK samples:          {agg.get('ok_samples', 0)}")
    print(f"  Error samples:       {agg.get('error_samples', 0)}")
    print(f"  Evaluation time:     {agg.get('evaluation_duration_s', 0):.1f}s")
    print()
    print("  STOI: pystoi 0.4.1 | 16 kHz | standard STOI (not extended)")
    print("  PESQ: pesq 0.0.4   | 16 kHz | wideband (wb) | no resampling")
    print()
    print("-" * 75)
    print(f"  {'Metric':<20} {'Noisy Input':>12} {'Enhanced':>12} {'Improvement':>12}  {'Target':>10}")
    print("-" * 75)

    snr_imp = agg.get("snr_improvement", {})
    stoi_imp = agg.get("stoi_improvement", {})
    pesq_imp = agg.get("pesq_improvement", {})

    snr_n_mean = agg.get("snr_noisy", {}).get("mean")
    snr_e_mean = agg.get("snr_enhanced", {}).get("mean")
    stoi_n_mean = agg.get("stoi_noisy", {}).get("mean")
    stoi_e_mean = agg.get("stoi_enhanced", {}).get("mean")
    pesq_n_mean = agg.get("pesq_noisy", {}).get("mean")
    pesq_e_mean = agg.get("pesq_enhanced", {}).get("mean")

    def _fv(v): return f"{v:.4f}" if v is not None else "N/A"
    def _fi(v): return (f"+{v:.4f}" if v > 0 else f"{v:.4f}") if v is not None else "N/A"

    snr_imp_v = (snr_e_mean - snr_n_mean) if (snr_e_mean is not None and snr_n_mean is not None) else None
    stoi_imp_v = (stoi_e_mean - stoi_n_mean) if (stoi_e_mean is not None and stoi_n_mean is not None) else None
    pesq_imp_v = (pesq_e_mean - pesq_n_mean) if (pesq_e_mean is not None and pesq_n_mean is not None) else None

    snr_met = "[NOT MET]" if snr_imp_v is None or snr_imp_v < 15 else "[MET]"
    stoi_met = "[NOT MET]" if stoi_e_mean is None or stoi_e_mean < 0.85 else "[MET]"
    pesq_met = "[NOT MET]" if pesq_e_mean is None or pesq_e_mean < 2.5 else "[MET]"

    print(f"  {'SNR (dB)':<20} {_fv(snr_n_mean):>12} {_fv(snr_e_mean):>12} {_fi(snr_imp_v):>12}  {'> +15 dB':>10}  {snr_met}")
    print(f"  {'STOI':<20} {_fv(stoi_n_mean):>12} {_fv(stoi_e_mean):>12} {_fi(stoi_imp_v):>12}  {'> 0.85':>10}  {stoi_met}")
    print(f"  {'PESQ':<20} {_fv(pesq_n_mean):>12} {_fv(pesq_e_mean):>12} {_fi(pesq_imp_v):>12}  {'> 2.5':>10}  {pesq_met}")
    print("-" * 75)

    print()
    print("  Detailed statistics (mean +/- std):")
    print(f"    SNR noisy:       {_fv(snr_n_mean)} +/- {_fv(agg.get('snr_noisy',{}).get('std'))}")
    print(f"    SNR enhanced:    {_fv(snr_e_mean)} +/- {_fv(agg.get('snr_enhanced',{}).get('std'))}")
    print(f"    STOI noisy:      {_fv(stoi_n_mean)} +/- {_fv(agg.get('stoi_noisy',{}).get('std'))}")
    print(f"    STOI enhanced:   {_fv(stoi_e_mean)} +/- {_fv(agg.get('stoi_enhanced',{}).get('std'))}")
    print(f"    PESQ noisy:      {_fv(pesq_n_mean)} +/- {_fv(agg.get('pesq_noisy',{}).get('std'))}")
    print(f"    PESQ enhanced:   {_fv(pesq_e_mean)} +/- {_fv(agg.get('pesq_enhanced',{}).get('std'))}")
    print(f"    PESQ noisy failures:    {agg.get('pesq_noisy_failure_count', 0)}")
    print(f"    PESQ enhanced failures: {agg.get('pesq_enhanced_failure_count', 0)}")
    print()
    print(f"  Classification Accuracy: {agg.get('classification_accuracy', 'N/A')}")
    print(f"    Stationary:     {agg.get('per_class_accuracy',{}).get('stationary','N/A')}")
    print(f"    Non-stationary: {agg.get('per_class_accuracy',{}).get('non-stationary','N/A')}")
    print(f"    Impulsive:      {agg.get('per_class_accuracy',{}).get('impulsive','N/A')}")
    print()
    print("  MANDATORY SUMMARY:")
    print(f"    STOI: noisy={_fv(stoi_n_mean)}, enhanced={_fv(stoi_e_mean)}, improvement={_fi(stoi_imp_v)}")
    print(f"    PESQ: noisy={_fv(pesq_n_mean)}, enhanced={_fv(pesq_e_mean)}, improvement={_fi(pesq_imp_v)}")
    print(f"    SNR:  noisy={_fv(snr_n_mean)}, enhanced={_fv(snr_e_mean)}, improvement={_fi(snr_imp_v)}")
    print()
    print("=" * 75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AETHEL123 Final Evaluation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    args = parser.parse_args()

    agg, all_results = run_evaluation(
        checkpoint_path=Path(args.checkpoint),
        max_samples=args.max_samples,
        device=args.device,
    )
    print_final_report(agg)
