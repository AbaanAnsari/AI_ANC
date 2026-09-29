"""
scripts/run_phase3_speech_quality.py
======================================
PHASE 3 — Speech Enhancement Quality Optimization.

EXPERIMENT MATRIX:
  A: Baseline reproduction (Step 6 loss, fresh init, 30 epochs)
  B: SI-SDR only (replace spectral loss with time-domain SI-SDR)
  C: Multi-Resolution STFT only
  D: Composite loss (mask + recon + SI-SDR + MRSTFT + energy)
  E: Composite loss + extended mask range [-2, +2]
  F: Composite loss + warm-start from Step 6 checkpoint (fine-tune 15 epochs)
  G: Best variant from D/E/F with 50-epoch extended training

RULES:
  - Model architecture: LightweightCNNGRUMaskModel, 70,789 params (FROZEN)
  - Training set only — validation set for model selection
  - Test set used ONCE at end (via evaluate_model.py)
  - No modifications to evaluation code
  - All results logged faithfully
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader, Phase1Dataset, custom_collate_fn
from src.features.stft import compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.targeted_crm_loss import MultiTaskTargetedCRMLoss
from src.training.phase3_losses import (
    Phase3MultiTaskLoss,
    SISDRLoss,
    MultiResolutionSTFTLoss,
    stft_to_waveform_torch,
)
from src.training.training_config import set_seed
from evaluation.snr import compute_snr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase3")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP_BASE       = PROJECT_ROOT / "experiments"
MANIFEST_DIR   = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST   = MANIFEST_DIR / "val_manifest.jsonl"
DATASET_ROOT   = PROJECT_ROOT / "data" / "raw" / "dataset"
STEP6_CHECKPOINT = EXP_BASE / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"

# Training defaults
SEED         = 123
BATCH_SIZE   = 16
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 5.0
SAMPLE_RATE  = 16000

# STOI/PESQ availability
try:
    from pystoi import stoi as _pystoi_fn
    HAS_STOI = True
except ImportError:
    HAS_STOI = False

try:
    from pesq import pesq as _pesq_fn, PesqError
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    PesqError = Exception


def _safe_stoi(clean, degraded, sr=16000):
    if not HAS_STOI:
        return None
    try:
        c = np.asarray(clean, dtype=np.float64).ravel()
        d = np.asarray(degraded, dtype=np.float64).ravel()
        min_len = min(len(c), len(d))
        c, d = c[:min_len], d[:min_len]
        if len(c) < 256 or not (np.all(np.isfinite(c)) and np.all(np.isfinite(d))):
            return None
        return float(_pystoi_fn(c, d, sr, extended=False))
    except Exception:
        return None


def _safe_pesq(clean, degraded, sr=16000):
    if not HAS_PESQ:
        return None
    try:
        c = np.asarray(clean, dtype=np.float32).ravel()
        d = np.asarray(degraded, dtype=np.float32).ravel()
        min_len = min(len(c), len(d))
        c, d = c[:min_len], d[:min_len]
        if len(c) < 8000 or not (np.all(np.isfinite(c)) and np.all(np.isfinite(d))):
            return None
        return float(_pesq_fn(sr, c, d, "wb"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "A": {
        "name": "Baseline Step6 Loss (Control)",
        "loss_type": "step6",
        "epochs": 30,
        "warm_start": None,
        "lr": 1e-3,
        "mask_clamp": (-1.0, 1.0),
        "description": "Reproduce Step 6 training as a controlled baseline.",
    },
    "B": {
        "name": "SI-SDR Dominant",
        "loss_type": "phase3",
        "epochs": 30,
        "warm_start": None,
        "lr": 1e-3,
        "mask_clamp": (-1.0, 1.0),
        "loss_weights": {
            "mask_weight": 0.20,
            "recon_weight": 0.05,
            "sisdr_weight": 0.50,
            "mrstft_weight": 0.15,
            "energy_weight": 0.10,
        },
        "description": "Heavy SI-SDR weighting to push time-domain quality.",
    },
    "C": {
        "name": "Multi-Res STFT Dominant",
        "loss_type": "phase3",
        "epochs": 30,
        "warm_start": None,
        "lr": 1e-3,
        "mask_clamp": (-1.0, 1.0),
        "loss_weights": {
            "mask_weight": 0.20,
            "recon_weight": 0.10,
            "sisdr_weight": 0.10,
            "mrstft_weight": 0.45,
            "energy_weight": 0.15,
        },
        "description": "Heavy multi-resolution STFT to capture spectral structure.",
    },
    "D": {
        "name": "Balanced Composite",
        "loss_type": "phase3",
        "epochs": 30,
        "warm_start": None,
        "lr": 1e-3,
        "mask_clamp": (-1.0, 1.0),
        "loss_weights": {
            "mask_weight": 0.30,
            "recon_weight": 0.15,
            "sisdr_weight": 0.30,
            "mrstft_weight": 0.15,
            "energy_weight": 0.10,
        },
        "description": "Balanced composite of all loss terms.",
    },
    "E": {
        "name": "Composite + Extended Mask [-2,+2]",
        "loss_type": "phase3",
        "epochs": 30,
        "warm_start": None,
        "lr": 1e-3,
        "mask_clamp": (-2.0, 2.0),
        "loss_weights": {
            "mask_weight": 0.30,
            "recon_weight": 0.15,
            "sisdr_weight": 0.30,
            "mrstft_weight": 0.15,
            "energy_weight": 0.10,
        },
        "description": "Same as D but with wider mask target clamp range.",
    },
    "F": {
        "name": "Composite + Warm Start (Step6)",
        "loss_type": "phase3",
        "epochs": 20,
        "warm_start": str(STEP6_CHECKPOINT),
        "lr": 3e-4,
        "mask_clamp": (-1.0, 1.0),
        "loss_weights": {
            "mask_weight": 0.25,
            "recon_weight": 0.10,
            "sisdr_weight": 0.35,
            "mrstft_weight": 0.15,
            "energy_weight": 0.15,
        },
        "description": "Fine-tune Step 6 checkpoint with Phase 3 composite loss.",
    },
    "H": {
        "name": "Composite + Warm Start (30 Epochs)",
        "loss_type": "phase3",
        "epochs": 30,
        "warm_start": str(STEP6_CHECKPOINT),
        "lr": 3e-4,
        "mask_clamp": (-1.0, 1.0),
        "loss_weights": {
            "mask_weight": 0.25,
            "recon_weight": 0.10,
            "sisdr_weight": 0.35,
            "mrstft_weight": 0.15,
            "energy_weight": 0.15,
        },
        "description": "30-epoch fine-tuning of Step 6 checkpoint with Phase 3 composite loss.",
    },
}


# ---------------------------------------------------------------------------
# Create loss function
# ---------------------------------------------------------------------------

def create_loss_fn(experiment: dict, device: str) -> nn.Module:
    """Create the loss function for the given experiment config."""
    loss_type = experiment["loss_type"]
    mask_clamp = experiment.get("mask_clamp", (-1.0, 1.0))

    if loss_type == "step6":
        return MultiTaskTargetedCRMLoss(
            enhancement_weight=1.0,
            classification_weight=0.10,
            mask_weight=0.60,
            recon_weight=0.35,
            energy_weight=0.05,
            l1_weight=0.7,
            l2_weight=0.3,
        ).to(device)
    elif loss_type == "phase3":
        weights = experiment.get("loss_weights", {})
        return Phase3MultiTaskLoss(
            enhancement_weight=1.0,
            classification_weight=0.10,
            mask_weight=weights.get("mask_weight", 0.30),
            recon_weight=weights.get("recon_weight", 0.15),
            sisdr_weight=weights.get("sisdr_weight", 0.30),
            mrstft_weight=weights.get("mrstft_weight", 0.15),
            energy_weight=weights.get("energy_weight", 0.10),
            mask_clamp_min=mask_clamp[0],
            mask_clamp_max=mask_clamp[1],
        ).to(device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ---------------------------------------------------------------------------
# Validation with per-sample metrics
# ---------------------------------------------------------------------------

def validate_with_metrics(
    model: nn.Module,
    val_loader,
    loss_fn: nn.Module,
    device: str,
    loss_type: str,
    num_metric_samples: int = 30,
) -> dict:
    """
    Run validation with SNR, STOI, PESQ on a subset of samples.
    Returns aggregated metrics.
    """
    model.eval()
    total_loss_sum = 0.0
    enh_loss_sum = 0.0
    cls_loss_sum = 0.0
    n_batches = 0

    snr_input_vals = []
    snr_enhanced_vals = []
    stoi_noisy_vals = []
    stoi_enhanced_vals = []
    pesq_noisy_vals = []
    pesq_enhanced_vals = []
    sample_count = 0

    with torch.no_grad():
        for batch in val_loader:
            noisy_features = batch["noisy_features"].to(device)
            target_stft = batch["target_stft"].to(device)
            noise_class_label = batch["noise_class_label"].to(device)

            enhanced_output, classification_logits, mask = model(noisy_features)

            # Compute loss
            if loss_type == "step6":
                loss_dict = loss_fn(
                    enhanced_output, mask, classification_logits,
                    noisy_features, target_stft, noise_class_label,
                )
            else:
                clean_waveform = batch.get("clean_target")
                if clean_waveform is not None:
                    clean_waveform = clean_waveform.to(device)
                loss_dict = loss_fn(
                    enhanced_output, mask, classification_logits,
                    noisy_features, target_stft, noise_class_label,
                    clean_waveform=clean_waveform,
                )

            total_loss_sum += loss_dict["total_loss"].item()
            enh_loss_sum += loss_dict["enhancement_loss"].item()
            cls_loss_sum += loss_dict["classification_loss"].item()
            n_batches += 1

            # Per-sample waveform metrics (on subset)
            if sample_count < num_metric_samples and "clean_target" in batch:
                B = noisy_features.shape[0]
                enhanced_np = enhanced_output.cpu().numpy()
                for i in range(B):
                    if sample_count >= num_metric_samples:
                        break

                    clean_wav = batch["clean_target"][i].cpu().numpy().astype(np.float64)

                    # Noisy waveform from M1
                    noisy_wav = batch["m1_waveform"][i].cpu().numpy().astype(np.float64) if "m1_waveform" in batch else None

                    # Enhanced waveform via ISTFT
                    enh_real = enhanced_np[i, 0]
                    enh_imag = enhanced_np[i, 1]
                    enh_complex = (enh_real + 1j * enh_imag).astype(np.complex64)
                    enhanced_wav = compute_istft(enh_complex, length=len(clean_wav)).astype(np.float64)

                    # Align
                    min_len = min(len(clean_wav), len(enhanced_wav))
                    clean_wav = clean_wav[:min_len]
                    enhanced_wav = enhanced_wav[:min_len]

                    if noisy_wav is not None:
                        noisy_wav = noisy_wav[:min_len].astype(np.float64)
                        snr_input = compute_snr(clean_wav, noisy_wav)
                        snr_input_vals.append(float(snr_input))

                        s_noisy = _safe_stoi(clean_wav, noisy_wav)
                        if s_noisy is not None:
                            stoi_noisy_vals.append(s_noisy)

                        p_noisy = _safe_pesq(clean_wav.astype(np.float32), noisy_wav.astype(np.float32))
                        if p_noisy is not None:
                            pesq_noisy_vals.append(p_noisy)

                    snr_enhanced = compute_snr(clean_wav, enhanced_wav)
                    snr_enhanced_vals.append(float(snr_enhanced))

                    s_enhanced = _safe_stoi(clean_wav, enhanced_wav)
                    if s_enhanced is not None:
                        stoi_enhanced_vals.append(s_enhanced)

                    p_enhanced = _safe_pesq(clean_wav.astype(np.float32), enhanced_wav.astype(np.float32))
                    if p_enhanced is not None:
                        pesq_enhanced_vals.append(p_enhanced)

                    sample_count += 1

    if n_batches == 0:
        return {}

    result = {
        "val_total_loss": total_loss_sum / n_batches,
        "val_enhancement_loss": enh_loss_sum / n_batches,
        "val_classification_loss": cls_loss_sum / n_batches,
        "val_samples_evaluated": sample_count,
    }

    if snr_input_vals:
        result["val_snr_input_mean"] = float(np.mean(snr_input_vals))
    if snr_enhanced_vals:
        result["val_snr_enhanced_mean"] = float(np.mean(snr_enhanced_vals))
        result["val_snr_improvement"] = float(np.mean(snr_enhanced_vals)) - (
            float(np.mean(snr_input_vals)) if snr_input_vals else 0.0
        )
    if stoi_noisy_vals:
        result["val_stoi_noisy_mean"] = float(np.mean(stoi_noisy_vals))
    if stoi_enhanced_vals:
        result["val_stoi_enhanced_mean"] = float(np.mean(stoi_enhanced_vals))
    if pesq_noisy_vals:
        result["val_pesq_noisy_mean"] = float(np.mean(pesq_noisy_vals))
    if pesq_enhanced_vals:
        result["val_pesq_enhanced_mean"] = float(np.mean(pesq_enhanced_vals))

    return result


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id: str,
    experiment: dict,
    device: str = "cpu",
) -> dict:
    """Run a single Phase 3 experiment."""
    exp_dir = EXP_BASE / f"phase3_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 3 EXPERIMENT %s: %s", exp_id, experiment["name"])
    logger.info("  Description: %s", experiment["description"])
    logger.info("  Epochs: %d | LR: %s | Loss: %s", experiment["epochs"], experiment["lr"], experiment["loss_type"])
    logger.info("  Mask clamp: %s", experiment.get("mask_clamp", (-1.0, 1.0)))
    logger.info("  Warm start: %s", experiment.get("warm_start", "None (fresh)"))
    logger.info("=" * 70)

    t_start = time.perf_counter()
    set_seed(SEED)

    # Create model
    model = LightweightCNNGRUMaskModel()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d", params)
    assert params == 70789, f"Expected 70,789 params, got {params}"

    # Warm start
    if experiment.get("warm_start"):
        ckpt_path = Path(experiment["warm_start"])
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded warm-start checkpoint: %s (epoch %s)", ckpt_path.name, ckpt.get("epoch", "?"))
        else:
            logger.warning("Warm-start checkpoint not found: %s — training from scratch!", ckpt_path)

    model.to(device)

    # Create data loaders
    train_loader = create_dataloader(
        TRAIN_MANIFEST, DATASET_ROOT,
        batch_size=BATCH_SIZE, shuffle=True, seed=SEED,
    )
    val_loader = create_dataloader(
        VAL_MANIFEST, DATASET_ROOT,
        batch_size=BATCH_SIZE, shuffle=False, seed=SEED,
    )

    # Create loss
    loss_fn = create_loss_fn(experiment, device)

    # Create optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=experiment["lr"],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )

    # Training loop
    history = []
    best_val_loss = float("inf")
    best_val_snr_imp = float("-inf")
    best_epoch = 0

    epochs = experiment["epochs"]
    loss_type = experiment["loss_type"]

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        train_loss_sum = 0.0
        train_enh_sum = 0.0
        train_cls_sum = 0.0
        n_train = 0

        for batch_idx, batch in enumerate(train_loader):
            noisy_features = batch["noisy_features"].to(device)
            target_stft = batch["target_stft"].to(device)
            noise_class_label = batch["noise_class_label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            enhanced_output, classification_logits, mask = model(noisy_features)

            if loss_type == "step6":
                loss_dict = loss_fn(
                    enhanced_output, mask, classification_logits,
                    noisy_features, target_stft, noise_class_label,
                )
            else:
                clean_waveform = batch.get("clean_target")
                if clean_waveform is not None:
                    clean_waveform = clean_waveform.to(device)
                loss_dict = loss_fn(
                    enhanced_output, mask, classification_logits,
                    noisy_features, target_stft, noise_class_label,
                    clean_waveform=clean_waveform,
                )

            total_loss = loss_dict["total_loss"]

            if not torch.isfinite(total_loss):
                logger.error("NaN/Inf loss at epoch %d batch %d — skipping", epoch, batch_idx)
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            train_loss_sum += loss_dict["total_loss"].item()
            train_enh_sum += loss_dict["enhancement_loss"].item()
            train_cls_sum += loss_dict["classification_loss"].item()
            n_train += 1

        if n_train == 0:
            logger.error("Epoch %d: zero training batches completed!", epoch)
            continue

        # Validation
        val_metrics = validate_with_metrics(model, val_loader, loss_fn, device, loss_type)
        val_loss = val_metrics.get("val_total_loss", float("inf"))
        scheduler.step(val_loss)

        # Track best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "experiment_id": exp_id,
                "experiment_name": experiment["name"],
            }, exp_dir / "best_checkpoint.pt")

        val_snr_imp = val_metrics.get("val_snr_improvement", float("-inf"))
        if val_snr_imp > best_val_snr_imp:
            best_val_snr_imp = val_snr_imp

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }, exp_dir / "last_checkpoint.pt")

        epoch_time = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_total_loss": train_loss_sum / n_train,
            "train_enhancement_loss": train_enh_sum / n_train,
            "train_classification_loss": train_cls_sum / n_train,
            **val_metrics,
            "learning_rate": lr,
            "epoch_time_s": round(epoch_time, 2),
            "is_best": is_best,
        }
        history.append(epoch_record)

        # Log
        snr_str = f"SNR_imp={val_metrics.get('val_snr_improvement', 'N/A'):.2f}" if isinstance(val_metrics.get('val_snr_improvement'), (int, float)) else "SNR_imp=N/A"
        stoi_str = f"STOI={val_metrics.get('val_stoi_enhanced_mean', 'N/A'):.4f}" if isinstance(val_metrics.get('val_stoi_enhanced_mean'), (int, float)) else "STOI=N/A"
        pesq_str = f"PESQ={val_metrics.get('val_pesq_enhanced_mean', 'N/A'):.4f}" if isinstance(val_metrics.get('val_pesq_enhanced_mean'), (int, float)) else "PESQ=N/A"

        logger.info(
            "Epoch %d/%d | loss=%.4f | val_loss=%.4f | %s | %s | %s | lr=%.1e | %.1fs%s",
            epoch, epochs,
            train_loss_sum / n_train, val_loss,
            snr_str, stoi_str, pesq_str,
            lr, epoch_time,
            " ★BEST" if is_best else "",
        )

    total_time = time.perf_counter() - t_start

    # Save history
    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    summary = {
        "experiment_id": exp_id,
        "experiment_name": experiment["name"],
        "description": experiment["description"],
        "loss_type": loss_type,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_snr_improvement": best_val_snr_imp,
        "total_time_s": round(total_time, 2),
        "trainable_params": params,
        "final_metrics": history[-1] if history else {},
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Experiment %s complete. Best epoch: %d, Best val_loss: %.6f", exp_id, best_epoch, best_val_loss)
    logger.info("Time: %.1fs | Results: %s", total_time, exp_dir)

    return summary


# ---------------------------------------------------------------------------
# Comparative report
# ---------------------------------------------------------------------------

def generate_comparison_report(summaries: dict, output_dir: Path) -> None:
    """Generate a markdown comparison report across all experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "PHASE3_COMPARISON.md"

    lines = [
        "# Phase 3 — Speech Enhancement Quality Optimization Results\n",
        "",
        "## Experiment Matrix\n",
        "",
        "| ID | Name | Loss | Epochs | Best Epoch | Val Loss | SNR Imp | STOI | PESQ |",
        "|:---|:-----|:-----|:------:|:----------:|:--------:|:-------:|:----:|:----:|",
    ]

    for exp_id, summary in sorted(summaries.items()):
        fm = summary.get("final_metrics", {})
        snr_imp = fm.get("val_snr_improvement")
        stoi = fm.get("val_stoi_enhanced_mean")
        pesq = fm.get("val_pesq_enhanced_mean")

        lines.append(
            f"| {exp_id} | {summary['experiment_name']} | {summary['loss_type']} | "
            f"{summary['epochs']} | {summary['best_epoch']} | "
            f"{summary['best_val_loss']:.4f} | "
            f"{snr_imp:.2f} dB" if isinstance(snr_imp, (int, float)) else "N/A" + f" | "
            f"{stoi:.4f}" if isinstance(stoi, (int, float)) else "N/A" + f" | "
            f"{pesq:.4f}" if isinstance(pesq, (int, float)) else "N/A" + f" |"
        )

    lines.extend([
        "",
        "## Targets\n",
        "- SNR Improvement: > +15 dB",
        "- Enhanced STOI: > 0.85",
        "- Enhanced PESQ: > 2.50",
        "- Classification: > 75%",
        "- Parameters: < 100,000 (actual: 70,789)",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison report: %s", report_path)

    # Also save as JSON
    json_path = output_dir / "phase3_all_summaries.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 Speech Quality Optimization")
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment IDs to run (e.g., A B D). Default: all (A-F)."
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have results")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("STOI available: %s | PESQ available: %s", HAS_STOI, HAS_PESQ)

    exp_ids = args.experiments or list(EXPERIMENTS.keys())
    logger.info("Experiments to run: %s", exp_ids)

    summaries = {}

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            logger.warning("Unknown experiment ID: %s — skipping", exp_id)
            continue

        exp_dir = EXP_BASE / f"phase3_{exp_id}"
        if args.skip_existing and (exp_dir / "summary.json").exists():
            logger.info("Experiment %s already has results — skipping", exp_id)
            with open(exp_dir / "summary.json") as f:
                summaries[exp_id] = json.load(f)
            continue

        try:
            summary = run_experiment(exp_id, EXPERIMENTS[exp_id], device=device)
            summaries[exp_id] = summary
        except Exception as e:
            logger.error("Experiment %s FAILED: %s", exp_id, e)
            logger.error(traceback.format_exc())
            summaries[exp_id] = {"experiment_id": exp_id, "status": "FAILED", "error": str(e)}

    # Generate comparison report
    report_dir = EXP_BASE / "phase3_analysis"
    generate_comparison_report(summaries, report_dir)

    logger.info("=" * 70)
    logger.info("PHASE 3 COMPLETE — All experiments finished")
    logger.info("Results: %s", report_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
