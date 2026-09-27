"""
scripts/run_phase2_baseline.py
================================
PHASE 2 STEP 1 — Full 30-Epoch Baseline Training Runner & Evaluator

Runs 30-epoch training with exact Phase 1 configuration:
- seed = 123
- batch_size = 16
- epochs = 30
- learning_rate = 0.001
- weight_decay = 0.0001
- gradient_clip_norm = 5.0
- optimizer = AdamW
- scheduler = ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-6)
- loss weights: enh=1.0, cls=0.10, l1=0.7, l2=0.3
- monitor = val_total_loss (min)

Saves all Phase 2 baseline artifacts in `experiments/phase2_baseline/`.
Does NOT overwrite Phase 1 checkpoints.
Does NOT use test set during training.
Loads best validation checkpoint for final test evaluation after 30 epochs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path

# ---- Ensure project root is on sys.path ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from evaluation.metrics import compute_classification_metrics
from evaluation.pesq import pesq_available
from evaluation.stoi import stoi_available
from src.data.dataset import create_dataloader
from src.features.stft import compute_istft
from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.losses import MultiTaskLoss
from src.training.training_config import TrainingConfig, set_seed
from src.training.validation import evaluate, load_model_from_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase2_baseline")

# Paths
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"
EXP_DIR = PROJECT_ROOT / "experiments" / "phase2_baseline"
PHASE1_BEST_CKPT = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"


def run_phase2_baseline_training() -> dict:
    """Execute Phase 2 Step 1: 30-Epoch Baseline Training."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 1 — FULL 30-EPOCH BASELINE TRAINING")
    logger.info("Experiment Directory: %s", EXP_DIR)
    logger.info("=" * 70)

    # 1. Configuration setup
    config = TrainingConfig.create(
        device="auto",
        batch_size=16,
        epochs=30,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip_norm=5.0,
        seed=123,
        num_workers=0,
        validation_frequency=1,
        log_frequency=10,
        mixed_precision=False,
    )

    # Save config.yaml
    config_dict = {
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.gradient_clip_norm,
        "seed": config.seed,
        "num_workers": config.num_workers,
        "validation_frequency": config.validation_frequency,
        "log_frequency": config.log_frequency,
        "mixed_precision": config.mixed_precision,
        "optimizer": "AdamW",
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-6,
        },
        "loss": {
            "enhancement_weight": config.loss.enhancement_weight,
            "classification_weight": config.loss.classification_weight,
            "enhancement_l1_weight": config.loss.enhancement_l1_weight,
            "enhancement_l2_weight": config.loss.enhancement_l2_weight,
        },
        "checkpoint": {
            "checkpoint_dir": str(EXP_DIR),
            "monitor": "val_total_loss",
            "mode": "min",
        },
    }
    with open(EXP_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Environment information
    device = torch.device(config.device)
    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "operating_system": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    logger.info("Environment: Python %s | PyTorch %s | Device: %s", sys.version.split()[0], torch.__version__, device)

    # 2. Set seed & DataLoader creation
    set_seed(config.seed)

    train_loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = create_dataloader(
        manifest_path=VAL_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    logger.info("Data loaded: %d train batches (%d samples) | %d val batches (%d samples)",
                n_train_batches, len(train_loader.dataset), n_val_batches, len(val_loader.dataset))

    # 3. Model & Loss initialization (Fresh init with seed=123)
    model = LightweightCNNTGRUModel().to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model created from fresh seed=123 init. Trainable params: %d", trainable_params)
    assert trainable_params == 70_789, f"Parameter mismatch: expected 70789, got {trainable_params}"

    loss_fn = MultiTaskLoss(
        enhancement_weight=config.loss.enhancement_weight,
        classification_weight=config.loss.classification_weight,
        enhancement_l1_weight=config.loss.enhancement_l1_weight,
        enhancement_l2_weight=config.loss.enhancement_l2_weight,
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # Extract deterministic fixed sample for output amplitude monitoring
    fixed_batch = next(iter(val_loader))
    fixed_noisy_features = fixed_batch["noisy_features"][0:1].to(device)  # (1, 2, 257, T)
    fixed_clean_target = fixed_batch["clean_target"][0].numpy()  # (N,)
    if "m1_waveform" in fixed_batch:
        fixed_noisy_wav = fixed_batch["m1_waveform"][0].numpy()
    else:
        n_real = fixed_batch["noisy_features"][0, 0].numpy()
        n_imag = fixed_batch["noisy_features"][0, 1].numpy()
        fixed_noisy_wav = compute_istft((n_real + 1j * n_imag).astype(np.complex64), length=len(fixed_clean_target))

    # Record fixed sample noisy & clean RMS
    sample_noisy_rms = float(np.sqrt(np.mean(fixed_noisy_wav ** 2)))
    sample_clean_rms = float(np.sqrt(np.mean(fixed_clean_target ** 2)))

    # Tracking variables
    history = []
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    best_val_snr_imp = -float("inf")
    best_val_snr_imp_epoch = -1
    nan_inf_occurred = False
    stop_reason = None

    training_start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.perf_counter()

        # -------------------------------------------------------------
        # TRAIN EPOCH
        # -------------------------------------------------------------
        model.train()
        train_total_loss_sum = 0.0
        train_enh_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        unclipped_norms = []
        clipped_norms = []

        for batch_idx, batch in enumerate(train_loader):
            noisy_features = batch["noisy_features"].to(device, non_blocking=True)
            target_stft = batch["target_stft"].to(device, non_blocking=True)
            noise_class_label = batch["noise_class_label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            enhanced_output, classification_logits = model(noisy_features)
            loss_dict = loss_fn(
                enhanced_output,
                classification_logits,
                target_stft,
                noise_class_label,
            )
            total_loss = loss_dict["total_loss"]

            # Health Check: Loss NaN/Inf
            loss_val = total_loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_inf_occurred = True
                stop_reason = f"NaN/Inf loss encountered at Epoch {epoch}, Batch {batch_idx + 1}: {loss_val}"
                logger.error(stop_reason)
                break

            total_loss.backward()

            # Health Check: Gradient NaN/Inf & Unclipped Norm
            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    g_norm = p.grad.detach().norm(2).item()
                    if math.isnan(g_norm) or math.isinf(g_norm):
                        nan_inf_occurred = True
                        stop_reason = f"NaN/Inf gradient encountered at Epoch {epoch}, Batch {batch_idx + 1}"
                        logger.error(stop_reason)
                        break
                    grad_norms.append(g_norm ** 2)

            if nan_inf_occurred:
                break

            unclipped_norm = math.sqrt(sum(grad_norms)) if grad_norms else 0.0
            unclipped_norms.append(unclipped_norm)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)

            # Clipped norm calculation
            clipped_grad_norms = [p.grad.detach().norm(2).item() ** 2 for p in model.parameters() if p.grad is not None]
            clipped_norm = math.sqrt(sum(clipped_grad_norms)) if clipped_grad_norms else 0.0
            clipped_norms.append(clipped_norm)

            optimizer.step()

            train_total_loss_sum += loss_val
            train_enh_loss_sum += loss_dict["enhancement_loss"].item()
            train_cls_loss_sum += loss_dict["classification_loss"].item()

        if nan_inf_occurred:
            logger.error("Training aborted due to NaN/Inf!")
            break

        avg_train_total_loss = train_total_loss_sum / n_train_batches
        avg_train_enh_loss = train_enh_loss_sum / n_train_batches
        avg_train_cls_loss = train_cls_loss_sum / n_train_batches
        avg_unclipped_norm = float(np.mean(unclipped_norms)) if unclipped_norms else 0.0
        avg_clipped_norm = float(np.mean(clipped_norms)) if clipped_norms else 0.0

        # -------------------------------------------------------------
        # VALIDATION & METRICS EVALUATION
        # -------------------------------------------------------------
        val_eval_res = evaluate(model, val_loader, loss_fn, device=device)

        val_total_loss = val_eval_res["mean_total_loss"]
        val_enh_loss = val_eval_res["mean_enhancement_loss"]
        val_cls_loss = val_eval_res["mean_classification_loss"]

        # Step Scheduler with val_total_loss
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_total_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------------------
        # FIXED SAMPLE OUTPUT AMPLITUDE MONITORING
        # -------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            fixed_enh_out, _ = model(fixed_noisy_features)
            enh_real = fixed_enh_out[0, 0].cpu().numpy()
            enh_imag = fixed_enh_out[0, 1].cpu().numpy()
            fixed_enh_complex = (enh_real + 1j * enh_imag).astype(np.complex64)
            fixed_enh_wav = compute_istft(fixed_enh_complex, length=len(fixed_clean_target))

        sample_enh_rms = float(np.sqrt(np.mean(fixed_enh_wav ** 2)))
        sample_min_enh = float(np.min(fixed_enh_wav))
        sample_max_enh = float(np.max(fixed_enh_wav))
        sample_mean_abs_enh = float(np.mean(np.abs(fixed_enh_wav)))
        sample_enh_to_clean_rms_ratio = sample_enh_rms / (sample_clean_rms + 1e-8)

        epoch_duration = time.perf_counter() - epoch_start_time

        # Check best checkpoint logic (val_total_loss min)
        is_best_loss = val_total_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_total_loss
            best_val_loss_epoch = epoch

            # Save best checkpoint
            ckpt_dict = {
                "epoch": epoch,
                "best_val_loss": val_total_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config_dict,
            }
            torch.save(ckpt_dict, EXP_DIR / "best_checkpoint.pt")
            logger.info("Epoch %2d | New BEST val_total_loss=%.6f — saved best_checkpoint.pt", epoch, val_total_loss)

        # Save last checkpoint every epoch
        ckpt_dict_last = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config_dict,
        }
        torch.save(ckpt_dict_last, EXP_DIR / "last_checkpoint.pt")

        # Track best SNR improvement epoch
        val_snr_imp = val_eval_res["mean_snr_improvement"]
        if val_snr_imp > best_val_snr_imp:
            best_val_snr_imp = val_snr_imp
            best_val_snr_imp_epoch = epoch

        # Record full epoch details
        epoch_record = {
            "epoch": epoch,
            "train_total_loss": float(avg_train_total_loss),
            "train_enhancement_loss": float(avg_train_enh_loss),
            "train_classification_loss": float(avg_train_cls_loss),
            "val_total_loss": float(val_total_loss),
            "val_enhancement_loss": float(val_enh_loss),
            "val_classification_loss": float(val_cls_loss),
            "learning_rate": float(current_lr),
            "epoch_duration_seconds": float(epoch_duration),
            "unclipped_grad_norm": float(avg_unclipped_norm),
            "clipped_grad_norm": float(avg_clipped_norm),
            # Fixed sample output monitoring
            "sample_noisy_rms": sample_noisy_rms,
            "sample_clean_rms": sample_clean_rms,
            "sample_enhanced_rms": sample_enh_rms,
            "sample_min_enhanced": sample_min_enh,
            "sample_max_enhanced": sample_max_enh,
            "sample_mean_abs_enhanced": sample_mean_abs_enh,
            "sample_enhanced_clean_rms_ratio": float(sample_enh_to_clean_rms_ratio),
            # Validation metrics
            "val_classification_accuracy": float(val_eval_res["classification_accuracy"]),
            "val_per_class_accuracy": {int(k): float(v) for k, v in val_eval_res["per_class_accuracy"].items()},
            "val_predicted_class_distribution": {int(k): int(v) for k, v in val_eval_res["class_counts"].items()},
            "val_confusion_matrix": val_eval_res["confusion_matrix"].tolist(),
            "val_input_snr_db": float(val_eval_res["mean_input_snr_db"]),
            "val_enhanced_snr_db": float(val_eval_res["mean_output_snr_db"]),
            "val_snr_improvement_db": float(val_eval_res["mean_snr_improvement"]),
            "val_noisy_stoi": float(val_eval_res["mean_noisy_stoi"]) if val_eval_res["stoi_available"] else "UNAVAILABLE",
            "val_enhanced_stoi": float(val_eval_res["mean_enhanced_stoi"]) if val_eval_res["stoi_available"] else "UNAVAILABLE",
            "val_noisy_pesq": float(val_eval_res["mean_noisy_pesq"]) if val_eval_res["pesq_available"] else "UNAVAILABLE",
            "val_enhanced_pesq": float(val_eval_res["mean_enhanced_pesq"]) if val_eval_res["pesq_available"] else "UNAVAILABLE",
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %2d/30 | Train Loss: %.6f (Enh: %.6f, Cls: %.6f) | Val Loss: %.6f (Enh: %.6f, Cls: %.6f) | "
            "Val Acc: %.2f%% | Val SNR Imp: %.2f dB | LR: %.2e | Enh/Clean RMS: %.4f | %.1fs",
            epoch, avg_train_total_loss, avg_train_enh_loss, avg_train_cls_loss,
            val_total_loss, val_enh_loss, val_cls_loss,
            val_eval_res["classification_accuracy"] * 100,
            val_eval_res["mean_snr_improvement"],
            current_lr, sample_enh_to_clean_rms_ratio, epoch_duration,
        )

    total_training_duration = time.perf_counter() - training_start_time
    logger.info("Training complete. Total time: %.1fs (%.2f mins)", total_training_duration, total_training_duration / 60.0)

    # Save history.json
    with open(EXP_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # -------------------------------------------------------------
    # FINAL TEST EVALUATION (Held-out Test Set)
    # -------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FINAL TEST EVALUATION USING BEST VALIDATION CHECKPOINT")
    logger.info("Loading checkpoint: %s", EXP_DIR / "best_checkpoint.pt")
    logger.info("=" * 70)

    test_model, test_ckpt = load_model_from_checkpoint(EXP_DIR / "best_checkpoint.pt", device=device)
    test_loader = create_dataloader(
        manifest_path=TEST_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    test_eval_res = evaluate(test_model, test_loader, loss_fn, device=device)

    final_test_results = {
        "checkpoint_evaluated": "best_checkpoint.pt",
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_loss": float(best_val_loss),
        "total_loss": float(test_eval_res["mean_total_loss"]),
        "enhancement_loss": float(test_eval_res["mean_enhancement_loss"]),
        "classification_loss": float(test_eval_res["mean_classification_loss"]),
        "input_snr_db": float(test_eval_res["mean_input_snr_db"]),
        "enhanced_snr_db": float(test_eval_res["mean_output_snr_db"]),
        "snr_improvement_db": float(test_eval_res["mean_snr_improvement"]),
        "stoi_available": test_eval_res["stoi_available"],
        "noisy_stoi": float(test_eval_res["mean_noisy_stoi"]) if test_eval_res["stoi_available"] else "UNAVAILABLE",
        "enhanced_stoi": float(test_eval_res["mean_enhanced_stoi"]) if test_eval_res["stoi_available"] else "UNAVAILABLE",
        "pesq_available": test_eval_res["pesq_available"],
        "noisy_pesq": float(test_eval_res["mean_noisy_pesq"]) if test_eval_res["pesq_available"] else "UNAVAILABLE",
        "enhanced_pesq": float(test_eval_res["mean_enhanced_pesq"]) if test_eval_res["pesq_available"] else "UNAVAILABLE",
        "classification_accuracy": float(test_eval_res["classification_accuracy"]),
        "per_class_accuracy": {int(k): float(v) for k, v in test_eval_res["per_class_accuracy"].items()},
        "confusion_matrix": test_eval_res["confusion_matrix"].tolist(),
        "class_counts": {int(k): int(v) for k, v in test_eval_res["class_counts"].items()},
    }

    metrics_dict = {
        "experiment_name": "phase2_baseline",
        "environment": env_info,
        "training_duration_seconds": float(total_training_duration),
        "completed_epochs": len(history),
        "nan_inf_occurred": nan_inf_occurred,
        "stop_reason": stop_reason,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_snr_imp_epoch": best_val_snr_imp_epoch,
        "best_val_snr_imp": float(best_val_snr_imp),
        "final_test_results": final_test_results,
    }

    with open(EXP_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info("Phase 2 Step 1 artifacts successfully saved to: %s", EXP_DIR)
    return metrics_dict


if __name__ == "__main__":
    run_phase2_baseline_training()
