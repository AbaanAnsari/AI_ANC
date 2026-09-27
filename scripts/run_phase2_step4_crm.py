"""
scripts/run_phase2_step4_crm.py
=================================
PHASE 2 STEP 4 — Full 30-Epoch CRM Enhancement Experiment.

Configuration:
  - Model: LightweightCNNGRUMaskModel (CRM, 70,789 params)
  - Epochs: 30
  - seed=123, batch=16, lr=0.001, wd=0.0001, clip=5.0
  - AdamW + ReduceLROnPlateau(min, factor=0.5, patience=3, min_lr=1e-6)
  - Loss: MultiTaskMaskLoss (l1=0.7, l2=0.3, energy=0.1, cls_wt=0.10)
  - Checkpoint dir: experiments/phase2_step4_crm/
  - Test set is held out; evaluated once after epoch 30.

Does NOT overwrite:
  - experiments/phase2_baseline/
  - experiments/phase2_step3/
  - models/checkpoints/best_checkpoint.pt
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

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_classification_metrics, NOISE_CLASS_NAMES
from evaluation.snr import compute_snr_improvement
from evaluation.stoi import compute_stoi, stoi_available
from evaluation.pesq import compute_pesq, pesq_available
from src.data.dataset import create_dataloader
from src.features.stft import compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.mask_losses import MultiTaskMaskLoss
from src.training.training_config import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase2_step4_crm")

EXP_DIR       = PROJECT_ROOT / "experiments" / "phase2_step4_crm"
MANIFEST_DIR  = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST   = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST  = MANIFEST_DIR / "test_manifest.jsonl"
DATASET_ROOT   = PROJECT_ROOT / "data" / "raw" / "dataset"

# Training configuration — same as 30-epoch baseline
SEED           = 123
BATCH_SIZE     = 16
EPOCHS         = 30
LR             = 0.001
WEIGHT_DECAY   = 0.0001
GRAD_CLIP      = 5.0
VAL_FREQ       = 1
LOG_FREQ       = 10

# Loss weights — Step 3 formulation
ENH_WEIGHT     = 1.0
CLS_WEIGHT     = 0.10
L1_WEIGHT      = 0.7
L2_WEIGHT      = 0.3
ENERGY_WEIGHT  = 0.1

# ReduceLROnPlateau
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 3
SCHED_MIN_LR   = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_sample_stats(model, device, fixed_noisy, fixed_clean_wav, fixed_noisy_wav):
    """Compute fixed-sample waveform statistics without modifying the model."""
    model.eval()
    with torch.no_grad():
        enh_out, _, mask_out = model(fixed_noisy.to(device))
    enh_np  = enh_out[0].cpu().numpy()
    mask_np = mask_out[0].cpu().numpy()
    enh_c   = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    enh_wav = compute_istft(enh_c, length=len(fixed_clean_wav))

    noisy_rms  = float(np.sqrt(np.mean(fixed_noisy_wav ** 2)))
    clean_rms  = float(np.sqrt(np.mean(fixed_clean_wav ** 2)))
    enh_rms    = float(np.sqrt(np.mean(enh_wav ** 2)))
    enh_min    = float(np.min(enh_wav))
    enh_max    = float(np.max(enh_wav))
    enh_mean_abs = float(np.mean(np.abs(enh_wav)))
    ratio      = enh_rms / (clean_rms + 1e-8)

    snr_res = compute_snr_improvement(fixed_clean_wav, fixed_noisy_wav, enh_wav)
    return {
        "noisy_rms": noisy_rms,
        "clean_rms": clean_rms,
        "enhanced_rms": enh_rms,
        "enhanced_min": enh_min,
        "enhanced_max": enh_max,
        "enhanced_mean_abs": enh_mean_abs,
        "enhanced_clean_rms_ratio": ratio,
        "input_snr_db":  snr_res["input_snr_db"],
        "enhanced_snr_db": snr_res["output_snr_db"],
        "snr_improvement_db": snr_res["snr_improvement"],
        "mask_mag_mean": float(np.sqrt(mask_np[0]**2 + mask_np[1]**2).mean()),
        "mask_abs_max":  float(np.abs(mask_np).max()),
    }


def evaluate_full(model, dataloader, loss_fn, device, sample_rate=16000):
    """
    Full validation/test evaluation with losses, SNR, STOI, PESQ, classification.
    Compatible with the mask model (forward returns 3 values).
    """
    model.eval()
    total_loss_sum = enh_loss_sum = cls_loss_sum = 0.0
    n_batches = n_samples = 0
    all_true = []
    all_pred = []
    input_snrs, output_snrs, snr_imps = [], [], []
    noisy_stois, enh_stois = [], []
    noisy_pesqs, enh_pesqs = [], []

    with torch.no_grad():
        for batch in dataloader:
            noisy_features = batch["noisy_features"].to(device)
            target_stft    = batch["target_stft"].to(device)
            labels         = batch["noise_class_label"].to(device)

            enhanced, logits, _ = model(noisy_features)
            loss_dict = loss_fn(enhanced, logits, target_stft, labels)

            B = noisy_features.shape[0]
            total_loss_sum += loss_dict["total_loss"].item()
            enh_loss_sum   += loss_dict["enhancement_loss"].item()
            cls_loss_sum   += loss_dict["classification_loss"].item()
            n_batches += 1
            n_samples += B

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            all_true.extend(true.tolist())
            all_pred.extend(pred.tolist())

            # Per-sample SNR/STOI/PESQ
            enh_np = enhanced.cpu().numpy()
            has_clean = "clean_target" in batch
            if has_clean:
                for i in range(B):
                    clean_wav = batch["clean_target"][i].numpy().astype(np.float32)
                    if "m1_waveform" in batch:
                        noisy_wav = batch["m1_waveform"][i].numpy().astype(np.float32)
                    else:
                        nf = batch["noisy_features"][i].numpy()
                        nc = (nf[0] + 1j * nf[1]).astype(np.complex64)
                        noisy_wav = compute_istft(nc, length=len(clean_wav))

                    ec = (enh_np[i, 0] + 1j * enh_np[i, 1]).astype(np.complex64)
                    enh_wav = compute_istft(ec, length=len(clean_wav))

                    snr = compute_snr_improvement(clean_wav, noisy_wav, enh_wav)
                    input_snrs.append(snr["input_snr_db"])
                    output_snrs.append(snr["output_snr_db"])
                    snr_imps.append(snr["snr_improvement"])

                    if stoi_available():
                        ns = compute_stoi(clean_wav, noisy_wav, sample_rate)
                        es = compute_stoi(clean_wav, enh_wav, sample_rate)
                        if ns is not None: noisy_stois.append(ns)
                        if es is not None: enh_stois.append(es)

                    if pesq_available():
                        np_ = compute_pesq(clean_wav, noisy_wav, sample_rate)
                        ep  = compute_pesq(clean_wav, enh_wav, sample_rate)
                        if np_ is not None: noisy_pesqs.append(np_)
                        if ep  is not None: enh_pesqs.append(ep)

    cls_metrics = compute_classification_metrics(all_true, all_pred)

    def safe_mean(lst): return float(np.mean(lst)) if lst else None

    return {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "mean_total_loss":         total_loss_sum / n_batches,
        "mean_enhancement_loss":   enh_loss_sum   / n_batches,
        "mean_classification_loss":cls_loss_sum   / n_batches,
        "mean_input_snr_db":    safe_mean(input_snrs),
        "mean_output_snr_db":   safe_mean(output_snrs),
        "mean_snr_improvement": safe_mean(snr_imps),
        "stoi_available":       stoi_available(),
        "mean_noisy_stoi":      safe_mean(noisy_stois),
        "mean_enhanced_stoi":   safe_mean(enh_stois),
        "pesq_available":       pesq_available(),
        "mean_noisy_pesq":      safe_mean(noisy_pesqs),
        "mean_enhanced_pesq":   safe_mean(enh_pesqs),
        "classification_accuracy":  cls_metrics["accuracy"],
        "per_class_accuracy":       {int(k): float(v) for k,v in cls_metrics["per_class_accuracy"].items()},
        "confusion_matrix":         cls_metrics["confusion_matrix"].tolist(),
        "class_counts":             {int(k): int(v) for k,v in cls_metrics["class_counts"].items()},
    }


def save_checkpoint(path, epoch, best_val_loss, model, optimizer, scheduler, cfg):
    torch.save({
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 4 — FULL 30-EPOCH CRM EXPERIMENT")
    logger.info("Experiment dir: %s", EXP_DIR)
    logger.info("=" * 70)

    # Verify prior experiments untouched
    assert (PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt").exists(), \
        "Phase 1 checkpoint missing!"
    assert (PROJECT_ROOT / "experiments" / "phase2_baseline" / "metrics.json").exists(), \
        "Phase 2 baseline metrics missing!"
    assert (PROJECT_ROOT / "experiments" / "phase2_step3" / "sanity_experiment.json").exists(), \
        "Phase 2 Step 3 sanity experiment missing!"
    logger.info("Prior experiments verified intact.")

    # Config
    cfg = {
        "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP, "seed": SEED,
        "num_workers": 0, "validation_frequency": VAL_FREQ,
        "mixed_precision": False, "optimizer": "AdamW",
        "scheduler": {"type": "ReduceLROnPlateau", "mode": "min",
                      "factor": SCHED_FACTOR, "patience": SCHED_PATIENCE,
                      "min_lr": SCHED_MIN_LR},
        "loss": {"enhancement_weight": ENH_WEIGHT,
                 "classification_weight": CLS_WEIGHT,
                 "enhancement_l1_weight": L1_WEIGHT,
                 "enhancement_l2_weight": L2_WEIGHT,
                 "energy_preservation_weight": ENERGY_WEIGHT},
        "checkpoint": {"dir": str(EXP_DIR), "monitor": "val_total_loss", "mode": "min"},
        "model": "LightweightCNNGRUMaskModel",
    }
    with open(EXP_DIR / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env_info = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "operating_system": platform.platform(),
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    logger.info("Python %s | PyTorch %s | Device: cpu", sys.version.split()[0], torch.__version__)

    # Seed + data
    set_seed(SEED)
    device = torch.device("cpu")

    train_loader = create_dataloader(TRAIN_MANIFEST, DATASET_ROOT, BATCH_SIZE, shuffle=True,  seed=SEED)
    val_loader   = create_dataloader(VAL_MANIFEST,   DATASET_ROOT, BATCH_SIZE, shuffle=False, seed=SEED)

    n_train = len(train_loader)
    n_val   = len(val_loader)
    logger.info("Train batches: %d | Val batches: %d", n_train, n_val)

    # Fixed validation sample for amplitude monitoring
    fixed_batch     = next(iter(val_loader))
    fixed_noisy_feat = fixed_batch["noisy_features"][0:1]          # (1, 2, 257, T)
    fixed_clean_wav  = fixed_batch["clean_target"][0].numpy().astype(np.float32)
    if "m1_waveform" in fixed_batch:
        fixed_noisy_wav = fixed_batch["m1_waveform"][0].numpy().astype(np.float32)
    else:
        nf = fixed_batch["noisy_features"][0].numpy()
        fixed_noisy_wav = compute_istft((nf[0]+1j*nf[1]).astype(np.complex64), length=len(fixed_clean_wav))

    # Model — fresh init with seed=123
    set_seed(SEED)
    model    = LightweightCNNGRUMaskModel().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: LightweightCNNGRUMaskModel | Params: %d", trainable)
    assert trainable == 70_789, f"Param mismatch: {trainable}"

    loss_fn = MultiTaskMaskLoss(
        enhancement_weight=ENH_WEIGHT, classification_weight=CLS_WEIGHT,
        enhancement_l1_weight=L1_WEIGHT, enhancement_l2_weight=L2_WEIGHT,
        energy_preservation_weight=ENERGY_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
    )

    # Training state
    history           = []
    best_val_loss     = float("inf")
    best_val_loss_ep  = -1
    best_val_snr_imp  = -float("inf")
    best_val_snr_ep   = -1
    nan_inf_occurred  = False
    stop_reason       = None
    total_start       = time.perf_counter()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()

        # --- Train ---
        model.train()
        t_total = t_enh = t_cls = 0.0
        unclipped_norms = []

        for bidx, batch in enumerate(train_loader):
            noisy = batch["noisy_features"].to(device)
            tgt   = batch["target_stft"].to(device)
            lbls  = batch["noise_class_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            enh_out, cls_logits, _ = model(noisy)
            ld = loss_fn(enh_out, cls_logits, tgt, lbls)
            loss_val = ld["total_loss"].item()

            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_inf_occurred = True
                stop_reason = f"NaN/Inf loss at Epoch {epoch} Batch {bidx+1}: {loss_val}"
                logger.error(stop_reason)
                break

            ld["total_loss"].backward()

            # Gradient health
            for p in model.parameters():
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    nan_inf_occurred = True
                    stop_reason = f"NaN/Inf gradient at Epoch {epoch} Batch {bidx+1}"
                    logger.error(stop_reason)
                    break
            if nan_inf_occurred:
                break

            unc = math.sqrt(sum(
                p.grad.detach().norm(2).item()**2
                for p in model.parameters() if p.grad is not None
            ))
            unclipped_norms.append(unc)

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            t_total += loss_val
            t_enh   += ld["enhancement_loss"].item()
            t_cls   += ld["classification_loss"].item()

            if (bidx + 1) % LOG_FREQ == 0:
                logger.info(
                    "  Epoch %d | Batch %d/%d | loss=%.6f enh=%.6f cls=%.6f",
                    epoch, bidx+1, n_train, loss_val,
                    ld["enhancement_loss"].item(), ld["classification_loss"].item(),
                )

        if nan_inf_occurred:
            break

        avg_train_total = t_total / n_train
        avg_train_enh   = t_enh   / n_train
        avg_train_cls   = t_cls   / n_train
        avg_grad_norm   = float(np.mean(unclipped_norms)) if unclipped_norms else 0.0

        # --- Validate ---
        val_res = evaluate_full(model, val_loader, loss_fn, device)
        val_total = val_res["mean_total_loss"]
        val_enh   = val_res["mean_enhancement_loss"]
        val_cls   = val_res["mean_classification_loss"]
        val_snr   = val_res["mean_snr_improvement"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_total)

        # Best loss checkpoint
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_val_loss_ep = epoch
            save_checkpoint(EXP_DIR / "best_checkpoint.pt", epoch, best_val_loss,
                            model, optimizer, scheduler, cfg)
            logger.info("Epoch %d | New BEST val_loss=%.6f — saved best_checkpoint.pt", epoch, val_total)

        # Best SNR epoch
        if val_snr is not None and val_snr > best_val_snr_imp:
            best_val_snr_imp = val_snr
            best_val_snr_ep  = epoch

        # Last checkpoint
        save_checkpoint(EXP_DIR / "last_checkpoint.pt", epoch, best_val_loss,
                        model, optimizer, scheduler, cfg)

        # Fixed-sample amplitude monitoring
        sample_stats = compute_sample_stats(model, device, fixed_noisy_feat, fixed_clean_wav, fixed_noisy_wav)

        ep_dur = time.perf_counter() - ep_start

        # Classification per-class distribution
        pred_dist = {}
        for k, v in val_res["class_counts"].items():
            pred_dist[int(k)] = int(v)

        epoch_rec = {
            "epoch": epoch,
            "train_total_loss":          float(avg_train_total),
            "train_enhancement_loss":    float(avg_train_enh),
            "train_classification_loss": float(avg_train_cls),
            "val_total_loss":            float(val_total),
            "val_enhancement_loss":      float(val_enh),
            "val_classification_loss":   float(val_cls),
            "learning_rate":             float(current_lr),
            "epoch_duration_seconds":    float(ep_dur),
            "unclipped_grad_norm":       float(avg_grad_norm),
            # Classification
            "val_classification_accuracy": float(val_res["classification_accuracy"]),
            "val_per_class_accuracy":      val_res["per_class_accuracy"],
            "val_predicted_class_distribution": pred_dist,
            "val_confusion_matrix":        val_res["confusion_matrix"],
            # SNR
            "val_input_snr_db":     val_res["mean_input_snr_db"],
            "val_enhanced_snr_db":  val_res["mean_output_snr_db"],
            "val_snr_improvement_db": val_res["mean_snr_improvement"],
            # STOI/PESQ
            "val_noisy_stoi":    val_res["mean_noisy_stoi"],
            "val_enhanced_stoi": val_res["mean_enhanced_stoi"],
            "val_noisy_pesq":    val_res["mean_noisy_pesq"],
            "val_enhanced_pesq": val_res["mean_enhanced_pesq"],
            # Fixed-sample amplitude monitoring
            **{f"sample_{k}": v for k, v in sample_stats.items()},
        }
        history.append(epoch_rec)

        logger.info(
            "Epoch %2d/30 | Train: %.6f (E=%.6f C=%.6f) | Val: %.6f (E=%.6f C=%.6f) "
            "| Acc=%.2f%% | SNR_imp=%.2f dB | LR=%.2e | Ratio=%.4f | %.1fs",
            epoch,
            avg_train_total, avg_train_enh, avg_train_cls,
            val_total, val_enh, val_cls,
            val_res["classification_accuracy"] * 100,
            val_snr if val_snr is not None else float("nan"),
            current_lr,
            sample_stats["enhanced_clean_rms_ratio"],
            ep_dur,
        )

    total_dur = time.perf_counter() - total_start
    logger.info("Training complete. Total: %.1fs (%.2f min)", total_dur, total_dur/60)

    # Save history
    with open(EXP_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # -----------------------------------------------------------------------
    # Final test evaluation (best checkpoint, test set, once only)
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FINAL TEST EVALUATION — loading best_checkpoint.pt (epoch %d)", best_val_loss_ep)
    logger.info("=" * 70)

    best_ckpt = torch.load(EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    test_model = LightweightCNNGRUMaskModel().to(device)
    test_model.load_state_dict(best_ckpt["model_state_dict"])
    test_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    assert test_params == 70_789

    test_loader = create_dataloader(TEST_MANIFEST, DATASET_ROOT, BATCH_SIZE, shuffle=False, seed=SEED)
    test_res    = evaluate_full(test_model, test_loader, loss_fn, device)

    # Fixed-sample stats from best checkpoint model
    test_sample_stats = compute_sample_stats(test_model, device, fixed_noisy_feat, fixed_clean_wav, fixed_noisy_wav)

    final_test = {
        "checkpoint_evaluated":   "best_checkpoint.pt",
        "best_val_loss_epoch":    best_val_loss_ep,
        "best_val_loss":          float(best_val_loss),
        "total_loss":             float(test_res["mean_total_loss"]),
        "enhancement_loss":       float(test_res["mean_enhancement_loss"]),
        "classification_loss":    float(test_res["mean_classification_loss"]),
        "input_snr_db":           test_res["mean_input_snr_db"],
        "enhanced_snr_db":        test_res["mean_output_snr_db"],
        "snr_improvement_db":     test_res["mean_snr_improvement"],
        "stoi_available":         test_res["stoi_available"],
        "noisy_stoi":             test_res["mean_noisy_stoi"],
        "enhanced_stoi":          test_res["mean_enhanced_stoi"],
        "pesq_available":         test_res["pesq_available"],
        "noisy_pesq":             test_res["mean_noisy_pesq"],
        "enhanced_pesq":          test_res["mean_enhanced_pesq"],
        "classification_accuracy":float(test_res["classification_accuracy"]),
        "per_class_accuracy":     test_res["per_class_accuracy"],
        "confusion_matrix":       test_res["confusion_matrix"],
        "class_counts":           {int(k): int(v) for k,v in test_res["class_counts"].items()},
        "sample_enhanced_clean_rms_ratio": test_sample_stats["enhanced_clean_rms_ratio"],
        "sample_enhanced_rms":    test_sample_stats["enhanced_rms"],
        "sample_clean_rms":       test_sample_stats["clean_rms"],
        "sample_noisy_rms":       test_sample_stats["noisy_rms"],
    }

    metrics = {
        "experiment_name":         "phase2_step4_crm",
        "environment":             env_info,
        "training_duration_seconds": float(total_dur),
        "completed_epochs":        len(history),
        "nan_inf_occurred":        nan_inf_occurred,
        "stop_reason":             stop_reason,
        "trainable_params":        int(trainable),
        "n_train_batches":         n_train,
        "n_val_batches":           n_val,
        "best_val_loss_epoch":     best_val_loss_ep,
        "best_val_loss":           float(best_val_loss),
        "best_val_snr_imp_epoch":  best_val_snr_ep,
        "best_val_snr_imp":        float(best_val_snr_imp),
        "final_test_results":      final_test,
    }
    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved to: %s", EXP_DIR)
    logger.info("Best val_loss epoch: %d | val_loss: %.6f", best_val_loss_ep, best_val_loss)
    logger.info("Best val SNR-imp epoch: %d | SNR-imp: %.2f dB", best_val_snr_ep, best_val_snr_imp)
    logger.info("Test SNR improvement: %.2f dB", final_test.get("snr_improvement_db") or float("nan"))
    logger.info("Test classification accuracy: %.2f%%", final_test["classification_accuracy"] * 100)
    logger.info("Test STOI noisy/enhanced: %.4f / %.4f",
                final_test.get("noisy_stoi") or 0.0, final_test.get("enhanced_stoi") or 0.0)
    logger.info("Test PESQ noisy/enhanced: %.4f / %.4f",
                final_test.get("noisy_pesq") or 0.0, final_test.get("enhanced_pesq") or 0.0)

    return metrics


if __name__ == "__main__":
    run()
