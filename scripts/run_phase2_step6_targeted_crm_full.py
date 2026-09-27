"""
scripts/run_phase2_step6_targeted_crm_full.py
=============================================
PHASE 2 STEP 6 — Full 30-Epoch Targeted CRM Validation Experiment.

Key Research Question:
Does the Step 5 targeted-CRM loss (direct mask supervision via clamped ideal CRM
M_target = clamp(S/(X+eps), -1, 1)) continue to improve over full 30 epochs?
Specifically:
  1. Does SNR improvement move toward zero or positive values?
  2. Does speech energy remain preserved without amplitude collapse?
  3. Does classification head recover useful multi-class discrimination?
  4. Does the model remain stable under the <100,000 parameter budget (70,789 params)?

Configuration:
  - Model: LightweightCNNGRUMaskModel (70,789 parameters)
  - Loss: MultiTaskTargetedCRMLoss
      - mask_weight = 0.60
      - recon_weight = 0.35
      - energy_weight = 0.05
      - classification_weight = 0.10
      - l1_weight = 0.7, l2_weight = 0.3
  - Optimizer: AdamW (lr=0.001, weight_decay=0.0001, grad_clip=5.0)
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6, mode=min)
  - Seed: 123 (fresh initialization)
  - Epochs: 30
  - Batch Size: 16
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

from evaluation.metrics import compute_classification_metrics
from evaluation.snr import compute_snr_improvement
from evaluation.stoi import compute_stoi, stoi_available
from evaluation.pesq import compute_pesq, pesq_available
from src.data.dataset import create_dataloader
from src.features.stft import compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.targeted_crm_loss import MultiTaskTargetedCRMLoss
from src.training.training_config import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase2_step6_targeted_crm_full")

EXP_DIR        = PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full"
MANIFEST_DIR   = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST   = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST  = MANIFEST_DIR / "test_manifest.jsonl"
DATASET_ROOT   = PROJECT_ROOT / "data" / "raw" / "dataset"

# Training configuration — strict matching to project requirements
SEED           = 123
BATCH_SIZE     = 16
EPOCHS         = 30
LR             = 0.001
WEIGHT_DECAY   = 0.0001
GRAD_CLIP      = 5.0
VAL_FREQ       = 1
LOG_FREQ       = 10

# Step 5 targeted CRM loss weights (frozen per instructions)
ENH_WEIGHT     = 1.0
CLS_WEIGHT     = 0.10
MASK_WEIGHT    = 0.60
RECON_WEIGHT   = 0.35
ENERGY_WEIGHT  = 0.05
L1_WEIGHT      = 0.7
L2_WEIGHT      = 0.3

# ReduceLROnPlateau
SCHED_FACTOR   = 0.5
SCHED_PATIENCE = 3
SCHED_MIN_LR   = 1e-6


def compute_sample_stats(model, device, fixed_noisy, fixed_clean_wav, fixed_noisy_wav):
    """Compute fixed-sample waveform statistics without modifying model weights."""
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
    ratio      = enh_rms / (clean_rms + 1e-8)

    snr_res = compute_snr_improvement(fixed_clean_wav, fixed_noisy_wav, enh_wav)

    sample_stoi = compute_stoi(fixed_clean_wav, enh_wav, 16000) if stoi_available() else None
    sample_pesq = compute_pesq(fixed_clean_wav, enh_wav, 16000) if pesq_available() else None

    return {
        "noisy_rms": noisy_rms,
        "clean_rms": clean_rms,
        "enhanced_rms": enh_rms,
        "enhanced_clean_rms_ratio": ratio,
        "enhanced_min": float(enh_wav.min()),
        "enhanced_max": float(enh_wav.max()),
        "enhanced_mean_abs": float(np.mean(np.abs(enh_wav))),
        "input_snr_db":  snr_res["input_snr_db"],
        "enhanced_snr_db": snr_res["output_snr_db"],
        "snr_improvement_db": snr_res["snr_improvement"],
        "sample_stoi": sample_stoi,
        "sample_pesq": sample_pesq,
        "mask_min": float(mask_np.min()),
        "mask_max": float(mask_np.max()),
        "mask_real_min": float(mask_np[0].min()),
        "mask_real_max": float(mask_np[0].max()),
        "mask_imag_min": float(mask_np[1].min()),
        "mask_imag_max": float(mask_np[1].max()),
        "mask_real_finite": bool(np.all(np.isfinite(mask_np[0]))),
        "mask_imag_finite": bool(np.all(np.isfinite(mask_np[1]))),
        "enhanced_finite": bool(np.all(np.isfinite(enh_np))),
        "mask_mean_abs": float(np.abs(mask_np).mean()),
    }


def evaluate_full(model, dataloader, loss_fn, device, sample_rate=16000):
    """Full evaluation on validation or test dataset."""
    model.eval()
    total_loss_sum = enh_loss_sum = cls_loss_sum = mask_loss_sum = recon_loss_sum = energy_loss_sum = 0.0
    n_batches = n_samples = 0
    all_true, all_pred = [], []
    input_snrs, output_snrs, snr_imps = [], [], []
    noisy_stois, enh_stois = [], []
    noisy_pesqs, enh_pesqs = [], []

    with torch.no_grad():
        for batch in dataloader:
            noisy_features = batch["noisy_features"].to(device)
            target_stft    = batch["target_stft"].to(device)
            labels         = batch["noise_class_label"].to(device)

            enhanced, logits, mask = model(noisy_features)
            loss_dict = loss_fn(enhanced, mask, logits, noisy_features, target_stft, labels)

            B = noisy_features.shape[0]
            total_loss_sum += loss_dict["total_loss"].item()
            enh_loss_sum   += loss_dict["enhancement_loss"].item()
            cls_loss_sum   += loss_dict["classification_loss"].item()
            mask_loss_sum  += loss_dict["mask_loss"].item()
            recon_loss_sum += loss_dict["reconstruction_loss"].item()
            energy_loss_sum+= loss_dict["energy_loss"].item()

            n_batches += 1
            n_samples += B

            pred = torch.argmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            all_true.extend(true.tolist())
            all_pred.extend(pred.tolist())

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
        "mean_mask_loss":          mask_loss_sum  / n_batches,
        "mean_reconstruction_loss":recon_loss_sum/ n_batches,
        "mean_energy_loss":        energy_loss_sum/ n_batches,
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


def run_step6_experiment():
    """Run full 30-epoch training run for Phase 2 Step 6."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 6 — FULL 30-EPOCH TARGETED CRM VALIDATION")
    logger.info("Experiment dir: %s", EXP_DIR)
    logger.info("Total Epochs: %d", EPOCHS)
    logger.info("=" * 70)

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
                 "mask_weight": MASK_WEIGHT,
                 "recon_weight": RECON_WEIGHT,
                 "energy_weight": ENERGY_WEIGHT,
                 "l1_weight": L1_WEIGHT,
                 "l2_weight": L2_WEIGHT},
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

    set_seed(SEED)
    device = torch.device("cpu")

    # STRICT DATA SEPARATION: Only train and val loaders are created here
    train_loader = create_dataloader(TRAIN_MANIFEST, DATASET_ROOT, BATCH_SIZE, shuffle=True,  seed=SEED)
    val_loader   = create_dataloader(VAL_MANIFEST,   DATASET_ROOT, BATCH_SIZE, shuffle=False, seed=SEED)

    n_train = len(train_loader)
    n_val   = len(val_loader)

    fixed_batch     = next(iter(val_loader))
    fixed_noisy_feat = fixed_batch["noisy_features"][0:1]
    fixed_clean_wav  = fixed_batch["clean_target"][0].numpy().astype(np.float32)
    if "m1_waveform" in fixed_batch:
        fixed_noisy_wav = fixed_batch["m1_waveform"][0].numpy().astype(np.float32)
    else:
        nf = fixed_batch["noisy_features"][0].numpy()
        fixed_noisy_wav = compute_istft((nf[0]+1j*nf[1]).astype(np.complex64), length=len(fixed_clean_wav))

    # Fresh model initialization with seed 123
    set_seed(SEED)
    model    = LightweightCNNGRUMaskModel().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 70_789, f"Param mismatch: {trainable}"

    loss_fn = MultiTaskTargetedCRMLoss(
        enhancement_weight=ENH_WEIGHT, classification_weight=CLS_WEIGHT,
        mask_weight=MASK_WEIGHT, recon_weight=RECON_WEIGHT, energy_weight=ENERGY_WEIGHT,
        l1_weight=L1_WEIGHT, l2_weight=L2_WEIGHT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR,
        patience=SCHED_PATIENCE, min_lr=SCHED_MIN_LR,
    )

    history           = []
    best_val_loss     = float("inf")
    best_val_loss_ep  = -1
    best_val_snr_imp  = -float("inf")
    best_val_snr_ep   = -1
    nan_inf_occurred  = False
    stop_reason       = None
    total_start       = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()

        model.train()
        t_total = t_enh = t_cls = t_mask = t_recon = t_energy = 0.0
        unclipped_norms = []

        for bidx, batch in enumerate(train_loader):
            noisy = batch["noisy_features"].to(device)
            tgt   = batch["target_stft"].to(device)
            lbls  = batch["noise_class_label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            enh_out, cls_logits, mask = model(noisy)
            ld = loss_fn(enh_out, mask, cls_logits, noisy, tgt, lbls)
            loss_val = ld["total_loss"].item()

            if math.isnan(loss_val) or math.isinf(loss_val):
                nan_inf_occurred = True
                stop_reason = f"NaN/Inf loss at Epoch {epoch} Batch {bidx+1}: {loss_val}"
                logger.error(stop_reason)
                break

            ld["total_loss"].backward()

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
            t_mask  += ld["mask_loss"].item()
            t_recon += ld["reconstruction_loss"].item()
            t_energy+= ld["energy_loss"].item()

            if (bidx + 1) % LOG_FREQ == 0:
                logger.info(
                    "  Epoch %d | Batch %d/%d | loss=%.6f enh=%.6f (mask=%.6f recon=%.6f) cls=%.6f",
                    epoch, bidx+1, n_train, loss_val,
                    ld["enhancement_loss"].item(), ld["mask_loss"].item(), ld["reconstruction_loss"].item(),
                    ld["classification_loss"].item(),
                )

        if nan_inf_occurred:
            break

        avg_train_total = t_total / n_train
        avg_train_enh   = t_enh   / n_train
        avg_train_cls   = t_cls   / n_train
        avg_train_mask  = t_mask  / n_train
        avg_train_recon = t_recon / n_train
        avg_train_energy= t_energy/ n_train
        avg_grad_norm   = float(np.mean(unclipped_norms)) if unclipped_norms else 0.0

        val_res = evaluate_full(model, val_loader, loss_fn, device)
        val_total = val_res["mean_total_loss"]
        val_enh   = val_res["mean_enhancement_loss"]
        val_cls   = val_res["mean_classification_loss"]
        val_snr   = val_res["mean_snr_improvement"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_total)

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_val_loss_ep = epoch
            torch.save({
                "epoch": epoch, "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": cfg,
            }, EXP_DIR / "best_checkpoint.pt")
            logger.info("Epoch %d | New BEST val_loss=%.6f — saved best_checkpoint.pt", epoch, val_total)

        if val_snr is not None and val_snr > best_val_snr_imp:
            best_val_snr_imp = val_snr
            best_val_snr_ep  = epoch

        torch.save({
            "epoch": epoch, "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        }, EXP_DIR / "last_checkpoint.pt")

        sample_stats = compute_sample_stats(model, device, fixed_noisy_feat, fixed_clean_wav, fixed_noisy_wav)
        ep_dur = time.perf_counter() - ep_start

        pred_dist = {int(k): int(v) for k, v in val_res["class_counts"].items()}

        epoch_rec = {
            "epoch": epoch,
            "train_total_loss":          float(avg_train_total),
            "train_enhancement_loss":    float(avg_train_enh),
            "train_classification_loss": float(avg_train_cls),
            "train_mask_loss":           float(avg_train_mask),
            "train_reconstruction_loss": float(avg_train_recon),
            "train_energy_loss":         float(avg_train_energy),
            "val_total_loss":            float(val_total),
            "val_enhancement_loss":      float(val_enh),
            "val_classification_loss":   float(val_cls),
            "val_mask_loss":             float(val_res["mean_mask_loss"]),
            "val_reconstruction_loss":   float(val_res["mean_reconstruction_loss"]),
            "val_energy_loss":           float(val_res["mean_energy_loss"]),
            "learning_rate":             float(current_lr),
            "epoch_duration_seconds":    float(ep_dur),
            "unclipped_grad_norm":       float(avg_grad_norm),
            "val_classification_accuracy": float(val_res["classification_accuracy"]),
            "val_per_class_accuracy":      val_res["per_class_accuracy"],
            "val_predicted_class_distribution": pred_dist,
            "val_confusion_matrix":        val_res["confusion_matrix"],
            "val_input_snr_db":     val_res["mean_input_snr_db"],
            "val_enhanced_snr_db":  val_res["mean_output_snr_db"],
            "val_snr_improvement_db": val_res["mean_snr_improvement"],
            "val_noisy_stoi":    val_res["mean_noisy_stoi"],
            "val_enhanced_stoi": val_res["mean_enhanced_stoi"],
            "val_noisy_pesq":    val_res["mean_noisy_pesq"],
            "val_enhanced_pesq": val_res["mean_enhanced_pesq"],
            **{f"sample_{k}": v for k, v in sample_stats.items()},
        }
        history.append(epoch_rec)

        logger.info(
            "Epoch %2d/%d | Train: %.6f (Mask=%.6f Rec=%.6f C=%.6f) | Val: %.6f (M=%.6f R=%.6f C=%.6f) "
            "| Acc=%.2f%% | SNR_imp=%.2f dB | LR=%.2e | Ratio=%.4f | %.1fs",
            epoch, EPOCHS,
            avg_train_total, avg_train_mask, avg_train_recon, avg_train_cls,
            val_total, val_res["mean_mask_loss"], val_res["mean_reconstruction_loss"], val_cls,
            val_res["classification_accuracy"] * 100,
            val_snr if val_snr is not None else float("nan"),
            current_lr,
            sample_stats["enhanced_clean_rms_ratio"],
            ep_dur,
        )

    total_dur = time.perf_counter() - total_start

    with open(EXP_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # FINAL HELD-OUT TEST EVALUATION — loading best_checkpoint.pt
    logger.info("=" * 70)
    logger.info("FINAL TEST EVALUATION — loading best_checkpoint.pt (epoch %d)", best_val_loss_ep)
    logger.info("=" * 70)

    best_ckpt = torch.load(EXP_DIR / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    test_model = LightweightCNNGRUMaskModel().to(device)
    test_model.load_state_dict(best_ckpt["model_state_dict"])

    test_loader = create_dataloader(TEST_MANIFEST, DATASET_ROOT, BATCH_SIZE, shuffle=False, seed=SEED)
    test_res    = evaluate_full(test_model, test_loader, loss_fn, device)
    test_sample_stats = compute_sample_stats(test_model, device, fixed_noisy_feat, fixed_clean_wav, fixed_noisy_wav)

    final_test = {
        "checkpoint_evaluated":   "best_checkpoint.pt",
        "best_val_loss_epoch":    best_val_loss_ep,
        "best_val_loss":          float(best_val_loss),
        "total_loss":             float(test_res["mean_total_loss"]),
        "enhancement_loss":       float(test_res["mean_enhancement_loss"]),
        "mask_loss":              float(test_res["mean_mask_loss"]),
        "reconstruction_loss":    float(test_res["mean_reconstruction_loss"]),
        "energy_loss":            float(test_res["mean_energy_loss"]),
        "classification_loss":    float(test_res["mean_classification_loss"]),
        "input_snr_db":           test_res["mean_input_snr_db"],
        "enhanced_snr_db":        test_res["mean_output_snr_db"],
        "snr_improvement_db":     test_res["mean_snr_improvement"],
        "stoi_available":         test_res["stoi_available"],
        "noisy_stoi":             test_res["mean_noisy_stoi"],
        "enhanced_stoi":          test_res["mean_enhanced_stoi"],
        "stoi_delta":             (test_res["mean_enhanced_stoi"] - test_res["mean_noisy_stoi"]) if (test_res["mean_enhanced_stoi"] is not None and test_res["mean_noisy_stoi"] is not None) else None,
        "pesq_available":         test_res["pesq_available"],
        "noisy_pesq":             test_res["mean_noisy_pesq"],
        "enhanced_pesq":          test_res["mean_enhanced_pesq"],
        "pesq_delta":             (test_res["mean_enhanced_pesq"] - test_res["mean_noisy_pesq"]) if (test_res["mean_enhanced_pesq"] is not None and test_res["mean_noisy_pesq"] is not None) else None,
        "classification_accuracy":float(test_res["classification_accuracy"]),
        "per_class_accuracy":     test_res["per_class_accuracy"],
        "confusion_matrix":       test_res["confusion_matrix"],
        "class_counts":           {int(k): int(v) for k,v in test_res["class_counts"].items()},
        "sample_enhanced_clean_rms_ratio": test_sample_stats["enhanced_clean_rms_ratio"],
        "sample_enhanced_rms":    test_sample_stats["enhanced_rms"],
        "sample_clean_rms":       test_sample_stats["clean_rms"],
        "sample_noisy_rms":       test_sample_stats["noisy_rms"],
        "sample_enhanced_min":    test_sample_stats["enhanced_min"],
        "sample_enhanced_max":    test_sample_stats["enhanced_max"],
        "sample_enhanced_mean_abs": test_sample_stats["enhanced_mean_abs"],
    }

    metrics = {
        "experiment_name":         "phase2_step6_targeted_crm_full",
        "environment":             env_info,
        "training_duration_seconds": float(total_dur),
        "completed_epochs":        len(history),
        "nan_inf_occurred":        nan_inf_occurred,
        "stop_reason":             stop_reason,
        "trainable_params":        int(trainable),
        "best_val_loss_epoch":     best_val_loss_ep,
        "best_val_loss":           float(best_val_loss),
        "best_val_snr_imp_epoch":  best_val_snr_ep,
        "best_val_snr_imp":        float(best_val_snr_imp),
        "final_test_results":      final_test,
    }

    with open(EXP_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Artifacts saved to: %s", EXP_DIR)
    return metrics


if __name__ == "__main__":
    run_step6_experiment()
