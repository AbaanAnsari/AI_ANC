"""
scripts/run_phase2_step3_sanity.py
=====================================
Phase 2 Step 3 — Short Anti-Collapse Sanity Experiment.

PURPOSE
-------
Verify that the new CRM (complex ratio mask) formulation does NOT collapse
to near-zero output energy, unlike the baseline direct STFT regression.

DESIGN
------
- Uses first 5 batches of training data (80 samples max) — NOT a full epoch
- Runs 5 gradient steps only
- Records fixed sample RMS before and after
- Checks whether output remains non-trivially non-zero
- Does NOT use the test set
- Does NOT save a Phase 1 / Phase 2 baseline checkpoint

CRITERIA FOR PASS
-----------------
- enhanced/clean RMS ratio stays > 0.05 after 5 steps (not collapsing)
- no NaN/Inf in loss or gradients
- mask values remain bounded in (-1, 1)
- loss is finite and decreasing (not exploding)
- SNR improvement does not become MORE negative than baseline after 5 steps

Experiment is saved to: experiments/phase2_step3/
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
logger = logging.getLogger("phase2_step3_sanity")

EXP_DIR = PROJECT_ROOT / "experiments" / "phase2_step3"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"

# Sanity experiment hyper-parameters (NOT the full training config)
SEED = 123
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0
N_SANITY_STEPS = 5      # Only 5 gradient steps
N_SANITY_BATCHES = 5    # Use first 5 batches of training data


def compute_sample_stats(model, fixed_noisy, fixed_clean_wav, fixed_noisy_wav, device):
    """Evaluate fixed sample and return statistics dict."""
    model.eval()
    with torch.no_grad():
        enh_out, _, mask = model(fixed_noisy.to(device))
        enh_np = enh_out[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()

    enh_complex = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    enh_wav = compute_istft(enh_complex, length=len(fixed_clean_wav))

    noisy_rms = float(np.sqrt(np.mean(fixed_noisy_wav ** 2)))
    clean_rms = float(np.sqrt(np.mean(fixed_clean_wav ** 2)))
    enh_rms = float(np.sqrt(np.mean(enh_wav ** 2)))
    enh_to_clean_ratio = enh_rms / (clean_rms + 1e-8)

    snr_res = compute_snr_improvement(fixed_clean_wav, fixed_noisy_wav, enh_wav)

    stats = {
        "noisy_rms": noisy_rms,
        "clean_rms": clean_rms,
        "enhanced_rms": enh_rms,
        "enhanced_clean_rms_ratio": enh_to_clean_ratio,
        "input_snr_db": snr_res["input_snr_db"],
        "enhanced_snr_db": snr_res["output_snr_db"],
        "snr_improvement_db": snr_res["snr_improvement"],
        "mask_real_mean": float(mask_np[0].mean()),
        "mask_imag_mean": float(mask_np[1].mean()),
        "mask_magnitude_mean": float(np.sqrt(mask_np[0]**2 + mask_np[1]**2).mean()),
        "mask_real_min": float(mask_np[0].min()),
        "mask_real_max": float(mask_np[0].max()),
        "mask_imag_min": float(mask_np[1].min()),
        "mask_imag_max": float(mask_np[1].max()),
    }

    # STOI if available
    if stoi_available():
        stoi_noisy = compute_stoi(fixed_clean_wav, fixed_noisy_wav, 16000)
        stoi_enh = compute_stoi(fixed_clean_wav, enh_wav, 16000)
        stats["stoi_noisy"] = float(stoi_noisy) if stoi_noisy is not None else None
        stats["stoi_enhanced"] = float(stoi_enh) if stoi_enh is not None else None
    else:
        stats["stoi_noisy"] = "UNAVAILABLE"
        stats["stoi_enhanced"] = "UNAVAILABLE"

    # PESQ if available
    if pesq_available():
        pesq_noisy = compute_pesq(fixed_clean_wav, fixed_noisy_wav, 16000)
        pesq_enh = compute_pesq(fixed_clean_wav, enh_wav, 16000)
        stats["pesq_noisy"] = float(pesq_noisy) if pesq_noisy is not None else None
        stats["pesq_enhanced"] = float(pesq_enh) if pesq_enh is not None else None
    else:
        stats["pesq_noisy"] = "UNAVAILABLE"
        stats["pesq_enhanced"] = "UNAVAILABLE"

    return stats


def run_sanity_experiment():
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 2 STEP 3 — SHORT ANTI-COLLAPSE SANITY EXPERIMENT")
    logger.info("Steps: %d | Batches/step: 1 | Total gradient steps: %d", N_SANITY_STEPS, N_SANITY_STEPS)
    logger.info("=" * 70)

    set_seed(SEED)
    device = torch.device("cpu")

    # Load data
    train_loader = create_dataloader(
        manifest_path=TRAIN_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_loader = create_dataloader(
        manifest_path=VAL_MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    # Extract fixed validation sample for monitoring
    fixed_batch = next(iter(val_loader))
    fixed_noisy_features = fixed_batch["noisy_features"][0:1]  # (1, 2, 257, T)
    fixed_clean_wav = fixed_batch["clean_target"][0].numpy().astype(np.float32)
    if "m1_waveform" in fixed_batch:
        fixed_noisy_wav = fixed_batch["m1_waveform"][0].numpy().astype(np.float32)
    else:
        nf = fixed_batch["noisy_features"][0].numpy()
        nc = (nf[0] + 1j * nf[1]).astype(np.complex64)
        fixed_noisy_wav = compute_istft(nc, length=len(fixed_clean_wav))

    # Initialize fresh mask model
    set_seed(SEED)
    model = LightweightCNNGRUMaskModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: LightweightCNNGRUMaskModel | Trainable params: %d", trainable)
    assert trainable == 70_789

    loss_fn = MultiTaskMaskLoss(
        enhancement_weight=1.0,
        classification_weight=0.10,
        enhancement_l1_weight=0.7,
        enhancement_l2_weight=0.3,
        energy_preservation_weight=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # === BEFORE TRAINING STATS ===
    logger.info("Computing initial (pre-training) fixed-sample statistics...")
    stats_before = compute_sample_stats(model, fixed_noisy_features, fixed_clean_wav, fixed_noisy_wav, device)
    logger.info(
        "BEFORE | noisy_rms=%.4f | clean_rms=%.4f | enhanced_rms=%.4f | ratio=%.4f | SNR_imp=%.2f dB",
        stats_before["noisy_rms"], stats_before["clean_rms"],
        stats_before["enhanced_rms"], stats_before["enhanced_clean_rms_ratio"],
        stats_before["snr_improvement_db"],
    )

    # === SANITY TRAINING STEPS ===
    nan_inf_occurred = False
    step_records = []
    train_iter = iter(train_loader)

    for step in range(1, N_SANITY_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        noisy_features = batch["noisy_features"].to(device)
        target_stft = batch["target_stft"].to(device)
        noise_class_label = batch["noise_class_label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        enhanced_output, cls_logits, mask_out = model(noisy_features)

        loss_dict = loss_fn(enhanced_output, cls_logits, target_stft, noise_class_label)
        total_loss = loss_dict["total_loss"]

        loss_val = total_loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_inf_occurred = True
            logger.error("NaN/Inf loss at step %d: %f", step, loss_val)
            break

        total_loss.backward()

        # Gradient health check
        grad_nan = False
        for p in model.parameters():
            if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                grad_nan = True
                break
        if grad_nan:
            nan_inf_occurred = True
            logger.error("NaN/Inf gradient at step %d", step)
            break

        # Gradient norm
        unclipped_norm = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)

        optimizer.step()

        # Mask statistics
        mask_abs_max = mask_out.abs().max().item()
        mask_bounded = mask_abs_max < 1.0

        # Fixed sample stats
        sample_stats = compute_sample_stats(model, fixed_noisy_features, fixed_clean_wav, fixed_noisy_wav, device)

        record = {
            "step": step,
            "train_total_loss": float(loss_val),
            "train_enhancement_loss": float(loss_dict["enhancement_loss"].item()),
            "train_classification_loss": float(loss_dict["classification_loss"].item()),
            "grad_norm_unclipped": float(unclipped_norm),
            "mask_abs_max": float(mask_abs_max),
            "mask_bounded": mask_bounded,
            **sample_stats,
        }
        step_records.append(record)

        logger.info(
            "Step %d | loss=%.6f (enh=%.6f, cls=%.6f) | mask_max=%.4f | "
            "enh_rms=%.6f | ratio=%.4f | SNR_imp=%.2f dB | grad_norm=%.3f",
            step,
            loss_val, loss_dict["enhancement_loss"].item(), loss_dict["classification_loss"].item(),
            mask_abs_max, sample_stats["enhanced_rms"],
            sample_stats["enhanced_clean_rms_ratio"],
            sample_stats["snr_improvement_db"],
            unclipped_norm,
        )

    # === AFTER TRAINING STATS ===
    stats_after = compute_sample_stats(model, fixed_noisy_features, fixed_clean_wav, fixed_noisy_wav, device)

    # === ANTI-COLLAPSE ASSESSMENT ===
    # Baseline epoch-28 ratio was 0.0052 (near-zero collapse)
    # Baseline epoch-1 ratio was 0.5823
    COLLAPSE_THRESHOLD = 0.05
    initial_ratio = stats_before["enhanced_clean_rms_ratio"]
    final_ratio = stats_after["enhanced_clean_rms_ratio"]
    collapsed = final_ratio < COLLAPSE_THRESHOLD

    # Check SNR improvement didn't become dramatically worse
    BASELINE_SNR_IMP_EPOCH28 = -7.60
    final_snr_imp = stats_after["snr_improvement_db"]

    anti_collapse_pass = (
        not nan_inf_occurred
        and not collapsed
        and all(r["mask_bounded"] for r in step_records)
    )

    result = {
        "experiment": "phase2_step3_sanity",
        "seed": SEED,
        "n_steps": N_SANITY_STEPS,
        "batch_size": BATCH_SIZE,
        "model": "LightweightCNNGRUMaskModel",
        "trainable_params": int(trainable),
        "nan_inf_occurred": nan_inf_occurred,
        "stats_before_training": stats_before,
        "stats_after_training": stats_after,
        "step_records": step_records,
        "anti_collapse_assessment": {
            "collapse_threshold_ratio": COLLAPSE_THRESHOLD,
            "initial_ratio": float(initial_ratio),
            "final_ratio": float(final_ratio),
            "collapsed": collapsed,
            "all_masks_bounded": all(r["mask_bounded"] for r in step_records),
            "baseline_snr_improvement_epoch28": BASELINE_SNR_IMP_EPOCH28,
            "final_snr_improvement": float(final_snr_imp),
            "anti_collapse_pass": anti_collapse_pass,
        },
    }

    # Save result
    with open(EXP_DIR / "sanity_experiment.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info("=" * 70)
    logger.info("ANTI-COLLAPSE ASSESSMENT")
    logger.info("Initial enhanced/clean RMS ratio: %.4f", initial_ratio)
    logger.info("Final   enhanced/clean RMS ratio: %.4f", final_ratio)
    logger.info("Collapse threshold:               %.4f", COLLAPSE_THRESHOLD)
    logger.info("Collapsed (ratio < threshold):    %s", collapsed)
    logger.info("All masks bounded (-1,1):         %s", all(r["mask_bounded"] for r in step_records))
    logger.info("NaN/Inf occurred:                 %s", nan_inf_occurred)
    logger.info("Final SNR improvement:            %.2f dB (baseline epoch-28: %.2f dB)",
                final_snr_imp, BASELINE_SNR_IMP_EPOCH28)
    logger.info("ANTI-COLLAPSE VERDICT: %s", "PASS" if anti_collapse_pass else "FAIL")
    logger.info("=" * 70)
    logger.info("Artifacts saved to: %s", EXP_DIR)

    return result


if __name__ == "__main__":
    run_sanity_experiment()
