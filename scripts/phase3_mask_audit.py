"""
scripts/phase3_mask_audit.py
==============================
Phase 3 Step 1: Audit the current mask distribution to assess
whether the [-1, +1] clamp constraint is limiting quality.

Loads the Step 6 checkpoint, runs inference on validation samples,
and analyzes:
  1. Predicted mask M_pred distribution (min, max, mean, saturation %)
  2. Target mask M_target_raw distribution (before clamping)
  3. Target mask M_target_clamped distribution (after clamping)
  4. Percentage of target mask values clipped by the clamp
  5. Energy ratio of enhanced vs clean waveforms

Outputs JSON to experiments/phase3_analysis/mask_distribution.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloader
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.losses import convert_complex_stft_to_real_imag
from src.training.training_config import set_seed

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
logger = logging.getLogger("mask_audit")

CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "phase2_step6_targeted_crm_full"
    / "best_checkpoint.pt"
)
VAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "val_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase3_analysis"

SEED = 123
BATCH_SIZE = 16
DEVICE = "cpu"
MAX_BATCHES = 20  # audit subset


def compute_raw_target_crm(noisy_features, target_stft, eps=1e-7):
    """Compute target CRM WITHOUT clamping to analyze full range."""
    target_ri = convert_complex_stft_to_real_imag(target_stft)
    X_r = noisy_features[:, 0]
    X_i = noisy_features[:, 1]
    S_r = target_ri[:, 0]
    S_i = target_ri[:, 1]

    power = X_r * X_r + X_i * X_i + eps
    M_r_raw = (S_r * X_r + S_i * X_i) / power
    M_i_raw = (S_i * X_r - S_r * X_i) / power

    return torch.stack([M_r_raw, M_i_raw], dim=1)  # (B, 2, F, T)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.to(DEVICE)
    logger.info("Loaded Step 6 checkpoint (epoch %s)", ckpt.get("epoch", "?"))

    # Load validation data
    val_loader = create_dataloader(
        VAL_MANIFEST,
        DATASET_ROOT,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
    )

    # Collect statistics
    all_pred_mask_real = []
    all_pred_mask_imag = []
    all_target_raw_real = []
    all_target_raw_imag = []
    all_target_clamped_real = []
    all_target_clamped_imag = []
    all_enh_energy = []
    all_clean_energy = []
    all_noisy_energy = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= MAX_BATCHES:
                break

            noisy_features = batch["noisy_features"].to(DEVICE)
            target_stft = batch["target_stft"].to(DEVICE)

            # Model inference
            enhanced_output, classification_logits, mask = model(noisy_features)

            # Predicted mask stats
            all_pred_mask_real.append(mask[:, 0].cpu().numpy().ravel())
            all_pred_mask_imag.append(mask[:, 1].cpu().numpy().ravel())

            # Target mask (raw, unclamped)
            target_raw = compute_raw_target_crm(noisy_features, target_stft)
            all_target_raw_real.append(target_raw[:, 0].cpu().numpy().ravel())
            all_target_raw_imag.append(target_raw[:, 1].cpu().numpy().ravel())

            # Target mask (clamped to [-1, +1])
            target_clamped = torch.clamp(target_raw, -1.0, 1.0)
            all_target_clamped_real.append(target_clamped[:, 0].cpu().numpy().ravel())
            all_target_clamped_imag.append(target_clamped[:, 1].cpu().numpy().ravel())

            # Energy
            target_ri = convert_complex_stft_to_real_imag(target_stft)
            all_enh_energy.append(torch.mean(enhanced_output**2).item())
            all_clean_energy.append(torch.mean(target_ri**2).item())
            all_noisy_energy.append(torch.mean(noisy_features**2).item())

            if (batch_idx + 1) % 5 == 0:
                logger.info("Processed %d/%d batches", batch_idx + 1, MAX_BATCHES)

    # Aggregate
    pred_r = np.concatenate(all_pred_mask_real)
    pred_i = np.concatenate(all_pred_mask_imag)
    raw_r = np.concatenate(all_target_raw_real)
    raw_i = np.concatenate(all_target_raw_imag)
    clamp_r = np.concatenate(all_target_clamped_real)
    clamp_i = np.concatenate(all_target_clamped_imag)

    # Clipping analysis
    clip_frac_r = float(np.mean(np.abs(raw_r) > 1.0))
    clip_frac_i = float(np.mean(np.abs(raw_i) > 1.0))
    clip_frac_total = float(np.mean((np.abs(raw_r) > 1.0) | (np.abs(raw_i) > 1.0)))

    # Saturation analysis (predicted mask near ±1)
    sat_threshold = 0.95
    sat_frac_r = float(np.mean(np.abs(pred_r) > sat_threshold))
    sat_frac_i = float(np.mean(np.abs(pred_i) > sat_threshold))

    def _stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p1": float(np.percentile(arr, 1)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    result = {
        "audit_info": {
            "checkpoint": str(CHECKPOINT),
            "epoch": ckpt.get("epoch"),
            "val_samples_audited": len(pred_r),
            "batches_processed": min(MAX_BATCHES, batch_idx + 1),
        },
        "predicted_mask_real": _stats(pred_r),
        "predicted_mask_imag": _stats(pred_i),
        "target_mask_raw_real": _stats(raw_r),
        "target_mask_raw_imag": _stats(raw_i),
        "target_mask_clamped_real": _stats(clamp_r),
        "target_mask_clamped_imag": _stats(clamp_i),
        "clipping_analysis": {
            "fraction_real_clipped": clip_frac_r,
            "fraction_imag_clipped": clip_frac_i,
            "fraction_any_component_clipped": clip_frac_total,
            "total_values_analyzed": len(raw_r),
        },
        "saturation_analysis": {
            "threshold": sat_threshold,
            "fraction_real_saturated": sat_frac_r,
            "fraction_imag_saturated": sat_frac_i,
        },
        "energy_analysis": {
            "mean_enhanced_energy": float(np.mean(all_enh_energy)),
            "mean_clean_energy": float(np.mean(all_clean_energy)),
            "mean_noisy_energy": float(np.mean(all_noisy_energy)),
            "enhanced_clean_ratio": float(np.mean(all_enh_energy))
            / (float(np.mean(all_clean_energy)) + 1e-10),
        },
    }

    out_path = OUTPUT_DIR / "mask_distribution.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("  PHASE 3 — MASK DISTRIBUTION AUDIT")
    print("=" * 70)
    print(f"  Checkpoint: epoch {ckpt.get('epoch')}")
    print(f"  Values analyzed: {len(pred_r):,}")
    print()
    print("  PREDICTED MASK (tanh-bounded):")
    print(
        f"    Real: [{pred_r.min():.4f}, {pred_r.max():.4f}] mean={pred_r.mean():.4f} std={pred_r.std():.4f}"
    )
    print(
        f"    Imag: [{pred_i.min():.4f}, {pred_i.max():.4f}] mean={pred_i.mean():.4f} std={pred_i.std():.4f}"
    )
    print(
        f"    Saturation (|M| > {sat_threshold}): real={sat_frac_r:.2%}, imag={sat_frac_i:.2%}"
    )
    print()
    print("  TARGET MASK (raw, unclamped S/X):")
    print(f"    Real: [{raw_r.min():.4f}, {raw_r.max():.4f}] mean={raw_r.mean():.4f}")
    print(f"    Imag: [{raw_i.min():.4f}, {raw_i.max():.4f}] mean={raw_i.mean():.4f}")
    print()
    print("  CLIPPING ANALYSIS (values outside [-1, +1]):")
    print(f"    Real component clipped:  {clip_frac_r:.2%}")
    print(f"    Imag component clipped:  {clip_frac_i:.2%}")
    print(f"    Any component clipped:   {clip_frac_total:.2%}")
    print()
    print("  ENERGY RATIO (enhanced / clean):")
    ratio = float(np.mean(all_enh_energy)) / (float(np.mean(all_clean_energy)) + 1e-10)
    print(f"    Enhanced/Clean energy: {ratio:.4f}")
    if ratio < 0.9:
        print(
            "    ⚠ Enhanced energy is LOWER than clean — mask may be attenuating speech!"
        )
    elif ratio > 1.1:
        print(
            "    ⚠ Enhanced energy is HIGHER than clean — possible amplification artifact!"
        )
    else:
        print("    ✓ Energy ratio is within acceptable range.")
    print()
    print(f"  Saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
