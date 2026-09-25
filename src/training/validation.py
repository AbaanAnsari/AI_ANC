"""
src/training/validation.py — Phase 1 Step 16 Validation Engine
===============================================================
Formal model evaluation on validation / test sets.

Design rules:
  - Loads from checkpoint; NEVER uses random weights silently.
  - Calls model.eval() + torch.no_grad(); never backward()/step().
  - Parameters are bit-exactly unchanged by evaluation.
  - All metrics use the actual clean target waveform.
  - STOI/PESQ report UNAVAILABLE if the package is missing.
  - Deterministic: same checkpoint + same seed → same results.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import (
    compute_classification_metrics,
    validate_labels,
    NOISE_CLASS_NAMES,
)
from evaluation.snr import compute_snr_improvement
from evaluation.stoi import compute_stoi, stoi_available, stoi_unavailable_message
from evaluation.pesq import compute_pesq, pesq_available, pesq_unavailable_message
from src.features.stft import compute_istft
from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)

EXPECTED_PARAM_COUNT = 70_789
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[LightweightCNNTGRUModel, dict]:
    """
    Load the CNN+GRU model from a saved checkpoint.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If model state_dict cannot be loaded.

    Returns
    -------
    (model, checkpoint_dict)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Cannot evaluate with random weights. "
            "Provide a valid checkpoint file."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} is missing 'model_state_dict'. "
            "File may be corrupt."
        )

    model = LightweightCNNTGRUModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params != EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"Loaded model has {trainable_params} trainable parameters; "
            f"expected {EXPECTED_PARAM_COUNT}. Checkpoint may be from a different architecture."
        )

    model.to(torch.device(device))
    logger.info(
        "Loaded checkpoint: %s | epoch=%s | best_val_loss=%s | params=%d",
        checkpoint_path.name,
        checkpoint.get("epoch", "?"),
        checkpoint.get("best_val_loss", "?"),
        trainable_params,
    )
    return model, checkpoint


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader,
    loss_fn: MultiTaskLoss,
    device: str | torch.device = "cpu",
    sample_rate: int = SAMPLE_RATE,
) -> Dict:
    """
    Run evaluation over a full DataLoader.

    Parameters
    ----------
    model : nn.Module
        CNN+GRU model (weights frozen during this call).
    dataloader : DataLoader
        Produces batches with keys:
            noisy_features    (B, 2, 257, T)  float32
            target_stft       (B, 257, T)     complex64
            noise_class_label (B,)            int64
            m1_waveform       (B, N)          float32  — used as noisy waveform
            clean_target      (B, N)          float32  — clean reference
    loss_fn : MultiTaskLoss
        Frozen loss function (existing weights).
    device : str | torch.device
    sample_rate : int

    Returns
    -------
    dict containing all aggregated evaluation results.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    # Clone params to verify they are unchanged after evaluation
    params_before = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }

    total_loss_sum = 0.0
    enh_loss_sum = 0.0
    cls_loss_sum = 0.0
    n_batches = 0
    n_samples = 0

    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []

    input_snr_vals: List[float] = []
    output_snr_vals: List[float] = []
    snr_improvement_vals: List[float] = []

    noisy_stoi_vals: List[float] = []
    enhanced_stoi_vals: List[float] = []
    noisy_pesq_vals: List[float] = []
    enhanced_pesq_vals: List[float] = []

    snr_distribution: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            noisy_features = batch["noisy_features"].to(device, non_blocking=True)
            target_stft = batch["target_stft"].to(device, non_blocking=True)
            noise_class_label = batch["noise_class_label"].to(device, non_blocking=True)

            # Forward pass
            enhanced_output, classification_logits = model(noisy_features)

            # Loss
            loss_dict = loss_fn(
                enhanced_output,
                classification_logits,
                target_stft,
                noise_class_label,
            )
            B = noisy_features.shape[0]
            total_loss_sum += loss_dict["total_loss"].item()
            enh_loss_sum += loss_dict["enhancement_loss"].item()
            cls_loss_sum += loss_dict["classification_loss"].item()
            n_batches += 1
            n_samples += B

            # Classification: argmax of logits
            pred_labels = torch.argmax(classification_logits, dim=1).cpu().numpy()
            true_labels = noise_class_label.cpu().numpy()
            all_true_labels.extend(true_labels.tolist())
            all_pred_labels.extend(pred_labels.tolist())

            # SNR-tracking from batch metadata
            if "target_snr_db" in batch:
                snr_vals = batch["target_snr_db"]
                if isinstance(snr_vals, torch.Tensor):
                    snr_distribution.extend(snr_vals.cpu().numpy().tolist())
                else:
                    snr_distribution.extend(float(v) for v in snr_vals)

            # Per-sample waveform-level metrics
            # We have: clean_target (B, N), noisy_features (B, 2, 257, T) → noisy STFT
            # Reconstruct noisy waveform from noisy_features via ISTFT if m1_waveform absent.
            has_clean = "clean_target" in batch
            has_noisy_wav = "m1_waveform" in batch
            has_noisy_stft = "noisy_features" in batch  # always true in our DataLoader

            enhanced_np = enhanced_output.cpu().numpy()  # (B, 2, 257, T)

            if has_clean:
                for i in range(B):
                    clean_wav = _to_numpy_wav(batch["clean_target"], i)

                    # Noisy waveform: prefer m1_waveform if available, else reconstruct
                    if has_noisy_wav:
                        noisy_wav = _to_numpy_wav(batch["m1_waveform"], i)
                    else:
                        # Reconstruct from noisy_features STFT
                        noisy_feat_np = batch["noisy_features"].cpu().numpy()  # (B, 2, 257, T)
                        n_real = noisy_feat_np[i, 0]  # (257, T)
                        n_imag = noisy_feat_np[i, 1]
                        n_complex = (n_real + 1j * n_imag).astype(np.complex64)
                        noisy_wav = compute_istft(n_complex, length=len(clean_wav))

                    # Reconstruct enhanced waveform via ISTFT
                    enh_real = enhanced_np[i, 0]  # (257, T)
                    enh_imag = enhanced_np[i, 1]
                    enh_complex = (enh_real + 1j * enh_imag).astype(np.complex64)
                    enhanced_wav = compute_istft(enh_complex, length=len(clean_wav))

                    # SNR
                    snr_res = compute_snr_improvement(clean_wav, noisy_wav, enhanced_wav)
                    input_snr_vals.append(snr_res["input_snr_db"])
                    output_snr_vals.append(snr_res["output_snr_db"])
                    snr_improvement_vals.append(snr_res["snr_improvement"])

                    # STOI (if available)
                    if stoi_available():
                        ns = compute_stoi(clean_wav, noisy_wav, sample_rate)
                        es = compute_stoi(clean_wav, enhanced_wav, sample_rate)
                        if ns is not None:
                            noisy_stoi_vals.append(ns)
                        if es is not None:
                            enhanced_stoi_vals.append(es)

                    # PESQ (if available)
                    if pesq_available():
                        np_val = compute_pesq(clean_wav, noisy_wav, sample_rate)
                        ep_val = compute_pesq(clean_wav, enhanced_wav, sample_rate)
                        if np_val is not None:
                            noisy_pesq_vals.append(np_val)
                        if ep_val is not None:
                            enhanced_pesq_vals.append(ep_val)

    if n_batches == 0:
        raise RuntimeError("DataLoader produced zero batches during evaluation.")

    # Verify parameters unchanged
    params_unchanged = True
    for name, param in model.named_parameters():
        if not torch.equal(param.detach(), params_before[name]):
            params_unchanged = False
            logger.error("Parameter '%s' was modified during evaluation!", name)
            break

    # Classification metrics
    assert validate_labels(all_true_labels, {0, 1, 2}), (
        "True labels contain values outside {0, 1, 2}"
    )
    assert validate_labels(all_pred_labels, {0, 1, 2}), (
        "Predicted labels contain values outside {0, 1, 2}"
    )
    cls_metrics = compute_classification_metrics(all_true_labels, all_pred_labels)

    # Build result dict
    result = {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "mean_total_loss": total_loss_sum / n_batches,
        "mean_enhancement_loss": enh_loss_sum / n_batches,
        "mean_classification_loss": cls_loss_sum / n_batches,
        "params_unchanged": params_unchanged,
        # SNR
        "mean_input_snr_db": float(np.mean(input_snr_vals)) if input_snr_vals else None,
        "mean_output_snr_db": float(np.mean(output_snr_vals)) if output_snr_vals else None,
        "mean_snr_improvement": float(np.mean(snr_improvement_vals)) if snr_improvement_vals else None,
        "median_snr_improvement": float(np.median(snr_improvement_vals)) if snr_improvement_vals else None,
        "std_snr_improvement": float(np.std(snr_improvement_vals)) if snr_improvement_vals else None,
        # STOI
        "stoi_available": stoi_available(),
        "stoi_unavailable_message": None if stoi_available() else stoi_unavailable_message(),
        "mean_noisy_stoi": float(np.mean(noisy_stoi_vals)) if noisy_stoi_vals else None,
        "mean_enhanced_stoi": float(np.mean(enhanced_stoi_vals)) if enhanced_stoi_vals else None,
        # PESQ
        "pesq_available": pesq_available(),
        "pesq_unavailable_message": None if pesq_available() else pesq_unavailable_message(),
        "mean_noisy_pesq": float(np.mean(noisy_pesq_vals)) if noisy_pesq_vals else None,
        "mean_enhanced_pesq": float(np.mean(enhanced_pesq_vals)) if enhanced_pesq_vals else None,
        # Classification
        "classification_accuracy": cls_metrics["accuracy"],
        "per_class_accuracy": cls_metrics["per_class_accuracy"],
        "per_class_precision": cls_metrics["per_class_precision"],
        "per_class_recall": cls_metrics["per_class_recall"],
        "per_class_f1": cls_metrics["per_class_f1"],
        "confusion_matrix": cls_metrics["confusion_matrix"],
        "macro_precision": cls_metrics["macro_precision"],
        "macro_recall": cls_metrics["macro_recall"],
        "macro_f1": cls_metrics["macro_f1"],
        "class_counts": cls_metrics["class_counts"],
        # SNR distribution summary
        "snr_distribution_evaluated": sorted(set(round(s, 1) for s in snr_distribution)),
        "snr_distribution_mean": float(np.mean(snr_distribution)) if snr_distribution else None,
    }
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy_wav(tensor_or_array, index: int) -> np.ndarray:
    """Extract item i from a batch tensor or list and return float32 numpy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array[index].cpu().numpy().astype(np.float32)
    # numpy array
    return np.asarray(tensor_or_array[index], dtype=np.float32)


# ---------------------------------------------------------------------------
# Evaluation report printer
# ---------------------------------------------------------------------------

def print_evaluation_report(result: Dict, title: str = "EVALUATION REPORT") -> None:
    """Print a formatted evaluation report."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Samples : {result['n_samples']}  |  Batches : {result['n_batches']}")
    print(f"  Parameters unchanged : {result['params_unchanged']}")
    print()
    print("  LOSSES")
    print(f"    Total loss       : {result['mean_total_loss']:.6f}")
    print(f"    Enhancement loss : {result['mean_enhancement_loss']:.6f}")
    print(f"    Class loss       : {result['mean_classification_loss']:.6f}")
    print()
    print("  SNR")
    inp = result['mean_input_snr_db']
    out = result['mean_output_snr_db']
    imp = result['mean_snr_improvement']
    print(f"    Input SNR   : {inp:.2f} dB" if inp is not None else "    Input SNR   : N/A")
    print(f"    Output SNR  : {out:.2f} dB" if out is not None else "    Output SNR  : N/A")
    print(f"    SNR Improv. : {imp:.2f} dB" if imp is not None else "    SNR Improv. : N/A")
    print()
    print("  STOI")
    if result["stoi_available"]:
        print(f"    Noisy STOI    : {result['mean_noisy_stoi']}")
        print(f"    Enhanced STOI : {result['mean_enhanced_stoi']}")
    else:
        print(f"    {result['stoi_unavailable_message']}")
    print()
    print("  PESQ")
    if result["pesq_available"]:
        print(f"    Noisy PESQ    : {result['mean_noisy_pesq']}")
        print(f"    Enhanced PESQ : {result['mean_enhanced_pesq']}")
    else:
        print(f"    {result['pesq_unavailable_message']}")
    print()
    print("  CLASSIFICATION")
    print(f"    Accuracy : {result['classification_accuracy']:.4f}")
    for c, name in NOISE_CLASS_NAMES.items():
        acc = result["per_class_accuracy"].get(c, float("nan"))
        count = result["class_counts"].get(c, 0)
        print(f"    Class {c} ({name:14s}) : acc={acc:.4f}  count={count}")
    print()
    print("  CONFUSION MATRIX (rows=true, cols=pred)")
    cm = result["confusion_matrix"]
    header = "         " + "  ".join(f"pred_{c}" for c in range(3))
    print(f"    {header}")
    for i in range(3):
        row = "  ".join(f"{cm[i,j]:7d}" for j in range(3))
        print(f"    true_{i}  {row}")
    print()
    print("  SNR Distribution Evaluated:", result["snr_distribution_evaluated"])
    print(sep)
