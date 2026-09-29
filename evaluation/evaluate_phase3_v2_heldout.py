"""
evaluation/evaluate_phase3_v2_heldout.py
========================================
Comprehensive evaluation of a trained model checkpoint on the held-out
TEST split (phase3_v2/test_manifest.jsonl) containing unseen speakers and noise recordings.

Computes:
- Input SNR, Output SNR, ΔSNR
- 95% Confidence interval for ΔSNR
- Breakdown by noise class (stationary, non-stationary, impulsive)
- Breakdown by input SNR levels
- Classification Accuracy, Macro-F1, Confusion Matrix
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import Phase1Dataset
from src.features.stft import compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("heldout_eval")

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "phase3_v2" / "test_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    min_len = min(len(clean), len(noisy))
    c = clean[:min_len]
    n = noisy[:min_len]
    p_clean = float(np.mean(c ** 2))
    p_err = float(np.mean((n - c) ** 2) + 1e-15)
    return 10.0 * np.log10(p_clean / p_err)


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: str = "cpu",
    max_samples: int | None = None,
) -> Dict[str, Any]:
    logger.info("Loading checkpoint from %s...", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    logger.info("Initializing held-out test dataset (frozen epoch=0)...")
    test_ds = Phase1Dataset(
        manifest_path=MANIFEST_PATH,
        dataset_root=DATASET_ROOT,
        seed=20260929,
        is_validation=True,  # deterministic epoch 0
    )
    n_total = len(test_ds)
    if max_samples:
        n_total = min(n_total, max_samples)
    logger.info("Evaluating %d test items...", n_total)

    delta_snrs = []
    input_snrs = []
    output_snrs = []
    class_trues = []
    class_preds = []

    # Per-class accumulators
    class_names = ["stationary", "non-stationary", "impulsive"]
    by_class = {c: [] for c in class_names}
    by_snr_bin = {"low (<0)": [], "mid (0-10)": [], "high (>10)": []}

    with torch.no_grad():
        for i in range(n_total):
            item = test_ds[i]
            feat = item["noisy_features"].unsqueeze(0).to(device)
            enh_ri, cls_logits, _ = model(feat)

            # Reconstruct enhanced waveform via ISTFT
            enh_ri_np = enh_ri.squeeze(0).cpu().numpy()
            enh_complex = (enh_ri_np[0] + 1j * enh_ri_np[1]).astype(np.complex64)
            clean_wav = item["clean_target"].numpy()
            m1_wav = item["m1_waveform"].numpy()
            enh_wav = compute_istft(enh_complex, length=len(clean_wav))

            # SNR
            snr_in = compute_snr(clean_wav, m1_wav)
            snr_out = compute_snr(clean_wav, enh_wav)
            d_snr = snr_out - snr_in

            delta_snrs.append(d_snr)
            input_snrs.append(snr_in)
            output_snrs.append(snr_out)

            # Class prediction
            pred_cls = int(torch.argmax(cls_logits, dim=-1).item())
            true_cls = int(item["noise_class_label"].item())
            class_trues.append(true_cls)
            class_preds.append(pred_cls)

            # Accumulate by class
            cname = class_names[true_cls] if true_cls < len(class_names) else "unknown"
            if cname in by_class:
                by_class[cname].append(d_snr)

            # Accumulate by SNR bin
            if snr_in < 0:
                by_snr_bin["low (<0)"].append(d_snr)
            elif snr_in <= 10:
                by_snr_bin["mid (0-10)"].append(d_snr)
            else:
                by_snr_bin["high (>10)"].append(d_snr)

            if (i + 1) % 50 == 0 or (i + 1) == n_total:
                logger.info(
                    "Processed %d/%d: mean ΔSNR=%.2f dB, acc=%.1f%%",
                    i + 1,
                    n_total,
                    np.mean(delta_snrs),
                    100.0 * np.mean(np.array(class_preds) == np.array(class_trues)),
                )

    d_arr = np.array(delta_snrs)
    mean_d = float(np.mean(d_arr))
    std_d = float(np.std(d_arr, ddof=1))
    ci_95 = float(1.96 * std_d / math.sqrt(len(d_arr)))

    # Classification accuracy & Macro-F1
    trues_arr = np.array(class_trues)
    preds_arr = np.array(class_preds)
    acc = float(np.mean(trues_arr == preds_arr))

    # Confusion matrix & per-class F1
    f1_list = []
    conf_mat = np.zeros((3, 3), dtype=int)
    for t, p in zip(class_trues, class_preds):
        if 0 <= t < 3 and 0 <= p < 3:
            conf_mat[t, p] += 1

    for c in range(3):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    macro_f1 = float(np.mean(f1_list))

    results = {
        "checkpoint": str(checkpoint_path),
        "total_samples": len(delta_snrs),
        "mean_input_snr_db": round(float(np.mean(input_snrs)), 2),
        "mean_output_snr_db": round(float(np.mean(output_snrs)), 2),
        "mean_delta_snr_db": round(mean_d, 2),
        "std_delta_snr_db": round(std_d, 2),
        "ci_95_delta_snr_db": round(ci_95, 2),
        "per_class_delta_snr": {
            k: round(float(np.mean(v)), 2) if len(v) > 0 else None
            for k, v in by_class.items()
        },
        "per_class_counts": {k: len(v) for k, v in by_class.items()},
        "per_snr_bin_delta_snr": {
            k: round(float(np.mean(v)), 2) if len(v) > 0 else None
            for k, v in by_snr_bin.items()
        },
        "classification_accuracy": round(acc * 100.0, 2),
        "classification_macro_f1": round(macro_f1, 4),
        "confusion_matrix": conf_mat.tolist(),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out Test Split Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test items")
    args = parser.parse_args()

    res = evaluate_checkpoint(Path(args.checkpoint), max_samples=args.max_samples)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved held-out test evaluation results to %s", out_p)
