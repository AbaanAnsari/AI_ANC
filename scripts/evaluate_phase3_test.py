"""
scripts/evaluate_phase3_test.py
===============================
Evaluates Step 6 Baseline and Phase 3 checkpoints (D and F) on the full frozen held-out test set (202 samples).
Computes exact SNR improvement, STOI, PESQ, classification accuracy, and per-class accuracy.
"""

import json
import logging
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

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
from scripts.run_phase2_step6_targeted_crm_full import TEST_MANIFEST, DATASET_ROOT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_phase3_test")


@torch.no_grad()
def evaluate_checkpoint_on_test(ckpt_path: Path, exp_name: str, device: str = "cpu"):
    logger.info("Evaluating %s: %s", exp_name, ckpt_path)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LightweightCNNGRUMaskModel().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    test_loader = create_dataloader(
        TEST_MANIFEST, DATASET_ROOT, batch_size=4, shuffle=False
    )

    all_input_snrs = []
    all_output_snrs = []
    all_snr_imps = []
    all_noisy_stois = []
    all_enhanced_stois = []
    all_noisy_pesqs = []
    all_enhanced_pesqs = []

    all_preds = []
    all_targets = []

    for batch in test_loader:
        noisy_feat = batch["noisy_features"].to(device)  # (B, 2, F, T)
        clean_target = batch["target_stft"].to(device)  # (B, F, T) complex
        labels = batch["noise_class_label"].to(device)  # (B,)

        enhanced_stft, logits, predicted_mask = model(noisy_feat)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        clean_wav = batch["clean_target"].numpy()
        B = noisy_feat.size(0)
        for b in range(B):
            enh_ri = enhanced_stft[b].cpu()  # (2, F, T)
            noisy_ri = noisy_feat[b].cpu()  # (2, F, T)
            enh_complex = (enh_ri[0] + 1j * enh_ri[1]).numpy()
            noisy_complex = (noisy_ri[0] + 1j * noisy_ri[1]).numpy()
            c_w = clean_wav[b]

            enh_wav = compute_istft(enh_complex, length=c_w.shape[0])
            n_w = compute_istft(noisy_complex, length=c_w.shape[0])
            e_w = enh_wav

            snr_res = compute_snr_improvement(c_w, n_w, e_w)
            all_input_snrs.append(snr_res["input_snr_db"])
            all_output_snrs.append(snr_res["output_snr_db"])
            all_snr_imps.append(snr_res["snr_improvement"])

            if stoi_available():
                n_s = compute_stoi(c_w, n_w)
                e_s = compute_stoi(c_w, e_w)
                if n_s is not None and e_s is not None:
                    all_noisy_stois.append(n_s)
                    all_enhanced_stois.append(e_s)

            if pesq_available():
                n_p = compute_pesq(c_w, n_w)
                e_p = compute_pesq(c_w, e_w)
                if n_p is not None and e_p is not None:
                    all_noisy_pesqs.append(n_p)
                    all_enhanced_pesqs.append(e_p)

    cls_metrics = compute_classification_metrics(all_preds, all_targets)

    mean_in_snr = float(torch.tensor(all_input_snrs).mean())
    mean_out_snr = float(torch.tensor(all_output_snrs).mean())
    mean_snr_imp = float(torch.tensor(all_snr_imps).mean())

    mean_noisy_stoi = (
        float(torch.tensor(all_noisy_stois).mean()) if all_noisy_stois else None
    )
    mean_enhanced_stoi = (
        float(torch.tensor(all_enhanced_stois).mean()) if all_enhanced_stois else None
    )

    mean_noisy_pesq = (
        float(torch.tensor(all_noisy_pesqs).mean()) if all_noisy_pesqs else None
    )
    mean_enhanced_pesq = (
        float(torch.tensor(all_enhanced_pesqs).mean()) if all_enhanced_pesqs else None
    )

    results = {
        "experiment": exp_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("epoch", "unknown"),
        "test_samples": len(all_preds),
        "classification_accuracy_pct": round(cls_metrics["accuracy"] * 100, 2),
        "per_class_accuracy": {
            k: round(v * 100, 2) for k, v in cls_metrics["per_class_accuracy"].items()
        },
        "input_snr_db": round(mean_in_snr, 2),
        "enhanced_snr_db": round(mean_out_snr, 2),
        "snr_improvement_db": round(mean_snr_imp, 2),
        "noisy_stoi": (
            round(mean_noisy_stoi, 4) if mean_noisy_stoi is not None else None
        ),
        "enhanced_stoi": (
            round(mean_enhanced_stoi, 4) if mean_enhanced_stoi is not None else None
        ),
        "stoi_improvement": (
            round(mean_enhanced_stoi - mean_noisy_stoi, 4)
            if (mean_enhanced_stoi is not None and mean_noisy_stoi is not None)
            else None
        ),
        "noisy_pesq": (
            round(mean_noisy_pesq, 4) if mean_noisy_pesq is not None else None
        ),
        "enhanced_pesq": (
            round(mean_enhanced_pesq, 4) if mean_enhanced_pesq is not None else None
        ),
        "pesq_improvement": (
            round(mean_enhanced_pesq - mean_noisy_pesq, 4)
            if (mean_enhanced_pesq is not None and mean_noisy_pesq is not None)
            else None
        ),
    }

    logger.info("=== HELD-OUT TEST RESULTS: %s ===", exp_name)
    logger.info(
        "Classification Accuracy : %.2f%%", results["classification_accuracy_pct"]
    )
    logger.info(
        "SNR (Noisy -> Enhanced)  : %.2f dB -> %.2f dB (Imp: %+.2f dB)",
        results["input_snr_db"],
        results["enhanced_snr_db"],
        results["snr_improvement_db"],
    )
    logger.info(
        "STOI (Noisy -> Enhanced) : %.4f -> %.4f (Imp: %+.4f)",
        results["noisy_stoi"] or 0,
        results["enhanced_stoi"] or 0,
        results["stoi_improvement"] or 0,
    )
    logger.info(
        "PESQ (Noisy -> Enhanced) : %.4f -> %.4f (Imp: %+.4f)",
        results["noisy_pesq"] or 0,
        results["enhanced_pesq"] or 0,
        results["pesq_improvement"] or 0,
    )

    return results


if __name__ == "__main__":
    eval_results = {}

    # Evaluate Step 6 baseline checkpoint
    step6_ckpt = Path("experiments/phase2_step6_targeted_crm_full/best_checkpoint.pt")
    if step6_ckpt.exists():
        eval_results["step6_baseline"] = evaluate_checkpoint_on_test(
            step6_ckpt, "Step 6 Baseline"
        )

    # Evaluate Phase 3 Exp D
    exp_d_ckpt = Path("experiments/phase3_D/best_checkpoint.pt")
    if exp_d_ckpt.exists():
        eval_results["phase3_exp_D"] = evaluate_checkpoint_on_test(
            exp_d_ckpt, "Phase 3 Exp D (Balanced Composite)"
        )

    # Evaluate Phase 3 Exp F
    exp_f_ckpt = Path("experiments/phase3_F/best_checkpoint.pt")
    if exp_f_ckpt.exists():
        eval_results["phase3_exp_F"] = evaluate_checkpoint_on_test(
            exp_f_ckpt, "Phase 3 Exp F (Warm Start)"
        )

    out_file = Path("experiments/phase3_analysis/phase3_heldout_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Heldout evaluation results written to: %s", out_file)
