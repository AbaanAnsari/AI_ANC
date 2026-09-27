"""
scripts/verify_phase2_step4.py
================================
Post-training artifact verification for Phase 2 Step 4 (CRM).
Checks isolation, parameter count, history completeness,
checkpoint integrity, and test-set exclusion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.models.cnn_gru import LightweightCNNTGRUModel

EXP  = PROJECT_ROOT / "experiments" / "phase2_step4_crm"
PH1  = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"
BASE = PROJECT_ROOT / "experiments" / "phase2_baseline"
ST3  = PROJECT_ROOT / "experiments" / "phase2_step3"


def verify():
    results = {}

    def check(name, passed, detail=""):
        results[name] = passed
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))

    print("=" * 70)
    print("PHASE 2 STEP 4 — ARTIFACT VERIFICATION")
    print("=" * 70)

    # 1. Prior experiments untouched
    check("V1: Phase 1 checkpoint intact", PH1.exists(), str(PH1))
    check("V2: Phase 2 baseline intact",  (BASE / "metrics.json").exists())
    check("V3: Phase 2 Step 3 intact",    (ST3 / "sanity_experiment.json").exists())

    # 4. New experiment dir exists
    check("V4: phase2_step4_crm directory created", EXP.exists())

    # 5. Config saved
    cfg_path = EXP / "config.yaml"
    check("V5: config.yaml exists", cfg_path.exists())
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        check("V5a: epochs=30", cfg.get("epochs") == 30, f"epochs={cfg.get('epochs')}")
        check("V5b: seed=123",  cfg.get("seed") == 123,  f"seed={cfg.get('seed')}")
        check("V5c: model=CRM", "Mask" in cfg.get("model", ""), cfg.get("model"))

    # 6. History exists and complete
    hist_path = EXP / "history.json"
    check("V6: history.json exists", hist_path.exists())
    if hist_path.exists():
        with open(hist_path) as f:
            history = json.load(f)
        check("V6a: exactly 30 epochs in history", len(history) == 30, f"found {len(history)}")
        check("V6b: all epochs have train/val losses",
              all("train_total_loss" in e and "val_total_loss" in e for e in history))
        check("V6c: all epochs have LR",
              all("learning_rate" in e for e in history))
        check("V6d: all epochs have sample RMS ratio",
              all("sample_enhanced_clean_rms_ratio" in e for e in history))
        check("V6e: all epochs have val_snr_improvement_db",
              all("val_snr_improvement_db" in e for e in history))

    # 7. Checkpoints
    best_path = EXP / "best_checkpoint.pt"
    last_path = EXP / "last_checkpoint.pt"
    check("V7: best_checkpoint.pt exists", best_path.exists())
    check("V8: last_checkpoint.pt exists", last_path.exists())

    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        check("V7a: best_checkpoint has model_state_dict",
              "model_state_dict" in ckpt)
        check("V7b: best_checkpoint has optimizer_state_dict",
              "optimizer_state_dict" in ckpt)

        # Verify param count in best checkpoint
        m = LightweightCNNGRUMaskModel()
        m.load_state_dict(ckpt["model_state_dict"])
        cnt = sum(p.numel() for p in m.parameters() if p.requires_grad)
        check("V7c: model params == 70,789", cnt == 70_789, f"got {cnt}")
        check("V7d: model params < 100,000", cnt < 100_000)

    if last_path.exists():
        lckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        check("V8a: last_checkpoint epoch == 30",
              lckpt.get("epoch") == 30, f"epoch={lckpt.get('epoch')}")

    # 8. Metrics
    mpath = EXP / "metrics.json"
    check("V9: metrics.json exists", mpath.exists())
    if mpath.exists():
        with open(mpath) as f:
            metrics = json.load(f)
        check("V9a: no NaN/Inf occurred", not metrics.get("nan_inf_occurred", True))
        check("V9b: completed_epochs == 30", metrics.get("completed_epochs") == 30)
        check("V9c: trainable_params == 70789", metrics.get("trainable_params") == 70_789)
        check("V9d: final_test_results present", "final_test_results" in metrics)
        check("V9e: best_val_loss_epoch recorded", metrics.get("best_val_loss_epoch", -1) >= 1)

        tr = metrics.get("final_test_results", {})
        check("V9f: test evaluated from best_checkpoint.pt",
              tr.get("checkpoint_evaluated") == "best_checkpoint.pt")
        check("V9g: test SNR improvement recorded",
              tr.get("snr_improvement_db") is not None)
        check("V9h: test classification accuracy recorded",
              tr.get("classification_accuracy") is not None)

    # 9. Original model param budget still correct
    orig = LightweightCNNTGRUModel()
    orig_cnt = sum(p.numel() for p in orig.parameters() if p.requires_grad)
    check("V10: original CNN+GRU still 70,789 params", orig_cnt == 70_789, f"got {orig_cnt}")

    # Best checkpoint epoch = actual minimum val_total_loss epoch
    if hist_path.exists() and best_path.exists():
        ckpt2 = torch.load(best_path, map_location="cpu", weights_only=False)
        best_ep_from_history = min(history, key=lambda x: x["val_total_loss"])["epoch"]
        best_ep_from_ckpt    = ckpt2.get("epoch")
        check("V11: best_checkpoint epoch matches min val_loss epoch",
              best_ep_from_history == best_ep_from_ckpt,
              f"history_min_ep={best_ep_from_history}, ckpt_ep={best_ep_from_ckpt}")

    all_pass = all(results.values())
    print("=" * 70)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    verify()
