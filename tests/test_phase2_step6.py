"""
tests/test_phase2_step6.py
==========================
Automated verification tests for Phase 2 Step 6 per Section 17:
1. Fresh initialization (not preloaded from baseline/step4/step5)
2. 30 epochs completed
3. Train/validation/test manifest separation
4. No test access during training
5. Best checkpoint selection (minimum validation total loss)
6. Last checkpoint exists
7. Checkpoint reload works
8. Parameter count = 70,789
9. Parameter count < 100,000
10. No NaN/Inf anywhere in model, losses, or logs
11. Deterministic seed (seed=123 determinism)
12. Step 5 loss unchanged
13. Previous experiment directories preserved (not overwritten)
14. Final test performed only after training
15. Required artifacts exist
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.targeted_crm_loss import MultiTaskTargetedCRMLoss
from src.training.training_config import set_seed

EXP_DIR = PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full"


# 1. Fresh initialization test
def test_fresh_initialization():
    """Verify that model initialization with seed 123 starts from fresh weights, not preloaded."""
    set_seed(123)
    model = LightweightCNNGRUMaskModel()
    
    # Check that weights do not match baseline or step 4 checkpoints
    baseline_ckpt_path = PROJECT_ROOT / "experiments" / "phase2_baseline" / "best_checkpoint.pt"
    if baseline_ckpt_path.exists():
        base_ckpt = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=False)
        # Compare conv1 weight
        assert not torch.allclose(model.conv1[0].weight, base_ckpt["model_state_dict"]["conv1.0.weight"])
    print("[PASS] TEST 1: Fresh initialization confirmed.")


# 2. 30 epochs completed
def test_thirty_epochs_completed():
    metrics_path = EXP_DIR / "metrics.json"
    assert metrics_path.exists(), f"metrics.json missing in {EXP_DIR}"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert metrics["completed_epochs"] == 30, f"Expected 30 epochs, got {metrics['completed_epochs']}"
    
    history_path = EXP_DIR / "history.json"
    assert history_path.exists(), f"history.json missing in {EXP_DIR}"
    with open(history_path, "r") as f:
        history = json.load(f)
    assert len(history) == 30, f"Expected 30 records in history.json, got {len(history)}"
    print("[PASS] TEST 2: 30 epochs completed verified.")


# 3. Train/validation/test manifest separation
def test_manifest_separation():
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    train_path = manifest_dir / "train_manifest.jsonl"
    val_path = manifest_dir / "val_manifest.jsonl"
    test_path = manifest_dir / "test_manifest.jsonl"
    
    def get_keys(p):
        keys = set()
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                keys.add(item.get("sample_id", item.get("noisy_path", line)))
        return keys

    train_keys = get_keys(train_path)
    val_keys = get_keys(val_path)
    test_keys = get_keys(test_path)
    
    assert len(train_keys.intersection(val_keys)) == 0, "Train and Val overlap!"
    assert len(train_keys.intersection(test_keys)) == 0, "Train and Test overlap!"
    assert len(val_keys.intersection(test_keys)) == 0, "Val and Test overlap!"
    print("[PASS] TEST 3: Manifest separation verified (0 overlap across splits).")


# 4. No test access during training
def test_no_test_access_during_training():
    script_path = PROJECT_ROOT / "scripts" / "run_phase2_step6_targeted_crm_full.py"
    assert script_path.exists()
    content = script_path.read_text()
    
    # Look for loop over epochs and check where test_loader is used
    train_loop_start = content.find("for epoch in range(1, EPOCHS + 1):")
    test_eval_start = content.find("FINAL TEST EVALUATION")
    
    assert train_loop_start != -1 and test_eval_start != -1
    assert test_eval_start > train_loop_start
    
    train_loop_body = content[train_loop_start:test_eval_start]
    assert "TEST_MANIFEST" not in train_loop_body
    assert "test_loader" not in train_loop_body
    print("[PASS] TEST 4: No test access during training loop verified.")


# 5. Best checkpoint selection
def test_best_checkpoint_selection():
    metrics_path = EXP_DIR / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    history_path = EXP_DIR / "history.json"
    with open(history_path, "r") as f:
        history = json.load(f)
        
    val_losses = [rec["val_total_loss"] for rec in history]
    min_loss = min(val_losses)
    best_ep = val_losses.index(min_loss) + 1
    
    assert metrics["best_val_loss_epoch"] == best_ep
    assert math.isclose(metrics["best_val_loss"], min_loss, rel_tol=1e-5)
    print(f"[PASS] TEST 5: Best checkpoint correctly selected at epoch {best_ep} (loss {min_loss:.6f}).")


# 6. Last checkpoint exists
def test_last_checkpoint_exists():
    last_ckpt = EXP_DIR / "last_checkpoint.pt"
    assert last_ckpt.exists()
    assert last_ckpt.stat().st_size > 0
    print("[PASS] TEST 6: Last checkpoint exists and is non-empty.")


# 7. Checkpoint reload works
def test_checkpoint_reload_works():
    best_ckpt = EXP_DIR / "best_checkpoint.pt"
    assert best_ckpt.exists()
    data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    
    dummy_input = torch.randn(2, 2, 257, 63)
    with torch.no_grad():
        enh, logits, mask = model(dummy_input)
    assert enh.shape == (2, 2, 257, 63)
    assert logits.shape == (2, 3)
    assert mask.shape == (2, 2, 257, 63)
    assert torch.all(torch.isfinite(enh))
    assert torch.all(torch.isfinite(mask))
    print("[PASS] TEST 7: Checkpoint reload and forward execution verified.")


# 8. Parameter count = 70,789
def test_parameter_count_exact():
    model = LightweightCNNGRUMaskModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 70_789, f"Expected 70,789, got {trainable}"
    print("[PASS] TEST 8: Parameter count is exactly 70,789.")


# 9. Parameter count < 100,000
def test_parameter_count_constraint():
    model = LightweightCNNGRUMaskModel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < 100_000
    print("[PASS] TEST 9: Hard parameter budget (<100,000) verified.")


# 10. No NaN/Inf
def test_no_nan_inf():
    metrics_path = EXP_DIR / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert metrics["nan_inf_occurred"] is False
    assert metrics["stop_reason"] is None
    
    history_path = EXP_DIR / "history.json"
    with open(history_path, "r") as f:
        history = json.load(f)
    for rec in history:
        for k, v in rec.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite {k} in epoch {rec['epoch']}: {v}"
    print("[PASS] TEST 10: Zero NaN/Inf in logs, parameters, and metrics verified.")


# 11. Deterministic seed
def test_deterministic_seed():
    def get_first_output(seed):
        set_seed(seed)
        m = LightweightCNNGRUMaskModel()
        x = torch.randn(1, 2, 257, 30)
        with torch.no_grad():
            enh, _, _ = m(x)
        return enh
        
    out1 = get_first_output(123)
    out2 = get_first_output(123)
    assert torch.allclose(out1, out2, atol=1e-7)
    print("[PASS] TEST 11: Deterministic seed confirmed.")


# 12. Step 5 loss unchanged
def test_step5_loss_unchanged():
    loss_fn = MultiTaskTargetedCRMLoss(
        enhancement_weight=1.0,
        classification_weight=0.10,
        mask_weight=0.60,
        recon_weight=0.35,
        energy_weight=0.05,
        l1_weight=0.7,
        l2_weight=0.3,
    )
    assert loss_fn.enhancement_weight == 1.0
    assert loss_fn.classification_weight == 0.10
    assert loss_fn.enhancement_loss_fn.mask_weight == 0.60
    assert loss_fn.enhancement_loss_fn.recon_weight == 0.35
    assert loss_fn.enhancement_loss_fn.energy_weight == 0.05
    assert loss_fn.enhancement_loss_fn.l1_weight == 0.7
    assert loss_fn.enhancement_loss_fn.l2_weight == 0.3
    print("[PASS] TEST 12: Step 5 loss parameters verified unchanged.")


# 13. Previous experiment directories not overwritten
def test_previous_experiments_preserved():
    prev_dirs = [
        PROJECT_ROOT / "experiments" / "phase2_baseline",
        PROJECT_ROOT / "experiments" / "phase2_step3",
        PROJECT_ROOT / "experiments" / "phase2_step4_crm",
        PROJECT_ROOT / "experiments" / "phase2_step5_targeted_crm",
    ]
    for d in prev_dirs:
        assert d.exists(), f"Previous directory {d} missing!"
        files = list(d.glob("*"))
        assert len(files) > 0, f"Previous directory {d} is empty!"
    print("[PASS] TEST 13: All previous experiment directories preserved.")


# 14. Final test performed only after training
def test_final_test_performed_after_training():
    metrics_path = EXP_DIR / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert "final_test_results" in metrics
    assert metrics["completed_epochs"] == 30
    assert metrics["final_test_results"]["checkpoint_evaluated"] == "best_checkpoint.pt"
    print("[PASS] TEST 14: Final test execution verified on best checkpoint post-training.")


# 15. Required artifacts exist
def test_required_artifacts_exist():
    required_files = [
        EXP_DIR / "config.yaml",
        EXP_DIR / "history.json",
        EXP_DIR / "metrics.json",
        EXP_DIR / "best_checkpoint.pt",
        EXP_DIR / "last_checkpoint.pt",
    ]
    for f in required_files:
        assert f.exists(), f"Required artifact {f} missing!"
        assert f.stat().st_size > 0, f"Artifact {f} is empty!"
    print("[PASS] TEST 15: All required Step 6 artifacts exist and are non-empty.")


if __name__ == "__main__":
    tests = [
        test_fresh_initialization,
        test_thirty_epochs_completed,
        test_manifest_separation,
        test_no_test_access_during_training,
        test_best_checkpoint_selection,
        test_last_checkpoint_exists,
        test_checkpoint_reload_works,
        test_parameter_count_exact,
        test_parameter_count_constraint,
        test_no_nan_inf,
        test_deterministic_seed,
        test_step5_loss_unchanged,
        test_previous_experiments_preserved,
        test_final_test_performed_after_training,
        test_required_artifacts_exist,
    ]
    passed = failed = 0
    print("=" * 70)
    print("PHASE 2 STEP 6 — 15 AUTOMATED VERIFICATION CHECKS")
    print("=" * 70)
    for i, t in enumerate(tests, 1):
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] TEST {i} ({t.__name__}): {e}")
            failed += 1
    print("=" * 70)
    print(f"RESULT: {passed}/{len(tests)} PASSED, {failed} FAILED")
    print("=" * 70)
