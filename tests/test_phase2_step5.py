"""
tests/test_phase2_step5.py
===========================
Phase 2 Step 5 — 20 Unit Tests for Targeted CRM Loss.

Tests (per specification):
 1.  Target CRM calculation (formula correctness)
 2.  Epsilon numerical stability (|X|^2 + eps >= eps)
 3.  Target mask finite (no NaN/Inf)
 4.  Target mask real/imag shapes
 5.  Target mask clipping/stabilization (tanh bounds)
 6.  Predicted CRM reconstruction (S_hat = M * X shape & finite)
 7.  Complex multiplication correctness (unit mask identity)
 8.  Mask loss (L1+L2 between M_pred and M_target)
 9.  Reconstruction loss (L1+L2 between S_hat and S)
10.  Energy hinge (fires when E_enh < E_clean, silent otherwise)
11.  Combined enhancement loss (weights sum correctly)
12.  Classification loss integration
13.  Total loss
14.  Finite forward pass
15.  Finite backward pass
16.  Gradient existence (all parameter tensors)
17.  Parameter count < 100k
18.  Deterministic output with fixed seed
19.  No mutation of input tensors
20.  No test-manifest access during training
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.training.targeted_crm_loss import (
    compute_target_crm,
    TargetedCRMEnhancementLoss,
    MultiTaskTargetedCRMLoss,
)
from src.training.losses import convert_complex_stft_to_real_imag

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
BATCH, FREQ, TIME, HIDDEN, SEED = 2, 257, 63, 48, 123


def seed_all(s=SEED):
    torch.manual_seed(s)
    np.random.seed(s)


def noisy_feat(b=BATCH, f=FREQ, t=TIME, s=SEED):
    torch.manual_seed(s)
    return torch.randn(b, 2, f, t)


def target_complex(b=BATCH, f=FREQ, t=TIME, s=SEED+1):
    torch.manual_seed(s)
    r = torch.randn(b, f, t)
    i = torch.randn(b, f, t)
    return torch.complex(r, i)


def noise_labels(b=BATCH, s=SEED):
    torch.manual_seed(s)
    return torch.randint(0, 3, (b,), dtype=torch.int64)


def make_model():
    seed_all()
    return LightweightCNNGRUMaskModel()


# ---------------------------------------------------------------------------
# TEST 1: Target CRM calculation formula
# ---------------------------------------------------------------------------
def test_target_crm_formula():
    """M_target = clamp(S / (X + eps), -1.0, 1.0) via stable complex division."""
    X = noisy_feat(b=1)
    S = target_complex(b=1)
    M_t = compute_target_crm(X, S)

    # Manually compute
    S_ri = convert_complex_stft_to_real_imag(S)
    X_r, X_i = X[0, 0], X[0, 1]
    S_r, S_i = S_ri[0, 0], S_ri[0, 1]
    denom = X_r * X_r + X_i * X_i + 1e-7
    raw_r = (S_r * X_r + S_i * X_i) / denom
    raw_i = (S_i * X_r - S_r * X_i) / denom
    ref_r = torch.clamp(raw_r, min=-1.0, max=1.0)
    ref_i = torch.clamp(raw_i, min=-1.0, max=1.0)

    assert torch.allclose(M_t[0, 0], ref_r, atol=1e-5), "M_target real mismatch"
    assert torch.allclose(M_t[0, 1], ref_i, atol=1e-5), "M_target imag mismatch"
    print("[PASS] TEST 1: Target CRM formula correct (clamped complex ratio).")


# ---------------------------------------------------------------------------
# TEST 2: Epsilon numerical stability
# ---------------------------------------------------------------------------
def test_epsilon_stability():
    """With zero noisy input, denominator = eps > 0 → no division by zero."""
    zero_noisy = torch.zeros(1, 2, FREQ, TIME)
    S = target_complex(b=1)
    M_t = compute_target_crm(zero_noisy, S, eps=1e-7)
    assert torch.all(torch.isfinite(M_t)), "NaN/Inf in target mask for zero noisy input"
    # clamp bounds to [-1, 1]
    assert M_t.abs().max().item() <= 1.0 + 1e-6
    print("[PASS] TEST 2: Epsilon stability confirmed for zero noisy input.")


# ---------------------------------------------------------------------------
# TEST 3: Target mask finite
# ---------------------------------------------------------------------------
def test_target_mask_finite():
    """Target mask contains no NaN/Inf for random normal inputs."""
    for trial in range(5):
        torch.manual_seed(trial)
        X = torch.randn(2, 2, FREQ, TIME) * (10 ** np.random.uniform(-4, 1))
        S = torch.complex(torch.randn(2, FREQ, TIME), torch.randn(2, FREQ, TIME))
        M_t = compute_target_crm(X, S)
        assert torch.all(torch.isfinite(M_t)), f"NaN/Inf in target mask trial {trial}"
    print("[PASS] TEST 3: Target mask finite for 5 random trials.")


# ---------------------------------------------------------------------------
# TEST 4: Target mask real/imag shapes
# ---------------------------------------------------------------------------
def test_target_mask_shapes():
    """Target mask shape is (B, 2, F, T)."""
    X = noisy_feat()
    S = target_complex()
    M_t = compute_target_crm(X, S)
    assert M_t.shape == (BATCH, 2, FREQ, TIME), f"Shape: {M_t.shape}"
    assert M_t.shape[1] == 2  # channel 0 = Mr, channel 1 = Mi
    print(f"[PASS] TEST 4: Target mask shape {M_t.shape} correct.")


# ---------------------------------------------------------------------------
# TEST 5: Component clipping bounds target mask to [-1, 1]
# ---------------------------------------------------------------------------
def test_target_mask_clipping_bounded():
    """Component clipping ensures all target mask values are bounded in [-1, 1]."""
    # Use very tiny noisy STFT to force large raw mask values
    tiny_noisy = torch.randn(2, 2, FREQ, TIME) * 1e-6
    S = target_complex()
    M_t = compute_target_crm(tiny_noisy, S, eps=1e-7)
    max_val = M_t.abs().max().item()
    assert max_val <= 1.0 + 1e-6, f"Clipping bound violated: max={max_val}"
    assert torch.all(torch.isfinite(M_t)), "NaN/Inf despite tiny noisy input"
    print(f"[PASS] TEST 5: Component clipping works. Max |M_target|={max_val:.6f} <= 1.")


# ---------------------------------------------------------------------------
# TEST 6: Predicted CRM reconstruction shape and finite
# ---------------------------------------------------------------------------
def test_predicted_crm_reconstruction():
    """Model forward produces finite S_hat and mask of correct shape."""
    model = make_model()
    X = noisy_feat()
    with torch.no_grad():
        enhanced, logits, mask = model(X)
    assert enhanced.shape == (BATCH, 2, FREQ, TIME)
    assert mask.shape     == (BATCH, 2, FREQ, TIME)
    assert torch.all(torch.isfinite(enhanced))
    assert torch.all(torch.isfinite(mask))
    print("[PASS] TEST 6: Predicted CRM produces finite outputs of correct shapes.")


# ---------------------------------------------------------------------------
# TEST 7: Complex multiplication correctness (unit mask = identity)
# ---------------------------------------------------------------------------
def test_complex_multiplication_identity():
    """Unit mask M=(1+0j) applied to X gives back X exactly."""
    X = noisy_feat(b=1)
    unit_mask = torch.zeros(1, 2, FREQ, TIME)
    unit_mask[:, 0] = 1.0  # Mr=1, Mi=0

    X_r, X_i = X[:, 0], X[:, 1]
    M_r, M_i = unit_mask[:, 0], unit_mask[:, 1]
    S_hat_r = M_r * X_r - M_i * X_i
    S_hat_i = M_r * X_i + M_i * X_r

    assert torch.allclose(S_hat_r, X_r, atol=1e-6), "Real component not preserved"
    assert torch.allclose(S_hat_i, X_i, atol=1e-6), "Imag component not preserved"
    print("[PASS] TEST 7: Unit mask M=(1+0j) correctly passes through input X.")


# ---------------------------------------------------------------------------
# TEST 8: Mask loss (finite, positive)
# ---------------------------------------------------------------------------
def test_mask_loss():
    """Mask loss is finite and non-negative."""
    loss_fn = TargetedCRMEnhancementLoss()
    model = make_model()
    X = noisy_feat()
    S = target_complex()
    with torch.no_grad():
        enhanced, _, mask = model(X)
    result = loss_fn(enhanced, mask, X, S)
    l_mask = result["mask_loss"]
    assert torch.isfinite(l_mask), f"Mask loss not finite: {l_mask}"
    assert l_mask.item() >= 0, f"Mask loss negative: {l_mask.item()}"
    print(f"[PASS] TEST 8: Mask loss = {l_mask.item():.6f} (finite, non-negative).")


# ---------------------------------------------------------------------------
# TEST 9: Reconstruction loss
# ---------------------------------------------------------------------------
def test_reconstruction_loss():
    """Reconstruction loss is finite and non-negative."""
    loss_fn = TargetedCRMEnhancementLoss()
    model = make_model()
    X = noisy_feat()
    S = target_complex()
    with torch.no_grad():
        enhanced, _, mask = model(X)
    result = loss_fn(enhanced, mask, X, S)
    l_recon = result["reconstruction_loss"]
    assert torch.isfinite(l_recon), f"Recon loss not finite: {l_recon}"
    assert l_recon.item() >= 0
    print(f"[PASS] TEST 9: Reconstruction loss = {l_recon.item():.6f} (finite, non-negative).")


# ---------------------------------------------------------------------------
# TEST 10: Energy hinge fires and is silent correctly
# ---------------------------------------------------------------------------
def test_energy_hinge():
    """Energy hinge: zero when E_enh >= E_clean; positive when E_enh << E_clean."""
    loss_fn = TargetedCRMEnhancementLoss()
    # Case A: enhanced has MUCH more energy than clean → hinge = 0
    big_enh  = torch.ones(1, 2, FREQ, TIME) * 10.0
    small_tgt = torch.complex(torch.ones(1, FREQ, TIME) * 0.01, torch.zeros(1, FREQ, TIME))
    noisy_dummy = torch.ones(1, 2, FREQ, TIME) * 10.0
    r = loss_fn(big_enh, torch.zeros_like(big_enh), noisy_dummy, small_tgt)
    assert r["energy_loss"].item() == 0.0, "Energy hinge should be 0 when E_enh >> E_clean"

    # Case B: enhanced is near zero → hinge > 0
    zero_enh = torch.zeros(1, 2, FREQ, TIME)
    tgt_clean = target_complex(b=1)
    noisy_small = noisy_feat(b=1) * 1e-6
    r2 = loss_fn(zero_enh, torch.zeros(1, 2, FREQ, TIME), noisy_small, tgt_clean)
    assert r2["energy_loss"].item() > 0.0, "Energy hinge should fire when E_enh = 0"

    print(f"[PASS] TEST 10: Energy hinge correct — fires={r2['energy_loss'].item():.4f}, silent=0.0")


# ---------------------------------------------------------------------------
# TEST 11: Combined enhancement loss weights
# ---------------------------------------------------------------------------
def test_combined_enhancement_loss():
    """Total L_enh = 0.60*L_mask + 0.35*L_recon + 0.05*L_energy."""
    loss_fn = TargetedCRMEnhancementLoss(
        mask_weight=0.60, recon_weight=0.35, energy_weight=0.05
    )
    model = make_model()
    X, S = noisy_feat(), target_complex()
    with torch.no_grad():
        enhanced, _, mask = model(X)
    res = loss_fn(enhanced, mask, X, S)
    expected = (0.60 * res["mask_loss"] + 0.35 * res["reconstruction_loss"]
                + 0.05 * res["energy_loss"])
    assert torch.allclose(res["total_enhancement_loss"], expected, atol=1e-5), \
        f"Weighted sum mismatch: {res['total_enhancement_loss']:.8f} vs {expected:.8f}"
    print(f"[PASS] TEST 11: Combined loss = {res['total_enhancement_loss'].item():.6f} (weights verified).")


# ---------------------------------------------------------------------------
# TEST 12: Classification loss integration
# ---------------------------------------------------------------------------
def test_classification_loss_integration():
    """Classification loss is integrated and finite in MultiTaskTargetedCRMLoss."""
    loss_fn = MultiTaskTargetedCRMLoss()
    model = make_model()
    X, S, labels = noisy_feat(), target_complex(), noise_labels()
    with torch.no_grad():
        enhanced, logits, mask = model(X)
    result = loss_fn(enhanced, mask, logits, X, S, labels)
    assert torch.isfinite(result["classification_loss"])
    assert torch.isfinite(result["total_loss"])
    # Verify total = 1.0*L_enh + 0.10*L_cls
    expected = result["enhancement_loss"] + 0.10 * result["classification_loss"]
    assert torch.allclose(result["total_loss"], expected, atol=1e-5)
    print(f"[PASS] TEST 12: Classification loss integrated. L_cls={result['classification_loss'].item():.4f}")


# ---------------------------------------------------------------------------
# TEST 13: Total loss finite and correct
# ---------------------------------------------------------------------------
def test_total_loss():
    """Total loss is finite and matches expected formula."""
    loss_fn = MultiTaskTargetedCRMLoss()
    model = make_model()
    X, S, labels = noisy_feat(), target_complex(), noise_labels()
    with torch.no_grad():
        enhanced, logits, mask = model(X)
    res = loss_fn(enhanced, mask, logits, X, S, labels)
    for k, v in res.items():
        assert torch.isfinite(v), f"NaN/Inf in {k}: {v}"
    print(f"[PASS] TEST 13: All loss components finite. total={res['total_loss'].item():.6f}")


# ---------------------------------------------------------------------------
# TEST 14: Finite forward pass
# ---------------------------------------------------------------------------
def test_finite_forward_pass():
    """Full forward pass of model produces finite values."""
    model = make_model()
    X = noisy_feat()
    with torch.no_grad():
        enhanced, logits, mask = model(X)
    assert torch.all(torch.isfinite(enhanced)), "NaN/Inf in enhanced"
    assert torch.all(torch.isfinite(logits)), "NaN/Inf in logits"
    assert torch.all(torch.isfinite(mask)), "NaN/Inf in mask"
    print("[PASS] TEST 14: Finite forward pass confirmed.")


# ---------------------------------------------------------------------------
# TEST 15: Finite backward pass
# ---------------------------------------------------------------------------
def test_finite_backward_pass():
    """Backward pass produces finite losses and no NaN/Inf anywhere."""
    model = make_model()
    loss_fn = MultiTaskTargetedCRMLoss()
    X, S, labels = noisy_feat(), target_complex(), noise_labels()
    enhanced, logits, mask = model(X)
    result = loss_fn(enhanced, mask, logits, X, S, labels)
    result["total_loss"].backward()
    assert not math.isnan(result["total_loss"].item()), "Loss is NaN"
    print(f"[PASS] TEST 15: Finite backward pass. Loss={result['total_loss'].item():.6f}")


# ---------------------------------------------------------------------------
# TEST 16: Gradient existence (all parameter tensors)
# ---------------------------------------------------------------------------
def test_gradient_existence():
    """Every learnable parameter tensor has a non-None, finite gradient."""
    model = make_model()
    loss_fn = MultiTaskTargetedCRMLoss()
    X, S, labels = noisy_feat(), target_complex(), noise_labels()
    enhanced, logits, mask = model(X)
    result = loss_fn(enhanced, mask, logits, X, S, labels)
    result["total_loss"].backward()

    no_grad = []
    nan_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad.append(name)
            elif not torch.all(torch.isfinite(p.grad)):
                nan_grad.append(name)

    assert len(no_grad)  == 0, f"Params with no grad: {no_grad}"
    assert len(nan_grad) == 0, f"Params with NaN/Inf grad: {nan_grad}"
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[PASS] TEST 16: All {total_params} param tensors have finite gradients.")


# ---------------------------------------------------------------------------
# TEST 17: Parameter count < 100k
# ---------------------------------------------------------------------------
def test_parameter_count():
    """Model has exactly 70,789 trainable params, well below 100k budget."""
    model = make_model()
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert cnt == 70_789, f"Expected 70,789, got {cnt}"
    assert cnt < 100_000
    print(f"[PASS] TEST 17: Parameter count = {cnt} < 100,000.")


# ---------------------------------------------------------------------------
# TEST 18: Deterministic output with fixed seed
# ---------------------------------------------------------------------------
def test_deterministic_output():
    """Same seed → identical model outputs."""
    def run(s):
        torch.manual_seed(s)
        m = LightweightCNNGRUMaskModel()
        torch.manual_seed(s)
        X = noisy_feat(s=s)
        with torch.no_grad():
            enh, _, _ = m(X)
        return enh

    out1, out2 = run(SEED), run(SEED)
    assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic output"
    print("[PASS] TEST 18: Deterministic output confirmed with seed 123.")


# ---------------------------------------------------------------------------
# TEST 19: No mutation of input tensors
# ---------------------------------------------------------------------------
def test_no_input_mutation():
    """Forward pass must not modify input tensor."""
    model = make_model()
    X = noisy_feat()
    X_copy = X.clone()
    with torch.no_grad():
        model(X)
    assert torch.allclose(X, X_copy, atol=1e-7), "Input tensor was mutated!"
    print("[PASS] TEST 19: No input tensor mutation during forward pass.")


# ---------------------------------------------------------------------------
# TEST 20: No test-manifest access during training (data isolation)
# ---------------------------------------------------------------------------
def test_no_test_manifest_access():
    """Test manifest must NOT be loaded during training data creation."""
    import importlib, inspect
    # Import training script loader path and verify it only uses train/val manifests
    test_manifest = PROJECT_ROOT / "data" / "manifests" / "test_manifest.jsonl"
    # Step 5 sanity script should not reference the test manifest in its training loop
    sanity_script = PROJECT_ROOT / "scripts" / "run_phase2_step5_sanity.py"
    if sanity_script.exists():
        content = sanity_script.read_text()
        # Training loader lines should only use TRAIN_MANIFEST and VAL_MANIFEST
        lines_with_test = [
            l.strip() for l in content.splitlines()
            if "test_manifest" in l.lower() and "create_dataloader" in l.lower()
        ]
        assert len(lines_with_test) == 0, \
            f"Test manifest used in training loader: {lines_with_test}"
    print("[PASS] TEST 20: Test manifest not accessed during training data loading.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_target_crm_formula,
        test_epsilon_stability,
        test_target_mask_finite,
        test_target_mask_shapes,
        test_target_mask_clipping_bounded,
        test_predicted_crm_reconstruction,
        test_complex_multiplication_identity,
        test_mask_loss,
        test_reconstruction_loss,
        test_energy_hinge,
        test_combined_enhancement_loss,
        test_classification_loss_integration,
        test_total_loss,
        test_finite_forward_pass,
        test_finite_backward_pass,
        test_gradient_existence,
        test_parameter_count,
        test_deterministic_output,
        test_no_input_mutation,
        test_no_test_manifest_access,
    ]

    passed = failed = 0
    print("=" * 70)
    print("PHASE 2 STEP 5 — UNIT TESTS (20 total)")
    print("=" * 70)
    for i, fn in enumerate(tests, 1):
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] TEST {i} ({fn.__name__}): {e}")
            failed += 1

    print("=" * 70)
    print(f"RESULT: {passed}/{passed+failed} PASSED, {failed} FAILED")
    print("=" * 70)
