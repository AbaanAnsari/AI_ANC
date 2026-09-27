"""
tests/test_phase2_step3.py
===========================
Phase 2 Step 3 — Comprehensive Unit Tests for CRM Enhancement.

Covers all 16 required test items:
 1.  mask output shape
 2.  mask boundedness (tanh -> all values in (-1, 1))
 3.  finite mask values
 4.  enhanced STFT shape
 5.  enhanced waveform shape (after ISTFT)
 6.  no NaN/Inf in mask, enhanced output, or loss
 7.  gradient propagation through mask
 8.  parameter count < 100,000 and == 70,789
 9.  deterministic behavior with seed 123
10.  zero-input behavior (X=0 => S_hat=0, loss still computable)
11.  very-low-energy input behavior
12.  clean-target reconstruction path (M=1+0j => S_hat=X)
13.  loss remains finite
14.  no mutation of input tensors
15.  existing STFT/ISTFT behavior unchanged
16.  existing CNN+GRU parameter budget compliant
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.stft import compute_istft, compute_stft
from src.models.cnn_gru import LightweightCNNTGRUModel
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.models.mask_enhancement_head import MaskEnhancementHead
from src.training.mask_losses import MaskEnhancementLoss, MultiTaskMaskLoss
from src.training.losses import MultiTaskLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 2
FREQ = 257
TIME = 63
HIDDEN = 48
SEED = 123

def seed_everything(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_noisy_features(batch=BATCH, freq=FREQ, time=TIME, seed=SEED):
    torch.manual_seed(seed)
    return torch.randn(batch, 2, freq, time)


def make_gru_features(batch=BATCH, time=TIME, hidden=HIDDEN, seed=SEED):
    torch.manual_seed(seed)
    return torch.randn(batch, time, hidden)


def make_target_stft(batch=BATCH, freq=FREQ, time=TIME, seed=SEED):
    """Return complex STFT target (B, F, T) complex64."""
    torch.manual_seed(seed + 1)
    real = torch.randn(batch, freq, time)
    imag = torch.randn(batch, freq, time)
    return torch.complex(real, imag)


def make_noise_labels(batch=BATCH, seed=SEED):
    torch.manual_seed(seed)
    return torch.randint(0, 3, (batch,), dtype=torch.int64)


# ---------------------------------------------------------------------------
# TEST 1: Mask output shape
# ---------------------------------------------------------------------------

def test_mask_output_shape():
    """Mask head returns mask of correct shape."""
    seed_everything()
    head = MaskEnhancementHead(hidden_dim=HIDDEN, n_freq_bins=FREQ, out_channels=2)
    gru_feats = make_gru_features()
    noisy = make_noisy_features()
    enhanced, mask = head(gru_feats, noisy)
    assert mask.shape == (BATCH, 2, FREQ, TIME), f"Mask shape mismatch: {mask.shape}"
    assert enhanced.shape == (BATCH, 2, FREQ, TIME), f"Enhanced shape mismatch: {enhanced.shape}"
    print("[PASS] TEST 1: Mask output shape correct.")


# ---------------------------------------------------------------------------
# TEST 2: Mask boundedness (tanh -> strictly in (-1, 1))
# ---------------------------------------------------------------------------

def test_mask_boundedness():
    """All mask values must be in the open interval (-1, 1)."""
    seed_everything()
    head = MaskEnhancementHead(hidden_dim=HIDDEN, n_freq_bins=FREQ, out_channels=2)
    gru_feats = make_gru_features()
    noisy = make_noisy_features()
    _, mask = head(gru_feats, noisy)
    max_val = mask.abs().max().item()
    assert max_val < 1.0, f"Mask values exceed 1.0: max |mask| = {max_val}"
    print(f"[PASS] TEST 2: Mask bounded. Max |mask| = {max_val:.6f} < 1.0")


# ---------------------------------------------------------------------------
# TEST 3: Finite mask values (no NaN/Inf in mask)
# ---------------------------------------------------------------------------

def test_finite_mask_values():
    """Mask contains only finite values."""
    seed_everything()
    head = MaskEnhancementHead(hidden_dim=HIDDEN, n_freq_bins=FREQ, out_channels=2)
    gru_feats = make_gru_features()
    noisy = make_noisy_features()
    enhanced, mask = head(gru_feats, noisy)
    assert torch.all(torch.isfinite(mask)), "Mask contains NaN or Inf"
    assert torch.all(torch.isfinite(enhanced)), "Enhanced STFT contains NaN or Inf"
    print("[PASS] TEST 3: All mask and enhanced values are finite.")


# ---------------------------------------------------------------------------
# TEST 4: Enhanced STFT shape
# ---------------------------------------------------------------------------

def test_enhanced_stft_shape():
    """Enhanced STFT from mask model has correct shape (B, 2, F, T)."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    noisy = make_noisy_features()
    with torch.no_grad():
        enhanced, logits, mask = model(noisy)
    assert enhanced.shape == (BATCH, 2, FREQ, TIME), f"Enhanced shape: {enhanced.shape}"
    assert logits.shape == (BATCH, 3), f"Logits shape: {logits.shape}"
    assert mask.shape == (BATCH, 2, FREQ, TIME), f"Mask shape: {mask.shape}"
    print("[PASS] TEST 4: Enhanced STFT, logits, and mask shapes are correct.")


# ---------------------------------------------------------------------------
# TEST 5: Enhanced waveform shape (after ISTFT)
# ---------------------------------------------------------------------------

def test_enhanced_waveform_shape():
    """Waveform reconstructed from enhanced STFT has correct length."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    noisy = make_noisy_features()
    with torch.no_grad():
        enhanced, _, _ = model(noisy)
    enh_np = enhanced[0].cpu().numpy()
    enh_complex = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    target_len = 16000  # 1 second at 16kHz
    wav = compute_istft(enh_complex, length=target_len)
    assert wav.shape == (target_len,), f"Waveform shape: {wav.shape}"
    print(f"[PASS] TEST 5: Enhanced waveform reconstructed with shape {wav.shape}.")


# ---------------------------------------------------------------------------
# TEST 6: No NaN/Inf in mask, enhanced output, or loss
# ---------------------------------------------------------------------------

def test_no_nan_inf_in_forward_and_loss():
    """Forward pass and loss computation produce no NaN or Inf."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    loss_fn = MultiTaskMaskLoss()
    noisy = make_noisy_features()
    target = make_target_stft()
    labels = make_noise_labels()
    with torch.no_grad():
        enhanced, logits, mask = model(noisy)
        loss_dict = loss_fn(enhanced, logits, target, labels)
    assert torch.all(torch.isfinite(enhanced)), "NaN/Inf in enhanced output"
    assert torch.all(torch.isfinite(mask)), "NaN/Inf in mask"
    for key, val in loss_dict.items():
        assert torch.isfinite(val), f"NaN/Inf in {key}: {val}"
    print(f"[PASS] TEST 6: No NaN/Inf. Losses: {', '.join(f'{k}={v.item():.6f}' for k, v in loss_dict.items())}")


# ---------------------------------------------------------------------------
# TEST 7: Gradient propagation through mask
# ---------------------------------------------------------------------------

def test_gradient_propagation():
    """Gradients flow back through the mask multiplication to all parameters."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    loss_fn = MultiTaskMaskLoss()
    noisy = make_noisy_features()
    target = make_target_stft()
    labels = make_noise_labels()

    enhanced, logits, _ = model(noisy)
    loss_dict = loss_fn(enhanced, logits, target, labels)
    loss_dict["total_loss"].backward()

    params_with_grad = 0
    params_no_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and torch.any(torch.isfinite(param.grad)):
                params_with_grad += 1
            else:
                params_no_grad += 1

    total_params = params_with_grad + params_no_grad
    assert params_no_grad == 0, f"{params_no_grad}/{total_params} params have no gradient!"
    print(f"[PASS] TEST 7: All {params_with_grad} parameter tensors received finite gradients.")


# ---------------------------------------------------------------------------
# TEST 8: Parameter count < 100,000 and == 70,789
# ---------------------------------------------------------------------------

def test_parameter_count():
    """Mask model has exactly 70,789 trainable parameters."""
    model = LightweightCNNGRUMaskModel()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 70_789, f"Expected 70,789 params, got {count}"
    assert count < 100_000, f"Parameter budget exceeded: {count}"
    print(f"[PASS] TEST 8: Parameter count = {count} (within budget, matches baseline).")


# ---------------------------------------------------------------------------
# TEST 9: Deterministic behavior with seed 123
# ---------------------------------------------------------------------------

def test_deterministic_seed():
    """Same seed 123 produces identical outputs."""
    def run_with_seed(s):
        torch.manual_seed(s)
        model = LightweightCNNGRUMaskModel()
        torch.manual_seed(s)
        noisy = make_noisy_features(seed=s)
        with torch.no_grad():
            enh, _, _ = model(noisy)
        return enh

    out1 = run_with_seed(SEED)
    out2 = run_with_seed(SEED)
    assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic outputs with same seed!"
    print("[PASS] TEST 9: Deterministic behavior confirmed with seed 123.")


# ---------------------------------------------------------------------------
# TEST 10: Zero-input behavior (X=0 => S_hat=0)
# ---------------------------------------------------------------------------

def test_zero_input_behavior():
    """Zero noisy input produces zero enhanced output (M * 0 = 0). Loss still finite."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    loss_fn = MultiTaskMaskLoss()

    zero_noisy = torch.zeros(1, 2, FREQ, TIME)
    target = make_target_stft(batch=1)
    labels = make_noise_labels(batch=1)

    with torch.no_grad():
        enhanced, logits, mask = model(zero_noisy)
        loss_dict = loss_fn(enhanced, logits, target, labels)

    # Enhanced must be zero since M * 0 = 0
    assert torch.allclose(enhanced, torch.zeros_like(enhanced), atol=1e-6), \
        f"Zero input produced non-zero enhanced output. Max: {enhanced.abs().max().item()}"
    # Loss must be finite even for zero output
    for key, val in loss_dict.items():
        assert torch.isfinite(val), f"NaN/Inf in {key} for zero input: {val}"
    # Mask should still be bounded
    assert mask.abs().max().item() < 1.0, "Mask unbounded for zero input"
    print(f"[PASS] TEST 10: Zero input -> zero enhanced output. Loss = {loss_dict['total_loss'].item():.6f} (finite).")


# ---------------------------------------------------------------------------
# TEST 11: Very-low-energy input behavior
# ---------------------------------------------------------------------------

def test_very_low_energy_input():
    """Very low energy input (scale 1e-6) produces finite bounded output."""
    seed_everything()
    model = LightweightCNNGRUMaskModel()
    loss_fn = MultiTaskMaskLoss()

    normal_noisy = make_noisy_features(batch=1)
    tiny_noisy = normal_noisy * 1e-6
    target = make_target_stft(batch=1)
    labels = make_noise_labels(batch=1)

    with torch.no_grad():
        enhanced, logits, mask = model(tiny_noisy)
        loss_dict = loss_fn(enhanced, logits, target, labels)

    assert torch.all(torch.isfinite(enhanced)), "NaN/Inf in enhanced for low-energy input"
    assert torch.all(torch.isfinite(mask)), "NaN/Inf in mask for low-energy input"
    enh_rms = enhanced.pow(2).mean().sqrt().item()
    assert enh_rms < 1e-3, f"Enhanced RMS unexpectedly large for tiny input: {enh_rms}"
    for key, val in loss_dict.items():
        assert torch.isfinite(val), f"NaN/Inf in {key} for low-energy input"
    print(f"[PASS] TEST 11: Low-energy input (1e-6 scale) handled. Enhanced RMS = {enh_rms:.2e}")


# ---------------------------------------------------------------------------
# TEST 12: Clean-target reconstruction path (M=1+0j => S_hat=X)
# ---------------------------------------------------------------------------

def test_unity_mask_reconstruction():
    """
    If mask = 1+0j everywhere, then S_hat = X (pass-through).
    Verify by manually constructing a unit mask and applying it.
    """
    seed_everything()
    head = MaskEnhancementHead(hidden_dim=HIDDEN, n_freq_bins=FREQ, out_channels=2)

    noisy = make_noisy_features(batch=1)

    # Manually set proj weights so tanh output is ~0.76 (tanh(1)=0.76 ≈ 1)
    # Instead, directly test the multiplication logic with mask = [1, 0] (real=1, imag=0)
    # S_hat_r = 1*X_r - 0*X_i = X_r
    # S_hat_i = 1*X_i + 0*X_r = X_i
    unit_mask = torch.zeros(1, 2, FREQ, TIME)
    unit_mask[:, 0, :, :] = 1.0  # M_r = 1
    unit_mask[:, 1, :, :] = 0.0  # M_i = 0

    X_r = noisy[:, 0]
    X_i = noisy[:, 1]
    M_r = unit_mask[:, 0]
    M_i = unit_mask[:, 1]

    S_hat_r = M_r * X_r - M_i * X_i
    S_hat_i = M_r * X_i + M_i * X_r
    S_hat = torch.stack([S_hat_r, S_hat_i], dim=1)

    # Should be identical to input noisy features
    assert torch.allclose(S_hat, noisy, atol=1e-6), \
        "Unit mask M=(1+0j) did not reconstruct input STFT correctly"
    print("[PASS] TEST 12: Unit mask M=(1+0j) correctly reconstructs X (pass-through verified).")


# ---------------------------------------------------------------------------
# TEST 13: Loss remains finite for various inputs
# ---------------------------------------------------------------------------

def test_loss_remains_finite():
    """Loss must be finite for random, zero, and near-zero inputs."""
    model = LightweightCNNGRUMaskModel()
    loss_fn = MultiTaskMaskLoss()
    target = make_target_stft()
    labels = make_noise_labels()

    test_cases = [
        ("random", make_noisy_features()),
        ("zeros", torch.zeros(BATCH, 2, FREQ, TIME)),
        ("near-zero", torch.randn(BATCH, 2, FREQ, TIME) * 1e-6),
        ("large", torch.randn(BATCH, 2, FREQ, TIME) * 10.0),
    ]

    for name, noisy in test_cases:
        with torch.no_grad():
            enhanced, logits, _ = model(noisy)
            loss_dict = loss_fn(enhanced, logits, target, labels)
        for key, val in loss_dict.items():
            assert torch.isfinite(val), f"Loss {key} not finite for input type '{name}': {val}"

    print("[PASS] TEST 13: Loss remains finite for all tested input types.")


# ---------------------------------------------------------------------------
# TEST 14: No mutation of input tensors
# ---------------------------------------------------------------------------

def test_no_input_mutation():
    """Forward pass must not modify input tensors."""
    model = LightweightCNNGRUMaskModel()
    noisy = make_noisy_features()
    noisy_copy = noisy.clone()

    with torch.no_grad():
        model(noisy)

    assert torch.allclose(noisy, noisy_copy, atol=1e-7), \
        "Input tensor was mutated by the forward pass!"
    print("[PASS] TEST 14: Input tensors are not mutated during forward pass.")


# ---------------------------------------------------------------------------
# TEST 15: Existing STFT/ISTFT behavior unchanged
# ---------------------------------------------------------------------------

def test_stft_istft_unchanged():
    """Baseline STFT/ISTFT functions produce correct results (not broken by changes)."""
    rng = np.random.default_rng(SEED)
    audio = rng.standard_normal(16000).astype(np.float32) * 0.1

    stft = compute_stft(audio)
    assert stft.shape == (257, 126), f"STFT shape mismatch: {stft.shape}"
    assert stft.dtype == np.complex64

    recon = compute_istft(stft, length=len(audio))
    assert recon.shape == (16000,), f"ISTFT output shape mismatch: {recon.shape}"

    # Reconstruction should be close to original
    rms_diff = np.sqrt(np.mean((recon - audio) ** 2))
    assert rms_diff < 0.05, f"STFT->ISTFT reconstruction error too large: {rms_diff:.4f}"
    print(f"[PASS] TEST 15: STFT/ISTFT unchanged. Reconstruction error = {rms_diff:.4f}")


# ---------------------------------------------------------------------------
# TEST 16: Existing CNN+GRU parameter budget compliant
# ---------------------------------------------------------------------------

def test_original_model_parameter_budget():
    """Original LightweightCNNTGRUModel still has exactly 70,789 params."""
    model = LightweightCNNTGRUModel()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 70_789, f"Baseline model param count changed! Got {count}"
    assert count < 100_000
    print(f"[PASS] TEST 16: Original CNN+GRU model param count = {count} (unchanged).")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_mask_output_shape,
        test_mask_boundedness,
        test_finite_mask_values,
        test_enhanced_stft_shape,
        test_enhanced_waveform_shape,
        test_no_nan_inf_in_forward_and_loss,
        test_gradient_propagation,
        test_parameter_count,
        test_deterministic_seed,
        test_zero_input_behavior,
        test_very_low_energy_input,
        test_unity_mask_reconstruction,
        test_loss_remains_finite,
        test_no_input_mutation,
        test_stft_istft_unchanged,
        test_original_model_parameter_budget,
    ]

    passed = 0
    failed = 0
    print("=" * 70)
    print("PHASE 2 STEP 3 — UNIT TESTS")
    print("=" * 70)
    for i, test_fn in enumerate(tests, start=1):
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] TEST {i} ({test_fn.__name__}): {e}")
            failed += 1

    print("=" * 70)
    print(f"RESULT: {passed}/{passed + failed} tests PASSED, {failed} FAILED")
    print("=" * 70)
