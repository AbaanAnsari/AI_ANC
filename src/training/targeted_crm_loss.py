"""
src/training/targeted_crm_loss.py
====================================
Phase 2 Step 5 — Targeted CRM Loss with Explicit Mask Supervision.

CONTEXT
-------
Step 4 showed that CRM (Complex Ratio Mask) prevents amplitude collapse and
improves PESQ, but SNR improvement remains negative (-7.47 dB). The Step 4
loss only compares reconstructed output S_hat to clean target S — it never
directly supervises the mask M_pred against an ideal mask M_target.

PROBLEM
-------
Without direct mask supervision, the model can satisfy L(S_hat, S) ≈ 0 by
learning a mask that generally scales the noisy signal by a near-constant
factor, rather than learning to selectively suppress noise-dominant bins and
preserve speech-dominant bins.

The ideal complex ratio mask M_target = S / X tells the model exactly which
bins to preserve and which to suppress:
  - In speech-dominant bins: |M_target| ≈ 1 → preserve
  - In noise-dominant bins: |M_target| ≈ 0 → suppress

STEP 5 CHANGE: Targeted CRM Loss
---------------------------------
New multi-component enhancement loss:

L_enh =
  0.60 * L_mask         <- Direct supervision of mask M_pred vs M_target
  + 0.35 * L_recon      <- Reconstruction fidelity S_hat vs S
  + 0.05 * L_energy     <- Collapse protection (reduced weight)

where:
  L_mask  = 0.7 * L1(M_pred, M_target) + 0.3 * L2(M_pred, M_target)
  L_recon = 0.7 * L1(S_hat, S)         + 0.3 * L2(S_hat, S)
  L_energy = max(0, E_clean - E_enhanced) / (E_clean + epsilon)

Total:
  L_total = 1.0 * L_enh + 0.10 * L_cls

NUMERICAL STABILITY OF M_TARGET
--------------------------------
M_target = S / X is numerically problematic when |X| ≈ 0.

Naive approach: M_target_r = (S_r*X_r + S_i*X_i) / (|X|^2 + eps)
                M_target_i = (S_i*X_r - S_r*X_i) / (|X|^2 + eps)

This is the Wiener-Hopf optimal complex ratio mask formulation.
The division uses |X|^2 + eps in the denominator (never zero).

The resulting raw mask values can be very large when |X| is small. Since
M_pred uses tanh (bounding each component to (-1,1)), we must bring M_target
into a compatible range for loss computation.

STABILIZATION STRATEGY: tanh(M_target)
We apply tanh to M_target before computing the mask loss:
  M_target_bounded = tanh(M_target_raw)

This is mathematically justified because:
1. M_pred is already tanh-bounded in (-1,1)
2. tanh is monotone: large |M_target_raw| maps to ±1 (saturation)
3. For bins where |X| is large and |S|/|X| is well-defined, tanh(M_target)
   ≈ M_target_raw (nearly linear for |M_target_raw| < 1)
4. For bins where |X| ≈ 0, M_target_raw → ∞, tanh saturates to ±1,
   which is a stable finite value
5. No NaN/Inf can occur: the denominator has eps > 0; tanh has bounded output

Epsilon choice:
  eps = 1e-7 for denominator (|X|^2 + eps)
  This is much smaller than typical STFT magnitudes (order 1e-3 to 1e-1)
  but large enough to prevent true zero denominators.

DO NOT use silent NaN replacement. The formulation itself prevents NaN.

MODEL OUTPUT UNCHANGED:
  M_pred = tanh(W h + b)  [Mr, Mi in (-1,1) each]
  S_hat  = M_pred * X     [complex multiplication, unchanged from Step 3/4]

No architecture changes. Parameter count: 70,789.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import (
    ClassificationLoss,
    convert_complex_stft_to_real_imag,
)


# ---------------------------------------------------------------------------
# Target mask computation
# ---------------------------------------------------------------------------

def compute_target_crm(
    noisy_features: torch.Tensor,
    target_stft: torch.Tensor,
    eps: float = 1e-7,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Compute the numerically stable, clamped ideal Complex Ratio Mask (CRM).

    Formula:
        M_target_raw_r = (S_r * X_r + S_i * X_i) / (|X|^2 + eps)
        M_target_raw_i = (S_i * X_r - S_r * X_i) / (|X|^2 + eps)
        M_target_stabilized_r = torch.clamp(M_target_raw_r, min=clamp_min, max=clamp_max)
        M_target_stabilized_i = torch.clamp(M_target_raw_i, min=clamp_min, max=clamp_max)

    Stabilization rationale:
        1. Numerical stability: |X|^2 + eps ensures the denominator is never zero.
        2. Compatibility: Clamping to [-1.0, 1.0] matches the tanh output dynamic range of M_pred.
        3. Linearity: For uncorrupted/speech bins where |M_raw| <= 1.0, the target mask remains the exact
           linear ratio S/X without compression distortion (unlike tanh(S/X) which warps linear gains).
        4. Bound: Bins with extreme ratios are safely bounded to [-1.0, 1.0] without NaN/Inf.

    Parameters
    ----------
    noisy_features : torch.Tensor
        Shape (B, 2, F, T) — [ch0=real, ch1=imag] of noisy STFT X.
    target_stft : torch.Tensor
        Shape (B, F, T) complex64 OR (B, 2, F, T) float32 — clean STFT S.
    eps : float
        Stabilization epsilon for denominator. Default 1e-7.
    clamp_min : float
        Lower bound for component clipping. Default -1.0.
    clamp_max : float
        Upper bound for component clipping. Default 1.0.

    Returns
    -------
    torch.Tensor
        Shape (B, 2, F, T) — clamped target CRM [ch0=Mr, ch1=Mi].
    """
    if noisy_features.ndim != 4 or noisy_features.shape[1] != 2:
        raise ValueError(
            f"noisy_features must be (B, 2, F, T), got {noisy_features.shape}"
        )

    # Convert target to (B, 2, F, T) float32 if complex
    target_ri = convert_complex_stft_to_real_imag(target_stft)

    X_r = noisy_features[:, 0]  # (B, F, T)
    X_i = noisy_features[:, 1]  # (B, F, T)
    S_r = target_ri[:, 0]       # (B, F, T)
    S_i = target_ri[:, 1]       # (B, F, T)

    # |X|^2 = X_r^2 + X_i^2 — never zero after adding eps
    power = X_r * X_r + X_i * X_i + eps  # (B, F, T), all > 0

    # Complex division: M_target = S / X = (S * conj(X)) / |X|^2
    M_r_raw = (S_r * X_r + S_i * X_i) / power  # (B, F, T)
    M_i_raw = (S_i * X_r - S_r * X_i) / power  # (B, F, T)

    # Safety check: no NaN/Inf should occur, but assert for debugging
    assert torch.all(torch.isfinite(M_r_raw)), "NaN/Inf in M_r_raw"
    assert torch.all(torch.isfinite(M_i_raw)), "NaN/Inf in M_i_raw"

    # Stabilize by clamping components to [-1.0, 1.0], matching M_pred tanh bounds
    M_r_clamped = torch.clamp(M_r_raw, min=clamp_min, max=clamp_max)
    M_i_clamped = torch.clamp(M_i_raw, min=clamp_min, max=clamp_max)

    # Stack to (B, 2, F, T)
    target_mask = torch.stack([M_r_clamped, M_i_clamped], dim=1)

    return target_mask


# ---------------------------------------------------------------------------
# Targeted CRM Enhancement Loss
# ---------------------------------------------------------------------------

class TargetedCRMEnhancementLoss(nn.Module):
    """
    Phase 2 Step 5 — Targeted Complex Ratio Mask Enhancement Loss.

    Combines three terms:
      1. L_mask:   L1+L2 between predicted mask M_pred and ideal mask M_target
      2. L_recon:  L1+L2 between reconstructed output S_hat and clean target S
      3. L_energy: Energy hinge (collapse protection)

    Total:
        L_enh = mask_weight * L_mask + recon_weight * L_recon + energy_weight * L_energy

    Parameters
    ----------
    mask_weight : float
        Weight for mask supervision term (default 0.60).
    recon_weight : float
        Weight for reconstruction fidelity term (default 0.35).
    energy_weight : float
        Weight for energy preservation hinge (default 0.05).
    l1_weight : float
        Weight for L1 within each sub-loss (default 0.7).
    l2_weight : float
        Weight for L2 within each sub-loss (default 0.3).
    eps : float
        Denominator epsilon for target mask computation (default 1e-7).
    """

    def __init__(
        self,
        mask_weight: float = 0.60,
        recon_weight: float = 0.35,
        energy_weight: float = 0.05,
        l1_weight: float = 0.7,
        l2_weight: float = 0.3,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        for name, val in [
            ("mask_weight", mask_weight), ("recon_weight", recon_weight),
            ("energy_weight", energy_weight), ("l1_weight", l1_weight),
            ("l2_weight", l2_weight), ("eps", eps),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")

        self.mask_weight   = float(mask_weight)
        self.recon_weight  = float(recon_weight)
        self.energy_weight = float(energy_weight)
        self.l1_weight     = float(l1_weight)
        self.l2_weight     = float(l2_weight)
        self.eps           = float(eps)

    def forward(
        self,
        enhanced_output: torch.Tensor,
        predicted_mask: torch.Tensor,
        noisy_features: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        enhanced_output : torch.Tensor
            Shape (B, 2, F, T) — S_hat = M_pred * X.
        predicted_mask : torch.Tensor
            Shape (B, 2, F, T) — tanh-bounded predicted mask M_pred.
        noisy_features : torch.Tensor
            Shape (B, 2, F, T) — noisy STFT X (for target mask computation).
        target_stft : torch.Tensor
            Shape (B, F, T) complex64 or (B, 2, F, T) float32 — clean S.

        Returns
        -------
        dict with keys:
            total_enhancement_loss, mask_loss, reconstruction_loss, energy_loss
        """
        if enhanced_output.ndim != 4 or enhanced_output.shape[1] != 2:
            raise ValueError(f"enhanced_output must be (B,2,F,T), got {enhanced_output.shape}")
        if predicted_mask.ndim != 4 or predicted_mask.shape[1] != 2:
            raise ValueError(f"predicted_mask must be (B,2,F,T), got {predicted_mask.shape}")

        target_ri = convert_complex_stft_to_real_imag(target_stft)

        # ---- A: Mask Loss ----
        target_mask = compute_target_crm(noisy_features, target_stft, eps=self.eps)
        mask_l1 = torch.mean(torch.abs(predicted_mask - target_mask))
        mask_l2 = torch.mean((predicted_mask - target_mask) ** 2)
        l_mask = self.l1_weight * mask_l1 + self.l2_weight * mask_l2

        # ---- B: Reconstruction Loss ----
        recon_l1 = torch.mean(torch.abs(enhanced_output - target_ri))
        recon_l2 = torch.mean((enhanced_output - target_ri) ** 2)
        l_recon  = self.l1_weight * recon_l1 + self.l2_weight * recon_l2

        # ---- C: Energy Hinge (collapse protection) ----
        enh_energy   = torch.mean(enhanced_output ** 2)
        clean_energy = torch.mean(target_ri ** 2)
        energy_deficit = F.relu(clean_energy - enh_energy)
        l_energy = energy_deficit / (clean_energy.detach() + 1e-8)

        # ---- Total Enhancement Loss ----
        l_enh = (
            self.mask_weight   * l_mask
            + self.recon_weight  * l_recon
            + self.energy_weight * l_energy
        )

        return {
            "total_enhancement_loss": l_enh,
            "mask_loss":              l_mask,
            "reconstruction_loss":    l_recon,
            "energy_loss":            l_energy,
        }


# ---------------------------------------------------------------------------
# Multi-Task Targeted CRM Loss
# ---------------------------------------------------------------------------

class MultiTaskTargetedCRMLoss(nn.Module):
    """
    Phase 2 Step 5 — Multi-Task Loss with Targeted CRM Supervision.

    L_total = enhancement_weight * L_enh + classification_weight * L_cls

    where L_enh uses TargetedCRMEnhancementLoss.

    Parameters
    ----------
    enhancement_weight : float
        Weight for enhancement loss (default 1.0).
    classification_weight : float
        Weight for classification loss (default 0.10).
    mask_weight : float
        Weight for mask supervision within L_enh (default 0.60).
    recon_weight : float
        Weight for reconstruction within L_enh (default 0.35).
    energy_weight : float
        Weight for energy hinge within L_enh (default 0.05).
    l1_weight : float
        L1 sub-weight within mask and recon losses (default 0.7).
    l2_weight : float
        L2 sub-weight within mask and recon losses (default 0.3).
    eps : float
        Denominator epsilon for target mask (default 1e-7).
    class_weights : torch.Tensor | None
        Optional class weights for CrossEntropy.
    """

    def __init__(
        self,
        enhancement_weight:    float = 1.0,
        classification_weight: float = 0.10,
        mask_weight:   float = 0.60,
        recon_weight:  float = 0.35,
        energy_weight: float = 0.05,
        l1_weight:     float = 0.7,
        l2_weight:     float = 0.3,
        eps: float = 1e-7,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.enhancement_weight    = float(enhancement_weight)
        self.classification_weight = float(classification_weight)

        self.enhancement_loss_fn = TargetedCRMEnhancementLoss(
            mask_weight=mask_weight,
            recon_weight=recon_weight,
            energy_weight=energy_weight,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            eps=eps,
        )
        self.classification_loss_fn = ClassificationLoss(class_weights=class_weights)

    def forward(
        self,
        enhanced_output:        torch.Tensor,
        predicted_mask:         torch.Tensor,
        classification_logits:  torch.Tensor,
        noisy_features:         torch.Tensor,
        target_stft:            torch.Tensor,
        noise_class_label:      torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        enhanced_output : torch.Tensor
            Shape (B, 2, F, T) — S_hat from mask application.
        predicted_mask : torch.Tensor
            Shape (B, 2, F, T) — tanh-bounded predicted mask.
        classification_logits : torch.Tensor
            Shape (B, 3) — noise class logits.
        noisy_features : torch.Tensor
            Shape (B, 2, F, T) — noisy STFT (for target mask).
        target_stft : torch.Tensor
            Shape (B, F, T) complex64 or (B, 2, F, T) float32 — clean STFT.
        noise_class_label : torch.Tensor
            Shape (B,) int64 — noise class labels.

        Returns
        -------
        dict with keys:
            total_loss, enhancement_loss, mask_loss, reconstruction_loss,
            energy_loss, classification_loss
        """
        enh_dict = self.enhancement_loss_fn(
            enhanced_output, predicted_mask, noisy_features, target_stft
        )
        l_enh = enh_dict["total_enhancement_loss"]
        l_cls = self.classification_loss_fn(classification_logits, noise_class_label)

        l_total = (
            self.enhancement_weight    * l_enh
            + self.classification_weight * l_cls
        )

        return {
            "total_loss":          l_total,
            "enhancement_loss":    l_enh,
            "mask_loss":           enh_dict["mask_loss"],
            "reconstruction_loss": enh_dict["reconstruction_loss"],
            "energy_loss":         enh_dict["energy_loss"],
            "classification_loss": l_cls,
        }
