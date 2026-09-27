"""
src/training/mask_losses.py
============================
Phase 2 Step 3 — Mask-Based Enhancement Loss.

FORMULATION
-----------
Loss is computed between:
    S_hat = M * X_noisy   (enhanced output from mask application)
    S_clean               (ground truth clean STFT)

using complex L1 + L2 loss, same weights as the baseline.

WHY THIS FORMULATION IS SPEECH-PRESERVING
------------------------------------------
The key difference from direct regression:
  1. S_hat = M * X  means enhanced energy is always proportional to input energy.
  2. The gradient of the mask loss w.r.t. M explicitly depends on X.
     When X is large (speech-dominant bins), gradients are larger.
     This naturally emphasizes learning to preserve speech-dominant bins.
  3. The trivial M=0 solution is still possible in theory, but:
     - At initialization (random M), output is random * X, producing non-zero energy.
     - The gradient pushes M toward S/X (complex division), the ideal ratio mask.
     - Unlike direct regression, the gradient does not simply push amplitudes to 0.

ADDITIONAL: SPEECH ENERGY PRESERVATION TERM
--------------------------------------------
We add an auxiliary term that penalizes low-energy enhanced output
relative to the clean target energy. This term explicitly prevents collapse:

    L_energy = max(0, E_clean - E_enhanced) / (E_clean + epsilon)

This is a hinge loss on energy: if enhanced energy >= clean energy, penalty = 0.
If enhanced energy < clean energy, penalty grows proportionally.

This term fires only when the model is collapsing (E_enhanced << E_clean).
It does NOT fire when output is correctly scaled or when noise is also present.

TOTAL LOSS
----------
L_mask_enh = l1_weight * L1(S_hat, S_clean)
           + l2_weight * L2(S_hat, S_clean)
           + energy_weight * L_energy(S_hat, S_clean)

Where energy_weight = 0.1 by default (light regularizer).

The MultiTaskMaskLoss combines this with the existing ClassificationLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import (
    ClassificationLoss,
    convert_complex_stft_to_real_imag,
)


class MaskEnhancementLoss(nn.Module):
    """
    Enhancement loss for mask-based (CRM) models.

    Computes loss between:
        enhanced_output (S_hat = M * X)  shape (B, 2, 257, T)
        target_stft (S_clean)            shape (B, 257, T) complex or (B, 2, 257, T) float

    Parameters
    ----------
    l1_weight : float
        Weight for complex L1 loss (default 0.7, matching baseline).
    l2_weight : float
        Weight for complex L2 (MSE) loss (default 0.3, matching baseline).
    energy_weight : float
        Weight for speech-energy preservation penalty (default 0.1).
    """

    def __init__(
        self,
        l1_weight: float = 0.7,
        l2_weight: float = 0.3,
        energy_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if l1_weight < 0 or l2_weight < 0 or energy_weight < 0:
            raise ValueError("All loss weights must be non-negative.")
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.energy_weight = float(energy_weight)

    def forward(
        self,
        enhanced_output: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        enhanced_output : torch.Tensor
            Shape (B, 2, n_freq_bins, T) — S_hat from mask application.
        target_stft : torch.Tensor
            Shape (B, n_freq_bins, T) complex64 OR (B, 2, n_freq_bins, T) float32.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if enhanced_output.ndim != 4 or enhanced_output.shape[1] != 2:
            raise ValueError(
                f"enhanced_output must be (B, 2, F, T), got {enhanced_output.shape}"
            )

        target_ri = convert_complex_stft_to_real_imag(target_stft)

        if enhanced_output.shape != target_ri.shape:
            raise ValueError(
                f"Shape mismatch: predicted {enhanced_output.shape} vs target {target_ri.shape}"
            )

        # Complex L1 loss: mean absolute error over real + imaginary
        l1_loss = torch.mean(torch.abs(enhanced_output - target_ri))

        # Complex L2 loss: mean squared error over real + imaginary
        l2_loss = torch.mean((enhanced_output - target_ri) ** 2)

        # Speech energy preservation penalty
        # Penalizes enhanced energy being below clean target energy
        # Computes power (squared magnitude) per sample and bin
        enh_energy = torch.mean(enhanced_output ** 2)
        clean_energy = torch.mean(target_ri ** 2)
        energy_deficit = F.relu(clean_energy - enh_energy)
        energy_loss = energy_deficit / (clean_energy.detach() + 1e-8)

        total = (
            self.l1_weight * l1_loss
            + self.l2_weight * l2_loss
            + self.energy_weight * energy_loss
        )
        return total


class MultiTaskMaskLoss(nn.Module):
    """
    Multi-task loss for the mask-based model.

    L_total = enhancement_weight * L_mask_enh + classification_weight * L_cls

    Parameters match baseline exactly except l1/l2/energy weights go to
    MaskEnhancementLoss instead of ComplexEnhancementLoss.
    """

    def __init__(
        self,
        enhancement_weight: float = 1.0,
        classification_weight: float = 0.10,
        enhancement_l1_weight: float = 0.7,
        enhancement_l2_weight: float = 0.3,
        energy_preservation_weight: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.enhancement_weight = float(enhancement_weight)
        self.classification_weight = float(classification_weight)

        self.enhancement_loss_fn = MaskEnhancementLoss(
            l1_weight=enhancement_l1_weight,
            l2_weight=enhancement_l2_weight,
            energy_weight=energy_preservation_weight,
        )
        self.classification_loss_fn = ClassificationLoss(
            class_weights=class_weights,
        )

    def forward(
        self,
        enhanced_output: torch.Tensor,
        classification_logits: torch.Tensor,
        target_stft: torch.Tensor,
        noise_class_label: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        enhanced_output : torch.Tensor
            Shape (B, 2, F, T) — masked enhanced STFT.
        classification_logits : torch.Tensor
            Shape (B, 3) — noise class logits.
        target_stft : torch.Tensor
            Shape (B, F, T) complex64 or (B, 2, F, T) float32 — clean target.
        noise_class_label : torch.Tensor
            Shape (B,) int64 — noise class labels.

        Returns
        -------
        dict with keys: total_loss, enhancement_loss, classification_loss
        """
        l_enh = self.enhancement_loss_fn(enhanced_output, target_stft)
        l_cls = self.classification_loss_fn(classification_logits, noise_class_label)

        l_total = (
            self.enhancement_weight * l_enh
            + self.classification_weight * l_cls
        )

        return {
            "total_loss": l_total,
            "enhancement_loss": l_enh,
            "classification_loss": l_cls,
        }
