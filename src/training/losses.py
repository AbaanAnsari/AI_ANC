from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_complex_stft_to_real_imag(target_stft: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex STFT tensor (B, 257, T) complex64 into a 2-channel
    float32 tensor (B, 2, 257, T) where:
        channel 0 = Real component
        channel 1 = Imaginary component

    If the input is already (B, 2, 257, T) float32, it is returned unchanged.
    """
    if target_stft.ndim == 4 and target_stft.shape[1] == 2:
        return target_stft.float()

    if target_stft.ndim != 3:
        raise ValueError(
            f"Expected complex STFT tensor of shape (B, 257, T), got shape {target_stft.shape}"
        )

    real = target_stft.real.float()
    imag = target_stft.imag.float()
    return torch.stack([real, imag], dim=1)


class ComplexEnhancementLoss(nn.Module):
    """
    Supervised Speech Enhancement Loss over Complex STFT (Real & Imaginary components).

    Formulation:
        L_enhancement = alpha * L1_complex + beta * L2_complex

    Default weights:
        alpha (l1_weight) = 0.7
        beta  (l2_weight) = 0.3
    """

    def __init__(self, l1_weight: float = 0.7, l2_weight: float = 0.3) -> None:
        super().__init__()
        if l1_weight < 0 or l2_weight < 0:
            raise ValueError("l1_weight and l2_weight must be non-negative")

        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)

    def forward(
        self,
        predicted_stft: torch.Tensor,
        target_stft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted_stft : torch.Tensor
            Predicted complex STFT features of shape (B, 2, 257, T) float32.
        target_stft : torch.Tensor
            Target complex STFT tensor of shape (B, 257, T) complex64
            or (B, 2, 257, T) float32.

        Returns
        -------
        torch.Tensor
            Scalar enhancement loss tensor.
        """
        if predicted_stft.ndim != 4 or predicted_stft.shape[1] != 2:
            raise ValueError(
                f"predicted_stft must be (B, 2, 257, T), got shape {predicted_stft.shape}"
            )

        target_real_imag = convert_complex_stft_to_real_imag(target_stft)

        if predicted_stft.shape != target_real_imag.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_stft.shape} vs target {target_real_imag.shape}"
            )

        # L1 Loss over real and imaginary components
        l1_loss = torch.mean(torch.abs(predicted_stft - target_real_imag))

        # L2 (MSE) Loss over real and imaginary components
        l2_loss = torch.mean((predicted_stft - target_real_imag) ** 2)

        total_enhancement_loss = (
            self.l1_weight * l1_loss + self.l2_weight * l2_loss
        )
        return total_enhancement_loss


class ClassificationLoss(nn.Module):
    """
    Supervised 3-Class Noise Classification Loss (Cross Entropy).

    Classes:
        0 = stationary
        1 = non-stationary
        2 = impulsive
    """

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        classification_logits: torch.Tensor,
        noise_class_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        classification_logits : torch.Tensor
            Predicted logits of shape (B, 3).
        noise_class_label : torch.Tensor
            Target class labels of shape (B,) int64.

        Returns
        -------
        torch.Tensor
            Scalar classification loss tensor.
        """
        if classification_logits.ndim != 2 or classification_logits.shape[1] != 3:
            raise ValueError(
                f"classification_logits must be (B, 3), got shape {classification_logits.shape}"
            )

        loss = self.cross_entropy(classification_logits, noise_class_label.long())
        return loss


class MultiTaskLoss(nn.Module):
    """
    Supervised Multi-Task Training Objective:

    L_total = lambda_enhancement * L_enhancement + lambda_classification * L_classification

    Default Weights:
        lambda_enhancement (enhancement_weight) = 1.0   (PRIMARY OBJECTIVE)
        lambda_classification (classification_weight) = 0.10 (SUPPORTING OBJECTIVE)
        enhancement_l1_weight = 0.7
        enhancement_l2_weight = 0.3
    """

    def __init__(
        self,
        enhancement_weight: float = 1.0,
        classification_weight: float = 0.10,
        enhancement_l1_weight: float = 0.7,
        enhancement_l2_weight: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.enhancement_weight = float(enhancement_weight)
        self.classification_weight = float(classification_weight)

        self.enhancement_loss_fn = ComplexEnhancementLoss(
            l1_weight=enhancement_l1_weight,
            l2_weight=enhancement_l2_weight,
        )
        self.classification_loss_fn = ClassificationLoss(
            class_weights=class_weights
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
            Predicted STFT features of shape (B, 2, 257, T).
        classification_logits : torch.Tensor
            Predicted logits of shape (B, 3).
        target_stft : torch.Tensor
            Target complex STFT tensor of shape (B, 257, T) complex64
            or (B, 2, 257, T) float32.
        noise_class_label : torch.Tensor
            Target class labels of shape (B,) int64.

        Returns
        -------
        dict[str, torch.Tensor]
            {
                "total_loss": scalar float32 tensor,
                "enhancement_loss": scalar float32 tensor,
                "classification_loss": scalar float32 tensor,
            }
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
