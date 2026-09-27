"""
src/models/mask_enhancement_head.py
=====================================
Phase 2 Step 3 — Mask-Based Enhancement Head.

DESIGN RATIONALE
----------------
The Phase 2 baseline (direct complex STFT regression) suffered from signal
extinction: the optimizer found a degenerate minimum of the L1/L2 spectral
loss by driving enhanced output amplitudes toward zero.

Root cause: L1 and L2 norms on complex STFT are minimized by a zero output
when the target clean STFT magnitudes are small relative to the noisy signal.
The trivial zero-output solution achieves finite (non-zero) loss only because
the clean target is non-zero, but the gradient still allows amplitude collapse.

FIX: Complex Ratio Mask (CRM) Prediction
-----------------------------------------
Instead of predicting the clean STFT directly, the model predicts a complex
ratio mask M that is applied to the noisy input STFT X to reconstruct the
enhanced STFT S_hat:

    S_hat = M * X

where:
    M      = M_r + j * M_i       (complex mask, real + imaginary parts)
    X      = X_r + j * X_i       (noisy input STFT)
    S_hat  = (M_r * X_r - M_i * X_i) + j * (M_r * X_i + M_i * X_r)

WHY ZERO-COLLAPSE IS PREVENTED
-------------------------------
Because S_hat = M * X, the model can only produce zero output if:
  (a) M = 0 (trivial zero mask), OR
  (b) X = 0 (zero input signal)

For any non-zero X (which holds for real noisy audio), predicting M = 0 is
the trivial solution. However, unlike direct regression where the target is
the clean STFT and zero output has a fixed L1/L2 cost, here:

    loss(M=0) = loss(S_hat=0 vs S=clean)
              = same as baseline zero-output loss

But the mask parametrization gives the gradient a clearer path toward the
optimal non-zero mask, because:
  1. The target mask M_target = S / X (complex division) is of order 1.
  2. The model learns to scale inputs proportionally, not to zero them out.
  3. We apply tanh activation to bound the mask components to (-1, 1).
     This prevents unbounded amplification while still allowing the model
     to implement near-unity gain (|M| ~ 1) where speech is dominant.

MASK BOUNDING
-------------
We apply tanh to the raw linear output:
    M_r = tanh(logit_r)
    M_i = tanh(logit_i)

This bounds each component in (-1, 1), preventing amplitude explosion.
The resulting complex mask has |M| ≤ sqrt(2) ≈ 1.414.

For scenarios where clean target magnitude > noisy magnitude (IRM > 1),
the mask cannot perfectly recover the signal — but it can still preserve
speech, and the loss gradient will push the mask toward the best feasible
bounded solution, which is speech-preserving rather than zero.

PARAMETER COUNT
---------------
The head has identical architecture to EnhancementHead:
    proj: Linear(hidden_dim=48, out=2*257=514)
    params: 48*514 + 514 = 25,186

Total model parameters: 70,789 (UNCHANGED from baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MaskEnhancementHead(nn.Module):
    """
    Complex Ratio Mask (CRM) Enhancement Head.

    Predicts a bounded complex mask M from GRU features, then applies:
        S_hat = M * X_noisy

    The mask components are bounded via tanh to prevent unbounded amplification.

    Parameters
    ----------
    hidden_dim : int
        GRU hidden dimension (default 48).
    n_freq_bins : int
        STFT frequency bins (default 257).
    out_channels : int
        Must be 2 (real + imaginary). Fixed.

    Input
    -----
    gru_features : (B, T, hidden_dim)
    noisy_features : (B, 2, n_freq_bins, T)  — the raw noisy STFT [real, imag]

    Output
    ------
    enhanced_stft : (B, 2, n_freq_bins, T)  — S_hat = M * X, bounded
    mask : (B, 2, n_freq_bins, T)           — the complex mask (for monitoring)
    """

    def __init__(
        self,
        hidden_dim: int = 48,
        n_freq_bins: int = 257,
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        assert out_channels == 2, "MaskEnhancementHead requires out_channels=2"
        self.hidden_dim = hidden_dim
        self.n_freq_bins = n_freq_bins
        self.out_channels = out_channels

        # Same projection as EnhancementHead — identical parameter count
        # Output: raw mask logits before activation
        self.proj = nn.Linear(hidden_dim, out_channels * n_freq_bins)

    def forward(
        self,
        gru_features: torch.Tensor,
        noisy_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        gru_features : torch.Tensor
            Shape (B, T, hidden_dim).
        noisy_features : torch.Tensor
            Shape (B, 2, n_freq_bins, T) — channel 0 = real, channel 1 = imag.

        Returns
        -------
        enhanced_stft : torch.Tensor
            Shape (B, 2, n_freq_bins, T) — masked enhanced STFT.
        mask : torch.Tensor
            Shape (B, 2, n_freq_bins, T) — the tanh-bounded complex mask.
        """
        batch_size, time_frames, _ = gru_features.shape

        # Project: (B, T, hidden_dim) -> (B, T, 2*n_freq_bins)
        raw = self.proj(gru_features)

        # Reshape: (B, T, 2*n_freq_bins) -> (B, T, 2, n_freq_bins)
        raw = raw.view(batch_size, time_frames, self.out_channels, self.n_freq_bins)

        # Permute: -> (B, 2, n_freq_bins, T)
        raw = raw.permute(0, 2, 3, 1).contiguous()

        # Apply tanh to bound mask components in (-1, 1)
        # This prevents unbounded amplification
        mask = torch.tanh(raw)  # (B, 2, n_freq_bins, T)

        # Apply complex mask: S_hat = M * X
        # M = mask[:, 0] + j * mask[:, 1]
        # X = noisy[:, 0] + j * noisy[:, 1]
        # S_hat_real = M_r * X_r - M_i * X_i
        # S_hat_imag = M_r * X_i + M_i * X_r
        M_r = mask[:, 0]          # (B, n_freq_bins, T)
        M_i = mask[:, 1]          # (B, n_freq_bins, T)
        X_r = noisy_features[:, 0]  # (B, n_freq_bins, T)
        X_i = noisy_features[:, 1]  # (B, n_freq_bins, T)

        S_hat_r = M_r * X_r - M_i * X_i  # (B, n_freq_bins, T)
        S_hat_i = M_r * X_i + M_i * X_r  # (B, n_freq_bins, T)

        # Stack back to (B, 2, n_freq_bins, T)
        enhanced_stft = torch.stack([S_hat_r, S_hat_i], dim=1)

        return enhanced_stft, mask

    def forward_mask_only(self, gru_features: torch.Tensor) -> torch.Tensor:
        """
        Return the tanh-bounded mask without applying it.
        Useful for mask analysis.

        Parameters
        ----------
        gru_features : torch.Tensor
            Shape (B, T, hidden_dim).

        Returns
        -------
        mask : torch.Tensor
            Shape (B, 2, n_freq_bins, T).
        """
        batch_size, time_frames, _ = gru_features.shape
        raw = self.proj(gru_features)
        raw = raw.view(batch_size, time_frames, self.out_channels, self.n_freq_bins)
        raw = raw.permute(0, 2, 3, 1).contiguous()
        return torch.tanh(raw)
