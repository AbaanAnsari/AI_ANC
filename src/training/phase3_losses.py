"""
src/training/phase3_losses.py
================================
Phase 3 — Speech Enhancement Quality Optimization Losses.

New loss components designed to improve SNR, STOI, and PESQ:

1. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) Loss
   - Directly optimizes time-domain signal reconstruction quality.
   - Addresses the problem that spectral L1/L2 doesn't directly penalize
     perceptually important time-domain distortions.

2. Multi-Resolution STFT Loss
   - Uses multiple FFT sizes (512, 1024, 256) to capture both fine
     and coarse spectral structures.
   - Combines spectral convergence + log-magnitude loss.

3. Phase 3 Composite Enhancement Loss
   - Combines: mask supervision + reconstruction + SI-SDR + multi-res STFT
     + energy hinge into one tunable loss.

RULES:
  - No architecture changes to the 70,789-parameter model.
  - No modifications to evaluation code.
  - All loss components are differentiable.
  - Compatible with existing CRM mask training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.training.losses import (
    ClassificationLoss,
    convert_complex_stft_to_real_imag,
)
from src.training.targeted_crm_loss import compute_target_crm


# ---------------------------------------------------------------------------
# 1. SI-SDR Loss (Time-Domain)
# ---------------------------------------------------------------------------

class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio Loss.

    SI-SDR = 10 * log10( ||s_target||^2 / ||e_noise||^2 )

    where:
      s_target = (<s_hat, s> / ||s||^2) * s
      e_noise  = s_hat - s_target

    The loss is: -SI-SDR  (we want to maximize SI-SDR, so minimize negative).

    For speech enhancement, s = clean waveform, s_hat = enhanced waveform.

    Reference: Le Roux et al., "SDR — Half-baked or Well Done?", ICASSP 2019.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        estimate : torch.Tensor
            Enhanced waveform, shape (B, N) or (B, 1, N).
        reference : torch.Tensor
            Clean reference waveform, shape (B, N) or (B, 1, N).

        Returns
        -------
        torch.Tensor
            Scalar negative SI-SDR loss (lower = better enhancement).
        """
        # Flatten to (B, N)
        est = estimate.reshape(estimate.shape[0], -1)
        ref = reference.reshape(reference.shape[0], -1)

        # Zero-mean normalization (per sample)
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)

        # s_target = (<s_hat, s> / ||s||^2) * s
        dot = torch.sum(est * ref, dim=-1, keepdim=True)
        s_ref_energy = torch.sum(ref * ref, dim=-1, keepdim=True) + self.eps
        # Speech enhancement requires non-negative projection (prevent 180 deg phase flip)
        alpha = torch.clamp(dot / s_ref_energy, min=0.0)
        s_target = alpha * ref

        # e_noise = s_hat - s_target
        e_noise = est - s_target

        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        si_sdr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=-1) /
            (torch.sum(e_noise ** 2, dim=-1) + self.eps)
            + self.eps
        )

        # Return negative mean SI-SDR (minimize = maximize SI-SDR)
        return -torch.mean(si_sdr)


# ---------------------------------------------------------------------------
# 2. Multi-Resolution STFT Loss
# ---------------------------------------------------------------------------

class SingleResolutionSTFTLoss(nn.Module):
    """
    Single-resolution STFT loss combining:
    - Spectral convergence: ||STFT(x) - STFT(y)|| / ||STFT(x)||
    - Log-magnitude loss: L1(log|STFT(x)|, log|STFT(y)|)
    """

    def __init__(
        self,
        fft_size: int = 512,
        hop_size: int = 128,
        win_length: int = 512,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.eps = eps

        # Register hann window as buffer
        self.register_buffer(
            "window",
            torch.hann_window(win_length),
            persistent=False,
        )

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        estimate : (B, N)
        reference : (B, N)

        Returns
        -------
        (spectral_convergence_loss, log_magnitude_loss)
        """
        # Compute STFT magnitudes
        est_stft = torch.stft(
            estimate, self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        ref_stft = torch.stft(
            reference, self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        est_mag = torch.abs(est_stft)
        ref_mag = torch.abs(ref_stft)

        # Spectral convergence
        sc_loss = torch.norm(ref_mag - est_mag, p="fro") / (
            torch.norm(ref_mag, p="fro") + self.eps
        )

        # Log magnitude
        log_mag_loss = F.l1_loss(
            torch.log(est_mag + self.eps),
            torch.log(ref_mag + self.eps),
        )

        return sc_loss, log_mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss across multiple FFT sizes.

    Uses 3 resolutions:
      - 256 / 64 / 256  (fine temporal, coarse spectral)
      - 512 / 128 / 512 (matched to model STFT config)
      - 1024 / 256 / 1024 (coarse temporal, fine spectral)
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (256, 512, 1024),
        hop_sizes: tuple[int, ...] = (64, 128, 256),
        win_lengths: tuple[int, ...] = (256, 512, 1024),
    ) -> None:
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.losses = nn.ModuleList([
            SingleResolutionSTFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths)
        ])

    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        estimate : (B, N)
        reference : (B, N)

        Returns
        -------
        Scalar combined multi-resolution STFT loss.
        """
        sc_total = 0.0
        mag_total = 0.0
        for loss_fn in self.losses:
            sc, mag = loss_fn(estimate, reference)
            sc_total += sc
            mag_total += mag

        n = len(self.losses)
        return (sc_total + mag_total) / n


# ---------------------------------------------------------------------------
# 3. STFT ↔ Waveform conversion (differentiable, PyTorch)
# ---------------------------------------------------------------------------

def stft_to_waveform_torch(
    stft_real_imag: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
    length: int | None = None,
) -> torch.Tensor:
    """
    Convert (B, 2, F, T) real/imag STFT to (B, N) waveform using torch.istft.

    This is differentiable — gradients flow through the ISTFT.
    """
    B, C, F, T_frames = stft_real_imag.shape
    assert C == 2, f"Expected 2-channel (real/imag), got {C}"

    # Build complex tensor
    real = stft_real_imag[:, 0]  # (B, F, T)
    imag = stft_real_imag[:, 1]  # (B, F, T)
    stft_complex = torch.complex(real, imag)  # (B, F, T)

    window = torch.hann_window(win_length, device=stft_real_imag.device)
    win_sum = float(window.sum().item())

    waveform = torch.istft(
        stft_complex * win_sum,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )

    return waveform  # (B, N)


# ---------------------------------------------------------------------------
# 4. Phase 3 Composite Loss
# ---------------------------------------------------------------------------

class Phase3CompositeLoss(nn.Module):
    """
    Phase 3 — Composite Enhancement Loss for Speech Quality Optimization.

    Combines:
      1. L_mask:    Direct CRM mask supervision (existing from Step 5)
      2. L_recon:   Spectral reconstruction L1+L2 (existing from Step 5)
      3. L_sisdr:   Time-domain SI-SDR loss (NEW)
      4. L_mrstft:  Multi-resolution STFT loss (NEW)
      5. L_energy:  Energy hinge collapse protection (existing, weight increased)

    L_enh = w_mask * L_mask
          + w_recon * L_recon
          + w_sisdr * L_sisdr
          + w_mrstft * L_mrstft
          + w_energy * L_energy

    Recommended weights for Phase 3:
      mask_weight    = 0.30   (reduced from 0.60 — still provides mask guidance)
      recon_weight   = 0.15   (reduced from 0.35 — spectral fidelity)
      sisdr_weight   = 0.30   (NEW — primary quality driver)
      mrstft_weight  = 0.15   (NEW — multi-scale spectral quality)
      energy_weight  = 0.10   (increased from 0.05 — stronger collapse prevention)

    Parameters: 70,789 (UNCHANGED)
    """

    def __init__(
        self,
        mask_weight: float = 0.30,
        recon_weight: float = 0.15,
        sisdr_weight: float = 0.30,
        mrstft_weight: float = 0.15,
        energy_weight: float = 0.10,
        l1_weight: float = 0.7,
        l2_weight: float = 0.3,
        eps: float = 1e-7,
        # CRM target mask clamp range
        mask_clamp_min: float = -1.0,
        mask_clamp_max: float = 1.0,
        # Waveform reconstruction parameters
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        segment_samples: int = 16000,
    ) -> None:
        super().__init__()

        self.mask_weight = float(mask_weight)
        self.recon_weight = float(recon_weight)
        self.sisdr_weight = float(sisdr_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.energy_weight = float(energy_weight)
        self.l1_weight = float(l1_weight)
        self.l2_weight = float(l2_weight)
        self.eps = float(eps)
        self.mask_clamp_min = float(mask_clamp_min)
        self.mask_clamp_max = float(mask_clamp_max)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.segment_samples = segment_samples

        # Sub-losses
        self.sisdr_loss = SISDRLoss(eps=eps)
        self.mrstft_loss = MultiResolutionSTFTLoss()

    def forward(
        self,
        enhanced_output: torch.Tensor,
        predicted_mask: torch.Tensor,
        noisy_features: torch.Tensor,
        target_stft: torch.Tensor,
        clean_waveform: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        enhanced_output : (B, 2, F, T)  — S_hat = M * X
        predicted_mask : (B, 2, F, T)   — tanh-bounded predicted mask
        noisy_features : (B, 2, F, T)   — noisy STFT X
        target_stft : (B, F, T) complex or (B, 2, F, T) float — clean STFT S
        clean_waveform : (B, N) float    — clean time-domain reference (optional)

        Returns
        -------
        dict with all loss components and total.
        """
        target_ri = convert_complex_stft_to_real_imag(target_stft)

        # ---- A: Mask Loss (from Step 5) ----
        target_mask = compute_target_crm(
            noisy_features, target_stft, eps=self.eps,
            clamp_min=self.mask_clamp_min, clamp_max=self.mask_clamp_max,
        )
        mask_l1 = torch.mean(torch.abs(predicted_mask - target_mask))
        mask_l2 = torch.mean((predicted_mask - target_mask) ** 2)
        l_mask = self.l1_weight * mask_l1 + self.l2_weight * mask_l2

        # ---- B: Reconstruction Loss (from Step 5) ----
        recon_l1 = torch.mean(torch.abs(enhanced_output - target_ri))
        recon_l2 = torch.mean((enhanced_output - target_ri) ** 2)
        l_recon = self.l1_weight * recon_l1 + self.l2_weight * recon_l2

        # ---- C: Energy Hinge (collapse protection, weight increased) ----
        enh_energy = torch.mean(enhanced_output ** 2)
        clean_energy = torch.mean(target_ri ** 2)
        energy_deficit = F.relu(clean_energy - enh_energy)
        l_energy = energy_deficit / (clean_energy.detach() + 1e-8)

        # ---- D: SI-SDR Loss (NEW — time-domain) ----
        # Reconstruct waveforms via differentiable ISTFT
        enhanced_wav = stft_to_waveform_torch(
            enhanced_output,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=self.segment_samples,
        )

        if clean_waveform is not None:
            clean_wav = clean_waveform
        else:
            # Reconstruct clean from target STFT
            clean_wav = stft_to_waveform_torch(
                target_ri,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                length=self.segment_samples,
            )

        # Align lengths
        min_len = min(enhanced_wav.shape[-1], clean_wav.shape[-1])
        enhanced_wav_aligned = enhanced_wav[..., :min_len]
        clean_wav_aligned = clean_wav[..., :min_len]

        l_sisdr = self.sisdr_loss(enhanced_wav_aligned, clean_wav_aligned)

        # ---- E: Multi-Resolution STFT Loss (NEW) ----
        l_mrstft = self.mrstft_loss(enhanced_wav_aligned, clean_wav_aligned)

        # ---- Total Enhancement Loss ----
        l_enh = (
            self.mask_weight * l_mask
            + self.recon_weight * l_recon
            + self.sisdr_weight * l_sisdr
            + self.mrstft_weight * l_mrstft
            + self.energy_weight * l_energy
        )

        return {
            "total_enhancement_loss": l_enh,
            "mask_loss": l_mask,
            "reconstruction_loss": l_recon,
            "sisdr_loss": l_sisdr,
            "mrstft_loss": l_mrstft,
            "energy_loss": l_energy,
        }


class Phase3MultiTaskLoss(nn.Module):
    """
    Phase 3 — Multi-Task Loss combining composite enhancement + classification.

    L_total = enh_weight * L_enh + cls_weight * L_cls
    """

    def __init__(
        self,
        enhancement_weight: float = 1.0,
        classification_weight: float = 0.10,
        mask_weight: float = 0.30,
        recon_weight: float = 0.15,
        sisdr_weight: float = 0.30,
        mrstft_weight: float = 0.15,
        energy_weight: float = 0.10,
        l1_weight: float = 0.7,
        l2_weight: float = 0.3,
        eps: float = 1e-7,
        mask_clamp_min: float = -1.0,
        mask_clamp_max: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.enhancement_weight = float(enhancement_weight)
        self.classification_weight = float(classification_weight)

        self.enhancement_loss_fn = Phase3CompositeLoss(
            mask_weight=mask_weight,
            recon_weight=recon_weight,
            sisdr_weight=sisdr_weight,
            mrstft_weight=mrstft_weight,
            energy_weight=energy_weight,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            eps=eps,
            mask_clamp_min=mask_clamp_min,
            mask_clamp_max=mask_clamp_max,
        )
        self.classification_loss_fn = ClassificationLoss(class_weights=class_weights)

    def forward(
        self,
        enhanced_output: torch.Tensor,
        predicted_mask: torch.Tensor,
        classification_logits: torch.Tensor,
        noisy_features: torch.Tensor,
        target_stft: torch.Tensor,
        noise_class_label: torch.Tensor,
        clean_waveform: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        enhanced_output : (B, 2, F, T)
        predicted_mask : (B, 2, F, T)
        classification_logits : (B, 3)
        noisy_features : (B, 2, F, T)
        target_stft : (B, F, T) complex or (B, 2, F, T)
        noise_class_label : (B,) int64
        clean_waveform : (B, N) float32, optional

        Returns
        -------
        dict with all loss components.
        """
        enh_dict = self.enhancement_loss_fn(
            enhanced_output, predicted_mask, noisy_features, target_stft,
            clean_waveform=clean_waveform,
        )
        l_enh = enh_dict["total_enhancement_loss"]
        l_cls = self.classification_loss_fn(classification_logits, noise_class_label)

        l_total = (
            self.enhancement_weight * l_enh
            + self.classification_weight * l_cls
        )

        return {
            "total_loss": l_total,
            "enhancement_loss": l_enh,
            "mask_loss": enh_dict["mask_loss"],
            "reconstruction_loss": enh_dict["reconstruction_loss"],
            "sisdr_loss": enh_dict["sisdr_loss"],
            "mrstft_loss": enh_dict["mrstft_loss"],
            "energy_loss": enh_dict["energy_loss"],
            "classification_loss": l_cls,
        }
