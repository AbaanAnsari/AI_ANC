"""
src/inference/ai_inference.py
==============================
AI Model Inference Wrapper for LightweightCNNGRUMaskModel.

Provides a clean, stateless interface for:
    - Loading checkpoint from disk
    - Running inference on STFT features
    - Returning enhanced STFT + classification logits + mask

Designed for both batch evaluation and streaming (frame-by-frame) use.

Usage:
    wrapper = AIInferenceWrapper.from_checkpoint(checkpoint_path)
    enhanced_stft, logits, mask = wrapper.infer(noisy_features)
    noise_class, confidence = wrapper.classify(logits)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_PARAM_COUNT = 70_789
NOISE_CLASS_NAMES = {0: "stationary", 1: "non-stationary", 2: "impulsive"}


# ---------------------------------------------------------------------------
# Inference Wrapper
# ---------------------------------------------------------------------------


class AIInferenceWrapper:
    """
    Inference wrapper for LightweightCNNGRUMaskModel.

    Handles:
    - Checkpoint loading with parameter count verification
    - Device management (CPU/CUDA)
    - eval() mode enforcement
    - no_grad() context
    - Input validation and shape handling
    - Latency measurement

    Parameters
    ----------
    model : LightweightCNNGRUMaskModel
        Loaded model in eval mode.
    device : torch.device
        Device to run inference on.
    checkpoint_path : str or None
        Source checkpoint path (for logging).
    """

    def __init__(
        self,
        model: LightweightCNNGRUMaskModel,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        classifier_smoothing_alpha: float = 0.15,
        hysteresis_frames: int = 5,
        confidence_threshold: float = 0.50,
        classifier_window_frames: int = 15,
    ) -> None:
        self.model = model
        self.device = device
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.classifier_smoothing_alpha = float(classifier_smoothing_alpha)
        self.hysteresis_frames = int(hysteresis_frames)
        self.confidence_threshold = float(confidence_threshold)
        self.classifier_window_frames = int(classifier_window_frames)

        # Ensure eval mode
        self.model.eval()
        self.model.to(self.device)

        # Verify parameter count
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable != EXPECTED_PARAM_COUNT:
            raise RuntimeError(
                "Production model parameter count mismatch: expected %d, got %d. "
                "Refusing to initialize inference with an unvalidated architecture.",
                EXPECTED_PARAM_COUNT,
                trainable,
            )
        self.param_count = trainable

        # Diagnostics & temporal state
        self._n_inferences: int = 0
        self._total_inference_time_s: float = 0.0
        self._hidden_state: Optional[torch.Tensor] = None
        self._gru_history: list = []
        self._smoothed_probs: Optional[np.ndarray] = None
        self._active_class: int = 0
        self._switch_counter: int = 0
        self._class_switch_count: int = 0

    def reset_state(self) -> None:
        """Reset the recurrent GRU hidden state and classifier smoothing history."""
        self._hidden_state = None
        self._gru_history.clear()
        self._smoothed_probs = None
        self._active_class = 0
        self._switch_counter = 0
        self._class_switch_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
        classifier_smoothing_alpha: float = 0.15,
        hysteresis_frames: int = 5,
        confidence_threshold: float = 0.50,
        classifier_window_frames: int = 15,
        classifier_hysteresis_threshold: Optional[float] = None,
        classifier_min_dwell_frames: Optional[int] = None,
        **kwargs,
    ) -> "AIInferenceWrapper":
        """
        Load a LightweightCNNGRUMaskModel from a checkpoint file.
        """
        if classifier_hysteresis_threshold is not None:
            confidence_threshold = classifier_hysteresis_threshold
        if classifier_min_dwell_frames is not None:
            hysteresis_frames = classifier_min_dwell_frames

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device == "auto":
            resolved_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved_device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing 'model_state_dict'. "
                f"Keys found: {list(checkpoint.keys())}"
            )

        model = LightweightCNNGRUMaskModel()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        logger.info(
            "Loaded checkpoint: %s (epoch=%s)",
            checkpoint_path,
            checkpoint.get("epoch", "unknown"),
        )

        return cls(
            model=model,
            device=resolved_device,
            checkpoint_path=checkpoint_path,
            classifier_smoothing_alpha=classifier_smoothing_alpha,
            hysteresis_frames=hysteresis_frames,
            confidence_threshold=confidence_threshold,
            classifier_window_frames=classifier_window_frames,
        )

    def infer(
        self,
        noisy_features: np.ndarray | torch.Tensor,
        stateful: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run AI inference on STFT features.

        Parameters
        ----------
        noisy_features : np.ndarray or torch.Tensor
            Shape (2, 257, T) or (1, 2, 257, T) or (B, 2, 257, T).
            Float32. Channel 0 = Real STFT, Channel 1 = Imaginary STFT.
            When T == 7, processes the rolling [t-3 ... t+3] temporal context window,
            producing the estimate corresponding to the center frame t (index 3).
        stateful : bool, optional
            If True, preserve recurrent GRU hidden state across consecutive calls.
            Default is True for streaming temporal continuity.

        Returns
        -------
        enhanced_stft : np.ndarray, shape (2, 257, T_out) — enhanced real/imag STFT
        classification_logits : np.ndarray, shape (3,) — noise class logits
        mask : np.ndarray, shape (2, 257, T_out) — predicted complex ratio mask
        """
        t_start = time.perf_counter()

        # Convert to tensor
        if isinstance(noisy_features, np.ndarray):
            x = torch.from_numpy(np.asarray(noisy_features, dtype=np.float32))
        else:
            x = noisy_features.float()

        # Handle shape: add batch dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, 2, 257, T)
        elif x.ndim == 4:
            pass  # (B, 2, 257, T)
        else:
            raise ValueError(
                f"noisy_features must be (2,257,T) or (B,2,257,T), got shape {x.shape}"
            )

        if x.shape[1] != 2:
            raise ValueError(
                f"Channel dimension must be 2 (real+imag), got {x.shape[1]}"
            )

        x = x.to(self.device)
        B, C, F, T = x.shape

        with torch.inference_mode():
            if T == 7:
                # 7-frame temporal context buffer [t-3, t-2, t-1, t, t+1, t+2, t+3]
                # Convolutional feature extraction across all 7 frames
                c1 = self.model.conv1(x)  # (B, 16, 129, 7)
                c2 = self.model.conv2(c1)  # (B, 32, 65, 7)
                c3 = self.model.conv3(c2)  # (B, 16, 33, 7)

                # Extract unpadded center frame at index 3 (frame t)
                c3_center = c3[:, :, :, 3:4]  # (B, 16, 33, 1)
                c3_flat = (
                    c3_center.permute(0, 3, 1, 2).contiguous().view(B, 1, -1)
                )  # (B, 1, 528)
                proj = self.model.proj(c3_flat)  # (B, 1, 32)

                # Recurrent GRU step
                if stateful:
                    h_in = (
                        self._hidden_state.to(self.device)
                        if self._hidden_state is not None
                        else None
                    )
                    gru_out, next_h = self.model.gru(proj, h_in)
                    self._hidden_state = next_h.detach()
                else:
                    gru_out, _ = self.model.gru(proj, None)

                # Enhancement head applied to center frame noisy STFT
                noisy_center = x[:, :, :, 3:4]  # (B, 2, 257, 1)
                enhanced_tensor, mask_tensor = self.model.enhancement_head(
                    gru_out, noisy_center
                )

                # Classification temporal context: maintain rolling buffer of GRU representations
                self._gru_history.append(gru_out.detach().squeeze(0))  # (1, 48)
                if len(self._gru_history) > self.classifier_window_frames:
                    self._gru_history.pop(0)

                # Temporal mean pooling over history window (matches training mean-pool)
                gru_seq = torch.cat(self._gru_history, dim=0).unsqueeze(0)  # (1, W, 48)
                pooled = torch.mean(gru_seq, dim=1)  # (1, 48)
                logits_tensor = self.model.classification_head.classifier(
                    pooled
                )  # (1, 3)

            elif T == 1:
                # Single-frame legacy path
                if stateful:
                    h_in = (
                        self._hidden_state.to(self.device)
                        if self._hidden_state is not None
                        else None
                    )
                    enhanced_tensor, logits_tensor, mask_tensor, next_h = self.model(
                        x, hidden_state=h_in, return_hidden=True
                    )
                    self._hidden_state = next_h.detach()
                else:
                    enhanced_tensor, logits_tensor, mask_tensor = self.model(
                        x, hidden_state=None, return_hidden=False
                    )
            else:
                # Full multi-frame batch sequence (T > 7, e.g. 1-second 126 frames)
                enhanced_tensor, logits_tensor, mask_tensor = self.model(
                    x, hidden_state=None, return_hidden=False
                )

        # Convert to numpy (CPU, float32)
        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        logits_np = logits_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        t_end = time.perf_counter()
        self._n_inferences += 1
        self._total_inference_time_s += t_end - t_start

        return enhanced_np, logits_np, mask_np

    def classify(
        self,
        logits: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Convert raw logits to noise class prediction and confidence (unsmoothed).

        Parameters
        ----------
        logits : np.ndarray, shape (3,) or (B, 3)
            Raw classifier logits.

        Returns
        -------
        noise_class : int — 0, 1, or 2
        confidence : float — max softmax probability in [0, 1]
        probabilities : np.ndarray, shape (3,) — softmax probabilities
        """
        logits = np.asarray(logits, dtype=np.float32).ravel()
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        noise_class = int(np.argmax(probs))
        confidence = float(probs[noise_class])
        return noise_class, confidence, probs

    def classify_smoothed(
        self,
        logits: np.ndarray,
    ) -> Tuple[int, float, np.ndarray, np.ndarray, int]:
        """
        Convert raw logits to smoothed noise class prediction and confidence using hysteresis.

        Maintains class probabilities rather than only the argmax class.
        Applies configurable exponential temporal smoothing.
        Requires a configurable confidence and dwell duration before changing active class.

        Parameters
        ----------
        logits : np.ndarray, shape (3,) or (B, 3)
            Raw classifier logits.

        Returns
        -------
        active_class : int — active stable noise class (0, 1, or 2)
        confidence : float — confidence of active class in [0, 1]
        raw_probs : np.ndarray, shape (3,) — unsmoothed softmax probabilities
        smoothed_probs : np.ndarray, shape (3,) — exponentially smoothed probabilities
        class_switches : int — total number of class switches so far
        """
        raw_class, raw_conf, raw_probs = self.classify(logits)

        # 1. Exponential temporal smoothing of probabilities
        if self._smoothed_probs is None:
            self._smoothed_probs = raw_probs.copy()
            self._active_class = raw_class
        else:
            alpha = self.classifier_smoothing_alpha
            self._smoothed_probs = (
                alpha * raw_probs + (1.0 - alpha) * self._smoothed_probs
            )

        # Renormalize smoothed probabilities
        s_sum = float(np.sum(self._smoothed_probs))
        if s_sum > 0:
            self._smoothed_probs /= s_sum

        # 2. Hysteresis logic
        candidate_class = int(np.argmax(self._smoothed_probs))
        candidate_conf = float(self._smoothed_probs[candidate_class])

        if candidate_class != self._active_class:
            if candidate_conf >= self.confidence_threshold:
                self._switch_counter += 1
                if self._switch_counter >= self.hysteresis_frames:
                    logger.info(
                        "Classifier switched active class: %d -> %d (conf=%.3f, after %d frames)",
                        self._active_class,
                        candidate_class,
                        candidate_conf,
                        self._switch_counter,
                    )
                    self._active_class = candidate_class
                    self._class_switch_count += 1
                    self._switch_counter = 0
            else:
                self._switch_counter = 0
        else:
            self._switch_counter = 0

        active_conf = float(self._smoothed_probs[self._active_class])
        return (
            self._active_class,
            active_conf,
            raw_probs,
            self._smoothed_probs.copy(),
            self._class_switch_count,
        )

    @property
    def switch_count(self) -> int:
        """Total number of noise class switches committed."""
        return self._class_switch_count

    @property
    def diagnostics(self) -> dict:
        avg_latency_ms = (
            1000.0 * self._total_inference_time_s / self._n_inferences
            if self._n_inferences > 0
            else 0.0
        )
        return {
            "param_count": self.param_count,
            "device": str(self.device),
            "n_inferences": self._n_inferences,
            "total_inference_time_s": round(self._total_inference_time_s, 4),
            "avg_inference_latency_ms": round(avg_latency_ms, 2),
            "checkpoint_path": self.checkpoint_path,
            "active_class": self._active_class,
            "class_switch_count": self._class_switch_count,
            "classifier_smoothing_alpha": self.classifier_smoothing_alpha,
            "hysteresis_frames": self.hysteresis_frames,
            "confidence_threshold": self.confidence_threshold,
            "classifier_window_frames": self.classifier_window_frames,
        }
