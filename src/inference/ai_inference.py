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
    ) -> None:
        self.model = model
        self.device = device
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None

        # Ensure eval mode
        self.model.eval()
        self.model.to(self.device)

        # Verify parameter count
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable != EXPECTED_PARAM_COUNT:
            logger.warning(
                "Parameter count mismatch: expected %d, got %d. "
                "Architecture may have changed.",
                EXPECTED_PARAM_COUNT, trainable,
            )
        self.param_count = trainable

        # Diagnostics
        self._n_inferences: int = 0
        self._total_inference_time_s: float = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> "AIInferenceWrapper":
        """
        Load a LightweightCNNGRUMaskModel from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the PyTorch checkpoint (.pt).
        device : str
            Target device: 'cpu', 'cuda', or 'auto'.

        Returns
        -------
        AIInferenceWrapper

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        RuntimeError
            If checkpoint is missing required keys.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device == "auto":
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(
            "Loaded checkpoint: %s (epoch=%s)",
            checkpoint_path,
            checkpoint.get("epoch", "unknown"),
        )

        return cls(model=model, device=resolved_device, checkpoint_path=checkpoint_path)

    def infer(
        self,
        noisy_features: np.ndarray | torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run AI inference on STFT features.

        Parameters
        ----------
        noisy_features : np.ndarray or torch.Tensor
            Shape (2, 257, T) or (1, 2, 257, T) or (B, 2, 257, T).
            Float32. Channel 0 = Real STFT, Channel 1 = Imaginary STFT.

        Returns
        -------
        enhanced_stft : np.ndarray, shape (2, 257, T) — enhanced real/imag STFT
        classification_logits : np.ndarray, shape (3,) — noise class logits
        mask : np.ndarray, shape (2, 257, T) — predicted complex ratio mask

        All outputs are float32 numpy arrays on CPU.

        Raises
        ------
        ValueError
            If input shape is invalid.
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

        # Inference
        with torch.no_grad():
            enhanced_tensor, logits_tensor, mask_tensor = self.model(x)

        # Convert to numpy (CPU, float32)
        enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        logits_np = logits_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        t_end = time.perf_counter()
        self._n_inferences += 1
        self._total_inference_time_s += (t_end - t_start)

        return enhanced_np, logits_np, mask_np

    def classify(
        self,
        logits: np.ndarray,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Convert raw logits to noise class prediction and confidence.

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
        # Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        noise_class = int(np.argmax(probs))
        confidence = float(probs[noise_class])
        return noise_class, confidence, probs

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
        }
