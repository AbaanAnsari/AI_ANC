from __future__ import annotations

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count total or trainable parameters in a PyTorch model.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_memory_bytes(model: nn.Module) -> int:
    """
    Estimate model parameter memory in bytes (assuming FP32 weights).
    """
    total_params = count_parameters(model, trainable_only=False)
    return total_params * 4
