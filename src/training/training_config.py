"""
Phase 1 Step 13 — Training Configuration
=========================================
Single-source defaults. All defaults here must match configs/training.yaml exactly.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Nested configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    type: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 3
    min_lr: float = 1e-6


@dataclass
class LossConfig:
    enhancement_weight: float = 1.0
    classification_weight: float = 0.10
    enhancement_l1_weight: float = 0.7
    enhancement_l2_weight: float = 0.3


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "models/checkpoints"
    save_best_only: bool = True
    monitor: str = "val_total_loss"
    mode: str = "min"
    save_last: bool = True


# ---------------------------------------------------------------------------
# Main Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Authoritative training configuration for Phase 1 Step 13.

    All defaults must match configs/training.yaml exactly.
    """
    # Core training
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 123
    num_workers: int = 0
    validation_frequency: int = 1
    log_frequency: int = 10
    mixed_precision: bool = False

    # Component configurations
    optimizer: str = "AdamW"
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Device (resolved by create(), not stored as "auto")
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate all configuration fields."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be > 0, got {self.gradient_clip_norm}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.validation_frequency <= 0:
            raise ValueError(f"validation_frequency must be > 0, got {self.validation_frequency}")
        if self.log_frequency <= 0:
            raise ValueError(f"log_frequency must be > 0, got {self.log_frequency}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

        # Loss weight validation
        if self.loss.enhancement_weight < 0:
            raise ValueError(f"enhancement_weight must be >= 0, got {self.loss.enhancement_weight}")
        if self.loss.classification_weight < 0:
            raise ValueError(f"classification_weight must be >= 0, got {self.loss.classification_weight}")
        if self.loss.enhancement_l1_weight < 0:
            raise ValueError(f"enhancement_l1_weight must be >= 0, got {self.loss.enhancement_l1_weight}")
        if self.loss.enhancement_l2_weight < 0:
            raise ValueError(f"enhancement_l2_weight must be >= 0, got {self.loss.enhancement_l2_weight}")
        if self.loss.enhancement_weight == 0.0 and self.loss.classification_weight == 0.0:
            raise ValueError(
                "At least one of enhancement_weight or classification_weight must be non-zero."
            )

    @classmethod
    def create(
        cls,
        device: Optional[str] = "auto",
        **kwargs,
    ) -> "TrainingConfig":
        """
        Create a TrainingConfig with explicit device resolution.

        device="auto"   → CUDA if available, otherwise CPU (silent fallback)
        device="cpu"    → CPU always
        device="cuda"   → CUDA if available, otherwise raise RuntimeError
        """
        resolved_device = _resolve_device(device)
        return cls(device=resolved_device, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """
        Load a TrainingConfig from configs/training.yaml.

        Only keys that correspond to TrainingConfig fields are consumed.
        Sub-keys are mapped to their respective nested dataclasses.
        """
        import yaml  # lazy import — not needed at module load time

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Training YAML not found: {path}")

        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            raw = {}

        # Build nested configs from YAML
        scheduler_cfg = SchedulerConfig(**raw["scheduler"]) if "scheduler" in raw else SchedulerConfig()
        loss_cfg = LossConfig(**raw["loss"]) if "loss" in raw else LossConfig()
        checkpoint_cfg = CheckpointConfig(**raw["checkpoint"]) if "checkpoint" in raw else CheckpointConfig()

        # Strip nested keys that we handle separately
        top_level_skip = {"scheduler", "loss", "checkpoint", "optimizer"}
        scalar_fields = {
            k: v for k, v in raw.items()
            if k not in top_level_skip
        }

        # Map only recognised TrainingConfig scalar fields
        recognised = {
            "batch_size", "epochs", "learning_rate", "weight_decay",
            "gradient_clip_norm", "seed", "num_workers", "validation_frequency",
            "log_frequency", "mixed_precision",
        }
        init_kwargs = {k: v for k, v in scalar_fields.items() if k in recognised}

        return cls(
            **init_kwargs,
            optimizer=raw.get("optimizer", "AdamW"),
            scheduler=scheduler_cfg,
            loss=loss_cfg,
            checkpoint=checkpoint_cfg,
            device="cpu",  # device is always resolved explicitly via create()
        )


# ---------------------------------------------------------------------------
# Device resolution helper
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    """
    Resolve device string according to Step 13 specification.

    "auto"  → cuda if available, else cpu
    "cpu"   → cpu
    "cuda"  → cuda if available, else RuntimeError
    None    → treated the same as "auto"
    """
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "device='cuda' was explicitly requested but CUDA is not available on this machine. "
            "Use device='auto' to allow silent CPU fallback."
        )
    raise ValueError(f"Unknown device specifier: '{device}'. Must be 'auto', 'cpu', or 'cuda'.")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Optimizer + Scheduler Factory
# ---------------------------------------------------------------------------

def create_optimizer_and_scheduler(
    model: nn.Module,
    config: TrainingConfig,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Create AdamW optimizer and ReduceLROnPlateau scheduler from the given model.

    IMPORTANT:
        - Only trainable parameters are registered with the optimizer.
        - scheduler.step() is NOT called here.
        - optimizer.step() is NOT called here.
        - No training is performed.

    Parameters
    ----------
    model : nn.Module
        The model whose trainable parameters will be registered.
    config : TrainingConfig
        Authoritative training configuration.

    Returns
    -------
    optimizer : torch.optim.AdamW
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    sched_cfg = config.scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.mode,
        factor=sched_cfg.factor,
        patience=sched_cfg.patience,
        min_lr=sched_cfg.min_lr,
    )

    return optimizer, scheduler
