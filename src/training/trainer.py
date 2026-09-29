"""
Phase 1 Step 14 — Training Loop
================================
Reusable Trainer class for the multi-task CNN+GRU speech enhancement model.

Design principles:
- Uses EXISTING: MultiTaskLoss, TrainingConfig, AdamW, ReduceLROnPlateau
- scheduler.step() is called with VALIDATION loss only
- Gradient clipping applied after backward(), before optimizer.step()
- Mixed precision (AMP) supported but disabled by default
- All model parameters preserved at 70,789 trainable
- No training happens during import
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import MultiTaskLoss
from src.training.training_config import TrainingConfig, create_optimizer_and_scheduler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Epoch result type alias
# ---------------------------------------------------------------------------

EpochMetrics = Dict[str, float]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """
    Reusable trainer for the multi-task CNN+GRU model.

    Parameters
    ----------
    model : nn.Module
        The CNN+GRU model.
    train_loader : DataLoader
        Training DataLoader producing batches with keys:
            noisy_features    (B, 2, 257, T)  float32
            target_stft       (B, 257, T)     complex64
            noise_class_label (B,)            int64
    val_loader : DataLoader
        Validation DataLoader (same format).
    optimizer : torch.optim.Optimizer
        AdamW optimizer (created by create_optimizer_and_scheduler).
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Scheduler stepped with validation total loss.
    loss_fn : MultiTaskLoss
        The frozen multi-task loss function.
    config : TrainingConfig
        Authoritative training configuration.
    device : str | torch.device, optional
        Device to run training on. Defaults to config.device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        device: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config

        # Resolve device
        if device is None:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device(device)

        # Move model and loss to device
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # AMP scaler — enabled only when mixed_precision=True and CUDA present
        self._use_amp = (
            config.mixed_precision
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        if self._use_amp:
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = torch.amp.GradScaler("cpu", enabled=False)

        # Checkpoint directory
        self._checkpoint_dir = Path(config.checkpoint.checkpoint_dir)

        # Training state
        self._best_val_loss: float = float("inf")
        self._history: List[EpochMetrics] = []

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        Run one full training epoch.

        Flow per batch:
          move tensors to device
          → model forward
          → MultiTaskLoss
          → loss.backward()
          → clip_grad_norm_
          → optimizer.step()

        Returns
        -------
        dict with keys:
            train_total_loss
            train_enhancement_loss
            train_classification_loss
        """
        self.model.train()

        # Propagate current epoch to dataset for deterministic epoch-varying mixture generation
        train_dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        total_loss_sum = 0.0
        enh_loss_sum = 0.0
        cls_loss_sum = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move tensors to device
            noisy_features = batch["noisy_features"].to(self.device, non_blocking=True)
            target_stft = batch["target_stft"].to(self.device, non_blocking=True)
            noise_class_label = batch["noise_class_label"].to(
                self.device, non_blocking=True
            )

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (optionally under AMP context)
            with torch.amp.autocast(
                device_type=self.device.type, enabled=self._use_amp
            ):
                enhanced_output, classification_logits = self.model(noisy_features)
                loss_dict = self.loss_fn(
                    enhanced_output,
                    classification_logits,
                    target_stft,
                    noise_class_label,
                )
                for loss_name, loss_value in loss_dict.items():
                    if not torch.isfinite(loss_value).all():
                        raise FloatingPointError(
                            f"Non-finite training loss {loss_name} at epoch {epoch}, batch {batch_idx + 1}"
                        )
                total_loss = loss_dict["total_loss"]

            # Backward
            self._scaler.scale(total_loss).backward()

            # Gradient clipping — AFTER backward, BEFORE optimizer.step()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm,
                error_if_nonfinite=True,
            )

            # Optimizer step
            self._scaler.step(self.optimizer)
            self._scaler.update()

            # Accumulate
            total_loss_sum += loss_dict["total_loss"].item()
            enh_loss_sum += loss_dict["enhancement_loss"].item()
            cls_loss_sum += loss_dict["classification_loss"].item()
            n_batches += 1

            if (batch_idx + 1) % self.config.log_frequency == 0:
                logger.info(
                    "Epoch %d | Batch %d/%d | loss=%.6f | enh=%.6f | cls=%.6f",
                    epoch,
                    batch_idx + 1,
                    len(self.train_loader),
                    loss_dict["total_loss"].item(),
                    loss_dict["enhancement_loss"].item(),
                    loss_dict["classification_loss"].item(),
                )

        if n_batches == 0:
            raise RuntimeError("Training DataLoader produced zero batches.")

        return {
            "train_total_loss": total_loss_sum / n_batches,
            "train_enhancement_loss": enh_loss_sum / n_batches,
            "train_classification_loss": cls_loss_sum / n_batches,
        }

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    def validate_epoch(self) -> EpochMetrics:
        """
        Run one full validation epoch under torch.no_grad().

        Guarantees:
            - model.eval() is set
            - backward() is NOT called
            - optimizer.step() is NOT called
            - model parameters are NOT modified

        Returns
        -------
        dict with keys:
            val_total_loss
            val_enhancement_loss
            val_classification_loss
        """
        self.model.eval()

        # Validation data must remain deterministic and stable across all epochs
        val_dataset = getattr(self.val_loader, "dataset", None)
        if hasattr(val_dataset, "set_epoch"):
            val_dataset.set_epoch(0)

        total_loss_sum = 0.0
        enh_loss_sum = 0.0
        cls_loss_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                noisy_features = batch["noisy_features"].to(
                    self.device, non_blocking=True
                )
                target_stft = batch["target_stft"].to(self.device, non_blocking=True)
                noise_class_label = batch["noise_class_label"].to(
                    self.device, non_blocking=True
                )

                enhanced_output, classification_logits = self.model(noisy_features)
                loss_dict = self.loss_fn(
                    enhanced_output,
                    classification_logits,
                    target_stft,
                    noise_class_label,
                )
                for loss_name, loss_value in loss_dict.items():
                    if not torch.isfinite(loss_value).all():
                        raise FloatingPointError(
                            f"Non-finite validation loss {loss_name}"
                        )

                total_loss_sum += loss_dict["total_loss"].item()
                enh_loss_sum += loss_dict["enhancement_loss"].item()
                cls_loss_sum += loss_dict["classification_loss"].item()
                n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Validation DataLoader produced zero batches.")

        return {
            "val_total_loss": total_loss_sum / n_batches,
            "val_enhancement_loss": enh_loss_sum / n_batches,
            "val_classification_loss": cls_loss_sum / n_batches,
        }

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def run(
        self,
        epochs: Optional[int] = None,
        checkpoint_dir: Optional[Any] = None,
    ) -> List[EpochMetrics]:
        """
        Run the full training loop for N epochs.

        Scheduler is stepped with VALIDATION total loss only:
            train_epoch → validate_epoch → scheduler.step(val_total_loss)

        Parameters
        ----------
        epochs : int, optional
            Number of epochs. Defaults to config.epochs.
        checkpoint_dir : str | Path, optional
            Override checkpoint directory.

        Returns
        -------
        List[EpochMetrics]
            Complete per-epoch history.
        """
        n_epochs = epochs if epochs is not None else self.config.epochs
        ckpt_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else self._checkpoint_dir
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        run_start = time.perf_counter()
        self._history = []

        for epoch in range(1, n_epochs + 1):
            epoch_start = time.perf_counter()

            # --- Training ---
            train_metrics = self.train_epoch(epoch)

            # --- Validation ---
            val_metrics: EpochMetrics = {}
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate_epoch()

                # Scheduler step uses VALIDATION loss (not training loss)
                val_total_loss = val_metrics["val_total_loss"]
                self.scheduler.step(val_total_loss)

                # Save best checkpoint
                is_best = val_total_loss < self._best_val_loss
                if is_best:
                    self._best_val_loss = val_total_loss
                    self.save_checkpoint(
                        epoch=epoch,
                        val_loss=val_total_loss,
                        filepath=ckpt_dir / "best_checkpoint.pt",
                    )
                    logger.info(
                        "Epoch %d | New best val_total_loss=%.6f — saved best checkpoint.",
                        epoch,
                        val_total_loss,
                    )

            # Save last checkpoint
            if self.config.checkpoint.save_last:
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_metrics.get("val_total_loss", float("inf")),
                    filepath=ckpt_dir / "last_checkpoint.pt",
                )

            epoch_duration = time.perf_counter() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_record: EpochMetrics = {
                "epoch": float(epoch),
                **train_metrics,
                **val_metrics,
                "learning_rate": current_lr,
                "epoch_duration_seconds": epoch_duration,
            }
            self._history.append(epoch_record)

            logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | lr=%.2e | %.1fs",
                epoch,
                n_epochs,
                train_metrics["train_total_loss"],
                val_metrics.get("val_total_loss", float("nan")),
                current_lr,
                epoch_duration,
            )

        total_duration = time.perf_counter() - run_start
        logger.info(
            "Training complete. Total time: %.1fs. Best val_total_loss: %.6f",
            total_duration,
            self._best_val_loss,
        )
        return self._history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filepath: Any,
    ) -> Path:
        """
        Save a checkpoint with all state required for resuming training.

        Saved keys:
            epoch, best_val_loss,
            model_state_dict, optimizer_state_dict, scheduler_state_dict,
            config (serialised scalar fields)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "best_val_loss": val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "gradient_clip_norm": self.config.gradient_clip_norm,
                "seed": self.config.seed,
                "device": self.config.device,
                "mixed_precision": self.config.mixed_precision,
            },
        }
        torch.save(checkpoint, filepath)
        logger.debug("Checkpoint saved: %s", filepath)
        return filepath

    @staticmethod
    def load_checkpoint(filepath: Any) -> Dict[str, Any]:
        """Load a checkpoint from disk and return the raw dict."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        return torch.load(filepath, map_location="cpu", weights_only=False)

    def restore_checkpoint(self, filepath: Any) -> int:
        """
        Restore model, optimizer, and scheduler state from a checkpoint.

        Returns the epoch stored in the checkpoint.
        """
        checkpoint = self.load_checkpoint(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return checkpoint["epoch"]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[EpochMetrics]:
        """Return a copy of the per-epoch training history."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    loss_fn: Optional[MultiTaskLoss] = None,
) -> Trainer:
    """
    Create a fully-configured Trainer from a TrainingConfig.

    Uses create_optimizer_and_scheduler() from Step 13.
    Creates MultiTaskLoss from config if not provided.
    """
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    if loss_fn is None:
        loss_fn = MultiTaskLoss(
            enhancement_weight=config.loss.enhancement_weight,
            classification_weight=config.loss.classification_weight,
            enhancement_l1_weight=config.loss.enhancement_l1_weight,
            enhancement_l2_weight=config.loss.enhancement_l2_weight,
        )

    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
    )
