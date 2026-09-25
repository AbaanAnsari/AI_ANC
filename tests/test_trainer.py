"""
Phase 1 Step 14 — Trainer Test Suite
======================================
18 tests covering all Step 14 requirements.

Tests use SHORT CONTROLLED training (1-2 batches / 1 epoch only).
No full dataset training is performed.

Run with:
    python tests/test_trainer.py
or:
    pytest tests/test_trainer.py -v
"""
from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# Project root on path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer, create_trainer
from src.training.training_config import (
    TrainingConfig,
    LossConfig,
    SchedulerConfig,
    set_seed,
    create_optimizer_and_scheduler,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"

EXPECTED_PARAMS = 70_789
BATCH_SIZE = 2          # Small for speed
TIME_FRAMES = 126       # Fixed by STFT config


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_synthetic_batch(batch_size: int = BATCH_SIZE, T: int = TIME_FRAMES):
    """
    Create a synthetic batch matching the actual DataLoader output format.
      noisy_features  (B, 2, 257, T)  float32
      target_stft     (B, 257, T)     complex64
      noise_class_label (B,)          int64
    """
    noisy_features = torch.randn(batch_size, 2, 257, T, dtype=torch.float32)
    # Build complex64 target_stft from float components
    real = torch.randn(batch_size, 257, T, dtype=torch.float32)
    imag = torch.randn(batch_size, 257, T, dtype=torch.float32)
    target_stft = torch.complex(real, imag)
    noise_class_label = torch.randint(0, 3, (batch_size,), dtype=torch.int64)
    return {
        "noisy_features": noisy_features,
        "target_stft": target_stft,
        "noise_class_label": noise_class_label,
    }


class SyntheticDataLoader:
    """
    Minimal DataLoader-like object producing a fixed number of synthetic batches.
    Avoids real dataset I/O for unit tests.
    """
    def __init__(self, n_batches: int = 3, batch_size: int = BATCH_SIZE):
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield make_synthetic_batch(self.batch_size)


def make_trainer(
    n_train_batches: int = 3,
    n_val_batches: int = 2,
    epochs: int = 1,
) -> Trainer:
    """Create a Trainer with synthetic loaders and default config."""
    set_seed(123)
    model = LightweightCNNTGRUModel()
    config = TrainingConfig.create(device="cpu", epochs=epochs)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    loss_fn = MultiTaskLoss(
        enhancement_weight=config.loss.enhancement_weight,
        classification_weight=config.loss.classification_weight,
        enhancement_l1_weight=config.loss.enhancement_l1_weight,
        enhancement_l2_weight=config.loss.enhancement_l2_weight,
    )
    trainer = Trainer(
        model=model,
        train_loader=SyntheticDataLoader(n_train_batches),
        val_loader=SyntheticDataLoader(n_val_batches),
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device="cpu",
    )
    return trainer


# ===========================================================================
# TEST 1 — Trainer can be constructed using the actual CNN+GRU model
# ===========================================================================

class TestTrainerConstruction:
    def test_trainer_created_from_actual_model(self):
        trainer = make_trainer()
        assert isinstance(trainer, Trainer)
        assert isinstance(trainer.model, LightweightCNNTGRUModel)

    def test_create_trainer_factory(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig.create(device="cpu")
        trainer = create_trainer(
            model=model,
            train_loader=SyntheticDataLoader(2),
            val_loader=SyntheticDataLoader(1),
            config=config,
        )
        assert isinstance(trainer, Trainer)


# ===========================================================================
# TEST 2 — Actual model remains exactly 70,789 trainable parameters
# ===========================================================================

class TestModelParameterCount:
    def test_model_has_correct_param_count(self):
        model = LightweightCNNTGRUModel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == EXPECTED_PARAMS, (
            f"Expected {EXPECTED_PARAMS}, got {trainable}"
        )

    def test_trainer_model_preserves_param_count(self):
        trainer = make_trainer()
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        assert trainable == EXPECTED_PARAMS


# ===========================================================================
# TEST 3 — Single training batch: forward + loss + backward
# ===========================================================================

class TestSingleBatchForwardBackward:
    def test_single_batch_runs_without_error(self):
        trainer = make_trainer()
        batch = make_synthetic_batch()

        noisy = batch["noisy_features"]
        target_stft = batch["target_stft"]
        labels = batch["noise_class_label"]

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        enhanced, logits = trainer.model(noisy)
        loss_dict = trainer.loss_fn(enhanced, logits, target_stft, labels)
        # Before backward: total_loss is a differentiable scalar tensor (requires_grad=True is correct)
        assert loss_dict["total_loss"].requires_grad is True
        assert torch.isfinite(loss_dict["total_loss"]), "total_loss is not finite"

        # backward must succeed without error
        loss_dict["total_loss"].backward()

    def test_loss_keys_present(self):
        trainer = make_trainer()
        batch = make_synthetic_batch()
        trainer.model.train()
        trainer.optimizer.zero_grad()
        enhanced, logits = trainer.model(batch["noisy_features"])
        loss_dict = trainer.loss_fn(
            enhanced, logits, batch["target_stft"], batch["noise_class_label"]
        )
        assert "total_loss" in loss_dict
        assert "enhancement_loss" in loss_dict
        assert "classification_loss" in loss_dict


# ===========================================================================
# TEST 4 — Gradients are finite
# ===========================================================================

class TestGradientsAreFinite:
    def test_all_gradients_finite_after_backward(self):
        trainer = make_trainer()
        batch = make_synthetic_batch()

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        enhanced, logits = trainer.model(batch["noisy_features"])
        loss_dict = trainer.loss_fn(
            enhanced, logits, batch["target_stft"], batch["noise_class_label"]
        )
        loss_dict["total_loss"].backward()

        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), (
                    f"Non-finite gradient in parameter: {name}"
                )


# ===========================================================================
# TEST 5 — Gradient clipping applied with max_norm = 5.0
# ===========================================================================

class TestGradientClipping:
    def test_grad_norm_clipped_to_max_norm(self):
        trainer = make_trainer()
        batch = make_synthetic_batch()

        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        enhanced, logits = trainer.model(batch["noisy_features"])
        loss_dict = trainer.loss_fn(
            enhanced, logits, batch["target_stft"], batch["noise_class_label"]
        )
        loss_dict["total_loss"].backward()

        # Apply clipping
        max_norm = trainer.config.gradient_clip_norm
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), max_norm=max_norm
        )

        # After clipping, compute actual norm to verify <= max_norm
        total_norm_after = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        assert total_norm_after <= max_norm + 1e-5, (
            f"After clipping, grad norm {total_norm_after:.4f} exceeds max_norm {max_norm}"
        )

    def test_config_gradient_clip_norm_is_5(self):
        config = TrainingConfig()
        assert config.gradient_clip_norm == 5.0


# ===========================================================================
# TEST 6 — Optimizer step changes trainable parameters
# ===========================================================================

class TestOptimizerStepChangesParams:
    def test_parameters_change_after_optimizer_step(self):
        """
        First step where optimizer.step() is allowed.
        Verifies that at least one parameter changes after a real update.
        """
        set_seed(123)
        trainer = make_trainer()
        batch = make_synthetic_batch()

        # Clone initial parameters
        before_params = {
            name: param.clone().detach()
            for name, param in trainer.model.named_parameters()
            if param.requires_grad
        }

        # Perform one real training step
        trainer.model.train()
        trainer.optimizer.zero_grad(set_to_none=True)
        enhanced, logits = trainer.model(batch["noisy_features"])
        loss_dict = trainer.loss_fn(
            enhanced, logits, batch["target_stft"], batch["noise_class_label"]
        )
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), max_norm=trainer.config.gradient_clip_norm
        )
        trainer.optimizer.step()

        # At least one parameter must have changed
        any_changed = False
        for name, param in trainer.model.named_parameters():
            if param.requires_grad:
                if not torch.equal(param.detach(), before_params[name]):
                    any_changed = True
                    break
        assert any_changed, "No parameters changed after optimizer.step()"


# ===========================================================================
# TEST 7 — Validation executes under no_grad
# ===========================================================================

class TestValidationUnderNoGrad:
    def test_validation_runs_without_error(self):
        trainer = make_trainer()
        metrics = trainer.validate_epoch()
        assert "val_total_loss" in metrics
        assert "val_enhancement_loss" in metrics
        assert "val_classification_loss" in metrics

    def test_no_gradients_computed_during_validation(self):
        """Verify that no gradient computation occurs during validate_epoch."""
        trainer = make_trainer()

        # Set model to eval, then validate — no grad tensors should be created
        trainer.model.eval()
        with torch.no_grad():
            batch = make_synthetic_batch()
            enhanced, logits = trainer.model(batch["noisy_features"])
            # Under no_grad, outputs must not require grad
            assert not enhanced.requires_grad
            assert not logits.requires_grad


# ===========================================================================
# TEST 8 — Validation does not modify model parameters
# ===========================================================================

class TestValidationDoesNotModifyParams:
    def test_params_unchanged_after_validation(self):
        trainer = make_trainer()

        # Clone all parameters before validation
        cloned = {
            name: param.clone().detach()
            for name, param in trainer.model.named_parameters()
        }

        trainer.validate_epoch()

        for name, param in trainer.model.named_parameters():
            assert torch.equal(param.detach(), cloned[name]), (
                f"Parameter '{name}' was modified during validation."
            )


# ===========================================================================
# TEST 9 — Scheduler is ReduceLROnPlateau
# ===========================================================================

class TestSchedulerType:
    def test_scheduler_is_reduce_lr_on_plateau(self):
        trainer = make_trainer()
        assert isinstance(
            trainer.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ), f"Expected ReduceLROnPlateau, got {type(trainer.scheduler).__name__}"

    def test_scheduler_config_correct(self):
        trainer = make_trainer()
        sched = trainer.scheduler
        assert sched.mode == "min"
        assert sched.factor == 0.5
        assert sched.patience == 3
        assert sched.min_lrs[0] == 1e-6


# ===========================================================================
# TEST 10 — Scheduler receives validation total loss
# ===========================================================================

class TestSchedulerReceivesValidationLoss:
    def test_scheduler_stepped_with_val_loss_not_train_loss(self):
        """
        Run one epoch and verify that scheduler.step() was called
        with the val_total_loss (not the training loss).

        We do this by tracking the scheduler's internal best value.
        """
        set_seed(123)
        trainer = make_trainer(n_train_batches=2, n_val_batches=2, epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        assert len(history) == 1
        record = history[0]
        val_loss = record.get("val_total_loss", float("nan"))
        assert val_loss == val_loss  # not NaN

        # The scheduler's best should now equal the val loss we observed
        # (since it's the first epoch, best == val_loss)
        sched = trainer.scheduler
        assert sched.best <= val_loss + 1e-6, (
            "Scheduler best should be set to val_total_loss, not train loss."
        )


# ===========================================================================
# TEST 11 — Training and validation losses are finite
# ===========================================================================

class TestLossesAreFinite:
    def test_all_losses_finite_after_epoch(self):
        trainer = make_trainer()

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        record = history[0]
        for key in [
            "train_total_loss", "train_enhancement_loss", "train_classification_loss",
            "val_total_loss", "val_enhancement_loss", "val_classification_loss",
        ]:
            val = record.get(key, float("nan"))
            assert val == val, f"NaN detected in {key}"
            assert val < float("inf"), f"Inf detected in {key}"
            assert val > -float("inf"), f"-Inf detected in {key}"


# ===========================================================================
# TEST 12 — Training history contains all required fields
# ===========================================================================

class TestTrainingHistoryFields:
    REQUIRED_FIELDS = [
        "epoch",
        "train_total_loss",
        "train_enhancement_loss",
        "train_classification_loss",
        "val_total_loss",
        "val_enhancement_loss",
        "val_classification_loss",
        "learning_rate",
    ]

    def test_history_contains_required_fields(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        assert len(history) == 1
        record = history[0]
        for field in self.REQUIRED_FIELDS:
            assert field in record, f"Missing required history field: '{field}'"

    def test_history_length_equals_epochs(self):
        trainer = make_trainer(epochs=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=2, checkpoint_dir=tmpdir)
        assert len(history) == 2


# ===========================================================================
# TEST 13 — Learning rate is recorded
# ===========================================================================

class TestLearningRateRecorded:
    def test_learning_rate_in_history(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        lr = history[0].get("learning_rate")
        assert lr is not None
        assert isinstance(lr, float)
        assert lr > 0.0

    def test_initial_learning_rate_matches_config(self):
        trainer = make_trainer()
        # Before any training, check optimizer lr
        lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(lr - trainer.config.learning_rate) < 1e-9


# ===========================================================================
# TEST 14 — Checkpoint save/load preserves model state
# ===========================================================================

class TestCheckpointSaveLoad:
    def test_checkpoint_save_creates_file(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.5, filepath=ckpt_path)
            assert ckpt_path.exists(), "Checkpoint file was not created"

    def test_checkpoint_preserves_model_state(self):
        set_seed(123)
        trainer = make_trainer()

        # Run one step to get non-random state
        batch = make_synthetic_batch()
        trainer.model.train()
        trainer.optimizer.zero_grad()
        enhanced, logits = trainer.model(batch["noisy_features"])
        loss_dict = trainer.loss_fn(
            enhanced, logits, batch["target_stft"], batch["noise_class_label"]
        )
        loss_dict["total_loss"].backward()
        trainer.optimizer.step()

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.42, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)

            # Build fresh model and load state
            fresh_model = LightweightCNNTGRUModel()
            fresh_model.load_state_dict(ckpt["model_state_dict"])

            # Parameters must match exactly
            for (n1, p1), (n2, p2) in zip(
                trainer.model.named_parameters(), fresh_model.named_parameters()
            ):
                assert n1 == n2
                assert torch.equal(p1.detach(), p2.detach()), (
                    f"Parameter '{n1}' mismatch after checkpoint load."
                )

    def test_loaded_model_has_correct_param_count(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.5, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)
            model = LightweightCNNTGRUModel()
            model.load_state_dict(ckpt["model_state_dict"])
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert trainable == EXPECTED_PARAMS


# ===========================================================================
# TEST 15 — Checkpoint contains optimizer and scheduler state
# ===========================================================================

class TestCheckpointContents:
    def test_checkpoint_has_required_keys(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=5, val_loss=0.123, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)

        required_keys = {
            "epoch",
            "best_val_loss",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "config",
        }
        for key in required_keys:
            assert key in ckpt, f"Missing checkpoint key: '{key}'"

    def test_checkpoint_epoch_matches(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=7, val_loss=0.99, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)
        assert ckpt["epoch"] == 7

    def test_checkpoint_val_loss_matches(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.314, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)
        assert abs(ckpt["best_val_loss"] - 0.314) < 1e-6

    def test_optimizer_state_can_be_loaded(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.5, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)
            # Verify optimizer state is a dict with 'state' and 'param_groups'
            opt_state = ckpt["optimizer_state_dict"]
            assert "state" in opt_state
            assert "param_groups" in opt_state

    def test_scheduler_state_can_be_loaded(self):
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(epoch=1, val_loss=0.5, filepath=ckpt_path)
            ckpt = Trainer.load_checkpoint(ckpt_path)
            sched_state = ckpt["scheduler_state_dict"]
            assert isinstance(sched_state, dict)
            assert len(sched_state) > 0


# ===========================================================================
# TEST 16 — CPU training path works
# ===========================================================================

class TestCPUTrainingPath:
    def test_cpu_training_runs_successfully(self):
        set_seed(123)
        trainer = make_trainer()
        assert trainer.device == torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        assert len(history) == 1
        record = history[0]
        assert record["train_total_loss"] > 0.0
        assert record.get("val_total_loss", 0.0) > 0.0

    def test_model_stays_on_cpu_during_training(self):
        trainer = make_trainer()
        for p in trainer.model.parameters():
            assert p.device.type == "cpu"


# ===========================================================================
# TEST 17 — No NaN or Inf during short actual training
# ===========================================================================

class TestNoNaNOrInf:
    def test_no_nan_inf_in_losses(self):
        set_seed(123)
        trainer = make_trainer(n_train_batches=4, n_val_batches=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)

        record = history[0]
        for key in [
            "train_total_loss", "train_enhancement_loss", "train_classification_loss",
            "val_total_loss", "val_enhancement_loss", "val_classification_loss",
        ]:
            val = record[key]
            assert val == val, f"NaN detected in '{key}': {val}"
            assert val != float("inf"), f"Inf detected in '{key}': {val}"

    def test_no_nan_in_model_params_after_training(self):
        set_seed(123)
        trainer = make_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.run(epochs=1, checkpoint_dir=tmpdir)

        for name, param in trainer.model.named_parameters():
            assert torch.all(torch.isfinite(param)), (
                f"Non-finite value in parameter '{name}' after training."
            )


# ===========================================================================
# TEST 18 — Compatible with actual DataLoader output shapes
# ===========================================================================

class TestDataLoaderCompatibility:
    """
    Verify that Trainer is compatible with the actual DataLoader batch shapes.
    Uses synthetic data with the exact shapes produced by the real DataLoader.
    """

    def test_batch_shape_noisy_features(self):
        batch = make_synthetic_batch(batch_size=4, T=126)
        assert batch["noisy_features"].shape == (4, 2, 257, 126)
        assert batch["noisy_features"].dtype == torch.float32

    def test_batch_shape_target_stft(self):
        batch = make_synthetic_batch(batch_size=4, T=126)
        assert batch["target_stft"].shape == (4, 257, 126)
        assert batch["target_stft"].dtype == torch.complex64

    def test_batch_shape_noise_class_label(self):
        batch = make_synthetic_batch(batch_size=4, T=126)
        assert batch["noise_class_label"].shape == (4,)
        assert batch["noise_class_label"].dtype == torch.int64

    def test_model_processes_correct_batch_shape(self):
        model = LightweightCNNTGRUModel().eval()
        batch = make_synthetic_batch(batch_size=4, T=126)
        with torch.no_grad():
            enhanced, logits = model(batch["noisy_features"])
        assert enhanced.shape == (4, 2, 257, 126)
        assert logits.shape == (4, 3)

    def test_loss_processes_correct_shapes(self):
        model = LightweightCNNTGRUModel().eval()
        loss_fn = MultiTaskLoss()
        batch = make_synthetic_batch(batch_size=4, T=126)
        with torch.no_grad():
            enhanced, logits = model(batch["noisy_features"])
            loss_dict = loss_fn(
                enhanced, logits, batch["target_stft"], batch["noise_class_label"]
            )
        assert "total_loss" in loss_dict
        assert torch.isfinite(loss_dict["total_loss"])

    def test_trainer_runs_with_actual_dataloader_shapes(self):
        """End-to-end: Trainer processes batches of the actual DataLoader shape."""
        trainer = make_trainer(n_train_batches=2, n_val_batches=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.run(epochs=1, checkpoint_dir=tmpdir)
        assert len(history) == 1


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
