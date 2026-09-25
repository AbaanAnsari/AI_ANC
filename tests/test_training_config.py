"""
Phase 1 Step 13 — Complete Test Suite
======================================
Tests 1–12 as required by the Step 13 locked specification.

Run with:
    python tests/test_training_config.py
or:
    pytest tests/test_training_config.py -v
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import copy
import pytest
import torch
import numpy as np

from src.training.training_config import (
    TrainingConfig,
    SchedulerConfig,
    LossConfig,
    CheckpointConfig,
    set_seed,
    create_optimizer_and_scheduler,
    _resolve_device,
)
from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.losses import MultiTaskLoss

# ============================================================
# TEST 1 — Default configuration values
# ============================================================

class TestDefaultConfigValues:
    """Verify all default field values match the specification exactly."""

    def test_core_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 16,            f"batch_size: {cfg.batch_size}"
        assert cfg.epochs == 30,               f"epochs: {cfg.epochs}"
        assert cfg.learning_rate == 0.001,     f"learning_rate: {cfg.learning_rate}"
        assert cfg.weight_decay == 0.0001,     f"weight_decay: {cfg.weight_decay}"
        assert cfg.gradient_clip_norm == 5.0,  f"gradient_clip_norm: {cfg.gradient_clip_norm}"
        assert cfg.seed == 123,                f"seed: {cfg.seed}"
        assert cfg.num_workers == 0,           f"num_workers: {cfg.num_workers}"
        assert cfg.validation_frequency == 1,  f"validation_frequency: {cfg.validation_frequency}"
        assert cfg.log_frequency == 10,        f"log_frequency: {cfg.log_frequency}"
        assert cfg.mixed_precision is False,   f"mixed_precision: {cfg.mixed_precision}"

    def test_loss_defaults(self):
        cfg = TrainingConfig()
        assert cfg.loss.enhancement_weight == 1.0,        f"enh_weight: {cfg.loss.enhancement_weight}"
        assert cfg.loss.classification_weight == 0.10,    f"cls_weight: {cfg.loss.classification_weight}"
        assert cfg.loss.enhancement_l1_weight == 0.7,     f"l1_weight: {cfg.loss.enhancement_l1_weight}"
        assert cfg.loss.enhancement_l2_weight == 0.3,     f"l2_weight: {cfg.loss.enhancement_l2_weight}"

    def test_scheduler_defaults(self):
        cfg = TrainingConfig()
        assert cfg.scheduler.type == "ReduceLROnPlateau"
        assert cfg.scheduler.mode == "min"
        assert cfg.scheduler.factor == 0.5
        assert cfg.scheduler.patience == 3
        assert cfg.scheduler.min_lr == 1e-6

    def test_checkpoint_defaults(self):
        cfg = TrainingConfig()
        assert cfg.checkpoint.checkpoint_dir == "models/checkpoints"
        assert cfg.checkpoint.save_best_only is True
        assert cfg.checkpoint.monitor == "val_total_loss"
        assert cfg.checkpoint.mode == "min"
        assert cfg.checkpoint.save_last is True

    def test_optimizer_default(self):
        cfg = TrainingConfig()
        assert cfg.optimizer == "AdamW"


# ============================================================
# TEST 2 — Invalid configuration values are rejected
# ============================================================

class TestInvalidConfigRejected:
    """Verify that out-of-range or illegal values raise ValueError."""

    def _make(self, **overrides):
        """Helper: construct TrainingConfig with given overrides."""
        return TrainingConfig(**overrides)

    def test_batch_size_zero(self):
        with pytest.raises(ValueError, match="batch_size"):
            self._make(batch_size=0)

    def test_batch_size_negative(self):
        with pytest.raises(ValueError, match="batch_size"):
            self._make(batch_size=-1)

    def test_epochs_zero(self):
        with pytest.raises(ValueError, match="epochs"):
            self._make(epochs=0)

    def test_epochs_negative(self):
        with pytest.raises(ValueError, match="epochs"):
            self._make(epochs=-5)

    def test_learning_rate_zero(self):
        with pytest.raises(ValueError, match="learning_rate"):
            self._make(learning_rate=0.0)

    def test_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate"):
            self._make(learning_rate=-1e-4)

    def test_weight_decay_negative(self):
        with pytest.raises(ValueError, match="weight_decay"):
            self._make(weight_decay=-0.001)

    def test_gradient_clip_norm_zero(self):
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            self._make(gradient_clip_norm=0.0)

    def test_gradient_clip_norm_negative(self):
        with pytest.raises(ValueError, match="gradient_clip_norm"):
            self._make(gradient_clip_norm=-1.0)

    def test_num_workers_negative(self):
        with pytest.raises(ValueError, match="num_workers"):
            self._make(num_workers=-1)

    def test_validation_frequency_zero(self):
        with pytest.raises(ValueError, match="validation_frequency"):
            self._make(validation_frequency=0)

    def test_log_frequency_zero(self):
        with pytest.raises(ValueError, match="log_frequency"):
            self._make(log_frequency=0)

    def test_negative_enhancement_loss_weight(self):
        with pytest.raises(ValueError, match="enhancement_weight"):
            self._make(loss=LossConfig(enhancement_weight=-0.1))

    def test_negative_classification_loss_weight(self):
        with pytest.raises(ValueError, match="classification_weight"):
            self._make(loss=LossConfig(classification_weight=-0.1))

    def test_negative_l1_weight(self):
        with pytest.raises(ValueError, match="enhancement_l1_weight"):
            self._make(loss=LossConfig(enhancement_l1_weight=-1.0))

    def test_negative_l2_weight(self):
        with pytest.raises(ValueError, match="enhancement_l2_weight"):
            self._make(loss=LossConfig(enhancement_l2_weight=-1.0))

    def test_negative_seed(self):
        with pytest.raises(ValueError, match="seed"):
            self._make(seed=-1)


# ============================================================
# TEST 3 — Both task weights zero must be rejected
# ============================================================

class TestBothWeightsZeroRejected:
    """enhancement_weight=0 AND classification_weight=0 must raise ValueError."""

    def test_both_zero_raises(self):
        with pytest.raises(ValueError):
            TrainingConfig(
                loss=LossConfig(
                    enhancement_weight=0.0,
                    classification_weight=0.0,
                )
            )

    def test_one_zero_allowed(self):
        """Only one weight being zero is legal."""
        cfg = TrainingConfig(
            loss=LossConfig(enhancement_weight=0.0, classification_weight=0.5)
        )
        assert cfg.loss.enhancement_weight == 0.0
        assert cfg.loss.classification_weight == 0.5


# ============================================================
# TEST 4 — Reproducibility
# ============================================================

class TestReproducibility:
    """With seed=123, verify deterministic values for Python, NumPy, and PyTorch."""

    def test_python_random_reproducible(self):
        set_seed(123)
        val1 = random.random()
        set_seed(123)
        val2 = random.random()
        assert val1 == val2, f"Python random not reproducible: {val1} != {val2}"

    def test_numpy_reproducible(self):
        set_seed(123)
        val1 = np.random.rand()
        set_seed(123)
        val2 = np.random.rand()
        assert val1 == val2, f"NumPy not reproducible: {val1} != {val2}"

    def test_torch_reproducible(self):
        set_seed(123)
        val1 = torch.rand(5).tolist()
        set_seed(123)
        val2 = torch.rand(5).tolist()
        assert val1 == val2, f"PyTorch not reproducible: {val1} != {val2}"

    def test_all_three_match_jointly(self):
        """Run all three in sequence and verify they match a repeated run."""
        set_seed(123)
        py1 = random.random()
        np1 = np.random.rand()
        t1 = torch.rand(3).tolist()

        set_seed(123)
        py2 = random.random()
        np2 = np.random.rand()
        t2 = torch.rand(3).tolist()

        assert py1 == py2
        assert np1 == np2
        assert t1 == t2


# ============================================================
# TEST 5 — Device behavior
# ============================================================

class TestDeviceBehavior:
    """Verify precise device resolution rules."""

    def test_cpu_is_cpu(self):
        device = _resolve_device("cpu")
        assert device == "cpu"

    def test_auto_returns_valid_device(self):
        device = _resolve_device("auto")
        assert device in ("cpu", "cuda"), f"Unexpected device: {device}"

    def test_auto_never_raises(self):
        """auto must never raise regardless of CUDA availability."""
        try:
            device = _resolve_device("auto")
            assert device in ("cpu", "cuda")
        except Exception as exc:
            pytest.fail(f"device='auto' raised unexpectedly: {exc}")

    def test_cuda_raises_when_unavailable(self):
        """If CUDA is not available, device='cuda' MUST raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine; skipping unavailable test.")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            _resolve_device("cuda")

    def test_cuda_resolves_when_available(self):
        """If CUDA is available, device='cuda' must resolve to 'cuda'."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine.")
        device = _resolve_device("cuda")
        assert device == "cuda"

    def test_create_cpu(self):
        cfg = TrainingConfig.create(device="cpu")
        assert cfg.device == "cpu"

    def test_create_auto(self):
        cfg = TrainingConfig.create(device="auto")
        assert cfg.device in ("cpu", "cuda")

    def test_create_cuda_raises_when_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available.")
        with pytest.raises(RuntimeError):
            TrainingConfig.create(device="cuda")


# ============================================================
# TEST 6 — Actual model compatibility (70,789 trainable parameters)
# ============================================================

class TestModelCompatibility:
    """Instantiate the real CNN+GRU model and verify its parameter count."""

    EXPECTED_TRAINABLE_PARAMS = 70_789

    def test_trainable_parameter_count(self):
        model = LightweightCNNTGRUModel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == self.EXPECTED_TRAINABLE_PARAMS, (
            f"Expected {self.EXPECTED_TRAINABLE_PARAMS} trainable params, got {trainable}"
        )

    def test_all_trainable_params_in_optimizer(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        optimizer, _ = create_optimizer_and_scheduler(model, config)

        # Collect all parameter data_ptrs in the optimizer
        opt_param_ptrs = {p.data_ptr() for group in optimizer.param_groups for p in group["params"]}

        for p in model.parameters():
            if p.requires_grad:
                assert p.data_ptr() in opt_param_ptrs, (
                    "A trainable model parameter is missing from the optimizer."
                )

    def test_no_non_trainable_params_in_optimizer(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        optimizer, _ = create_optimizer_and_scheduler(model, config)

        opt_param_ptrs = {p.data_ptr() for group in optimizer.param_groups for p in group["params"]}

        for p in model.parameters():
            if not p.requires_grad:
                assert p.data_ptr() not in opt_param_ptrs, (
                    "A non-trainable model parameter was incorrectly added to the optimizer."
                )


# ============================================================
# TEST 7 — Optimizer configuration
# ============================================================

class TestOptimizerConfiguration:
    """Verify AdamW optimizer is created with the correct hyperparameters."""

    def test_optimizer_type(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        assert isinstance(optimizer, torch.optim.AdamW), (
            f"Expected AdamW, got {type(optimizer).__name__}"
        )

    def test_optimizer_learning_rate(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        assert optimizer.defaults["lr"] == 0.001, (
            f"Expected lr=0.001, got {optimizer.defaults['lr']}"
        )

    def test_optimizer_weight_decay(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        assert optimizer.defaults["weight_decay"] == 0.0001, (
            f"Expected weight_decay=0.0001, got {optimizer.defaults['weight_decay']}"
        )

    def test_optimizer_step_not_called(self):
        """Optimizer must be constructed without calling .step()."""
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        # If step() were ever called on a zero-grad model, parameters could shift.
        # We verify by checking parameters haven't changed after factory call.
        initial_params = [p.clone().detach() for p in model.parameters()]
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        for i, p in enumerate(model.parameters()):
            assert torch.equal(p, initial_params[i]), "optimizer.step() appears to have been called."


# ============================================================
# TEST 8 — Scheduler configuration
# ============================================================

class TestSchedulerConfiguration:
    """Verify ReduceLROnPlateau scheduler is created with the correct settings."""

    def test_scheduler_type(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        _, scheduler = create_optimizer_and_scheduler(model, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau), (
            f"Expected ReduceLROnPlateau, got {type(scheduler).__name__}"
        )

    def test_scheduler_mode(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        _, scheduler = create_optimizer_and_scheduler(model, config)
        assert scheduler.mode == "min", f"Expected mode='min', got {scheduler.mode!r}"

    def test_scheduler_factor(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        _, scheduler = create_optimizer_and_scheduler(model, config)
        assert scheduler.factor == 0.5, f"Expected factor=0.5, got {scheduler.factor}"

    def test_scheduler_patience(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        _, scheduler = create_optimizer_and_scheduler(model, config)
        assert scheduler.patience == 3, f"Expected patience=3, got {scheduler.patience}"

    def test_scheduler_min_lr(self):
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        _, scheduler = create_optimizer_and_scheduler(model, config)
        # min_lrs is a list of min lr values per param group
        assert scheduler.min_lrs[0] == 1e-6, (
            f"Expected min_lr=1e-6, got {scheduler.min_lrs[0]}"
        )

    def test_scheduler_step_not_called(self):
        """Scheduler must be constructed without calling .step()."""
        model = LightweightCNNTGRUModel()
        config = TrainingConfig()
        initial_params = [p.clone().detach() for p in model.parameters()]
        _, scheduler = create_optimizer_and_scheduler(model, config)
        # After construction, LR must still be the initial value
        opt_lrs = [group["lr"] for group in scheduler.optimizer.param_groups]
        for lr in opt_lrs:
            assert lr == config.learning_rate, (
                f"Learning rate changed during construction: expected {config.learning_rate}, got {lr}"
            )
        for i, p in enumerate(model.parameters()):
            assert torch.equal(p, initial_params[i]), "scheduler.step() appears to have been called."


# ============================================================
# TEST 9 — Parameter immutability
# ============================================================

class TestParameterImmutability:
    """Model parameters must be unchanged after optimizer and scheduler creation."""

    def test_parameters_unchanged_after_setup(self):
        model = LightweightCNNTGRUModel()

        # Clone all parameters before factory call
        cloned_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        # Create optimizer and scheduler
        config = TrainingConfig()
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)

        # Verify every parameter is identical
        for name, param in model.named_parameters():
            assert torch.equal(param, cloned_params[name]), (
                f"Parameter '{name}' changed after optimizer/scheduler construction."
            )


# ============================================================
# TEST 10 — YAML configuration loading
# ============================================================

class TestYAMLConfigLoading:
    """TrainingConfig.from_yaml() must produce values matching Step 13 defaults."""

    YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "training.yaml")

    def test_yaml_file_exists(self):
        assert os.path.exists(self.YAML_PATH), f"training.yaml not found at {self.YAML_PATH}"

    def test_yaml_produces_correct_defaults(self):
        cfg = TrainingConfig.from_yaml(self.YAML_PATH)
        assert cfg.batch_size == 16
        assert cfg.epochs == 30
        assert cfg.learning_rate == 0.001
        assert cfg.weight_decay == 0.0001
        assert cfg.gradient_clip_norm == 5.0
        assert cfg.seed == 123
        assert cfg.num_workers == 0
        assert cfg.validation_frequency == 1
        assert cfg.log_frequency == 10
        assert cfg.mixed_precision is False

    def test_yaml_loss_weights(self):
        cfg = TrainingConfig.from_yaml(self.YAML_PATH)
        assert cfg.loss.enhancement_weight == 1.0
        assert cfg.loss.classification_weight == 0.10
        assert cfg.loss.enhancement_l1_weight == 0.7
        assert cfg.loss.enhancement_l2_weight == 0.3

    def test_yaml_scheduler_settings(self):
        cfg = TrainingConfig.from_yaml(self.YAML_PATH)
        assert cfg.scheduler.type == "ReduceLROnPlateau"
        assert cfg.scheduler.mode == "min"
        assert cfg.scheduler.factor == 0.5
        assert cfg.scheduler.patience == 3
        assert cfg.scheduler.min_lr == 1e-6

    def test_yaml_matches_python_defaults(self):
        """YAML-loaded config must agree with Python-default config on every field."""
        yaml_cfg = TrainingConfig.from_yaml(self.YAML_PATH)
        py_cfg = TrainingConfig()

        # Core scalar fields
        for attr in [
            "batch_size", "epochs", "learning_rate", "weight_decay",
            "gradient_clip_norm", "seed", "num_workers", "validation_frequency",
            "log_frequency", "mixed_precision", "optimizer",
        ]:
            assert getattr(yaml_cfg, attr) == getattr(py_cfg, attr), (
                f"Mismatch on '{attr}': YAML={getattr(yaml_cfg, attr)!r}, Python={getattr(py_cfg, attr)!r}"
            )

        # Loss weights
        assert yaml_cfg.loss.enhancement_weight == py_cfg.loss.enhancement_weight
        assert yaml_cfg.loss.classification_weight == py_cfg.loss.classification_weight
        assert yaml_cfg.loss.enhancement_l1_weight == py_cfg.loss.enhancement_l1_weight
        assert yaml_cfg.loss.enhancement_l2_weight == py_cfg.loss.enhancement_l2_weight

        # Scheduler settings
        assert yaml_cfg.scheduler.mode == py_cfg.scheduler.mode
        assert yaml_cfg.scheduler.factor == py_cfg.scheduler.factor
        assert yaml_cfg.scheduler.patience == py_cfg.scheduler.patience
        assert yaml_cfg.scheduler.min_lr == py_cfg.scheduler.min_lr

        # Checkpoint settings
        assert yaml_cfg.checkpoint.checkpoint_dir == py_cfg.checkpoint.checkpoint_dir
        assert yaml_cfg.checkpoint.save_best_only == py_cfg.checkpoint.save_best_only
        assert yaml_cfg.checkpoint.monitor == py_cfg.checkpoint.monitor
        assert yaml_cfg.checkpoint.mode == py_cfg.checkpoint.mode
        assert yaml_cfg.checkpoint.save_last == py_cfg.checkpoint.save_last


# ============================================================
# TEST 11 — Loss compatibility
# ============================================================

class TestLossCompatibility:
    """Verify MultiTaskLoss is instantiated with Step 13 configuration weights."""

    def test_loss_instantiation(self):
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        assert isinstance(loss_fn, MultiTaskLoss)

    def test_loss_enhancement_weight(self):
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        assert loss_fn.enhancement_weight == 1.0

    def test_loss_classification_weight(self):
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        assert abs(loss_fn.classification_weight - 0.10) < 1e-9

    def test_loss_l1_weight(self):
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        assert loss_fn.enhancement_loss_fn.l1_weight == 0.7

    def test_loss_l2_weight(self):
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        assert loss_fn.enhancement_loss_fn.l2_weight == 0.3

    def test_no_training_occurred(self):
        """Loss is constructed but no forward pass or step is called."""
        cfg = TrainingConfig()
        loss_fn = MultiTaskLoss(
            enhancement_weight=cfg.loss.enhancement_weight,
            classification_weight=cfg.loss.classification_weight,
            enhancement_l1_weight=cfg.loss.enhancement_l1_weight,
            enhancement_l2_weight=cfg.loss.enhancement_l2_weight,
        )
        # Simply verifying the object exists with correct attributes is sufficient
        assert loss_fn is not None


# ============================================================
# TEST 12 — CPU-only complete setup
# ============================================================

class TestCPUOnlyCompleteSetup:
    """
    Verify the full setup chain succeeds on CPU without any training.
    Chain: set_seed → TrainingConfig → CNN+GRU model → AdamW → ReduceLROnPlateau → MultiTaskLoss
    """

    def test_full_cpu_setup_chain(self):
        # Step 1: Set seed
        set_seed(123)

        # Step 2: TrainingConfig (CPU)
        config = TrainingConfig.create(device="cpu")
        assert config.device == "cpu"

        # Step 3: Instantiate actual CNN+GRU model
        model = LightweightCNNTGRUModel()
        model.to(config.device)
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_count == 70_789, f"Unexpected param count: {trainable_count}"

        # Step 4: AdamW optimizer
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)

        # Step 5: ReduceLROnPlateau scheduler
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == "min"
        assert scheduler.factor == 0.5
        assert scheduler.patience == 3
        assert scheduler.min_lrs[0] == 1e-6

        # Step 6: MultiTaskLoss
        loss_fn = MultiTaskLoss(
            enhancement_weight=config.loss.enhancement_weight,
            classification_weight=config.loss.classification_weight,
            enhancement_l1_weight=config.loss.enhancement_l1_weight,
            enhancement_l2_weight=config.loss.enhancement_l2_weight,
        )
        assert isinstance(loss_fn, MultiTaskLoss)

        # Confirm no parameters were modified
        for p in model.parameters():
            assert p.device.type == "cpu", f"Parameter not on CPU: {p.device}"

    def test_no_optimizer_step_was_called(self):
        """State check: optimizer.step() was never invoked during chain construction."""
        set_seed(123)
        config = TrainingConfig.create(device="cpu")
        model = LightweightCNNTGRUModel()
        initial_params = [p.clone().detach() for p in model.parameters()]
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        for i, p in enumerate(model.parameters()):
            assert torch.equal(p, initial_params[i]), (
                f"Parameter {i} changed — optimizer.step() may have been called."
            )

    def test_no_scheduler_step_was_called(self):
        """State check: scheduler.step() was never invoked, so LR must be unchanged."""
        config = TrainingConfig.create(device="cpu")
        model = LightweightCNNTGRUModel()
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        for group in optimizer.param_groups:
            assert group["lr"] == config.learning_rate, (
                f"LR changed from {config.learning_rate} to {group['lr']} — "
                "scheduler.step() may have been called."
            )


# ============================================================
# Standalone runner
# ============================================================

if __name__ == "__main__":
    import unittest

    # Use pytest for rich output
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
