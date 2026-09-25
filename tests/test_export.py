"""
tests/test_export.py — Phase 1 Step 17 Test Suite
==================================================
14 tests covering checkpoint verification, state restoration,
round-trip inference, and ONNX export.

Run with:
    python tests/test_export.py
or:
    pytest tests/test_export.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch

from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.training_config import (
    TrainingConfig,
    SchedulerConfig,
    LossConfig,
    create_optimizer_and_scheduler,
)
from deployment.export_onnx import (
    export_to_onnx,
    validate_onnx,
    run_onnx_inference,
    compare_pytorch_onnx,
    HAS_ONNX,
    HAS_ORT,
    EXPECTED_PARAM_COUNT,
    FIXED_INFERENCE_T,
)

BEST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"
LAST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "last_checkpoint.pt"
ONNX_OUTPUT = PROJECT_ROOT / "deployment" / "exports" / "cnn_gru.onnx"

EXPECTED_PARAMS = 70_789

# ---------------------------------------------------------------------------
# Expected model key set
# ---------------------------------------------------------------------------

def get_expected_model_keys():
    model = LightweightCNNTGRUModel()
    return set(model.state_dict().keys())


# ===========================================================================
# TEST 1 — Best checkpoint exists and loads
# ===========================================================================

class TestBestCheckpointExistsAndLoads:
    def test_best_checkpoint_file_exists(self):
        assert BEST_CHECKPOINT.exists(), (
            f"best_checkpoint.pt not found: {BEST_CHECKPOINT}"
        )

    def test_best_checkpoint_loads_without_error(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        assert isinstance(ckpt, dict)

    def test_best_checkpoint_has_all_required_keys(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        required = {"epoch", "best_val_loss", "model_state_dict",
                    "optimizer_state_dict", "scheduler_state_dict", "config"}
        for key in required:
            assert key in ckpt, f"Missing checkpoint key: '{key}'"


# ===========================================================================
# TEST 2 — Last checkpoint loads if present
# ===========================================================================

class TestLastCheckpointLoads:
    def test_last_checkpoint_loads_if_present(self):
        if not LAST_CHECKPOINT.exists():
            pytest.skip("last_checkpoint.pt not found — skipping.")
        ckpt = torch.load(LAST_CHECKPOINT, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt


# ===========================================================================
# TEST 3 — Model parameter count remains 70,789
# ===========================================================================

class TestModelParamCountAfterLoad:
    def test_best_checkpoint_model_has_70789_params(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == EXPECTED_PARAMS

    def test_fresh_model_has_same_param_count(self):
        model = LightweightCNNTGRUModel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == EXPECTED_PARAMS


# ===========================================================================
# TEST 4 — Checkpoint model keys are complete and exact
# ===========================================================================

class TestModelStateDictKeyCompleteness:
    def test_no_missing_keys(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        expected_keys = get_expected_model_keys()
        ckpt_keys = set(ckpt["model_state_dict"].keys())
        missing = expected_keys - ckpt_keys
        assert len(missing) == 0, f"Missing keys in checkpoint: {missing}"

    def test_no_unexpected_keys(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        expected_keys = get_expected_model_keys()
        ckpt_keys = set(ckpt["model_state_dict"].keys())
        unexpected = ckpt_keys - expected_keys
        assert len(unexpected) == 0, f"Unexpected keys in checkpoint: {unexpected}"


# ===========================================================================
# TEST 5 — Checkpoint tensor shapes match current architecture
# ===========================================================================

class TestCheckpointTensorShapes:
    def test_all_tensor_shapes_match_architecture(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        arch_sd = model.state_dict()
        ckpt_sd = ckpt["model_state_dict"]

        for key in arch_sd:
            assert arch_sd[key].shape == ckpt_sd[key].shape, (
                f"Shape mismatch for '{key}': "
                f"arch={arch_sd[key].shape}, ckpt={ckpt_sd[key].shape}"
            )


# ===========================================================================
# TEST 6 — Optimizer state restoration works
# ===========================================================================

class TestOptimizerStateRestoration:
    def test_optimizer_state_can_be_restored(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])

        config = TrainingConfig.create(device="cpu")
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Verify optimizer groups are preserved
        assert len(optimizer.param_groups) > 0
        assert "lr" in optimizer.param_groups[0]

    def test_optimizer_step_not_called_during_restoration(self):
        """State restoration must not call optimizer.step()."""
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])

        params_before = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        config = TrainingConfig.create(device="cpu")
        optimizer, _ = create_optimizer_and_scheduler(model, config)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Parameters must be unchanged — step() was not called
        for name, param in model.named_parameters():
            assert torch.equal(param.detach(), params_before[name])


# ===========================================================================
# TEST 7 — Scheduler state restoration works
# ===========================================================================

class TestSchedulerStateRestoration:
    def test_scheduler_state_can_be_restored(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])

        config = TrainingConfig.create(device="cpu")
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Scheduler should have valid internal state
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_scheduler_step_not_called_during_restoration(self):
        """State restoration must not call scheduler.step()."""
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])

        config = TrainingConfig.create(device="cpu")
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # LR must not have changed — step() was not called
        lr_after = optimizer.param_groups[0]["lr"]
        assert lr_before == lr_after


# ===========================================================================
# TEST 8 — Stored configuration is consistent with authoritative Step 13 config
# ===========================================================================

class TestStoredConfigurationConsistency:
    AUTHORITATIVE_CONFIG = {
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "seed": 123,
        "mixed_precision": False,
    }

    def test_stored_config_matches_authoritative(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        stored = ckpt["config"]
        for key, expected_val in self.AUTHORITATIVE_CONFIG.items():
            assert key in stored, f"Missing config key: '{key}'"
            assert stored[key] == expected_val, (
                f"Config mismatch for '{key}': expected {expected_val}, got {stored[key]}"
            )

    def test_authoritative_python_config_matches(self):
        """Verify TrainingConfig defaults match the stored checkpoint config."""
        auth_config = TrainingConfig()
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        stored = ckpt["config"]

        assert auth_config.batch_size == stored["batch_size"]
        assert abs(auth_config.learning_rate - stored["learning_rate"]) < 1e-9
        assert abs(auth_config.weight_decay - stored["weight_decay"]) < 1e-9
        assert abs(auth_config.gradient_clip_norm - stored["gradient_clip_norm"]) < 1e-9
        assert auth_config.seed == stored["seed"]
        assert auth_config.mixed_precision == stored["mixed_precision"]


# ===========================================================================
# TEST 9 — Checkpoint round-trip inference is deterministic
# ===========================================================================

class TestRoundTripInferenceDeterministic:
    def test_round_trip_produces_identical_outputs(self):
        """
        Load → infer → save to temp → reload → infer again.
        Outputs must be numerically identical.
        """
        # Load original
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model1 = LightweightCNNTGRUModel()
        model1.load_state_dict(ckpt["model_state_dict"])
        model1.eval()

        # Fixed deterministic input
        torch.manual_seed(42)
        x = torch.randn(1, 2, 257, FIXED_INFERENCE_T, dtype=torch.float32)

        with torch.no_grad():
            enh1, logits1 = model1(x)

        # Save to temp location
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "round_trip.pt"
            torch.save(ckpt, temp_path)

            # Reload
            ckpt2 = torch.load(temp_path, map_location="cpu", weights_only=False)
            model2 = LightweightCNNTGRUModel()
            model2.load_state_dict(ckpt2["model_state_dict"])
            model2.eval()

            with torch.no_grad():
                enh2, logits2 = model2(x)

        assert torch.equal(enh1, enh2), "Enhanced STFT outputs differ after round-trip."
        assert torch.equal(logits1, logits2), "Classification logits differ after round-trip."


# ===========================================================================
# TEST 10 — Export file exists after successful export
# ===========================================================================

class TestExportFileExists:
    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_export_creates_onnx_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            result_path = export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )
            assert result_path.exists(), f"ONNX file not created at {result_path}"

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_permanent_onnx_export_exists_after_export(self):
        """Export to the canonical deployment path and verify it exists."""
        result_path = export_to_onnx(
            checkpoint_path=BEST_CHECKPOINT,
            output_path=ONNX_OUTPUT,
        )
        assert result_path.exists()


# ===========================================================================
# TEST 11 — Export file is non-empty
# ===========================================================================

class TestExportFileNonEmpty:
    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_export_file_is_non_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )
            size = out_path.stat().st_size
            assert size > 0, f"Export file is empty: {out_path}"
            assert size > 1000, f"Export file suspiciously small: {size} bytes"


# ===========================================================================
# TEST 12 — Exported model structure is valid if ONNX tooling is installed
# ===========================================================================

class TestONNXModelStructureValid:
    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_onnx_model_passes_checker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )
            val = validate_onnx(out_path)
            assert val["onnx_check_passed"] is True, (
                f"ONNX model failed checker: {val.get('onnx_check_error')}"
            )
            assert val["file_size_bytes"] > 0

    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_onnx_model_has_correct_input_output_names(self):
        import onnx as _onnx
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )
            proto = _onnx.load(str(out_path))
            input_names = [inp.name for inp in proto.graph.input]
            output_names = [out.name for out in proto.graph.output]
            assert "noisy_features" in input_names
            assert "enhanced_stft" in output_names
            assert "classification_logits" in output_names


# ===========================================================================
# TEST 13 — PyTorch vs exported inference are numerically close
# ===========================================================================

class TestPyTorchVsONNXClose:
    @pytest.mark.skipif(not HAS_ONNX or not HAS_ORT, reason="onnx/onnxruntime not installed")
    def test_pytorch_and_onnx_outputs_are_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )
            cmp = compare_pytorch_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                onnx_path=out_path,
                atol=1e-4,
                rtol=1e-4,
            )
            print(f"\n  max_abs_diff_enhanced : {cmp['max_abs_diff_enhanced']:.2e}")
            print(f"  max_abs_diff_logits   : {cmp['max_abs_diff_logits']:.2e}")
            assert cmp["allclose_enhanced"], (
                f"Enhanced output differs. max_abs_diff={cmp['max_abs_diff_enhanced']:.2e}, "
                f"atol={cmp['atol']}"
            )
            assert cmp["allclose_logits"], (
                f"Logits differ. max_abs_diff={cmp['max_abs_diff_logits']:.2e}, "
                f"atol={cmp['atol']}"
            )


# ===========================================================================
# TEST 14 — Export does not modify the PyTorch model parameters
# ===========================================================================

class TestExportDoesNotModifyModelParams:
    @pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
    def test_params_unchanged_after_export(self):
        ckpt = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model = LightweightCNNTGRUModel()
        model.load_state_dict(ckpt["model_state_dict"])

        params_before = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "test_export.onnx"
            # Export using the checkpoint (internal model, not external reference)
            export_to_onnx(
                checkpoint_path=BEST_CHECKPOINT,
                output_path=out_path,
            )

        # Re-load the checkpoint and compare
        ckpt_after = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        model_after = LightweightCNNTGRUModel()
        model_after.load_state_dict(ckpt_after["model_state_dict"])

        for name in params_before:
            p_before = params_before[name]
            p_after = dict(model_after.named_parameters())[name].detach()
            assert torch.equal(p_before, p_after), (
                f"Parameter '{name}' was modified by export."
            )


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
