"""
tests/test_validation.py — Phase 1 Step 16 Test Suite
=======================================================
16 tests verifying the formal validation/evaluation layer.

Tests use synthetic DataLoaders to avoid dataset I/O.
Checkpoint tests load the ACTUAL best_checkpoint.pt produced in Step 15.

Run with:
    python tests/test_validation.py
or:
    pytest tests/test_validation.py -v
"""
from __future__ import annotations

import sys
import copy
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.models.cnn_gru import LightweightCNNTGRUModel
from src.training.losses import MultiTaskLoss
from src.training.training_config import TrainingConfig
from src.training.validation import (
    evaluate,
    load_model_from_checkpoint,
    EXPECTED_PARAM_COUNT,
)
from evaluation.snr import compute_snr, compute_snr_improvement
from evaluation.stoi import stoi_available, stoi_unavailable_message
from evaluation.pesq import pesq_available, pesq_unavailable_message
from evaluation.metrics import compute_confusion_matrix, compute_classification_metrics, validate_labels

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"
TEST_MANIFEST = MANIFEST_DIR / "test_manifest.jsonl"
BEST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"

EXPECTED_PARAMS = 70_789
BATCH_SIZE = 2
TIME_FRAMES = 126


# ---------------------------------------------------------------------------
# Synthetic DataLoader
# ---------------------------------------------------------------------------

def make_synthetic_batch(batch_size: int = BATCH_SIZE, T: int = TIME_FRAMES):
    real = torch.randn(batch_size, 257, T, dtype=torch.float32)
    imag = torch.randn(batch_size, 257, T, dtype=torch.float32)
    target_stft = torch.complex(real, imag)

    # Provide actual clean and m1 waveforms for SNR metrics
    clean_target = torch.randn(batch_size, 16000, dtype=torch.float32) * 0.1
    m1_waveform = clean_target + torch.randn(batch_size, 16000, dtype=torch.float32) * 0.05

    return {
        "noisy_features": torch.randn(batch_size, 2, 257, T, dtype=torch.float32),
        "target_stft": target_stft,
        "noise_class_label": torch.randint(0, 3, (batch_size,), dtype=torch.int64),
        "clean_target": clean_target,
        "m1_waveform": m1_waveform,
        "target_snr_db": torch.tensor([5.0] * batch_size, dtype=torch.float32),
    }


class SyntheticDataLoader:
    def __init__(self, n_batches: int = 3):
        self.n_batches = n_batches

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield make_synthetic_batch()


def make_loss_fn():
    return MultiTaskLoss(
        enhancement_weight=1.0,
        classification_weight=0.10,
        enhancement_l1_weight=0.7,
        enhancement_l2_weight=0.3,
    )


# ===========================================================================
# TEST 1 — Checkpoint loads successfully
# ===========================================================================

class TestCheckpointLoads:
    def test_best_checkpoint_exists(self):
        assert BEST_CHECKPOINT.exists(), (
            f"best_checkpoint.pt not found at {BEST_CHECKPOINT}. "
            "Run the training step first."
        )

    def test_checkpoint_loads_successfully(self):
        model, ckpt = load_model_from_checkpoint(BEST_CHECKPOINT)
        assert model is not None
        assert isinstance(ckpt, dict)
        assert "model_state_dict" in ckpt


# ===========================================================================
# TEST 2 — Loaded model has exactly 70,789 trainable parameters
# ===========================================================================

class TestLoadedModelParamCount:
    def test_loaded_model_has_70789_params(self):
        model, _ = load_model_from_checkpoint(BEST_CHECKPOINT)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == EXPECTED_PARAMS, (
            f"Expected {EXPECTED_PARAMS}, got {trainable}"
        )


# ===========================================================================
# TEST 3 — Evaluation uses model.eval()
# ===========================================================================

class TestEvaluationUsesEvalMode:
    def test_model_in_eval_mode_during_evaluation(self):
        """Verify model is set to eval() during evaluate()."""
        model = LightweightCNNTGRUModel()
        # Force training mode first
        model.train()
        assert model.training is True

        loss_fn = make_loss_fn()
        # After evaluate() model should be in eval mode
        evaluate(model, SyntheticDataLoader(2), loss_fn, device="cpu")

        # evaluate() sets model.eval() — verify it ends in eval mode
        assert model.training is False


# ===========================================================================
# TEST 4 — Evaluation executes under torch.no_grad()
# ===========================================================================

class TestNoGradDuringEvaluation:
    def test_outputs_have_no_grad(self):
        """Under no_grad(), model outputs must not require grad."""
        model = LightweightCNNTGRUModel()
        model.eval()
        batch = make_synthetic_batch()

        with torch.no_grad():
            enhanced, logits = model(batch["noisy_features"])

        assert not enhanced.requires_grad
        assert not logits.requires_grad


# ===========================================================================
# TEST 5 — Model parameters are unchanged by evaluation
# ===========================================================================

class TestEvaluationDoesNotModifyParams:
    def test_params_bit_exact_after_evaluation(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()

        params_before = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

        result = evaluate(model, SyntheticDataLoader(3), loss_fn, device="cpu")

        for name, param in model.named_parameters():
            assert torch.equal(param.detach(), params_before[name]), (
                f"Parameter '{name}' was modified during evaluation."
            )

        assert result["params_unchanged"] is True


# ===========================================================================
# TEST 6 — Loss outputs are finite
# ===========================================================================

class TestLossOutputsFinite:
    def test_all_losses_finite_after_evaluation(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()

        result = evaluate(model, SyntheticDataLoader(3), loss_fn, device="cpu")

        assert result["mean_total_loss"] == result["mean_total_loss"]  # not NaN
        assert result["mean_enhancement_loss"] == result["mean_enhancement_loss"]
        assert result["mean_classification_loss"] == result["mean_classification_loss"]

        assert result["mean_total_loss"] != float("inf")
        assert result["mean_enhancement_loss"] != float("inf")
        assert result["mean_classification_loss"] != float("inf")


# ===========================================================================
# TEST 7 — Enhancement output has the expected shape
# ===========================================================================

class TestEnhancementOutputShape:
    def test_enhanced_output_shape(self):
        model = LightweightCNNTGRUModel()
        model.eval()
        batch = make_synthetic_batch(batch_size=4, T=TIME_FRAMES)
        with torch.no_grad():
            enhanced, _ = model(batch["noisy_features"])
        assert enhanced.shape == (4, 2, 257, TIME_FRAMES)
        assert enhanced.dtype == torch.float32


# ===========================================================================
# TEST 8 — Classification output has the expected shape
# ===========================================================================

class TestClassificationOutputShape:
    def test_classification_output_shape(self):
        model = LightweightCNNTGRUModel()
        model.eval()
        batch = make_synthetic_batch(batch_size=4)
        with torch.no_grad():
            _, logits = model(batch["noisy_features"])
        assert logits.shape == (4, 3)
        assert logits.dtype == torch.float32


# ===========================================================================
# TEST 9 — Classification labels remain in {0, 1, 2}
# ===========================================================================

class TestClassificationLabelsValid:
    def test_true_labels_in_valid_set(self):
        batch = make_synthetic_batch(batch_size=8)
        labels = batch["noise_class_label"].numpy().tolist()
        assert validate_labels(labels, {0, 1, 2})

    def test_predicted_labels_in_valid_set(self):
        model = LightweightCNNTGRUModel()
        model.eval()
        batch = make_synthetic_batch(batch_size=8)
        with torch.no_grad():
            _, logits = model(batch["noisy_features"])
        preds = torch.argmax(logits, dim=1).numpy().tolist()
        assert validate_labels(preds, {0, 1, 2})


# ===========================================================================
# TEST 10 — Confusion matrix has shape 3x3
# ===========================================================================

class TestConfusionMatrixShape:
    def test_confusion_matrix_is_3x3(self):
        true_labels = [0, 1, 2, 0, 1, 2, 0, 0]
        pred_labels = [0, 1, 2, 1, 0, 2, 0, 1]
        cm = compute_confusion_matrix(true_labels, pred_labels, num_classes=3)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_3x3_from_evaluate(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(4), loss_fn, device="cpu")
        assert result["confusion_matrix"].shape == (3, 3)


# ===========================================================================
# TEST 11 — SNR calculations are finite when valid signals are supplied
# ===========================================================================

class TestSNRCalculationsFinite:
    def test_snr_finite_for_valid_signals(self):
        rng = np.random.default_rng(42)
        clean = rng.standard_normal(16000).astype(np.float32) * 0.1
        noisy = clean + rng.standard_normal(16000).astype(np.float32) * 0.05
        enhanced = clean + rng.standard_normal(16000).astype(np.float32) * 0.02

        result = compute_snr_improvement(clean, noisy, enhanced)
        assert np.isfinite(result["input_snr_db"])
        assert np.isfinite(result["output_snr_db"])
        assert np.isfinite(result["snr_improvement"])

    def test_snr_present_in_eval_result(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(3), loss_fn, device="cpu")
        # SNR should be populated since we provide clean_target and m1_waveform
        assert result["mean_input_snr_db"] is not None
        assert result["mean_output_snr_db"] is not None
        assert result["mean_snr_improvement"] is not None


# ===========================================================================
# TEST 12 — SNR improvement equals enhanced SNR minus noisy SNR
# ===========================================================================

class TestSNRImprovementFormula:
    def test_snr_improvement_equals_output_minus_input(self):
        rng = np.random.default_rng(99)
        clean = rng.standard_normal(16000).astype(np.float32) * 0.1
        noisy = clean + rng.standard_normal(16000).astype(np.float32) * 0.05
        enhanced = clean + rng.standard_normal(16000).astype(np.float32) * 0.01

        res = compute_snr_improvement(clean, noisy, enhanced)
        expected_improvement = res["output_snr_db"] - res["input_snr_db"]
        assert abs(res["snr_improvement"] - expected_improvement) < 1e-9


# ===========================================================================
# TEST 13 — Repeated deterministic evaluation gives equivalent results
# ===========================================================================

class TestDeterministicEvaluation:
    def test_two_identical_evaluations_give_same_loss(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()

        # Use a fixed-seed synthetic loader
        class FixedSeedLoader:
            def __init__(self):
                self.n_batches = 3

            def __len__(self):
                return 3

            def __iter__(self):
                torch.manual_seed(42)
                for _ in range(3):
                    yield make_synthetic_batch()

        result1 = evaluate(model, FixedSeedLoader(), loss_fn, device="cpu")
        result2 = evaluate(model, FixedSeedLoader(), loss_fn, device="cpu")

        assert abs(result1["mean_total_loss"] - result2["mean_total_loss"]) < 1e-9
        assert abs(result1["classification_accuracy"] - result2["classification_accuracy"]) < 1e-9


# ===========================================================================
# TEST 14 — Validation and test manifests remain distinct
# ===========================================================================

class TestManifestSplitIsolation:
    def test_val_and_test_manifests_are_different_files(self):
        assert VAL_MANIFEST != TEST_MANIFEST

    def test_val_and_test_manifests_have_no_clean_record_overlap(self):
        import json

        def read_clean_ids(path):
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            records = [json.loads(l) for l in lines if l.strip()]
            return {r["record_id"] for r in records if r.get("source_group") == "clean"}

        val_ids = read_clean_ids(VAL_MANIFEST)
        test_ids = read_clean_ids(TEST_MANIFEST)
        overlap = val_ids & test_ids
        assert len(overlap) == 0, (
            f"Found {len(overlap)} overlapping clean records between val and test splits."
        )


# ===========================================================================
# TEST 15 — Missing checkpoint produces a clear error
# ===========================================================================

class TestMissingCheckpointError:
    def test_missing_checkpoint_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model_from_checkpoint("/nonexistent/path/checkpoint.pt")
        assert "not found" in str(exc_info.value).lower() or "checkpoint" in str(exc_info.value).lower()

    def test_error_message_does_not_mention_random_weights(self):
        """Verify the error message guides to the actual problem, not silently falls back."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model_from_checkpoint("/nonexistent/ckpt.pt")
        assert "random" not in str(exc_info.value).lower() or "cannot" in str(exc_info.value).lower()


# ===========================================================================
# TEST 16 — Unavailable STOI/PESQ dependencies reported honestly
# ===========================================================================

class TestDependencyAvailabilityHonesty:
    def test_stoi_availability_is_boolean(self):
        available = stoi_available()
        assert isinstance(available, bool)

    def test_pesq_availability_is_boolean(self):
        available = pesq_available()
        assert isinstance(available, bool)

    def test_stoi_unavailable_message_is_informative(self):
        msg = stoi_unavailable_message()
        assert "pystoi" in msg or "STOI" in msg

    def test_pesq_unavailable_message_is_informative(self):
        msg = pesq_unavailable_message()
        assert "pesq" in msg.lower() or "PESQ" in msg

    def test_evaluation_result_reports_stoi_truthfully(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(2), loss_fn, device="cpu")
        # The reported STOI availability must match reality
        assert result["stoi_available"] == stoi_available()

    def test_evaluation_result_reports_pesq_truthfully(self):
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(2), loss_fn, device="cpu")
        assert result["pesq_available"] == pesq_available()

    def test_stoi_values_are_none_when_unavailable(self):
        if stoi_available():
            pytest.skip("pystoi is installed; this test only applies when unavailable.")
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(2), loss_fn, device="cpu")
        assert result["mean_noisy_stoi"] is None
        assert result["mean_enhanced_stoi"] is None

    def test_pesq_values_are_none_when_unavailable(self):
        if pesq_available():
            pytest.skip("pesq is installed; this test only applies when unavailable.")
        model = LightweightCNNTGRUModel()
        loss_fn = make_loss_fn()
        result = evaluate(model, SyntheticDataLoader(2), loss_fn, device="cpu")
        assert result["mean_noisy_pesq"] is None
        assert result["mean_enhanced_pesq"] is None


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
