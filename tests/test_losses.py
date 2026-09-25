import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.models.cnn_gru import LightweightCNNTGRUModel
from src.models.model_utils import count_parameters
from src.training.losses import (
    ClassificationLoss,
    ComplexEnhancementLoss,
    MultiTaskLoss,
    convert_complex_stft_to_real_imag,
)


def test_enhancement_loss_shape():
    pred = torch.randn(4, 2, 257, 126, dtype=torch.float32)
    target = torch.randn(4, 257, 126, dtype=torch.complex64)

    loss_fn = ComplexEnhancementLoss()
    loss = loss_fn(pred, target)

    assert loss.ndim == 0  # Scalar tensor
    assert loss.dtype == torch.float32
    assert torch.all(torch.isfinite(loss))


def test_enhancement_target_conversion():
    real_part = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    imag_part = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32)
    target_stft = torch.complex(real_part, imag_part)  # (1, 2, 2) complex64

    converted = convert_complex_stft_to_real_imag(target_stft)

    assert converted.shape == (1, 2, 2, 2)
    assert converted.dtype == torch.float32
    assert torch.equal(converted[:, 0, :, :], real_part)
    assert torch.equal(converted[:, 1, :, :], imag_part)


def test_enhancement_loss_zero_case():
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    target_real_imag = convert_complex_stft_to_real_imag(target_stft)

    loss_fn = ComplexEnhancementLoss()
    loss = loss_fn(target_real_imag, target_stft)

    assert torch.isclose(loss, torch.tensor(0.0, dtype=torch.float32), atol=1e-6)


def test_enhancement_loss_increases_with_error():
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    target_real_imag = convert_complex_stft_to_real_imag(target_stft)

    pred_perfect = target_real_imag.clone()
    pred_noisy = target_real_imag + 0.5 * torch.randn_like(target_real_imag)

    loss_fn = ComplexEnhancementLoss()
    loss_perfect = loss_fn(pred_perfect, target_stft)
    loss_noisy = loss_fn(pred_noisy, target_stft)

    assert loss_noisy > loss_perfect


def test_real_and_imaginary_components_both_matter():
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    target_real_imag = convert_complex_stft_to_real_imag(target_stft)

    loss_fn = ComplexEnhancementLoss()

    # Case A: Only Real component wrong
    case_a = target_real_imag.clone()
    case_a[:, 0, :, :] += 1.0

    # Case B: Only Imaginary component wrong
    case_b = target_real_imag.clone()
    case_b[:, 1, :, :] += 1.0

    # Case C: Both correct
    case_c = target_real_imag.clone()

    loss_a = loss_fn(case_a, target_stft)
    loss_b = loss_fn(case_b, target_stft)
    loss_c = loss_fn(case_c, target_stft)

    assert loss_a > 0.0, f"Expected loss_a > 0, got {loss_a}"
    assert loss_b > 0.0, f"Expected loss_b > 0, got {loss_b}"
    assert torch.isclose(loss_c, torch.tensor(0.0, dtype=torch.float32), atol=1e-6)


def test_classification_loss():
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    loss_fn = ClassificationLoss()
    loss = loss_fn(logits, labels)

    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert torch.all(torch.isfinite(loss))

    # Verify changing logits changes loss
    wrong_logits = torch.tensor([[0.1, 0.5, 3.0], [3.0, 0.1, 0.2]], dtype=torch.float32)
    wrong_loss = loss_fn(wrong_logits, labels)
    assert wrong_loss > loss


def test_classification_perfect_confidence_behavior():
    logits = torch.tensor([[100.0, -100.0, -100.0]], dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.int64)

    loss_fn = ClassificationLoss()
    loss = loss_fn(logits, labels)

    assert loss < 1e-4
    assert torch.all(torch.isfinite(loss))


def test_combined_loss_formula():
    loss_fn = MultiTaskLoss(
        enhancement_weight=1.0,
        classification_weight=0.10,
        enhancement_l1_weight=0.7,
        enhancement_l2_weight=0.3,
    )

    pred_stft = torch.randn(4, 2, 257, 126, dtype=torch.float32)
    logits = torch.randn(4, 3, dtype=torch.float32)
    target_stft = torch.randn(4, 257, 126, dtype=torch.complex64)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.int64)

    loss_dict = loss_fn(pred_stft, logits, target_stft, labels)

    total = loss_dict["total_loss"]
    enh = loss_dict["enhancement_loss"]
    cls = loss_dict["classification_loss"]

    expected_total = 1.0 * enh + 0.10 * cls
    assert torch.isclose(total, expected_total, atol=1e-6)


def test_configurable_loss_weights():
    pred_stft = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    logits = torch.randn(2, 3, dtype=torch.float32)
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    loss_fn1 = MultiTaskLoss(enhancement_weight=1.0, classification_weight=0.10)
    loss_fn2 = MultiTaskLoss(enhancement_weight=2.0, classification_weight=0.10)
    loss_fn3 = MultiTaskLoss(enhancement_weight=1.0, classification_weight=0.50)

    res1 = loss_fn1(pred_stft, logits, target_stft, labels)
    res2 = loss_fn2(pred_stft, logits, target_stft, labels)
    res3 = loss_fn3(pred_stft, logits, target_stft, labels)

    assert res2["total_loss"] > res1["total_loss"]
    assert res3["total_loss"] > res1["total_loss"]


def test_variable_temporal_dimension():
    loss_fn = MultiTaskLoss()

    for time_frames in (64, 126, 200):
        pred_stft = torch.randn(2, 2, 257, time_frames, dtype=torch.float32)
        logits = torch.randn(2, 3, dtype=torch.float32)
        target_stft = torch.randn(2, 257, time_frames, dtype=torch.complex64)
        labels = torch.tensor([0, 1], dtype=torch.int64)

        res = loss_fn(pred_stft, logits, target_stft, labels)
        assert torch.all(torch.isfinite(res["total_loss"]))


def test_batch_size_variation():
    loss_fn = MultiTaskLoss()

    for batch_size in (1, 2, 4):
        pred_stft = torch.randn(batch_size, 2, 257, 126, dtype=torch.float32)
        logits = torch.randn(batch_size, 3, dtype=torch.float32)
        target_stft = torch.randn(batch_size, 257, 126, dtype=torch.complex64)
        labels = torch.zeros(batch_size, dtype=torch.int64)

        res = loss_fn(pred_stft, logits, target_stft, labels)
        assert torch.all(torch.isfinite(res["total_loss"]))


def test_gradient_flow_with_actual_model():
    model = LightweightCNNTGRUModel()
    model.train()

    params_before = count_parameters(model, trainable_only=True)
    assert params_before == 70789

    loss_fn = MultiTaskLoss(
        enhancement_weight=1.0,
        classification_weight=0.10,
    )

    noisy_features = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    enhanced_output, classification_logits = model(noisy_features)
    loss_dict = loss_fn(enhanced_output, classification_logits, target_stft, labels)

    total_loss = loss_dict["total_loss"]
    assert torch.all(torch.isfinite(total_loss))

    total_loss.backward()

    # Verify parameters did not change
    params_after = count_parameters(model, trainable_only=True)
    assert params_before == params_after == 70789

    # Verify both branches received non-zero finite gradients
    enh_grad_found = False
    cls_grad_found = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} grad is None"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} grad non-finite"

            if "enhancement_head" in name and not torch.all(param.grad == 0):
                enh_grad_found = True
            if "classification_head" in name and not torch.all(param.grad == 0):
                cls_grad_found = True

    assert enh_grad_found, "Enhancement branch did not receive non-zero gradients"
    assert cls_grad_found, "Classification branch did not receive non-zero gradients"


def test_no_in_place_modification():
    pred_stft = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    logits = torch.randn(2, 3, dtype=torch.float32)
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    pred_clone = pred_stft.clone()
    logits_clone = logits.clone()
    target_clone = target_stft.clone()

    loss_fn = MultiTaskLoss()
    loss_fn(pred_stft, logits, target_stft, labels)

    assert torch.equal(pred_stft, pred_clone)
    assert torch.equal(logits, logits_clone)
    assert torch.equal(target_stft, target_clone)


def test_determinism():
    pred_stft = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    logits = torch.randn(2, 3, dtype=torch.float32)
    target_stft = torch.randn(2, 257, 126, dtype=torch.complex64)
    labels = torch.tensor([0, 1], dtype=torch.int64)

    loss_fn = MultiTaskLoss()
    res1 = loss_fn(pred_stft, logits, target_stft, labels)
    res2 = loss_fn(pred_stft, logits, target_stft, labels)

    assert torch.equal(res1["total_loss"], res2["total_loss"])
    assert torch.equal(res1["enhancement_loss"], res2["enhancement_loss"])
    assert torch.equal(res1["classification_loss"], res2["classification_loss"])


def test_cpu_compatibility():
    model = LightweightCNNTGRUModel().to("cpu")
    model.train()

    loss_fn = MultiTaskLoss().to("cpu")

    noisy_features = torch.randn(2, 2, 257, 126, device="cpu", dtype=torch.float32)
    target_stft = torch.randn(2, 257, 126, device="cpu", dtype=torch.complex64)
    labels = torch.tensor([0, 1], device="cpu", dtype=torch.int64)

    enhanced_output, classification_logits = model(noisy_features)
    loss_dict = loss_fn(enhanced_output, classification_logits, target_stft, labels)

    total_loss = loss_dict["total_loss"]
    total_loss.backward()

    assert total_loss.device == torch.device("cpu")
    assert torch.all(torch.isfinite(total_loss))


if __name__ == "__main__":
    test_enhancement_loss_shape()
    test_enhancement_target_conversion()
    test_enhancement_loss_zero_case()
    test_enhancement_loss_increases_with_error()
    test_real_and_imaginary_components_both_matter()
    test_classification_loss()
    test_classification_perfect_confidence_behavior()
    test_combined_loss_formula()
    test_configurable_loss_weights()
    test_variable_temporal_dimension()
    test_batch_size_variation()
    test_gradient_flow_with_actual_model()
    test_no_in_place_modification()
    test_determinism()
    test_cpu_compatibility()
    print("loss tests: PASS")
