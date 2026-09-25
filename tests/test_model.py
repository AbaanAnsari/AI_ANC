import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.models.cnn_gru import LightweightCNNTGRUModel
from src.models.model_utils import count_parameters, get_model_memory_bytes


def test_model_creation_and_parameter_budget():
    model = LightweightCNNTGRUModel()

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    memory_kb = round(get_model_memory_bytes(model) / 1024.0, 2)

    # HARD CONSTRAINT ASSERTION
    assert trainable_params < 100000, (
        f"HARD CONSTRAINT VIOLATED: Trainable parameters {trainable_params} >= 100,000"
    )

    assert total_params == trainable_params
    assert memory_kb < 500.0  # FP32 footprint under 500 KB

    summary = model.parameter_summary()
    assert summary["trainable_parameters"] < 100000


def test_forward_pass_batch_size_1():
    model = LightweightCNNTGRUModel()
    model.eval()

    # Input shape: (B=1, 2, 257, T=126)
    x = torch.randn(1, 2, 257, 126, dtype=torch.float32)

    with torch.no_grad():
        enhanced_stft, logits = model(x)

    assert enhanced_stft.shape == (1, 2, 257, 126)
    assert logits.shape == (1, 3)

    assert enhanced_stft.dtype == torch.float32
    assert logits.dtype == torch.float32

    assert torch.all(torch.isfinite(enhanced_stft))
    assert torch.all(torch.isfinite(logits))


def test_forward_pass_batch_size_4():
    model = LightweightCNNTGRUModel()
    model.eval()

    # Input shape: (B=4, 2, 257, T=126)
    x = torch.randn(4, 2, 257, 126, dtype=torch.float32)

    with torch.no_grad():
        enhanced_stft, logits = model(x)

    assert enhanced_stft.shape == (4, 2, 257, 126)
    assert logits.shape == (4, 3)

    assert torch.all(torch.isfinite(enhanced_stft))
    assert torch.all(torch.isfinite(logits))


def test_deterministic_inference():
    model = LightweightCNNTGRUModel()
    model.eval()

    x = torch.randn(2, 2, 257, 126, dtype=torch.float32)

    with torch.no_grad():
        out1_stft, out1_logits = model(x)
        out2_stft, out2_logits = model(x)

    assert torch.equal(out1_stft, out2_stft)
    assert torch.equal(out1_logits, out2_logits)


def test_no_accidental_parameter_growth():
    model = LightweightCNNTGRUModel()
    params_before = count_parameters(model, trainable_only=True)

    x = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    _ = model(x)

    params_after = count_parameters(model, trainable_only=True)
    assert params_before == params_after


def test_gradient_flow():
    model = LightweightCNNTGRUModel()
    model.train()

    x = torch.randn(2, 2, 257, 126, dtype=torch.float32)
    enhanced_stft, logits = model(x)

    # Dummy loss combining enhancement and classification outputs
    loss = enhanced_stft.pow(2).sum() + logits.pow(2).sum()
    loss.backward()

    # Verify every trainable parameter received a non-zero gradient
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} did not receive gradient"
            assert not torch.all(param.grad == 0), f"Parameter {name} gradient is zero"
            assert torch.all(torch.isfinite(param.grad)), f"Parameter {name} gradient is non-finite"
            grad_count += 1

    assert grad_count > 0


def test_cpu_forward_execution():
    model = LightweightCNNTGRUModel().to("cpu")
    model.eval()

    x = torch.randn(1, 2, 257, 126, device="cpu", dtype=torch.float32)

    with torch.no_grad():
        enhanced_stft, logits = model(x)

    assert enhanced_stft.device == torch.device("cpu")
    assert logits.device == torch.device("cpu")
    assert torch.all(torch.isfinite(enhanced_stft))


def test_layer_by_layer_parameter_audit():
    model = LightweightCNNTGRUModel()
    layer_breakdown = {}
    largest_layer = ("", 0)

    for name, param in model.named_parameters():
        count = param.numel()
        layer_breakdown[name] = count
        if count > largest_layer[1]:
            largest_layer = (name, count)

    total_params = sum(layer_breakdown.values())
    trainable_params = count_parameters(model, trainable_only=True)

    # Hard parameter check
    assert total_params == 70789, f"Expected 70,789 parameters, got {total_params}"
    assert trainable_params == 70789
    assert trainable_params < 100000

    # Verify largest layer
    assert largest_layer[0] == "enhancement_head.proj.weight"
    assert largest_layer[1] == 24672  # Weight matrix of Linear(48, 514)


def test_variable_time_dimension_preservation():
    model = LightweightCNNTGRUModel()
    model.eval()

    test_configs = [
        (1, 126),
        (4, 126),
        (2, 64),
        (2, 200),
    ]

    for batch_size, time_frames in test_configs:
        x = torch.randn(batch_size, 2, 257, time_frames, dtype=torch.float32)

        with torch.no_grad():
            enhanced_stft, logits = model(x)

        # Output time dimension MUST equal input time dimension
        assert enhanced_stft.shape == (batch_size, 2, 257, time_frames)
        assert logits.shape == (batch_size, 3)
        assert torch.all(torch.isfinite(enhanced_stft))
        assert torch.all(torch.isfinite(logits))


def test_frequency_preservation_and_channels():
    model = LightweightCNNTGRUModel()
    model.eval()

    x = torch.randn(2, 2, 257, 126, dtype=torch.float32)

    with torch.no_grad():
        enhanced_stft, logits = model(x)

    # 257 frequency bins & 2 output components (channel 0 = Real, channel 1 = Imag)
    assert enhanced_stft.shape[1] == 2
    assert enhanced_stft.shape[2] == 257


if __name__ == "__main__":
    test_model_creation_and_parameter_budget()
    test_forward_pass_batch_size_1()
    test_forward_pass_batch_size_4()
    test_deterministic_inference()
    test_no_accidental_parameter_growth()
    test_gradient_flow()
    test_cpu_forward_execution()
    test_layer_by_layer_parameter_audit()
    test_variable_time_dimension_preservation()
    test_frequency_preservation_and_channels()

    model = LightweightCNNTGRUModel()
    summary = model.parameter_summary()

    print("=" * 64)
    print("PHASE 1 STEP 11 — MODEL IMPLEMENTATION AUDIT")
    print("=" * 64)
    print(f"Total Parameters          : {summary['total_parameters']:,}")
    print(f"Trainable Parameters      : {summary['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters  : 0")
    print(f"Hard Constraint (<100,000): SATISFIED")
    print(f"FP32 Parameter Memory     : {summary['fp32_memory_bytes']:,} bytes ({summary['fp32_memory_kb']} KB)")
    print(f"Largest Layer             : enhancement_head.proj.weight (24,672 parameters)")
    print("=" * 64)
    print("cnn_gru audit model tests: PASS")
