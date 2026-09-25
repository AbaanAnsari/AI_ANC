"""
deployment/export_onnx.py — Phase 1 Step 17 ONNX Export
=========================================================
Exports the LightweightCNNTGRUModel to ONNX format for future inference.

Design rules:
  - Loads from checkpoint; never uses random weights.
  - Model architecture is NOT modified.
  - No quantization, no pruning.
  - No STM32-specific optimization.
  - Exports to: deployment/exports/cnn_gru.onnx
  - Does NOT overwrite the PyTorch checkpoint.

ONNX input shape:  (B, 2, 257, T)  — dynamic batch and T
ONNX outputs:
    enhanced_stft          (B, 2, 257, T)
    classification_logits  (B, 3)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Availability flags
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
    HAS_ORT = True
except ImportError:
    HAS_ONNX = False
    HAS_ORT = False

EXPORT_DIR = PROJECT_ROOT / "deployment" / "exports"
BEST_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoints" / "best_checkpoint.pt"
ONNX_PATH = EXPORT_DIR / "cnn_gru.onnx"

# Fixed T for export (used only if dynamic axes fail; dynamic is preferred)
FIXED_INFERENCE_T = 126
EXPECTED_PARAM_COUNT = 70_789


def export_to_onnx(
    checkpoint_path: str | Path = BEST_CHECKPOINT,
    output_path: str | Path = ONNX_PATH,
    opset_version: int = 14,
    fixed_T: int = FIXED_INFERENCE_T,
) -> Path:
    """
    Export the LightweightCNNTGRUModel to ONNX.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the PyTorch checkpoint.
    output_path : Path
        Destination ONNX file.
    opset_version : int
        ONNX opset version to use.
    fixed_T : int
        Fixed time dimension for the dummy input (dynamic axes still allowed).

    Returns
    -------
    Path
        Path to the exported ONNX file.

    Raises
    ------
    ImportError
        If onnx is not installed.
    FileNotFoundError
        If checkpoint does not exist.
    RuntimeError
        If model state_dict mismatch is detected.
    """
    if not HAS_ONNX:
        raise ImportError(
            "ONNX export requires 'onnx'. Install with: pip install onnx"
        )

    from src.models.cnn_gru import LightweightCNNTGRUModel

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint is missing 'model_state_dict'.")

    model = LightweightCNNTGRUModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    # Verify parameter count — architecture must be unchanged
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"Loaded model has {trainable} params; expected {EXPECTED_PARAM_COUNT}."
        )

    # Clone params to verify export doesn't mutate them
    params_before = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }

    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic dummy input: (1, 2, 257, T)
    dummy_input = torch.zeros(1, 2, 257, fixed_T, dtype=torch.float32)

    # Dynamic axes: batch and time
    dynamic_axes = {
        "noisy_features": {0: "batch_size", 3: "time_frames"},
        "enhanced_stft": {0: "batch_size", 3: "time_frames"},
        "classification_logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["noisy_features"],
        output_names=["enhanced_stft", "classification_logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        dynamo=False,  # use stable legacy TorchScript-based exporter
    )

    # Verify parameters unchanged after export
    params_unchanged = all(
        torch.equal(param.detach(), params_before[name])
        for name, param in model.named_parameters()
    )
    if not params_unchanged:
        raise RuntimeError("Model parameters were modified during ONNX export!")

    file_size = output_path.stat().st_size
    logger.info(
        "ONNX export complete: %s (%.1f KB)",
        output_path, file_size / 1024.0,
    )
    return output_path


def validate_onnx(
    onnx_path: str | Path = ONNX_PATH,
) -> dict:
    """
    Validate the exported ONNX model.

    Returns
    -------
    dict with:
        onnx_available     : bool
        ort_available      : bool
        file_exists        : bool
        file_size_bytes    : int or None
        onnx_check_passed  : bool or None
        onnx_ir_version    : int or None
        onnx_opset_version : int or None
    """
    onnx_path = Path(onnx_path)
    result = {
        "onnx_available": HAS_ONNX,
        "ort_available": HAS_ORT,
        "file_exists": onnx_path.exists(),
        "file_size_bytes": None,
        "onnx_check_passed": None,
        "onnx_ir_version": None,
        "onnx_opset_version": None,
    }

    if not onnx_path.exists():
        return result

    result["file_size_bytes"] = onnx_path.stat().st_size

    if HAS_ONNX:
        model_proto = onnx.load(str(onnx_path))
        try:
            onnx.checker.check_model(model_proto)
            result["onnx_check_passed"] = True
        except onnx.checker.ValidationError as e:
            result["onnx_check_passed"] = False
            result["onnx_check_error"] = str(e)

        result["onnx_ir_version"] = model_proto.ir_version
        if model_proto.opset_import:
            result["onnx_opset_version"] = model_proto.opset_import[0].version

    return result


def run_onnx_inference(
    onnx_path: str | Path,
    input_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run deterministic inference through the ONNX runtime.

    Parameters
    ----------
    onnx_path : Path
        Path to exported ONNX model.
    input_tensor : torch.Tensor
        Input of shape (B, 2, 257, T).

    Returns
    -------
    (enhanced_stft, classification_logits) as torch.Tensor
    """
    if not HAS_ORT:
        raise ImportError(
            "ONNX Runtime is required. Install with: pip install onnxruntime"
        )

    import numpy as np
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_np = input_tensor.cpu().numpy().astype(np.float32)
    outputs = session.run(
        ["enhanced_stft", "classification_logits"],
        {"noisy_features": input_np},
    )
    enhanced = torch.from_numpy(outputs[0])
    logits = torch.from_numpy(outputs[1])
    return enhanced, logits


def compare_pytorch_onnx(
    checkpoint_path: str | Path = BEST_CHECKPOINT,
    onnx_path: str | Path = ONNX_PATH,
    fixed_T: int = FIXED_INFERENCE_T,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict:
    """
    Compare PyTorch model output against ONNX runtime output.

    Uses a fixed deterministic input tensor.

    Returns
    -------
    dict with:
        pytorch_output_shape  : tuple
        onnx_output_shape     : tuple
        max_abs_diff_enhanced : float
        max_abs_diff_logits   : float
        allclose_enhanced     : bool
        allclose_logits       : bool
        atol                  : float
        rtol                  : float
    """
    from src.models.cnn_gru import LightweightCNNTGRUModel

    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path)

    # Load PyTorch model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LightweightCNNTGRUModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Fixed deterministic input
    torch.manual_seed(42)
    dummy = torch.randn(1, 2, 257, fixed_T, dtype=torch.float32)

    with torch.no_grad():
        pt_enhanced, pt_logits = model(dummy)

    ort_enhanced, ort_logits = run_onnx_inference(onnx_path, dummy)

    max_diff_enh = float(torch.max(torch.abs(pt_enhanced - ort_enhanced)))
    max_diff_logits = float(torch.max(torch.abs(pt_logits - ort_logits)))

    return {
        "pytorch_output_shape": tuple(pt_enhanced.shape),
        "onnx_output_shape": tuple(ort_enhanced.shape),
        "max_abs_diff_enhanced": max_diff_enh,
        "max_abs_diff_logits": max_diff_logits,
        "allclose_enhanced": bool(torch.allclose(pt_enhanced, ort_enhanced, atol=atol, rtol=rtol)),
        "allclose_logits": bool(torch.allclose(pt_logits, ort_logits, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Exporting model to ONNX...")
    path = export_to_onnx()
    print(f"Export path: {path}")

    print("\nValidating ONNX model...")
    val = validate_onnx(path)
    for k, v in val.items():
        print(f"  {k}: {v}")

    print("\nComparing PyTorch vs ONNX outputs...")
    cmp = compare_pytorch_onnx()
    for k, v in cmp.items():
        print(f"  {k}: {v}")
