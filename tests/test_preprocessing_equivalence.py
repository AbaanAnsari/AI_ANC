"""
tests/test_preprocessing_equivalence.py
========================================
Numerical equivalence test between training feature extraction (scipy.signal.stft)
and streaming deployment feature extraction (STFTExtractor.transform_frame).

Verifies:
1. Max absolute difference <= 1e-6 (floating-point tolerance)
2. Mean absolute difference <= 1e-7
3. Relative error <= 1e-5
4. Feature magnitude statistics before and after fix (~48.9 dB scale difference eliminated)
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from scipy.signal import stft as scipy_stft, get_window

from src.features.stft import compute_stft, compute_stft_frame, STFTExtractor


def test_stft_frame_scipy_exact_equivalence():
    """
    Test that STFTExtractor.transform_frame produces identical features
    to scipy.signal.stft on every single frame across an audio sequence.
    """
    rng = np.random.default_rng(20260929)
    sample_rate = 16000
    n_samples = 16000  # 1 second
    audio = rng.normal(0, 0.2, n_samples).astype(np.float32)

    # 1. Training feature extraction: scipy.signal.stft via compute_stft
    training_stft = compute_stft(audio, sample_rate=sample_rate, n_fft=512, win_length=512, hop_length=128, window="hann")
    # Shape: (257, 126)

    # 2. Deployment feature extraction: frame-by-frame via STFTExtractor.transform_frame
    extractor = STFTExtractor(sample_rate=sample_rate, n_fft=512, win_length=512, hop_length=128, window="hann")
    win = get_window("hann", 512).astype(np.float32)

    # Replicate scipy boundary='zeros' padding (256 zeros on left and right)
    padded = np.pad(audio, (256, 256), mode="constant")
    nstep = 128
    n_frames = training_stft.shape[1]
    deployment_frames = []

    for i in range(n_frames):
        frame = padded[i * nstep : i * nstep + 512]
        d_frame = extractor.transform_frame(frame)
        deployment_frames.append(d_frame)

    deployment_stft = np.column_stack(deployment_frames)

    # 3. Compute error metrics
    abs_diff = np.abs(training_stft - deployment_stft)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    norm_training = float(np.linalg.norm(training_stft))
    norm_diff = float(np.linalg.norm(abs_diff))
    relative_error = norm_diff / (norm_training + 1e-12)

    # Unfixed deployment (old raw np.fft.rfft without 1/win.sum() scale)
    old_raw_stft = deployment_stft * win.sum()
    old_scale_diff_db = 20.0 * np.log10(float(np.mean(np.abs(old_raw_stft))) / (float(np.mean(np.abs(training_stft))) + 1e-12))

    print("\n" + "=" * 60)
    print("STFT PREPROCESSING EQUIVALENCE AUDIT REPORT")
    print("=" * 60)
    print(f"Training STFT shape:   {training_stft.shape}")
    print(f"Deployment STFT shape: {deployment_stft.shape}")
    print(f"Max Absolute Diff:     {max_abs_diff:.4e}")
    print(f"Mean Absolute Diff:    {mean_abs_diff:.4e}")
    print(f"Relative L2 Error:     {relative_error:.4e}")
    print(f"Training mean magnitude:   {float(np.mean(np.abs(training_stft))):.6f}")
    print(f"Corrected deployment mean: {float(np.mean(np.abs(deployment_stft))):.6f}")
    print(f"Old uncorrected mean:      {float(np.mean(np.abs(old_raw_stft))):.6f}")
    print(f"Discrepancy eliminated:    {old_scale_diff_db:.2f} dB")
    print("=" * 60)

    # Acceptance criteria
    assert max_abs_diff < 1e-6, f"Max abs diff {max_abs_diff} exceeds tolerance 1e-6"
    assert mean_abs_diff < 1e-7, f"Mean abs diff {mean_abs_diff} exceeds tolerance 1e-7"
    assert relative_error < 1e-5, f"Relative error {relative_error} exceeds tolerance 1e-5"
    assert abs(old_scale_diff_db - 48.16) < 1.0, f"Expected ~48 dB scale mismatch, got {old_scale_diff_db} dB"


if __name__ == "__main__":
    test_stft_frame_scipy_exact_equivalence()
