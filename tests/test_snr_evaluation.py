"""
tests/test_snr_evaluation.py
=============================
Regression tests for robust SNR evaluation:
- Explicit zero/nearly-zero residual handling
- Never reporting artificial finite ~120 dB for machine precision limits
- Preserving standard SNR computation on physical nonzero residuals
"""
import math
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.snr import compute_snr, compute_snr_improvement


def test_perfect_clean_signal_returns_inf():
    clean = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
    # Exact duplicate
    snr = compute_snr(clean, clean)
    assert math.isinf(snr)
    assert snr > 0


def test_near_zero_float32_residual_returns_inf():
    clean = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
    # Tiny float32 quantization / roundoff error (< -100 dB)
    noisy_near_zero = clean + np.random.normal(0, 1e-8, size=len(clean)).astype(np.float32)
    snr = compute_snr(clean, noisy_near_zero)
    assert math.isinf(snr)


def test_finite_residual_returns_accurate_snr():
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1.0, 16000).astype(np.float32)
    # Add noise at exactly 10 dB SNR
    noise_10db = rng.normal(0, np.sqrt(0.1), 16000).astype(np.float32)
    deg_10db = clean + noise_10db
    snr = compute_snr(clean, deg_10db)
    assert not math.isinf(snr)
    assert snr == pytest.approx(10.0, abs=0.2)


def test_snr_improvement_with_oracle_inf():
    rng = np.random.default_rng(42)
    clean = rng.normal(0, 1.0, 16000).astype(np.float32)
    noisy = clean + rng.normal(0, 1.0, 16000).astype(np.float32)  # ~0 dB
    res = compute_snr_improvement(clean, noisy, clean)
    assert res["input_snr_db"] == pytest.approx(0.0, abs=0.2)
    assert math.isinf(res["output_snr_db"])
    assert math.isinf(res["snr_improvement"])
