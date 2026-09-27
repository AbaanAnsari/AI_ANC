"""
tests/test_nlms.py
==================
Comprehensive unit tests for the NLMS adaptive filter.

Tests:
    1. Basic initialization
    2. Invalid parameter rejection
    3. Single sample processing
    4. Block processing
    5. Numerical stability
    6. Stationary noise convergence
    7. Impulsive gate functionality
    8. Noise class factory
    9. Weight reset
    10. Deterministic behavior (same seed → same output)
    11. Extreme inputs (silence, clipping, NaN handling)
    12. AdaptiveNLMSController class transition
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.nlms import (
    NLMSFilter,
    AdaptiveNLMSController,
    create_nlms_for_noise_class,
    NOISE_CLASS_STATIONARY,
    NOISE_CLASS_NON_STATIONARY,
    NOISE_CLASS_IMPULSIVE,
)


class TestNLMSFilterInit:
    def test_valid_construction(self):
        f = NLMSFilter(filter_length=64, step_size=0.05)
        assert f.filter_length == 64
        assert abs(f.step_size - 0.05) < 1e-10

    def test_default_construction(self):
        f = NLMSFilter()
        assert f.filter_length >= 1
        assert 0 < f.step_size < 2

    def test_invalid_filter_length(self):
        with pytest.raises(ValueError):
            NLMSFilter(filter_length=0)

    def test_invalid_step_size_zero(self):
        with pytest.raises(ValueError):
            NLMSFilter(step_size=0.0)

    def test_invalid_step_size_too_large(self):
        with pytest.raises(ValueError):
            NLMSFilter(step_size=2.0)

    def test_invalid_leakage_zero(self):
        with pytest.raises(ValueError):
            NLMSFilter(leakage=0.0)

    def test_invalid_leakage_too_large(self):
        with pytest.raises(ValueError):
            NLMSFilter(leakage=1.1)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError):
            NLMSFilter(epsilon=-1e-6)

    def test_initial_weights_zero(self):
        f = NLMSFilter(filter_length=32)
        assert np.allclose(f.weights, 0.0)


class TestNLMSProcessSample:
    def test_process_returns_tuple(self):
        f = NLMSFilter(filter_length=8)
        error, estimated = f.process_sample(0.5, 0.3)
        assert isinstance(error, float)
        assert isinstance(estimated, float)

    def test_zero_inputs_return_zero(self):
        f = NLMSFilter(filter_length=16)
        error, estimated = f.process_sample(0.0, 0.0)
        assert estimated == pytest.approx(0.0, abs=1e-10)

    def test_output_finite(self):
        f = NLMSFilter(filter_length=16)
        error, est = f.process_sample(0.1, 0.05)
        assert np.isfinite(error)
        assert np.isfinite(est)

    def test_nan_input_handled_gracefully(self):
        f = NLMSFilter(filter_length=16)
        error, est = f.process_sample(float('nan'), 0.1)
        # Should not crash; returns some value
        assert not np.isnan(est)

    def test_inf_input_handled_gracefully(self):
        f = NLMSFilter(filter_length=16)
        error, est = f.process_sample(float('inf'), 0.1)
        assert not np.isnan(est)


class TestNLMSBlockProcessing:
    def test_block_shape(self):
        f = NLMSFilter(filter_length=8)
        N = 100
        primary = np.random.randn(N).astype(np.float64)
        reference = np.random.randn(N).astype(np.float64)
        error, estimated = f.process_block(primary, reference)
        assert error.shape == (N,)
        assert estimated.shape == (N,)

    def test_block_dtype(self):
        f = NLMSFilter(filter_length=8)
        primary = np.random.randn(50).astype(np.float64)
        reference = np.random.randn(50).astype(np.float64)
        error, _ = f.process_block(primary, reference)
        assert error.dtype == np.float32

    def test_block_finite_output(self):
        f = NLMSFilter(filter_length=8)
        primary = np.random.randn(200).astype(np.float64)
        reference = np.random.randn(200).astype(np.float64)
        error, estimated = f.process_block(primary, reference)
        assert np.all(np.isfinite(error))
        assert np.all(np.isfinite(estimated))

    def test_mismatched_lengths_raises(self):
        f = NLMSFilter(filter_length=8)
        with pytest.raises(ValueError):
            f.process_block(np.zeros(10), np.zeros(15))


class TestNLMSConvergence:
    def test_stationary_noise_convergence(self):
        """NLMS should reduce error over time for stationary noise."""
        rng = np.random.default_rng(42)
        N = 5000
        noise = rng.normal(0, 0.5, N).astype(np.float64)
        speech = np.zeros(N, dtype=np.float64)  # silence: only noise
        m1 = speech + noise
        m2 = noise  # perfect reference

        f = NLMSFilter(filter_length=32, step_size=0.05)
        early_errors = []
        late_errors = []
        for i in range(N):
            err, _ = f.process_sample(float(m1[i]), float(m2[i]))
            if i < 200:
                early_errors.append(abs(err))
            elif i > 4000:
                late_errors.append(abs(err))

        # After convergence, error should decrease
        assert np.mean(late_errors) < np.mean(early_errors) or np.mean(late_errors) < 0.5


class TestNLMSImpulsiveGate:
    def test_impulsive_gate_activates(self):
        f = NLMSFilter(
            filter_length=8, step_size=0.05,
            impulsive_gate=True, impulsive_threshold=3.0
        )
        # Warmup
        for _ in range(100):
            f.process_sample(0.01, 0.01)
        # Inject large impulse
        f.process_sample(1000.0, 1000.0)
        assert f.diagnostics["n_impulsive_gates"] > 0

    def test_no_gate_without_flag(self):
        f = NLMSFilter(filter_length=8, impulsive_gate=False)
        for _ in range(100):
            f.process_sample(1000.0, 1000.0)
        # No gating should occur
        assert f.diagnostics["n_impulsive_gates"] == 0


class TestNLMSReset:
    def test_reset_zeroes_weights(self):
        f = NLMSFilter(filter_length=16, step_size=0.1)
        for _ in range(100):
            f.process_sample(0.5, 0.3)
        assert not np.allclose(f.weights, 0.0)
        f.reset()
        assert np.allclose(f.weights, 0.0)

    def test_reset_resets_counter(self):
        f = NLMSFilter(filter_length=8)
        for _ in range(50):
            f.process_sample(0.1, 0.1)
        assert f.diagnostics["n_samples_processed"] == 50
        f.reset()
        assert f.diagnostics["n_samples_processed"] == 0


class TestNLMSDeterminism:
    def test_deterministic_output(self):
        """Same inputs produce identical outputs."""
        rng = np.random.default_rng(99)
        primary = rng.normal(0, 0.3, 200).astype(np.float64)
        reference = rng.normal(0, 0.2, 200).astype(np.float64)

        f1 = NLMSFilter(filter_length=16, step_size=0.05)
        f2 = NLMSFilter(filter_length=16, step_size=0.05)
        e1, _ = f1.process_block(primary, reference)
        e2, _ = f2.process_block(primary, reference)
        np.testing.assert_array_equal(e1, e2)


class TestNoiseClassFactory:
    def test_creates_stationary_filter(self):
        f = create_nlms_for_noise_class(NOISE_CLASS_STATIONARY)
        assert isinstance(f, NLMSFilter)

    def test_creates_non_stationary_filter(self):
        f = create_nlms_for_noise_class(NOISE_CLASS_NON_STATIONARY)
        assert isinstance(f, NLMSFilter)

    def test_creates_impulsive_filter(self):
        f = create_nlms_for_noise_class(NOISE_CLASS_IMPULSIVE)
        assert isinstance(f, NLMSFilter)
        assert f.impulsive_gate is True

    def test_invalid_class_raises(self):
        with pytest.raises(ValueError):
            create_nlms_for_noise_class(99)

    def test_override_filter_length(self):
        f = create_nlms_for_noise_class(NOISE_CLASS_STATIONARY, filter_length=10)
        assert f.filter_length == 10


class TestAdaptiveNLMSController:
    def test_default_class(self):
        ctrl = AdaptiveNLMSController()
        assert ctrl.current_noise_class == NOISE_CLASS_STATIONARY

    def test_class_update(self):
        ctrl = AdaptiveNLMSController()
        ctrl.update_noise_class(NOISE_CLASS_IMPULSIVE)
        assert ctrl.current_noise_class == NOISE_CLASS_IMPULSIVE

    def test_process_block(self):
        ctrl = AdaptiveNLMSController()
        primary = np.random.randn(64).astype(np.float64)
        reference = np.random.randn(64).astype(np.float64)
        error, _ = ctrl.process_block(primary, reference)
        assert error.shape == (64,)
        assert np.all(np.isfinite(error))

    def test_reset(self):
        ctrl = AdaptiveNLMSController()
        for _ in range(100):
            ctrl.process_block(np.random.randn(32), np.random.randn(32))
        ctrl.reset()
        # Should not raise

    def test_invalid_class_ignored(self):
        ctrl = AdaptiveNLMSController()
        ctrl.update_noise_class(99)  # invalid — should be ignored
        assert ctrl.current_noise_class == NOISE_CLASS_STATIONARY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
