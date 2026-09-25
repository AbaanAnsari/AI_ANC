import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.dual_mic import apply_delay, simulate_dual_mic
from src.dsp.gcc_phat import gcc_phat


def test_zero_delay():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0.0, 1.0, size=1000).astype(np.float32)
    s2 = s1.copy()

    res = gcc_phat(s1, s2, max_tau=20)
    assert res["estimated_delay_samples"] == 0
    assert np.isclose(res["estimated_delay_seconds"], 0.0)


def test_positive_known_delays():
    rng = np.random.default_rng(123)
    s1 = rng.normal(0.0, 1.0, size=2000).astype(np.float32)

    for known_delay in (3, 5, 10, 15):
        s2 = apply_delay(s1, known_delay)
        res = gcc_phat(s1, s2, max_tau=20)
        assert res["estimated_delay_samples"] == known_delay, (
            f"Expected positive delay {known_delay}, got {res['estimated_delay_samples']}"
        )


def test_negative_known_delays():
    rng = np.random.default_rng(456)
    s1 = rng.normal(0.0, 1.0, size=2000).astype(np.float32)

    for known_delay in (-3, -5, -10, -15):
        s2 = apply_delay(s1, known_delay)
        res = gcc_phat(s1, s2, max_tau=20)
        assert res["estimated_delay_samples"] == known_delay, (
            f"Expected negative delay {known_delay}, got {res['estimated_delay_samples']}"
        )


def test_maximum_lag_restriction():
    rng = np.random.default_rng(789)
    s1 = rng.normal(0.0, 1.0, size=2000).astype(np.float32)
    s2 = apply_delay(s1, 5)

    # Restrict max_tau to 10
    res = gcc_phat(s1, s2, max_tau=10)
    assert np.all(np.abs(res["lags"]) <= 10)
    assert res["estimated_delay_samples"] == 5


def test_unequal_input_lengths():
    rng = np.random.default_rng(101)
    s1 = rng.normal(0.0, 1.0, size=2000).astype(np.float32)
    s2 = apply_delay(s1, 7)[:1500]  # Shorter length

    res = gcc_phat(s1, s2, max_tau=20)
    assert res["estimated_delay_samples"] == 7


def test_zero_signal_edge_case():
    s1 = np.zeros(1000, dtype=np.float32)
    s2 = np.zeros(1000, dtype=np.float32)

    res = gcc_phat(s1, s2, max_tau=10)
    assert res["estimated_delay_samples"] == 0
    assert np.all(np.isfinite(res["gcc_cc"]))


def test_input_immutability():
    rng = np.random.default_rng(202)
    s1 = rng.normal(0.0, 1.0, size=1000).astype(np.float32)
    s2 = rng.normal(0.0, 1.0, size=1000).astype(np.float32)

    s1_orig = s1.copy()
    s2_orig = s2.copy()

    gcc_phat(s1, s2, max_tau=20)

    assert np.array_equal(s1, s1_orig)
    assert np.array_equal(s2, s2_orig)


def test_end_to_end_dual_mic_integration():
    """
    End-to-End integration test:
    known configured delay -> dual_mic simulation -> M1/M2 -> GCC-PHAT -> estimated delay
    """
    rng = np.random.default_rng(303)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=16000).astype(np.float32)

    for configured_delay in (-15, -7, 0, 5, 12, 18):
        m1, m2 = simulate_dual_mic(
            clean_speech=clean,
            scaled_noise=noise,
            relative_delay_samples=configured_delay,
            gain_mismatch=1.05,
            filter_coefficients=(0.95, 0.05),
            rng=rng,
        )

        res = gcc_phat(m1, m2, sample_rate=16000, max_tau=30)
        estimated_delay = res["estimated_delay_samples"]

        assert estimated_delay == configured_delay, (
            f"End-to-end failure: Configured delay {configured_delay}, "
            f"GCC-PHAT estimated {estimated_delay}"
        )


if __name__ == "__main__":
    test_zero_delay()
    test_positive_known_delays()
    test_negative_known_delays()
    test_maximum_lag_restriction()
    test_unequal_input_lengths()
    test_zero_signal_edge_case()
    test_input_immutability()
    test_end_to_end_dual_mic_integration()
    print("gcc_phat tests: PASS")
