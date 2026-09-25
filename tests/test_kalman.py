import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.dual_mic import simulate_dual_mic
from src.dsp.gcc_phat import gcc_phat
from src.dsp.kalman import DelayKalmanFilter, filter_delay_sequence


def test_initialization():
    kf = DelayKalmanFilter(
        initial_delay=5.0,
        initial_covariance=2.0,
        process_noise=0.05,
        measurement_noise=0.5,
    )
    assert kf.current_state == 5.0
    assert kf.current_covariance == 2.0


def test_constant_delay_convergence_positive():
    kf = DelayKalmanFilter(
        initial_delay=0.0,
        initial_covariance=1.0,
        process_noise=0.01,
        measurement_noise=1.0,
    )

    for _ in range(20):
        kf.update(7.0)

    assert np.isclose(kf.current_state, 7.0, atol=0.2), (
        f"Expected convergence to +7, got {kf.current_state:.4f}"
    )


def test_constant_delay_convergence_negative():
    kf = DelayKalmanFilter(
        initial_delay=0.0,
        initial_covariance=1.0,
        process_noise=0.01,
        measurement_noise=1.0,
    )

    for _ in range(20):
        kf.update(-7.0)

    assert np.isclose(kf.current_state, -7.0, atol=0.2), (
        f"Expected convergence to -7, got {kf.current_state:.4f}"
    )


def test_noisy_delay_smoothing():
    rng = np.random.default_rng(42)
    true_delay = 7.0
    raw_measurements = true_delay + rng.normal(0.0, 3.0, size=100)

    kf = DelayKalmanFilter(
        initial_delay=0.0,
        process_noise=0.01,
        measurement_noise=4.0,
    )

    filtered = [kf.update(m) for m in raw_measurements]

    # Ignore first 10 steps for warm-up
    raw_std = np.std(raw_measurements[10:])
    filtered_std = np.std(filtered[10:])

    assert filtered_std < raw_std, (
        f"Kalman filter did not smooth noise: filtered std {filtered_std:.4f} vs raw std {raw_std:.4f}"
    )


def test_slowly_changing_delay():
    kf = DelayKalmanFilter(
        initial_delay=0.0,
        process_noise=0.5,
        measurement_noise=0.5,
    )

    true_ramp = np.arange(15, dtype=float)
    filtered = [kf.update(val) for val in true_ramp]

    # Verify final estimate follows trend toward 14.0
    assert np.isclose(filtered[-1], 14.0, atol=1.5)


def test_outlier_handling():
    kf = DelayKalmanFilter(
        initial_delay=7.0,
        initial_covariance=0.1,
        process_noise=0.01,
        measurement_noise=4.0,
    )

    # Warm up around +7
    for _ in range(10):
        kf.update(7.0)

    state_before_outlier = kf.current_state

    # Insert huge outlier +40
    outlier_result = kf.update(40.0)

    # Verify estimate does not jump to +40 immediately
    assert outlier_result < 20.0, (
        f"Filter jumped excessively on outlier: got {outlier_result:.4f}"
    )

    # Subsequent normal observations bring it back toward +7
    for _ in range(25):
        kf.update(7.0)

    assert np.isclose(kf.current_state, 7.0, atol=0.75), (
        f"Filter failed to recover after outlier: got {kf.current_state:.4f}"
    )


def test_state_persistence():
    kf = DelayKalmanFilter(initial_delay=0.0)
    res1 = kf.update(10.0)
    res2 = kf.update(10.0)

    assert res2 != res1, "Second update should depend on updated state"


def test_reset():
    kf = DelayKalmanFilter(initial_delay=2.0, initial_covariance=0.5)

    kf.update(15.0)
    kf.update(20.0)
    assert kf.current_state != 2.0

    kf.reset()
    assert kf.current_state == 2.0
    assert kf.current_covariance == 0.5


def test_determinism():
    measurements = [0.0, 1.0, 5.0, 7.0, 7.0, 8.0, 7.0]

    out1 = filter_delay_sequence(measurements, initial_delay=0.0)
    out2 = filter_delay_sequence(measurements, initial_delay=0.0)

    assert np.array_equal(out1, out2)


def test_finite_output():
    kf = DelayKalmanFilter()
    measurements = [0.0, -10.0, 15.0, 100.0, -50.0]

    for m in measurements:
        state = kf.update(m)
        assert np.isfinite(state)
        assert np.isfinite(kf.current_covariance)


def test_gcc_phat_dual_mic_integration():
    """
    Full pipeline integration:
    dual_mic simulation -> GCC-PHAT -> raw delay -> add observation noise -> DelayKalmanFilter
    """
    rng = np.random.default_rng(2026)
    clean_speech = rng.normal(0.0, 0.2, size=16000).astype(np.float32)
    scaled_noise = rng.normal(0.0, 0.1, size=16000).astype(np.float32)

    tested_delays = [-15, -7, 0, 5, 12, 18]

    for true_delay in tested_delays:
        # 1. Dual Mic Simulation
        m1, m2 = simulate_dual_mic(
            clean_speech=clean_speech,
            scaled_noise=scaled_noise,
            relative_delay_samples=true_delay,
            gain_mismatch=1.05,
            filter_coefficients=(0.95, 0.05),
            rng=rng,
        )

        # 2. GCC-PHAT Raw Observation
        gcc_res = gcc_phat(m1, m2, sample_rate=16000, max_tau=30)
        raw_delay = gcc_res["estimated_delay_samples"]
        assert raw_delay == true_delay

        # 3. Add controlled observation noise to observations
        kf = DelayKalmanFilter(
            initial_delay=0.0,
            initial_covariance=5.0,
            process_noise=0.01,
            measurement_noise=2.0,
        )

        # Feed sequence of noisy observations of the raw GCC-PHAT delay
        noisy_observations = [raw_delay + rng.normal(0.0, 1.0) for _ in range(30)]
        for obs in noisy_observations:
            filtered_delay = kf.update(obs)

        # Final filtered estimate must recover true delay accurately
        assert np.isclose(kf.current_state, true_delay, atol=0.75), (
            f"True delay {true_delay}, got filtered estimate {kf.current_state:.4f}"
        )


if __name__ == "__main__":
    test_initialization()
    test_constant_delay_convergence_positive()
    test_constant_delay_convergence_negative()
    test_noisy_delay_smoothing()
    test_slowly_changing_delay()
    test_outlier_handling()
    test_state_persistence()
    test_reset()
    test_determinism()
    test_finite_output()
    test_gcc_phat_dual_mic_integration()
    print("kalman tests: PASS")
