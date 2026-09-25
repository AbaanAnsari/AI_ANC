import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.dual_mic import (
    apply_delay,
    apply_gain,
    apply_path_filter,
    calculate_cross_correlation,
    simulate_dual_mic,
)


def test_dual_mic_output_properties():
    rng = np.random.default_rng(42)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=16000).astype(np.float32)

    m1, m2 = simulate_dual_mic(clean, noise)

    assert m1.shape == (16000,)
    assert m2.shape == (16000,)
    assert m1.dtype == np.float32
    assert m2.dtype == np.float32
    assert np.all(np.isfinite(m1))
    assert np.all(np.isfinite(m2))


def test_relative_delay_behavior():
    signal = np.zeros(100, dtype=np.float32)
    signal[50] = 1.0  # Impulse at sample 50

    # Positive delay (shift right)
    delayed_pos = apply_delay(signal, delay_samples=10)
    assert delayed_pos.shape == (100,)
    assert delayed_pos[60] == 1.0
    assert delayed_pos[50] == 0.0

    # Negative delay (shift left)
    delayed_neg = apply_delay(signal, delay_samples=-10)
    assert delayed_neg.shape == (100,)
    assert delayed_neg[40] == 1.0
    assert delayed_neg[50] == 0.0

    # Test full range -20 to +20
    for delay in range(-20, 21):
        d_sig = apply_delay(signal, delay_samples=delay)
        assert d_sig.shape == (100,)
        assert np.all(np.isfinite(d_sig))


def test_gain_mismatch_behavior():
    signal = np.ones(100, dtype=np.float32)

    for gain in (0.8, 1.0, 1.2):
        gained = apply_gain(signal, gain)
        assert gained.shape == (100,)
        assert np.allclose(gained, gain)


def test_path_filtering_behavior():
    rng = np.random.default_rng(123)
    signal = rng.normal(0.0, 1.0, size=1000).astype(np.float32)

    filtered = apply_path_filter(signal, coefficients=(0.8, 0.2))
    assert filtered.shape == (1000,)
    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))


def test_noise_correlation():
    rng = np.random.default_rng(42)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=16000).astype(np.float32)

    m1, m2 = simulate_dual_mic(
        clean,
        noise,
        relative_delay_samples=5,
        gain_mismatch=1.1,
        filter_coefficients=(0.9, 0.1),
    )

    n1 = noise
    corr = calculate_cross_correlation(n1, m2)
    assert corr > 0.5, f"Expected strong correlation (>0.5), got {corr:.4f}"


def test_input_immutability():
    clean = np.ones(16000, dtype=np.float32)
    noise = np.ones(16000, dtype=np.float32)

    clean_orig = clean.copy()
    noise_orig = noise.copy()

    simulate_dual_mic(clean, noise, relative_delay_samples=10, gain_mismatch=0.9)

    assert np.array_equal(clean, clean_orig)
    assert np.array_equal(noise, noise_orig)


if __name__ == "__main__":
    test_dual_mic_output_properties()
    test_relative_delay_behavior()
    test_gain_mismatch_behavior()
    test_path_filtering_behavior()
    test_noise_correlation()
    test_input_immutability()
    print("dual mic tests: PASS")
