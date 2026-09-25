import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.audio_loader import load_audio
from src.data.mixer import (
    DEFAULT_SNR_SET,
    calculate_rms,
    calculate_snr_db,
    mix_signals,
)
from src.data.segmenter import segment_audio


def test_snr_scaling_accuracy():
    rng = np.random.default_rng(42)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=16000).astype(np.float32)

    for target_snr in DEFAULT_SNR_SET:
        noisy, scaled_noise = mix_signals(clean, noise, target_snr)

        measured_snr = calculate_snr_db(clean, scaled_noise)
        assert np.isclose(measured_snr, target_snr, atol=1e-3), (
            f"Expected {target_snr} dB, got {measured_snr:.4f} dB"
        )


def test_preserve_clean_length():
    rng = np.random.default_rng(123)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)

    # Test short noise (8000) and long noise (32000)
    for noise_len in (8000, 16000, 32000):
        noise = rng.normal(0.0, 0.05, size=noise_len).astype(np.float32)
        noisy, scaled_noise = mix_signals(clean, noise, target_snr_db=10.0, rng=rng)

        assert noisy.shape == (16000,)
        assert scaled_noise.shape == (16000,)
        assert noisy.dtype == np.float32
        assert scaled_noise.dtype == np.float32


def test_output_dtype_and_finiteness():
    rng = np.random.default_rng(999)
    clean = rng.normal(0.0, 0.1, size=16000).astype(np.float32)
    noise = rng.normal(0.0, 0.05, size=16000).astype(np.float32)

    noisy, scaled_noise = mix_signals(clean, noise, target_snr_db=0.0)

    assert noisy.dtype == np.float32
    assert scaled_noise.dtype == np.float32
    assert np.all(np.isfinite(noisy))
    assert np.all(np.isfinite(scaled_noise))


def test_input_immutability():
    clean = np.ones(16000, dtype=np.float32)
    noise = np.ones(16000, dtype=np.float32)

    clean_orig = clean.copy()
    noise_orig = noise.copy()

    mix_signals(clean, noise, target_snr_db=5.0)

    assert np.array_equal(clean, clean_orig)
    assert np.array_equal(noise, noise_orig)


def test_real_audio_mixing():
    clean_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dataset"
        / "clean"
        / "1098-133695-0000.wav"
    )
    noise_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dataset"
        / "noise"
        / "stationary"
        / "fan"
        / "fan_0.wav"
    )

    if clean_path.exists() and noise_path.exists():
        clean_raw = load_audio(clean_path)
        noise_raw = load_audio(noise_path)

        clean = segment_audio(clean_raw, rng=np.random.default_rng(42))
        noise = segment_audio(noise_raw, rng=np.random.default_rng(42))

        noisy, scaled_noise = mix_signals(clean, noise, target_snr_db=10.0)

        measured_snr = calculate_snr_db(clean, scaled_noise)
        assert np.isclose(measured_snr, 10.0, atol=1e-2)
        assert noisy.shape == (16000,)
        assert noisy.dtype == np.float32


if __name__ == "__main__":
    test_snr_scaling_accuracy()
    test_preserve_clean_length()
    test_output_dtype_and_finiteness()
    test_input_immutability()
    test_real_audio_mixing()
    print("mixer tests: PASS")
