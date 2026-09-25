import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.audio_loader import load_audio
from src.data.segmenter import segment_audio
from src.features.stft import (
    STFTExtractor,
    compute_istft,
    compute_stft,
    get_imaginary,
    get_magnitude,
    get_phase,
    get_real,
)


def test_basic_stft_shape_and_dtype():
    audio = np.ones(16000, dtype=np.float32)
    extractor = STFTExtractor(
        sample_rate=16000, n_fft=512, win_length=512, hop_length=128
    )

    stft_matrix = extractor.transform(audio)

    assert stft_matrix.ndim == 2
    assert stft_matrix.shape[0] == 257  # (n_fft // 2 + 1)
    assert stft_matrix.dtype == np.complex64


def test_deterministic_output():
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 1.0, size=16000).astype(np.float32)

    stft1 = compute_stft(audio)
    stft2 = compute_stft(audio)

    assert np.array_equal(stft1, stft2)


def test_input_immutability():
    audio = np.arange(16000, dtype=np.float32)
    audio_orig = audio.copy()

    compute_stft(audio)

    assert np.array_equal(audio, audio_orig)


def test_finite_real_imaginary():
    rng = np.random.default_rng(123)
    audio = rng.normal(0.0, 1.0, size=16000).astype(np.float32)

    stft_matrix = compute_stft(audio)

    real_part = get_real(stft_matrix)
    imag_part = get_imaginary(stft_matrix)
    mag_part = get_magnitude(stft_matrix)
    phase_part = get_phase(stft_matrix)

    assert np.all(np.isfinite(real_part))
    assert np.all(np.isfinite(imag_part))
    assert np.all(np.isfinite(mag_part))
    assert np.all(np.isfinite(phase_part))


def test_short_signal_handling():
    short_audio = np.ones(200, dtype=np.float32)
    stft_matrix = compute_stft(short_audio, n_fft=512, win_length=512, hop_length=128)

    assert stft_matrix.shape[0] == 257
    assert np.all(np.isfinite(stft_matrix))


def test_non_hop_aligned_length():
    odd_audio = np.ones(15347, dtype=np.float32)
    stft_matrix = compute_stft(odd_audio, n_fft=512, win_length=512, hop_length=128)

    rec_audio = compute_istft(
        stft_matrix,
        n_fft=512,
        win_length=512,
        hop_length=128,
        length=len(odd_audio),
    )

    assert rec_audio.shape == (15347,)
    assert np.all(np.isfinite(rec_audio))


def test_zero_silent_signal():
    silent_audio = np.zeros(16000, dtype=np.float32)
    stft_matrix = compute_stft(silent_audio)

    assert np.all(stft_matrix == 0.0)


def test_sinusoid_spectral_peak():
    sr = 16000
    n_fft = 512
    freq_hz = 1000.0  # 1 kHz sine wave
    t = np.arange(sr, dtype=np.float32) / sr
    sine_wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

    stft_matrix = compute_stft(sine_wave, sample_rate=sr, n_fft=n_fft)
    mag = get_magnitude(stft_matrix)

    # Average magnitude over time frames
    mean_mag = np.mean(mag, axis=1)
    peak_bin = int(np.argmax(mean_mag))

    expected_bin = int(round(freq_hz / (sr / n_fft)))
    assert peak_bin == expected_bin, (
        f"Expected spectral peak at bin {expected_bin}, got {peak_bin}"
    )


def test_stft_istft_reconstruction():
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 0.5, size=16000).astype(np.float32)

    stft_matrix = compute_stft(audio)
    reconstructed = compute_istft(stft_matrix, length=len(audio))

    mse = float(np.mean((audio - reconstructed) ** 2))
    max_err = float(np.max(np.abs(audio - reconstructed)))

    assert mse < 1e-6, f"Reconstruction MSE too high: {mse}"
    assert max_err < 1e-3, f"Reconstruction max error too high: {max_err}"


def test_multiple_parameter_configurations():
    audio = np.ones(16000, dtype=np.float32)

    configs = [
        {"n_fft": 256, "win_length": 256, "hop_length": 64},
        {"n_fft": 512, "win_length": 512, "hop_length": 128},
        {"n_fft": 1024, "win_length": 1024, "hop_length": 256},
    ]

    for cfg in configs:
        stft_matrix = compute_stft(audio, **cfg)
        rec = compute_istft(stft_matrix, length=len(audio), **cfg)

        assert stft_matrix.shape[0] == cfg["n_fft"] // 2 + 1
        assert rec.shape == (16000,)
        assert np.all(np.isfinite(stft_matrix))


def test_real_dataset_waveform_compatibility():
    clean_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dataset"
        / "clean"
        / "1098-133695-0000.wav"
    )

    if clean_path.exists():
        raw_audio = load_audio(clean_path)
        segmented = segment_audio(raw_audio, rng=np.random.default_rng(42))

        stft_matrix = compute_stft(segmented)
        reconstructed = compute_istft(stft_matrix, length=len(segmented))

        mse = float(np.mean((segmented - reconstructed) ** 2))
        assert mse < 1e-6, f"Real dataset reconstruction MSE too high: {mse}"
        assert reconstructed.shape == (16000,)
        assert reconstructed.dtype == np.float32


if __name__ == "__main__":
    test_basic_stft_shape_and_dtype()
    test_deterministic_output()
    test_input_immutability()
    test_finite_real_imaginary()
    test_short_signal_handling()
    test_non_hop_aligned_length()
    test_zero_silent_signal()
    test_sinusoid_spectral_peak()
    test_stft_istft_reconstruction()
    test_multiple_parameter_configurations()
    test_real_dataset_waveform_compatibility()
    print("stft tests: PASS")
