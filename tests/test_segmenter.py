import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.audio_loader import load_audio
from src.data.segmenter import segment_audio


DATA_ROOT = Path(r"D:\SIH\AI_ANC\data\raw\dataset")


def test_exact_length():

    audio = np.ones(
        16000,
        dtype=np.float32,
    )

    result = segment_audio(audio)

    assert result.shape == (16000,)
    assert result.dtype == np.float32
    assert np.all(result == 1.0)


def test_short_audio_is_zero_padded():

    audio = np.ones(
        8000,
        dtype=np.float32,
    )

    result = segment_audio(audio)

    assert result.shape == (16000,)
    assert result.dtype == np.float32

    assert np.all(result[:8000] == 1.0)
    assert np.all(result[8000:] == 0.0)


def test_long_audio_is_randomly_cropped():

    audio = np.arange(
        32000,
        dtype=np.float32,
    )

    rng = np.random.default_rng(12345)

    result = segment_audio(
        audio,
        rng=rng,
    )

    assert result.shape == (16000,)
    assert result.dtype == np.float32

    assert np.all(
        np.diff(result) == 1.0
    )


def test_input_is_not_modified():

    audio = np.arange(
        20000,
        dtype=np.float32,
    )

    original = audio.copy()

    rng = np.random.default_rng(123)

    segment_audio(
        audio,
        rng=rng,
    )

    assert np.array_equal(
        audio,
        original,
    )


def test_real_dataset_file():

    path = (
        DATA_ROOT
        / "clean"
        / "1098-133695-0000.wav"
    )

    audio = load_audio(path)

    result = segment_audio(
        audio,
        rng=np.random.default_rng(42),
    )

    assert result.shape == (16000,)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


if __name__ == "__main__":

    test_exact_length()
    test_short_audio_is_zero_padded()
    test_long_audio_is_randomly_cropped()
    test_input_is_not_modified()
    test_real_dataset_file()

    print("segmenter tests: PASS")