from __future__ import annotations

import numpy as np


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SEGMENT_SECONDS = 1.0


def segment_audio(
    audio: np.ndarray,
    segment_samples: int = 16000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Convert a canonical audio waveform into exactly one fixed-length segment.

    Expected input:
        - 1-D NumPy array
        - mono
        - float32
        - finite values
        - canonical sample rate: 16 kHz

    Behaviour:
        - Exactly segment_samples:
            returns a copy unchanged.
        - Longer than segment_samples:
            performs a random crop.
        - Shorter than segment_samples:
            zero-pads at the end.

    The input array is never modified.

    Parameters
    ----------
    audio:
        Canonical 1-D float32 audio waveform.

    segment_samples:
        Required output length in samples.
        Default: 16000 samples = 1 second at 16 kHz.

    rng:
        Optional NumPy random generator.
        Supplying one makes random cropping reproducible.

    Returns
    -------
    np.ndarray
        Exactly segment_samples samples, dtype float32.
    """

    if not isinstance(audio, np.ndarray):
        raise TypeError(
            f"audio must be a NumPy array, got {type(audio).__name__}"
        )

    if audio.ndim != 1:
        raise ValueError(
            f"audio must be 1-D mono audio, got shape {audio.shape}"
        )

    if audio.dtype != np.float32:
        raise ValueError(
            f"audio must have dtype float32, got {audio.dtype}"
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "audio contains NaN or Inf values"
        )

    if segment_samples <= 0:
        raise ValueError(
            f"segment_samples must be > 0, got {segment_samples}"
        )

    if rng is None:
        rng = np.random.default_rng()

    audio_length = audio.shape[0]

    # ---------------------------------------------------------
    # Case 1: exact length
    # ---------------------------------------------------------
    if audio_length == segment_samples:
        return audio.copy()

    # ---------------------------------------------------------
    # Case 2: audio is longer -> random crop
    # ---------------------------------------------------------
    if audio_length > segment_samples:

        max_start = audio_length - segment_samples

        start = int(
            rng.integers(
                0,
                max_start + 1,
            )
        )

        end = start + segment_samples

        return audio[start:end].copy()

    # ---------------------------------------------------------
    # Case 3: audio is shorter -> zero-pad at the end
    # ---------------------------------------------------------
    output = np.zeros(
        segment_samples,
        dtype=np.float32,
    )

    output[:audio_length] = audio

    return output