from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Load a WAV file and convert it to the project's canonical
    internal audio representation.

    Canonical output:
        - sample rate: 16 kHz
        - channels: mono
        - dtype: float32
        - shape: (samples,)

    The original source file is never modified.

    Processing:
        1. Read WAV PCM data.
        2. Convert PCM samples to float32.
        3. Downmix multi-channel audio to mono.
        4. Resample to target_sr when necessary.
        5. Return a 1-D float32 waveform.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()

        raw_data = wav.readframes(frame_count)

    if sample_rate <= 0:
        raise ValueError(
            f"Invalid sample rate: {sample_rate}"
        )

    if channels <= 0:
        raise ValueError(
            f"Invalid channel count: {channels}"
        )

    # ---------------------------------------------------------
    # Convert PCM bytes to float32.
    # ---------------------------------------------------------

    if sample_width == 1:
        # WAV 8-bit PCM is unsigned.
        audio = np.frombuffer(
            raw_data,
            dtype=np.uint8,
        ).astype(np.float32)

        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:
        audio = np.frombuffer(
            raw_data,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 4:
        audio = np.frombuffer(
            raw_data,
            dtype=np.int32,
        ).astype(np.float32)

        audio /= 2147483648.0

    else:
        raise ValueError(
            f"Unsupported WAV sample width: "
            f"{sample_width} bytes"
        )

    # ---------------------------------------------------------
    # Validate PCM sample count.
    # ---------------------------------------------------------

    if channels > 1:

        expected_samples = frame_count * channels

        if audio.size != expected_samples:
            raise ValueError(
                "WAV data size does not match the header: "
                f"expected {expected_samples} samples, "
                f"got {audio.size}"
            )

        audio = audio.reshape(
            frame_count,
            channels,
        )

        # Downmix all channels to mono.
        audio = audio.mean(axis=1)

    else:

        if audio.size != frame_count:
            raise ValueError(
                "WAV data size does not match the header: "
                f"expected {frame_count} samples, "
                f"got {audio.size}"
            )

    # ---------------------------------------------------------
    # Resample to canonical sample rate.
    # ---------------------------------------------------------

    if sample_rate != target_sr:

        audio = resample_poly(
            audio,
            target_sr,
            sample_rate,
        )

    # ---------------------------------------------------------
    # Canonical output validation.
    # ---------------------------------------------------------

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim != 1:
        raise RuntimeError(
            f"Expected 1-D mono output, got shape {audio.shape}"
        )

    if not np.all(np.isfinite(audio)):
        raise RuntimeError(
            "Audio contains NaN or Inf values."
        )

    return audio