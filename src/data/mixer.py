from __future__ import annotations

import numpy as np

DEFAULT_SNR_SET = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
EPSILON = 1e-12


def calculate_rms(signal: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) level of a 1-D signal.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(
            f"signal must be a NumPy array, got {type(signal).__name__}"
        )
    if signal.ndim != 1:
        raise ValueError(
            f"signal must be a 1-D array, got shape {signal.shape}"
        )
    mean_sq = float(np.mean(signal**2))
    return float(np.sqrt(mean_sq + EPSILON))


def calculate_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """
    Measure the actual Signal-to-Noise Ratio (SNR) in dB between
    clean speech and scaled noise.
    """
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)

    clean_power = clean_rms**2
    noise_power = noise_rms**2

    snr_linear = (clean_power + EPSILON) / (noise_power + EPSILON)
    return float(10.0 * np.log10(snr_linear))


def match_noise_length(
    noise: np.ndarray,
    target_length: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Match noise length to target_length by random cropping (if longer)
    or repeating/tiling (if shorter).
    """
    noise_len = noise.shape[0]

    if noise_len == target_length:
        return noise.copy()

    if noise_len > target_length:
        max_start = noise_len - target_length
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, max_start + 1))
        return noise[start : start + target_length].copy()

    # If shorter, tile noise to match target_length
    repeats = (target_length // noise_len) + 1
    tiled = np.tile(noise, repeats)
    return tiled[:target_length].copy()


def mix_signals(
    clean: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mix clean speech and noise at a target SNR (in dB).

    Parameters
    ----------
    clean : np.ndarray
        Clean speech waveform (1-D float32).
    noise : np.ndarray
        Noise waveform (1-D float32).
    target_snr_db : float
        Target SNR in decibels.
    rng : np.random.Generator | None
        Optional random number generator for length alignment.

    Returns
    -------
    noisy_speech : np.ndarray
        1-D float32 mixture array with length equal to clean.
    scaled_noise : np.ndarray
        1-D float32 scaled noise array matching clean length.
    """
    if not isinstance(clean, np.ndarray) or not isinstance(noise, np.ndarray):
        raise TypeError("clean and noise must be NumPy arrays")

    if clean.ndim != 1 or noise.ndim != 1:
        raise ValueError("clean and noise must be 1-D mono arrays")

    if clean.dtype != np.float32 or noise.dtype != np.float32:
        raise ValueError("clean and noise must have dtype float32")

    if not np.all(np.isfinite(clean)) or not np.all(np.isfinite(noise)):
        raise ValueError("Inputs contain NaN or Inf values")

    target_length = clean.shape[0]

    # Adjust noise length to match clean
    noise_aligned = match_noise_length(
        noise, target_length=target_length, rng=rng
    )

    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise_aligned)

    # Calculate required noise scaling factor
    # SNR_linear = 10 ** (target_snr_db / 10)
    # P_clean / P_noise_target = SNR_linear
    # P_noise_target = P_clean / SNR_linear
    # RMS_noise_target = clean_rms / sqrt(10 ** (target_snr_db / 10))
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    target_noise_rms = clean_rms / np.sqrt(snr_linear)

    scale_factor = target_noise_rms / (noise_rms + EPSILON)

    scaled_noise = (noise_aligned * scale_factor).astype(np.float32)
    noisy_speech = (clean + scaled_noise).astype(np.float32)

    if not np.all(np.isfinite(noisy_speech)):
        raise RuntimeError("Generated mixture contains non-finite values")

    return noisy_speech, scaled_noise
