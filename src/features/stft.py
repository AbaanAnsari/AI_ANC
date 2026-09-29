from __future__ import annotations

import numpy as np
from scipy.signal import get_window as scipy_get_window
from scipy.signal import istft as scipy_istft
from scipy.signal import stft as scipy_stft


class STFTExtractor:
    """
    Short-Time Fourier Transform (STFT) & ISTFT Feature Extractor.

    Canonical Configuration:
        - sample_rate: 16000 Hz
        - n_fft: 512
        - win_length: 512
        - hop_length: 128
        - window: "hann"
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 128,
        window: str = "hann",
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"n_fft must be > 0, got {n_fft}")
        if win_length <= 0 or win_length > n_fft:
            raise ValueError(
                f"win_length must be > 0 and <= n_fft ({n_fft}), got {win_length}"
            )
        if hop_length <= 0:
            raise ValueError(f"hop_length must be > 0, got {hop_length}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    def transform(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute complex STFT representation.

        Parameters
        ----------
        audio : np.ndarray
            1-D mono float32 audio waveform.

        Returns
        -------
        stft_matrix : np.ndarray
            Complex STFT matrix of shape (frequency_bins, time_frames), dtype complex64.
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError(
                f"audio must be a NumPy array, got {type(audio).__name__}"
            )
        if audio.ndim != 1:
            raise ValueError(
                f"audio must be 1-D mono audio, got shape {audio.shape}"
            )
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite (NaN/Inf) values")

        audio_in = audio.astype(np.float32)
        if audio_in.shape[0] < self.win_length:
            padded = np.zeros(self.win_length, dtype=np.float32)
            padded[: audio_in.shape[0]] = audio_in
            audio_in = padded

        noverlap = self.win_length - self.hop_length

        _, _, zxx = scipy_stft(
            audio_in,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.win_length,
            noverlap=noverlap,
            nfft=self.n_fft,
            padded=True,
            boundary="zeros",
        )

        stft_matrix = zxx.astype(np.complex64)
        if not np.all(np.isfinite(stft_matrix)):
            raise RuntimeError("STFT calculation produced non-finite values")

        return stft_matrix

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute complex STFT for a single analysis frame.
        Mathematically and numerically identical to scipy.signal.stft(..., scaling='spectrum').

        Parameters
        ----------
        frame : np.ndarray
            1-D audio frame of length win_length (512 samples).

        Returns
        -------
        np.ndarray of shape (frequency_bins,), complex64.
        """
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 1 or len(frame) != self.win_length:
            raise ValueError(
                f"frame must be 1-D of length {self.win_length}, got shape {frame.shape}"
            )
        win = scipy_get_window(self.window, self.win_length).astype(np.float32)
        scale = 1.0 / win.sum()
        spec = np.fft.rfft(frame.astype(np.float32) * win, n=self.n_fft) * scale
        return spec.astype(np.complex64)

    def inverse_frame(self, spec: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT scaled for overlap-add synthesis,
        matching scipy.signal.istft(..., scaling='spectrum').

        Parameters
        ----------
        spec : np.ndarray
            1-D complex spectrum of shape (frequency_bins,).

        Returns
        -------
        np.ndarray of shape (win_length,), float32.
        """
        win = scipy_get_window(self.window, self.win_length).astype(np.float32)
        time_frame = np.fft.irfft(spec, n=self.n_fft)[:self.win_length] * win.sum()
        return (time_frame * win).astype(np.float32)

    def inverse(
        self,
        stft_matrix: np.ndarray,
        length: int | None = None,
    ) -> np.ndarray:
        """
        Compute Inverse STFT (ISTFT) to reconstruct 1-D float32 waveform.

        Parameters
        ----------
        stft_matrix : np.ndarray
            Complex STFT matrix of shape (frequency_bins, time_frames).
        length : int | None
            Exact output length in samples to truncate/pad result.

        Returns
        -------
        audio_rec : np.ndarray
            Reconstructed 1-D float32 waveform.
        """
        if not isinstance(stft_matrix, np.ndarray):
            raise TypeError("stft_matrix must be a NumPy array")
        if stft_matrix.ndim != 2:
            raise ValueError(
                f"stft_matrix must be 2-D (freq_bins, time_frames), got shape {stft_matrix.shape}"
            )
        if not np.all(np.isfinite(stft_matrix)):
            raise ValueError("stft_matrix contains non-finite values")

        noverlap = self.win_length - self.hop_length

        _, x_rec = scipy_istft(
            stft_matrix,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.win_length,
            noverlap=noverlap,
            nfft=self.n_fft,
            time_axis=-1,
            freq_axis=-2,
            boundary=True,
        )

        audio_rec = np.asarray(x_rec, dtype=np.float32)

        if length is not None:
            if audio_rec.shape[0] >= length:
                audio_rec = audio_rec[:length]
            else:
                padded = np.zeros(length, dtype=np.float32)
                padded[: audio_rec.shape[0]] = audio_rec
                audio_rec = padded

        return audio_rec


def compute_stft(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    window: str = "hann",
) -> np.ndarray:
    extractor = STFTExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    return extractor.transform(audio)


def compute_istft(
    stft_matrix: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    window: str = "hann",
    length: int | None = None,
) -> np.ndarray:
    extractor = STFTExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    return extractor.inverse(stft_matrix, length=length)


def compute_stft_frame(
    frame: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
    window: str = "hann",
) -> np.ndarray:
    """Compute single-frame STFT identical to scipy.signal.stft."""
    extractor = STFTExtractor(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
    )
    return extractor.transform_frame(frame)



def transform_frame(frame: np.ndarray, window: np.ndarray = None, n_fft: int = 512) -> np.ndarray:
    """Compute single-frame STFT identical to scipy.signal.stft with 1.0/sum(window) scaling."""
    if window is None:
        window = scipy_get_window("hann", n_fft, fftbins=True)
    win_scale = 1.0 / float(np.sum(window))
    return np.fft.rfft(frame * window, n=n_fft) * win_scale


def inverse_frame(spec: np.ndarray, window: np.ndarray = None, n_fft: int = 512) -> np.ndarray:
    """Compute single-frame inverse FFT unscaled by sum(window)."""
    if window is None:
        window = scipy_get_window("hann", n_fft, fftbins=True)
    win_sum = float(np.sum(window))
    return np.fft.irfft(spec * win_sum, n=n_fft)[:n_fft]



def get_magnitude(stft_matrix: np.ndarray) -> np.ndarray:
    """Return magnitude spectrum |Zxx|."""
    return np.abs(stft_matrix).astype(np.float32)


def get_phase(stft_matrix: np.ndarray) -> np.ndarray:
    """Return phase spectrum angle(Zxx) in radians [-pi, pi]."""
    return np.angle(stft_matrix).astype(np.float32)


def get_real(stft_matrix: np.ndarray) -> np.ndarray:
    """Return real component Re{Zxx}."""
    return np.real(stft_matrix).astype(np.float32)


def get_imaginary(stft_matrix: np.ndarray) -> np.ndarray:
    """Return imaginary component Im{Zxx}."""
    return np.imag(stft_matrix).astype(np.float32)
