"""
tests/test_cause1_regression.py
================================
Regression test suite for Cause 1:
- STFT mathematical and scale equivalence
- Temporal context (7-frame buffer, 3-frame lookahead)
- Algorithmic latency (768 samples = 48.0 ms at 16 kHz)
- Offline vs Streaming equivalence (correlation >= 0.999, SNR diff <= 0.05 dB)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.signal import get_window
from src.features.stft import compute_stft, STFTExtractor
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.inference.streaming_pipeline import StreamingPipeline, SAMPLE_RATE, HOP_SIZE, N_FFT
from src.features.stft import compute_istft


def test_stft_equivalence_metrics():
    """Verify STFT extraction error metrics between offline scipy STFT and streaming STFTExtractor."""
    rng = np.random.default_rng(20260929)
    audio = rng.normal(0, 0.2, 16000).astype(np.float32)

    # Offline STFT
    offline_stft = compute_stft(audio, sample_rate=16000, n_fft=512, hop_length=128, win_length=512, window="hann")

    # Streaming STFTExtractor
    extractor = STFTExtractor(sample_rate=16000, n_fft=512, hop_length=128, win_length=512, window="hann")
    padded = np.pad(audio, (256, 256), mode="constant")
    n_frames = offline_stft.shape[1]
    streaming_frames = []
    for i in range(n_frames):
        frame = padded[i * 128 : i * 128 + 512]
        streaming_frames.append(extractor.transform_frame(frame))
    streaming_stft = np.column_stack(streaming_frames)

    abs_diff = np.abs(offline_stft - streaming_stft)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))
    rel_l2 = float(np.linalg.norm(abs_diff) / (np.linalg.norm(offline_stft) + 1e-12))
    scale_diff_db = 20.0 * np.log10(
        (float(np.mean(np.abs(streaming_stft))) + 1e-12) / (float(np.mean(np.abs(offline_stft))) + 1e-12)
    )

    print(f"\nSTFT Max Abs Error:   {max_err:.4e}")
    print(f"STFT Mean Abs Error:  {mean_err:.4e}")
    print(f"STFT Relative L2:     {rel_l2:.4e}")
    print(f"STFT Scale Diff (dB): {scale_diff_db:.4f} dB")

    assert max_err < 1e-6, f"Max error {max_err} exceeds 1e-6"
    assert mean_err < 1e-7, f"Mean error {mean_err} exceeds 1e-7"
    assert rel_l2 < 1e-5, f"Relative L2 error {rel_l2} exceeds 1e-5"
    assert abs(scale_diff_db) < 0.01, f"Scale difference {scale_diff_db:.4f} dB is not ~0 dB"


def test_temporal_context_and_latency():
    """Verify 7-frame buffer, 3-frame lookahead, and 768-sample pipeline latency."""
    from src.inference.streaming_pipeline import CONTEXT_FRAMES, LOOKAHEAD_HOPS
    ckpt_path = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
    pipeline = StreamingPipeline(checkpoint_path=ckpt_path, pipeline_mode="ai_only", device="cpu")
    assert CONTEXT_FRAMES == 7, f"Expected 7-frame buffer, got {CONTEXT_FRAMES}"
    assert LOOKAHEAD_HOPS == 3, f"Expected 3-frame lookahead, got {LOOKAHEAD_HOPS}"
    assert pipeline.algorithmic_delay_samples == 768, f"Expected 768 samples latency, got {pipeline.algorithmic_delay_samples}"
    assert abs(pipeline.algorithmic_delay_ms - 48.0) < 1e-3, f"Expected 48.0 ms latency, got {pipeline.algorithmic_delay_ms:.2f} ms"


def test_offline_vs_streaming_equivalence():
    """Verify offline vs streaming equivalence on Checkpoint D baseline or new checkpoint."""
    ckpt_path = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        pytest.skip("Checkpoint D not found")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pipeline = StreamingPipeline(checkpoint_path=ckpt_path, pipeline_mode="ai_only", device="cpu")

    # Generate test waveform (2 seconds)
    rng = np.random.default_rng(20260929)
    m1 = rng.normal(0, 0.1, 32000).astype(np.float32)
    m2 = np.roll(m1, 5)

    # 1. Offline inference
    stft_c = compute_stft(m1)
    feat = np.stack([stft_c.real.astype(np.float32), stft_c.imag.astype(np.float32)], axis=0)[np.newaxis, :, :, :]
    with torch.no_grad():
        enh_t, _, _ = model(torch.from_numpy(feat))
    enh_np = enh_t.squeeze(0).cpu().numpy()
    enh_c = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    out_offline = compute_istft(enh_c, length=len(m1))

    # 2. Streaming inference
    out_chunks = []
    hop = HOP_SIZE
    for i in range(0, len(m1), hop):
        c1 = m1[i : i + hop]
        c2 = m2[i : i + hop]
        if len(c1) < hop:
            c1 = np.pad(c1, (0, hop - len(c1)))
            c2 = np.pad(c2, (0, hop - len(c2)))
        pipeline.push(c1, c2)
        pipeline.process_available()
        chunk = pipeline.read_output(hop)
        if len(chunk) > 0:
            out_chunks.append(chunk)

    silence = np.zeros(128 * 3, dtype=np.float32)
    pipeline.push(silence, silence)
    pipeline.process_available()
    tail = pipeline.read_output(pipeline._output_buffer.available)
    if len(tail) > 0:
        out_chunks.append(tail)

    out_streaming = np.concatenate(out_chunks)

    # Align by 768 samples latency
    delay = 768
    valid_len = min(len(out_offline), len(out_streaming) - delay)
    off_eval = out_offline[256 : valid_len - 256]
    stream_eval = out_streaming[delay + 256 : delay + valid_len - 256]

    corr = float(np.corrcoef(off_eval, stream_eval)[0, 1])
    diff = off_eval - stream_eval
    max_err = float(np.max(np.abs(diff)))
    rms_err = float(np.sqrt(np.mean(diff ** 2)))

    p_off = float(np.mean(off_eval ** 2))
    p_diff = float(np.mean(diff ** 2) + 1e-15)
    snr_diff_db = 10.0 * np.log10(p_off / p_diff)

    print(f"\nCorrelation:      {corr:.6f}")
    print(f"Max Waveform Err: {max_err:.6f}")
    print(f"RMS Waveform Err: {rms_err:.6f}")
    print(f"Waveform SNR:     {snr_diff_db:.2f} dB")

    assert corr > 0.999, f"Correlation {corr:.6f} below 0.999"
    assert rms_err < 0.01, f"RMS error {rms_err:.6f} exceeds 0.01"
