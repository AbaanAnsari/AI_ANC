"""
evaluation/benchmark_12cases.py
=================================
Ablation and Benchmark Suite for the 12 Synthetic Cases:
    4 Noise Types: Fan (Stationary), Helicopter (Non-Stationary), Siren (Non-Stationary), Drone (Non-Stationary)
    x 3 Input SNRs: 0 dB, 5 dB, 10 dB

Configurations Compared:
    A. Original offline (1s full-sequence scipy STFT, no downstream DSP)
    B. Original streaming (unscaled FFT, 1-frame context, unaligned FIFO, destructive DSP)
    C. Corrected streaming AI-only (exact STFT scaling, 7-frame rolling buffer, phase-aligned, DSP OFF)
    D. Corrected streaming AI + NLMS (corrected AI + NLMS + configurable fusion, limiter OFF)
    E. Corrected full production (corrected AI + NLMS + smoothed VAD + speech protection + limiter)
    F. Oracle AI-only (clean speech directly, DSP OFF)
    G. Oracle full production (clean speech into downstream DSP)

    Plus Oracle DSP Ablations:
    - Oracle + no speech floor
    - Oracle + NLMS disabled
    - Oracle + fusion disabled
"""

from __future__ import annotations

import json
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.audio_loader import load_audio
from src.data.mixer import mix_signals, match_noise_length
from src.data.dual_mic import simulate_dual_mic
from src.features.stft import compute_stft, compute_istft
from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.inference.streaming_pipeline import (
    StreamingPipeline,
    HOP_SIZE,
    N_FFT,
    SAMPLE_RATE,
)
from src.inference.ai_inference import AIInferenceWrapper
from src.project_paths import PHASE3_V2_CHECKPOINT

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("benchmark_12cases")

CHECKPOINT_PATH = PHASE3_V2_CHECKPOINT
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"

NOISE_FILES = {
    "fan": DATASET_ROOT
    / "noise"
    / "stationary"
    / "fan"
    / "section_00_source_train_normal_0008_strength_1_ambient.wav",
    "helicopter": DATASET_ROOT
    / "noise"
    / "non-stationary"
    / "helicopter"
    / "25th_Huey_Helicopter_no_music_01.wav",
    "siren": DATASET_ROOT / "noise" / "non-stationary" / "siren" / "sound_601.wav",
    "drone": DATASET_ROOT
    / "noise"
    / "non-stationary"
    / "drone"
    / "B_S2_D1_067-bebop_000_.wav",
}
CLEAN_SPEECH_FILE = DATASET_ROOT / "clean" / "1098-133695-0000.wav"
SNR_LEVELS = [0.0, 5.0, 10.0]
DEFAULT_BENCHMARK_SEED = 20260929


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def compute_metrics(
    clean: np.ndarray, noisy: np.ndarray, enhanced: np.ndarray, delay: int = 0
) -> Tuple[float, float, float]:
    """Compute input SNR, output SNR, and SNR improvement with delay alignment."""
    min_len = min(len(clean), len(noisy), len(enhanced))
    if delay > 0 and min_len > delay + 256:
        c_eval = clean[: min_len - delay]
        n_eval = noisy[: min_len - delay]
        e_eval = enhanced[delay:min_len]
    else:
        c_eval = clean[:min_len]
        n_eval = noisy[:min_len]
        e_eval = enhanced[:min_len]

    p_clean = float(np.mean(c_eval**2))
    p_noise_in = float(np.mean((n_eval - c_eval) ** 2) + 1e-15)
    p_noise_out = float(np.mean((e_eval - c_eval) ** 2) + 1e-15)

    snr_in = 10.0 * np.log10(p_clean / p_noise_in)
    snr_out = 10.0 * np.log10(p_clean / p_noise_out)
    snr_imp = snr_out - snr_in
    return float(snr_in), float(snr_out), float(snr_imp)


def run_original_offline(
    model: LightweightCNNGRUMaskModel, m1: np.ndarray
) -> np.ndarray:
    """Original offline full-sequence 1-second inference with scipy STFT."""
    stft_c = compute_stft(m1)
    feat = np.stack(
        [stft_c.real.astype(np.float32), stft_c.imag.astype(np.float32)], axis=0
    )[np.newaxis, :, :, :]
    x = torch.from_numpy(feat)
    with torch.no_grad():
        enh_t, _, _ = model(x)
    enh_np = enh_t.squeeze(0).cpu().numpy()
    enh_c = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    return compute_istft(enh_c, length=len(m1))


def run_original_streaming(
    model: LightweightCNNGRUMaskModel, m1: np.ndarray, m2: np.ndarray
) -> np.ndarray:
    """Simulate the original flawed streaming implementation:
    - Raw FFT without 1/win.sum() scaling (+48.16 dB magnitude mismatch)
    - 1-frame context with zero padding at CNN layers
    - 384-sample FIFO delay (24 ms phase error with WOLA)
    - Unstable frame VAD with raw speech protection forcing 50% noisy leakage
    - Fixed 50% fusion weight
    """
    from src.dsp.vad import VoiceActivityDetector
    from src.dsp.nlms import AdaptiveNLMSController
    from src.realtime.ring_buffer import RingBuffer

    win = np.hanning(512).astype(np.float32)
    s_win = np.hanning(512).astype(np.float32)
    cola_norm = np.zeros(128, dtype=np.float32)
    for i in range(128):
        s = 0.0
        for k in range(4):
            s += win[i + k * 128] * s_win[i + k * 128]
        cola_norm[i] = s

    m1_buf = np.zeros(512, dtype=np.float32)
    ola_buf = np.zeros(512, dtype=np.float32)
    fifo = RingBuffer(2048)
    fifo.write(np.zeros(384, dtype=np.float32))  # Old 384-sample FIFO

    nlms = AdaptiveNLMSController()
    vad = VoiceActivityDetector()

    out_chunks = []
    n_hops = len(m1) // 128

    hidden = None
    for h in range(n_hops):
        m1_hop = m1[h * 128 : (h + 1) * 128]
        m2_hop = m2[h * 128 : (h + 1) * 128]

        m1_buf = np.roll(m1_buf, -128)
        m1_buf[-128:] = m1_hop

        # Old flawed STFT: raw rfft without win.sum() normalization
        raw_fft = np.fft.rfft(m1_buf * win, n=512)
        # Old 1-frame feature feeding
        feat = np.stack([raw_fft.real, raw_fft.imag], axis=0)[:, :, np.newaxis]
        x = torch.from_numpy(feat).unsqueeze(0).float()

        with torch.no_grad():
            enh_t, logits_t, _, next_h = model(
                x, hidden_state=hidden, return_hidden=True
            )
            hidden = next_h

        enh_np = enh_t.squeeze(0).cpu().numpy()
        enh_c = enh_np[0, :, 0] + 1j * enh_np[1, :, 0]

        # Old flawed iSTFT: raw irfft
        raw_rec = np.fft.irfft(enh_c, n=512)
        ola_buf += raw_rec * s_win
        ai_hop = ola_buf[:128] / cola_norm
        ola_buf[:-128] = ola_buf[128:]
        ola_buf[-128:] = 0.0

        # Old DSP branch
        dsp_hop, _ = nlms.process_hop(m1_hop, m2_hop)
        fifo.write(dsp_hop)
        aligned_dsp = fifo.read(128)

        # Old VAD & speech protection floor
        vad_res = vad.process_frame(m1_buf)
        sp_prob = vad_res["speech_probability"]
        # Old speech protection: 50% raw noisy signal forced into output
        prot_gain = 0.5 if sp_prob > 0.5 else 0.0
        ai_protected = (1.0 - prot_gain) * ai_hop + prot_gain * m1_hop

        # Old 50% fusion
        fused = 0.5 * ai_protected + 0.5 * aligned_dsp
        out_chunks.append(fused)

    return np.concatenate(out_chunks)


def run_streaming_pipeline(
    pipeline: StreamingPipeline,
    m1: np.ndarray,
    m2: np.ndarray,
    clean: np.ndarray = None,
) -> np.ndarray:
    """Feed audio through StreamingPipeline in 128-sample hops."""
    pipeline.reset()
    out_chunks = []
    n_hops = len(m1) // 128

    for h in range(n_hops):
        m1_h = m1[h * 128 : (h + 1) * 128]
        m2_h = m2[h * 128 : (h + 1) * 128]
        c_h = clean[h * 128 : (h + 1) * 128] if clean is not None else None
        pipeline.push(m1_h, m2_h, clean_samples=c_h)
        pipeline.process_available()
        out = pipeline.read_output(pipeline._output_buffer.available)
        if len(out) > 0:
            out_chunks.append(out)

    # Flush end of stream
    if pipeline.enable_ai:
        silence = np.zeros(128 * 3, dtype=np.float32)
        pipeline.push(
            silence, silence, clean_samples=silence if clean is not None else None
        )
        pipeline.process_available()
        out = pipeline.read_output(pipeline._output_buffer.available)
        if len(out) > 0:
            out_chunks.append(out)

    if not out_chunks:
        return np.zeros_like(m1)
    return np.concatenate(out_chunks)


def run_all_benchmarks(
    checkpoint_path: Path | None = None,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    logger.info("Loading model from %s...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    logger.info("Loading clean speech and noise files...")
    clean_audio_full = load_audio(CLEAN_SPEECH_FILE, target_sr=SAMPLE_RATE)
    # Take a 2.0-second segment (32000 samples) for good steady-state convergence
    seg_len = 32000
    clean_seg = clean_audio_full[:seg_len]

    # Pre-load noise sources
    noise_sources = {}
    source_hashes = {"clean": _sha256_file(CLEAN_SPEECH_FILE)}
    for noise_index, (ntype, npath) in enumerate(NOISE_FILES.items()):
        n_raw = load_audio(npath, target_sr=SAMPLE_RATE)
        noise_sources[ntype] = match_noise_length(
            n_raw, seg_len, rng=np.random.default_rng(seed + noise_index)
        )
        source_hashes[ntype] = _sha256_file(npath)

    # Instantiate pipelines
    pipe_ai_only = StreamingPipeline(ckpt_path, device="cpu", pipeline_mode="ai_only")
    pipe_ai_nlms = StreamingPipeline(
        ckpt_path, device="cpu", pipeline_mode="ai_plus_nlms", nlms_weight=0.2
    )
    pipe_prod = StreamingPipeline(
        ckpt_path, device="cpu", pipeline_mode="full_production"
    )

    # Oracle pipelines
    pipe_oracle_ai = StreamingPipeline(
        ckpt_path,
        device="cpu",
        pipeline_mode="oracle",
        enable_nlms=False,
        enable_fusion=False,
        enable_vad=False,
        enable_limiter=False,
    )
    pipe_oracle_prod = StreamingPipeline(
        ckpt_path,
        device="cpu",
        pipeline_mode="oracle",
        enable_nlms=True,
        enable_fusion=True,
        enable_vad=True,
        enable_limiter=True,
        speech_protection_enabled=True,
    )
    pipe_oracle_no_floor = StreamingPipeline(
        ckpt_path,
        device="cpu",
        pipeline_mode="oracle",
        enable_nlms=True,
        enable_fusion=True,
        enable_vad=True,
        enable_limiter=True,
        speech_protection_enabled=False,
    )
    pipe_oracle_no_nlms = StreamingPipeline(
        ckpt_path,
        device="cpu",
        pipeline_mode="oracle",
        enable_nlms=False,
        enable_fusion=True,
        enable_vad=True,
        enable_limiter=True,
        speech_protection_enabled=True,
    )
    pipe_oracle_no_fusion = StreamingPipeline(
        ckpt_path,
        device="cpu",
        pipeline_mode="oracle",
        enable_nlms=True,
        enable_fusion=False,
        enable_vad=True,
        enable_limiter=True,
        speech_protection_enabled=True,
    )

    results = []

    case_index = 0
    for ntype in ["fan", "helicopter", "siren", "drone"]:
        noise_seg = noise_sources[ntype]
        for snr_target in SNR_LEVELS:
            logger.info(
                "Evaluating Case: Noise=%s, Input SNR=%.1f dB...", ntype, snr_target
            )
            case_seed = seed + 100 + case_index
            case_rng = np.random.default_rng(case_seed)
            noisy_speech, scaled_noise = mix_signals(
                clean_seg, noise_seg, target_snr_db=snr_target, rng=case_rng
            )
            m1, m2 = simulate_dual_mic(
                clean_speech=clean_seg,
                scaled_noise=scaled_noise,
                relative_delay_samples=5,
                rng=case_rng,
            )

            # A. Original Offline
            out_a = run_original_offline(model, m1)
            sin_a, sout_a, imp_a = compute_metrics(clean_seg, m1, out_a, delay=0)

            # B. Original Streaming
            out_b = run_original_streaming(model, m1, m2)
            sin_b, sout_b, imp_b = compute_metrics(clean_seg, m1, out_b, delay=384)

            # C. Corrected Streaming AI-Only
            out_c = run_streaming_pipeline(pipe_ai_only, m1, m2)
            sin_c, sout_c, imp_c = compute_metrics(
                clean_seg, m1, out_c, delay=pipe_ai_only.algorithmic_delay_samples
            )

            # D. Corrected Streaming AI + NLMS
            out_d = run_streaming_pipeline(pipe_ai_nlms, m1, m2)
            sin_d, sout_d, imp_d = compute_metrics(
                clean_seg, m1, out_d, delay=pipe_ai_nlms.algorithmic_delay_samples
            )

            # E. Corrected Full Production
            out_e = run_streaming_pipeline(pipe_prod, m1, m2)
            sin_e, sout_e, imp_e = compute_metrics(
                clean_seg, m1, out_e, delay=pipe_prod.algorithmic_delay_samples
            )

            # F. Oracle AI-Only
            out_f = run_streaming_pipeline(pipe_oracle_ai, m1, m2, clean=clean_seg)
            sin_f, sout_f, imp_f = compute_metrics(
                clean_seg, m1, out_f, delay=pipe_oracle_ai.algorithmic_delay_samples
            )

            # G. Oracle Full Production
            out_g = run_streaming_pipeline(pipe_oracle_prod, m1, m2, clean=clean_seg)
            sin_g, sout_g, imp_g = compute_metrics(
                clean_seg, m1, out_g, delay=pipe_oracle_prod.algorithmic_delay_samples
            )

            # Oracle Ablations
            out_o_no_floor = run_streaming_pipeline(
                pipe_oracle_no_floor, m1, m2, clean=clean_seg
            )
            _, _, imp_o_no_floor = compute_metrics(
                clean_seg,
                m1,
                out_o_no_floor,
                delay=pipe_oracle_no_floor.algorithmic_delay_samples,
            )

            out_o_no_nlms = run_streaming_pipeline(
                pipe_oracle_no_nlms, m1, m2, clean=clean_seg
            )
            _, _, imp_o_no_nlms = compute_metrics(
                clean_seg,
                m1,
                out_o_no_nlms,
                delay=pipe_oracle_no_nlms.algorithmic_delay_samples,
            )

            out_o_no_fusion = run_streaming_pipeline(
                pipe_oracle_no_fusion, m1, m2, clean=clean_seg
            )
            _, _, imp_o_no_fusion = compute_metrics(
                clean_seg,
                m1,
                out_o_no_fusion,
                delay=pipe_oracle_no_fusion.algorithmic_delay_samples,
            )

            case_data = {
                "noise_type": ntype,
                "target_snr_db": snr_target,
                "seed": case_seed,
                "actual_input_snr_db": round(sin_c, 2),
                "A_orig_offline": round(imp_a, 2),
                "B_orig_streaming": round(imp_b, 2),
                "C_corrected_ai_only": round(imp_c, 2),
                "D_corrected_ai_nlms": round(imp_d, 2),
                "E_corrected_full_prod": round(imp_e, 2),
                "F_oracle_ai_only": round(imp_f, 2),
                "G_oracle_full_prod": round(imp_g, 2),
                "oracle_no_floor": round(imp_o_no_floor, 2),
                "oracle_no_nlms": round(imp_o_no_nlms, 2),
                "oracle_no_fusion": round(imp_o_no_fusion, 2),
                "class_switches": pipe_prod.diagnostics["class_switch_count"],
            }
            results.append(case_data)
            case_index += 1

    return {
        "benchmark": "synthetic_12_case",
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": _sha256_file(ckpt_path),
        "seed": seed,
        "source_paths": {
            "clean": CLEAN_SPEECH_FILE.relative_to(PROJECT_ROOT).as_posix(),
            **{
                name: path.relative_to(PROJECT_ROOT).as_posix()
                for name, path in NOISE_FILES.items()
            },
        },
        "source_sha256": source_hashes,
        "cases": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="12-Case Benchmark Suite")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "benchmark_12cases_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BENCHMARK_SEED,
        help="Deterministic benchmark seed",
    )
    args = parser.parse_args()

    benchmark_data = run_all_benchmarks(
        checkpoint_path=Path(args.checkpoint), seed=args.seed
    )
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info("Saved benchmark results to %s", out_file)
