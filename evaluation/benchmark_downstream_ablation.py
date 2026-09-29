"""
evaluation/benchmark_downstream_ablation.py
=============================================
Strict Downstream DSP Ablation Matrix (Configurations A through K)
and 12-Case Synthetic Evaluation for Checkpoint D.

Evaluates:
    4 Noise Types: Fan, Helicopter, Siren, Drone
    x 3 Input SNRs: 0 dB, 5 dB, 10 dB
    = 12 Cases

Ablation Configurations:
    A. AI only
    B. AI + fusion, NLMS branch disabled (w_dsp = 0.5)
    C. AI + NLMS, fusion weight = 0.0
    D. AI + NLMS, fusion weight = 0.1
    E. AI + NLMS, fusion weight = 0.2
    F. AI + NLMS, fusion weight = 0.3
    G. AI + NLMS, fusion weight = 0.5
    H. AI + speech protection only
    I. AI + limiter only
    J. AI + VAD only
    K. Full production
    + Baseline Comparisons: Original Offline, Original Streaming, Oracle AI-only, Oracle Full Production
"""
from __future__ import annotations

import json
import logging
import math
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
from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE, N_FFT, SAMPLE_RATE
from evaluation.snr import compute_snr, compute_snr_improvement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation_benchmark")

CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "phase3_D" / "best_checkpoint.pt"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "dataset"

NOISE_FILES = {
    "fan": DATASET_ROOT / "noise" / "stationary" / "fan" / "section_00_source_train_normal_0008_strength_1_ambient.wav",
    "helicopter": DATASET_ROOT / "noise" / "non-stationary" / "helicopter" / "25th_Huey_Helicopter_no_music_01.wav",
    "siren": DATASET_ROOT / "noise" / "non-stationary" / "siren" / "sound_601.wav",
    "drone": DATASET_ROOT / "noise" / "non-stationary" / "drone" / "B_S2_D1_067-bebop_000_.wav",
}
CLEAN_SPEECH_FILE = DATASET_ROOT / "clean" / "1098-133695-0000.wav"
SNR_LEVELS = [0.0, 5.0, 10.0]


def run_streaming(pipe: StreamingPipeline, m1: np.ndarray, m2: np.ndarray, clean: np.ndarray = None) -> np.ndarray:
    pipe.reset()
    out_chunks = []
    n_hops = len(m1) // HOP_SIZE

    for h in range(n_hops):
        m1_h = m1[h * HOP_SIZE : (h + 1) * HOP_SIZE]
        m2_h = m2[h * HOP_SIZE : (h + 1) * HOP_SIZE]
        c_h = clean[h * HOP_SIZE : (h + 1) * HOP_SIZE] if clean is not None else None
        pipe.push(m1_h, m2_h, clean_samples=c_h)
        pipe.process_available()
        out = pipe.read_output(pipe._output_buffer.available)
        if len(out) > 0:
            out_chunks.append(out)

    # Flush lookahead
    if pipe.enable_ai:
        silence = np.zeros(HOP_SIZE * 3, dtype=np.float32)
        pipe.push(silence, silence, clean_samples=silence if clean is not None else None)
        pipe.process_available()
        out = pipe.read_output(pipe._output_buffer.available)
        if len(out) > 0:
            out_chunks.append(out)

    if not out_chunks:
        return np.zeros_like(m1)
    return np.concatenate(out_chunks)


def run_original_offline(model: LightweightCNNGRUMaskModel, m1: np.ndarray) -> np.ndarray:
    stft_c = compute_stft(m1)
    feat = np.stack([stft_c.real.astype(np.float32), stft_c.imag.astype(np.float32)], axis=0)[np.newaxis, :, :, :]
    x = torch.from_numpy(feat)
    with torch.no_grad():
        enh_t, _, _ = model(x)
    enh_np = enh_t.squeeze(0).cpu().numpy()
    enh_c = (enh_np[0] + 1j * enh_np[1]).astype(np.complex64)
    return compute_istft(enh_c, length=len(m1))


def run_original_streaming(model: LightweightCNNGRUMaskModel, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
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
    fifo.write(np.zeros(384, dtype=np.float32))

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

        raw_fft = np.fft.rfft(m1_buf * win, n=512)
        feat = np.stack([raw_fft.real, raw_fft.imag], axis=0)[:, :, np.newaxis]
        x = torch.from_numpy(feat).unsqueeze(0).float()

        with torch.no_grad():
            enh_t, logits_t, _, next_h = model(x, hidden_state=hidden, return_hidden=True)
            hidden = next_h

        enh_np = enh_t.squeeze(0).cpu().numpy()
        enh_c = enh_np[0, :, 0] + 1j * enh_np[1, :, 0]

        raw_rec = np.fft.irfft(enh_c, n=512)
        ola_buf += raw_rec * s_win
        ai_hop = ola_buf[:128] / cola_norm
        ola_buf[:-128] = ola_buf[128:]
        ola_buf[-128:] = 0.0

        dsp_hop, _ = nlms.process_hop(m1_hop, m2_hop)
        fifo.write(dsp_hop)
        aligned_dsp = fifo.read(128)

        vad_res = vad.process_frame(m1_buf)
        sp_prob = vad_res["speech_probability"]
        prot_gain = 0.5 if sp_prob > 0.5 else 0.0
        ai_protected = (1.0 - prot_gain) * ai_hop + prot_gain * m1_hop

        fused = 0.5 * ai_protected + 0.5 * aligned_dsp
        out_chunks.append(fused)

    return np.concatenate(out_chunks)


def eval_aligned_snr(clean: np.ndarray, noisy: np.ndarray, out: np.ndarray, delay: int = 768) -> Tuple[float, float, float]:
    min_len = min(len(clean), len(noisy), len(out))
    if delay > 0 and min_len > delay + 256:
        c_eval = clean[: min_len - delay]
        n_eval = noisy[: min_len - delay]
        e_eval = out[delay : min_len]
    else:
        c_eval = clean[:min_len]
        n_eval = noisy[:min_len]
        e_eval = out[:min_len]

    sin = compute_snr(c_eval, n_eval)
    sout = compute_snr(c_eval, e_eval)
    simp = sout - sin if not math.isinf(sout) else float("inf")
    return sin, sout, simp


def run_full_ablation_suite(checkpoint_path: Path | None = None) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    logger.info("Loading model from %s...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = LightweightCNNGRUMaskModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info("Loading clean audio and 4 noise sources...")
    clean_audio_full = load_audio(CLEAN_SPEECH_FILE, target_sr=SAMPLE_RATE)
    seg_len = 32000
    clean_seg = clean_audio_full[:seg_len]

    noise_sources = {}
    for ntype, npath in NOISE_FILES.items():
        n_raw = load_audio(npath, target_sr=SAMPLE_RATE)
        noise_sources[ntype] = match_noise_length(n_raw, seg_len)

    # 11 Ablation configurations
    configs = {
        "A_ai_only": dict(enable_ai=True, enable_nlms=False, enable_vad=False, enable_fusion=False, speech_protection_enabled=False, enable_limiter=False),
        "B_ai_plus_fusion_no_nlms": dict(enable_ai=True, enable_nlms=False, enable_vad=False, enable_fusion=True, nlms_weight=0.5, speech_protection_enabled=False, enable_limiter=False),
        "C_ai_nlms_w00": dict(enable_ai=True, enable_nlms=True, enable_vad=False, enable_fusion=True, nlms_weight=0.0, speech_protection_enabled=False, enable_limiter=False),
        "D_ai_nlms_w01": dict(enable_ai=True, enable_nlms=True, enable_vad=False, enable_fusion=True, nlms_weight=0.1, speech_protection_enabled=False, enable_limiter=False),
        "E_ai_nlms_w02": dict(enable_ai=True, enable_nlms=True, enable_vad=False, enable_fusion=True, nlms_weight=0.2, speech_protection_enabled=False, enable_limiter=False),
        "F_ai_nlms_w03": dict(enable_ai=True, enable_nlms=True, enable_vad=False, enable_fusion=True, nlms_weight=0.3, speech_protection_enabled=False, enable_limiter=False),
        "G_ai_nlms_w05": dict(enable_ai=True, enable_nlms=True, enable_vad=False, enable_fusion=True, nlms_weight=0.5, speech_protection_enabled=False, enable_limiter=False),
        "H_ai_speech_prot_only": dict(enable_ai=True, enable_nlms=False, enable_vad=True, enable_fusion=True, nlms_weight=0.0, speech_protection_enabled=True, enable_limiter=False),
        "I_ai_limiter_only": dict(enable_ai=True, enable_nlms=False, enable_vad=False, enable_fusion=False, speech_protection_enabled=False, enable_limiter=True),
        "J_ai_vad_only": dict(enable_ai=True, enable_nlms=False, enable_vad=True, enable_fusion=False, speech_protection_enabled=False, enable_limiter=False),
        "K_full_production": dict(enable_ai=True, enable_nlms=True, enable_vad=True, enable_fusion=True, nlms_weight=None, speech_protection_enabled=True, enable_limiter=True),
    }

    pipes = {k: StreamingPipeline(ckpt_path, device="cpu", **v) for k, v in configs.items()}
    
    # Oracle pipelines
    pipe_oracle_ai = StreamingPipeline(CHECKPOINT_PATH, device="cpu", pipeline_mode="oracle", enable_nlms=False, enable_fusion=False, enable_vad=False, enable_limiter=False)
    pipe_oracle_prod = StreamingPipeline(CHECKPOINT_PATH, device="cpu", pipeline_mode="oracle", enable_nlms=True, enable_fusion=True, enable_vad=True, enable_limiter=True, speech_protection_enabled=True)

    all_case_results = []

    for ntype in ["fan", "helicopter", "siren", "drone"]:
        noise_seg = noise_sources[ntype]
        for snr_target in SNR_LEVELS:
            logger.info("Running ablation for Noise=%s, Input SNR=%.1f dB...", ntype, snr_target)
            noisy_speech, scaled_noise = mix_signals(clean_seg, noise_seg, target_snr_db=snr_target)
            m1, m2 = simulate_dual_mic(clean_speech=clean_seg, scaled_noise=scaled_noise, relative_delay_samples=5)

            # Baselines
            out_orig_off = run_original_offline(model, m1)
            _, _, imp_orig_off = eval_aligned_snr(clean_seg, m1, out_orig_off, delay=0)

            out_orig_stream = run_original_streaming(model, m1, m2)
            _, _, imp_orig_stream = eval_aligned_snr(clean_seg, m1, out_orig_stream, delay=384)

            out_oracle_ai = run_streaming(pipe_oracle_ai, m1, m2, clean=clean_seg)
            _, _, imp_oracle_ai = eval_aligned_snr(clean_seg, m1, out_oracle_ai, delay=768)

            out_oracle_prod = run_streaming(pipe_oracle_prod, m1, m2, clean=clean_seg)
            _, _, imp_oracle_prod = eval_aligned_snr(clean_seg, m1, out_oracle_prod, delay=768)

            case_dict = {
                "noise_type": ntype,
                "target_snr_db": snr_target,
                "orig_offline": round(imp_orig_off, 2),
                "orig_streaming": round(imp_orig_stream, 2),
                "oracle_ai_only": "inf" if math.isinf(imp_oracle_ai) else round(imp_oracle_ai, 2),
                "oracle_full_prod": round(imp_oracle_prod, 2),
                "ablations": {},
            }

            for cfg_name, pipe in pipes.items():
                out = run_streaming(pipe, m1, m2)
                sin, sout, simp = eval_aligned_snr(clean_seg, m1, out, delay=pipe.algorithmic_delay_samples)
                diag = pipe.diagnostics
                
                # Limiter gain reduction metric
                lim_red_db = 0.0
                if diag["limiter_enabled"] and diag["limiter_clipping_count"] > 0:
                    lim_red_db = round(float(20.0 * np.log10(max(1e-6, np.max(np.abs(out)) / max(1e-6, np.max(np.abs(m1)))))), 2)

                case_dict["ablations"][cfg_name] = {
                    "input_snr_db": round(sin, 2),
                    "output_snr_db": round(sout, 2) if not math.isinf(sout) else "inf",
                    "snr_improvement_db": round(simp, 2) if not math.isinf(simp) else "inf",
                    "output_rms": diag["fused_output_rms"],
                    "ai_branch_rms": diag["ai_output_rms"],
                    "nlms_branch_rms": diag["nlms_output_rms"],
                    "w_ai": diag["fusion_weights"].get("ai_weight", 1.0),
                    "w_dsp": diag["fusion_weights"].get("dsp_weight", 0.0),
                    "vad_probability": diag["smoothed_speech_prob"],
                    "speech_protection_state": diag["protection_floor_state"],
                    "applied_attenuation_floor_db": diag["applied_attenuation_floor_db"],
                    "limiter_clipping_count": diag["limiter_clipping_count"],
                    "limiter_gain_reduction_db": lim_red_db,
                }

            all_case_results.append(case_dict)

    return {"cases": all_case_results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Downstream DSP Ablation Suite")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "evaluation" / "benchmark_downstream_ablation_results.json"), help="Output JSON path")
    args = parser.parse_args()

    results = run_full_ablation_suite(checkpoint_path=Path(args.checkpoint))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved downstream ablation benchmark results to %s", out_path)
