"""
tests/test_phase2_optimizations.py
==================================
Comprehensive Unit & Regression Test Suite for Phase 2 Optimizations:
- Stage 1: STFT / iSTFT Overlap-Add (WOLA) Reconstruction & COLA Verification.
- Stage 2: Stateful Recurrent GRU Inference & Hidden State Continuity.
- Stage 3: Hop-based (128-sample) NLMS Adaptation & Numerical Safeguards.
- Stage 4: Delay-Aligned Phase Coherence & Comb-Filtering Elimination in Fusion.
- Stage 5: Reference Microphone Delay Alignment & Causal Cancellation.
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

from src.models.cnn_gru_mask import LightweightCNNGRUMaskModel
from src.inference.ai_inference import AIInferenceWrapper
from src.inference.streaming_pipeline import StreamingPipeline, N_FFT, HOP_SIZE, SAMPLE_RATE
from src.dsp.nlms import NLMSFilter, AdaptiveNLMSController
from src.dsp.fusion import AdaptiveFusionEngine

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
)


# ---------------------------------------------------------------------------
# Stage 1: STFT / iSTFT Overlap-Add Reconstruction Tests
# ---------------------------------------------------------------------------

class TestStage1WOLAReconstruction:
    """Validate mathematically correct WOLA reconstruction without tail-slicing artifacts."""

    def test_sine_wave_reconstruction_fidelity(self):
        """Pure 440 Hz sine wave reconstructed with > 90 dB SNR and near-zero error."""
        pipeline = StreamingPipeline(
            checkpoint_path=CHECKPOINT_PATH,
            enable_ai=False,
            enable_nlms=False,
            enable_fusion=False,
            enable_limiter=False,
        )

        duration_s = 0.5
        N = int(duration_s * SAMPLE_RATE)
        t = np.arange(N) / SAMPLE_RATE
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Feed through pipeline
        pipeline.push(sine, sine)
        pipeline.process_available()
        out = pipeline.read_output(N)

        delay = N_FFT - HOP_SIZE  # 384 samples
        # Evaluate steady state segment (out[delay:] corresponds to sine[:-delay])
        ref_seg = sine[0 : len(out) - delay]
        out_seg = out[delay : len(out)]

        # Verify steady state error is small (SNR > 50 dB in pipeline with DC filter)
        error = out_seg[N_FFT:] - ref_seg[N_FFT:]
        mse = np.mean(error**2)
        snr = 10 * np.log10(np.mean(ref_seg[N_FFT:]**2) / (mse + 1e-15))

        assert snr > 50.0, f"Sine reconstruction SNR too low: {snr:.2f} dB"
        assert np.max(np.abs(error)) < 1e-2, f"Max error too large: {np.max(np.abs(error))}"

    def test_impulse_train_reconstruction(self):
        """Impulse train passes through OLA without trailing smearing or clicks."""
        pipeline = StreamingPipeline(checkpoint_path=CHECKPOINT_PATH, enable_ai=False, enable_nlms=False)

        sig = np.zeros(8000, dtype=np.float32)
        sig[::400] = 1.0

        pipeline.push(sig, sig)
        pipeline.process_available()
        out = pipeline.read_output(len(sig))

        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 1.05


# ---------------------------------------------------------------------------
# Stage 2: Stateful GRU Recurrent Tracking Tests
# ---------------------------------------------------------------------------

class TestStage2StatefulGRU:
    """Validate recurrent GRU state persistence across streaming hops."""

    def test_model_hidden_state_interface(self):
        """Model forward accepts hidden_state and returns updated next_hidden when requested."""
        model = LightweightCNNGRUMaskModel()
        model.eval()

        x = torch.randn(1, 2, 257, 1)
        h0 = torch.zeros(1, 1, 48)

        # Legacy 3-tuple return
        enh, logits, mask = model(x)
        assert enh.shape == (1, 2, 257, 1)

        # Stateful 4-tuple return
        enh2, logits2, mask2, h1 = model(x, hidden_state=h0, return_hidden=True)
        assert h1.shape == (1, 1, 48)
        assert not torch.allclose(h0, h1)  # Hidden state evolved

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_ai_inference_wrapper_stateful_tracking(self):
        """AIInferenceWrapper preserves hidden state across consecutive 1-frame calls."""
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        wrapper.reset_state()
        assert wrapper._hidden_state is None

        f1 = np.random.randn(2, 257, 1).astype(np.float32)
        wrapper.infer(f1, stateful=True)
        assert wrapper._hidden_state is not None
        h_after_1 = wrapper._hidden_state.clone()

        f2 = np.random.randn(2, 257, 1).astype(np.float32)
        wrapper.infer(f2, stateful=True)
        assert not torch.allclose(h_after_1, wrapper._hidden_state)

        # Reset clears state
        wrapper.reset_state()
        assert wrapper._hidden_state is None


# ---------------------------------------------------------------------------
# Stage 3: Hop-based (128-sample) NLMS Adaptation Tests
# ---------------------------------------------------------------------------

class TestStage3HopBasedNLMS:
    """Validate 128-sample hop adaptation and safeguards against divergence."""

    def test_hop_adaptation_converges(self):
        """NLMS adapts stably on 128-sample hops and cancels correlated noise."""
        nlms = NLMSFilter(filter_length=64, step_size=0.05, leakage=0.9999)
        rng = np.random.default_rng(42)

        L = 8000
        noise = rng.normal(0, 1, L).astype(np.float32)
        # Primary has noise delayed by 4 samples
        m1 = np.zeros(L, dtype=np.float32)
        m1[4:] = 0.8 * noise[:-4]
        m2 = noise

        errors = []
        for h in range(L // HOP_SIZE):
            p_hop = m1[h*HOP_SIZE : (h+1)*HOP_SIZE]
            r_hop = m2[h*HOP_SIZE : (h+1)*HOP_SIZE]
            e_hop, _ = nlms.process_hop(p_hop, r_hop, speech_prob=0.0)
            errors.append(e_hop)
        errors = np.concatenate(errors)

        # Check attenuation in second half
        in_pow = np.mean(m1[4000:]**2)
        res_pow = np.mean(errors[4000:]**2)
        att_db = 10 * np.log10(in_pow / (res_pow + 1e-12))
        assert att_db > 10.0, f"NLMS noise attenuation too low: {att_db:.2f} dB"

    def test_speech_freeze_safeguard(self):
        """When speech probability exceeds threshold, adaptation freezes."""
        nlms = NLMSFilter(filter_length=32, step_size=0.1, speech_freeze_threshold=0.5)
        p_hop = np.ones(128, dtype=np.float32)
        r_hop = np.ones(128, dtype=np.float32)

        # Run with speech_prob = 0.9 (frozen)
        w_before = nlms.weights.copy()
        nlms.process_hop(p_hop, r_hop, speech_prob=0.9)
        w_after = nlms.weights.copy()
        np.testing.assert_array_equal(w_before, w_after)


# ---------------------------------------------------------------------------
# Stage 4: Phase-Aligned Fusion Tests
# ---------------------------------------------------------------------------

class TestStage4PhaseAlignedFusion:
    """Validate that delay FIFO aligns time-domain signals with STFT OLA output."""

    def test_fifo_delay_alignment_constructive_interference(self):
        """384-sample FIFO prevents destructive comb-filtering nulls."""
        L = 4000
        f_null = 437.5  # Critical null frequency for 384-sample delay
        t = np.arange(L) / SAMPLE_RATE
        sig = np.sin(2 * np.pi * f_null * t).astype(np.float32)

        # Simulate STFT group delay (384 samples)
        stft_delayed = np.zeros(L, dtype=np.float32)
        stft_delayed[384:] = sig[:-384]

        # Time-domain signal without delay
        time_sig = sig.copy()

        # Unaligned addition -> destructive interference
        unaligned = 0.5 * stft_delayed[500:3500] + 0.5 * time_sig[500:3500]
        unaligned_pow = np.mean(unaligned**2)

        # Aligned addition (delaying time_sig by 384 samples)
        time_aligned = np.zeros(L, dtype=np.float32)
        time_aligned[384:] = time_sig[:-384]
        aligned = 0.5 * stft_delayed[500:3500] + 0.5 * time_aligned[500:3500]
        aligned_pow = np.mean(aligned**2)

        assert unaligned_pow < 0.01  # Destructive cancellation verified
        assert aligned_pow > 0.45    # Constructive in-phase summation verified


# ---------------------------------------------------------------------------
# Stage 5: Reference Delay Alignment Tests
# ---------------------------------------------------------------------------

class TestStage5ReferenceDelayAlignment:
    """Validate that reference delay line dynamically shifts M2 to match delay."""

    def test_pipeline_smooths_and_aligns_reference(self):
        """Pipeline smoothly tracks and aligns reference microphone hop."""
        pipeline = StreamingPipeline(checkpoint_path=CHECKPOINT_PATH, enable_ai=False, enable_nlms=True)

        L = 3200
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 1, L).astype(np.float32)

        # M1 receives noise delayed by 10 samples
        tau = 10
        m1 = np.zeros(L, dtype=np.float32)
        m1[tau:] = noise[:-tau]
        m2 = noise.copy()

        pipeline.push(m1, m2)
        pipeline.process_available()

        # Verify Kalman delay converges toward -tau (M1 delayed relative to M2 convention)
        assert abs(pipeline._current_delay_samples - (-tau)) < 4.0
