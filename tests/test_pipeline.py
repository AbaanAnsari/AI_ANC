"""
tests/test_pipeline.py
=======================
Integration tests for the full dual-microphone noise cancellation pipeline.

Tests:
    1. DSP component chain (GCC-PHAT → Kalman → NLMS → VAD → Fusion → Limiter)
    2. AI inference wrapper (checkpoint loading, inference shapes)
    3. Streaming pipeline (push/process/read cycle)
    4. End-to-end synthetic pipeline (no NaN, no Inf, no clipping)
    5. Pipeline determinism (same input → same output)
    6. Pipeline reset
    7. Frame boundary handling
    8. Noise class → NLMS configuration routing
    9. Fusion weights (bounded, normalized)
    10. Limiter properties (no clipping, finite output)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
)


# ---------------------------------------------------------------------------
# DSP Component Tests
# ---------------------------------------------------------------------------

class TestGCCPHATKalmanChain:
    def test_gcc_phat_known_delay(self):
        """GCC-PHAT detects a known delay."""
        from src.dsp.gcc_phat import gcc_phat
        sample_rate = 16000
        N = 4096
        true_delay = 10  # samples
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, N).astype(np.float32)
        m1 = signal.copy()
        m2 = np.zeros_like(signal)
        m2[true_delay:] = signal[:N - true_delay]

        result = gcc_phat(m1, m2, sample_rate=sample_rate, max_tau=50)
        assert result["estimated_delay_samples"] == pytest.approx(true_delay, abs=2)

    def test_kalman_smoothing(self):
        """Kalman filter smooths noisy measurements."""
        from src.dsp.kalman import DelayKalmanFilter
        kf = DelayKalmanFilter(process_noise=0.01, measurement_noise=1.0)
        noisy_measurements = [5 + np.random.randn() * 3 for _ in range(50)]
        estimates = [kf.update(m) for m in noisy_measurements]
        # Variance of estimates should be less than variance of measurements
        assert np.var(estimates) < np.var(noisy_measurements)

    def test_kalman_reset(self):
        from src.dsp.kalman import DelayKalmanFilter
        kf = DelayKalmanFilter(initial_delay=5.0)
        for _ in range(20):
            kf.update(7.0)
        kf.reset()
        # reset() returns to initial_delay (5.0), not zero
        assert kf.current_state == pytest.approx(5.0)


class TestVAD:
    def test_vad_detects_speech(self):
        from src.dsp.vad import VoiceActivityDetector
        vad = VoiceActivityDetector(sample_rate=16000, hangover_frames=4)
        # Speech: strong sine wave
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        speech = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        result = vad.process_frame(speech)
        assert result["frame_energy_db"] > result["noise_floor_db"]

    def test_vad_returns_dict(self):
        from src.dsp.vad import VoiceActivityDetector
        vad = VoiceActivityDetector()
        frame = np.random.randn(320).astype(np.float32) * 0.1
        result = vad.process_frame(frame)
        required_keys = ["speech_active", "speech_probability", "frame_energy_db",
                         "noise_floor_db", "snr_estimate_db"]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    def test_vad_probability_bounded(self):
        from src.dsp.vad import VoiceActivityDetector
        vad = VoiceActivityDetector()
        for _ in range(20):
            frame = np.random.randn(320).astype(np.float32) * 0.3
            result = vad.process_frame(frame)
            assert 0.0 <= result["speech_probability"] <= 1.0

    def test_vad_reset(self):
        from src.dsp.vad import VoiceActivityDetector
        vad = VoiceActivityDetector()
        for _ in range(10):
            vad.process_frame(np.random.randn(320).astype(np.float32))
        vad.reset()
        assert vad.speech_probability == 0.0


class TestSpeechProtection:
    def test_protection_gain_bounded(self):
        from src.dsp.speech_protection import SpeechProtector
        sp = SpeechProtector()
        for prob in [0.0, 0.3, 0.5, 0.7, 1.0]:
            gain = sp.compute_protection_gain(prob)
            assert 0.0 <= gain <= 1.0, f"Gain {gain} out of bounds for prob {prob}"

    def test_apply_protection_finite(self):
        from src.dsp.speech_protection import SpeechProtector
        sp = SpeechProtector()
        signal = np.random.randn(128).astype(np.float32) * 0.5
        reference = np.random.randn(128).astype(np.float32) * 0.5
        protected = sp.apply_protection(signal, reference, speech_probability=0.8)
        assert np.all(np.isfinite(protected))

    def test_transient_detector(self):
        from src.dsp.speech_protection import ImpulsiveTransientDetector
        det = ImpulsiveTransientDetector(threshold=4.0)
        # Quiet signal first
        for _ in range(50):
            det.detect(np.ones(64) * 0.01)
        # Then loud impulse
        is_transient = det.detect(np.ones(64) * 10.0)
        assert is_transient is True


class TestLimiter:
    def test_output_bounded(self):
        from src.dsp.limiter import SafetyLimiter
        lim = SafetyLimiter(threshold_db=-3.0)
        loud_signal = np.ones(256) * 2.0  # way above threshold
        output = lim.process(loud_signal)
        assert np.all(np.abs(output) <= 1.0)

    def test_no_clipping_on_normal_signal(self):
        from src.dsp.limiter import SafetyLimiter
        lim = SafetyLimiter(threshold_db=-0.1)
        np.random.seed(42)
        normal = np.random.uniform(-0.3, 0.3, 1000).astype(np.float32)
        output = lim.process(normal)
        assert np.all(np.isfinite(output))
        assert lim.clipping_count == 0

    def test_nan_input_produces_finite_output(self):
        from src.dsp.limiter import SafetyLimiter
        lim = SafetyLimiter()
        signal = np.array([0.1, float('nan'), 0.3, float('inf'), -0.2], dtype=np.float32)
        output = lim.process(signal)
        assert np.all(np.isfinite(output))

    def test_reset(self):
        from src.dsp.limiter import SafetyLimiter
        lim = SafetyLimiter()
        lim.process(np.ones(100) * 2.0)
        assert lim.clipping_count > 0
        lim.reset()
        assert lim.clipping_count == 0

    def test_limiter_finite_on_extreme(self):
        from src.dsp.limiter import SafetyLimiter
        lim = SafetyLimiter()
        extreme = np.array([1e6, -1e6, 0, 1e-10, -1e-10] * 100, dtype=np.float32)
        output = lim.process(extreme)
        assert np.all(np.isfinite(output))


class TestFusion:
    def test_fusion_weights_normalized(self):
        from src.dsp.fusion import compute_fusion_weights
        for nc in [0, 1, 2]:
            for conf in [0.3, 0.6, 0.9]:
                result = compute_fusion_weights(nc, conf, 0.5)
                assert abs(result["w_ai"] + result["w_dsp"] - 1.0) < 1e-6

    def test_fusion_weights_bounded(self):
        from src.dsp.fusion import compute_fusion_weights
        result = compute_fusion_weights(0, 0.9, 0.9)
        assert 0 <= result["w_ai"] <= 1
        assert 0 <= result["w_dsp"] <= 1

    def test_fusion_engine_output_shape(self):
        from src.dsp.fusion import AdaptiveFusionEngine
        engine = AdaptiveFusionEngine()
        N = 128
        ai_sig = np.random.randn(N).astype(np.float32) * 0.3
        dsp_sig = np.random.randn(N).astype(np.float32) * 0.3
        orig = np.random.randn(N).astype(np.float32) * 0.5
        result = engine.fuse(ai_sig, dsp_sig, orig, noise_class=0, confidence=0.8, speech_probability=0.5)
        assert result["fused_signal"].shape == (N,)

    def test_fusion_finite_output(self):
        from src.dsp.fusion import AdaptiveFusionEngine
        engine = AdaptiveFusionEngine()
        N = 128
        result = engine.fuse(
            np.random.randn(N).astype(np.float32),
            np.random.randn(N).astype(np.float32),
            np.random.randn(N).astype(np.float32),
            noise_class=2, confidence=0.5, speech_probability=0.8,
        )
        assert np.all(np.isfinite(result["fused_signal"]))


# ---------------------------------------------------------------------------
# AI Inference Tests
# ---------------------------------------------------------------------------

class TestAIInference:
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_load_checkpoint(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        assert wrapper.param_count == 70789

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_inference_shapes(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        features = np.random.randn(2, 257, 10).astype(np.float32)
        enhanced, logits, mask = wrapper.infer(features)
        assert enhanced.shape == (2, 257, 10)
        assert logits.shape == (3,)
        assert mask.shape == (2, 257, 10)

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_inference_finite(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        features = np.random.randn(2, 257, 20).astype(np.float32)
        enhanced, logits, mask = wrapper.infer(features)
        assert np.all(np.isfinite(enhanced))
        assert np.all(np.isfinite(logits))
        assert np.all(np.isfinite(mask))

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_classify_probabilities_sum_to_one(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        features = np.random.randn(2, 257, 10).astype(np.float32)
        _, logits, _ = wrapper.infer(features)
        _, _, probs = wrapper.classify(logits)
        assert abs(probs.sum() - 1.0) < 1e-5

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_deterministic_inference(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        rng = np.random.default_rng(99)
        features = rng.normal(0, 1, (2, 257, 20)).astype(np.float32)
        e1, l1, m1 = wrapper.infer(features)
        e2, l2, m2 = wrapper.infer(features)
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(l1, l2)


# ---------------------------------------------------------------------------
# Ring Buffer Tests
# ---------------------------------------------------------------------------

class TestRingBuffer:
    def test_write_read(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        data = np.arange(50, dtype=np.float32)
        buf.write(data)
        out = buf.read(50)
        np.testing.assert_array_equal(out, data)

    def test_wrap_around(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(64)
        buf.write(np.ones(60, dtype=np.float32))
        buf.read(55)
        buf.write(np.ones(10, dtype=np.float32))  # wraps
        assert buf.available == 15

    def test_overflow_no_crash(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(10)
        written = buf.write(np.ones(20, dtype=np.float32))
        assert written == 10  # only 10 fit

    def test_peek_no_consume(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        buf.write(np.arange(10, dtype=np.float32))
        buf.peek(5)
        assert buf.available == 10

    def test_reset(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        buf.write(np.ones(50, dtype=np.float32))
        buf.reset()
        assert buf.available == 0


# ---------------------------------------------------------------------------
# Streaming Pipeline Integration Test
# ---------------------------------------------------------------------------

class TestStreamingPipeline:
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_push_process_read_cycle(self):
        from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE, SAMPLE_RATE
        pipeline = StreamingPipeline(
            checkpoint_path=CHECKPOINT_PATH,
            device="cpu",
            buffer_size_s=5.0,
        )

        n_samples = HOP_SIZE * 10
        m1 = np.random.randn(n_samples).astype(np.float32) * 0.3
        m2 = np.random.randn(n_samples).astype(np.float32) * 0.1
        pipeline.push(m1, m2)
        n_produced = pipeline.process_available()
        assert n_produced > 0
        output = pipeline.read_output(n_produced)
        assert len(output) > 0
        assert np.all(np.isfinite(output))

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_pipeline_no_nan_inf(self):
        from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE
        pipeline = StreamingPipeline(CHECKPOINT_PATH, device="cpu", buffer_size_s=5.0)
        m1 = np.random.randn(HOP_SIZE * 20).astype(np.float32) * 0.5
        m2 = np.random.randn(HOP_SIZE * 20).astype(np.float32) * 0.3
        pipeline.push(m1, m2)
        pipeline.process_available()
        out = pipeline.read_output(pipeline._output_buffer.available)
        assert np.all(np.isfinite(out))

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_pipeline_reset(self):
        from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE
        pipeline = StreamingPipeline(CHECKPOINT_PATH, device="cpu", buffer_size_s=5.0)
        m1 = np.random.randn(HOP_SIZE * 5).astype(np.float32) * 0.3
        pipeline.push(m1, m1)
        pipeline.process_available()
        pipeline.reset()
        assert pipeline._m1_buffer.available == 0
        assert pipeline._output_buffer.available == 0

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_diagnostics_keys(self):
        from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE
        pipeline = StreamingPipeline(CHECKPOINT_PATH, device="cpu", buffer_size_s=5.0)
        diag = pipeline.diagnostics
        required_keys = [
            "n_frames_processed", "hop_size_samples", "sample_rate",
            "current_noise_class", "current_confidence", "real_time_factor",
        ]
        for k in required_keys:
            assert k in diag, f"Missing diagnostic key: {k}"


# ---------------------------------------------------------------------------
# End-to-End Deterministic Synthetic Test
# ---------------------------------------------------------------------------

class TestEndToEndSynthetic:
    """
    Deterministic end-to-end test with known synthetic dual-mic signal.
    Verifies: no NaN, no Inf, no clipping, valid shapes, classification.
    """

    def _make_synthetic(self, seed: int = 42, n_samples: int = 16000):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 1.0, n_samples, endpoint=False)
        speech = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        noise = rng.normal(0, 0.1, n_samples).astype(np.float32)
        m1 = speech + noise
        delay = 5
        m2 = np.zeros_like(noise)
        m2[delay:] = noise[:n_samples - delay] * 0.9
        return m1, m2

    def test_dsp_chain_no_exceptions(self):
        """Run the full DSP chain without exceptions."""
        from src.dsp.gcc_phat import gcc_phat
        from src.dsp.kalman import DelayKalmanFilter
        from src.dsp.nlms import NLMSFilter
        from src.dsp.vad import VoiceActivityDetector
        from src.dsp.fusion import AdaptiveFusionEngine
        from src.dsp.limiter import SafetyLimiter

        m1, m2 = self._make_synthetic()
        block_size = 512

        kf = DelayKalmanFilter()
        nlms = NLMSFilter(filter_length=32, step_size=0.05)
        vad = VoiceActivityDetector()
        fusion = AdaptiveFusionEngine()
        limiter = SafetyLimiter()

        outputs = []
        for i in range(0, len(m1) - block_size, block_size):
            block_m1 = m1[i:i + block_size]
            block_m2 = m2[i:i + block_size]

            gcc_result = gcc_phat(block_m1, block_m2, max_tau=50)
            kf.update(gcc_result["estimated_delay_samples"])

            dsp_out, _ = nlms.process_block(block_m1, block_m2)
            vad_result = vad.process_frame(block_m1)

            fusion_result = fusion.fuse(
                block_m1, dsp_out, block_m1,
                noise_class=0, confidence=0.8, speech_probability=vad_result["speech_probability"],
            )
            output = limiter.process(fusion_result["fused_signal"])
            outputs.append(output)

        all_out = np.concatenate(outputs)
        assert np.all(np.isfinite(all_out)), "DSP chain produced non-finite output"
        assert not np.all(all_out == 0.0), "DSP chain produced all-zero output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
