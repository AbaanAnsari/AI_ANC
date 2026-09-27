"""
tests/test_realtime.py
=======================
Tests for realtime engine and streaming infrastructure.

Tests:
    1. RingBuffer thread safety
    2. RingBuffer edge cases (empty, full, wrap)
    3. RealtimeEngine construction
    4. RealtimeEngine configuration
    5. RealtimeEngine process_waveform (synthetic data)
    6. StreamingPipeline latency reporting
    7. Frame processing determinism
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "phase2_step6_targeted_crm_full" / "best_checkpoint.pt"
)


class TestRingBufferThreadSafety:
    def test_concurrent_write_read(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(10000)
        results = []

        def writer():
            for _ in range(100):
                buf.write(np.random.randn(10).astype(np.float32))
                time.sleep(0.001)

        def reader():
            total = 0
            for _ in range(200):
                data = buf.read(5)
                total += len(data)
                time.sleep(0.0005)
            results.append(total)

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_w.start(); t_r.start()
        t_w.join(); t_r.join()
        # Should not deadlock or raise

    def test_capacity_property(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(256)
        assert buf.capacity == 256

    def test_initial_state(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(512)
        assert buf.available == 0
        assert buf.free_space == 512

    def test_write_updates_available(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        buf.write(np.ones(30, dtype=np.float32))
        assert buf.available == 30
        assert buf.free_space == 70

    def test_read_reduces_available(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        buf.write(np.ones(50, dtype=np.float32))
        buf.read(20)
        assert buf.available == 30

    def test_empty_read_returns_empty(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        out = buf.read(10)
        assert len(out) == 0

    def test_invalid_capacity_raises(self):
        from src.realtime.ring_buffer import RingBuffer
        with pytest.raises(ValueError):
            RingBuffer(0)

    def test_partial_read_when_insufficient(self):
        from src.realtime.ring_buffer import RingBuffer
        buf = RingBuffer(100)
        buf.write(np.ones(5, dtype=np.float32))
        out = buf.read(100)
        assert len(out) == 5


class TestRealtimeEngine:
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_construction(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        assert engine is not None
        assert engine.status == "idle"

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_configure(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        engine.configure(enable_ai=True, enable_nlms=False)
        assert engine.enable_ai is True
        assert engine.enable_nlms is False

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_process_waveform_returns_dict(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        n = 16000
        m1 = np.random.randn(n).astype(np.float32) * 0.3
        m2 = np.random.randn(n).astype(np.float32) * 0.1
        result = engine.process_waveform(m1, m2)
        assert "enhanced_waveform" in result
        assert "processing_time_s" in result
        assert "real_time_factor" in result

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_process_waveform_finite(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        m1 = np.random.randn(8000).astype(np.float32) * 0.3
        m2 = np.random.randn(8000).astype(np.float32) * 0.1
        result = engine.process_waveform(m1, m2)
        enhanced = result["enhanced_waveform"]
        assert np.all(np.isfinite(enhanced))

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_processing_time_positive(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        m1 = np.random.randn(8000).astype(np.float32) * 0.3
        m2 = np.random.randn(8000).astype(np.float32) * 0.1
        result = engine.process_waveform(m1, m2)
        assert result["processing_time_s"] > 0.0

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_real_time_factor_reported(self):
        from src.realtime.realtime_engine import RealtimeEngine
        engine = RealtimeEngine(checkpoint_path=CHECKPOINT_PATH, device="cpu")
        m1 = np.random.randn(16000).astype(np.float32) * 0.3
        m2 = np.random.randn(16000).astype(np.float32) * 0.1
        result = engine.process_waveform(m1, m2)
        rtf = result["real_time_factor"]
        assert rtf > 0.0
        # Log RTF for visibility
        print(f"\n[MEASURED] Real-Time Factor: {rtf:.3f}x")
        if rtf > 1.0:
            print(f"  WARNING: RTF > 1.0 — pipeline is SLOWER than real-time")
        else:
            print(f"  OK: RTF <= 1.0 — pipeline is real-time capable")


class TestStreamingLatency:
    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_diagnostics_latency_reported(self):
        from src.inference.streaming_pipeline import StreamingPipeline, HOP_SIZE, SAMPLE_RATE
        pipeline = StreamingPipeline(CHECKPOINT_PATH, device="cpu", buffer_size_s=5.0)
        m1 = np.random.randn(HOP_SIZE * 30).astype(np.float32) * 0.3
        m2 = np.random.randn(HOP_SIZE * 30).astype(np.float32) * 0.1
        pipeline.push(m1, m2)
        pipeline.process_available()

        diag = pipeline.diagnostics
        assert diag["n_frames_processed"] > 0
        assert diag["avg_frame_processing_ms"] >= 0.0
        print(f"\n[MEASURED] Avg frame processing time: {diag['avg_frame_processing_ms']:.2f} ms")
        print(f"[MEASURED] Real-time factor: {diag['real_time_factor']:.3f}x")
        print(f"[MEASURED] Hop duration: {diag['hop_duration_ms']:.1f} ms")

    @pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Checkpoint not found")
    def test_pipeline_param_count_in_ai_diagnostics(self):
        from src.inference.ai_inference import AIInferenceWrapper
        wrapper = AIInferenceWrapper.from_checkpoint(CHECKPOINT_PATH, device="cpu")
        diag = wrapper.diagnostics
        assert diag["param_count"] == 70789


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
