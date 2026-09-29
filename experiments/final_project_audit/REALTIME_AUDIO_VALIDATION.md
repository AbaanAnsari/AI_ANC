# REAL-TIME AUDIO VALIDATION REPORT — AETHEL123
## AI-Driven Dual-Microphone Live Audio System
**Date**: 2026-09-27  
**Validation Status**: **HISTORICAL HOST AUDIO I/O REPORT — NOT PHYSICAL ANC VALIDATION**
**Host Environment**: Windows 11 (PortAudio V19.7.0 / sounddevice 0.5.5)  
**Model Checkpoint**: `experiments/phase2_step6_targeted_crm_full/best_checkpoint.pt` (70,789 params)  

---

> Audit note (2026-09-30): the logged host capture/playback run is retained as historical I/O evidence. It does not validate acoustic noise cancellation, microphone coupling, current Phase 3 held-out performance, or current hardware latency. Physical ANC validation remains **REQUIRES HARDWARE**.

The measurements below should be read as records of the earlier test session, not as current certification.

> [!IMPORTANT]
> **MANDATORY THREE-WAY DISTINCTION**:
> 1. **OFFLINE DATASET EVALUATION**: Measured over the 202 clean test samples from the held-out dataset (STOI, PESQ, SNR, Classification).
> 2. **SOFTWARE REAL-TIME PROCESSING TEST**: Algorithmic hop-by-hop pipeline benchmark measuring computation budget and RTF without hardware I/O.
> 3. **ACTUAL PHYSICAL AUDIO HARDWARE TEST**: Live microphone capture (`Microphone Array (Realtek(R) Audio)`) → RingBuffer → DSP/AI worker → RingBuffer → Live speaker playback (`Speakers (Realtek(R) Audio)`).

---

## 1. System Architecture

```
                       PHYSICAL AUDIO INPUT
               (Realtek Microphone Array: 2 Channels)
                                 │
                                 ▼
                     sounddevice.InputStream
                    (Non-blocking Audio Callback)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
        M1 Ring Buffer (Primary)        M2 Ring Buffer (Ref)
        [Capacity: 80,000 samples]      [Capacity: 80,000 samples]
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
               Worker Thread: StreamingPipeline (Hop=128)
           ┌───────────────────────────────────────────────┐
           │ 1. DC Removal & Preprocessing                 │
           │ 2. GCC-PHAT Inter-Mic Delay Estimation        │
           │ 3. Relative-Delay Kalman Filter               │
           │ 4. STFT (N_FFT=512, Hop=128, Hann Window)     │
           │ 5. Phase-2 Step-6 AI Model (70,789 params)    │
           │    ├─ Noise Classification (Stationary/Non/Imp)│
           │    └─ Complex Ratio Mask (CRM)                │
           │ 6. Inverse STFT (Overlap-Add Synthesis)       │
           │ 7. NLMS Adaptive Filter (Noise-Class Tuned)   │
           │ 8. Voice Activity Detection (VAD)             │
           │ 9. Speech Protection & Transient Recovery     │
           │ 10. AI / DSP Adaptive Fusion Engine           │
           │ 11. Lookahead Safety Limiter (-1 dBFS Guard)  │
           └───────────────────────┬───────────────────────┘
                                   │
                                   ▼
                          Output Ring Buffer
                      [Capacity: 80,000 samples]
                                   │
                                   ▼
                     sounddevice.OutputStream
                    (Non-blocking Audio Callback)
                                   │
                                   ▼
                        PHYSICAL AUDIO OUTPUT
                       (Realtek Stereo Output)
```

---

## 2. Hardware Environment & Audio Backend

- **Operating System**: Windows-10-10.0.26200-SP0 (Windows 11 Home/Pro)
- **CPU**: Intel64 Family 6 Model 151 (12 physical cores, 16 logical threads)
- **RAM**: 15.71 GB
- **Python**: 3.11.9
- **PyTorch**: 2.14.0+cpu (CPU inference mode; CUDA optional)
- **Audio Backend**: PortAudio V19.7.0-devel via `sounddevice 0.5.5`
- **Supported Host APIs**: MME, Windows DirectSound, Windows WASAPI, Windows WDM-KS

---

## 3. Physical Hardware Device Discovery & Selection

Total physical devices enumerated on host: **20 devices** (9 input endpoints, 11 output endpoints).

### Input Device Selected:
- **Device Index**: `1`
- **Name**: `Microphone Array (Realtek(R) Au)`
- **Host API**: MME
- **Hardware Channels**: 2 input channels
- **Channel Assignment**:
  - Primary Channel (M1): `Channel 0` (Speech + Noise)
  - Reference Channel (M2): `Channel 1` (Acoustic Noise Reference)
- **Dual-Microphone Operation**: **GENUINE DUAL-MICROPHONE HARDWARE** (no simulated duplication)

### Output Device Selected:
- **Device Index**: `3`
- **Name**: `Speakers (Realtek(R) Audio)`
- **Host API**: MME
- **Hardware Channels**: 2 output channels (Stereo)
- **Routing**: Processed mono enhanced signal duplicated to Left & Right channels

---

## 4. Measured Performance (Sustained 60-Second Physical Hardware Run)

The 60-second continuous streaming stress test was executed using `scripts/benchmark_live_stream.py` with physical audio devices active throughout:

| Metric | Target | Measured Result | Status |
|---|---|---|---|
| **Test Duration (Wall-Clock)** | ≥ 60.0 s | **60.82 s** | ✅ PASS |
| **Audio Duration Processed** | ≥ 59.0 s | **59.79 s** | ✅ PASS |
| **Total Processing Time** | — | **34.81 s** | — |
| **Real-Time Factor (RTF)** | **< 1.0** | **0.582×** | ✅ **PASS** |
| **Mean Hop Processing Time** | < 8.0 ms | **4.58 ms** | ✅ **PASS** (43% margin) |
| **p95 Block Latency** | < 12.0 ms | **6.17 ms** | ✅ **PASS** |
| **p99 Block Latency** | < 16.0 ms | **9.82 ms** | ✅ **PASS** |
| **Max Block Latency** | — | **10.19 ms** | No starvation |
| **Input Overflows** | 0 | **0** | ✅ PASS |
| **Dropped Blocks** | 0 | **0** | ✅ PASS |
| **Output Underflows (Steady-State)**| 0 | **0** | ✅ PASS |
| **Initial Startup Buffer Fill** | — | 20 blocks (startup only) | Normal startup |
| **Mean CPU Usage** | < 85% | **74.96%** | Stable |

---

## 5. Distinction Between Evaluation Modes

### A. Offline Dataset Evaluation (Held-Out Test Set)
Evaluated across all 202 samples in `data/manifests/test_manifest.jsonl`:
- **STOI**: Noisy = 0.8128 → Enhanced = **0.7936** (Δ = −0.0192)
- **PESQ (Wideband)**: Noisy = 1.5116 → Enhanced = **1.7745** (Δ = **+0.2629**)
- **SNR**: Noisy = 8.89 dB → Enhanced = **7.96 dB** (Δ = −0.925 dB)
- **Noise Classification**: **76.24%** accuracy on test set

### B. Software Real-Time Processing Benchmark
- **Streaming Pipeline Hop**: 128 samples (8.0 ms at 16 kHz)
- **Algorithm Computation**: 4.58 ms / hop on CPU
- **Algorithmic RTF**: 0.582× (real-time capable on 1 CPU thread)

### C. Actual Physical Audio Hardware Test
- Physical microphone array captured genuine live speech & background room noise
- Live audio passed through `AudioInputStream` → `RingBuffer` → AI/DSP worker → `AudioOutputStream`
- Audio was delivered through physical speakers/headphones in real time
- No WAV files, no synthetic noise, and no simulated waveforms in the execution path

---

## 6. Real-Time Application UI & Web Console

The real-time audio console is served at `http://127.0.0.1:8000`:
- **Physical Device Selection**: Dropdowns dynamically populated via `GET /api/devices`.
- **Channel Assignment**: Auto-configures M1/M2 or displays single-mic warning banner if `< 2` channels.
- **Acoustic Feedback Protection**: Master gain slider initialized conservatively to 50% with feedback warning banner.
- **Telemetry Streaming**: Real-time WebSocket (`/ws/live`) delivers live RMS, peak, VAD, noise classification, RTF, buffer metrics, and live oscilloscope waveforms at 25 Hz.
- **Lifecycle Control**: `Start Real-Time Processing` opens hardware streams; `Stop` safely flushes and closes streams without device locking.

---

## 7. Commands to Launch

```powershell
# 1. Launch the Real-Time Audio Console (Application Entry Point)
python -m src.realtime.live_audio_app

# 2. Alternative launcher:
python simulator/backend/main.py

# 3. Open browser:
# Navigate to http://127.0.0.1:8000

# 4. Run the 60-Second Sustained Hardware Streaming Stress Test:
python scripts/benchmark_live_stream.py 60.0

# 5. Run the Automated Test Suite (All 394 tests):
python -m pytest tests/ --ignore=tests/test_data.py -q
```
