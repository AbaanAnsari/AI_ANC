# STM32H753ZI — AI Noise Cancellation Deployment Architecture Notes
## Project AETHEL123
**Status**: PREPARATION NOTES ONLY — NOT YET EXECUTED ON HARDWARE

---

> [!CAUTION]
> These are architecture planning and preparation notes for future embedded deployment.
> The model has NOT been tested on STM32H753ZI hardware.
> These notes are documentation artifacts for the research prototype.

---

## 1. Target Hardware

| Parameter | Value |
|---|---|
| MCU | STM32H753ZI |
| CPU | ARM Cortex-M7 @ 480 MHz |
| RAM | 1 MB SRAM (564 KB DTCM + 256 KB AXI-SRAM + ...) |
| Flash | 2 MB |
| FPU | Double-precision FPU |
| DSP | ARM CMSIS-DSP library |
| Audio Interface | I2S / SAI |

---

## 2. AI Model Memory Budget

| Component | Size | Notes |
|---|---|---|
| Model weights (fp32) | ~275 KB | 70,789 params × 4 bytes |
| Model weights (fp16) | ~138 KB | With quantization |
| STFT input buffer | 512 × 4 = 2 KB | One frame |
| GRU hidden state | 48 × 4 = 192 bytes | 1 layer, 48 hidden |
| Output buffer | 512 × 4 = 2 KB | One enhanced frame |
| **Total fp32 footprint** | **~280 KB** | Tight but possible |
| **Total fp16 footprint** | **~142 KB** | Recommended |

**1 MB SRAM available. Footprint < 300 KB — potentially feasible with fp16.**

---

## 3. Required Pre-Processing Steps for STM32

### 3.1 STFT
- n_fft = 512, hop = 128, window = Hann
- ARM CMSIS-DSP: `arm_rfft_fast_f32` (512-point real FFT)
- Hann window coefficients stored in Flash (512 × 4 = 2 KB)

### 3.2 GCC-PHAT
- Cross-correlation in frequency domain
- CMSIS-DSP complex multiply + IFFT
- Output: integer delay estimate

### 3.3 Kalman
- Scalar Kalman filter — 4 FP operations per sample
- Trivially implementable on M7

### 3.4 NLMS
- `filter_length = 32` (impulsive mode) or 64
- Per sample: 32–64 MACs + division
- Cortex-M7 FPU: fully capable

### 3.5 VAD
- Frame energy comparison — minimal compute
- Exponential MA: 2–3 FP ops per frame

---

## 4. Inference Engine Options

| Option | Status | Notes |
|---|---|---|
| ST X-CUBE-AI | Recommended | Converts ONNX to C code for STM32 |
| CMSIS-NN | Alternative | Manual CNN mapping |
| TFLite Micro | Alternative | Requires TFLite conversion |
| Hand-coded C | Fallback | Full control, high effort |

**Recommended path**: 
1. Export `cnn_gru_mask.onnx` (done via `deployment/export_onnx.py`)  
2. Convert via STM32Cube.AI or X-CUBE-AI  
3. Validate output matches Python inference (tolerance: 1e-4)

---

## 5. Latency Budget

| Stage | Estimated Latency @ 480 MHz |
|---|---|
| STFT (512-point FFT) | ~50 μs |
| AI model inference | ~TBD (unverified) |
| NLMS (L=32) | ~10 μs |
| ISTFT (512-point IFFT) | ~50 μs |
| VAD, Kalman, Fusion | ~5 μs |
| **Total** | **~TBD** |
| **Budget (real-time @ 16 kHz, hop=128)** | **8 ms** |

**The hop size of 128 samples at 16 kHz = 8 ms budget per frame.**
**AI inference at 480 MHz: UNVERIFIED. Requires hardware measurement.**

---

## 6. Quantization

| Strategy | Status |
|---|---|
| FP32 → FP16 | Recommended first step |
| FP32 → INT8 | May cause accuracy loss (CRM is sensitive) |
| Mixed precision | Requires hardware evaluation |

**NOTE**: The model's CRM formulation uses tanh — well-supported in FP16.
INT8 quantization has NOT been validated and may degrade SNR.

---

## 7. Deployment Preparation Checklist

- [x] ONNX export updated (`deployment/export_onnx.py`)
- [x] Model architecture documented
- [x] Parameter count verified: 70,789 < 100,000
- [x] Memory budget estimated: ~280 KB fp32
- [ ] ONNX model validated (requires `onnx` + `onnxruntime` packages)
- [ ] STM32Cube.AI conversion (requires hardware)
- [ ] INT8 quantization accuracy validation
- [ ] On-device latency measurement
- [ ] I2S/SAI audio interface integration
- [ ] End-to-end hardware validation

---

## 8. Required Software Packages for Deployment Preparation

```
pip install onnx onnxruntime
# STM32Cube.AI: Download from ST website
# X-CUBE-AI: STM32CubeIDE plugin
```

---

## 9. Audio Interface Configuration

```
Audio rate:    16 kHz
Bit depth:     16-bit or 24-bit (convert to float32 internally)
Channels:      2 (MIC1=primary, MIC2=reference)
Frame size:    512 samples = 32 ms
Hop size:      128 samples = 8 ms
Overlap:       75% (4 frames per output)
```

---

## 10. Disclaimer

```
THIS IS A RESEARCH PROTOTYPE.
Hardware deployment requires:
    1. On-device validation
    2. Formal verification
    3. Safety analysis
    4. Latency certification
This documentation does NOT constitute certification for operational use.
```
