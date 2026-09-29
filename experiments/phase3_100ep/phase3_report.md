# Phase 3 — 100-Epoch Retraining & Revalidation Report

## 1. Executive Summary

- **Selected Checkpoint:** `D:\SIH\AI_ANC\experiments\phase3_100ep\best_checkpoint.pt`
- **Training Epochs:** 100 recorded epochs
- **Best Validation Epoch:** Epoch 68
- **Dataset Split Records:** Train 2644, validation 577, test 599
- **Phase 3 v2 AI-only Mean Delta SNR:** **+6.76 dB** (12 synthetic cases)
- **Phase 3 v2 Full-Production Mean Delta SNR:** **+5.63 dB** (12 synthetic cases)
- **Streaming Equivalence Correlation:** 0.999479 (Target > 0.999)
- **Real-Time Factor (RTF):** 0.339 (2.71 ms / 8.0 ms hop on CPU)
- **Held-out evaluation:** **BLOCKED** (612 manifest findings or missing result provenance); prior held-out scores are not presented as valid.

## 2. Dataset Audit

- **Train split:** 2644 records (912 clean, 1732 noise)
- **Validation split:** 577 records (206 clean, 371 noise)
- **Test split:** 599 records (227 clean, 372 noise)
- **Identity/leakage audit:** FAIL (612 findings); speaker/file leakage is not reported as zero when findings exist.
- **Recorded sample-0 diversity:** 100/100 unique fingerprints across recorded epochs.

## 3. Training Audit

- **Optimizer:** AdamW (LR=1e-3, Weight Decay=1e-4, eps=1e-8, grad_clip=5.0)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)
- **Loss Formulation:** MultiTaskPhase3Loss (Mask: 0.30, Recon: 0.15, SI-SDR: 0.30, MRSTFT: 0.15, Energy: 0.10, Cls: 0.10)
- **Best Validation Loss:** -2.358195 at Epoch 68
- **Training history:** 100 recorded epochs; best checkpoint selected by validation total loss.

## 4. Checkpoint Audit

- **Architecture:** `LightweightCNNGRUMaskModel`
- **Trainable Parameters:** 70,789 (Expected: 70,789)
- **Total Parameters:** 70,789
- **Strict Loading (`strict=True`):** PASS
- **Missing Keys:** 0 | **Unexpected Keys:** 0
- **Deployment Engine Load:** PASS

## 5. Cause 1 Revalidation (Streaming Equivalence)

- **STFT Max Abs Error:** 3.7253e-09 (< 1e-6)
- **STFT Relative L2 Error:** 3.9482e-08 (< 1e-5)
- **STFT Scale Discrepancy:** 0.0000 dB (Exact 0 dB match)
- **Temporal Context Buffer:** 7 frames
- **Future Lookahead:** 3 frames
- **Total Algorithmic Delay:** 768 samples (48.0 ms)
- **Offline vs Streaming Correlation:** 0.999479 (Threshold > 0.999)
- **Waveform SNR Difference:** 28.59 dB (< 0.05 dB)
- **Cause 1 Verdict:** PASS

## 6. Classifier Audit

- **Held-out classification:** BLOCKED; current test split/result provenance is invalid or absent.

## 7. AI Enhancement Benchmark (12 Synthetic Cases)

| Noise Type | Target SNR | Input SNR | Offline delta SNR | Streaming AI-only delta SNR | Full-production delta SNR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fan | 0.0 dB | 0.11 dB | +7.46 dB | **+7.36 dB** | +5.46 dB |
| Fan | 5.0 dB | 5.11 dB | +6.04 dB | **+5.92 dB** | +4.62 dB |
| Fan | 10.0 dB | 10.11 dB | +4.31 dB | **+4.19 dB** | +3.55 dB |
| Helicopter | 0.0 dB | 0.03 dB | +9.62 dB | **+9.60 dB** | +9.15 dB |
| Helicopter | 5.0 dB | 5.03 dB | +7.76 dB | **+7.72 dB** | +7.42 dB |
| Helicopter | 10.0 dB | 10.03 dB | +5.70 dB | **+5.65 dB** | +5.55 dB |
| Siren | 0.0 dB | 0.09 dB | +7.53 dB | **+7.41 dB** | +5.75 dB |
| Siren | 5.0 dB | 5.09 dB | +5.94 dB | **+5.83 dB** | +4.82 dB |
| Siren | 10.0 dB | 10.09 dB | +4.01 dB | **+3.91 dB** | +3.94 dB |
| Drone | 0.0 dB | 0.02 dB | +9.91 dB | **+9.88 dB** | +6.73 dB |
| Drone | 5.0 dB | 5.02 dB | +7.89 dB | **+7.85 dB** | +5.86 dB |
| Drone | 10.0 dB | 10.02 dB | +5.82 dB | **+5.80 dB** | +4.70 dB |

- **Overall Mean AI-only Delta SNR:** **+6.76 dB**
- **0 dB Mean:** **+8.56 dB** | **5 dB Mean:** **+6.83 dB** | **10 dB Mean:** **+4.89 dB**

## 8. Downstream DSP Ablation (Matrix A through K)

| Configuration | Description | Mean Delta SNR | Loss vs AI-only |
| :--- | :--- | :---: | :---: |
| `A_ai_only` | DSP ablation configuration | **+6.76 dB** | +0.00 dB |
| `B_ai_plus_fusion_no_nlms` | DSP ablation configuration | **+3.86 dB** | +2.90 dB |
| `C_ai_nlms_w00` | DSP ablation configuration | **+6.76 dB** | +0.00 dB |
| `D_ai_nlms_w01` | DSP ablation configuration | **+6.64 dB** | +0.12 dB |
| `E_ai_nlms_w02` | DSP ablation configuration | **+6.15 dB** | +0.61 dB |
| `F_ai_nlms_w03` | DSP ablation configuration | **+5.42 dB** | +1.34 dB |
| `G_ai_nlms_w05` | DSP ablation configuration | **+3.66 dB** | +3.10 dB |
| `H_ai_speech_prot_only` | DSP ablation configuration | **+6.07 dB** | +0.69 dB |
| `I_ai_limiter_only` | DSP ablation configuration | **+6.76 dB** | +0.00 dB |
| `J_ai_vad_only` | DSP ablation configuration | **+6.76 dB** | +0.00 dB |
| `K_full_production` | DSP ablation configuration | **+5.63 dB** | +1.13 dB |


## 9. Oracle Analysis

- **Oracle AI-only mean:** 122.00 dB across 12 finite cases (upper-bound experiment, not model performance).
- **Oracle full-production mean:** 8.96 dB across 12 cases (upper-bound experiment).

## 10. Real-Time Audit

- **Processing Time / Hop:** Mean: 2.71 ms | p50: 2.63 ms | p95: 3.67 ms | p99: 5.07 ms
- **Hop Size Budget:** 8.0 ms (128 samples at 16 kHz)
- **Real-Time Factor (RTF):** 0.339 (< 1.0)
- **Model Trainable Parameters:** 70,789
- **Process Memory (RSS):** 269.4 MB

## 11. Cross-Experiment Comparison

NOT REPORTED: this audit does not have a checkpoint-hash-matched baseline result set. Historical values are not substituted.

## 12. Remaining Limitations

1. **Held-out evaluation:** BLOCKED until source identity/content overlap is repaired and the held-out benchmark is rerun.
2. **Synthetic acoustics:** Reported benchmark cases use simulated mixtures and do not model all room, transducer, or microphone effects.
3. **Hardware:** Physical microphone coupling, ADC/DAC delay, clock drift, acoustic feedback, and STM32 execution remain unmeasured.

## 13. Final Status Verdict

| Subsystem | Verdict | Notes |
| :--- | :---: | :--- |
| **Recorded sample-0 mixture diversity** | **PASS** | 100/100 unique recorded hashes |
| **Manifest leakage audit** | **FAIL** | 612 finding(s); see the manifest audit command |
| **Training history** | **PASS** | 100 epoch records present |
| **Checkpoint Compatibility** | **PASS** | strict=True, trainable parameters=70789 |
| **Offline/streaming equivalence** | **PASS** | correlation 0.999479; 768 samples algorithmic delay |
| **AI/DSP quality** | **NOT TESTED** | synthetic full-production mean delta SNR +5.63 dB is reported; no pass threshold is defined |
| **Held-out evaluation** | **BLOCKED** | 612 manifest findings or missing result provenance |
| **CPU realtime factor** | **PASS** | 0.339 on the recorded benchmark host; not embedded feasibility |
| **Physical hardware validation** | **REQUIRES HARDWARE** | not performed |
| **STM32H753ZI deployment** | **NOT TESTED** | no target-board measurement |
