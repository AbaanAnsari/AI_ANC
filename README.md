# AI-DRIVEN NOISE CANCELLATION & SPEECH ENHANCEMENT

**Project ID:** AETHEL123  
**Development Strategy:** Synthetic simulation and desktop streaming → physical audio validation → possible embedded evaluation

## Current Status (2026-09-30)

The implemented enhancement model is `LightweightCNNGRUMaskModel` with 70,789 trainable parameters. The active Phase 3 experiment is `experiments/phase3_100ep/best_checkpoint.pt` (best validation epoch 68). A fresh seed-controlled synthetic benchmark records +6.76 dB AI-only and +5.63 dB full-production mean delta SNR across 12 cases. Offline/streaming correlation is 0.999479. A fresh 500-hop CPU software benchmark recorded mean 2.709 ms/hop, p95 3.668 ms, p99 5.073 ms, max 5.543 ms, RTF 0.3386, and RSS 269.43 MB. These are synthetic/software results, not physical ANC results.

The Phase 3 v2 manifests currently fail the stronger identity/hash leakage audit with 612 findings, including 56 cross-split file-content overlaps; clean filenames also do not provide verifiable speaker IDs for a subset of records. Held-out scoring is therefore blocked. A prior stored held-out result reports +5.06 dB and 47.14% classification accuracy, but it is not valid evidence for a leakage-free test set and is rejected by the dashboard API because it lacks matching provenance. Do not describe these manifests as speaker-disjoint or leakage-free until they are repaired and the model is re-evaluated. The historical checkpoints and reported scores have not been rewritten.

Physical microphone coupling, room acoustics, real ADC/DAC latency, and STM32H753ZI operation have not been validated. The algorithmic lookahead/WOLA delay is 768 samples (48 ms at 16 kHz); it is not total hardware or end-to-end system latency. See [Known Limitations](#known-limitations) before interpreting benchmark results.

The design sections below describe the intended architecture and workflow. Where they conflict with this current-status section, the measured implementation status above takes precedence.

---

## 1. Project Goal

The goal of this project is to develop a **real-time AI-guided adaptive noise cancellation and speech enhancement system** for defence and other mission-critical communication environments.

The system is designed to improve speech intelligibility in severe and changing acoustic conditions such as gunshots, artillery, helicopter/rotor noise, drones, vehicle engines, wind, sirens, pumps, fans and gearbox noise.

The project is **not an AI-only denoising system**. The central design is an **AI-guided adaptive noise cancellation architecture** in which:

- AI performs supervised speech enhancement as its **primary objective**.
- AI performs supervised 3-class noise classification as a **supporting objective**.
- The AI classification result provides control information to the adaptive DSP stage.
- The reference microphone supplies correlated-noise information to the adaptive NLMS canceller.
- GCC-PHAT, Kalman filtering, VAD, speech protection, adaptive fusion and a safety limiter support and protect the complete pipeline.

The desktop application supports continuous OS audio streaming. Physical microphone array behavior and acoustic ANC performance have not yet been validated; synthetic dual-microphone results must not be presented as hardware validation.

STM32H753ZI deployment is a future evaluation target, not a validated capability.

---

## 2. Fixed System Architecture

The following seven-block architecture is **LOCKED** and is the architecture that will be implemented.

```text
1. Dual Microphone Acquisition
              ↓
2. Preprocessing & Synchronization
              ↓
3. Kalman Filtering
              ↓
4. AI-Noise Analysis
              ↓
5. Conditional Noise Processing
              ↓
6. Adaptive Fusion & Speech Protection
              ↓
7. Safety Limiter
              ↓
        Enhanced Speech
```

### Block 1 — Dual Microphone Acquisition

- **M1:** Primary microphone containing Speech + Noise.
- **M2:** Reference microphone containing correlated noise.
- Desktop simulation uses the laptop's real 2-channel microphone array.
- The two channels are acquired continuously and synchronized.
- Output is passed to the streaming processing pipeline.

### Block 2 — Preprocessing & Synchronization

The preprocessing stage performs:

- DC removal
- Bandpass filtering
- Initial noise reduction where appropriate
- Framing/windowing
- Cross-correlation / GCC-PHAT
- Relative-delay estimation
- Synchronization of M1 and M2

GCC-PHAT is important because the reference microphone must provide a meaningful correlated-noise relationship for adaptive cancellation.

### Block 3 — Kalman Filtering

The Kalman stage receives the synchronized microphone signals and tracks the evolving signal/noise state.

It is intended to track:

- Signal state
- Noise variation
- Relative delay/state information
- Temporal changes in the acoustic environment

The Kalman output is passed to the AI analysis and downstream processing stages.

### Block 4 — AI-Noise Analysis

The AI stage uses a lightweight **CNN + GRU** architecture.

The AI has two supervised objectives:

#### Primary objective — Speech Enhancement

```text
Noisy speech → AI → Enhanced speech
                         ↑
                    Clean speech target
```

The enhancement model learns from clean speech ground truth and is responsible for the primary speech-quality improvement.

#### Supporting objective — Noise Classification

The model also classifies the acoustic environment into three noise regimes:

1. **Stationary**
2. **Non-stationary**
3. **Impulsive**

The classification confidence is used as a control signal for the adaptive DSP stage.

### Block 5 — Conditional Noise Processing

The adaptive noise-cancellation stage uses **NLMS** with parameters controlled according to the detected noise regime.

```text
Stationary      → stable NLMS parameters
Non-stationary  → adaptive NLMS parameters
Impulsive       → transient-aware AI + conservative NLMS
```

NLMS uses the reference microphone to suppress residual/correlated noise that remains after the AI enhancement stage.

### Block 6 — Adaptive Fusion & Speech Protection

This stage combines AI and DSP processing while protecting speech.

It includes:

- Confidence-guided AI + DSP control
- VAD
- Speech mask/protection
- Impulse handling
- Adaptive fusion
- Output protection logic

The purpose is to prevent aggressive noise cancellation from unnecessarily damaging speech.

### Block 7 — Safety Limiter

The final stage ensures that the output remains within safe signal limits.

It is responsible for:

- Preventing excessive output amplitude
- Reducing clipping/distortion risk
- Maintaining safe output levels

### Final Output

The final output is **enhanced speech** sent to the selected desktop audio output device, such as Bluetooth earphones/headphones.

**RF communication is intentionally excluded from the current desktop implementation.**

---

## 3. Real-Time Desktop System

The desktop implementation is a **continuous streaming test bench**.

```text
Laptop 2-Channel Microphone Array
              ↓
        M1 + M2 Acquisition
              ↓
     Preprocessing / Sync
              ↓
          GCC-PHAT
              ↓
           Kalman
              ↓
        STFT / Features
              ↓
          CNN + GRU
              ↓
       Conditional NLMS
              ↓
 Adaptive Fusion + VAD + Speech Protection
              ↓
        Safety Limiter
              ↓
     Bluetooth / Headphones
```

There is no completed recording that is processed afterward during real-time operation. Audio is processed continuously frame-by-frame/block-by-block.

### Desktop Audio Device Controls

The simulator/dashboard will provide an **Audio Devices** section containing:

- Input device selection
- Input channel mapping/selection
- M1 / M2 channel assignment
- Output device selection
- Sample-rate selection
- Input test
- Output test
- Connection/status indicators
- Processing status

Bluetooth/output-device latency will be treated separately from algorithmic processing latency.

---

## 4. Dataset

The working dataset is already prepared and is treated as the current frozen dataset for development.

```text
data/
└── raw/
    └── dataset/
        ├── clean/
        │   └── *.wav
        │
        └── noise/
            ├── impulsive/
            │   └── *.wav
            │
            ├── non-stationary/
            │   ├── drone/
            │   ├── helicopter/
            │   ├── vehicle/
            │   ├── siren/
            │   └── wind/
            │
            └── stationary/
                ├── fan/
                ├── gearbox/
                └── pump/
```

The raw dataset will remain untouched. Training mixtures and derived artifacts must not overwrite the original recordings.

### Dataset usage

Training data will be generated from:

```text
Clean Speech + Selected Noise + Target SNR
                     ↓
              Synthetic Noisy Speech
                     ↓
              Supervised Training
```

The dual-microphone training/simulation representation will model the physical relationship between the primary and reference microphones rather than simply copying the same noise waveform.

Conceptually:

```text
Noise source
   ├── primary acoustic path → delay/gain/filter → M1 = Speech + Noise
   └── reference path        → reference signal → M2 = Correlated Noise
```

---

## 5. Locked AI Design

### Model type

**Lightweight CNN + GRU**

### Input representation

Initial implementation:

- Noisy audio
- STFT representation
- Complex spectral information where required to preserve phase information
- Spectral and temporal features

### Enhancement output

The current implementation predicts a bounded complex ratio mask (CRM) over the input STFT. The selected model is `LightweightCNNGRUMaskModel`; its trainable parameter count is 70,789.

### Classification output

Three classes:

```text
0 → Stationary
1 → Non-stationary
2 → Impulsive
```

### Multi-task concept

```text
                 Noisy Complex STFT
                         ↓
                    Small CNN
                         ↓
                  Lightweight GRU
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
     Enhancement Head        Classification Head
             ↓                       ↓
      Enhanced Spectrum       3-Class Regime
             ↓                       ↓
           ISTFT             Control Signal
             ↓                       ↓
      Enhanced Speech       Conditional NLMS
```

### AI priority

**Speech enhancement is the primary AI objective.**

Noise classification is a supporting task used to guide adaptive processing.

---

## 6. Hard Model Constraint

The complete trainable AI model must remain:

> **UNDER 100,000 TRAINABLE PARAMETERS**

This is a design constraint from the beginning, not a pruning target after building an oversized model.

Every model iteration must report:

- Total trainable parameters
- Model size in FP32
- Estimated INT8 model size
- Parameter memory
- Runtime memory requirements
- Inference time
- CPU utilization where measurable
- Latency contribution

The model must be designed with eventual STM32H753ZI deployment in mind.

---

## 7. Initial Locked Signal-Processing Parameters

The following values are the initial implementation targets and may be tuned only through documented experiments.

| Parameter | Initial value |
|---|---:|
| Internal sample rate | **16 kHz** |
| Internal audio format | **float32** |
| Initial training segment | **1 second** |
| Samples per training segment | **16,000** |
| STFT FFT size | **512** |
| STFT hop length | **128** |
| STFT window length | **512** |
| STFT window | **Hann** |
| Training SNR choices | **-5, 0, 5, 10, 15, 20 dB** |
| Dataset split | **70% / 15% / 15%** |
| Split level | **Original recordings**, before segmentation |
| AI architecture | **CNN + lightweight GRU** |
| Noise classes | **3** |
| Trainable parameter limit | **< 100,000** |

Original dataset files will not be resampled or modified in-place. Audio is converted to the canonical internal representation during the data pipeline.

---

## 8. Supervised Training Strategy

Training will use paired clean/noisy data.

### Primary target

```text
Input:  Noisy speech
Target: Clean speech
```

### Supporting target

```text
Input:  Noisy speech
Target: Stationary / Non-stationary / Impulsive
```

### Phase 3 v2 combined loss

The Phase 3 training entry point uses a multi-objective loss with configured weights:

```text
L_total = λ1 L_enhancement
        + λ2 L_SI-SNR
        + λ3 L_spectral
        + λ4 L_classification
```

Mask 0.30, reconstruction 0.15, SI-SDR 0.30, multi-resolution STFT 0.15, energy 0.10, and classification 0.10. These settings are recorded in the checkpoint configuration; historical experiment settings may differ.

Potential evaluation metrics include:

- SNR
- SNR improvement
- STOI
- PESQ
- SI-SNR
- Processing latency
- Real-time factor
- Residual noise

Metrics must be measured from actual signals; the system must never fabricate performance values.

---

## 9. Dataset / Simulation Dual-Microphone Strategy

Two operating modes are required.

### Mode A — Training / Offline Simulation

Generate realistic dual-microphone mixtures using clean speech and categorized noise.

M1 contains:

```text
Speech + Primary-path Noise
```

M2 contains:

```text
Correlated Reference Noise
```

The two channels will have controlled differences such as:

- Relative delay
- Gain difference
- Filtering
- Small path variation

This gives GCC-PHAT and NLMS meaningful reference information.

### Mode B — Future Physical Desktop Validation

Use an actual two-channel microphone array only during a separately documented physical validation session. No such validation is currently claimed.

```text
Physical Mic Ch 1 → M1
Physical Mic Ch 2 → M2
```

Synthetic mixtures are used for the reported offline benchmarks. They do not establish real-world microphone coupling or physical noise cancellation.

---

## 10. Dashboard / Evaluation Requirements

The desktop dashboard will monitor the complete system in real time.

### Required live information

- Input SNR where ground truth is available
- Output SNR where ground truth is available
- SNR improvement
- STOI on a rolling evaluation window
- PESQ on a rolling evaluation window where appropriate
- AI confidence
- Detected noise regime
- Selected processing regime
- Algorithmic latency
- Audio levels
- Residual noise
- Processing-block status
- M1 waveform
- M2 waveform
- Spectrograms
- AI mask/output visualization
- NLMS parameters
- Model parameter count
- Model size
- Processing time

### Important metric rule

For dataset/simulation mode, clean speech ground truth is available, so SNR/STOI/PESQ-style evaluation can be calculated.

For live physical microphone operation, clean speech ground truth is normally unavailable. The dashboard must therefore distinguish **measured metrics** from metrics that require a reference signal and must not fabricate them.

### Required comparison

The evaluation framework should support:

```

Noisy Input
     vs
AI-only Output
     vs
AI + Adaptive NLMS Output
```

This comparison is important for demonstrating the contribution of the adaptive NLMS stage.

## Known Limitations

- The reported speech-enhancement and ablation values use synthetic dual-microphone mixtures. The Phase 3 v2 manifests currently fail a stronger content-hash audit; speaker-disjoint test performance is therefore **BLOCKED** pending data repair and re-evaluation.
- The classifier remains modest (47.14% accuracy and macro-F1 0.466 in the stored Phase 3 100-epoch report). Do not infer downstream benefit from classification accuracy alone.
- The synthetic acoustic path does not establish microphone frequency response, microphone self-noise, acoustic coupling, room reverberation, transducer nonlinearities, physical microphone spacing, clock drift, or acoustic-feedback stability.
- ADC/DAC, operating-system, and transducer latency have not been measured. The 768-sample / 48 ms value is algorithmic WOLA/lookahead delay only.
- The available CPU real-time benchmark is not a physical ANC or embedded benchmark. STM32H753ZI operator support, activation memory, peak RAM, MAC count, and on-target timing are **NOT TESTED**.
- `data/raw/**` is git-ignored, and no dataset download or external artifact-storage mechanism is configured. A fresh clone needs the exact source corpus provisioned separately; current manifest/source hashes should be checked before attempting reproduction.
- Historical reports contain older numbers and claims. Use results tied to the checkpoint hash and manifest hashes in `experiments/phase3_100ep/metadata.json`; the recorded training hardware and original package environment are unavailable.

## Reproduction and Audit Commands

Run from the repository root:

```powershell
python scripts/verify_dataset.py
python scripts/audit_manifest_leakage.py
python scripts/validate_checkpoint.py experiments/phase3_100ep/best_checkpoint.pt --metadata-output experiments/phase3_100ep/metadata.json
python -m pytest -q
python scripts/run_phase3_full_evaluation.py --checkpoint experiments/phase3_100ep/best_checkpoint.pt
```

The stronger leakage audit currently exits nonzero for the checked-in Phase 3 v2 manifests. Do not use the full-evaluation command to support speaker-disjoint claims until that audit passes and the evaluation is rerun on repaired splits.

---