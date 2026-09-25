# AI-DRIVEN NOISE CANCELLATION & SPEECH ENHANCEMENT

**Project ID:** AETHEL123  
**Development Strategy:** Desktop real-time simulation first → validated implementation → STM32H753ZI deployment

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

The first implementation will run as a **true continuous desktop streaming system using the laptop's physical 2-channel microphone array and a user-selected audio output device**. It will not use a record-then-play workflow.

After the desktop system is validated, the design will be optimized and mapped to the **STM32H753ZI** for embedded real-time operation.

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

The initial model will use a lightweight spectral enhancement formulation, with the exact mask/output parameterization finalized during Phase 2 model design and validation.

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
| STFT hop length | **256** |
| STFT window length | **512** |
| STFT window | **Hann** |
| Initial SNR range | **-5, 0, 5, 10, 15, 20 dB** |
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

### Candidate combined loss

The training framework will support a multi-objective loss of the form:

```text
L_total = λ1 L_enhancement
        + λ2 L_SI-SNR
        + λ3 L_spectral
        + λ4 L_classification
```

The exact loss weights are **not yet locked** and will be selected through controlled experiments.

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

### Mode B — Real-Time Desktop Hardware

Use the laptop's actual two-channel microphone array directly.

```text
Physical Mic Ch 1 → M1
Physical Mic Ch 2 → M2
```

No prerecorded audio is used to pretend that the physical microphone is live.

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

```text
Noisy Input
     vs
AI-only Output
     vs
AI + Adaptive NLMS Output
```

This comparison is important for demonstrating the contribution of the adaptive NLMS stage.

---

# 11. Locked Implementation Phases

The project will be implemented in the following phases. The phase order is fixed so that each major subsystem is validated before the next dependency is built.

---

## PHASE 0 — Project & Dataset Foundation

**Status:** Completed / prepared before AI training

Objectives:

- Freeze the working dataset.
- Establish the project folder structure.
- Establish canonical 16 kHz mono float32 internal processing.
- Establish configuration files.
- Create dataset manifests.
- Create recording-level train/validation/test splits.
- Prevent data leakage between splits.
- Verify the complete dataset pipeline.

Deliverables:

- Dataset manifest
- Train/validation/test manifests
- Configuration files
- Dataset verification scripts

---

## PHASE 1 — Data Pipeline & Supervised Training Dataset Generation

**Status:** NEXT PHASE

Objectives:

- Implement clean audio loading.
- Implement resampling/conversion to the canonical representation.
- Implement 1-second segmentation/random cropping.
- Implement noise selection by class.
- Implement SNR-controlled speech/noise mixing.
- Implement dual-microphone M1/M2 simulation.
- Implement relative delay, gain and filtering variation.
- Implement STFT feature extraction.
- Validate training batches visually and numerically.

Primary output:

```text
Clean + Noise + SNR
        ↓
M1 / M2 simulated pair
        ↓
STFT/features
        ↓
Supervised training sample
```

**Phase 1 does not train the final model yet.** It establishes a reliable supervised data pipeline.

---

## PHASE 2 — Lightweight AI Model Development

Objectives:

- Implement the CNN feature extractor.
- Implement the lightweight GRU temporal stage.
- Implement the speech-enhancement head.
- Implement the 3-class noise-classification head.
- Enforce the `<100,000 trainable parameter` constraint.
- Implement model parameter counting.
- Establish baseline model size and computational cost.
- Unit-test the forward pass.

Output:

```text
Noisy STFT
   ↓
CNN
   ↓
GRU
   ├── Enhancement Head
   └── Classification Head
```

---

## PHASE 3 — Supervised AI Training

Objectives:

- Train the enhancement task using clean speech targets.
- Train the 3-class noise classifier.
- Implement the combined multi-task training objective.
- Tune training hyperparameters.
- Validate against held-out data.
- Track loss curves and objective metrics.
- Save checkpoints and experiment metadata.
- Select a model only based on documented evaluation results and deployment constraints.

Required model checks:

- Parameter count `<100,000`
- Validation performance
- Generalization to noise categories
- Computational cost
- Latency

---

## PHASE 4 — Core DSP Implementation

Objectives:

- Implement preprocessing.
- Implement GCC-PHAT.
- Implement synchronization/delay estimation.
- Implement Kalman filtering.
- Implement NLMS.
- Implement VAD.
- Implement speech protection.
- Implement adaptive fusion.
- Implement safety limiter.

Each DSP module will be unit-tested independently before full pipeline integration.

---

## PHASE 5 — AI + Adaptive NLMS Integration

This phase creates the central **AI-guided adaptive noise cancellation system**.

Objectives:

- Connect AI enhancement to the DSP pipeline.
- Use AI noise classification to select/control NLMS behavior.
- Implement confidence-guided control.
- Implement stationary/non-stationary/impulsive processing modes.
- Combine AI output and adaptive NLMS output.
- Protect speech using VAD/speech-mask logic.
- Apply the safety limiter.

Target flow:

```text
M1 + M2
  ↓
Preprocessing + GCC-PHAT
  ↓
Kalman
  ↓
AI Enhancement + Noise Classification
  ↓
Conditional NLMS
  ↓
Adaptive Fusion + Speech Protection
  ↓
Safety Limiter
  ↓
Enhanced Speech
```

---

## PHASE 6 — Desktop Real-Time Streaming Simulator

Objectives:

- Connect the actual laptop 2-channel microphone array.
- Implement continuous audio capture.
- Implement real-time frame/ring-buffer processing.
- Connect the complete AI + DSP pipeline.
- Connect selectable audio output devices.
- Support Bluetooth earphones/headphones.
- Display algorithmic latency.
- Monitor audio levels and pipeline status.
- Provide live visualizations.
- Compare noisy, AI-only and AI+NLMS outputs.

The system must operate continuously without record-then-play processing.

---

## PHASE 7 — Evaluation & Optimization

Objectives:

- Evaluate SNR and SNR improvement.
- Evaluate STOI.
- Evaluate PESQ where applicable.
- Evaluate SI-SNR.
- Measure algorithmic latency.
- Measure CPU utilization.
- Measure memory use.
- Evaluate robustness across stationary, non-stationary and impulsive noise.
- Evaluate speech preservation.
- Compare AI-only versus AI+NLMS.
- Identify failure cases and document them.

The project targets from the problem statement are:

```text
SNR  > 15 dB
STOI > 0.85
PESQ > 2.5
Low latency suitable for real-time communication
```

These are **target requirements**, not assumed results. They must be demonstrated through measured evaluation.

---

## PHASE 8 — Embedded Optimization & STM32H753ZI Preparation

Objectives:

- Freeze the validated desktop model.
- Optimize inference for embedded execution.
- Evaluate quantization, especially INT8 where suitable.
- Reduce RAM/Flash requirements.
- Benchmark embedded-compatible inference.
- Convert/export the model into the selected STM32-compatible deployment format.
- Port required DSP blocks.
- Map the real-time pipeline to STM32 resources.
- Validate numerical behavior against the desktop reference implementation.

No embedded optimization should compromise the locked architecture or the `<100,000 trainable parameter` constraint without a documented design decision.

---

## PHASE 9 — STM32H753ZI Real-Time Prototype

Objectives:

- Connect primary and reference microphones to the STM32 system.
- Run the AI inference engine on-device.
- Run the adaptive DSP pipeline on-device.
- Implement real-time audio buffering.
- Validate end-to-end latency.
- Validate memory and CPU utilization.
- Validate speech enhancement under representative noise conditions.
- Compare embedded results against the desktop reference.

Final embedded path:

```text
Primary Mic + Reference Mic
            ↓
Acquisition
            ↓
Preprocessing + Synchronization
            ↓
Kalman
            ↓
AI Enhancement + Classification
            ↓
Conditional NLMS
            ↓
Fusion + Speech Protection
            ↓
Safety Limiter
            ↓
Enhanced Speech
```

---

# 12. Final Project Folder Structure

The `data/` folder is maintained separately because the dataset has already been prepared.

```text
AI-DRIVEN-NOISE-CANCELLATION-SPEECH-ENHANCEMENT/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── dataset.yaml
│   ├── training.yaml
│   ├── model.yaml
│   ├── dsp.yaml
│   └── realtime.yaml
│
├── data/                         # EXISTING DATASET — NOT MODIFIED BY SCAFFOLD
│   ├── raw/
│   ├── manifests/
│   ├── generated/
│   └── samples/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── create_manifest.py
│   ├── split_dataset.py
│   ├── generate_training_data.py
│   └── verify_dataset.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── audio_loader.py
│   │   ├── preprocessing.py
│   │   ├── mixer.py
│   │   ├── segmenter.py
│   │   └── dataset.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── stft.py
│   │   └── spectrogram.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_gru.py
│   │   ├── enhancement_head.py
│   │   ├── classification_head.py
│   │   └── model_utils.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── train.py
│   │   └── validation.py
│   │
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── gcc_phat.py
│   │   ├── kalman.py
│   │   ├── nlms.py
│   │   ├── vad.py
│   │   ├── speech_protection.py
│   │   ├── fusion.py
│   │   └── limiter.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── ai_inference.py
│   │   └── streaming_pipeline.py
│   │
│   └── realtime/
│       ├── __init__.py
│       ├── audio_input.py
│       ├── audio_output.py
│       ├── device_manager.py
│       ├── ring_buffer.py
│       └── realtime_engine.py
│
├── simulator/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api.py
│   │   └── state.py
│   │
│   └── frontend/
│       ├── index.html
│       ├── css/
│       │   └── dashboard.css
│       └── js/
│           ├── dashboard.js
│           ├── audio.js
│           ├── plots.js
│           └── metrics.js
│
├── evaluation/
│   ├── evaluate_model.py
│   ├── evaluate_realtime.py
│   ├── metrics.py
│   ├── snr.py
│   ├── stoi.py
│   ├── pesq.py
│   └── reports/
│       ├── training/
│       ├── validation/
│       └── realtime/
│
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── results/
│
├── models/
│   ├── checkpoints/
│   ├── exported/
│   └── quantized/
│
├── deployment/
│   ├── desktop/
│   │   ├── model/
│   │   └── runtime/
│   │
│   └── stm32/
│       ├── model/
│       ├── dsp/
│       ├── middleware/
│       └── firmware/
│
├── tests/
│   ├── test_data.py
│   ├── test_stft.py
│   ├── test_gcc_phat.py
│   ├── test_kalman.py
│   ├── test_nlms.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   └── test_realtime.py
│
└── docs/
    ├── architecture/
    ├── dataset/
    ├── model/
    ├── dsp/
    ├── realtime/
    └── deployment/
```

### Folder responsibilities

| Folder | Responsibility |
|---|---|
| `configs/` | Central project configuration |
| `data/` | Dataset, manifests and generated samples |
| `scripts/` | Dataset/build utility scripts |
| `src/data/` | Loading, preprocessing, mixing and segmentation |
| `src/features/` | STFT/spectral feature extraction |
| `src/models/` | CNN+GRU AI model |
| `src/training/` | Supervised training and validation |
| `src/dsp/` | GCC-PHAT, Kalman, NLMS, VAD, fusion and limiter |
| `src/inference/` | AI inference and complete streaming pipeline |
| `src/realtime/` | Physical audio I/O and real-time engine |
| `simulator/` | Desktop dashboard/backend |
| `evaluation/` | Metrics and evaluation reports |
| `experiments/` | Run configurations, logs and results |
| `models/` | Model checkpoints/exported/quantized artifacts |
| `deployment/` | Desktop and STM32 deployment material |
| `tests/` | Unit/integration tests |
| `docs/` | Technical documentation |

---

# 13. Architecture-to-Code Mapping

| Locked architecture block | Main implementation location |
|---|---|
| 1. Dual Microphone Acquisition | `src/realtime/audio_input.py`, `device_manager.py` |
| 2. Preprocessing & Synchronization | `src/data/preprocessing.py`, `src/dsp/gcc_phat.py` |
| 3. Kalman Filtering | `src/dsp/kalman.py` |
| 4. AI-Noise Analysis | `src/models/`, `src/features/`, `src/inference/` |
| 5. Conditional Noise Processing | `src/dsp/nlms.py` |
| 6. Adaptive Fusion & Speech Protection | `src/dsp/fusion.py`, `vad.py`, `speech_protection.py` |
| 7. Safety Limiter | `src/dsp/limiter.py` |
| Enhanced Speech Output | `src/realtime/audio_output.py` |

---

# 14. Non-Negotiable Design Rules

1. **The seven-block architecture is fixed.**
2. **Speech enhancement is the primary AI objective.**
3. **3-class noise classification is a supporting AI objective.**
4. **The AI model must remain below 100,000 trainable parameters.**
5. **NLMS is part of the core system, not an optional unrelated experiment.**
6. **The reference microphone must be meaningfully correlated with the primary-path noise.**
7. **The desktop system must use real continuous microphone input.**
8. **Record-then-play is not the real-time operating mode.**
9. **The laptop's two-channel microphone array is the desktop M1/M2 source.**
10. **Output must be selectable through the desktop audio device system.**
11. **RF communication is excluded from the current implementation.**
12. **Raw dataset files must remain untouched.**
13. **Train/validation/test splitting must occur at recording level to prevent leakage.**
14. **Performance metrics must be measured, not fabricated.**
15. **Bluetooth/output-device latency must be distinguished from algorithmic latency.**
16. **DSP and AI modules must be independently testable.**
17. **Desktop validation comes before STM32 deployment.**
18. **Embedded optimization must preserve the validated system behavior as closely as practical.**

---

# 15. Current Development Position

The project has completed the dataset preparation/foundation stage and the complete project scaffold has been established.

The next implementation step is:

> **PHASE 1 — Data Pipeline & Supervised Training Dataset Generation**

The immediate objective is to build and validate the data pipeline before training the CNN+GRU model.

```text
CURRENT
  ↓
PHASE 1: Data Pipeline
  ↓
PHASE 2: AI Model
  ↓
PHASE 3: AI Training
  ↓
PHASE 4: DSP
  ↓
PHASE 5: AI + NLMS Integration
  ↓
PHASE 6: Real-Time Desktop Simulator
  ↓
PHASE 7: Evaluation & Optimization
  ↓
PHASE 8: STM32 Optimization
  ↓
PHASE 9: STM32H753ZI Prototype
```

This README is the project-level reference for the current implementation plan. Any future change to the architecture, model constraint, real-time operating concept, or phase sequence should be explicitly documented rather than silently changing the design.
