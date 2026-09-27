# PHASE 2 STEP 6 REPORT

## 1. Experiment Objective
Phase 2 Step 6 is the decisive full 30-epoch validation of the **Targeted Complex Ratio Mask (CRM)** direction. In Step 5, supervising the predicted mask directly against a stabilized ideal ratio mask $M_{\text{target}} = \text{clamp}(S/(X+\epsilon), -1, 1)$ showed a promising upward trajectory in SNR improvement (moving from $-7.47\text{ dB}$ to $-6.07\text{ dB}$ in 5 epochs). The objective of Step 6 is to determine whether:
1. SNR improvement continues toward zero or positive values over a full 30-epoch training schedule.
2. Speech energy remains preserved without amplitude collapse.
3. The multi-task classification head recovers useful multi-class discrimination (resolving the Step 5 classification drop to 26.73%).
4. The model converges stably under the strict $<100,000$ trainable parameter budget (70,789 parameters).

---

## 2. Exact Configuration
- **Model**: `LightweightCNNGRUMaskModel` ([`src/models/cnn_gru_mask.py`](file:///d:/SIH/AI_ANC/src/models/cnn_gru_mask.py))
- **Parameter Count**: 70,789 trainable parameters
- **Optimizer**: AdamW ($lr = 0.001$, weight_decay $= 0.0001$, grad_clip_norm $= 5.0$)
- **Scheduler**: `ReduceLROnPlateau` (mode='min', factor=0.5, patience=3, min_lr=$10^{-6}$)
- **Batch Size**: 16
- **Epochs**: 30
- **Seed**: 123 (fresh initialization)
- **Loss Function**: `MultiTaskTargetedCRMLoss` ([`src/training/targeted_crm_loss.py`](file:///d:/SIH/AI_ANC/src/training/targeted_crm_loss.py))
  - Enhancement Loss: $L_{\text{enh}} = 0.60 \cdot L_{\text{mask}} + 0.35 \cdot L_{\text{recon}} + 0.05 \cdot L_{\text{energy}}$
  - Classification Loss: $L_{\text{cls}} = \text{CrossEntropy}(\text{logits}, y)$
  - Total Multi-Task Loss: $L_{\text{total}} = 1.0 \cdot L_{\text{enh}} + 0.10 \cdot L_{\text{cls}}$
  - Sub-loss weights: $0.7 \cdot L_1 + 0.3 \cdot L_2$ for both mask and reconstruction terms
  - Denominator stabilization: $\epsilon = 10^{-7}$
  - Target mask clipping: $\text{clamp}(M_{\text{raw}}, -1.0, 1.0)$ on real and imaginary components
- **Dataset Manifests**:
  - Training: [`data/manifests/train_manifest.jsonl`](file:///d:/SIH/AI_ANC/data/manifests/train_manifest.jsonl) (936 samples, 59 batches)
  - Validation: [`data/manifests/val_manifest.jsonl`](file:///d:/SIH/AI_ANC/data/manifests/val_manifest.jsonl) (202 samples, 13 batches)
  - Test: [`data/manifests/test_manifest.jsonl`](file:///d:/SIH/AI_ANC/data/manifests/test_manifest.jsonl) (202 samples, 13 batches) — strictly isolated, evaluated once post-training.

---

## 3. Architecture Verification
The CNN + GRU model architecture ([`LightweightCNNGRUMaskModel`](file:///d:/SIH/AI_ANC/src/models/cnn_gru_mask.py#L28-L178)) is **100% preserved**:
- **Input STFT**: $(B, 2, 257, T)$ [Channel 0: Real, Channel 1: Imaginary]
- **CNN Backbone**:
  - Conv2D Layer 1: $2 \to 16$, kernel $(5, 3)$, stride $(2, 1)$, padding $(2, 1)$, BatchNorm2D, ReLU $\to (B, 16, 129, T)$
  - Conv2D Layer 2: $16 \to 32$, kernel $(5, 3)$, stride $(2, 1)$, padding $(2, 1)$, BatchNorm2D, ReLU $\to (B, 32, 65, T)$
  - Conv2D Layer 3: $32 \to 16$, kernel $(5, 3)$, stride $(2, 1)$, padding $(2, 1)$, BatchNorm2D, ReLU $\to (B, 16, 33, T)$
- **Linear Feature Projection**: $(B, T, 528) \to (B, T, 32)$, ReLU
- **Temporal GRU**: Input size 32, hidden size 48, 1 layer, batch_first=True $\to (B, T, 48)$
- **Mask Enhancement Head**: Linear($48 \to 514$), reshape $(B, 2, 257, T)$, $\tanh$ component activation $\to M_{\text{pred}} \in (-1, 1)^2$
- **Complex Ratio Masking**: $\hat{S}_r = M_r X_r - M_i X_i$, $\hat{S}_i = M_r X_i + M_i X_r \to \hat{S} \in \mathbb{R}^{B \times 2 \times 257 \times T}$
- **Classification Head**: Bottleneck Linear($48 \to 16$), ReLU, Linear($16 \to 3$) $\to (B, 3)$ logits

---

## 4. Parameter Count
- **Trainable Parameters**: **70,789**
- **Non-Trainable Parameters**: 0
- **Total Parameters**: **70,789**
- **Hard Budget Constraint**: $< 100,000$ (Compliant, 29,211 parameters / 29.2% headroom)
- **FP32 Memory Footprint**: $276.52\text{ KB}$

---

## 5. Loss Verification
The exact Step 5 loss formulation is verified and unchanged:
$$M_{\text{raw}} = \frac{S}{X + \epsilon} = \frac{(S_r + j S_i)(X_r - j X_i)}{X_r^2 + X_i^2 + 10^{-7}}$$
$$M_{\text{target}, r} = \text{clamp}(M_{\text{raw}, r}, -1.0, 1.0), \quad M_{\text{target}, i} = \text{clamp}(M_{\text{raw}, i}, -1.0, 1.0)$$
$$L_{\text{mask}} = 0.7 \cdot \|M_{\text{pred}} - M_{\text{target}}\|_1 + 0.3 \cdot \|M_{\text{pred}} - M_{\text{target}}\|_2^2$$
$$L_{\text{recon}} = 0.7 \cdot \|\hat{S} - S\|_1 + 0.3 \cdot \|\hat{S} - S\|_2^2$$
$$L_{\text{energy}} = \frac{\max(0, E_{\text{clean}} - E_{\text{enhanced}})}{E_{\text{clean}} + 10^{-8}}$$
$$L_{\text{enh}} = 0.60 \cdot L_{\text{mask}} + 0.35 \cdot L_{\text{recon}} + 0.05 \cdot L_{\text{energy}}$$
$$L_{\text{total}} = 1.0 \cdot L_{\text{enh}} + 0.10 \cdot \text{CrossEntropy}(\text{logits}, y)$$

---

## 6. Training Results (Full 30-Epoch Trajectory)
Complete epoch-by-epoch log from [`history.json`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/history.json):

| Epoch | Train Total Loss | Val Total Loss | Val $L_{\text{enh}}$ | Val $L_{\text{mask}}$ | Val $L_{\text{recon}}$ | Val $L_{\text{energy}}$ | Val $L_{\text{cls}}$ | Val Acc (%) | Val SNR Imp (dB) | Sample RMS Ratio | Learning Rate | Duration (s) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.213301 | 0.237117 | 0.126849 | 0.134907 | 0.000579 | 0.914050 | 1.102677 | 31.19% | -6.81 | 0.1528 | $1.00 \times 10^{-3}$ | 17.4 |
| 2 | 0.195842 | 0.208038 | 0.097641 | 0.141849 | 0.000705 | 0.245701 | 1.103967 | 30.20% | -6.12 | 0.3724 | $1.00 \times 10^{-3}$ | 18.8 |
| 3 | 0.191293 | 0.214092 | 0.103995 | 0.137253 | 0.000622 | 0.428506 | 1.100971 | 33.17% | -5.74 | 0.2843 | $1.00 \times 10^{-3}$ | 17.5 |
| 4 | 0.187969 | 0.213686 | 0.103771 | 0.143882 | 0.000594 | 0.344667 | 1.099154 | 34.16% | -4.54 | 0.4145 | $1.00 \times 10^{-3}$ | 17.7 |
| 5 | 0.184846 | 0.220241 | 0.110566 | 0.139056 | 0.000572 | 0.538658 | 1.096742 | 36.14% | -4.96 | 0.3813 | $1.00 \times 10^{-3}$ | 17.7 |
| 6 | 0.182441 | 0.227068 | 0.117607 | 0.147598 | 0.000566 | 0.577000 | 1.094605 | 40.10% | -5.16 | 0.3263 | $1.00 \times 10^{-3}$ | 17.7 |
| 7 | 0.178740 | 0.233650 | 0.124402 | 0.162192 | 0.000568 | 0.537773 | 1.092475 | 40.10% | -5.10 | 0.3759 | $5.00 \times 10^{-4}$ | 17.9 |
| 8 | 0.176698 | 0.209718 | 0.100146 | 0.133836 | 0.000570 | 0.392904 | 1.095716 | 36.14% | -5.39 | 0.2947 | $5.00 \times 10^{-4}$ | 17.4 |
| 9 | 0.174490 | 0.200279 | 0.092064 | 0.130057 | 0.000529 | 0.276884 | 1.082153 | 40.59% | -3.57 | 0.5756 | $5.00 \times 10^{-4}$ | 17.9 |
| 10 | 0.170044 | 0.224973 | 0.115270 | 0.140038 | 0.000594 | 0.620787 | 1.097024 | 36.14% | -6.84 | 0.0811 | $5.00 \times 10^{-4}$ | 17.7 |
| 11 | 0.163089 | 0.236974 | 0.126725 | 0.163754 | 0.000540 | 0.565672 | 1.102491 | 42.57% | -5.46 | 0.2359 | $5.00 \times 10^{-4}$ | 17.5 |
| 12 | 0.154495 | 0.220302 | 0.111663 | 0.137008 | 0.000563 | 0.585217 | 1.086394 | 45.05% | -6.30 | 0.1356 | $5.00 \times 10^{-4}$ | 17.4 |
| 13 | 0.145233 | 0.230386 | 0.126511 | 0.166063 | 0.000558 | 0.533556 | 1.038754 | 43.56% | -5.92 | 0.1877 | $5.00 \times 10^{-4}$ | 17.5 |
| 14 | 0.136357 | 0.367296 | 0.161882 | 0.234397 | 0.000565 | 0.420930 | 2.054136 | 36.14% | -4.68 | 0.4297 | $2.50 \times 10^{-4}$ | 17.4 |
| 15 | 0.131836 | 0.151078 | 0.074905 | 0.121942 | 0.000521 | 0.031143 | 0.761734 | 64.85% | -1.01 | 0.9253 | $2.50 \times 10^{-4}$ | 17.4 |
| 16 | 0.129677 | 0.393496 | 0.136508 | 0.197151 | 0.000519 | 0.360713 | 2.569883 | 36.63% | -3.60 | 0.5146 | $2.50 \times 10^{-4}$ | 17.4 |
| 17 | 0.125984 | 0.332065 | 0.128414 | 0.178012 | 0.000560 | 0.428217 | 2.036509 | 41.09% | -5.22 | 0.2842 | $2.50 \times 10^{-4}$ | 17.9 |
| 18 | 0.124027 | 0.237490 | 0.094957 | 0.130888 | 0.000550 | 0.324618 | 1.425331 | 40.59% | -4.87 | 0.4413 | $2.50 \times 10^{-4}$ | 17.2 |
| 19 | 0.125260 | 0.249335 | 0.111095 | 0.144074 | 0.000575 | 0.488989 | 1.382399 | 39.11% | -6.15 | 0.2474 | $2.50 \times 10^{-4}$ | 17.4 |
| 20 | 0.117811 | 0.302504 | 0.113752 | 0.171299 | 0.000508 | 0.215898 | 1.887520 | 47.52% | -2.79 | 0.7137 | $1.25 \times 10^{-4}$ | 17.0 |
| 21 | 0.117698 | 0.338823 | 0.117649 | 0.184555 | 0.000508 | 0.134767 | 2.211740 | 46.53% | -1.95 | 0.8008 | $1.25 \times 10^{-4}$ | 17.5 |
| 22 | 0.114245 | 0.190364 | 0.089069 | 0.133787 | 0.000492 | 0.172484 | 1.012955 | 60.89% | -2.28 | 0.7587 | $1.25 \times 10^{-4}$ | 17.3 |
| 23 | 0.114316 | 0.381967 | 0.111525 | 0.159383 | 0.000526 | 0.314218 | 2.704425 | 44.55% | -3.92 | 0.5123 | $1.25 \times 10^{-4}$ | 17.4 |
| 24 | 0.112377 | 0.149928 | 0.073223 | 0.116258 | 0.000488 | 0.065947 | 0.767047 | 68.81% | -0.91 | 0.8704 | $6.25 \times 10^{-5}$ | 17.4 |
| 25 | 0.110708 | 0.140625 | 0.073170 | 0.117726 | 0.000495 | 0.047231 | 0.674549 | 69.31% | -0.61 | 0.8965 | $6.25 \times 10^{-5}$ | 17.9 |
| 26 | 0.110001 | 0.149975 | 0.073816 | 0.116648 | 0.000487 | 0.073127 | 0.761597 | 66.34% | -1.10 | 0.8399 | $6.25 \times 10^{-5}$ | 17.5 |
| 27 | 0.110063 | 0.184188 | 0.080460 | 0.120617 | 0.000497 | 0.158312 | 1.037281 | 60.40% | -2.35 | 0.7158 | $6.25 \times 10^{-5}$ | 18.4 |
| 28 | 0.108774 | 0.161531 | 0.078676 | 0.124239 | 0.000483 | 0.079264 | 0.828554 | 64.85% | -1.02 | 0.8536 | $6.25 \times 10^{-5}$ | 17.2 |
| **29** | **0.109964** | **0.138699** | **0.072905** | **0.116219** | **0.000484** | **0.060086** | **0.657940** | **73.27%** | **-0.65** | **0.8622** | **$6.25 \times 10^{-5}$** | **17.4** |
| 30 | 0.107750 | 0.139549 | 0.073734 | 0.116731 | 0.000476 | 0.070582 | 0.658147 | 70.79% | **-0.58** | 0.8492 | $6.25 \times 10^{-5}$ | 17.0 |

*Total Training Duration*: 526.9 seconds (~8.78 minutes).

---

## 7. Validation SNR Trajectory
- **Early Phase (Epochs 1–5)**: SNR improvement progressed from $-6.81\text{ dB}$ to $-4.96\text{ dB}$.
- **Mid Phase (Epochs 6–20)**: Oscillated between $-6.84\text{ dB}$ (Epoch 10) and $-1.01\text{ dB}$ (Epoch 15) as the model balanced multi-task gradients between the mask head and classification head.
- **Late Phase (Epochs 21–30)**: Converged sharply toward near-zero SNR degradation:
  - Epoch 24: $-0.91\text{ dB}$
  - Epoch 25: $-0.61\text{ dB}$
  - Epoch 29: $-0.65\text{ dB}$ (Best val loss checkpoint)
  - Epoch 30: **$-0.58\text{ dB}$** (Peak validation SNR improvement)
- **Comparison to Step 4 CRM**: Step 4 validation SNR improvement plateaued at $-7.35\text{ dB}$. Step 6 reached **$-0.58\text{ dB}$**, an improvement of **$+6.77\text{ dB}$**.

---

## 8. Validation Classification Trajectory
The Step 5 short run (5 epochs) saw classification accuracy drop to 26.73% due to transient gradient competition. In the full 30-epoch run:
- **Epochs 1–5**: Stagnated around chance (30.2% – 36.1%), with predictions collapsing mostly into Class 2.
- **Epochs 6–15**: As the learning rate decayed to $5 \times 10^{-4}$ and $2.5 \times 10^{-4}$, the representation diversified, reaching 64.85% at Epoch 15.
- **Epochs 24–30**: Classification accuracy stabilized between **66.3% and 73.3%**:
  - Epoch 29 (Best Checkpoint): Overall **73.27%** (Class 0: 70.3%, Class 1: 69.6%, Class 2: 79.7%).
  - Multi-class balance was completely restored across stationary, non-stationary, and impulsive noises.

---

## 9. Fixed-Sample RMS Trajectory
Monitored on fixed validation sample across all 30 epochs:
- Clean RMS: $0.08110$
- Noisy RMS: $0.08488$
- Enhanced RMS:
  - Epoch 1: $0.01239$ (ratio $0.1528$)
  - Epoch 5: $0.03093$ (ratio $0.3813$)
  - Epoch 15: $0.07505$ (ratio $0.9253$)
  - Epoch 25: $0.07271$ (ratio $0.8965$)
  - Epoch 29: $0.06993$ (ratio **$0.8622$**)
  - Epoch 30: $0.06887$ (ratio **$0.8492$**)
- Waveform Min / Max at Best Checkpoint: $[-0.2223, +0.4451]$ (Mean Absolute Value: $0.0389$).
- **Anti-Collapse Confirmation**: Zero amplitude collapse. Natural speech energy is fully preserved without scaling extinction.

---

## 10. Best Checkpoint
- **Selection Criterion**: Minimum validation total loss ($L_{\text{total}}$) on held-out validation set.
- **Best Validation Loss Epoch**: **Epoch 29**
- **Best Validation Loss**: **0.138699**
- **Best Validation SNR-Improvement Epoch**: **Epoch 30** ($-0.58\text{ dB}$)
- Saved to: [`experiments/phase2_step6_targeted_crm_full/best_checkpoint.pt`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/best_checkpoint.pt)
- Last Checkpoint (Epoch 30) saved to: [`experiments/phase2_step6_targeted_crm_full/last_checkpoint.pt`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/last_checkpoint.pt)

---

## 11. Final Held-Out Test Results
Evaluated **exactly once** on [`data/manifests/test_manifest.jsonl`](file:///d:/SIH/AI_ANC/data/manifests/test_manifest.jsonl) using `best_checkpoint.pt` (Epoch 29):

| Metric | Measured Value |
|---|---|
| **Checkpoint Evaluated** | `best_checkpoint.pt` (Epoch 29) |
| **Test Input SNR** | **+7.60 dB** |
| **Test Enhanced SNR** | **+6.86 dB** |
| **Test SNR Improvement** | **-0.73 dB** |
| **Noisy STOI** | Unavailable (library not installed) |
| **Enhanced STOI** | Unavailable (library not installed) |
| **STOI Delta** | Unavailable (library not installed) |
| **Noisy PESQ** | Unavailable (library not installed) |
| **Enhanced PESQ** | Unavailable (library not installed) |
| **PESQ Delta** | Unavailable (library not installed) |
| **Overall Classification Accuracy** | **79.70%** (161 / 202 correct) |
| **Class 0 (Stationary) Accuracy** | **78.13%** (50 / 64 correct) |
| **Class 1 (Non-Stationary) Accuracy** | **73.91%** (51 / 69 correct) |
| **Class 2 (Impulsive) Accuracy** | **86.96%** (60 / 69 correct) |
| **Test Total Loss** | 0.121721 |
| **Test Enhancement Loss** | 0.070825 (Mask: 0.113720, Recon: 0.000441, Energy: 0.048768) |
| **Test Classification Loss** | 0.508965 |
| **Sample Enhanced/Clean RMS Ratio** | **0.8622** |

### Test Set Confusion Matrix (Total: 202 samples):
```
Actual \ Predicted       Class 0 (Stat)    Class 1 (Non-Stat)    Class 2 (Impulsive)    Total
Class 0 (Stationary)           50                  12                     2               64
Class 1 (Non-Stationary)        3                  51                    15               69
Class 2 (Impulsive)             1                   8                    60               69
```

---

## 12. Direct Comparison Table
Comparison across the three verified Phase 2 experimental stages:

| Metric | Phase 2 Step 1 (Direct-STFT Baseline) | Phase 2 Step 4 (CRM Baseline) | Phase 2 Step 6 (Targeted CRM Full) | Step 6 vs Baseline Delta | Step 6 vs Step 4 Delta |
|---|:---:|:---:|:---:|:---:|:---:|
| **Training Epochs** | 30 | 30 | **30** | — | — |
| **Trainable Parameters** | 70,789 | 70,789 | **70,789** | 0 | 0 |
| **Loss Formulation** | Direct STFT Regression | CRM Reconstruction + Energy Hinge | **Targeted CRM ($0.60 L_{\text{mask}} + 0.35 L_{\text{recon}} + 0.05 L_{\text{en}}$)** | Supervised Mask Target | Supervised Mask Target |
| **Best Val Loss** | 0.110485 | 0.083693 | **0.138699** | +0.028214 | +0.055006 |
| **Best Val SNR Improvement** | N/A | -7.35 dB (Ep 28) | **-0.58 dB (Ep 30)** | — | **+6.77 dB** |
| **Final Test Input SNR** | +7.60 dB | +7.60 dB | **+7.60 dB** | 0.00 dB | 0.00 dB |
| **Final Test Enhanced SNR** | -0.01 dB | +0.12 dB | **+6.86 dB** | **+6.87 dB** | **+6.74 dB** |
| **Final Test SNR Improvement**| **-7.60 dB** | **-7.47 dB** | **-0.73 dB** | **+6.87 dB** | **+6.74 dB** |
| **Classification Accuracy** | 34.16% | 61.88% | **79.70%** | **+45.54%** | **+17.82%** |
| **Class 0 Accuracy** | 0.0% | 0.0% | **78.13%** | **+78.13%** | **+78.13%** |
| **Class 1 Accuracy** | 0.0% | 56.5% | **73.91%** | **+73.91%** | **+17.41%** |
| **Class 2 Accuracy** | 100.0% | 100.0% | **86.96%** | -13.04% (balanced) | -13.04% (balanced) |
| **Enhanced / Clean RMS Ratio** | 0.0052 (Collapse) | 0.8769 (Preserved) | **0.8622 (Preserved)** | **+0.8570** | -0.0147 |

---

## 13. Speech Quality
- **STOI**: `STOI unavailable` (not installed in environment).
- **PESQ**: `PESQ unavailable` (not installed in environment).
- Per project rules, no metric values are inferred, hallucinated, or claimed without direct measurement.

---

## 14. Classification
- Overall classification accuracy reached **79.70%** (161 / 202 samples correct on held-out test set).
- Per-class accuracy:
  - Class 0 (Stationary): 78.13%
  - Class 1 (Non-Stationary): 73.91%
  - Class 2 (Impulsive): 86.96%
- This confirms that the GRU temporal feature representations are rich enough to distinguish noise types while simultaneously predicting complex masks for enhancement, completely overcoming the 26.73% transient drop observed during the Step 5 5-epoch run.

---

## 15. Numerical Stability
- **NaN / Inf occurrences**: **0** throughout all 30 training epochs, 30 validation passes, final test evaluation, and automated test suite.
- **Gradient Norms**: Unclipped gradient norms remained strictly stable ($0.048 - 0.077$).
- **Parameters**: All 70,789 parameters remained strictly finite and bounded.

---

## 16. Reproducibility
- **Global Seed**: `123` (`torch.manual_seed(123)`, `np.random.seed(123)`)
- **Initialization**: Fresh random weights (no loading from baseline or previous checkpoint).
- **Environment**: Python 3.11.9, PyTorch 2.14.0+cu126, Windows-10 CPU execution.
- **Configuration Snapshot**: Saved in [`experiments/phase2_step6_targeted_crm_full/config.yaml`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/config.yaml).

---

## 17. Automated Verification
All 15 automated checks from [`tests/test_phase2_step6.py`](file:///d:/SIH/AI_ANC/tests/test_phase2_step6.py) passed cleanly:
1. `test_fresh_initialization`: PASSED
2. `test_thirty_epochs_completed`: PASSED
3. `test_manifest_separation`: PASSED
4. `test_no_test_access_during_training`: PASSED
5. `test_best_checkpoint_selection`: PASSED
6. `test_last_checkpoint_exists`: PASSED
7. `test_checkpoint_reload_works`: PASSED
8. `test_parameter_count_exact`: PASSED (70,789)
9. `test_parameter_count_constraint`: PASSED ($< 100,000$)
10. `test_no_nan_inf`: PASSED
11. `test_deterministic_seed`: PASSED
12. `test_step5_loss_unchanged`: PASSED
13. `test_previous_experiments_preserved`: PASSED
14. `test_final_test_performed_after_training`: PASSED
15. `test_required_artifacts_exist`: PASSED

---

## 18. Artifacts
All generated files are stored in [`experiments/phase2_step6_targeted_crm_full/`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/):
1. [`STEP6_REPORT.md`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/STEP6_REPORT.md) — This report.
2. [`config.yaml`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/config.yaml) — Training hyperparameters configuration.
3. [`history.json`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/history.json) — Full 30-epoch training and validation metrics.
4. [`metrics.json`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/metrics.json) — Final test results and experiment summary.
5. [`best_checkpoint.pt`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/best_checkpoint.pt) — Best model weights (Epoch 29).
6. [`last_checkpoint.pt`](file:///d:/SIH/AI_ANC/experiments/phase2_step6_targeted_crm_full/last_checkpoint.pt) — Final model weights (Epoch 30).

---

## 19. Verdict
**Verdict: B — PARTIAL / PROMISING BUT INSUFFICIENT**

### Evidence-Based Rationale:
1. **Meaningful Improvement Over Previous Models**:
   - SNR improvement rose dramatically from **$-7.60\text{ dB}$** (Direct-STFT Baseline) and **$-7.47\text{ dB}$** (Step 4 CRM) to **$-0.73\text{ dB}$** (Enhanced SNR $+6.86\text{ dB}$).
   - This represents a massive **$+6.74\text{ dB}$ gain** toward noise reduction while maintaining natural speech RMS ($0.8622$).
   - Classification accuracy reached an all-time high of **$79.70\%$** with balanced sensitivity across all 3 noise categories.
2. **Project Targets Remain Unmet**:
   - The project target is positive SNR improvement ($> +15\text{ dB}$).
   - Current test SNR improvement is **$-0.73\text{ dB}$**; although it is substantially closer to zero (net loss reduced by 90%), actual positive noise suppression has not yet been achieved.
   - Per Section 18 interpretation rules, this outcome is classified as **B**.

---

## 20. Next Step
**Recommend ONE Next Step Only**:
Perform a controlled ablation on the loss weighting of the Targeted CRM objective—specifically evaluating whether slightly increasing the mask supervision weight ($0.60 \to 0.75$) and lowering the energy hinge weight ($0.05 \to 0.01$), or transitioning to a signal-to-distortion ratio (SI-SDR / SNR) loss component, can provide the final directional gradient needed to cross zero and achieve positive SNR improvement ($> 0\text{ dB}$).
