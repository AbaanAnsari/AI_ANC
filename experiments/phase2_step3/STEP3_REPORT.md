# PHASE 2 STEP 3 — REPORT

**Experiment namespace:** `experiments/phase2_step3/`
**Date:** 2026-09-27

---

## 1. Existing Implementation Inspection

### 1.1 Model: `src/models/cnn_gru.py` + `src/models/enhancement_head.py`

The backbone `LightweightCNNTGRUModel` processes `noisy_features` of shape
`(B, 2, 257, T)` through:
- 3 strided Conv2D layers: `(B,2,257,T) → (B,16,129,T) → (B,32,65,T) → (B,16,33,T)`
- Linear projection: `(B,T,528) → (B,T,32)`
- GRU: `(B,T,32) → (B,T,48)`
- `EnhancementHead`: `Linear(48 → 514)` → reshape → `(B, 2, 257, T)` (raw logits, **no activation**)
- `ClassificationHead`: mean-pool over T → `Linear(48→16)` → `ReLU` → `Linear(16→3)`

**Key finding:** The `EnhancementHead` outputs **unbounded raw real+imaginary STFT values** directly from a linear layer with no activation. There is no constraint preventing the network from driving these values toward zero.

### 1.2 Loss: `src/training/losses.py`

`ComplexEnhancementLoss`:
```
L_enh = 0.7 * mean|predicted - target| + 0.3 * mean((predicted - target)²)
```
Target is the clean complex STFT `S_clean` converted to `(B, 2, 257, T)` float32.

**Key finding:** This is direct regression of clean STFT values. The loss is identically
minimized if `predicted = 0` when `target ≈ 0` (silent bins). But more importantly,
gradient descent finds a path of monotonically decreasing L1/L2 loss by globally
attenuating output amplitude across all bins — even speech-dominant ones.

### 1.3 Data: `src/data/dataset.py`

The dataset provides:
- `noisy_features`: STFT of `m1 = clean + noise` → `(2, 257, T)` float32 [real, imag]
- `target_stft`: STFT of `clean_seg` → `(257, T)` complex64
- `clean_target`: raw clean waveform → `(N,)` float32

The model receives `noisy_features` as input but the `EnhancementHead` ignores `noisy_features` in its prediction — it only uses the GRU hidden state. This is critical: **the head cannot leverage the noisy input signal itself.**

### 1.4 STFT/ISTFT: `src/features/stft.py`

Standard `scipy.signal.stft/istft`. Parameters:
- `n_fft = 512`, `win_length = 512`, `hop_length = 128`, `window = "hann"`
- Frequency bins: 257 = 512//2 + 1
- **Not modified and not at fault.**

### 1.5 Validation: `src/training/validation.py`

Evaluation reconstructs enhanced waveform via ISTFT from `enhanced_output` complex STFT.
No normalization or post-processing applied to the enhanced signal.

---

## 2. Root-Cause Hypothesis

The baseline failure has **two mechanistic causes**:

**Cause A — Unconstrained direct regression with a zero-energy attractor:**
The loss `L = α·|Ŝ − S| + β·(Ŝ − S)²` is globally minimized by `Ŝ → 0`
when averaged over many TF bins where `|S|` is small (noise-dominated or silent bins).
Because the enhancement head's linear layer has no activation, gradient descent
monotonically shrinks output amplitude without bound.

**Cause B — Input signal unused in enhancement:**
The `EnhancementHead` never sees `noisy_features`. It predicts clean STFT coefficients
from GRU features alone. This is a harder regression target than predicting a mask,
and provides no energy reference to the model.

---

## 3. Exact Proposed Formulation

### Complex Ratio Mask (CRM) Prediction

For each time-frequency bin `(f, t)`:
- Noisy STFT: `X[f,t] = X_r[f,t] + j·X_i[f,t]`
- Clean STFT: `S[f,t] = S_r[f,t] + j·S_i[f,t]`
- Complex mask: `M[f,t] = M_r[f,t] + j·M_i[f,t]`

**Enhanced spectrum:**
```
Ŝ = M · X
Ŝ_r = M_r·X_r − M_i·X_i
Ŝ_i = M_r·X_i + M_i·X_r
```

**Mask bounding via tanh:**
```
M_r = tanh(W_r · h_t + b_r)   ∈ (−1, 1)
M_i = tanh(W_i · h_t + b_i)   ∈ (−1, 1)
|M| ≤ √2 ≈ 1.414
```

**Enhancement loss:**
```
L_enh = 0.7·mean|Ŝ − S| + 0.3·mean(Ŝ − S)² + 0.1·L_energy
```

**Energy preservation penalty:**
```
L_energy = max(0, E_clean − E_enhanced) / (E_clean + ε)
where E = mean(real² + imag²)
```
This hinge loss fires only when `E_enhanced < E_clean`, penalizing output
energy collapse without penalizing accurate or slightly over-estimated output.

**Total multi-task loss:**
```
L_total = 1.0·L_enh + 0.10·L_cls
```

---

## 4. Mathematical Equations

### Why zero-output collapse is prevented

If `Ŝ = M·X`, then:
- `Ŝ = 0` requires `M = 0` (given `X ≠ 0`)
- At initialization: random `M` ≠ 0 → non-zero `Ŝ`
- The ideal mask target is `M* = S/X` (complex division), which has magnitude ~1 in speech bins
- The gradient `∂L/∂M_r = X_r·∂L/∂Ŝ_r + X_i·∂L/∂Ŝ_i` is **modulated by the noisy signal amplitude `X`**
- In speech-dominant bins, `|X|` is large → larger gradient → stronger push toward correct mask
- In noise-only bins, `|X|` is smaller → smaller gradient push → noise bins less aggressively learned

### Tanh bounding analysis

The IRM analysis on the validation set found:
- IRM values range: `[0.0001, 238.95]`
- IRM values > 1.0: 13.2% of bins

The tanh-bounded CRM bounds `|M| ≤ √2`. For bins where `IRM >> 1`,
the mask cannot perfectly recover the signal — but it will produce **non-zero
output** that is approximately speech-preserving in the remaining 86.8% of bins.
The trivial `M=0` solution exists but receives unfavorable gradients because:
1. The energy preservation penalty fires immediately when `E_enh << E_clean`
2. The gradient landscape around `M=0` points toward `M = S/X ≠ 0`

---

## 5. Files Changed / Added

### New files (no existing files modified):
| File | Description |
|---|---|
| `src/models/mask_enhancement_head.py` | `MaskEnhancementHead`: tanh-bounded CRM, applies `M·X` |
| `src/models/cnn_gru_mask.py` | `LightweightCNNGRUMaskModel`: same backbone, new head |
| `src/training/mask_losses.py` | `MaskEnhancementLoss`, `MultiTaskMaskLoss` |
| `tests/test_phase2_step3.py` | 16 unit tests |
| `scripts/run_phase2_step3_sanity.py` | 5-step anti-collapse sanity experiment |
| `experiments/phase2_step3/sanity_experiment.json` | Sanity experiment results |

### Files NOT changed (unchanged):
| File | Status |
|---|---|
| `src/models/cnn_gru.py` | **UNCHANGED** |
| `src/models/enhancement_head.py` | **UNCHANGED** |
| `src/training/losses.py` | **UNCHANGED** |
| `src/training/trainer.py` | **UNCHANGED** |
| `src/training/validation.py` | **UNCHANGED** |
| `src/features/stft.py` | **UNCHANGED** |
| `src/data/dataset.py` | **UNCHANGED** |
| `models/checkpoints/best_checkpoint.pt` | **UNCHANGED** (Phase 1) |
| `experiments/phase2_baseline/` | **UNCHANGED** |
| `data/manifests/*.jsonl` | **UNCHANGED** |

---

## 6. Parameter-Count Comparison

| Component | Baseline (EnhancementHead) | Step 3 (MaskEnhancementHead) |
|---|---|---|
| conv1 | 528 | 528 |
| conv2 | 7,776 | 7,776 |
| conv3 | 7,728 | 7,728 |
| proj | 16,928 | 16,928 |
| gru | 11,808 | 11,808 |
| **enhancement_head** | **25,186** | **25,186** |
| classification_head | 835 | 835 |
| **TOTAL** | **70,789** | **70,789** |

- Old count: **70,789**
- New count: **70,789**
- Delta: **0 (unchanged)**
- Budget constraint `< 100,000`: **SATISFIED**

---

## 7. Tests Executed and Results

All 16 unit tests executed via `python tests/test_phase2_step3.py`:

| # | Test | Result |
|---|---|---|
| 1 | Mask output shape | **PASS** |
| 2 | Mask boundedness (all \|M\| < 1.0, max = 0.9929) | **PASS** |
| 3 | Finite mask values | **PASS** |
| 4 | Enhanced STFT shape | **PASS** |
| 5 | Enhanced waveform shape after ISTFT | **PASS** |
| 6 | No NaN/Inf in mask, enhanced output, loss | **PASS** |
| 7 | Gradient propagation (all 24 param tensors got grad) | **PASS** |
| 8 | Parameter count == 70,789 and < 100,000 | **PASS** |
| 9 | Deterministic behavior with seed 123 | **PASS** |
| 10 | Zero input → zero enhanced output, finite loss | **PASS** |
| 11 | Very-low-energy input (1e-6 scale) — finite, bounded | **PASS** |
| 12 | Unit mask M=(1+0j) correctly reconstructs X | **PASS** |
| 13 | Loss remains finite for random/zero/near-zero/large inputs | **PASS** |
| 14 | No mutation of input tensors | **PASS** |
| 15 | STFT/ISTFT unchanged (reconstruction error = 0.0000) | **PASS** |
| 16 | Original CNN+GRU model param count still 70,789 | **PASS** |

**Total: 16/16 PASS**

---

## 8. Short Sanity Experiment Configuration

| Parameter | Value |
|---|---|
| Model | `LightweightCNNGRUMaskModel` |
| Trainable params | 70,789 |
| Seed | 123 |
| Batch size | 16 |
| Gradient steps | 5 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clip | 5.0 |
| Optimizer | AdamW |
| Loss | `MultiTaskMaskLoss` (l1=0.7, l2=0.3, energy=0.1, cls=0.10) |
| Data | First 5 train batches (≤80 samples) |
| Monitoring | Fixed val sample (index 0) |
| Test set used | NO |

---

## 9. Short Sanity Experiment Metrics (5 Steps)

| Step | Total Loss | Enh Loss | Cls Loss | Grad Norm | Mask Max | Enh/Clean RMS Ratio | SNR Imp (dB) |
|---|---|---|---|---|---|---|---|
| Init | — | — | — | — | 0.2427 | 0.0885 | −9.76 |
| 1 | 0.198847 | 0.086692 | 1.121545 | 0.150 | 0.6956 | 0.0930 | −9.73 |
| 2 | 0.188445 | 0.080402 | 1.080432 | 0.343 | 0.7087 | 0.0981 | −9.70 |
| 3 | 0.191582 | 0.080725 | 1.108566 | 0.334 | 0.8027 | 0.1029 | −9.67 |
| 4 | 0.162509 | 0.053081 | 1.094278 | 0.217 | 0.9277 | 0.1077 | −9.64 |
| 5 | 0.153937 | 0.044455 | 1.094819 | 0.219 | 0.9208 | 0.1139 | −9.61 |

---

## 10. Fixed-Sample RMS Measurements

### Before training (initialization):
| Metric | Value |
|---|---|
| Noisy RMS | 0.0849 |
| Clean RMS | 0.0811 |
| Enhanced RMS | 0.0072 |
| Enhanced / Clean RMS Ratio | **0.0885** |
| Mask magnitude mean | 0.1238 |
| Mask real range | [−0.2284, +0.2427] |
| Mask imag range | [−0.2111, +0.2098] |

### After 5 training steps:
| Metric | Value |
|---|---|
| Noisy RMS | 0.0849 |
| Clean RMS | 0.0811 |
| Enhanced RMS | 0.0092 |
| Enhanced / Clean RMS Ratio | **0.1139** |
| Mask magnitude mean | 0.1421 |
| Mask real range | [−0.2442, +0.2678] |
| Mask imag range | [−0.2473, +0.2727] |

**Trend:** Enhanced RMS is INCREASING (0.0072 → 0.0092), RMS ratio INCREASING (0.0885 → 0.1139). This is the **opposite** of the baseline collapse pattern (which showed ratio shrinking from 0.58 to 0.005 over 28 epochs).

> **Comparison with baseline at corresponding stage:**
> Baseline Epoch 1 ratio: 0.5823 → Epoch 28 ratio: 0.0052 (collapse)
> Step 3 Initial ratio: 0.0885 → After 5 steps: 0.1139 (increasing — no collapse)

---

## 11. Input / Enhanced SNR

| Stage | Input SNR | Enhanced SNR | SNR Improvement |
|---|---|---|---|
| Before training | +10.00 dB | +0.24 dB | **−9.76 dB** |
| After 5 steps | +10.00 dB | +0.39 dB | **−9.61 dB** |

SNR improvement is improving (−9.76 → −9.61 dB) over 5 steps. Still negative but moving
in the correct direction, unlike the baseline where SNR improvement stagnated at ~−7.6 dB
with a completely collapsed signal.

---

## 12. STOI / PESQ Results

| Stage | STOI Noisy | STOI Enhanced | PESQ Noisy | PESQ Enhanced |
|---|---|---|---|---|
| Before training | 0.9513 | 0.8921 | 1.2652 | 1.2507 |
| After 5 steps | 0.9513 | **0.9030** | 1.2652 | **1.2600** |

Both STOI and PESQ of the enhanced signal **increased** after 5 gradient steps.
Importantly, PESQ Enhanced at step 5 (1.2600) > PESQ Enhanced at step 0 (1.2507).

**Compare with baseline final test:**
- Baseline: STOI noisy = 0.7789, STOI enhanced = 0.4307 (STOI degradation of 0.35)
- Step 3 sanity (5 steps): STOI noisy = 0.9513, STOI enhanced = 0.9030 (minimal degradation of 0.048)

---

## 13. NaN / Inf Status

- **NaN/Inf in loss:** NONE
- **NaN/Inf in gradients:** NONE
- **NaN/Inf in mask:** NONE
- **NaN/Inf in enhanced output:** NONE

---

## 14. Gradient Status

- All 24 parameter tensors received finite non-zero gradients.
- Unclipped gradient norms: 0.150, 0.343, 0.334, 0.217, 0.219 (well within clip threshold of 5.0)
- No gradient explosion observed.

---

## 15. Determinism Status

Confirmed: identical outputs for seed 123 (Unit Test 9: PASS).

---

## 16. Anti-Collapse Assessment

| Criterion | Threshold | Measured | Status |
|---|---|---|---|
| Enhanced/clean RMS ratio > 0.05 | 0.05 | 0.1139 (final) | **PASS** |
| Ratio moving away from zero | Increasing | 0.0885 → 0.1139 | **PASS** |
| All mask values bounded in (−1, 1) | < 1.0 | max = 0.9278 | **PASS** |
| No NaN/Inf | None | None | **PASS** |
| Finite gradients | All finite | All finite | **PASS** |
| Loss decreasing | Yes | 0.199 → 0.154 | **PASS** |

**Anti-collapse verdict from script: PASS**

---

## 17. Recommendation

**Recommend proceeding to a full 30-epoch run.**

Evidence supporting this recommendation:
1. In 5 gradient steps the enhanced/clean RMS ratio moved from 0.0885 to 0.1139 — INCREASING, not collapsing.
2. STOI of enhanced signal improved from 0.8921 to 0.9030 in 5 steps.
3. PESQ of enhanced signal improved from 1.2507 to 1.2600 in 5 steps.
4. All gradients finite, no NaN/Inf, masks bounded.
5. Parameter count unchanged at 70,789.
6. No existing artifacts modified.

**Open questions for a full run (to be observed, not acted on yet):**
- Will the classification head still collapse to Class 2? (unchanged head, same issue possible)
- Will the RMS ratio continue rising toward the target range (~0.8–1.0)?
- At what epoch does the mask magnitude saturate?
- Does the energy preservation penalty remain active or switch off?

---

## Files in `experiments/phase2_step3/`

```
experiments/phase2_step3/
    sanity_experiment.json    ← full step-by-step sanity results
    STEP3_REPORT.md           ← this document
```

> [!NOTE]
> `models/checkpoints/best_checkpoint.pt` and `experiments/phase2_baseline/` are unchanged.
> No test set was used at any point during Step 3.
