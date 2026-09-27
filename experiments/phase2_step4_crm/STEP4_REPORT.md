# PHASE 2 STEP 4 — CRM 30-EPOCH EXPERIMENT REPORT

**Experiment namespace:** `experiments/phase2_step4_crm/`
**Date:** 2026-09-27
**Duration:** 1029.3 s (17.16 min)
**Artifact verification:** 31/31 checks PASS

---

## 1. Experiment Configuration

| Parameter | Value |
|---|---|
| Model | `LightweightCNNGRUMaskModel` (CRM) |
| Trainable parameters | **70,789** |
| Seed | 123 |
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clip | 5.0 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau (min, factor=0.5, patience=3, min_lr=1e-6) |
| Enhancement weight | 1.0 |
| Classification weight | 0.10 |
| L1 weight | 0.7 |
| L2 weight | 0.3 |
| Energy preservation weight | 0.1 |
| Device | CPU |
| NaN/Inf occurred | **NO** |

> [!NOTE]
> Configuration is identical to the Phase 2 30-epoch direct-STFT baseline in all parameters except the model and loss. This isolates the effect of the CRM formulation change.

---

## 2. Experiment Isolation

| Artifact | Status |
|---|---|
| `models/checkpoints/best_checkpoint.pt` | **UNCHANGED** (verified) |
| `experiments/phase2_baseline/` | **UNCHANGED** (verified) |
| `experiments/phase2_step3/` | **UNCHANGED** (verified) |
| `experiments/phase2_step4_crm/` | New — isolated directory |

---

## 3. Full Training History

| Epoch | Train Loss | Val Loss | Val Acc | SNR Imp (dB) | Enh/Clean Ratio | LR |
|---|---|---|---|---|---|---|
| 1  | 0.119779 | 0.136051 | 31.7% | −7.99 | 0.4343 | 1.00e-03 |
| 2  | 0.111118 | 0.110864 | 34.7% | −8.71 | 0.7508 | 1.00e-03 |
| 3  | 0.110535 | 0.111210 | 33.7% | −8.41 | 0.6691 | 1.00e-03 |
| 4  | 0.109441 | 0.124152 | 34.2% | −8.27 | 0.4425 | 1.00e-03 |
| 5  | 0.105730 | 0.135532 | 36.6% | −7.98 | 0.6235 | 1.00e-03 |
| 6  | 0.100018 | 0.149932 | 36.6% | −7.93 | 0.7261 | 1.00e-03 |
| **7**  | 0.095279 | 0.138687 | 41.1% | −8.02 | 0.7697 | **5.00e-04** |
| 8  | 0.090737 | 0.144665 | 41.1% | −7.91 | 0.7520 | 5.00e-04 |
| 9  | 0.087059 | 0.134677 | 47.0% | −7.88 | 0.7217 | 5.00e-04 |
| 10 | 0.086324 | 0.137185 | 45.5% | −7.79 | 0.7773 | 5.00e-04 |
| **11** | 0.080677 | 0.095111 | 54.0% | −7.66 | 0.8096 | **2.50e-04** |
| 12 | 0.078547 | 0.090820 | 50.5% | −7.63 | 0.8188 | 2.50e-04 |
| 13 | 0.076856 | 0.091373 | 55.4% | −7.63 | 0.8264 | 2.50e-04 |
| 14 | 0.075165 | 0.089178 | 57.9% | −7.61 | 0.8470 | 2.50e-04 |
| 15 | 0.073134 | 0.119053 | 54.0% | −7.59 | 0.8212 | 2.50e-04 |
| 16 | 0.072135 | 0.103284 | 51.0% | −7.69 | 0.8550 | 2.50e-04 |
| 17 | 0.070492 | 0.089290 | 55.9% | −7.61 | 0.8634 | 2.50e-04 |
| 18 | 0.069211 | 0.109881 | 55.4% | −7.59 | 0.8678 | 2.50e-04 |
| **★19** | 0.066632 | **0.083693** | 57.4% | −7.57 | 0.8769 | **1.25e-04** |
| 20 | 0.065110 | 0.088239 | 55.4% | −7.53 | 0.8741 | 1.25e-04 |
| 21 | 0.064392 | 0.090358 | 53.5% | −7.52 | **0.8781** | 1.25e-04 |
| 22 | 0.063713 | 0.088507 | 55.0% | −7.45 | 0.8607 | 1.25e-04 |
| 23 | 0.063969 | 0.087821 | 58.9% | −7.40 | 0.8499 | 1.25e-04 |
| **24** | 0.062023 | 0.086024 | 58.4% | −7.39 | 0.8580 | **6.25e-05** |
| 25 | 0.061081 | 0.085187 | 55.9% | −7.36 | 0.8446 | 6.25e-05 |
| 26 | 0.060135 | 0.087446 | 58.9% | −7.40 | 0.8667 | 6.25e-05 |
| 27 | 0.060641 | 0.096203 | 58.4% | −7.41 | 0.8648 | 6.25e-05 |
| **28** | 0.058920 | 0.087302 | 58.9% | **−7.35** | 0.8550 | **3.13e-05** |
| 29 | 0.059651 | 0.086550 | 58.4% | −7.35 | 0.8577 | 3.13e-05 |
| 30 | 0.059249 | 0.087704 | 58.4% | −7.37 | 0.8615 | 3.13e-05 |

★ = Best validation-loss checkpoint (saved as `best_checkpoint.pt`)

---

## 4. Learning Rate Schedule

| Epoch | LR |
|---|---|
| 1–6 | 1.00e-03 |
| 7–10 | 5.00e-04 |
| 11–18 | 2.50e-04 |
| 19–23 | 1.25e-04 |
| 24–27 | 6.25e-05 |
| 28–30 | 3.13e-05 |

Six LR reductions total. ReduceLROnPlateau correctly tracking validation loss.

---

## 5. Checkpoint Summary

| Checkpoint | Criterion | Epoch | Val Loss |
|---|---|---|---|
| `best_checkpoint.pt` | Lowest val_total_loss | **19** | **0.083693** |
| `last_checkpoint.pt` | Final epoch | 30 | 0.087704 |

**Best val-loss epoch:** 19
**Best val SNR-improvement epoch:** 28 (−7.35 dB)

> [!IMPORTANT]
> These differ. Checkpoint selection criterion is lowest val_total_loss (epoch 19), not SNR improvement. The test evaluation uses epoch 19.

---

## 6. Anti-Collapse Monitoring — Required RMS Ratio Checkpoints

| Epoch | Enh/Clean RMS Ratio | vs. Baseline |
|---|---|---|
| 1  | **0.4343** | Baseline epoch 1: ~0.58 |
| 5  | **0.6235** | Baseline epoch 5: ~0.42 |
| 10 | **0.7773** | Baseline epoch 10: ~0.15 |
| 15 | **0.8212** | Baseline epoch 15: ~0.05 |
| 20 | **0.8741** | Baseline epoch 20: ~0.01 |
| 25 | **0.8446** | Baseline epoch 25: ~0.006 |
| 28 | **0.8550** | Baseline epoch 28: **0.0052** |
| 30 | **0.8615** | — |

**Minimum ratio (30 epochs):** 0.4343 at epoch 1
**Maximum ratio (30 epochs):** 0.8781 at epoch 21

**Baseline epoch-28 ratio:** 0.0052
**CRM epoch-28 ratio:     ** 0.8550

> [!IMPORTANT]
> The CRM formulation completely prevents the near-zero-output collapse observed in the direct STFT regression baseline. The CRM model maintains enhanced/clean RMS ratios between 0.43–0.88 throughout the entire 30-epoch run. The ratio NEVER falls below 0.4343. The baseline's ratio fell monotonically to 0.0052 by epoch 28.

---

## 7. Final Test Results (Best Checkpoint, Epoch 19)

Test set was held out during all training. Evaluated exactly once using `best_checkpoint.pt`.

| Metric | Value |
|---|---|
| **Test total loss** | 0.081364 |
| Test enhancement loss | 0.000804 |
| Test classification loss | 0.805607 |
| **Input SNR** | +7.60 dB |
| **Enhanced SNR** | +0.12 dB |
| **SNR improvement** | **−7.47 dB** |
| Noisy STOI | 0.7789 |
| Enhanced STOI | **0.7266** |
| Noisy PESQ | 1.4911 |
| Enhanced PESQ | **1.5245** |
| **Classification accuracy** | **61.88%** |
| Fixed-sample Enh/Clean RMS ratio | **0.8769** |
| Fixed-sample enhanced RMS | 0.0711 |
| Fixed-sample clean RMS | 0.0811 |

---

## 8. Classification Results (Test)

| | Pred 0 (stat.) | Pred 1 (non-stat.) | Pred 2 (impulsive) |
|---|---|---|---|
| **True 0 (stat.)** | 45 | 13 | 6 |
| **True 1 (non-stat.)** | 31 | 19 | 19 |
| **True 2 (impulsive)** | 2 | 6 | 61 |

| Class | Accuracy |
|---|---|
| Class 0 (stationary) | 70.3% |
| Class 1 (non-stationary) | 27.5% |
| Class 2 (impulsive) | 88.4% |
| **Overall** | **61.88%** |

Class 1 (non-stationary) remains the hardest class. The classifier no longer predicts Class 2 exclusively — all three classes receive predictions.

---

## 9. STOI Validation Trajectory

| Epoch | Noisy STOI | Enhanced STOI |
|---|---|---|
| 1  | 0.7868 | 0.7341 |
| 5  | 0.7868 | 0.7180 |
| 10 | 0.7868 | 0.7263 |
| 15 | 0.7868 | 0.7281 |
| 20 | 0.7868 | 0.7348 |
| 25 | 0.7868 | 0.7414 |
| 30 | 0.7868 | 0.7417 |

STOI of enhanced signal improves from 0.718 (epoch 5) to 0.742 (epoch 30), converging toward but not reaching noisy STOI.

---

## 10. PESQ Validation Trajectory

| Epoch | Noisy PESQ | Enhanced PESQ |
|---|---|---|
| 1  | 1.4984 | 1.4498 |
| 5  | 1.4984 | 1.4245 |
| 10 | 1.4984 | 1.4958 |
| 15 | 1.4984 | 1.5060 |
| 20 | 1.4984 | 1.5273 |
| 25 | 1.4984 | 1.5386 |
| 30 | 1.4984 | 1.5449 |

PESQ of enhanced signal rises above noisy PESQ by epoch 15 and continues improving. By epoch 30, enhanced PESQ (1.545) exceeds noisy PESQ (1.498) by +0.047.

---

## 11. Direct Baseline Comparison

| Metric | Direct-STFT Baseline | CRM (Step 4) | Δ | Direction |
|---|---|---|---|---|
| Input SNR | +7.60 dB | +7.60 dB | 0.00 | Same |
| Enhanced SNR | −0.01 dB | +0.12 dB | **+0.13 dB** | ↑ |
| **SNR improvement** | **−7.60 dB** | **−7.47 dB** | **+0.13 dB** | ↑ |
| Noisy STOI | 0.7789 | 0.7789 | 0.00 | Same |
| Enhanced STOI | 0.4307 | **0.7266** | **+0.2959** | ↑ |
| Noisy PESQ | 1.4911 | 1.4911 | 0.00 | Same |
| Enhanced PESQ | 1.0442 | **1.5245** | **+0.4803** | ↑ |
| Classification accuracy | 34.16% | **61.88%** | **+27.7 pp** | ↑ |
| Enh/clean RMS (ep.28) | **0.0052** | **0.8550** | **+0.8498** | ↑ |

---

## 12. Project Target Comparison

| Target | Required | CRM Result | Status |
|---|---|---|---|
| SNR improvement | > +15 dB | −7.47 dB | **NOT MET** |
| STOI enhanced | > 0.85 | 0.7266 | **NOT MET** |
| PESQ enhanced | > 2.5 | 1.5245 | **NOT MET** |
| Params < 100k | < 100,000 | 70,789 | ✓ MET |

Project performance targets are not yet met. This is expected at this phase.

---

## 13. Analysis — Classification of Outcome

### Evidence for each outcome category:

**Evidence for Outcome B — PARTIAL IMPROVEMENT:**

1. **Collapse eliminated:** Enh/clean RMS ratio stayed between 0.43–0.88 across all 30 epochs. Baseline collapsed to 0.0052. This is a fundamental qualitative difference.

2. **Enhanced STOI: 0.7266 vs noisy STOI: 0.7789.** Still a degradation of −0.052 STOI, but dramatically better than the baseline's degradation of −0.348 STOI. The enhanced speech is no longer being largely destroyed.

3. **Enhanced PESQ: 1.5245 vs noisy PESQ: 1.4911.** The CRM model produces enhanced audio with **higher PESQ than the noisy input**. This is the first time in this experiment series that enhancement shows any objective speech quality gain over the raw noisy signal.

4. **SNR improvement: −7.47 dB** — still negative, meaning enhanced signal has lower SNR than input. However, SNR is only −0.13 dB worse than direct-STFT baseline, while delivering massively better STOI and PESQ.

5. **Classification accuracy: 61.88%** — genuine multi-class learning (Classes 0, 1, 2 all non-trivially predicted). Baseline produced 34.16% by predicting Class 2 for all samples.

6. **Val SNR improvement trajectory:** Went from −7.99 dB (epoch 1) to −7.35 dB (epoch 28) — a +0.64 dB improvement over the run, but still firmly negative.

**Evidence against Outcome A — MEANINGFUL ENHANCEMENT:**

1. SNR improvement remains negative throughout all 30 epochs. The system degrades SNR.
2. Enhanced STOI (0.7266) is below noisy STOI (0.7789) — intelligibility is still slightly degraded by enhancement.
3. SNR improvement of −7.47 dB is far from the +15 dB project target.

**Evidence against Outcome C — FAILURE:**

1. No collapse. Output energy is non-trivially non-zero throughout.
2. PESQ improves above noisy baseline — a genuinely positive speech quality signal.
3. Classification is meaningfully learning (61.88% vs 34.16% random baseline).
4. Training is stable: no NaN/Inf, smooth loss curves, gradient norms in range.

### Verdict:

**OUTCOME B — PARTIAL IMPROVEMENT**

The CRM formulation successfully eliminates the near-zero-output collapse that made the direct-STFT baseline completely non-functional. The model now preserves substantial signal energy (ratio ~0.86), achieves positive PESQ improvement over noisy input, and learns meaningful 3-class noise classification. However, SNR improvement remains negative throughout the run and STOI is still below the noisy baseline. The system does not yet achieve meaningful speech enhancement in the SNR or intelligibility sense.

---

## 14. Reproducibility Verification

| Item | Status |
|---|---|
| Seed = 123 | ✓ Confirmed |
| Fresh initialization | ✓ Confirmed |
| Deterministic (CPU) | ✓ Confirmed |
| Exact param count = 70,789 | ✓ Verified at init and in checkpoint |
| No NaN/Inf | ✓ Confirmed (31-check script) |
| No test leakage | ✓ Test evaluated once, after epoch 30 |
| Phase 1 checkpoint intact | ✓ Verified |
| Phase 2 baseline intact | ✓ Verified |
| Phase 2 Step 3 intact | ✓ Verified |

Artifact verification: **31/31 PASS**

---

## 15. Artifacts Saved

```
experiments/phase2_step4_crm/
    config.yaml               ← full training config
    history.json              ← 30-epoch epoch-by-epoch log
    metrics.json              ← summary + final test results
    best_checkpoint.pt        ← epoch 19 (lowest val_total_loss = 0.083693)
    last_checkpoint.pt        ← epoch 30 (final state)
    STEP4_REPORT.md           ← this document
```

---

## 16. Summary of Key Findings

1. **Zero-collapse confirmed eliminated** by CRM (ratio 0.43→0.88 vs baseline 0.58→0.005)
2. **PESQ improves** above noisy baseline (1.5245 vs 1.4911) — first genuine speech quality gain
3. **STOI still slightly degraded** relative to noisy (0.7266 vs 0.7789) but far better than baseline (0.4307)
4. **SNR improvement remains negative** (−7.47 dB) — enhancement does not yet improve SNR
5. **Classification working** (61.88% vs 34.16%) with genuine multi-class predictions
6. **Parameter constraint satisfied**: 70,789 < 100,000
7. **Training stable**: no NaN/Inf, smooth loss curves, 6 LR reductions
8. **Project targets not yet met**: SNR +15 dB, STOI >0.85, PESQ >2.5

---

## PHASE 2 STEP 4 — PASS
