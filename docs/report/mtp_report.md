# Edge-Based Single-Lead ECG Sensing and Multi-Label Cardiac Pathology Classification on a RAK11300 (RP2040)

**Lakshit Sachdeva (B23146), Nishant Nehra (B23163)**

Mentor: Dr. Varun Dutt

MTP report

## Abstract

We present a complete pipeline that classifies 10-second Lead I ECG windows into five PTB-XL
diagnostic superclasses (NORM, MI, STTC, CD, HYP) on a RAK11300 module, a Raspberry Pi RP2040
(dual Cortex-M0+, 133 MHz, 264 KB SRAM, no FPU). The hardware's three-electrode front end
(RA, LA, RL) records Lead I only, so every model was retrained on Lead I and compared with a
Leads I/II/III baseline. Across five model families, the 3-lead to Lead I drop is about 0.05 macro
AUC when the input features are the same. The deployed 5.7k-parameter INT8 1D-CNN reaches a macro
AUC of **0.829** on the held-out fold, matching a 2.5M-parameter ResNet with attention trained on
the same lead (0.831). A lead ablation with a fixed compact CNN puts the full 12-lead reference at
0.881 macro AUROC and shows that 50 Hz sampling costs under 0.01.

Running this on the device required three things beyond the model:

- **Interval delineation.** A median-beat delineator for PR/QRS/QT. Its Lead I QRS separates
  conduction disturbances as well as the 12-lead GE 12SL measurement (AUC 0.754 vs 0.755).
- **Quality gate.** A gate based on beat-to-template correlation. It passes 97.4 % of clean windows
  and rejects 82.5 % of windows corrupted with −6 dB electrode-motion noise from the MIT-BIH Noise
  Stress Test Database.
- **Heap-free C++ firmware.** It matches the Python reference on 21,387 of 21,388 PTB-XL windows
  (features) and on all 3,778 test windows (decisions).

The board was not available, so the unmodified firmware was run on the rp2040js emulator. All 100
randomly drawn test windows match the Python reference end-to-end, including TensorFlow Lite Micro
inference. The pipeline uses **27.7 KB of SRAM** and about **300 ms per 10 s window** at 125 MHz
(about 3 % of one core).

## 1. Introduction

Cardiovascular disease calls for continuous monitoring, but diagnostic ECG relies on stationary
12-lead machines. A low-cost wearable that screens a single lead and flags windows for review
needs three properties:

1. a classifier small enough for a microcontroller;
2. a way to know when its own input is unreliable;
3. outputs a clinician can interpret.

This project targets a RAK11300 module (RP2040 + SX1262 LoRa) with a three-electrode analog front
end.

**Contributions.**

1. A controlled comparison of five model families (gradient boosting, random forest, STFT 2D-CNN,
   lightweight 1D-CNN, ResNet with attention) on Leads I/II/III versus Lead I only, on PTB-XL's
   patient-disjoint test fold. A lead-set ablation with one fixed compact CNN (12, 6 limb, 3 limb,
   6 chest, Lead I; 100 and 50 Hz) quantifies what each lead set and sampling rate is worth.
2. An INT8 deployment recipe for single-lead ECG. Per-window z-scoring of the input reduces the
   INT8 accuracy loss from 0.05 to under 0.001 AUC.
3. A Lead I median-beat delineator for PR, QRS and QT, validated against GE 12SL global intervals
   from PTB-XL+.
4. A signal-quality gate and rule-based rhythm flags, validated against PTB-XL rhythm statements
   and against real recorded noise.
5. A heap-free C++ implementation of the whole pipeline with bit-level parity tests against Python,
   TensorFlow Lite Micro integration on the RP2040, and measured memory and latency.

All code, models and results are public at
<https://github.com/nishant-1-5/mtp-ecg-rak11300>.

## 2. Background

**Datasets.** PhysioNet/CinC 2021 (~88k recordings, 30+ classes) is broad but mixes sources and
record lengths. CPSC 2018 covers nine arrhythmia classes but little structural disease. MIT-BIH
Arrhythmia remains the beat-level reference but has no 10 s diagnostic windows. Icentia11k and CinC
2017 are single-lead but focus on rhythm. PTB-XL [1] provides 21,799 ten-second 12-lead records
from 18,869 patients, with SCP-ECG statements aggregated into five superclasses. It also ships a
patient-stratified ten-fold split, which makes it the best fit for multi-label diagnosis with
leak-free evaluation.

**State of the art and the deployment gap.** Ensembles of deep 1D-ResNets with self-attention
(e.g. the CinC 2021 winning entry) reach macro AUCs of about 0.92–0.94 on 12 leads. They need
megabytes of weights and activations, which is orders of magnitude beyond a Cortex-M0+.

**TinyML.** Depthwise-separable convolutions, global average pooling and post-training INT8
quantisation make CNNs feasible on microcontrollers. TensorFlow Lite Micro (TFLM) with CMSIS-NN
kernels is the standard runtime [4].

## 3. Hardware and data

**Target.** RAK11300: RP2040 with two Cortex-M0+ cores (the Pico SDK default is 125 MHz), 264 KB
SRAM and 2 MB of QSPI flash executed in place through a 16 KB cache. The chip has no FPU, and
floating point uses the optimised routines in the boot ROM. We kept a self-imposed SRAM target of
< 30 KB for the ECG pipeline, which leaves room for the LoRaWAN stack and both cores' stacks.
Earlier project documents assumed an ESP32. Its figures (520 KB SRAM, 240 MHz) do not apply.

**Lead.** The front end uses RA, LA and RL (reference) electrodes. Without LL it can measure only
the RA–LA difference, i.e. Lead I. Leads II and III, and hence the original 3-lead model, are not
available on this hardware.

**Data.**

- PTB-XL v1.0.3 at 100 Hz (1000 samples per window). Records with at least one diagnostic
  superclass: 19,230 for training (folds 1–9) and 2,158 for testing (fold 10). Fold 10 is never
  used for training, tuning or threshold selection.
- PTB-XL+ [2] supplies GE Marquette 12SL global intervals, used only as a reference for
  delineation.
- The MIT-BIH Noise Stress Test Database (NSTDB) [3] supplies real baseline-wander, electrode-motion
  and muscle-artifact recordings. Each record is split in half: one half for exploration, the other
  for evaluation.

PTB-XL stores samples at 1000 LSB/mV in 16 bits, so the firmware receives exactly the training data
as `int16` ADC counts.

## 4. Methods

### 4.1 System overview

![Per-window processing on the device](figures/fig_pipeline.png)

Each 10 s window passes through feature extraction, a quality gate, rule-based rhythm flags, and the
INT8 CNN (only for windows that pass the gate).

### 4.2 Features

**R peaks.** Pan-Tompkins style detection:

1. zero-phase 5–15 Hz Butterworth band-pass;
2. squared first difference;
3. 80 ms moving-window integration;
4. peak picking with a 350 ms refractory distance and a prominence threshold of 0.5 × the
   standard deviation of the integrated signal.

**Rhythm and quality features.**

- RR-based features: mean/min/max/std of HR, RR statistics, SDNN, RMSSD, pNN50.
- Hjorth complexity.
- Welch PSD (256-point Hann window, 50 % overlap), from which come the fraction of power below
  0.5 Hz (baseline wander) and an SNR proxy.

**Delineation.** The earlier interval code measured the R wave itself: median QT was 0.8 ms and
median PR 10 ms. Its "QRS" was the width of the energy envelope, about 96 ms for every class.
It was replaced by a median-beat delineator:

1. 0.5–40 Hz zero-phase band-pass.
2. Each detected peak moves to the largest |x − mean| within ±80 ms.
3. Sample-wise median beat from −300 to +700 ms around R; at least 3 beats are required.
4. QRS onset and offset: walk outwards from the steepest point on each side of R until |slope|
   stays below 8 % (onset) or 5 % (offset) of the maximum QRS slope for two samples. The edge is
   linearly interpolated.
5. T end: tangent method at the steepest return slope after the T peak.
6. P onset: where the P deviation falls below half its amplitude. P is treated as absent below
   0.03 mV, and PR is then reported as missing.

Thresholds were grid-searched on 2,000 training windows against 12SL.

**Quality index.** The mean Pearson correlation between each beat and the median beat. It is
computed from the same template, so it costs almost nothing extra.

### 4.3 Models

- **Tabular:** XGBoost and random forest (multi-output), on 24 Lead I features. The 3-lead baseline
  uses 36 features.
- **1D-CNN:** a strided Conv1D (16 filters), two separable Conv1D blocks (32 and 64 filters) with
  batch normalisation and pooling, global average pooling, Dense(32), and five sigmoid outputs.
  5,653 parameters.
- **STFT 2D-CNN:** 16 × 32 spectrogram grids with separable 2D convolutions.
- **ResNet + 4-head self-attention:** a performance ceiling, not deployable.

All models use binary cross-entropy, early stopping on validation macro AUC (last 10 % of the
training arrays), and a fixed seed.

### 4.4 INT8 quantisation

Full-integer post-training quantisation with 200 calibration windows. With raw-mV input, the INT8
input range had to cover rare ±7 mV windows. This left too few levels for normal ECG amplitudes,
and INT8 AUC fell to 0.78 (0.57 with more calibration data). The Lead I CNN therefore receives each
window z-scored as `(x − mean) / (std + 0.001 mV)`. The firmware computes the mean and standard
deviation from exact 64-bit integer moments of the ADC counts and writes the quantised window
directly into the TFLM input tensor.

![INT8 accuracy versus input normalisation](figures/fig_int8_normalisation.png)

### 4.5 Decision logic

**Quality gate.** A window is unreliable, and the CNN is skipped, if any of these holds:

- peak-to-peak < 0.1 mV;
- more than 1 % of samples sit on an ADC rail;
- fewer than 4 peaks, or mean HR outside 30–220 bpm;
- fewer than 3 beats are usable for the median beat;
- template correlation < 0.598 (2 % quantile of clean training windows);
- baseline-wander fraction > 0.43 (99.5 % quantile).

**Rhythm flags.** Standard clinical limits:

- TACHY: HR > 100 bpm;
- BRADY: HR < 60 bpm;
- WIDE_QRS: QRS ≥ 120 ms;
- LONG_PR: PR > 200 ms;
- IRREGULAR: RR coefficient of variation ≥ 0.095 (Youden-optimal for AF/flutter on the training
  folds);
- AF_SUSPECT: IRREGULAR and no P wave found.

**Class decisions.** Per-class thresholds on the CNN probabilities maximise F1 on the training
windows the CNN never trained on. A rule that also raises CD whenever WIDE_QRS is set did not
improve CD F1 on that data (0.655 vs 0.658), so CD comes from the CNN alone.

### 4.6 Firmware

- **Language and memory.** C++14, no heap and no STL. All scratch memory is one caller-owned
  workspace.
- **Filtering.** The filters reproduce `scipy.signal.filtfilt` exactly: odd padding, `lfilter_zi`
  initial state, direct-form II transposed.
- **Peak detection.** Reproduces `find_peaks`: plateau handling, distance rule applied before
  prominence.
- **Buffer reuse.** A single 1,030-float buffer is reused for both filters, the integrated signal
  and the FFT.
- **TFLM.** `pico-tflmicro` (TFLM + CMSIS-NN) registers only the ten operators the model uses. The
  5,780 B feature workspace is overlaid on the arena's activation region: features finish before
  the input tensor is written. The firmware verifies at start-up that the overlay fits.

**Parity testing.** Host builds of the same sources are compared with the Python reference on every
PTB-XL window. Firmware images embed reference outputs and report PASS/FAIL themselves.

Two problems appeared only on the RP2040:

1. **NaN handling.** The Pico SDK's fast float routines do not propagate NaN. All NaN tests
   therefore inspect the bit pattern, and NaN never enters arithmetic.
2. **Kernel rounding.** Desktop TFLite's XNNPACK kernels round requantisation differently from
   CMSIS-NN by up to 9 output LSB. The reference kernels (`BUILTIN_REF`) agree with the device to
   1 LSB and are used for all Python INT8 evaluation.

### 4.7 Evaluation

All numbers are on PTB-XL fold 10 (n = 2,158). We report:

- macro AUC and per-class AUC;
- F1, sensitivity and specificity at the fitted thresholds;
- for flags, sensitivity, specificity and PPV against PTB-XL statements (STACH, SBRAD, AFIB/AFLT,
  1AVB) and against 12SL QRS ≥ 120 ms.

## 5. Results

### 5.1 Lead I versus three leads

![Macro AUC per model family](figures/fig_lead_comparison.png)

| Model | Leads I–III | Lead I |
|---|---:|---:|
| XGBoost (tabular) | 0.795 | 0.786 |
| Random forest (tabular) | 0.786 | 0.778 |
| STFT 2D-CNN | 0.828 | 0.785 |
| 1D-CNN, float32 | 0.883 | 0.830 |
| **1D-CNN, INT8 (deployed)** | 0.866 | **0.829** |
| ResNet + attention | 0.894 | 0.831 |

- **Loss from fewer leads.** With identical feature definitions, every model family loses about
  0.05 macro AUC (tabular models with the original features: 0.744 and 0.732). MI suffers most
  (0.867 → 0.785, float), consistent with infarct localisation needing spatial information.
- **Model size.** On Lead I, the 5.7k-parameter CNN equals the ResNet ceiling, so a bigger network
  buys nothing here.
- **Per class (deployed INT8).** CD 0.812, HYP 0.791, MI 0.786, NORM 0.887, STTC 0.870.

### 5.2 Lead and sampling-rate ablation

A separate ablation asks how much each lead set is worth, independent of our deployed architecture.
It uses one compact separable CNN (2.75k–3.0k parameters), trained identically on every input
(`ecg_12lead_ablation_study.ipynb`, executed in full by `scripts/run_lead_ablation.py`). Each
configuration is a single run with seed 42 on fold 10. The 95 % CIs come from 1,000 bootstrap
resamples of the test records.

![Lead-set ablation with the same compact CNN](figures/fig_ablation.png)

| Input | Leads | Rate | Macro AUROC [95 % CI] | Macro F1 (0.5) |
|---|---:|---:|---|---:|
| 12 leads | 12 | 100 Hz | 0.881 [0.872, 0.890] | 0.619 |
| 6 limb (I, II, III, aVR, aVL, aVF) | 6 | 100 Hz | 0.854 [0.844, 0.864] | 0.541 |
| 3 limb (I, II, III) | 3 | 100 Hz | 0.855 [0.845, 0.865] | 0.527 |
| 6 chest (V1–V6) | 6 | 100 Hz | 0.848 [0.837, 0.859] | 0.535 |
| 6 limb | 6 | 50 Hz | 0.856 [0.846, 0.867] | 0.553 |
| **Lead I (hardware)** | 1 | 100 Hz | **0.803** [0.791, 0.814] | 0.410 |
| Lead I | 1 | 50 Hz | 0.794 [0.782, 0.806] | 0.415 |

Findings:

- **Going from 12 leads to Lead I costs 0.078 AUROC** with this model. The CIs of Lead I and every
  multi-lead set do not overlap.
- **Six limb leads add nothing over three.** aVR, aVL and aVF are linear combinations of I and II,
  so the CNN gains no information (0.854 vs 0.855).
- **The chest leads alone are not better than the limb leads** (0.848). The 12-lead gain comes from
  combining both views.
- **Halving the rate to 50 Hz barely matters** (−0.009 for Lead I, +0.002 for six limb leads), so
  100 Hz is not the accuracy bottleneck.
- **Per class**, Lead I loses most on MI (0.878 → 0.746) and CD (0.888 → 0.766). NORM and STTC
  stay within 0.06 of 12 leads, and HYP is weak for every input (0.77–0.79).
- **Simple waveform summaries are not enough.** Logistic regression and random forest on 5
  per-lead statistics (mean, std, min, max, energy) reach only 0.609 and 0.626 on Lead I (0.728 and
  0.772 on six limb leads). Our clinical Lead I features (rhythm, intervals, quality) give 0.786
  with XGBoost (§5.1).
- **Architecture matters on one lead.** The deployed 5.7k-parameter CNN with z-scored input
  reaches 0.830 float (0.829 INT8) on Lead I, against 0.803 for the 2.75k compact model. Its INT8
  size is 25 KB, against 18 KB for the compact model.

### 5.3 Delineation

![Median-beat delineation and agreement with 12SL](figures/fig_delineation.png)

| Interval | Coverage | Bias vs 12SL | MAE | r | CD AUC (Lead I) | CD AUC (12SL) | CD AUC (old) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QRS | 100 % | −5.8 ms | 13.9 ms | 0.69 | **0.754** | 0.755 | 0.483 |
| PR | 93.7 % | −23.9 ms | 28.3 ms | 0.67 | 0.603 | 0.610 | 0.538 |
| QT | 100 % | −25.8 ms | 40.4 ms | 0.52 | 0.550 | 0.556 | 0.546 |

- **QRS.** Lead I QRS ≥ 120 ms occurs in 33.1 % of CD windows and 1.7 % of NORM windows
  (12SL: 31.7 % and 0.5 %).
- **Effect on the tabular models.** Replacing the broken intervals raised Lead I XGBoost from 0.744
  to 0.786.
- **Interpretation.** 12SL global intervals span all 12 leads, so shorter single-lead values are
  expected.

### 5.4 Decisions and flags

At the fitted thresholds, macro F1 is 0.604 on gate-passed windows, against 0.515 with a flat 0.5
threshold.

| Class | F1 | Sensitivity | Specificity |
|---|---:|---:|---:|
| CD | 0.589 | 0.534 | 0.919 |
| HYP | 0.423 | 0.462 | 0.902 |
| MI | 0.559 | 0.606 | 0.814 |
| NORM | 0.792 | 0.954 | 0.627 |
| STTC | 0.657 | 0.686 | 0.871 |

| Flag | Reference | Sensitivity | Specificity | PPV |
|---|---|---:|---:|---:|
| TACHY | STACH | 0.939 | 0.960 | 0.481 |
| BRADY | SBRAD | 0.921 | 0.846 | 0.153 |
| IRREGULAR | AFIB/AFLT | 0.905 | 0.868 | 0.333 |
| AF_SUSPECT | AFIB/AFLT | 0.551 | 0.991 | 0.818 |
| WIDE_QRS | 12SL QRS ≥ 120 ms | 0.776 | 0.964 | 0.652 |
| LONG_PR | 1AVB | 0.380 | 0.946 | 0.211 |

The low PPVs for TACHY and BRADY largely reflect labelling: PTB-XL codes STACH and SBRAD only for
sinus rhythm. LONG_PR inherits the −24 ms PR bias.

### 5.5 Signal quality under real noise

![Gate rejection and accuracy under NSTDB noise](figures/fig_noise_stress.png)

PTB-XL's own noise annotations were not separable by any cheap index (AUC ≈ 0.5 for SNR, baseline
power and amplitude), so the gate was evaluated with NSTDB noise added to fold 10.

- **Clean windows.** The gate rejects 2.6 %.
- **Rejection grows with noise.** At −6 dB it rejects 82.5 % of electrode-motion windows, 47.4 %
  with baseline wander and 43.0 % with muscle artifact.
- **Passed windows score higher.** Under −6 dB electrode motion, the windows the gate passes reach
  macro AUC 0.694, against 0.653 for all windows.
- **Noise sensitivity of the CNN.** It tolerates baseline wander and muscle noise well. Electrode
  motion, the artifact most typical of wearables, degrades it most.

### 5.6 On the RP2040 (emulated)

![Measured SRAM and latency per window](figures/fig_memory_latency.png)

| Quantity | Value |
|---|---:|
| Correctness, 100 random fold-10 windows, full pipeline incl. TFLM | 100/100 match Python |
| TFLM arena used (activation head / persistent tail) | 20,788 B (16,000 / 4,788) |
| Peak core-0 stack | 1,636 B |
| Pipeline SRAM (arena incl. features, buffers, stack, interpreter) | **27,708 B** |
| Flash: model + our code / TFLM runtime | ~43 KB / 107 KB |
| Feature extraction | 112–126 ms |
| CNN inference (single core, CMSIS-NN) | 175 ms |
| Whole pipeline per 10 s window | 294–308 ms |

## 6. Discussion

**Corrections to the mid-term results.**

- The reported "0.8626 macro AUC" was a Keras validation-split metric. On the test fold, the 3-lead
  CNN scores 0.883 (float) and 0.866 (INT8).
- The claim that "Lead I QRS > 120 ms isolates CD" came from a plot that showed RR range under a
  QRS label.
- The original interval features did not measure intervals at all.

All three are corrected here, and the original artifacts are kept (marked "MISLABELED") for
transparency.

**Limitations.**

1. **Data and evaluation.** Results are from one public dataset recorded with clinical equipment,
   evaluated on a single held-out fold. Each ablation configuration is a single training run.
2. **Device timing.** The rp2040js emulator counts instruction cycles but not flash-cache misses,
   so on-board latency will be somewhat higher. Power was not measured.
3. **Sampling rate.** At 100 Hz, QRS resolution is coarse. About 4 % of windows sit at the
   measurement floor near 58 ms.
4. **QT.** The Lead I QT agrees only moderately with 12SL (r = 0.52) and is not used for any flag.
5. **Labels.** PTB-XL superclass labels come from cardiologist statements on 12 leads, which caps
   what any Lead I model can learn for MI and HYP.
6. **Clinical scope.** This is a screening prototype. Clinical use would require prospective
   validation, safety assessment and regulatory review.

## 7. Future work

- Run the existing `rak11300_pipeline` image on hardware to measure real latency and power.
- Record data with the lab front end. Set the ADC scale and rails, re-check the gate thresholds on
  device data, and fine-tune if the front end's noise floor differs from PTB-XL.
- Correct the PR bias (a ~176 ms threshold) and evaluate longer windows for HRV.

## 8. Conclusion

A Lead I-only screening pipeline fits comfortably on an RP2040:

- **Accuracy:** 0.829 macro AUC over five superclasses.
- **Interpretable flags:** rate, rhythm irregularity and QRS width, with QRS width as informative
  as a 12-lead commercial measurement.
- **Self-check:** a quality gate that withholds a verdict on unreliable input.
- **Footprint:** all within 27.7 KB of SRAM and about 3 % of one core.

## References

1. P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific
   Data* 7, 154 (2020).
2. N. Strodthoff et al., "PTB-XL+, a comprehensive electrocardiographic feature dataset,"
   *Scientific Data* 10, 279 (2023).
3. G. B. Moody, W. E. Muldrow, R. G. Mark, "A noise stress test for arrhythmia detectors,"
   *Computers in Cardiology* 11, 381–384 (1984).
4. R. David et al., "TensorFlow Lite Micro: Embedded machine learning for TinyML systems," *Proc.
   MLSys* (2021).
5. J. Pan, W. J. Tompkins, "A real-time QRS detection algorithm," *IEEE Trans. Biomed. Eng.* 32(3),
   230–236 (1985).
6. N. Strodthoff, P. Wagner, T. Schaeffter, W. Samek, "Deep learning for ECG analysis: Benchmarks
   and insights from PTB-XL," *IEEE J. Biomed. Health Inform.* 25(5), 1519–1528 (2021).
7. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation* 101(23),
   e215–e220 (2000).

## Appendix A: Reproducing the results

```
python scripts/run_notebook.py --lead-set lead1 notebooks/01_preprocess.ipynb
python scripts/fix_cache.py && python scripts/build_lead1_features.py
python scripts/run_notebook.py --lead-set lead1 notebooks/1*.ipynb notebooks/20_quantization.ipynb notebooks/30_metrics.ipynb
python scripts/compare_lead_sets.py
python scripts/run_lead_ablation.py          # 12/6/3/1-lead and 50 Hz ablation (~25 min)
python scripts/validate_delineation.py --split test
python scripts/orchestrator.py fit && python scripts/orchestrator.py evaluate
python scripts/compare_features.py --exe build/lead1_host.exe
python scripts/compare_pipeline.py
powershell -File firmware/rak11300_bench/build.ps1    # then run the .uf2 on rp2040js
python scripts/make_report_figures.py && python scripts/build_report.py
```

Design notes with full detail are in `docs/` (`sram_budget.md`, `lead1_retrain.md`,
`delineation.md`, `orchestrator.md`, `phase2_tflm.md`).
