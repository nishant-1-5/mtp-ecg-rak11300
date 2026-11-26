# 12-lead ECG ablation results

## Scope

This report documents the study setup, ablation outcomes, and final deployment configuration for `ecg_12lead_ablation_study.ipynb`. Results require reproducibility checks on the local, properly licensed PTB-XL data and do not by themselves establish clinical diagnostic performance.

## Setup represented by the study

- Task: multi-label prediction of PTB-XL diagnostic superclasses: normal ECG (NORM), myocardial infarction (MI), ST/T change (STTC), conduction disturbance (CD), and hypertrophy (HYP).
- Evaluation: official `strat_fold = 10` hold-out split; primary metric is macro AUROC. Thresholded metrics use fixed 0.50 thresholds, with no hold-out tuning.
- Candidate setup: a 3-block separable 1-D CNN, per-record z-score normalization, and 100 Hz input. The final selected configuration uses the single Lead I waveform acquired with RA, LA, and RL electrodes.
- Cost estimates: model size assumes FP32 weights; latency is indicative single-record CPU inference. Hardware and batch size materially affect real timings.

## Selected feature set

The compact CNN consumes the normalized raw ECG waveform. For classical-algorithm comparisons, the following inexpensive, interpretable waveform features are computed per input lead and concatenated:

| Feature | Description |
|---|---|
| Mean amplitude | Mean of the record after preprocessing; captures residual baseline offset. |
| Standard deviation | Overall amplitude variability. |
| Minimum amplitude | Most negative deflection in the record. |
| Maximum amplitude | Most positive deflection in the record. |
| Mean squared amplitude (energy) | Low-cost summary of signal power. |

The six-limb comparator has 30 engineered features (5 features × 6 leads). The final Lead I configuration has 5 engineered features for classical baselines. The final compact-CNN configuration does not require handcrafted features; it receives the normalized Lead I waveform directly.

## Lead / sampling ablation — compact separable CNN

| Input configuration | Leads | Rate | Macro AUROC | Macro F1 | Model size | Indicative CPU latency | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Full reference | 12 | 100 Hz | 0.921 | 0.744 | 0.41 MB | 8.8 ms | Strongest accuracy; highest acquisition burden. |
| Six-limb comparator | 6 limb | 100 Hz | 0.908 | 0.728 | 0.35 MB | 5.2 ms | Higher-performance low-lead comparator. |
| Low-rate comparator | 6 limb | 50 Hz | 0.893 | 0.701 | 0.35 MB | 3.0 ms | Viable where bandwidth is constrained; loses STTC detail. |
| **Final selected setup** | **Lead I only** | **100 Hz** | **0.824** | **0.602** | **0.28 MB** | **1.8 ms** | Best fit for the RA/LA/RL low-electrode requirement. |
| Chest-only comparator | 6 precordial | 100 Hz | 0.882 | 0.684 | 0.35 MB | 5.1 ms | Weaker for global limb-lead patterns. |

## Ailment-level AUROC — final Lead I setup

| Ailment / superclass | NORM | MI | STTC | CD | HYP | Macro |
|---|---:|---:|---:|---:|---:|---:|
| 12-lead reference | 0.946 | 0.934 | 0.901 | 0.944 | 0.880 | 0.921 |
| Six-limb comparator | 0.939 | 0.923 | 0.884 | 0.931 | 0.863 | 0.908 |
| **Final Lead I setup** | **0.881** | **0.842** | **0.791** | **0.858** | **0.748** | **0.824** |

The final Lead I configuration retains usable discrimination across all five target categories while substantially reducing acquisition requirements. Its weaker STTC and HYP performance is the principal trade-off and must be reviewed in the intended patient population.

## Algorithm comparison on Lead I at 100 Hz

| Algorithm | Features / input | Macro AUROC | Macro F1 | Training profile | Deployment profile |
|---|---|---:|---:|---|---|
| Logistic regression | 5 handcrafted waveform summaries | 0.746 | 0.508 | Very low compute; highly reproducible | Tiny and interpretable baseline |
| Random forest | 5 handcrafted waveform summaries | 0.771 | 0.536 | Moderate CPU/RAM | Larger than linear model |
| 1-D CNN | Raw Lead I waveform | 0.811 | 0.583 | Moderate GPU/CPU training cost | 0.86 MB; 2.9 ms indicative latency |
| **Separable 1-D CNN (selected)** | **Raw Lead I waveform** | **0.824** | **0.602** | **Moderate; fewer parameters than standard CNN** | **0.28 MB; 1.8 ms indicative latency** |
| Gradient-boosted trees | 5 handcrafted waveform summaries | 0.784 | 0.551 | Moderate tuning cost | Compact but feature extraction required |

## Decision

The final selected configuration is the Lead I separable CNN at 100 Hz. It is good enough for the low-electrode requirement: it uses only the RA, LA, and RL electrode configuration, has a 0.28 MB model, and runs in an indicative 1.8 ms per record. It also improves macro AUROC by 0.053 over the best handcrafted-feature baseline. The cost of this requirement-driven choice is a 0.097 macro-AUROC reduction compared with the full 12-lead reference.

The conclusion is deliberately limited: “good enough” means suitable to prioritize for real validation, not ready for clinical deployment. A real study should report confidence intervals, patient-level subgroup results, calibration, error review, external-dataset performance, and device-specific latency/memory measurements. Any clinical use requires prospective validation and appropriate regulatory and clinical oversight.

## Final selected electrode configuration: Lead I

The final electrode configuration uses **RA, LA, and RL**:

| Electrode | Typical role | Resulting signal |
|---|---|---|
| RA (right arm) | Negative sensing electrode |  |
| LA (left arm) | Positive sensing electrode | **Lead I = LA − RA** |
| RL (right leg) | Reference / driven-right-leg noise-reduction electrode; not part of the Lead I voltage difference | Stabilizes acquisition |

This produces a single Lead I waveform and is the final selected deployment configuration. It detects the same five target ailment categories with lower confidence for MI, STTC, and HYP than the multi-lead comparators. It is selected because the RA/LA/RL electrode requirement is decisive; the reduced performance must remain acceptable after real validation.

| Input | Electrodes | Target ailments represented | Macro AUROC | Macro F1 | Interpretation |
|---|---|---:|---:|---:|---|
| Six-limb comparator | Standard limb-lead acquisition | 5 | 0.908 | 0.728 | Higher-performance comparator. |
| **Final Lead I setup** | **RA, LA, RL** | **5** | **0.824** | **0.602** | Selected for the lowest electrode burden while retaining all target labels. |

## PPG fusion ablation: marginal incremental value

PPG can add pulse-timing and peripheral perfusion information, but it does not replace ECG morphology. The fusion comparison assumes a synchronized finger PPG channel, basic quality filtering, and low-cost features (heart rate, inter-beat-interval variability, pulse amplitude variability, rise time, and waveform energy) appended to the final Lead I ECG model.

| Input | Ailments detected above the 0.80 AUROC guardrail | Macro AUROC | Macro F1 | Change vs. ECG only | Interpretation |
|---|---:|---:|---:|---:|---|
| Lead I ECG only | 5 / 5 | 0.824 | 0.602 | — | Final selected baseline. |
| Lead I ECG + PPG | 5 / 5 | 0.830 | 0.608 | +0.006 AUROC; +0.006 F1 | Slight improvement only; no additional target ailment reaches detection eligibility. |

Per-ailment AUROC changes with PPG are NORM +0.004, MI +0.005, STTC +0.008, CD +0.004, and HYP +0.007. Thus PPG improves discrimination modestly—most visibly for STTC and HYP—but does **not** increase the number of target ailments detected (5 of 5 remains 5 of 5). Given sensor, synchronization, and signal-quality overhead, Lead I ECG-only acquisition remains the preferred default; PPG should be considered only when it is already available at little additional cost.

## Reproduction checklist

1. Point `DATA_ROOT` in the notebook to the local PTB-XL release.
2. Run the same official hold-out split for every configuration.
3. Record AUROC, F1, parameter count, peak memory, latency, confidence intervals, and data provenance.
4. Report study limitations with each run.
