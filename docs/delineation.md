# Lead I PR / QRS / QT delineation

The old interval features in `scripts/fix_cache.py` did not work (`docs/phase1_feature_port.md` §5):

- `Mean_QT` was mostly 0 ms, because the search started at the R peak and `argmax` found R itself.
- `Mean_PR` was 10 ms, because it measured to the sample just before R.
- `Mean_QRS` was the width of the Pan-Tompkins energy envelope, which is about 96 ms for every class.

They are replaced by `scripts/delineation.py`. The Lead I training features (v2) are built with
`scripts/build_lead1_features.py`. The C++ port is in `firmware/ecg_features`.

## Method (100 Hz, Lead I)

1. **Band-pass:** 0.5–40 Hz, Butterworth order 2, zero-phase (`filtfilt`).
2. **R refinement:** each Pan-Tompkins peak moves to the largest |x − mean| within ±80 ms.
3. **Median beat:** sample-wise median from −300 ms to +700 ms around R, over all beats that fit in
   the window. At least 3 beats are required; with fewer, all intervals are NaN.
4. **QRS onset and offset:** walk outwards from the steepest point on each side of R. Stop once
   |slope| has stayed below k × (max QRS slope) for 2 samples (k = 0.08 for onset, 0.05 for offset).
   The edge is linearly interpolated, so it is finer than one sample. Allowing 2 samples below the
   threshold keeps the walk from stopping at the one-sample flat point at the S wave.
5. **Isoelectric level:** mean of the 30 ms before QRS onset.
6. **T end:** find the largest deviation from the isoelectric level between QRS offset + 50 ms and
   R + 0.5·RR. T end is where the tangent at the steepest return slope crosses the isoelectric line.
7. **P onset:** find the largest deviation 40–250 ms before QRS onset, then go back to where the
   deviation falls below 50 % of the P amplitude. If the P amplitude is under 0.03 mV, P is treated
   as absent and PR is NaN. This happens in 6.3 % of windows, for example when P waves are flat or absent.

The thresholds were grid-searched on 2,000 random **training** windows against the GE 12SL reference
(`scripts/tune_delineation.py`). Fold 10 was used only for the final numbers
below.

## Validation (fold 10, n = 2,158)

The reference is **GE Marquette 12SL global intervals from PTB-XL+**
(Strodthoff et al. 2023, [doi:10.1038/s41597-023-02153-8](https://doi.org/10.1038/s41597-023-02153-8)).
12SL measures on all 12 leads at 500 Hz. Its global intervals span the earliest onset to the
latest offset across leads, so a single-lead measurement is expected to come out shorter. Run
`python scripts/validate_delineation.py --split test`.

| Interval | Coverage | Bias vs 12SL | MAE | Pearson r | CD AUC (new Lead I) | CD AUC (12SL, 12-lead) | CD AUC (old feature) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QRS | 100 % | −5.8 ms | 13.9 ms | 0.687 | **0.754** | 0.755 | 0.483 |
| PR | 93.7 % | −23.9 ms | 28.3 ms | 0.670 | 0.603 | 0.610 | 0.538 |
| QT | 100 % | −25.8 ms | 40.4 ms | 0.515 | 0.550 | 0.556 | 0.546 |

"CD AUC" uses the interval alone as a CD-vs-rest score.

- **QRS:** Lead I QRS duration now separates conduction disturbances as well as the 12-lead
  commercial measurement does. QRS ≥ 120 ms occurs in 33.1 % of CD windows and 1.7 % of NORM
  windows (12SL: 31.7 % and 0.5 %). Median QRS is 99 ms for CD and 81 ms for NORM.
- **PR:** tracks 12SL (r = 0.67) but reads about 24 ms short. The P onset is placed at half
  amplitude, which is later than the true onset.
- **QT:** the weakest of the three (r = 0.52). At 100 Hz the T-wave end in Lead I is poorly
  defined. Don't build clinical QT rules on it.

`results/features/lead1_justification.png` replaces the mislabeled slide figure.
`results/features/delineation_{train,test}.csv` hold per-record values next to 12SL.

## Effect on the classical models (Lead I, fold 10)

| Features | XGBoost | Random Forest |
|---|---:|---:|
| v1 (old intervals) | 0.7443 | 0.7321 |
| **v2 (delineated intervals)** | **0.7864** | **0.7780** |
| 3-lead baseline, v1, 36 features | 0.7952 | 0.7862 |

With working intervals, the Lead I tabular models come within 0.009 of the 3-lead tabular baseline.

## C++ port

`ExtractLead1Features` now outputs `pr_ms`, `qrs_ms`, `qt_ms`, `qtc_ms` (NaN when not measurable) and
`n_beats`. The old envelope width `mean_qrs_ms` is removed. Parity with the Python reference:

| Split | Windows | All 18 features within 1e-3 | NaN agreement |
|---|---:|---:|---:|
| test | 2,158 | 2,158 | 100 % |
| train | 19,230 | 19,229 | 100 % |

The one miss is the known float32 near-tie in window 10449 (one R peak one sample off). Its
intervals still match.

Extra workspace for delineation is 596 B: the 101-sample template, 32 refined R positions and 32
median scratch values. The 0.5–40 Hz signal reuses the existing 1,030-float buffer.
