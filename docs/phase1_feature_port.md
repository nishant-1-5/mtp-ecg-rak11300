# Phase 1: Lead I feature extraction in C++

## 1. What exists

| Path | Purpose |
|---|---|
| `firmware/ecg_features/ecg_features.{h,cpp}` | Portable C++14 extractor, no heap. Builds for the RP2040 and for the host. |
| `firmware/host_test/main.cpp` | Host harness: binary file of int16 windows in, CSV of features out. |
| `scripts/reference_features.py` | Python specification: `scripts/fix_cache.py::extract_ecg_features_corrected` restricted to Lead I. |
| `scripts/compare_features.py` | Exports PTB-XL Lead I as int16, runs the harness, compares with the reference. |

Memory layout and the reasons behind it are in `docs/sram_budget.md`.

## 2. Ported features

| C++ field | Python cache column | Method |
|---|---|---|
| `mean_hr`, `min_hr`, `max_hr`, `hr_std` | 0–3 | from RR intervals |
| `mean_rr`, `rr_std`, `rr_range`, `rr_cv` | 4–7 | |
| `sdnn`, `rmssd`, `pnn50` | 8–10 | |
| `hjorth_complexity` | 11 ("Sample_Entropy") | Hjorth complexity of raw Lead I |
| `mean_qrs_ms` | 13 | half-prominence width of the integrated peak |
| `snr_db`, `baseline_wander` | 16, 17 | Welch PSD, power below 0.5 Hz |

Pipeline, each step matching scipy's behaviour (padding, filter state, plateau handling, distance
then prominence order, width interpolation, Welch scaling):
`filtfilt(butter(2, 5–15 Hz))` → `diff²` → 80 ms moving sum → `find_peaks(distance=350 ms,
prominence=0.5·std)` → `peak_widths(rel_height=0.5)` → HRV → Hjorth → `welch(nperseg=256)`.

**Not ported:** `Mean_PR`, `Mean_QT`, `Mean_QTc` (columns 12, 14, 15) because the Python versions
are broken (§5.3), and the Lead II/III statistics (columns 18–35) because the hardware has no
Lead II/III.

## 3. Build and test (host)

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++14 -O2 -Wall -Wextra -Wdouble-promotion -Wfloat-conversion `
    -o build/lead1_host.exe firmware/host_test/main.cpp firmware/ecg_features/ecg_features.cpp
python scripts/compare_features.py --exe build/lead1_host.exe --split test
python scripts/compare_features.py --exe build/lead1_host.exe --split train
```

Add `-DECG_USE_DOUBLE` for a double-precision build (debugging only; on the RP2040 doubles are much
slower than floats).

`-Wdouble-promotion` is there to catch accidental double arithmetic, which would be slow on the M0+.

## 4. Parity results (float32 build, MinGW g++ 6.3, scipy 1.16.3)

| Split | Windows | Same R-peak count | All 15 features within 1e-3 (abs or rel) |
|---|---:|---:|---:|
| test (fold 10) | 2,158 | 100 % | 2,158 |
| train (folds 1–9) | 19,230 | 100 % | 19,229 |

The one failing window (train index 10449) has one R peak shifted by one sample (10 ms), which moves
the RR-based features by about 0.1 %. The double-precision build matches Python on that window exactly,
so this is float32 rounding at a near-tie, not a logic difference.

The reference was also checked against the cached `X_feat_*` arrays that the models were trained
on: max difference 5.8e-5, i.e. the C++ port reproduces the training features.

## 5. Issues found while porting (need a team decision)

### 5.1 The CNN is trained on 3 leads; the hardware records 1

`cnn.ipynb` (`LEADS = [0, 1, 2]`), `pre-process.ipynb` (`LEAD_INDICES = [0, 1, 2]`),
`tinyml.ipynb` (`input_shape=(1000, 3)`) and `quantization.ipynb` (same model) all use PTB-XL
Leads I, II and III. The RAK11300 front end has RA, LA and RL electrodes, which gives Lead I only
(slide 12 says the same). `model_data.h` therefore cannot be fed by this hardware. Feeding zeros or
copies of Lead I into channels 2–3 would put the model outside its training distribution.

Proposed fix: retrain the same architecture with `input_shape=(1000, 1)` on Lead I, then
re-quantise and re-measure the arena. Expect lower AUC for MI/HYP/STTC than 0.8626, and report the
new number.

### 5.2 `lead-1-just.py` plots the wrong columns

It takes `X_feat_train[:, :8]` and labels the columns `Mean_HR, HR_Std, SDNN, RMSSD,
Signal_Complexity, Mean_PR, Mean_QRS, Mean_QTc`. The real columns 0–7 are `Mean_HR, Min_HR, Max_HR,
HR_Std, Mean_RR, RR_Std, RR_Range, RR_CV`. So the "Mean_QRS" box plot in
`lead_1_justification_ppt.png` actually shows `RR_Range`.

With the correct column (13), on the training split:

- correlation of `Mean_QRS` with CD is 0.05;
- median `Mean_QRS` is 94.7 ms for CD and 96.6 ms for NORM.

The README claim that "Mean_QRS > 120 ms isolates CD" is not supported by the current extractor.
(`correlation.ipynb` uses the correct names; its heatmap is fine.)

### 5.3 PR and QT extraction do not work

On the training split, `Mean_PR` has median 10 ms and `Mean_QT` has median 0.8 ms (25th percentile
0 ms). The QT search window starts at the detected peak, so `argmax` usually returns the R wave
itself. For PR, the tallest sample in the 250 ms before the peak is usually the one immediately
before it (10 ms = 1 sample), i.e. the R-wave upstroke rather than the P wave. QTc inherits the
QT problem.

### 5.4 `Mean_QRS` is an energy-envelope width, not QRS duration

It is the half-height width of the 80 ms moving-sum envelope of the squared derivative, so the
integration window smears every beat and the values cluster around 96 ms for all classes. A
delineator working on the band-passed signal (QRS onset/offset from slope thresholds) would be
needed for a clinically meaningful QRS duration.

## 6. Next Phase 1 steps

1. Decide on §5.1 (retrain on Lead I) and on the triage rules, given §5.2–5.4.
2. Fix PR/QT/QRS delineation in Python first, re-validate against labels, then port with the same
   parity test.
3. Write the orchestrator (signal-quality gate → rule-based triage → CNN) once the rules are agreed.
4. Port the extractor into a Pico SDK project and measure latency per stage on the RAK11300.
