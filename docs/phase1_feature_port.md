# Phase 1: Lead I feature extraction in C++

## 1. What exists

| Path | Purpose |
|---|---|
| `firmware/ecg_features/ecg_features.{h,cpp}` | Portable C++14 extractor, no heap. Builds for the RP2040 and for the host. |
| `firmware/host_test/main.cpp` | Host harness: binary file of int16 windows in, CSV of features out. |
| `scripts/reference_features.py` | Python specification (v2): rhythm, HRV, Hjorth and Welch blocks from `scripts/fix_cache.py`, intervals from `scripts/delineation.py`. |
| `scripts/compare_features.py` | Exports PTB-XL Lead I as int16, runs the harness, compares with the reference and with the cached training features (`X_feat_v2_*`). |

Memory layout and the reasons behind it are in `docs/sram_budget.md`. The delineation method and its
validation are in `docs/delineation.md`.

## 2. Ported features (18)

| C++ field | `X_feat_v2` column | Method |
|---|---|---|
| `mean_hr`, `min_hr`, `max_hr`, `hr_std` | 0–3 | from RR intervals |
| `mean_rr`, `rr_std`, `rr_range`, `rr_cv` | 4–7 | |
| `sdnn`, `rmssd`, `pnn50` | 8–10 | |
| `hjorth_complexity` | 11 ("Sample_Entropy") | Hjorth complexity of raw Lead I |
| `pr_ms`, `qrs_ms`, `qt_ms`, `qtc_ms` | 12–15 | median-beat delineation, NaN if not measurable |
| `snr_db`, `baseline_wander` | 16, 17 | Welch PSD, power below 0.5 Hz |

Pipeline, each step matching scipy's behaviour (padding, filter state, plateau handling, distance
then prominence order, Welch scaling):
`filtfilt(5–15 Hz)` → `diff²` → 80 ms moving sum → `find_peaks(distance=350 ms,
prominence=0.5·std)` → HRV → `filtfilt(0.5–40 Hz)` → median beat → delineation → Hjorth →
`welch(nperseg=256)`.

**Not ported:** the Lead I statistics (columns 18–23: mean, std, skew, kurtosis, peak-to-peak, IQR).
Only the tabular models use them, and those don't run on the device.

## 3. Build and test (host)

```powershell
New-Item -ItemType Directory -Force build | Out-Null
g++ -std=c++14 -O2 -Wall -Wextra -Wdouble-promotion -Wfloat-conversion `
    -o build/lead1_host.exe firmware/host_test/main.cpp firmware/ecg_features/ecg_features.cpp
python scripts/build_lead1_features.py      # once, adds X_feat_v2_* to the cache
python scripts/compare_features.py --exe build/lead1_host.exe --split test
python scripts/compare_features.py --exe build/lead1_host.exe --split train
```

Add `-DECG_USE_DOUBLE` for a double-precision build (debugging only; on the RP2040 doubles are much
slower than floats).

`-Wdouble-promotion` is there to catch accidental double arithmetic, which would be slow on the M0+.

## 4. Parity results (float32 build, MinGW g++ 6.3, scipy 1.16.3)

| Split | Windows | Same R-peak count | All 18 features within 1e-3 (abs or rel), NaN in the same places |
|---|---:|---:|---:|
| test (fold 10) | 2,158 | 100 % | 2,158 |
| train (folds 1–9) | 19,230 | 100 % | 19,229 |

In the one failing window (train index 10449), one R peak is shifted by one sample (10 ms), which moves
the RR-based features by about 0.1 %. The double-precision build matches Python on that window
exactly, so this is float32 rounding at a near-tie, not a logic difference.

The reference matches the cached `X_feat_v2_*` training features to within 3e-5.

## 5. Issues found while porting, and their status

| Issue | Status |
|---|---|
| 5.1 The CNN was trained on Leads I, II, III, but the hardware records Lead I only | **Resolved:** all models retrained on Lead I (`docs/lead1_retrain.md`). Deployed INT8 model: macro AUC 0.8294. |
| 5.2 `lead-1-just.py` plotted the wrong columns ("Mean_QRS" was RR_Range) | **Resolved:** `scripts/lead1_justification_plot.py` selects columns by name. The old figures are kept as `results/features/*_MISLABELED.png`. |
| 5.3 PR / QT measured the R wave (medians 10 ms / 0.8 ms) | **Resolved:** median-beat delineation. PR r = 0.67 and QT r = 0.52 vs GE 12SL. |
| 5.4 `Mean_QRS` was an energy-envelope width (~96 ms for every class) | **Resolved:** QRS onset to offset, MAE 13.9 ms vs 12SL. Separates CD as well as 12-lead 12SL (AUC 0.754 vs 0.755). |

The original findings are in git history (commit `a825b85`).

## 6. Next Phase 1 steps

1. Orchestrator: signal-quality gate → rule-based triage → CNN (`docs/orchestrator.md`).
2. Port to a Pico SDK project and measure latency per stage on the RAK11300.
