// Lead I numerical feature extraction for the RAK11300 (RP2040, Cortex-M0+).
//
// C++ port of scripts/reference_features.py::extract_lead1_features (v2):
// rhythm/HRV/Hjorth/Welch blocks from fix_cache.py, PR/QRS/QT/QTc from the
// median-beat delineation in scripts/delineation.py. Lead I only, because the
// RA/LA/RL front end cannot record Leads II/III.
//
// Design rules (see docs/sram_budget.md):
//   * No heap, no exceptions, no STL containers. All scratch memory is a
//     caller-owned `Lead1Workspace` (5,780 B with float) that can live in .bss.
//   * Input is the raw int16 ADC window (2 B/sample) plus a mV-per-LSB scale,
//     so the acquisition buffer never has to be widened to float.
//   * Portable C++14: builds with the Pico SDK toolchain and with a host
//     compiler for parity testing (firmware/host_test).
#ifndef ECG_FEATURES_H_
#define ECG_FEATURES_H_

#include <stdint.h>

namespace ecg {

// Build with -DECG_USE_DOUBLE to run the whole pipeline in double precision
// (host parity debugging only; doubles are software-emulated on the RP2040).
#ifdef ECG_USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

constexpr int kFs = 100;           // PTB-XL records100 sampling rate (Hz)
constexpr int kWindowLen = 1000;   // 10 s analysis window
constexpr int kFiltPad = 15;       // scipy filtfilt padlen = 3 * max(len(a), len(b))
constexpr int kIntegLen = kWindowLen - 1;  // length of np.diff(filtered)
// Upper bound on strict local maxima in kIntegLen samples (they are >= 2 apart).
constexpr int kMaxCandidates = (kIntegLen - 1) / 2 + 1;
// Peaks that survive the 350 ms distance rule: floor((998 - 1) / 35) + 1 = 29.
constexpr int kMaxPeaks = 32;
constexpr int kWelchSeg = 256;     // scipy.signal.welch nperseg
constexpr int kWelchStep = 128;    // nperseg - noverlap
constexpr int kTemplatePre = 30;   // median beat: 300 ms before R ...
constexpr int kTemplatePost = 70;  // ... to 700 ms after R
constexpr int kTemplateLen = kTemplatePre + kTemplatePost + 1;

// Feature vector, same order as reference_features.FEATURE_NAMES and as
// columns 0-17 of the X_feat_v2_* training table.
struct Lead1Features {
  real_t mean_hr;            // bpm
  real_t min_hr;             // bpm
  real_t max_hr;             // bpm
  real_t hr_std;             // bpm
  real_t mean_rr;            // ms
  real_t rr_std;             // ms
  real_t rr_range;           // ms
  real_t rr_cv;              // unitless
  real_t sdnn;               // ms
  real_t rmssd;              // ms
  real_t pnn50;              // %
  real_t hjorth_complexity;  // "Sample_Entropy" column in the Python code
  real_t pr_ms;              // P onset -> QRS onset; NaN if no P wave found
  real_t qrs_ms;             // QRS onset -> offset; NaN if < 3 usable beats
  real_t qt_ms;              // QRS onset -> T end (tangent method); NaN if not found
  real_t qtc_ms;             // Bazett, using mean_rr
  real_t snr_db;             // Welch power outside vs inside < 0.5 Hz
  real_t baseline_wander;    // fraction of Welch power < 0.5 Hz
  // Not part of the model feature vector:
  real_t template_corr;      // mean beat-vs-median-beat correlation (signal quality); NaN if < 3 beats
  uint8_t n_peaks;           // detected R peaks (0 => HRV values are defaults)
  uint8_t n_beats;           // beats in the median template (< 3 => no intervals)
};

constexpr int kNumFeatures = 18;

// Caller-owned scratch memory. Reused for every window; contents are
// meaningless between calls. Keep one static instance.
struct Lead1Workspace {
  // 5-15 Hz filtfilt -> squared derivative -> integrated signal ->
  // 0.5-40 Hz filtfilt (delineation) -> FFT scratch + PSD accumulator.
  real_t buf[kWindowLen + 2 * kFiltPad];
  // Local-maximum positions; bits 14/15 hold the distance-filter state.
  uint16_t candidates[kMaxCandidates];
  uint16_t peaks[kMaxPeaks];   // Pan-Tompkins peaks (integrated-signal index)
  uint16_t r_pos[kMaxPeaks];   // refined R positions used in the template
  real_t templ[kTemplateLen];  // median beat
  real_t scratch[kMaxPeaks];   // per-sample values across beats (median)
};

enum Status : int {
  kOk = 0,
  kBadArgs = -1,
};

// Extracts Lead I features from one 10 s window.
//   adc        : kWindowLen raw samples, Lead I polarity (LA - RA)
//   mv_per_lsb : ADC scale; PTB-XL stores 1000 LSB/mV, i.e. 0.001
int ExtractLead1Features(const int16_t* adc, int n, real_t mv_per_lsb,
                         Lead1Workspace* ws, Lead1Features* out);

// Flattens `f` into `dst[kNumFeatures]` in struct order.
void FeaturesToArray(const Lead1Features& f, real_t* dst);

}  // namespace ecg

#endif  // ECG_FEATURES_H_
