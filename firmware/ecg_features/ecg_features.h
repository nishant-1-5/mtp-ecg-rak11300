// Lead I numerical feature extraction for the RAK11300 (RP2040, Cortex-M0+).
//
// C++ port of `extract_ecg_features_corrected()` in fix_cache.py, restricted to
// the features that only need Lead I (the RA/LA/RL front end cannot record
// Leads II/III). Python reference: tools/reference_features.py.
//
// Design rules (see docs/sram_budget.md):
//   * No heap, no exceptions, no STL containers. All scratch memory is a
//     caller-owned `Lead1Workspace` (5,184 B with float) that can live in .bss.
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

// Feature vector. Field order follows columns 0-11, 13, 16, 17 of the Python
// feature vector; Mean_PR/Mean_QT/Mean_QTc (12, 14, 15) are intentionally not
// ported yet (see docs/phase1_feature_port.md).
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
  real_t mean_qrs_ms;        // half-prominence width of the integrated peak
  real_t snr_db;             // Welch power outside vs inside < 0.5 Hz
  real_t baseline_wander;    // fraction of Welch power < 0.5 Hz
  uint8_t n_peaks;           // detected R peaks (0 => HRV values are defaults)
};

constexpr int kNumFeatures = 15;

// Caller-owned scratch memory. Reused for every window; contents are
// meaningless between calls. Keep one static instance.
struct Lead1Workspace {
  // filtfilt buffer -> squared derivative -> integrated signal -> FFT scratch.
  real_t buf[kWindowLen + 2 * kFiltPad];
  // Local-maximum positions; bits 14/15 hold the distance-filter state.
  uint16_t candidates[kMaxCandidates];
  uint16_t peaks[kMaxPeaks];
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
