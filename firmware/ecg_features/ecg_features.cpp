// Lead I feature extraction. Each stage mirrors one scipy/numpy call from
// fix_cache.py::extract_ecg_features_corrected; the matching Python is quoted
// above each block so the two can be diffed by eye.
#include "ecg_features.h"

#include <cmath>

namespace ecg {
namespace {

// ---------------------------------------------------------------------------
// Constants generated with scipy 1.16:
//   b, a = scipy.signal.butter(2, [5 / 50, 15 / 50], btype="bandpass")
//   zi   = scipy.signal.lfilter_zi(b, a)
// ---------------------------------------------------------------------------
#define ECG_R(v) static_cast<real_t>(v)  // intentional narrowing in float builds
const real_t kB[5] = {ECG_R(0.06745527388907191), ECG_R(0.0),
                      ECG_R(-0.13491054777814382), ECG_R(0.0),
                      ECG_R(0.06745527388907191)};
const real_t kA[5] = {ECG_R(1.0), ECG_R(-2.673578905120267),
                      ECG_R(2.9923618041278957), ECG_R(-1.6745773146352607),
                      ECG_R(0.41280159809618855)};
const real_t kZi[4] = {ECG_R(-0.06745527388907088), ECG_R(-0.06745527388907364),
                       ECG_R(0.06745527388907323), ECG_R(0.0674552738890715)};
#undef ECG_R

constexpr int kPeakDistance = 35;  // int(0.35 * fs)
constexpr int kIntegWin = 8;       // int(0.08 * fs)

// cos(2*pi*t/256) for t = 0..128. Used for the FFT twiddles and the periodic
// Hann window, so no trig runs on the device. 516 B of flash.
const float kCos256[129] = {
    1.000000000e+00f,  9.996988187e-01f,  9.987954562e-01f,  9.972904567e-01f,
    9.951847267e-01f,  9.924795346e-01f,  9.891765100e-01f,  9.852776424e-01f,
    9.807852804e-01f,  9.757021300e-01f,  9.700312532e-01f,  9.637760658e-01f,
    9.569403357e-01f,  9.495281806e-01f,  9.415440652e-01f,  9.329927988e-01f,
    9.238795325e-01f,  9.142097557e-01f,  9.039892931e-01f,  8.932243012e-01f,
    8.819212643e-01f,  8.700869911e-01f,  8.577286100e-01f,  8.448535652e-01f,
    8.314696123e-01f,  8.175848132e-01f,  8.032075315e-01f,  7.883464276e-01f,
    7.730104534e-01f,  7.572088465e-01f,  7.409511254e-01f,  7.242470830e-01f,
    7.071067812e-01f,  6.895405447e-01f,  6.715589548e-01f,  6.531728430e-01f,
    6.343932842e-01f,  6.152315906e-01f,  5.956993045e-01f,  5.758081914e-01f,
    5.555702330e-01f,  5.349976199e-01f,  5.141027442e-01f,  4.928981922e-01f,
    4.713967368e-01f,  4.496113297e-01f,  4.275550934e-01f,  4.052413140e-01f,
    3.826834324e-01f,  3.598950365e-01f,  3.368898534e-01f,  3.136817404e-01f,
    2.902846773e-01f,  2.667127575e-01f,  2.429801799e-01f,  2.191012402e-01f,
    1.950903220e-01f,  1.709618888e-01f,  1.467304745e-01f,  1.224106752e-01f,
    9.801714033e-02f,  7.356456360e-02f,  4.906767433e-02f,  2.454122852e-02f,
    0.000000000e+00f,  -2.454122852e-02f, -4.906767433e-02f, -7.356456360e-02f,
    -9.801714033e-02f, -1.224106752e-01f, -1.467304745e-01f, -1.709618888e-01f,
    -1.950903220e-01f, -2.191012402e-01f, -2.429801799e-01f, -2.667127575e-01f,
    -2.902846773e-01f, -3.136817404e-01f, -3.368898534e-01f, -3.598950365e-01f,
    -3.826834324e-01f, -4.052413140e-01f, -4.275550934e-01f, -4.496113297e-01f,
    -4.713967368e-01f, -4.928981922e-01f, -5.141027442e-01f, -5.349976199e-01f,
    -5.555702330e-01f, -5.758081914e-01f, -5.956993045e-01f, -6.152315906e-01f,
    -6.343932842e-01f, -6.531728430e-01f, -6.715589548e-01f, -6.895405447e-01f,
    -7.071067812e-01f, -7.242470830e-01f, -7.409511254e-01f, -7.572088465e-01f,
    -7.730104534e-01f, -7.883464276e-01f, -8.032075315e-01f, -8.175848132e-01f,
    -8.314696123e-01f, -8.448535652e-01f, -8.577286100e-01f, -8.700869911e-01f,
    -8.819212643e-01f, -8.932243012e-01f, -9.039892931e-01f, -9.142097557e-01f,
    -9.238795325e-01f, -9.329927988e-01f, -9.415440652e-01f, -9.495281806e-01f,
    -9.569403357e-01f, -9.637760658e-01f, -9.700312532e-01f, -9.757021300e-01f,
    -9.807852804e-01f, -9.852776424e-01f, -9.891765100e-01f, -9.924795346e-01f,
    -9.951847267e-01f, -9.972904567e-01f, -9.987954562e-01f, -9.996988187e-01f,
    -1.000000000e+00f};

inline real_t Cos256(int t) {  // t in [0, 256]
  return t <= 128 ? kCos256[t] : kCos256[256 - t];
}
inline real_t Sin256(int t) {  // t in [0, 128)
  const int d = 64 - t;
  return kCos256[d < 0 ? -d : d];
}

inline real_t Sqrt(real_t v) { return std::sqrt(v); }

// One lfilter() pass (direct form II transposed, the structure scipy uses),
// in place, starting with state zi * first_sample. dir = +1 or -1.
void FilterPass(real_t* x, int n, int dir) {
  int i = dir > 0 ? 0 : n - 1;
  const real_t first = x[i];
  real_t z0 = kZi[0] * first, z1 = kZi[1] * first;
  real_t z2 = kZi[2] * first, z3 = kZi[3] * first;
  for (int k = 0; k < n; ++k, i += dir) {
    const real_t xi = x[i];
    const real_t y = kB[0] * xi + z0;
    z0 = kB[1] * xi - kA[1] * y + z1;
    z1 = kB[2] * xi - kA[2] * y + z2;
    z2 = kB[3] * xi - kA[3] * y + z3;
    z3 = kB[4] * xi - kA[4] * y;
    x[i] = y;
  }
}

// In-place radix-2 complex FFT, N = 256, interleaved re/im.
void Fft256(real_t* d) {
  constexpr int N = 256;
  for (int i = 1, j = 0; i < N; ++i) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      real_t t = d[2 * i]; d[2 * i] = d[2 * j]; d[2 * j] = t;
      t = d[2 * i + 1]; d[2 * i + 1] = d[2 * j + 1]; d[2 * j + 1] = t;
    }
  }
  for (int len = 2; len <= N; len <<= 1) {
    const int half = len >> 1;
    const int step = N / len;
    for (int i = 0; i < N; i += len) {
      for (int k = 0; k < half; ++k) {
        const int t = k * step;
        const real_t wr = Cos256(t), wi = -Sin256(t);
        const int a = 2 * (i + k), b = 2 * (i + k + half);
        const real_t xr = d[b] * wr - d[b + 1] * wi;
        const real_t xim = d[b] * wi + d[b + 1] * wr;
        d[b] = d[a] - xr;
        d[b + 1] = d[a + 1] - xim;
        d[a] += xr;
        d[a + 1] += xim;
      }
    }
  }
}

// Population variance from integer sums s1 = sum(v), s2 = sum(v^2). The
// numerator n*s2 - s1^2 is exact in int64 for 16-bit samples and n <= 1000.
real_t IntVariance(int64_t s1, int64_t s2, int64_t n) {
  return static_cast<real_t>(n * s2 - s1 * s1) / static_cast<real_t>(n * n);
}

constexpr uint16_t kPosMask = 0x3FFF;
constexpr uint16_t kProcessed = 0x4000;
constexpr uint16_t kRemoved = 0x8000;

}  // namespace

int ExtractLead1Features(const int16_t* adc, int n, real_t mv_per_lsb,
                         Lead1Workspace* ws, Lead1Features* out) {
  if (adc == nullptr || ws == nullptr || out == nullptr || n != kWindowLen ||
      !(mv_per_lsb > 0)) {
    return kBadArgs;
  }
  real_t* const buf = ws->buf;

  // ---- 1. Band-pass -------------------------------------------------------
  // b, a = butter(2, [5/(fs/2), 15/(fs/2)], 'bandpass')
  // filtered = filtfilt(b, a, lead1)        # padtype='odd', padlen=15
  const real_t x0 = adc[0] * mv_per_lsb;
  const real_t xl = adc[n - 1] * mv_per_lsb;
  for (int k = 0; k < kFiltPad; ++k) {
    buf[k] = 2 * x0 - adc[kFiltPad - k] * mv_per_lsb;
    buf[kFiltPad + n + k] = 2 * xl - adc[n - 2 - k] * mv_per_lsb;
  }
  for (int i = 0; i < n; ++i) buf[kFiltPad + i] = adc[i] * mv_per_lsb;
  const int ext_len = n + 2 * kFiltPad;
  FilterPass(buf, ext_len, +1);
  FilterPass(buf, ext_len, -1);

  // ---- 2. Squared derivative + moving-window integration ------------------
  // integrated = np.convolve(np.diff(filtered)**2, np.ones(8), mode='same')
  // 'same' with an 8-tap kernel: out[i] = sum(d[i-4 .. i+3]).
  const int m = n - 1;
  for (int i = 0; i < m; ++i) {
    const real_t d = buf[kFiltPad + i + 1] - buf[kFiltPad + i];
    buf[i] = d * d;  // writes trail the reads, so in place is safe
  }
  {
    constexpr int kBack = kIntegWin / 2;      // 4 samples before i
    constexpr int kFwd = kIntegWin - kBack - 1;  // 3 samples after i
    real_t hist[kBack] = {};  // original d[i-4 .. i-1]
    for (int i = 0; i < m; ++i) {
      const real_t cur = buf[i];
      real_t s = cur;
      for (int k = 0; k < kBack; ++k) s += hist[k];
      for (int k = 1; k <= kFwd && i + k < m; ++k) s += buf[i + k];
      for (int k = 0; k < kBack - 1; ++k) hist[k] = hist[k + 1];
      hist[kBack - 1] = cur;
      buf[i] = s;
    }
  }
  const real_t* const x = buf;  // integrated signal, length m

  // ---- 3. R-peak detection ------------------------------------------------
  // peaks, _ = find_peaks(integrated, distance=35,
  //                       prominence=np.std(integrated) * 0.5)
  real_t mean = 0;
  for (int i = 0; i < m; ++i) mean += x[i];
  mean /= m;
  real_t var = 0;
  for (int i = 0; i < m; ++i) var += (x[i] - mean) * (x[i] - mean);
  const real_t min_prominence = Sqrt(var / m) * real_t(0.5);

  // 3a. Local maxima, plateau-aware (scipy _local_maxima_1d).
  uint16_t* const cand = ws->candidates;
  int n_cand = 0;
  for (int i = 1; i < m - 1; ++i) {
    if (x[i - 1] < x[i]) {
      int ahead = i + 1;
      while (ahead < m - 1 && x[ahead] == x[i]) ++ahead;
      if (x[ahead] < x[i]) {
        cand[n_cand++] = static_cast<uint16_t>((i + ahead - 1) / 2);
        i = ahead;
      }
    }
  }

  // 3b. Distance rule (scipy _select_by_peak_distance): visit candidates from
  // tallest to shortest, discard neighbours closer than 35 samples.
  // Selection by repeated max search (O(K^2), K <= 500) avoids a 1 KB sort index.
  for (;;) {
    int best = -1;
    real_t best_h = 0;
    for (int j = 0; j < n_cand; ++j) {
      if (cand[j] & (kProcessed | kRemoved)) continue;
      const real_t h = x[cand[j] & kPosMask];
      if (best < 0 || h >= best_h) { best = j; best_h = h; }
    }
    if (best < 0) break;
    cand[best] |= kProcessed;
    const int pb = cand[best] & kPosMask;
    for (int k = best - 1; k >= 0 && pb - (cand[k] & kPosMask) < kPeakDistance; --k)
      cand[k] |= kRemoved;
    for (int k = best + 1; k < n_cand && (cand[k] & kPosMask) - pb < kPeakDistance; ++k)
      cand[k] |= kRemoved;
  }

  // 3c. Prominence rule, then 3d. half-prominence width for survivors
  // (scipy _peak_prominences with wlen=None, then _peak_widths(rel_height=.5)).
  uint16_t* const peaks = ws->peaks;
  int n_peaks = 0;
  real_t width_sum = 0;
  for (int j = 0; j < n_cand && n_peaks < kMaxPeaks; ++j) {
    if (cand[j] & kRemoved) continue;
    const int p = cand[j] & kPosMask;
    const real_t xp = x[p];
    int i = p, left_base = p, right_base = p;
    real_t left_min = xp, right_min = xp;
    while (i >= 0 && x[i] <= xp) {
      if (x[i] < left_min) { left_min = x[i]; left_base = i; }
      --i;
    }
    i = p;
    while (i <= m - 1 && x[i] <= xp) {
      if (x[i] < right_min) { right_min = x[i]; right_base = i; }
      ++i;
    }
    const real_t prominence = xp - (left_min > right_min ? left_min : right_min);
    if (!(prominence >= min_prominence)) continue;

    const real_t height = xp - prominence * real_t(0.5);
    i = p;
    while (left_base < i && height < x[i]) --i;
    real_t left_ip = static_cast<real_t>(i);
    if (x[i] < height) left_ip += (height - x[i]) / (x[i + 1] - x[i]);
    i = p;
    while (i < right_base && height < x[i]) ++i;
    real_t right_ip = static_cast<real_t>(i);
    if (x[i] < height) right_ip -= (height - x[i]) / (x[i - 1] - x[i]);

    width_sum += right_ip - left_ip;
    peaks[n_peaks++] = static_cast<uint16_t>(p);
  }
  out->n_peaks = static_cast<uint8_t>(n_peaks);

  // ---- 4. HRV -------------------------------------------------------------
  // rr = diff(peaks) / fs * 1000 if len(peaks) >= 2 else [800.0]
  const int n_rr = n_peaks >= 2 ? n_peaks - 1 : 1;
  auto rr_at = [&](int k) -> real_t {
    return n_peaks >= 2
               ? static_cast<real_t>(peaks[k + 1] - peaks[k]) * (real_t(1000) / kFs)
               : real_t(800);
  };
  real_t rr_sum = 0, hr_sum = 0;
  real_t rr_min = rr_at(0), rr_max = rr_min;
  for (int k = 0; k < n_rr; ++k) {
    const real_t rr = rr_at(k);
    rr_sum += rr;
    hr_sum += real_t(60000) / rr;
    if (rr < rr_min) rr_min = rr;
    if (rr > rr_max) rr_max = rr;
  }
  const real_t mean_rr = rr_sum / n_rr;
  const real_t mean_hr = hr_sum / n_rr;
  real_t rr_var = 0, hr_var = 0;
  for (int k = 0; k < n_rr; ++k) {
    const real_t rr = rr_at(k);
    const real_t hr = real_t(60000) / rr;
    rr_var += (rr - mean_rr) * (rr - mean_rr);
    hr_var += (hr - mean_hr) * (hr - mean_hr);
  }
  const real_t rr_std = Sqrt(rr_var / n_rr);
  out->mean_hr = mean_hr;
  out->min_hr = real_t(60000) / rr_max;
  out->max_hr = real_t(60000) / rr_min;
  out->hr_std = Sqrt(hr_var / n_rr);
  out->mean_rr = mean_rr;
  out->rr_std = rr_std;
  out->rr_range = rr_max - rr_min;
  out->rr_cv = mean_rr > 0 ? rr_std / mean_rr : real_t(0);
  out->sdnn = rr_std;

  // rmssd = sqrt(mean(diff(rr)**2)); pnn50 = 100 * mean(|diff(rr)| > 50)
  const int n_diff = n_rr - 1;
  if (n_diff > 0) {
    real_t sq = 0;
    int over50 = 0;
    for (int k = 0; k < n_diff; ++k) {
      const real_t d = rr_at(k + 1) - rr_at(k);
      sq += d * d;
      if (d > 50 || d < -50) ++over50;
    }
    out->rmssd = Sqrt(sq / n_diff);
    out->pnn50 = real_t(100) * over50 / n_diff;
  } else {
    out->rmssd = 0;
    out->pnn50 = 0;
  }

  // ---- 5. QRS width -------------------------------------------------------
  // mean_qrs = mean(peak_widths(integrated, peaks, 0.5)[0]) / fs * 1000
  out->mean_qrs_ms =
      n_peaks > 0 ? width_sum / n_peaks * (real_t(1000) / kFs) : real_t(90);

  // ---- 6. Hjorth complexity on raw Lead I ---------------------------------
  // Scale-invariant, so computed exactly on integer ADC counts.
  {
    int64_t s1 = 0, s2 = 0, d1 = 0, d2 = 0, dd1 = 0, dd2 = 0;
    for (int i = 0; i < n; ++i) {
      const int64_t v = adc[i];
      s1 += v; s2 += v * v;
      if (i >= 1) {
        const int64_t d = v - adc[i - 1];
        d1 += d; d2 += d * d;
        if (i >= 2) {
          const int64_t dd = d - (adc[i - 1] - adc[i - 2]);
          dd1 += dd; dd2 += dd * dd;
        }
      }
    }
    const real_t var_y = IntVariance(s1, s2, n);
    const real_t var_dy = IntVariance(d1, d2, n - 1);
    const real_t var_ddy = IntVariance(dd1, dd2, n - 2);
    const real_t mobility = var_y > 0 ? Sqrt(var_dy / var_y) : real_t(0);
    out->hjorth_complexity = (var_dy > 0 && mobility > 0)
                                 ? Sqrt(var_ddy / var_dy) / mobility
                                 : real_t(0);
  }

  // ---- 7. Signal quality from Welch PSD -----------------------------------
  // freqs, psd = welch(lead1, fs=100, nperseg=256)   # hann, 50% overlap,
  // bw = sum(psd[freqs < 0.5]); total = sum(psd) + 1e-8      # constant detrend
  // snr = 10*log10((total - bw) / (bw + 1e-8)); baseline_wander = bw / total
  // The filter/integration buffer is no longer needed: reuse it as
  // FFT scratch [0, 512) and PSD accumulator [512, 641).
  {
    constexpr int kBins = kWelchSeg / 2 + 1;
    real_t* const fft = buf;
    real_t* const acc = buf + 2 * kWelchSeg;
    for (int k = 0; k < kBins; ++k) acc[k] = 0;
    const int n_seg = (n - (kWelchSeg - kWelchStep)) / kWelchStep;
    for (int s = 0; s < n_seg; ++s) {
      const int16_t* seg = adc + s * kWelchStep;
      int32_t sum = 0;
      for (int k = 0; k < kWelchSeg; ++k) sum += seg[k];
      const real_t seg_mean = static_cast<real_t>(sum) / kWelchSeg;
      for (int k = 0; k < kWelchSeg; ++k) {
        const real_t hann = real_t(0.5) - real_t(0.5) * Cos256(k);
        fft[2 * k] = (seg[k] - seg_mean) * mv_per_lsb * hann;
        fft[2 * k + 1] = 0;
      }
      Fft256(fft);
      for (int k = 0; k < kBins; ++k)
        acc[k] += fft[2 * k] * fft[2 * k] + fft[2 * k + 1] * fft[2 * k + 1];
    }
    // density scaling 1/(fs * sum(hann^2)); sum(hann^2) = 3N/8 = 96.
    const real_t scale = real_t(1) / (real_t(kFs) * (3 * kWelchSeg / 8) * n_seg);
    real_t total = 0, bw = 0;
    for (int k = 0; k < kBins; ++k) {
      real_t p = acc[k] * scale;
      if (k != 0 && k != kBins - 1) p *= 2;  // one-sided spectrum
      total += p;
      if (2 * k * kFs < kWelchSeg) bw += p;  // k * fs / nperseg < 0.5 Hz
    }
    total += real_t(1e-8);
    out->snr_db = real_t(10) * std::log10((total - bw) / (bw + real_t(1e-8)));
    out->baseline_wander = bw / total;
  }
  return kOk;
}

void FeaturesToArray(const Lead1Features& f, real_t* dst) {
  const real_t v[kNumFeatures] = {
      f.mean_hr, f.min_hr, f.max_hr, f.hr_std, f.mean_rr,
      f.rr_std,  f.rr_range, f.rr_cv, f.sdnn, f.rmssd,
      f.pnn50,   f.hjorth_complexity, f.mean_qrs_ms, f.snr_db,
      f.baseline_wander};
  for (int i = 0; i < kNumFeatures; ++i) dst[i] = v[i];
}

}  // namespace ecg
