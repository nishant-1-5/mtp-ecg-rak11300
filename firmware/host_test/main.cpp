// Host harness: runs the firmware feature extractor over a file of Lead I
// windows and writes one CSV row per window, so the output can be compared
// with the Python reference (tools/compare_features.py).
//
// Input file (little-endian): int32 n_records, int32 n_samples,
//                             int16 samples[n_records][n_samples]
// Usage: lead1_host <in.bin> <out.csv> [mv_per_lsb=0.001]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "../ecg_features/ecg_features.h"

static ecg::Lead1Workspace g_ws;  // static, exactly as on the device
static int16_t g_window[ecg::kWindowLen];

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <in.bin> <out.csv> [mv_per_lsb]\n", argv[0]);
    return 2;
  }
  const double mv_per_lsb = argc > 3 ? atof(argv[3]) : 0.001;
  FILE* in = fopen(argv[1], "rb");
  if (!in) { perror(argv[1]); return 1; }
  int32_t hdr[2];
  if (fread(hdr, sizeof(int32_t), 2, in) != 2 || hdr[1] != ecg::kWindowLen) {
    fprintf(stderr, "bad header (need n_samples == %d)\n", ecg::kWindowLen);
    return 1;
  }
  FILE* out = fopen(argv[2], "w");
  if (!out) { perror(argv[2]); return 1; }
  fprintf(out,
          "mean_hr,min_hr,max_hr,hr_std,mean_rr,rr_std,rr_range,rr_cv,sdnn,"
          "rmssd,pnn50,hjorth_complexity,mean_qrs_ms,snr_db,baseline_wander,"
          "n_peaks\n");

  clock_t cpu = 0;
  for (int r = 0; r < hdr[0]; ++r) {
    if (fread(g_window, sizeof(int16_t), ecg::kWindowLen, in) !=
        static_cast<size_t>(ecg::kWindowLen)) {
      fprintf(stderr, "short read at record %d\n", r);
      return 1;
    }
    ecg::Lead1Features f;
    const clock_t t0 = clock();
    const int rc = ecg::ExtractLead1Features(
        g_window, ecg::kWindowLen, static_cast<ecg::real_t>(mv_per_lsb), &g_ws, &f);
    cpu += clock() - t0;
    if (rc != ecg::kOk) { fprintf(stderr, "record %d: rc=%d\n", r, rc); return 1; }
    ecg::real_t v[ecg::kNumFeatures];
    ecg::FeaturesToArray(f, v);
    for (int i = 0; i < ecg::kNumFeatures; ++i) fprintf(out, "%.9g,", (double)v[i]);
    fprintf(out, "%d\n", f.n_peaks);
  }
  fclose(out);
  fclose(in);
  fprintf(stderr, "processed %d windows, sizeof(Lead1Workspace)=%u B, "
                  "host CPU %.3f ms/window\n",
          hdr[0], static_cast<unsigned>(sizeof(ecg::Lead1Workspace)),
          hdr[0] ? 1000.0 * cpu / CLOCKS_PER_SEC / hdr[0] : 0.0);
  return 0;
}
