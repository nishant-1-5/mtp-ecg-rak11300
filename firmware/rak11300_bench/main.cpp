// RAK11300 (RP2040) benchmark for the Lead I pipeline.
//
// For each embedded PTB-XL window (bench_vectors.h):
//   * runs ExtractLead1Features kRepeats times and reports per-stage time,
//   * checks the features, quality bits and rhythm flags against the Python
//     reference values baked into the header,
//   * times AssessQuality + RhythmFlags + PrepareCnnInput.
// The CNN is not included yet (TFLM integration is Phase 2).
//
// Output goes to USB CDC and UART0 (115200 8N1). Open a serial terminal after flashing.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

#include "../ecg_features/ecg_features.h"
#include "../ecg_pipeline/ecg_pipeline.h"
#include "bench_vectors.h"

namespace {

constexpr int kRepeats = 20;
constexpr int kStages = 6;  // marks 0..5 in ecg_features.cpp
const char* const kStageName[kStages - 1] = {
    "filter + R peaks", "HRV", "delineation + SQI", "Hjorth", "Welch PSD"};

uint64_t g_mark[kStages];
ecg::Lead1Workspace g_ws;      // 5,780 B in .bss
int8_t g_cnn_input[ecg::kWindowLen];

bool Close(float got, float want) {
  if (ecg::IsNan(want) || ecg::IsNan(got)) return ecg::IsNan(want) && ecg::IsNan(got);
  const float d = fabsf(got - want);
  return d <= 1e-3f || d <= 1e-3f * fabsf(want);
}

}  // namespace

extern "C" void EcgProfileMark(int stage) {
  if (stage >= 0 && stage < kStages) g_mark[stage] = time_us_64();
}

int main() {
  stdio_init_all();
  sleep_ms(3000);  // give the USB host time to enumerate the CDC port

  printf("\n=== RAK11300 Lead I pipeline benchmark ===\n");
  printf("clk_sys = %lu Hz, sizeof(Lead1Workspace) = %u B, sizeof(Lead1Features) = %u B\n",
         static_cast<unsigned long>(clock_get_hz(clk_sys)),
         static_cast<unsigned>(sizeof(ecg::Lead1Workspace)),
         static_cast<unsigned>(sizeof(ecg::Lead1Features)));

  const ecg::AdcInfo info = {0.001f, -32768, 32767};
  const ecg::CnnQuant q = {0.0541500449f, -25};  // models/lead1/model_data.h
  int failures = 0;

  for (int w = 0; w < BENCH_NUM_WINDOWS; ++w) {
    uint64_t stage_us[kStages - 1] = {0};
    uint64_t total_us = 0;
    ecg::Lead1Features f;
    for (int r = 0; r < kRepeats; ++r) {
      const uint64_t t0 = time_us_64();
      ecg::ExtractLead1Features(kBenchAdc[w], ecg::kWindowLen, 0.001f, &g_ws, &f);
      total_us += time_us_64() - t0;
      for (int s = 0; s + 1 < kStages; ++s) stage_us[s] += g_mark[s + 1] - g_mark[s];
    }

    const uint64_t t1 = time_us_64();
    const uint8_t quality = ecg::AssessQuality(kBenchAdc[w], ecg::kWindowLen, info, f);
    const uint8_t flags = ecg::RhythmFlags(f);
    ecg::PrepareCnnInput(kBenchAdc[w], ecg::kWindowLen, info.mv_per_lsb, q, g_cnn_input);
    const uint64_t post_us = time_us_64() - t1;

    float v[ecg::kNumFeatures];
    ecg::FeaturesToArray(f, v);
    int bad = 0;
    for (int k = 0; k < ecg::kNumFeatures; ++k) {
      if (!Close(v[k], kBenchExpected[w][k])) {
        ++bad;
        uint32_t gb, wb;
        memcpy(&gb, &v[k], 4);
        memcpy(&wb, &kBenchExpected[w][k], 4);
        printf("  feature %d: got 0x%08lx, expected 0x%08lx\n", k, static_cast<unsigned long>(gb),
               static_cast<unsigned long>(wb));
      }
    }
    const bool ok = bad == 0 && f.n_peaks == kBenchExpectedPeaks[w] &&
                    quality == kBenchExpectedQuality[w] && flags == kBenchExpectedFlags[w];
    failures += ok ? 0 : 1;

    printf("\n[%d] %s: %s (peaks %u, quality 0x%02x, flags 0x%02x)\n", w, kBenchLabel[w],
           ok ? "MATCH" : "MISMATCH", f.n_peaks, quality, flags);
    printf("  features total: %.2f ms\n", static_cast<double>(total_us) / kRepeats / 1000.0);
    for (int s = 0; s + 1 < kStages; ++s)
      printf("    %-20s %.2f ms\n", kStageName[s],
             static_cast<double>(stage_us[s]) / kRepeats / 1000.0);
    printf("  gate + flags + CNN input prep: %.2f ms\n", static_cast<double>(post_us) / 1000.0);
  }

  printf("\n=== %s: %d of %d windows match the Python reference ===\n",
         failures == 0 ? "PASS" : "FAIL", BENCH_NUM_WINDOWS - failures, BENCH_NUM_WINDOWS);
  while (true) sleep_ms(1000);
}
