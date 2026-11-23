// RAK11300 (RP2040) end-to-end pipeline check: features -> quality gate ->
// rhythm flags -> INT8 CNN under TensorFlow Lite Micro -> class decisions,
// on the windows in bench_vectors.h, compared with the Python/TFLite reference.
//
// Reports: TFLM arena use, peak core-0 stack, and per-window latency of the
// feature stage and of CNN inference. Output on UART0 (115200 8N1) and USB CDC.
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "pico/stdlib.h"

#include "../ecg_cnn/ecg_cnn.h"
#include "../ecg_features/ecg_features.h"
#include "../ecg_pipeline/ecg_pipeline.h"
#ifdef BENCH_VALIDATION
#include "bench_vectors_validation.h"  // generated: make_bench_vectors.py --validation N
#else
#include "bench_vectors.h"
#endif

extern "C" char __scratch_x_end__;  // pico-sdk linker script; core-0 stack sits above it
extern "C" char __StackTop;

namespace {

constexpr int kRepeats = BENCH_NUM_WINDOWS > 10 ? 1 : 10;
constexpr uint8_t kPaint = 0xA5;

alignas(16) uint8_t g_arena[ecg::kCnnArenaSize];
ecg::Lead1Workspace* g_ws = nullptr;  // overlaid on the arena head (CnnSharedWorkspace)

struct TimedCnn {
  uint64_t us = 0;
};

int TimedInvoke(const int8_t* input, int len, float* probs, void* ctx) {
  const uint64_t t0 = time_us_64();
  const int rc = ecg::CnnInvoke(input, len, probs, nullptr);
  static_cast<TimedCnn*>(ctx)->us += time_us_64() - t0;
  return rc;
}

// Fill the unused stack below the current SP with a pattern; later scan for
// the lowest overwritten byte.
__attribute__((noinline)) void PaintStack() {
  uint8_t* p = reinterpret_cast<uint8_t*>(&__scratch_x_end__);
  uint8_t* sp = static_cast<uint8_t*>(__builtin_frame_address(0)) - 64;
  while (p < sp) *p++ = kPaint;
}

size_t StackHighWater() {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&__scratch_x_end__);
  const uint8_t* top = reinterpret_cast<const uint8_t*>(&__StackTop);
  while (p < top && *p == kPaint) ++p;
  return static_cast<size_t>(top - p);
}

}  // namespace

int main() {
  stdio_init_all();
  sleep_ms(3000);
  printf("\n=== RAK11300 Lead I pipeline + TFLM ===\n");
  printf("clk_sys = %lu Hz\n", static_cast<unsigned long>(clock_get_hz(clk_sys)));

  ecg::CnnQuant q;
  if (!ecg::CnnInit(g_arena, sizeof(g_arena), &q)) {
    printf("CnnInit FAILED\n=== FAIL: CnnInit ===\n");
    while (true) sleep_ms(1000);
  }
  printf("TFLM arena: %u B used of %u B; input scale %.9g, zero point %d\n",
         static_cast<unsigned>(ecg::CnnArenaUsed()), static_cast<unsigned>(sizeof(g_arena)),
         static_cast<double>(q.input_scale), q.input_zero_point);
  g_ws = ecg::CnnSharedWorkspace();
  printf("activation head %u B; Lead1Workspace (%u B) %s\n",
         static_cast<unsigned>(ecg::CnnArenaHeadBytes()),
         static_cast<unsigned>(sizeof(ecg::Lead1Workspace)),
         g_ws ? "overlaid on the arena head" : "does NOT fit: abort");
  if (g_ws == nullptr) {
    printf("=== FAIL: workspace overlay ===\n");
    while (true) sleep_ms(1000);
  }

  const ecg::AdcInfo info = {0.001f, -32768, 32767};
  int failures = 0;
  size_t stack_peak = 0;
  for (int w = 0; w < BENCH_NUM_WINDOWS; ++w) {
    TimedCnn timer;
    ecg::PipelineResult res;
    uint64_t total_us = 0;
    int rc = 0;
    PaintStack();
    for (int r = 0; r < kRepeats; ++r) {
      const uint64_t t0 = time_us_64();
      rc |= ecg::RunPipeline(kBenchAdc[w], ecg::kWindowLen, info, q, TimedInvoke, &timer,
                             ecg::CnnInputBuffer(), g_ws, &res);
      total_us += time_us_64() - t0;
    }
    const size_t stack = StackHighWater();
    if (stack > stack_peak) stack_peak = stack;

    bool ok = rc == ecg::kPipelineOk && res.quality == kBenchExpectedQuality[w] &&
              res.rhythm_flags == kBenchExpectedFlags[w];
    float max_dp = 0;
    if (res.cnn_ran) {
      for (int k = 0; k < ecg::kNumClasses; ++k) {
        const float d = fabsf(res.probs[k] - kBenchExpectedProbs[w][k]);
        if (d > max_dp) max_dp = d;
      }
      // Reference is TFLite's BUILTIN_REF kernels, which match TFLM/CMSIS-NN to
      // within 1 output LSB (1/256).
      ok = ok && max_dp <= 1.0f / 256 + 1e-6f && res.classes == kBenchExpectedClasses[w];
    } else {
      ok = ok && kBenchExpectedClasses[w] == 0;
    }
    failures += ok ? 0 : 1;

    printf("\n[%d] %s: %s\n", w, kBenchLabel[w], ok ? "MATCH" : "MISMATCH");
    printf("  quality 0x%02x flags 0x%02x cnn %s classes 0x%02x (expected 0x%02x)\n", res.quality,
           res.rhythm_flags, res.cnn_ran ? "ran" : "skipped", res.classes, kBenchExpectedClasses[w]);
    if (res.cnn_ran) {
      printf("  probs CD %.3f HYP %.3f MI %.3f NORM %.3f STTC %.3f, max |diff| vs TFLite %.4f\n",
             static_cast<double>(res.probs[0]), static_cast<double>(res.probs[1]),
             static_cast<double>(res.probs[2]), static_cast<double>(res.probs[3]),
             static_cast<double>(res.probs[4]), static_cast<double>(max_dp));
      printf("  CNN invoke %.2f ms, ", static_cast<double>(timer.us) / kRepeats / 1000.0);
    } else {
      printf("  ");
    }
    printf("whole pipeline %.2f ms, peak stack %u B\n",
           static_cast<double>(total_us) / kRepeats / 1000.0, static_cast<unsigned>(stack));
  }
  printf("\npeak core-0 stack over all windows: %u B\n", static_cast<unsigned>(stack_peak));
  printf("=== %s: %d of %d windows match the Python reference ===\n", failures == 0 ? "PASS" : "FAIL",
         BENCH_NUM_WINDOWS - failures, BENCH_NUM_WINDOWS);
  while (true) sleep_ms(1000);
}
