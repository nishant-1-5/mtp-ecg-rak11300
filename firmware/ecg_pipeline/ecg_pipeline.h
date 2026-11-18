// On-device decision logic for one 10 s Lead I window (RAK11300).
//
// C++ port of scripts/orchestrator.py:
//   1. ExtractLead1Features (firmware/ecg_features)
//   2. AssessQuality  -> reason bitmask; nonzero = window unreliable, CNN skipped
//   3. RhythmFlags    -> rule-based rhythm / interval flags
//   4. PrepareCnnInput (z-score + INT8 quantisation) -> CNN callback -> DecideClasses
// Thresholds come from the generated ecg_pipeline_params.h (scripts/orchestrator.py fit).
//
// The CNN itself is a callback so this module has no TFLM dependency: the
// firmware passes a function that invokes the TFLM interpreter (Phase 2), the
// host test passes one that replays probabilities computed in Python.
#ifndef ECG_PIPELINE_H_
#define ECG_PIPELINE_H_

#include <stdint.h>

#include "../ecg_features/ecg_features.h"

namespace ecg {

// Quality reason bits (scripts/orchestrator.py Q_*).
enum QualityReason : uint8_t {
  kQualityFlat = 1,        // peak-to-peak < 0.1 mV: lead off / flat line
  kQualitySaturated = 2,   // > 1 % of samples on an ADC rail
  kQualityNoRhythm = 4,    // < 4 R peaks or mean HR outside 30-220 bpm
  kQualityNoTemplate = 8,  // < 3 beats usable for the median beat
  kQualityNoisy = 16,      // beats do not match the median beat (template correlation)
  kQualityBaseline = 32,   // too much power below 0.5 Hz
};

// Rhythm / interval flag bits (scripts/orchestrator.py F_*).
enum RhythmFlag : uint8_t {
  kFlagTachy = 1,       // mean HR > 100 bpm
  kFlagBrady = 2,       // mean HR < 60 bpm
  kFlagIrregular = 4,   // RR coefficient of variation above the fitted threshold
  kFlagAfSuspect = 8,   // irregular and no P wave found
  kFlagWideQrs = 16,    // QRS >= 120 ms
  kFlagLongPr = 32,     // PR > 200 ms
};

constexpr int kNumClasses = 5;  // bit k of PipelineResult::classes: CD, HYP, MI, NORM, STTC

struct AdcInfo {
  real_t mv_per_lsb;  // e.g. 0.001 for PTB-XL
  int32_t adc_min;    // ADC rails, for saturation detection
  int32_t adc_max;
};

struct CnnQuant {  // input quantisation of the INT8 model (model_data.h)
  float input_scale;
  int input_zero_point;
};

// Runs the INT8 CNN on `input` (kWindowLen values) and writes kNumClasses
// probabilities. Returns 0 on success.
typedef int (*CnnFn)(const int8_t* input, int len, float* probs, void* ctx);

struct PipelineResult {
  Lead1Features features;
  uint8_t quality;       // QualityReason bits, 0 = usable
  uint8_t rhythm_flags;  // RhythmFlag bits (computed even if quality != 0)
  bool cnn_ran;
  float probs[kNumClasses];  // valid if cnn_ran
  uint8_t classes;           // decided superclasses, valid if cnn_ran
};

uint8_t AssessQuality(const int16_t* adc, int n, const AdcInfo& info, const Lead1Features& f);
uint8_t RhythmFlags(const Lead1Features& f);

// z = (x - mean) / (std + 0.001 mV) per window, then q = round(z / scale) + zero_point
// (round half to even, as numpy). mean/std use exact integer moments of the ADC counts.
void PrepareCnnInput(const int16_t* adc, int n, real_t mv_per_lsb, const CnnQuant& q,
                     int8_t* out);

uint8_t DecideClasses(const float* probs, uint8_t rhythm_flags);

enum PipelineStatus : int { kPipelineOk = 0, kPipelineBadArgs = -1, kPipelineCnnError = -2 };

// `cnn_input` (kWindowLen bytes) may point straight at the TFLM input tensor,
// so the quantised window needs no extra buffer.
int RunPipeline(const int16_t* adc, int n, const AdcInfo& info, const CnnQuant& q,
                CnnFn cnn, void* cnn_ctx, int8_t* cnn_input, Lead1Workspace* ws,
                PipelineResult* out);

}  // namespace ecg

#endif  // ECG_PIPELINE_H_
