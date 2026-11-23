// Mirrors scripts/orchestrator.py (gate, rhythm_flags, cnn_input_int8, decide).
#include "ecg_pipeline.h"

#include <cmath>

#include "ecg_pipeline_params.h"

namespace ecg {

uint8_t AssessQuality(const int16_t* adc, int n, const AdcInfo& info, const Lead1Features& f) {
  uint8_t reasons = 0;
  int32_t lo = adc[0], hi = adc[0];
  int rails = 0;
  for (int i = 0; i < n; ++i) {
    if (adc[i] < lo) lo = adc[i];
    if (adc[i] > hi) hi = adc[i];
    if (adc[i] <= info.adc_min || adc[i] >= info.adc_max) ++rails;
  }
  if (static_cast<real_t>(hi - lo) * info.mv_per_lsb < params::kMinPtpMv) reasons |= kQualityFlat;
  if (static_cast<float>(rails) / static_cast<float>(n) > params::kMaxRailFraction)
    reasons |= kQualitySaturated;
  if (f.n_peaks < params::kMinPeaks ||
      !(f.mean_hr >= params::kHrMinBpm && f.mean_hr <= params::kHrMaxBpm))
    reasons |= kQualityNoRhythm;
  if (f.n_beats < params::kMinBeats)
    reasons |= kQualityNoTemplate;
  else if (IsNan(f.template_corr) || f.template_corr < params::kMinTemplateCorr)
    reasons |= kQualityNoisy;
  if (f.baseline_wander > params::kMaxBaselineWander) reasons |= kQualityBaseline;
  return reasons;
}

uint8_t RhythmFlags(const Lead1Features& f) {
  uint8_t flags = 0;
  if (f.mean_hr > params::kTachyBpm) flags |= kFlagTachy;
  if (f.mean_hr < params::kBradyBpm) flags |= kFlagBrady;
  if (f.rr_cv >= params::kIrregularRrCv) {
    flags |= kFlagIrregular;
    if (IsNan(f.pr_ms)) flags |= kFlagAfSuspect;  // no P wave
  }
  if (!IsNan(f.qrs_ms) && f.qrs_ms >= params::kWideQrsMs) flags |= kFlagWideQrs;
  if (!IsNan(f.pr_ms) && f.pr_ms > params::kLongPrMs) flags |= kFlagLongPr;
  return flags;
}

void PrepareCnnInput(const int16_t* adc, int n, real_t mv_per_lsb, const CnnQuant& q,
                     int8_t* out) {
  int64_t s1 = 0, s2 = 0;
  for (int i = 0; i < n; ++i) {
    s1 += adc[i];
    s2 += static_cast<int64_t>(adc[i]) * adc[i];
  }
  const real_t mean_counts = static_cast<real_t>(s1) / static_cast<real_t>(n);
  const real_t var_counts = static_cast<real_t>(n * s2 - s1 * s1) / static_cast<real_t>(int64_t(n) * n);
  const real_t std_mv = std::sqrt(var_counts) * mv_per_lsb;
  // z / scale folded into one multiplier per sample.
  const real_t k = mv_per_lsb / ((std_mv + real_t(1e-3)) * q.input_scale);
  for (int i = 0; i < n; ++i) {
    const real_t v = std::nearbyint((static_cast<real_t>(adc[i]) - mean_counts) * k) +
                     static_cast<real_t>(q.input_zero_point);
    out[i] = static_cast<int8_t>(v < -128 ? -128 : (v > 127 ? 127 : v));
  }
}

uint8_t DecideClasses(const float* probs, uint8_t rhythm_flags) {
  uint8_t out = 0;
  for (int k = 0; k < kNumClasses; ++k)
    if (probs[k] >= params::kClassThreshold[k]) out = static_cast<uint8_t>(out | (1u << k));
  if (params::kCdOrWideQrs && (rhythm_flags & kFlagWideQrs)) out |= 1u;  // CD is bit 0
  return out;
}

int RunPipeline(const int16_t* adc, int n, const AdcInfo& info, const CnnQuant& q,
                CnnFn cnn, void* cnn_ctx, int8_t* cnn_input, Lead1Workspace* ws,
                PipelineResult* out) {
  if (adc == nullptr || cnn == nullptr || cnn_input == nullptr || ws == nullptr ||
      out == nullptr) {
    return kPipelineBadArgs;
  }
  if (ExtractLead1Features(adc, n, info.mv_per_lsb, ws, &out->features) != kOk)
    return kPipelineBadArgs;
  out->quality = AssessQuality(adc, n, info, out->features);
  out->rhythm_flags = RhythmFlags(out->features);
  out->cnn_ran = false;
  out->classes = 0;
  for (int k = 0; k < kNumClasses; ++k) out->probs[k] = 0;
  if (out->quality != 0) return kPipelineOk;  // unreliable window: report, don't classify

  PrepareCnnInput(adc, n, info.mv_per_lsb, q, cnn_input);
  if (cnn(cnn_input, n, out->probs, cnn_ctx) != 0) return kPipelineCnnError;
  out->cnn_ran = true;
  out->classes = DecideClasses(out->probs, out->rhythm_flags);
  return kPipelineOk;
}

}  // namespace ecg
