// Host harness for firmware/ecg_pipeline (driven by scripts/compare_pipeline.py).
//
//   pipeline_host <windows.bin> <out.csv> <cnn_inputs_out.bin> <scale> <zero_point> [probs_in.bin]
//
// windows.bin: int32 n_records, int32 n_samples, int16 samples[n][n_samples] (PTB-XL scale 0.001 mV/LSB)
// The CNN callback appends every INT8 input it receives to cnn_inputs_out.bin.
// Without probs_in.bin it returns zeros (pass 1); with it, it replays float32[5]
// probabilities per CNN call, in call order (pass 2, computed by TFLite in Python).
#include <stdio.h>
#include <stdlib.h>

#include "../ecg_pipeline/ecg_pipeline.h"

namespace {

struct ReplayCnn {
  FILE* dump;
  FILE* probs;
};

int ReplayCnnFn(const int8_t* input, int len, float* probs, void* ctx) {
  ReplayCnn* c = static_cast<ReplayCnn*>(ctx);
  fwrite(input, 1, static_cast<size_t>(len), c->dump);
  if (c->probs) {
    if (fread(probs, sizeof(float), ecg::kNumClasses, c->probs) != ecg::kNumClasses) return 1;
  } else {
    for (int k = 0; k < ecg::kNumClasses; ++k) probs[k] = 0;
  }
  return 0;
}

ecg::Lead1Workspace g_ws;
int16_t g_window[ecg::kWindowLen];
int8_t g_cnn_input[ecg::kWindowLen];  // on the device: the TFLM input tensor

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s windows.bin out.csv cnn_inputs_out.bin scale zero_point [probs.bin]\n",
            argv[0]);
    return 2;
  }
  FILE* in = fopen(argv[1], "rb");
  FILE* out = fopen(argv[2], "w");
  ReplayCnn cnn = {fopen(argv[3], "wb"), argc > 6 ? fopen(argv[6], "rb") : nullptr};
  if (!in || !out || !cnn.dump || (argc > 6 && !cnn.probs)) { perror("open"); return 1; }
  const ecg::CnnQuant q = {static_cast<float>(atof(argv[4])), atoi(argv[5])};
  const ecg::AdcInfo info = {static_cast<ecg::real_t>(0.001), -32768, 32767};

  int32_t hdr[2];
  if (fread(hdr, sizeof(int32_t), 2, in) != 2 || hdr[1] != ecg::kWindowLen) return 1;
  fprintf(out, "quality,rhythm_flags,cnn_ran,p_CD,p_HYP,p_MI,p_NORM,p_STTC,classes,template_corr\n");
  for (int r = 0; r < hdr[0]; ++r) {
    if (fread(g_window, sizeof(int16_t), ecg::kWindowLen, in) != static_cast<size_t>(ecg::kWindowLen))
      return 1;
    ecg::PipelineResult res;
    const int rc = ecg::RunPipeline(g_window, ecg::kWindowLen, info, q, ReplayCnnFn, &cnn,
                                    g_cnn_input, &g_ws, &res);
    if (rc != ecg::kPipelineOk) { fprintf(stderr, "record %d: rc=%d\n", r, rc); return 1; }
    fprintf(out, "%d,%d,%d", res.quality, res.rhythm_flags, res.cnn_ran ? 1 : 0);
    for (int k = 0; k < ecg::kNumClasses; ++k) fprintf(out, ",%.7g", static_cast<double>(res.probs[k]));
    const double tc = static_cast<double>(res.features.template_corr);
    if (tc != tc) fprintf(out, ",%d,nan\n", res.classes);
    else fprintf(out, ",%d,%.7g\n", res.classes, tc);
  }
  fclose(in);
  fclose(out);
  fclose(cnn.dump);
  if (cnn.probs) fclose(cnn.probs);
  return 0;
}
