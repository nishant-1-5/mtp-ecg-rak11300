// TensorFlow Lite Micro wrapper for the Lead I INT8 1D-CNN (models/lead1/model_data.h).
//
// Usage on the device:
//   static uint8_t arena[kCnnArenaSize] __attribute__((aligned(16)));
//   ecg::CnnQuant q;
//   if (!ecg::CnnInit(arena, sizeof(arena), &q)) { ... }
//   ecg::RunPipeline(adc, n, info, q, ecg::CnnInvoke, nullptr, ecg::CnnInputBuffer(), &ws, &res);
//
// CnnInputBuffer() is the TFLM input tensor, so PrepareCnnInput writes the
// quantised window straight into the arena (no extra 1 KB buffer).
#ifndef ECG_CNN_H_
#define ECG_CNN_H_

#include <stddef.h>
#include <stdint.h>

#include "../ecg_pipeline/ecg_pipeline.h"

namespace ecg {

// Arena handed to TFLM. Measured use for models/lead1 (CnnArenaUsed, printed by
// rak11300_pipeline): 20,788 B. 21 KB leaves ~700 B of headroom.
constexpr size_t kCnnArenaSize = 21 * 1024;

// Builds the interpreter. Checks that the model's input/output shapes and the
// input quantisation match what PrepareCnnInput produces, and returns them in q.
bool CnnInit(uint8_t* arena, size_t arena_size, CnnQuant* q);

int8_t* CnnInputBuffer();

// CnnFn-compatible: runs the model on CnnInputBuffer() (input must point to it)
// and writes 5 dequantised sigmoid outputs.
int CnnInvoke(const int8_t* input, int len, float* probs, void* ctx);

size_t CnnArenaUsed();

// Feature workspace overlaid on the start of the arena, saving sizeof(Lead1Workspace).
//
// Safe because TFLM keeps per-inference activations (including the input
// tensor) at the head of the arena and persistent data at the tail, and the
// pipeline is strictly sequential: ExtractLead1Features is finished before
// PrepareCnnInput writes the input tensor, and CnnInvoke copies the outputs
// out before the next window. CnnInit verifies that the activation region
// covers the workspace; if it does not, this returns nullptr and the caller
// must use its own Lead1Workspace.
Lead1Workspace* CnnSharedWorkspace();

// Bytes from the arena start to the end of the highest activation tensor.
size_t CnnArenaHeadBytes();

}  // namespace ecg

#endif  // ECG_CNN_H_
