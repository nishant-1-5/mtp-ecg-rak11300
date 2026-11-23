#include "ecg_cnn.h"

#include <new>

#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "../../models/lead1/model_data.h"  // g_model, g_model_len, g_model_input_*

namespace ecg {
namespace {

// Exactly the 10 builtin ops in tinyml_1d_cnn_int8.tflite
// (tf.lite.experimental.Analyzer); registering only these keeps flash small.
using OpResolver = tflite::MicroMutableOpResolver<10>;

alignas(OpResolver) uint8_t g_resolver_mem[sizeof(OpResolver)];
alignas(tflite::MicroInterpreter) uint8_t g_interp_mem[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* g_interp = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
uint8_t* g_arena = nullptr;
tflite::SingleArenaBufferAllocator* g_buffers = nullptr;

}  // namespace

bool CnnInit(uint8_t* arena, size_t arena_size, CnnQuant* q) {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("model schema %d != %d", static_cast<int>(model->version()), TFLITE_SCHEMA_VERSION);
    return false;
  }
  OpResolver* r = new (g_resolver_mem) OpResolver();
  if (r->AddExpandDims() != kTfLiteOk || r->AddConv2D() != kTfLiteOk ||
      r->AddMul() != kTfLiteOk || r->AddAdd() != kTfLiteOk || r->AddReshape() != kTfLiteOk ||
      r->AddMaxPool2D() != kTfLiteOk || r->AddDepthwiseConv2D() != kTfLiteOk ||
      r->AddMean() != kTfLiteOk || r->AddFullyConnected() != kTfLiteOk ||
      r->AddLogistic() != kTfLiteOk) {
    return false;
  }
  // Same construction as MicroAllocator::Create(arena, size) (greedy planner),
  // but keeping the buffer allocator so the activation (head) size can be read.
  uint8_t* aligned = tflite::AlignPointerUp(arena, tflite::MicroArenaBufferAlignment());
  const size_t aligned_size = static_cast<size_t>(arena + arena_size - aligned);
  g_buffers = tflite::SingleArenaBufferAllocator::Create(aligned, aligned_size);
  uint8_t* planner_mem = g_buffers->AllocatePersistentBuffer(sizeof(tflite::GreedyMemoryPlanner),
                                                             alignof(tflite::GreedyMemoryPlanner));
  auto* planner = new (planner_mem) tflite::GreedyMemoryPlanner();
  tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(g_buffers, planner);
  g_interp = new (g_interp_mem) tflite::MicroInterpreter(model, *r, allocator);
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed (arena %u B)", static_cast<unsigned>(arena_size));
    return false;
  }
  g_input = g_interp->input(0);
  g_output = g_interp->output(0);
  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8 ||
      g_input->bytes != static_cast<size_t>(kWindowLen) ||
      g_output->bytes != static_cast<size_t>(kNumClasses)) {
    MicroPrintf("unexpected model I/O");
    return false;
  }
  q->input_scale = g_input->params.scale;
  q->input_zero_point = g_input->params.zero_point;
  // model_data.h records the same values; a mismatch means a stale header.
  if (q->input_scale != g_model_input_scale || q->input_zero_point != g_model_input_zero_point) {
    MicroPrintf("model_data.h quantisation constants do not match the flatbuffer");
    return false;
  }
  g_arena = aligned;
  return true;
}

size_t CnnArenaHeadBytes() {
  // Head = activations + kernel scratch, starting at the (aligned) arena start.
  return g_buffers ? g_buffers->GetNonPersistentUsedBytes() : 0;
}

Lead1Workspace* CnnSharedWorkspace() {
  if (g_arena == nullptr || CnnArenaHeadBytes() < sizeof(Lead1Workspace)) return nullptr;
  static_assert(alignof(Lead1Workspace) <= 16, "arena head is 16-byte aligned");
  return new (g_arena) Lead1Workspace;  // trivial type: no initialisation, just a typed view
}

int8_t* CnnInputBuffer() { return g_input ? g_input->data.int8 : nullptr; }

int CnnInvoke(const int8_t* input, int len, float* probs, void* /*ctx*/) {
  if (g_interp == nullptr || input != g_input->data.int8 || len != kWindowLen) return 1;
  if (g_interp->Invoke() != kTfLiteOk) return 2;
  const float s = g_output->params.scale;
  const int z = g_output->params.zero_point;
  for (int k = 0; k < kNumClasses; ++k) probs[k] = s * static_cast<float>(g_output->data.int8[k] - z);
  return 0;
}

size_t CnnArenaUsed() { return g_interp ? g_interp->arena_used_bytes() : 0; }

}  // namespace ecg
