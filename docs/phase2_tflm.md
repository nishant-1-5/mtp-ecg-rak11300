# Phase 2: TensorFlow Lite Micro on the RP2040 (without the board)

The RAK11300 is not available, so Phase 2 uses the same firmware the board would run, executed on
**rp2040js** ([wokwi/rp2040js](https://github.com/wokwi/rp2040js)). rp2040js is an open-source
RP2040 emulator that runs the unmodified `.uf2` image and simulates the Cortex-M0+ at 125 MHz.
Every number below comes from that emulator, from the linker map, or from TFLM itself. Board
measurements still need a board.

## What was built

| Path | Purpose |
|---|---|
| `firmware/ecg_cnn/ecg_cnn.{h,cpp}` | TFLM wrapper: interpreter setup with the 10 ops the model uses, a `CnnFn` callback for `RunPipeline`, input tensor exposed as the `PrepareCnnInput` target, workspace overlay |
| `firmware/cmake/pico_tflmicro.cmake` | Builds [raspberrypi/pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro) (TFLM + CMSIS-NN) from a checkout. Details below. |
| `firmware/rak11300_bench/pipeline_main.cpp` | `rak11300_pipeline`: full pipeline on the 6 bench windows. `rak11300_validation`: the same code on 100 random fold-10 windows. |
| `firmware/rak11300_bench/emulate/run.ts` | Runs a `.uf2` on rp2040js and exits on the firmware's `=== PASS/FAIL` line |
| `firmware/rak11300_bench/emulate/*_output.txt` | Captured emulator output for the numbers in this document |
| `scripts/map_sizes.py` | Flash/RAM per component from the linker map |

```powershell
python scripts/make_bench_vectors.py                    # 6 hand-picked windows (committed header)
python scripts/make_bench_vectors.py --validation 100   # 100 random fold-10 windows (build/ only)
powershell -ExecutionPolicy Bypass -File firmware/rak11300_bench/build.ps1
.toolchain/rp2040js/node_modules/.bin/tsx --tsconfig .toolchain/rp2040js/tsconfig.json `
    firmware/rak11300_bench/emulate/run.ts firmware/rak11300_bench/build/rak11300_validation.uf2 1700
```

## Correctness

Each window runs the complete firmware pipeline: features, gate, flags, INT8 input, TFLM inference
and decisions. The result is checked against the Python reference computed offline.

| Image | Windows | Result |
|---|---:|---|
| `rak11300_bench` (features only) | 6 | PASS: 18 features, peak count, gate bits, flags |
| `rak11300_pipeline` | 6 | PASS: plus probabilities within 1/256 and identical class decisions |
| `rak11300_validation` | 100 random fold-10 | **PASS 100/100** |

Two problems surfaced only on the RP2040. Neither shows up in the host build.

1. **NaN handling.** The Pico SDK's fast float routines are built without NaN propagation
   (`PICO_FLOAT_PROPAGATE_NANS=0`). Arithmetic on a NaN can therefore produce ±inf instead of NaN,
   and `x != x` is not a reliable NaN test. The first emulator run returned PR = −inf instead of NaN
   for a window without a P wave. The fix: `ecg::IsNan()` tests the bit pattern, NaN never enters
   arithmetic, and every NaN check in the firmware uses `IsNan()`.
2. **Which desktop kernels to compare against.** Desktop TFLite uses XNNPACK by default. XNNPACK
   rounds requantisation differently from TFLM/CMSIS-NN, by up to 9 output LSB on this model, and
   9 of the first 100 validation windows "failed" for that reason alone. TFLite's reference kernels
   (`BUILTIN_REF`) agree with the device to within 1 LSB. All Python INT8 evaluation now uses them
   (`orchestrator.Int8Cnn`, `compare_lead_sets.py`). The results changed by under 0.001: macro AUC
   0.8294 → 0.8292, macro F1 0.605 → 0.604. The fitted thresholds are unchanged.

## Memory (measured)

| Item | Bytes | How measured |
|---|---:|---|
| TFLM arena, used | **20,788** | `MicroInterpreter::arena_used_bytes()` |
| — activation head (input, activations, CMSIS-NN scratch) | 16,000 | `SingleArenaBufferAllocator::GetNonPersistentUsedBytes()` |
| — persistent tail (tensor metadata, op data, planner) | 4,788 | difference |
| Arena allocated (`kCnnArenaSize`) | 21,504 | 21 KB, ~700 B headroom |
| `Lead1Workspace` | 0 extra | overlaid on the activation head (below) |
| Interpreter + op resolver objects | 568 | `nm` |
| Peak core-0 stack, whole pipeline | **1,636** | painted stack, high-water mark |

**Workspace overlay.** The 5,780 B feature workspace lives in the first bytes of the TFLM arena.
This is safe because the pipeline is strictly sequential:

1. `ExtractLead1Features` finishes before `PrepareCnnInput` writes the input tensor.
2. TFLM keeps everything that must persist between inferences at the tail of the arena, never at
   the head.
3. `CnnInvoke` copies the outputs out before the next window.

`CnnInit` reads the activation-head size from TFLM's allocator, and `CnnSharedWorkspace()` returns
`nullptr` if the workspace would not fit. A retrained model therefore cannot silently break the
overlay. The 100-window validation runs with the overlay enabled.

**SRAM total for the ECG pipeline**

| Buffer | Bytes |
|---|---:|
| TFLM arena incl. feature workspace | 21,504 |
| Interpreter / resolver objects | 568 |
| Acquisition ping-pong buffers (2 × 1000 × int16, design) | 4,000 |
| Peak stack | 1,636 |
| **Total** | **27,708** (target < 30 KB) |

Without the overlay the total would be 33,488 B.

**Flash** (`scripts/map_sizes.py` on `rak11300_pipeline.elf.map`)

| Component | Flash bytes |
|---|---:|
| TFLM + CMSIS-NN runtime (10 ops) | 106,744 |
| Model array + wrapper (`ecg_cnn`) | 29,464 |
| Feature extraction + decision logic | 13,948 |
| pico-sdk, TinyUSB, runtime | 56,204 |
| Benchmark main + test vectors | 28,127 |
| Image total | 234,487 of 2 MB |

The original "< 50 KB flash" target covered the model and our own code: 25,184 + 13,948 +
~4,300 wrapper = **~43 KB**. The TFLM runtime adds another ~107 KB on top. That fits easily in
2 MB, but the slides should say so explicitly.

## Latency (emulator, 125 MHz)

| Stage | Time per 10 s window |
|---|---:|
| Filtering + R-peak detection | 33.5–35.2 ms |
| HRV | 0.1–0.4 ms |
| Delineation + template correlation | 27.4–40.0 ms |
| Hjorth | 1.7 ms |
| Welch PSD (6 × 256-point float FFT) | 48.9 ms |
| **Feature extraction total** | **112–126 ms** |
| Gate + flags + INT8 input prep | 7.2 ms |
| **CNN inference (TFLM, CMSIS-NN, single core)** | **175.2 ms** |
| **Whole pipeline, clean window** | **294–308 ms** |
| Whole pipeline, rejected window (CNN skipped) | 118 ms |

That is about **3 % CPU time** for one 10 s window, on one core, leaving core 1 free for
acquisition. The Welch PSD is the largest feature stage. If needed, a real-input FFT would roughly
halve it.

**Limits of these numbers.** rp2040js counts Cortex-M0+ instruction cycles at 125 MHz, but it does
not model flash (XIP) cache misses or bus contention. Real-board times will therefore be somewhat
higher, especially for TFLM code that does not fit in the 16 KB XIP cache. Treat them as lower
bounds until `rak11300_pipeline.uf2` runs on hardware. The ~45 ms CNN figure in the slides was for
an ESP32 at 240 MHz with a different model. The 175 ms measured here is the RP2040 figure.

## Build decisions

- **Single-core CMSIS-NN.** pico-tflmicro hard-codes `#define TF_LITE_PICO_MULTICORE` (marked
  experimental upstream), which runs half of each `CONV_2D` on core 1. The build compiles a copy of
  that one file without the define: core 1 stays free for acquisition, and rp2040js cannot emulate
  the inter-core FIFO. Use `-DECG_TFLM_MULTICORE=ON` to re-enable it on hardware.
- **`-O3` for TFLM.** Upstream's intended `-Os` never applied, because its `COMPILE_FLAGS` is
  overwritten. Measured: `-Os` saves 21.8 KB of flash but makes inference 37 % slower (240.9 vs
  175.2 ms). `-O3` is the default; `-DECG_TFLM_OPT=Os` switches.
- **GCC 15 compatibility.** The vendored flatbuffers headers use `uint32_t` without
  `#include <cstdint>`, so the CMake file force-includes it instead of patching the checkout.

## Still needs hardware

- Real latency, including XIP effects, and power consumption.
- RAK11300 board header (SX1262 and ADC pins) and live acquisition (Phase 3).
