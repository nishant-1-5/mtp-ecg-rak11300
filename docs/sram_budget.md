# SRAM budget and memory strategy

Living document. Update the tables whenever a buffer is added, resized, or measured on hardware.
"Measured" means `sizeof()` or a real profile; "estimate" means not yet checked on the device.

## 1. Target hardware

The device is the **RAK11300** WisDuo module: a Raspberry Pi **RP2040** plus an SX1262 LoRa radio
([RAK datasheet](https://docs.rakwireless.com/product-categories/wisduo/rak11300-module/datasheet/)).
The project was previously described as "ESP32"; the old numbers do not carry over.

| Item | RP2040 / RAK11300 | Consequence |
|---|---|---|
| CPU | 2 × Cortex-M0+, up to 133 MHz (Pico SDK default 125 MHz) | README's "~45 ms at 240 MHz" was an ESP32 figure. Latency must be re-measured. |
| SRAM | 264 KB ([Arm/Keil device page](https://www.keil.arm.com/devices/rpi-rp2040/features/)) | The slides' "~520 KB SRAM" is the ESP32 figure. |
| Flash | 2 MB external QSPI, executed in place through a 16 KB cache | Flash size is not a constraint; flash *speed* can be (model weights are read through XIP). |
| FPU / DSP | None. Floats run through the optimised software routines in the boot ROM. No SIMD. | `esp-dsp` does not apply. Prefer integer math where it is free; avoid `double` entirely. |

## 2. Budget

264 KB is far more than the pipeline needs. We **keep the original < 30 KB SRAM target** for the
ECG pipeline (acquisition + features + CNN). That leaves the rest for the LoRaWAN stack, USB,
logging, and stacks on both cores, and it keeps the design portable to smaller parts.

| Buffer | Bytes | Status | Notes |
|---|---:|---|---|
| Acquisition ping-pong, 2 × 1000 × `int16_t` | 4,000 | design | One window is analysed while the next fills. |
| TFLM tensor arena (`kCnnArenaSize`) | 21,504 | measured on rp2040js: 20,788 used | 16,000 B activation head + 4,788 B persistent tail. |
| `ecg::Lead1Workspace` | 0 extra (5,780) | measured | Overlaid on the arena's activation head; see `docs/phase2_tflm.md`. |
| TFLM interpreter + op resolver objects | 568 | measured (`nm`) | |
| `ecg::Lead1Features` / `PipelineResult` | ~100 | measured | on the stack |
| Peak core-0 stack, whole pipeline | 1,636 | measured (painted stack) | |
| **Total** | **27,708** | | Under the 30 KB target; 33,488 B without the overlay. |

Flash: Lead I model 25,184 B, feature + decision code 13,948 B, TFLM wrapper ~4.3 KB, so about
43 KB of our own code and data. The TFLM + CMSIS-NN runtime adds 106,744 B. The whole test image
is 234 KB of the 2 MB flash (`docs/phase2_tflm.md`).

## 3. Decisions taken in the Phase 1 feature port

1. **Raw samples stay `int16_t` (2 B/sample), never `float` (4 B).**
   PTB-XL stores Lead I at 1000 LSB/mV in 16 bits, so `round(mV × 1000)` reproduces the training data
   exactly (max error 1.2e-4 LSB over the test set). The device ADC is 12-bit, so `int16_t` also holds
   its counts. The extractor takes `(int16_t* adc, mv_per_lsb)` and converts on the fly.
   Saving: 2,000 B per window buffer, 4,000 B for the ping-pong pair.

2. **One float work buffer, reused by every stage.**
   `Lead1Workspace::buf` is 1,030 floats (1,000 samples + 2 × 15 filtfilt padding = 4,120 B). The
   stages run in place, one after another:
   - odd-extended copy of Lead I → forward IIR pass → backward IIR pass (filtfilt);
   - squared first difference, written over the same buffer (writes trail reads);
   - 8-tap moving sum, in place with a 4-value history;
   - R-peak search reads it;
   - the window is filtered again (0.5–40 Hz filtfilt) into the same buffer for delineation; the
     median beat goes to `templ[101]`, built one sample at a time through `scratch[32]`, so no
     beats × samples matrix is ever stored (+596 B in total with `r_pos[32]`);
   - Welch: `buf[0..511]` becomes the 256-point complex FFT scratch and `buf[512..640]` the PSD accumulator.

   A straightforward port (separate float input copy, filtered, squared-derivative, integrated and
   FFT arrays) would need about 18 KB.

3. **Peak candidates: 1,000 B, no sort index.**
   Strict local maxima in 999 samples are at least 2 apart, so 500 `uint16_t` always suffice (no
   overflow path). The distance filter keeps its state in bits 14–15 of each entry and picks the
   tallest remaining candidate by linear search instead of `argsort`, which avoids a second 1 KB
   index array. Cost is O(K²) comparisons, K ≤ 500.

4. **Hjorth complexity uses exact `int64_t` moments of the ADC counts.** No float buffer, no
   precision loss; the feature is scale-invariant so `mv_per_lsb` is not needed.

5. **No heap, no STL.** The workspace is a caller-owned static object, so the linker map shows the full
   cost and fragmentation cannot happen.

6. **float32, not fixed point (for now).** The float build matches the float64 Python reference on
   21,387 of 21,388 PTB-XL windows (see `docs/phase1_feature_port.md`). Fixed point would cut CPU
   time on the M0+ but needs its own validation; revisit only if on-device timing requires it.

## 4. Further savings, if the target tightens

| Option | Saves | Cost |
|---|---:|---|
| Single 1000-sample ring buffer + small ISR FIFO instead of ping-pong (core 1 does acquisition) | ~2,000 B | Analysis must finish, or copy out, before the oldest samples are overwritten. |
| Overlay `Lead1Workspace` with the non-persistent part of the tensor arena (features and CNN never run at the same time) | up to 5,780 B | Depends on TFLM's arena layout; fragile across TFLM versions. |
| Quantise CNN input straight from the `int16_t` window into the arena's `int8` input tensor | avoids a 4 KB float copy (Lead I) | None. Implemented: `ecg::PrepareCnnInput` writes into a caller-supplied buffer. |

## 5. Measurements and open items

Measured with the Arm GNU Toolchain 15.3 build of `firmware/rak11300_bench` (`-O3`, Release):

| Item | Value |
|---|---:|
| Feature extraction + delineation code (`ecg_features.cpp`) | 12,768 B flash |
| Gate / flags / CNN input prep (`ecg_pipeline.cpp`) | 1,376 B flash |
| Lead I INT8 model (`g_model_len`) | 25,184 B flash |
| **ECG code + model** | **39,328 B flash** (< 50 KB target; the TFLM runtime comes on top, Phase 2) |
| `Lead1Workspace` | 5,780 B .bss |
| Double-precision arithmetic in the ECG code | none (`nm -u` lists only `__aeabi_f*`) |

Latency on the rp2040js emulator at 125 MHz (`docs/phase2_tflm.md`): features 112–126 ms, CNN
175 ms, whole pipeline 294–308 ms per 10 s window (~3 % of one core).

Open:

- **Latency on the board:** flash `rak11300_pipeline.uf2` and read the times over USB serial
  (`firmware/rak11300_bench/README.md`). The emulator does not model XIP cache misses, so expect
  somewhat higher numbers on hardware.
- **XIP:** code runs from flash through a 16 KB cache. If the board numbers are worse than expected,
  move the hot loops (`FilterPass`, `Fft256`, CMSIS-NN conv) to RAM with `__not_in_flash_func`.
