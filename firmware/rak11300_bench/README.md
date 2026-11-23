# RAK11300 firmware (benchmark, full pipeline, emulator)

| Target | Contents |
|---|---|
| `rak11300_bench.uf2` | Feature extraction only, timing for each stage |
| `rak11300_pipeline.uf2` | Full pipeline: features, gate, flags, INT8 CNN under TFLM, decisions. Reports arena use, peak stack and latency. |
| `rak11300_validation.uf2` | The same as `rak11300_pipeline`, on 100 random fold-10 windows. Built only if `build/bench_vectors_validation.h` exists. |

Every target checks its own results against the Python reference baked into the header and ends
with `=== PASS` or `=== FAIL`. Results without a board are in `docs/phase2_tflm.md`.

## Run without a board (rp2040js emulator)

```powershell
git clone --depth 1 https://github.com/wokwi/rp2040js.git .toolchain/rp2040js
cd .toolchain/rp2040js; npm install; cd ../..
.toolchain/rp2040js/node_modules/.bin/tsx --tsconfig .toolchain/rp2040js/tsconfig.json `
    firmware/rak11300_bench/emulate/run.ts firmware/rak11300_bench/build/rak11300_pipeline.uf2 1500
```

The emulator runs at about a third of real time on this PC (≈1 min for the 6-window pipeline run).

---

The rest of this file covers the feature-stage benchmark. The pipeline targets build the same way
and also need `.toolchain/pico-tflmicro` (raspberrypi/pico-tflmicro, `main`).

# Feature-stage benchmark

This firmware runs the Lead I pipeline (`firmware/ecg_features`, `firmware/ecg_pipeline`) on the
RP2040, on 6 embedded PTB-XL windows (`bench_vectors.h`). For each window it:

- times every stage of `ExtractLead1Features`, averaged over 20 runs;
- times the gate, rhythm flags and INT8 CNN input preparation;
- checks features, R-peak count, quality bits and flags against the Python reference values baked
  into the header, then prints `MATCH` or `MISMATCH`.

It does not include the CNN. TFLM is Phase 2.

## Toolchain (portable, inside the repo, git-ignored)

```
.toolchain/
  bin/ ...                      Arm GNU Toolchain 15.3.rel1 (arm-none-eabi), zip from gitlab.arm.com
  cmake-3.31.6-windows-x86_64/  CMake
  ninja/ninja.exe               Ninja 1.12.1
  pico-sdk/                     raspberrypi/pico-sdk tag 2.2.0 (+ lib/tinyusb submodule)
  picotool/picotool/            prebuilt picotool 2.2.0-a4 (raspberrypi/pico-sdk-tools v2.2.0-4)
```

Nothing is installed system-wide. To remove the toolchain, delete `.toolchain/`.

## Build

```powershell
python scripts/make_bench_vectors.py          # only if the reference or thresholds changed
powershell -ExecutionPolicy Bypass -File firmware/rak11300_bench/build.ps1
```

Output: `firmware/rak11300_bench/build/rak11300_bench.uf2`.

The board is configured as `PICO_BOARD=pico`. The RAK11300 is an RP2040 with 2 MB of QSPI flash,
and this benchmark touches no board-specific pins. Phase 2 will need a RAK11300 board header for the
SX1262 and the ADC pins.

## Flash and read the results

1. Hold BOOT (BOOTSEL) on the RAK11300 carrier while connecting USB, or while pressing RESET. A
   drive named `RPI-RP2` appears.
2. Copy `rak11300_bench.uf2` onto that drive. The board reboots into the benchmark.
3. Open the new USB serial (CDC) port at any baud rate, e.g. PuTTY or
   `python -m serial.tools.miniterm COMx`. The firmware waits 3 s for the port, prints once, then
   idles. Press RESET to run it again. The same text also goes to UART0 (GP0/GP1, 115200 8N1).

Expected shape of the output:

```
clk_sys = 125000000 Hz, sizeof(Lead1Workspace) = 5780 B, ...
[0] NORM, regular: MATCH (peaks 13, quality 0x00, flags 0x00)
  features total: ... ms
    filter + R peaks     ... ms
    ...
=== PASS: 6 of 6 windows match the Python reference ===
```

Record the numbers in `docs/sram_budget.md` (§5, latency).

## Measured on the host build (no board needed)

| Item | Value | Source |
|---|---:|---|
| `ecg_features.cpp` code + const data | 12,768 B | `arm-none-eabi-size` on the object |
| `ecg_pipeline.cpp` code | 1,376 B | same |
| `Lead1Workspace` (.bss) | 5,780 B | `nm` / `sizeof` |
| Double-precision calls in the ECG code | none | `arm-none-eabi-nm -u`: only `__aeabi_f*`, which the SDK routes to the RP2040 ROM float routines |
| Whole benchmark image | 67,080 B text, 10,872 B bss | includes USB stack, printf and 12 KB of test vectors |
