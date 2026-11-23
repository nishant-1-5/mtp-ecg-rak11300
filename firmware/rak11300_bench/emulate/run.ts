// Runs a firmware .uf2 on the rp2040js emulator (https://github.com/wokwi/rp2040js)
// and prints UART0 until the benchmark's final "=== PASS/FAIL" line.
//
//   .toolchain/rp2040js/node_modules/.bin/tsx firmware/rak11300_bench/emulate/run.ts <image.uf2> [timeout_s]
//
// Timing: the emulator advances its clock by the cycle count it models for each
// Cortex-M0+ instruction at 125 MHz. It does not model flash (XIP) wait states,
// cache misses or bus contention, so on-board times will be somewhat higher.
import { bootromB1 } from '../../../.toolchain/rp2040js/demo/bootrom.js';
import { loadUF2 } from '../../../.toolchain/rp2040js/demo/load-flash.js';
import { Simulator } from '../../../.toolchain/rp2040js/src/simulator.js';
import { ConsoleLogger, LogLevel } from '../../../.toolchain/rp2040js/src/utils/logging.js';

const image = process.argv[2];
const timeoutS = Number(process.argv[3] ?? 900);
if (!image) {
  console.error('usage: run.ts <image.uf2> [timeout_s]');
  process.exit(2);
}

const simulator = new Simulator();
const mcu = simulator.rp2040;
mcu.logger = new ConsoleLogger(LogLevel.Error, true);  // hide per-instruction debug noise (SEV, unimplemented QSPI regs)
mcu.loadBootrom(bootromB1);
loadUF2(image, mcu);

let line = '';
let result: number | null = null;
mcu.uart[0].onByte = (value: number) => {
  const ch = String.fromCharCode(value);
  process.stdout.write(ch);
  if (ch === '\n') {
    if (line.startsWith('=== PASS')) result = 0;
    else if (line.startsWith('=== FAIL') || line.includes('FAILED')) result = 1;
    line = '';
  } else if (ch !== '\r') {
    line += ch;
  }
};

const started = Date.now();
mcu.core.PC = 0x10000000;
simulator.execute();
const poll = setInterval(() => {
  if (result !== null || Date.now() - started > timeoutS * 1000) {
    clearInterval(poll);
    simulator.stop();
    if (result === null) console.error(`\n[emulator] timeout after ${timeoutS} s`);
    console.error(`[emulator] wall time ${((Date.now() - started) / 1000).toFixed(0)} s`);
    process.exit(result ?? 3);
  }
}, 200);
