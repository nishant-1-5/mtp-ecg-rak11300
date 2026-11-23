"""Flash / RAM use per input file or library, from a GNU ld .map file.

    python scripts/map_sizes.py firmware/rak11300_bench/build/rak11300_pipeline.elf.map
"""
import re
import sys
from collections import defaultdict

FLASH = (".text", ".rodata", ".ARM.extab", ".ARM.exidx", ".binary_info", ".boot2")
RAM = (".data", ".bss", ".uninitialized_data", ".scratch_x", ".scratch_y")
# " .text.foo   0x10001234   0x40 path/to/lib.a(obj.o)"  (name may be on its own line)
LINE = re.compile(r"^\s(\.[\w.$]+)?\s*0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)$")


def group(path):
    path = path.replace("\\", "/")
    if "libpico-tflmicro.a" in path or "pico-tflmicro.dir" in path:
        return "TFLM + CMSIS-NN"
    if "ecg_cnn" in path:
        return "ecg_cnn (wrapper + model array)"
    if "ecg_features" in path or "ecg_pipeline" in path:
        return "ecg_features + ecg_pipeline"
    if "main.cpp" in path:
        return "benchmark main + test vectors"
    if any(k in path for k in ("libstdc++", "libsupc++", "libgcc", "libc.a", "libm.a", "libc_nano", "libnosys")):
        return "C/C++ runtime (newlib, libstdc++, libgcc)"
    if "tinyusb" in path or "pico_stdio_usb" in path:
        return "USB stdio (TinyUSB)"
    return "pico-sdk (other)"


def main(map_path):
    flash, ram = defaultdict(int), defaultdict(int)
    pending = None
    with open(map_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("Linker script and memory map"):
                pending = ""
                continue
            if pending is None:
                continue
            m = re.match(r"^\s(\.[\w.$]+)\s*$", line)
            if m:  # section name alone; address/size/file on the next line
                pending = m.group(1)
                continue
            m = LINE.match(line)
            if not m:
                continue
            sec = m.group(1) or pending or ""
            pending = ""
            size, obj = int(m.group(3), 16), m.group(4)
            if size == 0 or "(" not in obj and not obj.endswith((".o", ".obj")):
                continue
            g = group(obj)
            if sec.startswith(FLASH):
                flash[g] += size
            elif sec.startswith(RAM):
                ram[g] += size
    print(f"{'component':<44}{'flash B':>10}{'RAM B':>10}")
    for g in sorted(set(flash) | set(ram), key=lambda k: -flash[k]):
        print(f"{g:<44}{flash[g]:>10,}{ram[g]:>10,}")
    print(f"{'total':<44}{sum(flash.values()):>10,}{sum(ram.values()):>10,}")


if __name__ == "__main__":
    main(sys.argv[1])
