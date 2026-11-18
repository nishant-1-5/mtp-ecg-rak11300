"""Parity test: firmware/ecg_pipeline vs. scripts/orchestrator.py.

Windows: PTB-XL fold 10 (clean) plus NSTDB-noised copies, so that every gate
branch is exercised. Two passes of the host binary:
  pass 1: C++ gate / flags / INT8 inputs (CNN callback returns zeros)
  python: TFLite INT8 model on the C++ inputs -> probabilities
  pass 2: C++ replays those probabilities -> class decisions
Each C++ output is compared with the Python reference on the same windows.

    python scripts/compare_pipeline.py --exe build/pipeline_host.exe
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestrator as O  # noqa: E402
import project as P  # noqa: E402
from noise_stress import add_noise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=str(P.ROOT / "build" / "pipeline_host.exe"))
    ap.add_argument("--workdir", default=str(P.ROOT / "build"))
    args = ap.parse_args()
    wd = Path(args.workdir)

    p = json.loads(O.PARAMS_JSON.read_text(encoding="utf-8"))
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    clean = cache["X_test"][:, :, 0].astype(np.float64)
    rng = np.random.default_rng(123)
    sub = rng.choice(len(clean), 400, replace=False)
    parts = [clean] + [add_noise(clean[sub], k, s, rng, half=1)
                       for k, s in [("em", 0), ("em", -6), ("ma", -6), ("bw", -6)]]
    flat = np.zeros((20, 1000))  # lead-off
    flat[:, ::7] = 0.0005
    parts.append(flat)
    adc = O.to_adc(np.concatenate(parts))
    n = len(adc)

    win = wd / "pipeline_windows.bin"
    with open(win, "wb") as f:
        np.array([n, 1000], "<i4").tofile(f)
        adc.astype("<i2").tofile(f)

    cnn = O.Int8Cnn()
    scale, zp = f"{cnn.scale:.9g}", str(cnn.zero_point)
    csv1, dump1 = wd / "pipeline_pass1.csv", wd / "pipeline_inputs.bin"
    subprocess.run([args.exe, win, csv1, dump1, scale, zp], check=True)
    c1 = pd.read_csv(csv1)
    cpp_inputs = np.fromfile(dump1, dtype=np.int8).reshape(-1, 1000)
    probs_cpp_inputs = cnn.run(cpp_inputs)
    probs_bin = wd / "pipeline_probs.bin"
    probs_cpp_inputs.astype("<f4").tofile(probs_bin)
    csv2, dump2 = wd / "pipeline_pass2.csv", wd / "pipeline_inputs2.bin"
    subprocess.run([args.exe, win, csv2, dump2, scale, zp, probs_bin], check=True)
    c2 = pd.read_csv(csv2)

    ref, ref_inputs = O.run_pipeline(adc, p, cnn)
    passed = ref.quality.to_numpy() == 0

    print(f"{n} windows ({len(clean)} clean, {n - len(clean)} noisy/flat); "
          f"reference gate pass {passed.mean() * 100:.1f} %")
    same_q = (c1.quality.to_numpy() == ref.quality.to_numpy())
    same_f = (c1.rhythm_flags.to_numpy() == ref.rhythm_flags.to_numpy())
    print(f"quality bitmask identical: {same_q.mean() * 100:.2f} %  ({(~same_q).sum()} differ)")
    print(f"rhythm flags identical:    {same_f.mean() * 100:.2f} %  ({(~same_f).sum()} differ)")

    ran = c1.cnn_ran.to_numpy() == 1
    assert ran.sum() == len(cpp_inputs)
    both = ran & passed
    d = np.abs(cpp_inputs[both[ran]].astype(int) - ref_inputs[both].astype(int))
    print(f"INT8 CNN input: {both.sum()} windows, exact {np.mean(d == 0) * 100:.4f} % of samples, "
          f"max |diff| {d.max()} LSB, windows with any diff {np.mean(d.max(axis=1) > 0) * 100:.2f} %")

    pcols = [f"p_{c}" for c in O.CLASS_NAMES]
    dp = np.abs(c2.loc[both, pcols].to_numpy() - ref.loc[both, pcols].to_numpy())
    same_dec = c2.classes.to_numpy()[both] == ref.decision.to_numpy()[both]
    print(f"probabilities (C++ inputs vs Python inputs): max |diff| {dp.max():.4f}")
    print(f"class decisions identical: {same_dec.mean() * 100:.2f} %  ({(~same_dec).sum()} differ)")
    tc = np.abs(c1.template_corr.to_numpy() - ref.template_corr.to_numpy())
    print(f"template_corr max |diff|: {np.nanmax(tc):.2e}")
    if (~same_q).any():
        i = np.where(~same_q)[0][:5]
        print("quality mismatches (idx, cpp, ref, ref template_corr):",
              list(zip(i.tolist(), c1.quality.to_numpy()[i].tolist(), ref.quality.to_numpy()[i].tolist(),
                       ref.template_corr.to_numpy()[i].round(4).tolist())))
    return 0 if same_q.all() and same_f.all() and same_dec.all() else 1


if __name__ == "__main__":
    sys.exit(main())
