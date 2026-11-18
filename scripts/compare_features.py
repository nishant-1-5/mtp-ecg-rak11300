"""Parity test: firmware Lead I feature extractor vs. the Python reference.

1. Takes Lead I from ptbxl_preprocessed_cache.npz and writes it as int16 ADC
   counts (PTB-XL is stored at 1000 LSB/mV, so the round trip is lossless).
2. Runs the host build of firmware/ecg_features on that file.
3. Recomputes the features with scripts/reference_features.py and compares.

Usage (from the repo root):
    python scripts/compare_features.py --exe build/lead1_host.exe [--split test] [--limit N]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project import CACHE_FILE  # noqa: E402
from reference_features import CACHE_COLUMNS, FEATURE_NAMES, extract_lead1_features  # noqa: E402

LSB_PER_MV = 1000  # PTB-XL header gain
REL_TOL, ABS_TOL = 1e-3, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_FILE))
    ap.add_argument("--exe", default=os.path.join("build", "lead1_host.exe"))
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--workdir", default="build")
    args = ap.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    lead1 = cache[f"X_{args.split}"][:, :, 0]
    cached_feats = cache[f"X_feat_v2_{args.split}"]
    if args.limit:
        lead1, cached_feats = lead1[: args.limit], cached_feats[: args.limit]
    n = len(lead1)

    counts = np.round(lead1.astype(np.float64) * LSB_PER_MV)
    assert np.abs(counts - lead1 * LSB_PER_MV).max() < 1e-2, "not on the PTB-XL ADC grid"
    assert np.abs(counts).max() < 32768
    os.makedirs(args.workdir, exist_ok=True)
    bin_path = os.path.join(args.workdir, f"lead1_{args.split}.bin")
    csv_path = os.path.join(args.workdir, f"lead1_{args.split}_cpp.csv")
    with open(bin_path, "wb") as f:
        np.array([n, lead1.shape[1]], dtype="<i4").tofile(f)
        counts.astype("<i2").tofile(f)

    subprocess.run([args.exe, bin_path, csv_path, str(1.0 / LSB_PER_MV)], check=True)
    cpp = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    cpp_feats, cpp_peaks = cpp[:, :-2], cpp[:, -2].astype(int)

    ref = np.empty((n, len(FEATURE_NAMES)))
    ref_peaks = np.empty(n, dtype=int)
    for i in range(n):
        ref[i], ref_peaks[i] = extract_lead1_features(lead1[i])

    # The reference must reproduce what the models were trained on.
    cache_err = np.nanmax(np.abs(ref - cached_feats[:, CACHE_COLUMNS]), axis=0)
    print(f"reference vs cached X_feat_v2_{args.split}: max |diff| per feature")
    print("  " + ", ".join(f"{k}={v:.2g}" for k, v in zip(FEATURE_NAMES, cache_err)))

    same_peaks = cpp_peaks == ref_peaks
    print(f"\n{n} windows; identical R-peak count in {same_peaks.mean() * 100:.2f}%")
    print(f"{'feature':<18}{'max abs diff':>14}{'max rel diff':>14}{'within tol %':>14}{'NaN mismatch':>14}")
    worst = []
    for k, name in enumerate(FEATURE_NAMES):
        a, b = cpp_feats[:, k], ref[:, k]
        both_nan = np.isnan(a) & np.isnan(b)
        nan_mismatch = np.isnan(a) ^ np.isnan(b)
        d = np.where(both_nan | nan_mismatch, 0.0, np.abs(a - b))
        rel = d / np.maximum(np.abs(np.nan_to_num(b)), 1e-12)
        ok = ~nan_mismatch & ((d <= ABS_TOL) | (rel <= REL_TOL))
        worst.append(np.where(~ok)[0])
        print(f"{name:<18}{d.max():>14.3g}{rel.max():>14.3g}{ok.mean() * 100:>14.2f}{nan_mismatch.sum():>14d}")
    bad = np.unique(np.concatenate(worst))
    print(f"\nwindows with any feature out of tolerance: {len(bad)} / {n}")
    if len(bad):
        print("first few:", bad[:10].tolist())
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
