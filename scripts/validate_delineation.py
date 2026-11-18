"""Validate Lead I PR/QRS/QT against GE 12SL (PTB-XL+) and against the labels.

    python scripts/validate_delineation.py [--split train|test] [--limit N]

Reports, per interval: coverage, bias and MAE vs 12SL, Pearson r, and how well
the interval separates CD from the rest (single-feature AUC), next to the old
fix_cache.py value from the cache. Writes results/features/delineation_<split>.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delineation import DEFAULT_PARAMS, lead1_intervals  # noqa: E402
from project import CACHE_FILE, FEATURE_NAMES, FEATURE_RESULTS_DIR  # noqa: E402
from ptbxl_plus import cache_ecg_ids, reference_intervals  # noqa: E402
from reference_features import detect_r_peaks  # noqa: E402


def compute(lead1_all, params=DEFAULT_PARAMS):
    rows = []
    for x in lead1_all:
        peaks, _ = detect_r_peaks(x)
        rr = np.diff(peaks) * 10.0
        rr_ms = float(np.mean(rr)) if len(rr) else 800.0
        pr, qrs, qt, qtc, nb = lead1_intervals(x, peaks, rr_ms, params)
        rows.append((pr, qrs, qt, qtc, nb))
    return pd.DataFrame(rows, columns=["PR", "QRS", "QT", "QTc", "n_beats"])


def report(ours, ref, old, y, class_names):
    cd = y[:, list(class_names).index("CD")]
    lines = [f"{'interval':<6}{'coverage':>9}{'bias':>8}{'MAE':>7}{'r':>7}"
             f"{'AUC(CD) ours':>14}{'12SL':>7}{'old':>7}"]
    for k in ["QRS", "PR", "QT"]:
        ok = ours[k].notna() & ref[k].notna()
        err = ours[k][ok] - ref[k][ok]
        r = np.corrcoef(ours[k][ok], ref[k][ok])[0, 1]

        def auc(v):
            m = ~np.isnan(v)
            return roc_auc_score(cd[m], v[m])

        lines.append(f"{k:<6}{ours[k].notna().mean() * 100:>8.1f}%{err.mean():>8.1f}"
                     f"{err.abs().mean():>7.1f}{r:>7.3f}{auc(ours[k].to_numpy()):>14.3f}"
                     f"{auc(ref[k].to_numpy()):>7.3f}{auc(old[k]):>7.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = np.load(CACHE_FILE, allow_pickle=True)
    X = cache[f"X_{args.split}"][:, :, 0]
    y = cache[f"y_{args.split}"]
    feats = cache[f"X_feat_{args.split}"]
    ids = cache_ecg_ids()[0 if args.split == "train" else 1]
    if args.limit:
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(len(X), args.limit, replace=False))
        X, y, feats, ids = X[sel], y[sel], feats[sel], ids[sel]

    ref = reference_intervals(ids)
    ours = compute(X)
    old = {k: feats[:, FEATURE_NAMES.index(f"Mean_{k}")] for k in ["QRS", "PR", "QT"]}
    print(f"split={args.split}, n={len(X)}")
    print(report(ours, ref, old, y, cache["class_names"]))

    out = FEATURE_RESULTS_DIR / f"delineation_{args.split}.csv"
    pd.concat([ours.add_prefix("lead1_"), ref.add_prefix("12sl_")], axis=1).assign(ecg_id=ids).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
