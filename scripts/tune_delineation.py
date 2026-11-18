"""Grid-search delineation thresholds on a TRAIN subset (fold 10 is never used).

    python scripts/tune_delineation.py <n_windows> qrs|p|t [param=value ...]

Prints the 6 best settings for the chosen stage (by MAE vs GE 12SL), holding
the extra param=value overrides fixed. Used to pick DEFAULT_PARAMS in delineation.py.
"""
import itertools
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from delineation import DEFAULT_PARAMS, FS, bandpass, delineate_template, median_beat  # noqa
from project import CACHE_FILE  # noqa
from ptbxl_plus import cache_ecg_ids, reference_intervals  # noqa
from reference_features import detect_r_peaks  # noqa

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
cache = np.load(CACHE_FILE, allow_pickle=True)
rng = np.random.default_rng(1)
sel = np.sort(rng.choice(len(cache["X_train"]), N, replace=False))
X = cache["X_train"][sel, :, 0]
ref = reference_intervals(cache_ecg_ids()[0][sel])

tmpls, rrs = [], []
for x in X:
    peaks, _ = detect_r_peaks(x)
    rr = float(np.mean(np.diff(peaks)) * 10.0) if len(peaks) > 1 else 800.0
    t, _, _ = median_beat(bandpass(x), peaks)
    tmpls.append(t); rrs.append(rr)


def evaluate(p):
    out = {"QRS": [], "PR": [], "QT": []}
    for t, rr in zip(tmpls, rrs):
        f = delineate_template(t, rr, p) if t is not None else None
        nan = np.nan
        if f is None:
            out["QRS"].append(nan); out["PR"].append(nan); out["QT"].append(nan); continue
        out["QRS"].append((f["qrs_off"] - f["qrs_on"]) * 10)
        out["PR"].append((f["qrs_on"] - f["p_on"]) * 10 if f["p_on"] is not None else nan)
        out["QT"].append((f["t_end"] - f["qrs_on"]) * 10 if f["t_end"] is not None else nan)
    res = {}
    for k, v in out.items():
        v = np.array(v); ok = ~np.isnan(v) & ref[k].notna().to_numpy()
        e = v[ok] - ref[k].to_numpy()[ok]
        res[k] = (np.mean(np.abs(e)), np.mean(e), np.corrcoef(v[ok], ref[k].to_numpy()[ok])[0, 1], ok.mean())
    return res


what = sys.argv[2] if len(sys.argv) > 2 else "qrs"
if what == "qrs":
    grid = [dict(k_on=a, k_off=b, qrs_gap=g) for a, b, g in itertools.product([0.03, 0.05, 0.08, 0.12, 0.2], [0.03, 0.05, 0.08, 0.12, 0.2], [1, 2, 3])]
    key = "QRS"
elif what == "p":
    grid = [dict(p_frac=a, p_far=b, p_min_mv=c) for a, b, c in itertools.product([0.2, 0.3, 0.5], [22, 25, 30], [0.01, 0.02, 0.03])]
    key = "PR"
else:
    grid = [dict(t_rr_frac=a, t_min_gap=b) for a, b in itertools.product([0.5, 0.6, 0.7, 0.8], [3, 5, 8])]
    key = "QT"
base = dict(arg.split("=") for arg in sys.argv[3:])
base = {k: (int(v) if v.isdigit() else float(v)) for k, v in base.items()}
results = []
for g in grid:
    p = replace(DEFAULT_PARAMS, **base, **g)
    r = evaluate(p)
    results.append((r[key][0], g, r))
results.sort(key=lambda t: t[0])
for mae, g, r in results[:6]:
    print(g, " | ".join(f"{k}: MAE {v[0]:.1f} bias {v[1]:+.1f} r {v[2]:.3f} cov {v[3] * 100:.0f}%" for k, v in r.items()))
