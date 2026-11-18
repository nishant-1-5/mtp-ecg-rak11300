"""Writes the v2 Lead I feature table into the cache as X_feat_v2_train / X_feat_v2_test.

    python scripts/build_lead1_features.py

24 columns, same names/order as project.FEATURE_NAMES[:24]:
  0-17  reference_features.extract_lead1_features (PR/QRS/QT/QTc from delineation.py)
  18-23 Lead I mean, std, skew, kurtosis, peak-to-peak, IQR (unchanged from fix_cache.py)
Intervals are NaN when not measurable (XGBoost and sklearn forests accept NaN).
The 36-wide X_feat_* used by the 3-lead baseline is left untouched.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project import CACHE_FILE  # noqa: E402
from reference_features import extract_lead1_features  # noqa: E402


def lead1_row(lead1):
    f, _ = extract_lead1_features(lead1)
    stats = [np.mean(lead1), np.std(lead1), skew(lead1), kurtosis(lead1), np.ptp(lead1),
             np.percentile(lead1, 75) - np.percentile(lead1, 25)]
    return np.concatenate([f, stats]).astype(np.float32)


def table(X):
    with ProcessPoolExecutor() as ex:
        return np.stack(list(ex.map(lead1_row, X[:, :, 0], chunksize=256)))


def main():
    t0 = time.time()
    cache = dict(np.load(CACHE_FILE, allow_pickle=True))
    for split in ("train", "test"):
        cache[f"X_feat_v2_{split}"] = table(cache[f"X_{split}"])
        nan = np.isnan(cache[f"X_feat_v2_{split}"]).mean(axis=0)
        print(f"{split}: {cache[f'X_feat_v2_{split}'].shape}, NaN fraction PR/QRS/QT/QTc = "
              f"{nan[12]:.3f}/{nan[13]:.3f}/{nan[14]:.3f}/{nan[15]:.3f}")
    np.savez_compressed(CACHE_FILE, **cache)
    print(f"updated {CACHE_FILE} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
