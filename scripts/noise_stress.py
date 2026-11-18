"""Add real ECG noise (MIT-BIH Noise Stress Test Database) to clean PTB-XL windows.

Download once (~6 MB) into the ignored dataset/ folder:
    python -c "import wfdb; wfdb.dl_database('nstdb', 'dataset/nstdb', records=['bw', 'em', 'ma'])"

bw = baseline wander, em = electrode motion, ma = muscle (EMG) artifact.
Each 30 min record is resampled 360 -> 100 Hz. The first half is reserved for
threshold selection, the second half for evaluation, so the two never share
noise segments.

Citation: Moody GB, Muldrow WE, Mark RG. A noise stress test for arrhythmia
detectors. Computers in Cardiology 1984; 11:381-384.
"""
from functools import lru_cache

import numpy as np
import scipy.signal as signal

from project import DATASET_DIR

NSTDB_DIR = DATASET_DIR / "nstdb"
NOISE_TYPES = ("bw", "em", "ma")


@lru_cache(maxsize=None)
def noise_track(kind):
    import wfdb
    rec = wfdb.rdrecord(str(NSTDB_DIR / kind))
    x = rec.p_signal[:, 0]  # channel 1
    return signal.resample_poly(x, 5, 18)  # 360 Hz -> 100 Hz


def add_noise(clean, kind, snr_db, rng, half):
    """clean: (N, 1000) mV. Noise RMS is set so 10*log10(P_signal / P_noise) = snr_db,
    with P_signal the power of the de-meaned clean window. half: 0 = tuning, 1 = eval."""
    track = noise_track(kind)
    seg_len = clean.shape[1]
    lo = 0 if half == 0 else len(track) // 2
    hi = (len(track) // 2 if half == 0 else len(track)) - seg_len
    out = np.empty_like(clean, dtype=np.float64)
    for i, x in enumerate(clean):
        s = int(rng.integers(lo, hi))
        n = track[s:s + seg_len] - track[s:s + seg_len].mean()
        p_sig = np.var(x)
        p_noise = np.var(n) + 1e-12
        out[i] = x + n * np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return out
