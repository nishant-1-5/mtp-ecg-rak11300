"""Lead I QRS / PR / QT delineation on a median beat (100 Hz).

Replaces the PR/QT/QRS code in fix_cache.py, which measured the R wave itself
(median QT 0.8 ms, PR 10 ms) and an energy-envelope width instead of QRS
duration. Validated against GE 12SL global intervals from PTB-XL+
(scripts/validate_delineation.py, docs/delineation.md).

Steps (each is a plain loop over <= 1000 samples, written for a direct C++ port):
  1. Band-pass the raw window 0.5-40 Hz (zero-phase, Butterworth order 2).
  2. Refine each Pan-Tompkins peak to the R wave: max |x - window mean| within
     +-80 ms.
  3. Median beat: sample-wise median over beats that fit in [-300, +700] ms.
  4. QRS onset/offset: outermost samples around R whose |slope| reaches a
     fraction of the maximum QRS slope (linearly interpolated).
  5. T wave: largest deviation from the PR-segment level after the QRS; T end by
     the tangent method on the steepest point of the descending limb.
  6. P wave: largest deviation 40-250 ms before QRS onset; onset where the
     deviation falls below a fraction of the P amplitude.
"""
from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

FS = 100
PRE = 30    # median-beat samples before R (300 ms)
POST = 70   # samples after R (700 ms)


@dataclass(frozen=True)
class DelineationParams:
    r_search: int = 8          # +-80 ms around the Pan-Tompkins peak
    qrs_half_span: int = 12    # QRS edges searched within +-120 ms of R
    k_on: float = 0.08         # onset slope threshold (fraction of max QRS slope)
    k_off: float = 0.05        # offset slope threshold
    qrs_gap: int = 2           # sub-threshold run (samples) that ends the QRS
    t_min_gap: int = 5         # T search starts 50 ms after QRS offset
    t_rr_frac: float = 0.5     # ... and ends at R + 0.5 * RR
    p_far: int = 25            # P search window: 250 ms ...
    p_near: int = 4            # ... to 40 ms before QRS onset
    p_frac: float = 0.5        # P onset where deviation < p_frac * P amplitude
    p_min_mv: float = 0.03     # smaller P waves are treated as absent
    min_beats: int = 3


DEFAULT_PARAMS = DelineationParams()
_BP = signal.butter(2, [0.5 / (FS / 2), 40 / (FS / 2)], btype="bandpass")


def bandpass(lead1):
    return signal.filtfilt(*_BP, lead1)


def median_beat(x, peaks, params=DEFAULT_PARAMS):
    """Returns (template[PRE + POST + 1], r_positions, n_beats)."""
    n = len(x)
    ref = float(np.mean(x))  # ~0 after the 0.5 Hz high-pass; mean avoids a sort on the MCU
    r_pos = []
    for p in peaks:
        lo, hi = max(0, p - params.r_search), min(n - 1, p + params.r_search)
        r_pos.append(lo + int(np.argmax(np.abs(x[lo:hi + 1] - ref))))
    usable = [r for r in r_pos if r - PRE >= 0 and r + POST < n]
    if len(usable) < params.min_beats:
        return None, r_pos, len(usable)
    beats = np.stack([x[r - PRE:r + POST + 1] for r in usable])
    return np.median(beats, axis=0), r_pos, len(usable)


def _crossing(a, i_lo, i_hi, level):
    """Fractional index between i_lo and i_hi where a crosses level (linear, clamped)."""
    d = a[i_hi] - a[i_lo]
    frac = (level - a[i_lo]) / d if d != 0 else 0.0
    return i_lo + min(1.0, max(0.0, frac)) * (i_hi - i_lo)


def _walk(slope, start, limit, step, th, gap):
    """Walks from start towards limit; returns the last index with slope >= th
    before `gap` consecutive samples below th (QRS edge in whole samples)."""
    last = start
    below = 0
    j = start
    while j != limit:
        j += step
        if slope[j] >= th:
            last, below = j, 0
        else:
            below += 1
            if below >= gap:
                break
    return last


def delineate_template(t, rr_ms, params=DEFAULT_PARAMS):
    """Fiducials (fractional sample indices into t; R is at PRE) and amplitudes."""
    r = PRE
    m = len(t)
    slope = np.zeros(m)
    slope[1:-1] = np.abs(t[2:] - t[:-2]) / 2.0
    lo, hi = r - params.qrs_half_span, r + params.qrs_half_span

    # 4. QRS edges: walk outwards from the steepest point on each side of R
    # until |slope| drops below k * (max QRS slope).
    left_max = lo + int(np.argmax(slope[lo:r + 1]))
    right_max = r + int(np.argmax(slope[r:hi + 1]))
    s_max = max(slope[left_max], slope[right_max])
    if s_max <= 0:
        return None
    th_on = params.k_on * s_max
    j = _walk(slope, left_max, lo, -1, th_on, params.qrs_gap)
    qrs_on = _crossing(slope, j - 1, j, th_on) if j > 0 else float(j)
    th_off = params.k_off * s_max
    j = _walk(slope, right_max, hi, +1, th_off, params.qrs_gap)
    qrs_off = _crossing(slope, j, j + 1, th_off) if j + 1 < m else float(j)

    # PR-segment (isoelectric) level: mean of the 30 ms before QRS onset.
    i_on = int(np.floor(qrs_on))
    iso = float(np.mean(t[max(0, i_on - 3):max(1, i_on)]))

    # 5. T wave.
    t_lo = int(np.ceil(qrs_off)) + params.t_min_gap
    t_hi = min(m - 2, r + int(round(params.t_rr_frac * rr_ms / (1000 / FS))))
    t_end = None
    t_amp = 0.0
    if t_hi > t_lo + 2:
        dev = t[t_lo:t_hi + 1] - iso
        tp = t_lo + int(np.argmax(np.abs(dev)))
        t_amp = float(t[tp] - iso)
        sign = 1.0 if t_amp > 0 else -1.0
        # steepest descent back towards iso after the T peak
        best, best_k = 0.0, None
        for k in range(tp, min(m - 1, tp + 25)):
            s = -sign * (t[k + 1] - t[k - 1]) / 2.0
            if s > best:
                best, best_k = s, k
        if best_k is not None and best > 0:
            t_end = best_k + (t[best_k] - iso) * sign / best
            t_end = float(np.clip(t_end, tp, m - 1))

    # 6. P wave.
    p_on = None
    p_amp = 0.0
    p_lo, p_hi = max(1, i_on - params.p_far), i_on - params.p_near
    if p_hi > p_lo:
        dev = np.abs(t[p_lo:p_hi + 1] - iso)
        pp = p_lo + int(np.argmax(dev))
        p_amp = float(dev[pp - p_lo])
        if p_amp >= params.p_min_mv:
            level = params.p_frac * p_amp
            k = pp
            while k > p_lo and abs(t[k] - iso) >= level:
                k -= 1
            a = np.abs(t - iso)
            p_on = _crossing(a, k, k + 1, level)

    return {"qrs_on": qrs_on, "qrs_off": qrs_off, "t_end": t_end, "p_on": p_on,
            "iso": iso, "t_amp": t_amp, "p_amp": p_amp}


def lead1_intervals(lead1, peaks, rr_ms, params=DEFAULT_PARAMS):
    """Interval features in ms: (pr, qrs, qt, qtc, n_beats). NaN when not measurable."""
    x = bandpass(np.asarray(lead1, dtype=np.float64))
    tmpl, _, n_beats = median_beat(x, peaks, params)
    nan = float("nan")
    if tmpl is None:
        return nan, nan, nan, nan, n_beats
    f = delineate_template(tmpl, rr_ms, params)
    if f is None:
        return nan, nan, nan, nan, n_beats
    ms = 1000.0 / FS
    qrs = (f["qrs_off"] - f["qrs_on"]) * ms
    pr = (f["qrs_on"] - f["p_on"]) * ms if f["p_on"] is not None else nan
    qt = (f["t_end"] - f["qrs_on"]) * ms if f["t_end"] is not None else nan
    qtc = qt / np.sqrt(rr_ms / 1000.0) if (rr_ms > 0 and not np.isnan(qt)) else nan
    return pr, qrs, qt, qtc, n_beats
