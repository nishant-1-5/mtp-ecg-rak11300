"""Python reference for the Lead I features ported to firmware/ecg_features.

The body is copied from fix_cache.py::extract_ecg_features_corrected with the
Lead II/III statistics and the PR/QT/QTc block removed. Do not "improve" it
here: this function is the specification the C++ port is tested against.
"""
import numpy as np
import scipy.signal as signal

FEATURE_NAMES = [
    "mean_hr", "min_hr", "max_hr", "hr_std", "mean_rr", "rr_std", "rr_range",
    "rr_cv", "sdnn", "rmssd", "pnn50", "hjorth_complexity", "mean_qrs_ms",
    "snr_db", "baseline_wander",
]
# Column of each feature in the 36-wide X_feat_* arrays in the .npz cache.
CACHE_COLUMNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17]


def extract_lead1_features(lead1, fs=100):
    lead1 = np.asarray(lead1)

    b, a = signal.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, lead1)
    integrated = np.convolve(np.diff(filtered)**2, np.ones(int(0.08 * fs)), mode='same')
    peaks, _ = signal.find_peaks(integrated, distance=int(0.35 * fs), prominence=np.std(integrated) * 0.5)

    if len(peaks) < 2:
        rr_intervals = np.array([800.0])
    else:
        rr_intervals = (np.diff(peaks) / fs) * 1000.0

    hr_bpm = 60000.0 / rr_intervals
    mean_hr, min_hr, max_hr, hr_std = np.mean(hr_bpm), np.min(hr_bpm), np.max(hr_bpm), np.std(hr_bpm)
    mean_rr, rr_std = np.mean(rr_intervals), np.std(rr_intervals)
    rr_range = np.max(rr_intervals) - np.min(rr_intervals)
    rr_cv = (rr_std / mean_rr) if mean_rr > 0 else 0.0

    sdnn = np.std(rr_intervals)
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100.0 if len(rr_diff) > 0 else 0.0

    if len(peaks) > 0:
        widths, _, _, _ = signal.peak_widths(integrated, peaks, rel_height=0.5)
        mean_qrs = np.mean(widths) / fs * 1000.0
    else:
        mean_qrs = 90.0

    dy, ddy = np.diff(lead1), np.diff(np.diff(lead1))
    var_y, var_dy, var_ddy = np.var(lead1), np.var(dy), np.var(ddy)
    mobility = np.sqrt(var_dy / var_y) if var_y > 0 else 0.0
    complexity = (np.sqrt(var_ddy / var_dy) / mobility) if (var_dy > 0 and mobility > 0) else 0.0

    freqs, psd = signal.welch(lead1, fs=fs, nperseg=256)
    bw_power = np.sum(psd[freqs < 0.5])
    total_power = np.sum(psd) + 1e-8
    snr = 10 * np.log10((total_power - bw_power) / (bw_power + 1e-8))
    baseline_wander = bw_power / total_power

    return np.array([mean_hr, min_hr, max_hr, hr_std, mean_rr, rr_std, rr_range, rr_cv,
                     sdnn, rmssd, pnn50, complexity, mean_qrs, snr, baseline_wander],
                    dtype=np.float64), len(peaks)
