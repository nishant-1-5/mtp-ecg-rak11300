import numpy as np
import scipy.signal as signal
from scipy.stats import skew, kurtosis

from project import CACHE_FILE  # data/cache/ptbxl_preprocessed_cache.npz

print(f"Loading raw signals from {CACHE_FILE}...")
cached = np.load(CACHE_FILE, allow_pickle=True)
X_train = cached["X_train"]
X_test = cached["X_test"]
y_train = cached["y_train"]
y_test = cached["y_test"]
class_names = cached["class_names"]

def extract_ecg_features_corrected(signal_3lead, fs=100):
    lead1 = signal_3lead[:, 0]
    lead2 = signal_3lead[:, 1]
    lead3 = signal_3lead[:, 2]

    # --- 1. R-Peak Detection ---
    b, a = signal.butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype='bandpass')
    filtered = signal.filtfilt(b, a, lead1)
    integrated = np.convolve(np.diff(filtered)**2, np.ones(int(0.08 * fs)), mode='same')
    peaks, _ = signal.find_peaks(integrated, distance=int(0.35 * fs), prominence=np.std(integrated) * 0.5)

    if len(peaks) < 2:
        rr_intervals = np.array([800.0])
    else:
        rr_intervals = (np.diff(peaks) / fs) * 1000.0

    # --- 2. HRV Metrics ---
    hr_bpm = 60000.0 / rr_intervals
    mean_hr, min_hr, max_hr, hr_std = np.mean(hr_bpm), np.min(hr_bpm), np.max(hr_bpm), np.std(hr_bpm)
    mean_rr, rr_std = np.mean(rr_intervals), np.std(rr_intervals)
    rr_range = np.max(rr_intervals) - np.min(rr_intervals)
    rr_cv = (rr_std / mean_rr) if mean_rr > 0 else 0.0

    sdnn = np.std(rr_intervals)
    rr_diff = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0.0
    pnn50 = (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100.0 if len(rr_diff) > 0 else 0.0

    # --- 3. THE FIX: Dynamic Morphological Intervals ---
    # QRS Duration (Width of the R-peak integration window)
    if len(peaks) > 0:
        widths, _, _, _ = signal.peak_widths(integrated, peaks, rel_height=0.5)
        mean_qrs = np.mean(widths) / fs * 1000.0
    else:
        mean_qrs = 90.0

    # PR & QT Intervals via localized search windows
    pr_intervals, qt_intervals = [], []
    for p in peaks:
        # P-wave: Max peak in the 250ms window before R-peak
        p_start = max(0, p - int(0.25 * fs))
        if p > p_start:
            pr_intervals.append(((len(lead1[p_start:p]) - np.argmax(lead1[p_start:p])) / fs) * 1000.0)
        
        # T-wave: Max peak in the 400ms window after R-peak
        t_end = min(len(lead1), p + int(0.40 * fs))
        if t_end > p:
            qt_intervals.append((np.argmax(lead1[p:t_end]) / fs) * 1000.0)

    mean_pr = np.mean(pr_intervals) if pr_intervals else 160.0
    mean_qt = np.mean(qt_intervals) if qt_intervals else 380.0
    mean_qtc = mean_qt / np.sqrt(mean_rr / 1000.0) if mean_rr > 0 else 400.0

    # --- 4. THE FIX: Fast Signal Complexity (Hjorth proxy for Entropy) ---
    dy, ddy = np.diff(lead1), np.diff(np.diff(lead1))
    var_y, var_dy, var_ddy = np.var(lead1), np.var(dy), np.var(ddy)
    mobility = np.sqrt(var_dy / var_y) if var_y > 0 else 0.0
    sample_entropy = (np.sqrt(var_ddy / var_dy) / mobility) if (var_dy > 0 and mobility > 0) else 0.0

    # --- 5. Signal Quality & Spatial Channels ---
    freqs, psd = signal.welch(lead1, fs=fs, nperseg=256)
    bw_power = np.sum(psd[freqs < 0.5])
    total_power = np.sum(psd) + 1e-8
    snr = 10 * np.log10((total_power - bw_power) / (bw_power + 1e-8))
    baseline_wander = bw_power / total_power

    lead_stats = []
    for l in [lead1, lead2, lead3]:
        lead_stats.extend([np.mean(l), np.std(l), skew(l), kurtosis(l), np.ptp(l), np.percentile(l, 75) - np.percentile(l, 25)])

    return np.array([mean_hr, min_hr, max_hr, hr_std, mean_rr, rr_std, rr_range, rr_cv,
                     sdnn, rmssd, pnn50, sample_entropy, mean_pr, mean_qrs, mean_qt, mean_qtc,
                     snr, baseline_wander] + lead_stats, dtype=np.float32)

print("Calculating corrected dynamic features across all records (this takes ~1-2 mins)...")
X_feat_train_fixed = np.nan_to_num(np.array([extract_ecg_features_corrected(x) for x in X_train]))
X_feat_test_fixed = np.nan_to_num(np.array([extract_ecg_features_corrected(x) for x in X_test]))

print("Overwriting cache with corrected features...")
np.savez_compressed(
    CACHE_FILE,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    class_names=class_names, X_feat_train=X_feat_train_fixed, X_feat_test=X_feat_test_fixed
)
print("Features successfully fixed and cached!")