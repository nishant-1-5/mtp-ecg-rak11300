"""Shared paths and run configuration for the notebooks and scripts.

Notebooks locate this file by walking up from their working directory, so they
run the same from the repo root, from notebooks/, or headless.

Lead configuration
------------------
Set the environment variable MTP_LEAD_SET before starting the kernel:
    lead1  (default) - Lead I only, matches the RA/LA/RL hardware
    3lead            - Leads I, II, III, the original baseline (tag v0-3lead)
Models and results are written to models/<lead set>/ and results/<lead set>/.
"""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset"
CACHE_FILE = ROOT / "data" / "cache" / "ptbxl_preprocessed_cache.npz"
FEATURE_RESULTS_DIR = ROOT / "results" / "features"

SEED = 42

LEAD_SETS = {"lead1": [0], "3lead": [0, 1, 2]}
LEAD_SET = os.environ.get("MTP_LEAD_SET", "lead1")
if LEAD_SET not in LEAD_SETS:
    raise ValueError(f"MTP_LEAD_SET must be one of {list(LEAD_SETS)}, got {LEAD_SET!r}")
LEADS = LEAD_SETS[LEAD_SET]
MODELS_DIR = ROOT / "models" / LEAD_SET
RESULTS_DIR = ROOT / "results" / LEAD_SET

# Columns of X_feat_* in the cache (written by scripts/fix_cache.py).
FEATURE_NAMES = [
    "Mean_HR", "Min_HR", "Max_HR", "HR_Std", "Mean_RR", "RR_Std", "RR_Range", "RR_CV",
    "SDNN", "RMSSD", "pNN50", "Sample_Entropy", "Mean_PR", "Mean_QRS", "Mean_QT", "Mean_QTc",
    "SNR", "Baseline_Wander",
] + [f"{lead}_{stat}" for lead in ("I", "II", "III")
     for stat in ("Mean", "Std", "Skew", "Kurtosis", "PTP", "IQR")]
# Lead I only: the 18 rhythm/morphology/quality features + Lead I statistics.
FEATURE_COLUMNS = {"lead1": list(range(24)), "3lead": list(range(36))}[LEAD_SET]

# Input convention of the 1D-CNN (notebooks 11 and 20, and the firmware).
#   none   - raw mV (the original 3-lead model)
#   zscore - per window and channel: (x - mean) / (std + 1e-3 mV)
# With raw mV, INT8 calibration has to cover rare +-7 mV windows and the INT8
# Lead I model lost ~0.05 macro AUC; with zscore the loss is ~0.004.
# See docs/lead1_retrain.md.
CNN_INPUT_NORMS = {"lead1": "zscore", "3lead": "none"}
CNN_INPUT_NORM = CNN_INPUT_NORMS[LEAD_SET]
ZSCORE_EPS_MV = 1e-3


def normalize_cnn_input(X, mode=None):
    """Applies the CNN input convention to signals shaped (N, samples, channels).

    mode defaults to CNN_INPUT_NORM of the active lead set.
    """
    mode = CNN_INPUT_NORM if mode is None else mode
    if mode == "none":
        return X
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc / (Xc.std(axis=1, keepdims=True) + ZSCORE_EPS_MV)).astype(np.float32)


def load_cache(verbose=True):
    """Loads the cache and keeps only the leads/features of the active lead set.

    Returns X_train, y_train, X_test, y_test, class_names, X_feat_train, X_feat_test.
    """
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"{CACHE_FILE} not found. Run notebooks/01_preprocess.ipynb first.")
    cached = np.load(CACHE_FILE, allow_pickle=True)
    X_train = cached["X_train"][:, :, LEADS]
    X_test = cached["X_test"][:, :, LEADS]
    y_train, y_test = cached["y_train"], cached["y_test"]
    class_names = cached["class_names"]
    X_feat_train = cached["X_feat_train"][:, FEATURE_COLUMNS]
    X_feat_test = cached["X_feat_test"][:, FEATURE_COLUMNS]
    if verbose:
        print(f"Lead set  -> {LEAD_SET} (channels {LEADS})")
        print(f"Signals   -> Train: {X_train.shape} | Test: {X_test.shape}")
        print(f"Features  -> Train: {X_feat_train.shape} | Test: {X_feat_test.shape}")
        print(f"Targets   -> Classes ({len(class_names)}): {class_names.tolist()}")
    return X_train, y_train, X_test, y_test, class_names, X_feat_train, X_feat_test


def output_dirs():
    """Creates and returns (MODELS_DIR, RESULTS_DIR) for the active lead set."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR, RESULTS_DIR
