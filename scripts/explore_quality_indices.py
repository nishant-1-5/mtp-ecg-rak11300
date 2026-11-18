"""Exploration behind docs/orchestrator.md: cheap quality indices vs PTB-XL noise
annotations, and single-feature AUCs for rhythm statements (training folds only)."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as P  # noqa
from ptbxl_plus import cache_ecg_ids  # noqa
from reference_features import FEATURE_NAMES  # noqa

c = np.load(P.CACHE_FILE, allow_pickle=True)
F = pd.DataFrame(c["X_feat_v2_train"][:, :18], columns=FEATURE_NAMES)
x = c["X_train"][:, :, 0]
F["raw_std"] = x.std(axis=1)
F["raw_ptp"] = np.ptp(x, axis=1)
df = pd.read_csv(P.DATASET_DIR / "ptbxl_database.csv", index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
d = df.loc[cache_ecg_ids()[0]].reset_index()
for code in ["AFIB", "STACH", "SBRAD", "SR", "SARRH", "AFLT", "1AVB", "CLBBB", "CRBBB", "IVCD"]:
    F[code] = d.scp_codes.apply(lambda s: code in s).astype(int).to_numpy()


def lead1_noise(col):
    s = d[col].fillna("").astype(str)
    return s.str.contains(r"(?:^|[ ,])I(?:[ ,-]|$)|alles", regex=True).to_numpy()


F["noise_I"] = lead1_noise("static_noise") | lead1_noise("burst_noise") | lead1_noise("baseline_drift")

print(F[["snr_db", "baseline_wander", "raw_std", "raw_ptp", "mean_hr", "rr_cv", "rmssd", "pnn50"]]
      .describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(3).T.to_string())
print("n_beats<3:", (F.eval("qrs_ms != qrs_ms")).mean())
print("noise_I fraction", F.noise_I.mean())
for k in ["snr_db", "baseline_wander", "raw_std"]:
    print(f"AUC {k} vs noise_I annotation: {roc_auc_score(F.noise_I, F[k]):.3f}")
for code, feats in [("AFIB", ["rr_cv", "rmssd", "pnn50", "hr_std"]), ("STACH", ["mean_hr"]), ("SBRAD", ["mean_hr"]),
                    ("1AVB", ["pr_ms"]), ("CLBBB", ["qrs_ms"]), ("CRBBB", ["qrs_ms"])]:
    for f in feats:
        m = F[f].notna()
        print(f"{code:6s} n={F[code].sum():5d} AUC({f}) = {roc_auc_score(F[code][m], F[f][m]):.3f}")
afib = F[F.AFIB == 1]
print("AFIB: P absent (pr NaN) %.1f%% vs others %.1f%%" % (100 * afib.pr_ms.isna().mean(), 100 * F[F.AFIB == 0].pr_ms.isna().mean()))
print("STACH HR median", F[F.STACH == 1].mean_hr.median(), "SBRAD HR median", F[F.SBRAD == 1].mean_hr.median())
