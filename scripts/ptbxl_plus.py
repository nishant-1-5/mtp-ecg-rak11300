"""Reference interval measurements from PTB-XL+ (GE Marquette 12SL, 12-lead, 500 Hz).

Download once (93 MB, kept in the ignored dataset/ folder):
    https://physionet.org/files/ptb-xl-plus/1.0.1/features/12sl_features.csv
    -> dataset/ptb-xl-plus/features/12sl_features.csv

Citation: Strodthoff et al. (2023), PTB-XL+, a comprehensive electrocardiographic
feature dataset. Sci Data 10, 279. https://doi.org/10.1038/s41597-023-02153-8
"""
import ast

import numpy as np
import pandas as pd

from project import DATASET_DIR

TEST_FOLD = 10
FILE_12SL = DATASET_DIR / "ptb-xl-plus" / "features" / "12sl_features.csv"
REF_COLUMNS = {"QRS": "QRS_Dur_Global", "PR": "PR_Int_Global", "QT": "QT_Int_Global",
               "RR": "RR_Mean_Global"}


def cache_ecg_ids():
    """ecg_id of every row of X_train / X_test, reproducing notebooks/01_preprocess.ipynb."""
    df = pd.read_csv(DATASET_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(DATASET_DIR / "scp_statements.csv", index_col=0)
    diag = set(agg[agg.diagnostic == 1].index)
    df = df[df.scp_codes.apply(lambda d: any(k in diag for k in d))]
    train = df.index[df.strat_fold != TEST_FOLD].to_numpy()
    test = df.index[df.strat_fold == TEST_FOLD].to_numpy()
    return train, test


def reference_intervals(ecg_ids):
    """DataFrame (rows aligned with ecg_ids) of 12SL QRS / PR / QT / RR in ms; NaN if absent."""
    if not FILE_12SL.exists():
        raise FileNotFoundError(f"{FILE_12SL} missing, see module docstring")
    ref = pd.read_csv(FILE_12SL, usecols=["ecg_id", *REF_COLUMNS.values()]).set_index("ecg_id")
    ref = ref.reindex(ecg_ids)[list(REF_COLUMNS.values())]
    ref = ref.rename(columns={v: k for k, v in REF_COLUMNS.items()})
    ref.loc[ref.PR <= 0, "PR"] = np.nan  # 12SL reports 0 when no P wave was found
    return ref.reset_index(drop=True)
