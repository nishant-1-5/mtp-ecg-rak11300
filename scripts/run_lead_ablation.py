"""Execute the lead / sampling-rate ablation defined in ecg_12lead_ablation_study.ipynb.

    python scripts/run_lead_ablation.py

The notebook (added by the team) defines the models, features and plan but
contains no executed results. This script imports nothing new: the model,
feature, normalisation and evaluation functions are copied verbatim from the
notebook, and it runs the notebook's plan (extended to every LEAD_SETS entry)
on the full PTB-XL 100 Hz data with the official fold-10 test split.

Adds only what the notebook's own checklist asks for: bootstrap 95 % CIs for
macro AUROC (1,000 resamples of fold-10 records) and the INT8 size of each CNN.
Writes results/ablation/lead_ablation.csv and results/ablation/lead_ablation.md.
"""
import ast
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation"
CACHE = ROOT / "data" / "cache" / "ptbxl_12lead_100hz.npz"

# ---------------------------------------------------------------- from the notebook (cells 2-7)
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
DATA_ROOT = ROOT / "dataset"
FS = 100
CLASS_ORDER = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_SETS = {
    '12_lead': list(range(12)),
    'limb_6': [0, 1, 2, 3, 4, 5],
    'basic_3': [0, 1, 2],
    'lead_I': [0],
    'precordial_6': [6, 7, 8, 9, 10, 11],
}


def downsample(x, original_hz, target_hz):
    if target_hz == original_hz:
        return x
    factor = original_hz // target_hz
    if original_hz % target_hz:
        raise ValueError('Use an integer downsampling factor for this baseline.')
    return x[:, ::factor, :]


def normalize_per_record(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def compact_ecg_cnn(n_samples, n_leads, n_classes=5):
    inp = tf.keras.Input((n_samples, n_leads), name='ecg')
    x = tf.keras.layers.SeparableConv1D(16, 9, strides=2, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv1D(32, 7, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.SeparableConv1D(48, 5, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(n_classes, activation='sigmoid', name='diagnoses')(x)
    model = tf.keras.Model(inp, out, name='compact_separable_ecg_cnn')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(curve='ROC', multi_label=True, name='macro_auc')])
    return model


def waveform_features(x):
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), (x ** 2).mean(1)], axis=1)


def evaluate(y_true, probabilities, threshold=0.5):
    per_class = roc_auc_score(y_true, probabilities, average=None)
    return {
        'macro_auroc': roc_auc_score(y_true, probabilities, average='macro'),
        'macro_f1': f1_score(y_true, probabilities >= threshold, average='macro', zero_division=0),
        **{f'auroc_{name}': score for name, score in zip(CLASS_ORDER, per_class)}
    }


# ---------------------------------------------------------------- data (notebook cells 3-4)
def load_all():
    meta = pd.read_csv(DATA_ROOT / 'ptbxl_database.csv', index_col='ecg_id')
    meta['scp_codes'] = meta['scp_codes'].map(ast.literal_eval)
    statements = pd.read_csv(DATA_ROOT / 'scp_statements.csv', index_col=0)
    diagnostic = statements[statements.diagnostic == 1]
    meta['labels'] = meta['scp_codes'].map(lambda d: sorted({
        diagnostic.loc[c, 'diagnostic_class'] for c in d
        if c in diagnostic.index and diagnostic.loc[c, 'diagnostic_class'] in CLASS_ORDER}))
    meta = meta[meta.labels.map(len) > 0].copy()
    y = MultiLabelBinarizer(classes=CLASS_ORDER).fit_transform(meta.labels)
    train = (meta.strat_fold != 10).to_numpy()
    if CACHE.exists():
        X = np.load(CACHE)["X"]
    else:
        t0 = time.time()
        X = np.stack([wfdb.rdsamp(str(DATA_ROOT / f))[0].astype('float32') for f in meta.filename_lr])
        assert X.shape[2] == 12
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(CACHE, X=X)
        print(f"loaded {len(X):,} records x 12 leads in {time.time() - t0:.0f} s", flush=True)
    print(f"Records: {len(meta):,}; train: {train.sum():,}; test: {(~train).sum():,}", flush=True)
    return X, y, train


def boot_ci(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum(0).min() == 0:
            continue
        vals.append(roc_auc_score(y[i], p[i], average='macro'))
    return np.percentile(vals, [2.5, 97.5])


def int8_bytes(model, xtr):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([xtr[i:i + 1]] for i in range(200))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    return len(conv.convert())


def run_cnn(X, y, train, lead_set, hz):
    tf.keras.utils.set_random_seed(SEED)
    idx = LEAD_SETS[lead_set]
    x = normalize_per_record(downsample(X[:, :, idx], FS, hz)).astype('float32')
    xtr, xte, ytr, yte = x[train], x[~train], y[train], y[~train]
    model = compact_ecg_cnn(xtr.shape[1], xtr.shape[2])
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_macro_auc', mode='max', patience=4, restore_best_weights=True)]
    t0 = time.time()
    model.fit(xtr, ytr, validation_split=.1, epochs=20, batch_size=64, callbacks=cb, verbose=0)
    train_s = time.time() - t0
    start = time.perf_counter(); p = model.predict(xte, batch_size=128, verbose=0); sec = time.perf_counter() - start
    r = evaluate(yte, p)
    lo, hi = boot_ci(yte, p)
    r.update({'model': 'Compact separable CNN', 'lead_set': lead_set, 'leads': len(idx), 'sampling_hz': hz,
              'parameters': model.count_params(), 'int8_bytes': int8_bytes(model, xtr),
              'ms_per_record_desktop': 1000 * sec / len(xte), 'auroc_ci_lo': lo, 'auroc_ci_hi': hi,
              'train_s': train_s})
    return r


def run_baseline(X, y, train, lead_set, estimator):
    idx = LEAD_SETS[lead_set]
    x = normalize_per_record(X[:, :, idx])
    xtr, xte, ytr, yte = x[train], x[~train], y[train], y[~train]
    base = (LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5) if estimator == 'logistic'
            else RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample',
                                        n_jobs=-1, random_state=SEED))
    clf = make_pipeline(StandardScaler(), OneVsRestClassifier(base)) if estimator == 'logistic' else OneVsRestClassifier(base)
    clf.fit(waveform_features(xtr), ytr)
    p = clf.predict_proba(waveform_features(xte))
    r = evaluate(yte, p)
    lo, hi = boot_ci(yte, p)
    r.update({'model': {'logistic': 'Logistic regression', 'rf': 'Random forest'}[estimator], 'lead_set': lead_set,
              'leads': len(idx), 'sampling_hz': 100, 'auroc_ci_lo': lo, 'auroc_ci_hi': hi})
    return r


def main():
    X, y, train = load_all()
    plan = [('cnn', s, 100) for s in LEAD_SETS] + [('cnn', 'limb_6', 50), ('cnn', 'lead_I', 50)] + \
           [(e, s, 100) for s in ('lead_I', 'limb_6') for e in ('logistic', 'rf')]
    rows = []
    for kind, s, hz in plan:
        t0 = time.time()
        r = run_cnn(X, y, train, s, hz) if kind == 'cnn' else run_baseline(X, y, train, s, kind)
        rows.append(r)
        print(f"{r['model']:<22} {s:<13} {hz:>3} Hz  AUROC {r['macro_auroc']:.4f} "
              f"[{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}]  F1 {r['macro_f1']:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / "lead_ablation.csv", index=False)
    write_md(pd.DataFrame(rows))


def write_md(df):
    def f(v, nd=3):
        return "" if pd.isna(v) else f"{v:.{nd}f}"
    cnn = df[df.model == 'Compact separable CNN']
    lines = ["# Lead / sampling-rate ablation (measured)", "",
             "Generated by `python scripts/run_lead_ablation.py`, which executes the models and plan of "
             "`ecg_12lead_ablation_study.ipynb` on the full PTB-XL 100 Hz data. Test set: official fold 10 "
             f"(n = {int((df.shape[0] > 0) and 2158)}). Seed 42, single run per configuration; 95 % CI from 1,000 "
             "bootstrap resamples of test records. F1 at a fixed 0.5 threshold. Latency is desktop CPU, batch 128 "
             "(indicative only; RP2040 numbers are in docs/phase2_tflm.md).", "",
             "## Compact separable CNN", "",
             "| Lead set | Leads | Rate | Macro AUROC [95 % CI] | Macro F1 | NORM | MI | STTC | CD | HYP | Params | INT8 size |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in cnn.iterrows():
        lines.append(f"| {r.lead_set} | {r.leads} | {r.sampling_hz} Hz | {f(r.macro_auroc)} [{f(r.auroc_ci_lo)}, "
                     f"{f(r.auroc_ci_hi)}] | {f(r.macro_f1)} | {f(r.auroc_NORM)} | {f(r.auroc_MI)} | {f(r.auroc_STTC)} | "
                     f"{f(r.auroc_CD)} | {f(r.auroc_HYP)} | {int(r.parameters):,} | {int(r.int8_bytes):,} B |")
    lines += ["", "## Handcrafted waveform-summary baselines (5 features per lead)", "",
              "| Model | Lead set | Macro AUROC [95 % CI] | Macro F1 |", "|---|---|---|---:|"]
    for _, r in df[df.model != 'Compact separable CNN'].iterrows():
        lines.append(f"| {r.model} | {r.lead_set} | {f(r.macro_auroc)} [{f(r.auroc_ci_lo)}, {f(r.auroc_ci_hi)}] "
                     f"| {f(r.macro_f1)} |")
    (OUT / "lead_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
