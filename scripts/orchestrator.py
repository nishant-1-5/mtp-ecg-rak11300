"""Reference implementation of the on-device decision logic (spec for firmware/ecg_pipeline).

Per 10 s Lead I window:
  1. Features (reference_features.extract_lead1_features, incl. template correlation).
  2. Signal-quality gate. If it fails, the window is reported as unreliable and
     the CNN is not run.
  3. Rule-based rhythm/interval flags from the features (interpretable, cheap).
  4. INT8 1D-CNN on the z-scored window -> 5 superclass probabilities ->
     per-class thresholds. CD may additionally be raised by the wide-QRS rule.

    python scripts/orchestrator.py fit       # thresholds from training folds 1-9
    python scripts/orchestrator.py evaluate  # fold 10, clean + NSTDB noise

`fit` writes results/lead1/orchestrator_params.json and the generated header
firmware/ecg_pipeline/ecg_pipeline_params.h (single source of truth for the C++).
"""
import argparse
import ast
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as P  # noqa: E402
from ptbxl_plus import cache_ecg_ids, reference_intervals  # noqa: E402
from reference_features import FEATURE_NAMES, extract_lead1_features  # noqa: E402

CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
FI = {k: i for i, k in enumerate(FEATURE_NAMES)}
PARAMS_JSON = P.ROOT / "results" / "lead1" / "orchestrator_params.json"
HEADER = P.ROOT / "firmware" / "ecg_pipeline" / "ecg_pipeline_params.h"
TFLITE = P.ROOT / "models" / "lead1" / "tinyml_1d_cnn_int8.tflite"
MV_PER_LSB = 0.001  # PTB-XL

# Quality reason bits (0 = good). Must match firmware/ecg_pipeline/ecg_pipeline.h.
Q_FLAT, Q_SATURATED, Q_NO_RHYTHM, Q_NO_TEMPLATE, Q_NOISY, Q_BASELINE = 1, 2, 4, 8, 16, 32
# Rhythm / interval flag bits.
F_TACHY, F_BRADY, F_IRREGULAR, F_AF_SUSPECT, F_WIDE_QRS, F_LONG_PR = 1, 2, 4, 8, 16, 32

# Fixed (clinical or physical) parameters. Fitted ones are added by `fit`.
FIXED = {
    "min_ptp_mv": 0.1,          # below: flat line / lead off (PTB-XL 0.1 % quantile is 0.27 mV)
    "max_rail_fraction": 0.01,  # above: clipped ADC
    "adc_min": -32768, "adc_max": 32767,  # set to the device ADC range in firmware
    "min_peaks": 4,
    "hr_min_bpm": 30.0, "hr_max_bpm": 220.0,
    "min_beats": 3,
    "tachy_bpm": 100.0, "brady_bpm": 60.0,  # standard sinus tachy/brady limits
    "wide_qrs_ms": 120.0,                   # standard bundle-branch-block limit
    "long_pr_ms": 200.0,                    # standard 1st-degree AV block limit
}
CLEAN_REJECT_BUDGET = 0.02  # template-correlation gate may reject 2 % of clean training windows


# ----------------------------------------------------------------------------- logic
def gate(adc, feats, n_peaks, quality, p):
    """Quality reason bitmask for one window (0 = usable)."""
    reasons = 0
    ptp_mv = (int(adc.max()) - int(adc.min())) * MV_PER_LSB
    if ptp_mv < p["min_ptp_mv"]:
        reasons |= Q_FLAT
    rails = np.mean((adc <= p["adc_min"]) | (adc >= p["adc_max"]))
    if rails > p["max_rail_fraction"]:
        reasons |= Q_SATURATED
    hr = feats[FI["mean_hr"]]
    if n_peaks < p["min_peaks"] or not (p["hr_min_bpm"] <= hr <= p["hr_max_bpm"]):
        reasons |= Q_NO_RHYTHM
    if quality["n_beats"] < p["min_beats"]:
        reasons |= Q_NO_TEMPLATE
    elif not (quality["template_corr"] >= p["min_template_corr"]):
        reasons |= Q_NOISY
    if feats[FI["baseline_wander"]] > p["max_baseline_wander"]:
        reasons |= Q_BASELINE
    return reasons


def rhythm_flags(feats, p):
    flags = 0
    hr, rr_cv = feats[FI["mean_hr"]], feats[FI["rr_cv"]]
    qrs, pr = feats[FI["qrs_ms"]], feats[FI["pr_ms"]]
    if hr > p["tachy_bpm"]:
        flags |= F_TACHY
    if hr < p["brady_bpm"]:
        flags |= F_BRADY
    if rr_cv >= p["irregular_rr_cv"]:
        flags |= F_IRREGULAR
        if np.isnan(pr):
            flags |= F_AF_SUSPECT
    if qrs >= p["wide_qrs_ms"]:  # NaN compares False
        flags |= F_WIDE_QRS
    if pr > p["long_pr_ms"]:
        flags |= F_LONG_PR
    return flags


def cnn_input_int8(adc, scale, zero_point):
    """z-score in float32 exactly like project.normalize_cnn_input, then quantise."""
    x = (adc.astype(np.float32) * np.float32(MV_PER_LSB))[None, :, None]
    z = P.normalize_cnn_input(x, "zscore")
    return np.clip(np.round(z / scale + zero_point), -128, 127).astype(np.int8)[0, :, 0]


def decide(probs, flags, p):
    """Per-class decisions (bitmask over CLASS_NAMES)."""
    out = 0
    for k, name in enumerate(CLASS_NAMES):
        if probs[k] >= p["class_thresholds"][name]:
            out |= 1 << k
    if p["cd_or_wide_qrs"] and (flags & F_WIDE_QRS):
        out |= 1 << CLASS_NAMES.index("CD")
    return out


# ----------------------------------------------------------------------------- batch helpers
def _window(adc):
    f, npk, q = extract_lead1_features(adc.astype(np.float64) * MV_PER_LSB, with_quality=True)
    return f, npk, q["n_beats"], q["template_corr"]


def analyze_features(adc_windows):
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(_window, adc_windows, chunksize=128))
    feats = np.stack([r[0] for r in rows])
    n_peaks = np.array([r[1] for r in rows])
    quality = [{"n_beats": r[2], "template_corr": r[3]} for r in rows]
    return feats, n_peaks, quality


class Int8Cnn:
    """INT8 model on TFLite's *reference* kernels.

    The default desktop resolver uses XNNPACK, whose requantisation rounding
    differs from TFLM/CMSIS-NN by up to 9 output LSB on this model. The
    reference kernels agree with the device to within 1 LSB
    (docs/phase2_tflm.md), so they are the reference here."""

    def __init__(self, path=TFLITE):
        import tensorflow as tf
        self.tf = tf
        self.path = str(path)
        it = self._interpreter()
        d = it.get_input_details()[0]
        self.scale, self.zero_point = d["quantization"]

    def _interpreter(self):
        return self.tf.lite.Interpreter(
            model_path=self.path,
            experimental_op_resolver_type=self.tf.lite.experimental.OpResolverType.BUILTIN_REF)

    def run(self, q_inputs):
        """q_inputs: (N, 1000) int8 -> (N, 5) probabilities."""
        it = self._interpreter()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        it.resize_tensor_input(i["index"], [len(q_inputs), q_inputs.shape[1], 1])
        it.allocate_tensors()
        it.set_tensor(i["index"], q_inputs[:, :, None])
        it.invoke()
        so, zo = o["quantization"]
        return (it.get_tensor(o["index"]).astype(np.float32) - zo) * so


def to_adc(mv):
    return np.clip(np.round(mv / MV_PER_LSB), -32768, 32767).astype(np.int16)


def labels(split):
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    ids = cache_ecg_ids()[0 if split == "train" else 1]
    df = pd.read_csv(P.DATASET_DIR / "ptbxl_database.csv", index_col="ecg_id")
    codes = df.loc[ids].scp_codes.apply(ast.literal_eval).reset_index(drop=True)
    has = lambda *cs: codes.apply(lambda d: any(c in d for c in cs)).to_numpy()  # noqa: E731
    rhythm = {"STACH": has("STACH"), "SBRAD": has("SBRAD"), "AFIB": has("AFIB", "AFLT"),
              "1AVB": has("1AVB")}
    return cache, ids, rhythm


# ----------------------------------------------------------------------------- fit
def fit():
    cache, ids, rhythm = labels("train")
    adc = to_adc(cache["X_train"][:, :, 0].astype(np.float64))
    y = cache["y_train"]
    feats, n_peaks, quality = analyze_features(adc)
    p = dict(FIXED)

    tc = np.array([q["template_corr"] for q in quality], dtype=float)
    p["min_template_corr"] = round(float(np.nanquantile(tc, CLEAN_REJECT_BUDGET)), 3)
    p["max_baseline_wander"] = round(float(np.quantile(feats[:, FI["baseline_wander"]], 0.995)), 3)

    # Irregular rhythm: rr_cv threshold maximising Youden's J for AFIB/AFLT on training folds.
    cv = feats[:, FI["rr_cv"]]
    grid = np.round(np.arange(0.02, 0.30, 0.005), 3)
    j = [np.mean(cv[rhythm["AFIB"]] >= t) + np.mean(cv[~rhythm["AFIB"]] < t) - 1 for t in grid]
    p["irregular_rr_cv"] = float(grid[int(np.argmax(j))])

    cnn = Int8Cnn()
    q_in = np.stack([cnn_input_int8(a, cnn.scale, cnn.zero_point) for a in adc])
    probs = cnn.run(q_in)
    # Thresholds on the windows the CNN never trained on: Keras validation_split=0.1
    # holds out the last 10 % of the training arrays.
    val = np.arange(len(adc)) >= int(len(adc) * 0.9)
    th = {}
    for k, name in enumerate(CLASS_NAMES):
        cand = np.round(np.arange(0.05, 0.95, 0.01), 2)
        f1 = [f1_score(y[val, k], probs[val, k] >= t) for t in cand]
        th[name] = float(cand[int(np.argmax(f1))])
    p["class_thresholds"] = th

    k_cd = CLASS_NAMES.index("CD")
    wide = feats[:, FI["qrs_ms"]] >= p["wide_qrs_ms"]
    f1_cnn = f1_score(y[val, k_cd], probs[val, k_cd] >= th["CD"])
    f1_or = f1_score(y[val, k_cd], (probs[val, k_cd] >= th["CD"]) | wide[val])
    p["cd_or_wide_qrs"] = bool(f1_or > f1_cnn)
    p["fit_notes"] = {
        "template_corr_rule": f"{CLEAN_REJECT_BUDGET:.0%} quantile of clean training windows",
        "baseline_wander_rule": "99.5 % quantile of clean training windows",
        "irregular_rr_cv_rule": "max Youden J for AFIB/AFLT, training folds",
        "class_threshold_rule": "max F1 per class, INT8 model, on the last 10 % of training "
                                "windows (Keras validation_split, never trained on)",
        "cd_val_f1": {"cnn_only": round(f1_cnn, 4), "cnn_or_wide_qrs": round(f1_or, 4)},
        "cnn_input": {"scale": float(cnn.scale), "zero_point": int(cnn.zero_point)},
    }
    PARAMS_JSON.write_text(json.dumps(p, indent=2), encoding="utf-8")
    write_header(p)
    print(json.dumps(p, indent=2))
    print(f"wrote {PARAMS_JSON} and {HEADER}")


def write_header(p):
    th = p["class_thresholds"]
    lines = [
        "// GENERATED by scripts/orchestrator.py fit -- do not edit by hand.",
        "#ifndef ECG_PIPELINE_PARAMS_H_",
        "#define ECG_PIPELINE_PARAMS_H_",
        "",
        "namespace ecg {",
        "namespace params {",
        f"constexpr float kMinPtpMv = {p['min_ptp_mv']}f;",
        f"constexpr float kMaxRailFraction = {p['max_rail_fraction']}f;",
        f"constexpr int kMinPeaks = {p['min_peaks']};",
        f"constexpr float kHrMinBpm = {p['hr_min_bpm']}f;",
        f"constexpr float kHrMaxBpm = {p['hr_max_bpm']}f;",
        f"constexpr int kMinBeats = {p['min_beats']};",
        f"constexpr float kMinTemplateCorr = {p['min_template_corr']}f;",
        f"constexpr float kMaxBaselineWander = {p['max_baseline_wander']}f;",
        f"constexpr float kTachyBpm = {p['tachy_bpm']}f;",
        f"constexpr float kBradyBpm = {p['brady_bpm']}f;",
        f"constexpr float kIrregularRrCv = {p['irregular_rr_cv']}f;",
        f"constexpr float kWideQrsMs = {p['wide_qrs_ms']}f;",
        f"constexpr float kLongPrMs = {p['long_pr_ms']}f;",
        "// Order: " + ", ".join(CLASS_NAMES),
        "constexpr float kClassThreshold[5] = {" + ", ".join(f"{th[c]}f" for c in CLASS_NAMES) + "};",
        f"constexpr bool kCdOrWideQrs = {'true' if p['cd_or_wide_qrs'] else 'false'};",
        "}  // namespace params",
        "}  // namespace ecg",
        "",
        "#endif  // ECG_PIPELINE_PARAMS_H_",
        "",
    ]
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- evaluate
def run_pipeline(adc, p, cnn):
    """Batch version of the per-window logic. Returns a DataFrame."""
    feats, n_peaks, quality = analyze_features(adc)
    reasons = np.array([gate(a, f, n, q, p) for a, f, n, q in zip(adc, feats, n_peaks, quality)])
    flags = np.array([rhythm_flags(f, p) for f in feats])
    q_in = np.stack([cnn_input_int8(a, cnn.scale, cnn.zero_point) for a in adc])
    probs = cnn.run(q_in)  # computed for all windows; only used where the gate passes
    dec = np.array([decide(pr, fl, p) for pr, fl in zip(probs, flags)])
    df = pd.DataFrame(probs, columns=[f"p_{c}" for c in CLASS_NAMES])
    df["quality"], df["rhythm_flags"], df["decision"] = reasons, flags, dec
    df["template_corr"] = [q["template_corr"] for q in quality]
    for name in ("mean_hr", "rr_cv", "pr_ms", "qrs_ms"):
        df[name] = feats[:, FI[name]]
    return df, q_in


def sens_spec(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp, fn = np.sum(pred & truth), np.sum(~pred & truth)
    tn, fp = np.sum(~pred & ~truth), np.sum(pred & ~truth)
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp / max(tp + fp, 1), int(truth.sum())


def evaluate():
    from noise_stress import NOISE_TYPES, add_noise
    p = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    cache, ids, rhythm = labels("test")
    clean_mv = cache["X_test"][:, :, 0].astype(np.float64)
    y = cache["y_test"]
    cnn = Int8Cnn()
    md = ["# Orchestrator evaluation (fold 10, Lead I, INT8 CNN)", "",
          "Generated by `python scripts/orchestrator.py evaluate`. Parameters: "
          "`results/lead1/orchestrator_params.json` (fitted on folds 1-9 only).", ""]

    clean, _ = run_pipeline(to_adc(clean_mv), p, cnn)
    ok = clean.quality.to_numpy() == 0
    probs = clean[[f"p_{c}" for c in CLASS_NAMES]].to_numpy()
    md += ["## Clean PTB-XL windows", "",
           f"- Gate pass rate: **{ok.mean() * 100:.1f} %** ({(~ok).sum()} of {len(ok)} rejected)."]
    for bit, name in [(Q_FLAT, "flat"), (Q_SATURATED, "saturated"), (Q_NO_RHYTHM, "no rhythm"),
                      (Q_NO_TEMPLATE, "no template"), (Q_NOISY, "noisy (template corr)"),
                      (Q_BASELINE, "baseline wander")]:
        n = int(np.sum(clean.quality.to_numpy() & bit > 0))
        if n:
            md.append(f"  - {name}: {n}")
    md += [f"- Macro AUC, all windows: {roc_auc_score(y, probs, average='macro'):.4f}; "
           f"gate-passed only: {roc_auc_score(y[ok], probs[ok], average='macro'):.4f}.", ""]

    dec = clean.decision.to_numpy()
    md += ["### Class decisions (gate-passed windows)", "",
           "| Class | Threshold | F1 (fitted threshold) | F1 (0.5) | Sensitivity | Specificity |",
           "|---|---:|---:|---:|---:|---:|"]
    for k, c in enumerate(CLASS_NAMES):
        pred = (dec[ok] >> k) & 1
        se, sp, _, _ = sens_spec(pred, y[ok, k])
        md.append(f"| {c} | {p['class_thresholds'][c]:.2f} | {f1_score(y[ok, k], pred):.3f} | "
                  f"{f1_score(y[ok, k], probs[ok, k] >= 0.5):.3f} | {se:.3f} | {sp:.3f} |")
    f1_fit = f1_score(y[ok], np.stack([(dec[ok] >> k) & 1 for k in range(5)], 1), average="macro")
    md += ["", f"Macro F1: {f1_fit:.3f} with fitted thresholds vs "
           f"{f1_score(y[ok], (probs[ok] >= 0.5).astype(int), average='macro'):.3f} at 0.5. "
           f"`cd_or_wide_qrs` = {p['cd_or_wide_qrs']}.", ""]

    ref = reference_intervals(ids)
    flags = clean["rhythm_flags"].to_numpy()
    md += ["### Rhythm and interval flags vs PTB-XL statements (all clean windows)", "",
           "| Flag | Rule | Reference | n positive | Sensitivity | Specificity | PPV |",
           "|---|---|---|---:|---:|---:|---:|"]
    for bit, name, rule, truth, tname in [
            (F_TACHY, "TACHY", f"HR > {p['tachy_bpm']:.0f}", rhythm["STACH"], "STACH"),
            (F_BRADY, "BRADY", f"HR < {p['brady_bpm']:.0f}", rhythm["SBRAD"], "SBRAD"),
            (F_IRREGULAR, "IRREGULAR", f"RR CV >= {p['irregular_rr_cv']}", rhythm["AFIB"], "AFIB/AFLT"),
            (F_AF_SUSPECT, "AF_SUSPECT", "IRREGULAR and no P wave", rhythm["AFIB"], "AFIB/AFLT"),
            (F_WIDE_QRS, "WIDE_QRS", f"QRS >= {p['wide_qrs_ms']:.0f} ms", (ref.QRS >= 120).to_numpy(), "12SL QRS >= 120"),
            (F_LONG_PR, "LONG_PR", f"PR > {p['long_pr_ms']:.0f} ms", rhythm["1AVB"], "1AVB")]:
        se, sp, ppv, npos = sens_spec(flags & bit, truth)
        md.append(f"| {name} | {rule} | {tname} | {npos} | {se:.3f} | {sp:.3f} | {ppv:.3f} |")

    md += ["", "## Noise stress test (MIT-BIH NSTDB, evaluation half of each noise record)", "",
           "SNR is relative to the clean window's power. 'Rejected' = gate fails. AUC on passed "
           "windows shows what the device would report.", "",
           "| Noise | SNR (dB) | Rejected | Macro AUC, all | Macro AUC, passed |", "|---|---:|---:|---:|---:|"]
    rng = np.random.default_rng(10)
    for kind in NOISE_TYPES:
        for snr in (12, 6, 0, -6):
            noisy, _ = run_pipeline(to_adc(add_noise(clean_mv, kind, snr, rng, half=1)), p, cnn)
            okn = noisy.quality.to_numpy() == 0
            pr = noisy[[f"p_{c}" for c in CLASS_NAMES]].to_numpy()
            auc_pass = roc_auc_score(y[okn], pr[okn], average="macro") if okn.sum() > 50 else float("nan")
            md.append(f"| {kind} | {snr:+d} | {(~okn).mean() * 100:.1f} % | "
                      f"{roc_auc_score(y, pr, average='macro'):.4f} | {auc_pass:.4f} |")
    out = P.ROOT / "results" / "lead1" / "orchestrator_eval.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "evaluate"])
    a = ap.parse_args()
    fit() if a.command == "fit" else evaluate()
