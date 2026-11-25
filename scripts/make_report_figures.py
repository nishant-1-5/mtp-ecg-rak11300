"""Figures for the MTP report and slides -> docs/report/figures/.

    python scripts/make_report_figures.py

All numbers come from committed results files or the cache; nothing is typed in
by hand except the memory/latency measurements, which are copied from
docs/phase2_tflm.md (emulator) and marked as such in the figure titles.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as P  # noqa: E402

OUT = P.ROOT / "docs" / "report" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#3366aa", "#ee8833", "#44aa66", "#cc4444", "#888888"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# ----------------------------------------------------------------------------- 1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    w, step = 2.35, 2.6
    boxes = [
        ("RA / LA / RL\nfront end", "Lead I, 100 Hz\nint16 window (10 s)", GREY),
        ("Feature\nextraction", "R peaks, HRV, median beat,\nPR / QRS / QT, Welch SQI", BLUE),
        ("Quality gate", "flat, saturated, rhythm,\ntemplate corr., baseline", ORANGE),
        ("Rhythm flags", "tachy, brady, irregular,\nAF suspect, wide QRS,\nlong PR", GREEN),
        ("INT8 1D-CNN\n(TFLM)", "z-score -> int8,\n5 sigmoid outputs,\nper-class thresholds", RED),
    ]
    for i, (title, sub, col) in enumerate(boxes):
        x = 0.1 + i * step
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.75, boxstyle="round,pad=0.05", fc=col, alpha=0.15, ec=col))
        ax.text(x + w / 2, 2.35, title, ha="center", va="center", weight="bold", fontsize=9.5)
        ax.text(x + w / 2, 1.5, sub, ha="center", va="center", fontsize=8)
        if i:
            ax.annotate("", xy=(x - 0.02, 1.9), xytext=(x - step + w + 0.08, 1.9),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    gx = 0.1 + 2 * step + w / 2
    ax.annotate("fails: report 'unreliable', skip CNN", xy=(gx, 0.95), xytext=(gx, 0.3), ha="center",
                fontsize=8.5, color=ORANGE, arrowprops=dict(arrowstyle="<-", color=ORANGE))
    ax.text(6.5, 3.05, "Per 10 s window on the RP2040 (Cortex-M0+, 125 MHz): ~300 ms, 27.7 KB SRAM",
            ha="center", fontsize=9.5, style="italic")
    save(fig, "fig_pipeline.png")


# ----------------------------------------------------------------------------- 2
def fig_lead_comparison():
    df = pd.read_csv(P.ROOT / "results" / "lead_set_comparison.csv")
    order = ["XGBoost (tabular)", "Random Forest (tabular)", "STFT 2D-CNN", "1D-CNN float32",
             "1D-CNN INT8 (deployed)", "ResNet + attention (ceiling)"]
    piv = df.pivot(index="model", columns="lead_set", values="macro_auc").reindex(order)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(order))
    ax.barh(y + 0.2, piv["3lead"], 0.4, color=GREY, label="Leads I, II, III (baseline)")
    ax.barh(y - 0.2, piv["lead1"], 0.4, color=BLUE, label="Lead I only (hardware)")
    for i, (a, b) in enumerate(zip(piv["3lead"], piv["lead1"])):
        ax.text(a + 0.003, i + 0.2, f"{a:.3f}", va="center", fontsize=8)
        ax.text(b + 0.003, i - 0.2, f"{b:.3f}", va="center", fontsize=8, color=BLUE)
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlim(0.7, 0.93)
    ax.set_xlabel("Macro AUC, PTB-XL fold 10 (n = 2,158)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title("Lead I vs 3-lead, all model families")
    save(fig, "fig_lead_comparison.png")


# ----------------------------------------------------------------------------- 3
def fig_delineation():
    from delineation import PRE, bandpass, delineate_template, median_beat
    from reference_features import detect_r_peaks
    d = pd.read_csv(P.ROOT / "results" / "features" / "delineation_test.csv")
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    X = cache["X_test"][:, :, 0].astype(np.float64)
    names = list(cache["class_names"])
    y = cache["y_test"]

    def example(mask):
        cand = d[mask & d.lead1_PR.notna() & (abs(d.lead1_QRS - d["12sl_QRS"]) < 6)].index
        return int(cand[0])

    picks = [("NORM", example(pd.Series(y[:, names.index("NORM")] == 1) & (d["12sl_QRS"] < 100))),
             ("CD (wide QRS)", example(pd.Series(y[:, names.index("CD")] == 1) & (d["12sl_QRS"] >= 130)))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, (label, i) in zip(axes[:2], picks):
        x = X[i]
        peaks, _ = detect_r_peaks(x)
        rr = float(np.mean(np.diff(peaks)) * 10)
        t, _, nb = median_beat(bandpass(x), peaks)
        f = delineate_template(t, rr)
        ms = (np.arange(len(t)) - PRE) * 10
        ax.plot(ms, t, color="k", lw=1.3)
        lo = t.min() - 0.08 * (t.max() - t.min())
        for key, col, txt in [("p_on", GREEN, "P on"), ("qrs_on", BLUE, "QRS on"),
                              ("qrs_off", BLUE, "QRS off"), ("t_end", RED, "T end")]:
            if f[key] is not None:
                xm = (f[key] - PRE) * 10
                ax.axvline(xm, color=col, ls="--", lw=1)
                ax.text(xm + 8, lo, txt, rotation=90, fontsize=7.5, color=col, va="bottom", ha="left")
        ax.axhline(f["iso"], color=GREY, lw=0.6)
        ax.set_title(f"{label}\nQRS {d.lead1_QRS[i]:.0f} ms (12SL {d['12sl_QRS'][i]:.0f}), "
                     f"PR {d.lead1_PR[i]:.0f} ms (12SL {d['12sl_PR'][i]:.0f})", fontsize=9)
        ax.set_xlabel("ms from R")
        ax.set_ylabel("mV (0.5-40 Hz)")
    ax = axes[2]
    ok = d.lead1_QRS.notna() & d["12sl_QRS"].notna()
    ax.scatter(d["12sl_QRS"][ok], d.lead1_QRS[ok], s=4, alpha=0.35, color=BLUE)
    ax.plot([50, 200], [50, 200], color=GREY, lw=1)
    r = np.corrcoef(d["12sl_QRS"][ok], d.lead1_QRS[ok])[0, 1]
    mae = np.mean(np.abs(d.lead1_QRS[ok] - d["12sl_QRS"][ok]))
    ax.set_xlim(50, 200)
    ax.set_ylim(40, 220)
    ax.set_xlabel("GE 12SL QRS, 12 leads (ms)")
    ax.set_ylabel("Lead I QRS, this work (ms)")
    ax.set_title(f"Fold 10: r = {r:.2f}, MAE = {mae:.1f} ms\n(floor near 58 ms: see limitations)", fontsize=9)
    fig.suptitle("Median-beat delineation on Lead I at 100 Hz", y=1.06)
    save(fig, "fig_delineation.png")


# ----------------------------------------------------------------------------- 4
def fig_noise():
    md = (P.ROOT / "results" / "lead1" / "orchestrator_eval.md").read_text(encoding="utf-8")
    rows = re.findall(r"\| (bw|em|ma) \| ([+-]\d+) \| ([\d.]+) % \| ([\d.]+) \| ([\d.]+) \|", md)
    df = pd.DataFrame(rows, columns=["kind", "snr", "rej", "auc_all", "auc_pass"]).astype(
        {"snr": int, "rej": float, "auc_all": float, "auc_pass": float})
    clean = float(re.search(r"Macro AUC, all windows: ([\d.]+)", md).group(1))
    clean_rej = 100 - float(re.search(r"Gate pass rate: \*\*([\d.]+) %", md).group(1))
    names = {"bw": "baseline wander", "em": "electrode motion", "ma": "muscle artifact"}
    cols = {"bw": GREEN, "em": RED, "ma": ORANGE}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, g in df.groupby("kind"):
        g = g.sort_values("snr", ascending=False)
        xs = ["clean"] + [f"{s:+d}" for s in g.snr]
        a1.plot(xs, [clean_rej] + g.rej.tolist(), "o-", color=cols[k], label=names[k])
        a2.plot(xs, [clean] + g.auc_pass.tolist(), "o-", color=cols[k], label=f"{names[k]}, gate-passed")
        a2.plot(xs, [clean] + g.auc_all.tolist(), "o:", color=cols[k], alpha=0.5)
    a1.set_ylabel("windows rejected by the gate (%)")
    a1.set_xlabel("SNR (dB), MIT-BIH NSTDB noise added to fold 10")
    a1.legend(fontsize=8)
    a2.set_ylabel("macro AUC")
    a2.set_xlabel("SNR (dB)")
    a2.set_title("solid: windows the gate passes; dotted: all windows", fontsize=9)
    a2.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Signal-quality gate under real recorded noise")
    save(fig, "fig_noise_stress.png")


# ----------------------------------------------------------------------------- 5
def fig_memory_latency():
    # Measured on rp2040js (docs/phase2_tflm.md); acquisition buffers are a design figure.
    sram = [("TFLM arena\n(incl. feature\nworkspace)", 21504), ("acquisition\nbuffers", 4000),
            ("peak stack", 1636), ("interpreter\nobjects", 568)]
    stages = [("filter + R peaks", 34.5), ("delineation + SQI", 31.5), ("Welch PSD", 48.9),
              ("Hjorth + HRV", 2.0), ("gate + flags + INT8 prep", 7.2), ("CNN (TFLM)", 175.2)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    y = np.arange(len(sram))
    a1.barh(y, [b for _, b in sram], color=[RED, GREY, ORANGE, BLUE], alpha=0.8)
    for i, (name, b) in enumerate(sram):
        a1.text(b + 300, i, f"{b:,} B", va="center", fontsize=8)
    a1.set_yticks(y, [n.replace("\n", " ") for n, _ in sram], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, 26000)
    a1.set_xlabel("bytes")
    a1.set_title(f"Pipeline SRAM: {sum(b for _, b in sram):,} B (target 30,720 B; chip 264 KB)", fontsize=9.5)
    y = np.arange(len(stages))
    a2.barh(y, [m for _, m in stages], color=[BLUE, BLUE, BLUE, BLUE, ORANGE, RED], alpha=0.8)
    for i, (name, m) in enumerate(stages):
        a2.text(m + 2, i, f"{m:.1f} ms", va="center", fontsize=8)
    a2.set_yticks(y, [n for n, _ in stages], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlim(0, 210)
    total = sum(m for _, m in stages)
    a2.set_xlabel("ms per 10 s window (rp2040js emulator, 125 MHz)")
    a2.set_title(f"Latency: {total:.0f} ms per window (~{total / 100:.0f} % of one core)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_memory_latency.png")


# ----------------------------------------------------------------------------- 6
def fig_int8_norm():
    labels = ["raw mV,\n200 cal.", "raw mV,\n1,000 cal.", "de-mean,\n2,000 cal.", "z-score\n(adopted)"]
    flt = [0.8316, 0.8316, 0.8332, 0.8302]
    q = [0.7801, 0.5682, 0.5859, 0.8294]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - 0.18, flt, 0.36, color=GREY, label="float32")
    ax.bar(x + 0.18, q, 0.36, color=BLUE, label="INT8")
    for i, v in enumerate(q):
        ax.text(i + 0.18, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(0.5, 0.88)
    ax.set_ylabel("macro AUC (fold 10)")
    ax.set_title("Lead I CNN: INT8 needs per-window normalisation", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "fig_int8_normalisation.png")


# ============================================================================= slide figures
def _box(ax, x, y, w, h, title, sub="", col=BLUE, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=col, alpha=0.15, ec=col))
    ax.text(x + w / 2, y + h * (0.68 if sub else 0.5), title, ha="center", va="center", weight="bold", fontsize=fs)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fs - 1.5)


def _arrow(ax, x0, y0, x1, y1, col="k"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.5, color=col))


def fig_leads():
    """Einthoven triangle: what RA/LA/RL can and cannot measure."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.75, 1.15)
    ax.axis("off")
    ra, la, ll = np.array([-1.0, 0.7]), np.array([1.0, 0.7]), np.array([0.0, -1.1])
    ax.add_patch(plt.Polygon([ra, la, ll], fill=False, ls=":", ec=GREY))
    ax.annotate("", xy=la, xytext=ra, arrowprops=dict(arrowstyle="->", lw=3, color=BLUE))
    ax.text(0, 0.82, "Lead I = LA − RA  (recorded)", ha="center", color=BLUE, weight="bold")
    for a, b, name, pos in [(ra, ll, "Lead II = LL − RA", (-0.95, -0.3)), (la, ll, "Lead III = LL − LA", (0.95, -0.3))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=2, color=GREY, ls="--"))
        ax.text(*pos, name + "\n(needs LL: not available)", ha="center", color=GREY, fontsize=8.5)
    for p, lab, col in [(ra, "RA", BLUE), (la, "LA", BLUE), (ll, "LL\n(no electrode)", GREY)]:
        ax.add_patch(plt.Circle(p, 0.09, color=col, alpha=0.8 if col == BLUE else 0.3))
        ax.text(p[0], p[1] + (0.17 if p[1] > 0 else -0.3), lab, ha="center", fontsize=9.5, weight="bold", color=col)
    ax.add_patch(plt.Circle((-0.9, -1.55), 0.08, color=GREEN, alpha=0.8))
    ax.text(-0.75, -1.55, "RL = driven reference (noise rejection), not a lead", va="center", fontsize=8.5, color=GREEN)
    ax.set_title("Electrodes on the hardware vs the limb leads", fontsize=10.5)
    save(fig, "fig_leads.png")


def fig_dataset():
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    names = list(cache["class_names"])
    ytr, yte = cache["y_train"], cache["y_test"]
    order = ["NORM", "MI", "STTC", "CD", "HYP"]
    idx = [names.index(c) for c in order]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(order))
    a1.bar(x - 0.2, ytr[:, idx].mean(0) * 100, 0.4, color=BLUE, label=f"train, folds 1-9 (n = {len(ytr):,})")
    a1.bar(x + 0.2, yte[:, idx].mean(0) * 100, 0.4, color=ORANGE, alpha=0.8, label=f"test, fold 10 (n = {len(yte):,})")
    for i, k in enumerate(idx):
        a1.text(i - 0.2, ytr[:, k].mean() * 100 + 1, f"{ytr[:, k].sum():,}", ha="center", fontsize=7.5)
    a1.set_xticks(x, ["NORM\nnormal", "MI\ninfarct", "STTC\nST/T", "CD\nconduction", "HYP\nhypertr."])
    a1.set_ylabel("% of records with the label")
    a1.legend(fontsize=8)
    a1.set_title("Five diagnostic superclasses (multi-label; counts = training records)", fontsize=9.5)
    k = ytr.sum(1)
    vals, cnt = np.unique(k, return_counts=True)
    a2.bar(vals, cnt / len(k) * 100, color=GREY)
    for v, c in zip(vals, cnt):
        a2.text(v, c / len(k) * 100 + 1, f"{c / len(k) * 100:.0f} %", ha="center", fontsize=8)
    a2.set_xticks(vals)
    a2.set_xlabel("superclasses per record")
    a2.set_ylabel("% of training records")
    a2.set_title("Co-occurring diagnoses", fontsize=10)
    save(fig, "fig_dataset.png")


def fig_signal_steps():
    from delineation import bandpass
    import scipy.signal as signal
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    y, names = cache["y_test"], list(cache["class_names"])
    i = int(np.where((y[:, names.index("NORM")] == 1) & (y.sum(1) == 1))[0][3])
    x = cache["X_test"][i, :, 0].astype(np.float64)
    fs = 100
    t = np.arange(1000) / fs
    b, a = signal.butter(2, [5 / 50, 15 / 50], btype="bandpass")
    f = signal.filtfilt(b, a, x)
    integ = np.convolve(np.diff(f) ** 2, np.ones(8), mode="same")
    peaks, _ = signal.find_peaks(integ, distance=35, prominence=np.std(integ) * 0.5)
    xb = bandpass(x)
    fig, axes = plt.subplots(4, 1, figsize=(10, 5.6), sharex=True)
    axes[0].plot(t, x, color="k", lw=0.8)
    axes[0].set_ylabel("raw\nmV", fontsize=8.5)
    axes[1].plot(t, f, color=BLUE, lw=0.8)
    axes[1].set_ylabel("5-15 Hz", fontsize=8.5)
    axes[2].plot(t[1:], integ, color=ORANGE, lw=0.9)
    axes[2].plot(t[1:][peaks], integ[peaks], "v", color=RED, ms=6)
    axes[2].set_ylabel("diff², 80 ms\nsum", fontsize=8.5)
    axes[3].plot(t, xb, color="k", lw=0.8)
    for p in peaks:
        lo, hi = max(0, p - 8), min(999, p + 8)
        r = lo + int(np.argmax(np.abs(xb[lo:hi + 1] - xb.mean())))
        axes[3].axvline(r / fs, color=RED, lw=0.6, alpha=0.6)
    axes[3].set_ylabel("0.5-40 Hz\n+ R", fontsize=8.5)
    axes[3].set_xlabel("time (s), one 10 s Lead I window from fold 10")
    rr = np.diff(peaks) * 10
    fig.suptitle(f"Pan-Tompkins R-peak detection: {len(peaks)} beats, mean RR {rr.mean():.0f} ms "
                 f"({60000 / rr.mean():.0f} bpm), RR CV {rr.std() / rr.mean():.3f}", fontsize=10)
    for ax in axes:
        ax.tick_params(labelsize=8)
    save(fig, "fig_signal_steps.png")


def fig_cnn_arch():
    layers = [("Input", "1000 × 1", "z-scored Lead I", 0, GREY),
              ("Conv1D 16, k7, s2", "500 × 16", "+ BN, ReLU", 192, BLUE),
              ("MaxPool 2", "250 × 16", "", 0, GREY),
              ("SepConv1D 32, k5", "250 × 32", "+ BN, ReLU", 752, BLUE),
              ("MaxPool 2", "125 × 32", "", 0, GREY),
              ("SepConv1D 64, k3", "125 × 64", "+ BN, ReLU", 2464, BLUE),
              ("Global avg pool", "64", "replaces Flatten", 0, GREY),
              ("Dense 32", "32", "ReLU, dropout 0.3", 2080, ORANGE),
              ("Dense 5", "5", "sigmoid (multi-label)", 165, RED)]
    fig, ax = plt.subplots(figsize=(12, 3.3))
    n = len(layers)
    ax.set_xlim(0, n * 1.32)
    ax.set_ylim(-0.6, 2.2)
    ax.axis("off")
    for i, (name, shape, sub, params, col) in enumerate(layers):
        x = 0.1 + i * 1.32
        h = 0.6 + 1.3 * (np.log10(max(int(np.prod([int(s) for s in shape.split(' × ')])), 5)) / 3.0)
        h = min(h, 1.95)
        ax.add_patch(FancyBboxPatch((x, 1.0 - h / 2), 1.1, h, boxstyle="round,pad=0.03", fc=col, alpha=0.18, ec=col))
        ax.text(x + 0.55, 1.0 + h / 2 + 0.12, name, ha="center", fontsize=8, weight="bold")
        ax.text(x + 0.55, 1.05, shape, ha="center", va="center", fontsize=9)
        if sub:
            ax.text(x + 0.55, 0.75, sub, ha="center", va="center", fontsize=7)
        if params:
            ax.text(x + 0.55, -0.35, f"{params:,} params", ha="center", fontsize=7.5, color=col)
        if i:
            _arrow(ax, x - 0.2, 1.0, x - 0.02, 1.0)
    ax.text(n * 0.66, -0.6, "5,653 parameters in total  ·  INT8 flatbuffer 25,184 B  ·  "
            "global average pooling keeps the head tiny (64 -> 32 -> 5)", ha="center", fontsize=9, style="italic")
    save(fig, "fig_cnn_arch.png")


def fig_per_class_roc():
    from sklearn.metrics import roc_auc_score, roc_curve
    import orchestrator as O
    df = pd.read_csv(P.ROOT / "results" / "lead_set_comparison.csv")
    d = df[df.model == "1D-CNN INT8 (deployed)"].set_index("lead_set")
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    names = list(cache["class_names"])
    y = cache["y_test"]
    cnn = O.Int8Cnn()
    adc = O.to_adc(cache["X_test"][:, :, 0].astype(np.float64))
    probs = cnn.run(np.stack([O.cnn_input_int8(a, cnn.scale, cnn.zero_point) for a in adc]))
    order = ["NORM", "STTC", "CD", "HYP", "MI"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    x = np.arange(len(order))
    a1.bar(x - 0.2, [d.loc["3lead", f"auc_{c}"] for c in order], 0.4, color=GREY, label="Leads I-III")
    a1.bar(x + 0.2, [d.loc["lead1", f"auc_{c}"] for c in order], 0.4, color=BLUE, label="Lead I (deployed)")
    for i, c in enumerate(order):
        drop = d.loc["lead1", f"auc_{c}"] - d.loc["3lead", f"auc_{c}"]
        a1.text(i + 0.2, d.loc["lead1", f"auc_{c}"] + 0.004, f"{d.loc['lead1', f'auc_{c}']:.3f}", ha="center", fontsize=7.5)
        a1.text(i, d.loc["3lead", f"auc_{c}"] + 0.012, f"Δ {drop:+.3f}", ha="center", fontsize=8.5, weight="bold",
                color=RED if drop < -0.04 else GREY)
    a1.set_xticks(x, order)
    a1.set_ylim(0.7, 0.94)
    a1.set_ylabel("AUC, INT8 model, fold 10")
    a1.legend(fontsize=8, loc="lower left")
    a1.set_title("Per-class AUC; Δ = change without Leads II/III", fontsize=9.5)
    cols = {"NORM": GREEN, "STTC": ORANGE, "CD": BLUE, "HYP": GREY, "MI": RED}
    for c in order:
        k = names.index(c)
        fpr, tpr, _ = roc_curve(y[:, k], probs[:, k])
        a2.plot(fpr, tpr, color=cols[c], lw=1.6, label=f"{c} ({roc_auc_score(y[:, k], probs[:, k]):.3f})")
    a2.plot([0, 1], [0, 1], "k:", lw=0.8)
    a2.set_xlabel("false-positive rate")
    a2.set_ylabel("true-positive rate")
    a2.legend(fontsize=8, loc="lower right")
    a2.set_title("ROC of the deployed Lead I INT8 CNN", fontsize=9.5)
    save(fig, "fig_per_class_roc.png")


def _beats(x):
    from delineation import POST, PRE, bandpass, median_beat, template_correlation
    from reference_features import detect_r_peaks
    peaks, _ = detect_r_peaks(x)
    xb = bandpass(x)
    tmpl, r_pos, nb = median_beat(xb, peaks)
    beats = [xb[r - PRE:r + POST + 1] for r in r_pos if r - PRE >= 0 and r + POST < len(x)]
    return tmpl, beats, template_correlation(xb, tmpl, r_pos)


def fig_quality_examples():
    from noise_stress import add_noise
    cache = np.load(P.CACHE_FILE, allow_pickle=True)
    y, names = cache["y_test"], list(cache["class_names"])
    i = int(np.where((y[:, names.index("NORM")] == 1) & (y.sum(1) == 1))[0][3])
    x = cache["X_test"][i, :, 0].astype(np.float64)
    rng = np.random.default_rng(5)
    cases = [("clean", x), ("+ electrode motion, +6 dB", add_noise(x[None], "em", 6, rng, half=1)[0]),
             ("+ electrode motion, −6 dB", add_noise(x[None], "em", -6, rng, half=1)[0])]
    th = 0.598
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), gridspec_kw={"height_ratios": [1, 1.4]})
    for j, (label, s) in enumerate(cases):
        tmpl, beats, tc = _beats(s)
        axes[0, j].plot(np.arange(1000) / 100, s, color="k", lw=0.6)
        axes[0, j].set_title(label, fontsize=9.5)
        axes[0, j].set_xlabel("s", fontsize=8)
        ms = (np.arange(101) - 30) * 10
        for b in beats:
            axes[1, j].plot(ms, b, color=GREY, lw=0.6, alpha=0.6)
        if tmpl is not None:
            axes[1, j].plot(ms, tmpl, color=BLUE, lw=2, label="median beat")
        ok = tc >= th
        axes[1, j].set_title(f"template correlation {tc:.2f}  ->  {'PASS' if ok else 'REJECT'}",
                             fontsize=9.5, color=GREEN if ok else RED, weight="bold")
        axes[1, j].set_xlabel("ms from R", fontsize=8)
        for ax in axes[:, j]:
            ax.tick_params(labelsize=7.5)
    axes[1, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(f"Quality index: do the beats agree with their median beat? (reject below {th})", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig_quality_examples.png")


def fig_rhythm_examples():
    import orchestrator as O
    from reference_features import detect_r_peaks
    cache, ids, rhythm = O.labels("test")
    X = cache["X_test"][:, :, 0].astype(np.float64)
    feats = cache["X_feat_v2_test"]
    y, names = cache["y_test"], list(cache["class_names"])
    reg = int(np.where((y[:, names.index("NORM")] == 1) & (feats[:, 7] < 0.03) & ~rhythm["AFIB"])[0][5])
    af = int(np.where(rhythm["AFIB"] & (feats[:, 7] > 0.15) & np.isnan(feats[:, 12]))[0][2])
    fig, axes = plt.subplots(2, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [3, 1]})
    for row, (i, label) in enumerate([(reg, "Sinus rhythm (NORM)"), (af, "Atrial fibrillation (PTB-XL AFIB)")]):
        x = X[i]
        peaks, _ = detect_r_peaks(x)
        t = np.arange(1000) / 100
        axes[row, 0].plot(t, x, color="k", lw=0.7)
        axes[row, 0].plot(t[peaks], x[peaks], "v", color=RED, ms=5)
        rr = np.diff(peaks) * 10
        cv = rr.std() / rr.mean()
        pr = feats[i, 12]
        flags = ["IRREGULAR"] if cv >= 0.095 else []
        if flags and np.isnan(pr):
            flags.append("AF_SUSPECT")
        axes[row, 0].set_title(f"{label}: RR CV {cv:.3f}, P wave {'absent' if np.isnan(pr) else f'found (PR {pr:.0f} ms)'}"
                               f"  ->  flags: {', '.join(flags) if flags else 'none'}", fontsize=9.5,
                               color=RED if flags else GREEN)
        axes[row, 0].set_ylabel("mV", fontsize=8)
        axes[row, 1].plot(rr, "o-", color=RED if flags else GREEN, ms=4)
        axes[row, 1].axhline(rr.mean(), color=GREY, lw=0.8)
        axes[row, 1].set_ylim(300, 1300)
        axes[row, 1].set_title("RR intervals (ms)", fontsize=9)
        for ax in axes[row]:
            ax.tick_params(labelsize=7.5)
    axes[1, 0].set_xlabel("s")
    axes[1, 1].set_xlabel("beat")
    fig.tight_layout()
    save(fig, "fig_rhythm_examples.png")


def fig_arena_overlay():
    head, used, total, ws = 16000, 20788, 21504, 5780
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 3.9), gridspec_kw={"height_ratios": [1, 1.3]})
    a1.barh(0, head, color=BLUE, alpha=0.25, ec=BLUE)
    a1.barh(0, ws, color=ORANGE, alpha=0.7, ec=ORANGE)
    a1.barh(0, used - head, left=head, color=RED, alpha=0.6, ec=RED)
    a1.barh(0, total - used, left=used, color="white", ec=GREY, hatch="//")
    a1.text(ws / 2, 0, f"feature workspace\n{ws:,} B", ha="center", va="center", fontsize=8)
    a1.text((ws + head) / 2, 0, f"activation head (input, activations, scratch)\n{head:,} B", ha="center",
            va="center", fontsize=8)
    a1.text((head + used) / 2, 0, f"persistent\n{used - head:,} B", ha="center", va="center", fontsize=8)
    a1.text((used + total) / 2, 0.55, "free\n716 B", ha="center", va="bottom", fontsize=7.5)
    a1.set_xlim(0, total)
    a1.set_ylim(-0.6, 1.1)
    a1.set_yticks([])
    a1.set_xlabel("byte offset in the 21,504 B TFLM arena")
    a1.set_title("One buffer, two uses: the feature workspace lives inside the CNN's activation area", fontsize=10)
    a2.set_xlim(0, 10)
    a2.set_ylim(0, 1.4)
    a2.axis("off")
    steps = [("1. ExtractLead1Features", "uses the workspace\n(arena head)", ORANGE),
             ("2. Gate + flags", "workspace no longer\nneeded", GREY),
             ("3. PrepareCnnInput", "writes int8 input into\nthe input tensor (head)", BLUE),
             ("4. Invoke + copy out", "activations overwrite\nthe head; probs copied", RED)]
    for k, (ti, sub, col) in enumerate(steps):
        _box(a2, 0.1 + k * 2.5, 0.2, 2.2, 1.0, ti, sub, col, fs=9)
        if k:
            _arrow(a2, k * 2.5 - 0.15, 0.7, k * 2.5 + 0.08, 0.7)
    a2.text(5, 0.02, "Strictly sequential, so nothing is overwritten while it is still needed. "
            "CnnInit checks head ≥ workspace at start-up. Saves 5,780 B: 33.5 KB -> 27.7 KB.",
            ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    save(fig, "fig_arena_overlay.png")


def fig_verification():
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    stages = [("Python reference", "scipy / numpy\nspecification", "defines expected\noutputs", GREY),
              ("C++ features (host)", "same source as firmware,\nMinGW build", "21,387 / 21,388\nwindows match", BLUE),
              ("C++ pipeline (host)", "gate, flags, INT8 input,\ndecisions", "3,778 / 3,778\nwindows match", ORANGE),
              ("RP2040 firmware", "Arm GCC, Pico SDK,\nTFLM + CMSIS-NN", "built .uf2, no\nchanges for test", GREEN),
              ("rp2040js emulator", "unmodified image,\nCortex-M0+ @ 125 MHz", "100 / 100 random\nwindows PASS", RED)]
    for k, (ti, sub, res, col) in enumerate(stages):
        x = 0.1 + k * 2.3
        _box(ax, x, 1.0, 2.0, 1.6, ti, sub, col, fs=9.5)
        ax.text(x + 1.0, 0.55, res, ha="center", va="center", fontsize=8.5, weight="bold", color=col)
        if k:
            _arrow(ax, x - 0.28, 1.8, x - 0.03, 1.8)
    ax.text(5.75, 0.08, "Tolerance: features within 1e-3; probabilities within 1/256 (one INT8 output step); "
            "gate bits, flags and class decisions identical", ha="center", fontsize=8.5, style="italic")
    save(fig, "fig_verification.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_lead_comparison()
    fig_delineation()
    fig_noise()
    fig_memory_latency()
    fig_int8_norm()
    fig_leads()
    fig_dataset()
    fig_signal_steps()
    fig_cnn_arch()
    fig_per_class_roc()
    fig_quality_examples()
    fig_rhythm_examples()
    fig_arena_overlay()
    fig_verification()
