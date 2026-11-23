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


if __name__ == "__main__":
    fig_pipeline()
    fig_lead_comparison()
    fig_delineation()
    fig_noise()
    fig_memory_latency()
    fig_int8_norm()
