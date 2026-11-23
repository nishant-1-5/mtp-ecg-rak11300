"""Build docs/presentation/final_presentation.pptx from the mid-term deck's theme.

    python scripts/build_slides.py

Opens docs/presentation/midterm_presentation.pptx (theme, masters, layouts),
removes its slides, and adds the final-results deck. The mid-term file itself is
not modified. Figures come from docs/report/figures/ (make_report_figures.py).
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "presentation" / "midterm_presentation.pptx"
OUT = ROOT / "docs" / "presentation" / "final_presentation.pptx"
FIG = ROOT / "docs" / "report" / "figures"
RES = ROOT / "results"

TITLE, BODY, TITLE_ONLY = 0, 2, 4  # layout indices in the mid-term template


def clear_slides(prs):
    ids = prs.slides._sldIdLst
    for sld in list(ids):
        prs.part.drop_rel(sld.rId)
        ids.remove(sld)


def bullets(slide, items, size=14):
    body = slide.placeholders[1].text_frame
    body.clear()
    first = True
    for item in items:
        level = 0
        while item.startswith("  "):
            level += 1
            item = item[2:]
        p = body.paragraphs[0] if first else body.add_paragraph()
        first = False
        p.text = item
        p.level = level
        for r in p.runs:
            r.font.size = Pt(size - 2 * level)


def text_slide(prs, title, items, size=18):
    s = prs.slides.add_slide(prs.slide_layouts[BODY])
    s.shapes.title.text = title
    bullets(s, items, size)
    return s


def figure_slide(prs, title, image, note=None, top=Inches(1.3), max_h=Inches(3.9)):
    s = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
    s.shapes.title.text = title
    W = prs.slide_width
    max_h = min(max_h, prs.slide_height - top - Inches(0.75))  # leave room for the note
    pic = s.shapes.add_picture(str(image), 0, top)
    scale = min((W - Inches(0.6)) / pic.width, max_h / pic.height)
    pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
    pic.left = int((W - pic.width) / 2)
    if note:
        tb = s.shapes.add_textbox(Inches(0.4), pic.top + pic.height + Inches(0.05), W - Inches(0.8), Inches(0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = note
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.size = Pt(12)
    return s


def table_slide(prs, title, header, rows, col_w=None, note=None, size=11):
    s = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
    s.shapes.title.text = title
    W = prs.slide_width
    n_rows, n_cols = len(rows) + 1, len(header)
    width = W - Inches(0.8)
    shape = s.shapes.add_table(n_rows, n_cols, Inches(0.4), Inches(1.35), width, Inches(0.3) * n_rows)
    t = shape.table
    if col_w:
        tot = sum(col_w)
        for j, w in enumerate(col_w):
            t.columns[j].width = Emu(int(width * w / tot))
    for j, h in enumerate(header):
        t.cell(0, j).text = h
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            t.cell(i, j).text = str(v)
    for i in range(n_rows):
        for j in range(n_cols):
            for p in t.cell(i, j).text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    if i > 0:
                        r.font.color.rgb = RGBColor(0x26, 0x26, 0x26)
                    r.font.bold = i == 0 or (row_bold(rows, i))
    if note:
        tb = s.shapes.add_textbox(Inches(0.4), Inches(1.4) + Inches(0.3) * n_rows + Inches(0.1),
                                  W - Inches(0.8), Inches(0.8))
        tb.text_frame.word_wrap = True
        tb.text_frame.text = note
        for p in tb.text_frame.paragraphs:
            for r in p.runs:
                r.font.size = Pt(11)
    return s


def row_bold(rows, i):
    return str(rows[i - 1][0]).startswith("*")


def main():
    prs = Presentation(str(SRC))
    clear_slides(prs)

    s = prs.slides.add_slide(prs.slide_layouts[TITLE])
    s.shapes.title.text = "Single-Lead ECG Screening on the RP2040"
    s.placeholders[1].text = ("Multi-label cardiac pathology classification at the edge\n"
                              "Lakshit Sachdeva (B23146), Nishant Nehra (B23163)\nMentor: Dr. Varun Dutt")

    text_slide(prs, "Problem and goal", [
        "Continuous cardiac monitoring needs cheap wearables; diagnostic ECG relies on 12-lead machines",
        "Goal: classify 10 s ECG windows into 5 PTB-XL superclasses on a microcontroller",
        "  NORM, MI (infarction), STTC (ST/T change), CD (conduction disturbance), HYP (hypertrophy)",
        "Device: RAK11300 = RP2040 (2 x Cortex-M0+, 125 MHz, 264 KB SRAM, no FPU) + LoRa",
        "Front end: RA, LA, RL electrodes  ->  Lead I only",
        "Self-imposed budget: < 30 KB SRAM for the ECG pipeline",
    ])

    text_slide(prs, "Since the mid-term", [
        "Hardware has Lead I only: the 3-lead CNN could not be fed  ->  all models retrained on Lead I",
        "Device is an RP2040, not an ESP32: memory and timing re-measured",
        "Interval features were broken (QT median 0.8 ms; 'QRS' ~96 ms for every class)",
        "  replaced by median-beat delineation, validated against GE 12SL (PTB-XL+)",
        "'QRS > 120 ms isolates CD' came from a mislabeled plot (it showed RR range)",
        "'0.8626 AUC' was the Keras validation split; test-fold numbers are reported now",
        "Added: signal-quality gate, rhythm flags, C++ firmware, TFLM on the RP2040",
    ], size=16)

    figure_slide(prs, "System on the device", FIG / "fig_pipeline.png",
                 "Features -> quality gate -> rhythm flags -> INT8 CNN. Unreliable windows are reported, "
                 "not classified.", max_h=Inches(2.6))

    figure_slide(prs, "Lead I vs three leads", FIG / "fig_lead_comparison.png",
                 "Same feature definitions: ~0.05 AUC lost for every model family. On Lead I the 5.7k-parameter "
                 "CNN matches a 2.5M-parameter ResNet + attention (0.830 vs 0.831).")

    figure_slide(prs, "INT8 needs normalisation", FIG / "fig_int8_normalisation.png",
                 "Raw-mV input: the INT8 range must cover rare +-7 mV windows -> AUC 0.78. "
                 "Z-scoring each window: INT8 0.829 vs float 0.830. Also removes front-end gain dependence.",
                 max_h=Inches(3.4))

    figure_slide(prs, "PR / QRS / QT delineation", FIG / "fig_delineation.png",
                 "QRS vs GE 12SL (12 leads): MAE 13.9 ms, r = 0.69. Lead I QRS separates CD with AUC 0.754 "
                 "(12SL: 0.755; old feature: 0.483). Lead I XGBoost: 0.744 -> 0.786.", max_h=Inches(3.5))

    table_slide(prs, "Decisions and rhythm flags",
                ["Output", "Rule", "Sensitivity", "Specificity", "PPV / F1"],
                [["NORM (CNN)", "p >= 0.42", "0.954", "0.627", "F1 0.792"],
                 ["CD (CNN)", "p >= 0.24", "0.534", "0.919", "F1 0.589"],
                 ["MI (CNN)", "p >= 0.24", "0.606", "0.814", "F1 0.559"],
                 ["STTC (CNN)", "p >= 0.38", "0.686", "0.871", "F1 0.657"],
                 ["TACHY", "HR > 100", "0.939", "0.960", "PPV 0.481"],
                 ["IRREGULAR (vs AF)", "RR CV >= 0.095", "0.905", "0.868", "PPV 0.333"],
                 ["AF_SUSPECT", "irregular + no P", "0.551", "0.991", "PPV 0.818"],
                 ["WIDE_QRS", "QRS >= 120 ms", "0.776", "0.964", "PPV 0.652"]],
                col_w=[2.2, 2.2, 1.3, 1.3, 1.3],
                note="Class thresholds fitted on training windows the CNN never saw. Macro F1 0.604 "
                     "(0.515 at a flat 0.5).")

    figure_slide(prs, "Quality gate under noise", FIG / "fig_noise_stress.png",
                 "Gate passes 97.4 % of clean windows; rejects 82.5 % at -6 dB electrode motion. "
                 "Windows it passes keep higher AUC than the full set.", max_h=Inches(3.5))

    figure_slide(prs, "RP2040: memory and latency", FIG / "fig_memory_latency.png",
                 "Fold 10 (n = 2,158) unless noted. Unmodified firmware on the rp2040js emulator: 100/100 random test windows match Python "
                 "end-to-end incl. TFLM. Arena 20.8 KB, stack 1.6 KB, pipeline 27.7 KB SRAM. "
                 "~300 ms per 10 s window (~3 % of one core).", max_h=Inches(2.9))

    table_slide(prs, "Implementation summary",
                ["Item", "Value"],
                [["Deployed model", "INT8 1D-CNN, 5,653 params, 25,184 B, macro AUC 0.829"],
                 ["Feature code (C++14, no heap)", "13.9 KB flash; parity with Python on 21,387 / 21,388 windows"],
                 ["Decision logic parity", "3,778 / 3,778 windows (gate, flags, INT8 input, decisions)"],
                 ["TFLM + CMSIS-NN runtime", "107 KB flash, 10 operators, single core"],
                 ["Pipeline SRAM", "27,708 B (feature workspace overlaid on the TFLM arena)"],
                 ["Latency (emulated, 125 MHz)", "features 112-126 ms, CNN 175 ms, total ~300 ms"],
                 ["Issues found only on the RP2040", "SDK float routines do not propagate NaN; "
                                                     "XNNPACK vs CMSIS-NN rounding"]],
                col_w=[2.2, 5.0], size=11)

    text_slide(prs, "Limitations", [
        "Single public dataset (clinical recordings), one held-out fold",
        "Timing is emulated: flash-cache misses not modelled; power not measured",
        "100 Hz limits QRS resolution (floor near 58 ms); Lead I QT only moderately reliable (r = 0.52)",
        "Lead I cannot localise infarcts: MI AUC 0.785 vs 0.867 with three leads",
        "Screening prototype only: clinical use needs prospective validation and regulatory review",
    ])

    text_slide(prs, "Next steps", [
        "Flash rak11300_pipeline.uf2 on an RP2040 board: real latency and power",
        "Record with the lab front end; set ADC scale/rails; re-check gate thresholds on device data",
        "Bias-corrected PR threshold (~176 ms); longer windows for HRV",
        "Code, models, results: github.com/nishant-1-5/mtp-ecg-rak11300",
    ])

    s = prs.slides.add_slide(prs.slide_layouts[TITLE])
    s.shapes.title.text = "Thank you"
    s.placeholders[1].text = "github.com/nishant-1-5/mtp-ecg-rak11300"

    prs.save(str(OUT))
    print(f"wrote {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
