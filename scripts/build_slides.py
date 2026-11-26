"""Build docs/presentation/final_presentation.pptx on the mid-term deck's theme.

    python scripts/make_report_figures.py   # figures first
    python scripts/build_slides.py

Opens docs/presentation/midterm_presentation.pptx (theme, masters, layouts),
removes its slides and adds the final deck. The mid-term file is not modified.
Every slide gets speaker notes. Numbers are copied from docs/ and results/.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "presentation" / "midterm_presentation.pptx"
OUT = ROOT / "docs" / "presentation" / "final_presentation.pptx"
FIG = ROOT / "docs" / "report" / "figures"

TITLE, BODY, TITLE_ONLY = 0, 2, 4  # layout indices in the mid-term template
DARK = RGBColor(0x26, 0x26, 0x26)
GREY = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0x33, 0x66, 0xAA)
BAND = RGBColor(0xEE, 0xF2, 0xF8)
TOP = Inches(1.25)      # content starts below the title
BOTTOM = Inches(5.45)   # usable content bottom


class Deck:
    def __init__(self):
        self.prs = Presentation(str(SRC))
        ids = self.prs.slides._sldIdLst
        for sld in list(ids):
            self.prs.part.drop_rel(sld.rId)
            ids.remove(sld)
        self.W, self.H = self.prs.slide_width, self.prs.slide_height

    # ------------------------------------------------------------------ primitives
    def slide(self, title, notes, layout=TITLE_ONLY):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[layout])
        if title is not None:
            s.shapes.title.text = title
        s.notes_slide.notes_text_frame.text = notes
        return s

    @staticmethod
    def text(s, left, top, width, height, items, size=16, color=DARK, bullet=True, bold_first=False):
        """items: strings; leading '  ' = sub-bullet; '**bold**' anywhere in a line is bold."""
        tb = s.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        for k, item in enumerate(items):
            level = 0
            while item.startswith("  "):
                level, item = level + 1, item[2:]
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            parts = item.split("**")
            if bullet:
                parts[0] = ("•  " if level == 0 else "–  ") + parts[0]
            for j, part in enumerate(parts):
                if not part:
                    continue
                r = p.add_run()
                r.text = part
                r.font.size = Pt(size - 2 * level)
                r.font.color.rgb = color if level == 0 else GREY
                r.font.bold = (j % 2 == 1) or (bold_first and k == 0)
            p.space_after = Pt(6 if level == 0 else 3)
            if bullet:  # hanging indent so wrapped lines align with the text, not the bullet
                ind = Inches(0.25)
                ppr = p._p.get_or_add_pPr()
                ppr.set("marL", str(int(ind + Inches(0.35) * level)))
                ppr.set("indent", str(-int(ind)))
        return tb

    def takeaway(self, s, msg, top=None, size=15):
        top = top if top is not None else BOTTOM - Inches(0.55)
        box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), top, self.W - Inches(0.8), Inches(0.52))
        box.fill.solid()
        box.fill.fore_color.rgb = BAND
        box.line.fill.background()
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), top, Inches(0.08), Inches(0.52))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        r = p.add_run()
        r.text = msg
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = DARK

    def picture(self, s, image, left, top, max_w, max_h, center=True):
        pic = s.shapes.add_picture(str(image), left, top)
        k = min(max_w / pic.width, max_h / pic.height)
        pic.width, pic.height = int(pic.width * k), int(pic.height * k)
        if center:
            pic.left = int(left + (max_w - pic.width) / 2)
            pic.top = int(top + (max_h - pic.height) / 2)
        return pic

    # ------------------------------------------------------------------ slide types
    def figure(self, title, image, takeaway, notes, caption=None):
        s = self.slide(title, notes)
        h = BOTTOM - TOP - Inches(0.7) - (Inches(0.35) if caption else 0)
        self.picture(s, FIG / image, Inches(0.3), TOP, self.W - Inches(0.6), h)
        if caption:
            self.text(s, Inches(0.5), TOP + h, self.W - Inches(1.0), Inches(0.35), [caption], size=12,
                      color=GREY, bullet=False)
        self.takeaway(s, takeaway)
        return s

    def split(self, title, image, items, notes, takeaway=None, img_frac=0.56, size=16, img_left=False):
        s = self.slide(title, notes)
        bottom = BOTTOM - (Inches(0.65) if takeaway else 0)
        img_w = int((self.W - Inches(0.8)) * img_frac)
        txt_w = self.W - Inches(0.8) - img_w - Inches(0.2)
        if img_left:
            self.picture(s, FIG / image, Inches(0.4), TOP, img_w, bottom - TOP)
            self.text(s, Inches(0.4) + img_w + Inches(0.2), TOP, txt_w, bottom - TOP, items, size)
        else:
            self.text(s, Inches(0.4), TOP, txt_w, bottom - TOP, items, size)
            self.picture(s, FIG / image, Inches(0.4) + txt_w + Inches(0.2), TOP, img_w, bottom - TOP)
        if takeaway:
            self.takeaway(s, takeaway)
        return s

    def bullets(self, title, items, notes, takeaway=None, size=18):
        s = self.slide(title, notes)
        bottom = BOTTOM - (Inches(0.65) if takeaway else 0)
        self.text(s, Inches(0.5), TOP, self.W - Inches(1.0), bottom - TOP, items, size)
        if takeaway:
            self.takeaway(s, takeaway)
        return s

    def table(self, title, header, rows, notes, col_w, takeaway=None, size=13, row_h=Inches(0.36)):
        s = self.slide(title, notes)
        width = self.W - Inches(0.8)
        shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.4), TOP, width, row_h * (len(rows) + 1))
        t = shape.table
        tot = sum(col_w)
        for j, w in enumerate(col_w):
            t.columns[j].width = Emu(int(width * w / tot))
        for i, row in enumerate([header] + rows):
            for j, v in enumerate(row):
                cell = t.cell(i, j)
                cell.text = str(v)
                for p in cell.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(size)
                        r.font.bold = i == 0
                        if i:
                            r.font.color.rgb = DARK
        if takeaway:
            self.takeaway(s, takeaway)
        return s

    def section(self, title, subtitle, notes):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[1])  # SECTION_HEADER
        s.shapes.title.text = title
        t = s.shapes.title
        # subtitle in the band above the title placeholder (which sits low on this layout)
        self.text(s, t.left, Inches(1.0), t.width, Inches(0.8), [subtitle], size=22,
                  color=RGBColor(0xFF, 0xFF, 0xFF), bullet=False)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self):
        self.prs.save(str(OUT))
        print(f"wrote {OUT} ({len(self.prs.slides)} slides)")


def main():
    d = Deck()

    # 1 ------------------------------------------------------------------ title
    s = d.prs.slides.add_slide(d.prs.slide_layouts[TITLE])
    s.shapes.title.text = "Single-Lead ECG Screening on the RP2040"
    s.placeholders[1].text = ("Multi-label cardiac pathology classification on a RAK11300\n"
                              "Lakshit Sachdeva (B23146)  ·  Nishant Nehra (B23163)\nMentor: Dr. Varun Dutt")
    s.notes_slide.notes_text_frame.text = (
        "We built a complete ECG screening pipeline that runs on a low-cost microcontroller: it reads one ECG lead, "
        "checks whether the signal can be trusted, computes clinical measurements and classifies five diagnostic "
        "groups. Everything shown today is reproducible from our public repository.")

    # 2 ------------------------------------------------------------------ motivation
    d.bullets("Why single-lead screening", [
        "Cardiovascular disease is the leading cause of death; many abnormalities are intermittent",
        "Diagnostic ECG = 12-lead machine in a clinic: bulky, stationary, occasional",
        "Wearables can watch continuously, but they record one lead and are noisy",
        "**What a useful wearable must do on-device**",
        "  classify the rhythm and morphology it can see",
        "  know when its own signal is unreliable",
        "  report findings a clinician can check (rate, rhythm, QRS width)",
        "  fit in kilobytes of memory and a few % of a slow CPU",
    ], "Motivation: continuous monitoring needs on-device analysis because streaming raw ECG over LoRa is not "
       "practical. The four requirements on this slide structure the rest of the talk.",
       takeaway="Goal: screen, flag and self-check on a microcontroller, not replace a 12-lead ECG")

    # 3 ------------------------------------------------------------------ hardware / leads
    d.split("Hardware: what the device can measure", "fig_leads.png", [
        "**RAK11300 module**",
        "  RP2040: 2 × Cortex-M0+, 125 MHz",
        "  264 KB SRAM, 2 MB flash, no FPU",
        "  SX1262 LoRa radio",
        "**Front end: RA, LA, RL electrodes**",
        "  RA–LA difference = Lead I",
        "  no LL electrode -> no Lead II / III",
        "**Our budget:** < 30 KB SRAM for the whole ECG pipeline",
    ], "The electrode set decides what the models may use. RL is the driven-right-leg reference, so only Lead I is "
       "available. The mid-term models used Leads I, II and III, which this hardware cannot provide; that is why "
       "everything was retrained on Lead I. The 30 KB budget is self-imposed so the LoRa stack and both cores' "
       "stacks have room.", takeaway="Three electrodes give exactly one lead: every model must work on Lead I alone",
       img_frac=0.5)

    # 4 ------------------------------------------------------------------ since mid-term
    d.table("Since the mid-term: corrections", ["Mid-term claim", "What we found", "Now"],
            [["3-lead CNN, AUC 0.8626", "Hardware has Lead I only; 0.8626 was the Keras validation split",
              "Lead I models; test-fold AUC 0.829"],
             ["Target ESP32, 45 ms, 520 KB", "Device is an RP2040 (264 KB, no FPU)", "Measured: 175 ms CNN, 27.7 KB SRAM"],
             ["'QRS > 120 ms isolates CD'", "Plot was mislabeled: it showed RR range", "Real QRS: CD AUC 0.754"],
             ["PR / QT / QRS features", "Measured the R wave (QT median 0.8 ms)", "Median-beat delineation"],
             ["Model only", "No quality check, no interpretable output", "Quality gate + rhythm flags"]],
            "We want to be upfront: three mid-term results did not hold up. The lead mismatch made the old model "
            "unusable on the hardware; the AUC was a validation-split number; the QRS plot had shifted column labels. "
            "All originals are kept in the repository, marked, so the history is transparent.",
            col_w=[2.2, 3.0, 2.4], size=12, row_h=Inches(0.5),
            takeaway="All three issues are fixed and the original artifacts are kept, marked, in the repository")

    # 5 ------------------------------------------------------------------ data
    d.split("Data: PTB-XL and two references", "fig_dataset.png", [
        "**PTB-XL v1.0.3:** 21,799 ten-second 12-lead ECGs, 18,869 patients",
        "  100 Hz -> 1,000 samples per window; we use Lead I",
        "  patient-disjoint folds: train 1-9, test fold 10 only",
        "**PTB-XL+:** GE 12SL interval measurements (reference for PR/QRS/QT)",
        "**MIT-BIH Noise Stress Test DB:** real electrode-motion, muscle and baseline-wander noise",
    ], "Fold 10 was never used for training, tuning or picking thresholds. 24 % of records carry more than one "
       "superclass, so this is a multi-label problem with independent sigmoid outputs. PTB-XL+ gives a commercial "
       "12-lead algorithm's intervals, which we use as a reference, not as training labels.",
       takeaway="Every number in this talk is on held-out fold 10 (n = 2,158)", img_frac=0.55, size=14)

    # 6 ------------------------------------------------------------------ system
    d.figure("System on the device", "fig_pipeline.png",
             "Unreliable windows are reported as such and never classified",
             "Per 10 s window: features first (cheap and interpretable), then a quality gate. Only windows that pass "
             "go to the CNN. Rhythm flags are always computed. The whole chain is C++ with no dynamic memory, "
             "and the CNN runs under TensorFlow Lite Micro.",
             caption="Feature extraction and decision logic: C++14, no heap. CNN: TensorFlow Lite Micro with CMSIS-NN kernels.")

    d.section("1. Signals", "R peaks, delineation, rhythm and quality",
              "First the classical signal-processing part, which provides measurements and the quality check.")

    # 7 ------------------------------------------------------------------ R peaks
    d.figure("Finding the beats (Pan-Tompkins)", "fig_signal_steps.png",
             "Peak positions drive heart rate, RR variability and the median beat",
             "Raw Lead I, then a 5-15 Hz band-pass that keeps QRS energy, then the squared derivative summed over 80 ms. "
             "Peaks closer than 350 ms are suppressed and small ones removed by prominence. For delineation we "
             "re-filter at 0.5-40 Hz and move each peak to the true R wave. The C++ reproduces scipy exactly, "
             "including filtfilt padding.")

    # 8 ------------------------------------------------------------------ delineation
    d.figure("PR / QRS / QT from a median beat", "fig_delineation.png",
             "Lead I QRS separates conduction disturbances as well as 12-lead 12SL (AUC 0.754 vs 0.755)",
             "All beats in the window are aligned on R and the sample-wise median removes noise. QRS edges come from "
             "slope thresholds, T end from the tangent method, P onset at half the P amplitude. Against GE 12SL on "
             "fold 10: QRS MAE 13.9 ms, r 0.69; PR r 0.67 but about 24 ms short; QT is weakest at r 0.52. The old "
             "feature had CD AUC 0.48, i.e. no information. Fixing it raised Lead I XGBoost from 0.744 to 0.786.")

    # 9 ------------------------------------------------------------------ rhythm
    d.figure("Rhythm flags: interpretable findings", "fig_rhythm_examples.png",
             "IRREGULAR: sensitivity 0.91 for AF; with 'no P wave' added, PPV rises to 0.82",
             "Flags use textbook limits: tachy > 100 bpm, brady < 60, wide QRS >= 120 ms, long PR > 200 ms. "
             "The irregularity threshold (RR coefficient of variation 0.095) was chosen on training folds. "
             "AF_SUSPECT requires both irregular RR and no detectable P wave, trading sensitivity (0.55) for "
             "specificity (0.99).")

    # 10 ----------------------------------------------------------------- quality
    d.figure("Quality gate: trust this window?", "fig_quality_examples.png",
             "Clean: 97.4 % pass  ·  −6 dB electrode motion: 82.5 % rejected",
             "The key check: correlate each beat with the median beat. Clean beats agree (0.96); heavy electrode "
             "motion destroys the agreement (0.41). The threshold 0.598 is the 2 % quantile of clean training "
             "windows. Other checks: flat line, ADC saturation, implausible heart rate, too few beats, baseline wander. "
             "SNR-type indices did not separate PTB-XL's own noise annotations (AUC about 0.5), which is why we "
             "use beat agreement and test with real recorded noise.")

    d.section("2. Classifier", "1D-CNN, quantisation and results",
              "Next, the neural network and how well it works on one lead.")

    # 11 ----------------------------------------------------------------- CNN
    d.figure("A 5.7k-parameter 1D-CNN", "fig_cnn_arch.png",
             "Small enough for the RP2040: 25 KB of flash, 20.8 KB of working memory",
             "A strided convolution halves the length immediately, two depthwise-separable blocks extract morphology, "
             "and global average pooling replaces a large dense layer. Five independent sigmoid outputs allow "
             "co-occurring diagnoses. Trained with binary cross-entropy and early stopping on validation AUC.")

    # 12 ----------------------------------------------------------------- INT8
    d.split("INT8 quantisation needs normalisation", "fig_int8_normalisation.png", [
        "INT8 = 256 levels for the whole input range",
        "Raw mV: range must cover rare ±7 mV windows",
        "  normal ECG gets too few levels -> AUC 0.78",
        "  more calibration data made it worse (0.57)",
        "**Fix: z-score every window**",
        "  (x − mean) / (std + 0.001 mV)",
        "  INT8 0.829 vs float 0.830",
        "  also removes front-end gain differences",
    ], "This was the most important deployment detail. Quantising raw millivolts works for the 3-lead model but "
       "loses 0.05 AUC on Lead I. Per-window normalisation fixes it almost completely. The firmware computes "
       "mean and std from exact 64-bit integer sums of the ADC counts.",
       takeaway="Quantisation cost: 0.05 AUC -> under 0.001 AUC", img_frac=0.55, size=15)

    # 13 ----------------------------------------------------------------- lead comparison
    d.split("Lead I vs three leads", "fig_lead_comparison.png", [
        "Same models retrained on Lead I only",
        "~0.05 AUC lost with the same features, in every model family",
        "**On Lead I, the tiny CNN = the ceiling**",
        "  1D-CNN 0.830 vs ResNet + attention 0.831",
        "  5.7k vs 2.5M parameters",
        "Tabular models recover most of their loss once the intervals work (0.786)",
    ], "We kept the 3-lead models for comparison. The drop is consistent across families. The striking result is "
       "that a 440 times larger network gains nothing on one lead: the information, not the model, is the limit.",
       takeaway="Deployed INT8 model: macro AUC 0.829 on Lead I", img_frac=0.6, size=15)

    # 13b ---------------------------------------------------------------- ablation
    d.figure("Ablation: what each lead set is worth", "fig_ablation.png",
             "12 → 1 lead costs 0.078 AUROC; 50 Hz costs < 0.01; aVR/aVL/aVF add nothing",
             "One fixed compact separable CNN (about 2.8k parameters), trained identically on every input, fold 10, "
             "with 95 % bootstrap confidence intervals. 12 leads reach 0.881. Six limb leads equal three because "
             "aVR, aVL and aVF are computed from I and II. Chest leads alone are no better than limb leads; the "
             "12-lead gain comes from combining both. Lead I is clearly separated from every multi-lead set. "
             "Halving the sampling rate barely matters, so 100 Hz is not the bottleneck. Per class, MI and CD lose "
             "most; NORM and STTC hold up.")

    d.table("Ablation: design implications",
            ["Question", "Ablation result", "Design consequence"],
            [["More leads?", "12 lead 0.881 · 3 limb 0.855 · Lead I 0.803", "Hardware limits us; report the gap honestly"],
             ["Which multi-lead set?", "6 limb = 3 limb (0.854 vs 0.855)", "If an LL electrode is added, 3 leads are enough"],
             ["Sampling rate?", "Lead I: 100 Hz 0.803 · 50 Hz 0.794", "100 Hz is adequate; 50 Hz is a fallback"],
             ["Simple features?", "5 waveform stats: 0.61-0.63 on Lead I", "Use clinical features (XGBoost 0.786)"],
             ["Which CNN on Lead I?", "compact 2.8k: 0.803 · ours 5.7k: 0.830", "Keep the deployed CNN (25 KB INT8)"]],
            "Turning the ablation into decisions. The cheapest hardware upgrade with the largest gain would be a fourth "
            "electrode (LL), which gives Leads II and III and reaches the 3-limb result.",
            col_w=[1.6, 2.9, 2.7], size=12, row_h=Inches(0.55),
            takeaway="Cheapest upgrade: an LL electrode (3 limb leads, +0.05 AUROC)")

    # 14 ----------------------------------------------------------------- per class
    d.figure("Which classes lose most", "fig_per_class_roc.png",
             "Rhythm and repolarisation survive; MI loses most (−0.06), as infarct localisation needs spatial leads",
             "NORM and STTC lose about 0.015. CD, HYP and MI lose 0.04 to 0.06. MI is expected: infarcts are localised "
             "by comparing leads. These are INT8 numbers computed with the same kernels the device uses.")

    # 15 ----------------------------------------------------------------- decisions table
    d.table("From probabilities to decisions", ["Output", "Rule", "Sensitivity", "Specificity", "PPV / F1"],
            [["NORM (CNN)", "p ≥ 0.42", "0.954", "0.627", "F1 0.792"],
             ["STTC (CNN)", "p ≥ 0.38", "0.686", "0.871", "F1 0.657"],
             ["CD (CNN)", "p ≥ 0.24", "0.534", "0.919", "F1 0.589"],
             ["MI (CNN)", "p ≥ 0.24", "0.606", "0.814", "F1 0.559"],
             ["HYP (CNN)", "p ≥ 0.24", "0.462", "0.902", "F1 0.423"],
             ["TACHY", "HR > 100 bpm", "0.939", "0.960", "PPV 0.481"],
             ["IRREGULAR (vs AF)", "RR CV ≥ 0.095", "0.905", "0.868", "PPV 0.333"],
             ["WIDE_QRS (vs 12SL)", "QRS ≥ 120 ms", "0.776", "0.964", "PPV 0.652"]],
            "Thresholds were fitted to maximise F1 on training windows the CNN never trained on. That raises macro F1 "
            "from 0.515 at a flat 0.5 to 0.604. Low PPV for TACHY is partly labelling: PTB-XL codes sinus "
            "tachycardia only, so fast AF counts as a false positive.",
            col_w=[2.2, 1.8, 1.2, 1.2, 1.3], size=13, row_h=Inches(0.37),
            takeaway="Fitted thresholds: macro F1 0.604 (0.515 at a flat 0.5)")

    # 16 ----------------------------------------------------------------- noise
    d.figure("Robustness to real recorded noise", "fig_noise_stress.png",
             "Rejection rises with noise, and the windows that pass keep higher accuracy",
             "NSTDB noise was added to fold 10 at four SNRs. Baseline wander and muscle noise barely affect the CNN "
             "until -6 dB. Electrode motion, the typical wearable artifact, hurts most; the gate rejects it most "
             "aggressively. At -6 dB the windows that pass score 0.694 instead of 0.653.")

    d.section("3. On the device", "Firmware, memory, speed and verification",
              "Finally, the embedded implementation. The board was not available, so we ran the exact firmware image on an "
              "RP2040 emulator.")

    # 17 ----------------------------------------------------------------- verification
    d.figure("How we know the firmware is right", "fig_verification.png",
             "Same results from Python to the emulated RP2040, window by window",
             "Each stage is compared with the previous one on real data. The firmware images contain Python's "
             "expected outputs and print PASS or FAIL themselves. The emulator (rp2040js) runs the unmodified .uf2 "
             "file. 100 randomly drawn test windows passed end to end, including the neural network.")

    # 18 ----------------------------------------------------------------- arena overlay
    d.figure("Fitting in 30 KB: share one buffer", "fig_arena_overlay.png",
             "Pipeline SRAM 27.7 KB instead of 33.5 KB",
             "TFLM keeps activations at the start of its arena and permanent data at the end. Features are finished "
             "before the CNN input is written, so the 5.8 KB feature workspace can live in the activation area. "
             "The firmware asks TFLM's allocator for the activation size at start-up and refuses to run if the "
             "workspace would not fit, so a future model cannot break this silently.")

    # 19 ----------------------------------------------------------------- memory & latency
    d.figure("Memory and latency at 125 MHz", "fig_memory_latency.png",
             "~300 ms per 10 s window = about 3 % of one core; core 1 stays free",
             "Measured on the emulator: arena 20,788 B used, stack peak 1,636 B. The CNN is 175 ms and the features "
             "about 120 ms, of which the Welch spectrum is the largest part. The emulator counts instruction cycles "
             "but not flash-cache misses, so real hardware will be somewhat slower; still far within the 10 s budget.")

    # 20 ----------------------------------------------------------------- RP2040-only bugs
    d.bullets("Bugs only the RP2040 revealed", [
        "**NaN is not IEEE on the RP2040 fast float routines**",
        "  no P wave -> PR should be NaN; the device returned −inf",
        "  fix: test the bit pattern, never compute with NaN",
        "**Desktop TFLite rounds differently from the device**",
        "  XNNPACK vs CMSIS-NN: up to 9 output steps apart -> 9 % false 'failures'",
        "  fix: evaluate with TFLite's reference kernels (match within 1 step)",
        "  headline AUC changed by < 0.001",
    ], "Both bugs were invisible in host tests and showed up only when the real firmware image ran on the emulated "
       "chip. This is the main argument for testing the actual binary, not just the algorithm.",
       takeaway="Testing the real binary found bugs that host tests could not")

    # 21 ----------------------------------------------------------------- summary table
    d.table("Implementation summary",
            ["Item", "Value"],
            [["Deployed model", "INT8 1D-CNN, 5,653 params, 25,184 B, macro AUC 0.829"],
             ["Feature + decision code", "C++14, no heap, 13.9 KB flash"],
             ["TFLM + CMSIS-NN runtime", "107 KB flash, 10 operators, single core"],
             ["Pipeline SRAM", "27,708 B (target 30,720 B)"],
             ["Latency per 10 s window", "features 112-126 ms, CNN 175 ms, total ~300 ms"],
             ["Verified", "21,387 / 21,388 feature windows; 3,778 / 3,778 decisions; 100 / 100 on RP2040"],
             ["Open source", "github.com/nishant-1-5/mtp-ecg-rak11300"]],
            "One-slide summary of the engineering result. Flash total including the USB stack and test vectors is 234 "
            "KB of the 2 MB.", col_w=[2.2, 5.0], size=14, row_h=Inches(0.42))

    # 22 ----------------------------------------------------------------- limitations
    d.bullets("Limitations", [
        "One public dataset recorded with clinical equipment; one held-out fold",
        "Timing is emulated: flash-cache misses not modelled; power not measured",
        "100 Hz limits QRS resolution (floor near 58 ms for ~4 % of windows)",
        "Lead I QT agrees only moderately with 12SL (r = 0.52); not used for flags",
        "Lead I cannot localise infarcts: MI AUC 0.785 vs 0.867 with three leads",
        "Screening prototype: clinical use needs prospective validation and regulatory review",
    ], size=20, notes="We list these explicitly. The biggest open question is how the gate and model behave on our own front end, "
       "which may have a different noise floor than PTB-XL's clinical recordings.")

    # 23 ----------------------------------------------------------------- next steps
    d.bullets("Next steps", [
        "**Hardware:** flash rak11300_pipeline.uf2 on any RP2040 board -> real latency and power",
        "**Front end:** stream ADC samples, set scale and rails, record lab data",
        "  re-check gate thresholds; recalibrate or fine-tune if the noise floor differs",
        "**Algorithms:** bias-corrected PR threshold (~176 ms); faster real-input FFT",
        "**System:** send findings over LoRa (SX1262)",
    ], size=20, notes="The firmware for the hardware test already exists and checks itself, so the first step is short once a "
       "board is available.")

    # 24 ----------------------------------------------------------------- thanks
    s = d.prs.slides.add_slide(d.prs.slide_layouts[TITLE])
    s.shapes.title.text = "Thank you"
    s.placeholders[1].text = "Code, models and results:\ngithub.com/nishant-1-5/mtp-ecg-rak11300"
    s.notes_slide.notes_text_frame.text = "Questions."

    d.save()


if __name__ == "__main__":
    main()
