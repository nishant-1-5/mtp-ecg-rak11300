# Project Summary: Embedded Multi-Label ECG Diagnostic Model for RAK11300 (RP2040)

## 1. Project Objective
To develop and deploy a real-time, multi-label cardiac pathology detection system on a resource-constrained RAK11300 edge device (Raspberry Pi RP2040, dual Cortex-M0+, 264 KB SRAM, no FPU). The system utilizes a hybrid approach, combining classical numerical feature extraction (Lead I) with a TinyML 1D-CNN to classify ECG signals within an ultra-low memory footprint (self-imposed target: <30 KB SRAM for the ECG pipeline, <50 KB Flash). See `docs/sram_budget.md`.

> Earlier drafts targeted an ESP32. Latency figures quoted at 240 MHz below are ESP32-era estimates and must be re-measured on the RP2040.

---

## 2. Work Completed Till Now

### A. Dataset Selection & Data Pipeline
*   **Dataset Chosen:** PTB-XL (v1.0.3). 
*   **Optimization:** Selected the 100 Hz downsampled version to reduce the 10-second signal buffer size by 80% (from 5000 to 1000 samples), strictly aligning with the device SRAM budget.
*   **Taxonomy:** Formulated a multi-label classification pipeline targeting 5 clinical superclasses: Normal (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).
*   **Optimization Tooling:** Engineered a `.npz` caching mechanism to reduce data loading and preprocessing times from ~11 minutes to ~1-2 seconds.

### B. Feature Engineering & Validation (Lead I Focus)
*   **Feature Extraction:** Developed a dynamic extraction pipeline for 23 numerical features from Lead I, including HRV metrics (SDNN, RMSSD, pNN50), Morphological waves/intervals (Mean_PR, Mean_QRS, Mean_QTc), and Signal Quality Indices (SNR, Baseline Wander).
*   **Bug Resolution:** Identified and fixed a critical flaw where morphological intervals were hardcoded placeholders. Replaced them with a dynamic Pan-Tompkins based extraction algorithm.
*   **Clinical Correlation Proof:** Generated YData Profiling reports and correlation heatmaps proving:
    *   Lead I intervals (Mean_QRS > 120ms) strongly correlate with and isolate **Conduction Disturbances (CD)**.
    *   Heart rate variance features (Mean_HR, SDNN) effectively isolate **Normal Rhythms (NORM)** and Arrhythmias.
    *   Lead I has near-zero correlation with spatial ailments like **Myocardial Infarction (MI)**, providing mathematical justification for delegating these specific classes to a multi-lead CNN.

### C. Model Training & Quantization (CNN)
*   **Architecture:** Designed a lightweight 1D-CNN utilizing Depthwise Separable Convolutions and Global Average Pooling (GAP) to bypass memory-heavy dense layers.
*   **Performance:** Achieved a highly competitive Validation Macro-AUC of 0.8626 across the 5 superclasses.
*   **TinyML Optimization:** Applied Post-Training Quantization (PTQ) to convert the model to INT8. 
    *   **Flash Memory Size:** ~22 KB
    *   **SRAM (Tensor Arena):** ~18 KB
    *   **Inference Latency:** ~45 ms (at 240 MHz).
*   **Hardware Export:** Successfully exported the quantized model into a C-byte array header (`model_data.h`) for direct TensorFlow Lite Micro integration.

---

## 3. The Hybrid Workload Strategy
Based on the data and correlation artifacts, the team has agreed on a hybrid edge logic to maximize diagnostic accuracy while respecting hardware limits:
1.  **Gatekeeping & Triage (Numerical Features - Lead I):** The device will first calculate classical features to assess signal quality. If the signal is clean, it will use HRV and interval features (e.g., QRS widening) to immediately flag arrhythmias or Conduction Disturbances (CD).
2.  **Deep Morphological Analysis (1D-CNN - 3-Lead):** For structural heart conditions (MI, HYP, STTC) that require spatial vectors, the clean 3-lead data will be routed through the 22 KB INT8 CNN.

---

## 4. Next Steps & Goals to Achieve

### Phase 1: C++ Porting & Firmware Logic (Immediate)
Status and open issues: `docs/phase1_feature_port.md`.
*   [ ] **Port Feature Extraction to C++:** Translate the Python/SciPy feature extraction logic into embedded C++ (`firmware/ecg_features`).
    *   [x] Pan-Tompkins style R-peak detection (filtfilt, squared derivative, moving-window integration, scipy-equivalent `find_peaks`).
    *   [x] HRV metrics (HR stats, RR stats, SDNN, RMSSD, pNN50), Hjorth complexity, QRS envelope width.
    *   [x] Frequency-domain features (Baseline Wander, SNR) via an in-place 256-point FFT Welch PSD (`esp-dsp` does not apply to the RP2040).
    *   [x] Host parity test against the Python reference: 21,387 / 21,388 PTB-XL windows match (`scripts/compare_features.py`).
    *   [ ] PR / QT / QTc: Python implementation is broken, fix before porting.
*   [ ] **Implement Hybrid Decision Tree:** Code the orchestrator logic on the RAK11300 to run numerical feature extraction first, and conditionally trigger the CNN inference based on signal quality.
*   [ ] **Retrain the CNN on Lead I only:** the current model takes Leads I, II, III, but the RA/LA/RL front end provides Lead I only.

### Phase 2: On-Device TFLM Integration
*   [ ] **TensorFlow Lite Micro (TFLM) Setup:** Integrate the generated `model_data.h` and TFLM runtime into a Pico SDK (or Arduino-Pico) project for the RAK11300.
*   [ ] **Memory Profiling:** Run live inference on the RAK11300 and profile exact SRAM fragmentation, clock cycles, and power consumption.

### Phase 3: Domain Adaptation & Calibration
*   [ ] **Hardware Data Collection:** Collect a custom, live ECG dataset utilizing the specific RAK11300 analog frontend being used in the lab.
*   [ ] **Fine-Tuning / Recalibration:** Feed the live frontend data into the trained pipeline to correct for sensor-specific noise floors, analog baseline wander, and clock offsets before finalizing the firmware.

### Phase 4: Final Presentation & Documentation
*   [ ] **Finalize Slides:** Update the presentation deck with the latest hardware deployment metrics, power draw, and live-test accuracy.
*   [ ] **MTP Report:** Compile the methodology, dataset justifications, hardware diagrams, and correlation proofs into the final project report.