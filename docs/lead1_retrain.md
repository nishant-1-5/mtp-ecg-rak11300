# Lead I retrain

The RAK11300 front end (RA, LA, RL electrodes) records Lead I only, so every model was retrained on
PTB-XL Lead I. The 3-lead (I, II, III) models are kept unchanged for comparison:

- artifacts: `models/3lead/`, `results/3lead/`
- notebooks with their original outputs: git tag `v0-3lead`

## How to reproduce

```powershell
python scripts/run_notebook.py --lead-set lead1 notebooks/10_classical_xgb_rf.ipynb `
    notebooks/11_tinyml_1d_cnn.ipynb notebooks/12_stft_2d_cnn.ipynb `
    notebooks/13_resnet_attention.ipynb notebooks/20_quantization.ipynb notebooks/30_metrics.ipynb
python scripts/compare_lead_sets.py
```

`MTP_LEAD_SET` (set by `run_notebook.py`) selects the channels and feature columns in
`scripts/project.py`:

- **lead1:** channel 0, plus feature columns 0–23 (rhythm, morphology, signal quality and Lead I
  statistics).
- **3lead:** all 3 channels and all 36 feature columns.

Architectures and hyperparameters are unchanged. The Lead I runs set `tf.keras.utils.set_random_seed(42)`.

## Results (test fold 10, n = 2,158)

Full table: `results/lead_set_comparison.md`.

| Model | 3-lead | Lead I | Δ |
|---|---:|---:|---:|
| XGBoost (tabular) | 0.7952 | 0.7864 | −0.009 |
| Random Forest (tabular) | 0.7862 | 0.7780 | −0.008 |
| 1D-CNN float32 | 0.8832 | 0.8302 | −0.053 |
| **1D-CNN INT8 (deployed)** | 0.8652 | **0.8294** | −0.036 |
| STFT 2D-CNN | 0.8284 | 0.7851 | −0.043 |
| ResNet + attention (ceiling) | 0.8941 | 0.8310 | −0.063 |

The Lead I tabular models use the v2 features, with working PR/QRS/QT (`docs/delineation.md`). With
the original v1 features they scored 0.7443 (XGBoost) and 0.7321 (RF), the same ~0.05 drop as the
other models.

Deployed Lead I model, per class (INT8 AUC): CD 0.812, HYP 0.790, MI 0.787, NORM 0.887, STTC 0.870.

What this shows:

- **With identical features, losing Leads II/III costs about 0.05 macro AUC for every model type.**
  Fixing the interval features recovered most of that loss for the tabular models.
- **On Lead I, the 5.7k-parameter 1D-CNN matches the ResNet + attention ceiling** (0.8302 vs
  0.8310). The larger model has nothing more to extract from one lead, so there is no reason to
  deploy a bigger network.
- **MI loses the most** (3-lead 0.867 → Lead I 0.785, float), as the earlier correlation analysis
  predicted.

## Change: z-score input for the Lead I CNN

The first Lead I run fed raw mV into the CNN, as the 3-lead model did. Float AUC was 0.8316, but
**INT8 AUC was only 0.7801**. INT8 has 256 levels. Calibrating on 200 windows set the input range
to [−0.58, 1.87] mV. Calibrating on more data widened it to about ±7 mV to cover rare large
windows, which left too few levels for normal-amplitude ECGs (INT8 AUC 0.57–0.63).

Fix: the Lead I CNN now takes each window as `(x − mean) / (std + 0.001 mV)`, computed per window
(`project.normalize_cnn_input`, `CNN_INPUT_NORM = "zscore"`).

| Lead I 1D-CNN | float AUC | INT8 AUC |
|---|---:|---:|
| raw mV, 200 calibration windows | 0.8316 | 0.7801 |
| raw mV, 1,000 calibration windows | 0.8316 | 0.5682 |
| per-window de-mean only | 0.8332 | 0.59–0.68 |
| **per-window z-score (adopted)** | **0.8302** | **0.8294** (0.8262–0.8294 across calibration sets) |

Z-scoring also removes the dependence on the analog front end's absolute gain, which matters for
Phase 3. The firmware has to apply the same transform before quantising. `models/lead1/model_data.h`
records the convention together with `g_model_input_scale` (0.0541500449) and
`g_model_input_zero_point` (−25).

The 3-lead baseline still uses raw mV. Its INT8 model also loses 0.018 (0.8832 → 0.8652) for the
same reason.

## Model footprint (Lead I)

| | Value |
|---|---|
| Parameters | 5,653 |
| INT8 `.tflite` / `g_model_len` | 25,184 B |
| Input tensor | int8 [1, 1000, 1] |
| Tensor arena | not measured yet (Phase 2, TFLM `arena_used_bytes()`) |

## Notes on numbers quoted elsewhere

- The README's "Validation Macro-AUC 0.8626" for the 3-lead CNN was the Keras `validation_split`
  metric from training. On the held-out test fold the same model scores 0.8832 (float) and 0.8652
  (INT8). Quote the test-fold numbers above.
- `models/3lead/cnn_first_attempt_*` comes from `notebooks/archive/cnn_first_attempt.ipynb`, an
  earlier model without BatchNorm. It is not the 3-lead model in the comparison table.
- The Lead I ResNet stopped early after 10 epochs (patience 5), with its best epoch around epoch 5.
