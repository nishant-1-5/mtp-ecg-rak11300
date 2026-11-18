"""Lead I feature vs. superclass plots for the slides.

Fixed version of the original lead-1-just.py. The original took the first 8
cache columns but labelled them with the names below, so e.g. its "Mean_QRS"
box plot actually showed RR_Range (results/features/*_MISLABELED.png).
Columns are now selected by name from project.FEATURE_NAMES, from the v2
Lead I table (PR/QRS/QT from scripts/delineation.py).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from project import CACHE_FILE, FEATURE_NAMES, FEATURE_RESULTS_DIR

cache = np.load(CACHE_FILE, allow_pickle=True)
X_feat_train = cache["X_feat_v2_train"]  # 24 Lead I columns, names = FEATURE_NAMES[:24]
y_train = cache["y_train"]
class_names = cache["class_names"]  # ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Strictly Lead I features (display name -> cache column name)
lead_1_features = {
    "Mean_HR": "Mean_HR", "HR_Std": "HR_Std", "SDNN": "SDNN", "RMSSD": "RMSSD",
    "Hjorth_Complexity": "Sample_Entropy", "Mean_PR": "Mean_PR",
    "Mean_QRS": "Mean_QRS", "Mean_QTc": "Mean_QTc",
}
cols = [FEATURE_NAMES.index(c) for c in lead_1_features.values()]
df_features = pd.DataFrame(X_feat_train[:, cols], columns=list(lead_1_features))
df_targets = pd.DataFrame(y_train, columns=class_names)
df_combined = pd.concat([df_features, df_targets], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.set_theme(style="whitegrid")

# --- PLOT 1: absolute correlation of each Lead I feature with NORM / CD / MI ---
corr_matrix = df_combined.corr().abs()
target_corr = corr_matrix.loc[list(lead_1_features), ["NORM", "CD", "MI"]]
target_corr_melted = target_corr.reset_index().melt(
    id_vars="index", var_name="Superclass", value_name="Absolute Correlation")
sns.barplot(data=target_corr_melted, x="Absolute Correlation", y="index", hue="Superclass",
            palette={"NORM": "#2ecc71", "CD": "#3498db", "MI": "#e74c3c"}, ax=axes[0])
axes[0].set_title("A. Lead I Feature Predictive Power by Disease Class", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Absolute Pearson Correlation Magnitude", fontsize=12)
axes[0].set_ylabel("Lead I Extracted Features", fontsize=12)
axes[0].legend(title="Ailment Superclass", loc="upper right")


# --- PLOT 2: Mean_QRS by (mutually exclusive) primary class ---
def get_primary_class(row):
    if row["CD"] == 1: return "Conduction Disturbance (CD)"
    if row["NORM"] == 1: return "Normal (NORM)"
    if row["MI"] == 1: return "Myocardial Infarction (MI)"
    return "Other"


df_combined["Primary_Class"] = df_combined.apply(get_primary_class, axis=1)
df_filtered = df_combined[df_combined["Primary_Class"] != "Other"]
sns.boxplot(data=df_filtered, x="Primary_Class", y="Mean_QRS",
            palette={"Normal (NORM)": "#2ecc71", "Conduction Disturbance (CD)": "#3498db",
                     "Myocardial Infarction (MI)": "#e74c3c"},
            showfliers=False, ax=axes[1])
axes[1].axhline(120, color="black", linestyle="--", linewidth=1.5, label="120ms Threshold (Block)")
axes[1].set_title("B. Lead I Extracted QRS Width by Class", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Diagnostic Class", fontsize=12)
axes[1].set_ylabel("Extracted Mean QRS Duration (ms)", fontsize=12)
axes[1].legend(loc="upper right")

plt.tight_layout()
out = FEATURE_RESULTS_DIR / "lead1_justification.png"
plt.savefig(out, dpi=300)
print(f"saved {out}")
