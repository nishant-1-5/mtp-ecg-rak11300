import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the corrected cached data
cache = np.load("ptbxl_preprocessed_cache.npz", allow_pickle=True)
X_feat_train = cache["X_feat_train"]
y_train = cache["y_train"]
class_names = cache["class_names"] # ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Define strictly Lead I features
lead_1_features = [
    "Mean_HR", "HR_Std", "SDNN", "RMSSD", 
    "Signal_Complexity", "Mean_PR", "Mean_QRS", "Mean_QTc"
]

# Build DataFrame
df_features = pd.DataFrame(X_feat_train[:, :len(lead_1_features)], columns=lead_1_features)
df_targets = pd.DataFrame(y_train, columns=class_names)
df_combined = pd.concat([df_features, df_targets], axis=1)

# Set up the PPT visualization figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.set_theme(style="whitegrid")

# --- PLOT 1: Correlation Contrast (Lead I Strengths vs Weaknesses) ---
# Calculate absolute correlations for NORM/CD vs MI
corr_matrix = df_combined.corr().abs()
target_corr = corr_matrix.loc[lead_1_features, ["NORM", "CD", "MI"]]
target_corr_melted = target_corr.reset_index().melt(id_vars="index", var_name="Superclass", value_name="Absolute Correlation")

sns.barplot(
    data=target_corr_melted, 
    x="Absolute Correlation", 
    y="index", 
    hue="Superclass", 
    palette={"NORM": "#2ecc71", "CD": "#3498db", "MI": "#e74c3c"},
    ax=axes[0]
)
axes[0].set_title("A. Lead I Feature Predictive Power by Disease Class", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Absolute Pearson Correlation Magnitude", fontsize=12)
axes[0].set_ylabel("Lead I Extracted Features", fontsize=12)
axes[0].legend(title="Ailment Superclass", loc="lower right")

# --- PLOT 2: Clinical Class Separation Boxplot (Proof of CD Detection) ---
# Create a mutually exclusive categorical column for plotting
def get_primary_class(row):
    if row["CD"] == 1: return "Conduction Disturbance (CD)"
    if row["NORM"] == 1: return "Normal (NORM)"
    if row["MI"] == 1: return "Myocardial Infarction (MI)"
    return "Other"

df_combined["Primary_Class"] = df_combined.apply(get_primary_class, axis=1)
df_filtered = df_combined[df_combined["Primary_Class"] != "Other"]

sns.boxplot(
    data=df_filtered, 
    x="Primary_Class", 
    y="Mean_QRS", 
    palette={"Normal (NORM)": "#2ecc71", "Conduction Disturbance (CD)": "#3498db", "Myocardial Infarction (MI)": "#e74c3c"},
    showfliers=False, # Hide extreme outliers for a cleaner PPT chart
    ax=axes[1]
)
# Add a horizontal line at 120ms (Clinical threshold for Bundle Branch Blocks)
axes[1].axhline(120, color='black', linestyle='--', linewidth=1.5, label="120ms Threshold (Block)")
axes[1].set_title("B. Lead I QRS Widening in Conduction Disturbances", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Diagnostic Class", fontsize=12)
axes[1].set_ylabel("Extracted Mean QRS Duration (ms)", fontsize=12)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("lead_1_justification_ppt.png", dpi=300)
plt.show()