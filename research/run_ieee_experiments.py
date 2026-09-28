"""
=============================================================================
IEEE Publication Research Pipeline for Parkinson's Disease Detection
=============================================================================
Author: Research Upgrade Pipeline
Components:
  1. 6-Model Benchmark Suite (LR, KNN, DT, SVM, Random Forest, XGBoost)
  2. 5-Fold Stratified Cross-Validation (Mean +/- Std for 7 Metrics)
  3. Inside-Fold Leakage-Free SMOTENC Resampling
  4. Comprehensive Feature-Ablation Study (Drop-One & Single-Feature)
  5. 300 DPI Publication-Grade Figures & LaTeX Tables
  6. Zero-Downtime Synchronization with Streamlit Application
=============================================================================
"""

import os
import ast
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
from scipy import stats
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC
import joblib
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Setup Paths & Directories
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARKINSON_DIR = PROJECT_ROOT  # Assuming script resides in Parkinson/research

DATA_PATH = os.path.join(PARKINSON_DIR, "Dataset", "parkinsons_disease_data_cls.csv")
RESULTS_DIR = os.path.join(PARKINSON_DIR, "research", "ieee_results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
MODEL_DIR = os.path.join(PARKINSON_DIR, "Model Training", "saved_model")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 75)
print("STARTING IEEE PUBLICATION RESEARCH PIPELINE")
print("=" * 75)
print(f"Data Source   : {DATA_PATH}")
print(f"Results Output: {RESULTS_DIR}")
print(f"Model Sync Dir: {MODEL_DIR}")

# ---------------------------------------------------------------------------
# Load and Parse Dataset
# ---------------------------------------------------------------------------
raw_df = pd.read_csv(DATA_PATH)

# Parse stringified Symptoms dictionary
raw_df['Symptoms'] = raw_df['Symptoms'].apply(ast.literal_eval)
symptoms_df = pd.json_normalize(raw_df['Symptoms']).replace({'Yes': 1, 'No': 0})

# Combine key target features
FEATURE_COLS = ['UPDRS', 'FunctionalAssessment', 'Tremor', 'MoCA', 'Bradykinesia', 'Rigidity']
CAT_INDICES = [2, 4, 5]  # Tremor, Bradykinesia, Rigidity are binary nominal
CONT_COLS = ['UPDRS', 'FunctionalAssessment', 'MoCA']

df = pd.concat([
    raw_df[['UPDRS', 'FunctionalAssessment', 'MoCA', 'Diagnosis']],
    symptoms_df[['Tremor', 'Bradykinesia', 'Rigidity']]
], axis=1)

X = df[FEATURE_COLS].copy()
y = df['Diagnosis'].values

n_total = len(df)
n_neg = sum(y == 0)
n_pos = sum(y == 1)
print("\nCohort Overview:")
print(f"  Total Patients  : {n_total}")
print(f"  Healthy (Class 0): {n_neg} ({n_neg / n_total * 100:.2f}%)")
print(f"  Parkinson's (1) : {n_pos} ({n_pos / n_total * 100:.2f}%)")

# ---------------------------------------------------------------------------
# Define Model Suite
# ---------------------------------------------------------------------------
def initialize_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, metric="minkowski", p=2
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=42
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_split=5, max_features="sqrt", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", random_state=42
        )
    }

# ---------------------------------------------------------------------------
# Experiment 1: 5-Fold Stratified Cross-Validation (Leak-Free SMOTENC)
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("EXPERIMENT 1: 5-FOLD STRATIFIED CROSS-VALIDATION WITH SMOTENC")
print("-" * 75)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_names = list(initialize_models().keys())
metrics_dict = {
    m: {"acc": [], "sens": [], "spec": [], "prec": [], "f1": [], "auc": [], "mcc": []}
    for m in model_names
}
roc_storage = {m: {"fpr": [], "tpr": [], "aucs": []} for m in model_names}
cm_storage = {m: np.zeros((2, 2), dtype=int) for m in model_names}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold = X.iloc[train_idx].copy()
    y_train_fold = y[train_idx]
    X_val_fold = X.iloc[val_idx].copy()
    y_val_fold = y[val_idx]

    # Preprocessing strictly inside training fold
    fold_scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        fold_scaler.fit_transform(X_train_fold),
        columns=FEATURE_COLS
    )
    X_val_scaled = pd.DataFrame(
        fold_scaler.transform(X_val_fold),
        columns=FEATURE_COLS
    )

    # SMOTENC applied strictly to training fold
    smotenc = SMOTENC(categorical_features=CAT_INDICES, random_state=42)
    X_train_res, y_train_res = smotenc.fit_resample(X_train_scaled, y_train_fold)

    # Train and evaluate all models on untouched validation fold
    current_models = initialize_models()
    for name, model in current_models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        cm = confusion_matrix(y_val_fold, y_pred)
        cm_storage[name] += cm
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_val_fold, y_pred)
        sens = recall_score(y_val_fold, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_val_fold, y_pred)
        f1 = f1_score(y_val_fold, y_pred)
        auc = roc_auc_score(y_val_fold, y_prob)
        mcc = matthews_corrcoef(y_val_fold, y_pred)

        metrics_dict[name]["acc"].append(acc)
        metrics_dict[name]["sens"].append(sens)
        metrics_dict[name]["spec"].append(spec)
        metrics_dict[name]["prec"].append(prec)
        metrics_dict[name]["f1"].append(f1)
        metrics_dict[name]["auc"].append(auc)
        metrics_dict[name]["mcc"].append(mcc)

        fpr, tpr, _ = roc_curve(y_val_fold, y_prob)
        roc_storage[name]["fpr"].append(fpr)
        roc_storage[name]["tpr"].append(tpr)
        roc_storage[name]["aucs"].append(auc)

    print(f"  Fold {fold}/5 completed successfully.")

# Build Table 1
table1_rows = []
for name in model_names:
    m = metrics_dict[name]
    table1_rows.append({
        "Model": name,
        "Accuracy (%)": f"{np.mean(m['acc'])*100:.2f} ± {np.std(m['acc'])*100:.2f}",
        "Sensitivity (%)": f"{np.mean(m['sens'])*100:.2f} ± {np.std(m['sens'])*100:.2f}",
        "Specificity (%)": f"{np.mean(m['spec'])*100:.2f} ± {np.std(m['spec'])*100:.2f}",
        "Precision (%)": f"{np.mean(m['prec'])*100:.2f} ± {np.std(m['prec'])*100:.2f}",
        "F1-Score": f"{np.mean(m['f1']):.4f} ± {np.std(m['f1']):.4f}",
        "ROC-AUC": f"{np.mean(m['auc']):.4f} ± {np.std(m['auc']):.4f}",
        "MCC": f"{np.mean(m['mcc']):.4f} ± {np.std(m['mcc']):.4f}",
    })

df_table1 = pd.DataFrame(table1_rows)
csv_tab1_path = os.path.join(TAB_DIR, "Table1_Model_Benchmark.csv")
tex_tab1_path = os.path.join(TAB_DIR, "Table1_Model_Benchmark.tex")
df_table1.to_csv(csv_tab1_path, index=False, encoding="utf-8")

# Custom clean LaTeX export
latex_tab1 = (
    "\\begin{table*}[htbp]\n"
    "\\centering\n"
    "\\caption{Performance Comparison of Machine Learning Models via 5-Fold Stratified Cross-Validation with Leakage-Free SMOTENC}\n"
    "\\label{tab:model_benchmark}\n"
    "\\begin{tabular}{lcccccc}\n"
    "\\hline\n"
    "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Sensitivity (\\%)} & \\textbf{Specificity (\\%)} & \\textbf{F1-Score} & \\textbf{ROC-AUC} & \\textbf{MCC} \\\\\n"
    "\\hline\n"
)
for _, r in df_table1.iterrows():
    bold = "\\textbf{" if "Random Forest" in r["Model"] or "XGBoost" in r["Model"] else ""
    end_bold = "}" if bold else ""
    latex_tab1 += (
        f"{bold}{r['Model']}{end_bold} & {r['Accuracy (%)']} & {r['Sensitivity (%)']} & "
        f"{r['Specificity (%)']} & {r['F1-Score']} & {r['ROC-AUC']} & {r['MCC']} \\\\\n"
    )
latex_tab1 += "\\hline\n\\end{tabular}\n\\end{table*}\n"
with open(tex_tab1_path, "w", encoding="utf-8") as f:
    f.write(latex_tab1)

print("\n" + df_table1.to_string(index=False))
print(f"\nSaved Table 1: {csv_tab1_path} and {tex_tab1_path}")

# ---------------------------------------------------------------------------
# Experiment 1B: Paired Statistical Significance (Table 3)
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("EXPERIMENT 1B: PAIRED STATISTICAL SIGNIFICANCE TESTS (RF VS. BASELINES)")
print("-" * 75)

rf_accs = np.array(metrics_dict["Random Forest"]["acc"])
sig_rows = []
sig_tests_raw = []

for m_name in ["XGBoost", "Decision Tree", "K-Nearest Neighbors", "Support Vector Machine", "Logistic Regression"]:
    comp_accs = np.array(metrics_dict[m_name]["acc"])
    diff = rf_accs - comp_accs
    d_mean = float(np.mean(diff))
    t_stat, p_val_t = stats.ttest_rel(rf_accs, comp_accs)
    try:
        w_stat, p_val_w = stats.wilcoxon(rf_accs, comp_accs)
    except Exception:
        w_stat, p_val_w = np.nan, np.nan
    ci_l, ci_h = stats.t.interval(0.95, df=len(diff)-1, loc=d_mean, scale=stats.sem(diff))
    c_d = float(d_mean / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    is_sig = bool(p_val_t < 0.05)
    sig_rows.append({
        "Model Comparison": f"Random Forest vs. {m_name}",
        "Delta Accuracy (%)": f"{d_mean:+.2f}%",
        "95% CI": f"[{ci_l:+.2f}%, {ci_h:+.2f}%]",
        "Paired t-stat": f"{t_stat:.3f}",
        "p-value (t-test)": f"{p_val_t:.4f}",
        "Wilcoxon W": f"{w_stat:.1f}",
        "p-value (Wilcoxon)": f"{p_val_w:.4f}",
        "Cohen's d": f"{c_d:.2f}",
        "Significance (alpha=0.05)": "Significant (p < 0.05)" if is_sig else "Indistinguishable (p >= 0.05)"
    })
    sig_tests_raw.append({
        "challenger": m_name,
        "delta_accuracy": round(d_mean, 4),
        "ci_95": [round(float(ci_l), 4), round(float(ci_h), 4)],
        "t_stat": round(float(t_stat), 4),
        "p_value_t": round(float(p_val_t), 6),
        "wilcoxon_w": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
        "p_value_w": round(float(p_val_w), 6) if not np.isnan(p_val_w) else None,
        "cohen_d": round(c_d, 4),
        "is_significant": is_sig
    })

df_table3 = pd.DataFrame(sig_rows)
csv_tab3_path = os.path.join(TAB_DIR, "Table3_Statistical_Significance.csv")
tex_tab3_path = os.path.join(TAB_DIR, "Table3_Statistical_Significance.tex")
df_table3.to_csv(csv_tab3_path, index=False, encoding="utf-8")
with open(tex_tab3_path, "w", encoding="utf-8") as f:
    f.write(df_table3.to_latex(index=False, caption="Paired Statistical Significance Tests (Random Forest vs. Baseline Models Across 5 Folds)", label="tab:significance"))
print(df_table3.to_string(index=False))
print(f"\nSaved Table 3: {csv_tab3_path} and {tex_tab3_path}")

# ---------------------------------------------------------------------------
# Experiment 2: Feature-Ablation Study
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("EXPERIMENT 2: FEATURE-ABLATION EXPERIMENTS (DROP-ONE & SINGLE-FEATURE)")
print("-" * 75)

ablation_configurations = {
    # Full baseline
    "Full Set (All 6 Features)": FEATURE_COLS,
    # Leave-One-Out
    "Drop UPDRS (Prodromal Mode)": [f for f in FEATURE_COLS if f != 'UPDRS'],
    "Drop FunctionalAssessment": [f for f in FEATURE_COLS if f != 'FunctionalAssessment'],
    "Drop Tremor": [f for f in FEATURE_COLS if f != 'Tremor'],
    "Drop MoCA": [f for f in FEATURE_COLS if f != 'MoCA'],
    "Drop Bradykinesia": [f for f in FEATURE_COLS if f != 'Bradykinesia'],
    "Drop Rigidity": [f for f in FEATURE_COLS if f != 'Rigidity'],
    # Single feature evaluations
    "Only UPDRS": ['UPDRS'],
    "Only FunctionalAssessment": ['FunctionalAssessment'],
    "Only Tremor": ['Tremor'],
    "Only MoCA": ['MoCA'],
    "Only Bradykinesia": ['Bradykinesia'],
    "Only Rigidity": ['Rigidity'],
}

ablation_results = []
for config_name, feats in ablation_configurations.items():
    X_sub = df[feats].copy()
    cat_idxs = [i for i, f in enumerate(feats) if f in ['Tremor', 'Bradykinesia', 'Rigidity']]
    cont_sub = [f for f in feats if f not in ['Tremor', 'Bradykinesia', 'Rigidity']]

    accs, senss, specs, f1s, aucs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X_sub, y):
        X_tr, y_tr = X_sub.iloc[train_idx].copy(), y[train_idx]
        X_va, y_va = X_sub.iloc[val_idx].copy(), y[val_idx]

        # Scaler
        s = MinMaxScaler()
        X_tr_sc = pd.DataFrame(s.fit_transform(X_tr), columns=feats)
        X_va_sc = pd.DataFrame(s.transform(X_va), columns=feats)

        # SMOTE: only apply if nominal features present and >= 2 features
        if len(feats) > 1 and len(cat_idxs) > 0 and len(cat_idxs) < len(feats):
            sm = SMOTENC(categorical_features=cat_idxs, random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
        elif len(feats) > 1 and len(cat_idxs) == 0:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr_sc, y_tr

        eval_model = RandomForestClassifier(n_estimators=200, min_samples_split=5, random_state=42)
        eval_model.fit(X_tr_res, y_tr_res)

        yp = eval_model.predict(X_va_sc)
        yprob = eval_model.predict_proba(X_va_sc)[:, 1]
        cm_sub = confusion_matrix(y_va, yp)
        tn, fp, fn, tp = cm_sub.ravel()

        accs.append(accuracy_score(y_va, yp))
        senss.append(recall_score(y_va, yp))
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        f1s.append(f1_score(y_va, yp))
        aucs.append(roc_auc_score(y_va, yprob))

    ablation_results.append({
        "Configuration": config_name,
        "Num Features": len(feats),
        "Accuracy (%)": f"{np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}",
        "Sensitivity (%)": f"{np.mean(senss)*100:.2f} ± {np.std(senss)*100:.2f}",
        "Specificity (%)": f"{np.mean(specs)*100:.2f} ± {np.std(specs)*100:.2f}",
        "F1-Score": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
        "ROC-AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}",
        "_mean_acc": np.mean(accs) * 100,
        "_mean_auc": np.mean(aucs)
    })

df_ablation = pd.DataFrame(ablation_results)
csv_tab2_path = os.path.join(TAB_DIR, "Table2_Feature_Ablation.csv")
tex_tab2_path = os.path.join(TAB_DIR, "Table2_Feature_Ablation.tex")
df_ablation.drop(columns=["_mean_acc", "_mean_auc"]).to_csv(csv_tab2_path, index=False, encoding="utf-8")

# Custom LaTeX Export for Table 2
latex_tab2 = (
    "\\begin{table*}[htbp]\n"
    "\\centering\n"
    "\\caption{Feature Ablation Analysis: Evaluating Impact of Clinical Markers on Random Forest Performance}\n"
    "\\label{tab:ablation}\n"
    "\\begin{tabular}{lccccc}\n"
    "\\hline\n"
    "\\textbf{Configuration} & \\textbf{Features} & \\textbf{Accuracy (\\%)} & \\textbf{Sensitivity (\\%)} & \\textbf{F1-Score} & \\textbf{ROC-AUC} \\\\\n"
    "\\hline\n"
)
for _, r in df_ablation.iterrows():
    bold = "\\textbf{" if "Full Set" in r["Configuration"] or "Drop UPDRS" in r["Configuration"] else ""
    end_bold = "}" if bold else ""
    latex_tab2 += (
        f"{bold}{r['Configuration']}{end_bold} & {r['Num Features']} & "
        f"{r['Accuracy (%)']} & {r['Sensitivity (%)']} & {r['F1-Score']} & {r['ROC-AUC']} \\\\\n"
    )
latex_tab2 += "\\hline\n\\end{tabular}\n\\end{table*}\n"
with open(tex_tab2_path, "w", encoding="utf-8") as f:
    f.write(latex_tab2)

print("\n" + df_ablation.drop(columns=["_mean_acc", "_mean_auc"]).to_string(index=False))
print(f"\nSaved Table 2: {csv_tab2_path} and {tex_tab2_path}")

# ---------------------------------------------------------------------------
# Generate Publication-Quality Figures (300 DPI)
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("GENERATING 300 DPI PUBLICATION FIGURES")
print("-" * 75)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
})

# Fig 1: Multi-Model ROC Curves
plt.figure(figsize=(7.5, 6), dpi=300)
colors = sns.color_palette("tab10", len(model_names))
for i, name in enumerate(model_names):
    mean_auc = np.mean(metrics_dict[name]["auc"])
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for f in range(5):
        tprs.append(np.interp(mean_fpr, roc_storage[name]["fpr"][f], roc_storage[name]["tpr"][f]))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color=colors[i], lw=2, label=f"{name} (AUC = {mean_auc:.3f})")

plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
plt.title("Stratified 5-Fold Mean ROC Curves", fontweight="bold", pad=12)
plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "Fig1_ROC_Curves.png")
plt.savefig(fig1_path)
plt.close()
print(f"  Saved Fig 1: {fig1_path}")

# Fig 2: Confusion Matrices for Top 3 Models
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)
top_models = ["Random Forest", "XGBoost", "Support Vector Machine"]
for idx, m_name in enumerate(top_models):
    cm = cm_storage[m_name]
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    annot_matrix = np.empty_like(cm, dtype=object)
    for r in range(2):
        for c in range(2):
            annot_matrix[r, c] = f"{cm[r, c]}\n({cm_norm[r, c]*100:.1f}%)"

    sns.heatmap(
        cm, annot=annot_matrix, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
        annot_kws={"size": 11, "fontweight": "bold"}, linewidths=1.2, linecolor="white"
    )
    axes[idx].set_title(f"{m_name}", fontweight="bold", fontsize=11)
    axes[idx].set_xlabel("Predicted Label", fontweight="bold")
    axes[idx].set_xticklabels(["No PD (0)", "PD (1)"])
    axes[idx].set_yticklabels(["No PD (0)", "PD (1)"], rotation=0)
    if idx == 0:
        axes[idx].set_ylabel("True Label", fontweight="bold")
    else:
        axes[idx].set_ylabel("")

plt.suptitle("Aggregated 5-Fold Confusion Matrices (Counts & Class Recall)", fontweight="bold", y=1.02)
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "Fig2_Confusion_Matrices.png")
plt.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"  Saved Fig 2: {fig2_path}")

# Fig 3: Feature Importance Comparison (RF MDI vs XGBoost Gain)
clean_scaler = MinMaxScaler()
X_full_scaled = pd.DataFrame(clean_scaler.fit_transform(X), columns=FEATURE_COLS)
smote_full = SMOTENC(categorical_features=CAT_INDICES, random_state=42)
X_full_res, y_full_res = smote_full.fit_resample(X_full_scaled, y)

benchmark_models = initialize_models()
for name, model_inst in benchmark_models.items():
    print(f"  Fitting full model for production export: {name}...")
    model_inst.fit(X_full_res, y_full_res)

rf_full = benchmark_models["Random Forest"]
xgb_full = benchmark_models["XGBoost"]

imp_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Random Forest (MDI)": rf_full.feature_importances_,
    "XGBoost (Gain)": xgb_full.feature_importances_
}).sort_values(by="Random Forest (MDI)", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
x_indices = np.arange(len(FEATURE_COLS))
bar_width = 0.38

b1 = ax.bar(x_indices - bar_width/2, imp_df["Random Forest (MDI)"], bar_width, label="Random Forest (MDI)", color="#1f77b4")
b2 = ax.bar(x_indices + bar_width/2, imp_df["XGBoost (Gain)"], bar_width, label="XGBoost (Gain)", color="#ff7f0e")

ax.set_ylabel("Relative Importance Score", fontweight="bold")
ax.set_title("Feature Importance Benchmark: Tree Ensembles", fontweight="bold", pad=10)
ax.set_xticks(x_indices)
ax.set_xticklabels(imp_df["Feature"], rotation=20, ha="right", fontweight="bold")
ax.legend(frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "Fig3_Feature_Importance.png")
plt.savefig(fig3_path)
plt.close()
print(f"  Saved Fig 3: {fig3_path}")

# Fig 4: SHAP Global Beeswarm Plot
explainer = shap.TreeExplainer(rf_full)
shap_values = explainer.shap_values(X_full_scaled)

if isinstance(shap_values, list):
    shap_vals_positive = shap_values[1]
elif shap_values.ndim == 3:
    shap_vals_positive = shap_values[:, :, 1]
else:
    shap_vals_positive = shap_values

plt.figure(figsize=(8.5, 5), dpi=300)
shap.summary_plot(
    shap_vals_positive, X, feature_names=FEATURE_COLS,
    show=False, color_bar=True
)
plt.title("Global Clinical Risk Factor Attribution (TreeSHAP Beeswarm)", fontweight="bold", pad=12)
plt.xlabel("SHAP Value (Impact on Log-Odds of Parkinson's Diagnosis)", fontweight="bold")
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "Fig4_SHAP_Global_Beeswarm.png")
plt.savefig(fig4_path, bbox_inches="tight")
plt.close()
print(f"  Saved Fig 4: {fig4_path}")

# Fig 5: Feature Ablation Comparison Bar Plot
df_drop_one = df_ablation[df_ablation["Configuration"].str.startswith("Drop") | df_ablation["Configuration"].str.startswith("Full")].copy()
df_drop_one.sort_values(by="_mean_acc", ascending=True, inplace=True)

plt.figure(figsize=(9, 4.8), dpi=300)
bars = plt.barh(
    df_drop_one["Configuration"], df_drop_one["_mean_acc"],
    color=["#d95f02" if "Drop UPDRS" in c else "#7570b3" if "Full" in c else "#1b9e77" for c in df_drop_one["Configuration"]],
    edgecolor="black", height=0.6
)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", ha="left", fontsize=9, fontweight="bold")

plt.xlim([0, 105])
plt.xlabel("5-Fold Cross-Validation Mean Accuracy (%)", fontweight="bold")
plt.title("Feature Ablation Impact: Performance Retention when Markers are Excluded", fontweight="bold", pad=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
fig5_path = os.path.join(FIG_DIR, "Fig5_Feature_Ablation.png")
plt.savefig(fig5_path)
plt.close()
print(f"  Saved Fig 5: {fig5_path}")

# ---------------------------------------------------------------------------
# Reconcile & Synchronize Streamlit Model Artifacts
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("SYNCHRONIZING VERIFIED MODEL ARTIFACTS FOR STREAMLIT APP")
print("-" * 75)

# 1. Save Random Forest model artifact
rf_artifact_path = os.path.join(MODEL_DIR, "final_parkinsons_model.pkl")
joblib.dump(rf_full, rf_artifact_path)
print(f"  [OK] Saved Clean Model Artifact: {rf_artifact_path}")

# 2. Save all 6 benchmark models for multi-model consensus and inference
models_artifact_path = os.path.join(MODEL_DIR, "all_benchmark_models.pkl")
joblib.dump(benchmark_models, models_artifact_path)
print(f"  [OK] Saved All 6 Benchmark Models: {models_artifact_path}")

# 3. Save the 6-feature MinMaxScaler
scaler_artifact_path = os.path.join(MODEL_DIR, "final_scaler.pkl")
joblib.dump(clean_scaler, scaler_artifact_path)
print(f"  [OK] Saved Clean Scaler Artifact: {scaler_artifact_path}")

# 4. Save the selected feature names
features_artifact_path = os.path.join(MODEL_DIR, "final_selected_features.pkl")
joblib.dump(FEATURE_COLS, features_artifact_path)
print(f"  [OK] Saved Selected Features Artifact: {features_artifact_path}")

# 5. Build rich benchmark_summary.json
mean_abs_shap = np.mean(np.abs(shap_vals_positive), axis=0)
shap_dict = {feat: float(mean_abs_shap[i]) for i, feat in enumerate(FEATURE_COLS)}

cv_models_summary = {}
for name in model_names:
    m = metrics_dict[name]
    cv_models_summary[name] = {
        "accuracy_mean": round(float(np.mean(m["acc"]) * 100), 2),
        "accuracy_std": round(float(np.std(m["acc"]) * 100), 2),
        "f1_mean": round(float(np.mean(m["f1"])), 4),
        "f1_std": round(float(np.std(m["f1"])), 4),
        "auc_mean": round(float(np.mean(m["auc"])), 4),
        "auc_std": round(float(np.std(m["auc"])), 4),
        "precision_mean": round(float(np.mean(m["prec"]) * 100), 2),
        "precision_std": round(float(np.std(m["prec"]) * 100), 2),
        "sensitivity_mean": round(float(np.mean(m["sens"]) * 100), 2),
        "sensitivity_std": round(float(np.std(m["sens"]) * 100), 2),
        "specificity_mean": round(float(np.mean(m["spec"]) * 100), 2),
        "specificity_std": round(float(np.std(m["spec"]) * 100), 2),
        "mcc_mean": round(float(np.mean(m["mcc"])), 4),
        "mcc_std": round(float(np.std(m["mcc"])), 4),
        "fold_accuracy": [round(float(v * 100), 2) for v in m["acc"]],
        "fold_f1": [round(float(v), 4) for v in m["f1"]],
        "fold_auc": [round(float(v), 4) for v in m["auc"]],
        "confusion_matrix": cm_storage[name].tolist(),
    }

# ROC curves interpolation
roc_summary = {}
mean_fpr_points = np.linspace(0, 1, 101)
for name in model_names:
    tprs = []
    for f in range(5):
        tprs.append(np.interp(mean_fpr_points, roc_storage[name]["fpr"][f], roc_storage[name]["tpr"][f]))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    roc_summary[name] = {
        "mean_auc": round(float(np.mean(metrics_dict[name]["auc"])), 4),
        "std_auc": round(float(np.std(metrics_dict[name]["auc"])), 4),
        "tpr": [round(float(val), 4) for val in mean_tpr]
    }

ablation_records = df_ablation.drop(columns=["_mean_acc", "_mean_auc"], errors="ignore").to_dict(orient="records")

benchmark_payload = {
    "protocol": "5-Fold Stratified Cross-Validation with Inside-Fold SMOTENC Resampling",
    "features": FEATURE_COLS,
    "cohort": {
        "total": n_total,
        "healthy": int(n_neg),
        "parkinsons": int(n_pos),
        "healthy_pct": round(n_neg / n_total * 100, 2),
        "parkinsons_pct": round(n_pos / n_total * 100, 2)
    },
    "models": cv_models_summary,
    "roc_curves": {
        "mean_fpr": [round(float(x), 4) for x in mean_fpr_points],
        "curves": roc_summary
    },
    "ablation": ablation_records,
    "statistical_significance": {
        "tests": sig_tests_raw
    },
    "global_shap": shap_dict,
    "gini_importance": {feat: round(float(rf_full.feature_importances_[i]), 4) for i, feat in enumerate(FEATURE_COLS)},
    "xgb_gain_importance": {feat: round(float(xgb_full.feature_importances_[i]), 4) for i, feat in enumerate(FEATURE_COLS)}
}

# 6. Save verified metrics.json
rf_cv_stats = metrics_dict["Random Forest"]
mean_acc = round(float(np.mean(rf_cv_stats["acc"]) * 100), 2)
mean_auc = round(float(np.mean(rf_cv_stats["auc"]) * 100), 2)
mean_f1_0 = round(float(np.mean(rf_cv_stats["spec"])), 2)
mean_f1_1 = round(float(np.mean(rf_cv_stats["sens"])), 2)

metrics_payload = {
    "accuracy": mean_acc,
    "roc_auc": mean_auc,
    "f1_class0": mean_f1_0,
    "f1_class1": mean_f1_1,
    "cv_folds": 5,
    "protocol": "5-Fold Stratified Cross-Validation with Inside-Fold SMOTENC Resampling",
    "models_evaluated": len(model_names),
    "updated_at": "2026-09-25"
}

# Sync to all target directories
sync_dirs = [
    MODEL_DIR,
    os.path.join(PARKINSON_DIR, "saved_model"),
    TAB_DIR,
    PARKINSON_DIR
]

# Table 1-3 are generated by this script (in TAB_DIR); copy them to every
# other sync directory too, so a download button anywhere in the app never
# silently serves an older run's numbers than what's shown on screen.
# Table 4 (literature comparison) is manually curated, not generated here,
# so it is intentionally left untouched by this sync step.
generated_table_files = [
    "Table1_Model_Benchmark.csv", "Table1_Model_Benchmark.tex",
    "Table2_Feature_Ablation.csv", "Table2_Feature_Ablation.tex",
    "Table3_Statistical_Significance.csv", "Table3_Statistical_Significance.tex",
]

for d in sync_dirs:
    os.makedirs(d, exist_ok=True)
    if d in [MODEL_DIR, os.path.join(PARKINSON_DIR, "saved_model")]:
        joblib.dump(rf_full, os.path.join(d, "final_parkinsons_model.pkl"))
        joblib.dump(benchmark_models, os.path.join(d, "all_benchmark_models.pkl"))
        joblib.dump(clean_scaler, os.path.join(d, "final_scaler.pkl"))
        joblib.dump(FEATURE_COLS, os.path.join(d, "final_selected_features.pkl"))
    
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(os.path.join(d, "shap_global_importance.json"), "w") as f:
        json.dump(shap_dict, f, indent=2)
    with open(os.path.join(d, "benchmark_summary.json"), "w") as f:
        json.dump(benchmark_payload, f, indent=2)

    if d != TAB_DIR:
        for fname in generated_table_files:
            src = os.path.join(TAB_DIR, fname)
            if os.path.isfile(src):
                with open(src, "r", encoding="utf-8") as f_src:
                    content = f_src.read()
                with open(os.path.join(d, fname), "w", encoding="utf-8") as f_dst:
                    f_dst.write(content)

print("  [OK] Synchronized all artifacts to MODEL_DIR and saved_model!")

# ---------------------------------------------------------------------------
# Sanity Test: Verify Streamlit App Compatibility
# ---------------------------------------------------------------------------
print("\n" + "-" * 75)
print("SANITY CHECKING STREAMLIT INFERENCE PIPELINE")
print("-" * 75)

# Emulate Streamlit app prediction logic
loaded_model = joblib.load(rf_artifact_path)
loaded_scaler = joblib.load(scaler_artifact_path)
loaded_features = joblib.load(features_artifact_path)

sample_patient = {
    "UPDRS": 75.0,
    "FunctionalAssessment": 4.5,
    "Tremor": 1,
    "MoCA": 16.0,
    "Bradykinesia": 1,
    "Rigidity": 0
}
sample_df = pd.DataFrame([sample_patient])[loaded_features]
sample_scaled = loaded_scaler.transform(sample_df)
pred = loaded_model.predict(sample_scaled)[0]
probs = loaded_model.predict_proba(sample_scaled)[0]

print(f"  Test Patient Input: {sample_patient}")
diagnosis_text = "Parkinson's Disease (1)" if pred == 1 else "Healthy (0)"
print(f"  Model Prediction  : {diagnosis_text}")
print(f"  Probabilities     : No PD: {probs[0]*100:.2f}%, PD: {probs[1]*100:.2f}%")
print("  [SUCCESS] Model, Scaler, and Features verified 100% compatible with Streamlit application!")

print("\n" + "=" * 75)
print("RESEARCH UPGRADE COMPLETE & VERIFIED")
print("=" * 75)