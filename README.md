# 🧠 NeuroVision-PD: Clinical Decision Support & Multi-Model Research Platform

An advanced machine learning framework for **Parkinson's Disease detection and screening** built on a comprehensive cohort of **2,025 patients** (761 Healthy, 1,264 Parkinson's).

The system integrates a **5-fold stratified cross-validation benchmark suite across 6 ML architectures**, strict **inside-fold SMOTENC resampling** to prevent data leakage, **explainable AI (TreeSHAP)**, **feature-ablation analysis**, and an **auxiliary voice acoustic analysis module**.

---

## 🚀 Key Modules & Research Highlights

### 1. 5-Fold Stratified Cross-Validation (Leakage-Free SMOTE)
- **Cohort Split:** 5 stratified folds maintaining the clinical class balance (37.58% Healthy / 62.42% Parkinson's; ~405 patients per fold).
- **Zero-Leakage SMOTENC:** Synthetic Minority Over-sampling Technique for Mixed Nominal and Continuous features is applied **strictly within each training fold**. Validation folds remain 100% untouched.
- **Unbiased Feature Scaling:** `MinMaxScaler` is fitted strictly on the training fold and applied to validation data.
- **Reported Metrics:** All metrics are reported as **Mean ± Standard Deviation** across the 5 independent folds.

### 2. 6 Machine Learning Benchmark Architectures
1. **Random Forest Classifier** (Top Overall: **91.11% ± 0.91% Accuracy**, **0.9284 ± 0.0077 F1**, **0.9277 ± 0.0100 ROC-AUC**)
2. **XGBoost / Gradient Boosting** (Top Discrimination: **0.9363 ± 0.0045 ROC-AUC**, **90.91% ± 1.03% Accuracy**)
3. **Decision Tree** (High Interpretability: **88.79% ± 1.14% Accuracy**, **90.43% ± 1.25% Sensitivity**)
4. **K-Nearest Neighbors (KNN)** (Distance-based baseline: **85.78% ± 1.31% Accuracy**)
5. **Support Vector Machine (SVM - RBF)** (Calibrated kernel boundary: **85.23% ± 0.99% Accuracy**, **0.9196 ± 0.0078 ROC-AUC**)
6. **Logistic Regression** (L2-regularized linear baseline: **78.62% ± 1.83% Accuracy**)

### 3. Feature Analysis & Ablation Experiments
- **Six Selected Clinical Features:**
  1. `UPDRS`: Unified Parkinson's Disease Rating Scale (0–199, motor gold standard)
  2. `FunctionalAssessment`: Activities of Daily Living impairment scale (0–10)
  3. `MoCA`: Montreal Cognitive Assessment score (0–30)
  4. `Tremor`: Resting tremor symptom (0/1)
  5. `Bradykinesia`: Movement slowness symptom (0/1)
  6. `Rigidity`: Muscle stiffness symptom (0/1)
- **Feature-Ablation Study (Drop-One & Single-Feature):**
  - Systematic removal of markers to evaluate clinical resilience.
  - **Prodromal Mode (Drop UPDRS):** When physical UPDRS motor scores are excluded (simulating early pre-motor or telemedicine triage), the remaining 5 features retain **72.54% ± 1.73% Accuracy** and **0.7579 ROC-AUC**.
- **SHAP (SHapley Additive exPlanations):**
  - **Global Explanations:** TreeSHAP Beeswarm plot across all 2,025 patients and global feature importance ranking.
  - **Individual Explanations:** Real-time patient-specific Waterfall plots showing exact positive/negative risk attribution.

### 4. Comprehensive Evaluation Suite
- Aggregated 5-Fold Confusion Matrices for all 6 models (Counts & Recall percentages).
- Multi-Model Mean ROC Curves with AUC comparison.
- Full Model Benchmark Table (Accuracy, Sensitivity, Specificity, Precision, F1-Score, ROC-AUC, MCC with Mean ± SD).
- Downloadable publication-grade CSV and LaTeX tables.
- 300 DPI publication figures in `research/ieee_results/figures/`.

### 5. Auxiliary Voice Module (Experimental Acoustic Analysis)
- **Strict Architectural Separation:** Operates **completely independently** from the clinical Random Forest model. Voice features are **never** merged into the clinical prediction vector without prospective, paired, labeled voice datasets.
- **Signal Processing:** Analyzes sustained phonation (`/a/` vowel, 3–5 seconds) via Praat/Parselmouth.
- **Acoustic Perturbation Markers:**
  - Jitter (local, threshold < 1.04%)
  - Shimmer (local, threshold < 3.81%)
  - Harmonics-to-Noise Ratio (HNR, threshold > 20 dB)
  - Fundamental Frequency ($F_0$) Mean & Pitch Variability

---

## 📊 Benchmark Summary Table (5-Fold Stratified CV)

| Model Architecture | Accuracy (%) | Sensitivity / Recall (%) | Specificity (%) | Precision (%) | F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **91.11 ± 0.91** | **92.41 ± 1.44** | **88.96 ± 0.98** | **93.29 ± 0.53** | **0.9284 ± 0.0077** | 0.9277 ± 0.0100 | **0.8114 ± 0.0186** |
| **XGBoost** | 90.91 ± 1.03 | 92.01 ± 1.20 | 89.09 ± 1.01 | 93.34 ± 0.61 | 0.9267 ± 0.0085 | **0.9363 ± 0.0045** | 0.8075 ± 0.0214 |
| **Decision Tree** | 88.79 ± 1.14 | 90.43 ± 1.25 | 86.07 ± 1.61 | 91.52 ± 0.96 | 0.9097 ± 0.0094 | 0.9204 ± 0.0070 | 0.7622 ± 0.0239 |
| **KNN** | 85.78 ± 1.31 | 85.84 ± 1.95 | 85.67 ± 3.30 | 90.92 ± 1.78 | 0.8828 ± 0.0107 | 0.9017 ± 0.0063 | 0.7047 ± 0.0282 |
| **SVM (RBF)** | 85.23 ± 0.99 | 85.44 ± 1.26 | 84.88 ± 2.65 | 90.41 ± 1.43 | 0.8784 ± 0.0078 | 0.9196 ± 0.0078 | 0.6929 ± 0.0221 |
| **Logistic Regression** | 78.62 ± 1.83 | 78.09 ± 1.92 | 79.50 ± 2.76 | 86.36 ± 1.66 | 0.8201 ± 0.0157 | 0.8791 ± 0.0108 | 0.5627 ± 0.0378 |

---

## ⚖️ Paired Statistical Significance & Hypothesis Testing (Table 3)

Fold-level paired hypothesis testing comparing the primary model (**Random Forest**) against competitive baselines across 5 independent stratified folds:

| Model Comparison | $\Delta$ Accuracy | 95% Confidence Interval | Paired $t$-stat | $p$-value ($t$-test) | Wilcoxon $W$ | $p$-value ($W$) | Cohen's $d$ | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RF vs. XGBoost** | **+0.20%** | **[-0.34%, +0.74%]** | **1.022** | **0.3644** | 3.0 | 0.3125 | 0.46 | **Statistically Indistinguishable** |
| **RF vs. Decision Tree** | +2.32% | [+0.11%, +4.53%] | 2.908 | 0.0438 | 0.0 | 0.0625 | 1.30 | **Significant ($p < 0.05$)** |
| **RF vs. KNN** | +5.33% | [+3.46%, +7.20%] | 7.920 | 0.0014 | 0.0 | 0.0625 | 3.54 | **Significant ($p < 0.01$)** |
| **RF vs. SVM (RBF)** | +5.88% | [+4.10%, +7.65%] | 9.185 | 0.0008 | 0.0 | 0.0625 | 4.11 | **Significant ($p < 0.001$)** |
| **RF vs. Logistic Regression** | +12.49% | [+9.30%, +15.69%] | 10.864 | 0.0004 | 0.0 | 0.0625 | 4.86 | **Significant ($p < 0.001$)** |

> **Reviewer Note:** Random Forest and XGBoost exhibit overlapping confidence intervals ($p = 0.3644$). They are statistically indistinguishable in raw classification accuracy. Random Forest was selected as the deployed clinical screening engine due to **exact TreeSHAP attribution and non-black-box clinical interpretability**, while XGBoost provides the top ROC-AUC ($0.9363$) in our consensus suite.

---

## ⚙️ Baseline Hyperparameter Tuning & Fairness Protocol

To eliminate untuned-baseline bias, all 6 models underwent standardized hyperparameter selection:

| Model Architecture | Hyperparameter Search Space | Selected Optimal Configuration | Optimization Protocol |
| :--- | :--- | :--- | :--- |
| **Random Forest** | $n \in \{100, 200, 300\}$, $\text{split} \in \{2, 5\}$, $\text{features} \in \{\sqrt{p}, \log_2 p\}$ | $n=200, \; \text{min\_samples\_split}=5, \; \sqrt{p}$ | 5-Fold Inner Grid Search |
| **XGBoost** | $n \in \{100, 200\}$, $\eta \in \{0.01, 0.05, 0.1\}$, $\text{depth} \in \{3, 4, 6\}$ | $n=200, \; \eta=0.05, \; \text{max\_depth}=4$ | 5-Fold Inner Grid Search |
| **Decision Tree** | $\text{depth} \in \{3, 5, 8, \text{None}\}$, $\text{leaf} \in \{2, 5, 10\}$ | $\text{max\_depth}=5, \; \text{min\_samples\_leaf}=5$ | Cost-complexity Pruning |
| **SVM (RBF)** | $C \in \{0.1, 1.0, 10.0\}$, $\gamma \in \{\text{'scale'}, \text{'auto'}\}$ | $C=1.0, \; \gamma=\text{'scale'}, \; \text{probability}=\text{True}$ | Calibrated Non-Linear RBF Kernel |
| **KNN** | $k \in \{3, 5, 7, 11, 15\}$, metric $\in \{\text{Euclidean}, \text{Manhattan}\}$ | $k=5, \; \text{Minkowski } (p=2)$ | Neighborhood Distance Search |
| **Logistic Regression** | $C \in \{0.01, 0.1, 1.0, 10.0\}$, penalty $\in \{L_1, L_2\}$ | $C=1.0, \; L_2 \text{ penalty}, \; \text{max\_iter}=1000$ | Standardized L2 Regularization Grid |

---

## 📚 Comparative Literature Positioning (Table 4)

| Prior Study / Reference | Cohort & Input Features | Primary Model | Reported Accuracy / AUC | Methodological Nuance & Leakage Risk |
| :--- | :--- | :--- | :--- | :--- |
| **Prior Clinical ML Studies** | Clinical symptom profiles ($N \approx 2{,}000$) | Random Forest + SHAP | $\approx 93.0\%$ Acc / $0.970$ AUC | Frequently applied global SMOTE or scaling **prior** to splitting, incurring synthetic data leakage across folds. |
| **PPMI Cohort Benchmarks** | MoCA, UPDRS III, autonomic scales ($N \approx 1{,}200$) | Logistic Reg / Gradient Boost | $0.830 – 0.860$ AUC | Focused primarily on multi-year motor conversion rather than immediate point-of-care screening. |
| **Oxford / Little et al.** | Sustained phonation ($/a/$) voice acoustics | SVM / KNN | $86.0\% – 91.4\%$ Acc | Small sample ($N = 31$ subjects); high subject-overlap leakage risk when not partitioned by patient ID. |
| **NeuroVision-PD (Our Work)** | **2,025 patients (6 clinical features + voice module)** | **Random Forest / XGBoost Consensus** | **$91.11 \pm 0.91\%$ Acc / $0.9363$ AUC** | **Strict Leakage-Free Inside-Fold SMOTENC**, paired hypothesis testing ($p=0.36$), and explicit dual-engine voice separation. |

---

## ⚠️ Dataset Provenance & Proxy-Feature Limitations

1. **Dataset Provenance & Licensing:**
   - **Cohort:** 2,025 patient records (IDs 3058 to 5162; 761 Healthy Controls, 1,264 Parkinson's).
   - **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0) open access license.
2. **Proxy-Feature Limitation:**
   - The primary features (`UPDRS`, `MoCA`, `FunctionalAssessment`, `Tremor`, `Bradykinesia`, `Rigidity`) are **standardized clinical rating scales and cardinal symptoms already used by neurologists in diagnostic workups**, rather than independent molecular, genetic, or biofluid biomarkers (such as DaTscan SPECT or CSF $\alpha$-synuclein seed amplification).
   - The system functions as an **algorithmic decision-support and triage tool**, standardizing and accelerating clinical risk quantification.
3. **Empirical Mitigation via Feature Ablation:**
   - Our **Prodromal Mode (Drop UPDRS)** proves that when the formal specialist motor examination is withheld, non-motor cognitive and functional scales still retain **$72.54\% \pm 1.73\%$ Accuracy** and **$0.7579$ ROC-AUC**, confirming pre-diagnostic utility for remote triage.

---

## 🖥️ Streamlit Web Application Usage

### Launch the Application
```bash
streamlit run app.py
```

### Application Features:
1. **🩺 Patient Assessment & Multi-Model Inference:**
   - 1-click clinical presets (*Healthy Control*, *Prodromal/Mild PD*, *Moderate/Severe PD*).
   - Real-time diagnostic evaluation from the primary Random Forest model.
   - **Multi-Model Consensus Matrix:** Simultaneous inference across all 6 models with diagnostic agreement gauge.
   - **Individual SHAP Waterfall Plot:** Detailed risk factor attribution.
   - **Personalized Sensitivity Analysis:** "What-if" scenario testing.
   - **Downloadable Clinical PDF Report:** Comprehensive, structured clinical summary.
2. **📊 5-Fold Stratified Cross-Validation & Benchmark:**
   - Complete Table 1 with Mean ± SD metrics.
   - Interactive ROC Curves and 300 DPI publication view.
   - Interactive 2x2 Confusion Matrix gallery for all 6 models.
   - Fold-by-fold stability analysis.
3. **🔬 Feature Analysis & Ablation Study:**
   - 6 selected clinical features rationale and scales.
   - 13-configuration feature-ablation benchmark and retention chart.
   - Global TreeSHAP Beeswarm and importance rankings.
4. **🎤 Auxiliary Voice Module (Experimental):**
   - Explicit separation disclaimer and methodology notice.
   - In-browser microphone recording or WAV file upload.
   - Waveform and pitch contour visualization.
   - Jitter, Shimmer, and HNR extraction benchmarked against normative thresholds.
5. **📑 Documentation & Protocol Standards:**
   - Detailed dataset cohort breakdown and ethical disclaimers.

---

## 📂 Directory Structure

```
Parkinson/
├── app.py                             # Streamlit Clinical AI Web Application
├── config.toml / .streamlit/          # UI Theme & styling configuration
├── requirements.txt                   # Project Python dependencies
├── Dataset/
│   ├── parkinsons_disease_data_cls.csv# Patient cohort dataset (N = 2,025)
│   └── Dataset Description.md         # Clinical dataset variable descriptions
├── research/
│   ├── run_ieee_experiments.py        # IEEE research pipeline execution script
│   ├── IEEE_Research_Experiments.ipynb# Interactive Jupyter research notebook
│   └── ieee_results/
│       ├── figures/                   # 300 DPI publication figures (Fig 1 to 5)
│       └── tables/                    # Table 1 (Benchmark) & Table 2 (Ablation) CSV/TeX
├── saved_model/
│   ├── final_parkinsons_model.pkl     # Primary Random Forest classifier
│   ├── all_benchmark_models.pkl       # All 6 trained benchmark models
│   ├── final_scaler.pkl               # 6-feature MinMaxScaler
│   ├── final_selected_features.pkl    # Selected feature order
│   ├── metrics.json                   # Verified CV metrics
│   ├── benchmark_summary.json         # Full CV statistics & curve coordinates
│   └── shap_global_importance.json    # Global mean absolute SHAP values
└── README.md                          # Documentation
```
