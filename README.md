# 🩸 Predicting Massive Transfusion in Trauma Patients Using Early Clinical Parameters

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning pipeline for early prediction of **Massive Transfusion (MT)** in trauma patients, using clinical parameters available within the first hour of hospital admission.

Developed as part of the **Technion AI Developer Course** (2026).

---

## 🎯 Clinical Problem

Massive Transfusion — defined as ≥10 units of packed red blood cells within 24 hours — occurs in ~3–5% of trauma patients but accounts for a disproportionate share of trauma mortality. Early identification is critical: delays in activating a Massive Transfusion Protocol (MTP) worsen outcomes.

The current clinical standard, the **ABC Score**, achieves high precision (0.90) but misses **62% of MT patients** — making it unsuitable as a standalone screening tool.

**Goal:** Build a model that catches more MT patients early, while keeping false alarms at a clinically acceptable level.

---

## 📊 Dataset

Synthetic dataset inspired by the CRASH-2 trial design (n = 1,000 patients).

| Split | Size | MT+ | MT− |
|-------|------|-----|-----|
| Train | 800  | 200 | 600 |
| Test  | 200  | 50  | 150 |

Class ratio: **1:3 (MT : Non-MT)** — reflecting realistic trauma registry prevalence.

**Features used for modeling (6):**

| Feature | Clinical Meaning |
|---------|-----------------|
| `Systolic_BP_mmHg` | Hemodynamic instability marker |
| `Heart_Rate_BPM` | Tachycardia — early shock sign |
| `Shock_Index` | HR / SBP — composite instability score |
| `Lactate_mmol_L` | Tissue hypoperfusion / hemorrhagic shock |
| `Arterial_Base_Excess` | Metabolic acidosis severity |
| `Injury_Type` | Penetrating (1) vs Blunt (0) |

Features excluded: `GCS_Score` (40% missing, not MT-specific), `Units_Transfused` (data leakage), `Time_to_Hospital_min` (not predictive).

---

## 🔬 Pipeline

```
Raw Data → EDA → Preprocessing → Model Training → Threshold Optimization → Evaluation
```

| Notebook | Description |
|----------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis — distributions, correlations, class balance |
| `02_Preprocessing.ipynb` | Feature selection, encoding, train/test split, StandardScaler |
| `03_LogisticRegression.ipynb` | Baseline model with class_weight='balanced' |
| `04_RandomForest.ipynb` | Ensemble with min_samples_leaf tuning |
| `05_XGBoost.ipynb` | Two-stage hyperparameter search (manual → GridSearchCV, PR-AUC scoring) |
| `06_ChoosingTheModel.ipynb` | Voting & Stacking ensembles, final model selection |

---

## 📈 Results

### Model Comparison

| Model | ROC-AUC | Recall (MT) | Precision (MT) | F1 (MT) |
|-------|---------|-------------|----------------|---------|
| ABC Score (clinical baseline) | 0.747 | 0.38 | **0.90** | 0.54 |
| Logistic Regression | 0.880 | 0.78 | 0.62 | 0.69 |
| Random Forest | 0.883 | 0.80 | 0.53 | 0.63 |
| Voting Classifier | 0.899 | 0.76 | 0.62 | 0.68 |
| Stacking Classifier | 0.901 | 0.80 | 0.61 | 0.69 |
| **XGBoost (final)** | **0.902** | **0.80** | **0.62** | **0.70** |

### Final Model: XGBoost (threshold = 0.44)

```
Confusion Matrix (test set, n=200):

              Predicted Non-MT   Predicted MT
Actual Non-MT      125               25
Actual MT           10               40
```

- **Sensitivity (Recall): 0.80** — finds 4 out of 5 MT patients
- **Specificity: 0.833** — 83% of non-MT correctly cleared
- **vs ABC Score: +42 percentage points in Recall**

### Key Hyperparameters

```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=1.0,
    scale_pos_weight=3.0,   # handles 1:3 class imbalance
    eval_metric='logloss'
)
```

Threshold selected via **Out-of-Fold (OOF) F1 optimization with Recall ≥ 0.80 constraint**.

---

## 🏥 Clinical Interpretation

> The model is designed as a **screening tool**, not a replacement for clinical judgment.

| | ABC Score | XGBoost | Combined workflow |
|--|-----------|---------|------------------|
| Role | Confirmation | **Screening** | Screen → Confirm |
| Recall | 0.38 | **0.80** | — |
| Precision | 0.90 | 0.62 | — |
| Miss rate | 62% | **20%** | — |

**Proposed workflow:** XGBoost flags high-risk patients → ABC Score used for confirmation → MTP activation decision.

---

## ⚠️ Limitations

1. **Synthetic dataset** — performance on real institutional data is unknown
2. **No GCS** — excluded due to 40% missingness; real models would impute or include it
3. **Precision = 0.62** — 1 in 3 MT alerts is a false positive; insufficient for standalone clinical use
4. **Gray zone overlap** — ~43% of cases intentionally created with overlapping feature distributions, simulating real clinical uncertainty

---

## 🚀 Next Steps

- [ ] Validate pipeline on real institutional dataset (~500 patients)
- [ ] Ablation study: performance with vs without Lactate
- [ ] SHAP values for individual patient explanations
- [ ] Calibration tuning (Platt scaling / isotonic regression)

---

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/Yuliana-wein/Predict-Massive-Transfusion-in-Trauma-PatieUsing-Early-Clinical-Parameters.git
cd Predict-Massive-Transfusion-in-Trauma-PatieUsing-Early-Clinical-Parameters

# Create environment
conda env create -f environment.yaml
conda activate trauma-ml

# Launch notebooks
jupyter notebook
```

---

## 📁 Project Structure

```
├── data/
│   └── CRASH2_Final_Dataset.xlsx      # Synthetic dataset (n=1000)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_LogisticRegression.ipynb
│   ├── 04_RandomForest.ipynb
│   ├── 05_XGBoost.ipynb
│   └── 06_ChoosingTheModel.ipynb
├── outputs/
│   └── figures/                       # Saved plots
├── environment.yaml
└── README.md
```

---

## 👩‍💻 Authors

**Yuliana Weinstein**  
**Dima Spektor**  
**Aysha Azaizeh**  

Technion AI Developer Course, 2026

---

## 📚 References

- Roberts I. et al. *The CRASH-2 trial: a randomised controlled trial.* Lancet, 2010.
- Nunez T.C. et al. *Simple calculation of the ABC Score.* J Trauma, 2009.
- Chen T., Guestrin C. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.
