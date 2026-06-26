"""
logistic_regression_workflow.py

Logistic Regression workflow for predicting Massive Transfusion (MT = 1)
in trauma patients using early clinical parameters.

Scope (this file):
  - Data loading and EDA
  - Leakage-safe preprocessing
  - GCS_Score strategy experiment (drop vs median-impute inside pipeline)
  - Feature diagnostics: VIF and correlation (descriptive only)
  - Baseline and tuned Logistic Regression (L1 / L2 / ElasticNet)
  - Threshold selection on OOF predictions from training data only
  - Single held-out test evaluation
  - Artifact saving: figures, metrics CSV, coefficients CSV,
    hyperparameters JSON, final report

Out of scope (teammates):
  - Random Forest and XGBoost models
  A placeholder comparison table is written to final_report.txt.

Usage:
    python scripts/logistic_regression_workflow.py

Outputs (created at runtime):
    outputs/logistic_regression/
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import textwrap
import warnings
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for script mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance as sk_perm_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED     = 42
TEST_SIZE       = 0.20
MIN_SPECIFICITY = 0.75

# Relative to the project root (run script from project root directory)
DATA_PATH    = Path("data/CRASH2_Final_1000.xlsx")
TARGET       = "MT"
OUTPUT_BASE  = Path("outputs/logistic_regression")
FIGURES_DIR  = OUTPUT_BASE / "figures"

# Columns removed before any analysis
# - Patient_ID:           row identifier — no predictive value
# - Units_Transfused:     directly defines the MT label — data leakage
# - Time_to_Hospital_min: excluded by project scope decision
DROP_ALWAYS = ["Patient_ID", "Units_Transfused", "Time_to_Hospital_min"]

# Deterministic categorical encodings (no statistics learned from data)
CATEGORICAL_COLS = {
    "Sex":         {"Female": 1, "Male": 0},
    "Injury_Type": {"Penetrating": 1, "Blunt": 0},
}

# Physiological survivable bounds — clinical constants, not data-derived.
# Applying these same constants to the test set is NOT leakage.
CLIP_BOUNDS = {
    "Systolic_BP_mmHg": (40, 250),
    "Heart_Rate_BPM":   (20, 220),
}

# Single feature set used throughout all modeling steps.
# Shock_Index (= HR / SBP) is present in the data and kept in the dataframe
# for EDA and correlation reporting only.  It is intentionally excluded here
# because Systolic_BP_mmHg and Heart_Rate_BPM are already included — using
# all three creates perfect multicollinearity that destabilises LR coefficients.
# GCS_Score is added to this list programmatically if the GCS experiment
# determines that imputation is the better strategy.
BASE_FEATURES = [
    "Systolic_BP_mmHg",
    "Heart_Rate_BPM",
    "Age",
    "Respiratory_Rate_BPM",
    "Lactate_mmol_L",
    "Arterial_Base_Excess",
    "Injury_Type_Coded",
    "Sex_Coded",
]

VIF_THRESHOLD = 5.0    # VIF above this is flagged in the report (descriptive)
CORR_FLAG     = 0.70   # |r| above this is flagged in EDA interpretation

# Continuous columns shown in histograms and violin plots.
# Shock_Index is included for EDA only; binary-encoded columns are excluded.
CONTINUOUS_COLS = [
    "Age", "Systolic_BP_mmHg", "Heart_Rate_BPM",
    "Respiratory_Rate_BPM", "GCS_Score",
    "Lactate_mmol_L", "Arterial_Base_Excess", "Shock_Index",
]


# ════════════════════════════════════════════════════════════════
# SECTION 1 — Data loading
# ════════════════════════════════════════════════════════════════

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"\n[load_data] Path   : {path}")
    print(f"[load_data] Shape  : {df.shape}")
    print(f"[load_data] Columns: {df.columns.tolist()}")
    print(f"[load_data] Dtypes :\n{df.dtypes.to_string()}")
    print(f"[load_data] Duplicates     : {df.duplicated().sum()}")
    id_col = df.columns[0]
    print(f"[load_data] Unique {id_col}: {df[id_col].nunique()}")
    return df


# ════════════════════════════════════════════════════════════════
# SECTION 2 — Early column removal and categorical encoding
# ════════════════════════════════════════════════════════════════

def drop_early_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    reasons = {
        "Patient_ID":           "row identifier — no predictive value",
        "Units_Transfused":     "directly defines the MT label — data leakage",
        "Time_to_Hospital_min": "excluded by project scope decision",
    }
    to_drop = [c for c in cols if c in df.columns]
    print("\n[drop_early_columns]")
    for c in to_drop:
        print(f"  Dropping '{c}': {reasons.get(c, 'scheduled for removal')}")
    return df.drop(columns=to_drop)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[encode_categoricals]")
    for col, mapping in CATEGORICAL_COLS.items():
        if col not in df.columns:
            continue
        new_col = col + "_Coded"
        df[new_col] = df[col].map(mapping)
        df = df.drop(columns=[col])
        print(f"  '{col}' → '{new_col}'  mapping: {mapping}")
    return df


# ════════════════════════════════════════════════════════════════
# SECTION 3 — Exploratory data analysis
# ════════════════════════════════════════════════════════════════

def run_eda(df: pd.DataFrame, target: str, out_dir: Path) -> str:
    """
    Run all EDA plots and generate interpretation text.
    Saves figures to out_dir/figures/ and a written report to eda_report.txt.
    Returns the combined interpretation text (inserted into final_report.txt).
    """
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    cont_cols    = [c for c in CONTINUOUS_COLS if c in df.columns]
    all_numeric  = [c for c in df.columns
                    if c != target and pd.api.types.is_numeric_dtype(df[c])]

    texts = [
        _plot_class_distribution(df, target, fdir),
        _plot_missing_values(df, fdir),
        _plot_feature_histograms(df, target, cont_cols, fdir),
        _plot_violin_by_target(df, target, cont_cols, fdir),
        _plot_correlation_heatmap(df, all_numeric, fdir),
        _eda_summary_text(df, target, cont_cols),
    ]

    combined = "\n\n".join(t for t in texts if t)
    report_path = out_dir / "eda_report.txt"
    report_path.write_text(combined, encoding="utf-8")
    print(f"\n[EDA] Report saved → {report_path}")
    return combined


def _plot_class_distribution(df, target, fdir) -> str:
    counts = df[target].value_counts().sort_index()
    pcts   = df[target].value_counts(normalize=True).sort_index() * 100
    labels = ["Non-MT (0)", "MT (1)"]
    colors = ["#5B9BD5", "#ED7D31"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor="black", linewidth=0.7)
    for bar, pct in zip(bars, pcts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_title("Target Class Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fdir / "01_class_distribution.png", dpi=150)
    plt.close()

    text = (
        "CLASS DISTRIBUTION\n"
        f"  Non-MT (0): {counts.get(0, 0)} patients "
        f"({pcts.get(0, 0):.1f}%)\n"
        f"  MT     (1): {counts.get(1, 0)} patients "
        f"({pcts.get(1, 0):.1f}%)\n"
        f"  Imbalance ratio: "
        f"{counts.get(0, 0) / max(counts.get(1, 1), 1):.1f}:1\n"
        "  Implication: accuracy alone is misleading.\n"
        "  Recall and PR-AUC are primary metrics.\n"
        "  class_weight='balanced' is applied in all LR configurations."
    )
    print(f"\n{text}")
    return text


def _plot_missing_values(df, fdir) -> str:
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        text = "MISSING VALUES\n  No missing values detected."
        print(f"\n{text}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Missing Values", fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(fdir / "02_missing_values.png", dpi=150)
        plt.close()
        return text

    colors = [
        "#C00000" if v > 30 else "#ED7D31" if v > 10 else "#5B9BD5"
        for v in miss.values
    ]
    fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.55)))
    ax.barh(miss.index, miss.values, color=colors,
            edgecolor="black", linewidth=0.6)
    ax.axvline(30, color="darkred", ls="--", lw=1.2, label="30% threshold")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Values by Column", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.4)
    for i, (col, val) in enumerate(miss.items()):
        ax.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fdir / "02_missing_values.png", dpi=150)
    plt.close()

    lines = ["MISSING VALUES"]
    for col, val in miss.items():
        lines.append(f"  {col}: {val:.1f}% missing")
    if "GCS_Score" in miss.index:
        lines.append(
            "  GCS_Score dominates at 37.9% missing.\n"
            "  The GCS experiment (Section 5 of main) tests whether\n"
            "  median imputation inside a CV pipeline improves PR-AUC\n"
            "  over simply dropping the column."
        )
    text = "\n".join(lines)
    print(f"\n{text}")
    return text


def _plot_feature_histograms(df, target, cont_cols, fdir) -> str:
    if not cont_cols:
        return "FEATURE HISTOGRAMS\n  No continuous columns to plot."

    ncols = 3
    nrows = int(np.ceil(len(cont_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    palette = {0: "#5B9BD5", 1: "#ED7D31"}

    for i, col in enumerate(cont_cols):
        ax = axes[i]
        for cls in [0, 1]:
            vals = df[df[target] == cls][col].dropna()
            ax.hist(vals, bins=25, alpha=0.55,
                    color=palette[cls],
                    label=f"MT={cls}", edgecolor="none")
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(cont_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by MT Class",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(fdir / "03_feature_histograms.png", dpi=150)
    plt.close()

    lines = ["FEATURE HISTOGRAMS (continuous variables)"]
    lines.append(
        f"  {'Feature':<28} {'Skewness':>9}  "
        f"{'Mean MT=0':>10}  {'Mean MT=1':>10}  {'Diff':>8}"
    )
    lines.append("  " + "-" * 70)
    for col in cont_cols:
        if col not in df.columns:
            continue
        skew = df[col].skew()
        m0   = df[df[target] == 0][col].mean()
        m1   = df[df[target] == 1][col].mean()
        lines.append(
            f"  {col:<28} {skew:>+9.2f}  {m0:>10.2f}  {m1:>10.2f}  {m1-m0:>+8.2f}"
        )
    lines.append(
        "\n  Interpretation: Features with large absolute mean differences\n"
        "  between MT=0 and MT=1 (e.g., Lactate, Arterial_Base_Excess,\n"
        "  Heart_Rate) provide stronger univariate signal for LR.\n"
        "  Features with high skewness may benefit from log-transform\n"
        "  in a future iteration (see Iteration Ideas in final_report.txt)."
    )
    text = "\n".join(lines)
    print(f"\n{text}")
    return text


def _plot_violin_by_target(df, target, cont_cols, fdir) -> str:
    if not cont_cols:
        return "VIOLIN PLOTS\n  No continuous columns to plot."

    ncols = 3
    nrows = int(np.ceil(len(cont_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    palette = {0: "#5B9BD5", 1: "#ED7D31"}

    for i, col in enumerate(cont_cols):
        ax = axes[i]
        plot_df = df[[target, col]].dropna()
        sns.violinplot(
            data=plot_df, x=target, y=col,
            palette=palette, inner="quartile",
            ax=ax, linewidth=0.8,
        )
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_xlabel("MT (0 = No, 1 = Yes)")
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(cont_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by MT Class — Violin Plots",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(fdir / "04_violin_by_target.png", dpi=150)
    plt.close()

    text = (
        "VIOLIN PLOTS\n"
        "  Show distributional shape, spread, and quartiles per MT class.\n"
        "  Wide separation between MT=0 and MT=1 violin bodies indicates\n"
        "  strong univariate predictive value.\n"
        "  Substantial overlap (e.g., Age, Respiratory_Rate) means those\n"
        "  features contribute less individually but may still add value\n"
        "  in combination with others inside the LR model."
    )
    print(f"\n{text}")
    return text


def _plot_correlation_heatmap(df, all_numeric, fdir) -> str:
    if len(all_numeric) < 2:
        return "CORRELATION HEATMAP\n  Too few numeric columns to plot."

    corr = df[all_numeric].corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    size = max(8, len(all_numeric))

    fig, ax = plt.subplots(figsize=(size, size - 1))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title(
        "Pearson Correlation Heatmap (all numeric features + target)",
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fdir / "05_correlation_heatmap.png", dpi=150)
    plt.close()

    lines = [f"CORRELATION HEATMAP (|r| ≥ {CORR_FLAG} flagged — descriptive only)"]
    flagged = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= CORR_FLAG:
                flagged.append((cols[i], cols[j], r))

    if flagged:
        flagged.sort(key=lambda x: -abs(x[2]))
        for a, b, r in flagged:
            lines.append(f"  {a}  ↔  {b}:  r = {r:+.3f}")
        lines.append(
            "\n  Note: Shock_Index = HR / SBP, so its high correlation with\n"
            "  those columns is expected and algebraically determined.\n"
            "  Shock_Index is retained in the dataframe for EDA only and\n"
            "  excluded from all LR model fitting.\n"
            "  Correlation findings are reported here for interpretation;\n"
            "  no features are removed or model variants built because of them.\n"
            "  Regularization in LR handles residual multicollinearity."
        )
    else:
        lines.append(f"  No pairs exceed |r| = {CORR_FLAG}.")

    text = "\n".join(lines)
    print(f"\n{text}")
    return text


def _eda_summary_text(df, target, cont_cols) -> str:
    lines = ["DESCRIPTIVE STATISTICS"]
    present = [c for c in cont_cols if c in df.columns]
    if present:
        lines.append(df[present].describe().round(2).to_string())

    lines.append("\nEstimated task difficulty:")
    n_mt   = (df[target] == 1).sum()
    n_tot  = len(df)
    ratio  = (n_tot - n_mt) / max(n_mt, 1)
    n_gcs_miss = df["GCS_Score"].isnull().sum() if "GCS_Score" in df.columns else 0
    lines.append(
        f"  - Class imbalance   : {ratio:.1f}:1  (manageable)\n"
        f"  - GCS_Score missing : {n_gcs_miss / n_tot * 100:.1f}%\n"
        f"    (handled by GCS strategy experiment)\n"
        f"  - Distributional overlap: moderate for Age and Respiratory_Rate;\n"
        f"    stronger separation for Lactate and Arterial_Base_Excess\n"
        f"  - LR performance ceiling: depends on linear separability in\n"
        f"    log-odds space; calibration plot will confirm quality\n"
        f"  - Overall: moderately difficult binary classification task.\n"
        f"    LR establishes a transparent, interpretable linear baseline."
    )
    text = "\n".join(lines)
    print(f"\n{text}")
    return text


# ════════════════════════════════════════════════════════════════
# SECTION 4 — Preprocessing
# ════════════════════════════════════════════════════════════════

def stratified_split(X: pd.DataFrame, y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_SEED):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    print(f"\n[split] Train: {X_train.shape}  MT rate: {y_train.mean():.1%}")
    print(f"[split] Test : {X_test.shape}   MT rate: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def apply_clinical_clipping(X_train: pd.DataFrame,
                            X_test:  pd.DataFrame,
                            bounds:  dict):
    """
    Clips physiological variables to survivable bounds.
    Bounds are clinical constants, not derived from training data,
    so applying them to both sets is not leakage.
    """
    print("\n[clinical clipping]")
    for col, (lo, hi) in bounds.items():
        if col not in X_train.columns:
            continue
        n_tr = ((X_train[col] < lo) | (X_train[col] > hi)).sum()
        n_te = ((X_test[col]  < lo) | (X_test[col]  > hi)).sum()
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col]  = X_test[col].clip(lo, hi)
        print(f"  '{col}' clipped to [{lo}, {hi}]: "
              f"{n_tr} train rows, {n_te} test rows affected")
    return X_train, X_test


def recalculate_shock_index(X: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute Shock_Index = HR / SBP from clipped values so the stored
    column stays consistent with the clipped raw vitals.
    Called separately on X_train and X_test after clipping.
    Shock_Index is not used as a model feature.
    """
    if "Systolic_BP_mmHg" in X.columns and "Heart_Rate_BPM" in X.columns:
        X = X.copy()
        X["Shock_Index"] = X["Heart_Rate_BPM"] / X["Systolic_BP_mmHg"]
    return X


def build_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_SEED,
        )),
    ])


# ════════════════════════════════════════════════════════════════
# SECTION 5 — GCS_Score strategy experiment
# ════════════════════════════════════════════════════════════════

def compare_gcs_strategies(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           skf: StratifiedKFold) -> tuple:
    """
    Compare dropping GCS_Score vs including it with median imputation.

    Both pipelines use default LR (C=1, L2, lbfgs, class_weight='balanced').
    Imputer is fitted on training folds only inside cross_val_score.
    X_test is never seen here.

    Decision rule: choose "impute" only if it improves CV PR-AUC by > 0.005.
    Otherwise choose "drop" — 37.9% missingness makes imputation risky without
    a meaningful performance gain.

    Returns (decision: str, results: dict).
    """
    print("\n[GCS experiment] drop vs median-impute inside pipeline ...")
    results = {}

    for strategy in ("drop", "impute"):
        feats = list(BASE_FEATURES)                     # copy
        if strategy == "impute" and "GCS_Score" in X_train.columns:
            feats = feats + ["GCS_Score"]
        feats = [f for f in feats if f in X_train.columns]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("lr",      LogisticRegression(
                C=1, penalty="l2", solver="lbfgs",
                class_weight="balanced",
                max_iter=5000, random_state=RANDOM_SEED,
            )),
        ])

        scores = cross_val_score(
            pipe, X_train[feats], y_train,
            cv=skf, scoring="average_precision", n_jobs=-1,
        )
        results[strategy] = {
            "mean":     float(scores.mean()),
            "std":      float(scores.std()),
            "features": feats,
        }
        print(f"  {strategy:>6}: CV PR-AUC = "
              f"{scores.mean():.4f} ± {scores.std():.4f}  "
              f"(n_features = {len(feats)})")

    delta = results["impute"]["mean"] - results["drop"]["mean"]
    print(f"\n  Δ (impute − drop) = {delta:+.4f}")

    if delta > 0.005:
        decision = "impute"
        print("  → Decision: IMPUTE  "
              "(GCS adds > 0.005 PR-AUC; safe to include with pipeline imputer)")
    else:
        decision = "drop"
        print("  → Decision: DROP  "
              "(Δ ≤ 0.005; 37.9% missingness not justified by performance gain)")

    return decision, results


def get_feature_list(gcs_strategy: str, X_train: pd.DataFrame) -> list:
    if gcs_strategy == "impute" and "GCS_Score" in X_train.columns:
        return BASE_FEATURES + ["GCS_Score"]
    return [f for f in BASE_FEATURES if f in X_train.columns]


# ════════════════════════════════════════════════════════════════
# SECTION 6 — Feature diagnostics (descriptive only)
# ════════════════════════════════════════════════════════════════

def compute_vif(X_train: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Variance Inflation Factor for the model's numeric features (training set).

    Tries statsmodels first; falls back to sklearn LinearRegression R² formula.
    Binary-encoded columns (Injury_Type_Coded, Sex_Coded) are excluded by the
    caller because VIF is less meaningful for binary predictors.

    VIF findings are DESCRIPTIVE ONLY — no features are removed based on VIF.
    """
    X_num = X_train[feature_cols].dropna()

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant
        X_arr = add_constant(X_num.values.astype(float))
        vif_df = pd.DataFrame({
            "Feature": ["const"] + feature_cols,
            "VIF":     [variance_inflation_factor(X_arr, i)
                        for i in range(X_arr.shape[1])],
        })
        vif_df = (vif_df[vif_df["Feature"] != "const"]
                  .reset_index(drop=True))
        method = "statsmodels"
    except ImportError:
        arr     = X_num.values.astype(float)
        records = []
        for i, col in enumerate(feature_cols):
            y_col  = arr[:, i]
            X_rest = np.delete(arr, i, axis=1)
            if X_rest.shape[1] == 0:
                records.append({"Feature": col, "VIF": np.nan})
                continue
            r2 = LinearRegression().fit(X_rest, y_col).score(X_rest, y_col)
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            records.append({"Feature": col, "VIF": round(vif, 2)})
        vif_df = pd.DataFrame(records)
        method = "sklearn R² fallback (statsmodels not installed)"

    vif_df["VIF"] = vif_df["VIF"].round(2)
    vif_df = (vif_df.sort_values("VIF", ascending=False)
              .reset_index(drop=True))

    print(f"\n[VIF — {method}]")
    print(vif_df.to_string(index=False))

    flagged = vif_df[vif_df["VIF"] > VIF_THRESHOLD]
    if not flagged.empty:
        print(f"\n  Features with VIF > {VIF_THRESHOLD} (flagged, descriptive only):")
        for _, row in flagged.iterrows():
            print(f"    {row['Feature']}: {row['VIF']:.2f}")
    else:
        print(f"  All VIF ≤ {VIF_THRESHOLD} — no strong multicollinearity detected.")

    print(
        "\n  Note: VIF is reported for interpretation only.\n"
        "  No features are removed based on this result.\n"
        "  L1/L2/ElasticNet regularization handles multicollinearity in LR."
    )
    return vif_df


# ════════════════════════════════════════════════════════════════
# SECTION 7 — Metrics helpers
# Threshold-sweep logic and metric calculations reused from
# MT_ML.ipynb cells 16 (compute_metrics, select_threshold_f2,
# select_threshold_spec); re-implemented here as named functions.
# ════════════════════════════════════════════════════════════════

def compute_metrics(y_true, probs, threshold, label="") -> tuple:
    preds            = (probs >= threshold).astype(int)
    cm               = confusion_matrix(y_true, preds)
    tn, fp, fn, tp   = cm.ravel()
    recall  = tp / (tp + fn)          if (tp + fn) > 0          else 0.0
    spec    = tn / (tn + fp)          if (tn + fp) > 0          else 0.0
    prec    = tp / (tp + fp)          if (tp + fp) > 0          else 0.0
    f1      = 2*prec*recall / (prec+recall)   if (prec+recall) > 0  else 0.0
    f2      = (5*prec*recall)/(4*prec+recall) if (4*prec+recall)>0  else 0.0
    bal_acc = (recall + spec) / 2
    pr_auc  = average_precision_score(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    brier   = brier_score_loss(y_true, probs)
    metrics = {
        "Model":        label,
        "Threshold":    round(threshold, 4),
        "PR-AUC":       round(pr_auc,    3),
        "ROC-AUC":      round(roc_auc,   3),
        "Recall":       round(recall,    3),
        "Specificity":  round(spec,      3),
        "Precision":    round(prec,      3),
        "F1":           round(f1,        3),
        "F2":           round(f2,        3),
        "Bal-Accuracy": round(bal_acc,   3),
        "Brier":        round(brier,     3),
    }
    return metrics, cm


def select_threshold_f2(oof_probs, y_true) -> float:
    best_f2, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 500):
        preds          = (oof_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f2   = (5*prec*rec) / (4*prec+rec) if (4*prec+rec) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_t = f2, t
    print(f"  [F2 threshold]       t = {best_t:.3f}   OOF F2 = {best_f2:.3f}")
    return float(best_t)


def select_threshold_spec(oof_probs, y_true,
                          min_spec: float = MIN_SPECIFICITY):
    best_rec, best_t = 0.0, None
    for t in np.linspace(0.05, 0.95, 500):
        preds          = (oof_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if spec >= min_spec and rec > best_rec:
            best_rec, best_t = rec, t
    if best_t is None:
        print(f"  [Spec≥{min_spec}] no threshold satisfies this constraint")
    else:
        print(f"  [Spec≥{min_spec}] t = {best_t:.3f}   OOF Recall = {best_rec:.3f}")
    return float(best_t) if best_t is not None else None


# ════════════════════════════════════════════════════════════════
# SECTION 8 — Logistic Regression modeling
# ════════════════════════════════════════════════════════════════

def build_baseline_lr(features: list,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      skf: StratifiedKFold) -> dict:
    """
    Default LR: C=1, L2, lbfgs, class_weight='balanced'.
    Evaluated via OOF predictions from cross_val_predict on training data.
    X_test is never used here.
    """
    print("\n[Baseline LR]")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=1, penalty="l2", solver="lbfgs",
            class_weight="balanced",
            max_iter=5000, random_state=RANDOM_SEED,
        )),
    ])

    cv_scores = cross_val_score(
        pipe, X_train[features], y_train,
        cv=skf, scoring="average_precision", n_jobs=-1,
    )
    print(f"  CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    oof = cross_val_predict(
        pipe, X_train[features], y_train,
        cv=skf, method="predict_proba",
    )[:, 1]

    thr_f2   = select_threshold_f2(oof, y_train)
    thr_spec = select_threshold_spec(oof, y_train)

    m_f2, _ = compute_metrics(y_train, oof, thr_f2, "Baseline OOF (F2-thr)")
    print("\n  OOF metrics at F2-threshold (training data only):")
    for k, v in m_f2.items():
        if k != "Model":
            print(f"    {k:<15}: {v}")

    return {
        "cv_prauc":        float(cv_scores.mean()),
        "thr_f2":          thr_f2,
        "thr_spec":        thr_spec,
        "oof_metrics_f2":  m_f2,
    }


def build_param_grids() -> list:
    """
    Three grids with strictly valid solver / penalty / l1_ratio combinations.
    l1_ratio appears only in the ElasticNet grid.
    """
    C_range = [0.001, 0.01, 0.1, 1, 10, 100]
    common  = {"lr__class_weight": ["balanced"], "lr__max_iter": [5000]}

    grid_l1 = {**common,
        "lr__penalty": ["l1"],
        "lr__solver":  ["liblinear"],
        "lr__C":       C_range,
    }
    grid_l2 = {**common,
        "lr__penalty": ["l2"],
        "lr__solver":  ["lbfgs"],
        "lr__C":       C_range,
    }
    grid_en = {**common,
        "lr__penalty":  ["elasticnet"],
        "lr__solver":   ["saga"],
        "lr__C":        C_range,
        "lr__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return [("L1 / liblinear",   grid_l1),
            ("L2 / lbfgs",       grid_l2),
            ("ElasticNet / saga", grid_en)]


def run_hyperparameter_search(features: list,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              skf: StratifiedKFold) -> tuple:
    """
    GridSearchCV across three separate grids (L1, L2, ElasticNet).
    Selects the overall best estimator by CV PR-AUC across all grids.
    ConvergenceWarnings are captured and reported; they do not halt execution.

    Returns (best_pipeline, best_params, best_cv_prauc).
    """
    print("\n[Hyperparameter search]")
    pipe   = build_lr_pipeline()
    grids  = build_param_grids()

    best_score  = -1.0
    best_est    = None
    best_params = None

    for name, grid in grids:
        print(f"\n  ── Grid: {name}")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            gs = GridSearchCV(
                pipe, grid, cv=skf,
                scoring="average_precision",
                n_jobs=-1, refit=True,
            )
            gs.fit(X_train[features], y_train)

        n_warn = sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning)
        )
        if n_warn:
            print(
                f"    ⚠ {n_warn} ConvergenceWarning(s) during '{name}' search.\n"
                f"      max_iter=5000 may need to be increased if warnings persist."
            )

        print(f"  Best params : {gs.best_params_}")
        print(f"  CV PR-AUC   : {gs.best_score_:.4f}")

        if gs.best_score_ > best_score:
            best_score  = gs.best_score_
            best_est    = gs.best_estimator_
            best_params = gs.best_params_

    print(f"\n  ═══ Overall winner ═══")
    print(f"  Params      : {best_params}")
    print(f"  CV PR-AUC   : {best_score:.4f}")
    return best_est, best_params, float(best_score)


# ════════════════════════════════════════════════════════════════
# SECTION 9 — Threshold selection (training data only)
# ════════════════════════════════════════════════════════════════

def select_thresholds_on_oof(pipeline: Pipeline,
                             features: list,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             skf: StratifiedKFold) -> tuple:
    """
    Generate OOF probabilities via cross_val_predict on training data.
    Thresholds are selected here; X_test is never used in this function.
    """
    print("\n[Threshold selection on OOF predictions — training data only]")
    oof = cross_val_predict(
        pipeline, X_train[features], y_train,
        cv=skf, method="predict_proba",
    )[:, 1]
    thr_f2   = select_threshold_f2(oof, y_train)
    thr_spec = select_threshold_spec(oof, y_train)
    return thr_f2, thr_spec


# ════════════════════════════════════════════════════════════════
# SECTION 10 — Final test evaluation (called exactly once)
# ════════════════════════════════════════════════════════════════

def evaluate_on_test(pipeline: Pipeline,
                     features: list,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test:  pd.DataFrame,
                     y_test:  pd.Series,
                     thr_f2:  float,
                     thr_spec,
                     out_dir: Path) -> tuple:
    """
    Fit the final pipeline on all training data; evaluate once on the
    held-out test set. No further model adjustments are made after this step.
    """
    print("\n[Final test evaluation — held-out test set]")
    pipeline.fit(X_train[features], y_train)
    probs = pipeline.predict_proba(X_test[features])[:, 1]

    m_f2, cm_f2 = compute_metrics(y_test, probs, thr_f2, "LR (F2-threshold)")
    print(f"\n  Results at F2-threshold ({thr_f2:.4f}):")
    for k, v in m_f2.items():
        if k != "Model":
            print(f"    {k:<15}: {v}")
    print(f"\n  Classification report (F2-threshold = {thr_f2:.4f}):")
    print(classification_report(
        y_test, (probs >= thr_f2).astype(int),
        target_names=["Non-MT", "MT"],
    ))

    m_spec, cm_spec = {}, None
    if thr_spec is not None:
        m_spec, cm_spec = compute_metrics(
            y_test, probs, thr_spec,
            f"LR (Spec≥{MIN_SPECIFICITY} threshold)"
        )
        print(f"\n  Results at Spec≥{MIN_SPECIFICITY} threshold ({thr_spec:.4f}):")
        for k, v in m_spec.items():
            if k != "Model":
                print(f"    {k:<15}: {v}")
        print(f"\n  Classification report (Spec-threshold = {thr_spec:.4f}):")
        print(classification_report(
            y_test, (probs >= thr_spec).astype(int),
            target_names=["Non-MT", "MT"],
        ))

    _save_test_plots(probs, y_test, thr_f2, thr_spec,
                     cm_f2, cm_spec, out_dir)
    return m_f2, m_spec


def _save_test_plots(probs, y_test, thr_f2, thr_spec,
                     cm_f2, cm_spec, out_dir):
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    # ── Confusion matrix: F2 threshold
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_f2, annot=True, fmt="d", cmap="Blues", linewidths=1,
                xticklabels=["Non-MT", "MT"],
                yticklabels=["Non-MT", "MT"], ax=ax)
    ax.set_title(f"Confusion Matrix — F2 threshold ({thr_f2:.3f})",
                 fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fdir / "06_confusion_matrix_f2.png", dpi=150)
    plt.close()

    # ── Confusion matrix: Spec threshold
    if cm_spec is not None and thr_spec is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_spec, annot=True, fmt="d", cmap="Blues", linewidths=1,
                    xticklabels=["Non-MT", "MT"],
                    yticklabels=["Non-MT", "MT"], ax=ax)
        ax.set_title(
            f"Confusion Matrix — Spec≥{MIN_SPECIFICITY} threshold ({thr_spec:.3f})",
            fontweight="bold",
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(fdir / "07_confusion_matrix_spec.png", dpi=150)
        plt.close()

    # ── ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val      = roc_auc_score(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2E75B6", lw=2, label=f"LR (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    tn, fp, fn, tp = confusion_matrix(y_test, (probs >= thr_f2).astype(int)).ravel()
    ax.scatter(fp / max(fp + tn, 1), tp / max(tp + fn, 1),
               color="#ED7D31", s=90, zorder=5,
               label=f"F2 threshold ({thr_f2:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — Logistic Regression (Test Set)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fdir / "08_roc_curve.png", dpi=150)
    plt.close()

    # ── Precision-Recall curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs)
    pr_auc     = average_precision_score(y_test, probs)
    prevalence = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, color="#2E75B6", lw=2,
            label=f"LR (PR-AUC = {pr_auc:.3f})")
    ax.axhline(prevalence, color="grey", ls=":", lw=1.5,
               label=f"No-skill baseline ({prevalence:.2f})")
    tn, fp, fn, tp = confusion_matrix(y_test, (probs >= thr_f2).astype(int)).ravel()
    p_pt = tp / max(tp + fp, 1)
    r_pt = tp / max(tp + fn, 1)
    ax.scatter(r_pt, p_pt, color="#ED7D31", s=90, zorder=5,
               label=f"F2 threshold ({thr_f2:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Logistic Regression (Test Set)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fdir / "09_pr_curve.png", dpi=150)
    plt.close()

    # ── Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test, probs, n_bins=8, strategy="quantile"
    )
    brier = brier_score_loss(y_test, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", color="#2E75B6", lw=2,
            label="LR calibration")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_title(f"Calibration Curve — LR  |  Brier = {brier:.3f}",
                 fontweight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed MT fraction")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fdir / "10_calibration_curve.png", dpi=150)
    plt.close()

    print(f"\n[plots] Saved confusion matrices, ROC, PR, calibration → {fdir}")


# ════════════════════════════════════════════════════════════════
# SECTION 11 — Feature interpretation plots
# ════════════════════════════════════════════════════════════════

def extract_lr_coefficients(pipeline: Pipeline,
                            feature_names: list) -> pd.DataFrame:
    coefs  = pipeline.named_steps["lr"].coef_[0]
    df_c   = pd.DataFrame({
        "Feature":         feature_names,
        "Coefficient":     coefs.round(4),
        "Abs_Coefficient": np.abs(coefs).round(4),
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    df_c["Rank"] = range(1, len(df_c) + 1)
    return df_c


def plot_coefficients(coef_df: pd.DataFrame, out_dir: Path) -> None:
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    colors = ["#ED7D31" if v > 0 else "#2E75B6"
              for v in coef_df["Coefficient"]]
    fig, ax = plt.subplots(figsize=(8, max(4, len(coef_df) * 0.5)))
    ax.barh(
        coef_df["Feature"][::-1],
        coef_df["Coefficient"][::-1],
        color=colors[::-1], edgecolor="black", linewidth=0.5,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_title("LR Coefficients (standardized scale)\n"
                 "Orange = positive (↑ MT probability)  |  "
                 "Blue = negative (↓ MT probability)",
                 fontweight="bold")
    ax.set_xlabel("Coefficient value")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fdir / "11_coefficients.png", dpi=150)
    plt.close()
    print(f"[plot] Coefficients → {fdir / '11_coefficients.png'}")


def compute_lr_permutation_importance(pipeline: Pipeline,
                                      features: list,
                                      X_test:   pd.DataFrame,
                                      y_test:   pd.Series) -> pd.DataFrame:
    result = sk_perm_importance(
        pipeline, X_test[features], y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    df_p = pd.DataFrame({
        "Feature":             features,
        "Mean_Decrease_PRAUC": result.importances_mean.round(4),
        "Std":                 result.importances_std.round(4),
    }).sort_values("Mean_Decrease_PRAUC", ascending=False).reset_index(drop=True)
    return df_p


def plot_permutation_importance(perm_df: pd.DataFrame, out_dir: Path) -> None:
    fdir = out_dir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(perm_df) * 0.5)))
    ax.barh(
        perm_df["Feature"][::-1],
        perm_df["Mean_Decrease_PRAUC"][::-1],
        xerr=perm_df["Std"][::-1],
        color="#2E75B6", edgecolor="black", linewidth=0.5,
    )
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_title(
        "Permutation Importance — LR\n"
        "Mean decrease in PR-AUC when feature values are shuffled (test set)",
        fontweight="bold",
    )
    ax.set_xlabel("Mean decrease in PR-AUC")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fdir / "12_permutation_importance.png", dpi=150)
    plt.close()
    print(f"[plot] Permutation importance → {fdir / '12_permutation_importance.png'}")


# ════════════════════════════════════════════════════════════════
# SECTION 12 — Save artifacts
# ════════════════════════════════════════════════════════════════

def save_metrics_csv(metrics_list: list, path: Path) -> None:
    pd.DataFrame(metrics_list).to_csv(path, index=False)
    print(f"[save] Metrics CSV → {path}")


def save_coefficients_csv(coef_df: pd.DataFrame, path: Path) -> None:
    coef_df.to_csv(path, index=False)
    print(f"[save] Coefficients CSV → {path}")


def save_hyperparameters_json(info: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    print(f"[save] Hyperparameters JSON → {path}")


def write_final_report(sections: dict,
                       lr_row:   dict,
                       path:     Path) -> None:
    W = 72

    def bar(title):
        return ["=" * W, title.upper(), "=" * W]

    lines = []
    lines += bar("Logistic Regression Workflow — Final Report")
    lines += [
        f"Generated : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Seed      : {RANDOM_SEED}   |   Test size: {TEST_SIZE}",
        f"Data file : {DATA_PATH}",
        "",
    ]

    for title, content in sections.items():
        lines += bar(title)
        lines.append(content)
        lines.append("")

    # ── Placeholder comparison table
    lines += bar("7. Future comparison with teammate models")
    lines += [
        "Model selection criterion: CV PR-AUC on training set only.",
        "This table will be completed when teammates submit their models.",
        "",
    ]
    col_w  = [24, 10, 8, 9, 8, 12, 6, 7, 10]
    hdrs   = ["Model", "CV PR-AUC", "PR-AUC", "ROC-AUC",
              "Recall", "Specificity", "F2", "Brier", "Threshold"]
    sep    = "  ".join("-" * w for w in col_w)
    lines.append("  ".join(h.ljust(w) for h, w in zip(hdrs, col_w)))
    lines.append(sep)

    lr_vals = [
        "Logistic Regression",
        str(lr_row.get("CV_PR-AUC", "—")),
        str(lr_row.get("PR-AUC",    "—")),
        str(lr_row.get("ROC-AUC",   "—")),
        str(lr_row.get("Recall",    "—")),
        str(lr_row.get("Specificity","—")),
        str(lr_row.get("F2",        "—")),
        str(lr_row.get("Brier",     "—")),
        str(lr_row.get("Threshold", "—")),
    ]
    lines.append("  ".join(v.ljust(w) for v, w in zip(lr_vals, col_w)))
    lines.append("  ".join(["Random Forest".ljust(col_w[0])] +
                            ["[TBD]".ljust(w) for w in col_w[1:]]))
    lines.append("  ".join(["XGBoost".ljust(col_w[0])] +
                            ["[TBD]".ljust(w) for w in col_w[1:]]))
    lines.append("")

    # ── Iteration ideas
    lines += bar("8. Iteration ideas")
    lines.append(
        "Based on LR results, consider revisiting the following:\n\n"
        "1. Log-transform for skewed features\n"
        "   Lactate_mmol_L and Arterial_Base_Excess show right skew.\n"
        "   Log-transforming them before scaling may help LR separate\n"
        "   classes more linearly. Test via the existing OOF CV framework.\n\n"
        "2. Probability calibration\n"
        "   If the calibration curve deviates noticeably from the diagonal,\n"
        "   apply sklearn CalibratedClassifierCV (method='sigmoid') on top\n"
        "   of the best pipeline. Brier score will confirm improvement.\n\n"
        "3. GCS strategy revisit\n"
        "   If the impute strategy was chosen but the Δ PR-AUC was close to\n"
        "   the 0.005 cutoff, discuss with the team whether GCS data could be\n"
        "   collected more completely in a future dataset version.\n\n"
        "4. Explicit class weight tuning\n"
        "   Replace class_weight='balanced' with a manual ratio search\n"
        "   (e.g., {0:1, 1:3}, {0:1, 1:4}) inside the existing grid.\n"
        "   No structural changes required.\n\n"
        "5. Threshold clinical alignment\n"
        "   If neither threshold achieves Recall ≥ 0.90 with acceptable\n"
        "   Specificity, consult the clinical team about the minimum\n"
        "   acceptable missed-MT rate and set an explicit Recall floor.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] Final report → {path}")


# ════════════════════════════════════════════════════════════════
# SECTION 13 — main
# ════════════════════════════════════════════════════════════════

def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Logistic Regression Workflow — MT Prediction in Trauma")
    print("=" * 60)

    # ── 1. Load
    df = load_data(DATA_PATH)

    # ── 2. Early removal + encoding (before EDA)
    df = drop_early_columns(df, DROP_ALWAYS)
    df = encode_categoricals(df)

    # ── 3. EDA (Time_to_Hospital_min already removed; Shock_Index present
    #          for descriptive correlation only)
    print("\n[EDA] Running exploratory data analysis ...")
    eda_text = run_eda(df, TARGET, OUTPUT_BASE)

    # ── 4. Split X / y
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # ── 5. Clipping + Shock_Index refresh on clipped values
    X_train, X_test = apply_clinical_clipping(X_train, X_test, CLIP_BOUNDS)
    X_train = recalculate_shock_index(X_train)
    X_test  = recalculate_shock_index(X_test)

    # ── 6. Cross-validation splitter (shared across all CV steps)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # ── 7. GCS experiment  (uses X_train only)
    gcs_strategy, gcs_results = compare_gcs_strategies(X_train, y_train, skf)
    features = get_feature_list(gcs_strategy, X_train)
    print(f"\n[Features] Final set ({len(features)} features): {features}")

    # ── 8. VIF — numeric model features only, training set, descriptive
    numeric_feats = [f for f in features
                     if f not in ("Injury_Type_Coded", "Sex_Coded")]
    vif_df = compute_vif(X_train, numeric_feats)

    # ── 9. Baseline LR
    baseline = build_baseline_lr(features, X_train, y_train, skf)

    # ── 10. Hyperparameter search
    best_pipeline, best_params, best_cv_prauc = run_hyperparameter_search(
        features, X_train, y_train, skf,
    )

    # ── 11. Threshold selection on OOF (training data only)
    thr_f2, thr_spec = select_thresholds_on_oof(
        best_pipeline, features, X_train, y_train, skf,
    )

    # ── 12. Final evaluation — test set called exactly once
    m_f2, m_spec = evaluate_on_test(
        best_pipeline, features,
        X_train, y_train,
        X_test, y_test,
        thr_f2, thr_spec,
        OUTPUT_BASE,
    )

    # ── 13. Coefficient table + plot
    print("\n[Interpretation] LR coefficients ...")
    coef_df = extract_lr_coefficients(best_pipeline, features)
    print(coef_df.to_string(index=False))
    plot_coefficients(coef_df, OUTPUT_BASE)

    # ── 14. Permutation importance on test set
    print("\n[Interpretation] Permutation importance ...")
    perm_df = compute_lr_permutation_importance(
        best_pipeline, features, X_test, y_test,
    )
    print(perm_df.to_string(index=False))
    plot_permutation_importance(perm_df, OUTPUT_BASE)

    # ── 15. Save artefacts
    metrics_rows = []
    m_f2_save = dict(m_f2)
    m_f2_save["CV_PR-AUC"] = round(best_cv_prauc, 3)
    metrics_rows.append(m_f2_save)
    if m_spec:
        m_sp_save = dict(m_spec)
        m_sp_save["CV_PR-AUC"] = round(best_cv_prauc, 3)
        metrics_rows.append(m_sp_save)

    save_metrics_csv(metrics_rows, OUTPUT_BASE / "metrics.csv")
    save_coefficients_csv(coef_df,  OUTPUT_BASE / "model_coefficients.csv")
    save_hyperparameters_json(
        {
            "features":     features,
            "gcs_strategy": gcs_strategy,
            "best_params":  best_params,
            "cv_prauc":     round(best_cv_prauc, 3),
            "thr_f2":       round(thr_f2, 4),
            "thr_spec":     round(thr_spec, 4) if thr_spec is not None else None,
        },
        OUTPUT_BASE / "best_hyperparameters.json",
    )

    # ── 16. Final report
    gcs_text = (
        f"GCS_Score strategy experiment:\n"
        f"  drop  : CV PR-AUC = {gcs_results['drop']['mean']:.4f} "
        f"± {gcs_results['drop']['std']:.4f}\n"
        f"  impute: CV PR-AUC = {gcs_results['impute']['mean']:.4f} "
        f"± {gcs_results['impute']['std']:.4f}\n"
        f"  Δ = {gcs_results['impute']['mean'] - gcs_results['drop']['mean']:+.4f}\n"
        f"  Decision: {gcs_strategy.upper()}\n\n"
        f"Physiological clipping (clinical constants, not data-derived):\n"
        f"  Systolic_BP_mmHg → [40, 250] mmHg\n"
        f"  Heart_Rate_BPM   → [20, 220] bpm\n\n"
        f"Categorical encoding (deterministic, no statistics learned):\n"
        f"  Sex         : Female=1, Male=0\n"
        f"  Injury_Type : Penetrating=1, Blunt=0\n\n"
        f"Shock_Index: retained in dataframe for EDA correlation reporting.\n"
        f"  Excluded from all model fitting because SBP and HR are already\n"
        f"  present — including all three creates perfect multicollinearity."
    )

    feat_text = (
        f"Single feature set (selected before any modeling):\n"
        f"  {features}\n\n"
        f"VIF analysis (training set, descriptive only):\n"
        f"{vif_df.to_string(index=False)}\n\n"
        f"Correlation findings: see eda_report.txt and "
        f"figures/05_correlation_heatmap.png\n"
        f"No features were removed based on VIF or correlation.\n"
        f"Regularization (L1/L2/ElasticNet) handles multicollinearity in LR."
    )

    model_text = (
        f"Baseline LR CV PR-AUC : {baseline['cv_prauc']:.4f}\n"
        f"Best LR CV PR-AUC     : {best_cv_prauc:.4f}\n"
        f"Improvement over base : {best_cv_prauc - baseline['cv_prauc']:+.4f}\n\n"
        f"Best hyperparameters  : {best_params}\n"
        f"Features used         : {features}\n\n"
        f"Thresholds (selected on OOF — training data only):\n"
        f"  F2-optimal     : {thr_f2:.4f}\n"
        f"  Spec≥{MIN_SPECIFICITY}     : "
        + (f"{thr_spec:.4f}" if thr_spec else "constraint not achievable") + "\n\n"
        f"Test set results (F2-threshold):\n"
        + "\n".join(f"  {k:<15}: {v}"
                    for k, v in m_f2.items() if k != "Model")
    )

    diff_text = (
        f"Class imbalance   : 3:1 — compensated with class_weight='balanced'\n"
        f"GCS missingness   : 37.9% — {gcs_strategy} strategy used\n"
        f"Distributional overlap: moderate (see violin plots)\n"
        f"Calibration       : see figures/10_calibration_curve.png "
        f"(Brier = {m_f2.get('Brier', '—')})\n"
        f"Overall difficulty: moderate — LR is a principled, interpretable\n"
        f"  linear baseline; Recall and PR-AUC are the primary metrics."
    )

    lr_row_for_table = {
        "CV_PR-AUC":   round(best_cv_prauc, 3),
        "PR-AUC":      m_f2.get("PR-AUC",      "—"),
        "ROC-AUC":     m_f2.get("ROC-AUC",     "—"),
        "Recall":      m_f2.get("Recall",       "—"),
        "Specificity": m_f2.get("Specificity",  "—"),
        "F2":          m_f2.get("F2",           "—"),
        "Brier":       m_f2.get("Brier",        "—"),
        "Threshold":   round(thr_f2, 3),
    }

    write_final_report(
        sections={
            "1. EDA summary":                  eda_text,
            "2. Preprocessing decisions":       gcs_text,
            "3. Feature engineering decisions": feat_text,
            "4. Model selection":               model_text,
            "5. Estimated task difficulty":     diff_text,
        },
        lr_row=lr_row_for_table,
        path=OUTPUT_BASE / "final_report.txt",
    )

    print("\n" + "=" * 60)
    print("Workflow complete.")
    print(f"Outputs: {OUTPUT_BASE.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
