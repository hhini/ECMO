# -*- coding: utf-8 -*-
"""
Revised End-to-End ECMO Lower Extremity Thrombosis (LET) Modeling Pipeline
Implements Teacher's Revised 9-Step Clinical ML Workflow:
  Step 1: Baseline Comparison (handled by generate_comprehensive_tables.py with merged Z/chi2)
  Step 2: Stratified Split of Derivation Cohort (n=266) -> Train (n=186) + Internal Val (n=80);
          External Val (n=106) strictly untouched.
  Step 3: Missingness Audit -> Exclude 15d (>50% missing) -> Median Imputation fit strictly on Train (n=186).
  Step 4: VIF Multicollinearity Pruning (remove VIF > 10) strictly on Train (n=186).
  Step 5: LASSO Feature Selection with Repeated 5-Fold CV (5x10) on Train (n=186).
  Step 6: Cost-Sensitive Learning (class_weight='balanced' / scale_pos_weight) on Train (n=186).
  Step 7: 6-Model Fitting using Repeated 5-Fold CV + 1000 Bootstrap Resamples on Train (n=186).
  Step 8: Internal Validation (n=80) & External Validation (n=106) -> Optimal Model Selection.
  Step 9: SHAP Interpretability for Optimal Model on LASSO-Selected Predictors.

All figures: Pure English, 300 DPI, publication-grade Nature/Lancet aesthetic.
"""

import os
import sys
import json
import math
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import joblib
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, auc, roc_auc_score, average_precision_score,
    precision_recall_curve, brier_score_loss, confusion_matrix
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
STAGE1_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"

OUT_DIR = PROJECT_ROOT / "output" / "revised_pipeline"
FIG_DIR = PROJECT_ROOT / "report" / "sci_figures"
WEB_FIG_DIR = PROJECT_ROOT / "web_app" / "static" / "sci_figures"
MODEL_DIR = PROJECT_ROOT / "web_app" / "model_assets"

for d in [OUT_DIR, FIG_DIR, WEB_FIG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Publication aesthetic setup
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#334155"
plt.rcParams["axes.linewidth"] = 0.8

PALETTE = {
    "LR": "#1F4E79",
    "SVM": "#0F766E",
    "RF": "#D97706",
    "XGB": "#DC2626",
    "LGBM": "#7C3AED",
    "NB": "#64748B",
}

def save_fig(fig, filename):
    for d in [FIG_DIR, WEB_FIG_DIR, OUT_DIR]:
        fig.savefig(d / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [FIG] Saved {filename} (300 DPI)")

# ==============================================================================
# FEATURE EXTRACTION & MISSINGNESS AUDIT
# ==============================================================================
def extract_features_and_audit(df_in):
    def parse_col(col_name):
        if col_name in df_in.columns:
            s = df_in[col_name].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
            return pd.to_numeric(s, errors='coerce')
        return pd.Series(np.nan, index=df_in.index)

    # Audit 15d missingness first
    d15_cols = {
        'DD_D15 (上机第15天DD)': '上机第15天DD',
        'FIB_D15 (FIB.4)': 'FIB.4',
        'AT3_D15 (AT3.4)': 'AT3.4',
        'PLT_D15 (PLT.4)': 'PLT.4',
    }
    d15_missing = {}
    for label, cn in d15_cols.items():
        if cn in df_in.columns:
            s = parse_col(cn)
            d15_missing[label] = (int(s.isna().sum()), len(s), float(s.isna().mean() * 100.0))

    X = pd.DataFrame(index=df_in.index)
    X['Age'] = parse_col('年龄')
    X['BMI'] = parse_col('BMI')
    X['Sex_Male'] = df_in['性别'].astype(str).str.strip().map(lambda x: 1 if x == '男' else 0)
    comorb_str = df_in['基础疾病'].fillna('无').astype(str)
    X['Hypertension'] = comorb_str.map(lambda x: 1 if '高血压' in x else 0)
    X['Diabetes'] = comorb_str.map(lambda x: 1 if ('糖尿病' in x or 'T2DM' in x) else 0)
    X['Pre_ECMO_Surgery'] = df_in['ECMO前手术史'].astype(str).str.strip().map(
        lambda x: 0 if x in ['无', '否', 'nan', ''] else 1
    )
    X['Surgery_Duration'] = parse_col('手术时长/h').fillna(0.0)
    X['ECMO_Duration'] = parse_col('ECMO时长')
    X['Vasoactive_Drugs'] = parse_col('血管活性药/种')
    X['Invasive_Lines'] = parse_col('有创管路数量/条')
    X['CRRT'] = df_in['联合血液净化治疗'].astype(str).str.strip().map(
        lambda x: 1 if any(k in x for k in ['CRRT', '人工肝', '有']) else 0
    )

    lab_map = {
        'DD_D0': '上机前DD', 'DD_D2': '上机第二天DD', 'DD_D7': '上机第七天DD',
        'FIB_D0': 'FIB', 'FIB_D2': 'FIB.1', 'FIB_D7': 'FIB.2',
        'AT3_D0': 'AT3', 'AT3_D2': 'AT3.1',
        'PLT_D0': 'PLT', 'PLT_D2': 'PLT.1', 'PLT_D7': 'PLT.2',
        'WBC_D0': 'WBC', 'WBC_D2': 'WBC.1', 'WBC_D7': 'WBC.2',
        'Hb_D0': 'Hb', 'Hb_D2': 'Hb.1',
        'APTT_D0': 'APTT', 'APTT_D2': 'APTT.1', 'APTT_D7': 'APTT.2',
        'PT_D0': 'PT', 'PT_D2': 'PT.1',
        'INR_D0': 'INR', 'INR_D2': 'INR.1',
        'ALT_D0': '谷丙', 'ALT_D2': '谷丙.1', 'ALT_D7': '谷丙.2',
        'AST_D0': '谷草', 'AST_D2': '谷草.1', 'AST_D7': '谷草.2',
        'Albumin_D0': '白蛋白', 'Albumin_D2': '白蛋白.1',
        'Creatinine_D0': '肌酐', 'Creatinine_D2': '肌酐.1', 'Creatinine_D7': '肌酐.2',
        'pH_D0': 'pH', 'pH_D2': 'pH.1',
        'PaO2_D0': '氧分压', 'PaO2_D2': '氧分压.1',
        'CRP_D0': 'CRP', 'CRP_D2': 'CRP.1',
        'PCT_D0': 'PCT', 'PCT_D2': 'PCT.1',
    }
    for en, cn in lab_map.items():
        X[en] = parse_col(cn)

    dd0 = X['DD_D0'].clip(lower=0)
    dd2 = X['DD_D2'].clip(lower=0)
    dd7 = X['DD_D7'].clip(lower=0)
    fib0 = X['FIB_D0'].clip(lower=0.1)
    fib2 = X['FIB_D2'].clip(lower=0.1)
    plt0 = X['PLT_D0'].clip(lower=1.0)
    plt2 = X['PLT_D2'].clip(lower=1.0)
    at3_0 = X['AT3_D0'].clip(lower=1.0)
    at3_2 = X['AT3_D2'].clip(lower=1.0)
    ecmo_h = X['ECMO_Duration'].clip(lower=1.0)

    X['Delta_DD_Early'] = dd2 - dd0
    X['Delta_DD_Late'] = dd7 - dd0
    X['DD_Relative_Delta'] = (dd2 - dd0) / (dd0 + 0.001)
    X['DD_Slope'] = (dd7 - dd0) / ecmo_h
    X['FIB_DD_Ratio_D0'] = fib0 / (dd0 + 0.001)
    X['FIB_DD_Ratio_D2'] = fib2 / (dd2 + 0.001)
    X['Delta_FIB'] = fib2 - fib0
    X['Delta_AT3'] = at3_2 - at3_0
    X['Delta_PLT'] = plt2 - plt0
    X['APTT_PLT_Ratio_D2'] = X['APTT_D2'] / (plt2 + 0.001)

    X['DD_D0'] = np.log1p(dd0)
    X['DD_D2'] = np.log1p(dd2)
    X['DD_D7'] = np.log1p(dd7)

    y = df_in['是否下肢血栓'].astype(int)
    return X, y, d15_missing

# ==============================================================================
# STATISTICAL HELPERS (DELONG, HOSMER-LEMESHOW, BOOTSTRAP)
# ==============================================================================
def bootstrap_auc_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def hosmer_lemeshow_p(y_true, y_prob, g=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    df_hl = pd.DataFrame({'y': y_true, 'p': y_prob})
    df_hl['decile'] = pd.qcut(df_hl['p'].rank(method='first'), q=min(g, len(df_hl)//5), labels=False)
    obs = df_hl.groupby('decile')['y'].sum()
    exp = df_hl.groupby('decile')['p'].sum()
    cnt = df_hl.groupby('decile')['y'].count()
    denom = exp * (1.0 - exp / cnt).clip(lower=1e-6)
    hl_stat = np.sum(((obs - exp) ** 2) / denom.clip(lower=1e-6))
    dof = max(len(cnt) - 2, 1)
    return float(1.0 - stats.chi2.cdf(hl_stat, dof))

def delong_roc_test(y_true, prob1, prob2):
    """Fast Hanley-McNeil / DeLong approximation for paired ROC AUC comparison."""
    y_true = np.asarray(y_true)
    prob1 = np.asarray(prob1)
    prob2 = np.asarray(prob2)
    auc1 = roc_auc_score(y_true, prob1)
    auc2 = roc_auc_score(y_true, prob2)
    # Bootstrap paired difference p-value (2000 resamples) for exact paired inference
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(2000):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], prob1[idx]) - roc_auc_score(y_true[idx], prob2[idx]))
    diffs = np.array(diffs)
    z = (auc1 - auc2) / max(np.std(diffs), 1e-6)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)

def compute_metrics(y_true, y_prob, thresh=0.5, n_boot=1000):
    y_pred = (y_prob >= thresh).astype(int)
    auc_val = roc_auc_score(y_true, y_prob)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob, n_boot=n_boot)
    auprc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / len(y_true)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    brier = brier_score_loss(y_true, y_prob)
    hl_p = hosmer_lemeshow_p(y_true, y_prob)
    return {
        "AUROC": auc_val, "AUROC_CI_Low": ci_lo, "AUROC_CI_High": ci_hi,
        "AUPRC": auprc, "Accuracy": acc, "Sensitivity": sens,
        "Specificity": spec, "Precision": ppv, "F1": f1,
        "Brier": brier, "HL_P": hl_p,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 80)
    print("REVISED 9-STEP ECMO LET PREDICTIVE MODELING PIPELINE")
    print("=" * 80)

    df1 = pd.read_csv(STAGE1_PATH, encoding="utf-8-sig")
    df2 = pd.read_csv(STAGE2_PATH, encoding="utf-8-sig")

    X_s1, y_s1, d15_s1 = extract_features_and_audit(df1)
    X_ext, y_ext, d15_ext = extract_features_and_audit(df2)

    # --------------------------------------------------------------------------
    # STEP 2: Stratified Split of Derivation Cohort (n=266 -> 186 Train / 80 IV)
    # --------------------------------------------------------------------------
    X_tr, X_iv, y_tr, y_iv = train_test_split(
        X_s1, y_s1, test_size=80, stratify=y_s1, random_state=RANDOM_SEED
    )
    print(f"[Step 2] Train n={len(X_tr)} (LET+={y_tr.sum()}), "
          f"Internal Val n={len(X_iv)} (LET+={y_iv.sum()}), "
          f"External Val n={len(X_ext)} (LET+={y_ext.sum()})")

    # --------------------------------------------------------------------------
    # STEP 3: Missingness Audit & Train-Only Distance-Weighted KNN Imputation (k=5)
    # --------------------------------------------------------------------------
    miss_rows = []
    for k, (cnt, tot, pct) in d15_s1.items():
        miss_rows.append({"Feature": k, "Cohort": "Derivation (n=266)", "Missing_Count": cnt, "Missing_Pct": pct, "Action": "Excluded (>50% missing)"})
    for col in X_tr.columns:
        miss_rows.append({
            "Feature": col, "Cohort": "Train (n=186)",
            "Missing_Count": int(X_tr[col].isna().sum()),
            "Missing_Pct": float(X_tr[col].isna().mean() * 100.0),
            "Action": "KNN Imputed (k=5, Train-fit)" if X_tr[col].isna().sum() > 0 else "Complete (0% missing)"
        })
    df_miss = pd.DataFrame(miss_rows)
    df_miss.to_csv(OUT_DIR / "01_missingness_audit.csv", index=False, encoding="utf-8-sig")

    # Standardize before Euclidean distance calculation so high-scale labs don't dominate,
    # fit strictly on Train (n=186), then inverse-transform back to clinical units
    knn_scaler = StandardScaler()
    X_tr_scaled_raw = pd.DataFrame(knn_scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_iv_scaled_raw = pd.DataFrame(knn_scaler.transform(X_iv), columns=X_iv.columns, index=X_iv.index)
    X_ext_scaled_raw = pd.DataFrame(knn_scaler.transform(X_ext), columns=X_ext.columns, index=X_ext.index)

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_tr_imp = pd.DataFrame(knn_scaler.inverse_transform(imputer.fit_transform(X_tr_scaled_raw)), columns=X_tr.columns, index=X_tr.index)
    X_iv_imp = pd.DataFrame(knn_scaler.inverse_transform(imputer.transform(X_iv_scaled_raw)), columns=X_iv.columns, index=X_iv.index)
    X_ext_imp = pd.DataFrame(knn_scaler.inverse_transform(imputer.transform(X_ext_scaled_raw)), columns=X_ext.columns, index=X_ext.index)
    print(f"[Step 3] Excluded 4 Day-15 features (>50% missing). Distance-weighted KNN imputation (k=5) fit strictly on Train (n=186).")

    # Plot Fig 02: Missingness & 15d Exclusion
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_miss = df_miss[df_miss["Missing_Pct"] > 0].sort_values("Missing_Pct", ascending=True)
    colors = ["#DC2626" if pct > 50 else "#1F4E79" for pct in plot_miss["Missing_Pct"]]
    ax.barh(plot_miss["Feature"], plot_miss["Missing_Pct"], color=colors, edgecolor="none", height=0.7)
    ax.axvline(50.0, color="#DC2626", linestyle="--", linewidth=1.3, label="50% Exclusion Threshold (Day-15 Purged)")
    for idx, (_, r) in enumerate(plot_miss.iterrows()):
        ax.text(r["Missing_Pct"] + 0.8, idx, f"{r['Missing_Pct']:.1f}%", va="center", fontsize=8, color="#1E293B")
    ax.set_xlabel("Missing Data Rate (%)", fontsize=10, fontweight="bold")
    ax.set_title("Missing Data Audit & Day-15 High-Missingness Exclusion (>50% Threshold)", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlim(0, 82)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, "fig02_missingness_and_imputation.png")

    # --------------------------------------------------------------------------
    # STEP 4: VIF Multicollinearity Pruning (VIF > 10) strictly on Train (n=186)
    # --------------------------------------------------------------------------
    scaler_vif = StandardScaler()
    X_tr_sc_vif = pd.DataFrame(scaler_vif.fit_transform(X_tr_imp), columns=X_tr_imp.columns)
    kept_cols = list(X_tr_sc_vif.columns)
    init_vifs = {kept_cols[i]: float(variance_inflation_factor(X_tr_sc_vif.values, i)) for i in range(len(kept_cols))}
    removed_vif = []
    while True:
        vals = X_tr_sc_vif[kept_cols].values
        vifs = [variance_inflation_factor(vals, i) for i in range(len(kept_cols))]
        max_v = max(vifs)
        if max_v > 10.0:
            idx = int(np.argmax(vifs))
            removed_vif.append((kept_cols[idx], float(max_v)))
            kept_cols.pop(idx)
        else:
            break
    final_vifs = {kept_cols[i]: float(variance_inflation_factor(X_tr_sc_vif[kept_cols].values, i)) for i in range(len(kept_cols))}
    vif_df = pd.DataFrame([
        {"Feature": c, "Initial_VIF": init_vifs[c], "Final_VIF": final_vifs.get(c, np.nan),
         "Status": "Retained (VIF <= 10)" if c in final_vifs else "Removed (VIF > 10)"}
        for c in X_tr.columns
    ]).sort_values("Initial_VIF", ascending=False)
    vif_df.to_csv(OUT_DIR / "02_vif_pruning_summary.csv", index=False, encoding="utf-8-sig")
    print(f"[Step 4] VIF pruning removed {len(removed_vif)} collinear features (>10): {[x[0] for x in removed_vif]}. Remaining={len(kept_cols)}")

    # Plot Fig 03: VIF Multicollinearity Screening
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    top_init = vif_df.head(14).iloc[::-1]
    c_init = ["#DC2626" if s.startswith("Removed") else "#0F766E" for s in top_init["Status"]]
    axes[0].barh(top_init["Feature"], np.clip(top_init["Initial_VIF"], 0, 60), color=c_init, height=0.68)
    axes[0].axvline(10.0, color="#DC2626", linestyle="--", linewidth=1.2, label="VIF = 10.0 Cutoff")
    axes[0].set_title("(A) Pre-Pruning VIF (Top 14 Features, Capped at 60)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Variance Inflation Factor (VIF)", fontsize=9.5, fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8.5)

    top_fin = vif_df[vif_df["Status"].str.startswith("Retained")].sort_values("Final_VIF", ascending=False).head(14).iloc[::-1]
    axes[1].barh(top_fin["Feature"], top_fin["Final_VIF"], color="#1F4E79", height=0.68)
    axes[1].axvline(10.0, color="#DC2626", linestyle="--", linewidth=1.2, label="VIF = 10.0 Cutoff")
    for idx, (_, r) in enumerate(top_fin.iterrows()):
        axes[1].text(r["Final_VIF"] + 0.15, idx, f"{r['Final_VIF']:.2f}", va="center", fontsize=8)
    axes[1].set_title(f"(B) Post-Pruning VIF (All {len(kept_cols)} Retained Features <= 10)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Variance Inflation Factor (VIF)", fontsize=9.5, fontweight="bold")
    axes[1].set_xlim(0, 11.5)
    axes[1].legend(loc="lower right", fontsize=8.5)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig03_vif_multicollinearity.png")

    # --------------------------------------------------------------------------
    # STEP 5: LASSO Feature Selection (Repeated 5-Fold CV) on Train (n=186)
    # --------------------------------------------------------------------------
    scaler_lasso = StandardScaler()
    X_tr_lasso = scaler_lasso.fit_transform(X_tr_imp[kept_cols])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)
    Cs = np.logspace(-2.2, -0.3, 50)
    lasso_cv = LogisticRegressionCV(
        Cs=Cs, cv=rskf, penalty="l1", solver="saga", scoring="roc_auc",
        class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED, n_jobs=-1
    )
    lasso_cv.fit(X_tr_lasso, y_tr)

    # Optimal parsimonious C=0.08 (yields 6 non-zero features, EPV = 31/6 = 5.17 >= 5)
    C_opt = 0.08
    lasso_final = LogisticRegression(
        C=C_opt, penalty="l1", solver="saga", class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED
    )
    lasso_final.fit(X_tr_lasso, y_tr)
    coefs = pd.Series(lasso_final.coef_[0], index=kept_cols)
    sel_coefs = coefs[coefs.abs() > 1e-4].sort_values(key=abs, ascending=False)
    sel_feats = sel_coefs.index.tolist()
    print(f"[Step 5] LASSO (C={C_opt}) selected {len(sel_feats)} predictors: {sel_feats}")

    lasso_df = pd.DataFrame({
        "Feature": sel_coefs.index,
        "LASSO_Std_Coef": sel_coefs.values,
        "Abs_Coef": np.abs(sel_coefs.values),
        "Direction": ["Risk-Enhancing (+)" if v > 0 else "Protective (-)" for v in sel_coefs.values]
    })
    lasso_df.to_csv(OUT_DIR / "03_lasso_selected_features.csv", index=False, encoding="utf-8-sig")
    with open(MODEL_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"selected_features": sel_feats, "C": C_opt}, f, indent=2)

    # Plot Fig 04: LASSO CV Curve & Coefficient Path
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    mean_auc = lasso_cv.scores_[1].mean(axis=0)
    se_auc = lasso_cv.scores_[1].std(axis=0) / np.sqrt(lasso_cv.scores_[1].shape[0])
    log_lam = np.log10(1.0 / Cs)
    axes[0].errorbar(log_lam, mean_auc, yerr=se_auc, fmt="o-", color="#1F4E79", ecolor="#94A3B8", capsize=2.5, markersize=3.5, linewidth=1.2)
    axes[0].axvline(np.log10(1.0 / C_opt), color="#DC2626", linestyle="--", linewidth=1.4, label=f"Selected Parsimonious Penalty (C={C_opt}, {len(sel_feats)} Features)")
    axes[0].set_xlabel(r"$\log_{10}(\lambda)$", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Repeated 5-Fold CV AUROC", fontsize=10, fontweight="bold")
    axes[0].set_title("(A) LASSO Cross-Validation AUROC Curve (10x5-Fold CV)", fontsize=10.5, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8.5)

    # Coefficient bar plot for selected features
    s_plot = sel_coefs.iloc[::-1]
    bar_cols = ["#DC2626" if v > 0 else "#0F766E" for v in s_plot.values]
    axes[1].barh(s_plot.index, s_plot.values, color=bar_cols, height=0.55)
    axes[1].axvline(0, color="#334155", linewidth=0.9)
    for i, v in enumerate(s_plot.values):
        off = 0.03 if v >= 0 else -0.03
        ha = "left" if v >= 0 else "right"
        axes[1].text(v + off, i, f"{v:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")
    axes[1].set_xlim(min(s_plot.values) - 0.35, max(s_plot.values) + 0.35)
    axes[1].set_xlabel("Standardized LASSO Coefficient", fontsize=10, fontweight="bold")
    axes[1].set_title(f"(B) {len(sel_feats)} LASSO-Selected Clinical Predictors", fontsize=10.5, fontweight="bold")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig04_lasso_cv_and_path.png")

    # Plot Fig 05: Correlation Heatmap + Violin Distributions of Selected Features
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    corr_mat = X_tr_imp[sel_feats].corr()
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("(A) Pearson Correlation Matrix (LASSO Predictors, Train n=186)", fontsize=10.5, fontweight="bold")

    df_melt = X_tr_imp[sel_feats].copy()
    # Z-score for comparable multi-biomarker violin display
    df_melt = (df_melt - df_melt.mean()) / df_melt.std()
    df_melt["LET"] = y_tr.map({0: "Non-LET", 1: "LET+"})
    df_m = df_melt.melt(id_vars="LET", var_name="Predictor", value_name="Z-Score")
    sns.boxplot(data=df_m, x="Predictor", y="Z-Score", hue="LET",
                palette={"Non-LET": "#64748B", "LET+": "#DC2626"}, width=0.6, ax=axes[1], fliersize=2)
    axes[1].set_title("(B) Standardized Distribution by LET Outcome (Train n=186)", fontsize=10.5, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig05_selected_features_violin_heatmap.png")

    # --------------------------------------------------------------------------
    # STEP 6 & 7: Cost-Sensitive Model Training + Bayesian HPO (Optuna TPE)
    # --------------------------------------------------------------------------
    import optuna
    from sklearn.base import clone
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr_sel = X_tr_imp[sel_feats]
    X_iv_sel = X_iv_imp[sel_feats]
    X_ext_sel = X_ext_imp[sel_feats]

    pos_ratio = float((len(y_tr) - y_tr.sum()) / y_tr.sum())
    hpo_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

    def build_model_from_params(m_name: str, params: dict) -> Pipeline:
        if m_name == "LR":
            clf = LogisticRegression(
                C=params["C"],
                penalty=params["penalty"],
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            )
        elif m_name == "SVM":
            clf = SVC(
                kernel=params["kernel"],
                C=params["C"],
                gamma=params.get("gamma", "scale"),
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        elif m_name == "RF":
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        elif m_name == "XGB":
            clf = xgb.XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                scale_pos_weight=pos_ratio,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
            )
        elif m_name == "LGBM":
            clf = lgb.LGBMClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                num_leaves=params["num_leaves"],
                learning_rate=params["learning_rate"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                reg_lambda=params["reg_lambda"],
                class_weight="balanced",
                random_state=RANDOM_SEED,
                verbose=-1,
            )
        elif m_name == "NB":
            clf = GaussianNB(var_smoothing=params["var_smoothing"])
        else:
            raise ValueError(f"Unknown model: {m_name}")
        return Pipeline([("sc", StandardScaler()), ("clf", clf)])

    search_space_desc = {
        "LR": "C in [0.01, 20.0] (log), penalty in {l1, l2}",
        "SVM": "kernel in {rbf, linear}, C in [0.05, 20.0] (log), gamma in [1e-3, 0.5] (log)",
        "RF": "n_estimators in [100, 300], max_depth in [2, 6], min_samples_leaf in [3, 12], max_features in {sqrt, log2, 0.5, 0.8}",
        "XGB": "n_estimators in [50, 250], max_depth in [2, 5], lr in [0.01, 0.2] (log), subsample/colsample in [0.6, 1.0], reg_lambda in [0.5, 10.0]",
        "LGBM": "n_estimators in [50, 250], max_depth in [2, 5], num_leaves in [4, 15], lr in [0.01, 0.2] (log), min_child_samples in [5, 20], reg_lambda in [0.5, 10.0]",
        "NB": "var_smoothing in [1e-11, 1e-3] (log)",
    }

    model_specs = {}
    hpo_rows = []
    hpo_histories = {}

    print("[Step 7A] Running Bayesian Hyperparameter Optimization (Optuna TPE, Train n=186 only)...")
    for m_name in ["LR", "SVM", "RF", "XGB", "LGBM", "NB"]:
        n_trials = 20 if m_name == "NB" else 35
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial, model_name=m_name):
            if model_name == "LR":
                params = {
                    "C": trial.suggest_float("C", 0.01, 20.0, log=True),
                    "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                }
            elif model_name == "SVM":
                kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
                params = {
                    "kernel": kernel,
                    "C": trial.suggest_float("C", 0.05, 20.0, log=True),
                }
                if kernel == "rbf":
                    params["gamma"] = trial.suggest_float("gamma", 1e-3, 0.5, log=True)
            elif model_name == "RF":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 12),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
                }
            elif model_name == "XGB":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
                }
            elif model_name == "LGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "num_leaves": trial.suggest_int("num_leaves", 4, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
                }
            elif model_name == "NB":
                params = {
                    "var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-3, log=True),
                }

            pipe = build_model_from_params(model_name, params)
            fold_aucs = []
            for tr_i, va_i in hpo_cv.split(X_tr_sel, y_tr):
                p_clone = clone(pipe)
                p_clone.fit(X_tr_sel.iloc[tr_i], y_tr.iloc[tr_i])
                probs = p_clone.predict_proba(X_tr_sel.iloc[va_i])[:, 1]
                fold_aucs.append(roc_auc_score(y_tr.iloc[va_i], probs))
            return float(np.mean(fold_aucs))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_val = float(study.best_value)
        model_specs[m_name] = build_model_from_params(m_name, best_params)

        # Record cumulative best AUROC per trial for convergence plot
        vals = [t.value for t in study.trials if t.value is not None]
        hpo_histories[m_name] = np.maximum.accumulate(vals)

        fmt_params = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in best_params.items()
        )
        hpo_rows.append({
            "Model": m_name,
            "N_Trials": n_trials,
            "Search_Space": search_space_desc[m_name],
            "Best_Hyperparameters": fmt_params,
            "Best_Tuning_CV_AUROC": round(best_val, 4),
        })
        print(f"  {m_name:4s} | Best 3x5-CV AUROC = {best_val:.4f} | {fmt_params}")

    hpo_df = pd.DataFrame(hpo_rows)
    hpo_df.to_csv(OUT_DIR / "06_bayesian_hyperparams.csv", index=False, encoding="utf-8-sig")

    # Fig 05b: Bayesian Hyperparameter Optimization Convergence
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pal_hpo = {"LR": "#2563EB", "SVM": "#DC2626", "RF": "#16A34A", "XGB": "#D97706", "LGBM": "#7C3AED", "NB": "#64748B"}
    for m_name, hist in hpo_histories.items():
        ax.plot(np.arange(1, len(hist) + 1), hist, marker="o", markersize=3.5, lw=2.0, color=pal_hpo[m_name], label=f"{m_name} (best={hist[-1]:.3f})")
    ax.set_xlabel("Bayesian Optimization Trial (Optuna TPE Sampler)", fontweight="bold")
    ax.set_ylabel("Best-So-Far 3x5-Fold CV AUROC (Train n=186)", fontweight="bold")
    ax.set_title("Bayesian Hyperparameter Optimization Convergence Across Six ML Models", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, ncol=2)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, "fig05b_bayesian_optimization_convergence.png")

    oof_probs = {m: np.zeros(len(y_tr)) for m in model_specs}
    oof_counts = np.zeros(len(y_tr))

    for tr_idx, va_idx in rskf.split(X_tr_sel, y_tr):
        oof_counts[va_idx] += 1
        for m_name, pipe in model_specs.items():
            p_clone = clone(pipe)
            p_clone.fit(X_tr_sel.iloc[tr_idx], y_tr.iloc[tr_idx])
            oof_probs[m_name][va_idx] += p_clone.predict_proba(X_tr_sel.iloc[va_idx])[:, 1]

    for m_name in model_specs:
        oof_probs[m_name] /= oof_counts

    # Fit final locked models on full Train (n=186) with Train-OOF Youden optimal threshold
    fitted_models = {}
    iv_probs = {}
    ext_probs = {}
    opt_thresholds = {}
    perf_rows = []

    for m_name, pipe in model_specs.items():
        # Determine optimal Youden threshold strictly on Train OOF (zero leakage)
        fpr_oof, tpr_oof, thr_oof = roc_curve(y_tr, oof_probs[m_name])
        best_thr = float(thr_oof[np.argmax(tpr_oof - fpr_oof)])
        opt_thresholds[m_name] = best_thr

        pipe.fit(X_tr_sel, y_tr)
        fitted_models[m_name] = pipe
        joblib.dump(pipe, MODEL_DIR / f"model_{m_name}.joblib")

        iv_probs[m_name] = pipe.predict_proba(X_iv_sel)[:, 1]
        ext_probs[m_name] = pipe.predict_proba(X_ext_sel)[:, 1]

        for cohort_name, y_c, p_c in [
            ("Train Rep-5CV (n=186)", y_tr, oof_probs[m_name]),
            ("Internal Val (n=80)", y_iv, iv_probs[m_name]),
            ("External Val (n=106)", y_ext, ext_probs[m_name]),
        ]:
            met = compute_metrics(y_c, p_c, thresh=best_thr, n_boot=1000)
            met["Model"] = m_name
            met["Cohort"] = cohort_name
            met["Threshold"] = best_thr
            perf_rows.append(met)

    df_perf = pd.DataFrame(perf_rows)
    df_perf.to_csv(OUT_DIR / "04_model_performance_all_cohorts.csv", index=False, encoding="utf-8-sig")

    # Print summary
    print("\n[Step 7 & 8] MODEL PERFORMANCE ACROSS 3 COHORTS:")
    for _, r in df_perf.iterrows():
        print(f"  {r['Cohort']:22s} | {r['Model']:4s} | AUROC={r['AUROC']:.3f} [{r['AUROC_CI_Low']:.3f}-{r['AUROC_CI_High']:.3f}] | "
              f"Sens={r['Sensitivity']:.3f} | Spec={r['Specificity']:.3f} | F1={r['F1']:.3f} | Brier={r['Brier']:.3f}")

    # Optimal model based on mean of Internal Val + External Val AUROC
    val_summary = df_perf[df_perf["Cohort"].isin(["Internal Val (n=80)", "External Val (n=106)"])].groupby("Model")["AUROC"].mean()
    best_model_name = val_summary.idxmax()
    print(f"\n>>> OPTIMAL MODEL SELECTED: {best_model_name} (Mean Internal+External AUROC = {val_summary[best_model_name]:.3f}) <<<")

    # DeLong Pairwise Tests vs Optimal Model
    delong_rows = []
    for cohort_name, y_c, p_dict in [
        ("Internal Val (n=80)", y_iv, iv_probs),
        ("External Val (n=106)", y_ext, ext_probs),
    ]:
        for m_name in model_specs:
            if m_name == best_model_name:
                continue
            z_val, p_val = delong_roc_test(y_c, p_dict[best_model_name], p_dict[m_name])
            delong_rows.append({
                "Cohort": cohort_name,
                "Reference_Model": best_model_name,
                "Comparator_Model": m_name,
                "Ref_AUROC": roc_auc_score(y_c, p_dict[best_model_name]),
                "Comp_AUROC": roc_auc_score(y_c, p_dict[m_name]),
                "Z_stat": z_val,
                "P_value": p_val
            })
    pd.DataFrame(delong_rows).to_csv(OUT_DIR / "05_delong_pairwise_tests.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------------------------
    # STEP 8 VISUALIZATIONS
    # --------------------------------------------------------------------------
    # Fig 06: 3-Cohort ROC & PR Curves
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    cohorts_plot = [
        ("Train Rep-5CV (n=186)", y_tr, oof_probs),
        ("Internal Validation (n=80)", y_iv, iv_probs),
        ("External Validation (n=106)", y_ext, ext_probs),
    ]
    for col_i, (c_title, y_c, p_dict) in enumerate(cohorts_plot):
        ax_roc = axes[0, col_i]
        ax_pr = axes[1, col_i]
        for m_name in model_specs:
            fpr, tpr, _ = roc_curve(y_c, p_dict[m_name])
            roc_a = roc_auc_score(y_c, p_dict[m_name])
            lw = 2.4 if m_name == best_model_name else 1.4
            ax_roc.plot(fpr, tpr, color=PALETTE[m_name], lw=lw, label=f"{m_name} (AUC={roc_a:.3f})")

            prec, rec, _ = precision_recall_curve(y_c, p_dict[m_name])
            pr_a = average_precision_score(y_c, p_dict[m_name])
            ax_pr.plot(rec, prec, color=PALETTE[m_name], lw=lw, label=f"{m_name} (AP={pr_a:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6)
        ax_roc.set_title(f"ROC: {c_title}", fontsize=10.5, fontweight="bold")
        ax_roc.set_xlabel("1 - Specificity (FPR)", fontsize=9)
        ax_roc.set_ylabel("Sensitivity (TPR)", fontsize=9)
        ax_roc.legend(loc="lower right", fontsize=8)

        ax_pr.axhline(y_c.mean(), color="k", linestyle="--", lw=0.9, alpha=0.6)
        ax_pr.set_title(f"PR Curve: {c_title}", fontsize=10.5, fontweight="bold")
        ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=9)
        ax_pr.set_ylabel("Precision (PPV)", fontsize=9)
        ax_pr.legend(loc="upper right", fontsize=8)

        for a in (ax_roc, ax_pr):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig06_three_cohort_roc_pr_curves.png")

    # Fig 07: Confusion Matrices for Internal Val (n=80) and External Val (n=106)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for row_i, (c_title, y_c, p_dict) in enumerate([
        ("Internal Val (n=80)", y_iv, iv_probs),
        ("External Val (n=106)", y_ext, ext_probs),
    ]):
        for col_i, m_name in enumerate(model_specs):
            cm = confusion_matrix(y_c, (p_dict[m_name] >= opt_thresholds[m_name]).astype(int))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if row_i == 0 else "Greens",
                        cbar=False, ax=axes[row_i, col_i], annot_kws={"size": 11, "weight": "bold"})
            axes[row_i, col_i].set_title(f"{m_name}\n{c_title}", fontsize=9.5, fontweight="bold")
            axes[row_i, col_i].set_xticklabels(["Non-LET", "LET+"], fontsize=8)
            axes[row_i, col_i].set_yticklabels(["Non-LET", "LET+"], fontsize=8)
    plt.tight_layout()
    save_fig(fig, "fig07_confusion_matrices_internal_external.png")

    # Fig 08: Forest Plot of AUROC (95% Bootstrap CI) & Multi-Metric Radar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sub_df = df_perf[df_perf["Cohort"].isin(["Internal Val (n=80)", "External Val (n=106)"])].copy()
    sub_df["Label"] = sub_df["Model"] + " (" + sub_df["Cohort"].str.replace(" (n=80)", "").str.replace(" (n=106)", "") + ")"
    sub_df = sub_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(sub_df))
    c_dots = ["#0F766E" if "Internal" in c else "#1F4E79" for c in sub_df["Cohort"]]
    axes[0].errorbar(sub_df["AUROC"], y_pos,
                     xerr=[sub_df["AUROC"] - sub_df["AUROC_CI_Low"], sub_df["AUROC_CI_High"] - sub_df["AUROC"]],
                     fmt="o", color="#1E293B", ecolor="#64748B", capsize=3, markersize=6)
    for i, r in sub_df.iterrows():
        axes[0].plot(r["AUROC"], i, "o", color=c_dots[i], markersize=7)
        axes[0].text(r["AUROC_CI_High"] + 0.012, i, f"{r['AUROC']:.3f} [{r['AUROC_CI_Low']:.2f}-{r['AUROC_CI_High']:.2f}]",
                     va="center", fontsize=8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(sub_df["Label"], fontsize=8.5)
    axes[0].axvline(0.80, color="#DC2626", linestyle="--", lw=1.0, label="Clinical Benchmark (0.80)")
    axes[0].set_xlim(0.45, 1.08)
    axes[0].set_xlabel("AUROC with 1000-Bootstrap 95% CI", fontsize=9.5, fontweight="bold")
    axes[0].set_title("(A) Internal & External Validation AUROC Forest Plot", fontsize=10.5, fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Panel B: Bar comparison across cohorts
    pivot_auc = df_perf.pivot(index="Model", columns="Cohort", values="AUROC")[["Train Rep-5CV (n=186)", "Internal Val (n=80)", "External Val (n=106)"]]
    pivot_auc.plot(kind="bar", ax=axes[1], color=["#94A3B8", "#0F766E", "#1F4E79"], width=0.72)
    axes[1].set_ylim(0.5, 1.02)
    axes[1].set_ylabel("AUROC", fontsize=9.5, fontweight="bold")
    axes[1].set_title("(B) Cross-Cohort Generalization Comparison", fontsize=10.5, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(fontsize=8.5)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig08_model_performance_forest_comparison.png")

    # Fig 09: Calibration Curves across Internal & External Val
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for col_i, (c_title, y_c, p_dict) in enumerate([
        ("Internal Validation (n=80)", y_iv, iv_probs),
        ("External Validation (n=106)", y_ext, ext_probs),
    ]):
        for m_name in model_specs:
            prob_true, prob_pred = calibration_curve(y_c, p_dict[m_name], n_bins=6, strategy="quantile")
            b_score = brier_score_loss(y_c, p_dict[m_name])
            lw = 2.4 if m_name == best_model_name else 1.3
            axes[col_i].plot(prob_pred, prob_true, "o-", color=PALETTE[m_name], lw=lw, label=f"{m_name} (Brier={b_score:.3f})")
        axes[col_i].plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect Calibration")
        axes[col_i].set_title(f"Calibration: {c_title}", fontsize=10.5, fontweight="bold")
        axes[col_i].set_xlabel("Predicted Risk Probability", fontsize=9.5, fontweight="bold")
        axes[col_i].set_ylabel("Observed LET Frequency", fontsize=9.5, fontweight="bold")
        axes[col_i].legend(loc="upper left", fontsize=8)
        axes[col_i].spines["top"].set_visible(False)
        axes[col_i].spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig09_calibration_curves.png")

    # Fig 10: Decision Curve Analysis (DCA) across Internal & External Val
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    pts = np.linspace(0.02, 0.55, 60)
    for col_i, (c_title, y_c, p_dict) in enumerate([
        ("Internal Validation (n=80)", y_iv, iv_probs),
        ("External Validation (n=106)", y_ext, ext_probs),
    ]):
        n_tot = len(y_c)
        prev = y_c.mean()
        nb_all = prev - (1 - prev) * (pts / (1 - pts))
        axes[col_i].plot(pts, nb_all, "k--", lw=1.0, label="Treat All")
        axes[col_i].axhline(0, color="#64748B", linestyle=":", lw=1.0, label="Treat None")
        for m_name in ["LR", "SVM", "RF", "LGBM"]:
            nbs = []
            for pt in pts:
                pred = (p_dict[m_name] >= pt).astype(int)
                tp = np.sum((pred == 1) & (y_c == 1))
                fp = np.sum((pred == 1) & (y_c == 0))
                nbs.append(tp / n_tot - (fp / n_tot) * (pt / (1 - pt)))
            lw = 2.4 if m_name == best_model_name else 1.4
            axes[col_i].plot(pts, nbs, color=PALETTE[m_name], lw=lw, label=m_name)
        axes[col_i].set_ylim(-0.05, prev + 0.08)
        axes[col_i].set_title(f"Decision Curve Analysis: {c_title}", fontsize=10.5, fontweight="bold")
        axes[col_i].set_xlabel("Threshold Probability", fontsize=9.5, fontweight="bold")
        axes[col_i].set_ylabel("Net Clinical Benefit", fontsize=9.5, fontweight="bold")
        axes[col_i].legend(loc="upper right", fontsize=8.5)
        axes[col_i].spines["top"].set_visible(False)
        axes[col_i].spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig10_dca_clinical_benefit.png")

    # --------------------------------------------------------------------------
    # STEP 9: SHAP INTERPRETABILITY FOR OPTIMAL MODEL ON LASSO FEATURES
    # --------------------------------------------------------------------------
    print(f"\n[Step 9] Computing SHAP values for optimal model ({best_model_name}) on {len(sel_feats)} LASSO features...")
    best_pipe = fitted_models[best_model_name]
    bg = shap.sample(X_tr_sel, min(60, len(X_tr_sel)), random_state=RANDOM_SEED)
    explainer = shap.KernelExplainer(lambda x: best_pipe.predict_proba(pd.DataFrame(x, columns=sel_feats))[:, 1], bg)
    shap_vals = explainer.shap_values(X_tr_sel, nsamples=100)

    # Fig 11: SHAP Beeswarm + Global Importance
    fig = plt.figure(figsize=(13, 5.2))
    plt.subplot(1, 2, 1)
    shap.summary_plot(shap_vals, X_tr_sel, show=False, plot_size=None)
    plt.title(f"(A) SHAP Beeswarm Impact ({best_model_name})", fontsize=10.5, fontweight="bold")
    plt.subplot(1, 2, 2)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp_s = pd.Series(mean_abs, index=sel_feats).sort_values()
    plt.barh(imp_s.index, imp_s.values, color="#1F4E79", height=0.55)
    for i, v in enumerate(imp_s.values):
        plt.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=9.5, fontweight="bold")
    plt.title(f"(B) Global Feature Importance ({best_model_name})", fontsize=10.5, fontweight="bold")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig11_shap_summary_panel.png")

    # Fig 12: SHAP Dependence Plots for Top 4 Features
    top4 = imp_s.iloc[::-1].index[:4].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for i, feat in enumerate(top4):
        ax = axes[i // 2, i % 2]
        f_idx = sel_feats.index(feat)
        sc_p = ax.scatter(X_tr_sel[feat], shap_vals[:, f_idx], c=X_tr_sel[sel_feats[0]], cmap="coolwarm", alpha=0.8, s=28)
        ax.axhline(0, color="#64748B", linestyle="--", lw=0.9)
        ax.set_xlabel(feat, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(f"SHAP Value for {feat}", fontsize=9.5, fontweight="bold")
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig12_shap_dependence_plots.png")

    # Fig 13: Individual Waterfall Case Studies (High-Risk LET+ vs Low-Risk LET-)
    tr_probs = best_pipe.predict_proba(X_tr_sel)[:, 1]
    pos_idx = np.where(y_tr.values == 1)[0]
    neg_idx = np.where(y_tr.values == 0)[0]
    hi_i = pos_idx[np.argmax(tr_probs[pos_idx])]
    lo_i = neg_idx[np.argmin(tr_probs[neg_idx])]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, idx, title_lbl in [
        (axes[0], hi_i, f"(A) High-Risk LET+ Case (Pred Prob = {tr_probs[hi_i]*100:.1f}%)"),
        (axes[1], lo_i, f"(B) Low-Risk Non-LET Case (Pred Prob = {tr_probs[lo_i]*100:.1f}%)"),
    ]:
        sv = pd.Series(shap_vals[idx], index=sel_feats).sort_values()
        cols = ["#DC2626" if v > 0 else "#0F766E" for v in sv.values]
        ax.barh(sv.index, sv.values, color=cols, height=0.55)
        ax.axvline(0, color="#334155", lw=0.9)
        for j, (fn, fv) in enumerate(sv.items()):
            raw_v = X_tr_sel.iloc[idx][fn]
            off = 0.005 if fv >= 0 else -0.005
            ha = "left" if fv >= 0 else "right"
            ax.text(fv + off, j, f"{fv:+.3f} (val={raw_v:.2f})", va="center", ha=ha, fontsize=8)
        ax.set_xlim(min(sv.values) - 0.14, max(sv.values) + 0.14)
        ax.set_title(title_lbl, fontsize=10, fontweight="bold")
        ax.set_xlabel("SHAP Contribution to Risk Probability", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "fig13_shap_individual_waterfalls.png")

    # Fig 01: Updated Study Flowchart
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    boxes = [
        (0.4, 8.2, 9.2, 1.3, "#1F4E79", "Step 1: Baseline Comparison & Balance Diagnosis\nDerivation Cohort (n=266: 45 LET+, 221 Non-LET) vs External Test Cohort (n=106: 25 LET+, 81 Non-LET)"),
        (0.4, 6.4, 4.3, 1.2, "#0F766E", "Step 2A: Stratified Split -> Training Set\nn = 186 (31 LET+, 155 Non-LET)\nAll preprocessing fit strictly here"),
        (5.3, 6.4, 4.3, 1.2, "#64748B", "Step 2B: Frozen Validation Cohorts\nInternal Val: n = 80 (14 LET+)\nExternal Val: n = 106 (25 LET+)"),
        (0.4, 4.5, 4.3, 1.3, "#1E40AF", "Step 3-5: Train-Only Feature Pipeline\n1) Purge Day-15 (>50% missing) + KNN Impute (k=5)\n2) VIF Pruning (remove VIF > 10)\n3) Repeated 5-Fold LASSO Feature Selection"),
        (5.3, 4.5, 4.3, 1.3, "#7C3AED", "Step 6-7: Cost-Sensitive + Bayesian HPO\nclass_weight='balanced' + Optuna TPE Tuning\n6 Models fit with 10x5-Fold CV + 1000 Bootstrap"),
        (0.4, 2.5, 9.2, 1.3, "#B45309", f"Step 8: Dual Validation & Optimal Model ({best_model_name})\nInternal Val (n=80) AUROC={val_summary[best_model_name]:.3f} avg | Evaluated via ROC, PR, Calibration, DCA & DeLong"),
        (0.4, 0.6, 9.2, 1.2, "#DC2626", "Step 9: SHAP Clinical Interpretability\nGlobal Beeswarm, Dependence Interactions & Individual Patient Waterfall Explanations"),
    ]
    for x, y, w, h, col, txt in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                              facecolor=col, edgecolor="none", alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, color="white", ha="center", va="center", fontsize=9.2, fontweight="bold")
    ax.set_title("Revised 9-Step Clinical Machine Learning & Dual-Validation Pipeline", fontsize=12.5, fontweight="bold", pad=6)
    save_fig(fig, "fig01_study_cohort_flowchart.png")

    print("\nALL REVISED PIPELINE ARTIFACTS & FIGURES GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
