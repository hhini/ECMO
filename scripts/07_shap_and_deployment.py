# -*- coding: utf-8 -*-
r"""ECMO Thrombosis Stage 2 Machine Learning Validation & Deployment
Milestone 7 (Requirement R7): Publication-Grade SHAP Interpretability & Web Assets

Description:
This script performs rigorous SHAP (SHapley Additive exPlanations) interpretability
analysis on the locked clinical models (focusing on the benchmark-leading Random Forest
and gradient boosted ensembles), and generates publication-grade visualizations:
1. Global SHAP Summary Beeswarm Plot (04_shap_summary.png, shap_beeswarm.png)
2. Global SHAP Feature Importance Bar Chart (04_shap_importance_bar.png)
3. Individual Waterfall Attribution Plot for High-Risk Patient (04_shap_waterfall_high_risk.png)
4. Individual Waterfall Attribution Plot for Low-Risk Patient (04_shap_waterfall_low_risk.png)
5. Background Reference Matrix Generation (shap_background.npy for fast sub-second bedside KernelExplainer)

Methodological Compliance:
- Zero data fabrication: all values computed from authentic clinical cohort (data2.csv).
- Multi-directory mirroring to satisfy project layout:
  * output/07_shap_interpretability/
  * output/04_shap_interpretability/
  * output/
  * report/figures/
- High-contrast Lancet / Nature publication aesthetics at 300+ DPI.
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ==============================================================================
# 1. PATH RESOLUTION & OUTPUT CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STAGE1_DATA_PATH = DATA_DIR / "data2.csv"

DIR_M7_SHAP = PROJECT_ROOT / "output" / "07_shap_interpretability"
DIR_M4_SHAP = PROJECT_ROOT / "output" / "04_shap_interpretability"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_FIG_DIR = PROJECT_ROOT / "report" / "figures"
REPORT_IMG_DIR = PROJECT_ROOT / "report" / "images"
MODEL_ASSETS_DIR = PROJECT_ROOT / "web_app" / "model_assets"

for d in [DIR_M7_SHAP, DIR_M4_SHAP, REPORT_FIG_DIR, REPORT_IMG_DIR, MODEL_ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Setup Logging
logger = logging.getLogger("07_shap_and_deployment")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_file = DIR_M7_SHAP / "07_shap_interpretability.log"
fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Set matplotlib publication aesthetics (Lancet / Nature standard)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

TARGET_COLUMN = "是否下肢血栓"
CONSENSUS_FEATURES = ["APTT_D0", "PLT_D2", "WBC_D0", "FIB_D2"]
FEATURE_DISPLAY_NAMES = {
    "APTT_D0": "Day 0 APTT (s)",
    "PLT_D2": "Day 2 Platelets (×10⁹/L)",
    "WBC_D0": "Day 0 WBC (×10⁹/L)",
    "FIB_D2": "Day 2 Fibrinogen (g/L)",
}

# ==============================================================================
# 2. DATA INGESTION & COHORT HARMONIZATION
# ==============================================================================
def load_cohort_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load authentic clinical cohort data and extract consensus features."""
    if not STAGE1_DATA_PATH.exists():
        raise FileNotFoundError(f"Cohort data file not found: {STAGE1_DATA_PATH}")
    
    logger.info(f"Loading clinical data from: {STAGE1_DATA_PATH}")
    df_raw = pd.read_csv(STAGE1_DATA_PATH, encoding="utf-8-sig")
    
    if TARGET_COLUMN not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")
    y = df_raw[TARGET_COLUMN].astype(int)
    
    col_map = {
        "APTT": "APTT_D0",
        "PLT.1": "PLT_D2",
        "WBC": "WBC_D0",
        "FIB.1": "FIB_D2",
    }
    
    X_dict = {}
    for feat in CONSENSUS_FEATURES:
        if feat in df_raw.columns:
            X_dict[feat] = df_raw[feat].values
        else:
            matched = False
            for k, v in col_map.items():
                if v == feat and k in df_raw.columns:
                    X_dict[feat] = df_raw[k].values
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Could not map feature '{feat}' from dataset columns.")
                
    X = pd.DataFrame(X_dict, index=df_raw.index)
    medians = X.median()
    X = X.fillna(medians)
    
    logger.info(f"Loaded cohort N={len(X)} with features={list(X.columns)}, LET Positives={int(np.sum(y==1))}, Negatives={int(np.sum(y==0))}")
    return X, y


# ==============================================================================
# 3. EXTRACT MODEL & SCALER
# ==============================================================================
def load_top_model() -> Tuple[Any, Any, Any]:
    """Load Random Forest model and extract underlying estimator and preprocessor."""
    rf_path = MODEL_ASSETS_DIR / "model_RF.joblib"
    if not rf_path.exists():
        rf_path = PROJECT_ROOT / "output" / "05_model_training" / "model_RF.joblib"
    if not rf_path.exists():
        raise FileNotFoundError("model_RF.joblib could not be located in model_assets or output.")
        
    logger.info(f"Loading top-performing model from: {rf_path}")
    full_pipeline = joblib.load(rf_path)
    
    preprocessor = full_pipeline.named_steps["preprocessor"]
    clf = full_pipeline.named_steps["classifier"]
    
    # Extract base tree estimator from CalibratedClassifierCV
    if hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
        base_estimator = clf.calibrated_classifiers_[0].estimator
    else:
        base_estimator = clf
        
    return full_pipeline, preprocessor, base_estimator


# ==============================================================================
# 4. SHAP EXPLAINER & PLOTTING PIPELINE
# ==============================================================================
def run_shap_analysis():
    logger.info("=" * 70)
    logger.info("STARTING PUBLICATION-GRADE SHAP INTERPRETABILITY ANALYSIS")
    logger.info("=" * 70)
    
    X, y = load_cohort_data()
    full_pipeline, preprocessor, base_estimator = load_top_model()
    
    # Predict probabilities for the entire cohort
    probs = full_pipeline.predict_proba(X)[:, 1]
    
    # Preprocess inputs for TreeExplainer
    X_scaled = preprocessor.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=CONSENSUS_FEATURES)
    
    # Initialize TreeExplainer
    logger.info(f"Initializing SHAP TreeExplainer on {base_estimator.__class__.__name__}...")
    explainer = shap.TreeExplainer(base_estimator)
    shap_explanation = explainer(X_scaled_df)
    
    # Extract positive class (thrombosis risk) explanations
    if len(shap_explanation.shape) == 3:
        exp_pos = shap_explanation[:, :, 1]
    else:
        exp_pos = shap_explanation
        
    # Replace feature names with publication-grade clinical labels
    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in CONSENSUS_FEATURES]
    exp_pos.feature_names = display_names
    
    # --------------------------------------------------------------------------
    # 4.1 Global SHAP Beeswarm Summary Plot (04_shap_summary.png)
    # --------------------------------------------------------------------------
    logger.info("Generating Global SHAP Beeswarm Summary Plot...")
    plt.figure(figsize=(9, 6), dpi=300)
    shap.plots.beeswarm(exp_pos, max_display=10, show=False, plot_size=(9, 6))
    plt.title("Global SHAP Attribution Summary (Random Forest Ensemble)", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("SHAP Value (Impact on Log-Odds of Lower Extremity Thrombosis)", fontsize=11, labelpad=8)
    plt.tight_layout()
    
    summary_plot_path = DIR_M7_SHAP / "04_shap_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved beeswarm summary plot to: {summary_plot_path}")
    
    # Mirror as shap_beeswarm.png
    shutil.copyfile(summary_plot_path, DIR_M7_SHAP / "shap_beeswarm.png")
    
    # --------------------------------------------------------------------------
    # 4.2 Global SHAP Feature Importance Bar Chart (04_shap_importance_bar.png)
    # --------------------------------------------------------------------------
    logger.info("Generating Global Feature Importance Bar Chart...")
    mean_abs_shap = np.mean(np.abs(exp_pos.values), axis=0)
    importance_df = pd.DataFrame({
        "Feature": display_names,
        "Raw_Feature": CONSENSUS_FEATURES,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=True)
    
    # Save importance metrics CSV
    metrics_csv_path = DIR_M7_SHAP / "07_shap_importance_metrics.csv"
    importance_df.sort_values(by="Mean_Abs_SHAP", ascending=False).to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved feature importance metrics to: {metrics_csv_path}")
    
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Mean_Abs_SHAP"],
        color="#2563EB",
        edgecolor="#1D4ED8",
        height=0.6,
        alpha=0.9
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=10, fontweight="bold", color="#1E293B")
        
    ax.set_title("Global Clinical Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Average Absolute Marginal Contribution)", fontsize=11, labelpad=8)
    ax.set_xlim(0, max(importance_df["Mean_Abs_SHAP"]) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    
    bar_plot_path = DIR_M7_SHAP / "04_shap_importance_bar.png"
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance bar chart to: {bar_plot_path}")
    shutil.copyfile(bar_plot_path, DIR_M7_SHAP / "shap_importance_bar.png")
    
    # --------------------------------------------------------------------------
    # 4.3 Individual Waterfall Attribution Plots (High & Low Risk)
    # --------------------------------------------------------------------------
    # Find representative high-risk (y=1 with top predicted probability)
    pos_indices = np.where(y == 1)[0]
    high_risk_idx = int(pos_indices[np.argmax(probs[pos_indices])])
    
    # Find representative low-risk (y=0 with low predicted probability)
    neg_indices = np.where(y == 0)[0]
    low_risk_idx = int(neg_indices[np.argmin(probs[neg_indices])])
    
    logger.info(f"High-Risk Patient Selected: ID={high_risk_idx}, Prob={probs[high_risk_idx]:.4f}, Actual LET={y.iloc[high_risk_idx]}")
    logger.info(f"Low-Risk Patient Selected: ID={low_risk_idx}, Prob={probs[low_risk_idx]:.4f}, Actual LET={y.iloc[low_risk_idx]}")
    
    # High-Risk Waterfall
    plt.figure(figsize=(9, 5.5), dpi=300)
    shap.plots.waterfall(exp_pos[high_risk_idx], max_display=10, show=False)
    plt.title(
        f"Individual Waterfall Attribution: High-Risk Case (Predicted Risk = {probs[high_risk_idx]*100:.1f}%, Actual LET=Yes)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.tight_layout()
    high_waterfall_path = DIR_M7_SHAP / "04_shap_waterfall_high_risk.png"
    plt.savefig(high_waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved high-risk waterfall plot to: {high_waterfall_path}")
    
    # Low-Risk Waterfall
    plt.figure(figsize=(9, 5.5), dpi=300)
    shap.plots.waterfall(exp_pos[low_risk_idx], max_display=10, show=False)
    plt.title(
        f"Individual Waterfall Attribution: Low-Risk Case (Predicted Risk = {probs[low_risk_idx]*100:.1f}%, Actual LET=No)",
        fontsize=12, fontweight="bold", pad=14
    )
    plt.tight_layout()
    low_waterfall_path = DIR_M7_SHAP / "04_shap_waterfall_low_risk.png"
    plt.savefig(low_waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved low-risk waterfall plot to: {low_waterfall_path}")
    
    # --------------------------------------------------------------------------
    # 4.4 Save Background Reference Matrix for Web App (shap_background.npy)
    # --------------------------------------------------------------------------
    logger.info("Generating representative background matrix (N=40) for web calculator KernelExplainer...")
    np.random.seed(30)
    bg_sample = shap.sample(X, 40, random_state=30)
    bg_path = MODEL_ASSETS_DIR / "shap_background.npy"
    np.save(bg_path, bg_sample.values)
    logger.info(f"Saved shap_background.npy (shape {bg_sample.shape}) to: {bg_path}")
    
    # Also save background to DIR_M7_SHAP
    np.save(DIR_M7_SHAP / "shap_background.npy", bg_sample.values)
    
    # --------------------------------------------------------------------------
    # 4.5 Mirror Artifacts to Complementary Directories
    # --------------------------------------------------------------------------
    artifacts_to_mirror = [
        "04_shap_summary.png",
        "shap_beeswarm.png",
        "04_shap_importance_bar.png",
        "shap_importance_bar.png",
        "04_shap_waterfall_high_risk.png",
        "04_shap_waterfall_low_risk.png",
        "07_shap_importance_metrics.csv",
    ]
    
    mirror_targets = [DIR_M4_SHAP, OUTPUT_DIR, REPORT_FIG_DIR, REPORT_IMG_DIR]
    for target_dir in mirror_targets:
        target_dir.mkdir(parents=True, exist_ok=True)
        for art in artifacts_to_mirror:
            src = DIR_M7_SHAP / art
            if src.exists():
                dst = target_dir / art
                shutil.copyfile(src, dst)
                
    logger.info("Successfully mirrored all SHAP figures and metrics to output/, output/04_shap_interpretability/, and report/figures/")
    logger.info("Milestone 7 SHAP Interpretability pipeline completed successfully!")


if __name__ == "__main__":
    run_shap_analysis()
