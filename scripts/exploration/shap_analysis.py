# -*- coding: utf-8 -*-
"""
SHAP Explainability Analysis for ECMO Thrombosis Prediction Model
Author: Antigravity Team
Date: 2026-07-14
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Set Lancet/publication typography globally
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Configure logging
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "False").lower() in ["true", "1", "yes"]
base_dir = r"D:\acode\ECMO\output\data_process_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_process"
OUTPUT_DIR = base_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "04_shap_analysis.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("shap_analysis")

def run_shap_analysis(track_name, base_output_dir):
    logger.info(f"=== Starting SHAP Analysis for Track: {track_name} ===")
    
    # Load model and data
    model_path = os.path.join(base_output_dir, "model_RF.joblib")
    train_path = os.path.join(base_output_dir, "train_preprocessed.csv")
    test_path = os.path.join(base_output_dir, "test_preprocessed.csv")
    selected_feat_path = os.path.join(base_output_dir, "selected_features.json")
    
    if not all(os.path.exists(p) for p in [model_path, train_path, test_path, selected_feat_path]):
        logger.error(f"Required files not found in {base_output_dir}. Skipping track {track_name}.")
        return
        
    model = joblib.load(model_path)
    df_train = pd.read_csv(train_path, encoding="utf-8-sig")
    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    
    with open(selected_feat_path, "r", encoding="utf-8") as f:
        feat_config = json.load(f)
    selected_features = feat_config["final_features"]
    
    target_col = "是否下肢血栓"
    X_train = df_train[selected_features]
    X_test = df_test[selected_features]
    
    logger.info(f"Loaded Random Forest model. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Define prediction function for probability of thrombosis
    def predict_prob(X_df):
        # Convert to DataFrame if it is a numpy array
        if isinstance(X_df, np.ndarray):
            X_df = pd.DataFrame(X_df, columns=selected_features)
        return model.predict_proba(X_df)[:, 1]
        
    # Use KernelExplainer with a representative background dataset to speed up
    # We use shap.kmeans to summarize the background dataset
    background = shap.kmeans(X_train, 15) if len(X_train) > 50 else X_train
    
    logger.info("Computing SHAP values using KernelExplainer...")
    explainer = shap.KernelExplainer(predict_prob, background)
    shap_values = explainer.shap_values(X_test)
    
    # Check shape of shap_values
    logger.info(f"SHAP values computed. Shape: {np.shape(shap_values)}")
    
    # Generate SHAP Summary Plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None, color_bar_label="Feature Value (Raw Scale)")
    plt.title(f"SHAP Global Feature Importance - {track_name}", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    summary_plot_path = os.path.join(base_output_dir, "04_shap_summary.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP summary plot to: {summary_plot_path}")
    
    # Generate SHAP Bar Plot (Mean Absolute SHAP Value)
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, plot_size=None)
    plt.title(f"SHAP Feature Impact Magnitude - {track_name}", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=11)
    plt.tight_layout()
    bar_plot_path = os.path.join(base_output_dir, "04_shap_bar.png")
    plt.savefig(bar_plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP bar plot to: {bar_plot_path}")
    
    # Generate SHAP Dependence Plots for top features
    # Sort features by mean absolute SHAP value
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]
    
    for rank, idx in enumerate(top_indices[:3]):  # Plot top 3 features
        feature_name = selected_features[idx]
        logger.info(f"Generating SHAP dependence plot for top feature #{rank+1}: {feature_name}")
        
        plt.figure(figsize=(8, 6))
        # Find index of interaction feature (highest correlation in SHAP values)
        shap.dependence_plot(feature_name, shap_values, X_test, show=False, ax=plt.gca(), interaction_index="auto")
        plt.title(f"SHAP Dependence Plot: {feature_name} ({track_name})", fontsize=13, fontweight="bold", pad=15)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        dep_plot_path = os.path.join(base_output_dir, f"04_shap_dependence_{feature_name}.png")
        plt.savefig(dep_plot_path, dpi=300)
        plt.close()
        logger.info(f"Saved dependence plot for {feature_name} to: {dep_plot_path}")
        
    logger.info(f"SHAP Analysis Completed Successfully for Track: {track_name}")

def main():
    clear_dir = r"D:\acode\ECMO\output\data_clear_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_clear"
    process_dir = r"D:\acode\ECMO\output\data_process_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_process"
    
    run_shap_analysis("data_clear", clear_dir)
    run_shap_analysis("data_process", process_dir)

if __name__ == "__main__":
    main()
