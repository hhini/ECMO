# -*- coding: utf-8 -*-
"""
Visual Theme and Publication-Grade Plotting Utilities
Author: Antigravity Team
Date: 2026-07-14
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# Cohesive Lancet-inspired color palette
CLINICAL_COLORS = {
    "Control": "#3C5488FF",      # Classic Navy / Steel Blue
    "Thrombosis": "#E64B35FF",    # Premium Coral Red
    "Balanced": "#00A087FF",      # Teal / Mint
    "Other": "#3c763d",
    "Model_LR": "#4DBBD5FF",      # Light Blue
    "Model_SVM": "#3C5488FF",     # Deep Navy
    "Model_RF": "#E64B35FF",      # Coral Red
    "Model_DT": "#F39B7FFF",      # Muted Peach
    "Model_NB": "#8491B4FF"       # Muted Blue-Grey
}

MODEL_PALETTE = {
    "LR": "#4DBBD5FF",
    "SVM": "#3C5488FF",
    "RF": "#E64B35FF",
    "DT": "#F39B7FFF",
    "NB": "#8491B4FF"
}

def set_publication_style():
    """Sets Matplotlib global parameters for premium, scientific journal publications."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # Text sizes
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9.5
    plt.rcParams["ytick.labelsize"] = 9.5
    plt.rcParams["legend.fontsize"] = 9.5
    
    # Line styles
    plt.rcParams["grid.color"] = "#EBEBEB"
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.8
    
    # Spine clean up
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

def clean_spines(ax):
    """Removes top and right spines from an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

# 1. Missingness Ratio Plot
def plot_missingness_ratio(missing_series, output_path, title_suffix=""):
    set_publication_style()
    plt.figure(figsize=(10, 5.5))
    
    # Palette gradient from dark to light
    n_bars = len(missing_series)
    colors = sns.color_palette("flare_r", n_colors=n_bars)
    
    ax = sns.barplot(x=missing_series.index, y=missing_series.values, palette=colors, hue=missing_series.index, legend=False)
    
    # Add values on top of bars
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val*100:.1f}%", 
                        (p.get_x() + p.get_width() / 2., val), 
                        ha='center', va='center', 
                        xytext=(0, 6), 
                        textcoords='offset points', 
                        fontsize=8.5, color='#444444', fontweight='semibold')
            
    plt.xticks(rotation=45, ha='right', fontsize=9.5)
    plt.ylabel("Missingness Ratio", fontsize=11, fontweight='semibold')
    plt.xlabel("Clinical Features", fontsize=11, fontweight='semibold')
    plt.title(f"Missingness Profile of Key Features {title_suffix}", fontsize=13, fontweight="bold", pad=15)
    plt.ylim(0, max(missing_series.values) * 1.15 if len(missing_series) > 0 else 1.0)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 2. Class Distribution Plot
def plot_class_distribution(y_raw, y_balanced, y_test, output_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    
    classes = ["No Thrombosis", "Thrombosis"]
    x = np.arange(len(classes))
    width = 0.25
    
    original = [sum(y_raw == 0), sum(y_raw == 1)]
    balanced = [sum(y_balanced == 0), sum(y_balanced == 1)]
    test = [sum(y_test == 0), sum(y_test == 1)]
    
    rects1 = ax.bar(x - width, original, width, label=f"Train (Raw, N={sum(original)})", color="#C0C0C0", edgecolor="#888888", linewidth=0.5)
    rects2 = ax.bar(x, balanced, width, label=f"Train (SMOTE+ENN, N={sum(balanced)})", color=CLINICAL_COLORS["Balanced"], edgecolor="none")
    rects3 = ax.bar(x + width, test, width, label=f"Test (Raw, N={sum(test)})", color=CLINICAL_COLORS["Control"], edgecolor="none")
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='semibold')
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    ax.set_ylabel('Sample Size', fontsize=11, fontweight='semibold')
    ax.set_title('Dataset Balance Optimization - Cohort Distribution', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    clean_spines(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 3. Spaghetti Coagulation Trajectory Plot
def plot_coagulation_trajectory(df_traj_wide, feat_d0, feat_d2, label_name, output_path, title_name):
    """
    Plots a combined Boxplot + Individual Spaghetti Plot.
    Shows individual trajectories connected by thin lines, colored by thrombosis status.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    
    df_plot = df_traj_wide[[feat_d0, feat_d2, label_name]].copy()
    df_plot.columns = ["D0", "D2", "Group"]
    df_plot["ID"] = range(len(df_plot))
    
    # Coagulation markers can be noisy, let's keep it visible
    # Map group to color
    color_map = {"LET": CLINICAL_COLORS["Thrombosis"], "No LET": CLINICAL_COLORS["Control"]}
    
    # Draw spaghetti lines first
    for _, row in df_plot.iterrows():
        color = color_map[row["Group"]]
        alpha = 0.15 if row["Group"] == "No LET" else 0.35
        lw = 0.7 if row["Group"] == "No LET" else 1.2
        ax.plot(["Baseline (D0)", "Post-ECMO (D2)"], [row["D0"], row["D2"]], 
                color=color, alpha=alpha, linewidth=lw, zorder=1)
        
    # Melt for boxplot/violin overlay
    df_long = pd.melt(df_plot, id_vars=["ID", "Group"], value_vars=["D0", "D2"], var_name="Timepoint", value_name="Val")
    df_long["Timepoint"] = df_long["Timepoint"].map({"D0": "Baseline (D0)", "D2": "Post-ECMO (D2)"})
    
    # Overlay a beautiful boxplot
    sns.boxplot(data=df_long, x="Timepoint", y="Val", hue="Group", 
                palette=[CLINICAL_COLORS["Control"], CLINICAL_COLORS["Thrombosis"]],
                width=0.45, ax=ax, zorder=10, showfliers=False,
                boxprops=dict(alpha=0.85, edgecolor='black', linewidth=0.8),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                medianprops=dict(color='black', linewidth=1.2))
    
    ax.set_title(title_name, fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel(title_name.split(" Dynamics")[0], fontsize=11, fontweight='semibold')
    ax.set_xlabel("")
    
    # Title name cleans
    ax.legend(frameon=True, facecolor="white", edgecolor="none", title="Outcome")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    clean_spines(ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 4. Imputation Density Comparison
def plot_mice_imputation(observed, imputed, label_text, output_path):
    set_publication_style()
    plt.figure(figsize=(7.5, 5))
    
    sns.kdeplot(data=observed, label="Observed Distribution", 
                color=CLINICAL_COLORS["Control"], fill=True, alpha=0.25, linewidth=2)
    sns.kdeplot(data=imputed, label="MICE Imputed Distribution", 
                color=CLINICAL_COLORS["Thrombosis"], linestyle="--", linewidth=2.2)
    
    plt.title(f"MICE Multi-Imputation Quality - {label_text}", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel(label_text, fontsize=11, fontweight='semibold')
    plt.ylabel("Probability Density", fontsize=11, fontweight='semibold')
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 5. Boruta Feature Selection Boxplot with Shadow Features
def plot_boruta_importance(df_plot, output_path, feat_history=None, shadow_max_history=None, shadow_mean_history=None, shadow_min_history=None, green_feats=None, blue_feats=None):
    set_publication_style()
    
    if feat_history is not None and shadow_max_history is not None:
        # Detailed Boruta boxplot with shadow features
        medians = {col: np.median(vals) for col, vals in feat_history.items()}
        sorted_feats = sorted(medians.keys(), key=lambda k: medians[k], reverse=True)
        top_feats = sorted_feats[:25]
        
        rows = []
        # Add original features
        for feat in top_feats:
            status = "Confirmed" if feat in green_feats else ("Tentative" if feat in blue_feats else "Rejected")
            for val in feat_history[feat]:
                rows.append({"Feature": feat, "Importance": val, "Type": status})
                
        # Add shadow variables
        for val in shadow_max_history:
            rows.append({"Feature": "Shadow_Max", "Importance": val, "Type": "Shadow_Max"})
        for val in shadow_mean_history:
            rows.append({"Feature": "Shadow_Mean", "Importance": val, "Type": "Shadow_Mean"})
        for val in shadow_min_history:
            rows.append({"Feature": "Shadow_Min", "Importance": val, "Type": "Shadow_Min"})
            
        df_long = pd.DataFrame(rows)
        order = top_feats + ["Shadow_Max", "Shadow_Mean", "Shadow_Min"]
        
        plt.figure(figsize=(11, 7.5))
        colors = {
            "Confirmed": CLINICAL_COLORS["Balanced"],
            "Tentative": "#E69F00",
            "Rejected": "#999999",
            "Shadow_Max": CLINICAL_COLORS["Thrombosis"], # Distinctive Red
            "Shadow_Mean": CLINICAL_COLORS["Control"],   # Navy
            "Shadow_Min": "#8491B4FF"                    # Soft Grey-Blue
        }
        
        ax = sns.boxplot(
            data=df_long,
            y="Feature",
            x="Importance",
            hue="Type",
            palette=colors,
            order=order,
            orient="h",
            dodge=False,
            width=0.7,
            showfliers=False,
            boxprops=dict(alpha=0.85, linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            medianprops=dict(linewidth=1.2, color="black")
        )
        
        # Draw horizontal separating line for shadow variables
        ax.axhline(len(top_feats) - 0.5, color="#CCCCCC", linestyle="--", linewidth=0.8)
        
        plt.title("Multi-dimensional Feature Selection: Boruta Shadow Features Comparison (Top 25)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Variable Importance (Gini Importance across Shadow Iterations)", fontsize=11, fontweight='semibold')
        plt.ylabel("Clinical Predictors & Shadow References", fontsize=11, fontweight='semibold')
        
        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), frameon=True, facecolor="white", edgecolor="none", title="Variable Type")
        
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        clean_spines(ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Fallback to barplot if history is not provided
        plt.figure(figsize=(10.5, 6.5))
        colors = {"Confirmed": CLINICAL_COLORS["Balanced"], "Tentative": "#E69F00", "Rejected": "#999999"}
        ax = sns.barplot(data=df_plot.head(25), y="Feature", x="Importance", hue="Type", palette=colors, dodge=False)
        plt.title("Multi-dimensional Feature Selection: Boruta Core Ranking (Top 25)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Median Variable Importance Value (Z-Score)", fontsize=11, fontweight='semibold')
        plt.ylabel("Clinical Predictor Variable", fontsize=11, fontweight='semibold')
        plt.legend(frameon=True, facecolor="white", edgecolor="none", title="Boruta Status")
        plt.grid(axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# 6. LASSO Parameter Tuning Curves
def plot_lasso_cv(Cs_vals, mean_scores, std_scores, best_c, se_c, output_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(Cs_vals, mean_scores, color=CLINICAL_COLORS["Thrombosis"], lw=2, label="Mean CV AUC")
    ax.fill_between(Cs_vals, mean_scores - std_scores, mean_scores + std_scores, color=CLINICAL_COLORS["Thrombosis"], alpha=0.15)
    
    ax.axvline(best_c, color=CLINICAL_COLORS["Balanced"], linestyle="--", lw=1.5, label=f"Best C ({best_c:.4f})")
    ax.axvline(se_c, color=CLINICAL_COLORS["Control"], linestyle=":", lw=1.8, label=f"1-SE C ({se_c:.4f})")
    
    ax.set_xlabel("Inverse of Regularization Parameter (C)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Mean CV ROC AUC", fontsize=11, fontweight='semibold')
    ax.set_title("L1 Logistic Regression (LASSO) Parameter Optimization", fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    clean_spines(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 7. LASSO Coefficients Path
def plot_lasso_coef(Cs_path, coefs_path, best_c, output_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(Cs_path, coefs_path, alpha=0.7, lw=1.2)
    ax.axvline(best_c, color=CLINICAL_COLORS["Balanced"], linestyle="--", lw=1.5, label="Optimal C")
    
    ax.set_xlabel("Inverse of Regularization Parameter (C)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Coefficient Value", fontsize=11, fontweight='semibold')
    ax.set_title("L1 Regularized (LASSO) Coefficient Convergence Path", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    clean_spines(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 8. ROC Curve with Bootstrapped 95% Confidence Interval for Best Model
def plot_roc_curves_with_ci(best_models, X_test, y_test, colors, best_model_name, output_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    
    # 1. Plot all curves
    for name, model in best_models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = auc(fpr, tpr)
        
        lw = 2.5 if name == best_model_name else 1.5
        alpha = 1.0 if name == best_model_name else 0.7
        ax.plot(fpr, tpr, color=colors.get(name, "#777777"), lw=lw, alpha=alpha, label=f"{name} (AUC = {auc_val:.3f})")
        
        # Calculate bootstrapped CI for the best model
        if name == best_model_name:
            rng = np.random.default_rng(42)
            bootstrapped_tprs = []
            base_fpr = np.linspace(0, 1, 100)
            
            for _ in range(200):
                # Bootstrap test set
                indices = rng.choice(len(y_test), len(y_test), replace=True)
                if len(np.unique(y_test.values[indices])) < 2:
                    continue
                boot_prob = prob[indices]
                boot_y = y_test.values[indices]
                
                b_fpr, b_tpr, _ = roc_curve(boot_y, boot_prob)
                b_tpr_interp = np.interp(base_fpr, b_fpr, b_tpr)
                b_tpr_interp[0] = 0.0
                bootstrapped_tprs.append(b_tpr_interp)
                
            if len(bootstrapped_tprs) > 0:
                tprs_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
                tprs_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
                ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=colors.get(name), alpha=0.15, 
                                label=f"{best_model_name} 95% Confidence Interval")
                
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1.2, linestyle="--")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='semibold')
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight='semibold')
    ax.set_title("Receiver Operating Characteristic (ROC) Curves - Validation Set", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    clean_spines(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 9. Calibration Curve with Marginal Histograms
def plot_calibration_with_margin(best_models, X_test, y_test, colors, best_model_name, output_path):
    set_publication_style()
    
    # Set up layout: Main plot on top, marginal density at the bottom
    fig = plt.figure(figsize=(8.5, 9.5))
    grid = plt.GridSpec(4, 1, hspace=0.12)
    
    ax_cal = fig.add_subplot(grid[:3, 0])
    ax_dens = fig.add_subplot(grid[3, 0])
    
    # 1. Calibration curves on main axis
    for name, model in best_models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob, n_bins=5)
        
        lw = 2.5 if name == best_model_name else 1.5
        ax_cal.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors.get(name, "#777777"), 
                    lw=lw, label=f"{name}")
        
    ax_cal.plot([0, 1], [0, 1], color="#95a5a6", lw=1.2, linestyle="--")
    ax_cal.set_xlim([-0.01, 1.01])
    ax_cal.set_ylim([-0.01, 1.01])
    ax_cal.set_ylabel("Fraction of Positives (Observed)", fontsize=11, fontweight='semibold')
    ax_cal.set_title("Clinical Probability Calibration curves - Validation Set", fontsize=13, fontweight="bold", pad=15)
    ax_cal.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax_cal.grid(True, linestyle=":", alpha=0.5)
    clean_spines(ax_cal)
    
    # Hide x-labels on top plot
    ax_cal.set_xticklabels([])
    
    # 2. Predicted probability distributions on bottom axis for the best model
    best_prob = best_models[best_model_name].predict_proba(X_test)[:, 1]
    
    sns.kdeplot(best_prob[y_test == 0], ax=ax_dens, color=CLINICAL_COLORS["Control"], 
                fill=True, alpha=0.3, label="No Thrombosis (Control)")
    sns.kdeplot(best_prob[y_test == 1], ax=ax_dens, color=CLINICAL_COLORS["Thrombosis"], 
                fill=True, alpha=0.3, label="Thrombosis (Case)")
    
    ax_dens.set_xlabel("Mean Predicted Probability of Thrombosis", fontsize=11, fontweight='semibold')
    ax_dens.set_ylabel("Density", fontsize=10)
    ax_dens.set_xlim([-0.01, 1.01])
    ax_dens.legend(frameon=True, facecolor="white", edgecolor="none")
    ax_dens.grid(True, linestyle=":", alpha=0.5)
    clean_spines(ax_dens)
    
    # Adjust layout
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 10. DCA with Highlighted Benefit Window
def plot_dca_curves_with_highlight(best_models, X_test, y_test, colors, best_model_name, output_path):
    set_publication_style()
    fig, ax = plt.subplots(figsize=(8, 7.5))
    
    thresholds = np.arange(0.01, 0.99, 0.01)
    n_samples = len(y_test)
    total_pos = sum(y_test == 1)
    
    # 1. All Strategy
    net_benefit_all = []
    for pt in thresholds:
        nb = (total_pos / n_samples) - ((n_samples - total_pos) / n_samples) * (pt / (1 - pt))
        net_benefit_all.append(nb)
    ax.plot(thresholds, net_benefit_all, color="#95a5a6", linestyle=":", lw=1.8, label="Treat All Patients")
    
    # 2. None Strategy
    ax.plot(thresholds, [0] * len(thresholds), color="black", linestyle="-", lw=1.2, label="Treat None (Baseline)")
    
    # 3. Model Strategies
    best_nb = None
    for name, model in best_models.items():
        prob = model.predict_proba(X_test)[:, 1]
        net_benefit = []
        for pt in thresholds:
            tp = sum((prob >= pt) & (y_test == 1))
            fp = sum((prob >= pt) & (y_test == 0))
            nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
            net_benefit.append(nb)
            
        lw = 2.5 if name == best_model_name else 1.5
        ax.plot(thresholds, net_benefit, color=colors.get(name, "#777777"), lw=lw, label=f"Model: {name}")
        
        if name == best_model_name:
            best_nb = np.array(net_benefit)
            
    # Highlight Clinical Decision range for Best Model where it beats BOTH All and None
    if best_nb is not None:
        nb_all_arr = np.array(net_benefit_all)
        # Find threshold indices where best model net benefit is positive AND greater than treat all
        eligible = (best_nb > 0) & (best_nb > nb_all_arr)
        if sum(eligible) > 0:
            start_t = thresholds[np.where(eligible)[0][0]]
            end_t = thresholds[np.where(eligible)[0][-1]]
            ax.axvspan(start_t, end_t, color=colors.get(best_model_name), alpha=0.08, 
                       label=f"Clinical Benefit Range ({start_t:.2f} - {end_t:.2f})")
            
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.15, max(max(net_benefit_all), max(best_nb)) * 1.15 if best_nb is not None else 0.5])
    ax.set_xlabel("Thrombosis Decision Threshold Probability", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Clinical Net Benefit", fontsize=11, fontweight='semibold')
    ax.set_title("Decision Curve Analysis (DCA) - Clinical Utility Profile", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    clean_spines(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 11. PCA & t-SNE with Marginal KDE Distributions (Joint plots)
def plot_dimensionality_reduction_joint(X_proj, y, title_text, output_path, is_tsne=False):
    set_publication_style()
    
    # Format data for plotting
    df_proj = pd.DataFrame(X_proj, columns=["Dim 1", "Dim 2"])
    df_proj["Thrombosis"] = y.map({0: "No LET (Control)", 1: "LET (Case)"}).values
    
    # Construct JointGrid for custom styling
    g = sns.JointGrid(data=df_proj, x="Dim 1", y="Dim 2", hue="Thrombosis",
                      palette=[CLINICAL_COLORS["Control"], CLINICAL_COLORS["Thrombosis"]])
    
    # Plot scatter in the center
    g.plot_joint(sns.scatterplot, s=55, alpha=0.75, edgecolor='w', linewidth=0.6)
    
    # Plot filled KDE on margins
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3, common_norm=False, lw=1.2)
    
    # Styling and adjustments
    g.figure.set_size_inches(8, 8)
    g.ax_joint.set_xlabel(f"{'t-SNE' if is_tsne else 'PCA'} Component 1", fontsize=11, fontweight='semibold')
    g.ax_joint.set_ylabel(f"{'t-SNE' if is_tsne else 'PCA'} Component 2", fontsize=11, fontweight='semibold')
    g.ax_joint.grid(True, linestyle=":", alpha=0.5)
    clean_spines(g.ax_joint)
    
    # Place legend inside joint plot
    g.ax_joint.legend(title="Thrombosis Status", frameon=True, facecolor="white", edgecolor="none", loc="best")
    
    plt.suptitle(title_text, fontsize=13, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
