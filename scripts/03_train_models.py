import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


# ==========================================
# 1. CONFIGURATION & PARAMETERS (解耦配置块)
# ==========================================
TRAIN_DATA_PATH = r"D:\acode\ECMO\output\01_preprocessing\train_preprocessed.csv"
TEST_DATA_PATH = r"D:\acode\ECMO\output\01_preprocessing\test_preprocessed.csv"
SELECTED_FEATURES_PATH = r"D:\acode\ECMO\output\02_feature_selection\selected_features.json"
OUTPUT_DIR = r"D:\acode\ECMO\output\03_model_training"
RANDOM_SEED = 30
# ==========================================

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
log_file = os.path.join(OUTPUT_DIR, "03_train_models.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_training")

logger.info("Starting Machine Learning Model Training & Evaluation Pipeline...")

# ==========================================
# 2. LOAD DATA & FILTER SELECTED FEATURES
# ==========================================
# Load selected features config
with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
    feat_config = json.load(f)
selected_features = feat_config["final_features"]

# Load preprocessing config to identify continuous columns
PREPROCESS_CONFIG_PATH = r"D:\acode\ECMO\output\01_preprocessing\preprocess_config.json"
with open(PREPROCESS_CONFIG_PATH, "r", encoding="utf-8") as f:
    prep_config = json.load(f)
continuous_features = prep_config.get("continuous_features", [])

logger.info(f"Loaded {len(selected_features)} final selected features: {selected_features}")

# Load datasets
df_train = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")
df_test = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")

target_col = "是否下肢血栓"
X_train = df_train[selected_features]
y_train = df_train[target_col]
X_test = df_test[selected_features]
y_test = df_test[target_col]

X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

logger.info(f"Filtered Train X: {X_train.shape}, Test X: {X_test.shape}, Full Cohort X: {X_full.shape}")

# Set matplotlib aesthetics for publication quality
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ==========================================
# 3. DEFINE MODELS & HYPERPARAMETER GRIDS
# ==========================================
from sklearn.pipeline import Pipeline

models = {
    "LR": LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED),
    "RF": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
    "DT": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
    "NB": GaussianNB()
}

param_grids = {
    "LR": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l1", "l2"]
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1],
        "classifier__kernel": ["rbf"]
    },
    "RF": {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [2, 3, 4],
        "classifier__min_samples_split": [4, 8, 12],
        "classifier__min_samples_leaf": [2, 4, 6]
    },
    "DT": {
        "classifier__max_depth": [2, 3, 4],
        "classifier__min_samples_split": [4, 8, 12],
        "classifier__min_samples_leaf": [2, 4, 6]
    },
    "NB": {
        "classifier__var_smoothing": np.logspace(-9, -1, 9)
    }
}

# ==========================================
# 4. GRID SEARCH HYPERPARAMETER TUNING
# ==========================================
best_models = {}
cv_results = {}

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, model in models.items():
    logger.info(f"Tuning hyperparameters for {name} using 5-Fold CV with Class-Weight Balancing...")
    
    from sklearn.compose import ColumnTransformer
    continuous_features_in_subset = [f for f in selected_features if f in continuous_features]
    
    if len(continuous_features_in_subset) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), continuous_features_in_subset)
            ],
            remainder="passthrough"
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline([
            ("classifier", model)
        ])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[name],
        cv=cv_strat,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_base_model = grid_search.best_estimator_
    
    # Wrap in CalibratedClassifierCV using Platt scaling (sigmoid method) to calibrate probabilities.
    # All models (including LR) are calibrated to correct for the probability shift introduced by class-weight balancing.
    logger.info(f"Applying Platt scaling probability calibration to {name}...")
    calibrated_model = CalibratedClassifierCV(estimator=best_base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)
    best_models[name] = calibrated_model
        
    cv_results[name] = grid_search.best_score_
    
    logger.info(f"{name} best params: {grid_search.best_params_}")
    logger.info(f"{name} best 5-Fold CV AUC: {grid_search.best_score_:.4f}")
    
    # Save the calibrated fitted model binary
    model_save_path = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
    joblib.dump(best_models[name], model_save_path)
    logger.info(f"Saved calibrated {name} model to {model_save_path}")

# ==========================================
# 5. PERFORMANCE EVALUATION & STATISTICS
# ==========================================
performance_metrics = []
predictions_proba_test = {}
predictions_class_test = {}

# Hosmer-Lemeshow goodness-of-fit test function
def hosmer_lemeshow_test(y_true, y_prob, g=10):
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    data = data.sort_values("y_prob")
    data["group"] = pd.qcut(data["y_prob"].rank(method="first"), g, labels=False)
    
    obs_events = data.groupby("group")["y_true"].sum()
    obs_nonevents = data.groupby("group")["y_true"].count() - obs_events
    
    exp_events = data.groupby("group")["y_prob"].sum()
    exp_nonevents = data.groupby("group")["y_prob"].count() - exp_events
    
    # Compute chi-square statistic
    # Add a small epsilon to denominator to prevent division by zero
    eps = 1e-15
    chi2_val = 0.0
    for i in range(g):
        if exp_events[i] > 0:
            chi2_val += ((obs_events[i] - exp_events[i])**2) / (exp_events[i] + eps)
        if exp_nonevents[i] > 0:
            chi2_val += ((obs_nonevents[i] - exp_nonevents[i])**2) / (exp_nonevents[i] + eps)
            
    p_val = 1.0 - scipy.stats.chi2.cdf(chi2_val, g - 2)
    return chi2_val, p_val

logger.info("Evaluating models on Train and Test sets...")

for name, model in best_models.items():
    # Predictions (probabilities)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Optimize classification threshold using Out-Of-Fold Youden's index to prevent threshold overfitting
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_oof_prob = cross_val_predict(model, X_train, y_train, cv=cv_strat, method="predict_proba", n_jobs=-1)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_train, y_oof_prob)
    youden_idx = np.argmax(tpr - fpr)
    opt_thresh = float(thresholds[youden_idx]) if len(thresholds) > 0 else 0.5
    # Keep threshold within reasonable bounds
    opt_thresh = max(0.1, min(0.9, opt_thresh))
    
    # Apply threshold
    y_train_pred = (y_train_prob >= opt_thresh).astype(int)
    y_test_pred = (y_test_prob >= opt_thresh).astype(int)
    
    logger.info(f"[{name}] Optimized classification threshold (OOF Youden): {opt_thresh:.4f}")
    
    predictions_proba_test[name] = y_test_prob
    predictions_class_test[name] = y_test_pred
    
    # Calculate metrics
    # Train metrics
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred).ravel()
    spec_tr = tn_tr / (tn_tr + fp_tr) if (tn_tr + fp_tr) > 0 else 0
    auc_tr = roc_auc_score(y_train, y_train_prob)
    
    # Test metrics
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()
    spec_te = tn_te / (tn_te + fp_te) if (tn_te + fp_te) > 0 else 0
    auc_te = roc_auc_score(y_test, y_test_prob)
    
    # Hosmer-Lemeshow test on Test set
    hl_stat, hl_p = hosmer_lemeshow_test(y_test, y_test_prob)
    
    # Log results
    logger.info(f"[{name} Test Set Evaluation] AUC={auc_te:.4f}, Accuracy={accuracy_score(y_test, y_test_pred):.4f}, Sensitivity={recall_score(y_test, y_test_pred):.4f}, Specificity={spec_te:.4f}, HL p-value={hl_p:.4f}")
    
    # Save to metrics list
    performance_metrics.append({
        "Model": name,
        "Train_AUC": auc_tr,
        "Train_Accuracy": accuracy_score(y_train, y_train_pred),
        "Train_Sensitivity": recall_score(y_train, y_train_pred),
        "Train_Specificity": spec_tr,
        "Train_Precision": precision_score(y_train, y_train_pred, zero_division=0),
        "Train_F1": f1_score(y_train, y_train_pred, zero_division=0),
        
        "Test_AUC": auc_te,
        "Test_Accuracy": accuracy_score(y_test, y_test_pred),
        "Test_Sensitivity": recall_score(y_test, y_test_pred),
        "Test_Specificity": spec_te,
        "Test_Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Test_F1": f1_score(y_test, y_test_pred, zero_division=0),
        
        "HL_Stat": hl_stat,
        "HL_p_value": hl_p
    })

df_metrics = pd.DataFrame(performance_metrics)
metrics_path = os.path.join(OUTPUT_DIR, "03_model_performance_metrics.csv")
df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
logger.info(f"Saved model performance metrics table to: {metrics_path}")

# ==========================================
# 5.2. FULL COHORT OUT-OF-FOLD (OOF) CROSS-VALIDATION EVALUATION
# ==========================================
logger.info("Running 5-Fold Cross-Validation on the full cohort (n=266)...")
X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof_proba_full = {}
oof_class_full = {}
cv_performance_metrics = []

for name, model_pipeline in best_models.items():
    # Clone the best estimator (pipeline) to avoid data leakage
    clf_clone = clone(model_pipeline)
    
    # Generate out-of-fold predicted probabilities on the full cohort
    y_oof_prob = cross_val_predict(clf_clone, X_full, y_full, cv=cv_strat_full, method="predict_proba", n_jobs=-1)[:, 1]
    oof_proba_full[name] = y_oof_prob
    
    # Calculate Youden index threshold on these OOF predictions
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_full, y_oof_prob)
    youden_idx_cv = np.argmax(tpr_cv - fpr_cv)
    opt_thresh_cv = float(thresholds_cv[youden_idx_cv]) if len(thresholds_cv) > 0 else 0.5
    opt_thresh_cv = max(0.1, min(0.9, opt_thresh_cv))
    
    y_oof_pred = (y_oof_prob >= opt_thresh_cv).astype(int)
    oof_class_full[name] = y_oof_pred
    
    # Compute metrics
    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_full, y_oof_pred).ravel()
    spec_cv = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else 0
    auc_cv = roc_auc_score(y_full, y_oof_prob)
    hl_stat_cv, hl_p_cv = hosmer_lemeshow_test(y_full, y_oof_prob)
    
    logger.info(f"[{name} 5-Fold OOF CV Evaluation] AUC={auc_cv:.4f}, Accuracy={accuracy_score(y_full, y_oof_pred):.4f}, Sensitivity={recall_score(y_full, y_oof_pred):.4f}, Specificity={spec_cv:.4f}, HL p-value={hl_p_cv:.4f}")
    
    cv_performance_metrics.append({
        "Model": name,
        "CV_AUC": auc_cv,
        "CV_Accuracy": accuracy_score(y_full, y_oof_pred),
        "CV_Sensitivity": recall_score(y_full, y_oof_pred),
        "CV_Specificity": spec_cv,
        "CV_Precision": precision_score(y_full, y_oof_pred, zero_division=0),
        "CV_F1": f1_score(y_full, y_oof_pred, zero_division=0),
        "HL_Stat": hl_stat_cv,
        "HL_p_value": hl_p_cv
    })

df_cv_metrics = pd.DataFrame(cv_performance_metrics)
cv_metrics_path = os.path.join(OUTPUT_DIR, "03_model_cv_performance_metrics.csv")
df_cv_metrics.to_csv(cv_metrics_path, index=False, encoding="utf-8-sig")
logger.info(f"Saved model CV performance metrics table to: {cv_metrics_path}")

# ==========================================
# 6. DELONG pairwise AUC comparison test
# ==========================================
# DeLong's method for comparing AUCs of two models
# Vectorized implementation for fast computing
def delong_roc_test(ground_truth, predictions_one, predictions_two):
    # Vectorized compute for AUC variance and covariance
    # Ground truth must be 0/1
    ground_truth = np.array(ground_truth)
    pred1 = np.array(predictions_one)
    pred2 = np.array(predictions_two)
    
    m = sum(ground_truth == 1)
    n = sum(ground_truth == 0)
    
    x1 = pred1[ground_truth == 1]
    y1 = pred1[ground_truth == 0]
    x2 = pred2[ground_truth == 1]
    y2 = pred2[ground_truth == 0]
    
    # Structuring components
    def compute_v(x, y):
        # Structural component V10 for positive cases, V01 for negative cases
        v10 = np.zeros(len(x))
        v01 = np.zeros(len(y))
        for i in range(len(x)):
            v10[i] = np.mean(y < x[i]) + 0.5 * np.mean(y == x[i])
        for j in range(len(y)):
            v01[j] = np.mean(x > y[j]) + 0.5 * np.mean(x == y[j])
        return v10, v01
        
    v10_1, v01_1 = compute_v(x1, y1)
    v10_2, v01_2 = compute_v(x2, y2)
    
    auc1 = np.mean(v10_1)
    auc2 = np.mean(v10_2)
    
    # Variances and Covariances
    var1 = (np.var(v10_1, ddof=1) / m) + (np.var(v01_1, ddof=1) / n)
    var2 = (np.var(v10_2, ddof=1) / m) + (np.var(v01_2, ddof=1) / n)
    
    cov = (np.cov(v10_1, v10_2)[0, 1] / m) + (np.cov(v01_1, v01_2)[0, 1] / n)
    
    # Z-statistic
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2 * cov + 1e-15)
    p = 2 * (1.0 - scipy.stats.norm.cdf(abs(z)))
    return z, p

logger.info("Running pairwise DeLong AUC comparison tests on Test Set...")
delong_results = []
model_names = list(best_models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1 = model_names[i]
        m2 = model_names[j]
        z, p = delong_roc_test(y_test, predictions_proba_test[m1], predictions_proba_test[m2])
        delong_results.append({
            "Model_A": m1,
            "Model_B": m2,
            "AUC_A": performance_metrics[i]["Test_AUC"],
            "AUC_B": performance_metrics[j]["Test_AUC"],
            "Z_Statistic": z,
            "p_value": p
        })

df_delong = pd.DataFrame(delong_results).sort_values("p_value")
delong_path = os.path.join(OUTPUT_DIR, "03_delong_pairwise_tests.csv")
df_delong.to_csv(delong_path, index=False, encoding="utf-8-sig")
logger.info(f"Pairwise DeLong test results saved to: {delong_path}")

# ==========================================
# 7. SCI-GRADE VISUALIZATIONS (ROC, Calibration, DCA)
# ==========================================
colors = {
    "LR": "#128A76",   # Premium Muted Teal
    "SVM": "#1F3A60",  # Navy Blue
    "RF": "#4A7C59",   # Sage/Forest Green
    "DT": "#D4A373",   # Warm Gold
    "NB": "#8E7C93"    # Muted Lavender/Purple
}

# 7.1 ROC Curves Plot (SCI standard, English)
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name, model in best_models.items():
    prob = predictions_proba_test[name]
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[name], lw=2.5, label=f"{name} (AUC = {roc_auc:.3f})")
    
plt.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Reference Line")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("Receiver Operating Characteristic (ROC) - Test Set", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
roc_plot_path = os.path.join(OUTPUT_DIR, "03_roc_curves_test.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
logger.info(f"Saved ROC curves plot to: {roc_plot_path}")

# 7.2 Calibration Curves Plot (SCI standard, English)
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name, model in best_models.items():
    prob = predictions_proba_test[name]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob, n_bins=5)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors[name], lw=2, markersize=5, label=f"{name}")
    
plt.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Perfectly Calibrated")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Fraction of Positives (Actual)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("Model Calibration Curves - Test Set", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
cal_plot_path = os.path.join(OUTPUT_DIR, "03_calibration_curves_test.png")
plt.savefig(cal_plot_path, dpi=300)
plt.close()
logger.info(f"Saved calibration curves plot to: {cal_plot_path}")

# 7.3 Decision Curve Analysis (DCA) Plot (SCI standard, English)
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
thresholds = np.arange(0.01, 1.0, 0.01)
n_samples = len(y_test)
total_pos = sum(y_test == 1)

# Plot Treat All & Treat None
net_benefit_all = []
for pt in thresholds:
    # Treat all means all predicted positive
    tp = total_pos
    fp = n_samples - total_pos
    nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
    net_benefit_all.append(nb)

plt.plot(thresholds, net_benefit_all, color="#bdc3c7", lw=1.8, linestyle="--", label="Treat All")
plt.plot(thresholds, np.zeros(len(thresholds)), color="#7f8c8d", lw=1.8, linestyle="-", label="Treat None")

# Plot each model's DCA
for name, model in best_models.items():
    prob = predictions_proba_test[name]
    net_benefit_model = []
    
    for pt in thresholds:
        # Classify as positive if prob >= pt
        y_pred_pt = (prob >= pt).astype(int)
        
        # Calculate True Positives & False Positives
        tp = sum((y_pred_pt == 1) & (y_test == 1))
        fp = sum((y_pred_pt == 1) & (y_test == 0))
        
        nb = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
        net_benefit_model.append(nb)
        
    plt.plot(thresholds, net_benefit_model, color=colors[name], lw=2.5, label=f"{name}")

plt.xlim([0.0, 1.0])
plt.ylim([-0.05, 0.35]) # Cap y-axis to focus on clinically relevant range
plt.xlabel("Threshold Probability ($p_t$)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Net Benefit", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("Decision Curve Analysis (DCA) - Test Set", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
dca_plot_path = os.path.join(OUTPUT_DIR, "03_dca_curves_test.png")
plt.savefig(dca_plot_path, dpi=300)
plt.close()
logger.info(f"Saved DCA curves plot to: {dca_plot_path}")

# ==========================================
# 7.4. CV ROC Curves Plot (Mean ± SD across folds, Full Cohort)
# ==========================================
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")

for name, model_pipeline in best_models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_strat_full.split(X_full, y_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
        
        fpr_f, tpr_f, _ = roc_curve(y_val, y_val_prob)
        roc_auc_f = auc(fpr_f, tpr_f)
        aucs.append(roc_auc_f)
        
        interp_tpr = np.interp(mean_fpr, fpr_f, tpr_f)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color=colors[name], lw=2.5, 
             label=f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

plt.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Reference Line")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("5-Fold Cross-Validation Mean ROC Curves - Full Cohort", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
roc_cv_plot_path = os.path.join(OUTPUT_DIR, "03_roc_curves_cv.png")
plt.savefig(roc_cv_plot_path, dpi=300)
plt.close()
logger.info(f"Saved CV ROC curves plot to: {roc_cv_plot_path}")

# ==========================================
# 7.5. CV Calibration Curves Plot (Mean across folds, Full Cohort)
# ==========================================
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")

for name, model_pipeline in best_models.items():
    mean_pred_val = np.linspace(0.1, 0.9, 5) # 5 bins
    fraction_positives_list = []
    
    cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_prob, n_bins=5)
        # Interpolate to common x grid
        interp_frac = np.interp(mean_pred_val, mean_predicted_value, fraction_of_positives)
        fraction_positives_list.append(interp_frac)
        
    mean_fraction_positives = np.mean(fraction_positives_list, axis=0)
    plt.plot(mean_pred_val, mean_fraction_positives, "s-", color=colors[name], lw=2, markersize=5, label=f"{name}")
    
plt.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Perfectly Calibrated")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Fraction of Positives (Observed)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("5-Fold Cross-Validation Calibration Curves", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
cal_cv_plot_path = os.path.join(OUTPUT_DIR, "03_calibration_curves_cv.png")
plt.savefig(cal_cv_plot_path, dpi=300)
plt.close()
logger.info(f"Saved CV calibration curves plot to: {cal_cv_plot_path}")

# ==========================================
# 7.6. CV DCA Curves Plot (Mean across folds, Full Cohort)
# ==========================================
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
thresholds_dca = np.linspace(0.01, 0.50, 100) # focus on early warning relevant range

# Calculate the mean net benefit for each model across folds
for name, model_pipeline in best_models.items():
    net_benefits_folds = []
    
    cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
        
        net_benefits_fold = []
        n_samples_f = len(y_val)
        total_pos_f = sum(y_val == 1)
        
        for pt in thresholds_dca:
            tp = sum((y_val_prob >= pt) & (y_val == 1))
            fp = sum((y_val_prob >= pt) & (y_val == 0))
            if pt == 1.0:
                nb = 0.0
            else:
                nb = (tp / n_samples_f) - (fp / n_samples_f) * (pt / (1 - pt))
            net_benefits_fold.append(nb)
            
        net_benefits_folds.append(net_benefits_fold)
        
    mean_net_benefits = np.mean(net_benefits_folds, axis=0)
    plt.plot(thresholds_dca, mean_net_benefits, color=colors[name], lw=2.5, label=f"{name}")

# Plot Treat All and Treat None across folds
net_benefit_all_folds = []
cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
    y_val = y_full.iloc[val_idx]
    n_samples_f = len(y_val)
    total_pos_f = sum(y_val == 1)
    
    net_benefit_all_fold = []
    for pt in thresholds_dca:
        if pt == 1.0:
            nb = 0.0
        else:
            nb = (total_pos_f / n_samples_f) - ((n_samples_f - total_pos_f) / n_samples_f) * (pt / (1 - pt))
        net_benefit_all_fold.append(nb)
    net_benefit_all_folds.append(net_benefit_all_fold)

mean_net_benefit_all = np.mean(net_benefit_all_folds, axis=0)

plt.plot(thresholds_dca, mean_net_benefit_all, color="#bdc3c7", lw=1.8, linestyle="--", label="Treat All")
plt.plot(thresholds_dca, np.zeros(len(thresholds_dca)), color="#7f8c8d", lw=1.8, linestyle="-", label="Treat None")

plt.xlim([0.0, 0.5])
plt.ylim([-0.05, 0.20])
plt.xlabel("Threshold Probability ($p_t$)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Net Benefit", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("5-Fold Cross-Validation Decision Curve Analysis (DCA)", fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
dca_cv_plot_path = os.path.join(OUTPUT_DIR, "03_dca_curves_cv.png")
plt.savefig(dca_cv_plot_path, dpi=300)
plt.close()
logger.info(f"Saved CV DCA curves plot to: {dca_cv_plot_path}")

# Run pairwise DeLong AUC comparison tests on Full Cohort OOF CV Predictions
logger.info("Running pairwise DeLong AUC comparison tests on Full Cohort OOF CV Predictions...")
delong_cv_results = []
model_names = list(best_models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1 = model_names[i]
        m2 = model_names[j]
        z, p = delong_roc_test(y_full, oof_proba_full[m1], oof_proba_full[m2])
        delong_cv_results.append({
            "Model_A": m1,
            "Model_B": m2,
            "AUC_A": cv_performance_metrics[i]["CV_AUC"],
            "AUC_B": cv_performance_metrics[j]["CV_AUC"],
            "Z_Statistic": z,
            "p_value": p
        })

df_delong_cv = pd.DataFrame(delong_cv_results).sort_values("p_value")
delong_cv_path = os.path.join(OUTPUT_DIR, "03_delong_pairwise_tests_cv.csv")
df_delong_cv.to_csv(delong_cv_path, index=False, encoding="utf-8-sig")
logger.info(f"Pairwise DeLong test results on CV saved to: {delong_cv_path}")

logger.info("ECMO Thrombosis Model Training and Evaluation Pipeline Completed Successfully.")
