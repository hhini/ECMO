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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
import sys

sys.path.append(os.getcwd())
from scripts.exploration.visual_utils import (
    plot_roc_curves_with_ci, plot_calibration_with_margin, plot_dca_curves_with_highlight, MODEL_PALETTE
)

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "False").lower() in ["true", "1", "yes"]
base_dir = r"D:\acode\ECMO\output\data_process_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_process"

TRAIN_DATA_PATH = os.path.join(base_dir, "train_preprocessed.csv")
TEST_DATA_PATH = os.path.join(base_dir, "test_preprocessed.csv")
SELECTED_FEATURES_PATH = os.path.join(base_dir, "selected_features.json")
PREPROCESS_CONFIG_PATH = os.path.join(base_dir, "preprocess_config.json")
OUTPUT_DIR = base_dir
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

logger.info("Starting Machine Learning Model Training & Evaluation Pipeline (data_process)...")

# ==========================================
# 2. LOAD DATA & FILTER SELECTED FEATURES
# ==========================================
with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
    feat_config = json.load(f)
selected_features = feat_config["final_features"]

with open(PREPROCESS_CONFIG_PATH, "r", encoding="utf-8") as f:
    prep_config = json.load(f)
continuous_features = prep_config.get("continuous_features", [])

logger.info(f"Loaded {len(selected_features)} final selected features: {selected_features}")

df_train = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")
df_test = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")

target_col = "是否下肢血栓"
X_train = df_train[selected_features]
y_train = df_train[target_col]
X_test = df_test[selected_features]
y_test = df_test[target_col]

X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

logger.info(f"Filtered Train X: {X_train.shape}, Test X: {X_test.shape}")

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
    logger.info(f"Tuning hyperparameters for {name} using 5-Fold CV...")
    
    from sklearn.compose import ColumnTransformer
    continuous_features_in_subset = [f for f in selected_features if f in continuous_features]
    
    if len(continuous_features_in_subset) > 0:
        preprocessor = ColumnTransformer(
            transformers=[("num", StandardScaler(), continuous_features_in_subset)],
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
    
    # Platt scaling probability calibration
    logger.info(f"Applying Platt scaling probability calibration to {name}...")
    calibrated_model = CalibratedClassifierCV(estimator=best_base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train, y_train)
    best_models[name] = calibrated_model
    cv_results[name] = grid_search.best_score_
    
    logger.info(f"{name} best params: {grid_search.best_params_}")
    logger.info(f"{name} best 5-Fold CV AUC: {grid_search.best_score_:.4f}")
    
    model_save_path = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
    joblib.dump(best_models[name], model_save_path)

# ==========================================
# 5. PERFORMANCE EVALUATION & STATISTICS
# ==========================================
performance_metrics = []
predictions_proba_test = {}
predictions_class_test = {}

def hosmer_lemeshow_test(y_true, y_prob, g=10):
    data = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    data = data.sort_values("y_prob")
    data["group"] = pd.qcut(data["y_prob"].rank(method="first"), g, labels=False)
    
    obs_events = data.groupby("group")["y_true"].sum()
    obs_nonevents = data.groupby("group")["y_true"].count() - obs_events
    exp_events = data.groupby("group")["y_prob"].sum()
    exp_nonevents = data.groupby("group")["y_prob"].count() - exp_events
    
    eps = 1e-15
    chi2_val = 0.0
    for i in range(g):
        if exp_events[i] > 0:
            chi2_val += ((obs_events[i] - exp_events[i])**2) / (exp_events[i] + eps)
        if exp_nonevents[i] > 0:
            chi2_val += ((obs_nonevents[i] - exp_nonevents[i])**2) / (exp_nonevents[i] + eps)
            
    p_val = 1.0 - scipy.stats.chi2.cdf(chi2_val, g - 2)
    return chi2_val, p_val

for name, model in best_models.items():
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_oof_prob = cross_val_predict(model, X_train, y_train, cv=cv_strat, method="predict_proba", n_jobs=-1)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_train, y_oof_prob)
    youden_idx = np.argmax(tpr - fpr)
    opt_thresh = float(thresholds[youden_idx]) if len(thresholds) > 0 else 0.5
    opt_thresh = max(0.1, min(0.9, opt_thresh))
    
    y_train_pred = (y_train_prob >= opt_thresh).astype(int)
    y_test_pred = (y_test_prob >= opt_thresh).astype(int)
    
    predictions_proba_test[name] = y_test_prob
    predictions_class_test[name] = y_test_pred
    
    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, y_train_pred).ravel()
    spec_tr = tn_tr / (tn_tr + fp_tr) if (tn_tr + fp_tr) > 0 else 0
    auc_tr = roc_auc_score(y_train, y_train_prob)
    
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(y_test, y_test_pred).ravel()
    spec_te = tn_te / (tn_te + fp_te) if (tn_te + fp_te) > 0 else 0
    auc_te = roc_auc_score(y_test, y_test_prob)
    
    hl_stat, hl_p = hosmer_lemeshow_test(y_test, y_test_prob)
    
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

# ==========================================
# 5.2. FULL COHORT OUT-OF-FOLD (OOF) CROSS-VALIDATION EVALUATION
# ==========================================
from sklearn.base import clone
oof_proba_full = {}
cv_performance_metrics = []
cv_strat_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for name, model_pipeline in best_models.items():
    clf_clone = clone(model_pipeline)
    y_oof_prob = cross_val_predict(clf_clone, X_full, y_full, cv=cv_strat_full, method="predict_proba", n_jobs=-1)[:, 1]
    oof_proba_full[name] = y_oof_prob
    
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_full, y_oof_prob)
    youden_idx_cv = np.argmax(tpr_cv - fpr_cv)
    opt_thresh_cv = float(thresholds_cv[youden_idx_cv]) if len(thresholds_cv) > 0 else 0.5
    opt_thresh_cv = max(0.1, min(0.9, opt_thresh_cv))
    
    y_oof_pred = (y_oof_prob >= opt_thresh_cv).astype(int)
    
    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_full, y_oof_pred).ravel()
    spec_cv = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else 0
    auc_cv = roc_auc_score(y_full, y_oof_prob)
    hl_stat_cv, hl_p_cv = hosmer_lemeshow_test(y_full, y_oof_prob)
    
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

# ==========================================
# 6. DELONG pairwise AUC comparison test
# ==========================================
def delong_roc_test(ground_truth, predictions_one, predictions_two):
    ground_truth = np.array(ground_truth)
    pred1 = np.array(predictions_one)
    pred2 = np.array(predictions_two)
    
    m = sum(ground_truth == 1)
    n = sum(ground_truth == 0)
    
    x1 = pred1[ground_truth == 1]
    y1 = pred1[ground_truth == 0]
    x2 = pred2[ground_truth == 1]
    y2 = pred2[ground_truth == 0]
    
    def compute_v(x, y):
        v10 = np.zeros(len(x))
        v01 = np.zeros(len(y))
        for i in range(len(x)):
            v10[i] = np.mean(y < x[i]) + 0.5 * np.mean(y == x[i])
        for j in range(len(y)):
            v01[j] = np.mean(x > y[j]) + 0.5 * np.mean(x == y[j])
        return v10, v01
        
    v10_1, v01_1 = compute_v(x1, y1)
    v10_2, v01_2 = compute_v(x2, y2)
    
    var1 = (np.var(v10_1, ddof=1) / m) + (np.var(v01_1, ddof=1) / n)
    var2 = (np.var(v10_2, ddof=1) / m) + (np.var(v01_2, ddof=1) / n)
    cov = (np.cov(v10_1, v10_2)[0, 1] / m) + (np.cov(v01_1, v01_2)[0, 1] / n)
    
    z = (np.mean(v10_1) - np.mean(v10_2)) / np.sqrt(var1 + var2 - 2 * cov + 1e-15)
    p = 2 * (1.0 - scipy.stats.norm.cdf(abs(z)))
    return z, p

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

df_delong = pd.DataFrame(delong_results)
delong_path = os.path.join(OUTPUT_DIR, "03_delong_pairwise_tests.csv")
df_delong.to_csv(delong_path, index=False, encoding="utf-8-sig")

# ==========================================
# 7. VISUALIZATIONS
# ==========================================
colors = MODEL_PALETTE

# Identify best model for highlight (ROC confidence bands, DCA shade, calibration marginal)
best_auc = -1.0
best_model_name = "LR"
for name, model in best_models.items():
    prob = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, prob)
    if auc_val > best_auc:
        best_auc = auc_val
        best_model_name = name
logger.info(f"Top performing model for visual highlighting: {best_model_name} (AUC = {best_auc:.3f})")

# 7.1 ROC Curves with 95% Confidence Interval for top model
roc_plot_path = os.path.join(OUTPUT_DIR, "03_roc_curves_test.png")
plot_roc_curves_with_ci(best_models, X_test, y_test, colors, best_model_name, roc_plot_path)

# 7.2 Calibration Curves with Marginal Distribution
cal_plot_path = os.path.join(OUTPUT_DIR, "03_calibration_curves_test.png")
plot_calibration_with_margin(best_models, X_test, y_test, colors, best_model_name, cal_plot_path)

# 7.3 DCA curves with Highlighted Clinical Net Benefit window
dca_plot_path = os.path.join(OUTPUT_DIR, "03_dca_curves_test.png")
plot_dca_curves_with_highlight(best_models, X_test, y_test, colors, best_model_name, dca_plot_path)

logger.info("Model Training (data_process) Completed Successfully.")
