import os
import json
import logging
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from sklearn.base import clone
import joblib
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Define directories
BASE_DIR = r"D:\acode\ECMO"
DATA_DIR = os.path.join(BASE_DIR, "output", "01_preprocessing")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "option3_stability")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logger
logger = logging.getLogger("stability_pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "03_stability_pipeline.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

RANDOM_SEED = 30
np.random.seed(RANDOM_SEED)

# Load datasets
train_path = os.path.join(DATA_DIR, "train_preprocessed.csv")
test_path = os.path.join(DATA_DIR, "test_preprocessed.csv")

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Preprocessed datasets not found. Please run preprocessing first.")

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
target_col = "是否下肢血栓"

logger.info(f"Loaded train set: {df_train.shape[0]} rows, test set: {df_test.shape[0]} rows.")

# Combine train and test for the Nested 10-Fold CV evaluation (entire cohort)
df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
X_full = df_full.drop(columns=[target_col])
y_full = df_full[target_col]

# Define continuous features list (needed for standard scaling)
continuous_features = [
    "Age", "BMI", "Cannulation_Size", "Vasoactive_Drugs_Count", "Anticoagulant_Types_Count",
    "Anticoagulant_Duration", "Invasive_Lines_Count", "Transfusion_Types_Count",
    "Ventilation_Duration", "ICU_Duration", "ECMO_Duration",
    "DD_D0", "FIB_D0", "TT_D0", "APTT_D0", "PT_D0", "INR_D0", "AT3_D0", "PLT_D0", "WBC_D0", "RBC_D0", "Hb_D0", "PCT_D0", "CRP_D0", "ALT_D0", "AST_D0", "Albumin_D0", "Creatinine_D0", "pH_D0", "PaO2_D0", "PaCO2_D0",
    "DD_D2", "FIB_D2", "TT_D2", "APTT_D2", "PT_D2", "INR_D2", "AT3_D2", "PLT_D2", "WBC_D2", "RBC_D2", "Hb_D2", "PCT_D2", "CRP_D2", "ALT_D2", "AST_D2", "Albumin_D2", "Creatinine_D2", "pH_D2", "PaO2_D2", "PaCO2_D2",
    "DD_D7", "FIB_D7", "TT_D7", "APTT_D7", "PT_D7", "INR_D7", "AT3_D7", "PLT_D7", "WBC_D7", "RBC_D7", "Hb_D7", "PCT_D7", "CRP_D7", "ALT_D7", "AST_D7", "Albumin_D7", "Creatinine_D7", "pH_D7", "PaO2_D7", "PaCO2_D7",
    "DD_Wean", "FIB_Wean", "TT_Wean", "APTT_Wean", "PT_Wean", "INR_Wean", "AT3_Wean", "PLT_Wean", "WBC_Wean", "RBC_Wean", "Hb_Wean", "PCT_Wean", "CRP_Wean", "ALT_Wean", "AST_Wean", "Albumin_Wean", "Creatinine_Wean", "pH_Wean", "PaO2_Wean", "PaCO2_Wean",
    "DD_Delta", "DD_max", "DD_slope", "DD_abnormality_ratio",
    "PLT_Delta", "PLT_min", "PLT_slope", "PLT_abnormality_ratio",
    "FIB_Delta", "FIB_min", "FIB_slope", "FIB_abnormality_ratio",
    "AT3_Delta", "AT3_min", "AT3_slope", "AT3_abnormality_ratio",
    "APTT_Delta", "APTT_max", "APTT_slope", "APTT_abnormality_ratio",
    "PT_Delta", "PT_max", "PT_slope", "PT_abnormality_ratio",
    "INR_Delta", "INR_max", "INR_slope", "INR_abnormality_ratio",
    "TT_Delta", "TT_max", "TT_slope", "TT_abnormality_ratio",
    "Max_DD", "Min_PLT", "Min_AT3", "Mean_APTT", "PLT_Pct_Change", "DD_Ratio", "Hb_Delta"
]

# Identify columns to drop (administrative, calendar, demographic noise and late timepoints)
cols = list(X_full.columns)
late_cols = [c for c in cols if "_D7" in c or "_Wean" in c or "Max_DD" in c or "Min_PLT" in c or "Min_AT3" in c or "Mean_APTT" in c]
admin_cols = [
    "Work_Status", "Smoking", "Drinking",
    "Residence_1.0", "Residence_2.0", "Residence_3.0",
    "Occupation_1.0", "Occupation_2.0", "Occupation_3.0"
]
post_cols = ["ICU_Duration", "ECMO_Duration", "Ventilation_Duration", "Anticoagulant_Duration"]

drop_cols = list(set(late_cols + admin_cols + post_cols))
candidate_features = [c for c in cols if c not in drop_cols]
logger.info(f"Initial candidate pool for stability selection: {len(candidate_features)} features.")

# Define clinical backbone features (forced in)
CLINICAL_BACKBONE = ["FIB_D0", "AST_D2"]
# Exclude clinical backbone from stability candidate selection pool
candidate_features = [c for c in candidate_features if c not in CLINICAL_BACKBONE]
logger.info(f"Forced Clinical Backbone features: {CLINICAL_BACKBONE}")
logger.info(f"Stability candidate features to screen: {len(candidate_features)}")

def drop_collinear_features(df_train_fold, candidate_cols, threshold=0.80):
    """
    Remove highly collinear features inside the fold to prevent L1 parameter sharing.
    """
    corr_matrix = df_train_fold[candidate_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for col in upper.columns:
        high_corr_features = upper.index[upper[col] > threshold].tolist()
        if high_corr_features:
            to_drop.append(col)
            
    reduced_candidates = [c for c in candidate_cols if c not in to_drop]
    return reduced_candidates

def stability_selection(X_train_fold, y_train_fold, candidate_cols, forced_cols, n_bootstrap=100, selection_threshold=0.40, random_state=30):
    """
    Perform Bootstrap Stability Selection with L1 Logistic Regression inside the CV fold by averaging over a C-grid.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X_train_fold)
    
    # 1. Drop collinear features first
    reduced_candidates = drop_collinear_features(X_train_fold, candidate_cols, threshold=0.80)
    
    cols_to_use = list(reduced_candidates) + list(forced_cols)
    
    # 2. Scale features on this fold training data
    X_scaled = X_train_fold[cols_to_use].copy()
    continuous_in_fold = [c for c in cols_to_use if c in continuous_features]
    if len(continuous_in_fold) > 0:
        scaler = StandardScaler()
        X_scaled[continuous_in_fold] = scaler.fit_transform(X_scaled[continuous_in_fold])
        
    # 3. Define C-grid
    C_grid = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    # 4. Bootstrap runs
    selected_counts = {col: 0 for col in reduced_candidates}
    total_runs = n_bootstrap * len(C_grid)
    
    for boot_idx in range(n_bootstrap):
        boot_indices = rng.choice(n_samples, size=int(0.8 * n_samples), replace=True)
        X_boot = X_scaled.iloc[boot_indices]
        y_boot = y_train_fold.iloc[boot_indices]
        
        for C_val in C_grid:
            lr = LogisticRegression(
                penalty="l1",
                C=C_val,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state + boot_idx,
                max_iter=5000
            )
            try:
                lr.fit(X_boot, y_boot)
                coefs = lr.coef_[0]
                for col_idx, col_name in enumerate(cols_to_use):
                    if col_name in reduced_candidates and coefs[col_idx] != 0.0:
                        selected_counts[col_name] += 1
            except Exception:
                continue
            
    selection_probs = {col: count / total_runs for col, count in selected_counts.items()}
    selected_candidates = [col for col, prob in selection_probs.items() if prob >= selection_threshold]
    
    sorted_probs = sorted(selection_probs.items(), key=lambda x: x[1], reverse=True)
    return selected_candidates, sorted_probs

# ==========================================
# 4. DEFINE MODELS & HYPERPARAMETER GRIDS
# ==========================================
models = {
    "LR": LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED),
    "RF": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED),
    "DT": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
    "XGB": XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", scale_pos_weight=(len(y_full) - sum(y_full)) / sum(y_full)),
    "NB": GaussianNB()
}

param_grids = {
    "LR": {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"]
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "classifier__kernel": ["rbf"]
    },
    "RF": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 8, None],
        "classifier__min_samples_split": [2, 5, 10]
    },
    "DT": {
        "classifier__max_depth": [3, 5, 8, 12, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__criterion": ["gini", "entropy"]
    },
    "XGB": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.8, 1.0]
    },
    "NB": {
        "classifier__var_smoothing": np.logspace(-9, -1, 9)
    }
}

# ==========================================
# 5. NESTED Stratified 10-Fold CV WITH INNER SEARCH & NESTED SELECTION
# ==========================================
cv_strat_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

oof_proba_full = {}
oof_class_full = {}
cv_performance_metrics = []
performance_metrics = []

# To keep track of features selected in each fold (for reporting)
fold_selected_features = []

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

logger.info("Starting Nested Stratified 10-Fold CV Pipeline with Bootstrap Stability Selection...")

# Store out-of-fold selected features for analysis
fold_features_store = {}

for fold, (train_idx, val_idx) in enumerate(cv_strat_outer.split(X_full, y_full)):
    X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
    
    logger.info(f"--- FOLD {fold+1}/10 ---")
    
    # Run Stability Selection on outer loop fold training split
    selected_cands, sorted_probs = stability_selection(
        X_tr, y_tr, candidate_features, CLINICAL_BACKBONE,
        n_bootstrap=100, selection_threshold=0.40, random_state=RANDOM_SEED + fold
    )
    
    fold_selected_feats = list(CLINICAL_BACKBONE) + list(selected_cands)
    fold_features_store[fold] = fold_selected_feats
    logger.info(f"Fold {fold+1} selected features (n={len(fold_selected_feats)}): {fold_selected_feats}")
    logger.info(f"Top 5 candidates by probability: {sorted_probs[:5]}")

# Run OOF modeling using the features selected within each fold
oof_fold_results = {name: [] for name in models.keys()}

for name, model in models.items():
    logger.info(f"Training and validating {name} using fold-specific stability-selected features...")
    oof_probs = np.zeros(len(y_full))
    
    for fold, (train_idx, val_idx) in enumerate(cv_strat_outer.split(X_full, y_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        # Get selected features for this fold
        selected_feats = fold_features_store[fold]
        X_tr_sub = X_tr[selected_feats]
        X_val_sub = X_val[selected_feats]
        
        continuous_in_subset = [f for f in selected_feats if f in continuous_features]
        if len(continuous_in_subset) > 0:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), continuous_in_subset)
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
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_tr_sub, y_tr)
        
        y_val_prob = grid_search.best_estimator_.predict_proba(X_val_sub)[:, 1]
        oof_probs[val_idx] = y_val_prob
        oof_fold_results[name].append((y_val.copy(), y_val_prob))
        
    oof_proba_full[name] = oof_probs
    
    # Determine threshold via Youden's Index on OOF probabilities
    fpr_cv, tpr_cv, thresholds_cv = roc_curve(y_full, oof_probs)
    youden_idx_cv = np.argmax(tpr_cv - fpr_cv)
    opt_thresh_cv = float(thresholds_cv[youden_idx_cv]) if len(thresholds_cv) > 0 else 0.5
    opt_thresh_cv = max(0.1, min(0.9, opt_thresh_cv))
    
    y_oof_pred = (oof_probs >= opt_thresh_cv).astype(int)
    oof_class_full[name] = y_oof_pred
    
    tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(y_full, y_oof_pred).ravel()
    spec_cv = tn_cv / (tn_cv + fp_cv) if (tn_cv + fp_cv) > 0 else 0
    auc_cv = roc_auc_score(y_full, oof_probs)
    hl_stat_cv, hl_p_cv = hosmer_lemeshow_test(y_full, oof_probs)
    
    logger.info(f"[{name} 10-Fold OOF CV] AUC={auc_cv:.4f}, Accuracy={accuracy_score(y_full, y_oof_pred):.4f}, Sensitivity={recall_score(y_full, y_oof_pred):.4f}, Specificity={spec_cv:.4f}, HL p-value={hl_p_cv:.4f}")
    
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

# ==========================================
# 6. RUN STABILITY SELECTION ON FULL COHORT & TRAIN FINAL MODELS
# ==========================================
logger.info("=== RUNNING FINAL STABILITY SELECTION ON FULL COHORT ===")
final_selected_cands, final_sorted_probs = stability_selection(
    X_full, y_full, candidate_features, CLINICAL_BACKBONE,
    n_bootstrap=100, selection_threshold=0.40, random_state=RANDOM_SEED
)

final_selected_features = list(CLINICAL_BACKBONE) + list(final_selected_cands)
logger.info(f"Final Selected Features on Full Cohort (n={len(final_selected_features)}): {final_selected_features}")
logger.info(f"Full Cohort Top 15 Selection Probabilities:")
for col, prob in final_sorted_probs[:15]:
    logger.info(f"  {col}: {prob:.3f}")

# Save final feature set
feature_config = {
    "clinical_backbone": CLINICAL_BACKBONE,
    "stability_selected_candidates": final_selected_cands,
    "final_features": final_selected_features,
    "selection_probabilities": {col: prob for col, prob in final_sorted_probs}
}
feat_config_path = os.path.join(OUTPUT_DIR, "selected_features.json")
with open(feat_config_path, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=4, ensure_ascii=False)
logger.info(f"Saved final feature config to: {feat_config_path}")

best_models = {}
for name, model in models.items():
    logger.info(f"Tuning final hyperparameters for {name} on full cohort with final selected features...")
    X_full_sub = X_full[final_selected_features]
    
    continuous_in_subset = [f for f in final_selected_features if f in continuous_features]
    if len(continuous_in_subset) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), continuous_in_subset)
            ],
            remainder="passthrough"
        )
        pipeline_full = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    else:
        pipeline_full = Pipeline([
            ("classifier", model)
        ])
        
    grid_search_full = GridSearchCV(
        estimator=pipeline_full,
        param_grid=param_grids[name],
        cv=cv_strat_outer,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search_full.fit(X_full_sub, y_full)
    best_models[name] = grid_search_full.best_estimator_
    
    # Apparent performance of final model
    y_full_prob = grid_search_full.best_estimator_.predict_proba(X_full_sub)[:, 1]
    
    # Find Youden threshold for final model
    fpr_f, tpr_f, thresholds_f = roc_curve(y_full, y_full_prob)
    youden_idx_f = np.argmax(tpr_f - fpr_f)
    opt_thresh_f = float(thresholds_f[youden_idx_f]) if len(thresholds_f) > 0 else 0.5
    opt_thresh_f = max(0.1, min(0.9, opt_thresh_f))
    
    y_full_pred = (y_full_prob >= opt_thresh_f).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_full, y_full_pred).ravel()
    spec_f = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0
    auc_f = roc_auc_score(y_full, y_full_prob)
    hl_stat_f, hl_p_f = hosmer_lemeshow_test(y_full, y_full_prob)
    
    performance_metrics.append({
        "Model": name,
        "Train_AUC": auc_f,
        "Train_Accuracy": accuracy_score(y_full, y_full_pred),
        "Train_Sensitivity": recall_score(y_full, y_full_pred),
        "Train_Specificity": spec_f,
        "Test_AUC": auc_f,
        "Test_Accuracy": accuracy_score(y_full, y_full_pred),
        "Test_Sensitivity": recall_score(y_full, y_full_pred),
        "Test_Specificity": spec_f,
        "Test_Precision": precision_score(y_full, y_full_pred, zero_division=0),
        "Test_F1": f1_score(y_full, y_full_pred, zero_division=0),
        "HL_Stat": hl_stat_f,
        "HL_p_value": hl_p_f
    })
    
    # Save the fitted model
    model_save_path = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
    joblib.dump(grid_search_full.best_estimator_, model_save_path)
    logger.info(f"Saved final model to {model_save_path}")

# Save CSV reports
df_cv_metrics = pd.DataFrame(cv_performance_metrics)
cv_metrics_path = os.path.join(OUTPUT_DIR, "model_cv_performance_metrics.csv")
df_cv_metrics.to_csv(cv_metrics_path, index=False, encoding="utf-8-sig")

df_metrics = pd.DataFrame(performance_metrics)
metrics_path = os.path.join(OUTPUT_DIR, "model_performance_metrics.csv")
df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

# ==========================================
# 7. SCI VISUALIZATIONS (ROC, Calibration, DCA)
# ==========================================
colors = {
    "LR": "#128A76", "SVM": "#1F3A60", "RF": "#4A7C59", 
    "DT": "#D4A373", "XGB": "#C05C46", "NB": "#8E7C93"
}

# 7.1 10-Fold CV Mean ROC Curves
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name in best_models.keys():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold, (y_val, y_val_prob) in enumerate(oof_fold_results[name]):
        fpr_f, tpr_f, _ = roc_curve(y_val, y_val_prob)
        aucs.append(auc(fpr_f, tpr_f))
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
plt.title("10-Fold CV Mean ROC Curves (Stability Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
roc_plot_path = os.path.join(OUTPUT_DIR, "roc_curves_cv.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()

# 7.2 10-Fold CV Calibration Curves
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name in best_models.keys():
    mean_pred_val = np.linspace(0.1, 0.9, 5)
    fraction_positives_list = []
    for fold, (y_val, y_val_prob) in enumerate(oof_fold_results[name]):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_val_prob, n_bins=5)
        interp_frac = np.interp(mean_pred_val, mean_predicted_value, fraction_of_positives)
        fraction_positives_list.append(interp_frac)
    mean_fraction_positives = np.mean(fraction_positives_list, axis=0)
    plt.plot(mean_pred_val, mean_fraction_positives, "s-", color=colors[name], lw=2, markersize=5, label=f"{name}")

plt.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Perfectly Calibrated")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Fraction of Positives (Observed)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("10-Fold CV Calibration Curves (Stability Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
cal_plot_path = os.path.join(OUTPUT_DIR, "calibration_curves_cv.png")
plt.savefig(cal_plot_path, dpi=300)
plt.close()

# 7.3 10-Fold CV Decision Curve Analysis (DCA)
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
thresholds_dca = np.linspace(0.01, 0.50, 100)

for name in best_models.keys():
    net_benefits_folds = []
    for fold, (y_val, y_val_prob) in enumerate(oof_fold_results[name]):
        net_benefits_fold = []
        n_samples_f = len(y_val)
        for pt in thresholds_dca:
            tp = sum((y_val_prob >= pt) & (y_val == 1))
            fp = sum((y_val_prob >= pt) & (y_val == 0))
            nb = (tp / n_samples_f) - (fp / n_samples_f) * (pt / (1 - pt))
            net_benefits_fold.append(nb)
        net_benefits_folds.append(net_benefits_fold)
    mean_net_benefits = np.mean(net_benefits_folds, axis=0)
    plt.plot(thresholds_dca, mean_net_benefits, color=colors[name], lw=2.5, label=f"{name}")

# Plot Treat All & Treat None
net_benefit_all_folds = []
for fold, (y_val, y_val_prob) in enumerate(oof_fold_results[list(best_models.keys())[0]]):
    n_samples_f = len(y_val)
    total_pos_f = sum(y_val == 1)
    net_benefit_all_fold = []
    for pt in thresholds_dca:
        nb = (total_pos_f / n_samples_f) - ((n_samples_f - total_pos_f) / n_samples_f) * (pt / (1 - pt))
        net_benefit_all_fold.append(nb)
    net_benefit_all_folds.append(net_benefit_all_fold)
mean_net_benefit_all = np.mean(net_benefit_all_folds, axis=0)

plt.plot(thresholds_dca, mean_net_benefit_all, color="#bdc3c7", lw=1.8, linestyle="--", label="Treat All")
plt.plot(thresholds_dca, np.zeros(len(thresholds_dca)), color="#7f8c8d", lw=1.8, label="Treat None")
plt.xlim([0.0, 0.5])
plt.ylim([-0.05, 0.20])
plt.xlabel("Threshold Probability ($p_t$)", fontsize=12, fontweight="bold", color="#2c3e50")
plt.ylabel("Net Benefit", fontsize=12, fontweight="bold", color="#2c3e50")
plt.title("10-Fold CV DCA Curves (Stability Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
dca_plot_path = os.path.join(OUTPUT_DIR, "dca_curves_cv.png")
plt.savefig(dca_plot_path, dpi=300)
plt.close()

logger.info("Stability Selection Pipeline Finished Successfully.")
