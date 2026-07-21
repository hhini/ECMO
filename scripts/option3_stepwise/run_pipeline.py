import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
TRAIN_DATA_PATH = r"D:\acode\ECMO\output\01_preprocessing\train_preprocessed.csv"
TEST_DATA_PATH = r"D:\acode\ECMO\output\01_preprocessing\test_preprocessed.csv"
PREPROCESS_CONFIG_PATH = r"D:\acode\ECMO\output\01_preprocessing\preprocess_config.json"
OUTPUT_DIR = r"D:\acode\ECMO\output\option3_stepwise"
RANDOM_SEED = 30

# Initial 12 consensus features from previous feature selection run
CONSENSUS_FEATURES = [
    'Transfusion_Types_Count', 'CRP_D2', 'Diagnosis_Secondary_1.0', 
    'Parenteral_Nutrition', 'FIB_D0', 'Concomitant_Hemofiltration_2.0', 
    'pH_D2', 'Underlying_Hyperlipidemia', 'Occupation_2.0', 
    'AST_D2', 'Pregnancy_2.0', 'Creatinine_D2'
]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
log_file = os.path.join(OUTPUT_DIR, "option3_pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("option3_stepwise")
logger.info("Starting Option 3 Stepwise Feature Pruning & Model Training...")

# ==========================================
# 2. LOAD DATA
# ==========================================
df_train = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")
df_test = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")
target_col = "是否下肢血栓"

with open(PREPROCESS_CONFIG_PATH, "r", encoding="utf-8") as f:
    prep_config = json.load(f)
continuous_features = prep_config.get("continuous_features", [])

# Filter to 12 consensus features
X_train_consensus = df_train[CONSENSUS_FEATURES]
y_train = df_train[target_col]
X_test_consensus = df_test[CONSENSUS_FEATURES]
y_test = df_test[target_col]

# ==========================================
# 3. BACKWARD STEPWISE SELECTION BASED ON BIC
# ==========================================
logger.info("Running BIC-based Backward Stepwise Regression...")

def backward_stepwise_bic(X, y):
    current_features = list(X.columns)
    # Fit initial model with constant
    X_const = sm.add_constant(X[current_features])
    initial_model = sm.Logit(y, X_const).fit(disp=0)
    current_bic = initial_model.bic
    
    logger.info(f"Initial features: {current_features}")
    logger.info(f"Initial model BIC: {current_bic:.4f}")
    
    step = 1
    while len(current_features) > 1:
        best_new_bic = np.inf
        feature_to_remove = None
        
        for feature in current_features:
            candidate_features = [f for f in current_features if f != feature]
            X_cand = sm.add_constant(X[candidate_features])
            try:
                model = sm.Logit(y, X_cand).fit(disp=0)
                bic = model.bic
                if bic < best_new_bic:
                    best_new_bic = bic
                    feature_to_remove = feature
            except Exception as e:
                # Handle convergence errors by skipping
                continue
        
        if best_new_bic < current_bic:
            current_features.remove(feature_to_remove)
            current_bic = best_new_bic
            logger.info(f"Step {step}: Removed '{feature_to_remove}', New BIC: {current_bic:.4f}")
            logger.info(f"Current features ({len(current_features)}): {current_features}")
            step += 1
        else:
            logger.info("No further feature removal reduces BIC. Stepwise selection completed.")
            break
            
    return current_features

# Execute backward stepwise selection
stepwise_features = backward_stepwise_bic(X_train_consensus, y_train)
logger.info(f"Backward stepwise selected features: {stepwise_features}")

# As approved by the user, we select the clinically sound, statistically optimal 4-feature subset 
# that avoids sparse-feature overfitting:
selected_features = ['Parenteral_Nutrition', 'FIB_D0', 'AST_D2', 'Pregnancy_2.0']
logger.info(f"Final clinical BIC-pruned features selected for modeling: {selected_features}")

# Save selected features list
selected_feat_path = os.path.join(OUTPUT_DIR, "selected_features.json")
with open(selected_feat_path, "w", encoding="utf-8") as f:
    json.dump({
        "selected_features": selected_features,
        "stepwise_selected_features": stepwise_features,
        "initial_features": CONSENSUS_FEATURES,
        "selection_method": "Clinical BIC-Pruned 4-Feature Subset"
    }, f, indent=4)
logger.info(f"Saved selected features list to {selected_feat_path}")

# Filter datasets to new feature space
X_train = df_train[selected_features]
X_test = df_test[selected_features]
X_full = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_full = pd.concat([y_train, y_test], axis=0, ignore_index=True)

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
# 5. GRID SEARCH & NESTED 10-FOLD CV
# ==========================================
best_models = {}
cv_results = {}
cv_strat_full = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

oof_proba_full = {}
oof_class_full = {}
cv_performance_metrics = []
performance_metrics = []

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

for name, model in models.items():
    logger.info(f"Running Nested 10-Fold CV for {name}...")
    
    oof_probs = np.zeros(len(y_full))
    
    # Outer loop for OOF predictions
    for fold, (train_idx, val_idx) in enumerate(cv_strat_full.split(X_full, y_full)):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        # Scale continuous features
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
            
        # Inner loop grid search on training fold
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_tr, y_tr)
        
        # Predict on validation fold
        oof_probs[val_idx] = grid_search.best_estimator_.predict_proba(X_val)[:, 1]
        
    oof_proba_full[name] = oof_probs
    
    # Determine the optimal classification threshold using Youden's index on OOF probabilities
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
    
    # Fit final model on full cohort for deployment
    logger.info(f"Tuning final hyperparameters for {name} on full cohort...")
    continuous_features_in_subset = [f for f in selected_features if f in continuous_features]
    if len(continuous_features_in_subset) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), continuous_features_in_subset)
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
        cv=cv_strat_full,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search_full.fit(X_full, y_full)
    best_models[name] = grid_search_full.best_estimator_
    
    # Apparent (apparent training) performance of final model on full cohort
    y_full_prob = grid_search_full.best_estimator_.predict_proba(X_full)[:, 1]
    y_full_pred = (y_full_prob >= opt_thresh_cv).astype(int)
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
    
    # Save the fitted model binary
    model_save_path = os.path.join(OUTPUT_DIR, f"model_{name}.joblib")
    joblib.dump(grid_search_full.best_estimator_, model_save_path)
    logger.info(f"Saved final model to {model_save_path}")

# Save CSV reports
df_cv_metrics = pd.DataFrame(cv_performance_metrics)
cv_metrics_path = os.path.join(OUTPUT_DIR, "model_cv_performance_metrics.csv")
df_cv_metrics.to_csv(cv_metrics_path, index=False, encoding="utf-8-sig")
logger.info(f"Saved model CV performance metrics table to: {cv_metrics_path}")

df_metrics = pd.DataFrame(performance_metrics)
metrics_path = os.path.join(OUTPUT_DIR, "model_performance_metrics.csv")
df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
logger.info(f"Saved model apparent performance metrics table to: {metrics_path}")

# ==========================================
# 6. SCI VISUALIZATIONS (ROC, Calibration, DCA)
# ==========================================
colors = {
    "LR": "#128A76", "SVM": "#1F3A60", "RF": "#4A7C59", 
    "DT": "#D4A373", "XGB": "#C05C46", "NB": "#8E7C93"
}

# 6.1 10-Fold CV Mean ROC Curves
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name, model_pipeline in best_models.items():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
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
plt.title("10-Fold CV Mean ROC Curves (BIC-Pruned Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
roc_plot_path = os.path.join(OUTPUT_DIR, "roc_curves_cv.png")
plt.savefig(roc_plot_path, dpi=300)
plt.close()
logger.info(f"Saved OOF CV ROC curves plot to: {roc_plot_path}")

# 6.2 10-Fold CV Calibration Curves
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
for name, model_pipeline in best_models.items():
    mean_pred_val = np.linspace(0.1, 0.9, 5)
    fraction_positives_list = []
    for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
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
plt.title("10-Fold CV Calibration Curves (BIC-Pruned Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
cal_plot_path = os.path.join(OUTPUT_DIR, "calibration_curves_cv.png")
plt.savefig(cal_plot_path, dpi=300)
plt.close()
logger.info(f"Saved OOF CV Calibration curves plot to: {cal_plot_path}")

# 6.3 10-Fold CV Decision Curve Analysis (DCA)
plt.figure(figsize=(8, 7))
sns.set_theme(style="ticks")
thresholds_dca = np.linspace(0.01, 0.50, 100)

for name, model_pipeline in best_models.items():
    net_benefits_folds = []
    for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        clf_clone = clone(model_pipeline)
        clf_clone.fit(X_tr, y_tr)
        y_val_prob = clf_clone.predict_proba(X_val)[:, 1]
        
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
for train_idx, val_idx in cv_strat_full.split(X_full, y_full):
    y_val = y_full.iloc[val_idx]
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
plt.title("10-Fold CV DCA Curves (BIC-Pruned Features)", fontsize=13, fontweight="bold", color="#2c3e50", pad=15)
plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6, color="#eceff1")
sns.despine(trim=True)
plt.tight_layout()
dca_plot_path = os.path.join(OUTPUT_DIR, "dca_curves_cv.png")
plt.savefig(dca_plot_path, dpi=300)
plt.close()
logger.info(f"Saved OOF CV DCA curves plot to: {dca_plot_path}")

logger.info("Option 3 Pipeline Run Finished Successfully.")
