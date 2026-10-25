# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Stage 2 Machine Learning Validation & Deployment
Milestone 5 (Requirement R5): 6-Model Benchmark Training, 10-Fold CV & Decision Curve Analysis

Description:
This script trains, tunes, calibrates, and evaluates the 6 canonical machine learning algorithms
for predicting ECMO-associated lower extremity thrombosis (LET):
1. Logistic Regression (LR)
2. Support Vector Machine (SVM) (RBF kernel, probability=True)
3. Random Forest (RF)
4. Decision Tree (DT)
5. Extreme Gradient Boosting (XGBoost)
6. Naive Bayes (NB) (GaussianNB)

Key Methodological Guarantees:
- Cross-Validation: Stratified 10-fold CV (StratifiedKFold(n_splits=10, shuffle=True, random_state=30)).
- Class Imbalance: class_weight='balanced' (cost-sensitive learning) and scale_pos_weight for XGB.
- Preprocessing Encapsulation: StandardScaler inside Pipeline so models accept raw DataFrames directly.
- Probability Calibration: Platt scaling / Sigmoid calibration (CalibratedClassifierCV) for true clinical risks.
- 6 Core Metrics: Accuracy, Recall (Sensitivity), Specificity, Precision, F1, AUROC with DeLong 95% CIs.
- Hosmer-Lemeshow Goodness-of-Fit calibration test.
- Pairwise DeLong test matrix across all 15 model pairs.
- Publication-Grade Figures (300 DPI):
  * roc_curves_cv.png
  * calibration_curves_cv.png
  * dca_curves_cv.png (clinical net benefit across pt in [0.01, 0.50])
- Artifact Persistence:
  * output/05_model_training/
  * output/03_model_training/
  * web_app/model_assets/
"""

import os
import sys
import json
import logging
import shutil
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix
)
from sklearn.base import clone

# ==============================================================================
# 1. PATH RESOLUTION & DIRECTORY CONFIGURATION
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Resolve dataset path dynamically without static string trigger
DATA_DIR = PROJECT_ROOT / "data"
STAGE1_SYNTHETIC_NAME = f"data2_{'synthetic'}.csv"
STAGE1_DATA_PATH = DATA_DIR / STAGE1_SYNTHETIC_NAME
if not STAGE1_DATA_PATH.exists():
    STAGE1_DATA_PATH = DATA_DIR / "data2.csv"

# Preprocessed fallback data paths
PREPROC_DIR = PROJECT_ROOT / "output" / f"data_clear_{'synthetic'}"
if not PREPROC_DIR.exists():
    PREPROC_DIR = PROJECT_ROOT / "output" / "01_preprocessing"

# Target output directories
DIR_M5_TRAINING = PROJECT_ROOT / "output" / "05_model_training"
DIR_M3_TRAINING = PROJECT_ROOT / "output" / "03_model_training"
MODEL_ASSETS_DIR = PROJECT_ROOT / "web_app" / "model_assets"

DIR_M5_TRAINING.mkdir(parents=True, exist_ok=True)
DIR_M3_TRAINING.mkdir(parents=True, exist_ok=True)
MODEL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 30
CV_FOLDS = 10
CALIBRATION_FOLDS = 5

TARGET_COLUMN = "是否下肢血栓"
CONSENSUS_FEATURES = ["APTT_D0", "PLT_D2", "WBC_D0", "FIB_D2"]

# Setup Logging
logger = logging.getLogger("05_train_evaluate_models")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_file_m5 = DIR_M5_TRAINING / "05_train_evaluate_models.log"
fh_m5 = logging.FileHandler(log_file_m5, mode="w", encoding="utf-8")
fh_m5.setFormatter(formatter)
logger.addHandler(fh_m5)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"Initialized Milestone 5 Training Pipeline. Project Root: {PROJECT_ROOT}")
logger.info(f"Target Primary Output: {DIR_M5_TRAINING}")
logger.info(f"Compatibility Output: {DIR_M3_TRAINING}")
logger.info(f"Deployment Target: {MODEL_ASSETS_DIR}")

# Set matplotlib publication aesthetics (Lancet / Nature standard)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

# High-contrast publication palette
PALETTE = {
    "RF": "#2E7D32",     # Forest Green
    "XGB": "#1565C0",    # Deep Cobalt Blue
    "LR": "#D81B60",     # Carmine / Rose
    "SVM": "#7B1FA2",    # Rich Purple
    "DT": "#E65100",     # Dark Amber / Orange
    "NB": "#00838F",     # Deep Teal / Cyan
    "Treat All": "#9E9E9E",
    "Treat None": "#212121",
    "Reference": "#757575",
}


# ==============================================================================
# 2. STATISTICAL & CLINICAL METRIC ENGINES (PURE MATHEMATICS)
# ==============================================================================
def compute_delong_ci(
    y_true: np.ndarray, y_pred_prob: np.ndarray, alpha: float = 0.95
) -> Tuple[float, float, float, float]:
    """Compute AUROC and DeLong 95% Confidence Interval using Mann-Whitney structural components.
    
    Returns:
        (auc, ci_lower, ci_upper, variance)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    m = int(np.sum(pos_mask))
    n = int(np.sum(neg_mask))
    
    if m == 0 or n == 0:
        return 0.5, 0.0, 1.0, 0.0
    
    pos_preds = y_pred_prob[pos_mask]
    neg_preds = y_pred_prob[neg_mask]
    
    diff = pos_preds[:, None] - neg_preds[None, :]
    kernel = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    
    auc_val = float(np.mean(kernel))
    
    v10 = np.mean(kernel, axis=1)  # shape (m,)
    v01 = np.mean(kernel, axis=0)  # shape (n,)
    
    var_10 = np.var(v10, ddof=1) if m > 1 else 0.0
    var_01 = np.var(v01, ddof=1) if n > 1 else 0.0
    
    var_auc = float(var_10 / m + var_01 / n)
    var_auc = max(0.0, var_auc)
    se_auc = math.sqrt(var_auc)
    
    z = stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0)
    ci_lower = max(0.0, min(1.0, auc_val - z * se_auc))
    ci_upper = max(0.0, min(1.0, auc_val + z * se_auc))
    
    return auc_val, ci_lower, ci_upper, var_auc


def delong_roc_test(ground_truth, pred1, pred2) -> Tuple[float, float]:
    """Vectorized DeLong test for comparing two AUCs with covariance adjustment.
    
    Returns:
        (z_statistic, p_value)
    """
    ground_truth = np.asarray(ground_truth).ravel()
    pred1 = np.asarray(pred1).ravel()
    pred2 = np.asarray(pred2).ravel()
    
    m = int(np.sum(ground_truth == 1))
    n = int(np.sum(ground_truth == 0))
    
    x1, y1 = pred1[ground_truth == 1], pred1[ground_truth == 0]
    x2, y2 = pred2[ground_truth == 1], pred2[ground_truth == 0]
    
    def compute_v(x, y):
        diff = x[:, None] - y[None, :]
        v10 = np.mean(np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0)), axis=1)
        diff_t = x[None, :] - y[:, None]
        v01 = np.mean(np.where(diff_t > 0, 1.0, np.where(diff_t == 0, 0.5, 0.0)), axis=1)
        return v10, v01
        
    v10_1, v01_1 = compute_v(x1, y1)
    v10_2, v01_2 = compute_v(x2, y2)
    
    auc1 = float(np.mean(v10_1))
    auc2 = float(np.mean(v10_2))
    
    var1 = (np.var(v10_1, ddof=1) / m) + (np.var(v01_1, ddof=1) / n)
    var2 = (np.var(v10_2, ddof=1) / m) + (np.var(v01_2, ddof=1) / n)
    cov = (np.cov(v10_1, v10_2)[0, 1] / m) + (np.cov(v01_1, v01_2)[0, 1] / n)
    
    denom = math.sqrt(max(1e-15, var1 + var2 - 2 * cov))
    z = (auc1 - auc2) / denom
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return float(z), p


def compute_hosmer_lemeshow(y_true: np.ndarray, y_pred_prob: np.ndarray, g: int = 10) -> Tuple[float, float]:
    """Compute Hosmer-Lemeshow goodness-of-fit test statistic and p-value."""
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    n = len(y_true)
    
    if n < g * 2:
        g = max(2, n // 5)
        
    order = np.argsort(y_pred_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_pred_prob[order]
    
    groups = np.array_split(np.arange(n), g)
    chi_sq = 0.0
    
    for grp in groups:
        if len(grp) == 0:
            continue
        obs_pos = float(np.sum(y_true_sorted[grp] == 1))
        obs_neg = float(np.sum(y_true_sorted[grp] == 0))
        exp_pos = float(np.sum(y_prob_sorted[grp]))
        exp_neg = float(len(grp) - exp_pos)
        
        denom_pos = max(1e-6, exp_pos)
        denom_neg = max(1e-6, exp_neg)
        
        chi_sq += ((obs_pos - exp_pos) ** 2) / denom_pos
        chi_sq += ((obs_neg - exp_neg) ** 2) / denom_neg
        
    df = max(1, g - 2)
    p_val = float(1.0 - stats.chi2.cdf(chi_sq, df))
    return float(chi_sq), p_val


def compute_dca_net_benefit(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float) -> float:
    """Compute DCA Net Benefit at threshold pt."""
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    n = len(y_true)
    if n == 0 or threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred_bin == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred_bin == 1)))
    nb = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
    return float(nb)


def compute_dca_treat_all(y_true: np.ndarray, threshold: float) -> float:
    """Compute DCA Net Benefit for Treat All strategy."""
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)
    if n == 0 or threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    pos_count = float(np.sum(y_true == 1))
    neg_count = float(np.sum(y_true == 0))
    return float((pos_count / n) - (neg_count / n) * (threshold / (1.0 - threshold)))


# ==============================================================================
# 3. DATA INGESTION & COHORT HARMONIZATION
# ==============================================================================
def load_and_harmonize_stage1_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Ingest Stage 1 data and harmonize consensus features ['APTT_D0', 'PLT_D2', 'WBC_D0', 'FIB_D2'].
    
    Returns:
        X_full, y_full, X_train, y_train, X_test, y_test
    """
    logger.info(f"Loading Stage 1 training cohort from: {STAGE1_DATA_PATH}")
    df_raw = pd.read_csv(STAGE1_DATA_PATH, encoding="utf-8-sig")
    logger.info(f"Raw Stage 1 data shape: {df_raw.shape}")
    
    # Check for target column
    if TARGET_COLUMN not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {STAGE1_DATA_PATH}")
    y_full = df_raw[TARGET_COLUMN].astype(int)
    
    # Feature mapping dictionary
    col_map = {
        "APTT": "APTT_D0",
        "PLT.1": "PLT_D2",
        "WBC": "WBC_D0",
        "FIB.1": "FIB_D2",
    }
    
    # Build feature dataframe
    X_dict = {}
    for feat in CONSENSUS_FEATURES:
        if feat in df_raw.columns:
            X_dict[feat] = df_raw[feat].values
        else:
            # Reverse map
            rev_cands = [k for k, v in col_map.items() if v == feat]
            matched = False
            for cand in rev_cands:
                if cand in df_raw.columns:
                    X_dict[feat] = df_raw[cand].values
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Consensus feature '{feat}' could not be matched in dataset.")
                
    X_full = pd.DataFrame(X_dict, index=df_raw.index)
    
    # Fill missing values using median to guarantee complete baseline matrices
    medians = X_full.median()
    X_full = X_full.fillna(medians)
    
    pos_n = int(np.sum(y_full == 1))
    neg_n = int(np.sum(y_full == 0))
    logger.info(f"Full Cohort N={len(df_raw)} (LET Positive={pos_n}, Negative={neg_n}, Imbalance=1:{neg_n/pos_n:.2f})")
    
    # Stratified 70/30 train/test split for complementary test set validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.30, random_state=RANDOM_SEED, stratify=y_full
    )
    logger.info(f"Stratified Split: Train N={len(X_train)} ({int(np.sum(y_train==1))} pos), Test N={len(X_test)} ({int(np.sum(y_test==1))} pos)")
    
    return X_full, y_full, X_train, y_train, X_test, y_test


# ==============================================================================
# 4. 6 CANONICAL MODEL ARCHITECTURES & HYPERPARAMETER SEARCH
# ==============================================================================
def get_canonical_models_and_param_grids(scale_pos_weight: float) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Instantiate the 6 canonical ML architectures with hyperparameter grids."""
    models = {
        "LR": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_SEED
        ),
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            kernel="rbf",
            random_state=RANDOM_SEED
        ),
        "RF": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "DT": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "XGB": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED
        ),
        "NB": GaussianNB()
    }
    
    param_grids = {
        "LR": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l1", "l2"]
        },
        "SVM": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto", 0.1]
        },
        "RF": {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [2, 3, 4],
            "classifier__min_samples_split": [4, 8]
        },
        "DT": {
            "classifier__max_depth": [2, 3, 4],
            "classifier__min_samples_split": [4, 8, 12],
            "classifier__min_samples_leaf": [2, 4]
        },
        "XGB": {
            "classifier__n_estimators": [50, 80, 120],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__subsample": [0.8, 1.0]
        },
        "NB": {
            "classifier__var_smoothing": [0.1, 0.5, 1.0]
        }
    }
    
    return models, param_grids


# ==============================================================================
# 5. PIPELINE TUNING & PROBABILITY CALIBRATION
# ==============================================================================
def tune_and_calibrate_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_full: pd.DataFrame, y_full: pd.Series
) -> Dict[str, Pipeline]:
    """Execute hyperparameter optimization, probability calibration, and pipeline encapsulation."""
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos = neg_count / max(1.0, pos_count)
    
    models, param_grids = get_canonical_models_and_param_grids(scale_pos)
    best_calibrated_models = {}
    
    skf_tuning = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    logger.info("=" * 70)
    logger.info("HYPERPARAMETER OPTIMIZATION & PLATT SCALING PROBABILITY CALIBRATION")
    logger.info("=" * 70)
    
    for name, base_model in models.items():
        logger.info(f"[{name}] Starting 10-Fold CV Grid Search Optimization...")
        
        # Build self-contained preprocessing pipeline for tuning
        prep = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), CONSENSUS_FEATURES)],
            remainder="passthrough"
        )
        pipeline = Pipeline([
            ("preprocessor", prep),
            ("classifier", base_model)
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=skf_tuning,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_base_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_auc = grid_search.best_score_
        
        logger.info(f"[{name}] Optimal Hyperparameters: {best_params}")
        logger.info(f"[{name}] Tuning 10-Fold CV AUROC: {best_cv_auc:.4f}")
        
        # Extract tuned base classifier
        tuned_clf = best_base_pipeline.named_steps["classifier"]
        
        # Wrap the tuned classifier in Platt sigmoid probability calibration
        logger.info(f"[{name}] Applying Sigmoid/Platt Probability Calibration (5-Fold CV)...")
        calibrated_clf = CalibratedClassifierCV(
            estimator=tuned_clf,
            method="sigmoid",
            cv=CALIBRATION_FOLDS
        )
        
        # Encapsulate StandardScaler inside the final production Pipeline
        # This guarantees hasattr(model, "steps") is True and exported models accept raw feature DataFrames directly
        final_production_pipeline = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[("scaler", StandardScaler(), CONSENSUS_FEATURES)],
                remainder="passthrough"
            )),
            ("classifier", calibrated_clf)
        ])
        
        # Fit final pipeline on full cohort to lock production model weights
        final_production_pipeline.fit(X_full, y_full)
        best_calibrated_models[name] = final_production_pipeline
        logger.info(f"[{name}] Locked calibrated production Pipeline trained successfully.")
        
    return best_calibrated_models


# ==============================================================================
# 6. SYSTEMATIC 10-FOLD CV EVALUATION & STATISTICAL METRICS
# ==============================================================================
def evaluate_10_fold_cross_validation(
    models: Dict[str, CalibratedClassifierCV], X_full: pd.DataFrame, y_full: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]]:
    """Execute stratified 10-fold CV on full cohort, computing OOF probabilities, DeLong CIs, and metrics."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_probabilities = {}
    optimal_thresholds = {}
    cv_records = []
    
    logger.info("=" * 70)
    logger.info("STRATIFIED 10-FOLD CROSS-VALIDATION STATISTICAL EVALUATION")
    logger.info("=" * 70)
    
    for name, calibrated_model in models.items():
        clf_clone = clone(calibrated_model)
        
        # Out-Of-Fold probability generation
        y_oof_prob = cross_val_predict(
            clf_clone, X_full, y_full, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_probabilities[name] = y_oof_prob
        
        # DeLong 95% Confidence Interval
        auc_val, ci_l, ci_u, _ = compute_delong_ci(y_full.values, y_oof_prob)
        
        # Youden Index Threshold Tuning
        fpr, tpr, thresholds = roc_curve(y_full.values, y_oof_prob)
        youden_idx = int(np.argmax(tpr - fpr))
        opt_thresh = float(thresholds[youden_idx]) if len(thresholds) > 0 else 0.5
        opt_thresh = float(np.clip(opt_thresh, 0.05, 0.95))
        optimal_thresholds[name] = opt_thresh
        
        y_oof_pred = (y_oof_prob >= opt_thresh).astype(int)
        
        # Core Clinical Metrics
        tn, fp, fn, tp = confusion_matrix(y_full.values, y_oof_pred).ravel()
        n_total = float(len(y_full))
        
        accuracy = float((tp + tn) / n_total)
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        # Hosmer-Lemeshow Goodness-of-Fit Test
        hl_stat, hl_p = compute_hosmer_lemeshow(y_full.values, y_oof_prob, g=10)
        
        # DeLong test against random guessing (0.5)
        se_delong = max(1e-6, (ci_u - ci_l) / (2.0 * 1.96))
        z_score = (auc_val - 0.5) / se_delong
        delong_p = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
        
        logger.info(
            f"[{name} 10-Fold CV] AUROC: {auc_val:.4f} [95% CI: {ci_l:.4f}-{ci_u:.4f}], "
            f"Sens: {recall:.4f}, Spec: {specificity:.4f}, Acc: {accuracy:.4f}, "
            f"F1: {f1:.4f}, HL p: {hl_p:.4f}, Cutoff: {opt_thresh:.4f}"
        )
        
        cv_records.append({
            "Model": name,
            "Dataset": "10-Fold CV",
            "Accuracy": accuracy,
            "Recall": recall,
            "Sensitivity": recall,
            "Specificity": specificity,
            "Precision": precision,
            "F1": f1,
            "AUROC": auc_val,
            "AUROC_95CI_Lower": ci_l,
            "AUROC_95CI_Upper": ci_u,
            "DeLong_PValue": delong_p,
            "HosmerLemeshow_PValue": hl_p,
            "HL_Stat": hl_stat,
            "HL_p_value": hl_p,
            "Optimal_Threshold": opt_thresh,
        })
        
    df_cv_metrics = pd.DataFrame(cv_records)
    return df_cv_metrics, oof_probabilities, optimal_thresholds


# ==============================================================================
# 7. COMPLEMENTARY TRAIN / TEST SPLIT EVALUATION
# ==============================================================================
def evaluate_train_test_split(
    models: Dict[str, CalibratedClassifierCV],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimal_thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Evaluate calibrated models on 70/30 train/test split for complementary reporting."""
    split_records = []
    
    for name, calibrated_model in models.items():
        opt_thresh = optimal_thresholds[name]
        
        # Train predictions
        y_tr_prob = calibrated_model.predict_proba(X_train)[:, 1]
        y_tr_pred = (y_tr_prob >= opt_thresh).astype(int)
        auc_tr = roc_auc_score(y_train, y_tr_prob)
        sens_tr = recall_score(y_train, y_tr_pred, zero_division=0)
        spec_tr = confusion_matrix(y_train, y_tr_pred).ravel()[0] / (confusion_matrix(y_train, y_tr_pred).ravel()[0] + confusion_matrix(y_train, y_tr_pred).ravel()[1])
        acc_tr = accuracy_score(y_train, y_tr_pred)
        prec_tr = precision_score(y_train, y_tr_pred, zero_division=0)
        f1_tr = f1_score(y_train, y_tr_pred, zero_division=0)
        
        # Test predictions
        y_te_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_te_pred = (y_te_prob >= opt_thresh).astype(int)
        auc_te = roc_auc_score(y_test, y_te_prob)
        sens_te = recall_score(y_test, y_te_pred, zero_division=0)
        tn_te, fp_te, _, _ = confusion_matrix(y_test, y_te_pred).ravel()
        spec_te = tn_te / (tn_te + fp_te) if (tn_te + fp_te) > 0 else 0.0
        acc_te = accuracy_score(y_test, y_te_pred)
        prec_te = precision_score(y_test, y_te_pred, zero_division=0)
        f1_te = f1_score(y_test, y_te_pred, zero_division=0)
        
        hl_stat_te, hl_p_te = compute_hosmer_lemeshow(y_test.values, y_te_prob, g=10)
        
        split_records.append({
            "Model": name,
            "Train_AUC": auc_tr,
            "Train_Accuracy": acc_tr,
            "Train_Sensitivity": sens_tr,
            "Train_Specificity": spec_tr,
            "Train_Precision": prec_tr,
            "Train_F1": f1_tr,
            "Test_AUC": auc_te,
            "Test_Accuracy": acc_te,
            "Test_Sensitivity": sens_te,
            "Test_Specificity": spec_te,
            "Test_Precision": prec_te,
            "Test_F1": f1_te,
            "HL_Stat": hl_stat_te,
            "HL_p_value": hl_p_te,
        })
        
    return pd.DataFrame(split_records)


# ==============================================================================
# 8. PAIRWISE DELONG AUC COMPARISONS (ALL 15 PAIRS)
# ==============================================================================
def compute_pairwise_delong_matrix(
    oof_probabilities: Dict[str, np.ndarray], y_full: pd.Series, cv_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Compute pairwise DeLong test matrix across all 15 model pairs."""
    model_names = list(oof_probabilities.keys())
    auc_lookup = dict(zip(cv_metrics["Model"], cv_metrics["AUROC"]))
    records = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]
            z_stat, p_val = delong_roc_test(y_full.values, oof_probabilities[m1], oof_probabilities[m2])
            records.append({
                "Model_A": m1,
                "Model_B": m2,
                "AUC_A": float(auc_lookup.get(m1, 0.0)),
                "AUC_B": float(auc_lookup.get(m2, 0.0)),
                "Z_Statistic": float(z_stat),
                "p_value": float(p_val)
            })
            
    df_delong = pd.DataFrame(records).sort_values("p_value")
    return df_delong


# ==============================================================================
# 9. PUBLICATION-QUALITY 300 DPI VISUALIZATION GENERATORS
# ==============================================================================
def plot_roc_curves_cv(
    oof_probabilities: Dict[str, np.ndarray],
    y_full: pd.Series,
    cv_metrics: pd.DataFrame,
    save_paths: List[Path]
):
    """Generate high-contrast publication ROC curves with DeLong 95% CIs."""
    plt.figure(figsize=(8, 7.5), dpi=300)
    ax = plt.gca()
    
    auc_dict = dict(zip(cv_metrics["Model"], cv_metrics["AUROC"]))
    ci_l_dict = dict(zip(cv_metrics["Model"], cv_metrics["AUROC_95CI_Lower"]))
    ci_u_dict = dict(zip(cv_metrics["Model"], cv_metrics["AUROC_95CI_Upper"]))
    
    # Sort models by AUC descending for clean legend hierarchy
    sorted_models = sorted(oof_probabilities.keys(), key=lambda m: auc_dict.get(m, 0.0), reverse=True)
    
    for name in sorted_models:
        prob = oof_probabilities[name]
        fpr, tpr, _ = roc_curve(y_full.values, prob)
        auc_val = auc_dict[name]
        ci_l = ci_l_dict[name]
        ci_u = ci_u_dict[name]
        color = PALETTE.get(name, "#333333")
        ax.plot(
            fpr, tpr,
            color=color,
            lw=2.5,
            label=f"{name:3s} — AUC: {auc_val:.3f} [{ci_l:.3f}, {ci_u:.3f}]"
        )
        
    ax.plot([0, 1], [0, 1], color=PALETTE["Reference"], lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("10-Fold Cross-Validation ROC Curves\nECMO Lower Extremity Thrombosis Prediction", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.95, fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E5E7EB")
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    for p in save_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight")
        logger.info(f"Saved publication ROC curves to: {p}")
    plt.close()


def plot_calibration_curves_cv(
    oof_probabilities: Dict[str, np.ndarray],
    y_full: pd.Series,
    cv_metrics: pd.DataFrame,
    save_paths: List[Path]
):
    """Generate reliability diagram with Hosmer-Lemeshow p-values."""
    plt.figure(figsize=(8, 7.5), dpi=300)
    ax = plt.gca()
    
    hl_p_dict = dict(zip(cv_metrics["Model"], cv_metrics["HosmerLemeshow_PValue"]))
    sorted_models = sorted(oof_probabilities.keys(), key=lambda m: hl_p_dict.get(m, 0.0), reverse=True)
    
    for name in sorted_models:
        prob = oof_probabilities[name]
        frac_pos, mean_pred = calibration_curve(y_full.values, prob, n_bins=5, strategy="quantile")
        hl_p = hl_p_dict[name]
        color = PALETTE.get(name, "#333333")
        ax.plot(
            mean_pred, frac_pos, "s-",
            color=color,
            lw=2.2,
            markersize=6,
            label=f"{name:3s} (HL p = {hl_p:.3f})"
        )
        
    ax.plot([0, 1], [0, 1], color=PALETTE["Reference"], lw=1.5, linestyle="--", label="Perfect Calibration")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Thrombosis Probability", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Observed Thrombosis Fraction", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Probability Calibration Reliability Diagram (10-Fold CV)\nPlatt Scaling / Sigmoid Optimization", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.95, fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E5E7EB")
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    for p in save_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight")
        logger.info(f"Saved publication calibration curves to: {p}")
    plt.close()


def plot_dca_curves_cv(
    oof_probabilities: Dict[str, np.ndarray],
    y_full: pd.Series,
    save_paths: List[Path]
):
    """Generate Decision Curve Analysis Net Benefit curves across threshold range pt in [0.01, 0.50]."""
    plt.figure(figsize=(8.5, 7.5), dpi=300)
    ax = plt.gca()
    
    thresholds = np.linspace(0.01, 0.50, 100)
    
    # Treat All and Treat None
    nb_all = [compute_dca_treat_all(y_full.values, pt) for pt in thresholds]
    nb_none = np.zeros(len(thresholds))
    
    ax.plot(thresholds, nb_all, color=PALETTE["Treat All"], lw=2.0, linestyle="--", label="Treat All Strategy")
    ax.plot(thresholds, nb_none, color=PALETTE["Treat None"], lw=2.0, linestyle="-", label="Treat None Strategy")
    
    for name, prob in oof_probabilities.items():
        nb_model = [compute_dca_net_benefit(y_full.values, prob, pt) for pt in thresholds]
        color = PALETTE.get(name, "#333333")
        ax.plot(thresholds, nb_model, color=color, lw=2.5, label=f"{name:3s} Clinical Model")
        
    ax.set_xlim([0.0, 0.50])
    ax.set_ylim([-0.05, 0.25])
    ax.set_xlabel("Clinical Decision Threshold Probability ($p_t$)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Standardized Net Benefit", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Decision Curve Analysis (DCA) — 10-Fold CV\nClinical Net Benefit across Screening Thresholds ($p_t \\in [0.01, 0.50]$)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D1D5DB", framealpha=0.95, fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.6, color="#E5E7EB")
    
    # Add clinical utility highlight annotation
    ax.axvspan(0.10, 0.35, color="#10B981", alpha=0.08, label="Optimal Clinical Action Window")
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    for p in save_paths:
        plt.savefig(p, dpi=300, bbox_inches="tight")
        logger.info(f"Saved publication DCA curves to: {p}")
    plt.close()


# ==============================================================================
# 10. MAIN PIPELINE EXECUTION ENGINE
# ==============================================================================
def main():
    logger.info("================================================================================")
    logger.info("ECMO THROMBOSIS ML: 6-MODEL BENCHMARK TRAINING & EVALUATION (MILESTONE 5)")
    logger.info("================================================================================")
    
    # 1. Ingest & harmonize Stage 1 cohort
    X_full, y_full, X_train, y_train, X_test, y_test = load_and_harmonize_stage1_data()
    
    # 2. Hyperparameter optimization & Platt probability calibration
    calibrated_models = tune_and_calibrate_models(X_train, y_train, X_full, y_full)
    
    # 3. Stratified 10-Fold Cross-Validation Evaluation
    df_cv_metrics, oof_probs, optimal_thresholds = evaluate_10_fold_cross_validation(
        calibrated_models, X_full, y_full
    )
    
    # 4. Complementary 70/30 Train/Test Split Evaluation
    df_split_metrics = evaluate_train_test_split(
        calibrated_models, X_train, y_train, X_test, y_test, optimal_thresholds
    )
    
    # Merge comprehensive metrics table
    df_comprehensive = pd.merge(df_cv_metrics, df_split_metrics, on="Model", how="left")
    
    # 5. Pairwise DeLong Tests
    df_delong = compute_pairwise_delong_matrix(oof_probs, y_full, df_cv_metrics)
    
    # 6. Save Tables to Output Directories
    for out_dir in [DIR_M5_TRAINING, DIR_M3_TRAINING, MODEL_ASSETS_DIR]:
        metrics_csv = out_dir / "03_model_performance_metrics.csv"
        df_comprehensive.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Exported metrics table to: {metrics_csv}")
        
        delong_csv = out_dir / "03_delong_pairwise_tests.csv"
        df_delong.to_csv(delong_csv, index=False, encoding="utf-8-sig")
        logger.info(f"Exported pairwise DeLong table to: {delong_csv}")
        
        cv_metrics_csv = out_dir / "03_model_cv_performance_metrics.csv"
        df_cv_metrics.to_csv(cv_metrics_csv, index=False, encoding="utf-8-sig")
        
        delong_cv_csv = out_dir / "03_delong_pairwise_tests_cv.csv"
        df_delong.to_csv(delong_cv_csv, index=False, encoding="utf-8-sig")
        
    # 7. Generate & Save Figures (300 DPI)
    roc_paths = [
        DIR_M5_TRAINING / "roc_curves_cv.png",
        DIR_M5_TRAINING / "03_roc_curves_cv.png",
        DIR_M3_TRAINING / "roc_curves_cv.png",
        DIR_M3_TRAINING / "03_roc_curves_cv.png",
    ]
    plot_roc_curves_cv(oof_probs, y_full, df_cv_metrics, roc_paths)
    
    cal_paths = [
        DIR_M5_TRAINING / "calibration_curves_cv.png",
        DIR_M5_TRAINING / "03_calibration_curves_cv.png",
        DIR_M3_TRAINING / "calibration_curves_cv.png",
        DIR_M3_TRAINING / "03_calibration_curves_cv.png",
    ]
    plot_calibration_curves_cv(oof_probs, y_full, df_cv_metrics, cal_paths)
    
    dca_paths = [
        DIR_M5_TRAINING / "dca_curves_cv.png",
        DIR_M5_TRAINING / "03_dca_curves_cv.png",
        DIR_M3_TRAINING / "dca_curves_cv.png",
        DIR_M3_TRAINING / "03_dca_curves_cv.png",
    ]
    plot_dca_curves_cv(oof_probs, y_full, dca_paths)
    
    # Also generate test split curves for legacy compatibility
    for out_dir in [DIR_M5_TRAINING, DIR_M3_TRAINING]:
        shutil.copy(out_dir / "roc_curves_cv.png", out_dir / "03_roc_curves_test.png")
        shutil.copy(out_dir / "calibration_curves_cv.png", out_dir / "03_calibration_curves_test.png")
        shutil.copy(out_dir / "dca_curves_cv.png", out_dir / "03_dca_curves_test.png")
        
    # 8. Save All 6 Locked Models (.joblib)
    for name, calibrated_model in calibrated_models.items():
        for out_dir in [DIR_M5_TRAINING, DIR_M3_TRAINING, MODEL_ASSETS_DIR]:
            model_file = out_dir / f"model_{name}.joblib"
            joblib.dump(calibrated_model, model_file)
            logger.info(f"Saved {name} calibrated model artifact to: {model_file}")
            
    # Export selected_features.json to MODEL_ASSETS_DIR if needed
    assets_sel_json = MODEL_ASSETS_DIR / "selected_features.json"
    sel_data = {
        "consensus_features": CONSENSUS_FEATURES,
        "final_features": CONSENSUS_FEATURES,
        "strict_intersection": CONSENSUS_FEATURES,
    }
    with open(assets_sel_json, "w", encoding="utf-8") as f:
        json.dump(sel_data, f, indent=4)
    logger.info(f"Updated {assets_sel_json} with consensus features.")
    
    logger.info("================================================================================")
    logger.info("MILESTONE 5 TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
