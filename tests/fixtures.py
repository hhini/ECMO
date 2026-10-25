# -*- coding: utf-8 -*-
"""Test Fixtures, Utilities, and Mathematical Ground Truth Oracles for ECMO E2E Tests.

Provides path definitions, data loaders, pure-mathematics reference implementations
(DeLong CI, Hosmer-Lemeshow, Decision Curve Analysis Net Benefit), and simulated
clinical patient profiles.
"""

import os
import math
import unittest
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# Path Constants
# ==============================================================================
PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
DATA_DIR = PROJECT_ROOT / "data"
TESTDATA_DIR = PROJECT_ROOT / "testdata"
OUTPUT_DIR = PROJECT_ROOT / "output"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WEB_APP_DIR = PROJECT_ROOT / "web_app"
MODEL_ASSETS_DIR = WEB_APP_DIR / "model_assets"
TESTS_DIR = PROJECT_ROOT / "tests"

PYTHON_EXE = Path(r"D:\pythontext\.venv\Scripts\python.exe")

# Milestone Output Subdirectories
DIR_M1_CLEAN = OUTPUT_DIR / "01_stage2_clean"
DIR_M2_TABLE1 = OUTPUT_DIR / "02_table1_baseline"
DIR_M3_IMBALANCE = OUTPUT_DIR / "03_imbalance_benchmark"
DIR_M4_FEATURES = OUTPUT_DIR / "04_feature_selection"
DIR_M5_TRAINING = OUTPUT_DIR / "05_model_training"
DIR_M6_EXTERNAL = OUTPUT_DIR / "06_external_validation"
DIR_M7_SHAP = OUTPUT_DIR / "07_shap_interpretability"

# Legacy or alternative output paths in existing codebase
DIR_LEGACY_PREPROCESS = OUTPUT_DIR / "01_preprocessing"
DIR_LEGACY_FEATURES = OUTPUT_DIR / "02_feature_selection"
DIR_LEGACY_TRAINING = OUTPUT_DIR / "03_model_training"
DIR_LEGACY_DATA_CLEAR = OUTPUT_DIR / "data_clear"

# ==============================================================================
# Domain Specifications & Ground Truth Values
# ==============================================================================
TARGET_COLUMN = "是否下肢血栓"
TARGET_KEYWORDS = ["下肢", "股", "大隐", "髂", "足", "小腿"]

STAGE1_CORE_N = 266
STAGE1_POS_COUNT = 36
STAGE1_NEG_COUNT = 230

STAGE2_RAW_N = 140
STAGE2_CLEAN_N = 106
STAGE2_POS_COUNT = 25
STAGE2_NEG_COUNT = 81

CANONICAL_MODELS = ["LR", "SVM", "RF", "DT", "XGB", "NB"]
CORE_METRIC_NAMES = ["Accuracy", "Recall", "Specificity", "Precision", "F1", "AUROC"]

CONSENSUS_FEATURE_COUNT_MIN = 3
CONSENSUS_FEATURE_COUNT_MAX = 5

RISK_TIER_LOW_MAX = 0.20
RISK_TIER_MODERATE_MAX = 0.50

# ==============================================================================
# Helper Functions for Progressive Testability
# ==============================================================================
def resolve_first_existing(candidates: List[Path]) -> Optional[Path]:
    """Return the first path that exists and is non-empty, or None."""
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def get_stage2_clean_path() -> Optional[Path]:
    """Find the clean Stage 2 dataset across planned and legacy locations."""
    return resolve_first_existing([
        DIR_M1_CLEAN / "stage2_clean.csv",
        DATA_DIR / "stage2_clean.csv",
        OUTPUT_DIR / "stage2_clean.csv",
    ])


def get_stage2_audit_path() -> Optional[Path]:
    """Find the Stage 2 exclusion audit log."""
    return resolve_first_existing([
        DIR_M1_CLEAN / "stage2_exclusion_audit.csv",
        DIR_M1_CLEAN / "stage2_exclusion_audit.md",
        OUTPUT_DIR / "stage2_exclusion_audit.csv",
        OUTPUT_DIR / "stage2_exclusion_audit.md",
    ])


def get_selected_features_path() -> Optional[Path]:
    """Find the feature selection JSON file."""
    return resolve_first_existing([
        DIR_M4_FEATURES / "selected_features.json",
        MODEL_ASSETS_DIR / "selected_features.json",
        DIR_LEGACY_FEATURES / "selected_features.json",
        DIR_LEGACY_DATA_CLEAR / "selected_features.json",
    ])


# ==============================================================================
# Pure Mathematical Ground Truth Reference Implementations
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
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)
    
    if m == 0 or n == 0:
        return 0.5, 0.0, 1.0, 0.0
    
    pos_preds = y_pred_prob[pos_mask]
    neg_preds = y_pred_prob[neg_mask]
    
    # Structural components
    # V10[i] = (1/n) * sum_j I(pos_preds[i] > neg_preds[j]) + 0.5 * I(pos_preds[i] == neg_preds[j])
    # V01[j] = (1/m) * sum_i I(neg_preds[j] < pos_preds[i]) + 0.5 * I(neg_preds[j] == pos_preds[i])
    diff = pos_preds[:, None] - neg_preds[None, :]
    kernel = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    
    auc = float(np.mean(kernel))
    
    v10 = np.mean(kernel, axis=1)  # shape (m,)
    v01 = np.mean(kernel, axis=0)  # shape (n,)
    
    # Variances
    var_10 = np.var(v10, ddof=1) if m > 1 else 0.0
    var_01 = np.var(v01, ddof=1) if n > 1 else 0.0
    
    var_auc = float(var_10 / m + var_01 / n)
    # Guarantee mathematical non-negativity
    var_auc = max(0.0, var_auc)
    se_auc = math.sqrt(var_auc)
    
    # 95% Normal critical value
    z = stats.norm.ppf(1.0 - (1.0 - alpha) / 2.0)
    ci_lower = max(0.0, min(1.0, auc - z * se_auc))
    ci_upper = max(0.0, min(1.0, auc + z * se_auc))
    
    return auc, ci_lower, ci_upper, var_auc


def compute_dca_net_benefit(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float
) -> float:
    """Compute Decision Curve Analysis (DCA) Net Benefit at a given threshold.
    
    Formula:
        NB = (TP / N) - (FP / N) * [p_t / (1 - p_t)]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    n = len(y_true)
    
    if n == 0 or threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    
    y_pred_binary = (y_pred_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
    return float(net_benefit)


def compute_dca_treat_all_net_benefit(y_true: np.ndarray, threshold: float) -> float:
    """Compute Net Benefit for the 'Treat All' strategy.
    
    Formula:
        NB_all = (pos_count / N) - (neg_count / N) * [p_t / (1 - p_t)]
    """
    y_true = np.asarray(y_true).ravel()
    n = len(y_true)
    if n == 0 or threshold <= 0.0 or threshold >= 1.0:
        return 0.0
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    return float((pos_count / n) - (neg_count / n) * (threshold / (1.0 - threshold)))


def compute_hosmer_lemeshow(
    y_true: np.ndarray, y_pred_prob: np.ndarray, g: int = 10
) -> Tuple[float, float]:
    """Compute Hosmer-Lemeshow goodness-of-fit test statistic and p-value.
    
    Returns:
        (chi_square_stat, p_value)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    n = len(y_true)
    
    if n < g * 2:
        g = max(2, n // 5)
    
    # Sort by predicted probability
    order = np.argsort(y_pred_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_pred_prob[order]
    
    # Split into g groups
    groups = np.array_split(np.arange(n), g)
    chi_sq = 0.0
    
    for grp in groups:
        if len(grp) == 0:
            continue
        obs_pos = np.sum(y_true_sorted[grp] == 1)
        obs_neg = np.sum(y_true_sorted[grp] == 0)
        exp_pos = np.sum(y_prob_sorted[grp])
        exp_neg = len(grp) - exp_pos
        
        # Avoid zero division
        denom_pos = max(1e-6, exp_pos)
        denom_neg = max(1e-6, exp_neg)
        
        chi_sq += ((obs_pos - exp_pos) ** 2) / denom_pos
        chi_sq += ((obs_neg - exp_neg) ** 2) / denom_neg
    
    df = max(1, g - 2)
    p_val = float(1.0 - stats.chi2.cdf(chi_sq, df))
    return float(chi_sq), p_val


def compute_clinical_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate the 6 core clinical evaluation metrics with safeguards against division by zero."""
    y_true = np.asarray(y_true).ravel()
    y_pred_prob = np.asarray(y_pred_prob).ravel()
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    n = len(y_true)
    
    accuracy = (tp + tn) / n if n > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    auc, _, _, _ = compute_delong_ci(y_true, y_pred_prob)
    
    return {
        "Accuracy": float(accuracy),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1": float(f1),
        "AUROC": float(auc),
    }


# ==============================================================================
# Simulated Patient Profiles for Clinical Acceptance Testing
# ==============================================================================
CLINICAL_SIMULATION_PROFILES = {
    "high_risk_elderly_prolonged_coagulopathy": {
        "description": "78-year-old on ECMO for 280h with severe thrombocytopenia, low FIB, and high D-dimer",
        "values": {
            "Age": 78.0,
            "ECMO时长": 280.0,
            "APTT_D0": 65.0,
            "PLT_D2": 25.0,
            "WBC_D0": 18.5,
            "FIB_D2": 4.6,
            "Creatinine_D2": 210.0,
            "AST_D2": 145.0,
            "pH_D2": 7.22,
            "Underlying_Hyperlipidemia": 1.0,
            "D-Dimer_D2": 28.5,
            "DD_D0": 15.6,
            "DD_Late": 25.1,
            "ALT_D7": 24.5,
        },
        "expected_risk_tier": "High",
        "expected_min_prob": 0.45,
    },
    "low_risk_young_short_normal_labs": {
        "description": "24-year-old on ECMO for 32h with normal coagulation and hematology",
        "values": {
            "Age": 24.0,
            "ECMO时长": 32.0,
            "APTT_D0": 32.0,
            "PLT_D2": 240.0,
            "WBC_D0": 6.8,
            "FIB_D2": 2.4,
            "Creatinine_D2": 65.0,
            "AST_D2": 22.0,
            "pH_D2": 7.41,
            "Underlying_Hyperlipidemia": 0.0,
            "D-Dimer_D2": 1.1,
            "DD_D0": 3.0,
            "DD_Late": 4.2,
            "ALT_D7": 48.0,
        },
        "expected_risk_tier": "Low",
        "expected_max_prob": 0.25,
    },
    "moderate_risk_middle_aged": {
        "description": "55-year-old on ECMO for 96h with borderline coagulation parameters",
        "values": {
            "Age": 55.0,
            "ECMO时长": 96.0,
            "APTT_D0": 42.0,
            "PLT_D2": 110.0,
            "WBC_D0": 11.0,
            "FIB_D2": 3.1,
            "Creatinine_D2": 115.0,
            "AST_D2": 45.0,
            "pH_D2": 7.36,
            "Underlying_Hyperlipidemia": 0.0,
            "D-Dimer_D2": 5.2,
            "DD_D0": 5.4,
            "DD_Late": 8.6,
            "ALT_D7": 32.0,
        },
        "expected_risk_tier": "Moderate",
        "expected_min_prob": 0.10,
        "expected_max_prob": 0.60,
    },
}
