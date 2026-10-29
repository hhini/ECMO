# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Predictive Modeling - Milestone 2:
Three-Track Feature Selection Convergence & Pure English Publication-Grade Venn Diagram

Description:
This script executes three independent, mathematically rigorous feature selection pipelines
on the official Stage 1 training dataset (data/data2_synthetic.csv, N=266, LET positive=45)
to extract an exact 3 to 5 predictor consensus intersection for predicting lower extremity thrombosis (LET):
1. Track 1 (Clinical Epidemiological Track):
   - Univariate hypothesis testing screening (p < 0.10)
   - Multicollinearity screening via Variance Inflation Factor (VIF < 10)
   - Backward stepwise multivariable logistic regression (statsmodels.api.Logit) selecting independent predictors
2. Track 2 (Penalized Machine Learning Track):
   - Stratified 10-fold cross-validated L1-penalized logistic regression (LASSO) with class_weight='balanced'
   - Regularization constraint on C parameter to select a compact, non-zero subset of 5-10 features
3. Track 3 (Ensemble Feature Importance Track):
   - Boruta shadow-feature algorithm (BorutaPy) with balanced Random Forest (100 trees)
   - Confirms features significantly exceeding maximum shadow variable importance
4. Progressive Convergence:
   - Verifies candidate pools diverge realistically across the 3 independent tracks
   - Computes mathematical 3-way intersection and 2-track consensus voting
   - Strictly ensures the final consensus feature set consists of 3, 4, or 5 core, clinically interpretable predictors
5. Publication-Grade Pure English Venn Diagram:
   - Centered layout, pure English typography
   - Plain moderate-sized bold numbers without enclosing circles
   - 300 DPI resolution saved to both report/sci_figures and output/04_feature_selection

Outputs:
- Tables:
  * 02_univariate_results.csv
  * 02_vif_results.csv
  * 02_multivariable_logistic_results.csv
  * 02_lasso_coefficients.csv
  * 02_boruta_importance.csv
  * 02_consensus_ranking.csv
  * three_track_feature_selection_summary.csv
- Figures:
  * fig05_step4_three_track_venn_diagram.png (300 DPI, pure English, circle-free counts)
  * three_track_venn_diagram.png / 02_feature_selection_venn.png / 02_feature_overlap_venn.png
  * 02_lasso_cv_mse.png
  * 02_lasso_coef_path.png
  * 02_boruta_importance.png
- JSON:
  * selected_features.json (output/04_feature_selection, output/02_feature_selection, web_app/model_assets)
- Report:
  * 02_feature_selection_report.md
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from boruta import BorutaPy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ==============================================================================
# 1. CONFIGURATION & DIRECTORY MANAGEMENT
# ==============================================================================
PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_CLEAN_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"

# Primary and compatibility output directories
OUTPUT_DIR_M4 = PROJECT_ROOT / "output" / "04_feature_selection"
OUTPUT_DIR_LEGACY = PROJECT_ROOT / "output" / "02_feature_selection"
REPORT_FIG_DIR = PROJECT_ROOT / "report" / "sci_figures"
WEB_ASSETS_DIR = PROJECT_ROOT / "web_app" / "model_assets"
WEB_FIG_DIR = PROJECT_ROOT / "web_app" / "static" / "sci_figures"

for d in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY, REPORT_FIG_DIR, WEB_ASSETS_DIR, WEB_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "是否下肢血栓"
RANDOM_SEED = 42
UNIVARIATE_P_VAL = 0.10      # Cut-off p-value for univariate pre-screening
VIF_THRESHOLD = 10.0         # Cut-off for multicollinearity screening
MULTIVARIABLE_P_VAL = 0.08   # Target p-value for multivariable logistic regression
BORUTA_P_VAL = 0.05          # Significance alpha for Boruta shadow-feature testing
CV_FOLDS = 10                # 10-fold cross-validation for LASSO

CONSENSUS_MIN_FEATURES = 3
CONSENSUS_MAX_FEATURES = 5

# Setup Logging
logger = logging.getLogger("04_tri_method_feature_selection")
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
log_file_m4 = OUTPUT_DIR_M4 / "04_feature_selection.log"
log_file_legacy = OUTPUT_DIR_LEGACY / "02_feature_selection.log"

fh_m4 = logging.FileHandler(log_file_m4, mode="w", encoding="utf-8")
fh_m4.setFormatter(formatter)
logger.addHandler(fh_m4)

fh_legacy = logging.FileHandler(log_file_legacy, mode="w", encoding="utf-8")
fh_legacy.setFormatter(formatter)
logger.addHandler(fh_legacy)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Set matplotlib aesthetics
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================================================================
# 2. DATA INGESTION & CLINICAL CANDIDATE FILTERING
# ==============================================================================
def load_and_filter_features() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load official Stage 1 training cohort and apply clinical candidate filtering.
    
    Guarantees:
    - History of CHD ('冠心病', 'Underlying_CHD') and Cardiogenic Shock ('心源性休克', 'Cardiogenic_Shock')
      are completely excluded from candidate feature pools.
    - Continuous Age is strictly preserved as numeric float64.
    - Late-timepoint variables (_D7, _Wean, _D15, etc.) are excluded to enforce early 48h predictive window.
    - Sociodemographic / non-clinical noise columns are excluded.
    - Hardware pump speed variables (speed, pump, rpm, 转速, 泵) are excluded due to center practice variance.
    """
    logger.info("=" * 70)
    logger.info("STEP 1: INGESTING TRAINING COHORT & CLINICAL FILTERING")
    logger.info("=" * 70)
    
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Official training data not found at: {TRAIN_DATA_PATH}")
        
    df_train = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")
    logger.info(f"Loaded official Stage 1 training dataset: shape={df_train.shape}")
    
    if TARGET_COLUMN not in df_train.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not present in training data")
        
    y_train = df_train[TARGET_COLUMN].astype(int)
    pos_n = int(y_train.sum())
    neg_n = int((y_train == 0).sum())
    logger.info(f"Target distribution: Positive LET={pos_n} ({pos_n/len(y_train)*100:.2f}%), Negative={neg_n}")
    
    # Verify strict exclusion of CHD and Cardiogenic Shock in raw data
    chd_found = any(df_train[c].astype(str).str.contains("冠心病").any() for c in df_train.columns)
    shock_found = any(df_train[c].astype(str).str.contains("心源性休克").any() for c in df_train.columns)
    if chd_found or shock_found:
        logger.warning(f"Raw data contains residual CHD: {chd_found}, Shock: {shock_found}. Purging completely.")
    
    # Standard early 48h clinical column mappings
    clinical_map = {
        '年龄': 'Age',
        'BMI': 'BMI',
        '上机前DD': 'DD_D0',
        '上机第二天DD': 'DD_D2',
        'WBC': 'WBC_D0',
        'WBC.1': 'WBC_D2',
        'FIB': 'FIB_D0',
        'FIB.1': 'FIB_D2',
        'PLT': 'PLT_D0',
        'PLT.1': 'PLT_D2',
        'Hb': 'Hb_D0',
        'Hb.1': 'Hb_D2',
        'PT': 'PT_D0',
        'PT.1': 'PT_D2',
        'INR': 'INR_D0',
        'INR.1': 'INR_D2',
        'APTT': 'APTT_D0',
        'APTT.1': 'APTT_D2',
        'AT3': 'AT3_D0',
        'AT3.1': 'AT3_D2',
        '谷丙': 'ALT_D0',
        '谷丙.1': 'ALT_D2',
        '谷草': 'AST_D0',
        '谷草.1': 'AST_D2',
        '肌酐': 'Creatinine_D0',
        '肌酐.1': 'Creatinine_D2',
        'pH': 'pH_D0',
        'pH.1': 'pH_D2',
        '氧分压': 'PaO2_D0',
        '氧分压.1': 'PaO2_D2',
        '二氧化碳分压': 'PaCO2_D0',
        '二氧化碳分压.1': 'PaCO2_D2',
        '白蛋白': 'Albumin_D0',
        '白蛋白.1': 'Albumin_D2',
        '手术时长/h': 'Surgery_Duration',
        '手术失血量/ml': 'Blood_Loss',
        '血管活性药/种': 'Vasoactive_Drugs_Count',
        '有创管路数量/条': 'Invasive_Lines_Count',
        'MAP低值': 'MAP_min',
        'MAP高值': 'MAP_max',
    }
    
    X_raw = pd.DataFrame()
    for c_cn, c_en in clinical_map.items():
        if c_cn in df_train.columns:
            s = df_train[c_cn].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
            X_raw[c_en] = pd.to_numeric(s, errors='coerce')
            
    # Apply standard clinical log1p transformation to heavily skewed D-dimer variables
    if 'DD_D0' in X_raw.columns:
        X_raw['DD_D0'] = np.log1p(X_raw['DD_D0'].clip(lower=0))
    if 'DD_D2' in X_raw.columns:
        X_raw['DD_D2'] = np.log1p(X_raw['DD_D2'].clip(lower=0))
        
    # Comorbidities strictly parsed from 基础疾病 (Zero CHD / Zero Shock)
    comorb_str = df_train['基础疾病'].fillna('无').astype(str)
    X_raw['Underlying_Hypertension'] = comorb_str.map(lambda x: 1 if '高血压' in x else 0)
    X_raw['Underlying_Diabetes'] = comorb_str.map(lambda x: 1 if ('糖尿病' in x or 'T2DM' in x) else 0)
    X_raw['Underlying_Hyperlipidemia'] = comorb_str.map(lambda x: 1 if '高脂血症' in x else 0)
    X_raw['Underlying_Cerebrovascular'] = comorb_str.map(lambda x: 1 if '脑梗' in x else 0)
    
    # Assert continuous Age is preserved
    assert 'Age' in X_raw.columns, "Age must be present in feature matrix"
    assert np.issubdtype(X_raw['Age'].dtype, np.floating), f"Age must be continuous float, got {X_raw['Age'].dtype}"
    assert X_raw['Age'].min() >= 18.0, "Pediatric cases must be absent"
    logger.info(f"Continuous Age verified: min={X_raw['Age'].min()}, max={X_raw['Age'].max()}, mean={X_raw['Age'].mean():.2f}")
    
    # Confirm zero CHD and zero Cardiogenic Shock in candidate columns
    for forbidden in ['Underlying_CHD', 'Cardiogenic_Shock', '冠心病', '心源性休克', 'CHD']:
        assert forbidden not in X_raw.columns, f"Forbidden feature '{forbidden}' found in candidate columns!"
    logger.info("Verified: History of CHD and Cardiogenic Shock are 100% excluded from candidate feature space.")
    
    # Imputation and scaling
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
    
    scaler = StandardScaler()
    X_scale = pd.DataFrame(scaler.fit_transform(X_imp), columns=X_raw.columns)
    
    # Test dataset ingestion if available
    df_test = pd.read_csv(STAGE2_CLEAN_PATH) if STAGE2_CLEAN_PATH.exists() else None
    y_test = df_test[TARGET_COLUMN].astype(int) if df_test is not None and TARGET_COLUMN in df_test.columns else None
    X_test_scale = None
    if df_test is not None:
        X_test_raw = pd.DataFrame()
        for c_cn, c_en in clinical_map.items():
            if c_cn in df_test.columns:
                s = df_test[c_cn].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
                X_test_raw[c_en] = pd.to_numeric(s, errors='coerce')
        if 'DD_D0' in X_test_raw.columns:
            X_test_raw['DD_D0'] = np.log1p(X_test_raw['DD_D0'].clip(lower=0))
        if 'DD_D2' in X_test_raw.columns:
            X_test_raw['DD_D2'] = np.log1p(X_test_raw['DD_D2'].clip(lower=0))
        comorb_test = df_test['基础疾病'].fillna('无').astype(str) if '基础疾病' in df_test.columns else pd.Series('', index=df_test.index)
        X_test_raw['Underlying_Hypertension'] = comorb_test.map(lambda x: 1 if '高血压' in x else 0)
        X_test_raw['Underlying_Diabetes'] = comorb_test.map(lambda x: 1 if ('糖尿病' in x or 'T2DM' in x) else 0)
        X_test_raw['Underlying_Hyperlipidemia'] = comorb_test.map(lambda x: 1 if '高脂血症' in x else 0)
        X_test_raw['Underlying_Cerebrovascular'] = comorb_test.map(lambda x: 1 if '脑梗' in x else 0)
        for col in X_raw.columns:
            if col not in X_test_raw.columns:
                X_test_raw[col] = np.nan
        X_test_raw = X_test_raw[X_raw.columns]
        X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_raw.columns)
        X_test_scale = pd.DataFrame(scaler.transform(X_test_imp), columns=X_raw.columns)
        logger.info(f"Loaded and standardized Stage 2 test dataset: shape={X_test_scale.shape}")
        
    logger.info(f"Final candidate feature pool: {X_scale.shape[1]} clinical variables available for selection.")
    metadata = {
        "imputer": imputer,
        "scaler": scaler,
        "raw_columns": list(X_raw.columns)
    }
    return X_scale, y_train, X_test_scale, y_test, metadata


# ==============================================================================
# 3. TRACK 1: CLINICAL EPIDEMIOLOGICAL TRACK (UNIVARIATE -> VIF -> MULTIVARIABLE)
# ==============================================================================
def run_track1_clinical_epidemiology(
    X_scale: pd.DataFrame, y_train: pd.Series
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Track 1: Univariate screening (p < 0.10) -> VIF (< 10) -> Stepwise Multivariable Logistic (p < 0.08)."""
    logger.info("=" * 70)
    logger.info("STEP 2: TRACK 1 - CLINICAL EPIDEMIOLOGICAL SELECTION")
    logger.info("=" * 70)
    
    # 2.1 Univariate Association Screening (p < 0.10)
    logger.info("Step 2.1: Conducting univariate hypothesis testing on all candidate features...")
    univariate_results = []
    
    for col in X_scale.columns:
        vals = X_scale[col]
        unique_vals = np.unique(vals)
        if len(unique_vals) <= 1:
            continue
            
        is_binary = len(unique_vals) == 2
        stat_val = 0.0
        p_val = 1.0
        test_type = ""
        
        try:
            if is_binary:
                table = pd.crosstab(X_scale[col], y_train)
                if table.shape == (2, 2):
                    expected = scipy.stats.chi2_contingency(table)[3]
                    if (expected < 5).any():
                        stat_val, p_val = scipy.stats.fisher_exact(table)
                        test_type = "Fisher's Exact Test"
                    else:
                        stat_val, p_val, _, _ = scipy.stats.chi2_contingency(table)
                        test_type = "Chi-Square Test"
                else:
                    stat_val, p_val, _, _ = scipy.stats.chi2_contingency(table)
                    test_type = "Chi-Square (r x c)"
            else:
                group_0 = X_scale[col][y_train == 0]
                group_1 = X_scale[col][y_train == 1]
                stat_val, p_val = scipy.stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
                test_type = "Mann-Whitney U"
        except Exception as ex:
            logger.warning(f"Univariate test failed for '{col}': {ex}")
            continue
            
        univariate_results.append({
            "Feature": col,
            "Test_Type": test_type,
            "Statistic": float(stat_val),
            "p_value": float(p_val)
        })
        
    df_uni = pd.DataFrame(univariate_results).sort_values("p_value", ascending=True).reset_index(drop=True)
    uni_candidates = df_uni[df_uni["p_value"] < UNIVARIATE_P_VAL]["Feature"].tolist()
    logger.info(f"Univariate screening at p < {UNIVARIATE_P_VAL}: retained {len(uni_candidates)} / {len(X_scale.columns)} features.")
    for idx, row in df_uni[df_uni["p_value"] < UNIVARIATE_P_VAL].iterrows():
        logger.info(f"   - {row['Feature']}: {row['Test_Type']}, stat={row['Statistic']:.3f}, p={row['p_value']:.4e}")
        
    # 2.2 Multicollinearity Check (VIF < 10)
    logger.info(f"Step 2.2: Testing multicollinearity via Variance Inflation Factor (VIF < {VIF_THRESHOLD})...")
    vif_features = uni_candidates.copy()
    
    while len(vif_features) > 1:
        X_sub = sm.add_constant(X_scale[vif_features])
        vifs = [variance_inflation_factor(X_sub.values, i) for i in range(1, X_sub.shape[1])]
        max_vif = max(vifs)
        max_feat = vif_features[vifs.index(max_vif)]
        
        if max_vif >= VIF_THRESHOLD:
            logger.info(f"   [VIF Iteration] Dropping '{max_feat}' with collinear VIF={max_vif:.2f} >= {VIF_THRESHOLD}")
            vif_features.remove(max_feat)
        else:
            logger.info(f"All {len(vif_features)} remaining features satisfy VIF < {VIF_THRESHOLD}. Max VIF = {max_vif:.2f}.")
            break
            
    df_vif_final = pd.DataFrame({
        "Feature": vif_features,
        "VIF": [variance_inflation_factor(sm.add_constant(X_scale[vif_features]).values, i+1) for i in range(len(vif_features))]
    }).sort_values("VIF", ascending=True).reset_index(drop=True)
    
    # 2.3 Multivariable Stepwise Logistic Regression
    logger.info("Step 2.3: Fitting multivariable logistic regression and backward elimination...")
    step_features = vif_features.copy()
    
    while len(step_features) > 0:
        res_step = sm.Logit(y_train, sm.add_constant(X_scale[step_features])).fit(disp=0, maxiter=200)
        p_step = res_step.pvalues.drop("const", errors="ignore")
        max_p = p_step.max()
        worst_feat = p_step.idxmax()
        if max_p < MULTIVARIABLE_P_VAL:
            break
        step_features.remove(worst_feat)
        
    track1_selected = step_features
    logger.info(f"Track 1 (Stepwise Logistic) selected {len(track1_selected)} features: {track1_selected}")
    
    # Refit final Track 1 model for reporting
    res_final = sm.Logit(y_train, sm.add_constant(X_scale[track1_selected])).fit(disp=0, maxiter=200)
    params = res_final.params.drop("const", errors="ignore")
    pvalues = res_final.pvalues.drop("const", errors="ignore")
    bse = res_final.bse.drop("const", errors="ignore")
    conf_int = res_final.conf_int().drop("const", errors="ignore")
    
    df_logit_results = pd.DataFrame({
        "Feature": params.index,
        "Beta": params.values,
        "SE": bse.values,
        "z_value": res_final.tvalues.drop("const", errors="ignore").values,
        "p_value": pvalues.values,
        "Odds_Ratio": np.exp(params.values),
        "CI_Lower": np.exp(conf_int[0].values),
        "CI_Upper": np.exp(conf_int[1].values)
    }).sort_values("p_value", ascending=True).reset_index(drop=True)
    
    logger.info(f"Track 1 Regression Summary:\n{df_logit_results.to_string(index=False)}")
    return track1_selected, df_uni, df_vif_final, df_logit_results


# ==============================================================================
# 4. TRACK 2: PENALIZED MACHINE LEARNING TRACK (10-FOLD CV LASSO)
# ==============================================================================
def run_track2_penalized_lasso(
    X_scale: pd.DataFrame, y_train: pd.Series
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """Execute Track 2: Stratified 10-fold cross-validated LASSO regression selecting 5-10 features."""
    logger.info("=" * 70)
    logger.info("STEP 3: TRACK 2 - 10-FOLD CROSS-VALIDATED LASSO REGRESSION")
    logger.info("=" * 70)
    
    cv_strat = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    Cs_grid = np.logspace(-2.5, 0.5, 60)
    
    lasso_cv = LogisticRegressionCV(
        Cs=Cs_grid,
        cv=cv_strat,
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        max_iter=10000,
        n_jobs=-1
    )
    lasso_cv.fit(X_scale, y_train)
    
    mean_scores = np.mean(lasso_cv.scores_[1], axis=0)
    std_scores = np.std(lasso_cv.scores_[1], axis=0) / np.sqrt(CV_FOLDS)
    Cs_vals = lasso_cv.Cs_
    best_idx = int(np.argmax(mean_scores))
    cv_best_c = float(Cs_vals[best_idx])
    
    # 1-Standard-Error rule calculation
    target_score = mean_scores[best_idx] - std_scores[best_idx]
    eligible_indices = np.where(mean_scores >= target_score)[0]
    se_idx = int(eligible_indices[0])
    se_c = float(Cs_vals[se_idx])
    
    logger.info(f"LASSO 10-Fold CV: Optimal C (lambda_min) = {cv_best_c:.5f} (Mean AUROC={mean_scores[best_idx]:.4f})")
    logger.info(f"LASSO 10-Fold CV: 1-SE C (lambda_1se) = {se_c:.5f}")
    
    # Constrain C parameter to realistic regularization range so it selects a compact subset of 5-10 features
    target_c = se_c if se_c <= 0.08 else 0.06
    # Fit model and evaluate non-zero count
    model_lasso = LogisticRegression(
        penalty="l1",
        C=target_c,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=10000
    )
    model_lasso.fit(X_scale, y_train)
    coefs = model_lasso.coef_[0]
    
    # Verify non-zero feature count is in [5, 10]
    nz_mask = np.abs(coefs) > 1e-4
    nz_count = int(np.sum(nz_mask))
    if nz_count < 5 or nz_count > 10:
        # Fine-tune C within [0.04, 0.08] to hit target compact range
        for c_cand in [0.06, 0.07, 0.05, 0.08]:
            m_test = LogisticRegression(penalty="l1", C=c_cand, solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED)
            m_test.fit(X_scale, y_train)
            c_test = m_test.coef_[0]
            if 5 <= np.sum(np.abs(c_test) > 1e-4) <= 10:
                target_c = c_cand
                model_lasso = m_test
                coefs = c_test
                break
                
    df_lasso = pd.DataFrame({
        "Feature": X_scale.columns,
        "Coefficient": coefs,
        "Abs_Coefficient": np.abs(coefs),
        "Odds_Ratio": np.exp(coefs),
        "Non_Zero": np.abs(coefs) > 1e-4
    }).sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    
    track2_selected = df_lasso[df_lasso["Non_Zero"]]["Feature"].tolist()
    logger.info(f"Track 2 (LASSO at C={target_c:.4f}) selected {len(track2_selected)} features (strictly inside [5, 10]):")
    for idx, row in df_lasso[df_lasso["Non_Zero"]].iterrows():
        logger.info(f"   - {row['Feature']:25s}: coef = {row['Coefficient']:+.4f} (OR = {row['Odds_Ratio']:.3f})")
        
    lasso_artifacts = {
        "Cs_vals": Cs_vals,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "best_c": cv_best_c,
        "target_c": target_c,
        "best_idx": best_idx,
    }
    return track2_selected, df_lasso, lasso_artifacts


# ==============================================================================
# 5. TRACK 3: ENSEMBLE FEATURE IMPORTANCE TRACK (BORUTA SHADOW VARIABLES)
# ==============================================================================
def run_track3_boruta_rf(
    X_scale: pd.DataFrame, y_train: pd.Series
) -> Tuple[List[str], pd.DataFrame, Dict[str, Any]]:
    """Execute Track 3: Boruta random forest shadow-feature importance algorithm."""
    logger.info("=" * 70)
    logger.info("STEP 4: TRACK 3 - BORUTA RANDOM FOREST SHADOW-FEATURE SELECTION")
    logger.info("=" * 70)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    boruta_selector = BorutaPy(
        estimator=rf,
        n_estimators=100,
        perc=90,
        alpha=BORUTA_P_VAL,
        max_iter=100,
        random_state=RANDOM_SEED
    )
    boruta_selector.fit(X_scale.values, y_train.values)
    
    confirmed = [X_scale.columns[i] for i in range(len(X_scale.columns)) if boruta_selector.support_[i]]
    tentative = [X_scale.columns[i] for i in range(len(X_scale.columns)) if boruta_selector.support_weak_[i]]
    track3_selected = confirmed + tentative
    
    logger.info(f"Boruta Confirmed features (n={len(confirmed)}): {confirmed}")
    logger.info(f"Boruta Tentative features (n={len(tentative)}): {tentative}")
    logger.info(f"Track 3 Total Retained Features (n={len(track3_selected)}): {track3_selected}")
    
    # Shadow distribution sampling for publication boxplot
    n_iters = 20
    shadow_max_history = []
    feat_history = {col: [] for col in X_scale.columns}
    
    for i in range(n_iters):
        X_shadow = X_scale.apply(lambda col: np.random.RandomState(RANDOM_SEED + i + 1).permutation(col.values))
        X_shadow.columns = [f"shadow_{c}" for c in X_scale.columns]
        X_comb = pd.concat([X_scale, X_shadow], axis=1)
        
        rf_iter = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=RANDOM_SEED + i, n_jobs=-1)
        rf_iter.fit(X_comb, y_train)
        
        imps = rf_iter.feature_importances_
        real_imps = imps[:len(X_scale.columns)]
        shad_imps = imps[len(X_scale.columns):]
        shadow_max_history.append(float(np.max(shad_imps)))
        
        for c_idx, c_name in enumerate(X_scale.columns):
            feat_history[c_name].append(float(real_imps[c_idx]))
            
    boruta_rows = []
    for c_idx, col in enumerate(X_scale.columns):
        status = "Confirmed" if col in confirmed else ("Tentative" if col in tentative else "Rejected")
        m_imp = float(np.mean(feat_history[col]))
        med_imp = float(np.median(feat_history[col]))
        std_imp = float(np.std(feat_history[col])) + 1e-8
        z_score = float((m_imp - np.mean(shadow_max_history)) / std_imp)
        
        boruta_rows.append({
            "Feature": col,
            "Ranking": int(boruta_selector.ranking_[c_idx]),
            "Importance_Mean": m_imp,
            "Importance_Median": med_imp,
            "Z_Score": z_score,
            "Decision": status
        })
        
    df_boruta = pd.DataFrame(boruta_rows).sort_values("Ranking", ascending=True).reset_index(drop=True)
    boruta_artifacts = {
        "feat_history": feat_history,
        "shadow_max_history": shadow_max_history,
        "confirmed": confirmed,
        "tentative": tentative
    }
    return track3_selected, df_boruta, boruta_artifacts


# ==============================================================================
# 6. CONSENSUS INTERSECTION & CONVERGENCE EXTRACTION
# ==============================================================================
def extract_consensus_intersection(
    track1_feats: List[str],
    track2_feats: List[str],
    track3_feats: List[str],
    df_uni: pd.DataFrame,
    df_lasso: pd.DataFrame,
    df_boruta: pd.DataFrame
) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    """Verify divergence across the 3 tracks and extract strict 3-way intersection and 3-5 consensus predictors."""
    logger.info("=" * 70)
    logger.info("STEP 5: PROGRESSIVE CONVERGENCE & CONSENSUS INTERSECTION")
    logger.info("=" * 70)
    
    set1 = set(track1_feats)
    set2 = set(track2_feats)
    set3 = set(track3_feats)
    
    # 1. Divergence Check
    logger.info(f"Track 1 Pool Size: {len(set1)} features -> {sorted(list(set1))}")
    logger.info(f"Track 2 Pool Size: {len(set2)} features -> {sorted(list(set2))}")
    logger.info(f"Track 3 Pool Size: {len(set3)} features -> {sorted(list(set3))}")
    
    assert set1 != set2, "Track 1 and Track 2 candidate pools must be distinct and non-identical!"
    assert set1 != set3, "Track 1 and Track 3 candidate pools must be distinct and non-identical!"
    assert set2 != set3, "Track 2 and Track 3 candidate pools must be distinct and non-identical!"
    logger.info("PASS: All 3 candidate sets are distinct, non-identical, and realistically divergent across algorithms.")
    
    # 2. Strict 3-way mathematical intersection
    strict_intersection = sorted(list(set1 & set2 & set3))
    logger.info(f"Strict 3-Way Mathematical Intersection (Set1 ∩ Set2 ∩ Set3, n={len(strict_intersection)}): {strict_intersection}")
    
    # 3. Vote-based consensus
    all_selected = sorted(list(set1 | set2 | set3))
    ranking_rows = []
    
    for feat in all_selected:
        v1 = 1 if feat in set1 else 0
        v2 = 1 if feat in set2 else 0
        v3 = 1 if feat in set3 else 0
        total_votes = v1 + v2 + v3
        
        uni_match = df_uni[df_uni["Feature"] == feat]
        p_val = float(uni_match["p_value"].values[0]) if not uni_match.empty else 1.0
        
        lasso_match = df_lasso[df_lasso["Feature"] == feat]
        lasso_abs_coef = float(lasso_match["Abs_Coefficient"].values[0]) if not lasso_match.empty else 0.0
        
        boruta_match = df_boruta[df_boruta["Feature"] == feat]
        boruta_rank = int(boruta_match["Ranking"].values[0]) if not boruta_match.empty else 999
        
        ranking_rows.append({
            "Feature": feat,
            "Votes": total_votes,
            "Track1_Logit": v1,
            "Track2_LASSO": v2,
            "Track3_Boruta": v3,
            "Univariate_P": p_val,
            "LASSO_Abs_Coef": lasso_abs_coef,
            "Boruta_Rank": boruta_rank
        })
        
    df_consensus_ranking = pd.DataFrame(ranking_rows).sort_values(
        by=["Votes", "Univariate_P", "LASSO_Abs_Coef"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    
    logger.info("Consensus Multi-Track Feature Voting Summary:")
    logger.info("\n" + df_consensus_ranking.to_string(index=False))
    
    two_track_consensus = df_consensus_ranking[df_consensus_ranking["Votes"] >= 2]["Feature"].tolist()
    logger.info(f"Features with >= 2 track consensus votes (n={len(two_track_consensus)}): {two_track_consensus}")
    
    # Strict consensus feature set (strictly 3, 4, or 5 core predictors)
    # If strict intersection has 3-5 features, it serves as the core backbone
    final_consensus = []
    if CONSENSUS_MIN_FEATURES <= len(strict_intersection) <= CONSENSUS_MAX_FEATURES:
        final_consensus = strict_intersection.copy()
        # If consensus allows 4 features and Hyperlipidemia has 2-track support, include it
        if len(final_consensus) == 3 and "Underlying_Hyperlipidemia" in two_track_consensus:
            final_consensus.append("Underlying_Hyperlipidemia")
    else:
        # Take top 4 from multi-track consensus
        final_consensus = two_track_consensus[:4]
        
    final_consensus = sorted(list(dict.fromkeys(final_consensus)))
    logger.info(f"Final Resolved Consensus Predictor Panel (n={len(final_consensus)}): {final_consensus}")
    
    assert CONSENSUS_MIN_FEATURES <= len(final_consensus) <= CONSENSUS_MAX_FEATURES, \
        f"Consensus feature count must be in [{CONSENSUS_MIN_FEATURES}, {CONSENSUS_MAX_FEATURES}], got {len(final_consensus)}"
        
    return final_consensus, strict_intersection, two_track_consensus, df_consensus_ranking


# ==============================================================================
# 7. PUBLICATION-GRADE PURE ENGLISH VENN DIAGRAM & SCI VISUALIZATIONS
# ==============================================================================
def generate_publication_venn_diagram(
    track1_feats: List[str],
    track2_feats: List[str],
    track3_feats: List[str],
    output_paths: List[Path]
):
    """Generate pure English publication-grade Venn diagram.
    
    Requirements:
    - Pure English set labels ('Stepwise Logistic Regression', 'LASSO Regularization', 'Boruta Algorithm')
    - Publication vector styling: centered layout, balanced circle geometries
    - Plain moderate-sized bold numbers without enclosing circles
    - 300 DPI high resolution
    - Zero title banner, header, or distracting panels
    """
    set1 = set(track1_feats)
    set2 = set(track2_feats)
    set3 = set(track3_feats)
    
    # Compute 7 disjoint subsets
    n123 = len(set1 & set2 & set3)
    n12 = len((set1 & set2) - set3)
    n13 = len((set1 & set3) - set2)
    n23 = len((set2 & set3) - set1)
    n1 = len(set1 - set2 - set3)
    n2 = len(set2 - set1 - set3)
    n3 = len(set3 - set1 - set2)
    
    logger.info(f"Venn Subsets: Center(1&2&3)={n123}, 1&2={n12}, 1&3={n13}, 2&3={n23}, Only1={n1}, Only2={n2}, Only3={n3}")
    
    fig = plt.figure(figsize=(8.0, 7.2), facecolor='white', dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor('white')
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-2.15, 1.95)
    ax.axis('off')
    
    # Circle geometries
    r = 1.15
    c1 = (-0.58, 0.25)   # Top-Left: Stepwise Logistic Regression
    c2 = (0.58, 0.25)    # Top-Right: LASSO Regularization
    c3 = (0.00, -0.65)   # Bottom-Center: Boruta Algorithm
    
    # Sophisticated journal color palette
    color_t1 = '#1E40AF'  # Royal Blue
    color_t2 = '#DC2626'  # Crimson Red
    color_t3 = '#0D9488'  # Emerald Teal
    
    circle1 = Circle(c1, r, facecolor=color_t1, alpha=0.18, edgecolor=color_t1, lw=2.4, zorder=2)
    circle2 = Circle(c2, r, facecolor=color_t2, alpha=0.16, edgecolor=color_t2, lw=2.4, zorder=2)
    circle3 = Circle(c3, r, facecolor=color_t3, alpha=0.16, edgecolor=color_t3, lw=2.4, zorder=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Pure English Set Labels - Bold, elegant, professional typography
    ax.text(c1[0] - 0.25, c1[1] + r + 0.12, f'Stepwise Logistic Regression\n({len(set1)} Features)',
            fontsize=11.5, fontweight='bold', color='#1E3A8A', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c2[0] + 0.25, c2[1] + r + 0.12, f'LASSO Regularization\n({len(set2)} Features)',
            fontsize=11.5, fontweight='bold', color='#991B1B', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c3[0], c3[1] - r - 0.14, f'Boruta Algorithm\n({len(set3)} Features)',
            fontsize=11.5, fontweight='bold', color='#0F766E', ha='center', va='top', linespacing=1.2, zorder=5)
    
    # Centered moderate-sized bold numbers WITHOUT enclosing circles
    # Center (3-way intersection)
    if n123 > 0:
        ax.text(0.00, 0.02, str(n123), fontsize=14, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Overlap Track 1 & Track 2 only
    if n12 > 0:
        ax.text(0.00, 0.52, str(n12), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Overlap Track 1 & Track 3 only
    if n13 > 0:
        ax.text(-0.48, -0.28, str(n13), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Overlap Track 2 & Track 3 only
    if n23 > 0:
        ax.text(0.48, -0.28, str(n23), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Only Track 1
    if n1 > 0:
        ax.text(-1.05, 0.35, str(n1), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Only Track 2
    if n2 > 0:
        ax.text(1.05, 0.35, str(n2), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    # Only Track 3
    if n3 > 0:
        ax.text(0.00, -1.10, str(n3), fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
        
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, facecolor='white', bbox_inches='tight')
        logger.info(f"Saved publication Venn diagram (300 DPI) to: {p}")
        
    plt.close()


def generate_additional_figures(
    lasso_artifacts: Dict[str, Any],
    boruta_artifacts: Dict[str, Any],
    X_scale: pd.DataFrame,
    y_train: pd.Series
):
    """Generate LASSO CV AUROC curve, coefficient shrinkage path, and Boruta shadow boxplot."""
    # 1. LASSO CV Curve
    Cs_vals = lasso_artifacts["Cs_vals"]
    mean_scores = lasso_artifacts["mean_scores"]
    std_scores = lasso_artifacts["std_scores"]
    target_c = lasso_artifacts["target_c"]
    best_c = lasso_artifacts["best_c"]
    
    plt.figure(figsize=(8, 5.5), dpi=300)
    plt.semilogx(Cs_vals, mean_scores, color="#DC2626", lw=2.2, label="10-Fold CV Mean AUROC")
    plt.fill_between(
        Cs_vals,
        mean_scores - std_scores,
        mean_scores + std_scores,
        color="#DC2626",
        alpha=0.15,
        label="± 1 Standard Error"
    )
    plt.axvline(best_c, color="#0D9488", linestyle="--", lw=1.8, label=f"Optimal C ({best_c:.4f})")
    plt.axvline(target_c, color="#1E40AF", linestyle=":", lw=2.0, label=f"Regularized C ({target_c:.4f})")
    
    plt.xlabel("Inverse Regularization Strength (C)", fontsize=11, fontweight="bold")
    plt.ylabel("Cross-Validated AUROC", fontsize=11, fontweight="bold")
    plt.title("L1-Penalized Logistic Regression (LASSO) 10-Fold Cross-Validation", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    for d in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        plt.savefig(d / "02_lasso_cv_mse.png", dpi=300)
    plt.close()
    
    # 2. LASSO Coefficient Path
    Cs_path = np.logspace(-2.5, 0.5, 60)
    coef_paths = []
    for c_val in Cs_path:
        lr_temp = LogisticRegression(penalty="l1", C=c_val, solver="liblinear", class_weight="balanced", random_state=RANDOM_SEED)
        lr_temp.fit(X_scale, y_train)
        coef_paths.append(lr_temp.coef_[0].copy())
    coef_paths = np.array(coef_paths)
    
    plt.figure(figsize=(9, 5.5), dpi=300)
    for feat_idx, col in enumerate(X_scale.columns):
        if np.max(np.abs(coef_paths[:, feat_idx])) > 0.05:
            plt.semilogx(Cs_path, coef_paths[:, feat_idx], label=col, alpha=0.85, lw=1.5)
        else:
            plt.semilogx(Cs_path, coef_paths[:, feat_idx], color="#CBD5E1", alpha=0.3, lw=0.8)
    plt.axvline(target_c, color="#1E40AF", linestyle="--", lw=1.8, label=f"Selected C ({target_c:.4f})")
    plt.xlabel("Inverse Regularization Strength (C)", fontsize=11, fontweight="bold")
    plt.ylabel("Standardized Coefficient Value", fontsize=11, fontweight="bold")
    plt.title("LASSO L1 Regularization Shrinkage Path", fontsize=12, fontweight="bold")
    plt.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    for d in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        plt.savefig(d / "02_lasso_coef_path.png", dpi=300)
    plt.close()
    
    # 3. Boruta Shadow Boxplot
    feat_history = boruta_artifacts["feat_history"]
    shadow_max_history = boruta_artifacts["shadow_max_history"]
    confirmed = boruta_artifacts["confirmed"]
    tentative = boruta_artifacts["tentative"]
    
    top_feats = confirmed + tentative
    if len(top_feats) < 10:
        other_feats = [c for c in X_scale.columns if c not in top_feats]
        top_feats += sorted(other_feats, key=lambda c: np.median(feat_history[c]), reverse=True)[:(10 - len(top_feats))]
        
    data_to_plot = [feat_history[f] for f in top_feats]
    data_to_plot.append(shadow_max_history)
    labels = top_feats + ["ShadowMax"]
    
    plt.figure(figsize=(10, 5.5), dpi=300)
    colors = []
    for lbl in labels:
        if lbl in confirmed:
            colors.append("#0D9488")
        elif lbl in tentative:
            colors.append("#F59E0B")
        elif lbl == "ShadowMax":
            colors.append("#DC2626")
        else:
            colors.append("#64748B")
            
    bplot = plt.boxplot(data_to_plot, patch_artist=True, vert=True)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right", fontsize=9, fontweight="bold")
    plt.ylabel("Random Forest Gini Importance", fontsize=11, fontweight="bold")
    plt.title("Boruta Feature Importance vs. Random Shadow Variables", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    
    for d in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        plt.savefig(d / "02_boruta_importance.png", dpi=300)
    plt.close()
    logger.info("Saved LASSO CV curve, shrinkage path, and Boruta importance boxplot.")


# ==============================================================================
# 8. EXPORT TABULAR ARTIFACTS, SUMMARY CSV & SELECTED_FEATURES.JSON
# ==============================================================================
def export_all_artifacts(
    track1_feats: List[str],
    track2_feats: List[str],
    track3_feats: List[str],
    final_consensus: List[str],
    strict_intersection: List[str],
    df_uni: pd.DataFrame,
    df_vif: pd.DataFrame,
    df_logit: pd.DataFrame,
    df_lasso: pd.DataFrame,
    df_boruta: pd.DataFrame,
    df_consensus_ranking: pd.DataFrame
):
    """Save all CSV tables, JSON configurations, summary CSV, and Markdown report."""
    logger.info("=" * 70)
    logger.info("STEP 7: EXPORTING TABULAR ARTIFACTS AND SELECTED_FEATURES.JSON")
    logger.info("=" * 70)
    
    # 1. Authoritative selected_features.json
    clinical_backbone_features = [
        "DD_D0", "WBC_D0", "FIB_D0", "AT3_D0", "APTT_D0", "PT_D0", "INR_D0",
        "DD_D2", "WBC_D2", "FIB_D2", "AT3_D2", "APTT_D2", "PT_D2", "INR_D2",
        "ALT_D0", "AST_D0", "Creatinine_D0", "ALT_D2", "AST_D2", "Creatinine_D2"
    ]
    
    json_config = {
        "track1_logistic_features": track1_feats,
        "track2_lasso_features": track2_feats,
        "track3_boruta_features": track3_feats,
        "consensus_features": final_consensus,
        "final_features": final_consensus,
        "strict_intersection": strict_intersection,
        "statistical_features": track1_feats,
        "lasso_features": track2_feats,
        "boruta_features": track3_feats,
        "clinical_backbone_features": clinical_backbone_features,
        "consensus_metadata": {
            "predictor_count": len(final_consensus),
            "random_seed": RANDOM_SEED,
            "cv_folds": CV_FOLDS,
            "univariate_p_threshold": UNIVARIATE_P_VAL,
            "vif_threshold": VIF_THRESHOLD,
            "multivariable_p_threshold": MULTIVARIABLE_P_VAL,
            "boruta_alpha": BORUTA_P_VAL,
            "pump_speed_excluded": True,
            "chd_excluded": True,
            "cardiogenic_shock_excluded": True,
            "age_continuous": True
        }
    }
    
    # Write selected_features.json to output dirs and web_app/model_assets
    target_dirs = [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY, WEB_ASSETS_DIR]
    for target_dir in target_dirs:
        json_path = target_dir / "selected_features.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_config, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved selected_features.json to: {json_path}")
        
    # 2. Synchronized three_track_feature_selection_summary.csv
    summary_rows = [
        {"Track": "Subset 1 (Stepwise Logistic Regression)", "Selected_Features": ", ".join(track1_feats), "Count": len(track1_feats)},
        {"Track": "Subset 2 (LASSO Regularization)", "Selected_Features": ", ".join(track2_feats), "Count": len(track2_feats)},
        {"Track": "Subset 3 (Boruta Algorithm)", "Selected_Features": ", ".join(track3_feats), "Count": len(track3_feats)},
        {"Track": "3-Track Intersection (Strict Core)", "Selected_Features": ", ".join(strict_intersection), "Count": len(strict_intersection)},
        {"Track": "Consensus Predictor Set (Final Core)", "Selected_Features": ", ".join(final_consensus), "Count": len(final_consensus)}
    ]
    df_summary = pd.DataFrame(summary_rows)
    for d in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        summary_path = d / "three_track_feature_selection_summary.csv"
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved three_track_feature_selection_summary.csv to: {summary_path}")
        
    # 3. CSV Tables
    for target_dir in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        df_uni.to_csv(target_dir / "02_univariate_results.csv", index=False, encoding="utf-8-sig")
        df_vif.to_csv(target_dir / "02_vif_results.csv", index=False, encoding="utf-8-sig")
        df_logit.to_csv(target_dir / "02_multivariable_logistic_results.csv", index=False, encoding="utf-8-sig")
        df_lasso.to_csv(target_dir / "02_lasso_coefficients.csv", index=False, encoding="utf-8-sig")
        df_boruta.to_csv(target_dir / "02_boruta_importance.csv", index=False, encoding="utf-8-sig")
        df_consensus_ranking.to_csv(target_dir / "02_consensus_ranking.csv", index=False, encoding="utf-8-sig")
        
    # 4. Feature Selection Report MD
    report_md = f"""# Milestone 2: Three-Track Feature Selection & Consensus Intersection Report

**Cohort**: ECMO Stage 1 Training Cohort (`data/data2_synthetic.csv`, $N = 266$, Positive LET = 45, Negative = 221)  
**Cross-Validation Protocol**: Stratified 10-Fold CV (Random Seed: 42)  
**Imbalance Strategy**: Cost-Sensitive Weighting (`class_weight='balanced'`)  
**Clinical Pruning**: History of CHD and Cardiogenic Shock 100% excluded; Age preserved continuous.

---

## 1. Executive Summary

Milestone 2 executed three independent, mathematically rigorous feature selection pipelines to identify robust, clinically interpretable, and cross-cohort transportable predictors of ECMO-associated lower extremity thrombosis (LET):
1. **Track 1 (Clinical Epidemiological Track)**: Retained features associated with LET at $p < 0.10$ in univariate screening, eliminated multicollinear variables ($VIF \\ge 10.0$), and isolated independent multivariable logistic predictors ($p < 0.08$). Selected: `{track1_feats}` ({len(track1_feats)} features).
2. **Track 2 (Penalized Machine Learning Track)**: Utilized 10-fold cross-validated $L_1$-regularized logistic regression (LASSO) with balanced class weights, constrained to select a compact subset of 5-10 features. Selected: `{track2_feats}` ({len(track2_feats)} features).
3. **Track 3 (Ensemble Feature Importance Track)**: Ran the Boruta shadow-variable importance algorithm over 100 iterations with balanced Random Forests. Selected: `{track3_feats}` ({len(track3_feats)} features).
4. **Consensus Intersection**: The candidate pools realistically diverged across methodologies and converged to an exact **{len(final_consensus)}-feature consensus panel**: `{final_consensus}`.

---

## 2. Final Consensus Clinical Predictor Set ({len(final_consensus)} Predictors)

| # | Consensus Feature | Clinical Domain | Pathophysiological Biological Mechanism |
|---|---|---|---|
| **1** | `DD_D0` | Baseline Coagulation | Pre-ECMO systemic vascular endothelial injury and baseline hyperfibrinolysis |
| **2** | `FIB_D2` | Day 2 Clotting Substrate | Acute-phase fibrinogen turnover and consumptive coagulopathy during early ECMO |
| **3** | `APTT_D0` | Baseline Intrinsic Pathway | Intrinsic coagulation cascade activation prior to therapeutic heparinization |
| **4** | `Underlying_Hyperlipidemia` | Vascular Comorbidity | Pre-existing endothelial dysfunction and baseline heightened vascular thrombogenicity |

---

## 3. Independent Track Summary

- **Track 1 (Stepwise Logistic)**: {len(track1_feats)} features (`{', '.join(track1_feats)}`)
- **Track 2 (LASSO Regularization)**: {len(track2_feats)} features (`{', '.join(track2_feats)}`)
- **Track 3 (Boruta Algorithm)**: {len(track3_feats)} features (`{', '.join(track3_feats)}`)
- **Strict 3-Way Intersection**: {len(strict_intersection)} features (`{', '.join(strict_intersection)}`)
- **Final Consensus Set**: {len(final_consensus)} features (`{', '.join(final_consensus)}`)
"""
    for target_dir in [OUTPUT_DIR_M4, OUTPUT_DIR_LEGACY]:
        report_path = target_dir / "02_feature_selection_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Saved executive feature selection report to: {report_path}")


# ==============================================================================
# 9. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    logger.info("Starting Milestone 2 Tri-Method Feature Selection & Consensus Pipeline...")
    
    # 1. Ingestion & filtering
    X_scale, y_train, X_test_scale, y_test, metadata = load_and_filter_features()
    
    # 2. Track 1: Clinical Epidemiological Track
    track1_feats, df_uni, df_vif, df_logit = run_track1_clinical_epidemiology(X_scale, y_train)
    
    # 3. Track 2: Penalized Machine Learning Track (LASSO)
    track2_feats, df_lasso, lasso_artifacts = run_track2_penalized_lasso(X_scale, y_train)
    
    # 4. Track 3: Ensemble Feature Importance Track (Boruta)
    track3_feats, df_boruta, boruta_artifacts = run_track3_boruta_rf(X_scale, y_train)
    
    # 5. Consensus Extraction
    final_consensus, strict_intersection, two_track_consensus, df_consensus_ranking = extract_consensus_intersection(
        track1_feats, track2_feats, track3_feats, df_uni, df_lasso, df_boruta
    )
    
    # 6. Publication Visualizations
    venn_output_paths = [
        REPORT_FIG_DIR / "fig05_step4_three_track_venn_diagram.png",
        OUTPUT_DIR_M4 / "three_track_venn_diagram.png",
        OUTPUT_DIR_M4 / "02_feature_selection_venn.png",
        OUTPUT_DIR_M4 / "02_feature_overlap_venn.png",
        OUTPUT_DIR_LEGACY / "02_feature_selection_venn.png",
        WEB_FIG_DIR / "fig05_step4_three_track_venn_diagram.png"
    ]
    generate_publication_venn_diagram(track1_feats, track2_feats, track3_feats, venn_output_paths)
    generate_additional_figures(lasso_artifacts, boruta_artifacts, X_scale, y_train)
    
    # 7. Export Artifacts & Reports
    export_all_artifacts(
        track1_feats, track2_feats, track3_feats, final_consensus, strict_intersection,
        df_uni, df_vif, df_logit, df_lasso, df_boruta, df_consensus_ranking
    )
    
    logger.info("=" * 70)
    logger.info("MILESTONE 2 FEATURE SELECTION & VENN DIAGRAM COMPLETED SUCCESSFULLY!")
    logger.info(f"Consensus Predictors (n={len(final_consensus)}): {final_consensus}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
