import os
import json
import logging
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from matplotlib_venn import venn3

sys.path.append(os.getcwd())
from scripts.exploration.visual_utils import (
    plot_lasso_cv, plot_lasso_coef, plot_boruta_importance
)

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "False").lower() in ["true", "1", "yes"]
base_dir = r"D:\acode\ECMO\output\data_process_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_process"

TRAIN_DATA_PATH = os.path.join(base_dir, "train_preprocessed.csv")
TEST_DATA_PATH = os.path.join(base_dir, "test_preprocessed.csv")
OUTPUT_DIR = base_dir
RANDOM_SEED = 30

# Thresholds for feature selection
UNIVARIATE_P_VAL = 0.20          # Cut-off p-value for univariate pre-screening
VIF_THRESHOLD = 10.0            # Cut-off for multicollinearity screening
MULTIVARIABLE_P_VAL = 0.15      # Cut-off p-value in multivariable logistic regression
LASSO_RULE = "min"              # "min" for MSE minimizing alpha, "1se" for 1-std-error rule alpha
BORUTA_P_VAL = 0.15             # Cut-off p-value for Boruta shadow features (using alpha=0.15 for better sensitivity)
SELECTION_MODE = "automated_consensus" # "clinical_backbone", "automated_consensus", "strict_intersection", "clinical_selected", or "all"
EXCLUDE_LATE_FEATURES = True     # Exclude Day 7 and Wean variables to build an early warning model
# ==========================================

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup Logging
log_file = os.path.join(OUTPUT_DIR, "02_feature_selection.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_selection")

logger.info("Starting Multi-Track Feature Selection Pipeline (data_process)...")

# ==========================================
# 2. LOAD DATA
# ==========================================
logger.info(f"Loading train data from: {TRAIN_DATA_PATH}")
df_train = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8-sig")
logger.info(f"Loading test data from: {TEST_DATA_PATH}")
df_test = pd.read_csv(TEST_DATA_PATH, encoding="utf-8-sig")

target_col = "是否下肢血栓"
X_train = df_train.drop(columns=[target_col])
y_train = df_train[target_col]
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

# Exclude late-timepoint features (Day 7 and Wean) to build an early warning model
if EXCLUDE_LATE_FEATURES:
    late_cols = [c for c in X_train.columns if "_D7" in c or "_Wean" in c or "Max_DD" in c or "Min_PLT" in c or "Min_AT3" in c or "Mean_APTT" in c]
    logger.info(f"Excluding {len(late_cols)} late-timepoint (D7/Wean) features for early-warning predictive modeling.")
    X_train = X_train.drop(columns=late_cols)
    X_test = X_test.drop(columns=late_cols)

# Drop non-clinical/sociodemographic and calendar noise columns, and features with >30% missingness (like CRP)
cols_to_drop = [c for c in X_train.columns if 
                "Occupation" in c or 
                "Residence" in c or 
                "Work_Status" in c or 
                "Pregnancy" in c or 
                "Diagnosis_Primary" in c or 
                "Diagnosis_Secondary" in c or 
                "Transfusion_Types_Count" in c or
                "Admission_" in c or 
                "CRP" in c or
                c in ["Smoking", "Drinking", "Sex"]]
if cols_to_drop:
    logger.info(f"Purging non-clinical / sociodemographic and calendar columns: {cols_to_drop}")
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)

logger.info(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")

# Set matplotlib aesthetics
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ==========================================
# 3. TRACK 1: MEDICAL STATISTICAL SELECTION
# ==========================================
logger.info("=== TRACK 1: MEDICAL STATISTICAL SELECTION ===")

# 3.1 Univariate Pre-screening
logger.info("Step 1.1: Running Univariate Association Tests (t-test / Wilcoxon / Chi-square / Fisher's)...")
univariate_results = []
significant_features = []

for col in X_train.columns:
    vals = X_train[col].dropna()
    unique_vals = vals.unique()
    
    # Skip zero variance columns
    if len(unique_vals) <= 1:
        continue
        
    # Check if binary (0/1)
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0.0, 1.0, 0, 1})
    
    p_val = 1.0
    test_type = ""
    stat_val = 0.0
    
    try:
        if is_binary:
            contingency_table = pd.crosstab(X_train[col], y_train)
            if contingency_table.shape == (2, 2):
                expected = scipy.stats.chi2_contingency(contingency_table)[3]
                if (expected < 5).any():
                    stat_val, p_val = scipy.stats.fisher_exact(contingency_table)
                    test_type = "Fisher's Exact Test"
                else:
                    stat_val, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                    test_type = "Chi-Square Test"
            else:
                stat_val, p_val, dof, expected = scipy.stats.chi2_contingency(contingency_table)
                test_type = "Chi-Square Test (non-2x2)"
        else:
            group_0 = X_train[col][y_train == 0]
            group_1 = X_train[col][y_train == 1]
            
            _, p_norm = scipy.stats.shapiro(vals)
            
            if p_norm > 0.05:
                stat_val, p_val = scipy.stats.ttest_ind(group_0, group_1, equal_var=False)
                test_type = "Welch's t-test"
            else:
                stat_val, p_val = scipy.stats.mannwhitneyu(group_0, group_1, alternative="two-sided")
                test_type = "Mann-Whitney U"
    except Exception as e:
        logger.warning(f"Univariate test failed for feature '{col}': {e}. Skipping.")
        continue
        
    univariate_results.append({
        "Feature": col,
        "Test_Type": test_type,
        "Statistic": stat_val,
        "p_value": p_val
    })
    
    if p_val < UNIVARIATE_P_VAL:
        significant_features.append(col)

df_uni = pd.DataFrame(univariate_results).sort_values("p_value")
uni_path = os.path.join(OUTPUT_DIR, "02_univariate_results.csv")
df_uni.to_csv(uni_path, index=False, encoding="utf-8-sig")
logger.info(f"Univariate screening completed. Saved to: {uni_path}")
logger.info(f"Features significant at p < {UNIVARIATE_P_VAL}: {len(significant_features)} / {X_train.shape[1]}")

# 3.2 Multicollinearity Check (VIF)
logger.info("Step 1.2: Checking Multicollinearity via Variance Inflation Factor (VIF)...")
vif_features = significant_features.copy()

iteration = 1
while len(vif_features) > 1:
    X_vif = sm.add_constant(X_train[vif_features])
    
    vif_data = []
    for i in range(1, X_vif.shape[1]):
        col_name = X_vif.columns[i]
        try:
            vif_val = variance_inflation_factor(X_vif.values, i)
            vif_data.append({"Feature": col_name, "VIF": vif_val})
        except Exception as e:
            vif_data.append({"Feature": col_name, "VIF": 999.0})
            
    df_vif = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    max_vif = df_vif.iloc[0]["VIF"]
    max_feat = df_vif.iloc[0]["Feature"]
    
    if max_vif > VIF_THRESHOLD:
        logger.info(f"[VIF Iteration {iteration}] Dropping '{max_feat}' with VIF = {max_vif:.2f}")
        vif_features.remove(max_feat)
        iteration += 1
    else:
        vif_path = os.path.join(OUTPUT_DIR, "02_vif_results.csv")
        df_vif.to_csv(vif_path, index=False, encoding="utf-8-sig")
        logger.info(f"Multicollinearity check complete. Final VIF table saved to: {vif_path}")
        break

logger.info(f"Features remaining after VIF screening: {len(vif_features)}")

# 3.3 Multivariable Logistic Regression
logger.info("Step 1.3: Fitting Multivariable Logistic Regression...")
track1_features = []
if len(vif_features) > 0:
    try:
        X_multi = sm.add_constant(X_train[vif_features])
        logit_model = sm.Logit(y_train, X_multi).fit(disp=0)
        
        p_vals = logit_model.pvalues.drop("const", errors="ignore")
        df_logit = pd.DataFrame({
            "Feature": p_vals.index,
            "Beta": logit_model.params.drop("const", errors="ignore"),
            "OR": np.exp(logit_model.params.drop("const", errors="ignore")),
            "p_value": p_vals.values
        }).sort_values("p_value")
        
        logit_path = os.path.join(OUTPUT_DIR, "02_multivariable_logistic_results.csv")
        df_logit.to_csv(logit_path, index=False, encoding="utf-8-sig")
        logger.info(f"Logistic regression completed. Results saved to: {logit_path}")
        
        track1_features = df_logit[df_logit["p_value"] < MULTIVARIABLE_P_VAL]["Feature"].tolist()
        logger.info(f"Track 1 (Statistical) Selected Features (p < {MULTIVARIABLE_P_VAL}): {track1_features}")
    except Exception as e:
        logger.error(f"Multivariable Logistic Regression failed: {e}. Falling back to VIF filtered features.")
        track1_features = vif_features.copy()
else:
    logger.warning("No features remained for Multivariable Regression!")
    track1_features = []

# ==========================================
# 4. TRACK 2: LASSO REGRESSION SELECTION
# ==========================================
logger.info("=== TRACK 2: LASSO REGRESSION SELECTION ===")
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

lasso_cv = LogisticRegressionCV(
    Cs=np.logspace(-2.5, 1.5, 50), 
    cv=cv_strat, 
    penalty="l1", 
    solver="liblinear", 
    class_weight="balanced", 
    scoring="roc_auc", 
    random_state=RANDOM_SEED, 
    max_iter=10000, 
    n_jobs=-1
)
lasso_cv.fit(X_train[significant_features], y_train)

mean_scores = np.mean(lasso_cv.scores_[1.0], axis=0)
std_scores = np.std(lasso_cv.scores_[1.0], axis=0) / np.sqrt(5)
Cs_vals = lasso_cv.Cs_

best_idx = np.argmax(mean_scores)
best_c = float(Cs_vals[best_idx])

target_score = mean_scores[best_idx] - std_scores[best_idx]
eligible_indices = np.where(mean_scores >= target_score)[0]
se_idx = eligible_indices[0]
se_c = float(Cs_vals[se_idx])

logger.info(f"L1 Logistic Regression CV: Best C = {best_c:.6f}, 1-SE C = {se_c:.6f}")
selected_c = se_c if LASSO_RULE == "1se" else best_c
logger.info(f"Using LASSO rule: {LASSO_RULE} (Selected C = {selected_c:.6f})")

model_selected = LogisticRegression(
    penalty="l1", C=selected_c, solver="liblinear", 
    class_weight="balanced", random_state=RANDOM_SEED, max_iter=10000
)
model_selected.fit(X_train[significant_features], y_train)

coefs = model_selected.coef_[0]
lasso_min_features = [significant_features[i] for i in range(len(significant_features)) if coefs[i] != 0]
track2_features = lasso_min_features
logger.info(f"Track 2 (LASSO) Selected Features: {track2_features}")

lasso_plot_path = os.path.join(OUTPUT_DIR, "02_lasso_cv_mse.png")
plot_lasso_cv(Cs_vals, mean_scores, std_scores, best_c, se_c, lasso_plot_path)

Cs_path = np.logspace(-3, 3, 100)
coefs_path = []
for c_val in Cs_path:
    l_temp = LogisticRegression(
        penalty="l1", C=c_val, solver="liblinear", 
        class_weight="balanced", random_state=RANDOM_SEED, max_iter=10000
    )
    l_temp.fit(X_train[significant_features], y_train)
    coefs_path.append(l_temp.coef_[0])

lasso_path_plot_path = os.path.join(OUTPUT_DIR, "02_lasso_coef_path.png")
plot_lasso_coef(Cs_path, coefs_path, best_c, lasso_path_plot_path)

# ==========================================
# 5. TRACK 3: BORUTA FEATURE SELECTION
# ==========================================
logger.info("=== TRACK 3: BORUTA FEATURE SELECTION ===")
rf = RandomForestClassifier(n_estimators="auto", max_depth=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
boruta_selector = BorutaPy(rf, n_estimators="auto", perc=90, alpha=BORUTA_P_VAL, max_iter=100, random_state=RANDOM_SEED)
boruta_selector.fit(X_train[significant_features].values, y_train.values)

track3_features = [significant_features[i] for i in range(len(significant_features)) if boruta_selector.support_[i] or boruta_selector.support_weak_[i]]
tentative_features = [significant_features[i] for i in range(len(significant_features)) if boruta_selector.support_weak_[i]]

if len(track3_features) == 0:
    logger.info("Boruta Support is empty. Extracting top 10 features based on Random Forest feature importances...")
    rf_temp = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
    rf_temp.fit(X_train[significant_features], y_train)
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1]
    track3_features = [significant_features[i] for i in indices[:10]]

logger.info(f"Boruta Confirmed features: {np.sum(boruta_selector.support_)}")
logger.info(f"Boruta Tentative features: {len(tentative_features)}")
logger.info(f"Track 3 (Boruta/RF) Selected Features: {track3_features}")

green_feats = [significant_features[i] for i in range(len(significant_features)) if boruta_selector.support_[i]]
blue_feats = [significant_features[i] for i in range(len(significant_features)) if boruta_selector.support_weak_[i]]
red_feats = [significant_features[i] for i in range(len(significant_features)) if not (boruta_selector.support_[i] or boruta_selector.support_weak_[i])]

logger.info("Running custom Random Forest iterations to collect shadow feature statistics for boxplot...")
n_shadow_iters = 20
shadow_max_history = []
shadow_mean_history = []
shadow_min_history = []
feat_history = {col: [] for col in significant_features}

for iter_idx in range(n_shadow_iters):
    X_significant = X_train[significant_features]
    X_shadow = X_significant.apply(lambda x: np.random.permutation(x.values))
    X_shadow.columns = [f"shadow_{col}" for col in significant_features]
    
    X_combined = pd.concat([X_significant, X_shadow], axis=1)
    
    rf_temp = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", n_jobs=-1, random_state=iter_idx)
    rf_temp.fit(X_combined, y_train)
    
    imps = rf_temp.feature_importances_
    orig_imps = imps[:len(significant_features)]
    shadow_imps = imps[len(significant_features):]
    
    shadow_max_history.append(np.max(shadow_imps))
    shadow_mean_history.append(np.mean(shadow_imps))
    shadow_min_history.append(np.min(shadow_imps))
    
    for col_idx, col_name in enumerate(significant_features):
        feat_history[col_name].append(orig_imps[col_idx])

data_list = []
for col_name in significant_features:
    status = "Confirmed" if col_name in green_feats else ("Tentative" if col_name in blue_feats else "Rejected")
    data_list.append({"Feature": col_name, "Importance": np.median(feat_history[col_name]), "Type": status})

df_plot = pd.DataFrame(data_list).sort_values("Importance", ascending=False)
boruta_plot_path = os.path.join(OUTPUT_DIR, "02_boruta_importance.png")
plot_boruta_importance(
    df_plot=df_plot,
    output_path=boruta_plot_path,
    feat_history=feat_history,
    shadow_max_history=shadow_max_history,
    shadow_mean_history=shadow_mean_history,
    shadow_min_history=shadow_min_history,
    green_feats=green_feats,
    blue_feats=blue_feats
)

# ==========================================
# 6. CONSENSUS FUSION & OVERLAP ANALYSIS
# ==========================================
logger.info("=== FEATURE FUSION & OVERLAP ANALYSIS ===")
set_stat = set(track1_features)
set_lasso = set(track2_features)
set_boruta = set(track3_features)

strict_intersection = list(set_stat.intersection(set_lasso).intersection(set_boruta))
consensus_features = []
all_selected = list(set_stat.union(set_lasso).union(set_boruta))
for feat in all_selected:
    vote_count = 0
    if feat in set_stat: vote_count += 1
    if feat in set_lasso: vote_count += 1
    if feat in set_boruta: vote_count += 1
    if vote_count >= 2:
        consensus_features.append(feat)

logger.info(f"Strict Intersection Features: {strict_intersection}")
logger.info(f"Consensus Features: {consensus_features}")

plt.figure(figsize=(7, 7))
venn = venn3(
    [set_stat, set_lasso, set_boruta],
    set_labels=("Statistical\n(Track 1)", "LASSO\n(Track 2)", "Boruta\n(Track 3)"),
    set_colors=("#2980b9", "#c0392b", "#27ae60"),
    alpha=0.6
)
plt.title("Feature Selection Track Overlap Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
venn_plot_path = os.path.join(OUTPUT_DIR, "02_feature_overlap_venn.png")
plt.savefig(venn_plot_path, dpi=300)
plt.close()

# Clinical backbone definition for mapping fallback
clinical_backbone_features = [
    "ALT_D0", "AST_D0", "Albumin_D0", "Creatinine_D0", "WBC_D0", "RBC_D0", "Hb_D0",
    "DD_D0", "PLT_D0", "FIB_D0", "AT3_D0", "APTT_D0", "PT_D0", "INR_D0", "TT_D0",
    "PLT_min", "FIB_min", "AT3_min", "DD_max", "APTT_max", "PT_max", "INR_max", "TT_max",
    "DD_slope", "PLT_slope", "FIB_slope", "AT3_slope", "APTT_slope", "PT_slope", "INR_slope", "TT_slope",
    "DD_abnormality_ratio", "PLT_abnormality_ratio", "FIB_abnormality_ratio", "AT3_abnormality_ratio",
    "APTT_abnormality_ratio", "PT_abnormality_ratio", "INR_abnormality_ratio", "TT_abnormality_ratio"
]
clinical_backbone_features = [f for f in clinical_backbone_features if f in X_train.columns]

MAX_FEATURES = 4
feature_ranking = []
for feat in all_selected:
    vote_count = 0
    if feat in set_stat: vote_count += 1
    if feat in set_lasso: vote_count += 1
    if feat in set_boruta: vote_count += 1
    
    p_val = 1.0
    if feat in df_uni["Feature"].values:
        p_val = df_uni[df_uni["Feature"] == feat]["p_value"].values[0]
        
    feature_ranking.append({
        "Feature": feat,
        "Votes": vote_count,
        "p_value": p_val
    })
df_ranking = pd.DataFrame(feature_ranking).sort_values(by=["Votes", "p_value"], ascending=[False, True])
logger.info(f"Consensus feature ranking:\n{df_ranking.to_string(index=False)}")

if SELECTION_MODE == "clinical_backbone":
    candidates = [f for f in df_ranking["Feature"].tolist() if f in clinical_backbone_features]
    final_features = candidates[:MAX_FEATURES]
elif SELECTION_MODE == "automated_consensus":
    candidates = df_ranking[df_ranking["Votes"] >= 2]["Feature"].tolist()
    final_features = candidates[:MAX_FEATURES]
    if len(final_features) < MAX_FEATURES:
        extra_cands = df_ranking[df_ranking["Votes"] == 1]["Feature"].tolist()
        final_features.extend(extra_cands[:(MAX_FEATURES - len(final_features))])
elif SELECTION_MODE == "strict_intersection":
    candidates = df_ranking[df_ranking["Votes"] == 3]["Feature"].tolist()
    final_features = candidates[:MAX_FEATURES]
else:
    final_features = X_train.columns.tolist()[:MAX_FEATURES]

logger.info(f"Final Selected Predictive Features for Downstream Modeling (n={len(final_features)}): {final_features}")

# Save final feature set
feature_config = {
    "statistical_features": list(set_stat),
    "lasso_features": list(set_lasso),
    "boruta_features": list(set_boruta),
    "strict_intersection": strict_intersection,
    "consensus_features": consensus_features,
    "clinical_backbone_features": clinical_backbone_features,
    "final_features": final_features,
    "lasso_rule": LASSO_RULE,
    "selection_mode": SELECTION_MODE
}

feat_config_path = os.path.join(OUTPUT_DIR, "selected_features.json")
with open(feat_config_path, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, indent=4, ensure_ascii=False)
logger.info(f"Saved final features list config to: {feat_config_path}")
logger.info("Feature Selection Pipeline (data_process) Completed Successfully.")
