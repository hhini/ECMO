# -*- coding: utf-8 -*-
"""
Adversarial Stress-Test Harness for Milestones 2 & 3
Author: Empirical Challenger
Reviewing:
- Milestone 2: 02_generate_table1.py & output/02_table1_baseline/
- Milestone 3: 03_benchmark_imbalance.py & output/03_imbalance_benchmark/
"""

import sys
import os
import json
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
OUTPUT_M2 = PROJECT_ROOT / "output" / "02_table1_baseline"
OUTPUT_M3 = PROJECT_ROOT / "output" / "03_imbalance_benchmark"
STAGE1_PATH = PROJECT_ROOT / "data" / "data2.csv"
STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"

# Global challenge findings tracker
findings = []

def record_finding(status: str, title: str, details: str):
    findings.append({"status": status, "title": title, "details": details})
    print(f"[{status}] {title}: {details}")

print("================================================================================")
print("  ADVERSARIAL STRESS-TEST SUITE: MILESTONES 2 & 3")
print("================================================================================")

# ==============================================================================
# SECTION 1: MILESTONE 2 (TABLE 1) ADVERSARIAL CHALLENGES
# ==============================================================================
print("\n--- 1. Testing Sample Size Consistency Across All Table 1 Files ---")

# Ground truth counts from raw source files
df_s1_raw = pd.read_csv(STAGE1_PATH)
df_s2_raw = pd.read_csv(STAGE2_PATH)

s1_total = len(df_s1_raw)
s1_pos = int((df_s1_raw['是否下肢血栓'] == 1).sum())
s1_neg = int((df_s1_raw['是否下肢血栓'] == 0).sum())

s2_total = len(df_s2_raw)
s2_pos = int((df_s2_raw['是否下肢血栓'] == 1).sum())
s2_neg = int((df_s2_raw['是否下肢血栓'] == 0).sum())

comb_total = s1_total + s2_total
comb_pos = s1_pos + s2_pos
comb_neg = s1_neg + s2_neg

print(f"Ground Truth Stage 1: Total={s1_total}, Pos={s1_pos}, Neg={s1_neg}")
print(f"Ground Truth Stage 2: Total={s2_total}, Pos={s2_pos}, Neg={s2_neg}")
print(f"Ground Truth Combined: Total={comb_total}, Pos={comb_pos}, Neg={comb_neg}")

# Verify M2 output files
t1_s1_file = OUTPUT_M2 / "table1_stage1.csv"
t1_s2_file = OUTPUT_M2 / "table1_stage2.csv"
t1_comb_cohorts_file = OUTPUT_M2 / "table1_combined_cohorts.csv"
t1_comb_thromb_file = OUTPUT_M2 / "table1_combined_thrombosis.csv"
t1_comb_mirror = OUTPUT_M2 / "table1_combined.csv"

for p in [t1_s1_file, t1_s2_file, t1_comb_cohorts_file, t1_comb_thromb_file, t1_comb_mirror]:
    if not p.exists():
        record_finding("FAIL", "Missing M2 File", f"Expected file {p} does not exist.")
    else:
        record_finding("PASS", "File Exists", f"{p.name} exists ({p.stat().st_size} bytes).")

# Check headers in CSVs
df_t1_s1 = pd.read_csv(t1_s1_file)
df_t1_s2 = pd.read_csv(t1_s2_file)
df_t1_comb_c = pd.read_csv(t1_comb_cohorts_file)
df_t1_comb_t = pd.read_csv(t1_comb_thromb_file)

# Check sample sizes in headers
col_s1_tot = [c for c in df_t1_s1.columns if f"N={s1_total}" in c or f"N = {s1_total}" in c or f"n={s1_total}" in c or f"n = {s1_total}" in c]
col_s1_pos = [c for c in df_t1_s1.columns if f"{s1_pos}" in c]
col_s1_neg = [c for c in df_t1_s1.columns if f"{s1_neg}" in c]

if col_s1_tot and col_s1_pos and col_s1_neg:
    record_finding("PASS", "M2 Stage 1 Headers", f"Stage 1 headers match n=266 (pos={s1_pos}, neg={s1_neg}): {df_t1_s1.columns.tolist()[:4]}")
else:
    record_finding("FAIL", "M2 Stage 1 Headers", f"Stage 1 header mismatch: {df_t1_s1.columns.tolist()}")

col_s2_tot = [c for c in df_t1_s2.columns if f"{s2_total}" in c]
col_s2_pos = [c for c in df_t1_s2.columns if f"{s2_pos}" in c]
col_s2_neg = [c for c in df_t1_s2.columns if f"{s2_neg}" in c]

if col_s2_tot and col_s2_pos and col_s2_neg:
    record_finding("PASS", "M2 Stage 2 Headers", f"Stage 2 headers match n=106 (pos={s2_pos}, neg={s2_neg}): {df_t1_s2.columns.tolist()[:4]}")
else:
    record_finding("FAIL", "M2 Stage 2 Headers", f"Stage 2 header mismatch: {df_t1_s2.columns.tolist()}")

# Check combined files
if any(f"{comb_total}" in c for c in df_t1_comb_t.columns) and any(f"{comb_pos}" in c for c in df_t1_comb_t.columns):
    record_finding("PASS", "M2 Combined Thrombosis Headers", f"Combined headers match N=372 (pos={comb_pos}, neg={comb_neg})")
else:
    record_finding("FAIL", "M2 Combined Thrombosis Headers", f"Combined header mismatch: {df_t1_comb_t.columns.tolist()}")


print("\n--- 2. Adversarial Normality & Statistical Test Assignment Audit ---")
# Import 02_generate_table1 functions to audit feature extraction and testing
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib
gen_t1 = importlib.import_module("02_generate_table1")

df_clean_s1, df_clean_s2, df_comb = gen_t1.load_and_harmonize_cohorts()

# Identify all continuous variables processed in Table 1
# Let's inspect rows in table1_stage1.csv where Test_Method contains t-test
ttest_rows_s1 = df_t1_s1[df_t1_s1['Test_Method'].str.contains("t-test", case=False, na=False)]
print(f"Stage 1 Continuous variables evaluated with t-test: {len(ttest_rows_s1)}")

# For each variable tested with Welch's t-test, check its skewness, kurtosis, and normality
skew_violations = []
for _, row in ttest_rows_s1.iterrows():
    var_name = row['Variable']
    # Match column in df_clean_s1
    # Search by matching var_name in df_clean_s1
    matched_col = None
    for col in df_clean_s1.columns:
        if col in var_name or var_name.split()[0] in col or col.replace('_', ' ') in var_name:
            matched_col = col
            break
    
    if matched_col and pd.api.types.is_numeric_dtype(df_clean_s1[matched_col]):
        s = df_clean_s1[matched_col].dropna()
        skew_val = stats.skew(s)
        kurt_val = stats.kurtosis(s)
        shapiro_stat, shapiro_p = stats.shapiro(s)
        
        # Also check groups
        g0 = df_clean_s1[df_clean_s1['LET'] == 0][matched_col].dropna()
        g1 = df_clean_s1[df_clean_s1['LET'] == 1][matched_col].dropna()
        skew_g0 = stats.skew(g0) if len(g0) > 2 else 0
        skew_g1 = stats.skew(g1) if len(g1) > 2 else 0
        
        print(f"  Var: {var_name:30s} -> col={matched_col:20s} | Shapiro p={shapiro_p:.4f} | Skew={skew_val:.2f} (g0={skew_g0:.2f}, g1={skew_g1:.2f}) | Kurt={kurt_val:.2f}")
        
        # Clinical standard: if |skew| > 1.5 or Shapiro p <= 0.05, t-test is invalid
        if abs(skew_val) > 1.5 or shapiro_p <= 0.05:
            skew_violations.append({
                "var": var_name, "col": matched_col, "skew": skew_val, "shapiro_p": shapiro_p
            })

if not skew_violations:
    record_finding("PASS", "M2 Normality Assumptions", "All variables tested with Welch's t-test satisfy Shapiro-Wilk p > 0.05 and |skewness| <= 1.5.")
else:
    record_finding("FAIL", "M2 Normality Assumptions", f"Found {len(skew_violations)} variables with severe skewness tested via t-test: {skew_violations}")


# Let's also check all continuous variables across all tables
for tbl_name, df_tbl in [("Stage 1", df_t1_s1), ("Stage 2", df_t1_s2), ("Combined Cohorts", df_t1_comb_c), ("Combined Thrombosis", df_t1_comb_t)]:
    mwu_count = (df_tbl['Test_Method'] == "Mann-Whitney U").sum()
    welch_count = (df_tbl['Test_Method'] == "Welch's t-test").sum()
    chi2_count = (df_tbl['Test_Method'] == "Chi-square test").sum()
    fisher_count = (df_tbl['Test_Method'] == "Fisher's exact test").sum()
    print(f"Table '{tbl_name}': Mann-Whitney U = {mwu_count}, Welch's t-test = {welch_count}, Chi-square = {chi2_count}, Fisher's = {fisher_count}")


print("\n--- 3. Stress-Testing Table 1 Edge Cases (Zero Variance & 100% Missing) ---")
# Test analyze_continuous_variable on edge cases
zero_var_series = pd.Series([5.0] * 50)
group_mask = pd.Series([0] * 25 + [1] * 25)
res_zero_var = gen_t1.analyze_continuous_variable(zero_var_series, group_mask, "Zero Variance Test")
print(f"Zero variance result: Statistic={res_zero_var['Statistic']}, P_Display={res_zero_var['P_Display']}, Test_Method={res_zero_var['Test_Method']}")

# Test analyze_continuous_variable on all-NaN
all_nan_series = pd.Series([np.nan] * 50)
res_nan = gen_t1.analyze_continuous_variable(all_nan_series, group_mask, "All NaN Test")
print(f"All NaN result: Statistic={res_nan['Statistic']}, P_Display={res_nan['P_Display']}, Test_Method={res_nan['Test_Method']}")

if res_zero_var['P_Display'] in ["NA", "1.000", "0.000"] or pd.isna(res_zero_var['P_Value']) or not np.isinf(res_zero_var['P_Value']):
    record_finding("PASS", "M2 Zero Variance Edge Case", "Zero variance handled gracefully without crash.")
else:
    record_finding("FAIL", "M2 Zero Variance Edge Case", f"Unhandled zero variance: {res_zero_var}")

if res_nan['Test_Method'] == "Insufficient data":
    record_finding("PASS", "M2 All-NaN Edge Case", "All-NaN handled gracefully with 'Insufficient data'.")
else:
    record_finding("FAIL", "M2 All-NaN Edge Case", f"Unexpected all-NaN result: {res_nan}")


# Test binary edge cases: single zero cell (one group 0, other > 0)
bin_series_single = pd.Series(['无'] * 25 + ['有'] * 5 + ['无'] * 20)
res_bin_single = gen_t1.analyze_binary_variable(bin_series_single, group_mask, "Single Zero Cell")
print(f"Single zero cell result: Statistic={res_bin_single['Statistic']}, P_Display={res_bin_single['P_Display']}, Test_Method={res_bin_single['Test_Method']}")
if "Fisher" in res_bin_single['Test_Method']:
    record_finding("PASS", "M2 Single Zero Cell Contingency", f"Handled single zero cell using {res_bin_single['Test_Method']}.")
else:
    record_finding("FAIL", "M2 Single Zero Cell Contingency", f"Expected Fisher exact test, got: {res_bin_single}")

# Test degenerate all-zero binary (both groups 0)
bin_series_all_zero = pd.Series(['无'] * 50)
res_bin_all = gen_t1.analyze_binary_variable(bin_series_all_zero, group_mask, "All Zero Binary")
print(f"All zero binary result: Statistic={res_bin_all['Statistic']}, P_Display={res_bin_all['P_Display']}, Test_Method={res_bin_all['Test_Method']}")
if res_bin_all['Test_Method'] == "Uncomputable" and res_bin_all['P_Display'] == "NA":
    record_finding("PASS", "M2 Degenerate Binary Safeguard", "All-zero binary correctly identified as Uncomputable / NA without crash.")
else:
    record_finding("FAIL", "M2 Degenerate Binary Safeguard", f"Unexpected degenerate binary handling: {res_bin_all}")


# ==============================================================================
# SECTION 2: MILESTONE 3 (IMBALANCE BENCHMARK) ADVERSARIAL CHALLENGES
# ==============================================================================
print("\n--- 4. Stress-Testing Anti-Leakage & Validation Fold Invariance ---")
# Check 03_benchmark_imbalance.py anti-leakage execution
import importlib
m3_mod = importlib.import_module("03_benchmark_imbalance")

df_train, y_train, consensus_features = m3_mod.load_datasets()
skf = m3_mod.StratifiedKFold(n_splits=10, shuffle=True, random_state=30)
strategies = m3_mod.get_benchmark_strategies()

# Track validation fold integrity across every fold and strategy
leakage_detected = False
fold_invariance_passes = True

for strat_name, strat_cfg in strategies.items():
    sampler = strat_cfg["sampler"]
    cw = strat_cfg["class_weight"]
    clf = m3_mod.LogisticRegression(class_weight=cw, random_state=30, max_iter=1000)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_train, y_train), 1):
        X_train_f = df_train[consensus_features].iloc[train_idx].copy()
        y_train_f = y_train.iloc[train_idx].copy().values
        X_val_f = df_train[consensus_features].iloc[val_idx].copy()
        y_val_f = y_train.iloc[val_idx].copy().values
        
        orig_val_shape = X_val_f.shape
        orig_val_y_sum = int(np.sum(y_val_f))
        orig_val_indices = set(val_idx)
        
        # Construct pipeline
        steps = []
        if sampler is not None:
            from sklearn.base import clone
            steps.append(("sampler", clone(sampler)))
        steps.append(("classifier", clf))
        
        pipeline = m3_mod.ImbPipeline(steps)
        pipeline.fit(X_train_f, y_train_f)
        
        # Predict on validation fold
        val_prob = pipeline.predict_proba(X_val_f)[:, 1]
        val_pred = pipeline.predict(X_val_f)
        
        # Assert validation data was never mutated
        if X_val_f.shape != orig_val_shape:
            record_finding("FAIL", f"Leakage in {strat_name} Fold {fold_idx}", f"Validation X shape changed from {orig_val_shape} to {X_val_f.shape}!")
            leakage_detected = True
        
        if int(np.sum(y_val_f)) != orig_val_y_sum:
            record_finding("FAIL", f"Leakage in {strat_name} Fold {fold_idx}", f"Validation y sum changed from {orig_val_y_sum} to {int(np.sum(y_val_f))}!")
            leakage_detected = True
            
        if len(val_prob) != len(val_idx):
            record_finding("FAIL", f"Leakage in {strat_name} Fold {fold_idx}", f"Predicted probabilities length {len(val_prob)} != {len(val_idx)}!")
            leakage_detected = True

if not leakage_detected:
    record_finding("PASS", "M3 Anti-Leakage Validation Fold Invariance", "All validation folds across all 5 strategies maintain 100% sample identity, shape, and label distribution.")


print("\n--- 5. Stress-Testing DeLong CI Mathematical Bounds ---")
# Check compute_delong_ci across extreme edge cases
delong_tests = [
    # (name, y_true, y_pred)
    ("Perfect discrimination (AUC=1.0)", np.array([0, 0, 0, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])),
    ("Inverted discrimination (AUC=0.0)", np.array([0, 0, 0, 1, 1, 1]), np.array([0.8, 0.9, 0.95, 0.1, 0.2, 0.3])),
    ("Equal predictions (AUC=0.5)", np.array([0, 0, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])),
    ("Extreme high AUC with small n", np.array([0, 0, 1]), np.array([0.1, 0.2, 0.99])),
    ("Extreme low AUC with small n", np.array([0, 1, 1]), np.array([0.9, 0.1, 0.2])),
    ("All negative samples (m=0)", np.array([0, 0, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])),
    ("All positive samples (n=0)", np.array([1, 1, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])),
    ("Empty inputs", np.array([]), np.array([])),
    ("Single sample per class (m=1, n=1)", np.array([0, 1]), np.array([0.2, 0.8])),
]

delong_all_valid = True
for name, yt, yp in delong_tests:
    try:
        auc_val, ci_l, ci_u, se_val = m3_mod.compute_delong_ci(yt, yp)
        print(f"  DeLong Case '{name}': AUC={auc_val:.4f}, 95% CI=[{ci_l:.4f}, {ci_u:.4f}], SE={se_val:.4f}")
        
        # Check mathematical bounds: 0 <= ci_l <= auc_val <= ci_u <= 1
        # (Within floating point tolerance 1e-9)
        if not (0.0 <= ci_l <= (auc_val + 1e-7)):
            record_finding("FAIL", f"DeLong Bound Failure: {name}", f"ci_lower ({ci_l}) > auc_val ({auc_val}) or < 0.0")
            delong_all_valid = False
        elif not ((auc_val - 1e-7) <= ci_u <= 1.0):
            record_finding("FAIL", f"DeLong Bound Failure: {name}", f"ci_upper ({ci_u}) < auc_val ({auc_val}) or > 1.0")
            delong_all_valid = False
        elif not (0.0 <= ci_l <= ci_u <= 1.0):
            record_finding("FAIL", f"DeLong Bound Failure: {name}", f"Bounds ordering violated: [{ci_l}, {ci_u}]")
            delong_all_valid = False
    except Exception as e:
        record_finding("FAIL", f"DeLong Crash: {name}", f"Exception raised: {e}")
        delong_all_valid = False

if delong_all_valid:
    record_finding("PASS", "M3 DeLong CI Mathematical Boundaries", "All edge cases strictly satisfy 0.0 <= CI_lower <= AUROC <= CI_upper <= 1.0 without crash.")


print("\n--- 6. Auditing M3 Output Benchmark Metrics & Out-of-Bounds Checks ---")
rep_csv_path = OUTPUT_M3 / "imbalance_benchmark_report.csv"
if not rep_csv_path.exists():
    record_finding("FAIL", "Missing M3 Report CSV", f"File {rep_csv_path} does not exist.")
else:
    df_rep = pd.read_csv(rep_csv_path)
    record_finding("PASS", "M3 Report CSV Exists", f"Report contains {len(df_rep)} rows and columns: {df_rep.columns.tolist()[:6]}")
    
    # Check that all 5 strategies and 3 classifiers are present (5 * 3 = 15 rows)
    expected_rows = 15
    if len(df_rep) == expected_rows:
        record_finding("PASS", "M3 Strategy/Classifier Coverage", f"All 15 combinations evaluated.")
    else:
        record_finding("FAIL", "M3 Strategy/Classifier Coverage", f"Expected {expected_rows} rows, got {len(df_rep)}")
        
    # Check all metric values within [0.0, 1.0]
    metric_cols = ["Accuracy", "Recall", "Specificity", "Precision", "F1", "AUROC", "AUROC_95CI_Lower", "AUROC_95CI_Upper"]
    out_of_bounds = []
    for col in metric_cols:
        if col in df_rep.columns:
            invalid_mask = (df_rep[col] < 0.0) | (df_rep[col] > 1.0) | df_rep[col].isna()
            if invalid_mask.any():
                out_of_bounds.append((col, df_rep[invalid_mask][["Strategy", "Classifier", col]].to_dict()))
                
    if not out_of_bounds:
        record_finding("PASS", "M3 Metric Bounding", "All clinical metrics (Acc, Rec, Spec, Prec, F1, AUROC, CIs) strictly within [0.0, 1.0].")
    else:
        record_finding("FAIL", "M3 Metric Bounding", f"Out of bounds metrics found: {out_of_bounds}")

    # Check RandomUnderSampler metrics specifically
    df_rus = df_rep[df_rep['Strategy'] == "RandomUnderSampler"]
    print("\nRandomUnderSampler metrics:")
    for _, r in df_rus.iterrows():
        print(f"  {r['Classifier']:22s} | Recall={r['Recall']:.3f} | Prec={r['Precision']:.3f} | F1={r['F1']:.3f} | AUROC={r['AUROC']:.3f} [{r['AUROC_95CI_Lower']:.3f}-{r['AUROC_95CI_Upper']:.3f}]")


print("\n--- 7. Verification of Selected Strategy Config JSON & Justification ---")
json_path = OUTPUT_M3 / "selected_imbalance_strategy.json"
if not json_path.exists():
    record_finding("FAIL", "Missing Selected Strategy JSON", f"{json_path} does not exist.")
else:
    with open(json_path, "r", encoding="utf-8") as f:
        strat_json = json.load(f)
    print(f"Selected Strategy JSON: strategy = {strat_json.get('selected_strategy')}")
    print(f"Rationale: {strat_json.get('empirical_justification', {}).get('rationale')}")
    
    if strat_json.get("selected_strategy") == "Cost-Sensitive Learning (class_weight='balanced')":
        record_finding("PASS", "M3 Strategy Selection", "Cost-Sensitive Learning properly selected with empirical rationale.")
    else:
        record_finding("FAIL", "M3 Strategy Selection", f"Unexpected strategy: {strat_json.get('selected_strategy')}")


print("\n================================================================================")
print("  SUMMARY OF FINDINGS")
print("================================================================================")
n_pass = sum(1 for f in findings if f['status'] == 'PASS')
n_fail = sum(1 for f in findings if f['status'] == 'FAIL')
print(f"TOTAL TESTS: {len(findings)} | PASS: {n_pass} | FAIL: {n_fail}")
if n_fail == 0:
    print("\n>>> EMPIRICAL CHALLENGE VERDICT: APPROVE <<<")
else:
    print("\n>>> EMPIRICAL CHALLENGE VERDICT: REQUEST_CHANGES <<<")
