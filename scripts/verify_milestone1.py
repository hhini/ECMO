# -*- coding: utf-8 -*-
"""
Verification Script for Milestone 1:
Data Recalibration, Pruning, MICE Imputation, Baseline Balance & Table 1 Generation
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
STAGE1_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
TABLE_DIR = PROJECT_ROOT / "output" / "02_table1_baseline"

def parse_num(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    if s in ['无', '否', '未查', '未测', 'nan', '', '未手术']: return np.nan
    m = re.search(r'([0-9]+\.?[0-9]*)', s)
    return float(m.group(1)) if m else np.nan

def verify_milestone1():
    print("=" * 80)
    print("STARTING MILESTONE 1 INDEPENDENT VERIFICATION AUDIT")
    print("=" * 80)
    all_passed = True
    
    # Check 1: Ingest datasets
    df_s1 = pd.read_csv(STAGE1_PATH)
    df_s2 = pd.read_csv(STAGE2_PATH)
    print(f"[DATA LOAD] Stage 1 Shape: {df_s1.shape}, Stage 2 Shape: {df_s2.shape}")
    assert len(df_s1) == 266, f"Stage 1 N != 266: {len(df_s1)}"
    assert len(df_s2) == 106, f"Stage 2 N != 106: {len(df_s2)}"

    # Check 2 (a): Age Representation in Data & Table 1
    print("\n--- CHECK 2A: Continuous Age in Modeling Dataset ---")
    age_s1 = df_s1['年龄']
    is_float = np.issubdtype(age_s1.dtype, np.floating)
    print(f"Age dtype: {age_s1.dtype} (is float: {is_float})")
    print(f"Age range: min={age_s1.min()}, max={age_s1.max()}, mean={age_s1.mean():.2f}")
    if is_float and age_s1.isna().sum() == 0 and age_s1.min() >= 18.0:
        print("PASS: Age is strictly continuous float64 without missing or pediatric values.")
    else:
        print("FAIL: Age continuous format violation!")
        all_passed = False

    print("\n--- CHECK 2B: Dual-Reported Age in Table 1 ---")
    t1_comb = pd.read_csv(TABLE_DIR / "table1_combined_cohorts.csv")
    age_rows = t1_comb[t1_comb['Variable'].str.contains('年龄|Age', na=False)]
    print(f"Found {len(age_rows)} Age-related rows in table1_combined_cohorts.csv:")
    for _, row in age_rows.iterrows():
        p_val_display = row.get('P-value', row.get('P_Display', 'NA'))
        print(f"  - {row['Variable']}: Stage 1='{row['Stage 1 (n=266)']}', Stage 2='{row['Stage 2 (n=106)']}', P='{p_val_display}'")
    
    has_mean_sd = any('均值' in str(v) or 'Mean' in str(v) for v in age_rows['Variable'])
    has_median_iqr = any('中位数' in str(v) or 'Median' in str(v) for v in age_rows['Variable'])
    has_stratified = any('年龄分段' in str(v) or 'stratification' in str(v) for v in age_rows['Variable'])
    
    if has_mean_sd and has_median_iqr and has_stratified:
        print("PASS: Age is dual-reported in Table 1: Mean ± SD, Median (IQR), and 4 clinical strata.")
    else:
        print("FAIL: Table 1 Age dual-reporting missing required components!")
        all_passed = False

    # Check 3: Feature Pruning (CHD & Cardiogenic Shock)
    print("\n--- CHECK 3: Total Absence of CHD & Cardiogenic Shock ---")
    chd_shock_terms = ['冠心病', '心源性休克', 'Coronary_Heart_Disease', 'Cardiogenic_Shock', 'Underlying_CHD']
    
    # Check data files
    s1_text = df_s1.to_string()
    s2_text = df_s2.to_string()
    data_violations = []
    for term in chd_shock_terms:
        c1 = s1_text.count(term)
        c2 = s2_text.count(term)
        if c1 > 0 or c2 > 0:
            data_violations.append((term, c1, c2))
    
    if not data_violations:
        print("PASS: data2_synthetic.csv and stage2_clean.csv contain ZERO occurrences of CHD and Cardiogenic Shock.")
    else:
        print(f"FAIL: CHD/Shock found in data files: {data_violations}")
        all_passed = False

    # Check table files in output/02_table1_baseline/
    table_files = [f for f in TABLE_DIR.glob("*.csv")]
    table_violations = []
    for tf in table_files:
        df_tbl = pd.read_csv(tf)
        content_str = df_tbl.to_string()
        for term in ['冠心病', '心源性休克', 'Coronary', 'Shock']:
            # Ignore standard text words if unrelated, but check if CHD or Cardiogenic shock exists
            cnt = content_str.count(term)
            if term in ['冠心病', '心源性休克'] and cnt > 0:
                table_violations.append((tf.name, term, cnt))
            elif term == 'Coronary' and content_str.count('Coronary heart disease') > 0:
                table_violations.append((tf.name, term, cnt))
            elif term == 'Shock' and content_str.count('Cardiogenic') > 0:
                table_violations.append((tf.name, term, cnt))

    if not table_violations:
        print("PASS: All tables in output/02_table1_baseline/ have ZERO occurrences of CHD and Cardiogenic Shock.")
    else:
        print(f"FAIL: CHD/Shock found in tables: {table_violations}")
        all_passed = False

    # Check 4: Intraoperative Blood Loss Imputation & Strict Numeric Sanitization
    print("\n--- CHECK 4: Intraoperative Blood Loss Imputation (MICE) & Numeric Hygiene ---")
    surg_hist = df_s1['ECMO前手术史'].fillna('无').astype(str).str.strip()
    has_surgery = surg_hist.map(lambda x: 0 if x in ['无', '否', 'nan', ''] else 1)
    
    # Strict numeric conversion without regex masking
    bl_vals = pd.to_numeric(df_s1['手术失血量/ml'], errors='coerce')
    surg_zero = (has_surgery == 1) & (bl_vals.isna() | (bl_vals <= 0.0))
    surg_bl = bl_vals[has_surgery == 1]
    
    print(f"Surgical patients total: {has_surgery.sum()}")
    print(f"Surgical patients with <= 0 / missing blood loss (strict numeric): {surg_zero.sum()}")
    print(f"Surgical blood loss distribution: Min={surg_bl.min()} mL, Median={surg_bl.median():.1f} mL, Mean={surg_bl.mean():.1f} mL, Max={surg_bl.max()} mL")
    
    # Also verify sedation duration has no alphanumeric traps like '1d'
    sed_traps = df_s1['镇静药物使用时长/d'].astype(str).str.contains(r'^\s*\d+(?:\.\d+)?[a-zA-Z]+\s*$', regex=True).sum()
    print(f"Sedation duration alphanumeric traps count: {sed_traps}")
    
    if surg_zero.sum() == 0 and surg_bl.min() >= 50.0 and sed_traps == 0:
        print("PASS: Zero unrecorded blood loss values remain among surgical patients under strict pd.to_numeric (bounded physiologically >= 50 mL).")
    else:
        print(f"FAIL: Unrecorded blood loss or unit traps remain! Zeros count: {surg_zero.sum()}, traps: {sed_traps}")
        all_passed = False

    # Check 5: Target Endpoint Harmonization
    print("\n--- CHECK 5: Target Endpoint Harmonization (LET) ---")
    s1_pos = int(df_s1['是否下肢血栓'].sum())
    s2_pos = int(df_s2['是否下肢血栓'].sum())
    ct_let = [[s1_pos, s2_pos], [len(df_s1) - s1_pos, len(df_s2) - s2_pos]]
    chi2_let, p_let, _, _ = stats.chi2_contingency(ct_let, correction=False)
    _, p_let_fisher = stats.fisher_exact(ct_let)
    
    print(f"Stage 1 LET Positive: {s1_pos}/{len(df_s1)} ({s1_pos/len(df_s1)*100:.2f}%)")
    print(f"Stage 2 LET Positive: {s2_pos}/{len(df_s2)} ({s2_pos/len(df_s2)*100:.2f}%)")
    print(f"LET 2x2 Contingency Matrix: {ct_let}")
    print(f"Chi-Square Test: Chi2={chi2_let:.4f}, P={p_let:.4f} (Fisher exact P={p_let_fisher:.4f})")
    
    if s1_pos == 45 and p_let >= 0.05:
        print("PASS: Stage 1 LET calibrated to 45/266 (16.92%), cross-cohort divergence eliminated (P >= 0.05).")
    else:
        print(f"FAIL: LET endpoint balance not achieved! S1_pos={s1_pos}, P={p_let:.4f}")
        all_passed = False

    # Check 6: Baseline Demographics & Comorbidities Balance
    print("\n--- CHECK 6: Baseline Demographics & Comorbidities Cross-Cohort Balance ---")
    baseline_vars = [
        ("Age Continuous (t-test)", stats.ttest_ind(df_s1['年龄'], df_s2['年龄'], equal_var=False)[1]),
        ("Age Continuous (MWU)", stats.mannwhitneyu(df_s1['年龄'], df_s2['年龄'])[1]),
        ("Sex (Male %)", stats.chi2_contingency([
            [(df_s1['性别']=='男').sum(), (df_s2['性别']=='男').sum()],
            [(df_s1['性别']=='女').sum(), (df_s2['性别']=='女').sum()]
        ], correction=False)[1]),
        ("BMI", stats.mannwhitneyu(df_s1['BMI'].apply(parse_num).dropna(), df_s2['BMI'].apply(parse_num).dropna())[1]),
        ("Smoking History", stats.chi2_contingency([
            [df_s1['吸烟'].astype(str).str.contains('吸烟|是|1').sum(), df_s2['吸烟'].astype(str).str.contains('吸烟|是|1').sum()],
            [(~df_s1['吸烟'].astype(str).str.contains('吸烟|是|1')).sum(), (~df_s2['吸烟'].astype(str).str.contains('吸烟|是|1')).sum()]
        ])[1]),
        ("Drinking History", stats.chi2_contingency([
            [df_s1['饮酒'].astype(str).str.contains('饮酒|有|是|1').sum(), df_s2['饮酒'].astype(str).str.contains('饮酒|有|是|1').sum()],
            [(~df_s1['饮酒'].astype(str).str.contains('饮酒|有|是|1')).sum(), (~df_s2['饮酒'].astype(str).str.contains('饮酒|有|是|1')).sum()]
        ])[1]),
        ("Hypertension", stats.chi2_contingency([
            [df_s1['基础疾病'].astype(str).str.contains('高血压').sum(), df_s2['基础疾病'].astype(str).str.contains('高血压').sum()],
            [(~df_s1['基础疾病'].astype(str).str.contains('高血压')).sum(), (~df_s2['基础疾病'].astype(str).str.contains('高血压')).sum()]
        ])[1]),
        ("Diabetes", stats.chi2_contingency([
            [df_s1['基础疾病'].astype(str).str.contains('糖尿病|T2DM|DM').sum(), df_s2['基础疾病'].astype(str).str.contains('糖尿病|T2DM|DM').sum()],
            [(~df_s1['基础疾病'].astype(str).str.contains('糖尿病|T2DM|DM')).sum(), (~df_s2['基础疾病'].astype(str).str.contains('糖尿病|T2DM|DM')).sum()]
        ])[1]),
        ("Cerebrovascular Disease", stats.fisher_exact([
            [df_s1['基础疾病'].astype(str).str.contains('脑梗|脑出血|中风').sum(), df_s2['基础疾病'].astype(str).str.contains('脑梗|脑出血|中风').sum()],
            [(~df_s1['基础疾病'].astype(str).str.contains('脑梗|脑出血|中风')).sum(), (~df_s2['基础疾病'].astype(str).str.contains('脑梗|脑出血|中风')).sum()]
        ])[1]),
    ]
    
    # Age 4-categories Chi2
    def age_cat(x):
        if x < 40: return '18-40'
        elif x < 60: return '40-60'
        elif x < 70: return '60-70'
        else: return '>=70'
    c1 = df_s1['年龄'].apply(age_cat).value_counts()
    c2 = df_s2['年龄'].apply(age_cat).value_counts()
    cats = ['18-40', '40-60', '60-70', '>=70']
    ct_age = [[c1.get(c, 0), c2.get(c, 0)] for c in cats]
    p_age_cat = stats.chi2_contingency(ct_age)[1]
    baseline_vars.insert(2, ("Age 4 Categories (Chi2)", p_age_cat))

    for var_name, p_val in baseline_vars:
        status = "PASS" if p_val >= 0.05 else "FAIL"
        print(f"  {var_name:<28}: P = {p_val:.4f} [{status}]")
        if p_val < 0.05:
            all_passed = False

    # Check 7: Artifact Files Presence
    print("\n--- CHECK 7: Artifact Files Existence ---")
    required_artifacts = [
        TABLE_DIR / "table1_stage1.csv",
        TABLE_DIR / "table1_stage2.csv",
        TABLE_DIR / "table1_combined_cohorts.csv",
        TABLE_DIR / "table1_combined_thrombosis.csv",
        TABLE_DIR / "table1_publication_report.md",
        TABLE_DIR / "table1_three_line.docx",
        TABLE_DIR / "Table1_Baseline_Balance_Diagnosis.csv",
        TABLE_DIR / "Table2_Univariate_Cross_Cohort_Consistency.csv",
        TABLE_DIR / "Table2B_Univariate_Overall_Thrombosis.csv"
    ]
    for art in required_artifacts:
        exists = art.exists() and art.stat().st_size > 0
        status = "PASS" if exists else "FAIL"
        sz = f"{art.stat().st_size} bytes" if art.exists() else "MISSING"
        print(f"  {art.name:<45}: [{status}] ({sz})")
        if not exists:
            all_passed = False

    # Check 8: Table 2 & Table 2B Dynamic Subgroup Sample Sizes
    print("\n--- CHECK 8: Table 2 & Table 2B Dynamic Subgroup Sample Sizes ---")
    df_t2 = pd.read_csv(TABLE_DIR / "Table2_Univariate_Cross_Cohort_Consistency.csv")
    df_t2b = pd.read_csv(TABLE_DIR / "Table2B_Univariate_Overall_Thrombosis.csv")
    
    t2_cols = list(df_t2.columns)
    t2b_cols = list(df_t2b.columns)
    
    expected_t2_s1_non = f"训练集 非血栓组 (n={len(df_s1[df_s1['是否下肢血栓']==0])})"
    expected_t2_s1_thr = f"训练集 血栓组 (n={len(df_s1[df_s1['是否下肢血栓']==1])})"
    expected_t2b_non = f"非血栓对照组 (n={len(df_s1[df_s1['是否下肢血栓']==0]) + len(df_s2[df_s2['是否下肢血栓']==0])})"
    expected_t2b_thr = f"下肢深静脉血栓组 (n={len(df_s1[df_s1['是否下肢血栓']==1]) + len(df_s2[df_s2['是否下肢血栓']==1])})"
    
    has_t2_correct = expected_t2_s1_non in t2_cols and expected_t2_s1_thr in t2_cols
    has_t2b_correct = expected_t2b_non in t2b_cols and expected_t2b_thr in t2b_cols
    
    if has_t2_correct and has_t2b_correct:
        print(f"PASS: Table 2 and Table 2B headers accurately reflect dynamic sample sizes ({expected_t2_s1_non}, {expected_t2_s1_thr}, {expected_t2b_non}, {expected_t2b_thr}).")
    else:
        print(f"FAIL: Table 2 or Table 2B headers do not match expected dynamic counts!")
        print(f"  Table 2 columns: {t2_cols}")
        print(f"  Table 2B columns: {t2b_cols}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL MILESTONE 1 VERIFICATION CHECKS PASSED WITH FLYING COLORS!")
    else:
        print("SOME MILESTONE 1 CHECKS FAILED! INVESTIGATE OUTPUT ABOVE.")
    print("=" * 80)

if __name__ == "__main__":
    verify_milestone1()
