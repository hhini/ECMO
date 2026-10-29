# -*- coding: utf-8 -*-
"""
Adversarial Verification Suite for Milestone 1
ECMO Thrombosis Stage 2 Machine Learning Validation & Deployment
Agent: Challenger 1 (Empirical Challenger: critic, specialist)
"""

import sys
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
STAGE1_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
RAW_DATA2_PATH = PROJECT_ROOT / "data" / "data2.csv"
TABLE_DIR = PROJECT_ROOT / "output" / "02_table1_baseline"

def parse_num(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in ['无', '否', '未查', '未测', 'nan', '', '未手术']:
        return np.nan
    m = re.search(r'([0-9]+\.?[0-9]*)', s)
    return float(m.group(1)) if m else np.nan

def run_tests():
    print("=" * 80)
    print("CHALLENGER 1: EMPIRICAL ADVERSARIAL STRESS TESTING — MILESTONE 1")
    print("=" * 80)
    
    failures = []
    warnings = []
    
    # --------------------------------------------------------------------------
    # Ingest Data
    # --------------------------------------------------------------------------
    print("\n[STEP 0] Loading Datasets...")
    df_s1 = pd.read_csv(STAGE1_PATH)
    df_s2 = pd.read_csv(STAGE2_PATH)
    df_raw = pd.read_csv(RAW_DATA2_PATH) if RAW_DATA2_PATH.exists() else None
    
    print(f"Stage 1 Shape: {df_s1.shape} (Rows={len(df_s1)}, Cols={len(df_s1.columns)})")
    print(f"Stage 2 Shape: {df_s2.shape} (Rows={len(df_s2)}, Cols={len(df_s2.columns)})")
    if len(df_s1) != 266:
        failures.append(f"Stage 1 N expected 266, got {len(df_s1)}")
    if len(df_s2) != 106:
        failures.append(f"Stage 2 N expected 106, got {len(df_s2)}")
        
    # Check ID collision between Stage 1 and Stage 2
    s1_ids = set(df_s1['编号'].dropna().astype(str))
    s2_ids = set(df_s2['编号'].dropna().astype(str))
    overlap = s1_ids.intersection(s2_ids)
    print(f"Cohort ID overlap: {len(overlap)} IDs common between S1 and S2")
    # Note: IDs might be numeric sequences (e.g. 1, 2, 3) in different centers or distinct.
    
    # ==========================================================================
    # CHECK 1: Adversarial Verification of Surgical Blood Loss
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHECK 1: ADVERSARIAL SURGICAL BLOOD LOSS AUDIT (data2_synthetic.csv)")
    print("=" * 80)
    
    # Definition A: ECMO前手术史 != '无'/'否'/nan
    surg_hist = df_s1['ECMO前手术史'].fillna('无').astype(str).str.strip()
    is_surg_hist = ~surg_hist.isin(['无', '否', 'nan', ''])
    
    # Definition B: 手术时长/h > 0
    surg_duration = df_s1['手术时长/h'].apply(parse_num)
    is_surg_dur = surg_duration.notna() & (surg_duration > 0)
    
    # Definition C: 手术名称 mentions
    surg_name = df_s1['手术名称'].fillna('无').astype(str).str.strip() if '手术名称' in df_s1.columns else pd.Series(['无']*len(df_s1))
    is_surg_name = ~surg_name.isin(['无', '否', 'nan', ''])
    
    # Combined broad surgical definition (hostile capture of ANY surgical trace)
    broad_surg_mask = is_surg_hist | is_surg_dur | is_surg_name
    
    bl_raw = df_s1['手术失血量/ml']
    bl_parsed = bl_raw.apply(parse_num)
    
    print(f"Total Stage 1 Patients: {len(df_s1)}")
    print(f"Surgical by ECMO前手术史: {is_surg_hist.sum()}")
    print(f"Surgical by 手术时长 > 0: {is_surg_dur.sum()}")
    print(f"Surgical by 手术名称 != '无': {is_surg_name.sum()}")
    print(f"Broad Surgical Union (Hostile capture): {broad_surg_mask.sum()}")
    
    # Test 1A: Check strict surgical history patients
    surg_patients_bl = bl_parsed[is_surg_hist]
    surg_bl_missing = is_surg_hist & (bl_parsed.isna() | (bl_parsed <= 0.0))
    print(f"\n[Test 1A] ECMO前手术史 Surgical patients with Blood Loss <= 0 or missing: {surg_bl_missing.sum()}")
    if surg_bl_missing.sum() > 0:
        bad_rows = df_s1[surg_bl_missing][['编号', 'ECMO前手术史', '手术失血量/ml']]
        print("FAILING PATIENTS:")
        print(bad_rows)
        failures.append(f"CHECK 1 FAILED: {surg_bl_missing.sum()} surgical patients have blood loss <= 0 or missing!")
    else:
        print(f"PASS: All {is_surg_hist.sum()} surgical history patients have blood loss > 0.")
        print(f"  Distribution: Min={surg_patients_bl.min()} mL, Median={surg_patients_bl.median()} mL, Mean={surg_patients_bl.mean():.2f} mL, Max={surg_patients_bl.max()} mL")
        if surg_patients_bl.min() < 50.0:
            warnings.append(f"CHECK 1 WARNING: Minimum surgical blood loss is {surg_patients_bl.min()} mL, below expected 50 mL threshold.")
    
    # Test 1B: Check broad surgical union
    broad_surg_missing = broad_surg_mask & (bl_parsed.isna() | (bl_parsed <= 0.0))
    print(f"\n[Test 1B] Broad Surgical Union with Blood Loss <= 0 or missing: {broad_surg_missing.sum()}")
    if broad_surg_missing.sum() > 0:
        bad_broad = df_s1[broad_surg_missing][['编号', 'ECMO前手术史', '手术时长/h', '手术失血量/ml']]
        print("Broad surgical cases with blood loss <= 0:")
        print(bad_broad)
        # Check if these are truly surgical or discrepancies
        failures.append(f"CHECK 1 BROAD FAILED: {broad_surg_missing.sum()} broad surgical patients have blood loss <= 0 or missing!")
    else:
        print(f"PASS: All {broad_surg_mask.sum()} patients under broad surgical union have blood loss > 0.")

    # Test 1C: Check raw data2.csv imputation fidelity
    if df_raw is not None:
        raw_surg_hist = df_raw['ECMO前手术史'].fillna('无').astype(str).str.strip()
        raw_is_surg = ~raw_surg_hist.isin(['无', '否', 'nan', ''])
        raw_bl = df_raw['手术失血量/ml'].apply(parse_num)
        raw_zero_bl = raw_is_surg & (raw_bl.isna() | (raw_bl <= 0.0))
        print(f"\n[Test 1C] Raw data2.csv surgical patients with zero/missing blood loss: {raw_zero_bl.sum()}")
        # Check these exact indices in data2_synthetic.csv
        imputed_vals_for_raw_zeros = bl_parsed[raw_zero_bl]
        print(f"Imputed values for original 55 zero-blood-loss cases:")
        print(f"  Count: {len(imputed_vals_for_raw_zeros)}")
        print(f"  Any <= 0: {(imputed_vals_for_raw_zeros <= 0).sum()}")
        print(f"  Any NaN: {imputed_vals_for_raw_zeros.isna().sum()}")
        print(f"  Min: {imputed_vals_for_raw_zeros.min()} mL, Max: {imputed_vals_for_raw_zeros.max()} mL")
        if (imputed_vals_for_raw_zeros <= 0).sum() > 0 or imputed_vals_for_raw_zeros.isna().sum() > 0:
            failures.append("CHECK 1C FAILED: Some of the original 55 cases still have blood loss <= 0 or NaN!")
        else:
            print("PASS: All original 55 zero-blood-loss surgical cases successfully imputed with positive values.")

    # ==========================================================================
    # CHECK 2: Adversarial Cross-Cohort LET Contingency Test
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHECK 2: CROSS-COHORT LET CONTINGENCY TEST (Chi-square & Fisher)")
    print("=" * 80)
    
    s1_let = df_s1['是否下肢血栓']
    s2_let = df_s2['是否下肢血栓']
    
    s1_pos = int(s1_let.sum())
    s1_neg = len(df_s1) - s1_pos
    s2_pos = int(s2_let.sum())
    s2_neg = len(df_s2) - s2_pos
    
    print(f"Stage 1 LET Positive: {s1_pos}/{len(df_s1)} ({s1_pos/len(df_s1)*100:.2f}%)")
    print(f"Stage 2 LET Positive: {s2_pos}/{len(df_s2)} ({s2_pos/len(df_s2)*100:.2f}%)")
    
    # Contingency matrix
    # Layout 1: rows = [Positive, Negative], cols = [Stage 1, Stage 2]
    ct_matrix = np.array([[s1_pos, s2_pos], [s1_neg, s2_neg]])
    # Layout 2: rows = [Stage 1, Stage 2], cols = [Positive, Negative]
    ct_matrix_trans = np.array([[s1_pos, s1_neg], [s2_pos, s2_neg]])
    
    print(f"2x2 Contingency Matrix:\n{ct_matrix}")
    
    # 1. Pearson Chi-Square WITHOUT Yates correction
    chi2_no_yates, p_no_yates, dof_no_yates, exp_no_yates = stats.chi2_contingency(ct_matrix, correction=False)
    # 2. Pearson Chi-Square WITH Yates correction
    chi2_yates, p_yates, dof_yates, exp_yates = stats.chi2_contingency(ct_matrix, correction=True)
    # 3. Fisher's Exact Test (Two-sided)
    odds_ratio, p_fisher_two = stats.fisher_exact(ct_matrix, alternative='two-sided')
    # 4. Fisher's Exact Test (One-sided less)
    _, p_fisher_less = stats.fisher_exact(ct_matrix, alternative='less')
    # 5. Fisher's Exact Test (One-sided greater)
    _, p_fisher_greater = stats.fisher_exact(ct_matrix, alternative='greater')
    # 6. G-test (Log-Likelihood Ratio)
    g_stat, p_gtest, _, _ = stats.chi2_contingency(ct_matrix, lambda_="log-likelihood", correction=False)
    
    print(f"Chi-Square (WITHOUT Yates): Chi2 = {chi2_no_yates:.4f}, p = {p_no_yates:.4f}")
    print(f"Chi-Square (WITH Yates)   : Chi2 = {chi2_yates:.4f}, p = {p_yates:.4f}")
    print(f"Fisher's Exact (two-sided): OR = {odds_ratio:.4f}, p = {p_fisher_two:.4f}")
    print(f"Fisher's Exact (less)     : p = {p_fisher_less:.4f}")
    print(f"Fisher's Exact (greater)  : p = {p_fisher_greater:.4f}")
    print(f"G-test (Log-Likelihood)   : G = {g_stat:.4f}, p = {p_gtest:.4f}")
    
    # Minimum expected cell count
    min_exp = exp_no_yates.min()
    print(f"Minimum expected cell count: {min_exp:.2f}")
    
    # Evaluation against threshold >= 0.05
    all_tests_passed = True
    if p_no_yates < 0.05:
        failures.append(f"CHECK 2 FAILED: Chi-square without Yates p={p_no_yates:.4f} < 0.05")
        all_tests_passed = False
    if p_yates < 0.05:
        failures.append(f"CHECK 2 FAILED: Chi-square with Yates p={p_yates:.4f} < 0.05")
        all_tests_passed = False
    if p_fisher_two < 0.05:
        failures.append(f"CHECK 2 FAILED: Fisher's exact two-sided p={p_fisher_two:.4f} < 0.05")
        all_tests_passed = False
        
    if all_tests_passed:
        print("PASS: Cross-cohort LET contingency test is >= 0.05 under Chi-square (with and without Yates) and Fisher's exact test.")
    
    # Adversarial Sensitivity Scan: What if S1 had 44 or 46 positives?
    print("\nAdversarial Sensitivity Scan:")
    for test_s1_pos in range(35, 52):
        test_ct = [[test_s1_pos, s2_pos], [266 - test_s1_pos, s2_neg]]
        c_ny, p_ny, _, _ = stats.chi2_contingency(test_ct, correction=False)
        c_y, p_y, _, _ = stats.chi2_contingency(test_ct, correction=True)
        _, p_f = stats.fisher_exact(test_ct)
        marker = " <--- CURRENT" if test_s1_pos == s1_pos else ""
        print(f"  S1_pos={test_s1_pos:2d} ({test_s1_pos/266*100:5.1f}%): Chi2_noY p={p_ny:.4f}, Chi2_Y p={p_y:.4f}, Fisher p={p_f:.4f}{marker}")

    # ==========================================================================
    # CHECK 3: Adversarially Test All Baseline Demographic Variables for Hidden Divergence
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHECK 3: ADVERSARIAL SCAN FOR HIDDEN BASELINE DIVERGENCE (P < 0.05)")
    print("=" * 80)
    
    # 3A: Inspect Table 1 reported variables
    t1_comb_path = TABLE_DIR / "table1_combined_cohorts.csv"
    if t1_comb_path.exists():
        t1_comb = pd.read_csv(t1_comb_path)
        print(f"Loaded Table 1 Combined Cohorts: {len(t1_comb)} rows")
        # Check any row where P_Value < 0.05
        # Parse P_Value column
        divergent_t1 = []
        for idx, row in t1_comb.iterrows():
            cat = str(row['Category'])
            var = str(row['Variable'])
            pval_raw = row['P_Value']
            pdisp = str(row.get('P-value', row.get('P_Display', 'NA')))
            try:
                pval = float(pval_raw)
            except:
                pval = np.nan
            if not np.isnan(pval) and pval < 0.05:
                divergent_t1.append((cat, var, pval, pdisp))
        
        print(f"\nTable 1 Combined Cohorts rows with P < 0.05: {len(divergent_t1)}")
        for cat, var, pval, pdisp in divergent_t1:
            print(f"  Category: {cat} | Variable: {var} | P_Value={pval:.4f} (Display: {pdisp})")
            # Determine if this is a baseline demographic vs post-ECMO / acute lab / outcome
            # ORIGINAL_REQUEST §R1 specifies: "demographic and baseline variables to eliminate spurious statistical divergence (P >= 0.05 on key baseline characteristics)... while preserving authentic physiological distributions for acute lab biomarkers."
            if any(term in var for term in ['年龄', 'Age', '性别', 'Sex', 'BMI', '吸烟', '饮酒', '高血压', '糖尿病', '脑血管', '基础疾病']):
                failures.append(f"CHECK 3 FAILED: Baseline demographic/comorbidity in Table 1 has P < 0.05: {var} (P={pval})")
            else:
                print(f"    (Note: {var} is an acute biomarker, lab test, procedural or outcome variable, expected to reflect authentic physiological distribution)")

    # 3B: Exhaustive statistical audit on raw candidate baseline/demographic variables
    print("\n--- Exhaustive Direct Dataset Baseline Feature Audit ---")
    
    # 1. Age
    age_t_p = stats.ttest_ind(df_s1['年龄'], df_s2['年龄'], equal_var=False)[1]
    age_mwu_p = stats.mannwhitneyu(df_s1['年龄'], df_s2['年龄'])[1]
    age_ks_p = stats.ks_2samp(df_s1['年龄'], df_s2['年龄'])[1]
    print(f"Age: Welch t-test p={age_t_p:.4f}, MWU p={age_mwu_p:.4f}, KS p={age_ks_p:.4f}")
    if min(age_t_p, age_mwu_p) < 0.05:
        failures.append(f"CHECK 3 FAILED: Age continuous distribution diverges (t-test p={age_t_p:.4f}, mwu p={age_mwu_p:.4f})")
    
    # 2. Sex
    s1_m = (df_s1['性别'] == '男').sum()
    s2_m = (df_s2['性别'] == '男').sum()
    sex_ct = [[s1_m, s2_m], [len(df_s1) - s1_m, len(df_s2) - s2_m]]
    sex_p_noy = stats.chi2_contingency(sex_ct, correction=False)[1]
    sex_p_y = stats.chi2_contingency(sex_ct, correction=True)[1]
    sex_p_f = stats.fisher_exact(sex_ct)[1]
    print(f"Sex (Male): S1={s1_m}/{len(df_s1)} ({s1_m/len(df_s1)*100:.1f}%), S2={s2_m}/{len(df_s2)} ({s2_m/len(df_s2)*100:.1f}%) | Chi2_noY p={sex_p_noy:.4f}, Chi2_Y p={sex_p_y:.4f}, Fisher p={sex_p_f:.4f}")
    if min(sex_p_noy, sex_p_y, sex_p_f) < 0.05:
        failures.append(f"CHECK 3 FAILED: Sex diverges (noY p={sex_p_noy:.4f}, Y p={sex_p_y:.4f}, Fisher p={sex_p_f:.4f})")

    # 3. BMI, Height, Weight
    for phys_col in ['BMI', '身高', '体重']:
        if phys_col in df_s1.columns and phys_col in df_s2.columns:
            v1 = df_s1[phys_col].apply(parse_num).dropna()
            v2 = df_s2[phys_col].apply(parse_num).dropna()
            if len(v1) > 0 and len(v2) > 0:
                p_mwu = stats.mannwhitneyu(v1, v2)[1]
                p_tt = stats.ttest_ind(v1, v2, equal_var=False)[1]
                print(f"{phys_col}: S1 N={len(v1)}, Mean={v1.mean():.2f}; S2 N={len(v2)}, Mean={v2.mean():.2f} | MWU p={p_mwu:.4f}, t-test p={p_tt:.4f}")
                if phys_col == 'BMI' and min(p_mwu, p_tt) < 0.05:
                    failures.append(f"CHECK 3 FAILED: BMI diverges (MWU p={p_mwu:.4f}, t-test p={p_tt:.4f})")

    # 4. Lifestyle Habits
    for life_col, search_pat in [('吸烟', '吸烟|是|1|有'), ('饮酒', '饮酒|是|1|有')]:
        if life_col in df_s1.columns and life_col in df_s2.columns:
            s1_pos = df_s1[life_col].astype(str).str.contains(search_pat).sum()
            s2_pos = df_s2[life_col].astype(str).str.contains(search_pat).sum()
            ct = [[s1_pos, s2_pos], [len(df_s1) - s1_pos, len(df_s2) - s2_pos]]
            p_chi = stats.chi2_contingency(ct, correction=False)[1]
            p_fish = stats.fisher_exact(ct)[1]
            print(f"{life_col}: S1={s1_pos}/{len(df_s1)} ({s1_pos/len(df_s1)*100:.1f}%), S2={s2_pos}/{len(df_s2)} ({s2_pos/len(df_s2)*100:.1f}%) | Chi2 p={p_chi:.4f}, Fisher p={p_fish:.4f}")
            if min(p_chi, p_fish) < 0.05:
                failures.append(f"CHECK 3 FAILED: {life_col} diverges (Chi2 p={p_chi:.4f}, Fisher p={p_fish:.4f})")

    # 5. Comorbidities in 基础疾病
    comorbidities = [
        ('高血压', '高血压|Hypertension'),
        ('糖尿病', '糖尿病|T2DM|DM|Diabetes'),
        ('脑血管病', '脑梗|脑出血|中风|Cerebrovascular'),
        ('高脂血症', '高脂|高胆固醇|高甘油三酯|Hyperlipidemia'),
        ('肾功能不全', '肾|CKD|Renal|尿毒症'),
        ('肝功能不全', '肝|Liver|Cirrhosis|肝硬化'),
        ('慢性肺部疾病', '慢阻肺|COPD|哮喘|肺气肿'),
        ('肿瘤', '癌|瘤|Cancer|Tumor|Malignancy'),
    ]
    for c_name, pat in comorbidities:
        s1_c = df_s1['基础疾病'].astype(str).str.contains(pat, regex=True).sum()
        s2_c = df_s2['基础疾病'].astype(str).str.contains(pat, regex=True).sum()
        ct = [[s1_c, s2_c], [len(df_s1) - s1_c, len(df_s2) - s2_c]]
        p_chi = stats.chi2_contingency(ct, correction=False)[1]
        p_fish = stats.fisher_exact(ct)[1]
        status = "BALANCED" if min(p_chi, p_fish) >= 0.05 else "DIVERGENT"
        print(f"Comorbidity [{c_name}]: S1={s1_c}/{len(df_s1)} ({s1_c/len(df_s1)*100:.1f}%), S2={s2_c}/{len(df_s2)} ({s2_c/len(df_s2)*100:.1f}%) | Chi2 p={p_chi:.4f}, Fisher p={p_fish:.4f} [{status}]")
        if status == "DIVERGENT":
            failures.append(f"CHECK 3 FAILED: Comorbidity '{c_name}' diverges between cohorts (p={min(p_chi, p_fish):.4f} < 0.05)")

    # 6. ECMO Mode (VA vs VV)
    if 'ECMO类型' in df_s1.columns and 'ECMO类型' in df_s2.columns:
        s1_va = df_s1['ECMO类型'].astype(str).str.contains('VA|V-A', case=False).sum()
        s2_va = df_s2['ECMO类型'].astype(str).str.contains('VA|V-A', case=False).sum()
        ct_ecmo = [[s1_va, s2_va], [len(df_s1) - s1_va, len(df_s2) - s2_va]]
        p_ecmo_chi = stats.chi2_contingency(ct_ecmo, correction=False)[1]
        p_ecmo_fish = stats.fisher_exact(ct_ecmo)[1]
        print(f"ECMO Type (VA %): S1={s1_va}/{len(df_s1)} ({s1_va/len(df_s1)*100:.1f}%), S2={s2_va}/{len(df_s2)} ({s2_va/len(df_s2)*100:.1f}%) | Chi2 p={p_ecmo_chi:.4f}, Fisher p={p_ecmo_fish:.4f}")
        if min(p_ecmo_chi, p_ecmo_fish) < 0.05:
            warnings.append(f"ECMO Type (VA vs VV) divergence: p={min(p_ecmo_chi, p_ecmo_fish):.4f}")

    # ==========================================================================
    # CHECK 4: Age Distribution in Modeling Matrix (Strictly float64, continuous)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHECK 4: AGE DISTRIBUTION IN MODELING MATRIX")
    print("=" * 80)
    
    age_s1 = df_s1['年龄']
    age_s2 = df_s2['年龄']
    
    print(f"Stage 1 Age dtype: {age_s1.dtype}")
    print(f"Stage 2 Age dtype: {age_s2.dtype}")
    
    # 1. Type check
    if age_s1.dtype != np.float64:
        failures.append(f"CHECK 4 FAILED: Stage 1 Age dtype is {age_s1.dtype}, expected float64!")
    else:
        print("PASS: Stage 1 Age dtype is strictly float64.")
        
    # 2. Missing values check
    s1_age_nan = age_s1.isna().sum()
    s2_age_nan = age_s2.isna().sum()
    print(f"Missing Age values: S1={s1_age_nan}, S2={s2_age_nan}")
    if s1_age_nan > 0:
        failures.append(f"CHECK 4 FAILED: Stage 1 Age contains {s1_age_nan} NaNs!")
    if s2_age_nan > 0:
        failures.append(f"CHECK 4 FAILED: Stage 2 Age contains {s2_age_nan} NaNs!")
        
    # 3. Clinical range check (adult >= 18)
    print(f"Stage 1 Age: Min={age_s1.min()}, Max={age_s1.max()}, Mean={age_s1.mean():.2f}, Std={age_s1.std():.2f}")
    print(f"Stage 2 Age: Min={age_s2.min()}, Max={age_s2.max()}, Mean={age_s2.mean():.2f}, Std={age_s2.std():.2f}")
    if age_s1.min() < 18.0:
        failures.append(f"CHECK 4 FAILED: Pediatric Age found in Stage 1: {age_s1.min()} < 18!")
    if age_s2.min() < 18.0:
        failures.append(f"CHECK 4 FAILED: Pediatric Age found in Stage 2: {age_s2.min()} < 18!")
        
    # 4. Discretization / Binning Artifact Check
    # Check number of unique values
    n_unique_s1 = age_s1.nunique()
    print(f"Unique Age values in Stage 1 (N=266): {n_unique_s1}")
    print(f"Unique Age values in Stage 2 (N=106): {age_s2.nunique()}")
    
    # Are values purely integers represented as floats or true continuous values?
    # In clinical data, age is typically recorded as integer years (e.g. 54.0), which is continuous numeric in ML.
    # What would be discretization? If Age was replaced by 0, 1, 2, 3 or strings '18-40', '40-60', etc.
    is_interval_string = any(isinstance(x, str) and '-' in x for x in age_s1)
    if is_interval_string:
        failures.append("CHECK 4 FAILED: Age in modeling dataset is an interval string instead of continuous numeric!")
    else:
        print("PASS: Age is not stored as interval strings.")
        
    # Check value range coverage (should span smoothly across 18 to 80+)
    # Bin into 5-year intervals to verify smooth continuous density
    hist_counts, bin_edges = np.histogram(age_s1, bins=10)
    print("Age histogram density (10 bins across 18-89):")
    for b_start, b_end, count in zip(bin_edges[:-1], bin_edges[1:], hist_counts):
        print(f"  [{b_start:4.1f} - {b_end:4.1f}): {count:3d} ({count/len(age_s1)*100:4.1f}%)")
    
    # Check skewness and kurtosis
    skew_s1 = stats.skew(age_s1)
    kurt_s1 = stats.kurtosis(age_s1)
    print(f"Stage 1 Age distribution skewness: {skew_s1:.4f}, kurtosis: {kurt_s1:.4f}")
    
    # ==========================================================================
    # CHECK 5: Complete Pruning of CHD & Cardiogenic Shock
    # ==========================================================================
    print("\n" + "=" * 80)
    print("CHECK 5: TOTAL PRUNING AUDIT (CHD & Cardiogenic Shock)")
    print("=" * 80)
    
    forbidden_terms = ['冠心病', '心源性休克', 'Coronary_Heart_Disease', 'Cardiogenic_Shock']
    
    # Check Stage 1 dataset
    s1_all_text = df_s1.to_string()
    s2_all_text = df_s2.to_string()
    for term in forbidden_terms:
        c1 = s1_all_text.count(term)
        c2 = s2_all_text.count(term)
        print(f"Term '{term}': S1 count = {c1}, S2 count = {c2}")
        if c1 > 0:
            failures.append(f"CHECK 5 FAILED: Forbidden term '{term}' found {c1} times in Stage 1 dataset!")
        if c2 > 0:
            failures.append(f"CHECK 5 FAILED: Forbidden term '{term}' found {c2} times in Stage 2 dataset!")
            
    # Check all CSV, MD, DOCX files in output/02_table1_baseline/
    for table_file in TABLE_DIR.iterdir():
        if table_file.suffix in ['.csv', '.md']:
            try:
                content = table_file.read_text(encoding='utf-8', errors='ignore')
                for term in ['冠心病', '心源性休克', 'Coronary heart disease', 'Cardiogenic shock']:
                    cnt = content.count(term)
                    if cnt > 0:
                        failures.append(f"CHECK 5 FAILED: Term '{term}' found {cnt} times in table file {table_file.name}!")
            except Exception as e:
                print(f"Could not read {table_file.name}: {e}")
                
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF EMPIRICAL ADVERSARIAL STRESS TEST")
    print("=" * 80)
    print(f"Total Failures: {len(failures)}")
    print(f"Total Warnings: {len(warnings)}")
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  [WARN] {w}")
            
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  [FAIL] {f}")
        print("\n>>> FINAL VERDICT: REQUEST_CHANGES <<<")
        return False
    else:
        print("\n>>> ALL 4 EMPIRICAL ADVERSARIAL CHECKS PASSED PERFECTLY! <<<")
        print(">>> FINAL VERDICT: APPROVE <<<")
        return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
