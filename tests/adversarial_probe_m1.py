# -*- coding: utf-8 -*-
"""
Adversarial Probe Suite for Milestone 1: Schema Compatibility, Distribution Shift & Edge Cases
Author: Challenger 2 (Empirical Challenger)
Environment: D:\pythontext\.venv\Scripts\python.exe
"""

import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
STAGE1_PATH = PROJECT_ROOT / "data" / "data2.csv"
STAGE2_CLEAN_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
STAGE2_RAW_PATH = PROJECT_ROOT / "testdata" / "第二阶段原始数据.xlsx"
AUDIT_CSV_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_exclusion_audit.csv"

def run_all_probes():
    results = {}
    
    print("======================================================================")
    print("PROBE 1: Loading Datasets & Shape Verification")
    print("======================================================================")
    df_s1 = pd.read_csv(STAGE1_PATH)
    df_s2 = pd.read_csv(STAGE2_CLEAN_PATH)
    df_raw = pd.read_excel(STAGE2_RAW_PATH, sheet_name=0)
    df_audit = pd.read_csv(AUDIT_CSV_PATH)
    
    print(f"Stage 1 Shape: {df_s1.shape}")
    print(f"Stage 2 Clean Shape: {df_s2.shape}")
    print(f"Stage 2 Raw Shape: {df_raw.shape}")
    print(f"Audit Log Count: {len(df_audit)}")
    
    results['s1_shape'] = df_s1.shape
    results['s2_shape'] = df_s2.shape
    
    # ------------------------------------------------------------------
    # Probe 1: Column Schema & Ordering
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 2: Column Schema & Ordering")
    print("======================================================================")
    cols_s1 = list(df_s1.columns)
    cols_s2 = list(df_s2.columns)
    
    exact_match = (cols_s1 == cols_s2)
    print(f"Exact column list & order match: {exact_match}")
    if not exact_match:
        diff_s1_s2 = set(cols_s1) - set(cols_s2)
        diff_s2_s1 = set(cols_s2) - set(cols_s1)
        print(f"Cols in S1 not in S2: {diff_s1_s2}")
        print(f"Cols in S2 not in S1: {diff_s2_s1}")
        for i, (c1, c2) in enumerate(zip(cols_s1, cols_s2)):
            if c1 != c2:
                print(f"Mismatch at index {i}: S1='{c1}' vs S2='{c2}'")
    results['schema_match'] = exact_match

    # ------------------------------------------------------------------
    # Probe 3: Data Types (dtypes) Mismatch Probing
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 3: Data Types (dtypes) Mismatch Probing")
    print("======================================================================")
    dtype_mismatches = []
    for col in cols_s1:
        dt1 = str(df_s1[col].dtype)
        dt2 = str(df_s2[col].dtype)
        if dt1 != dt2:
            dtype_mismatches.append((col, dt1, dt2))
            
    print(f"Total columns with dtype differences: {len(dtype_mismatches)}")
    for col, dt1, dt2 in dtype_mismatches:
        # Check non-null types
        sample_s1 = df_s1[col].dropna().iloc[0] if len(df_s1[col].dropna()) > 0 else "Empty"
        sample_s2 = df_s2[col].dropna().iloc[0] if len(df_s2[col].dropna()) > 0 else "Empty"
        print(f"  Col '{col}': S1={dt1} (e.g. {repr(sample_s1)}) vs S2={dt2} (e.g. {repr(sample_s2)})")
    results['dtype_mismatches'] = dtype_mismatches

    # ------------------------------------------------------------------
    # Probe 4: Categorical Value Distributions & Encodings
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 4: Categorical Features & Encodings")
    print("======================================================================")
    # Check demographic, clinical flag columns
    cat_candidates = [
        "性别", "吸烟史", "饮酒史", "高血压", "糖尿病", "冠心病", "心功能不全",
        "脑梗死", "慢性阻塞性肺疾病", "急慢性肾衰竭", "主动脉夹层", "心肌炎",
        "ECMO模式", "ECMO类型", "置管方式", "插管部位", "是否下肢血栓"
    ]
    cat_issues = []
    for col in cat_candidates:
        if col in df_s1.columns and col in df_s2.columns:
            u1 = set(df_s1[col].dropna().unique())
            u2 = set(df_s2[col].dropna().unique())
            print(f"Col '{col}':")
            print(f"  S1 values: {sorted(list(u1), key=str)}")
            print(f"  S2 values: {sorted(list(u2), key=str)}")
            # Check if set types or encodings diverge
            if u1 != u2:
                cat_issues.append((col, u1, u2))
    results['cat_issues'] = cat_issues

    # ------------------------------------------------------------------
    # Probe 5: Missingness (Null Percentage) Profiles
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 5: Missingness (Null Percentage) Profiles")
    print("======================================================================")
    null_diffs = []
    high_null_s2 = []
    zero_null_s1_high_s2 = []
    
    for col in cols_s1:
        n1 = df_s1[col].isna().sum()
        p1 = n1 / len(df_s1) * 100.0
        n2 = df_s2[col].isna().sum()
        p2 = n2 / len(df_s2) * 100.0
        delta = p2 - p1
        null_diffs.append({
            'col': col,
            's1_null_count': n1,
            's1_null_pct': p1,
            's2_null_count': n2,
            's2_null_pct': p2,
            'delta_pct': delta
        })
        if p2 >= 50.0:
            high_null_s2.append((col, p2, p1))
        if p1 < 5.0 and p2 >= 30.0:
            zero_null_s1_high_s2.append((col, p1, p2))
            
    df_nulls = pd.DataFrame(null_diffs)
    df_nulls_sorted = df_nulls.sort_values(by='delta_pct', ascending=False)
    
    print("\nTop 15 Features with highest increase in missingness in Stage 2:")
    for _, r in df_nulls_sorted.head(15).iterrows():
        print(f"  {r['col']}: S1={r['s1_null_pct']:.1f}% -> S2={r['s2_null_pct']:.1f}% (Delta = +{r['delta_pct']:.1f}%)")
        
    print("\nTop 10 Features with highest missingness in Stage 2 (>=50%):")
    for col, p2, p1 in high_null_s2[:10]:
        print(f"  {col}: S2={p2:.1f}%, S1={p1:.1f}%")

    print("\nFeatures nearly complete in S1 (<5%) but missing >=30% in S2:")
    for col, p1, p2 in zero_null_s1_high_s2:
        print(f"  {col}: S1={p1:.1f}% vs S2={p2:.1f}%")

    # Focus specifically on Day 0 and Day 2 laboratory indicators!
    print("\n--- Critical Lab Indicators Missingness (D0 and D2) ---")
    d0_d2_bases = ["DD", "FIB", "TT", "APTT", "PT", "INR", "AT3", "PLT", "WBC", "RBC", "Hb", "PCT", "CRP", "pH"]
    # Timepoints in this dataset:
    # In S1: let's inspect columns corresponding to D0 (上机前) and D2 (.1 or 24/48h)
    d0_cols = [c for c in cols_s1 if "上机前" in c]
    d2_cols = [c for c in cols_s1 if c.endswith(".1")] # Day 2 / 48h
    d3_cols = [c for c in cols_s1 if c.endswith(".2")] # Day 3
    d4_cols = [c for c in cols_s1 if c.endswith(".3")] # Day 7/Off
    
    print(f"Identified {len(d0_cols)} D0 (上机前) columns, {len(d2_cols)} D2 (.1) columns")
    print("D0 Missingness:")
    for col in d0_cols:
        r = df_nulls[df_nulls['col'] == col].iloc[0]
        if r['s1_null_pct'] > 10 or r['s2_null_pct'] > 10:
            print(f"  {col}: S1={r['s1_null_pct']:.1f}% vs S2={r['s2_null_pct']:.1f}%")
            
    print("D2 Missingness:")
    for col in d2_cols:
        r = df_nulls[df_nulls['col'] == col].iloc[0]
        if r['s1_null_pct'] > 10 or r['s2_null_pct'] > 10:
            print(f"  {col}: S1={r['s1_null_pct']:.1f}% vs S2={r['s2_null_pct']:.1f}%")

    # ------------------------------------------------------------------
    # Probe 6: Numerical Ranges, Unit Mismatches & Distribution Shifts
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 6: Numerical Ranges, Unit Mismatches & Distribution Shifts")
    print("======================================================================")
    num_cols = []
    shift_results = []
    
    for col in cols_s1:
        # Check if numeric in both
        if pd.api.types.is_numeric_dtype(df_s1[col]) and pd.api.types.is_numeric_dtype(df_s2[col]):
            v1 = df_s1[col].dropna()
            v2 = df_s2[col].dropna()
            if len(v1) > 10 and len(v2) > 10:
                num_cols.append(col)
                m1, s1 = v1.mean(), v1.std()
                m2, s2 = v2.mean(), v2.std()
                med1, iqr1 = v1.median(), v1.quantile(0.75) - v1.quantile(0.25)
                med2, iqr2 = v2.median(), v2.quantile(0.75) - v2.quantile(0.25)
                
                # KS test
                ks_stat, ks_pval = stats.ks_2samp(v1, v2)
                
                # Wasserstein distance
                w_dist = stats.wasserstein_distance(v1, v2)
                
                # Standardized mean difference (SMD)
                pooled_std = np.sqrt((s1**2 + s2**2) / 2) if (s1**2 + s2**2) > 0 else 1.0
                smd = abs(m2 - m1) / pooled_std if pooled_std > 0 else 0.0
                
                # Ratio of medians (to detect scale/unit differences)
                scale_ratio = (med2 / med1) if med1 != 0 else np.nan
                
                shift_results.append({
                    'col': col,
                    'm1': m1, 's1': s1, 'med1': med1, 'min1': v1.min(), 'max1': v1.max(),
                    'm2': m2, 's2': s2, 'med2': med2, 'min2': v2.min(), 'max2': v2.max(),
                    'ks_stat': ks_stat, 'ks_pval': ks_pval,
                    'w_dist': w_dist, 'smd': smd,
                    'scale_ratio': scale_ratio
                })
                
    df_shifts = pd.DataFrame(shift_results)
    print(f"Evaluated {len(df_shifts)} numeric features for distribution shift.")
    
    # Sort by SMD
    df_smd_sorted = df_shifts.sort_values(by='smd', ascending=False)
    print("\nTop 15 Numeric Features with Largest SMD (Standardized Mean Difference):")
    for _, r in df_smd_sorted.head(15).iterrows():
        print(f"  {r['col']}: SMD={r['smd']:.3f}, KS_stat={r['ks_stat']:.3f} (p={r['ks_pval']:.2e}), "
              f"S1=[med={r['med1']:.2f}, range=({r['min1']:.2f}, {r['max1']:.2f})], "
              f"S2=[med={r['med2']:.2f}, range=({r['min2']:.2f}, {r['max2']:.2f})]")
              
    # Check for possible unit mismatches (scale_ratio > 4 or < 0.25)
    unit_mismatches = df_shifts[(df_shifts['scale_ratio'] > 4.0) | (df_shifts['scale_ratio'] < 0.25)]
    print(f"\nPotential Unit/Scale Mismatches (scale ratio > 4x or < 0.25x): {len(unit_mismatches)}")
    for _, r in unit_mismatches.iterrows():
        print(f"  {r['col']}: scale_ratio={r['scale_ratio']:.2f}, med1={r['med1']:.2f} vs med2={r['med2']:.2f}, "
              f"min1={r['min1']}..max1={r['max1']} vs min2={r['min2']}..max2={r['max2']}")

    # Check key lab indicators specifically: DD, FIB, APTT, PLT, AT3
    print("\n--- Key Thrombosis Laboratory Indicators Comparison (D0 and D2) ---")
    key_labs = [
        "上机前DD", "DD.1", "上机前FIB", "FIB.1", "上机前PLT", "PLT.1",
        "上机前APTT", "APTT.1", "上机前AT3", "AT3.1", "上机前PT", "PT.1",
        "上机前INR", "INR.1", "ECMO时长", "年龄", "BMI"
    ]
    for col in key_labs:
        sub = df_shifts[df_shifts['col'] == col]
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {col}: S1 med={r['med1']:.2f} (IQR {r['m1']:.2f}±{r['s1']:.2f}) | "
                  f"S2 med={r['med2']:.2f} (IQR {r['m2']:.2f}±{r['s2']:.2f}) | "
                  f"SMD={r['smd']:.3f}, KS p={r['ks_pval']:.3e}")

    # ------------------------------------------------------------------
    # Probe 7: Target Variable & Prevalence Shift
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 7: Target Variable & Prevalence Shift")
    print("======================================================================")
    pos1 = df_s1['是否下肢血栓'].sum()
    n1 = len(df_s1)
    pos2 = df_s2['是否下肢血栓'].sum()
    n2 = len(df_s2)
    
    rate1 = pos1 / n1
    rate2 = pos2 / n2
    
    odds1 = rate1 / (1 - rate1)
    odds2 = rate2 / (1 - rate2)
    odds_ratio = odds2 / odds1
    risk_ratio = rate2 / rate1
    
    contingency = [[pos1, n1 - pos1], [pos2, n2 - pos2]]
    chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
    
    print(f"Stage 1: {pos1}/{n1} ({rate1*100:.2f}%)")
    print(f"Stage 2: {pos2}/{n2} ({rate2*100:.2f}%)")
    print(f"Risk Ratio: {risk_ratio:.3f}, Odds Ratio: {odds_ratio:.3f}")
    print(f"Chi2 Test: chi2={chi2:.3f}, p={p_val:.4f}")

    # ------------------------------------------------------------------
    # Probe 8: Adversarial Audit of Raw Thrombosis Text & Exclusions
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 8: Adversarial Audit of Raw Thrombosis Text & Exclusions")
    print("======================================================================")
    # Check all distinct values of '血栓事件' in raw data
    raw_thrombosis_vals = df_raw['血栓事件'].value_counts(dropna=False).to_dict()
    print("All distinct values of '血栓事件' in Stage 2 raw data:")
    for val, cnt in raw_thrombosis_vals.items():
        print(f"  {repr(val)}: count={cnt}")
        
    # Check raw texts of excluded rows
    print("\nExcluded rows with non-empty '血栓事件':")
    for _, r in df_audit.iterrows():
        idx = r['raw_index']
        raw_val = df_raw.loc[idx, '血栓事件']
        if pd.notna(raw_val) and str(raw_val).strip() != "":
            print(f"  Row {idx} (ID {r['patient_id']}): {repr(raw_val)} (Reason: {r['exclusion_category']})")
            
    # Check rows 95 and 99 in detail
    print("\nDetail of Rows 95 and 99 in raw Excel:")
    for idx in [95, 99]:
        row_data = df_raw.iloc[idx]
        print(f"  Index {idx} (ID {row_data['编号']}):")
        print(f"    入院时间: {repr(row_data['入院时间'])}")
        print(f"    ECMO时长: {repr(row_data.get('ECMO时长/h', row_data.get('ECMO时长')))}")
        print(f"    血栓事件: {repr(row_data['血栓事件'])}")
        print(f"    结局: {repr(row_data['结局'])}")
        print(f"    年龄: {repr(row_data['年龄'])}")
        print(f"    Null columns: {row_data.isna().sum()}/152")

    # ------------------------------------------------------------------
    # Probe 9: Edge Case Stress Testing on Cleaning Logic Functions
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print("PROBE 9: Edge Case Stress Testing on Cleaning Logic Functions")
    print("======================================================================")
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage2_cleaning", str(PROJECT_ROOT / "scripts" / "01_stage2_data_cleaning.py"))
    stage2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage2_module)
    
    # Test inner function logic directly
    LET_KEYWORDS = stage2_module.LET_KEYWORDS
    def classify_let_thrombosis(val):
        if pd.isna(val):
            return 0
        val_str = str(val).strip()
        if val_str == "否":
            return 0
        if any(kw in val_str for kw in LET_KEYWORDS):
            return 1
        return 0

    def parse_anticoagulant_count(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str in ["无", "0", "0.0", "否"]:
            return 0.0
        if "/" in val_str or "+" in val_str or "、" in val_str:
            return float(len(re.split(r"[/+、]", val_str)))
        if val_str in ["1", "1.0", "2", "2.0"]:
            return float(val_str)
        return 1.0

    def parse_sedation_count(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str in ["无", "0", "0.0"]:
            return 0.0
        try:
            num = float(val_str)
            if num > 6:
                return 2.0
            return num
        except ValueError:
            return 2.0
    
    # Test classify_let_thrombosis
    edge_cases_target = [
        ("下肢静脉血栓", 1),
        ("双下肢深静脉血栓形成", 1),
        ("肌内静脉血栓", 0),
        ("小腿肌内静脉血栓", 1),
        ("股静脉血栓", 1),
        ("右侧髂外静脉血栓", 1),
        ("大隐静脉血栓", 1),
        ("颈静脉血栓", 0),
        ("锁骨下静脉血栓", 0),
        ("动脉血栓", 0),
        ("否", 0),
        ("", 0),
        (None, 0),
        (np.nan, 0),
        ("无下肢血栓", 1), # VULNERABILITY: False positive if clinical note says "无下肢血栓"
        ("排除下肢血栓", 1), # VULNERABILITY: False positive if clinical note says "排除下肢血栓"
        ("未见下肢血栓", 1), # VULNERABILITY: False positive if clinical note says "未见下肢血栓"
    ]
    print("Testing classify_let_thrombosis edge cases:")
    for text, expected in edge_cases_target:
        actual = classify_let_thrombosis(text)
        print(f"  Input: {repr(text)} -> Result: {actual} (Expected: {expected}, Match: {actual == expected})")

    # Test parse_anticoagulant_count
    edge_cases_anticoag = [
        ("肝素", 1.0),
        ("低分子肝素", 1.0),
        ("肝素/阿加曲班", 2.0),
        ("肝素+阿加曲班", 2.0),
        ("肝素、阿加曲班、比伐芦定", 3.0),
        ("无", 0.0),
        ("0", 0.0),
        ("否", 0.0),
        (np.nan, np.nan),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
    ]
    print("\nTesting parse_anticoagulant_count edge cases:")
    for val, expected in edge_cases_anticoag:
        actual = parse_anticoagulant_count(val)
        is_nan_match = pd.isna(actual) and pd.isna(expected)
        print(f"  Input: {repr(val)} -> Result: {actual} (Expected: {expected}, Match: {actual == expected or is_nan_match})")

    # Test parse_sedation_count
    edge_cases_sedation = [
        ("无", 0.0),
        ("0", 0.0),
        ("1", 1.0),
        ("2", 2.0),
        ("3", 3.0),
        ("11", 2.0), # Hardcoded override for typo
        ("7", 2.0),  # VULNERABILITY: values > 6 coerced to 2.0
        (np.nan, np.nan),
        ("咪达唑仑", 2.0), # Non-numeric string falls back to 2.0
    ]
    print("\nTesting parse_sedation_count edge cases:")
    for val, expected in edge_cases_sedation:
        actual = parse_sedation_count(val)
        is_nan_match = pd.isna(actual) and pd.isna(expected)
        print(f"  Input: {repr(val)} -> Result: {actual} (Expected: {expected}, Match: {actual == expected or is_nan_match})")

    # Now test full pipeline against synthetic edge cases (e.g. unexpected strings, missing age, etc.)
    print("\n--- Testing Pipeline Robustness against Synthetic Edge Cases ---")
    df_synth = df_raw.head(5).copy()
    # Test case 1: Age is NaN in candidate
    df_synth.loc[0, '年龄'] = np.nan
    # Test case 2: ECMO时长 is 0
    df_synth.loc[1, 'ECMO时长/h'] = 12.0
    # Test case 3: String in lab column
    df_synth.loc[2, '上机前DD'] = '无法测出'
    df_synth.loc[3, '抗凝血制剂/种'] = '肝素/低分子肝素/阿加曲班/比伐芦定'
    
    try:
        # Running through standardization functions
        res_std = stage2_module.standardize_clinical_types(df_synth)
        print("Pipeline handled synthetic data types successfully!")
        print("  Coerced '无法测出' in 上机前DD to:", res_std.loc[2, '上机前DD'])
        print("  Parsed 4 drugs count to:", res_std.loc[3, '抗凝血制剂/种'])
    except Exception as e:
        print("Pipeline crashed on synthetic edge cases:", e)


if __name__ == "__main__":
    run_all_probes()
