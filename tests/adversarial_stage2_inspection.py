# -*- coding: utf-8 -*-
"""
Adversarial Exploration and Stress Analysis for Milestone 1.
Author: Challenger 1
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"D:\acode\ECMO")
RAW_PATH = PROJECT_ROOT / "testdata" / "第二阶段原始数据.xlsx"
CLEAN_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
AUDIT_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_exclusion_audit.csv"
STAGE1_PATH = PROJECT_ROOT / "data" / "data2.csv"

def inspect_all():
    print("=== 1. RAW EXCEL INSPECTION ===")
    df_raw = pd.read_excel(RAW_PATH, sheet_name=0)
    print(f"Raw shape: {df_raw.shape}")
    
    # Check '血栓事件' unique values across all 140 rows
    print("\n--- Unique values of '血栓事件' across all raw rows (including counts) ---")
    print(df_raw['血栓事件'].value_counts(dropna=False))
    
    # Check row 108-122 details
    print("\n--- Details of rows 108 to 122 (Pre-flagged exclusions) ---")
    sub_108_122 = df_raw.iloc[108:123][['编号', '住院号', '入院时间', '年龄', 'ECMO时长/h', '血栓事件', '结局']]
    print(sub_108_122.to_string())
    
    # Check rows 123 to 139 (Template rows)
    print("\n--- Details of rows 123 to 139 (Template rows) ---")
    sub_123_139 = df_raw.iloc[123:140][['编号', '住院号', '入院时间', '年龄', 'ECMO时长/h', '血栓事件', '结局']]
    print(sub_123_139.head(5).to_string())
    print("Non-null counts in rows 123-139 per row:")
    print(df_raw.iloc[123:140].notna().sum(axis=1).values)
    
    # Check candidate rows 0-107
    print("\n--- Rows 0-107 missingness in outcomes ---")
    cand = df_raw.iloc[:108]
    missing_thromb = cand[cand['血栓事件'].isna()]
    print(f"Rows with NaN in '血栓事件' in 0-107: index {missing_thromb.index.tolist()}")
    print(missing_thromb[['编号', '住院号', '入院时间', '年龄', 'ECMO时长/h', '血栓事件', '结局']].to_string())
    
    # Check if there are any candidate rows with age < 18 or ECMO < 24
    age_under_18 = cand[pd.to_numeric(cand['年龄'], errors='coerce') < 18]
    print(f"\nCandidate rows with age < 18: {len(age_under_18)}")
    ecmo_under_24 = cand[pd.to_numeric(cand['ECMO时长/h'], errors='coerce') < 24]
    print(f"Candidate rows with ECMO < 24h: {len(ecmo_under_24)}")
    if len(ecmo_under_24) > 0:
        print(ecmo_under_24[['编号', '住院号', '年龄', 'ECMO时长/h']])
        
    print("\n=== 2. THROMBOSIS KEYWORD AUDIT IN STAGE 1 & STAGE 2 ===")
    df_s1 = pd.read_csv(STAGE1_PATH)
    print("\nStage 1 unique '血栓事件':")
    print(df_s1['血栓事件'].value_counts(dropna=False))
    
    # Cross reference Stage 1 '血栓事件' vs '是否下肢血栓'
    print("\nStage 1 '血栓事件' vs '是否下肢血栓' mapping:")
    s1_map = df_s1.groupby(['血栓事件', '是否下肢血栓']).size().unstack(fill_value=0)
    print(s1_map.to_string())

    print("\n=== 3. STAGE 2 CLEAN TARGET LABELING AUDIT ===")
    df_clean = pd.read_csv(CLEAN_PATH)
    s2_map = df_clean.groupby(['血栓事件', '是否下肢血栓']).size().unstack(fill_value=0)
    print("Stage 2 clean '血栓事件' vs '是否下肢血栓' mapping:")
    print(s2_map.to_string())
    
    print("\n=== 4. SCHEMA AND DTYPE COMPARISON ===")
    print(f"Stage 1 shape: {df_s1.shape}, Stage 2 shape: {df_clean.shape}")
    cols_s1 = list(df_s1.columns)
    cols_s2 = list(df_clean.columns)
    print(f"Columns identical in order and name: {cols_s1 == cols_s2}")
    
    # Dtype comparison
    dtype_diffs = []
    for c in cols_s1:
        dt1 = df_s1[c].dtype
        dt2 = df_clean[c].dtype
        if dt1 != dt2:
            dtype_diffs.append((c, dt1, dt2))
    print(f"\nNumber of columns with different dtypes between Stage 1 and Stage 2: {len(dtype_diffs)}")
    if dtype_diffs:
        print("Dtype differences (Col, Stage 1, Stage 2):")
        for diff in dtype_diffs:
            print(f"  {diff[0]}: S1={diff[1]} vs S2={diff[2]}")

if __name__ == "__main__":
    inspect_all()
