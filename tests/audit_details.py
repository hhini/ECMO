# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np

# Force UTF-8 on Windows console
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_s1 = pd.read_csv(f"{PROJECT_ROOT}/data/data2.csv")
df_s2 = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)

print("=== STAGE 2 RAW '血栓事件' ALL VALUES ===")
for val, count in df_raw['血栓事件'].value_counts(dropna=False).items():
    print(f"  {repr(val)}: count={count}")

print("\n=== STAGE 2 CLEAN '血栓事件' vs '是否下肢血栓' ===")
for (evt, target), count in df_s2.groupby(['血栓事件', '是否下肢血栓']).size().items():
    print(f"  Event: {repr(evt):<30} Target: {target} (count={count})")

print("\n=== STAGE 1 '血栓事件' vs '是否下肢血栓' ===")
for (evt, target), count in df_s1.groupby(['血栓事件', '是否下肢血栓']).size().items():
    print(f"  Event: {repr(evt):<30} Target: {target} (count={count})")

print("\n=== DTYPE DIFFERENCE DETAILS ===")
cols_diff = [c for c in df_s1.columns if df_s1[c].dtype != df_s2[c].dtype]
for c in cols_diff:
    print(f"\nColumn: {c}")
    print(f"  Stage 1 dtype: {df_s1[c].dtype}")
    print(f"  Stage 2 dtype: {df_s2[c].dtype}")
    # Inspect non-numeric values in Stage 1
    s1_vals = df_s1[c].dropna()
    non_numeric_s1 = [v for v in s1_vals if not str(v).replace('.', '', 1).isdigit()]
    print(f"  Stage 1 non-numeric count: {len(non_numeric_s1)}, samples: {non_numeric_s1[:10]}")
