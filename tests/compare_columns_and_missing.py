# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_s1 = pd.read_csv(f"{PROJECT_ROOT}/data/data2.csv")
df_s2 = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")

print("=== COLUMN-BY-COLUMN COMPARISON ===")
print(f"Total columns: S1={len(df_s1.columns)}, S2={len(df_s2.columns)}")

object_cols_s2 = df_s2.select_dtypes(include=['object']).columns.tolist()
print(f"\nStage 2 object columns ({len(object_cols_s2)}):")
for col in object_cols_s2:
    s2_vals = df_s2[col].dropna().unique()
    s1_vals = df_s1[col].dropna().unique() if col in df_s1.columns else "NOT IN S1"
    print(f"  Col: {col:<25} | S2 unique ({len(s2_vals)}): {list(s2_vals)[:5]} | S1 unique: {list(s1_vals)[:5] if isinstance(s1_vals, (list, np.ndarray)) else s1_vals}")

# Check missing rates
print("\n=== MISSING RATES COMPARISON (Cols with > 50% missingness) ===")
s2_null_rates = df_s2.isna().mean()
s1_null_rates = df_s1.isna().mean()

high_missing_s2 = s2_null_rates[s2_null_rates > 0.5]
print(f"Stage 2 high missing cols (>50%): {len(high_missing_s2)}")
for col, rate in high_missing_s2.items():
    s1_rate = s1_null_rates.get(col, np.nan)
    print(f"  {col:<30}: S2={rate*100:.1f}%, S1={s1_rate*100:.1f}%")
