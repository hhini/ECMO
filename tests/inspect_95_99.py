# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)

for idx in [95, 99]:
    row = df_raw.iloc[idx]
    non_nulls = row.dropna()
    print(f"\n=== ROW {idx} (Patient ID: {row['编号']}, Hospital ID: {row['住院号']}) ===")
    print(f"Total non-null columns: {len(non_nulls)} / {len(row)}")
    print(f"Non-null columns and values:")
    for col, val in non_nulls.items():
        print(f"  {col}: {val}")
