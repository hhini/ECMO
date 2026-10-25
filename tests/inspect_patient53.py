# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)
row = df_raw[df_raw['血栓事件'] == '动脉血栓'].iloc[0]

print("=== PATIENT 53 (动脉血栓) DETAILS ===")
for col in df_raw.columns[:30]:
    val = row[col]
    if pd.notna(val):
        print(f"  {col}: {val}")

for col in df_raw.columns:
    if any(k in col for k in ['模式', '置管', '插管', '血栓', '诊断', '部位', '转归']):
        print(f"  {col}: {row[col]}")
