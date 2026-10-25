# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)
df_s1 = pd.read_csv(f"{PROJECT_ROOT}/data/data2.csv")

print("=== RAW '镇静镇痛药物/种' VALUES ===")
print(df_raw['镇静镇痛药物/种'].value_counts(dropna=False))

print("\n=== STAGE 1 '镇静镇痛药物/种' VALUES ===")
print(df_s1['镇静镇痛药物/种'].value_counts(dropna=False))

print("\n=== ROWS WITH '镇静镇痛药物/种' > 6 IN RAW ===")
high_sed = df_raw[pd.to_numeric(df_raw['镇静镇痛药物/种'], errors='coerce') > 6]
print(high_sed[['编号', '住院号', '镇静镇痛药物/种', '镇静药物使用时长/d', '抗凝血制剂/种', '抗凝血制剂使用时长/d']])
