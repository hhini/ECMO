# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)
df_clean = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")

print("=== ALL NON-EMPTY '血栓事件' IN RAW EXCEL (140 ROWS) ===")
for idx, row in df_raw.iterrows():
    evt = row['血栓事件']
    if pd.notna(evt) and str(evt).strip() != '否':
        pid = row['编号']
        adm = row['入院时间']
        age = row['年龄']
        dur = row.get('ECMO时长/h', row.get('ECMO时长', None))
        in_clean = pid in df_clean['编号'].values
        target_val = df_clean.loc[df_clean['编号'] == pid, '是否下肢血栓'].values[0] if in_clean else "EXCLUDED"
        print(f"Row {idx:3d} | ID: {str(pid):<5} | Event: {repr(evt):<32} | In Clean: {str(in_clean):<5} | Target: {target_val}")
