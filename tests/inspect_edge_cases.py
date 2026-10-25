# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)

# Check row with '动脉血栓'
row_arterial = df_raw[df_raw['血栓事件'] == '动脉血栓']
print("=== ROW WITH '动脉血栓' ===")
for col in ['编号', '住院号', '年龄', '主要诊断', '手术名称', '血栓事件', '结局']:
    if col in df_raw.columns:
        print(f"  {col}: {row_arterial[col].values}")

# Check row with '颈静脉血栓'
row_jugular = df_raw[df_raw['血栓事件'] == '颈静脉血栓']
print("\n=== ROW WITH '颈静脉血栓' ===")
for col in ['编号', '住院号', '年龄', '主要诊断', '手术名称', '血栓事件', '结局']:
    if col in df_raw.columns:
        print(f"  {col}: {row_jugular[col].values}")

# Check the two missing outcome rows: 95 and 99
print("\n=== ROWS 95 AND 99 ===")
print("Row 95:")
for col in ['编号', '住院号', '年龄', '入院时间', '主要诊断', '手术名称', '血栓事件', '结局', 'ECMO时长/h']:
    if col in df_raw.columns:
        print(f"  {col}: {df_raw.iloc[95][col]}")
print("Row 99:")
for col in ['编号', '住院号', '年龄', '入院时间', '主要诊断', '手术名称', '血栓事件', '结局', 'ECMO时长/h']:
    if col in df_raw.columns:
        print(f"  {col}: {df_raw.iloc[99][col]}")
