# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)
df_s1 = pd.read_csv(f"{PROJECT_ROOT}/data/data2.csv")

lower_limb_anatomical_terms = [
    "腘", "胫", "腓", "比目鱼", "足", "趾", "大隐", "小隐", "股", "髂", "小腿", "大腿", "下肢", "肌间"
]

print("=== CHECKING ALL EVENTS IN RAW STAGE 2 FOR ANATOMICAL TERMS ===")
for idx, row in df_raw.iterrows():
    evt = str(row['血栓事件'])
    if any(term in evt for term in lower_limb_anatomical_terms):
        has_current_kw = any(kw in evt for kw in ["下肢", "股", "大隐", "髂", "足", "小腿"])
        print(f"Row {idx:3d} (ID: {row['编号']}): Event={repr(evt):<30} Matched current LET_KEYWORDS: {has_current_kw}")

print("\n=== CHECKING ALL EVENTS IN STAGE 1 FOR ANATOMICAL TERMS ===")
for evt in df_s1['血栓事件'].unique():
    evt_str = str(evt)
    if any(term in evt_str for term in lower_limb_anatomical_terms):
        has_current_kw = any(kw in evt_str for kw in ["下肢", "股", "大隐", "髂", "足", "小腿"])
        s1_target = df_s1.loc[df_s1['血栓事件'] == evt, '是否下肢血栓'].iloc[0]
        print(f"S1 Event: {repr(evt_str):<35} Matched: {has_current_kw:<5} S1 Target: {s1_target}")
