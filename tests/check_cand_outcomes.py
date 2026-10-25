# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_raw = pd.read_excel(f"{PROJECT_ROOT}/testdata/第二阶段原始数据.xlsx", sheet_name=0)
cand = df_raw.iloc[:108]

print(f"Cand rows where 结局 is NaN: {cand[cand['结局'].isna()].index.tolist()}")
print(f"Cand rows where 血栓事件 is NaN: {cand[cand['血栓事件'].isna()].index.tolist()}")
