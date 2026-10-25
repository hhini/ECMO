# -*- coding: utf-8 -*-
import sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_clean = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")

high_ecmo = df_clean[df_clean['ECMO时长'] > 300]
print(f"Number of patients with ECMO duration > 300h: {len(high_ecmo)}")
for idx, row in high_ecmo.iterrows():
    print(f"ID: {row['编号']}, Age: {row['年龄']}, ECMO Duration: {row['ECMO时长']}, LET: {row['是否下肢血栓']}, Event: {row['血栓事件']}")
