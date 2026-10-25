# -*- coding: utf-8 -*-
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('output/02_table1_baseline/table1_stage1.csv')
for feat in ['ECMO', 'D-二聚体', '肌酐', 'CRP', '谷草', '谷丙', '降钙素原']:
    matches = df[df['Variable'].str.contains(feat, case=False, na=False)]
    for _, r in matches.iterrows():
        val = str(r.iloc[2])
        test = str(r['Test_Method'])
        p = str(r['P_Display'])
        print(f"{r['Variable']:35s} | {val:30s} | {test:15s} | p={p}")
