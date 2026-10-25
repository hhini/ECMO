# -*- coding: utf-8 -*-
import sys
import pandas as pd
import scipy.stats as stats
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

tables = ['table1_stage1.csv', 'table1_stage2.csv', 'table1_combined_cohorts.csv', 'table1_combined_thrombosis.csv']

for name in tables:
    df = pd.read_csv(f'output/02_table1_baseline/{name}')
    welch = df[df['Test_Method'] == "Welch's t-test"]
    print(f"\n=== {name}: {len(welch)} Welch variables ===")
    for _, r in welch.iterrows():
        print(f"  {r['Variable']:40s} | Tot={r.iloc[2]} | G0={r.iloc[3]} | G1={r.iloc[4]} | {r['Statistic']} | p={r['P_Display']}")

