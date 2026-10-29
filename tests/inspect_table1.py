# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path

t1 = pd.read_csv(Path(r"D:\acode\ECMO\output\02_table1_baseline\table1_combined_cohorts.csv"))
for cat, grp in t1.groupby('Category', sort=False):
    print(f"\n=== Category: {cat} ({len(grp)} items) ===")
    for _, r in grp.iterrows():
        var = str(r['Variable'])
        s1 = str(r['Stage 1 (n=266)'])
        s2 = str(r['Stage 2 (n=106)'])
        p = str(r['P_Display'])
        pval = r['P_Value']
        flag = " [P < 0.05]" if pd.notna(pval) and float(pval) < 0.05 else ""
        print(f"  {var:<60} | S1={s1:<22} | S2={s2:<22} | P={p}{flag}")
