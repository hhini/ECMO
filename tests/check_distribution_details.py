# -*- coding: utf-8 -*-
import sys
import pandas as pd
import scipy.stats as stats
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, 'scripts')
import importlib
gen_t1 = importlib.import_module("02_generate_table1")
df_s1, df_s2, df_comb = gen_t1.load_and_harmonize_cohorts()

def check_normality_details(df, col_name, label):
    s = df[col_name].dropna()
    skew_val = stats.skew(s)
    kurt_val = stats.kurtosis(s)
    stat, p_val = stats.shapiro(s)
    print(f"[{label}] {col_name:20s}: N={len(s)}, Shapiro_p={p_val:.4f}, Skew={skew_val:.3f}, Kurt={kurt_val:.3f}")
    # Per group
    if 'LET' in df.columns:
        g0 = df[df['LET'] == 0][col_name].dropna()
        g1 = df[df['LET'] == 1][col_name].dropna()
        p0 = stats.shapiro(g0)[1] if len(g0) >= 3 else np.nan
        p1 = stats.shapiro(g1)[1] if len(g1) >= 3 else np.nan
        print(f"   Group 0 (N={len(g0)}): p={p0:.4f}, Skew={stats.skew(g0):.3f} | Group 1 (N={len(g1)}): p={p1:.4f}, Skew={stats.skew(g1):.3f}")

print("=== Stage 1 Welch Variables ===")
check_normality_details(df_s1, 'AT3_D0', 'Stage 1')
check_normality_details(df_s1, 'AT3_D2', 'Stage 1')

print("\n=== Stage 2 Welch Variables ===")
check_normality_details(df_s2, 'MAP_High', 'Stage 2')
check_normality_details(df_s2, 'MAP_Low', 'Stage 2')
check_normality_details(df_s2, 'Albumin_D0', 'Stage 2')
check_normality_details(df_s2, 'AT3_D2', 'Stage 2')
check_normality_details(df_s2, 'RBC_D2', 'Stage 2')
check_normality_details(df_s2, 'Hb_D2', 'Stage 2')

print("\n=== Combined Thrombosis Welch Variables ===")
check_normality_details(df_comb, 'AT3_D2', 'Combined')
check_normality_details(df_comb, 'Hb_D2', 'Combined')

