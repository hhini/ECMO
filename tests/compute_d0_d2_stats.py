# -*- coding: utf-8 -*-
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(r"D:\acode\ECMO")
df1 = pd.read_csv(PROJECT_ROOT / "data" / "data2.csv")
df2 = pd.read_csv(PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv")

# Exact D0 features:
d0_features = [
    "上机前DD", "FIB", "TT", "APTT", "PT", "INR", "AT3", "PLT", "WBC", "Hb", "PCT", "CRP", "谷丙", "肌酐"
]

# Exact D2 features:
d2_features = [
    "上机第二天DD", "FIB.1", "TT.1", "APTT.1", "PT.1", "INR.1", "AT3.1", "PLT.1", "WBC.1", "Hb.1", "PCT.1", "CRP.1", "谷丙.1", "肌酐.1"
]

# Demographics & continuous features:
demo_features = [
    "年龄", "BMI", "收缩压", "舒张压", "平均动脉压", "最高转速", "最高血流速", "最低转速", "最低血流速",
    "ECMO时长", "ICU住院时长/d", "呼吸机使用时长/d", "镇静药物使用时长/d"
]

all_target_features = [("Day 0 Baseline", d0_features), ("Day 2 (48h)", d2_features), ("Demographic & Physiological", demo_features)]

report_rows = []

for section_name, feats in all_target_features:
    for feat in feats:
        if feat not in df1.columns or feat not in df2.columns:
            continue
        v1 = df1[feat]
        v2 = df2[feat]
        
        # If string in v1, extract numeric
        if v1.dtype == 'object':
            v1_num = pd.to_numeric(v1.astype(str).str.extract(r'([0-9\.]+)', expand=False), errors='coerce')
        else:
            v1_num = pd.to_numeric(v1, errors='coerce')
            
        v2_num = pd.to_numeric(v2, errors='coerce')
        
        n1_null = v1_num.isna().sum()
        p1_null = n1_null / len(df1) * 100.0
        n2_null = v2_num.isna().sum()
        p2_null = n2_null / len(df2) * 100.0
        
        s1_clean = v1_num.dropna()
        s2_clean = v2_num.dropna()
        
        m1, std1 = s1_clean.mean(), s1_clean.std()
        med1 = s1_clean.median()
        iqr1_l, iqr1_u = s1_clean.quantile(0.25), s1_clean.quantile(0.75)
        
        m2, std2 = s2_clean.mean(), s2_clean.std()
        med2 = s2_clean.median()
        iqr2_l, iqr2_u = s2_clean.quantile(0.25), s2_clean.quantile(0.75)
        
        # Stats tests
        if len(s1_clean) > 5 and len(s2_clean) > 5:
            ks_res = stats.ks_2samp(s1_clean, s2_clean)
            ks_stat, ks_p = ks_res.statistic, ks_res.pvalue
            w_dist = stats.wasserstein_distance(s1_clean, s2_clean)
            pooled_s = np.sqrt((std1**2 + std2**2)/2) if (std1**2 + std2**2) > 0 else 1.0
            smd = abs(m2 - m1) / pooled_s
            # Mann-Whitney U test
            mwu_res = stats.mannwhitneyu(s1_clean, s2_clean, alternative='two-sided')
            mwu_p = mwu_res.pvalue
        else:
            ks_stat, ks_p, w_dist, smd, mwu_p = np.nan, np.nan, np.nan, np.nan, np.nan
            
        report_rows.append({
            "section": section_name,
            "feature": feat,
            "s1_null_pct": p1_null,
            "s2_null_pct": p2_null,
            "delta_null_pct": p2_null - p1_null,
            "s1_med_iqr": f"{med1:.2f} ({iqr1_l:.2f}-{iqr1_u:.2f})",
            "s2_med_iqr": f"{med2:.2f} ({iqr2_l:.2f}-{iqr2_u:.2f})",
            "s1_mean_std": f"{m1:.2f}±{std1:.2f}",
            "s2_mean_std": f"{m2:.2f}±{std2:.2f}",
            "smd": smd,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "mwu_p": mwu_p,
            "w_dist": w_dist
        })

df_report = pd.DataFrame(report_rows)
df_report.to_json(PROJECT_ROOT / "tests" / "detailed_lab_d0_d2_stats.json", orient="records", force_ascii=False, indent=2)

# Generate Markdown table
md_lines = ["# Detailed D0, D2 & Physiological Features Comparison: Stage 1 vs Stage 2\n"]
for section in ["Day 0 Baseline", "Day 2 (48h)", "Demographic & Physiological"]:
    md_lines.append(f"## {section}\n")
    md_lines.append("| Feature | S1 Null% | S2 Null% | S1 Median (IQR) | S2 Median (IQR) | SMD | KS p-value | MWU p-value |")
    md_lines.append("|---|---|---|---|---|---|---|---|")
    sub = df_report[df_report['section'] == section]
    for _, r in sub.iterrows():
        p_str = f"{r['ks_p']:.3e}" if pd.notna(r['ks_p']) else "—"
        mwu_str = f"{r['mwu_p']:.3e}" if pd.notna(r['mwu_p']) else "—"
        md_lines.append(
            f"| {r['feature']} | {r['s1_null_pct']:.1f}% | {r['s2_null_pct']:.1f}% | "
            f"{r['s1_med_iqr']} | {r['s2_med_iqr']} | {r['smd']:.3f} | {p_str} | {mwu_str} |"
        )
    md_lines.append("")

with open(PROJECT_ROOT / "tests" / "d0_d2_distribution_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Saved detailed reports: tests/detailed_lab_d0_d2_stats.json and tests/d0_d2_distribution_report.md")
