# -*- coding: utf-8 -*-
"""
Empirical Challenge Test Suite: Milestone 1 Recalibration & Table 1 Baseline
Agent: Challenger 2 (teamwork_preview_challenger_m1_r2_2)
Date: 2026-10-29
"""

import os
import re
import sys
from pathlib import Path
import docx
import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
STAGE1_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
TABLE_DIR = PROJECT_ROOT / "output" / "02_table1_baseline"

def test_chd_and_shock_eradication():
    print("\n" + "=" * 80)
    print("CHALLENGE 1: SEARCHING FOR CHD AND CARDIOGENIC SHOCK")
    print("=" * 80)
    
    terms_to_check = {
        'CHD_chinese': '冠心病',
        'Coronary_english': 'coronary',
        'Shock_chinese': '休克',
        'Shock_cardiogenic_chinese': '心源性休克',
        'Shock_english': 'shock',
        'Cardiogenic_english': 'cardiogenic'
    }
    
    findings = []
    
    # 1. Inspect data/data2_synthetic.csv
    print(f"\n1.1 Scanning {STAGE1_PATH}...")
    df_s1 = pd.read_csv(STAGE1_PATH)
    for col in df_s1.columns:
        # Check column name
        for term_name, term in terms_to_check.items():
            if term.lower() in col.lower():
                findings.append({
                    'file': 'data/data2_synthetic.csv',
                    'location': f'Column Header: {col}',
                    'term_key': term_name,
                    'term': term,
                    'context': col
                })
        # Check values
        s_col = df_s1[col].astype(str)
        for term_name, term in terms_to_check.items():
            matches = s_col[s_col.str.contains(term, case=False, na=False)]
            for idx, val in matches.items():
                findings.append({
                    'file': 'data/data2_synthetic.csv',
                    'location': f'Row {idx}, Col {col}',
                    'term_key': term_name,
                    'term': term,
                    'context': val
                })

    # 1.2 Inspect output/01_stage2_clean/stage2_clean.csv
    print(f"1.2 Scanning {STAGE2_PATH}...")
    df_s2 = pd.read_csv(STAGE2_PATH)
    for col in df_s2.columns:
        # Check column name
        for term_name, term in terms_to_check.items():
            if term.lower() in col.lower():
                findings.append({
                    'file': 'output/01_stage2_clean/stage2_clean.csv',
                    'location': f'Column Header: {col}',
                    'term_key': term_name,
                    'term': term,
                    'context': col
                })
        # Check values
        s_col = df_s2[col].astype(str)
        for term_name, term in terms_to_check.items():
            matches = s_col[s_col.str.contains(term, case=False, na=False)]
            for idx, val in matches.items():
                findings.append({
                    'file': 'output/01_stage2_clean/stage2_clean.csv',
                    'location': f'Row {idx}, Col {col}',
                    'term_key': term_name,
                    'term': term,
                    'context': val
                })

    # 1.3 Inspect all tables in output/02_table1_baseline/
    print(f"1.3 Scanning all files in {TABLE_DIR}...")
    for file_path in TABLE_DIR.iterdir():
        if file_path.name.endswith('.csv'):
            try:
                df_tbl = pd.read_csv(file_path)
                for col in df_tbl.columns:
                    for term_name, term in terms_to_check.items():
                        if term.lower() in col.lower():
                            findings.append({
                                'file': f'output/02_table1_baseline/{file_path.name}',
                                'location': f'Column Header: {col}',
                                'term_key': term_name,
                                'term': term,
                                'context': col
                            })
                    s_col = df_tbl[col].astype(str)
                    for term_name, term in terms_to_check.items():
                        matches = s_col[s_col.str.contains(term, case=False, na=False)]
                        for idx, val in matches.items():
                            findings.append({
                                'file': f'output/02_table1_baseline/{file_path.name}',
                                'location': f'Row {idx}, Col {col}',
                                'term_key': term_name,
                                'term': term,
                                'context': val
                            })
            except Exception as e:
                print(f"Error reading CSV {file_path.name}: {e}")
        elif file_path.name.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    for term_name, term in terms_to_check.items():
                        if term.lower() in line.lower():
                            findings.append({
                                'file': f'output/02_table1_baseline/{file_path.name}',
                                'location': f'Line {line_no}',
                                'term_key': term_name,
                                'term': term,
                                'context': line.strip()
                            })
        elif file_path.name.endswith('.docx'):
            try:
                doc = docx.Document(str(file_path))
                for p_no, p in enumerate(doc.paragraphs, 1):
                    for term_name, term in terms_to_check.items():
                        if term.lower() in p.text.lower():
                            findings.append({
                                'file': f'output/02_table1_baseline/{file_path.name}',
                                'location': f'Paragraph {p_no}',
                                'term_key': term_name,
                                'term': term,
                                'context': p.text.strip()
                            })
                for t_no, t in enumerate(doc.tables, 1):
                    for r_no, row in enumerate(t.rows, 1):
                        for c_no, cell in enumerate(row.cells, 1):
                            cell_text = cell.text
                            for term_name, term in terms_to_check.items():
                                if term.lower() in cell_text.lower():
                                    findings.append({
                                        'file': f'output/02_table1_baseline/{file_path.name}',
                                        'location': f'Table {t_no}, Row {r_no}, Cell {c_no}',
                                        'term_key': term_name,
                                        'term': term,
                                        'context': cell_text.strip()
                                    })
            except Exception as e:
                print(f"Error reading DOCX {file_path.name}: {e}")

    print(f"\nTotal occurrences of target terms detected: {len(findings)}")
    # Classify findings
    chd_findings = [f for f in findings if f['term_key'] in ['CHD_chinese', 'Coronary_english']]
    shock_findings = [f for f in findings if f['term_key'] in ['Shock_chinese', 'Shock_cardiogenic_chinese', 'Shock_english', 'Cardiogenic_english']]
    
    print("\n--- Summary of CHD Findings ---")
    if not chd_findings:
        print("PASS: Zero CHD ('冠心病', 'Coronary') found anywhere!")
    else:
        print(f"ATTENTION: Found {len(chd_findings)} occurrences of Coronary / 冠心病:")
        for item in chd_findings:
            print(f"  [{item['file']}] {item['location']} | Term: '{item['term']}' | Context: {item['context'][:100]}")
            
    print("\n--- Summary of Shock Findings ---")
    if not shock_findings:
        print("PASS: Zero Cardiogenic Shock ('心源性休克', 'Shock') found anywhere!")
    else:
        print(f"ATTENTION: Found {len(shock_findings)} occurrences of Shock / 休克:")
        for item in shock_findings:
            print(f"  [{item['file']}] {item['location']} | Term: '{item['term']}' | Context: {item['context'][:100]}")

    return findings


def test_table1_age_representation():
    print("\n" + "=" * 80)
    print("CHALLENGE 2: EMPIRICAL VERIFICATION OF AGE IN DATA & TABLE 1")
    print("=" * 80)
    
    df_s1 = pd.read_csv(STAGE1_PATH)
    df_s2 = pd.read_csv(STAGE2_PATH)
    
    # 2.1 Check continuous age in modeling data
    age_s1 = pd.to_numeric(df_s1['年龄'], errors='coerce')
    age_s2 = pd.to_numeric(df_s2['年龄'], errors='coerce')
    
    print("2.1 Empirical Ground Truth Stats:")
    print(f"  Stage 1 (N={len(df_s1)}):")
    print(f"    Missing: {age_s1.isna().sum()}")
    print(f"    Min: {age_s1.min()}, Max: {age_s1.max()}")
    print(f"    Mean +/- SD: {age_s1.mean():.2f} +/- {age_s1.std(ddof=1):.2f}")
    print(f"    Median (IQR): {age_s1.median():.2f} ({age_s1.quantile(0.25):.2f} - {age_s1.quantile(0.75):.2f})")
    
    print(f"  Stage 2 (N={len(df_s2)}):")
    print(f"    Missing: {age_s2.isna().sum()}")
    print(f"    Min: {age_s2.min()}, Max: {age_s2.max()}")
    print(f"    Mean +/- SD: {age_s2.mean():.2f} +/- {age_s2.std(ddof=1):.2f}")
    print(f"    Median (IQR): {age_s2.median():.2f} ({age_s2.quantile(0.25):.2f} - {age_s2.quantile(0.75):.2f})")
    
    # Stratified 4 tiers: 18-40, 40-60, 60-70, >=70
    def categorize_age(a):
        if a < 40: return '18-40'
        elif a < 60: return '40-60'
        elif a < 70: return '60-70'
        else: return '>=70'
        
    s1_cat = age_s1.apply(categorize_age).value_counts()[['18-40', '40-60', '60-70', '>=70']]
    s2_cat = age_s2.apply(categorize_age).value_counts()[['18-40', '40-60', '60-70', '>=70']]
    
    print("  Stratified Ground Truth Frequencies:")
    for cat in ['18-40', '40-60', '60-70', '>=70']:
        c1, p1 = s1_cat[cat], s1_cat[cat]/len(df_s1)*100
        c2, p2 = s2_cat[cat], s2_cat[cat]/len(df_s2)*100
        print(f"    {cat:<6}: Stage 1 = {c1} ({p1:.1f}%), Stage 2 = {c2} ({p2:.1f}%)")
        
    # Check table files
    print("\n2.2 Inspecting Table 1 files for Age rows...")
    t1_files = ['table1_stage1.csv', 'table1_stage2.csv', 'table1_combined_cohorts.csv', 'table1_combined_thrombosis.csv', 'Table1_Baseline_Balance_Diagnosis.csv']
    
    for t_name in t1_files:
        p = TABLE_DIR / t_name
        if not p.exists():
            print(f"  WARNING: {t_name} does not exist!")
            continue
        df_tbl = pd.read_csv(p)
        var_col = 'Variable' if 'Variable' in df_tbl.columns else df_tbl.columns[0]
        age_rows = df_tbl[df_tbl[var_col].astype(str).str.contains('年龄|Age|18-40|40-60|60-70|≥70|>=70', na=False)]
        print(f"\n  In {t_name}: Found {len(age_rows)} Age-related rows:")
        for _, r in age_rows.iterrows():
            cols_to_print = [c for c in df_tbl.columns if any(k in c for k in ['Variable', 'Total', 'Stage', 'Thrombosis', 'P_Display', 'Statistic', 'Test_Method'])]
            disp = {c: str(r[c]) for c in cols_to_print[:5]}
            print(f"    - {disp}")


def test_mice_imputation_logic():
    print("\n" + "=" * 80)
    print("CHALLENGE 3: MICE IMPUTATION LOGIC & BLOOD LOSS AUDIT")
    print("=" * 80)
    
    df_s1 = pd.read_csv(STAGE1_PATH)
    
    # Surgery status
    def parse_surgery(s):
        s_str = str(s).strip()
        return 0 if s_str in ['无', '否', 'nan', ''] else 1
    
    has_surgery = df_s1['ECMO前手术史'].apply(parse_surgery)
    surg_count = has_surgery.sum()
    non_surg_count = len(df_s1) - surg_count
    print(f"Patients with prior surgery: {surg_count}")
    print(f"Patients without prior surgery: {non_surg_count}")
    
    # Blood loss parsing
    def parse_num(v):
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if s in ['无', '否', '未查', '未测', 'nan', '', '未手术']: return np.nan
        m = re.search(r'([0-9]+\.?[0-9]*)', s)
        return float(m.group(1)) if m else np.nan

    bl_vals = df_s1['手术失血量/ml'].apply(parse_num)
    
    print("\n3.1 Blood loss integrity check:")
    print(f"  Any negative blood loss across entire cohort? {(bl_vals < 0).any()} (Count: {(bl_vals < 0).sum()})")
    
    # Surgical patients blood loss
    surg_bl = bl_vals[has_surgery == 1]
    print(f"  Surgical patients missing blood loss: {surg_bl.isna().sum()}")
    print(f"  Surgical patients blood loss <= 0: {(surg_bl <= 0).sum()}")
    print(f"  Surgical patients blood loss min: {surg_bl.min()} mL")
    print(f"  Surgical patients blood loss max: {surg_bl.max()} mL")
    print(f"  Surgical patients blood loss mean: {surg_bl.mean():.2f} mL")
    print(f"  Surgical patients blood loss median: {surg_bl.median():.2f} mL")
    
    # Non-surgical patients blood loss
    non_surg_bl = bl_vals[has_surgery == 0]
    print(f"  Non-surgical patients blood loss missing/NaN: {non_surg_bl.isna().sum()}/{len(non_surg_bl)}")
    print(f"  Non-surgical raw values: {df_s1.loc[has_surgery == 0, '手术失血量/ml'].unique()[:5]}")
    
    # Check recalibration audit log to see exactly which 55 patients were imputed
    audit_path = PROJECT_ROOT / "data" / "data2_synthetic_audit.csv"
    if audit_path.exists():
        df_audit = pd.read_csv(audit_path)
        bl_audits = df_audit[df_audit['Feature_Name'] == '手术失血量/ml']
        print(f"\n3.2 Audit log records for blood loss: {len(bl_audits)}")
        imputed_audits = bl_audits[bl_audits['Clinical_Justification'].str.contains('MICE', na=False)]
        print(f"  MICE imputed records: {len(imputed_audits)}")
        imputed_vals = imputed_audits['Modified_Value'].astype(float)
        print(f"  Imputed values summary:")
        print(f"    Min: {imputed_vals.min()}, Max: {imputed_vals.max()}")
        print(f"    Mean: {imputed_vals.mean():.2f}, Std: {imputed_vals.std():.2f}")
        print(f"    Unique imputed values: {imputed_vals.nunique()} out of {len(imputed_vals)}")
        print(f"    Sample values: {imputed_vals.head(10).tolist()}")
        
        # Test multivariate association:
        # Check if imputed blood loss correlates with surgical duration, anticoagulation, BMI, Age
        imputed_indices = imputed_audits['Patient_Row_Index'].values
        df_sub = df_s1.loc[imputed_indices].copy()
        df_sub['Imputed_BL'] = imputed_vals.values
        df_sub['Age_num'] = pd.to_numeric(df_sub['年龄'], errors='coerce')
        df_sub['BMI_num'] = pd.to_numeric(df_sub['BMI'], errors='coerce')
        df_sub['Surg_Dur'] = df_sub['手术时长/h'].apply(parse_num).fillna(0)
        df_sub['ECMO_Dur'] = df_sub['ECMO时长'].apply(parse_num).fillna(0)
        
        corrs = {
            'Surg_Dur': df_sub['Imputed_BL'].corr(df_sub['Surg_Dur']),
            'Age': df_sub['Imputed_BL'].corr(df_sub['Age_num']),
            'BMI': df_sub['Imputed_BL'].corr(df_sub['BMI_num']),
            'ECMO_Dur': df_sub['Imputed_BL'].corr(df_sub['ECMO_Dur'])
        }
        print(f"  Correlations of imputed blood loss with covariates: {corrs}")
    else:
        print(f"  Audit log missing: {audit_path}")


if __name__ == '__main__':
    test_chd_and_shock_eradication()
    test_table1_age_representation()
    test_mice_imputation_logic()
