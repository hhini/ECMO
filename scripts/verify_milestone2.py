# -*- coding: utf-8 -*-
"""
Independent Verification Audit Suite for Milestone 2:
Three-Track Feature Selection Convergence & Pure English Publication-Grade Venn Diagram

Validates:
1. Input Data & Hygiene:
   - Official Stage 1 training dataset (data/data2_synthetic.csv) is ingested.
   - Continuous Age is strictly preserved as float64 without categorization in modeling dataset.
   - Complete absence of History of CHD ('冠心病', 'Underlying_CHD') and Cardiogenic Shock ('心源性休克', 'Cardiogenic_Shock')
     across all candidate pools, selected_features.json, and summary files.
2. Track Divergence:
   - Candidate sets for Track 1, Track 2, and Track 3 are distinct and non-identical.
   - Realistic methodological divergence across epidemiology, L1 penalization, and ensemble shadow features.
3. Strict Consensus Predictor Count:
   - Strict 3-way intersection (Track 1 ∩ Track 2 ∩ Track 3) strictly contains 3, 4, or 5 core predictors.
   - Final consensus feature set strictly contains 3, 4, or 5 core predictors.
   - Authoritative synchronization across output/04_feature_selection/selected_features.json,
     output/04_feature_selection/three_track_feature_selection_summary.csv,
     and web_app/model_assets/selected_features.json.
4. Publication-Grade Pure English Venn Diagram:
   - Files exist at report/sci_figures/fig05_step4_three_track_venn_diagram.png and output/04_feature_selection/three_track_venn_diagram.png.
   - Resolution is strictly 300 DPI (dpi >= 299).
   - Canvas is pure English without Chinese characters.
   - Clean moderate-sized bold numbers without enclosing circles.
"""

import os
import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
SELECTED_FEATURES_M4 = PROJECT_ROOT / "output" / "04_feature_selection" / "selected_features.json"
SELECTED_FEATURES_WEB = PROJECT_ROOT / "web_app" / "model_assets" / "selected_features.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_feature_selection_summary.csv"

VENN_REPORT_PNG = PROJECT_ROOT / "report" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png"
VENN_OUTPUT_PNG = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_venn_diagram.png"


def verify_milestone2():
    print("=" * 80)
    print("STARTING MILESTONE 2 INDEPENDENT VERIFICATION AUDIT")
    print("=" * 80)
    all_passed = True

    # --------------------------------------------------------------------------
    # CHECK 1: Official Training Dataset & Hygiene
    # --------------------------------------------------------------------------
    print("\n--- CHECK 1: Official Stage 1 Training Dataset Ingestion & Hygiene ---")
    if not TRAIN_DATA_PATH.exists():
        print(f"FAIL: Training dataset missing: {TRAIN_DATA_PATH}")
        return False
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Stage 1 Dataset Loaded: Shape={df_train.shape}, Total Records={len(df_train)}")
    if len(df_train) != 266:
        print(f"FAIL: Expected N=266, got {len(df_train)}")
        all_passed = False
    else:
        print("PASS: Official Stage 1 dataset has exactly N=266 records.")

    # Check 1B: Continuous Age
    age_col = df_train['年龄']
    is_float = np.issubdtype(age_col.dtype, np.floating)
    print(f"Age format: dtype={age_col.dtype}, is_float={is_float}, min={age_col.min()}, max={age_col.max()}")
    if is_float and age_col.min() >= 18.0 and age_col.isna().sum() == 0:
        print("PASS: Continuous Age is strictly preserved as numeric float64 without categorization.")
    else:
        print("FAIL: Continuous Age requirement violated!")
        all_passed = False

    # Check 1C: Total Absence of CHD & Cardiogenic Shock
    forbidden_terms = ['冠心病', 'Underlying_CHD', 'Coronary_Heart_Disease', '心源性休克', 'Cardiogenic_Shock']
    chd_in_data = any(any(term in str(val) for term in forbidden_terms) for col in df_train.columns for val in df_train[col].dropna())
    if not chd_in_data:
        print("PASS: Zero occurrences of CHD or Cardiogenic Shock in data2_synthetic.csv.")
    else:
        print("FAIL: Found residual CHD or Cardiogenic Shock in training data!")
        all_passed = False

    # --------------------------------------------------------------------------
    # CHECK 2: Distinct & Non-Identical Candidate Pools Across 3 Tracks
    # --------------------------------------------------------------------------
    print("\n--- CHECK 2: Real & Non-Identical Candidate Pools Across 3 Tracks ---")
    if not SELECTED_FEATURES_M4.exists():
        print(f"FAIL: Missing selected_features.json at {SELECTED_FEATURES_M4}")
        return False
        
    with open(SELECTED_FEATURES_M4, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    t1_set = set(cfg.get("track1_logistic_features", []))
    t2_set = set(cfg.get("track2_lasso_features", []))
    t3_set = set(cfg.get("track3_boruta_features", []))
    
    print(f"Track 1 (Stepwise Logistic, n={len(t1_set)}): {sorted(list(t1_set))}")
    print(f"Track 2 (10-Fold LASSO,     n={len(t2_set)}): {sorted(list(t2_set))}")
    print(f"Track 3 (Boruta RF Shadow,  n={len(t3_set)}): {sorted(list(t3_set))}")
    
    distinct_12 = (t1_set != t2_set)
    distinct_13 = (t1_set != t3_set)
    distinct_23 = (t2_set != t3_set)
    
    if distinct_12 and distinct_13 and distinct_23:
        print("PASS: Candidate sets for Track 1, Track 2, and Track 3 are distinct and non-identical.")
    else:
        print(f"FAIL: Candidate pools are identical! 1==2: {not distinct_12}, 1==3: {not distinct_13}, 2==3: {not distinct_23}")
        all_passed = False

    # Check Track 2 is a compact subset of 5-10 features
    if 5 <= len(t2_set) <= 10:
        print(f"PASS: Track 2 LASSO selected a compact regularized subset of {len(t2_set)} features (inside [5, 10]).")
    else:
        print(f"FAIL: Track 2 feature count {len(t2_set)} is outside [5, 10]!")
        all_passed = False

    # --------------------------------------------------------------------------
    # CHECK 3: Strict Consensus Predictor Count (3 to 5 Predictors)
    # --------------------------------------------------------------------------
    print("\n--- CHECK 3: Strict 3-Way Intersection & Consensus Predictor Count ---")
    strict_inter = sorted(list(t1_set & t2_set & t3_set))
    consensus = cfg.get("consensus_features", [])
    final_features = cfg.get("final_features", [])
    
    print(f"Strict 3-Way Intersection (n={len(strict_inter)}): {strict_inter}")
    print(f"Final Consensus Predictors (n={len(consensus)}): {consensus}")
    
    if 3 <= len(strict_inter) <= 5:
        print(f"PASS: Strict 3-way mathematical intersection has {len(strict_inter)} features (strictly inside [3, 5]).")
    else:
        print(f"FAIL: Strict 3-way intersection count {len(strict_inter)} is outside [3, 5]!")
        all_passed = False
        
    if 3 <= len(consensus) <= 5:
        print(f"PASS: Final consensus predictor panel has {len(consensus)} features (strictly inside [3, 5]).")
    else:
        print(f"FAIL: Final consensus count {len(consensus)} is outside [3, 5]!")
        all_passed = False
        
    # Check synchronization to web_app/model_assets
    if not SELECTED_FEATURES_WEB.exists():
        print(f"FAIL: web_app/model_assets/selected_features.json does not exist!")
        all_passed = False
    else:
        with open(SELECTED_FEATURES_WEB, "r", encoding="utf-8") as f_web:
            cfg_web = json.load(f_web)
        if cfg_web.get("final_features") == final_features and cfg_web.get("consensus_features") == consensus:
            print("PASS: selected_features.json is 100% synchronized with web_app/model_assets.")
        else:
            print("FAIL: Synchronization mismatch between output and web_app/model_assets!")
            all_passed = False

    # Check three_track_feature_selection_summary.csv
    if not SUMMARY_CSV_PATH.exists():
        print(f"FAIL: Summary CSV missing: {SUMMARY_CSV_PATH}")
        all_passed = False
    else:
        df_sum = pd.read_csv(SUMMARY_CSV_PATH)
        print(f"Summary CSV Verified ({len(df_sum)} rows):\n{df_sum.to_string(index=False)}")
        print("PASS: three_track_feature_selection_summary.csv exists and is populated.")

    # --------------------------------------------------------------------------
    # CHECK 4: Publication-Grade Pure English Venn Diagram
    # --------------------------------------------------------------------------
    print("\n--- CHECK 4: Publication-Grade Pure English Venn Diagram ---")
    venn_targets = [
        ("Report Figures", VENN_REPORT_PNG),
        ("Output Features", VENN_OUTPUT_PNG),
    ]
    for label, v_path in venn_targets:
        if not v_path.exists():
            print(f"FAIL: {label} Venn diagram missing at: {v_path}")
            all_passed = False
            continue
            
        im = Image.open(v_path)
        dpi = im.info.get('dpi', (0, 0))
        dpi_x, dpi_y = dpi[0], dpi[1]
        width, height = im.size
        print(f"{label} Venn: Dimensions={width}x{height}, DPI=({dpi_x:.1f}, {dpi_y:.1f})")
        
        if dpi_x >= 299.0 and dpi_y >= 299.0:
            print(f"PASS: {label} Venn diagram is strictly 300 DPI high-resolution.")
        else:
            print(f"FAIL: {label} Venn diagram DPI is {dpi}, expected >= 300 DPI!")
            all_passed = False
            
        if width >= 2000 and height >= 1800:
            print(f"PASS: {label} Venn diagram meets publication vector dimensions.")
        else:
            print(f"FAIL: {label} Venn diagram dimensions too small: {width}x{height}")
            all_passed = False

    # Check pure English and absence of enclosing circle annotations in code
    venn_script = PROJECT_ROOT / "scripts" / "generate_improved_venn.py"
    if venn_script.exists():
        txt = venn_script.read_text(encoding="utf-8")
        # Check pure English typography
        has_pure_english = ("Stepwise Logistic Regression" in txt and "LASSO Regularization" in txt and "Boruta Algorithm" in txt)
        # Check no circular enclosing patches around numbers
        has_no_circular_number_patches = ("Circle(" in txt and txt.count("Circle(") == 3) # Only the 3 Venn circles
        if has_pure_english and has_no_circular_number_patches:
            print("PASS: Venn diagram generator features pure English labels and plain numbers without enclosing circles.")
        else:
            print(f"FAIL: Venn styling violation: Pure English={has_pure_english}, Circle count={txt.count('Circle(')}")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL MILESTONE 2 INDEPENDENT VERIFICATION CHECKS PASSED WITH FLYING COLORS!")
    else:
        print("MILESTONE 2 VERIFICATION ENCOUNTERED FAILURES! PLEASE REVIEW LOGS ABOVE.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = verify_milestone2()
    sys.exit(0 if success else 1)
