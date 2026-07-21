# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Predict Pipeline Automation - Run All (Original & Synthetic)
Author: Antigravity Team
Date: 2026-07-14

This script automates the end-to-end execution of the preprocessing, 
feature selection, model training, and feature exploration scripts 
for both the original cohort (data2.csv) and the semi-synthetic cohort (data2_synthetic.csv).
"""

import os
import subprocess
import sys

def run_script(python_exe, script_relative_path, use_synthetic):
    script_path = os.path.join(r"D:\acode\ECMO", script_relative_path)
    if not os.path.exists(script_path):
        print(f"[ERROR] Script not found: {script_path}")
        return False
        
    env = os.environ.copy()
    env["USE_SYNTHETIC"] = "True" if use_synthetic else "False"
    
    print(f"\nRunning {script_relative_path} (USE_SYNTHETIC={use_synthetic})...")
    result = subprocess.run(
        [python_exe, script_path],
        env=env,
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] Script {script_relative_path} failed with exit code {result.returncode}")
        return False
    return True

def main():
    python_exe = r"D:\pythontext\.venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # Fallback
        
    print(f"Using Python executable: {python_exe}")
    
    pipeline_steps = [
        r"scripts\data_clear\01_preprocess.py",
        r"scripts\data_clear\02_feature_selection.py",
        r"scripts\data_clear\03_train_models.py",
        r"scripts\data_process\01_preprocess.py",
        r"scripts\data_process\02_feature_selection.py",
        r"scripts\data_process\03_train_models.py",
        r"scripts\exploration\shap_analysis.py",
        r"scripts\exploration\feature_exploration.py"
    ]
    
    # 1. Run for Original Cohort
    print("\n" + "="*80)
    print("RUNNING END-TO-END PIPELINE FOR ORIGINAL COHORT (data2.csv)")
    print("="*80)
    for step in pipeline_steps:
        success = run_script(python_exe, step, use_synthetic=False)
        if not success:
            sys.exit(1)
            
    # 2. Run for Semi-Synthetic Cohort
    print("\n" + "="*80)
    print("RUNNING END-TO-END PIPELINE FOR SEMI-SYNTHETIC COHORT (data2_synthetic.csv)")
    print("="*80)
    for step in pipeline_steps:
        success = run_script(python_exe, step, use_synthetic=True)
        if not success:
            sys.exit(1)
            
    print("\n" + "="*80)
    print("All tasks completed successfully for both original and synthetic cohorts!")
    print("="*80)

if __name__ == "__main__":
    main()
