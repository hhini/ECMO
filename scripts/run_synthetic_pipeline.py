# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Semi-Synthetic Pipeline Automation Runner
Author: Antigravity Team
Date: 2026-07-14

This script automates the execution of the preprocessing, feature selection, 
and model training scripts for both data_clear and data_process tracks 
using the semi-synthetic dataset.
"""

import os
import subprocess
import sys

def main():
    python_exe = r"D:\pythontext\.venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # Fallback
        
    print(f"Using Python executable: {python_exe}")
    
    # Define tasks to run
    tasks = [
        # Track 1: data_clear (synthetic)
        ("Preprocessing (data_clear)", r"scripts\data_clear\01_preprocess.py"),
        ("Feature Selection (data_clear)", r"scripts\data_clear\02_feature_selection.py"),
        ("Model Training (data_clear)", r"scripts\data_clear\03_train_models.py"),
        
        # Track 2: data_process (synthetic)
        ("Preprocessing (data_process)", r"scripts\data_process\01_preprocess.py"),
        ("Feature Selection (data_process)", r"scripts\data_process\02_feature_selection.py"),
        ("Model Training (data_process)", r"scripts\data_process\03_train_models.py"),
    ]
    
    # Setup environment
    env = os.environ.copy()
    env["USE_SYNTHETIC"] = "True"
    
    for name, script_path in tasks:
        print("\n" + "="*60)
        print(f"Running {name}...")
        print(f"Script: {script_path}")
        print("="*60)
        
        full_path = os.path.join(r"D:\acode\ECMO", script_path)
        if not os.path.exists(full_path):
            print(f"Error: Script not found at {full_path}")
            sys.exit(1)
            
        result = subprocess.run(
            [python_exe, full_path],
            env=env,
            capture_output=False, # Print straight to terminal
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n[ERROR] Task '{name}' failed with return code {result.returncode}")
            sys.exit(result.returncode)
            
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
