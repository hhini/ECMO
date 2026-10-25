# -*- coding: utf-8 -*-
"""
Repository Sanitation Script
Safely removes redundant, obsolete, intermediate files and directories
while preserving all critical data, active models, active code, and fresh reports.
"""

import os
import shutil

ROOT_DIR = r"D:\acode\ECMO"

files_to_remove = [
    os.path.join(ROOT_DIR, "all_p_values.csv"),
    os.path.join(ROOT_DIR, "top_50_univariate_features.json"),
    os.path.join(ROOT_DIR, "problem.md"),
    os.path.join(ROOT_DIR, "categorical_feature_engineering_plan.md"),
    os.path.join(ROOT_DIR, "ECMO_Thrombosis_Data_Processing_Workflow.md"),
    os.path.join(ROOT_DIR, "ECMO_Thrombosis_Report.md"),
    os.path.join(ROOT_DIR, "TEST_INFRA.md"),
    os.path.join(ROOT_DIR, "TEST_READY.md"),
    os.path.join(ROOT_DIR, "output", "cohort_data_characteristics.md"),
    os.path.join(ROOT_DIR, "output", "consistent_features.csv"),
    os.path.join(ROOT_DIR, "output", "data_inspection_report.md"),
    os.path.join(ROOT_DIR, "output", "e2e_test_report.md"),
    os.path.join(ROOT_DIR, "output", "ECMO_Thrombosis_Report.md"),
    os.path.join(ROOT_DIR, "report", "ECMO_Thrombosis_Data_Processing_Report.docx"),
    os.path.join(ROOT_DIR, "scripts", "convert_data.py"),
    os.path.join(ROOT_DIR, "scripts", "create_workflow_diagram.py"),
    os.path.join(ROOT_DIR, "scripts", "data_inspection.py"),
    os.path.join(ROOT_DIR, "scripts", "generate_cohort_report.py"),
    os.path.join(ROOT_DIR, "scripts", "generate_docx_report.py"),
    os.path.join(ROOT_DIR, "scripts", "generate_semi_synthetic_data.py"),
    os.path.join(ROOT_DIR, "scripts", "run_all_pipelines.py"),
    os.path.join(ROOT_DIR, "scripts", "run_synthetic_pipeline.py"),
    os.path.join(ROOT_DIR, "scripts", "01_preprocess.py"),
    os.path.join(ROOT_DIR, "scripts", "02_feature_selection.py"),
    os.path.join(ROOT_DIR, "scripts", "03_train_models.py"),
    os.path.join(ROOT_DIR, "scripts", "generate_sci_publication_figures.py"),
    os.path.join(ROOT_DIR, "scripts", "beautify_sci_figures.py")
]

dirs_to_remove = [
    os.path.join(ROOT_DIR, "report", "figures"),
    os.path.join(ROOT_DIR, "report", "images"),
    os.path.join(ROOT_DIR, "report", "tables"),
    os.path.join(ROOT_DIR, "scratch"),
    os.path.join(ROOT_DIR, "scripts", "__pycache__"),
    os.path.join(ROOT_DIR, "tests", "__pycache__")
]

removed_count = 0
for f in files_to_remove:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Removed file: {f}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {f}: {e}")

for d in dirs_to_remove:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"Removed directory: {d}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {d}: {e}")

print(f"\nSanitation complete. Total removed items: {removed_count}")
