# Code Guidelines & Technical Standards (skill.md)

This document defines the coding standards, structural organization, logging constraints, and visualization requirements for the ECMO Lower Extremity Thrombosis predictive modeling project. All Python and R scripts developed for this project must strictly adhere to these guidelines.

---

## 1. Code Decoupling & Parameterization (解耦与参数化)
- **Configuration Block**: All directory paths, input/output file names, random seeds, hyperparameter ranges, and processing thresholds must be defined at the very beginning of each script.
- **No Hardcoding**: No hard-coded paths or magic numbers are allowed in the middle of the analytical pipelines.
- **Example Structure**:
  ```python
  # ==========================================
  # CONFIGURATION & PARAMETERS
  # ==========================================
  DATA_PATH = "../data/data2.csv"
  OUTPUT_DIR = "../output/01_preprocessing"
  RANDOM_SEED = 42
  MISSING_THRESHOLD = 0.30
  # ==========================================
  ```

## 2. High Code Readability & Comments (高可读性与详细注释)
- **Detailed Comments**: Every logical section, major function, and data transformation step must have clear, human-readable Chinese and English comments explaining the clinical or statistical rationale.
- **Self-Documenting Code**: Use descriptive variable names (e.g., `missing_percent` instead of `mp`).

## 3. Strict Logging System (严格日志系统)
- **Console & File Logging**: Every script must output detailed progress logs to both the console and a physical `.log` file.
- **Log Location**: Logs must be saved in `output/logs/` or the corresponding step-specific subdirectory, named after the script (e.g., `01_preprocess.log`).
- **Key Information to Log**:
  - Script start and completion timestamps.
  - Intermediate dataset shapes (rows, columns) at each filtering step.
  - Missingness statistics and imputation status.
  - Hyperparameter tuning progress and selected parameters.
  - Evaluation metrics (AUC, sensitivity, specificity, etc.).

## 4. SCI-Grade Visualization (科研级可视化规范)
- **Language**: All figures must use **English** labels, axes, titles, and legends to be suitable for SCI journal publication.
- **Aesthetic Theme**: Use publication-grade color palettes (e.g., Nature Publishing Group `npg`, Lancet `lancet`, or JAMA theme from R's `ggsci` package). Avoid default high-contrast red/green/blue.
- **Layout & Quality**:
  - High resolution (minimum 300 DPI for raster formats like PNG, vector formats like PDF/SVG are preferred where applicable).
  - Clear, readable fonts (e.g., Arial, Helvetica, or Times New Roman).
  - Explicit legends and annotations for statistical significance (e.g., p-values).
- **Aesthetic Density**: Generate a rich set of exploratory and diagnostic plots (e.g., distribution plots, correlation heatmaps, lasso trace plots, ROC curves, calibration curves, DCA, and SHAP plots) to ensure comprehensiveness, even if some are kept only as supplementary materials.

## 5. Directory & Output Structure (清晰的研究目录结构)
Outputs must be organized into numbered step subdirectories inside the `output/` folder:
- `output/01_preprocessing/`: Imputed datasets, scale parameters, and missingness visualizations.
- `output/02_feature_selection/`: Selected feature lists, VIF tables, LASSO/Boruta diagnostic plots.
- `output/03_model_training/`: Trained model binaries, ROC/calibration curves, DCA plots, metrics summaries.
- `output/04_shap_explain/`: SHAP summary and dependence plots.
- `output/05_shiny_app/`: UI/Server files for the Shiny application.
- All files must follow systematic naming (e.g., `01_univariate_results.csv`, `02_roc_curves.png`).

## 6. Scientific Integrity & Data Policy (学术真实性 - 绝不造假)
- **Zero Fabrication**: Under no circumstances should synthetic, simulated, or fake data be generated or used for modeling.
- **Empirical Pipeline**: All results must be derived directly from the real patient records in `data2.csv` (and the external validation cohort).
- **Auditability**: Every analysis step must be completely reproducible using the scripts in `scripts/` folder.
