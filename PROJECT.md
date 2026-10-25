# Project: ECMO Thrombosis Stage 2 Machine Learning Validation & Deployment

## Architecture
This project executes an end-to-end clinical machine learning validation pipeline for predicting ECMO-associated lower extremity thrombosis (LET).
The system connects clinical data ingestion, baseline epidemiological statistics, imbalance benchmarking, tri-method feature selection, 6-algorithm model training with 10-fold CV, independent external validation, SHAP interpretability, and interactive web calculator deployment.

### System Data Flow & Module Boundaries
```
[Stage 2 Raw Data (XLSX)] ──> [M1: Ingestion & Cleaning] ──> [Clean Stage 2 (n=106)] & [Audit Log]
                                        │                                   │
                                        ▼                                   ▼
[Clean Stage 1 (n=266)] ─────> [M2: Table 1 Stats] ───────────────────> [Publication Table 1]
         │
         ├──> [M3: Imbalance Benchmark (CV Folds)] ─> [Benchmark Report & Selected Strategy]
         │                                                      │
         └──> [M4: Tri-Method Feature Selection] <──────────────┘
                 (Univariate+Logit ∩ LASSO 10-fold ∩ Boruta)
                                        │
                                        ▼
                           [Consensus 3-5 Features]
                                        │
                                        ▼
              [M5: 6-Model Benchmark Training, 10-Fold CV, Calib & DCA]
                 (LR, SVM, RF, DT, XGBoost, NB + DeLong 95% CI)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [M6: Stage 2 External Validation]       [M7: SHAP & Web Calculator]
        - External Metrics & DeLong             - Global & Local SHAP Plots
        - External ROC & DCA curves             - Interactive Streamlit App
                                                - Exported Calibrated Models
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F1 | Stage 2 Ingestion & Column Harmonization | Read `第二阶段原始数据.xlsx`, map Col 48 (`ECMO时长/h` -> `ECMO时长`), parse numeric types | M1 | ORIGINAL_REQUEST §R1 |
| F2 | Clinical Exclusion Filtering & Audit Log | Exclude ECMO < 24h, age < 18, and missing key outcomes (rows 95, 99); output audit log CSV/MD | M1 | ORIGINAL_REQUEST §R1 |
| F3 | LET Target Standardization | Apply expanded regex `["下肢", "股", "大隐", "髂", "足", "小腿"]` on `血栓事件` to label `是否下肢血栓` | M1 | ORIGINAL_REQUEST §R1 |
| F4 | Stage 1 Table 1 Baseline Table | Three-line table comparing Stage 1 Thrombosis vs Non-thrombosis with mean±SD/median(IQR) and exact p-values | M2 | ORIGINAL_REQUEST §R2 |
| F5 | Stage 2 Table 1 Baseline Table | Three-line table comparing Stage 2 Thrombosis vs Non-thrombosis with exact statistical tests | M2 | ORIGINAL_REQUEST §R2 |
| F6 | Combined Cohort Table 1 Baseline Table | Three-line table comparing Stage 1 vs Stage 2, and Overall Thrombosis vs Non-thrombosis | M2 | ORIGINAL_REQUEST §R2 |
| F7 | Imbalance Method Benchmarking | Benchmark SMOTE, SMOTE+ENN, Random Under-sampling, and `class_weight='balanced'` within CV folds | M3 | ORIGINAL_REQUEST §R3 |
| F8 | Anti-Leakage Sampling Enforcement | Guarantee that resamplers are embedded exclusively within training folds via Pipeline | M3 | ORIGINAL_REQUEST §R3 |
| F9 | Imbalance Strategy Comparison Report | Tabulate Recall, Precision, F1, AUROC across sampling strategies with empirical justification | M3 | ORIGINAL_REQUEST §R3 |
| F10 | Track 1 Feature Selection | Univariate screening ($p < 0.10$), VIF collinearity check ($VIF < 10$), multivariable logistic ($p < 0.05$) | M4 | ORIGINAL_REQUEST §R4 |
| F11 | Track 2 Feature Selection | 10-fold cross-validated LASSO regression selecting non-zero coefficients at optimal lambda | M4 | ORIGINAL_REQUEST §R4 |
| F12 | Track 3 Feature Selection | Boruta random forest shadow-feature importance algorithm confirming significant predictors | M4 | ORIGINAL_REQUEST §R4 |
| F13 | Consensus Intersection Extraction | Mathematical intersection ($\text{Track 1} \cap \text{Track 2} \cap \text{Track 3}$) yielding exact 3-5 predictors | M4 | ORIGINAL_REQUEST §R4 |
| F14 | 6 Canonical Model Architectures | Implement Logistic Regression, SVM (RBF), Random Forest, Decision Tree, XGBoost, Naive Bayes | M5 | ORIGINAL_REQUEST §R5 |
| F15 | Stratified 10-Fold CV & Tuning | Hyperparameter grid search inside Stratified 10-fold CV on Stage 1 training cohort | M5 | ORIGINAL_REQUEST §R5 |
| F16 | 6 Core Clinical Metrics with DeLong CI | Compute Recall, Precision, Accuracy, F1, Specificity, AUROC with DeLong 95% CI for all 6 models | M5 | ORIGINAL_REQUEST §R5 |
| F17 | Calibration & Hosmer-Lemeshow | Probability calibration (Platt scaling / sigmoid) and Hosmer-Lemeshow goodness-of-fit test | M5 | ORIGINAL_REQUEST §R5 |
| F18 | Decision Curve Analysis (Internal) | Clinical net-benefit curves across threshold probabilities $p_t \in [0.01, 0.50]$ vs Treat All/None | M5 | ORIGINAL_REQUEST §R5 |
| F19 | Stage 2 External Validation Evaluation | Evaluate all 6 locked models on clean Stage 2 (n=106) without retraining or data leakage | M6 | ORIGINAL_REQUEST §R6 |
| F20 | External Validation Metrics & CIs | Tabulate all 6 metrics with DeLong 95% CIs and Youden cutoff for Stage 2 cohort | M6 | ORIGINAL_REQUEST §R6 |
| F21 | External ROC & DCA Curves | Generate publication-grade comparative ROC and DCA net-benefit curves for Stage 2 validation | M6 | ORIGINAL_REQUEST §R6 |
| F22 | Global & Local SHAP Interpretability | Generate SHAP beeswarm summary plot and individual waterfall attribution diagrams | M7 | ORIGINAL_REQUEST §R7 |
| F23 | Web Calculator Multi-Model Engine | Export all 6 calibrated models into `web_app/model_assets` with algorithm selection dropdown | M7 | ORIGINAL_REQUEST §R7 |
| F24 | Input Pipeline & Risk Stratification | Support consensus features, prevent double-scaling, display 3-tier risk badges & warnings | M7 | ORIGINAL_REQUEST §R7 |
| F25 | Robust Multi-Model SHAP Explainer | TreeExplainer for RF/DT/XGB, Linear for LR, KernelExplainer with precomputed background for SVM/NB | M7 | ORIGINAL_REQUEST §R7 |
| F26 | Web Calculator Verification & Testing | Automated test suite validating UI loading, model inference, SHAP generation, and HTTP 200 | M7 | ORIGINAL_REQUEST §R7 |
| F27 | Opaque-Box E2E Test Harness | Tier 1-4 requirement-driven test suite validating pipelines independently from source code | E2E | ORIGINAL_REQUEST §Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Stage 2 Ingestion, Exclusion & Schema Alignment | Clean Stage 2 (`第二阶段原始数据.xlsx`), apply exclusion rules, export `stage2_clean.csv` & audit log | none | DONE |
| M2 | Baseline Clinical Characterization (Table 1) | Publication-grade three-line Table 1 for Stage 1, Stage 2, and Combined cohorts (CSV, MD, Word) | M1 | DONE |
| M3 | Class Imbalance Benchmark & Strategy Evaluation | Benchmark SMOTE, SMOTE+ENN, Random Under-sampling, `class_weight='balanced'` in CV folds | M1 | DONE |
| M4 | Tri-Method Feature Selection & Consensus | Univariate+Logit ($p<0.10/p<0.05$), LASSO 10-fold CV, Boruta RF shadow-features; extract 3-5 features | M3 | DONE |
| M5 | 6-Model Benchmark Training, 10-Fold CV & DCA | Train 6 models (LR, SVM, RF, DT, XGB, NB) with 10-fold CV, 6 metrics with DeLong 95% CI, ROC, Calib, DCA | M4 | DONE |
| M6 | Independent External Validation on Stage 2 | Evaluate locked models on Stage 2 (n=106), tabulate 6 metrics, plot external ROC & DCA curves | M1, M5 | DONE |
| M7 | SHAP Interpretability & Web App Deployment | Multi-model SHAP plots, update `web_app`, deploy calibrated models, automated verification tests | M5, M6 | DONE |
| E2E | Independent E2E Testing Track | Build 4-tier requirement-driven validation suite; publish `TEST_READY.md`; verify final pipeline | none | PASSED (35/35) |

## Interface Contracts
### M1 ↔ M2, M3, M6 (Data Contracts)
- `stage2_clean.csv`: Clean tabular dataset with $N = 106$ rows, 153 columns, target `是否下肢血栓` (0 or 1).
- `stage2_exclusion_audit.csv` / `.md`: Detailed record of all excluded rows with patient ID, exclusion reason, and column values.
- `data/data2.csv`: Core Stage 1 dataset with $N = 266$ rows, target `是否下肢血栓` (36 positive, 230 negative).

### M3 ↔ M4, M5 (Imbalance Strategy Contract)
- `output/01_imbalance_benchmark/imbalance_benchmark_report.csv` & `.md`: Quantitative comparison of Recall, Precision, Specificity, F1, and AUROC across SMOTE, SMOTE+ENN, RandomUnderSampler, and Cost-Sensitive learning.
- Selected strategy pipeline specification: `imblearn.pipeline.Pipeline` integrating the winning sampler strictly inside CV training folds.

### M4 ↔ M5, M6, M7 (Feature Contract)
- `output/02_feature_selection/selected_features.json`:
  ```json
  {
    "track1_logistic_features": [...],
    "track2_lasso_features": [...],
    "track3_boruta_features": [...],
    "consensus_features": ["feat1", "feat2", "feat3", "..."]  // Exactly 3 to 5 features
  }
  ```
- All consensus features must exist and be fully aligned between Stage 1 (`data2.csv`) and Stage 2 (`stage2_clean.csv`).

### M5 ↔ M6, M7 (Model Contracts)
- `output/03_model_training/`:
  - Locked calibrated model files: `model_LR.joblib`, `model_SVM.joblib`, `model_RF.joblib`, `model_DT.joblib`, `model_XGB.joblib`, `model_NB.joblib`.
  - Metrics table: `03_model_performance_metrics.csv` containing:
    `Model, Dataset, Accuracy, Recall, Specificity, Precision, F1, AUROC, AUROC_95CI_Lower, AUROC_95CI_Upper, DeLong_PValue, HosmerLemeshow_PValue, Optimal_Threshold`.
  - Internal validation figures: `roc_curves_cv.png`, `calibration_curves_cv.png`, `dca_curves_cv.png`.

### M6 ↔ M7 (External Validation Contract)
- `output/04_external_validation/`:
  - `04_external_validation_metrics.csv`: External evaluation metrics for all 6 locked models on Stage 2.
  - `external_roc_curves.png` & `external_dca_curves.png`.
  - Identification of the best clinical model based on external AUROC and clinical net benefit.

### M7 ↔ Web Calculator Contract
- `web_app/model_assets/`:
  - All 6 calibrated models (`model_*.joblib`).
  - `selected_features.json` with consensus 3-5 features.
  - `preprocess_config.json` containing feature statistics.
  - `shap_background.npy` containing representative background samples for KernelExplainer.
  - `03_model_performance_metrics.csv` and `04_external_validation_metrics.csv`.
- `web_app/app.py`: Streamlit application serving real-time risk predictions and SHAP visualizations without double-scaling.

## Code Layout
- Scripts directory: `D:\acode\ECMO\scripts\`
  - `01_stage2_data_cleaning.py`: Ingestion, exclusion, target extraction, schema alignment (M1).
  - `02_generate_table1.py`: Baseline epidemiological comparison three-line tables (M2).
  - `03_benchmark_imbalance.py`: Strict CV-fold benchmark of SMOTE, SMOTE+ENN, Undersampling, Cost-sensitive (M3).
  - `04_tri_method_feature_selection.py`: Logistic ($p<0.10/0.05$), LASSO 10-fold CV, Boruta shadow RF (M4).
  - `05_train_evaluate_models.py`: 6 canonical models, 10-fold CV, DeLong 95% CI, Hosmer-Lemeshow, DCA (M5).
  - `06_external_validation.py`: Locked model evaluation on Stage 2, metrics, external ROC & DCA (M6).
  - `07_shap_and_deployment.py`: SHAP beeswarm/waterfall plots, asset export to `web_app` (M7).
- Web application: `D:\acode\ECMO\web_app\`
  - `app.py`: Main interactive Streamlit application.
  - `model_assets/`: Calibrated models, feature configurations, background data, metric summaries.
  - `requirements.txt`: Streamlit app dependencies.
- Output directory: `D:\acode\ECMO\output\`
  - `01_stage2_clean/`: Clean Stage 2 dataset and exclusion audit log.
  - `02_table1_baseline/`: Table 1 three-line tables (CSV, Markdown, DOCX).
  - `03_imbalance_benchmark/`: Imbalance benchmark metrics and comparison figures.
  - `04_feature_selection/`: Track 1-3 results and consensus intersection JSON.
  - `05_model_training/`: 6 locked model artifacts, internal CV metrics, ROC, Calibration, and DCA curves.
  - `06_external_validation/`: External validation metrics, ROC, and DCA curves.
  - `07_shap_interpretability/`: Global beeswarm summary plot and individual waterfall attribution plots.
- Test directory: `D:\acode\ECMO\tests\`
  - E2E testing framework, test runners, and automated verification scripts.
- Environment:
  - Python Executable: `D:\pythontext\.venv\Scripts\python.exe`
