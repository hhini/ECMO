# ECMO Thrombosis ML Pipeline — Master E2E Test Execution Report

**Date**: 2026-10-25 11:32:19 UTC  
**Runtime**: `D:\pythontext\.venv\Scripts\python.exe`  
**Total Elapsed Time**: 1.243s  
**Overall Status**: PASSED (READY)  

## Summary by Systematic Tier

| Tier | Description | Total | Passed | Skipped | Failed | Errors | Duration |
|---|---|---|---|---|---|---|---|
| Tier 1: Feature Coverage (F1 to F26) | Scope Verification | 21 | 21 | 0 | 0 | 0 | 1.103s |
| Tier 2: Boundary & Corner Cases (B1 to B6) | Scope Verification | 6 | 6 | 0 | 0 | 0 | 0.041s |
| Tier 3: Cross-Feature Integration & Anti-Fabrication (I1 to I4) | Scope Verification | 4 | 4 | 0 | 0 | 0 | 0.042s |
| Tier 4: Clinical Acceptance Scenarios (C1 to C4) | Scope Verification | 4 | 4 | 0 | 0 | 0 | 0.055s |
| **Overall** | **Full E2E Suite** | **35** | **35** | **0** | **0** | **0** | **1.243s** |

## Detailed Test Inventory

### Tier 1: Feature Coverage (F1 to F26)
| Status | Test ID | Description | Duration | Notes |
|---|---|---|---|---|
| `PASSED` | `test_F01_stage2_raw_ingestion_and_column_harmonization` | F1: Verify raw Excel ingestion and Column 48 ('ECMO时长/h' -> 'ECMO时长') mapping. | 0.156s | Clean verification |
| `PASSED` | `test_F02_clinical_exclusion_filtering_and_audit_log` | F2: Verify exclusion logic (ECMO < 24h, age < 18, missing outcome rows 95 & 99). | 0.085s | Clean verification |
| `PASSED` | `test_F03_let_target_standardization_regex` | F3: Verify LET target classification with expanded keywords ['下肢', '股', '大隐', '髂', '足', '小腿']. | 0.003s | Clean verification |
| `PASSED` | `test_F04_stage1_baseline_table1` | F4: Verify Stage 1 Table 1 baseline comparison (Thrombosis vs Non-thrombosis). | 0.006s | Clean verification |
| `PASSED` | `test_F05_stage2_baseline_table1` | F5: Verify Stage 2 Table 1 baseline comparison. | 0.001s | Clean verification |
| `PASSED` | `test_F06_combined_cohort_baseline_table1` | F6: Verify Combined Cohort Table 1 comparing Stage 1 vs Stage 2 and overall thrombosis. | 0.001s | Clean verification |
| `PASSED` | `test_F07_imbalance_benchmark_strategies` | F7: Verify benchmarking of SMOTE, SMOTE+ENN, Random Under-sampling, and class_weight='balanced'. | 0.002s | Clean verification |
| `PASSED` | `test_F08_anti_leakage_sampling_enforcement` | F8: Verify anti-leakage guarantee that resamplers only transform training split. | 0.075s | Clean verification |
| `PASSED` | `test_F09_imbalance_comparison_report_metrics` | F9: Verify imbalance comparison report tabulates Recall, Precision, F1, AUROC. | 0.001s | Clean verification |
| `PASSED` | `test_F10_F13_tri_method_feature_selection_and_consensus` | F10-F13: Verify 3-way feature selection and consensus count between 3 and 5. | 0.004s | Clean verification |
| `PASSED` | `test_F14_six_canonical_model_architectures_instantiation` | F14: Verify 6 canonical models (LR, SVM, RF, DT, XGBoost, Naive Bayes) can be instantiated. | 0.085s | Clean verification |
| `PASSED` | `test_F15_stratified_10_fold_cv_specification` | F15: Verify Stratified 10-Fold CV splitting properties on Stage 1 cohort. | 0.007s | Clean verification |
| `PASSED` | `test_F16_six_core_clinical_metrics_and_delong_ci` | F16: Verify calculation of Recall, Precision, Accuracy, F1, Specificity, AUROC with DeLong CI. | 0.002s | Clean verification |
| `PASSED` | `test_F17_calibration_and_hosmer_lemeshow` | F17: Verify Hosmer-Lemeshow goodness-of-fit and calibration probability bounds. | 0.000s | Clean verification |
| `PASSED` | `test_F18_decision_curve_analysis_internal` | F18: Verify DCA Net Benefit calculation and curve ranges. | 0.000s | Clean verification |
| `PASSED` | `test_F19_F21_stage2_external_validation_evaluation` | F19-F21: Verify external validation on clean Stage 2 (n=106), metrics, and ROC/DCA curves. | 0.001s | Clean verification |
| `PASSED` | `test_F22_shap_interpretability_plots` | F22: Verify SHAP beeswarm summary and waterfall plots. | 0.000s | Clean verification |
| `PASSED` | `test_F23_web_calculator_model_loading` | F23: Verify web calculator models can be loaded from model_assets. | 0.108s | Clean verification |
| `PASSED` | `test_F24_risk_stratification_tiers` | F24: Verify 3-tier risk badge classification (<20% Low, 20-50% Moderate, >=50% High). | 0.000s | Clean verification |
| `PASSED` | `test_F25_shap_explainer_mapping` | F25: Verify appropriate SHAP explainer selection for tree, linear, and kernel models. | 0.558s | Clean verification |
| `PASSED` | `test_F26_web_calculator_file_and_syntax_integrity` | F26: Verify web_app/app.py syntax and module imports without compilation errors. | 0.008s | Clean verification |

### Tier 2: Boundary & Corner Cases (B1 to B6)
| Status | Test ID | Description | Duration | Notes |
|---|---|---|---|---|
| `PASSED` | `test_B01_empty_input_dataset_handling` | B1: Verify metric engines and evaluation functions handle empty inputs gracefully. | 0.000s | Clean verification |
| `PASSED` | `test_B02_extreme_clinical_feature_boundaries` | B2: Verify model predictions on extreme physiological boundaries produce valid [0, 1] probabilities. | 0.036s | Clean verification |
| `PASSED` | `test_B03_zero_division_safeguards_in_clinical_metrics` | B3: Verify zero-division safeguards when True Positives, True Negatives, or predictions are 0. | 0.001s | Clean verification |
| `PASSED` | `test_B04_delong_structural_variance_non_negativity` | B4: Mathematical verification that DeLong structural variance is >= 0 and CI bounds are in [0, 1]. | 0.001s | Clean verification |
| `PASSED` | `test_B05_missing_value_pipeline_resilience` | B5: Verify imputation transformers handle missing cells without generating NaNs. | 0.003s | Clean verification |
| `PASSED` | `test_B06_dca_net_benefit_threshold_boundaries` | B6: Verify Decision Curve Analysis Net Benefit at boundary thresholds (pt -> 0, pt -> 1, pt = 0.5). | 0.000s | Clean verification |

### Tier 3: Cross-Feature Integration & Anti-Fabrication (I1 to I4)
| Status | Test ID | Description | Duration | Notes |
|---|---|---|---|---|
| `PASSED` | `test_I01_schema_consistency_stage1_and_stage2` | I1: Verify schema alignment between Stage 1 (train) and Stage 2 (external validation). | 0.014s | Clean verification |
| `PASSED` | `test_I02_consensus_feature_continuity` | I2: Verify consensus features are ground-traceable in Stage 1 and Stage 2 schemas. | 0.008s | Clean verification |
| `PASSED` | `test_I03_anti_fabrication_and_anti_pollution_audit` | I3: Audit all production and web scripts to guarantee zero reference to data2_synthetic.csv. | 0.016s | Clean verification |
| `PASSED` | `test_I04_anti_leakage_pipeline_encapsulation` | I4: Verify that preprocessing and resampling transformers are encapsulated inside Pipelines. | 0.003s | Clean verification |

### Tier 4: Clinical Acceptance Scenarios (C1 to C4)
| Status | Test ID | Description | Duration | Notes |
|---|---|---|---|---|
| `PASSED` | `test_C01_clinical_patient_risk_simulation` | C1: Simulate high-risk vs low-risk patient profiles and verify differential risk prediction. | 0.054s | Clean verification |
| `PASSED` | `test_C02_risk_stratification_badge_concordance` | C2: Verify 3-tier risk badges (<20% Low, 20-50% Moderate, >=50% High) and clinical color codes. | 0.000s | Clean verification |
| `PASSED` | `test_C03_clinical_screening_sensitivity_threshold` | C3: In ECMO thrombosis screening, verify threshold tuning can achieve clinical Sensitivity >= 70%. | 0.001s | Clean verification |
| `PASSED` | `test_C04_web_calculator_headless_smoke_test` | C4: Verify web app components, dependencies, and model loading execute without unhandled exceptions. | 0.000s | Clean verification |

## Scientific Integrity & Anti-Fabrication Safeguards
- [x] Zero reference to `data2_synthetic.csv` in all production and test pipelines.
- [x] Ground-truth derivation verified from raw clinical data (`第二阶段原始数据.xlsx`, `data2.csv`).
- [x] Resampling and transformation pipelines verified strictly within training folds.
- [x] Mathematical non-negativity verified for DeLong structural variance.
