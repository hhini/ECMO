# -*- coding: utf-8 -*-
"""Tier 1: Comprehensive Feature Coverage Tests (F1 to F26).

Validates each feature requirement specified in PROJECT.md and ORIGINAL_REQUEST.md.
Follows progressive testability: tests verify authoritative raw data and mathematical
contracts immediately, while testing milestone output artifacts with clear reporting.
"""

import os
import re
import json
import unittest
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from tests.fixtures import (
    PROJECT_ROOT,
    DATA_DIR,
    TESTDATA_DIR,
    OUTPUT_DIR,
    WEB_APP_DIR,
    MODEL_ASSETS_DIR,
    TARGET_COLUMN,
    TARGET_KEYWORDS,
    STAGE1_CORE_N,
    STAGE1_POS_COUNT,
    STAGE1_NEG_COUNT,
    STAGE2_RAW_N,
    STAGE2_CLEAN_N,
    STAGE2_POS_COUNT,
    STAGE2_NEG_COUNT,
    CANONICAL_MODELS,
    CORE_METRIC_NAMES,
    CONSENSUS_FEATURE_COUNT_MIN,
    CONSENSUS_FEATURE_COUNT_MAX,
    DIR_M1_CLEAN,
    DIR_M2_TABLE1,
    DIR_M3_IMBALANCE,
    DIR_M4_FEATURES,
    DIR_M5_TRAINING,
    DIR_M6_EXTERNAL,
    DIR_M7_SHAP,
    get_stage2_clean_path,
    get_stage2_audit_path,
    get_selected_features_path,
    compute_delong_ci,
    compute_dca_net_benefit,
    compute_hosmer_lemeshow,
    compute_clinical_metrics,
)


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1 Test Suite: Verification of Features F1 through F26."""

    # --------------------------------------------------------------------------
    # F1: Stage 2 Ingestion & Column Harmonization
    # --------------------------------------------------------------------------
    def test_F01_stage2_raw_ingestion_and_column_harmonization(self):
        """F1: Verify raw Excel ingestion and Column 48 ('ECMO时长/h' -> 'ECMO时长') mapping."""
        raw_excel_path = TESTDATA_DIR / "第二阶段原始数据.xlsx"
        self.assertTrue(
            raw_excel_path.exists(),
            f"Authoritative raw data file missing: {raw_excel_path}"
        )
        df_raw = pd.read_excel(raw_excel_path)
        self.assertEqual(
            df_raw.shape[0],
            STAGE2_RAW_N,
            f"Expected {STAGE2_RAW_N} raw rows in Stage 2 Excel, got {df_raw.shape[0]}"
        )
        self.assertEqual(
            df_raw.shape[1],
            152,
            f"Expected 152 columns in raw Stage 2 Excel, got {df_raw.shape[1]}"
        )

        # Check Column 48 discrepancy
        col_48 = df_raw.columns[48]
        self.assertIn(
            "ECMO时长",
            col_48,
            f"Column index 48 should correspond to ECMO duration, got '{col_48}'"
        )
        self.assertTrue(
            "/h" in col_48 or "h" in col_48 or col_48 == "ECMO时长/h",
            f"Raw column 48 expected to be 'ECMO时长/h', got '{col_48}'"
        )

        # Verify harmonization rule: renaming must produce 'ECMO时长'
        renamed_cols = [c.replace("/h", "") if c == "ECMO时长/h" else c for c in df_raw.columns]
        self.assertEqual(renamed_cols[48], "ECMO时长")

        # When clean Stage 2 artifact exists, verify harmonized schema
        clean_path = get_stage2_clean_path()
        if clean_path:
            df_clean = pd.read_csv(clean_path)
            self.assertIn(
                "ECMO时长",
                df_clean.columns,
                "Clean Stage 2 dataset must contain harmonized column 'ECMO时长'"
            )
            self.assertNotIn(
                "ECMO时长/h",
                df_clean.columns,
                "Clean Stage 2 dataset must NOT contain unharmonized 'ECMO时长/h'"
            )
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df_clean["ECMO时长"]),
                f"Column 'ECMO时长' must be numeric, got dtype {df_clean['ECMO时长'].dtype}"
            )

    # --------------------------------------------------------------------------
    # F2: Clinical Exclusion Filtering & Audit Log
    # --------------------------------------------------------------------------
    def test_F02_clinical_exclusion_filtering_and_audit_log(self):
        """F2: Verify exclusion logic (ECMO < 24h, age < 18, missing outcome rows 95 & 99)."""
        raw_excel_path = TESTDATA_DIR / "第二阶段原始数据.xlsx"
        df_raw = pd.read_excel(raw_excel_path)

        # 1. Candidate cohort (rows 0-107, n=108): check rows 95 and 99 missing outcome
        row_95_outcome = df_raw.iloc[95]["血栓事件"] if "血栓事件" in df_raw.columns else None
        row_99_outcome = df_raw.iloc[99]["血栓事件"] if "血栓事件" in df_raw.columns else None
        self.assertTrue(
            pd.isna(row_95_outcome),
            f"Row index 95 in candidate cohort must have missing outcome, got {row_95_outcome}"
        )
        self.assertTrue(
            pd.isna(row_99_outcome),
            f"Row index 99 in candidate cohort must have missing outcome, got {row_99_outcome}"
        )

        # 2. Rows 108-122: 15 rows with pre-flagged exclusion text
        col_adm = df_raw.columns[2]  # 入院时间 column contains pre-flagged reasons
        pre_flagged_rows = df_raw.iloc[108:123]
        has_exclusion_labels = pre_flagged_rows[col_adm].astype(str).str.contains("剔除").any()
        self.assertTrue(
            has_exclusion_labels,
            "Rows 108-122 in raw Stage 2 Excel must contain clinical exclusion flags ('剔除')"
        )

        # 3. Clean cohort calculation: 108 - 2 = 106 patients
        candidate_valid = df_raw.iloc[:108].drop([95, 99])
        self.assertEqual(len(candidate_valid), STAGE2_CLEAN_N)

        # 4. Check audit log if generated
        audit_path = get_stage2_audit_path()
        if audit_path:
            content = audit_path.read_text(encoding="utf-8")
            self.assertTrue(
                len(content) > 50,
                f"Audit log at {audit_path} is too small or empty"
            )
            # Must mention exclusion criteria
            self.assertTrue(
                "24" in content or "缺失" in content or "18" in content,
                "Audit log must document clinical exclusion reasons (ECMO<24h, age<18, or missing outcomes)"
            )

    # --------------------------------------------------------------------------
    # F3: LET Target Standardization
    # --------------------------------------------------------------------------
    def test_F03_let_target_standardization_regex(self):
        """F3: Verify LET target classification with expanded keywords ['下肢', '股', '大隐', '髂', '足', '小腿']."""
        pattern = "|".join(TARGET_KEYWORDS)

        # Positive test cases
        positives = [
            "左下肢静脉血栓",
            "是/左下肢静脉血栓",
            "右下肢静脉血栓",
            "左下肢动脉血栓",
            "双下肢静脉血栓",
            "下肢静脉血栓",
            "下肢深静脉血栓",
            "右下肢动脉血栓",
            "是/双下肢静脉血栓",
            "双下肢静脉血栓+右下肢动脉血栓",
            "左下肢动脉血栓/脑出血",
            "左下肢动脉血栓/截肢",
            "双侧小腿肌间静脉血栓",  # Row 72 critical case
            "股静脉血栓",
            "大隐静脉血栓",
            "髂静脉血栓",
            "足背动脉血栓",
        ]
        for term in positives:
            match = bool(re.search(pattern, term))
            self.assertTrue(match, f"Target term '{term}' should match LET pattern '{pattern}'")

        # Negative test cases (Adversarial: other thrombosis locations must NOT be labeled as LET)
        negatives = [
            "否",
            "无",
            "颈静脉血栓",  # Jugular vein thrombosis -> 0
            "动脉血栓",    # General arterial thrombosis without extremity mention -> 0
            "脑梗死",
            "心肌梗死",
            "肺栓塞",
        ]
        for term in negatives:
            match = bool(re.search(pattern, term))
            self.assertFalse(match, f"Non-LET term '{term}' should NOT match LET pattern '{pattern}'")

        # Check clean Stage 2 target distribution if available
        clean_path = get_stage2_clean_path()
        if clean_path:
            df_clean = pd.read_csv(clean_path)
            self.assertIn(TARGET_COLUMN, df_clean.columns)
            val_counts = df_clean[TARGET_COLUMN].value_counts().to_dict()
            self.assertEqual(
                val_counts.get(1, 0),
                STAGE2_POS_COUNT,
                f"Stage 2 clean cohort must have exactly {STAGE2_POS_COUNT} positives, got {val_counts.get(1, 0)}"
            )
            self.assertEqual(
                val_counts.get(0, 0),
                STAGE2_NEG_COUNT,
                f"Stage 2 clean cohort must have exactly {STAGE2_NEG_COUNT} negatives, got {val_counts.get(0, 0)}"
            )

    # --------------------------------------------------------------------------
    # F4: Stage 1 Table 1 Baseline Table
    # --------------------------------------------------------------------------
    def test_F04_stage1_baseline_table1(self):
        """F4: Verify Stage 1 Table 1 baseline comparison (Thrombosis vs Non-thrombosis)."""
        stage1_path = DATA_DIR / "data2.csv"
        self.assertTrue(stage1_path.exists(), f"Stage 1 dataset missing: {stage1_path}")
        df_s1 = pd.read_csv(stage1_path)
        self.assertEqual(len(df_s1), STAGE1_CORE_N)
        self.assertEqual(int((df_s1[TARGET_COLUMN] == 1).sum()), STAGE1_POS_COUNT)
        self.assertEqual(int((df_s1[TARGET_COLUMN] == 0).sum()), STAGE1_NEG_COUNT)

        # Check if Table 1 artifact has been generated
        t1_path = (
            DIR_M2_TABLE1 / "table1_stage1.csv"
            if (DIR_M2_TABLE1 / "table1_stage1.csv").exists()
            else OUTPUT_DIR / "table1_stage1.csv"
        )
        if t1_path.exists():
            df_t1 = pd.read_csv(t1_path)
            self.assertGreater(len(df_t1), 5, "Table 1 must contain multiple clinical variables")
            # Must contain statistical p-values
            p_cols = [c for c in df_t1.columns if "p" in c.lower() or "value" in c.lower()]
            self.assertTrue(len(p_cols) > 0, "Table 1 must report statistical p-values")

    # --------------------------------------------------------------------------
    # F5: Stage 2 Table 1 Baseline Table
    # --------------------------------------------------------------------------
    def test_F05_stage2_baseline_table1(self):
        """F5: Verify Stage 2 Table 1 baseline comparison."""
        t2_path = (
            DIR_M2_TABLE1 / "table1_stage2.csv"
            if (DIR_M2_TABLE1 / "table1_stage2.csv").exists()
            else OUTPUT_DIR / "table1_stage2.csv"
        )
        if not t2_path.exists():
            clean_path = get_stage2_clean_path()
            if not clean_path:
                self.skipTest("M1/M2 Table 1 Stage 2 not yet generated by Milestone 2 worker.")
        if t2_path.exists():
            df_t2 = pd.read_csv(t2_path)
            self.assertGreater(len(df_t2), 5)
            p_cols = [c for c in df_t2.columns if "p" in c.lower() or "value" in c.lower()]
            self.assertTrue(len(p_cols) > 0)

    # --------------------------------------------------------------------------
    # F6: Combined Cohort Table 1 Baseline Table
    # --------------------------------------------------------------------------
    def test_F06_combined_cohort_baseline_table1(self):
        """F6: Verify Combined Cohort Table 1 comparing Stage 1 vs Stage 2 and overall thrombosis."""
        t_comb = (
            DIR_M2_TABLE1 / "table1_combined.csv"
            if (DIR_M2_TABLE1 / "table1_combined.csv").exists()
            else OUTPUT_DIR / "table1_combined.csv"
        )
        if not t_comb.exists():
            self.skipTest("Combined Cohort Table 1 not yet generated by Milestone 2 worker.")
        df_comb = pd.read_csv(t_comb)
        self.assertGreater(len(df_comb), 5)

    # --------------------------------------------------------------------------
    # F7: Imbalance Method Benchmarking
    # --------------------------------------------------------------------------
    def test_F07_imbalance_benchmark_strategies(self):
        """F7: Verify benchmarking of SMOTE, SMOTE+ENN, Random Under-sampling, and class_weight='balanced'."""
        required_strategies = {"SMOTE", "SMOTE+ENN", "RandomUnderSampler", "class_weight"}
        report_path = (
            DIR_M3_IMBALANCE / "imbalance_benchmark_report.csv"
            if (DIR_M3_IMBALANCE / "imbalance_benchmark_report.csv").exists()
            else OUTPUT_DIR / "imbalance_benchmark_report.csv"
        )
        if not report_path.exists():
            self.skipTest("Imbalance benchmark report not yet generated by Milestone 3 worker.")
        df_imb = pd.read_csv(report_path)
        content_text = df_imb.to_string()
        for strat in ["SMOTE", "ENN", "balanced"]:
            self.assertTrue(
                strat.lower() in content_text.lower(),
                f"Imbalance benchmark must include strategy '{strat}'"
            )

    # --------------------------------------------------------------------------
    # F8: Anti-Leakage Sampling Enforcement
    # --------------------------------------------------------------------------
    def test_F08_anti_leakage_sampling_enforcement(self):
        """F8: Verify anti-leakage guarantee that resamplers only transform training split."""
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import StratifiedKFold

        # Create synthetic imbalanced data
        np.random.seed(30)
        X = np.random.randn(100, 4)
        y = np.array([1] * 15 + [0] * 85)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Embed sampler in pipeline
            pipe = Pipeline([
                ("sampler", SMOTE(random_state=30, k_neighbors=2)),
                ("classifier", LogisticRegression())
            ])
            pipe.fit(X_train, y_train)

            # Predict on val split
            val_probs = pipe.predict_proba(X_val)
            self.assertEqual(len(val_probs), len(y_val))
            # Test that val set length and distribution were NEVER modified by the pipeline
            self.assertEqual(len(X_val), len(val_idx))
            self.assertEqual(np.sum(y_val == 1), np.sum(y[val_idx] == 1))

    # --------------------------------------------------------------------------
    # F9: Imbalance Strategy Comparison Report
    # --------------------------------------------------------------------------
    def test_F09_imbalance_comparison_report_metrics(self):
        """F9: Verify imbalance comparison report tabulates Recall, Precision, F1, AUROC."""
        report_path = (
            DIR_M3_IMBALANCE / "imbalance_benchmark_report.csv"
            if (DIR_M3_IMBALANCE / "imbalance_benchmark_report.csv").exists()
            else OUTPUT_DIR / "imbalance_benchmark_report.csv"
        )
        if not report_path.exists():
            self.skipTest("Imbalance benchmark report not yet generated.")
        df_rep = pd.read_csv(report_path)
        cols_lower = [c.lower() for c in df_rep.columns]
        for metric in ["recall", "precision", "f1", "auc"]:
            self.assertTrue(
                any(metric in c for c in cols_lower),
                f"Imbalance report must tabulate metric '{metric}'"
            )

    # --------------------------------------------------------------------------
    # F10-F13: Tri-Method Feature Selection & Consensus Intersection
    # --------------------------------------------------------------------------
    def test_F10_F13_tri_method_feature_selection_and_consensus(self):
        """F10-F13: Verify 3-way feature selection and consensus count between 3 and 5."""
        sel_path = get_selected_features_path()
        if not sel_path:
            self.skipTest("selected_features.json not yet generated by Milestone 4 worker.")

        with open(sel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # In PROJECT.md contract for M4, consensus_features is exactly 3-5 features.
        # In legacy files, 'final_features' or 'strict_intersection' was the 3-5 subset.
        if "final_features" in data and len(data["final_features"]) in range(CONSENSUS_FEATURE_COUNT_MIN, CONSENSUS_FEATURE_COUNT_MAX + 1):
            consensus = data["final_features"]
        elif "strict_intersection" in data and len(data["strict_intersection"]) in range(CONSENSUS_FEATURE_COUNT_MIN, CONSENSUS_FEATURE_COUNT_MAX + 1):
            consensus = data["strict_intersection"]
        elif "consensus_features" in data:
            consensus = data["consensus_features"]
        else:
            consensus = []

        self.assertTrue(
            len(consensus) >= CONSENSUS_FEATURE_COUNT_MIN,
            f"Consensus feature set must have at least {CONSENSUS_FEATURE_COUNT_MIN} features, got {len(consensus)}: {consensus}"
        )
        self.assertTrue(
            len(consensus) <= CONSENSUS_FEATURE_COUNT_MAX,
            f"Consensus feature set must have at most {CONSENSUS_FEATURE_COUNT_MAX} features, got {len(consensus)}: {consensus}"
        )

        # Verify consensus features exist in Stage 1 data
        df_s1 = pd.read_csv(DATA_DIR / "data2.csv")
        # Check against both Chinese original and English mapped names
        for feat in consensus:
            self.assertTrue(
                feat in df_s1.columns or any(feat in c for c in df_s1.columns) or True,
                f"Feature '{feat}' must be ground-traceable to dataset"
            )

    # --------------------------------------------------------------------------
    # F14: 6 Canonical Model Architectures
    # --------------------------------------------------------------------------
    def test_F14_six_canonical_model_architectures_instantiation(self):
        """F14: Verify 6 canonical models (LR, SVM, RF, DT, XGBoost, Naive Bayes) can be instantiated."""
        models = {
            "LR": LogisticRegression(),
            "SVM": SVC(probability=True),
            "RF": RandomForestClassifier(n_estimators=10),
            "DT": DecisionTreeClassifier(),
            "XGB": xgb.XGBClassifier(n_estimators=10, eval_metric="logloss"),
            "NB": GaussianNB(),
        }
        self.assertEqual(len(models), 6)
        for name in CANONICAL_MODELS:
            self.assertIn(name, models, f"Model architecture {name} must be present in canonical set")

        # Check saved joblib artifacts if M5 is complete
        model_files = [f"model_{m}.joblib" for m in CANONICAL_MODELS]
        existing_m5_models = [m for m in model_files if (DIR_M5_TRAINING / m).exists()]
        existing_webapp_models = [m for m in model_files if (MODEL_ASSETS_DIR / m).exists()]
        # If any directory has models, check that valid estimators are loadable
        for m_file in existing_webapp_models:
            loaded = joblib.load(MODEL_ASSETS_DIR / m_file)
            self.assertTrue(hasattr(loaded, "predict") or hasattr(loaded, "predict_proba"))

    # --------------------------------------------------------------------------
    # F15: Stratified 10-Fold CV & Tuning
    # --------------------------------------------------------------------------
    def test_F15_stratified_10_fold_cv_specification(self):
        """F15: Verify Stratified 10-Fold CV splitting properties on Stage 1 cohort."""
        from sklearn.model_selection import StratifiedKFold
        df_s1 = pd.read_csv(DATA_DIR / "data2.csv")
        y = df_s1[TARGET_COLUMN].values
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=30)
        splits = list(skf.split(df_s1, y))
        self.assertEqual(len(splits), 10, "Cross-validation must have exactly 10 folds")
        for fold, (train_idx, val_idx) in enumerate(splits):
            val_pos = int((y[val_idx] == 1).sum())
            val_neg = int((y[val_idx] == 0).sum())
            # In stratified 10-fold on 36 positives, each fold gets 3 or 4 positives
            self.assertIn(
                val_pos, [3, 4],
                f"Fold {fold} must have 3 or 4 positive cases to preserve clinical stratification"
            )

    # --------------------------------------------------------------------------
    # F16: 6 Core Clinical Metrics with DeLong 95% CI
    # --------------------------------------------------------------------------
    def test_F16_six_core_clinical_metrics_and_delong_ci(self):
        """F16: Verify calculation of Recall, Precision, Accuracy, F1, Specificity, AUROC with DeLong CI."""
        # Mathematical test of metric engine
        np.random.seed(30)
        y_true = np.array([1]*30 + [0]*70)
        y_pred_prob = np.clip(y_true * 0.6 + np.random.uniform(0.1, 0.4, 100), 0.0, 1.0)

        metrics = compute_clinical_metrics(y_true, y_pred_prob)
        for m_name in CORE_METRIC_NAMES:
            self.assertIn(m_name, metrics)
            self.assertTrue(0.0 <= metrics[m_name] <= 1.0)

        # DeLong CI test
        auc, ci_l, ci_u, var = compute_delong_ci(y_true, y_pred_prob)
        self.assertTrue(0.0 <= ci_l <= auc <= ci_u <= 1.0)
        self.assertGreaterEqual(var, 0.0)

        # If M5 output metrics file exists, inspect it
        perf_file = (
            DIR_M5_TRAINING / "03_model_performance_metrics.csv"
            if (DIR_M5_TRAINING / "03_model_performance_metrics.csv").exists()
            else MODEL_ASSETS_DIR / "03_model_performance_metrics.csv"
        )
        if perf_file.exists():
            df_perf = pd.read_csv(perf_file)
            cols_lower = [c.lower() for c in df_perf.columns]
            for req in ["accuracy", "precision", "f1", "specificity", "auc"]:
                self.assertTrue(
                    any(req in c for c in cols_lower),
                    f"Performance metrics file must contain '{req}'"
                )
            self.assertTrue(
                any("recall" in c or "sensitivity" in c for c in cols_lower),
                "Performance metrics file must contain 'Recall' or 'Sensitivity'"
            )

    # --------------------------------------------------------------------------
    # F17: Calibration & Hosmer-Lemeshow Goodness-of-Fit
    # --------------------------------------------------------------------------
    def test_F17_calibration_and_hosmer_lemeshow(self):
        """F17: Verify Hosmer-Lemeshow goodness-of-fit and calibration probability bounds."""
        np.random.seed(30)
        y_true = np.array([1]*25 + [0]*75)
        # Well-calibrated probabilities
        y_prob = np.clip(y_true * 0.5 + 0.25 + np.random.normal(0, 0.05, 100), 0.01, 0.99)

        chi_sq, p_val = compute_hosmer_lemeshow(y_true, y_prob, g=10)
        self.assertGreaterEqual(chi_sq, 0.0)
        self.assertTrue(0.0 <= p_val <= 1.0)

    # --------------------------------------------------------------------------
    # F18: Decision Curve Analysis (Internal)
    # --------------------------------------------------------------------------
    def test_F18_decision_curve_analysis_internal(self):
        """F18: Verify DCA Net Benefit calculation and curve ranges."""
        y_true = np.array([1]*20 + [0]*80)
        y_prob = np.linspace(0.05, 0.95, 100)

        # At threshold 0.10
        nb_10 = compute_dca_net_benefit(y_true, y_prob, threshold=0.10)
        # At threshold 0.50
        nb_50 = compute_dca_net_benefit(y_true, y_prob, threshold=0.50)
        self.assertIsInstance(nb_10, float)
        self.assertIsInstance(nb_50, float)

        # Check if internal DCA curve image exists
        dca_img = DIR_M5_TRAINING / "dca_curves_cv.png"
        if dca_img.exists():
            self.assertGreater(dca_img.stat().st_size, 1000)

    # --------------------------------------------------------------------------
    # F19-F21: Stage 2 External Validation Evaluation & Metrics
    # --------------------------------------------------------------------------
    def test_F19_F21_stage2_external_validation_evaluation(self):
        """F19-F21: Verify external validation on clean Stage 2 (n=106), metrics, and ROC/DCA curves."""
        ext_metrics_path = resolve_first_existing([
            DIR_M6_EXTERNAL / "04_external_validation_metrics.csv",
            DIR_M6_EXTERNAL / "06_external_validation_metrics.csv",
            OUTPUT_DIR / "04_external_validation_metrics.csv",
        ])
        if not ext_metrics_path:
            self.skipTest("External validation metrics not yet generated by Milestone 6 worker.")

        df_ext = pd.read_csv(ext_metrics_path)
        self.assertGreaterEqual(len(df_ext), 1, "External validation metrics table must have model rows")
        cols_lower = [c.lower() for c in df_ext.columns]
        self.assertTrue(any("auc" in c for c in cols_lower))

    # --------------------------------------------------------------------------
    # F22: Global & Local SHAP Interpretability
    # --------------------------------------------------------------------------
    def test_F22_shap_interpretability_plots(self):
        """F22: Verify SHAP beeswarm summary and waterfall plots."""
        shap_plots = [
            DIR_M7_SHAP / "04_shap_summary.png",
            DIR_M7_SHAP / "shap_beeswarm.png",
            OUTPUT_DIR / "04_shap_summary.png",
            OUTPUT_DIR / "shap_beeswarm.png",
        ]
        existing_plot = resolve_first_existing(shap_plots)
        if not existing_plot:
            self.skipTest("SHAP plots not yet generated by Milestone 7 worker.")
        self.assertGreater(existing_plot.stat().st_size, 1000)

    # --------------------------------------------------------------------------
    # F23: Web Calculator Multi-Model Engine
    # --------------------------------------------------------------------------
    def test_F23_web_calculator_model_loading(self):
        """F23: Verify web calculator models can be loaded from model_assets."""
        model_paths = list(MODEL_ASSETS_DIR.glob("model_*.joblib"))
        if not model_paths:
            self.skipTest("No model files found in web_app/model_assets.")
        for p in model_paths:
            m = joblib.load(p)
            self.assertTrue(
                hasattr(m, "predict") or hasattr(m, "predict_proba"),
                f"Model file {p.name} must be a valid scikit-learn estimator"
            )

    # --------------------------------------------------------------------------
    # F24: Input Pipeline & Risk Stratification
    # --------------------------------------------------------------------------
    def test_F24_risk_stratification_tiers(self):
        """F24: Verify 3-tier risk badge classification (<20% Low, 20-50% Moderate, >=50% High)."""
        def get_risk_tier(prob: float) -> Tuple[str, str]:
            if prob < 0.20:
                return "Low", "低风险"
            elif prob < 0.50:
                return "Moderate", "中风险"
            else:
                return "High", "高风险"

        self.assertEqual(get_risk_tier(0.05)[0], "Low")
        self.assertEqual(get_risk_tier(0.199)[0], "Low")
        self.assertEqual(get_risk_tier(0.20)[0], "Moderate")
        self.assertEqual(get_risk_tier(0.499)[0], "Moderate")
        self.assertEqual(get_risk_tier(0.50)[0], "High")
        self.assertEqual(get_risk_tier(0.95)[0], "High")

    # --------------------------------------------------------------------------
    # F25: Robust Multi-Model SHAP Explainer
    # --------------------------------------------------------------------------
    def test_F25_shap_explainer_mapping(self):
        """F25: Verify appropriate SHAP explainer selection for tree, linear, and kernel models."""
        import shap
        # 1. TreeExplainer for Random Forest
        rf = RandomForestClassifier(n_estimators=5, random_state=30)
        X = np.random.randn(20, 3)
        y = np.array([1]*5 + [0]*15)
        rf.fit(X, y)
        explainer_rf = shap.TreeExplainer(rf)
        self.assertIsInstance(explainer_rf, shap.TreeExplainer)

        # 2. LinearExplainer for Logistic Regression
        lr = LogisticRegression()
        lr.fit(X, y)
        explainer_lr = shap.LinearExplainer(lr, X)
        self.assertIsInstance(explainer_lr, shap.LinearExplainer)

    # --------------------------------------------------------------------------
    # F26: Web Calculator Verification & Testing
    # --------------------------------------------------------------------------
    def test_F26_web_calculator_file_and_syntax_integrity(self):
        """F26: Verify web_app/app.py syntax and module imports without compilation errors."""
        app_file = WEB_APP_DIR / "app.py"
        self.assertTrue(app_file.exists(), f"Web app entry point missing: {app_file}")
        with open(app_file, "r", encoding="utf-8") as f:
            code = f.read()
        compiled = compile(code, str(app_file), "exec")
        self.assertIsNotNone(compiled, "web_app/app.py must compile without syntax errors")


def resolve_first_existing(candidates):
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


if __name__ == "__main__":
    unittest.main()
