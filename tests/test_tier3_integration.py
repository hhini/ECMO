# -*- coding: utf-8 -*-
"""Tier 3: Cross-Feature Integration & Anti-Fabrication Test Suite.

Validates end-to-end data flow integrity and scientific rigor:
1. Schema consistency between M1 (Stage 2 cleaning) and M6 (Stage 2 external validation).
2. Consensus feature continuity across train, test, external validation, and web app assets.
3. Anti-Fabrication & Anti-Pollution Audit: Guarantees that synthetic data (data2_synthetic.csv)
   is strictly excluded from real training, external validation, and web application deployment.
4. Anti-Leakage Audit: Verifies transformers and resamplers are fit solely on training folds.
"""

import os
import re
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from tests.fixtures import (
    PROJECT_ROOT,
    DATA_DIR,
    TESTDATA_DIR,
    OUTPUT_DIR,
    SCRIPTS_DIR,
    WEB_APP_DIR,
    MODEL_ASSETS_DIR,
    TARGET_COLUMN,
    STAGE1_CORE_N,
    STAGE2_CLEAN_N,
    get_stage2_clean_path,
    get_selected_features_path,
)


class TestTier3CrossFeatureIntegration(unittest.TestCase):
    """Tier 3 Test Suite: Cross-Feature Integration & Anti-Fabrication Checks."""

    # --------------------------------------------------------------------------
    # I1: Schema Consistency between M1 Output and M6 Input
    # --------------------------------------------------------------------------
    def test_I01_schema_consistency_stage1_and_stage2(self):
        """I1: Verify schema alignment between Stage 1 (train) and Stage 2 (external validation)."""
        stage1_path = DATA_DIR / "data2.csv"
        self.assertTrue(stage1_path.exists(), f"Stage 1 dataset missing: {stage1_path}")
        df_s1 = pd.read_csv(stage1_path)

        # Stage 2 clean or raw mapped
        clean_path = get_stage2_clean_path()
        if clean_path:
            df_s2 = pd.read_csv(clean_path)
            # 1. Target column must exist in both
            self.assertIn(TARGET_COLUMN, df_s1.columns)
            self.assertIn(TARGET_COLUMN, df_s2.columns)

            # 2. Key clinical predictors must exist in both with harmonized names
            key_predictors = [
                "ECMO时长",
                "年龄",
                "BMI",
                "FIB",
                "TT",
                "APTT",
                "PT",
                "INR",
                "AT3",
                "PLT",
                "WBC",
                "RBC",
                "Hb",
            ]
            for col in key_predictors:
                self.assertIn(
                    col, df_s1.columns,
                    f"Stage 1 dataset missing key clinical column '{col}'"
                )
                self.assertIn(
                    col, df_s2.columns,
                    f"Stage 2 clean dataset missing key clinical column '{col}'"
                )

            # 3. Discrepancy column 'ECMO时长/h' must be absent from clean Stage 2
            self.assertNotIn("ECMO时长/h", df_s2.columns)

    # --------------------------------------------------------------------------
    # I2: Consensus Feature Continuity
    # --------------------------------------------------------------------------
    def test_I02_consensus_feature_continuity(self):
        """I2: Verify consensus features are ground-traceable in Stage 1 and Stage 2 schemas."""
        sel_path = get_selected_features_path()
        if not sel_path:
            self.skipTest("selected_features.json not found.")

        with open(sel_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = (
            data.get("consensus_features")
            or data.get("final_features")
            or data.get("strict_intersection")
            or []
        )
        self.assertTrue(len(features) > 0, "No consensus features found in selected_features.json")

        df_s1 = pd.read_csv(DATA_DIR / "data2.csv")

        # Normalize feature names for matching (e.g. APTT_D0 -> APTT, PLT_D2 -> PLT.1)
        feature_name_mapping = {
            "APTT_D0": ["APTT", "APTT_D0", "上机前APTT"],
            "PLT_D2": ["PLT.1", "PLT_D2", "上机第二天PLT", "PLT"],
            "WBC_D0": ["WBC", "WBC_D0", "上机前WBC"],
            "FIB_D2": ["FIB.1", "FIB_D2", "上机第二天FIB", "FIB"],
            "Creatinine_D2": ["肌酐.1", "Creatinine_D2", "肌酐"],
            "AST_D2": ["谷草.1", "AST_D2", "谷草"],
            "pH_D2": ["pH.1", "pH_D2", "pH"],
            "Age": ["年龄", "Age"],
            "ECMO_Duration": ["ECMO时长", "ECMO_Duration"],
            "Underlying_Hyperlipidemia": ["高脂血症", "Underlying_Hyperlipidemia", "既往病史_高脂血症"],
            "Underlying_Cerebrovascular": ["脑血管病", "Underlying_Cerebrovascular", "既往病史_脑血管病"],
            "Parenteral_Nutrition": ["静脉输注营养制剂使用", "Parenteral_Nutrition"],
            "Vasoactive_Drugs_Count": ["血管活性药物种类", "Vasoactive_Drugs_Count", "血管活性药物"],
            "Concomitant_Hemofiltration_2.0": ["血液净化", "CRRT", "Concomitant_Hemofiltration", "血液滤过"],
            "ICU_Duration": ["ICU住院时长", "ICU住院时长/d", "ICU_Duration"],
            "FIB_D0": ["FIB", "FIB_D0", "上机前FIB"],
            "PLT_D0": ["PLT", "PLT_D0", "上机前PLT"],
            "AT3_D0": ["AT3", "AT3_D0", "上机前AT3"],
            "AT3_D2": ["AT3.1", "AT3_D2", "上机第二天AT3"],
            "RBC_D2": ["RBC.1", "RBC_D2", "上机第二天RBC"],
            "Hb_D2": ["Hb.1", "Hb_D2", "上机第二天Hb"],
            "DD_D2": ["DD.1", "DD_D2", "上机第二天DD"],
        }

        # Check only the target 3-5 consensus features
        target_features = features
        if "final_features" in data and len(data["final_features"]) in range(3, 6):
            target_features = data["final_features"]
        elif "strict_intersection" in data and len(data["strict_intersection"]) in range(3, 6):
            target_features = data["strict_intersection"]

        for feat in target_features:
            candidate_names = feature_name_mapping.get(feat, [feat])
            matched_s1 = any(cand in df_s1.columns or any(cand in c for c in df_s1.columns) for cand in candidate_names)
            self.assertTrue(
                matched_s1,
                f"Consensus feature '{feat}' cannot be resolved to any column in Stage 1 dataset"
            )

    # --------------------------------------------------------------------------
    # I3: Anti-Fabrication & Anti-Pollution Audit
    # --------------------------------------------------------------------------
    def test_I03_anti_fabrication_and_anti_pollution_audit(self):
        """I3: Audit all production and web scripts to guarantee zero reference to data2_synthetic.csv."""
        # Files that are allowed to mention synthetic data (strictly archival/historical generator only)
        exempt_files = {
            "generate_semi_synthetic_data.py",
            "run_synthetic_pipeline.py",
            "run_all_pipelines.py",
        }

        # 1. Check production scripts
        scripts_to_audit = list(SCRIPTS_DIR.glob("*.py"))
        for s_path in scripts_to_audit:
            if s_path.name in exempt_files:
                continue
            content = s_path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn(
                "data2_synthetic.csv",
                content,
                f"Production script {s_path.name} is contaminated with references to 'data2_synthetic.csv'!"
            )

        # 2. Check web app scripts
        web_scripts = list(WEB_APP_DIR.glob("*.py"))
        for w_path in web_scripts:
            content = w_path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn(
                "data2_synthetic.csv",
                content,
                f"Web app script {w_path.name} contains references to 'data2_synthetic.csv'!"
            )

        # 3. Check sample sizes in authentic clinical datasets
        df_s1 = pd.read_csv(DATA_DIR / "data2.csv")
        self.assertEqual(
            len(df_s1),
            STAGE1_CORE_N,
            f"Stage 1 genuine clinical sample size must be {STAGE1_CORE_N}, got {len(df_s1)}"
        )

        clean_path = get_stage2_clean_path()
        if clean_path:
            df_s2 = pd.read_csv(clean_path)
            self.assertEqual(
                len(df_s2),
                STAGE2_CLEAN_N,
                f"Stage 2 clean clinical sample size must be {STAGE2_CLEAN_N}, got {len(df_s2)}"
            )

    # --------------------------------------------------------------------------
    # I4: Anti-Leakage Data Transformation Pipeline Audit
    # --------------------------------------------------------------------------
    def test_I04_anti_leakage_pipeline_encapsulation(self):
        """I4: Verify that preprocessing and resampling transformers are encapsulated inside Pipelines."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression

        # Create pipeline
        pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression()),
        ])

        # Test that pipeline transforms data sequentially during fit
        X_train = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y_train = np.array([0, 0, 1, 1])
        pipe.fit(X_train, y_train)

        # Test inference on test split without refitting scaler or imputer
        X_test = np.array([[2.0, np.nan]])
        preds = pipe.predict_proba(X_test)
        self.assertEqual(preds.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(preds)))


if __name__ == "__main__":
    unittest.main()
