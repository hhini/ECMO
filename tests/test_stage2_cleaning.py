r"""
Unit and Integration Test Suite for Milestone 1: Stage 2 Data Cleaning & Schema Alignment
Author: Worker M1
Date: 2026-10-25
Environment: D:\pythontext\.venv\Scripts\python.exe

Tests the ingestion, clinical exclusion filtering, schema harmonization, target variable
extraction, and clinical type standardization for the Stage 2 ECMO validation cohort.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from tests.fixtures import (
    PROJECT_ROOT,
    DIR_M1_CLEAN,
    STAGE2_CLEAN_N,
    STAGE2_POS_COUNT,
    STAGE2_NEG_COUNT,
    TARGET_COLUMN,
    TARGET_KEYWORDS,
    get_stage2_clean_path,
    get_stage2_audit_path,
)


class TestStage2DataCleaning(unittest.TestCase):
    """Test suite validating Milestone 1 outputs and data transformations."""

    @classmethod
    def setUpClass(cls):
        """Locate and load output artifacts."""
        cls.clean_csv_path = get_stage2_clean_path()
        cls.audit_csv_path = get_stage2_audit_path()
        cls.audit_md_path = DIR_M1_CLEAN / "stage2_exclusion_audit.md"
        cls.stage1_csv_path = PROJECT_ROOT / "data" / "data2.csv"

        assert cls.clean_csv_path is not None, "stage2_clean.csv could not be found!"
        assert cls.audit_csv_path is not None, "stage2_exclusion_audit.csv could not be found!"

        cls.df_clean = pd.read_csv(cls.clean_csv_path)
        cls.df_audit = pd.read_csv(cls.audit_csv_path)
        cls.df_stage1 = pd.read_csv(cls.stage1_csv_path)

    def test_stage2_clean_shape_and_columns_match_stage1(self):
        """Verifies clean Stage 2 dataset has shape (106, 153) and matches Stage 1 column names."""
        self.assertEqual(
            self.df_clean.shape,
            (STAGE2_CLEAN_N, 153),
            f"Expected clean cohort shape ({STAGE2_CLEAN_N}, 153), got {self.df_clean.shape}"
        )
        self.assertEqual(
            list(self.df_clean.columns),
            list(self.df_stage1.columns),
            "Stage 2 column names or order do not match Stage 1 data2.csv exactly!"
        )

    def test_stage2_column_48_renamed(self):
        """Verifies that column index 48 is 'ECMO时长' and 'ECMO时长/h' is not present."""
        self.assertEqual(self.df_clean.columns[48], "ECMO时长")
        self.assertNotIn("ECMO时长/h", self.df_clean.columns)

    def test_stage2_target_distribution_and_keywords(self):
        """Verifies target distribution has 25 positive and 81 negative, and handles edge cases."""
        target_counts = self.df_clean[TARGET_COLUMN].value_counts().to_dict()
        expected_dist = {0: STAGE2_NEG_COUNT, 1: STAGE2_POS_COUNT}
        self.assertEqual(
            target_counts,
            expected_dist,
            f"Target distribution mismatch: expected {expected_dist}, got {target_counts}"
        )

        # Check positive cases mention at least one keyword
        pos_df = self.df_clean[self.df_clean[TARGET_COLUMN] == 1]
        for _, row in pos_df.iterrows():
            event_text = str(row["血栓事件"])
            has_kw = any(kw in event_text for kw in TARGET_KEYWORDS)
            self.assertTrue(has_kw, f"Positive case does not contain LET keywords: {event_text}")

        # Check negative non-empty cases (e.g. jugular or aortic thrombi) are properly labeled 0
        neg_df = self.df_clean[self.df_clean[TARGET_COLUMN] == 0]
        non_no_negatives = neg_df[neg_df["血栓事件"] != "否"]["血栓事件"].tolist()
        self.assertIn("颈静脉血栓", non_no_negatives)
        self.assertIn("动脉血栓", non_no_negatives)

    def test_stage2_clinical_exclusion_criteria_enforced(self):
        """Verifies that 100% of included patients meet age >= 18, ECMO >= 24h, and complete outcomes."""
        self.assertTrue((self.df_clean["年龄"] >= 18).all(), "Found underage (<18) patients in clean cohort!")
        self.assertTrue((self.df_clean["ECMO时长"] >= 24.0).all(), "Found short duration (<24h) patients in clean cohort!")
        self.assertTrue(self.df_clean["血栓事件"].notna().all(), "Found missing '血栓事件' in clean cohort!")
        self.assertTrue(self.df_clean["结局"].notna().all(), "Found missing '结局' in clean cohort!")

    def test_stage2_exclusion_audit_log_completeness(self):
        """Verifies that audit log has exactly 34 records and MD report exists."""
        self.assertEqual(len(self.df_audit), 34, f"Expected 34 audit records, got {len(self.df_audit)}")
        
        cat_counts = self.df_audit["exclusion_category"].value_counts().to_dict()
        self.assertEqual(cat_counts.get("Template Empty Row", 0), 17)
        self.assertEqual(cat_counts.get("Clinical Exclusion (ECMO < 24h)", 0), 13)
        self.assertEqual(cat_counts.get("Clinical Exclusion (Age < 18)", 0), 1)
        self.assertEqual(cat_counts.get("Clinical Exclusion (Prior Thrombosis)", 0), 1)
        self.assertEqual(cat_counts.get("Missing Primary Outcome", 0), 2)

        # Check MD report exists and is non-empty
        self.assertTrue(self.audit_md_path.exists(), "Markdown audit report does not exist!")
        self.assertGreater(self.audit_md_path.stat().st_size, 1000, "Markdown audit report is suspiciously small!")

    def test_stage2_numeric_features_cast_without_string_artifacts(self):
        """Verifies laboratory and continuous features are float and do not contain string artifacts."""
        lab_base = ["DD", "FIB", "TT", "APTT", "PT", "INR", "AT3", "PLT", "WBC", "RBC"]
        lab_cols = [c for c in self.df_clean.columns if any(k in c for k in lab_base) and c != "血栓事件"]
        
        for col in lab_cols:
            self.assertIn(
                self.df_clean[col].dtype,
                [np.dtype("float64"), np.dtype("int64")],
                f"Laboratory column '{col}' is not numeric! Dtype: {self.df_clean[col].dtype}"
            )
            # Ensure no strings remain
            for val in self.df_clean[col].dropna():
                self.assertIsInstance(val, (float, int, np.floating, np.integer))


if __name__ == "__main__":
    unittest.main()
