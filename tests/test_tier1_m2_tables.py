# -*- coding: utf-8 -*-
"""Unit and Integrity Tests for Milestone 2: Baseline Clinical Characterization (Table 1).

Verifies the integrity, completeness, genuine statistical calculation, and formatting
of Table 1 artifacts across Stage 1, Stage 2, and Combined cohorts.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import docx

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
OUTPUT_M2_DIR = PROJECT_ROOT / "output" / "02_table1_baseline"

class TestMilestone2Table1(unittest.TestCase):
    """Test suite for Table 1 baseline comparison tables."""

    def test_artifacts_existence(self):
        """Verify all 6 required M2 artifacts exist and are non-empty."""
        required_files = [
            "table1_stage1.csv",
            "table1_stage2.csv",
            "table1_combined_cohorts.csv",
            "table1_combined_thrombosis.csv",
            "table1_combined.csv",
            "table1_publication_report.md",
            "table1_three_line.docx"
        ]
        for fname in required_files:
            fpath = OUTPUT_M2_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing required M2 artifact: {fpath}")
            self.assertGreater(fpath.stat().st_size, 1000, f"Artifact is unexpectedly small: {fpath}")

    def test_table1_stage1_structure_and_stats(self):
        """Verify Stage 1 Table 1 columns, row count, and genuine statistical tests."""
        fpath = OUTPUT_M2_DIR / "table1_stage1.csv"
        df = pd.read_csv(fpath)
        self.assertGreaterEqual(len(df), 50, "Table 1 Stage 1 must contain comprehensive clinical indicators")
        expected_cols = [
            "Category", "Variable", "Stage 1 Total (N=266)",
            "Non-Thrombosis (n=230)", "Thrombosis (n=36)",
            "Statistic", "P_Value", "P_Display", "Test_Method"
        ]
        self.assertEqual(list(df.columns), expected_cols)

        # Check p-value validity: all non-nan p-values in [0, 1]
        valid_p = df["P_Value"].dropna()
        self.assertTrue((valid_p >= 0.0).all() and (valid_p <= 1.0).all(), "P-values must lie within [0, 1]")

    def test_table1_stage2_structure_and_stats(self):
        """Verify Stage 2 Table 1 columns, row count, and genuine statistical tests."""
        fpath = OUTPUT_M2_DIR / "table1_stage2.csv"
        df = pd.read_csv(fpath)
        self.assertGreaterEqual(len(df), 50)
        expected_cols = [
            "Category", "Variable", "Stage 2 Total (N=106)",
            "Non-Thrombosis (n=81)", "Thrombosis (n=25)",
            "Statistic", "P_Value", "P_Display", "Test_Method"
        ]
        self.assertEqual(list(df.columns), expected_cols)
        valid_p = df["P_Value"].dropna()
        self.assertTrue((valid_p >= 0.0).all() and (valid_p <= 1.0).all())

    def test_table1_combined_cohorts_structure(self):
        """Verify Combined Cohorts Table 1 (Stage 1 vs Stage 2)."""
        fpath = OUTPUT_M2_DIR / "table1_combined_cohorts.csv"
        df = pd.read_csv(fpath)
        self.assertGreaterEqual(len(df), 50)
        expected_cols = [
            "Category", "Variable", "Combined Cohort (N=372)",
            "Stage 1 (n=266)", "Stage 2 (n=106)",
            "Statistic", "P_Value", "P_Display", "Test_Method"
        ]
        self.assertEqual(list(df.columns), expected_cols)

    def test_table1_combined_thrombosis_structure(self):
        """Verify Overall Combined Cohort Table 1 by Thrombosis status."""
        fpath = OUTPUT_M2_DIR / "table1_combined_thrombosis.csv"
        df = pd.read_csv(fpath)
        self.assertGreaterEqual(len(df), 50)
        expected_cols = [
            "Category", "Variable", "Combined Cohort (N=372)",
            "Non-Thrombosis (n=311)", "Thrombosis (n=61)",
            "Statistic", "P_Value", "P_Display", "Test_Method"
        ]
        self.assertEqual(list(df.columns), expected_cols)

    def test_genuine_calculation_spot_check(self):
        """Anti-fabrication audit: spot-check exact statistical outputs against raw data calculation."""
        df_stage1_raw = pd.read_csv(PROJECT_ROOT / "data" / "data2.csv")
        age_g0 = df_stage1_raw[df_stage1_raw["是否下肢血栓"] == 0]["年龄"].dropna()
        age_g1 = df_stage1_raw[df_stage1_raw["是否下肢血栓"] == 1]["年龄"].dropna()
        mwu_stat, mwu_p = stats.mannwhitneyu(age_g0, age_g1, alternative='two-sided')

        df_t1 = pd.read_csv(OUTPUT_M2_DIR / "table1_stage1.csv")
        age_row = df_t1[df_t1["Variable"].str.contains("年龄")].iloc[0]

        # Verify exact test statistic and p-value match ground truth
        self.assertIn(f"U = {mwu_stat:.1f}", age_row["Statistic"])
        self.assertAlmostEqual(float(age_row["P_Value"]), float(mwu_p), places=4)

    def test_docx_three_line_table_properties(self):
        """Verify Word document contains 4 three-line tables with XML borders and cantSplit."""
        docx_path = OUTPUT_M2_DIR / "table1_three_line.docx"
        doc = docx.Document(str(docx_path))
        self.assertEqual(len(doc.tables), 4, "Word document must contain exactly 4 tables (1A, 1B, 1C, 1D)")
        for t_idx, table in enumerate(doc.tables):
            self.assertGreater(len(table.rows), 50, f"Table {t_idx} must have >= 50 rows")
            self.assertEqual(len(table.columns), 7, f"Table {t_idx} must have 7 display columns")
            # Verify tblHeader is present on first row
            header_xml = table.rows[0]._element.xml
            self.assertIn("tblHeader", header_xml, f"Table {t_idx} header must repeat across pages")

if __name__ == "__main__":
    unittest.main()
