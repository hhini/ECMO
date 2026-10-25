# -*- coding: utf-8 -*-
r"""
Adversarial Stress Test Suite for Milestone 1 (Stage 2 Data Cleaning & Schema Alignment)
Author: Challenger 1 (Empirical Challenger)
Environment: D:\pythontext\.venv\Scripts\python.exe

Tests data filtering partitions, target NLP matching, schema integrity, and ML modeling readiness.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
RAW_STAGE2_PATH = PROJECT_ROOT / "testdata" / "第二阶段原始数据.xlsx"
CLEAN_STAGE2_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
AUDIT_CSV_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_exclusion_audit.csv"
AUDIT_MD_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_exclusion_audit.md"
STAGE1_DATA2_PATH = PROJECT_ROOT / "data" / "data2.csv"

LET_KEYWORDS = ["下肢", "股", "大隐", "髂", "足", "小腿"]


class TestMilestone1AdversarialChallenge(unittest.TestCase):
    """Rigorous empirical challenger test suite for Milestone 1."""

    @classmethod
    def setUpClass(cls):
        assert RAW_STAGE2_PATH.exists(), f"Raw Excel missing: {RAW_STAGE2_PATH}"
        assert CLEAN_STAGE2_PATH.exists(), f"Clean CSV missing: {CLEAN_STAGE2_PATH}"
        assert AUDIT_CSV_PATH.exists(), f"Audit CSV missing: {AUDIT_CSV_PATH}"
        assert STAGE1_DATA2_PATH.exists(), f"Stage 1 CSV missing: {STAGE1_DATA2_PATH}"

        cls.df_raw = pd.read_excel(RAW_STAGE2_PATH, sheet_name=0)
        cls.df_clean = pd.read_csv(CLEAN_STAGE2_PATH)
        cls.df_audit = pd.read_csv(AUDIT_CSV_PATH)
        cls.df_stage1 = pd.read_csv(STAGE1_DATA2_PATH)

    # ==========================================================================
    # 1. EXCLUSION FILTERING PARTITION & BOUNDARY PROBING
    # ==========================================================================
    def test_strict_cohort_partition(self):
        """Verifies that raw (140) = clean (106) + audit (34) with zero overlap."""
        self.assertEqual(len(self.df_raw), 140, "Raw row count mismatch!")
        self.assertEqual(len(self.df_clean), 106, "Clean cohort count mismatch!")
        self.assertEqual(len(self.df_audit), 34, "Audit count mismatch!")

        raw_indices = set(range(len(self.df_raw)))
        audit_raw_indices = set(self.df_audit["raw_index"].tolist())
        
        # Verify that excluded indices are exactly 34 unique indices
        self.assertEqual(len(audit_raw_indices), 34, "Audit raw_index contains duplicates!")
        self.assertTrue(audit_raw_indices.issubset(raw_indices), "Audit contains out-of-bounds indices!")

        # Candidate clean indices
        clean_raw_indices = raw_indices - audit_raw_indices
        self.assertEqual(len(clean_raw_indices), 106, "Clean index set is not exactly 106!")

        # Verify all 106 clean patients correspond to clean_raw_indices in df_raw
        raw_clean_pids = [str(self.df_raw.iloc[i]["编号"]) for i in clean_raw_indices]
        clean_pids = [str(pid) for pid in self.df_clean["编号"]]
        self.assertEqual(sorted(raw_clean_pids), sorted(clean_pids), "Clean patient IDs diverge from unexcluded raw rows!")

    def test_every_excluded_row_justification(self):
        """Adversarially probes all 34 excluded rows to ensure valid justification."""
        for _, row in self.df_audit.iterrows():
            raw_idx = int(row["raw_index"])
            raw_row = self.df_raw.iloc[raw_idx]
            cat = row["exclusion_category"]

            if cat == "Template Empty Row":
                # Must have >= 140 missing features out of 152
                null_count = raw_row.isna().sum()
                self.assertGreaterEqual(null_count, 140, f"Template row {raw_idx} has too many non-null values ({152-null_count})!")
                self.assertTrue(pd.isna(raw_row["血栓事件"]), f"Template row {raw_idx} has non-null 血栓事件!")
            elif "Clinical Exclusion" in cat:
                # Must have admission time flag or severe missingness
                flag = str(raw_row["入院时间"])
                has_exclusion_flag = any(k in flag for k in ["18", "24", "既往", "剔除"])
                self.assertTrue(has_exclusion_flag, f"Clinical exclusion row {raw_idx} lacks explicit exclusion flag in 入院时间: {flag}")
            elif cat == "Missing Primary Outcome":
                # Must be missing both thrombosis event and clinical outcome
                self.assertTrue(pd.isna(raw_row["血栓事件"]), f"Row {raw_idx} excluded for missing outcome but 血栓事件 is not NaN!")
                self.assertTrue(pd.isna(raw_row["结局"]), f"Row {raw_idx} excluded for missing outcome but 结局 is not NaN!")
            else:
                self.fail(f"Unrecognized exclusion category: {cat}")

    def test_clean_cohort_inclusion_criteria_zero_defect(self):
        """Probes all 106 clean cases for zero violation of Stage 1 clinical inclusion criteria."""
        # Age >= 18
        self.assertTrue((self.df_clean["年龄"] >= 18.0).all(), "Clean cohort contains pediatric (<18) patients!")
        self.assertFalse(self.df_clean["年龄"].isna().any(), "Clean cohort contains missing age!")

        # ECMO Duration >= 24h
        self.assertTrue((self.df_clean["ECMO时长"] >= 24.0).all(), "Clean cohort contains ECMO duration < 24h!")
        self.assertFalse(self.df_clean["ECMO时长"].isna().any(), "Clean cohort contains missing ECMO duration!")

        # No prior thrombosis history
        self.assertTrue(((self.df_clean["既往血栓史"] == "无") | (self.df_clean["既往血栓史"].isna())).all(),
                        "Clean cohort contains patients with prior thrombosis!")

        # Complete outcomes
        self.assertFalse(self.df_clean["血栓事件"].isna().any(), "Clean cohort contains missing 血栓事件!")
        self.assertFalse(self.df_clean["结局"].isna().any(), "Clean cohort contains missing 结局!")
        self.assertFalse(self.df_clean["是否下肢血栓"].isna().any(), "Clean cohort contains missing 是否下肢血栓!")

    # ==========================================================================
    # 2. TARGET NLP & KEYWORD MATCHING STRESS TESTS
    # ==========================================================================
    def test_target_classification_empirical_audit(self):
        """Audits every single row's '血栓事件' in clean cohort against '是否下肢血栓'."""
        for idx, row in self.df_clean.iterrows():
            evt = str(row["血栓事件"]).strip()
            target = row["是否下肢血栓"]

            if any(kw in evt for kw in LET_KEYWORDS):
                self.assertEqual(target, 1, f"Row {idx} ('{evt}') contains LET keyword but target is {target}!")
            else:
                self.assertEqual(target, 0, f"Row {idx} ('{evt}') lacks LET keyword but target is {target}!")

        # Exact target counts
        self.assertEqual(int(self.df_clean["是否下肢血栓"].sum()), 25, "Positive LET count is not 25!")
        self.assertEqual(int((self.df_clean["是否下肢血栓"] == 0).sum()), 81, "Negative LET count is not 81!")

    def test_adversarial_synthetic_keyword_matrix(self):
        """Tests the keyword classifier against adversarial synthetic thrombosis strings."""
        def classify(val):
            if pd.isna(val) or str(val).strip() == "否":
                return 0
            if any(kw in str(val).strip() for kw in LET_KEYWORDS):
                return 1
            return 0

        # Adversarial Lower Extremity Positives (must be 1)
        synthetic_positives = [
            "左下肢深静脉血栓",
            "右侧股深静脉血栓",
            "双下肢动脉栓塞坏死",
            "大隐静脉血栓形成",
            "右侧髂外动脉血栓",
            "左足背动脉血栓",
            "双侧小腿肌间静脉血栓",
            "股动静脉血栓"
        ]
        for s in synthetic_positives:
            self.assertEqual(classify(s), 1, f"Adversarial positive '{s}' failed to match!")

        # Adversarial Non-Lower Extremity Negatives (must be 0)
        synthetic_negatives = [
            "否",
            "无",
            "颈静脉血栓",
            "右侧颈内静脉血栓",
            "锁骨下静脉血栓",
            "上肢深静脉血栓",
            "左心房血栓",
            "膜肺血栓",
            "管路血栓",
            "肺动脉血栓",
            "肠系膜上动脉血栓",
            "脑梗死/脑血栓",
            "门静脉血栓"
        ]
        for s in synthetic_negatives:
            self.assertEqual(classify(s), 0, f"Adversarial negative '{s}' incorrectly labeled as positive!")

    # ==========================================================================
    # 3. SCHEMA ALIGNMENT & DTYPE INTEGRITY
    # ==========================================================================
    def test_schema_names_and_order_exact_match(self):
        """Verifies 100% exact match of 153 column names and order against Stage 1."""
        self.assertEqual(
            list(self.df_clean.columns),
            list(self.df_stage1.columns),
            "Clean Stage 2 columns do not match Stage 1 data2.csv!"
        )
        self.assertEqual(self.df_clean.columns[48], "ECMO时长", "Column 48 is not ECMO时长!")
        self.assertEqual(self.df_clean.columns[-1], "是否下肢血栓", "Last column is not 是否下肢血栓!")

    def test_numeric_columns_convertibility_and_zero_string_traps(self):
        """Verifies all 123 numeric columns are pure numeric dtypes with no string objects."""
        categorical_object_cols = {
            '入院时间', '性别', '居住地', '工作状态', '职业', '吸烟', '饮酒', '妊娠状态',
            'ECMO前意识', '首要诊断', '次要诊断', '基础疾病', '既往血栓史', '既往血管性疾病史',
            '既往手术史', '家族血管性遗传病史', 'ECMO前手术史', '自体血回输', 'ECMO治疗期间意识',
            '静脉输注营养制剂使用', '联合血液净化治疗', 'ECMO模式', '置管部位', '置管管径',
            '模式转换', '多次置管', '侧支循环', '血型', '血栓事件', '结局'
        }
        numeric_cols = [c for c in self.df_clean.columns if c not in categorical_object_cols]
        self.assertEqual(len(numeric_cols), 123, f"Expected 123 numeric columns, got {len(numeric_cols)}")

        for col in numeric_cols:
            self.assertIn(
                self.df_clean[col].dtype,
                [np.dtype("float64"), np.dtype("int64")],
                f"Column '{col}' is not numeric dtype: {self.df_clean[col].dtype}"
            )
            # Ensure no infinite values
            self.assertFalse(np.isinf(self.df_clean[col]).any(), f"Column '{col}' contains inf/-inf!")

    def test_scikit_learn_pipeline_consumption(self):
        """Simulates machine learning ingestion of candidate Day 0/Day 2 features."""
        candidate_features = [
            "年龄", "BMI", "ECMO时长", "上机前DD", "PLT", "WBC", "FIB", "APTT",
            "上机第二天DD", "PLT.1", "WBC.1", "FIB.1", "APTT.1"
        ]
        X = self.df_clean[candidate_features]
        y = self.df_clean["是否下肢血栓"]

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42))
        ])

        try:
            pipeline.fit(X, y)
            preds = pipeline.predict_proba(X)[:, 1]
            self.assertEqual(len(preds), 106, "Prediction output length mismatch!")
            self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all(), "Invalid probability range!")
        except Exception as e:
            self.fail(f"Scikit-learn pipeline crashed on clean Stage 2 data: {e}")


if __name__ == "__main__":
    unittest.main()
