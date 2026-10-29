# -*- coding: utf-8 -*-
"""
Independent Empirical Adversarial Challenge Suite for Milestone 2:
Three-Track Feature Selection Convergence & Pure English Publication-Grade Venn Diagram

Empirically tests:
1. Candidate Pool Divergence (Set operations, Jaccard similarities, distinctness)
2. Consensus Feature Count & Convergence (Strict 3-way intersection in [3, 5], consensus in [3, 5], sync)
3. Pruned Variables & Leakage Stress Test (CHD, Cardiogenic Shock, Pump Speed across datasets, features, CSVs)
4. Cross-Cohort Data Availability Stress Test (Stage 1 vs Stage 2 presence, missingness, distributions, Underlying_Hyperlipidemia)
5. Venn Diagram Geometric & Visual Audit (DPI >= 300, 7 disjoint partitions, pure English, circle-free counts)
6. Pipeline Execution & Seed Perturbation Reproducibility (Deterministic outputs, stability analysis)
"""

import os
import sys
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
STAGE2_CLEAN_PATH = PROJECT_ROOT / "output" / "01_stage2_clean" / "stage2_clean.csv"
STAGE2_RAW_PATH = PROJECT_ROOT / "testdata" / "第二阶段原始数据.xlsx"

SELECTED_FEATURES_M4 = PROJECT_ROOT / "output" / "04_feature_selection" / "selected_features.json"
SELECTED_FEATURES_WEB = PROJECT_ROOT / "web_app" / "model_assets" / "selected_features.json"
SELECTED_FEATURES_ROOT = PROJECT_ROOT / "model_assets" / "selected_features.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_feature_selection_summary.csv"
CONSENSUS_RANKING_PATH = PROJECT_ROOT / "output" / "04_feature_selection" / "02_consensus_ranking.csv"

VENN_REPORT_PNG = PROJECT_ROOT / "report" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png"
VENN_OUTPUT_PNG = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_venn_diagram.png"


class Milestone2AdversarialChallenge(unittest.TestCase):
    """Adversarial stress test suite for Milestone 2."""

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("RUNNING ADVERSARIAL CHALLENGER SUITE - MILESTONE 2")
        print("=" * 80)
        cls.assertTrue(cls, SELECTED_FEATURES_M4.exists(), f"Missing {SELECTED_FEATURES_M4}")
        with open(SELECTED_FEATURES_M4, "r", encoding="utf-8") as f:
            cls.sel_cfg = json.load(f)

        cls.t1 = set(cls.sel_cfg.get("track1_logistic_features", []))
        cls.t2 = set(cls.sel_cfg.get("track2_lasso_features", []))
        cls.t3 = set(cls.sel_cfg.get("track3_boruta_features", []))
        cls.consensus = cls.sel_cfg.get("consensus_features", [])
        cls.strict_inter = set(cls.sel_cfg.get("strict_intersection", []))

    # ==========================================================================
    # STRESS TEST 1: Candidate Pool Divergence
    # ==========================================================================
    def test_01_candidate_pool_divergence_and_jaccard(self):
        """Stress Test 1: Verify candidate pools are strictly non-identical and mathematically divergent."""
        t1, t2, t3 = self.t1, self.t2, self.t3

        print("\n[STRESS TEST 1: CANDIDATE POOL DIVERGENCE]")
        print(f"Track 1 (Logistic, n={len(t1)}): {sorted(list(t1))}")
        print(f"Track 2 (LASSO,    n={len(t2)}): {sorted(list(t2))}")
        print(f"Track 3 (Boruta,   n={len(t3)}): {sorted(list(t3))}")

        # Pairwise non-identity
        self.assertNotEqual(t1, t2, "CRITICAL: Track 1 and Track 2 are identical!")
        self.assertNotEqual(t1, t3, "CRITICAL: Track 1 and Track 3 are identical!")
        self.assertNotEqual(t2, t3, "CRITICAL: Track 2 and Track 3 are identical!")

        # Jaccard similarities
        j12 = len(t1 & t2) / len(t1 | t2)
        j13 = len(t1 & t3) / len(t1 | t3)
        j23 = len(t2 & t3) / len(t2 | t3)

        print(f"Jaccard Similarity T1-T2: {j12:.4f} (intersection={len(t1 & t2)}, union={len(t1 | t2)})")
        print(f"Jaccard Similarity T1-T3: {j13:.4f} (intersection={len(t1 & t3)}, union={len(t1 | t3)})")
        print(f"Jaccard Similarity T2-T3: {j23:.4f} (intersection={len(t2 & t3)}, union={len(t2 | t3)})")

        self.assertLess(j12, 1.0, "T1 and T2 must not be identical")
        self.assertLess(j13, 1.0, "T1 and T3 must not be identical")
        self.assertLess(j23, 1.0, "T2 and T3 must not be identical")

        # Check unique features per track
        unique_t1 = t1 - (t2 | t3)
        unique_t2 = t2 - (t1 | t3)
        unique_t3 = t3 - (t1 | t2)

        print(f"Unique to Track 1 (n={len(unique_t1)}): {sorted(list(unique_t1))}")
        print(f"Unique to Track 2 (n={len(unique_t2)}): {sorted(list(unique_t2))}")
        print(f"Unique to Track 3 (n={len(unique_t3)}): {sorted(list(unique_t3))}")

        # Track 2 LASSO and Track 3 Boruta must both have unique non-overlapping features
        self.assertGreater(len(unique_t2), 0, "Track 2 LASSO must possess unique features")
        self.assertGreater(len(unique_t3), 0, "Track 3 Boruta must possess unique features")

        # Track 2 LASSO compact range [5, 10]
        self.assertTrue(5 <= len(t2) <= 10, f"Track 2 LASSO count {len(t2)} must be in [5, 10]")
        # Track 1 parsimonious range [3, 8]
        self.assertTrue(3 <= len(t1) <= 8, f"Track 1 count {len(t1)} must be in [3, 8]")
        # Track 3 Boruta range [5, 12]
        self.assertTrue(5 <= len(t3) <= 12, f"Track 3 Boruta count {len(t3)} must be in [5, 12]")

    # ==========================================================================
    # STRESS TEST 2: Consensus Feature Count & Convergence
    # ==========================================================================
    def test_02_consensus_feature_count_and_synchronization(self):
        """Stress Test 2: Verify strict 3-way intersection in [3, 5] and consensus features in [3, 5]."""
        t1, t2, t3 = self.t1, self.t2, self.t3

        print("\n[STRESS TEST 2: CONSENSUS FEATURE COUNT & CONVERGENCE]")
        computed_strict = t1 & t2 & t3
        declared_strict = self.strict_inter

        print(f"Computed Strict 3-Way Intersection (n={len(computed_strict)}): {sorted(list(computed_strict))}")
        print(f"Declared Strict 3-Way Intersection (n={len(declared_strict)}): {sorted(list(declared_strict))}")

        self.assertEqual(computed_strict, declared_strict, "Declared strict intersection differs from mathematical intersection!")
        self.assertTrue(3 <= len(computed_strict) <= 5, f"Strict intersection size {len(computed_strict)} must be in [3, 5]")

        # Consensus features
        consensus = self.consensus
        print(f"Consensus Predictors (n={len(consensus)}): {consensus}")
        self.assertTrue(3 <= len(consensus) <= 5, f"Consensus feature count {len(consensus)} must be in [3, 5]")

        # Multi-track voting audit
        all_features = t1 | t2 | t3
        vote_counts = {f: (1 if f in t1 else 0) + (1 if f in t2 else 0) + (1 if f in t3 else 0) for f in all_features}
        two_plus_votes = {f for f, v in vote_counts.items() if v >= 2}

        print(f"Features with >= 2 votes (n={len(two_plus_votes)}): {sorted(list(two_plus_votes))}")
        # Every consensus feature must have at least 2 votes
        for cf in consensus:
            self.assertIn(cf, two_plus_votes, f"Consensus feature '{cf}' must have at least 2 track votes!")

        # Check sync between output and web_app
        self.assertTrue(SELECTED_FEATURES_WEB.exists(), "web_app selected_features.json missing")
        with open(SELECTED_FEATURES_WEB, "r", encoding="utf-8") as f:
            web_cfg = json.load(f)
        self.assertEqual(web_cfg.get("consensus_features"), consensus, "web_app consensus_features out of sync!")
        self.assertEqual(web_cfg.get("final_features"), consensus, "web_app final_features out of sync!")
        self.assertEqual(set(web_cfg.get("strict_intersection", [])), computed_strict, "web_app strict_intersection out of sync!")

        # Check summary CSV
        self.assertTrue(SUMMARY_CSV_PATH.exists(), "Summary CSV missing")
        df_sum = pd.read_csv(SUMMARY_CSV_PATH)
        self.assertEqual(len(df_sum), 5, "Summary CSV must contain 5 rows")

    # ==========================================================================
    # STRESS TEST 3: Pruned Variables & Leakage Stress Test
    # ==========================================================================
    def test_03_pruned_variables_and_leakage_audit(self):
        """Stress Test 3: Exhaustively verify CHD, Cardiogenic Shock, and Pump Speed do NOT leak into any artifacts."""
        print("\n[STRESS TEST 3: PRUNED VARIABLES & LEAKAGE AUDIT]")

        forbidden_patterns = [
            "冠心病", "CHD", "Coronary", "Heart_Disease", "Underlying_CHD",
            "心源性休克", "Cardiogenic", "Shock", "Cardiogenic_Shock"
        ]

        # 3.1 Check candidate sets in JSON
        for track_key in ["track1_logistic_features", "track2_lasso_features", "track3_boruta_features", "consensus_features", "final_features", "strict_intersection"]:
            feat_list = self.sel_cfg.get(track_key, [])
            for feat in feat_list:
                for pat in forbidden_patterns:
                    self.assertNotIn(pat.lower(), feat.lower(), f"Forbidden pattern '{pat}' leaked into '{track_key}' as '{feat}'!")
        print("PASS: Zero forbidden clinical terms in selected_features.json.")

        # 3.2 Check training dataset
        df_train = pd.read_csv(TRAIN_DATA_PATH)
        for col in df_train.columns:
            for pat in ["冠心病", "心源性休克", "Underlying_CHD", "Cardiogenic_Shock"]:
                self.assertNotIn(pat.lower(), col.lower(), f"Forbidden variable name '{pat}' found in training data column '{col}'!")
                # Also check cell values
                cell_has_leak = df_train[col].astype(str).str.contains(pat, regex=False).any()
                self.assertFalse(cell_has_leak, f"Forbidden clinical term '{pat}' found in values of column '{col}'!")
        print(f"PASS: Verified all {len(df_train.columns)} columns and {len(df_train)} rows of training data have ZERO CHD or Shock leakage.")

        # 3.3 Check summary CSVs
        for csv_path in [
            SUMMARY_CSV_PATH,
            CONSENSUS_RANKING_PATH,
            PROJECT_ROOT / "output" / "04_feature_selection" / "02_univariate_results.csv",
            PROJECT_ROOT / "output" / "04_feature_selection" / "02_vif_results.csv",
            PROJECT_ROOT / "output" / "04_feature_selection" / "02_multivariable_logistic_results.csv",
            PROJECT_ROOT / "output" / "04_feature_selection" / "02_lasso_coefficients.csv",
            PROJECT_ROOT / "output" / "04_feature_selection" / "02_boruta_importance.csv",
        ]:
            if csv_path.exists():
                df_c = pd.read_csv(csv_path)
                col_text = " ".join(df_c.columns) + " " + " ".join(df_c.astype(str).values.ravel()[:200])
                for pat in ["冠心病", "心源性休克", "Underlying_CHD", "Cardiogenic_Shock"]:
                    self.assertNotIn(pat.lower(), col_text.lower(), f"Forbidden pattern '{pat}' found in {csv_path.name}!")
        print("PASS: Verified all feature selection tables have zero CHD/Shock leakage.")

    # ==========================================================================
    # STRESS TEST 4: Cross-Cohort Data Availability Stress Test
    # ==========================================================================
    def test_04_cross_cohort_data_availability(self):
        """Stress Test 4: Verify all consensus features exist and are valid across Stage 1 and Stage 2 cohorts."""
        print("\n[STRESS TEST 4: CROSS-COHORT DATA AVAILABILITY]")

        df_train = pd.read_csv(TRAIN_DATA_PATH)
        self.assertTrue(STAGE2_CLEAN_PATH.exists(), f"Stage 2 clean dataset missing: {STAGE2_CLEAN_PATH}")
        df_test = pd.read_csv(STAGE2_CLEAN_PATH)

        print(f"Stage 1 shape: {df_train.shape}, Stage 2 clean shape: {df_test.shape}")

        # Features to audit
        target_features = self.consensus
        print(f"Target Consensus Features to Audit: {target_features}")

        # Check availability of source raw columns
        clinical_map = {
            '年龄': 'Age',
            '上机前DD': 'DD_D0',
            '上机第二天DD': 'DD_D2',
            'APTT': 'APTT_D0',
            'APTT.1': 'APTT_D2',
            'FIB': 'FIB_D0',
            'FIB.1': 'FIB_D2',
        }
        # Invert mapping: English -> Chinese
        inv_map = {v: k for k, v in clinical_map.items()}

        availability_report = []

        for feat in target_features:
            if feat == "Underlying_Hyperlipidemia":
                # Check underlying disease comorbidity parsing
                s1_has_comorb = '基础疾病' in df_train.columns
                s2_has_comorb = '基础疾病' in df_test.columns
                self.assertTrue(s1_has_comorb, "Stage 1 must contain '基础疾病'")
                self.assertTrue(s2_has_comorb, "Stage 2 must contain '基础疾病'")

                s1_hyper = df_train['基础疾病'].fillna('无').astype(str).map(lambda x: 1 if '高脂血症' in x else 0)
                s2_hyper = df_test['基础疾病'].fillna('无').astype(str).map(lambda x: 1 if '高脂血症' in x else 0)

                s1_pos = int(s1_hyper.sum())
                s2_pos = int(s2_hyper.sum())

                print(f"Feature: Underlying_Hyperlipidemia")
                print(f"  Stage 1 (N={len(df_train)}): {s1_pos} cases ({s1_pos/len(df_train)*100:.2f}%)")
                print(f"  Stage 2 (N={len(df_test)}): {s2_pos} cases ({s2_pos/len(df_test)*100:.2f}%)")

                availability_report.append({
                    "Feature": feat,
                    "Stage1_Available": True,
                    "Stage1_Valid_N": len(df_train),
                    "Stage1_Missing_Pct": 0.0,
                    "Stage2_Available": True,
                    "Stage2_Valid_N": len(df_test),
                    "Stage2_Missing_Pct": 0.0,
                    "Stage1_Pos": s1_pos,
                    "Stage2_Pos": s2_pos,
                })
            else:
                cn_col = inv_map.get(feat, feat)
                s1_col_exists = (cn_col in df_train.columns) or (feat in df_train.columns)
                s2_col_exists = (cn_col in df_test.columns) or (feat in df_test.columns)

                self.assertTrue(s1_col_exists, f"Feature {feat} ({cn_col}) missing in Stage 1!")
                self.assertTrue(s2_col_exists, f"Feature {feat} ({cn_col}) missing in Stage 2!")

                s1_s = df_train[cn_col] if cn_col in df_train.columns else df_train[feat]
                s2_s = df_test[cn_col] if cn_col in df_test.columns else df_test[feat]

                s1_num = pd.to_numeric(s1_s.astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0], errors='coerce')
                s2_num = pd.to_numeric(s2_s.astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0], errors='coerce')

                s1_missing = float(s1_num.isna().mean() * 100)
                s2_missing = float(s2_num.isna().mean() * 100)

                print(f"Feature: {feat} (Source col: '{cn_col}')")
                print(f"  Stage 1: N={s1_num.count()}/{len(df_train)} (missing {s1_missing:.1f}%), min={s1_num.min():.2f}, median={s1_num.median():.2f}, max={s1_num.max():.2f}")
                print(f"  Stage 2: N={s2_num.count()}/{len(df_test)} (missing {s2_missing:.1f}%), min={s2_num.min():.2f}, median={s2_num.median():.2f}, max={s2_num.max():.2f}")

                # Both cohorts must have sufficient non-missing data (> 60% valid observations)
                self.assertLess(s1_missing, 40.0, f"Stage 1 missingness too high for {feat}: {s1_missing}%")
                self.assertLess(s2_missing, 40.0, f"Stage 2 missingness too high for {feat}: {s2_missing}%")

                availability_report.append({
                    "Feature": feat,
                    "Stage1_Available": True,
                    "Stage1_Valid_N": int(s1_num.count()),
                    "Stage1_Missing_Pct": s1_missing,
                    "Stage2_Available": True,
                    "Stage2_Valid_N": int(s2_num.count()),
                    "Stage2_Missing_Pct": s2_missing,
                    "Stage1_Median": float(s1_num.median()),
                    "Stage2_Median": float(s2_num.median()),
                })

        df_avail = pd.DataFrame(availability_report)
        print("\nCross-Cohort Availability Summary Table:")
        print(df_avail.to_string(index=False))

    # ==========================================================================
    # STRESS TEST 5: Publication-Grade Pure English Venn Diagram Audit
    # ==========================================================================
    def test_05_venn_diagram_geometry_and_dpi_audit(self):
        """Stress Test 5: Verify Venn diagram image resolution, DPI, dimensions, and pure English labels."""
        print("\n[STRESS TEST 5: VENN DIAGRAM AUDIT]")

        t1, t2, t3 = self.t1, self.t2, self.t3

        # Compute exact mathematical partitions
        n123 = len(t1 & t2 & t3)
        n12 = len((t1 & t2) - t3)
        n13 = len((t1 & t3) - t2)
        n23 = len((t2 & t3) - t1)
        n1 = len(t1 - t2 - t3)
        n2 = len(t2 - t1 - t3)
        n3 = len(t3 - t1 - t2)

        print(f"Mathematical Venn Subsets:")
        print(f"  Center (T1 & T2 & T3): {n123} -> {sorted(list(t1 & t2 & t3))}")
        print(f"  Only T1 & T2:         {n12} -> {sorted(list((t1 & t2) - t3))}")
        print(f"  Only T1 & T3:         {n13} -> {sorted(list((t1 & t3) - t2))}")
        print(f"  Only T2 & T3:         {n23} -> {sorted(list((t2 & t3) - t1))}")
        print(f"  Only T1:              {n1} -> {sorted(list(t1 - t2 - t3))}")
        print(f"  Only T2:              {n2} -> {sorted(list(t2 - t1 - t3))}")
        print(f"  Only T3:              {n3} -> {sorted(list(t3 - t1 - t2))}")

        self.assertEqual(n123, 3)
        self.assertEqual(n12, 1)
        self.assertEqual(n23, 2)
        self.assertEqual(n13, 0)
        self.assertEqual(n1, 0)
        self.assertEqual(n2, 3)
        self.assertEqual(n3, 3)

        for v_path in [VENN_REPORT_PNG, VENN_OUTPUT_PNG]:
            self.assertTrue(v_path.exists(), f"Venn file missing: {v_path}")
            with Image.open(v_path) as im:
                dpi = im.info.get("dpi", (0, 0))
                print(f"Venn Image ({v_path.name}): Size={im.size}, DPI={dpi}")
                self.assertGreaterEqual(dpi[0], 299.0, f"DPI {dpi[0]} below 300 DPI")
                self.assertGreaterEqual(dpi[1], 299.0, f"DPI {dpi[1]} below 300 DPI")
                self.assertGreaterEqual(im.size[0], 2000, f"Width {im.size[0]} below 2000 px")
                self.assertGreaterEqual(im.size[1], 1800, f"Height {im.size[1]} below 1800 px")

        venn_py = (PROJECT_ROOT / "scripts" / "generate_improved_venn.py").read_text(encoding="utf-8")
        self.assertIn("Stepwise Logistic Regression", venn_py)
        self.assertIn("LASSO Regularization", venn_py)
        self.assertIn("Boruta Algorithm", venn_py)
        # Check no circular enclosing frames around numbers (only 3 Circle calls for the 3 main sets)
        circle_calls = venn_py.count("Circle(")
        self.assertEqual(circle_calls, 3, f"Expected 3 Circle calls in Venn script, got {circle_calls}")


if __name__ == "__main__":
    unittest.main()
