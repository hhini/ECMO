# -*- coding: utf-8 -*-
"""
Milestone 2 Unit & Acceptance Test Suite:
Three-Track Feature Selection Convergence & Pure English Publication-Grade Venn Diagram
"""

import os
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "data2_synthetic.csv"
SELECTED_FEATURES_M4 = PROJECT_ROOT / "output" / "04_feature_selection" / "selected_features.json"
SELECTED_FEATURES_WEB = PROJECT_ROOT / "web_app" / "model_assets" / "selected_features.json"
SUMMARY_CSV_PATH = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_feature_selection_summary.csv"

VENN_REPORT_PNG = PROJECT_ROOT / "report" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png"
VENN_OUTPUT_PNG = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_venn_diagram.png"


class TestMilestone2Verification(unittest.TestCase):
    """Milestone 2 Unit & Acceptance Test Suite."""

    def test_01_training_data_ingestion_and_hygiene(self):
        """M2.1: Verify official training data ingestion, continuous age, and zero CHD/Shock."""
        self.assertTrue(TRAIN_DATA_PATH.exists(), f"Official training dataset missing: {TRAIN_DATA_PATH}")
        df = pd.read_csv(TRAIN_DATA_PATH)
        self.assertEqual(len(df), 266, f"Expected N=266, got {len(df)}")
        self.assertEqual(int(df["是否下肢血栓"].sum()), 45, f"Expected 45 positive cases, got {df['是否下肢血栓'].sum()}")
        
        # Continuous Age
        age = df["年龄"]
        self.assertTrue(np.issubdtype(age.dtype, np.floating), f"Age must be continuous float64, got {age.dtype}")
        self.assertGreaterEqual(float(age.min()), 18.0, "Pediatric cases must be excluded")
        self.assertEqual(int(age.isna().sum()), 0, "Age must not contain NaNs in modeling dataset")
        
        # Zero CHD / Cardiogenic Shock
        for forbidden in ["冠心病", "Underlying_CHD", "Coronary_Heart_Disease", "心源性休克", "Cardiogenic_Shock"]:
            for col in df.columns:
                has_term = df[col].astype(str).str.contains(forbidden).any()
                self.assertFalse(has_term, f"Forbidden clinical term '{forbidden}' found in column '{col}'!")

    def test_02_track_candidate_sets_distinct_and_divergent(self):
        """M2.2: Verify candidate sets for Track 1, Track 2, and Track 3 are distinct and non-identical."""
        self.assertTrue(SELECTED_FEATURES_M4.exists(), f"selected_features.json missing: {SELECTED_FEATURES_M4}")
        with open(SELECTED_FEATURES_M4, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        t1 = set(cfg.get("track1_logistic_features", []))
        t2 = set(cfg.get("track2_lasso_features", []))
        t3 = set(cfg.get("track3_boruta_features", []))
        
        self.assertGreater(len(t1), 0, "Track 1 set must not be empty")
        self.assertGreater(len(t2), 0, "Track 2 set must not be empty")
        self.assertGreater(len(t3), 0, "Track 3 set must not be empty")
        
        # Distinct and non-identical
        self.assertNotEqual(t1, t2, "Track 1 and Track 2 must be distinct")
        self.assertNotEqual(t1, t3, "Track 1 and Track 3 must be distinct")
        self.assertNotEqual(t2, t3, "Track 2 and Track 3 must be distinct")
        
        # LASSO compact constraint: 5-10 features
        self.assertTrue(5 <= len(t2) <= 10, f"Track 2 LASSO count {len(t2)} must be in [5, 10]")

    def test_03_consensus_feature_count_and_synchronization(self):
        """M2.3: Verify strict 3-way intersection and consensus predictor panel counts in [3, 5]."""
        self.assertTrue(SELECTED_FEATURES_M4.exists())
        with open(SELECTED_FEATURES_M4, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        t1 = set(cfg["track1_logistic_features"])
        t2 = set(cfg["track2_lasso_features"])
        t3 = set(cfg["track3_boruta_features"])
        
        strict_inter = t1 & t2 & t3
        self.assertTrue(3 <= len(strict_inter) <= 5, f"Strict 3-way intersection {len(strict_inter)} must be in [3, 5]: {strict_inter}")
        
        consensus = cfg.get("consensus_features", [])
        final_features = cfg.get("final_features", [])
        self.assertTrue(3 <= len(consensus) <= 5, f"Consensus features count {len(consensus)} must be in [3, 5]: {consensus}")
        self.assertEqual(consensus, final_features, "consensus_features and final_features must match")
        
        # Check summary CSV
        self.assertTrue(SUMMARY_CSV_PATH.exists())
        df_sum = pd.read_csv(SUMMARY_CSV_PATH)
        self.assertEqual(len(df_sum), 5, "Summary CSV must contain 5 rows (3 tracks, intersection, consensus)")
        
        # Check web_app sync
        self.assertTrue(SELECTED_FEATURES_WEB.exists())
        with open(SELECTED_FEATURES_WEB, "r", encoding="utf-8") as f_web:
            cfg_web = json.load(f_web)
        self.assertEqual(cfg_web.get("final_features"), final_features)
        self.assertEqual(cfg_web.get("consensus_features"), consensus)

    def test_04_publication_venn_diagram_resolution_and_styling(self):
        """M2.4: Verify Venn diagram exists, is 300 DPI, pure English, and has plain numbers without enclosing circles."""
        for v_path in [VENN_REPORT_PNG, VENN_OUTPUT_PNG]:
            self.assertTrue(v_path.exists(), f"Venn diagram missing at: {v_path}")
            with Image.open(v_path) as im:
                dpi = im.info.get("dpi", (0, 0))
                self.assertGreaterEqual(dpi[0], 299.0, f"Venn diagram DPI {dpi[0]} must be >= 300 DPI")
                self.assertGreaterEqual(dpi[1], 299.0, f"Venn diagram DPI {dpi[1]} must be >= 300 DPI")
                self.assertGreaterEqual(im.size[0], 2000, f"Venn width {im.size[0]} must be >= 2000 px")
                self.assertGreaterEqual(im.size[1], 1800, f"Venn height {im.size[1]} must be >= 1800 px")
            
        venn_code = (PROJECT_ROOT / "scripts" / "generate_improved_venn.py").read_text(encoding="utf-8")
        self.assertIn("Stepwise Logistic Regression", venn_code)
        self.assertIn("LASSO Regularization", venn_code)
        self.assertIn("Boruta Algorithm", venn_code)
        # Verify no circular patches enclosing numbers (only 3 circle calls in file for the 3 main sets)
        self.assertEqual(venn_code.count("Circle("), 3)


if __name__ == "__main__":
    unittest.main()
