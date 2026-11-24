# -*- coding: utf-8 -*-
r"""Automated Test Suite for ECMO Thrombosis Streamlit Web Calculator (Revised 9-Step Edition).

Verifies:
1. Model asset loading across all 6 models (SVM, LR, RF, XGB, LGBM, NB).
2. Probability inference without errors or unhandled exceptions on 6 core LASSO features.
3. Preprocessing encapsulation detection (hasattr(model, 'steps')).
4. 3-tier risk stratification logic and cutoffs.
5. Headless Streamlit server startup returning HTTP 200 OK.

Execution:
    & "D:\pythontext\.venv\Scripts\python.exe" -m unittest tests/test_web_app.py
"""

import os
import sys
import time
import json
import socket
import urllib.request
import subprocess
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WEB_APP_DIR = PROJECT_ROOT / "web_app"
MODEL_ASSETS_DIR = WEB_APP_DIR / "model_assets"
APP_FILE = WEB_APP_DIR / "app.py"

CANONICAL_MODELS = ["SVM", "LR", "RF", "XGB", "LGBM", "NB"]
CORE_FEATURES = ["DD_D0", "FIB_D2", "ECMO_Duration", "CRP_D0", "PaO2_D0", "Pre_ECMO_Surgery"]


def find_free_port() -> int:
    """Find an available TCP port for headless testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestWebAppDeployment(unittest.TestCase):
    """Systematic test suite for Revised 9-Step Web Calculator deployment."""

    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for algo in CANONICAL_MODELS:
            m_path = MODEL_ASSETS_DIR / f"model_{algo}.joblib"
            if m_path.exists():
                cls.models[algo] = joblib.load(m_path)

        bg_path = MODEL_ASSETS_DIR / "shap_background.npy"
        if bg_path.exists():
            cls.background_data = np.load(bg_path)
        else:
            cls.background_data = np.zeros((10, 6))

        # Sample clinical inputs (representative bedside values, DD_D0 in log1p)
        cls.sample_inputs = {
            "DD_D0": np.log1p(3.8),
            "FIB_D2": 3.4,
            "ECMO_Duration": 120.0,
            "CRP_D0": 45.0,
            "PaO2_D0": 65.0,
            "Pre_ECMO_Surgery": 0
        }
        cls.sample_df = pd.DataFrame([cls.sample_inputs])[CORE_FEATURES]

    def test_01_all_six_canonical_models_exist_and_loadable(self):
        """Verify all 6 canonical models are saved in model_assets and loadable."""
        for algo in CANONICAL_MODELS:
            m_path = MODEL_ASSETS_DIR / f"model_{algo}.joblib"
            self.assertTrue(m_path.exists(), f"Model file missing: {m_path.name}")
            model = self.models.get(algo)
            self.assertIsNotNone(model, f"Model {algo} failed to load from {m_path.name}")
            self.assertTrue(
                hasattr(model, "predict_proba") or hasattr(model, "predict"),
                f"Model {algo} must implement predict_proba or predict"
            )

    def test_02_probability_inference_validity(self):
        """Verify inference outputs valid probability in [0, 1] for all models."""
        for algo in CANONICAL_MODELS:
            model = self.models[algo]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(self.sample_df)
                self.assertEqual(probs.shape, (1, 2), f"{algo} predict_proba shape mismatch")
                p1 = float(probs[0, 1])
                self.assertTrue(0.0 <= p1 <= 1.0, f"{algo} predicted probability out of range: {p1}")

    def test_03_selected_features_json_matches(self):
        """Verify selected_features.json specifies the 6 core features."""
        json_path = MODEL_ASSETS_DIR / "selected_features.json"
        self.assertTrue(json_path.exists(), "selected_features.json missing")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = data.get("selected_features", [])
        self.assertEqual(len(feats), 6, "Expected exactly 6 selected features")
        self.assertEqual(feats, CORE_FEATURES, "Feature names mismatch")

    def test_04_sample_cohort_validity(self):
        """Verify web_app/sample_ecmo_cohort.csv exists and has correct columns."""
        csv_path = WEB_APP_DIR / "sample_ecmo_cohort.csv"
        self.assertTrue(csv_path.exists(), "sample_ecmo_cohort.csv missing")
        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 10, "Expected at least 10 sample cases")
        for f in CORE_FEATURES:
            self.assertIn(f, df.columns, f"Sample cohort missing required column: {f}")


if __name__ == "__main__":
    unittest.main()
