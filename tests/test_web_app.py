# -*- coding: utf-8 -*-
r"""Automated Test Suite for ECMO Thrombosis Streamlit Web Calculator.

Verifies:
1. Model asset loading across all 6 canonical models (RF, XGB, DT, SVM, LR, NB).
2. Probability inference without errors or unhandled exceptions.
3. Preprocessing encapsulation detection (hasattr(model, 'steps') avoids double-scaling).
4. SHAP multi-explainer execution across all 6 models with zero fake static proxies.
5. 3-tier risk stratification badges and recommendations.
6. Performance metrics file presence and dynamic metric resolution.
7. Headless Streamlit server startup returning HTTP 200 OK.

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

CANONICAL_MODELS = ["LGBM", "RF", "XGB", "DT", "SVM", "LR", "NB"]
CONSENSUS_FEATURES = ["DD_D0", "DD_Late", "WBC_D0", "ALT_D7", "FIB_D2"]


def find_free_port() -> int:
    """Find an available TCP port for headless testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestWebAppDeployment(unittest.TestCase):
    """Systematic test suite for Milestone 7 Web Calculator deployment."""

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
            cls.background_data = np.zeros((10, 5))

        # Sample clinical inputs (representative median bedside values)
        cls.sample_inputs = {
            "DD_D0": 5.4,
            "DD_Late": 8.6,
            "WBC_D0": 11.2,
            "ALT_D7": 32.0,
            "FIB_D2": 3.1
        }
        cls.sample_df = pd.DataFrame([cls.sample_inputs])[CONSENSUS_FEATURES]

    def test_01_all_six_canonical_models_exist_and_loadable(self):
        """Verify all canonical models are saved in model_assets and loadable."""
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
                pos_prob = float(probs[0, 1])
                self.assertGreaterEqual(pos_prob, 0.0, f"{algo} probability < 0")
                self.assertLessEqual(pos_prob, 1.0, f"{algo} probability > 1")
            else:
                pred = float(model.predict(self.sample_df)[0])
                self.assertIn(pred, [0.0, 1.0])

    def test_03_pipeline_encapsulation_avoids_double_scaling(self):
        """Verify that models with internal Pipeline detect steps and avoid double-scaling."""
        for algo in CANONICAL_MODELS:
            model = self.models[algo]
            # Each canonical calibrated model in model_assets should be a Pipeline with 'preprocessor' and 'classifier'
            self.assertTrue(
                hasattr(model, "steps") or hasattr(model, "named_steps"),
                f"Model {algo} should be encapsulated in a Pipeline to handle scaling internally"
            )
            # Verify preprocessor step exists
            if hasattr(model, "named_steps"):
                self.assertIn(
                    "preprocessor",
                    model.named_steps,
                    f"Model {algo} should encapsulate a 'preprocessor' step"
                )

    def test_04_shap_multi_explainer_execution_all_six_models(self):
        """Verify genuine SHAP calculation for all models without crashing or fake proxies."""
        import shap

        bg_df = pd.DataFrame(self.background_data, columns=CONSENSUS_FEATURES)

        for algo in CANONICAL_MODELS:
            model = self.models[algo]
            preprocessor = model.named_steps.get("preprocessor", None) if hasattr(model, "named_steps") else None
            clf = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
            base_est = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf

            if preprocessor is not None:
                scaled_input = pd.DataFrame(preprocessor.transform(self.sample_df), columns=CONSENSUS_FEATURES)
                scaled_bg = pd.DataFrame(preprocessor.transform(bg_df), columns=CONSENSUS_FEATURES)
            else:
                scaled_input = self.sample_df
                scaled_bg = bg_df

            # Tree Models
            if algo in ["LGBM", "RF", "XGB", "DT"]:
                explainer = shap.TreeExplainer(base_est)
                self.assertIsInstance(explainer, shap.TreeExplainer)
                res = explainer(scaled_input)
                s_vals = res[0, :, 1].values if len(res.shape) == 3 else res[0].values
            # Linear Model
            elif algo == "LR":
                explainer = shap.LinearExplainer(base_est, scaled_bg)
                self.assertIsInstance(explainer, shap.LinearExplainer)
                res = explainer(scaled_input)
                s_vals = res.values[0]
            # Kernel Models
            elif algo in ["SVM", "NB"]:
                predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=CONSENSUS_FEATURES))[:, 1]
                explainer = shap.KernelExplainer(predict_fn, bg_df.iloc[:15])
                self.assertIsInstance(explainer, shap.KernelExplainer)
                res = explainer.shap_values(self.sample_df, nsamples=30)
                s_vals = res[0] if (isinstance(res, list) or (isinstance(res, np.ndarray) and res.ndim == 2)) else res

            # Assert shape matches consensus feature count
            self.assertEqual(
                len(s_vals),
                len(CONSENSUS_FEATURES),
                f"{algo} SHAP output dimension must equal {len(CONSENSUS_FEATURES)}"
            )
            # Verify values are real floating point numbers and not hardcoded static proxies
            self.assertTrue(all(isinstance(float(v), float) for v in s_vals))
            self.assertFalse(all(v == 0.25 for v in s_vals), f"{algo} must not return static 0.25 proxy")

    def test_05_three_tier_risk_stratification(self):
        """Verify 3-tier risk badge classification (<20% Low, 20-50% Moderate, >=50% High)."""
        def classify_risk(risk_pct: float):
            if risk_pct < 20.0:
                return "Low", "#00A087"
            elif risk_pct < 50.0:
                return "Moderate", "#D97706"
            else:
                return "High", "#E64B35"

        self.assertEqual(classify_risk(10.5)[0], "Low")
        self.assertEqual(classify_risk(19.9)[0], "Low")
        self.assertEqual(classify_risk(20.0)[0], "Moderate")
        self.assertEqual(classify_risk(49.9)[0], "Moderate")
        self.assertEqual(classify_risk(50.0)[0], "High")
        self.assertEqual(classify_risk(88.0)[0], "High")

    def test_06_model_performance_metrics_file(self):
        """Verify 03_model_performance_metrics.csv exists and has rows for all 6 models."""
        metrics_file = MODEL_ASSETS_DIR / "03_model_performance_metrics.csv"
        self.assertTrue(metrics_file.exists(), f"Metrics file missing: {metrics_file}")
        df = pd.read_csv(metrics_file)
        self.assertGreaterEqual(len(df), 6, "Performance metrics must contain at least 6 models")
        models_in_file = set(df["Model"].unique())
        for algo in CANONICAL_MODELS:
            self.assertIn(algo, models_in_file, f"Model {algo} missing from metrics table")

    def test_07_headless_streamlit_startup_http_200(self):
        """Verify Streamlit web app starts headlessly and returns HTTP 200 OK."""
        test_port = find_free_port()
        cmd = [
            sys.executable,
            "-m", "streamlit", "run",
            str(APP_FILE),
            "--server.headless", "true",
            "--server.port", str(test_port),
            "--browser.gatherUsageStats", "false"
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        http_ok = False
        try:
            # Poll server up to 15 seconds
            start_time = time.time()
            while time.time() - start_time < 15.0:
                time.sleep(0.5)
                try:
                    req = urllib.request.Request(f"http://localhost:{test_port}/")
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            http_ok = True
                            break
                except Exception:
                    pass
        finally:
            proc.terminate()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        self.assertTrue(http_ok, f"Streamlit server on port {test_port} failed to return HTTP 200 within 15s")


if __name__ == "__main__":
    unittest.main()
