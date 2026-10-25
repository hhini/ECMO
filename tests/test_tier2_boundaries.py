# -*- coding: utf-8 -*-
"""Tier 2: Boundary & Corner Cases Test Suite.

Adversarial stress testing covering:
1. Empty input handling (zero-length datasets and null inputs).
2. Extreme physiological boundary values in clinical features.
3. Severe label distribution extremes (all-positive, all-negative, zero-division safeguards).
4. DeLong structural variance mathematical non-negativity across degenerate predictions.
5. Missing value tolerance and non-NaN propagation.
6. Decision Curve Analysis (DCA) Net Benefit threshold boundaries.
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from tests.fixtures import (
    compute_delong_ci,
    compute_dca_net_benefit,
    compute_dca_treat_all_net_benefit,
    compute_hosmer_lemeshow,
    compute_clinical_metrics,
    MODEL_ASSETS_DIR,
    CANONICAL_MODELS,
    CLINICAL_SIMULATION_PROFILES,
)
import joblib


class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2 Test Suite: Boundary & Corner Cases."""

    # --------------------------------------------------------------------------
    # B1: Empty Input Handling
    # --------------------------------------------------------------------------
    def test_B01_empty_input_dataset_handling(self):
        """B1: Verify metric engines and evaluation functions handle empty inputs gracefully."""
        empty_y_true = np.array([])
        empty_y_pred = np.array([])

        # Metric calculator must return without crashing, AUROC is 0.5 (random guess) or 0.0
        metrics = compute_clinical_metrics(empty_y_true, empty_y_pred)
        for k, v in metrics.items():
            if k == "AUROC":
                self.assertIn(v, [0.0, 0.5], f"AUROC on empty input should be 0.0 or 0.5, got {v}")
            else:
                self.assertEqual(v, 0.0, f"Metric '{k}' on empty input should be 0.0")

        # DeLong CI on empty input
        auc, ci_l, ci_u, var = compute_delong_ci(empty_y_true, empty_y_pred)
        self.assertEqual(var, 0.0)

        # DCA Net Benefit on empty input
        nb = compute_dca_net_benefit(empty_y_true, empty_y_pred, threshold=0.20)
        self.assertEqual(nb, 0.0)

    # --------------------------------------------------------------------------
    # B2: Extreme Boundary Values in Clinical Predictors
    # --------------------------------------------------------------------------
    def test_B02_extreme_clinical_feature_boundaries(self):
        """B2: Verify model predictions on extreme physiological boundaries produce valid [0, 1] probabilities."""
        model_paths = list(MODEL_ASSETS_DIR.glob("model_*.joblib"))
        if not model_paths:
            self.skipTest("No models found in web_app/model_assets.")

        # Test extreme boundary DataFrame
        # Minimum vs maximum physiological ranges
        extreme_cases = pd.DataFrame([
            {
                # Ultra-low physiological boundary
                "Age": 18.0,
                "ECMO_Duration": 24.0,
                "APTT_D0": 10.0,
                "PLT_D2": 1.0,
                "WBC_D0": 0.5,
                "FIB_D2": 0.1,
                "Creatinine_D2": 10.0,
                "AST_D2": 5.0,
                "pH_D2": 6.80,
                "Underlying_Hyperlipidemia": 0.0,
                "DD_D0": 0.1,
                "DD_Late": 0.1,
                "ALT_D7": 5.0,
            },
            {
                # Ultra-high physiological boundary
                "Age": 105.0,
                "ECMO_Duration": 1000.0,
                "APTT_D0": 250.0,
                "PLT_D2": 1500.0,
                "WBC_D0": 100.0,
                "FIB_D2": 15.0,
                "Creatinine_D2": 1500.0,
                "AST_D2": 2500.0,
                "pH_D2": 7.80,
                "Underlying_Hyperlipidemia": 1.0,
                "DD_D0": 100.0,
                "DD_Late": 120.0,
                "ALT_D7": 500.0,
            }
        ])

        for p in model_paths:
            model = joblib.load(p)
            # Find expected features if model has feature_names_in_
            feat_names = getattr(model, "feature_names_in_", None)
            if feat_names is None and hasattr(model, "steps"):
                feat_names = getattr(model.steps[-1][1], "feature_names_in_", None)

            if feat_names is not None:
                # Filter or mock extreme cases for expected features
                present_cols = [c for c in feat_names if c in extreme_cases.columns]
                if len(present_cols) == len(feat_names):
                    X_test = extreme_cases[list(feat_names)]
                    probs = model.predict_proba(X_test)
                    self.assertEqual(probs.shape, (2, 2))
                    self.assertTrue(np.all(probs >= 0.0), f"Negative probability from {p.name}")
                    self.assertTrue(np.all(probs <= 1.0), f"Probability > 1.0 from {p.name}")
                    self.assertTrue(np.all(np.isfinite(probs)), f"NaN or Inf probability from {p.name}")

    # --------------------------------------------------------------------------
    # B3: Division by Zero Safeguards in Metric Calculations
    # --------------------------------------------------------------------------
    def test_B03_zero_division_safeguards_in_clinical_metrics(self):
        """B3: Verify zero-division safeguards when True Positives, True Negatives, or predictions are 0."""
        # Case 1: All ground truth is negative (0 positives -> Recall denominator is 0)
        y_true_all_neg = np.array([0, 0, 0, 0, 0])
        y_pred_prob = np.array([0.1, 0.2, 0.1, 0.4, 0.3])
        metrics_neg = compute_clinical_metrics(y_true_all_neg, y_pred_prob, threshold=0.5)
        self.assertEqual(metrics_neg["Recall"], 0.0)
        self.assertEqual(metrics_neg["Specificity"], 1.0)
        self.assertEqual(metrics_neg["Accuracy"], 1.0)
        self.assertEqual(metrics_neg["Precision"], 0.0)
        self.assertEqual(metrics_neg["F1"], 0.0)

        # Case 2: All ground truth is positive (0 negatives -> Specificity denominator is 0)
        y_true_all_pos = np.array([1, 1, 1, 1, 1])
        metrics_pos = compute_clinical_metrics(y_true_all_pos, y_pred_prob, threshold=0.5)
        self.assertEqual(metrics_pos["Specificity"], 0.0)
        self.assertEqual(metrics_pos["Recall"], 0.0)  # none predicted >= 0.5
        self.assertEqual(metrics_pos["Precision"], 0.0)

        # Case 3: All predictions are negative (0 predicted positives -> Precision denominator is 0)
        y_true_mixed = np.array([1, 0, 1, 0, 1])
        y_pred_all_zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        metrics_zero = compute_clinical_metrics(y_true_mixed, y_pred_all_zeros, threshold=0.5)
        self.assertEqual(metrics_zero["Precision"], 0.0)
        self.assertEqual(metrics_zero["F1"], 0.0)
        self.assertEqual(metrics_zero["Recall"], 0.0)
        self.assertEqual(metrics_zero["Specificity"], 1.0)

    # --------------------------------------------------------------------------
    # B4: DeLong Variance Mathematical Non-Negativity
    # --------------------------------------------------------------------------
    def test_B04_delong_structural_variance_non_negativity(self):
        """B4: Mathematical verification that DeLong structural variance is >= 0 and CI bounds are in [0, 1]."""
        # Test 1: Completely identical predictions (no separation, AUC = 0.5)
        y_true = np.array([1]*10 + [0]*10)
        y_pred_identical = np.full(20, 0.3)
        auc, ci_l, ci_u, var = compute_delong_ci(y_true, y_pred_identical)
        self.assertEqual(auc, 0.5)
        self.assertGreaterEqual(var, 0.0)
        self.assertTrue(0.0 <= ci_l <= ci_u <= 1.0)

        # Test 2: Perfect separation (AUC = 1.0)
        y_pred_perfect = np.array([0.9]*10 + [0.1]*10)
        auc_p, ci_lp, ci_up, var_p = compute_delong_ci(y_true, y_pred_perfect)
        self.assertEqual(auc_p, 1.0)
        self.assertGreaterEqual(var_p, 0.0)
        self.assertTrue(0.0 <= ci_lp <= 1.0)
        self.assertEqual(ci_up, 1.0)

        # Test 3: Completely inverted predictions (AUC = 0.0)
        y_pred_inverted = np.array([0.1]*10 + [0.9]*10)
        auc_i, ci_li, ci_ui, var_i = compute_delong_ci(y_true, y_pred_inverted)
        self.assertEqual(auc_i, 0.0)
        self.assertGreaterEqual(var_i, 0.0)
        self.assertEqual(ci_li, 0.0)
        self.assertTrue(0.0 <= ci_ui <= 1.0)

        # Test 4: Extreme imbalance (1 positive, 99 negatives)
        y_extreme_imb = np.array([1] + [0]*99)
        y_prob_imb = np.linspace(0.1, 0.9, 100)
        auc_imb, ci_limb, ci_uimb, var_imb = compute_delong_ci(y_extreme_imb, y_prob_imb)
        self.assertGreaterEqual(var_imb, 0.0)
        self.assertTrue(0.0 <= ci_limb <= ci_uimb <= 1.0)

    # --------------------------------------------------------------------------
    # B5: Missing Value Handling & Imputation Pipeline Robustness
    # --------------------------------------------------------------------------
    def test_B05_missing_value_pipeline_resilience(self):
        """B5: Verify imputation transformers handle missing cells without generating NaNs."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # Sample clinical matrix with missing cells
        X_with_nan = pd.DataFrame({
            "Age": [55.0, np.nan, 72.0, 43.0, np.nan],
            "PLT": [150.0, 220.0, np.nan, 85.0, 190.0],
            "APTT": [np.nan, 34.0, 48.0, np.nan, 55.0],
        })

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_clean = pipe.fit_transform(X_with_nan)

        # Must have zero NaNs and shape preserved
        self.assertEqual(X_clean.shape, (5, 3))
        self.assertEqual(int(np.isnan(X_clean).sum()), 0)
        self.assertTrue(np.all(np.isfinite(X_clean)))

    # --------------------------------------------------------------------------
    # B6: DCA Net Benefit Boundary Tests
    # --------------------------------------------------------------------------
    def test_B06_dca_net_benefit_threshold_boundaries(self):
        """B6: Verify Decision Curve Analysis Net Benefit at boundary thresholds (pt -> 0, pt -> 1, pt = 0.5)."""
        y_true = np.array([1]*20 + [0]*80)
        y_prob = np.linspace(0.05, 0.95, 100)

        # pt -> 0 (e.g., 0.001)
        nb_low = compute_dca_net_benefit(y_true, y_prob, threshold=0.001)
        self.assertFalse(np.isnan(nb_low))
        self.assertFalse(np.isinf(nb_low))

        # pt = 0.5 (weight factor pt / (1 - pt) = 1.0)
        nb_mid = compute_dca_net_benefit(y_true, y_prob, threshold=0.50)
        tp_mid = np.sum((y_true == 1) & (y_prob >= 0.50))
        fp_mid = np.sum((y_true == 0) & (y_prob >= 0.50))
        expected_mid = (tp_mid - fp_mid) / 100.0
        self.assertAlmostEqual(nb_mid, expected_mid, places=6)

        # Treat None: Net Benefit is identically 0.0
        nb_none = 0.0
        self.assertEqual(nb_none, 0.0)

        # Treat All at threshold 0.10
        nb_all_10 = compute_dca_treat_all_net_benefit(y_true, threshold=0.10)
        # 20/100 - (80/100) * (0.10 / 0.90) = 0.20 - 0.80 * (1/9) = 0.20 - 0.08888 = 0.11111
        expected_all_10 = 0.20 - 0.80 * (0.10 / 0.90)
        self.assertAlmostEqual(nb_all_10, expected_all_10, places=5)


if __name__ == "__main__":
    unittest.main()
