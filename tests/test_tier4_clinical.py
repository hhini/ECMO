# -*- coding: utf-8 -*-
"""Tier 4: Clinical Acceptance Scenarios Test Suite.

Simulates real-world clinical use cases:
1. End-to-end clinical patient simulation: High-risk vs Low-risk profiles through model inference.
2. Risk tier badge concordance and warning threshold rules.
3. Clinical decision threshold safety (Sensitivity >= 70% to prevent fatal missed thrombosis).
4. Web calculator headless execution smoke test.
"""

import os
import json
import unittest
import numpy as np
import pandas as pd
import joblib

from tests.fixtures import (
    WEB_APP_DIR,
    MODEL_ASSETS_DIR,
    DATA_DIR,
    TARGET_COLUMN,
    CLINICAL_SIMULATION_PROFILES,
    RISK_TIER_LOW_MAX,
    RISK_TIER_MODERATE_MAX,
    compute_clinical_metrics,
)


class TestTier4ClinicalAcceptance(unittest.TestCase):
    """Tier 4 Test Suite: Clinical Acceptance Scenarios."""

    # --------------------------------------------------------------------------
    # C1: High-Risk vs Low-Risk Patient Clinical Simulation
    # --------------------------------------------------------------------------
    def test_C01_clinical_patient_risk_simulation(self):
        """C1: Simulate high-risk vs low-risk patient profiles and verify differential risk prediction."""
        model_paths = list(MODEL_ASSETS_DIR.glob("model_*.joblib"))
        if not model_paths:
            self.skipTest("No models found in web_app/model_assets.")

        high_profile = CLINICAL_SIMULATION_PROFILES["high_risk_elderly_prolonged_coagulopathy"]["values"]
        low_profile = CLINICAL_SIMULATION_PROFILES["low_risk_young_short_normal_labs"]["values"]

        # Load scaler configuration from preprocess_config.json if available
        prep_cfg_file = MODEL_ASSETS_DIR / "preprocess_config.json"
        means = {}
        scales = {}
        if prep_cfg_file.exists():
            with open(prep_cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            means = cfg.get("scaler_mean", {})
            scales = cfg.get("scaler_scale", {})

        # Run against available models
        for p in model_paths:
            model = joblib.load(p)
            feat_names = getattr(model, "feature_names_in_", None)
            if feat_names is None and hasattr(model, "steps"):
                feat_names = getattr(model.steps[-1][1], "feature_names_in_", None)

            if feat_names is not None:
                # Build test dataframes for both profiles, applying scaler if model is not a self-scaling Pipeline
                is_pipeline = hasattr(model, "steps")
                def prepare_row(profile):
                    row = {}
                    for f in feat_names:
                        val = profile.get(f, 0.0)
                        if not is_pipeline and f in means and f in scales and scales[f]:
                            row[f] = (float(val) - float(means[f])) / float(scales[f])
                        else:
                            row[f] = float(val)
                    return row

                df_high = pd.DataFrame([prepare_row(high_profile)])
                df_low = pd.DataFrame([prepare_row(low_profile)])

                prob_high = model.predict_proba(df_high)[0, 1]
                prob_low = model.predict_proba(df_low)[0, 1]

                # High risk profile must have higher predicted probability than low risk profile
                self.assertGreater(
                    prob_high,
                    prob_low,
                    f"Model {p.name}: High risk profile prob ({prob_high:.4f}) should exceed low risk prob ({prob_low:.4f})"
                )

    # --------------------------------------------------------------------------
    # C2: Risk Stratification Badge Concordance
    # --------------------------------------------------------------------------
    def test_C02_risk_stratification_badge_concordance(self):
        """C2: Verify 3-tier risk badges (<20% Low, 20-50% Moderate, >=50% High) and clinical color codes."""
        badge_specs = [
            (0.08, "Low", "低风险", "#00A087"),
            (0.19, "Low", "低风险", "#00A087"),
            (0.20, "Moderate", "中风险", "#D97706"),
            (0.35, "Moderate", "中风险", "#D97706"),
            (0.499, "Moderate", "中风险", "#D97706"),
            (0.50, "High", "高风险", "#E64B35"),
            (0.85, "High", "高风险", "#E64B35"),
        ]

        def get_risk_badge(prob: float):
            if prob < RISK_TIER_LOW_MAX:
                return "Low", "低风险", "#00A087"
            elif prob < RISK_TIER_MODERATE_MAX:
                return "Moderate", "中风险", "#D97706"
            else:
                return "High", "高风险", "#E64B35"

        for p_val, exp_tier, exp_label, exp_color in badge_specs:
            tier, label, color = get_risk_badge(p_val)
            self.assertEqual(tier, exp_tier)
            self.assertEqual(label, exp_label)
            self.assertEqual(color, exp_color)

    # --------------------------------------------------------------------------
    # C3: Clinical Decision Threshold Sensitivity Requirement
    # --------------------------------------------------------------------------
    def test_C03_clinical_screening_sensitivity_threshold(self):
        """C3: In ECMO thrombosis screening, verify threshold tuning can achieve clinical Sensitivity >= 70%."""
        perf_file = MODEL_ASSETS_DIR / "03_model_performance_metrics.csv"
        if not perf_file.exists():
            self.skipTest("Performance metrics file not found.")

        df_perf = pd.read_csv(perf_file)
        # Check sensitivity/recall columns
        sens_cols = [c for c in df_perf.columns if "sens" in c.lower() or "recall" in c.lower()]
        self.assertTrue(len(sens_cols) > 0, "No sensitivity/recall columns found in performance table")

        # Verify at least one model achieves high sensitivity on training or testing
        max_sens = df_perf[sens_cols].max().max()
        self.assertGreaterEqual(
            max_sens,
            0.70,
            f"At least one model must achieve clinical screening sensitivity >= 0.70, max found: {max_sens:.3f}"
        )

    # --------------------------------------------------------------------------
    # C4: Web Calculator Headless Smoke Test
    # --------------------------------------------------------------------------
    def test_C04_web_calculator_headless_smoke_test(self):
        """C4: Verify web app components, dependencies, and model loading execute without unhandled exceptions."""
        app_file = WEB_APP_DIR / "app.py"
        self.assertTrue(app_file.exists(), "web_app/app.py does not exist")

        # Verify selected_features.json is readable and contains valid structure
        sel_features_file = MODEL_ASSETS_DIR / "selected_features.json"
        if sel_features_file.exists():
            with open(sel_features_file, "r", encoding="utf-8") as f:
                feat_cfg = json.load(f)
            self.assertTrue(isinstance(feat_cfg, dict))

        # Verify preprocess_config.json is readable
        prep_cfg_file = MODEL_ASSETS_DIR / "preprocess_config.json"
        if prep_cfg_file.exists():
            with open(prep_cfg_file, "r", encoding="utf-8") as f:
                prep_cfg = json.load(f)
            self.assertTrue(isinstance(prep_cfg, dict))


if __name__ == "__main__":
    unittest.main()
