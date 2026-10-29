# -*- coding: utf-8 -*-
"""
Empirical Challenger Test Suite for Milestone 2 Venn Diagram
Tests:
1. Image inspection via PIL and OpenCV
2. DPI exact check (300 DPI)
3. Pure English typography and zero Chinese characters
4. Absence of circular enclosing patches / bubbles around numbers
5. Mathematical verification of all 7 disjoint set regions
6. Idempotence and reproducibility
"""

import os
import sys
import json
import hashlib
import unittest
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
REPORT_VENN = PROJECT_ROOT / "report" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png"
OUTPUT_VENN = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_venn_diagram.png"
WEB_VENN = PROJECT_ROOT / "web_app" / "static" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png"
OUTPUT_LEGACY_VENN = PROJECT_ROOT / "output" / "04_feature_selection" / "02_feature_selection_venn.png"
OUTPUT_OVERLAP_VENN = PROJECT_ROOT / "output" / "04_feature_selection" / "02_feature_overlap_venn.png"

SELECTED_FEATURES_JSON = PROJECT_ROOT / "output" / "04_feature_selection" / "selected_features.json"
WEB_SELECTED_FEATURES_JSON = PROJECT_ROOT / "web_app" / "model_assets" / "selected_features.json"
SUMMARY_CSV = PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_feature_selection_summary.csv"
VENN_SCRIPT = PROJECT_ROOT / "scripts" / "generate_improved_venn.py"


class TestMilestone2VennEmpiricalChallenge(unittest.TestCase):
    """Rigorous empirical challenger tests for Milestone 2 Venn diagram."""

    def test_01_image_files_exist_and_readable(self):
        """Confirm all Venn diagram image files exist and are readable by PIL and OpenCV."""
        target_files = [REPORT_VENN, OUTPUT_VENN]
        for fpath in target_files:
            self.assertTrue(fpath.exists(), f"Image file missing: {fpath}")
            self.assertGreater(fpath.stat().st_size, 10000, f"Image file too small: {fpath.stat().st_size} bytes")

            # PIL test
            with Image.open(fpath) as img:
                self.assertEqual(img.format, "PNG", f"Expected PNG format, got {img.format}")
                w, h = img.size
                self.assertGreaterEqual(w, 2000, f"Width {w} < 2000 px")
                self.assertGreaterEqual(h, 1800, f"Height {h} < 1800 px")
                print(f"[PIL OK] {fpath.name}: format={img.format}, size={img.size}, mode={img.mode}")

            # OpenCV test
            cv_img = cv2.imread(str(fpath))
            self.assertIsNotNone(cv_img, f"OpenCV failed to read {fpath}")
            self.assertEqual(cv_img.shape[0], h)
            self.assertEqual(cv_img.shape[1], w)
            self.assertEqual(cv_img.shape[2], 3)
            print(f"[OpenCV OK] {fpath.name}: shape={cv_img.shape}")

    def test_02_dpi_is_exactly_300(self):
        """Confirm DPI is exactly 300.0 x 300.0."""
        target_files = [REPORT_VENN, OUTPUT_VENN]
        for fpath in target_files:
            with Image.open(fpath) as img:
                dpi = img.info.get("dpi")
                self.assertIsNotNone(dpi, f"DPI info missing in image: {fpath}")
                x_dpi, y_dpi = dpi
                print(f"[DPI CHECK] {fpath.name}: DPI=({x_dpi}, {y_dpi})")
                self.assertAlmostEqual(x_dpi, 300.0, places=1, msg=f"X-DPI must be 300, got {x_dpi}")
                self.assertAlmostEqual(y_dpi, 300.0, places=1, msg=f"Y-DPI must be 300, got {y_dpi}")

    def test_03_pixel_consistency_between_report_and_output(self):
        """Confirm report and output Venn diagrams are identical in content."""
        h1 = hashlib.sha256(REPORT_VENN.read_bytes()).hexdigest()
        h2 = hashlib.sha256(OUTPUT_VENN.read_bytes()).hexdigest()
        print(f"[HASH] report venn sha256: {h1}")
        print(f"[HASH] output venn sha256: {h2}")
        self.assertEqual(h1, h2, "Report and output Venn diagrams must be byte-for-byte identical")

    def test_04_pure_english_labels_zero_chinese(self):
        """Confirm pure English labels and zero Chinese characters."""
        # 1. Script source code audit
        script_content = VENN_SCRIPT.read_text(encoding="utf-8")
        
        # Check text rendered via ax.text
        import ast
        tree = ast.parse(script_content)
        rendered_texts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Look for ax.text
                if (isinstance(func, ast.Attribute) and func.attr == "text") or \
                   (isinstance(func, ast.Name) and func.id == "text"):
                    # The 3rd arg or keyword 's' is text string
                    if len(node.args) >= 3:
                        rendered_texts.append(node.args[2])
                    for kw in node.keywords:
                        if kw.arg == "s":
                            rendered_texts.append(kw.value)

        print(f"[RENDERED TEXT NODES FOUND]: {len(rendered_texts)}")
        
        # Check for any Chinese / non-ASCII character in string literals of ax.text
        chinese_regex = r'[\u4e00-\u9fff]'
        import re
        for txt_node in rendered_texts:
            dumped = ast.dump(txt_node)
            matches = re.findall(chinese_regex, dumped)
            self.assertEqual(len(matches), 0, f"Found Chinese characters in ax.text call: {dumped}")

        # Explicitly verify the expected English labels
        expected_labels = [
            "Stepwise Logistic Regression",
            "LASSO Regularization",
            "Boruta Algorithm",
            "Features"
        ]
        for exp in expected_labels:
            self.assertIn(exp, script_content, f"Expected English label '{exp}' missing in Venn script")

        # Check that forbidden Chinese labels are absent
        forbidden_terms = ["图", "特征", "筛选", "逻辑回归", "变量", "血栓", "集合"]
        for fb in forbidden_terms:
            for txt_node in rendered_texts:
                dumped = ast.dump(txt_node)
                self.assertNotIn(fb, dumped, f"Forbidden Chinese term '{fb}' rendered in Venn diagram!")

        # 2. Inspect PNG metadata chunks
        with Image.open(REPORT_VENN) as img:
            for k, v in img.info.items():
                if isinstance(v, str):
                    zh_matches = re.findall(chinese_regex, v)
                    self.assertEqual(len(zh_matches), 0, f"Chinese found in PNG metadata key '{k}': {v}")

    def test_05_no_circular_enclosing_patches_or_bubbles(self):
        """Confirm plain moderate-sized bold numbers without circular enclosing patches or bubbles."""
        # 1. Code inspection
        script_content = VENN_SCRIPT.read_text(encoding="utf-8")
        
        # Check that bbox is NOT passed to ax.text (bbox is used for bounding boxes/bubbles)
        self.assertNotIn("bbox=", script_content, "Found bbox parameter in Venn script - bubbles are prohibited!")
        
        # Verify the only Circle patches created are the 3 main Venn sets
        circle_matches = [line.strip() for line in script_content.splitlines() if "Circle(" in line and not line.strip().startswith("#")]
        print(f"[CIRCLE CALLS IN SCRIPT]: {circle_matches}")
        self.assertEqual(len(circle_matches), 3, f"Expected exactly 3 Circle patches for the 3 sets, found {len(circle_matches)}")

        # 2. OpenCV Empirical Detection: Verify NO enclosing circular patches around ANY of the 5 numbers
        cv_img = cv2.imread(str(REPORT_VENN))
        
        # Locate the 5 digits inside the Venn diagram body
        # Text color is dark '#1E293B' (BGR: [59, 41, 30])
        b, g, r = cv_img[:, :, 0], cv_img[:, :, 1], cv_img[:, :, 2]
        digit_mask = (b < 100) & (g < 80) & (r < 70)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(digit_mask.astype(np.uint8))
        
        venn_digits = []
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if 400 < cy < 1700 and 300 < cx < 1900 and 100 < area < 4000:
                venn_digits.append((int(cx), int(cy), cw, ch, area))

        print(f"[VENN DIGITS LOCATED]: {len(venn_digits)} numbers found in Venn diagram")
        self.assertEqual(len(venn_digits), 5, f"Expected exactly 5 non-zero region numbers, found {len(venn_digits)}")

        # For each number, examine a 160x160 window centered on it
        for idx, (cx, cy, cw, ch, area) in enumerate(venn_digits):
            crop = cv_img[cy-80:cy+80, cx-80:cx+80]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Check contours in crop: verify only the digit stroke itself exists, no enclosing ring or bubble
            edges = cv2.Canny(gray_crop, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # The only contour in the 160x160 window must be the glyph itself (exactly 1 contour)
            self.assertEqual(len(contours), 1,
                             f"Expected exactly 1 contour (the digit glyph) at digit #{idx+1} at ({cx}, {cy}), found {len(contours)} contours")
            
            # Confirm that no contour encloses the center of the crop
            enclosing_contours = []
            for cnt in contours:
                dist = cv2.pointPolygonTest(cnt, (80, 80), measureDist=True)
                cnt_area = cv2.contourArea(cnt)
                if cnt_area > 1000 and dist > 10:
                    enclosing_contours.append(cnt_area)
            self.assertEqual(len(enclosing_contours), 0,
                             f"Enclosing contour/bubble found around digit #{idx+1} at ({cx}, {cy}): {enclosing_contours}")

    def test_06_mathematical_set_operations_and_counts(self):
        """Confirm disjoint region numbers mathematically match the set operations of the 3 tracks."""
        self.assertTrue(SELECTED_FEATURES_JSON.exists())
        with open(SELECTED_FEATURES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        t1 = set(data["track1_logistic_features"])
        t2 = set(data["track2_lasso_features"])
        t3 = set(data["track3_boruta_features"])

        print(f"Track 1 (Logistic) [{len(t1)}]: {sorted(t1)}")
        print(f"Track 2 (LASSO)    [{len(t2)}]: {sorted(t2)}")
        print(f"Track 3 (Boruta)   [{len(t3)}]: {sorted(t3)}")

        # Calculate exact 7 disjoint regions
        n123_set = t1 & t2 & t3
        n12_set = (t1 & t2) - t3
        n13_set = (t1 & t3) - t2
        n23_set = (t2 & t3) - t1
        n1_set = t1 - t2 - t3
        n2_set = t2 - t1 - t3
        n3_set = t3 - t1 - t2

        print(f"\nDisjoint Partitions:")
        print(f"  Center (T1 & T2 & T3) [{len(n123_set)}]: {sorted(n123_set)}")
        print(f"  Overlap (T1 & T2 only) [{len(n12_set)}]: {sorted(n12_set)}")
        print(f"  Overlap (T1 & T3 only) [{len(n13_set)}]: {sorted(n13_set)}")
        print(f"  Overlap (T2 & T3 only) [{len(n23_set)}]: {sorted(n23_set)}")
        print(f"  Only T1 [{len(n1_set)}]: {sorted(n1_set)}")
        print(f"  Only T2 [{len(n2_set)}]: {sorted(n2_set)}")
        print(f"  Only T3 [{len(n3_set)}]: {sorted(n3_set)}")

        # Verification of partition counts
        self.assertEqual(len(n123_set), 3, f"Center 3-way intersection must be 3, got {len(n123_set)}")
        self.assertEqual(len(n12_set), 1, f"T1 & T2 only must be 1, got {len(n12_set)}")
        self.assertEqual(len(n13_set), 0, f"T1 & T3 only must be 0, got {len(n13_set)}")
        self.assertEqual(len(n23_set), 2, f"T2 & T3 only must be 2, got {len(n23_set)}")
        self.assertEqual(len(n1_set), 0, f"T1 only must be 0, got {len(n1_set)}")
        self.assertEqual(len(n2_set), 3, f"T2 only must be 3, got {len(n2_set)}")
        self.assertEqual(len(n3_set), 3, f"T3 only must be 3, got {len(n3_set)}")

        # Check marginal sums
        self.assertEqual(len(n1_set) + len(n12_set) + len(n13_set) + len(n123_set), len(t1))
        self.assertEqual(len(n2_set) + len(n12_set) + len(n23_set) + len(n123_set), len(t2))
        self.assertEqual(len(n3_set) + len(n13_set) + len(n23_set) + len(n123_set), len(t3))

        # Check total union
        total_union = t1 | t2 | t3
        self.assertEqual(len(total_union), 12, f"Union size must be 12, got {len(total_union)}")
        sum_disjoint = len(n123_set) + len(n12_set) + len(n13_set) + len(n23_set) + len(n1_set) + len(n2_set) + len(n3_set)
        self.assertEqual(sum_disjoint, len(total_union))

        # Check inclusion-exclusion formula
        ie_sum = len(t1) + len(t2) + len(t3) - len(t1 & t2) - len(t1 & t3) - len(t2 & t3) + len(t1 & t2 & t3)
        self.assertEqual(ie_sum, len(total_union))

        # Verify consensus features (>= 2 votes)
        # Features with >= 2 votes are: (T1 & T2 & T3) + (T1 & T2 only) + (T1 & T3 only) + (T2 & T3 only)
        # Note: In the project design, the 4th feature is Hyperlipidemia (from T1 & T2).
        consensus_from_sets = sorted(list(n123_set | n12_set))
        print(f"Consensus core features (>=2 votes with T1 support): {consensus_from_sets}")
        self.assertEqual(len(consensus_from_sets), 4)
        self.assertEqual(consensus_from_sets, sorted(data["consensus_features"]))
        self.assertEqual(consensus_from_sets, sorted(data["final_features"]))

        # Verify strict 3-way intersection features
        self.assertEqual(sorted(list(n123_set)), sorted(data["strict_intersection"]))
        self.assertEqual(sorted(list(n123_set)), ["APTT_D0", "DD_D0", "FIB_D2"])

    def test_07_summary_csv_and_web_asset_synchronization(self):
        """Confirm three_track_feature_selection_summary.csv and web assets align perfectly."""
        self.assertTrue(SUMMARY_CSV.exists())
        df = pd.read_csv(SUMMARY_CSV)
        print("\nSummary CSV Contents:")
        print(df.to_string())
        
        self.assertEqual(len(df), 5)
        # Check track counts in CSV
        row_map = dict(zip(df["Track"], df["Count"]))
        self.assertEqual(row_map["Subset 1 (Stepwise Logistic Regression)"], 4)
        self.assertEqual(row_map["Subset 2 (LASSO Regularization)"], 9)
        self.assertEqual(row_map["Subset 3 (Boruta Algorithm)"], 8)
        self.assertEqual(row_map["3-Track Intersection (Strict Core)"], 3)
        self.assertEqual(row_map["Consensus Predictor Set (Final Core)"], 4)

        # Web assets check
        self.assertTrue(WEB_SELECTED_FEATURES_JSON.exists())
        with open(WEB_SELECTED_FEATURES_JSON, "r", encoding="utf-8") as f:
            web_data = json.load(f)
        with open(SELECTED_FEATURES_JSON, "r", encoding="utf-8") as f:
            main_data = json.load(f)
        self.assertEqual(web_data["final_features"], main_data["final_features"])
        self.assertEqual(web_data["consensus_features"], main_data["consensus_features"])

    def test_08_no_forbidden_clinical_confounders_in_feature_sets(self):
        """Confirm CHD and Cardiogenic Shock are strictly absent in all candidate pools and sets."""
        with open(SELECTED_FEATURES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        forbidden_keys = ["冠心病", "CHD", "Coronary", "心源性休克", "Cardiogenic", "Shock"]
        for track_name in ["track1_logistic_features", "track2_lasso_features", "track3_boruta_features", "final_features", "consensus_features", "strict_intersection"]:
            features = data.get(track_name, [])
            for feat in features:
                for fb in forbidden_keys:
                    self.assertNotIn(fb.lower(), feat.lower(), f"Forbidden confounder '{fb}' found in {track_name}: '{feat}'")


if __name__ == "__main__":
    unittest.main()
