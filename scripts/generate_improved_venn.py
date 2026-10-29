# -*- coding: utf-8 -*-
"""
Generate Pure English Publication-Grade Three-Track Feature Selection Venn Diagram
Aesthetics: Nature Medicine / Lancet Digital Health / CNS
Requirements:
- Dynamically reads feature subsets from output/04_feature_selection/selected_features.json
- 100% English typography ('Stepwise Logistic Regression', 'LASSO Regularization', 'Boruta Algorithm')
- No 'Figure 5' title or header banner
- No right-side panel
- Pure Venn diagram only
- Clean, moderate-sized bold numbers without enclosing circles
- 300 DPI publication resolution saved to report/sci_figures and output/04_feature_selection
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Set, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_ROOT = Path(r"D:\acode\ECMO").resolve()
SELECTED_FEATURES_PATH = PROJECT_ROOT / "output" / "04_feature_selection" / "selected_features.json"

DEFAULT_OUTPUT_PATHS = [
    PROJECT_ROOT / "report" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png",
    PROJECT_ROOT / "report" / "figures" / "fig05_step4_three_track_venn_diagram.png",
    PROJECT_ROOT / "output" / "04_feature_selection" / "three_track_venn_diagram.png",
    PROJECT_ROOT / "output" / "04_feature_selection" / "02_feature_selection_venn.png",
    PROJECT_ROOT / "output" / "04_feature_selection" / "02_feature_overlap_venn.png",
    PROJECT_ROOT / "web_app" / "static" / "sci_figures" / "fig05_step4_three_track_venn_diagram.png",
]


def load_feature_sets() -> Tuple[Set[str], Set[str], Set[str]]:
    """Load the 3 independent feature selection subsets from selected_features.json."""
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(f"selected_features.json not found at: {SELECTED_FEATURES_PATH}")
        
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    set1 = set(data.get("track1_logistic_features") or data.get("statistical_features") or [])
    set2 = set(data.get("track2_lasso_features") or data.get("lasso_features") or [])
    set3 = set(data.get("track3_boruta_features") or data.get("boruta_features") or [])
    
    if not set1 or not set2 or not set3:
        raise ValueError(f"One or more feature tracks in {SELECTED_FEATURES_PATH} is empty: Track1={len(set1)}, Track2={len(set2)}, Track3={len(set3)}")
        
    return set1, set2, set3


def create_pure_english_venn(output_paths: Optional[List[Path]] = None):
    """Render publication-grade vector Venn diagram."""
    if output_paths is None:
        output_paths = DEFAULT_OUTPUT_PATHS
        
    set1, set2, set3 = load_feature_sets()
    
    # Calculate 7 disjoint subset counts
    n123 = len(set1 & set2 & set3)
    n12 = len((set1 & set2) - set3)
    n13 = len((set1 & set3) - set2)
    n23 = len((set2 & set3) - set1)
    n1 = len(set1 - set2 - set3)
    n2 = len(set2 - set1 - set3)
    n3 = len(set3 - set1 - set2)
    
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 300 DPI publication resolution
    fig = plt.figure(figsize=(7.8, 7.0), facecolor='white', dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor('white')
    ax.set_xlim(-2.10, 2.10)
    ax.set_ylim(-2.10, 1.95)
    ax.axis('off')
    
    # Geometry of the 3 circles - perfectly balanced & centered
    r = 1.15
    c1 = (-0.58, 0.25)   # Top-Left: Stepwise Logistic Regression
    c2 = (0.58, 0.25)    # Top-Right: LASSO Regularization
    c3 = (0.00, -0.65)   # Bottom-Center: Boruta Algorithm
    
    # Sophisticated journal color palette: Navy Slate, Crimson Wine, Emerald Sage
    color_t1 = '#1D3557'  # Deep Navy Slate
    color_t2 = '#9E2A2B'  # Crimson Wine
    color_t3 = '#2D6A4F'  # Emerald Sage
    
    circle1 = Circle(c1, r, facecolor=color_t1, alpha=0.19, edgecolor=color_t1, lw=2.2, zorder=2)
    circle2 = Circle(c2, r, facecolor=color_t2, alpha=0.17, edgecolor=color_t2, lw=2.2, zorder=2)
    circle3 = Circle(c3, r, facecolor=color_t3, alpha=0.17, edgecolor=color_t3, lw=2.2, zorder=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Pure English Set Labels - Bold, elegant, professional typography
    ax.text(c1[0] - 0.25, c1[1] + r + 0.12, f'Stepwise Logistic Regression\n({len(set1)} Features)',
            fontsize=11.0, fontweight='bold', color='#1D3557', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c2[0] + 0.25, c2[1] + r + 0.12, f'LASSO Regularization\n({len(set2)} Features)',
            fontsize=11.0, fontweight='bold', color='#9E2A2B', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c3[0], c3[1] - r - 0.14, f'Boruta Algorithm\n({len(set3)} Features)',
            fontsize=11.0, fontweight='bold', color='#2D6A4F', ha='center', va='top', linespacing=1.2, zorder=5)
    
    # Centered moderate-sized bold numbers WITHOUT enclosing circles
    # Center (3-way intersection)
    if n123 > 0:
        ax.text(0.00, 0.02, str(n123), fontsize=14, fontweight='bold', color='#0F172A', ha='center', va='center', zorder=6)
        
    # Overlap Track 1 & Track 2 only
    if n12 > 0:
        ax.text(0.00, 0.52, str(n12), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    # Overlap Track 1 & Track 3 only
    if n13 > 0:
        ax.text(-0.48, -0.28, str(n13), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    # Overlap Track 2 & Track 3 only
    if n23 > 0:
        ax.text(0.48, -0.28, str(n23), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    # Only Track 1
    if n1 > 0:
        ax.text(-1.05, 0.35, str(n1), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    # Only Track 2
    if n2 > 0:
        ax.text(1.05, 0.35, str(n2), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    # Only Track 3
    if n3 > 0:
        ax.text(0.00, -1.10, str(n3), fontsize=12.5, fontweight='semibold', color='#334155', ha='center', va='center', zorder=6)
        
    for p in output_paths:
        p_obj = Path(p)
        p_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p_obj, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"[OK] Pure English Venn diagram (300 DPI) saved to: {p_obj}")
        
    plt.close()


if __name__ == "__main__":
    create_pure_english_venn()
