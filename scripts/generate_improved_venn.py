# -*- coding: utf-8 -*-
"""
Generate Pure Publication-Grade Three-Track Feature Selection Venn Diagram
Style: Nature / Lancet Digital Health / CNS
Requirements:
- All English (全英文)
- No 'Figure 5' title or header banner (不要figure5这样的东西)
- No right-side panel (右边的不需要)
- Pure Venn diagram only (就是venn图就行)
- No small annotations above or below (上面和下面的小字也不要)
- Clean, moderate-sized numbers without any wrapping circles (数字别太大，别是圆包裹，不大不小)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def create_pure_english_venn(output_paths):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Clean balanced aspect ratio, 300 DPI publication resolution
    fig = plt.figure(figsize=(8.0, 7.2), facecolor='white', dpi=300)
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
    ax.set_facecolor('white')
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-2.15, 1.95)
    ax.axis('off')
    
    # Geometry of the 3 circles - perfectly balanced & centered
    r = 1.15
    c1 = (-0.58, 0.25)   # Top-Left: Stepwise Logistic Regression
    c2 = (0.58, 0.25)    # Top-Right: LASSO Regularization
    c3 = (0.00, -0.65)   # Bottom-Center: Boruta Algorithm
    
    # Sophisticated journal color palette
    color_t1 = '#1E40AF'  # Royal Blue
    color_t2 = '#DC2626'  # Crimson Red
    color_t3 = '#0D9488'  # Emerald Teal
    
    circle1 = Circle(c1, r, facecolor=color_t1, alpha=0.18, edgecolor=color_t1, lw=2.4, zorder=2)
    circle2 = Circle(c2, r, facecolor=color_t2, alpha=0.16, edgecolor=color_t2, lw=2.4, zorder=2)
    circle3 = Circle(c3, r, facecolor=color_t3, alpha=0.16, edgecolor=color_t3, lw=2.4, zorder=2)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    
    # Pure English Set Labels - Bold, elegant, professional typography
    ax.text(c1[0] - 0.25, c1[1] + r + 0.12, 'Stepwise Logistic Regression\n(5 Features)',
            fontsize=11.5, fontweight='bold', color='#1E3A8A', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c2[0] + 0.25, c2[1] + r + 0.12, 'LASSO Regularization\n(4 Features)',
            fontsize=11.5, fontweight='bold', color='#991B1B', ha='center', va='bottom', linespacing=1.2, zorder=5)
    
    ax.text(c3[0], c3[1] - r - 0.14, 'Boruta Algorithm\n(5 Features)',
            fontsize=11.5, fontweight='bold', color='#0F766E', ha='center', va='top', linespacing=1.2, zorder=5)
    
    # Center 3-way intersection: Plain moderate-sized number without any enclosing circle
    ax.text(0.00, 0.05, '4', fontsize=14, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
    
    # 2-way overlap between Track 1 & Track 3: Plain moderate-sized number without any enclosing circle
    ax.text(-0.55, -0.30, '1', fontsize=13, fontweight='bold', color='#1E293B', ha='center', va='center', zorder=6)
    
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        print(f"[OK] Pure English Venn diagram saved to: {path}")
        
    plt.close()

if __name__ == "__main__":
    out1 = os.path.join(r"D:\acode\ECMO\report\sci_figures", "fig05_step4_three_track_venn_diagram.png")
    out2 = os.path.join(r"D:\acode\ECMO\web_app\static\sci_figures", "fig05_step4_three_track_venn_diagram.png")
    out3 = os.path.join(r"C:\Users\Administrator\.gemini\antigravity\brain\8135a483-dc36-4da3-997c-c300a510ee39\sci_figures", "fig05_step4_three_track_venn_diagram.png")
    create_pure_english_venn([out1, out2, out3])
