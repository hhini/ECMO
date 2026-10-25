# ECMO Thrombosis Risk AI — Clinical Decision Support Platform
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Enterprise-Grade Clinical Informatics Platform for Individualized Thrombosis Risk Stratification in Patients on Extracorporeal Membrane Oxygenation (ECMO).**

---

## 🌟 Overview & Key Highlights

This repository contains the full clinical machine learning pipeline and interactive web decision support system for predicting **Lower Extremity Thrombosis (LET)** and systemic thrombosis during ECMO support.

* **Champion Model**: **LightGBM (LGBM)** demonstrated superior discrimination and calibration:
  * **Stage 1 10-Fold Cross-Validation AUC**: **0.881** (95% CI: 0.842 – 0.920)
  * **Stage 2 Blind External Validation AUC**: **0.829** (95% CI: 0.735 – 0.923) across $N=106$ pure real-world patients (25 positive vs 81 negative) without refitting.
  * **Clinical Utility**: Outperformed LR ($P=0.018$), RF ($P=0.031$), and SVM ($P=0.012$) in DeLong non-parametric testing with optimal Net Benefit in Decision Curve Analysis (DCA).
* **Consensus 5 Biomarkers**: Screened via **3-Track Consensus (Logistic + LASSO + 60-iteration Boruta Shadow)**:
  1. `DD_D0`: Baseline D-dimer (ug/mL)
  2. `DD_Late`: Peak/Late D-dimer during ECMO run (ug/mL)
  3. `WBC_D0`: Baseline White Blood Cell Count ($10^9$/L)
  4. `ALT_D7`: Day 7 Alanine Aminotransferase (U/L)
  5. `FIB_D2`: Day 2 Fibrinogen (g/L)
* **Comprehensive 17-Figure SCI Visual Suite**: 300-DPI publication-grade visualizations for CONSORT flowchart, Boruta shadow boxplots, Venn diagrams, split-violins, DCA/CIC, and collision-free SHAP interpretability.

---

## 🚀 Quick Start (Local Run)

Clone the repository and install dependencies:

```bash
git clone https://github.com/hhini/ECMO.git
cd ECMO
pip install -r requirements.txt
```

Launch the interactive web platform:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment Guide

To deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Log in to **Streamlit Community Cloud** with your GitHub account.
2. Click **New app**.
3. Configure the deployment settings:
   * **Repository**: `hhini/ECMO`
   * **Branch**: `main`
   * **Main file path**: `app.py` (or `web_app/app.py`)
4. Click **Deploy**!

The application includes zero-dependency fallbacks, self-contained pipeline transformers, and pre-packaged model assets for 100% crash-free cloud execution.

---

## 📁 Repository Structure

```text
ECMO/
├── app.py                          # Root entry point (redirects seamlessly to web_app/app.py)
├── requirements.txt                # Production dependency specification
├── README.md                       # Comprehensive project documentation
├── PROJECT.md                      # Engineering design manifesto & clinical specifications
│
├── web_app/                        # Interactive Enterprise Clinical Web Application
│   ├── app.py                      # Main Streamlit application with 5 functional tabs
│   ├── requirements.txt            # Web application dependencies
│   ├── sample_ecmo_cohort.csv      # Sample multi-patient cohort for batch screening
│   ├── model_assets/               # Pre-trained ML pipelines, metrics, and SHAP backgrounds
│   └── static/sci_figures/         # 17 publication-grade high-resolution figures
│
├── report/                         # Full Clinical Research Report & Artifacts
│   ├── ECMO_Thrombosis_Full_Research_Report.docx # Complete manuscript with 20 tables & 17 figures
│   ├── ECMO_Thrombosis_Full_Research_Report.md   # Markdown summary report
│   └── sci_figures/                # 17 high-resolution publication PNGs (fig01 to fig17)
│
├── scripts/                        # Reproducible Research Pipeline Scripts
│   ├── 01_stage2_data_cleaning.py
│   ├── 02_generate_table1.py
│   ├── 03_benchmark_imbalance.py
│   ├── 04_tri_method_feature_selection.py
│   ├── 05_train_evaluate_models.py
│   ├── 06_external_validation.py
│   ├── 07_shap_and_deployment.py
│   ├── build_all_inclusive_report.py
│   └── generate_all_sci_visualizations.py
│
└── tests/                          # 63 automated tests & 35 E2E integration verification tests
```

---

## 🔬 Web Application Modules

* **Tab 1: 🏥 Individual Risk Calculator**: Interactive sliders, clinical presets, real-time risk gauge, 4-tier alert badge, patient-level SHAP waterfalls, and evidence-based anticoagulation guidance.
* **Tab 2: 📊 Batch Cohort Screening**: Upload CSV/Excel cohorts, automatic median imputation, batch risk stratification, and one-click summary report export.
* **Tab 3: 🔬 6-Model Benchmark Arena**: Performance comparisons (AUC, Sensitivity, Specificity, Brier score), DeLong matrices, radar charts, and confusion matrix heatmaps.
* **Tab 4: 📈 17 SCI Visualizations Gallery**: Publication gallery with categorized filters, clinical interpretability explanations, and one-click 300-DPI downloads.
* **Tab 5: 📑 Methodology Whitepaper**: Full CONSORT audit, transparent Model Card, hyperparameter configurations, and clinical governance notes.

---

## 📄 Citation & License

This project is licensed under the MIT License. Academic research citing this system should reference the accompanying clinical research report in `report/ECMO_Thrombosis_Full_Research_Report.docx`.
