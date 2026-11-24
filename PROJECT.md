# Project: ECMO Support Lower Extremity Thrombosis (LET) Predictive Modeling & Bedside Decision Support

## 1. System Architecture & The 9-Step Protocol
This project executes an end-to-end clinical machine learning validation pipeline adhering strictly to **TRIPOD+AI** reporting standards.

```
[Derivation Cohort (n=266)] ────────────────────────┐
                                                    ▼
[External Test Cohort (n=106)] ────────> [Step 1: Baseline Comparison & Balance Diagnosis (Merged Z / χ²)]
                                                    │
                                                    ▼
                                         [Step 2: Stratified Split]
                                                    │
                                ┌───────────────────┴───────────────────┐
                                ▼                                       ▼
                    [Training Cohort (n=186)]            [Frozen Internal Val (n=80)] & [Frozen External Val (n=106)]
                                │                                       │
                                ▼                                       │
            [Step 3: Missing Audit & Distance-Weighted KNN (k=5)]       │
            (Purge Day-15 >50% missing; Fit KNN strictly on Train)      │
                                │                                       │
                                ▼                                       │
            [Step 4: Train-Only VIF > 10 Collinearity Pruning]          │
            (Remove Delta_FIB, PLT_D0, AT3_D0, pH_D0 -> 59 remain)      │
                                │                                       │
                                ▼                                       │
            [Step 5: 10×5-Fold Repeated CV LASSO Regression]            │
            (Lock exactly 6 Core Predictors: DD_D0, FIB_D2, ...)        │
                                │                                       │
                                ▼                                       │
            [Step 6: Cost-Sensitive Balancing (class_weight='balanced')]│
                                │                                       │
                                ▼                                       │
            [Step 7: Optuna TPE Bayesian Hyperparameter Optimization]   │
            (3x5-CV TPE Tuning on Train -> 10x5-CV + 1000 Bootstrap)   │
                                │                                       │
                                └───────────────────┬───────────────────┘
                                                    ▼
                                [Step 8: Dual-Blind Cohort Validation & Optimal Model Selection]
                                - Model Champion: SVM (RBF kernel, C=5.658, gamma=0.05223)
                                - Internal Val AUROC = 0.884, External Val AUROC = 0.771
                                - DeLong Tests, Calibration Brier, DCA Net Benefit
                                                    │
                                                    ▼
                                [Step 9: Global & Local SHAP Interpretability]
                                - Beeswarm Summary, Dependence Curves & Individual Patient Waterfalls
                                - Interactive Streamlit Clinical Web Calculator & Batch Screener
```

---

## 2. Core Feature Inventory (6 LASSO Predictors)
| Feature | Clinical Name | Role | Transform / Unit | Pathophysiological Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `DD_D0` | Baseline D-Dimer | Predictor 1 | $\log(1+x)$, $\mu$g/mL | Pre-existing microthrombosis & hyperfibrinolysis |
| `FIB_D2` | Day 2 Fibrinogen | Predictor 2 | Continuous, g/L | Core structural substrate of dense fibrin meshwork |
| `ECMO_Duration` | ECMO Duration | Predictor 3 | Continuous, hours | Cumulative foreign circuit & cannula shear exposure |
| `CRP_D0` | Baseline CRP | Predictor 4 | Continuous, mg/L | Acute inflammatory storm & endothelial procoagulant activation |
| `PaO2_D0` | Baseline PaO2 | Predictor 5 | Continuous, mmHg | Severe hypoxia-induced prothrombotic state |
| `Pre_ECMO_Surgery`| Pre-ECMO Surgery | Predictor 6 | Binary, 0/1 | Surgical trauma-induced tissue factor & extrinsic coagulation surge |

---

## 3. Implementation Scripts & Interfaces
- **`scripts/run_revised_pipeline.py`**: The complete reproducible engine executing Steps 2 through 9.
- **`scripts/generate_comprehensive_tables.py`**: Generates Table 1 and Table 2 with merged Z / χ² statistics.
- **`scripts/build_chinese_research_report.py`**: Compiles `report/ECMO_Thrombosis_Full_Research_Report.docx` and `.md` using live model outputs.
- **`web_app/app.py`**: Streamlit bedside decision support application hosting all 6 models, batch screening, benchmark arena, 14 SCI figures, and clinical whitepaper.
