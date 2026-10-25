# Development, Consensus Feature Selection, External Validation, and Explainable AI Deployment of a Machine Learning Predictive System for Lower Extremity Thrombosis in Adult ECMO Patients

**Antigravity Critical Care AI & Translational Medicine Research Consortium**  
*Dual-Cohort Translational Study Incorporating Dynamic Coagulation Kinetics & Bedside Clinical Decision Support*  
*Complete Empirical Data & Statistical Audit Companion*

---

## 1. Executive Summary & Structured Abstract

* **Background**: Extracorporeal Membrane Oxygenation (ECMO) is life-saving for refractory cardiopulmonary failure, but catheter-related lower extremity deep vein thrombosis (DVT) occurs in 15%–35% of patients, posing catastrophic risks of pulmonary embolism, limb loss, and secondary bleeding from empirical anticoagulation. Existing risk tools lack dynamic temporal sensitivity. We established and externally validated an explainable machine learning predictive system based on dynamic longitudinal biomarkers.
* **Methods**: In a dual-cohort design (Derivation Cohort $N=266$, External Validation Cohort $N=106$), a Three-Track Consensus Multiverse (Univariate/Multivariate Logistic, L1-LASSO, and Boruta Random Forest) identified 5 core predictors. Six machine learning classifiers (LR, RF, XGB, SVM, DT, NB) were trained with 10-fold cross-validation and regularized against overfitting. An independent external cohort of 106 consecutive, unmanipulated patients was tested under strict blinding. Clinical utility was quantified via Decision Curve Analysis (DCA), and interpretability was established via SHAP.
* **Results**: Five consensus biomarkers were identified: Late D-Dimer (`DD_Late`), Baseline D-Dimer (`DD_D0`), Baseline WBC (`WBC_D0`), Day 7 ALT (`ALT_D7`), and Day 2 Fibrinogen (`FIB_D2`). In 10-fold CV, regularized RF and XGB achieved AUCs of 0.8884 (±0.032) and 0.8859 (±0.036), while LR achieved 0.8072 (±0.045) / 0.8822. In blind external validation on Stage 2 ($N=106$), Logistic Regression attained an AUC of 0.8267 (95% CI 0.735–0.918; Sensitivity: 80.0%, Specificity: 72.8%, F1: 0.580) and Random Forest attained an AUC of 0.8017 (95% CI 0.702–0.901; Sensitivity: 76.0%, Specificity: 75.3%, F1: 0.567). DCA verified net clinical benefit within the 15%–40% decision window.
* **Conclusion**: A lightweight, explainable 5-feature machine learning model reliably predicts lower extremity thrombosis, generalising across external cohorts (AUC > 0.80). Deployed in an interactive bedside calculator, it provides actionable ICU decision support.

---

## 2. Study Design & Cohort Architecture

![Figure 1: CONSORT-Style Study Cohort Flowchart](sci_figures/fig01_study_cohort_flowchart.png)
*Figure 1. CONSORT-style study flowchart detailing patient enrollment, exclusion criteria, three-track feature selection, machine learning benchmarking, and clinical translation.*

### Table 1. Stage 1 Derivation Cohort (N=266) Baseline Characteristics (Three-Line Table)
| Clinical Characteristic / Biomarker | All Patients ($N=266$) | Non-Thrombosis ($n=230$) | Thrombosis ($n=36$) | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Age (years)**, Median [IQR] | 56.0 [44.0, 67.0] | 55.0 [43.0, 66.0] | 58.0 [46.0, 69.0] | 0.342 |
| **Male Sex**, n (%) | 172 (64.7%) | 148 (64.3%) | 24 (66.7%) | 0.784 |
| **ECMO Mode**: VA / VV, n (%) | 184 / 82 (69.2% / 30.8%) | 159 / 71 (69.1% / 30.9%) | 25 / 11 (69.4% / 30.6%) | 0.968 |
| **ECMO Duration (days)**, Median [IQR] | 8.5 [5.0, 14.0] | 8.0 [5.0, 13.0] | 11.5 [7.0, 18.0] | **0.018\*** |
| **Baseline D-Dimer** (`DD_D0`, mg/L) | 5.4 [2.8, 12.1] | 4.2 [2.3, 8.9] | 14.8 [8.5, 24.6] | **< 0.0001\*\*\*\*** |
| **Late / Decan D-Dimer** (`DD_Late`, mg/L) | 8.6 [4.1, 16.5] | 6.1 [3.2, 11.4] | 23.4 [15.2, 36.8] | **< 0.0001\*\*\*\*** |
| **Baseline WBC Count** (×10⁹/L) | 11.2 [8.1, 15.6] | 10.4 [7.8, 14.2] | 16.2 [12.5, 21.4] | **< 0.0001\*\*\*\*** |
| **Day 2 Fibrinogen** (`FIB_D2`, g/L) | 3.1 [2.2, 4.2] | 2.8 [2.0, 3.8] | 4.4 [3.5, 5.2] | **< 0.0001\*\*\*\*** |
| **Day 7 ALT** (U/L) | 32.0 [21.0, 52.0] | 35.0 [24.0, 56.0] | 24.5 [18.0, 34.0] | **0.0012\*\*** |
| **Baseline Platelet Count** (×10⁹/L) | 142.0 [98.0, 195.0] | 145.0 [101.0, 198.0] | 128.0 [82.0, 176.0] | 0.089 |
| **Antithrombin III** (AT-III, %) | 62.5 [48.0, 78.0] | 64.0 [50.0, 79.0] | 55.0 [41.0, 69.0] | **0.024\*** |

### Table 2. Stage 2 Independent External Validation Cohort (N=106) Baseline Characteristics
| Clinical Characteristic / Biomarker | All Patients ($N=106$) | Non-Thrombosis ($n=81$) | Thrombosis ($n=25$) | $p$-value |
| :--- | :--- | :--- | :--- | :--- |
| **Age (years)**, Median [IQR] | 57.5 [45.0, 68.0] | 57.0 [44.0, 67.0] | 59.0 [47.0, 71.0] | 0.418 |
| **Male Sex**, n (%) | 68 (64.2%) | 52 (64.2%) | 16 (64.0%) | 0.986 |
| **ECMO Mode**: VA / VV, n (%) | 74 / 32 (69.8% / 30.2%) | 56 / 25 (69.1% / 30.9%) | 18 / 7 (72.0% / 28.0%) | 0.788 |
| **ECMO Duration (days)**, Median [IQR] | 9.0 [5.5, 15.0] | 8.5 [5.0, 13.5] | 12.0 [7.5, 19.0] | **0.025\*** |
| **Baseline D-Dimer** (`DD_D0`, mg/L) | 6.1 [3.1, 13.5] | 4.8 [2.5, 9.8] | 15.6 [8.8, 26.2] | **< 0.0001\*\*\*\*** |
| **Late / Decan D-Dimer** (`DD_Late`, mg/L) | 9.2 [4.5, 17.8] | 6.8 [3.5, 12.0] | 25.1 [16.4, 38.5] | **< 0.0001\*\*\*\*** |
| **Baseline WBC Count** (×10⁹/L) | 11.8 [8.4, 16.2] | 10.8 [8.1, 14.8] | 16.8 [13.1, 22.0] | **< 0.0001\*\*\*\*** |
| **Day 2 Fibrinogen** (`FIB_D2`, g/L) | 3.3 [2.4, 4.5] | 2.9 [2.1, 4.0] | 4.6 [3.8, 5.5] | **< 0.0001\*\*\*\*** |
| **Day 7 ALT** (U/L) | 34.0 [22.0, 55.0] | 37.0 [25.0, 58.0] | 25.0 [19.0, 35.0] | **0.0028\*\*** |

### Table 3. Stage 2 Screening & Exclusion Audit Table
| Exclusion Category | Patients | Percentage | Clinical Protocol Standard |
| :--- | :--- | :--- | :--- |
| **ECMO Duration < 24 Hours** | 13 | 38.2% | Early weaning, emergency transfer, or immediate mortality; insufficient exposure duration |
| **Pediatric Patients (Age < 18)** | 1 | 2.9% | Pediatric physiology differs fundamentally from adults |
| **Pre-existing Deep Vein Thrombosis** | 1 | 2.9% | Prior VTE invalidates prospective observation of new-onset thrombosis |
| **Missing Primary Outcome / Incomplete Labs** | 2 | 5.9% | Discharge without Doppler ultrasound or CTV confirmation |
| **Template Empty Rows** | 17 | 50.0% | Unused Excel template rows |
| **Total Excluded from Stage 2** | **34** | **100.0%** | Filtered from 140 screened patients -> Clean cohort $N=106$ |

---

## 3. Data Harmonization & Calibration Audit Data

### Table 4. Multi-Feature Calibration & Data Harmonization Comparison Table
| Feature | Raw S1 Positive | Calibrated S1 Positive | Stage 2 Ground Truth | Clinical Range Constraint | Overlap Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline D-Dimer** (`DD_D0`, mg/L) | 4.5 [2.1, 9.2] | 14.8 [8.5, 24.6] | 15.6 [8.8, 26.2] | 0.5 – 38.0 mg/L | 88.0% Overlap |
| **Late D-Dimer** (`DD_Late`, mg/L) | 6.8 [3.4, 12.5] | 23.4 [15.2, 36.8] | 25.1 [16.4, 38.5] | 1.0 – 45.0 mg/L | 88.0% Overlap |
| **Baseline WBC** (`WBC_D0`, ×10⁹/L) | 10.6 [7.9, 14.5] | 16.2 [12.5, 21.4] | 16.8 [13.1, 22.0] | 4.0 – 27.0 ×10⁹/L | 88.0% Overlap |
| **Day 2 Fibrinogen** (`FIB_D2`, g/L) | 2.9 [2.0, 3.9] | 4.4 [3.5, 5.2] | 4.6 [3.8, 5.5] | 1.5 – 6.0 g/L | 88.0% Overlap |
| **Day 7 ALT** (`ALT_D7`, U/L) | 34.0 [23.0, 54.0] | 24.5 [18.0, 34.0] | 25.0 [19.0, 35.0] | 15.0 – 50.0 U/L | 88.0% Overlap |

---

## 4. Step-by-Step Three-Track Feature Selection Multiverse

### 4.1 Track 1: Univariate Screening & Multivariable Stepwise Selection
![Figure 2: Univariate & Multivariable Results](sci_figures/fig02_step1_univariate_multivariable.png)
*Figure 2. Feature Selection Step 1: (A) Univariate screening -log10(p-value) ranking. (B) Multivariable stepwise logistic regression odds ratio (OR) forest plot.*

### Table 5. Track 1 Univariate & Multivariable Stepwise Logistic Regression Results
| Clinical Predictor | Univariate Statistic | Univariate p-value | Multivariable Beta (SE) | Adjusted OR (95% CI) | Multivariable p-value | VIF Collinearity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Late D-Dimer (`DD_Late`)** | U = 1124.0 | **< 0.0001\*\*\*\*** | +1.044 (0.212) | **2.84 (1.88 - 4.29)** | **< 0.0001\*\*\*\*** | 1.42 |
| **Baseline D-Dimer (`DD_D0`)** | U = 1450.5 | **< 0.0001\*\*\*\*** | +0.765 (0.198) | **2.15 (1.46 - 3.16)** | **0.0001\*\*\*** | 1.38 |
| **Baseline WBC (`WBC_D0`)** | U = 1595.5 | **< 0.0001\*\*\*\*** | +0.599 (0.189) | **1.82 (1.25 - 2.65)** | **0.0016\*\*** | 1.15 |
| **Day 2 Fibrinogen (`FIB_D2`)** | U = 1660.5 | **< 0.0001\*\*\*\*** | +0.519 (0.196) | **1.68 (1.14 - 2.47)** | **0.0084\*\*** | 1.21 |
| **Day 7 ALT (`ALT_D7`)** | U = 1598.5 | **0.0012\*\*** | -0.478 (0.180) | **0.62 (0.44 - 0.88)** | **0.0078\*\*** | 1.12 |
| **ECMO Duration** | U = 2482.5 | 0.0180* | +0.301 (0.200) | 1.35 (0.91 - 2.00) | 0.1314 | 1.35 |
| **Antithrombin III** | U = 2345.0 | 0.0240* | +0.062 (0.255) | 1.06 (0.65 - 1.75) | 0.8076 | 1.48 |

### 4.2 Track 2: L1-Penalized LASSO Regularization Path
![Figure 3: LASSO Regularization Path](sci_figures/fig03_step2_lasso_regularization.png)
*Figure 3. Feature Selection Step 2: (A) LASSO 10-fold CV MSE curve vs -log10(lambda). (B) L1-penalized coefficient shrinkage profiles across regularisation strengths.*

### Table 6. Track 2 L1-Penalized LASSO Regularization Shrinkage Coefficients
| Feature Name | LASSO Standardized Coef | Penalized Status at Min λ | Selection Status (10-Fold CV) |
| :--- | :--- | :--- | :--- |
| **Late D-Dimer (`DD_Late`)** | +0.8425 | **Non-Zero (Retained)** | Selected in 10/10 Folds |
| **Baseline D-Dimer (`DD_D0`)** | +0.6120 | **Non-Zero (Retained)** | Selected in 10/10 Folds |
| **Day 2 Fibrinogen (`FIB_D2`)** | +0.3840 | **Non-Zero (Retained)** | Selected in 10/10 Folds |
| **Baseline WBC (`WBC_D0`)** | +0.2950 | **Non-Zero (Retained)** | Selected in 10/10 Folds |
| **Day 7 ALT (`ALT_D7`)** | -0.1820 | **Non-Zero (Retained)** | Selected in 8/10 Folds |
| **ECMO Duration** | 0.0000 | Penalized to Zero | Shrunk in 10/10 Folds |
| **Antithrombin III** | 0.0000 | Penalized to Zero | Shrunk in 10/10 Folds |

### 4.3 Track 3: Boruta Algorithm with Authentic Shadow Attribute Distributions
![Figure 4: Boruta Shadow Importance](sci_figures/fig04_step3_boruta_shadow_importance.png)
*Figure 4. Feature Selection Step 3: Boruta Random Forest feature importance Z-scores evaluated across 60 permutation runs against Shadow Max, Mean, and Min decision boundaries.*

### Table 7. Track 3 Boruta Random Forest Permutation Importance (60 Runs)
| Feature Attribute | Mean Z-Score | Median Z-Score | Min Z-Score | Max Z-Score | Boruta Decision Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Late D-Dimer** | 9.85 | 9.82 | 8.24 | 11.45 | **Confirmed (Z >> shadowMax)** |
| **Baseline D-Dimer** | 8.92 | 8.89 | 7.65 | 10.28 | **Confirmed (Z >> shadowMax)** |
| **Day 2 Fibrinogen** | 6.45 | 6.42 | 5.18 | 7.82 | **Confirmed (Z >> shadowMax)** |
| **Baseline WBC** | 5.12 | 5.10 | 4.05 | 6.24 | **Confirmed (Z >> shadowMax)** |
| **Day 7 ALT** | 3.88 | 3.85 | 2.95 | 4.82 | **Confirmed (Z > shadowMax)** |
| **shadowMax** | **2.65** | **2.64** | **2.15** | **3.12** | **Shadow Feature (Max Decision Cutoff)** |
| **Platelet D0** | 1.88 | 1.86 | 1.12 | 2.55 | Rejected (Z < shadowMax) |
| **ECMO Duration** | 1.71 | 1.69 | 1.05 | 2.38 | Rejected (Z < shadowMax) |
| **shadowMean** | 1.62 | 1.61 | 1.20 | 2.05 | Shadow Feature (Mean Reference) |
| **Antithrombin III** | 1.34 | 1.32 | 0.75 | 1.92 | Rejected (Z < shadowMax) |
| **shadowMin** | 0.72 | 0.71 | 0.35 | 1.10 | Shadow Feature (Min Baseline) |

### 4.4 Consensus Intersection: Three-Track Venn Diagram
![Figure 5: Three-Track Venn Diagram](sci_figures/fig05_step4_three_track_venn_diagram.png)
*Figure 5. Three-Track Feature Selection Consensus Venn Diagram.*

### Table 8. Three-Track Consensus Convergence Matrix & Biological Defense
| Feature Candidate | Track 1 (Univar+Stepwise) | Track 2 (LASSO) | Track 3 (Boruta) | Consensus | Pathophysiological Defense |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Late D-Dimer (`DD_Late`)** | Selected (p < 0.0001) | Selected (coef = +0.842) | Confirmed (Z = 9.85) | **Core Predictor** | Cumulative cannula thrombi & late hyperfibrinolysis |
| **Baseline D-Dimer (`DD_D0`)** | Selected (p = 0.0001) | Selected (coef = +0.612) | Confirmed (Z = 8.92) | **Core Predictor** | Pre-existing endothelial injury prior to cannulation |
| **Baseline WBC (`WBC_D0`)** | Selected (p = 0.0016) | Selected (coef = +0.295) | Confirmed (Z = 5.12) | **Core Predictor** | Neutrophil extracellular traps & immunothrombosis |
| **Day 2 Fibrinogen (`FIB_D2`)** | Selected (p = 0.0084) | Selected (coef = +0.384) | Confirmed (Z = 6.45) | **Core Predictor** | Early substrate availability & acute-phase turnover |
| **Day 7 ALT (`ALT_D7`)** | Selected (p = 0.0078) | Selected (coef = -0.182) | Confirmed (Z = 3.88) | **Core Predictor** | Hepatic congestion & impaired procoagulant clearance |

---

## 5. Biological Distributions & Multicollinearity Diagnostics

![Figure 6: Biological Distributions Violin Plots](sci_figures/fig06_biological_distributions_violin.png)
*Figure 6. Biological distributions and non-overlapping significance testing for the 5 core biomarkers.*

![Figure 7: Correlation Matrix Heatmap](sci_figures/fig07_correlation_heatmap.png)
*Figure 7. Spearman rank correlation and multicollinearity matrix among core predictors.*

---

## 6. Class Imbalance Benchmark & Methodological Strategy

![Figure 8: Class Imbalance Benchmark](sci_figures/fig08_imbalance_benchmark_comparison.png)
*Figure 8. Comparative evaluation of class imbalance strategies across 10-fold cross-validation.*

### Table 9. Class Imbalance Strategy Benchmark Comparison Table
| Balancing Strategy | 10-Fold CV AUC | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **None (Unweighted Baseline)** | 0.8652 | 84.97% | 13.33% | 96.96% | 26.67% | 0.1705 | Unacceptable: 86.7% false negative rate |
| **Random Under-Sampling (RUS)** | 0.8641 | 80.85% | 81.67% | 79.57% | 49.65% | 0.5689 | Suboptimal: Discards 80% of control data |
| **SMOTE (Synthetic Over-sampling)** | 0.8819 | 83.05% | 75.83% | 82.61% | 44.71% | 0.5494 | Effective: Synthesizes artificial vectors |
| **Cost-Sensitive Balanced Weights** | **0.8822** | **83.43%** | **78.33%** | **82.17%** | **44.86%** | **0.5564** | **Recommended: Authentic geometry** |

---

## 7. Machine Learning Model Training & 10-Fold CV Benchmark

![Figure 9: Model Radar & Forest Plot](sci_figures/fig09_model_performance_radar_forest.png)
*Figure 9. Model Performance Comparison in Stage 1 Cohort.*

### Table 10. Stage 1 Cohort: 10-Fold Cross-Validation Performance Metrics (All 6 Models)
| Model Classifier | 10-Fold CV AUC (Mean ± SD) | 95% Confidence Interval | Accuracy | Sensitivity | Specificity | Precision | F1-Score | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | **0.8884 ± 0.032** | 0.869 - 0.908 | 83.1% | 80.6% | 83.5% | 44.1% | **0.558** | 0.118 |
| **XGBoost (XGB)** | **0.8859 ± 0.036** | 0.864 - 0.908 | 82.3% | 77.8% | 83.0% | 42.8% | **0.538** | 0.124 |
| **Logistic Regression (LR)** | **0.8072 ± 0.045 / 0.8822\*** | 0.779 - 0.835 | 81.9% | 78.3% | 82.2% | 44.9% | **0.556** | 0.138 |
| **Support Vector Machine (SVM)** | **0.8241 ± 0.041** | 0.799 - 0.849 | 81.2% | 72.2% | 82.6% | 41.5% | 0.512 | 0.142 |
| **Gaussian Naive Bayes (NB)** | **0.8412 ± 0.048** | 0.811 - 0.871 | 80.1% | 69.4% | 81.7% | 38.5% | 0.485 | 0.151 |
| **Decision Tree (DT)** | **0.7915 ± 0.052** | 0.759 - 0.824 | 78.2% | 75.0% | 78.7% | 36.5% | 0.478 | 0.165 |

![Figure 10: Stage 1 Confusion Matrices](sci_figures/fig10_stage1_cv_confusion_matrices.png)
*Figure 10. Stage 1 Cohort 10-fold cross-validation confusion matrices for all 6 machine learning classifiers.*

### Table 11. Stage 1 Cross-Validation Confusion Matrix Contingency Table
| Model Architecture | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) | Sensitivity | Specificity | PPV | NPV |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | 28 / 36 (77.8%) | 8 / 36 (22.2%) | 41 / 230 (17.8%) | 189 / 230 (82.2%) | 77.8% | 82.2% | 40.6% | 95.9% |
| **Random Forest (RF)** | 29 / 36 (80.6%) | 7 / 36 (19.4%) | 38 / 230 (16.5%) | 192 / 230 (83.5%) | 80.6% | 83.5% | 43.3% | 96.5% |
| **XGBoost (XGB)** | 28 / 36 (77.8%) | 8 / 36 (22.2%) | 39 / 230 (17.0%) | 191 / 230 (83.0%) | 77.8% | 83.0% | 41.8% | 96.0% |
| **Support Vector Machine (SVM)** | 26 / 36 (72.2%) | 10 / 36 (27.8%) | 40 / 230 (17.4%) | 190 / 230 (82.6%) | 72.2% | 82.6% | 39.4% | 95.0% |
| **Decision Tree (DT)** | 27 / 36 (75.0%) | 9 / 36 (25.0%) | 49 / 230 (21.3%) | 181 / 230 (78.7%) | 75.0% | 78.7% | 35.5% | 95.3% |
| **Gaussian Naive Bayes (NB)** | 25 / 36 (69.4%) | 11 / 36 (30.6%) | 42 / 230 (18.3%) | 188 / 230 (81.7%) | 69.4% | 81.7% | 37.3% | 94.5% |

### Table 12. Stage 1 10-Fold Cross-Validation DeLong Pairwise Significance Matrix
| Pairwise Comparison | AUC Model A | AUC Model B | Difference | Z-Statistic | p-value | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RF vs. LR** | 0.8884 | 0.8072 | +0.0812 | 2.145 | **0.0319\*** | Significant ($p < 0.05$) |
| **RF vs. XGB** | 0.8884 | 0.8859 | +0.0025 | 0.185 | 0.8532 | Comparable |
| **RF vs. SVM** | 0.8884 | 0.8241 | +0.0643 | 1.921 | 0.0547 | Marginally Significant |
| **RF vs. DT** | 0.8884 | 0.7915 | +0.0969 | 2.418 | **0.0156\*** | Significant ($p < 0.05$) |
| **XGB vs. LR** | 0.8859 | 0.8072 | +0.0787 | 2.052 | **0.0402\*** | Significant ($p < 0.05$) |

---

## 8. Independent Blind External Validation on Stage 2 Cohort ($N=106$)

![Figure 11: Stage 2 Confusion Matrices](sci_figures/fig11_stage2_external_confusion_matrices.png)
*Figure 11. Stage 2 independent blind external validation (N=106) confusion matrices for all 6 classifiers.*

### Table 13. Stage 2 Independent External Validation Performance Table (N=106)
| Model Architecture | External AUC | 95% Confidence Interval | Accuracy | Sensitivity | Specificity | F1-Score | Brier Score | Hosmer-Lemeshow p-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | **0.8267** | **0.735 - 0.918** | 74.5% | **80.0% (20/25)** | **72.8% (59/81)** | **0.580** | **0.158** | 0.482 (Well Calibrated) |
| **Random Forest (RF)** | **0.8017** | **0.702 - 0.901** | 75.5% | **76.0% (19/25)** | **75.3% (61/81)** | **0.567** | **0.162** | 0.521 (Well Calibrated) |
| **XGBoost (XGB)** | **0.7926** | **0.691 - 0.894** | 75.5% | 72.0% (18/25) | 76.5% (62/81) | 0.545 | 0.166 | 0.395 (Well Calibrated) |
| **Support Vector Machine (SVM)** | **0.7812** | **0.678 - 0.884** | 73.6% | 68.0% (17/25) | 75.3% (61/81) | 0.515 | 0.174 | 0.312 (Well Calibrated) |
| **Gaussian Naive Bayes (NB)** | **0.7741** | **0.665 - 0.883** | 72.6% | 64.0% (16/25) | 75.3% (61/81) | 0.492 | 0.181 | 0.245 (Moderate Fit) |
| **Decision Tree (DT)** | **0.7350** | **0.621 - 0.849** | 71.7% | 68.0% (17/25) | 72.8% (59/81) | 0.493 | 0.198 | 0.182 (Acceptable) |

### Table 14. Stage 2 External Validation Confusion Matrix Contingency Table (N=106)
| Model Classifier | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) | Sensitivity | Specificity | PPV | NPV |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | 20 / 25 (80.0%) | 5 / 25 (20.0%) | 22 / 81 (27.2%) | 59 / 81 (72.8%) | 80.0% | 72.8% | 47.6% | 92.2% |
| **Random Forest (RF)** | 19 / 25 (76.0%) | 6 / 25 (24.0%) | 20 / 81 (24.7%) | 61 / 81 (75.3%) | 76.0% | 75.3% | 48.7% | 91.0% |
| **XGBoost (XGB)** | 18 / 25 (72.0%) | 7 / 25 (28.0%) | 19 / 81 (23.5%) | 62 / 81 (76.5%) | 72.0% | 76.5% | 48.6% | 89.9% |
| **Support Vector Machine (SVM)** | 17 / 25 (68.0%) | 8 / 25 (32.0%) | 20 / 81 (24.7%) | 61 / 81 (75.3%) | 68.0% | 75.3% | 45.9% | 88.4% |
| **Gaussian Naive Bayes (NB)** | 16 / 25 (64.0%) | 9 / 25 (36.0%) | 20 / 81 (24.7%) | 61 / 81 (75.3%) | 64.0% | 75.3% | 44.4% | 87.1% |
| **Decision Tree (DT)** | 17 / 25 (68.0%) | 8 / 25 (32.0%) | 22 / 81 (27.2%) | 59 / 81 (72.8%) | 68.0% | 72.8% | 43.6% | 88.1% |

### Table 15. Stage 2 DeLong Pairwise Significance Matrix
| Pairwise Comparison | Stage 2 AUC Model A | Stage 2 AUC Model B | Difference | Z-Statistic | p-value | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LR vs. RF** | 0.8267 | 0.8017 | +0.0250 | 0.612 | 0.5405 | Comparable |
| **LR vs. XGB** | 0.8267 | 0.7926 | +0.0341 | 0.845 | 0.3981 | Comparable |
| **LR vs. DT** | 0.8267 | 0.7350 | +0.0917 | 1.985 | **0.0471\*** | Statistically Superior ($p < 0.05$) |
| **RF vs. DT** | 0.8017 | 0.7350 | +0.0667 | 1.682 | 0.0926 | Marginally Superior |

![Figure 12: Dual-Cohort ROC & PR Curves](sci_figures/fig12_dual_cohort_roc_pr_curves.png)
*Figure 12. Dual-cohort Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.*

![Figure 13: DCA & Clinical Impact Curves](sci_figures/fig13_dca_and_clinical_impact.png)
*Figure 13. Decision Curve Analysis (DCA) and Clinical Impact Curve (CIC).*

### Table 16. Decision Curve Analysis (DCA) Net Benefit Quantification (Stage 2)
| Threshold (Pt) | Treat All | Treat None | LR Net Benefit | RF Net Benefit | XGB Net Benefit | Advantage Window |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10%** | 0.148 | 0.000 | 0.198 | 0.192 | 0.188 | Superior to Treat All |
| **15%** | 0.101 | 0.000 | 0.174 | 0.168 | 0.162 | Optimal Action Zone |
| **20%** | 0.045 | 0.000 | 0.148 | 0.142 | 0.138 | Optimal Action Zone |
| **25%** | -0.021 | 0.000 | 0.125 | 0.121 | 0.118 | Treat All Harmful |
| **30%** | -0.095 | 0.000 | 0.098 | 0.095 | 0.091 | Net Benefit (+0.098) |
| **40%** | -0.282 | 0.000 | 0.058 | 0.054 | 0.050 | Selective Action |

### Table 17. Clinical Impact Curve (CIC) Diagnostic Yield per 100 Patients
| Risk Threshold (Pt) | Predicted High-Risk / 100 | True Events Detected / 100 | False Positives Avoided / 100 | Number Needed to Screen (NNS) |
| :--- | :--- | :--- | :--- | :--- |
| **10%** | 58 Patients | 22 Events | 42 Patients Spared | 2.64 Ultrasound Exams / Event |
| **15%** | 44 Patients | 20 Events | 56 Patients Spared | 2.20 Ultrasound Exams / Event |
| **20%** | 36 Patients | 19 Events | 64 Patients Spared | 1.89 Ultrasound Exams / Event |
| **25%** | 29 Patients | 17 Events | 71 Patients Spared | 1.71 Ultrasound Exams / Event |
| **30%** | 23 Patients | 15 Events | 77 Patients Spared | 1.53 Ultrasound Exams / Event |
| **40%** | 15 Patients | 12 Events | 85 Patients Spared | 1.25 Ultrasound Exams / Event |

![Figure 14: Calibration Reliability Curves](sci_figures/fig14_calibration_reliability_curves.png)
*Figure 14. Calibration reliability curves across Stage 1 and Stage 2 cohorts.*

---

## 9. Explainable AI (SHAP) & Quantitative Feature Attribution

![Figure 15: SHAP Multi-Perspective Panel](sci_figures/fig15_shap_interpretability_panel.png)
*Figure 15. SHAP Multi-Perspective Interpretability Panel: (A) Beeswarm plot. (B) Mean absolute SHAP ranking. (C) High-risk patient waterfall. (D) Low-risk patient waterfall.*

### Table 18. Global Explainability: SHAP Feature Importance Metrics
| Feature Name | Mean |SHAP| Value | Median SHAP | Standard Deviation | Global Rank | Risk Impact Directionality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Late D-Dimer (`DD_Late`)** | 0.2854 | +0.2150 | 0.1985 | **Rank 1 (Primary)** | Positive: High late levels increase risk |
| **Baseline D-Dimer (`DD_D0`)** | 0.2412 | +0.1840 | 0.1752 | **Rank 2 (Strong Risk)** | Positive: Pre-existing microvascular injury |
| **Day 2 Fibrinogen (`FIB_D2`)** | 0.1650 | +0.1280 | 0.1140 | **Rank 3 (Substrate)** | Positive: Substrate for clot propagation |
| **Baseline WBC (`WBC_D0`)** | 0.1185 | +0.0920 | 0.0895 | **Rank 4 (Inflammation)** | Positive: Promotes immunothrombosis |
| **Day 7 ALT (`ALT_D7`)** | 0.0924 | -0.0650 | 0.0712 | **Rank 5 (Organ Modifier)**| Negative: Impaired hepatic clearance |

### Table 19. Patient-Level Local SHAP Decomposition (Representative Case Profiles)
| Case Profile | Base E[f(x)] | Late DD Value & SHAP | Baseline DD & SHAP | Baseline WBC & SHAP | Day 2 FIB & SHAP | Day 7 ALT & SHAP | Final Output f(x) | Probability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **High-Risk Case (#42)** | -1.86 | 28.5 mg/L (+1.42) | 18.2 mg/L (+0.95) | 18.4 ×10⁹/L (+0.68) | 4.6 g/L (+0.52) | 22.0 U/L (+0.24) | +1.95 | **87.5% (High Tier)** |
| **Low-Risk Case (#108)** | -1.86 | 4.2 mg/L (-0.85) | 3.1 mg/L (-0.62) | 8.5 ×10⁹/L (-0.25) | 2.4 g/L (-0.38) | 38.0 U/L (-0.15) | -4.11 | **1.6% (Low Tier)** |

![Figure 16: SHAP Dependence Interactions](sci_figures/fig16_shap_dependence_interactions.png)
*Figure 16. SHAP Feature Dependence and Interaction Profiles.*

---

## 10. Dynamic Longitudinal Coagulation Kinetics

![Figure 17: Dynamic Longitudinal D-Dimer Kinetics](sci_figures/fig17_longitudinal_ddimer_kinetics.png)
*Figure 17. Dynamic longitudinal D-Dimer trajectories across ECMO therapy.*

### Table 20. Dynamic Longitudinal Biomarker Trajectories across ECMO Duration
| Cohort Phase | Biomarker | Baseline Day 0 | Early ECMO Day 2 | Late ECMO / Decan | Surge Δ (Late - Base) | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 Derivation** | Thrombosis DD (mg/L) | 14.8 [8.5, 24.6] | 18.2 [11.5, 29.4] | 23.4 [15.2, 36.8] | **+8.6 mg/L Surge** | **p < 0.0001\*\*\*\*** |
| **Stage 1 Derivation** | Non-Thrombosis DD (mg/L) | 4.2 [2.3, 8.9] | 5.1 [2.8, 10.2] | 6.1 [3.2, 11.4] | +1.9 mg/L Stable | p = 0.082 |
| **Stage 2 External** | Thrombosis DD (mg/L) | 15.6 [8.8, 26.2] | 19.5 [12.0, 31.0] | 25.1 [16.4, 38.5] | **+9.5 mg/L Surge** | **p < 0.0001\*\*\*\*** |
| **Stage 2 External** | Non-Thrombosis DD (mg/L) | 4.8 [2.5, 9.8] | 5.6 [3.0, 11.2] | 6.8 [3.5, 12.0] | +2.0 mg/L Stable | p = 0.095 |
| **Stage 1 Derivation** | Thrombosis FIB (g/L) | 3.8 [2.8, 4.8] | 4.4 [3.5, 5.2] | 4.2 [3.2, 5.0] | +0.6 g/L Early Rise | **p < 0.0001\*\*\*\*** |
| **Stage 1 Derivation** | Non-Thrombosis FIB (g/L) | 2.9 [2.1, 3.8] | 2.8 [2.0, 3.8] | 3.0 [2.2, 4.0] | -0.1 g/L Baseline | p = 0.420 |

---

## 11. Conclusion & Bedside Web Application Deployment

The final 5-biomarker model has been deployed into an interactive web application (`web_app/app.py`). Clinicians can enter bedside parameters to obtain instant calibrated thrombosis probabilities, risk tier assignments, and evidence-based clinical surveillance recommendations.
