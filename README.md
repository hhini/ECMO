# 成人体外膜肺氧合（ECMO）下肢血栓智能风险预警系统 (ECMO-LET-AI)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **基于严格九步法科研架构：数据分层隔离划分（186/80/106）、Day-15高缺失剔除、训练集距离加权KNN插补、VIF共线性筛除、重复5折LASSO精简筛选、代价敏感学习、Optuna TPE 贝叶斯超参数优化与SHAP可解释性分析。**

---

## 🌟 研究亮点与核心结论 (Overview & Highlights)

遵循国际医学顶刊 **TRIPOD+AI** 规范与导师指导的严谨统计学标准，本项目构建了面向成人 ECMO 下肢血栓形成（Lower Extremity Thrombosis, LET）的智能风险预警系统：

* **核心研究队列架构**:
  * **推导队列 ($N=266$)**: 严格按分层抽样划分为 **训练集 ($n=186$, 血栓 31 例)** 与 **内部验证集 ($n=80$, 血栓 14 例)**。
  * **独立外部验证集 ($N=106$, 血栓 25 例)**: 全程绝对冻结、盲态测试，零数据泄露。
* **数据质控与单边插补**:
  * 剔除危重症撤机转归导致的 Day-15 结构性高缺失指标（缺失率均 $>50\%$）。
  * 采用标准化距离加权 **KNN 插补 ($k=5$, weights='distance')**，参数严格仅在 186 例训练集上拟合并单向映射至内部与外部验证集。
* **VIF 共线性筛除与 LASSO 核心预测因子**:
  * 在训练集上实施方差膨胀因子检验，剔除 VIF $> 10.0$ 的强共线特征（$4$ 个），保留 $59$ 个独立候选特征。
  * 经 $10$ 次重复 $5$ 折交叉验证 LASSO 精简筛选出 **$6$ 个独立核心预测因子**：
    1. `DD_D0`: ECMO 上机前基线 D-二聚体 ($\mu$g/mL, $\log(1+x)$ 转换)
    2. `FIB_D2`: ECMO 第 2 天纤维蛋白原 (g/L)
    3. `ECMO_Duration`: ECMO 辅助支持总时长 (hours)
    4. `CRP_D0`: ECMO 上机前基线 C 反应蛋白 (mg/L)
    5. `PaO2_D0`: ECMO 上机前动脉氧分压 (mmHg)
    6. `Pre_ECMO_Surgery`: ECMO 上机前急诊手术史 ($0$=无, $1$=有)
* **贝叶斯超参数优化与金标准模型 (SVM Champion)**:
  * 严格在训练集上采用 **Optuna TPE 贝叶斯优化** 锁定各算法最优超参数。
  * **支持向量机 (SVM, RBF核, $C=5.658, \gamma=0.05223$)** 综合跨队列泛化能力位列榜首：
    * **训练集 10×5折 CV AUROC**: **0.918** (95% CI: 0.870 – 0.958) | Brier: 0.087
    * **内部验证集 ($n=80$) AUROC**: **0.884** (95% CI: 0.798 – 0.957) | 敏感度: 78.6% | 特异度: 84.8% | Brier: 0.104
    * **独立外部验证集 ($n=106$) AUROC**: **0.771** (95% CI: 0.651 – 0.867) | 特异度: 79.0% | Brier: 0.166
    * **双验证集平均 AUROC**: **0.828**（显著优于外部验证泛化衰减明显的树模型，DeLong $P < 0.05$）。
* **14 幅 300 DPI 顶刊级科研全景图谱**: 涵盖技术路线流程图、缺失值审计、VIF诊断、LASSO路径、贝叶斯收敛轨迹、三队列ROC/PR曲线、混淆矩阵、森林图、校准曲线、DCA净获益及全局/个体SHAP瀑布图。

---

## 🚀 本地快速启动 (Quick Start)

### 1. 环境安装
```bash
git clone https://github.com/hhini/ECMO.git
cd ECMO
pip install -r requirements.txt
```

### 2. 启动交互式床旁临床决策平台
```bash
streamlit run app.py
```
浏览器将自动打开 `http://localhost:8501`，即可使用：
- **Tab 1: 临床个体化风险评估器**（输入 6 项指标即时获得血栓概率、分层预警及床旁干预建议）
- **Tab 2: 批量患者筛查与预测**（支持上传 CSV 或一键加载 25 例外部验证集数据）
- **Tab 3: 机器学习算法对决看板**（6 种算法三队列全指标对比与 DeLong 检验）
- **Tab 4: 14 幅科研高清全景图谱画廊**（300 DPI 顶刊图谱）
- **Tab 5: 九步法全流程研究白皮书**（严谨科研规范与模型卡片）

---

## 📁 代码库整洁目录结构 (Repository Structure)

```text
ECMO/
├── app.py                          # 根目录启动入口（无缝重定向至 web_app/app.py）
├── requirements.txt                # 运行依赖规范
├── README.md                       # 项目总体说明与研究亮点文档
├── PROJECT.md                      # 九步法科研架构与技术规范
│
├── web_app/                        # 交互式生产级临床 Web 应用
│   ├── app.py                      # Streamlit 5 大功能模块主程序
│   ├── sample_ecmo_cohort.csv      # 25 例外部验证集样本（含 6 核心特征与真实结局）
│   ├── model_assets/               # 6 种预训练 Pipeline 模型及特征元数据
│   │   ├── model_SVM.joblib        # 最优模型 (SVM RBF 核, 贝叶斯调优)
│   │   ├── model_LR.joblib         # 逻辑回归模型
│   │   ├── model_RF.joblib         # 随机森林模型
│   │   ├── model_XGB.joblib        # XGBoost 模型
│   │   ├── model_LGBM.joblib       # LightGBM 模型
│   │   ├── model_NB.joblib         # 高斯朴素贝叶斯模型
│   │   ├── selected_features.json  # 6 个核心特征定义
│   │   └── shap_background.npy     # 训练集 SHAP 经验背景分布
│   └── static/sci_figures/         # 14 幅 300 DPI 出版级科研图表
│
├── report/                         # 完整学术报告与图表成果
│   ├── ECMO_Thrombosis_Full_Research_Report.docx # 最新版 4.3 MB 完整学术报告 Word 文档
│   ├── ECMO_Thrombosis_Full_Research_Report.md   # 最新版完整学术报告 Markdown 文档
│   └── sci_figures/                # 14 幅 300 DPI 出版级科研高清图 (fig01 ~ fig13, fig05b)
│
├── scripts/                        # 核心复现科研脚本（精炼无冗余）
│   ├── run_revised_pipeline.py     # 修订版九步法全流程主脚本（KNN插补/VIF/LASSO/贝叶斯/SHAP）
│   ├── generate_comprehensive_tables.py # 基线特征对比三线表（Table 1 & 2，Z/χ²合并）
│   └── build_chinese_research_report.py # 自动生成最新版 Word & Markdown 学术报告
│
├── data/                           # 核心建模推导数据 (N=266)
│   ├── data2_synthetic.csv         # 推导队列数据
│   └── ...
│
├── testdata/                       # 独立外部验证原始数据
│   └── 第二阶段原始数据.xlsx        # 外部验证原始 Excel
│
└── output/
    ├── 01_stage2_clean/            # 外部验证集清洗输出 (stage2_clean.csv, n=106)
    ├── 02_table1_baseline/         # Table 1 & Table 2 基线表 (含合并后 Z / χ² 统计量)
    └── revised_pipeline/           # 九步法运行成果 (缺失审计/VIF/LASSO/贝叶斯参数/三队列性能/DeLong检验)
```

---

## 🔬 六大核心临床预测因子病理生理学依据

| 核心特征 | 临床名称 | 单位 | 效应方向 | 病理生理学现实意义 |
| :--- | :--- | :--- | :--- | :--- |
| `DD_D0` | 基线 D-二聚体 | $\mu$g/mL | 正向促栓 | 反映 ECMO 启动前机体已存在的缺血缺氧、凝血微血栓形成与纤溶预激活状态。 |
| `FIB_D2` | 第 2 天纤维蛋白原 | g/L | 正向促栓 | 早期稳态凝血网架底物丰富度。高浓度纤维蛋白原增强血栓的机械剪切阻力与稳固性。 |
| `ECMO_Duration` | ECMO 支持总时长 | hours | 正向促栓 | 体外回路异物接触与粗大置管对股静脉管壁机械压迫的累积暴露时长。 |
| `CRP_D0` | 基线 C 反应蛋白 | mg/L | 正向促栓 | 急性期全身炎症风暴标志物，诱导内皮细胞促凝表型转化与血小板超活化。 |
| `PaO2_D0` | 基线动脉氧分压 | mmHg | 负向保护 | 反映上机前极重度缺氧程度。组织低氧与静脉淤滞协同促发血栓形成。 |
| `Pre_ECMO_Surgery`| ECMO 前手术史 | 0/1 | 正向促栓 | 手术创伤激活外源性凝血途径与组织因子释放，叠加 ECMO 回路致高凝状态。 |

---

## 📜 许可证 (License)

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源许可。
