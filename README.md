# ECMO Thrombosis Risk AI — Clinical Decision Support Web Application

## 📁 目录结构 (Directory Structure)

```text
web_app/
├── app.py                     # Streamlit 主程序 (包含高对比度 UI、双控件输入、Plotly 动态 SHAP 归因)
├── model_assets/              # 最佳预测模型与归因配置文件
│   ├── model_RF.joblib        # 随机森林最佳拟合模型 (AUC = 0.810, Sens = 81.82%)
│   ├── model_LR.joblib        # Logistic 回归对比模型
│   ├── model_SVM.joblib       # 支持向量机对比模型
│   ├── preprocess_config.json # Z-score 归一化均值与标准差参数 (Scaler Stats)
│   ├── selected_features.json # Boruta/LASSO 筛选出的核心临床因子
│   └── 03_model_performance_metrics.csv # 模型交叉验证与测试集性能矩阵
└── README.md                  # 应用部署说明文档
```

## 🚀 启动方式 (How to Run)

在命令行中直接运行以下命令启动服务：

```bash
python -m streamlit run D:\acode\ECMO\web_app\app.py
```

应用将在本地浏览器打开：`http://localhost:8501`
