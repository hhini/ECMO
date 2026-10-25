# -*- coding: utf-8 -*-
import os
import json
import joblib
import pandas as pd

ASSET_DIR = r"D:\acode\ECMO\web_app\model_assets"
meta_path = os.path.join(ASSET_DIR, "model_metadata.json")

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

feats = meta["features"]

# 1. selected_features.json
with open(os.path.join(ASSET_DIR, "selected_features.json"), "w", encoding="utf-8") as f:
    json.dump({"final_features": feats}, f, indent=4)

# 2. preprocess_config.json
scaler = joblib.load(os.path.join(ASSET_DIR, "scaler.pkl"))
prep_config = {
    "features": feats,
    "scaler_mean": {f: float(m) for f, m in zip(feats, scaler.mean_)},
    "scaler_scale": {f: float(s) for f, s in zip(feats, scaler.scale_)}
}
with open(os.path.join(ASSET_DIR, "preprocess_config.json"), "w", encoding="utf-8") as f:
    json.dump(prep_config, f, indent=4)

# 3. Create joblib aliases (model_RF.joblib, model_LR.joblib, etc.)
for code in ['rf', 'lr', 'xgb', 'svm', 'dt', 'nb']:
    src = os.path.join(ASSET_DIR, f"{code}_model.pkl")
    dst = os.path.join(ASSET_DIR, f"model_{code.upper()}.joblib")
    if os.path.exists(src):
        joblib.dump(joblib.load(src), dst)
        print(f"Saved {dst}")

# 4. 03_model_performance_metrics.csv
records = []
for name in meta['stage1_10fold_cv_auc']:
    code = name.split(' (')[1].replace(')', '')
    records.append({
        'Model': code,
        'AUROC': f"{meta['stage1_10fold_cv_auc'][name]:.4f}",
        'Test_AUC': f"{meta['stage2_external_auc'][name]:.4f}",
        'External_AUC': f"{meta['stage2_external_auc'][name]:.4f}"
    })
pd.DataFrame(records).to_csv(os.path.join(ASSET_DIR, "03_model_performance_metrics.csv"), index=False)
print("Updated web app assets compatibility files.")
