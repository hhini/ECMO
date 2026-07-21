import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import json

with open("D:/acode/ECMO/output/01_preprocessing/preprocess_config.json", encoding="utf-8") as f:
    config = json.load(f)

from sklearn.model_selection import train_test_split
df = pd.read_csv("D:/acode/ECMO/data/data2.csv")

# Clean
for col in ["手术时长/h", "手术失血量/ml", "镇静药物使用时长/d", "呼吸机使用时长/d", "ICU住院时长/d", "ECMO时长"]:
    import re
    def parse_numeric(val):
        if pd.isnull(val): return 0.0
        val_str = str(val).strip()
        if val_str in ["无", "未手术", "否"]: return 0.0
        match = re.search(r"([0-9\.]+)", val_str)
        if match: return float(match.group(1))
        return 0.0
    df[col] = df[col].apply(parse_numeric)

X = df.drop(columns=["编号", "住院号", "入院时间", "血栓事件", "结局", "是否下肢血栓", "首要诊断", "次要诊断", "基础疾病", "既往手术史", "ECMO前手术史"])
X = X.drop(columns=config["dropped_missing_features"])
y = df["是否下肢血栓"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Impute and Scale
X_train = X_train.fillna(X_train.mean(numeric_only=True))
num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)

print(f"Original shape: {X_train_scaled.shape}")

for k in [3, 5, 7, 9, 11, 15, 20]:
    enn = EditedNearestNeighbours(n_neighbors=k, sampling_strategy='all')
    smote_enn = SMOTEENN(smote=SMOTE(random_state=42), enn=enn, random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_train_scaled, y_train)
    print(f"ENN k={k}: Class 0={sum(y_res==0)}, Class 1={sum(y_res==1)}")
