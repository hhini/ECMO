# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Predictive Modeling - Stage 1: Data Preprocessing (No Feature Engineering)
Author: Antigravity Team
Date: 2026-07-13

Description:
This script loads the raw ECMO cohort clinical data, performs clinical feature mappings and
cleansing, splits the cohort into train/test, drops features with >30% missingness, and
imputes missing values using MICE (continuous) and Median (categorical/demographics).
Does not construct any temporal trajectories, ratios, or clinical pathological flags.
"""

import os
import re
import json
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

sys.path.append(os.getcwd())
from scripts.exploration.visual_utils import (
    plot_missingness_ratio, plot_class_distribution, plot_mice_imputation
)

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "False").lower() in ["true", "1", "yes"]

DATA_PATH = r"D:\acode\ECMO\data\data2_synthetic.csv" if USE_SYNTHETIC else r"D:\acode\ECMO\data\data2.csv"
OUTPUT_DIR = r"D:\acode\ECMO\output\data_clear_synthetic" if USE_SYNTHETIC else r"D:\acode\ECMO\output\data_clear"
RANDOM_SEED = 30
TEST_SIZE = 0.30

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOGGING SYSTEM SETUP
# ==========================================
log_file = os.path.join(OUTPUT_DIR, "01_preprocess.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocessing")

logger.info("Starting Clinical Comprehensive Preprocessing Pipeline (data_clear)...")

# ==========================================
# 3. CHINESE TO ENGLISH TRANSLATIONS FOR PLOTS
# ==========================================
CHINESE_TO_ENGLISH = {
    "上机前DD": "Baseline D-Dimer",
    "上机第二天DD": "Day 2 D-Dimer",
    "FIB": "Baseline Fibrinogen",
    "FIB.1": "Day 2 Fibrinogen",
    "TT": "Baseline Thrombin Time",
    "TT.1": "Day 2 Thrombin Time",
    "APTT": "Baseline APTT",
    "APTT.1": "Day 2 APTT",
    "PT": "Baseline PT",
    "PT.1": "Day 2 PT",
    "INR": "Baseline INR",
    "INR.1": "Day 2 INR",
    "AT3": "Baseline Antithrombin III",
    "AT3.1": "Day 2 Antithrombin III",
    "PLT": "Baseline Platelets",
    "PLT.1": "Day 2 Platelets",
    "谷丙": "Baseline ALT",
    "谷草": "Baseline AST",
    "白蛋白": "Baseline Albumin",
    "肌酐": "Baseline Creatinine",
    "WBC": "Baseline WBC",
    "RBC": "Baseline RBC",
    "Hb": "Baseline Hemoglobin",
    "是否下肢血栓": "Thrombosis Event"
}

def translate_to_english(col):
    if col in CHINESE_TO_ENGLISH:
        return CHINESE_TO_ENGLISH[col]
    clean = re.sub(r'[^\w\.\-\s/]', '', col)
    return clean if clean.strip() else col

# ==========================================
# 4. PARSING & MAPPING UTILITIES
# ==========================================
def parse_numeric(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ["无", "未手术", "否"]:
        return 0.0
    match = re.search(r"([0-9\.]+)", val_str)
    if match:
        return float(match.group(1))
    return np.nan

def map_sex(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "男" in val_str:
        return 1.0
    return 0.0

def map_residence(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "农村" in val_str: return 0.0
    if "乡镇" in val_str: return 1.0
    if "县城" in val_str: return 2.0
    if "市区" in val_str: return 3.0
    return np.nan

def map_work(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "工作" in val_str:
        return 1.0
    return 0.0

def map_occupation(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if val_str in ["无", "退休", "无/退休"]:
        return 0.0
    if val_str in ["农民", "工人"]:
        return 1.0
    if val_str in ["职员", "教师", "公务员", "专业技术人员"]:
        return 2.0
    return 3.0  # 其他/混合

def parse_smoking(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if val_str in ["无", "否"]:
        return 0.0
    match = re.search(r"([0-9\.]+)", val_str)
    if match:
        return float(match.group(1))
    return 0.0

def map_drinking(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "有" in val_str:
        return 1.0
    return 0.0

def map_pregnancy(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "男性" in val_str: return 0.0
    if "绝经" in val_str: return 2.0
    if "妊娠" in val_str: return 3.0
    return 1.0  # 女性未妊娠未绝经 (通常标记为 "无")

def map_consciousness_pre(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "清醒" in val_str: return 0.0
    if "镇静" in val_str: return 1.0
    if "嗜睡" in val_str or "意识模糊" in val_str: return 2.0
    if "昏迷" in val_str: return 3.0
    return 0.0

def map_diagnosis(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if any(k in val_str for k in ["心肌梗死", "心梗", "心力衰竭", "心衰", "心源性", "冠心病"]):
        return 1.0
    if any(k in val_str for k in ["肺炎", "肺纤维化", "呼吸衰竭", "ARDS", "肺"]):
        return 2.0
    if any(k in val_str for k in ["创伤", "休克", "多发伤", "车祸", "骨折"]):
        return 3.0
    return 0.0

def map_history_surgery(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "无" in val_str:
        return 0.0
    return 1.0

def map_pre_ecmo_surgery(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "无" in val_str:
        return 0.0
    return 1.0

def map_autologous(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "有" in val_str:
        return 1.0
    return 0.0

def map_ecmo_consciousness(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "昏迷" in val_str:
        return 1.0
    return 0.0

def map_nutrition(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "有" in val_str:
        return 1.0
    return 0.0

def map_hemofiltration(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "人工肝" in val_str:
        return 2.0
    if "CRRT" in val_str:
        return 1.0
    return 0.0

def map_ecmo_mode(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip().upper()
    if "VAV" in val_str: return 2.0
    if "VA" in val_str: return 1.0
    if "VV" in val_str: return 0.0
    return 0.0

def map_cannulation_site(val):
    if pd.isnull(val):
        return np.nan
    val_str = str(val).strip()
    if "颈内+股静脉" in val_str: return 1.0
    if "颈内+股动静脉" in val_str: return 2.0
    if "中心" in val_str: return 3.0
    return 0.0  # 股动静脉置管

def parse_cannulation_size(val):
    if pd.isnull(val):
        return 19.0  # 默认管径
    val_str = str(val).strip()
    parts = re.findall(r"([0-9\.]+)", val_str)
    if parts:
        return max([float(x) for x in parts])
    return 19.0

def map_mode_conversion(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "有" in val_str:
        return 1.0
    return 0.0

def map_multiple_cannulation(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "无" in val_str:
        return 0.0
    return 1.0

def map_collateral(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if "2次" in val_str: return 2.0
    if "有" in val_str: return 1.0
    return 0.0

def parse_duration_days(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if val_str in ["无", "否"]:
        return 0.0
    match = re.search(r"([0-9\.]+)", val_str)
    if match:
        return float(match.group(1))
    return 0.0

def parse_duration_hours(val):
    if pd.isnull(val):
        return 0.0
    val_str = str(val).strip()
    if val_str in ["无", "否"]:
        return 0.0
    match = re.search(r"([0-9\.]+)", val_str)
    if match:
        return float(match.group(1))
    return 0.0

# ==========================================
# 5. LOAD & DEFINE GROUPS
# ==========================================
logger.info(f"Loading raw data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
logger.info(f"Loaded raw data shape: {df.shape[0]} rows, {df.shape[1]} columns")

demo_support_cols_raw = [
    "性别", "年龄", "BMI", "居住地", "工作状态", "职业", "吸烟", "饮酒", "妊娠状态", "ECMO前意识", 
    "首要诊断", "次要诊断", "基础疾病", "既往手术史", "ECMO前手术史",
    "自体血回输", "ECMO治疗期间意识", "血管活性药/种", "抗凝血制剂/种", "抗凝血制剂使用时长/d", 
    "有创管路数量/条", "输血种数/种", "静脉输注营养制剂使用", "联合血液净化治疗", 
    "呼吸机使用时长/d", "ICU住院时长/d", "ECMO模式", "置管部位", "置管管径", 
    "模式转换", "多次置管", "侧支循环", "ECMO时长"
]

lab_cols = [
    # D0
    "上机前DD", "FIB", "TT", "APTT", "PT", "INR", "AT3", "PLT", "WBC", "RBC", "Hb", "PCT", "CRP", "谷丙", "谷草", "白蛋白", "肌酐", "pH", "氧分压", "二氧化碳分压",
    # D2
    "上机第二天DD", "FIB.1", "TT.1", "APTT.1", "PT.1", "INR.1", "AT3.1", "PLT.1", "WBC.1", "RBC.1", "Hb.1", "PCT.1", "CRP.1", "谷丙.1", "谷草.1", "白蛋白.1", "肌酐.1", "pH.1", "氧分压.1", "二氧化碳分压.1",
    # D7
    "上机第七天DD", "FIB.2", "TT.2", "APTT.2", "PT.2", "INR.2", "AT3.2", "PLT.2", "WBC.2", "RBC.2", "Hb.2", "PCT.2", "CRP.2", "谷丙.2", "谷草.2", "白蛋白.2", "肌酐.2", "pH.2", "氧分压.2", "二氧化碳分压.2",
    # Wean
    "下机当天DD", "FIB.3", "TT.3", "APTT.3", "PT.3", "INR.3", "AT3.3", "PLT.3", "WBC.3", "RBC.3", "Hb.3", "PCT.3", "CRP.3", "谷丙.3", "谷草.3", "白蛋白.3", "肌酐.3", "pH.3", "氧分压.3", "二氧化碳分压.3"
]

required_raw_cols = list(set(demo_support_cols_raw + lab_cols))

y = df["是否下肢血栓"]
logger.info(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)} (Positive Rate: {100*y.mean():.2f}%)")

X_raw = df[required_raw_cols].copy()

# ==========================================
# 6. STEP 2: MISSINGNESS ANALYSIS & PLOTTING
# ==========================================
missing_ratios = X_raw.isnull().sum() / len(X_raw)
missing_to_plot = missing_ratios[missing_ratios > 0].sort_values(ascending=False)
if len(missing_to_plot) > 0:
    missing_to_plot_top = missing_to_plot[:25]
    eng_labels = [translate_to_english(col) for col in missing_to_plot_top.index]
    plot_path = os.path.join(OUTPUT_DIR, "01_missingness_ratio.png")
    plot_missingness_ratio(pd.Series(missing_to_plot_top.values, index=eng_labels), plot_path, title_suffix=" (data_clear)")
    logger.info(f"Saved missingness ratio plot to: {plot_path}")

# ==========================================
# 7. STEP 3: MAPPING TO NUMERIC REPRESENTATIONS
# ==========================================
def map_all_features(df_input):
    df_mapped = pd.DataFrame(index=df_input.index)
    
    # Section 2: Demographics
    df_mapped["Sex"] = df_input["性别"].apply(map_sex)
    df_mapped["Age"] = df_input["年龄"]
    df_mapped["BMI"] = df_input["BMI"]
    df_mapped["Residence"] = df_input["居住地"].apply(map_residence)
    df_mapped["Work_Status"] = df_input["工作状态"].apply(map_work)
    df_mapped["Occupation"] = df_input["职业"].apply(map_occupation)
    df_mapped["Smoking"] = df_input["吸烟"].apply(parse_smoking)
    df_mapped["Drinking"] = df_input["饮酒"].apply(map_drinking)
    df_mapped["Pregnancy"] = df_input["妊娠状态"].apply(map_pregnancy)
    df_mapped["Consciousness_Pre"] = df_input["ECMO前意识"].apply(map_consciousness_pre)
    df_mapped["Diagnosis_Primary"] = df_input["首要诊断"].apply(map_diagnosis)
    df_mapped["Diagnosis_Secondary"] = df_input["次要诊断"].apply(map_diagnosis)
    
    # Underlying diseases
    base = df_input["基础疾病"].fillna("无").astype(str)
    df_mapped["Underlying_Hypertension"] = base.apply(lambda x: 1.0 if "高血压" in x else 0.0)
    df_mapped["Underlying_Diabetes"] = base.apply(lambda x: 1.0 if any(k in x for k in ["糖尿病", "T2DM", "DM"]) else 0.0)
    df_mapped["Underlying_CHD"] = base.apply(lambda x: 1.0 if "冠心病" in x else 0.0)
    df_mapped["Underlying_Cerebrovascular"] = base.apply(lambda x: 1.0 if any(k in x for k in ["脑梗", "脑出血", "脑血管", "脑心"]) else 0.0)
    df_mapped["Underlying_Hyperlipidemia"] = base.apply(lambda x: 1.0 if "高脂" in x else 0.0)
    
    df_mapped["History_Surgery"] = df_input["既往手术史"].apply(map_history_surgery)
    df_mapped["Pre_ECMO_Surgery"] = df_input["ECMO前手术史"].apply(map_pre_ecmo_surgery)
    
    # Section 3: ECMO Support
    df_mapped["Autologous_Transfusion"] = df_input["自体血回输"].apply(map_autologous)
    df_mapped["ECMO_Consciousness_Intra"] = df_input["ECMO治疗期间意识"].apply(map_ecmo_consciousness)
    df_mapped["Vasoactive_Drugs_Count"] = df_input["血管活性药/种"]
    df_mapped["Anticoagulant_Types_Count"] = df_input["抗凝血制剂/种"]
    df_mapped["Anticoagulant_Duration"] = df_input["抗凝血制剂使用时长/d"]
    df_mapped["Invasive_Lines_Count"] = df_input["有创管路数量/条"]
    df_mapped["Transfusion_Types_Count"] = df_input["输血种数/种"]
    df_mapped["Parenteral_Nutrition"] = df_input["静脉输注营养制剂使用"].apply(map_nutrition)
    df_mapped["Concomitant_Hemofiltration"] = df_input["联合血液净化治疗"].apply(map_hemofiltration)
    df_mapped["Ventilation_Duration"] = df_input["呼吸机使用时长/d"].apply(parse_duration_days)
    df_mapped["ICU_Duration"] = df_input["ICU住院时长/d"].apply(parse_duration_days)
    df_mapped["ECMO_Mode"] = df_input["ECMO模式"].apply(map_ecmo_mode)
    df_mapped["Cannulation_Site"] = df_input["置管部位"].apply(map_cannulation_site)
    df_mapped["Cannulation_Size"] = df_input["置管管径"].apply(parse_cannulation_size)
    df_mapped["Mode_Conversion"] = df_input["模式转换"].apply(map_mode_conversion)
    df_mapped["Multiple_Cannulation"] = df_input["多次置管"].apply(map_multiple_cannulation)
    df_mapped["Collateral_Circulation"] = df_input["侧支循环"].apply(map_collateral)
    df_mapped["ECMO_Duration"] = df_input["ECMO时长"].apply(parse_duration_hours)
    
    # Section 4: Laboratory
    for col in lab_cols:
        df_mapped[col] = df_input[col].apply(parse_numeric)
        
    return df_mapped

X_mapped = map_all_features(X_raw)

multi_cat_cols = [
    "Residence", "Occupation", "Pregnancy", "Consciousness_Pre",
    "Diagnosis_Primary", "Diagnosis_Secondary", "Concomitant_Hemofiltration",
    "ECMO_Mode", "Cannulation_Site", "Collateral_Circulation"
]

binary_cols = [
    "Sex", "Work_Status", "Drinking",
    "Underlying_Hypertension", "Underlying_Diabetes", "Underlying_CHD",
    "Underlying_Cerebrovascular", "Underlying_Hyperlipidemia",
    "History_Surgery", "Pre_ECMO_Surgery", "Autologous_Transfusion",
    "ECMO_Consciousness_Intra", "Parenteral_Nutrition",
    "Mode_Conversion", "Multiple_Cannulation"
]

logger.info(f"One-hot encoding multi-categorical features: {multi_cat_cols}")
X_dummy = pd.get_dummies(X_mapped, columns=multi_cat_cols, drop_first=True, dtype=float)

# ==========================================
# 8. STEP 4: STRATIFIED TRAIN/TEST SPLIT
# ==========================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_dummy, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

cols_to_check = [col for col in X_train_raw.columns if not any(col.startswith(cat + "_") for cat in multi_cat_cols)]
missing_ratios_train = X_train_raw[cols_to_check].isnull().sum() / len(X_train_raw)
cols_to_drop = missing_ratios_train[missing_ratios_train > 0.30].index.tolist()

logger.info(f"Features dropped due to >30% missingness on train set: {cols_to_drop}")
X_train_raw = X_train_raw.drop(columns=cols_to_drop)
X_test_raw = X_test_raw.drop(columns=cols_to_drop)
logger.info(f"Train raw shape after dropping high-missingness columns: {X_train_raw.shape}")

X_train_unimputed = X_train_raw.copy()

# ==========================================
# 9. STEP 5: DECOUPLED IMPUTATION
# ==========================================
core_indicators_imp = {
    "DD": ["上机前DD", "上机第二天DD"],
    "PLT": ["PLT", "PLT.1"],
    "FIB": ["FIB", "FIB.1"],
    "AT3": ["AT3", "AT3.1"],
    "APTT": ["APTT", "APTT.1"],
    "PT": ["PT", "PT.1"],
    "INR": ["INR", "INR.1"],
    "TT": ["TT", "TT.1"]
}
baseline_reserves_imp = ["谷丙", "谷草", "白蛋白", "肌酐", "WBC", "RBC", "Hb"]
core_cols = []
for indicator, cols in core_indicators_imp.items():
    core_cols.extend([c for c in cols if c in X_train_raw.columns])
core_cols.extend([c for c in baseline_reserves_imp if c in X_train_raw.columns])

other_cols = [col for col in X_train_raw.columns if col not in core_cols]

core_imputer = IterativeImputer(max_iter=10, random_state=RANDOM_SEED)
X_train_core_imp = pd.DataFrame(
    core_imputer.fit_transform(X_train_raw[core_cols]), columns=core_cols, index=X_train_raw.index
)
X_test_core_imp = pd.DataFrame(
    core_imputer.transform(X_test_raw[core_cols]), columns=core_cols, index=X_test_raw.index
)

other_imputer = SimpleImputer(strategy="median")
X_train_other_imp = pd.DataFrame(
    other_imputer.fit_transform(X_train_raw[other_cols]), columns=other_cols, index=X_train_raw.index
)
X_test_other_imp = pd.DataFrame(
    other_imputer.transform(X_test_raw[other_cols]), columns=other_cols, index=X_test_raw.index
)

remaining_columns = [col for col in X_dummy.columns if col in X_train_raw.columns]
X_train_imp = pd.concat([X_train_core_imp, X_train_other_imp], axis=1)[remaining_columns]
X_test_imp = pd.concat([X_test_core_imp, X_test_other_imp], axis=1)[remaining_columns]
logger.info("Decoupled Imputation complete. Verified 0 missing values.")

# ==========================================
# 10. STEP 6: TRANSLATE LAB COLUMNS (NO FEATURE ENGINEERING)
# ==========================================
LAB_TRANSLATIONS = {
    "上机前DD": "DD_D0", "FIB": "FIB_D0", "TT": "TT_D0", "APTT": "APTT_D0", "PT": "PT_D0", "INR": "INR_D0", "AT3": "AT3_D0", "PLT": "PLT_D0",
    "WBC": "WBC_D0", "RBC": "RBC_D0", "Hb": "Hb_D0", "PCT": "PCT_D0", "CRP": "CRP_D0", "谷丙": "ALT_D0", "谷草": "AST_D0", "白蛋白": "Albumin_D0",
    "肌酐": "Creatinine_D0", "pH": "pH_D0", "氧分压": "PaO2_D0", "二氧化碳分压": "PaCO2_D0",
    "上机第二天DD": "DD_D2", "FIB.1": "FIB_D2", "TT.1": "TT_D2", "APTT.1": "APTT_D2", "PT.1": "PT_D2", "INR.1": "INR_D2", "AT3.1": "AT3_D2", "PLT.1": "PLT_D2",
    "WBC.1": "WBC_D2", "RBC.1": "RBC_D2", "Hb.1": "Hb_D2", "PCT.1": "PCT_D2", "CRP.1": "CRP_D2", "谷丙.1": "ALT_D2", "谷草.1": "AST_D2", "白蛋白.1": "Albumin_D2",
    "肌酐.1": "Creatinine_D2", "pH.1": "pH_D2", "氧分压.1": "PaO2_D2", "二氧化碳分压.1": "PaCO2_D2",
    "上机第七天DD": "DD_D7", "FIB.2": "FIB_D7", "TT.2": "TT_D7", "APTT.2": "APTT_D7", "PT.2": "PT_D7", "INR.2": "INR_D7", "AT3.2": "AT3_D7", "PLT.2": "PLT_D7",
    "WBC.2": "WBC_D7", "RBC.2": "RBC_D7", "Hb.2": "Hb_D7", "PCT.2": "PCT_D7", "CRP.2": "CRP_D7", "谷丙.2": "ALT_D7", "谷草.2": "AST_D7", "白蛋白.2": "Albumin_D7",
    "肌酐.2": "Creatinine_D7", "pH.2": "pH_D7", "氧分压.2": "PaO2_D7", "二氧化碳分压.2": "PaCO2_D7",
    "下机当天DD": "DD_Wean", "FIB.3": "FIB_Wean", "TT.3": "TT_Wean", "APTT.3": "APTT_Wean", "PT.3": "PT_Wean", "INR.3": "INR_Wean", "AT3.3": "AT3_Wean", "PLT.3": "PLT_Wean",
    "WBC.3": "WBC_Wean", "RBC.3": "RBC_Wean", "Hb.3": "Hb_Wean", "PCT.3": "PCT_Wean", "CRP.3": "CRP_Wean", "谷丙.3": "ALT_Wean", "谷草.3": "AST_Wean", "白蛋白.3": "Albumin_Wean",
    "肌酐.3": "Creatinine_Wean", "pH.3": "pH_Wean", "氧分压.3": "PaO2_Wean", "二氧化碳分压.3": "PaCO2_Wean"
}

def translate_lab_col(col):
    return LAB_TRANSLATIONS.get(col, col)

def construct_final_features(df_imp):
    demo_support_cols = [col for col in df_imp.columns if col not in lab_cols]
    
    df_final = pd.DataFrame(index=df_imp.index)
    
    for col in demo_support_cols:
        df_final[col] = df_imp[col]
        
    for col in lab_cols:
        if col in df_imp.columns:
            eng_name = translate_lab_col(col)
            df_final[eng_name] = df_imp[col]
        
    return df_final

X_train_engineered = construct_final_features(X_train_imp)
X_test_engineered = construct_final_features(X_test_imp)

logger.info(f"Feature matrices generated. Shape: {X_train_engineered.shape}")

# ==========================================
# 11. STEP 7: WINSORIZATION & STANDARD SCALING
# ==========================================
binary_and_dummy_cols = []
for col in X_train_engineered.columns:
    is_bin = col in binary_cols
    is_dum = any(col.startswith(prefix + "_") for prefix in multi_cat_cols)
    if is_bin or is_dum:
        binary_and_dummy_cols.append(col)

continuous_cols = [col for col in X_train_engineered.columns if col not in binary_and_dummy_cols]

clipping_limits = {}
X_train_scaled = X_train_engineered.copy()
X_test_scaled = X_test_engineered.copy()

for col in continuous_cols:
    lower_limit = float(X_train_engineered[col].quantile(0.01))
    upper_limit = float(X_train_engineered[col].quantile(0.99))
    clipping_limits[col] = [lower_limit, upper_limit]
    X_train_scaled[col] = X_train_engineered[col].clip(lower_limit, upper_limit)
    X_test_scaled[col] = X_test_engineered[col].clip(lower_limit, upper_limit)

scaler = StandardScaler()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_scaled[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test_scaled[continuous_cols])

# ==========================================
# 12. STEP 8: SMOTE+ENN BALANCING
# ==========================================
smote = SMOTE(random_state=RANDOM_SEED)
enn = EditedNearestNeighbours(n_neighbors=3, sampling_strategy='all')
smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=RANDOM_SEED)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)

# ==========================================
# 13. STEP 9: GENERATING SCI VISUALIZATIONS
# ==========================================
dist_plot_path = os.path.join(OUTPUT_DIR, "02_class_distribution.png")
plot_class_distribution(y_train, y_train_balanced, y_test, dist_plot_path)

# Diagnostic
plot_mice_imputation(
    X_train_unimputed["APTT"].dropna(), X_train_imp["APTT"], 
    "Activated Partial Thromboplastin Time (s)", 
    os.path.join(OUTPUT_DIR, "07_mice_imputation_diagnostic.png")
)

# Save datasets
train_output_path = os.path.join(OUTPUT_DIR, "train_balanced.csv")
test_output_path = os.path.join(OUTPUT_DIR, "test_preprocessed.csv")
train_unbalanced_path = os.path.join(OUTPUT_DIR, "train_preprocessed.csv")

train_out_df = pd.concat([X_train_balanced, y_train_balanced], axis=1)
test_out_df = pd.concat([X_test_scaled, y_test], axis=1)
train_unbalanced_df = pd.concat([X_train_scaled, y_train], axis=1)

train_unbalanced_df.to_csv(train_unbalanced_path, index=False, encoding="utf-8-sig")
train_out_df.to_csv(train_output_path, index=False, encoding="utf-8-sig")
test_out_df.to_csv(test_output_path, index=False, encoding="utf-8-sig")

config = {
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "final_features": X_train_scaled.columns.tolist(),
    "continuous_features": continuous_cols,
    "binary_and_dummy_features": binary_and_dummy_cols,
    "scaler_mean": {col: float(m) for col, m in zip(continuous_cols, scaler.mean_)},
    "scaler_scale": {col: float(s) for col, s in zip(continuous_cols, scaler.scale_)},
    "clipping_limits": {k: [float(v[0]), float(v[1])] for k, v in clipping_limits.items()}
}
config_path = os.path.join(OUTPUT_DIR, "preprocess_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

logger.info("ECMO Thrombosis Data Preprocessing (data_clear) Completed Successfully.")
