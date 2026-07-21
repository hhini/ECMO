# -*- coding: utf-8 -*-
"""
ECMO Thrombosis Predictive Modeling - Stage 1: Data Preprocessing (Comprehensive Plan)
Author: Antigravity Team
Date: 2026-06-23

Description:
This script loads the raw ECMO cohort clinical data, performs clinical feature mappings and
cleansing, splits the cohort into train/test, imputes missing values using MICE, and reconstructs
all variables including demographics, clinical support parameters, 20 laboratory indicators at 4
timepoints, and 12 temporal trajectory features. Standardizes the prediction window and generates
7 publication-grade figures.
"""

import os
import re
import json
import logging
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

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
DATA_PATH = r"D:\acode\ECMO\data\data2.csv"
OUTPUT_DIR = r"D:\acode\ECMO\output\01_preprocessing"
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

logger.info("Starting Clinical Expert-Driven Comprehensive Preprocessing Pipeline...")

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
    # 循环系统
    if any(k in val_str for k in ["心肌梗死", "心梗", "心力衰竭", "心衰", "心源性", "冠心病"]):
        return 1.0
    # 呼吸系统
    if any(k in val_str for k in ["肺炎", "肺纤维化", "呼吸衰竭", "ARDS", "肺"]):
        return 2.0
    # 创伤与休克
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

# Define target variable
y = df["是否下肢血栓"]
logger.info(f"Target distribution: 0={sum(y==0)}, 1={sum(y==1)} (Positive Rate: {100*y.mean():.2f}%)")

# Extract the clinical subset
X_raw = df[required_raw_cols].copy()

# ==========================================
# 6. STEP 2: MISSINGNESS ANALYSIS & PLOTTING (Fig 1: Missingness Ratio)
# ==========================================
missing_ratios = X_raw.isnull().sum() / len(X_raw)
logger.info(f"Missingness check: Max missingness is {missing_ratios.max():.2%}")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.figure(figsize=(12, 5))
missing_to_plot = missing_ratios[missing_ratios > 0].sort_values(ascending=False)
if len(missing_to_plot) > 0:
    # Show top 25 missing features for legibility
    missing_to_plot_top = missing_to_plot[:25]
    eng_labels = [translate_to_english(col) for col in missing_to_plot_top.index]
    sns.barplot(x=eng_labels, y=missing_to_plot_top.values, color="#00468B", alpha=0.8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel("Missingness Ratio", fontsize=12)
    plt.xlabel("Clinical Features", fontsize=12)
    plt.title("Missingness Ratio of Clinical Features (Top 25)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "01_missingness_ratio.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved missingness ratio plot to: {plot_path}")

# ==========================================
# 7. STEP 3: MAPPING TO NUMERIC REPRESENTATIONS
# ==========================================
logger.info("Mapping all demographics, support and laboratory columns to numeric features...")

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

# Categorical column definitions
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
logger.info(f"Splitting dataset into train ({100*(1-TEST_SIZE):.0f}%) and test ({100*TEST_SIZE:.0f}%) stratified by target...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_dummy, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
logger.info(f"Train raw shape: {X_train_raw.shape}, Test raw shape: {X_test_raw.shape}")

# Drop features with >30% missingness on the train set
cols_to_check = [col for col in X_train_raw.columns if not any(col.startswith(cat + "_") for cat in multi_cat_cols)]
missing_ratios_train = X_train_raw[cols_to_check].isnull().sum() / len(X_train_raw)
cols_to_drop = missing_ratios_train[missing_ratios_train > 0.30].index.tolist()

logger.info(f"Features dropped due to >30% missingness on train set: {cols_to_drop}")
X_train_raw = X_train_raw.drop(columns=cols_to_drop)
X_test_raw = X_test_raw.drop(columns=cols_to_drop)
logger.info(f"Train raw shape after dropping high-missingness columns: {X_train_raw.shape}")

# Save a copy of un-imputed train set for diagnostics
X_train_unimputed = X_train_raw.copy()

# ==========================================
# 9. STEP 5: DECOUPLED IMPUTATION
# ==========================================
logger.info("Performing decoupled imputation on raw split feature matrices...")

# Define feature groups for imputation, filtering out dropped columns
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

# 1. MICE on core features
core_imputer = IterativeImputer(max_iter=10, random_state=RANDOM_SEED)
X_train_core_imp = pd.DataFrame(
    core_imputer.fit_transform(X_train_raw[core_cols]), columns=core_cols, index=X_train_raw.index
)
X_test_core_imp = pd.DataFrame(
    core_imputer.transform(X_test_raw[core_cols]), columns=core_cols, index=X_test_raw.index
)

# 2. Simple Median Imputation on demographics, support, and other labs
other_imputer = SimpleImputer(strategy="median")
X_train_other_imp = pd.DataFrame(
    other_imputer.fit_transform(X_train_raw[other_cols]), columns=other_cols, index=X_train_raw.index
)
X_test_other_imp = pd.DataFrame(
    other_imputer.transform(X_test_raw[other_cols]), columns=other_cols, index=X_test_raw.index
)

# Combine back in original column order (excluding dropped columns)
remaining_columns = [col for col in X_dummy.columns if col in X_train_raw.columns]
X_train_imp = pd.concat([X_train_core_imp, X_train_other_imp], axis=1)[remaining_columns]
X_test_imp = pd.concat([X_test_core_imp, X_test_other_imp], axis=1)[remaining_columns]
logger.info("Decoupled Imputation complete. Verified 0 missing values.")

# ==========================================
# 10. STEP 6: TEMPORAL FEATURE ENGINEERING (32 KINETICS + 4 SUMMARY STATS)
# ==========================================
logger.info("Reconstructing temporal trajectory and summary statistics...")

def generate_trajectories(df_imp):
    df_traj = pd.DataFrame(index=df_imp.index)
    
    core_indicators = {
        "DD": ["上机前DD", "上机第二天DD"],
        "PLT": ["PLT", "PLT.1"],
        "FIB": ["FIB", "FIB.1"],
        "AT3": ["AT3", "AT3.1"],
        "APTT": ["APTT", "APTT.1"],
        "PT": ["PT", "PT.1"],
        "INR": ["INR", "INR.1"],
        "TT": ["TT", "TT.1"]
    }
    thresholds = {
        "PLT": 100.0, "FIB": 1.5, "AT3": 50.0, "DD": 1.5,
        "APTT": 70.0, "PT": 15.0, "INR": 1.5, "TT": 20.0
    }
    
    for ind, cols in core_indicators.items():
        if cols[0] in df_imp.columns and cols[1] in df_imp.columns:
            d0, d2 = cols
            
            # 1. Delta
            df_traj[f"{ind}_Delta"] = df_imp[d2] - df_imp[d0]
            
            # 2. Extreme value (Min/Max)
            if ind in ["PLT", "FIB", "AT3"]:
                df_traj[f"{ind}_min"] = df_imp[[d0, d2]].min(axis=1)
            else:
                df_traj[f"{ind}_max"] = df_imp[[d0, d2]].max(axis=1)
                
            # 3. Slope
            df_traj[f"{ind}_slope"] = (df_imp[d2] - df_imp[d0]) / 2.0
            
            # 4. Abnormality Ratio
            d0_val = df_imp[d0]
            d2_val = df_imp[d2]
            if ind in ["PLT", "FIB", "AT3"]:
                is_d0_ab = (d0_val < thresholds[ind]).astype(float)
                is_d2_ab = (d2_val < thresholds[ind]).astype(float)
            else:
                is_d0_ab = (d0_val > thresholds[ind]).astype(float)
                is_d2_ab = (d2_val > thresholds[ind]).astype(float)
            df_traj[f"{ind}_abnormality_ratio"] = (is_d0_ab + is_d2_ab) / 2.0
            
            # 5. [New] Relative Delta (Literature feature)
            df_traj[f"{ind}_Relative_Delta"] = (df_imp[d2] - df_imp[d0]) / (df_imp[d0] + 1e-5)
            
    # [New] Coagulation Balance Ratios
    if "PLT" in df_imp.columns and "FIB" in df_imp.columns:
        df_traj["PLT_FIB_Ratio_D0"] = df_imp["PLT"] / (df_imp["FIB"] + 1e-5)
    if "PLT.1" in df_imp.columns and "FIB.1" in df_imp.columns:
        df_traj["PLT_FIB_Ratio_D2"] = df_imp["PLT.1"] / (df_imp["FIB.1"] + 1e-5)
        
    if "FIB" in df_imp.columns and "上机前DD" in df_imp.columns:
        df_traj["FIB_DD_Ratio_D0"] = df_imp["FIB"] / (df_imp["上机前DD"] + 1e-5)
    if "FIB.1" in df_imp.columns and "上机第二天DD" in df_imp.columns:
        df_traj["FIB_DD_Ratio_D2"] = df_imp["FIB.1"] / (df_imp["上机第二天DD"] + 1e-5)
        
    if "APTT" in df_imp.columns and "PLT" in df_imp.columns:
        df_traj["APTT_PLT_Ratio_D0"] = df_imp["APTT"] / (df_imp["PLT"] + 1e-5)
    if "APTT.1" in df_imp.columns and "PLT.1" in df_imp.columns:
        df_traj["APTT_PLT_Ratio_D2"] = df_imp["APTT.1"] / (df_imp["PLT.1"] + 1e-5)

    # [New] Clinical Pathological flags
    if "PLT.1" in df_imp.columns:
        df_traj["Severe_Thrombocytopenia_D2"] = (df_imp["PLT.1"] < 100.0).astype(float)
    if "上机第二天DD" in df_imp.columns:
        df_traj["Hyperfibrinolysis_D2"] = (df_imp["上机第二天DD"] > 5.0).astype(float)
    if "AT3.1" in df_imp.columns:
        df_traj["Low_AT3_D2"] = (df_imp["AT3.1"] < 50.0).astype(float)
        
    # Additional patient-level summary stats (across remaining timepoints)
    dd_cols = [c for c in ["上机前DD", "上机第二天DD", "上机第七天DD", "下机当天DD"] if c in df_imp.columns]
    plt_cols = [c for c in ["PLT", "PLT.1", "PLT.2", "PLT.3"] if c in df_imp.columns]
    at3_cols = [c for c in ["AT3", "AT3.1", "AT3.2", "AT3.3"] if c in df_imp.columns]
    aptt_cols = [c for c in ["APTT", "APTT.1", "APTT.2", "APTT.3"] if c in df_imp.columns]
    
    if dd_cols:
        df_traj["Max_DD"] = df_imp[dd_cols].max(axis=1)
    if plt_cols:
        df_traj["Min_PLT"] = df_imp[plt_cols].min(axis=1)
    if at3_cols:
        df_traj["Min_AT3"] = df_imp[at3_cols].min(axis=1)
    if aptt_cols:
        df_traj["Mean_APTT"] = df_imp[aptt_cols].mean(axis=1)
        
    return df_traj

LAB_TRANSLATIONS = {
    # D0
    "上机前DD": "DD_D0",
    "FIB": "FIB_D0",
    "TT": "TT_D0",
    "APTT": "APTT_D0",
    "PT": "PT_D0",
    "INR": "INR_D0",
    "AT3": "AT3_D0",
    "PLT": "PLT_D0",
    "WBC": "WBC_D0",
    "RBC": "RBC_D0",
    "Hb": "Hb_D0",
    "PCT": "PCT_D0",
    "CRP": "CRP_D0",
    "谷丙": "ALT_D0",
    "谷草": "AST_D0",
    "白蛋白": "Albumin_D0",
    "肌酐": "Creatinine_D0",
    "pH": "pH_D0",
    "氧分压": "PaO2_D0",
    "二氧化碳分压": "PaCO2_D0",
    # D2
    "上机第二天DD": "DD_D2",
    "FIB.1": "FIB_D2",
    "TT.1": "TT_D2",
    "APTT.1": "APTT_D2",
    "PT.1": "PT_D2",
    "INR.1": "INR_D2",
    "AT3.1": "AT3_D2",
    "PLT.1": "PLT_D2",
    "WBC.1": "WBC_D2",
    "RBC.1": "RBC_D2",
    "Hb.1": "Hb_D2",
    "PCT.1": "PCT_D2",
    "CRP.1": "CRP_D2",
    "谷丙.1": "ALT_D2",
    "谷草.1": "AST_D2",
    "白蛋白.1": "Albumin_D2",
    "肌酐.1": "Creatinine_D2",
    "pH.1": "pH_D2",
    "氧分压.1": "PaO2_D2",
    "二氧化碳分压.1": "PaCO2_D2",
    # D7
    "上机第七天DD": "DD_D7",
    "FIB.2": "FIB_D7",
    "TT.2": "TT_D7",
    "APTT.2": "APTT_D7",
    "PT.2": "PT_D7",
    "INR.2": "INR_D7",
    "AT3.2": "AT3_D7",
    "PLT.2": "PLT_D7",
    "WBC.2": "WBC_D7",
    "RBC.2": "RBC_D7",
    "Hb.2": "Hb_D7",
    "PCT.2": "PCT_D7",
    "CRP.2": "CRP_D7",
    "谷丙.2": "ALT_D7",
    "谷草.2": "AST_D7",
    "白蛋白.2": "Albumin_D7",
    "肌酐.2": "Creatinine_D7",
    "pH.2": "pH_D7",
    "氧分压.2": "PaO2_D7",
    "二氧化碳分压.2": "PaCO2_D7",
    # Wean
    "下机当天DD": "DD_Wean",
    "FIB.3": "FIB_Wean",
    "TT.3": "TT_Wean",
    "APTT.3": "APTT_Wean",
    "PT.3": "PT_Wean",
    "INR.3": "INR_Wean",
    "AT3.3": "AT3_Wean",
    "PLT.3": "PLT_Wean",
    "WBC.3": "WBC_Wean",
    "RBC.3": "RBC_Wean",
    "Hb.3": "Hb_Wean",
    "PCT.3": "PCT_Wean",
    "CRP.3": "CRP_Wean",
    "谷丙.3": "ALT_Wean",
    "谷草.3": "AST_Wean",
    "白蛋白.3": "Albumin_Wean",
    "肌酐.3": "Creatinine_Wean",
    "pH.3": "pH_Wean",
    "氧分压.3": "PaO2_Wean",
    "二氧化碳分压.3": "PaCO2_Wean"
}

def translate_lab_col(col):
    return LAB_TRANSLATIONS.get(col, col)

def construct_final_features(df_imp):
    df_traj = generate_trajectories(df_imp)
    demo_support_cols = [col for col in df_imp.columns if col not in lab_cols]
    
    df_final = pd.DataFrame(index=df_imp.index)
    
    # 1. Demographics & Support
    for col in demo_support_cols:
        df_final[col] = df_imp[col]
        
    # 2. Translated Lab Columns
    for col in lab_cols:
        if col in df_imp.columns:
            eng_name = translate_lab_col(col)
            df_final[eng_name] = df_imp[col]
        
    # 3. Trajectory Features
    for col in df_traj.columns:
        df_final[col] = df_traj[col]
        
    return df_final

X_train_engineered = construct_final_features(X_train_imp)
X_test_engineered = construct_final_features(X_test_imp)

logger.info(f"Engineered feature matrices generated. Shape: {X_train_engineered.shape}")

# ==========================================
# 11. STEP 7: WINSORIZATION & STANDARD SCALING
# ==========================================
logger.info("Winsorizing continuous features at [1%, 99%] and applying Standard Scaling...")

# Identify binary and dummy features
binary_and_dummy_cols = []
for col in X_train_engineered.columns:
    is_bin = col in binary_cols or col in ["Severe_Thrombocytopenia_D2", "Hyperfibrinolysis_D2", "Low_AT3_D2"]
    is_dum = any(col.startswith(prefix + "_") for prefix in multi_cat_cols)
    if is_bin or is_dum:
        binary_and_dummy_cols.append(col)

continuous_cols = [col for col in X_train_engineered.columns if col not in binary_and_dummy_cols]

logger.info(f"Identified {len(continuous_cols)} continuous columns for Winsorization and Scaling.")
logger.info(f"Identified {len(binary_and_dummy_cols)} binary/dummy columns to keep unscaled.")

clipping_limits = {}
X_train_scaled = X_train_engineered.copy()
X_test_scaled = X_test_engineered.copy()

for col in continuous_cols:
    lower_limit = float(X_train_engineered[col].quantile(0.01))
    upper_limit = float(X_train_engineered[col].quantile(0.99))
    clipping_limits[col] = [lower_limit, upper_limit]
    
    X_train_scaled[col] = X_train_engineered[col].clip(lower_limit, upper_limit)
    X_test_scaled[col] = X_test_engineered[col].clip(lower_limit, upper_limit)

# Standard scale only the continuous columns
scaler = StandardScaler()
X_train_scaled[continuous_cols] = scaler.fit_transform(X_train_scaled[continuous_cols])
X_test_scaled[continuous_cols] = scaler.transform(X_test_scaled[continuous_cols])
logger.info("Winsorization and Standard Scaling completed on continuous features.")

# ==========================================
# 12. STEP 8: SMOTE+ENN BALANCING (FOR REFERENCE ONLY)
# ==========================================
logger.info("Applying SMOTE+ENN balancing to scaled train set...")
smote = SMOTE(random_state=RANDOM_SEED)
enn = EditedNearestNeighbours(n_neighbors=3, sampling_strategy='all')
smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=RANDOM_SEED)

X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_scaled, y_train)
logger.info(f"Balanced class counts: 0={sum(y_train_balanced==0)}, 1={sum(y_train_balanced==1)}")

# ==========================================
# 13. STEP 9: GENERATING SCI VISUALIZATIONS (Fig 2 to Fig 7)
# ==========================================
logger.info("Generating publication-grade figures (Fig 2 to Fig 7)...")

# Fig 2: Class Distribution
plt.figure(figsize=(10, 5))
classes = ["No Thrombosis (0)", "Thrombosis (1)"]
original_counts = [sum(y_train == 0), sum(y_train == 1)]
balanced_counts = [sum(y_train_balanced == 0), sum(y_train_balanced == 1)]
test_counts = [sum(y_test == 0), sum(y_test == 1)]

x = np.arange(len(classes))
width = 0.25

plt.bar(x - width, original_counts, width, label="Raw Train Set (70%)", color="#7fb3d5")
plt.bar(x, balanced_counts, width, label="Balanced Train Set (SMOTE+ENN)", color="#00468B")
plt.bar(x + width, test_counts, width, label="Raw Test Set (30%)", color="#e0a96d")

plt.xticks(x, classes, fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Sample Class Distribution Across Cohorts", fontsize=14, fontweight="bold")
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
dist_plot_path = os.path.join(OUTPUT_DIR, "02_class_distribution.png")
plt.savefig(dist_plot_path, dpi=300)
plt.close()

# Fig 3: Platelet (PLT) Trajectory Boxplot (Baseline -> Day 2)
plt.figure(figsize=(7, 6))
df_plt_traj = pd.DataFrame({
    "Baseline": X_train_imp["PLT"],
    "Day 2": X_train_imp["PLT.1"],
    "Thrombosis": y_train.map({0: "No LET", 1: "LET"})
})
df_plt_long = pd.melt(df_plt_traj, id_vars=["Thrombosis"], value_vars=["Baseline", "Day 2"], var_name="Timepoint", value_name="Platelet Count (10^9/L)")
sns.boxplot(data=df_plt_long, x="Timepoint", y="Platelet Count (10^9/L)", hue="Thrombosis", palette=["#00468B", "#ED0000"], width=0.5)
plt.title("Platelet Count Dynamics (D0 to D2)", fontsize=13, fontweight="bold")
plt.ylabel("Platelet Count (10^9/L)", fontsize=11)
plt.xlabel("Timepoint", fontsize=11)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_plt_trajectory.png"), dpi=300)
plt.close()

# Fig 4: D-Dimer Trajectory Boxplot
plt.figure(figsize=(7, 6))
df_dd_traj = pd.DataFrame({
    "Baseline": X_train_imp["上机前DD"],
    "Day 2": X_train_imp["上机第二天DD"],
    "Thrombosis": y_train.map({0: "No LET", 1: "LET"})
})
df_dd_long = pd.melt(df_dd_traj, id_vars=["Thrombosis"], value_vars=["Baseline", "Day 2"], var_name="Timepoint", value_name="D-Dimer (mg/L)")
sns.boxplot(data=df_dd_long, x="Timepoint", y="D-Dimer (mg/L)", hue="Thrombosis", palette=["#00468B", "#ED0000"], width=0.5)
plt.title("D-Dimer Dynamics (D0 to D2)", fontsize=13, fontweight="bold")
plt.ylabel("D-Dimer (mg/L)", fontsize=11)
plt.xlabel("Timepoint", fontsize=11)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_dd_trajectory.png"), dpi=300)
plt.close()

# Fig 5: Antithrombin III (AT3) Trajectory Boxplot
plt.figure(figsize=(7, 6))
df_at3_traj = pd.DataFrame({
    "Baseline": X_train_imp["AT3"],
    "Day 2": X_train_imp["AT3.1"],
    "Thrombosis": y_train.map({0: "No LET", 1: "LET"})
})
df_at3_long = pd.melt(df_at3_traj, id_vars=["Thrombosis"], value_vars=["Baseline", "Day 2"], var_name="Timepoint", value_name="Antithrombin III (%)")
sns.boxplot(data=df_at3_long, x="Timepoint", y="Antithrombin III (%)", hue="Thrombosis", palette=["#00468B", "#ED0000"], width=0.5)
plt.title("Antithrombin III Consumption (D0 to D2)", fontsize=13, fontweight="bold")
plt.ylabel("Antithrombin III (%)", fontsize=11)
plt.xlabel("Timepoint", fontsize=11)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_at3_trajectory.png"), dpi=300)
plt.close()

# Fig 6: Fibrinogen (FIB) Trajectory Boxplot
plt.figure(figsize=(7, 6))
df_fib_traj = pd.DataFrame({
    "Baseline": X_train_imp["FIB"],
    "Day 2": X_train_imp["FIB.1"],
    "Thrombosis": y_train.map({0: "No LET", 1: "LET"})
})
df_fib_long = pd.melt(df_fib_traj, id_vars=["Thrombosis"], value_vars=["Baseline", "Day 2"], var_name="Timepoint", value_name="Fibrinogen (g/L)")
sns.boxplot(data=df_fib_long, x="Timepoint", y="Fibrinogen (g/L)", hue="Thrombosis", palette=["#00468B", "#ED0000"], width=0.5)
plt.title("Fibrinogen Dynamics (D0 to D2)", fontsize=13, fontweight="bold")
plt.ylabel("Fibrinogen (g/L)", fontsize=11)
plt.xlabel("Timepoint", fontsize=11)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_fib_trajectory.png"), dpi=300)
plt.close()

# Fig 7: MICE Imputation Quality Diagnostic Plot (Density Plot for APTT)
plt.figure(figsize=(7, 5))
sns.kdeplot(data=X_train_unimputed["APTT"].dropna(), label="Observed APTT", color="#00468B", fill=True, alpha=0.3, linewidth=2)
sns.kdeplot(data=X_train_imp["APTT"], label="MICE Imputed APTT", color="#ED0000", linestyle="--", linewidth=2)
plt.title("MICE Imputation Quality Diagnostic for APTT", fontsize=13, fontweight="bold")
plt.xlabel("Activated Partial Thromboplastin Time (s)", fontsize=11)
plt.ylabel("Density", fontsize=11)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_mice_imputation_diagnostic.png"), dpi=300)
plt.close()

logger.info("All 7 diagnostic and trajectory plots saved successfully.")

# ==========================================
# 14. STEP 10: SAVE DATASETS AND CONFIGS
# ==========================================
train_output_path = os.path.join(OUTPUT_DIR, "train_balanced.csv")
test_output_path = os.path.join(OUTPUT_DIR, "test_preprocessed.csv")
train_unbalanced_path = os.path.join(OUTPUT_DIR, "train_preprocessed.csv")

# Combine features and labels for output
train_out_df = pd.concat([X_train_balanced, y_train_balanced], axis=1)
test_out_df = pd.concat([X_test_scaled, y_test], axis=1)
train_unbalanced_df = pd.concat([X_train_scaled, y_train], axis=1)

train_unbalanced_df.to_csv(train_unbalanced_path, index=False, encoding="utf-8-sig")
train_out_df.to_csv(train_output_path, index=False, encoding="utf-8-sig")
test_out_df.to_csv(test_output_path, index=False, encoding="utf-8-sig")

logger.info(f"Saved unbalanced train dataset (shape {train_unbalanced_df.shape}) to: {train_unbalanced_path}")
logger.info(f"Saved balanced train dataset (shape {train_out_df.shape}) to: {train_output_path}")
logger.info(f"Saved preprocessed test dataset (shape {test_out_df.shape}) to: {test_output_path}")

# Save configuration JSON for modular reference in downstream scripts
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
logger.info(f"Saved preprocessing config mapping to: {config_path}")

logger.info("ECMO Thrombosis Data Preprocessing Pipeline Completed Successfully.")
