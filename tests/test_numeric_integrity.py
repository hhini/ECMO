# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_clean = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")

print("=== TESTING ALL 153 COLUMNS FOR DATA TYPES & CONVERTIBILITY ===")
errors = []
for col in df_clean.columns:
    s = df_clean[col]
    # If column is laboratory or physiologic numeric
    # Check if dtype is float/int
    if col not in [
        '入院时间', '性别', '居住地', '工作状态', '职业', '吸烟', '饮酒', '妊娠状态',
        'ECMO前意识', '首要诊断', '次要诊断', '基础疾病', '既往血栓史', '既往血管性疾病史',
        '既往手术史', '家族血管性遗传病史', 'ECMO前手术史', '自体血回输', 'ECMO治疗期间意识',
        '静脉输注营养制剂使用', '联合血液净化治疗', 'ECMO模式', '置管部位', '置管管径',
        '模式转换', '多次置管', '侧支循环', '血型', '血栓事件', '结局'
    ]:
        if s.dtype not in [np.dtype('float64'), np.dtype('int64')]:
            errors.append((col, s.dtype, "Not numeric dtype!"))
        # Check if contains any string
        str_mask = s.apply(lambda x: isinstance(x, str))
        if str_mask.any():
            errors.append((col, s.dtype, f"Contains strings: {s[str_mask].unique()}"))

print(f"Total numeric columns tested: {153 - 30} = 123 columns.")
print(f"Errors found: {len(errors)}")
if errors:
    for err in errors:
        print(f"  ERROR: {err}")
else:
    print("ALL 123 NUMERIC COLUMNS ARE STRICTLY NUMERIC DTYPES WITH ZERO HIDDEN STRINGS!")
