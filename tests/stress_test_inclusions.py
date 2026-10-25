# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = r"D:\acode\ECMO"
df_clean = pd.read_csv(f"{PROJECT_ROOT}/output/01_stage2_clean/stage2_clean.csv")

print("=== INCLUSION CRITERIA STRESS TEST ON CLEAN DATASET (N=106) ===")

# 1. Age check
ages = df_clean['年龄']
print(f"Age range: min={ages.min()}, max={ages.max()}, nulls={ages.isna().sum()}")
assert (ages >= 18).all(), "FAIL: Found age < 18!"
assert (ages <= 120).all(), "FAIL: Found impossible age > 120!"
print("Age check: PASS (All patients >= 18 and <= 120)")

# 2. ECMO duration check
durations = df_clean['ECMO时长']
print(f"ECMO duration range: min={durations.min()}, max={durations.max()}, nulls={durations.isna().sum()}")
assert (durations >= 24.0).all(), "FAIL: Found ECMO duration < 24h!"
print("ECMO duration check: PASS (All patients >= 24h)")

# 3. Prior thrombosis history check
prior_thromb = df_clean['既往血栓史']
print(f"Prior thrombosis history values: {prior_thromb.value_counts(dropna=False).to_dict()}")
assert ((prior_thromb == '无') | (prior_thromb.isna())).all(), "FAIL: Found prior thrombosis history in clean cohort!"
print("Prior thrombosis check: PASS (No prior thrombosis history)")

# 4. Target variable completeness and valid values
target = df_clean['是否下肢血栓']
print(f"Target distribution: {target.value_counts(dropna=False).to_dict()}")
assert set(target.unique()) == {0, 1}, "FAIL: Target contains values other than 0 and 1!"
assert target.isna().sum() == 0, "FAIL: Target contains NaNs!"
print("Target variable check: PASS (Binary 0/1 without missingness)")

# 5. Primary outcomes completeness
assert df_clean['血栓事件'].notna().all(), "FAIL: '血栓事件' contains NaNs!"
assert df_clean['结局'].notna().all(), "FAIL: '结局' contains NaNs!"
print("Primary outcomes check: PASS (100% complete)")
