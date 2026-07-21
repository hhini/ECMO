import pandas as pd
import os

def convert_excel_to_csv():
    raw_excel_path = r"D:\acode\ECMO\data\原始数据2.xlsx"
    data1_csv_path = r"D:\acode\ECMO\data\data1.csv"
    data2_csv_path = r"D:\acode\ECMO\data\data2.csv"
    
    print(f"Reading raw Excel file from: {raw_excel_path}")
    df = pd.read_excel(raw_excel_path, sheet_name=0)
    
    # 1. Save data1.csv (all 382 rows)
    print(f"Saving full raw data (data1.csv) with {df.shape[0]} rows and {df.shape[1]} columns...")
    df.to_csv(data1_csv_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved data1.csv to: {data1_csv_path}")
    
    # 2. Process data2.csv (core cohort of 266 rows)
    print("Slicing first 266 rows for core cohort (data2.csv)...")
    core_df = df.iloc[:266].copy()
    
    # Define Lower Extremity Thrombosis target column
    lower_limb_keywords = ["下肢", "股", "大隐", "髂", "足"]
    
    def classify_thrombosis(val):
        if pd.isnull(val):
            return 0
        val_str = str(val).strip()
        if val_str == "无":
            return 0
        if any(kw in val_str for kw in lower_limb_keywords):
            return 1
        return 0
        
    core_df['是否下肢血栓'] = core_df['血栓事件'].apply(classify_thrombosis)
    
    positive_count = core_df['是否下肢血栓'].sum()
    negative_count = len(core_df) - positive_count
    print(f"Class distribution in core cohort:")
    print(f"  - Lower Extremity Thrombosis (1): {positive_count} ({positive_count/len(core_df)*100:.2f}%)")
    print(f"  - No Lower Extremity Thrombosis (0): {negative_count} ({negative_count/len(core_df)*100:.2f}%)")
    
    print(f"Saving core cohort data (data2.csv) with {core_df.shape[0]} rows and {core_df.shape[1]} columns...")
    core_df.to_csv(data2_csv_path, index=False, encoding='utf-8-sig')
    print(f"Successfully saved data2.csv to: {data2_csv_path}")

if __name__ == "__main__":
    convert_excel_to_csv()
