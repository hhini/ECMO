import os
import pandas as pd
import numpy as np

def inspect_data(file_path, output_report_path):
    print(f"Loading data from: {file_path}")
    
    # Load Excel file (we assume the first sheet is the main dataset)
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names
    print(f"Sheet names in Excel: {sheet_names}")
    
    # Load the first sheet
    df = pd.read_excel(file_path, sheet_name=0)
    
    num_rows, num_cols = df.shape
    print(f"Data shape: {num_rows} rows (samples), {num_cols} columns (features)")
    
    report_lines = []
    report_lines.append("# Data Inspection Report")
    report_lines.append(f"- **Data File Path**: `{file_path}`")
    report_lines.append(f"- **Active Sheet Name**: `{sheet_names[0]}`")
    report_lines.append(f"- **Total Samples (Rows)**: {num_rows}")
    report_lines.append(f"- **Total Features (Columns)**: {num_cols}")
    report_lines.append("\n## Sheets Found in Excel File")
    for name in sheet_names:
        report_lines.append(f"- {name}")
    
    report_lines.append("\n## Feature-by-Feature Detailed Inspection")
    report_lines.append("| No. | Feature Name | Data Type | Classification | Missing Count | Missing % | Unique Values Count | Sample Unique Values |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    
    for idx, col in enumerate(df.columns, 1):
        col_data = df[col]
        dtype = str(col_data.dtype)
        
        # Missing values
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / num_rows) * 100
        
        # Unique values
        unique_vals = col_data.dropna().unique()
        unique_count = len(unique_vals)
        
        # Classification: numeric vs character
        if pd.api.types.is_numeric_dtype(col_data):
            classification = "Numeric"
        else:
            classification = "Character/Categorical"
            
        # Sample of unique values
        if unique_count == 0:
            sample_vals_str = "N/A (All missing)"
        elif unique_count <= 15:
            # Sort if possible for cleaner presentation
            try:
                sorted_vals = sorted(unique_vals)
            except Exception:
                sorted_vals = unique_vals
            sample_vals_str = ", ".join([str(x) for x in sorted_vals])
        else:
            # Show a subset
            try:
                sorted_vals = sorted(unique_vals)
            except Exception:
                sorted_vals = unique_vals
            sample_vals_str = ", ".join([str(x) for x in sorted_vals[:7]]) + " ... [and " + str(unique_count - 10) + " more] ... " + ", ".join([str(x) for x in sorted_vals[-3:]])
            
        # Truncate sample string if too long for markdown table
        if len(sample_vals_str) > 100:
            sample_vals_str = sample_vals_str[:97] + "..."
            
        # Escape markdown characters in column names and values to avoid formatting break
        col_escaped = str(col).replace("|", "\\|")
        sample_vals_str_escaped = str(sample_vals_str).replace("|", "\\|").replace("\n", " ")
        
        report_lines.append(
            f"| {idx} | {col_escaped} | {dtype} | {classification} | {missing_count} | {missing_pct:.2f}% | {unique_count} | {sample_vals_str_escaped} |"
        )
        
    # Write report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"Data inspection report successfully saved to: {output_report_path}")

if __name__ == "__main__":
    raw_data_path = r"D:\acode\ECMO\data\原始数据2.xlsx"
    report_path = r"D:\acode\ECMO\output\data_inspection_report.md"
    inspect_data(raw_data_path, report_path)
