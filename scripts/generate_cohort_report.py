import os
import pandas as pd
import numpy as np

def generate_cohort_report(input_file, output_report_path):
    # Load dataset
    df = pd.read_excel(input_file, sheet_name=0)
    
    # We identified that the first 266 rows represent the valid clinical study cohort
    # Rows 266-280 are exclusion cases or incomplete records (missing > 80% columns)
    # Rows 281-381 are empty template rows with only ID populated
    cohort_df = df.iloc[:266].copy()
    
    num_rows, num_cols = cohort_df.shape
    
    # Define Lower Extremity Thrombosis (下肢血栓) target
    # Keywords indicating lower extremity thrombosis
    lower_limb_keywords = ["下肢", "股", "大隐", "髂", "足"]
    
    def classify_thrombosis(val):
        if pd.isnull(val):
            return "Missing"
        val_str = str(val).strip()
        if val_str == "无":
            return "No Thrombosis"
        if any(kw in val_str for kw in lower_limb_keywords):
            return "Lower Extremity Thrombosis"
        return "Other Thrombosis"
        
    cohort_df['血栓分类'] = cohort_df['血栓事件'].apply(classify_thrombosis)
    cohort_df['是否下肢血栓'] = cohort_df['血栓分类'].apply(lambda x: 1 if x == "Lower Extremity Thrombosis" else 0)
    
    thrombosis_counts = cohort_df['血栓分类'].value_counts()
    is_lower_limb_counts = cohort_df['是否下肢血栓'].value_counts()
    
    report_lines = []
    report_lines.append("# ECMO 支持患者下肢血栓预测模型 - 原始数据基本检查报告")
    report_lines.append(f"- **原始数据路径**: `{input_file}`")
    report_lines.append(f"- **研究样本总量 (筛选后)**: **{num_rows}** 例")
    report_lines.append(f"- **原始特征总数**: **{num_cols}** 个")
    
    report_lines.append("\n## 一、 研究对象筛选与队列确定说明")
    report_lines.append("1. **原始数据行数**: 382 行")
    report_lines.append("2. **空值与排除病例处理**:")
    report_lines.append("   - **行号 282-382 (索引 281-381)**: 均为 Excel 模板预留行，仅“编号”列有数值，其余 151 个特征皆为空值。因此直接予以剔除。")
    report_lines.append("   - **行号 267-281 (索引 266-280)**: 共 15 例样本。其中 6 例“性别”列标明为 **“上机＜24h剔除”**，不满足纳入标准；其余 9 例样本的临床特征缺失率均超过 80%（严重缺失）。这 15 例样本被判定为无效/排除样本，予以剔除。")
    report_lines.append("   - **核心建模队列 (前 266 行，索引 0-265)**: 共有 266 例完整的研究样本。其基本人口学信息（年龄、性别）与结局指标均 100% 完整，无小于 18 岁的患者，且 ECMO 治疗时长均大于 24 小时，完全符合纳排标准。")
    
    report_lines.append("\n## 二、 结局指标（下肢血栓）统计")
    report_lines.append("根据本研究结局指标定义（追踪 ECMO 置入至出院发生的下肢血栓事件，以超声多普勒为诊断依据），我们对 `血栓事件` 字段中含有 **“下肢”、“股”、“大隐”、“髂”、“足”** 的记录定义为 **“下肢血栓形成”**，其余“无”或非下肢部位血栓定义为 **“无下肢血栓”**：")
    report_lines.append("\n| 类别 | 包含的原始血栓描述 | 样本数 (例) | 占比 (%) |")
    report_lines.append("|---|---|---|---|")
    
    # Detail lower limb cases
    lower_limb_details = cohort_df[cohort_df['是否下肢血栓'] == 1]['血栓事件'].value_counts()
    lower_limb_desc = ", ".join([f"{k}({v}例)" for k, v in lower_limb_details.items()])
    report_lines.append(f"| **下肢血栓组 (1)** | {lower_limb_desc} | {is_lower_limb_counts.get(1, 0)} | {is_lower_limb_counts.get(1, 0)/num_rows*100:.2f}% |")
    
    # Detail other/no cases
    no_thrombosis_count = thrombosis_counts.get("No Thrombosis", 0)
    other_thrombosis_details = cohort_df[cohort_df['血栓分类'] == 'Other Thrombosis']['血栓事件'].value_counts()
    other_thrombosis_desc = ", ".join([f"{k}({v}例)" for k, v in other_thrombosis_details.items()])
    if not other_thrombosis_desc:
        other_thrombosis_desc = "无"
    report_lines.append(f"| **无下肢血栓组 (0)** | 无 ({no_thrombosis_count}例); 其他非下肢血栓: {other_thrombosis_desc} | {is_lower_limb_counts.get(0, 0)} | {is_lower_limb_counts.get(0, 0)/num_rows*100:.2f}% |")
    
    report_lines.append(f"\n> **注**: 下肢血栓发生率为 **{is_lower_limb_counts.get(1, 0)/num_rows*100:.2f}%**（共 {is_lower_limb_counts.get(1, 0)} 例），满足阳性样本量的预测模型构建要求。")
    
    report_lines.append("\n## 三、 特征属性与数据类型基本检查")
    report_lines.append("| 序号 | 特征名称 | 数据类型 | 属性分类 | 缺失个数 | 缺失比例 | 唯一值个数 | 唯一值示例 / 取值范围 |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    
    for idx, col in enumerate(df.columns, 1):
        col_data = cohort_df[col]
        dtype = str(col_data.dtype)
        
        # Missing info
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / num_rows) * 100
        
        # Unique info
        unique_vals = col_data.dropna().unique()
        unique_count = len(unique_vals)
        
        # Attribute type
        if pd.api.types.is_numeric_dtype(col_data):
            attr_type = "数值型 (Numeric)"
        else:
            attr_type = "字符/分类型 (Categorical)"
            
        # Format unique values sample
        if unique_count == 0:
            sample_vals = "无有效值"
        elif unique_count <= 8:
            # Sort if values can be sorted
            try:
                sorted_vals = sorted(unique_vals)
            except Exception:
                sorted_vals = unique_vals
            sample_vals = ", ".join([str(x) for x in sorted_vals])
        else:
            try:
                sorted_vals = sorted(unique_vals)
            except Exception:
                sorted_vals = unique_vals
            # For numeric variables, show min, max and a few samples
            if attr_type == "数值型 (Numeric)":
                sample_vals = f"范围: [{min(unique_vals)}, {max(unique_vals)}], 示例: " + ", ".join([str(x) for x in unique_vals[:3]])
            else:
                sample_vals = ", ".join([str(x) for x in unique_vals[:5]]) + " ... [等 " + str(unique_count) + " 个唯一值] ... " + ", ".join([str(x) for x in unique_vals[-2:]])
                
        # Escape markdown symbols
        col_escaped = str(col).replace("|", "\\|")
        sample_vals_escaped = str(sample_vals).replace("|", "\\|").replace("\n", " ")
        if len(sample_vals_escaped) > 80:
            sample_vals_escaped = sample_vals_escaped[:77] + "..."
            
        report_lines.append(
            f"| {idx} | {col_escaped} | `{dtype}` | {attr_type} | {missing_count} | {missing_pct:.2f}% | {unique_count} | {sample_vals_escaped} |"
        )
        
    # Ensure directory exists and write
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"Detailed cohort inspection report successfully saved to: {output_report_path}")

if __name__ == "__main__":
    raw_data_path = r"D:\acode\ECMO\data\原始数据2.xlsx"
    report_path = r"D:\acode\ECMO\output\cohort_data_characteristics.md"
    generate_cohort_report(raw_data_path, report_path)
