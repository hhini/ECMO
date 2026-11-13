# -*- coding: utf-8 -*-
"""
Comprehensive Chinese Publication Research Report Generator (Revised 9-Step Edition)
Reads live outputs from:
  - output/02_table1_baseline/Table1_Baseline_Balance_Diagnosis.csv (merged Z / chi2)
  - output/02_table1_baseline/Table2_Univariate_Cross_Cohort_Consistency.csv (merged Z / chi2)
  - output/02_table1_baseline/Table2B_Univariate_Overall_Thrombosis.csv
  - output/revised_pipeline/*.csv (Missingness, VIF, LASSO, 3-Cohort Performance, DeLong)
  - report/sci_figures/fig01..fig13 (300 DPI, pure English)

Produces:
  - report/ECMO_Thrombosis_Full_Research_Report.docx
  - report/ECMO_Thrombosis_Full_Research_Report.md
  - Artifact MD copy
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import numpy as np
import pandas as pd

BASE_DIR = r"D:\acode\ECMO"
REPORT_DIR = os.path.join(BASE_DIR, "report")
FIG_DIR = os.path.join(REPORT_DIR, "sci_figures")
REV_DIR = os.path.join(BASE_DIR, "output", "revised_pipeline")
T1_DIR = os.path.join(BASE_DIR, "output", "02_table1_baseline")
DOCX_PATH = os.path.join(REPORT_DIR, "ECMO_Thrombosis_Full_Research_Report.docx")
MD_PATH = os.path.join(REPORT_DIR, "ECMO_Thrombosis_Full_Research_Report.md")
ARTIFACT_DIR = r"C:\Users\Administrator\.gemini\antigravity\brain\8135a483-dc36-4da3-997c-c300a510ee39"
ARTIFACT_MD_PATH = os.path.join(ARTIFACT_DIR, "ECMO_Thrombosis_Full_Research_Report.md")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def set_cell_background(cell, fill_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=65, bottom=65, left=80, right=80):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_three_line_borders(table, color="1F4E79", sz="12"):
    tblPr = table._element.xpath('w:tblPr')
    if tblPr:
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CBD5E0"/>
                <w:insideV w:val="none"/>
                <w:left w:val="none"/>
                <w:right w:val="none"/>
            </w:tblBorders>
        ''')
        tblPr[0].append(borders)

def add_custom_table(doc, table_data, title=None, subtitle=None):
    if title:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_before = Pt(12)
        p_t.paragraph_format.space_after = Pt(2)
        p_t.paragraph_format.keep_with_next = True
        r_t = p_t.add_run(title)
        r_t.bold = True
        r_t.font.size = Pt(10.5)
        r_t.font.color.rgb = RGBColor(31, 78, 121)

    if subtitle:
        p_sub = doc.add_paragraph()
        p_sub.paragraph_format.space_before = Pt(0)
        p_sub.paragraph_format.space_after = Pt(4)
        p_sub.paragraph_format.keep_with_next = True
        r_s = p_sub.add_run(subtitle)
        r_s.font.size = Pt(8.5)
        r_s.font.italic = True
        r_s.font.color.rgb = RGBColor(90, 100, 110)

    if not table_data or len(table_data) < 2:
        return

    n_cols = len(table_data[0])
    table = doc.add_table(rows=len(table_data), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line_borders(table)

    for r_idx, row_vals in enumerate(table_data):
        row = table.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_margins(cell)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.size = Pt(8.0 if n_cols >= 7 else 8.5)
            if r_idx == 0:
                set_cell_background(cell, "1F4E79")
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                if r_idx % 2 == 0:
                    set_cell_background(cell, "F8FAFC")
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER

def table_to_markdown(table_data, title=None, subtitle=None):
    lines = []
    if title:
        lines.append(f"\n**{title}**\n")
    if subtitle:
        lines.append(f"*{subtitle}*\n")
    if not table_data:
        return ""
    header = "| " + " | ".join(str(x) for x in table_data[0]) + " |"
    sep = "| " + " | ".join("---" for _ in table_data[0]) + " |"
    lines.append(header)
    lines.append(sep)
    for row in table_data[1:]:
        lines.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    lines.append("\n")
    return "\n".join(lines)

def build_full_report():
    print("Building Revised 9-Step Chinese Research Report (.docx & .md)...")
    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.7)
        section.right_margin = Inches(0.7)

    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Microsoft YaHei'
    style_normal.font.size = Pt(9.5)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("成人体外膜肺氧合（ECMO）下肢血栓形成智能风险预警模型研发、内部交叉验证与独立外部验证全流程学术报告")
    r_title.bold = True
    r_title.font.size = Pt(15.5)
    r_title.font.color.rgb = RGBColor(27, 54, 93)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("基于严格数据隔离划分（186/80/106）、Day-15高缺失剔除与训练集单边插补、VIF共线性筛除、重复5折LASSO精简筛选、代价敏感学习及SHAP可解释性分析")
    r_sub.font.size = Pt(9.5)
    r_sub.font.color.rgb = RGBColor(100, 110, 120)

    md_lines = [
        "# 成人体外膜肺氧合（ECMO）下肢血栓形成智能风险预警模型研发、内部交叉验证与独立外部验证全流程学术报告\n",
        "> **基于严格数据隔离划分（186/80/106）、Day-15高缺失剔除与训练集单边插补、VIF共线性筛除、重复5折LASSO精简筛选、代价敏感学习及SHAP可解释性分析**\n\n---\n"
    ]

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(12.5)
        r.font.color.rgb = RGBColor(31, 78, 121)
        md_lines.append(f"\n# {text}\n")

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.bold = True
        r.font.size = Pt(10.8)
        r.font.color.rgb = RGBColor(43, 84, 126)
        md_lines.append(f"\n## {text}\n")

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.25
        r = p.add_run(text)
        r.font.size = Pt(9.5)
        md_lines.append(f"{text}\n")

    def add_callout(title, text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(7)
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.right_indent = Inches(0.18)
        r1 = p.add_run(f"【{title}】 ")
        r1.bold = True
        r1.font.color.rgb = RGBColor(0, 102, 102)
        r2 = p.add_run(text)
        r2.font.size = Pt(9.0)
        r2.font.color.rgb = RGBColor(55, 65, 81)
        md_lines.append(f"\n> **【{title}】**\n> {text.replace(chr(10), chr(10) + '> ')}\n")

    def add_img(fig_name, caption):
        fpath = os.path.join(FIG_DIR, fig_name)
        if os.path.exists(fpath):
            p_i = doc.add_paragraph()
            p_i.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_i.paragraph_format.space_before = Pt(6)
            p_i.paragraph_format.space_after = Pt(3)
            p_i.paragraph_format.keep_with_next = True
            doc.add_picture(fpath, width=Inches(6.2))
            p_c = doc.add_paragraph()
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_after = Pt(8)
            rc = p_c.add_run(caption)
            rc.font.size = Pt(8.5)
            rc.font.italic = True
            rc.font.color.rgb = RGBColor(95, 105, 115)
            md_lines.append(f"\n![{caption}](sci_figures/{fig_name})\n*{caption}*\n")

    # Load CSVs
    df_t1 = pd.read_csv(os.path.join(T1_DIR, "Table1_Baseline_Balance_Diagnosis.csv"), encoding="utf-8-sig")
    df_t2 = pd.read_csv(os.path.join(T1_DIR, "Table2_Univariate_Cross_Cohort_Consistency.csv"), encoding="utf-8-sig")
    df_t2b = pd.read_csv(os.path.join(T1_DIR, "Table2B_Univariate_Overall_Thrombosis.csv"), encoding="utf-8-sig")
    df_miss = pd.read_csv(os.path.join(REV_DIR, "01_missingness_audit.csv"), encoding="utf-8-sig")
    df_vif = pd.read_csv(os.path.join(REV_DIR, "02_vif_pruning_summary.csv"), encoding="utf-8-sig")
    df_lasso = pd.read_csv(os.path.join(REV_DIR, "03_lasso_selected_features.csv"), encoding="utf-8-sig")
    df_perf = pd.read_csv(os.path.join(REV_DIR, "04_model_performance_all_cohorts.csv"), encoding="utf-8-sig")
    df_delong = pd.read_csv(os.path.join(REV_DIR, "05_delong_pairwise_tests.csv"), encoding="utf-8-sig")

    # =========================================================================
    # 总览流程图
    # =========================================================================
    add_h1("【导言】修订后九步法全流程科研技术架构")
    add_body(
        "遵循临床预测模型国际规范（TRIPOD+AI）与统计学严谨性要求，本研究确立了九步闭环研究流程："
        "① 两阶段队列基线平衡性与单因素血栓关联对比（Z值与χ²值合并呈现）；"
        "② 将 266 例推导队列按分层抽样严格划分为训练集（n=186，血栓 31 例）与内部验证集（n=80，血栓 14 例），"
        "此后内部验证集（n=80）与外部验证集（n=106，血栓 25 例）全程绝对冻结、零接触；"
        "③ 缺失值审计：剔除缺失率 >50% 的第 15 天（15d）指标，其余缺失特征严格仅在训练集（n=186）上拟合中位数插补器；"
        "④ 在训练集上实施方差膨胀因子（VIF）多重共线性检验，逐步剔除 VIF > 10 的强共线变量；"
        "⑤ 将通过 VIF 检验的无共线特征纳入 10 次重复 5 折交叉验证 LASSO 回归（替代复杂的单因素-多因素-Boruta三轨叠加，避免 II 型错误增高及漏筛有效因子），精简得出 6 个独立预测因子；"
        "⑥ 在训练集上采用代价敏感学习（Cost-Sensitive Learning）应对类别不平衡；"
        "⑦ 基于 10 次重复 5 折交叉验证与 1000 次 Bootstrap 重抽样拟合 6 种机器学习模型（有效缓解中小样本方差）；"
        "⑧ 在冻结的内部验证集（n=80）与外部验证集（n=106）上开展全方位性能、校准与决策曲线（DCA）评价，优选出最佳模型（SVM）；"
        "⑨ 基于 LASSO 筛选出的 6 个核心预测因子构建最优模型 SHAP 全局与个体级可解释性归因图谱。"
    )
    add_img("fig01_study_cohort_flowchart.png", "图1. 修订版九步法临床机器学习建模、数据隔离（186/80/106）及双队列验证全流程技术路线图")

    # =========================================================================
    # 第一步：基线对比（含合并后的 Z / χ² 栏）
    # =========================================================================
    add_h1("一、 第一步：基线对比与队列平衡性诊断（Z / χ²值合并呈现）")
    add_body(
        "推导队列（n=266，下肢血栓形成 45 例 [16.92%]，非血栓 221 例 [83.08%]）与独立外部测试队列（n=106，下肢血栓形成 25 例 [23.58%]，非血栓 81 例 [76.42%]）"
        "在人口学四分段年龄（18-40、40-60、60-70、≥70岁）、性别、居住地、术中分层补齐失血量（杜绝 0 mL 伪影）及终点下肢血栓形成发生率上均保持良好均衡性（P ≥ 0.1375～0.99），"
        "并已彻底剔除冠心病史与心源性休克两个强临床共线性冗余变量。下表已按规范将 Z 值与 χ² 值合并至同一统计量列中："
    )

    # Table 1
    t1_header = ["项目", "类别", "训练集 (n=266)", "测试集 (n=106)", "Z / χ²值", "P值"]
    t1_data = [t1_header]
    curr_cat = ""
    for _, r in df_t1.iterrows():
        cat = str(r["领域类别"])
        if cat != curr_cat:
            curr_cat = cat
            t1_data.append([f"【{cat}】", "—", "—", "—", "—", "—"])
        t1_data.append([r["项目"], r["类别"], r["训练集 (n=266)"], r["测试集 (n=106)"], r["Z / χ²值"], r["P值"]])
    add_custom_table(doc, t1_data, "Table 1. 推导队列 (n=266) 与独立外部测试队列 (n=106) 基线特征及平衡性诊断三线表", "[注：Z值（Mann-Whitney U检验）与χ²值（Pearson卡方检验）已合并呈现]")
    md_lines.append(table_to_markdown(t1_data, "Table 1. 推导队列与独立外部测试队列基线特征三线表"))

    # Table 2
    c_tr0 = [c for c in df_t2.columns if "训练集" in c and "非血栓" in c][0]
    c_tr1 = [c for c in df_t2.columns if "训练集" in c and "血栓" in c and "非" not in c][0]
    c_te0 = [c for c in df_t2.columns if "测试集" in c and "非血栓" in c][0]
    c_te1 = [c for c in df_t2.columns if "测试集" in c and "血栓" in c and "非" not in c][0]
    t2_header = ["项目", "类别", c_tr0, c_tr1, "训练集 Z / χ²", "训练集 P值", c_te0, c_te1, "测试集 Z / χ²", "测试集 P值"]
    t2_data = [t2_header]
    curr_cat = ""
    for _, r in df_t2.iterrows():
        cat = str(r["领域类别"])
        if cat != curr_cat:
            curr_cat = cat
            t2_data.append([f"【{cat}】"] + ["—"]*9)
        t2_data.append([r["项目"], r["类别"], r[c_tr0], r[c_tr1], r["训练集 Z / χ²值"], r["训练集 P值"], r[c_te0], r[c_te1], r["测试集 Z / χ²值"], r["测试集 P值"]])
    add_custom_table(doc, t2_data, "Table 2. 两队列下肢血栓形成单因素特征对比及跨队列一致性三线表")
    md_lines.append(table_to_markdown(t2_data, "Table 2. 两队列下肢血栓形成单因素特征对比三线表"))

    # Table 2B
    c_b0 = [c for c in df_t2b.columns if "非血栓" in c][0]
    c_b1 = [c for c in df_t2b.columns if "血栓" in c and "非" not in c][0]
    t2b_header = ["项目", "类别", c_b0, c_b1, "χ²值 / Z值", "P值", "临床病理生理学意义"]
    t2b_data = [t2b_header]
    curr_cat = ""
    for _, r in df_t2b.iterrows():
        cat = str(r["领域类别"])
        if cat != curr_cat:
            curr_cat = cat
            t2b_data.append([f"【{cat}】"] + ["—"]*6)
        t2b_data.append([r["项目"], r["类别"], r[c_b0], r[c_b1], r["χ²值 / Z值"], r["P值"], r["临床病理生理学现实意义与医学参考范围"]])
    add_custom_table(doc, t2b_data, "Table 2B. 全队列 (N=372) 下肢血栓形成单因素分析与病理生理学基准表")
    md_lines.append(table_to_markdown(t2b_data, "Table 2B. 全队列下肢血栓形成单因素分析基准表"))

    # =========================================================================
    # 第二、三步：数据划分（186 / 80 / 106）与缺失值处理
    # =========================================================================
    add_h1("二、 第二与第三步：数据严格隔离划分（186/80/106）、Day-15高缺失剔除与训练集单边插补")
    add_body(
        "1. 数据划分（零泄露隔离）：将 266 例推导队列按结局分层随机划分为训练集（n = 186 例，其中下肢血栓 31 例 [16.67%]，非血栓 155 例）"
        "与内部验证集（n = 80 例，其中下肢血栓 14 例 [17.50%]，非血栓 66 例）。独立外部验证集（n = 106 例，血栓 25 例 [23.58%]）保持绝对独立。"
        "划分完成后，内部验证集（n=80）与外部验证集（n=106）全程处于冻结状态，绝不参与任何缺失值插补参数计算、VIF 筛选、LASSO 特征选择或模型训练。\n"
        "2. 第 15 天（15d）高缺失指标剔除：经缺失率审计，由于多数危重症患者在置管 15 天前已完成撤机或转归，"
        "Day-15 实验室指标呈现极高比例结构性缺失（上机第15天DD 缺失 59.4%、FIB_D15 缺失 66.2%、AT3_D15 缺失 71.4%、PLT_D15 缺失 51.9%，均 >50% 阈值），"
        "强行插补会引入严重人为偏倚，故在建模前予以彻底剔除。\n"
        "3. 训练集单边中位数插补（SimpleImputer, strategy='median'）：对其余缺失率 <35% 的基线及 D0/D2/D7 特征，"
        "采用单变量稳健中位数插补法（仅在 186 例训练集上拟合各特征中位数，随后冻结参数并向内部验证集 80 例与外部验证集 106 例单向映射），"
        "有效抗衡急危重症检验指标的重度偏态极端值干扰。"
    )
    add_img("fig02_missingness_and_imputation.png", "图2. 临床指标缺失率审计与 Day-15 高缺失特征（>50%）剔除决策图")

    # =========================================================================
    # 第四步：VIF 多重共线性检验
    # =========================================================================
    add_h1("三、 第四步：训练集方差膨胀因子（VIF）多重共线性筛除")
    add_body(
        "在 186 例训练集插补后矩阵上实施迭代方差膨胀因子（VIF）共线性诊断，逐步剔除 VIF > 10.0 的高度共线变量。"
        "共精准识别并剔除 4 个强共线冗余特征（Delta_FIB, pH_D0, PLT_D0, AT3_D0），最终保留 59 个彼此统计独立的候选特征（所有保留特征均满足 VIF ≤ 10.0），为后续 LASSO 稀疏筛选奠定稳定基础。"
    )
    vif_removed = df_vif[df_vif["Status"].str.startswith("Removed")]
    vif_tbl = [["变量名称", "剔除前初始 VIF", "剔除后最终 VIF", "VIF 检验处置结论"]]
    for _, r in pd.concat([vif_removed, df_vif[df_vif["Status"].str.startswith("Retained")].head(10)]).iterrows():
        fin_v = f"{r['Final_VIF']:.2f}" if pd.notna(r["Final_VIF"]) else "—"
        init_v = f"{r['Initial_VIF']:.2f}" if np.isfinite(r["Initial_VIF"]) else "> 1000 (完全共线)"
        vif_tbl.append([r["Feature"], init_v, fin_v, r["Status"]])
    add_custom_table(doc, vif_tbl, "表3. 训练集 (n=186) 方差膨胀因子 (VIF) 共线性剔除与保留特征摘要表")
    md_lines.append(table_to_markdown(vif_tbl, "表3. 训练集 VIF 共线性剔除摘要表"))
    add_img("fig03_vif_multicollinearity.png", "图3. 训练集剔除前/后方差膨胀因子（VIF > 10.0 截断线）多重共线性诊断图")

    # =========================================================================
    # 第五步：重复 5 折交叉验证 LASSO 特征筛选
    # =========================================================================
    add_h1("四、 第五步：10 次重复 5 折交叉验证 LASSO 特征筛选")
    add_body(
        "采纳统计学专家意见，摒弃易导致 II 型错误增高及漏筛有效变量的'单因素+多因素'繁琐筛选，以及惩罚收缩力不足的 Boruta 算法；"
        "直接将通过 VIF ≤ 10 检验的 59 个无共线特征全部投入 L1 正则化逻辑回归（LASSO），在 186 例训练集上采用 10 次重复 5 折交叉验证（50 次重抽样）"
        "并结合临床事件数每变量比（EPV ≥ 5）约束，最终精简筛选出 6 个最具独立预测价值的非零系数核心因子："
        "DD_D0（上机前 D-二聚体）、FIB_D2（第2天纤维蛋白原）、ECMO_Duration（ECMO支持时长）、Pre_ECMO_Surgery（ECMO前急诊手术史）、PaO2_D0（上机前动脉氧分压）及 FIB_D0（基线纤维蛋白原）。"
    )
    lasso_tbl = [["筛选入选核心预测因子", "LASSO 标准化回归系数", "绝对权重排序", "临床效应方向"]]
    for idx, (_, r) in enumerate(df_lasso.iterrows(), 1):
        lasso_tbl.append([r["Feature"], f"{r['LASSO_Std_Coef']:+.4f}", f"Top {idx}", r["Direction"]])
    add_custom_table(doc, lasso_tbl, "表4. 10×5折交叉验证 LASSO 最终筛选出的 6 个核心临床预测因子及系数表")
    md_lines.append(table_to_markdown(lasso_tbl, "表4. LASSO 筛选核心临床预测因子表"))
    add_img("fig04_lasso_cv_and_path.png", "图4. LASSO 10×5折交叉验证 AUROC 曲线与 6 个核心预测因子标准化系数柱状图")
    add_img("fig05_selected_features_violin_heatmap.png", "图5. LASSO 筛选出的 6 个预测因子间的 Pearson 相关性热图与血栓/非血栓分布对比图")

    # =========================================================================
    # 第六步：代价敏感学习（仅说明方法与理由，不放表格和结果）
    # =========================================================================
    add_h1("五、 第六步：训练集类别不平衡处理策略（代价敏感学习）")
    add_callout(
        "代价敏感学习（Cost-Sensitive Learning）原理与临床选择依据",
        "在 186 例训练集中，下肢血栓阳性患者仅占 16.67%（31 例 vs 155 例阴性），存在显著类别不平衡。本研究严格在训练集模型拟合阶段启用代价敏感学习"
        "（在 LR、SVM、RF、LGBM 中设置 class_weight='balanced'，在 XGBoost 中设置 scale_pos_weight = N_neg / N_pos = 5.00）。\n"
        "【核心理由】：① 100% 忠于真实世界患者流形：不同于 SMOTE 等合成过采样技术会在模糊临床边界人工捏造脱离生理规律的虚假样本，代价敏感学习不增删任何真实病案；"
        "② 杜绝跨折数据泄露：损失函数内嵌逆频数惩罚权重，完全规避了采样导致的数据泄露与过拟合风险；"
        "③ 精准契合重症漏诊代价：通过成倍放大血栓阳性漏诊（False Negative）的惩罚代价，在维持高特异性的同时显著保障血栓高危患者的临床检出率。"
    )

    # =========================================================================
    # 第七、八步：模型拟合（重复5折+Bootstrap）及内部/外部双队列验证
    # =========================================================================
    add_h1("六、 第七与第八步：重复5折+Bootstrap模型拟合与内部/外部独立验证")
    add_body(
        "基于筛选出的 6 个核心预测因子，在 186 例训练集上采用【10 次重复 5 折交叉验证 + 1000 次 Bootstrap 重抽样】（解决中小样本性能估计方差大问题）"
        "训练并对比 6 种经典机器学习模型（LR、SVM、RF、XGBoost、LightGBM、Naive Bayes），以训练集 OOF Youden 指数锁定最佳临床决策阈值。"
        "随后在完全冻结模型参数的前提下，分别在内部验证集（n=80）与独立外部验证集（n=106）开展双盲态全面评价。"
    )

    perf_tbl = [["评价队列", "模型", "AUROC (95% Bootstrap CI)", "AUPRC", "准确率", "敏感度", "特异性", "F1分数", "Brier分数"]]
    for _, r in df_perf.iterrows():
        perf_tbl.append([
            r["Cohort"], r["Model"],
            f"{r['AUROC']:.3f} [{r['AUROC_CI_Low']:.3f}-{r['AUROC_CI_High']:.3f}]",
            f"{r['AUPRC']:.3f}", f"{r['Accuracy']:.3f}", f"{r['Sensitivity']:.3f}",
            f"{r['Specificity']:.3f}", f"{r['F1']:.3f}", f"{r['Brier']:.3f}"
        ])
    add_custom_table(doc, perf_tbl, "表5. 6 种机器学习模型在训练集 (10×5折CV, n=186)、内部验证集 (n=80) 及外部验证集 (n=106) 全指标对比三线表")
    md_lines.append(table_to_markdown(perf_tbl, "表5. 6种模型三队列全面验证指标对比三线表"))

    delong_tbl = [["验证队列", "最优基准模型", "对比模型", "基准 AUROC", "对比 AUROC", "Z 统计量", "DeLong P 值"]]
    for _, r in df_delong.iterrows():
        delong_tbl.append([
            r["Cohort"], r["Reference_Model"], r["Comparator_Model"],
            f"{r['Ref_AUROC']:.3f}", f"{r['Comp_AUROC']:.3f}", f"{r['Z_stat']:+.3f}", f"{r['P_value']:.4f}"
        ])
    add_custom_table(doc, delong_tbl, "表6. 最优模型 (SVM) 与其余 5 种算法在内部及外部验证集上的成对 DeLong 检验结果表")
    md_lines.append(table_to_markdown(delong_tbl, "表6. 最优模型成对 DeLong 检验表"))

    add_body(
        "【最优模型确立结论】：综合内部验证与独立外部验证的跨队列泛化稳健性，支持向量机（SVM, RBF核）表现最卓越且均衡："
        "训练集 10×5 折交叉验证 AUROC 达 0.940 [0.903–0.971]；在完全冻结的内部验证集（n=80）上 AUROC 达 0.870 [0.772–0.959]（敏感度 78.6%、特异性 80.3%、Brier 0.106）；"
        "在独立外部验证集（n=106）上 AUROC 稳居全模型榜首 0.797 [0.698–0.877]（特异性 80.2%、Brier 0.158），且跨队列衰减幅度最小（显著优于出现轻度过拟合的树模型 RF/XGB/LGBM，外部验证集 DeLong 检验 P < 0.05）；"
        "同时逻辑回归（LR）亦展现稳健基线（内部 AUROC 0.833、外部 AUROC 0.788），共同印证了 LASSO 所选 6 特征的强跨中心泛化能力。"
    )

    add_img("fig06_three_cohort_roc_pr_curves.png", "图6. 6 种模型在训练集 (n=186 Rep-5CV)、内部验证集 (n=80) 及外部验证集 (n=106) 的 ROC 与 PR 曲线")
    add_img("fig07_confusion_matrices_internal_external.png", "图7. 6 种模型在内部验证集 (n=80) 与外部验证集 (n=106) 的混淆矩阵（基于训练集锁定 Youden 阈值）")
    add_img("fig08_model_performance_forest_comparison.png", "图8. 内部与外部验证集 AUROC (1000次 Bootstrap 95% CI) 森林图及跨队列泛化柱状对比")
    add_img("fig09_calibration_curves.png", "图9. 内部验证集 (n=80) 与外部验证集 (n=106) 模型概率校准可靠性曲线与 Brier 评分")
    add_img("fig10_dca_clinical_benefit.png", "图10. 内部验证集 (n=80) 与外部验证集 (n=106) 临床决策曲线分析（DCA 净获益）")

    # =========================================================================
    # 第九步：SHAP 可解释性分析
    # =========================================================================
    add_h1("七、 第九步：基于 LASSO 筛选特征的最优模型（SVM）SHAP 可解释性剖析")
    add_body(
        "为彻底打开机器学习'黑盒'，本研究基于 LASSO 筛选出的 6 个核心临床预测因子（DD_D0, FIB_D2, ECMO_Duration, Pre_ECMO_Surgery, PaO2_D0, FIB_D0），"
        "对最优模型 SVM 构建了 SHAP（SHapley Additive exPlanations）全局与个体级归因解释体系："
        "全局蜂群图与重要性排序表明，上机前基础 D-二聚体（DD_D0）、ECMO 运行第 2 天纤维蛋白原激增水平（FIB_D2）以及 ECMO 辅助支持总时长（ECMO_Duration）"
        "是驱动下肢血栓形成的三大核心支柱因子；个体级瀑布图可为临床医师直观呈现单个患者预测风险的定量正/负贡献明细，赋能床旁精准抗凝与超声筛查决策。"
    )
    add_img("fig11_shap_summary_panel.png", "图11. 最优模型 (SVM) 全局 SHAP 蜂群分布图与 6 大 LASSO 核心预测因子平均绝对贡献排序")
    add_img("fig12_shap_dependence_plots.png", "图12. Top 4 核心临床预测因子与下肢血栓风险的非线性 SHAP 依赖关系图")
    add_img("fig13_shap_individual_waterfalls.png", "图13. 典型个体化临床案例微观 SHAP 瀑布归因图：(A) 高危 LET+ 患者 vs (B) 低危非血栓患者")

    doc.save(DOCX_PATH)
    print(f"[SUCCESS] Word report saved: {DOCX_PATH} ({os.path.getsize(DOCX_PATH):,} bytes)")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    with open(ARTIFACT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[SUCCESS] Markdown report saved: {MD_PATH}")

if __name__ == "__main__":
    build_full_report()
