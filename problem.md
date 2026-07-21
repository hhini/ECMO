I will inspect the preprocessing script `scripts/01_preprocess.py` to identify the input raw data paths and see if missingness analysis is performed there.

Viewed 01_preprocess.py:1-60

I will run a python command to calculate the exact missingness rates of all columns in the raw dataset `data2.csv`.

Ran command: `D:\anaconda\python.exe -c "import pandas as pd; df = pd.read_csv(r'D:\acode\ECMO\data\data2.csv'); missing = df.isna().mean().sort_values(ascending=False); print(missing[missing > 0.05].to_string())"`

I will view the translations section of `scripts/01_preprocess.py` to understand the mapping of Chinese columns to English names.

Viewed 01_preprocess.py:60-120

I will run a python command to load the raw data with the correct Chinese encoding and print out the missingness rates of all columns in a readable format.

Ran command: `D:\anaconda\python.exe -c "import pandas as pd; df = pd.read_csv(r'D:\acode\ECMO\data\data2.csv', encoding='utf-8-sig'); missing = df.isna().mean().sort_values(ascending=False); [print(f'{col}: {val:.2%}') for col, val in missing.items() if val > 0.05]"`

I will run the command with Chinese GB18030 encoding to resolve the garbled characters and output the exact clinical column names.

Ran command: `D:\anaconda\python.exe -c "import pandas as pd; df = pd.read_csv(r'D:\acode\ECMO\data\data2.csv', encoding='gb18030'); missing = df.isna().mean().sort_values(ascending=False); [print(f'{col}: {val:.2%}') for col, val in missing.items() if val > 0.05]"`

I will write the missingness results to a text file with UTF-8 encoding so we can view all Chinese and English clinical column names correctly without console encoding issues.

Ran command: `D:\anaconda\python.exe -c "import pandas as pd, os; os.makedirs(r'D:\acode\ECMO\scratch', exist_ok=True); df = pd.read_csv(r'D:\acode\ECMO\data\data2.csv', encoding='utf-8-sig'); missing = df.isna().mean().sort_values(ascending=False); f = open(r'D:\acode\ECMO\scratch\missingness.txt', 'w', encoding='utf-8'); [f.write(f'{col}: {val:.2%}\n') for col, val in missing.items() if val > 0.0]"`

I will now view the generated `missingness.txt` file to read the exact missingness rates.

Viewed missingness.txt:1-110

根据对原始数据集 `data2.csv` 的直接统计，以下是早期时间点（上机前 Day 0 和 上机第二天 Day 2）各主要临床特征的**原始缺失率统计**。

这些高缺失率直接导致了多重插补（MICE）过度平滑个体差异，从而稀释了预测信号。

---

### 一、 早期时间点缺失率极高（$>20\%$）的特征

这部分特征在临床上属于“高缺失度特征”，即便插补也存在很大的不确定性：

| 中文特征名 | 对应英文名 | 时间点 (时间后缀) | 原始缺失率 | 临床解释 |
| :--- | :--- | :---: | :---: | :--- |
| **CRP** | C-Reactive Protein | Day 2 (`CRP.1`) | **57.89%** | C反应蛋白。代表全身急性炎症反应，但由于缺失过半被我们**硬性剔除**。 |
| **CRP** | C-Reactive Protein | Day 0 (`CRP`) | **56.02%** | 上机前 C反应蛋白（同上，剔除）。 |
| **PCT** | Procalcitonin | Day 0 (`PCT`) | **28.57%** | 降钙素原。重症感染指标。 |
| **PCT** | Procalcitonin | Day 2 (`PCT.1`) | **25.94%** | Day 2 降钙素原。 |
| **白蛋白** | Albumin | Day 0 (无后缀) | **30.45%** | 基线白蛋白，反映病患的基线营养与渗透压状态。 |
| **谷丙转氨酶** | ALT | Day 0 (无后缀) | **22.93%** | 基线谷丙转氨酶，反映肝脏细胞储备。 |
| **谷草转氨酶** | AST | Day 0 (无后缀) | **22.56%** | 基线谷草转氨酶。 |
| **二氧化碳分压** | PaCO2 | Day 0 (无后缀) | **19.55%** | 基线血气分析指标，指示无氧代谢与通气储备。 |
| **氧分压/pH** | PaO2 / pH | Day 0 (无后缀) | **19.17%** | 基线氧分压与酸碱度。 |

---

### 二、 早期时间点中度缺失（$10\% \sim 20\%$）的特征

这部分特征有一定程度的缺失，虽然可以用插补找回，但会引入不可避免的统计噪声：

| 中文特征名 | 对应英文名 | 时间点 (时间后缀) | 原始缺失率 | 临床解释 |
| :--- | :--- | :---: | :---: | :--- |
| **AT3** | Antithrombin III | Day 0 (无后缀) | **18.05%** | 抗凝血酶 III（肝素抗凝的基础活性物）。 |
| **AT3** | Antithrombin III | Day 2 (`AT3.1`) | **14.29%** | Day 2 抗凝血酶 III。 |
| **白蛋白** | Albumin | Day 2 (`白蛋白.1`) | **15.41%** | Day 2 白蛋白。 |
| **谷丙转氨酶** | ALT | Day 2 (`谷丙.1`) | **12.03%** | Day 2 谷丙转氨酶。 |
| **谷草转氨酶** | AST | Day 2 (`谷草.1`) | **11.65%** | Day 2 谷草转氨酶。 |
| **肌酐** | Creatinine | Day 0 (无后缀) | **10.15%** | 基线肌酐，反映肾功能水平。 |

---

### 三、 中后期时间点（Day 7 / Day 15）的“极高缺失”（参考）
另外，如果引入更晚的时间点，缺失率会发生断崖式上升（因为部分病患在 Day 7 或 Day 15 前已经撤机或出院，产生大量**非随机缺失**）：
*   `CRP.4` (Day 15 CRP): **81.58%** 缺失
*   `AT3.4` (Day 15 抗凝血酶): **71.43%** 缺失
*   `TT.4` / `APTT.4` / `FIB.4` (Day 15 凝血指标): **66.17% ~ 66.92%** 缺失
*   `pH.4` / `氧分压.4` / `二氧化碳分压.4` (Day 15 血气): **63.91%** 缺失