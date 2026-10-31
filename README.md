# 药材烘干问题：论文与计算代码

论文《基于热质耦合退化扩散方程与 ALE 动边界方法的圆柱药材热风烘干模型》的 PDF、赛题文件、求解代码和已有计算结果。

## 文件结构

```text
.
├── paper.pdf                 # 论文 PDF，含附录
├── problems/                 # 完整赛题原文件
│   ├── A题.pdf
│   ├── A题.txt
│   └── 附件/
│       ├── 附件1.xlsx        # 烘房温度、含水率输入
│       ├── 附件2.xlsx        # 半径变化输入
│       └── 附件3/            # result1—result4 原始空白模板
├── codes/
│   ├── main.py               # 主入口
│   ├── problem1.py           # 问题一
│   ├── problem2.py           # 问题二
│   ├── problem3.py           # 问题三
│   ├── problem4.py           # 问题四
│   ├── params.py             # 参数和文件路径
│   ├── props.py              # 物性公式
│   ├── utils.py              # 公共求解函数、读写函数
│   ├── structural_tests.py   # 结构性检验
│   ├── sensitivity_analysis.py
│   ├── global_q4_analysis.py # 问题四扩展分析
│   └── ansys/                # ANSYS APDL 输入文件（.dat）
├── results/
│   ├── result1.xlsx          # 已计算的四问结果
│   ├── result2.xlsx
│   ├── result3.xlsx
│   ├── result4.xlsx
│   ├── all_results.json      # 跨问汇总
│   ├── problem_*_results.json
│   ├── structural_tests.json
│   ├── sensitivity_results.json
│   └── ansys/                # 已有仿真指标和结果表
├── requirements.txt
├── LICENSE
└── .gitignore
```

`problems/附件/附件3` 中是赛题提供的模板，`results/` 中是已填写的计算结果。求解器直接读取 `problems/附件/附件1.xlsx` 和 `附件2.xlsx`，不需要预先处理数据。

## 运行方法

建议使用 Python 3.11 或更新版本。整理时在 Python 3.12.3、NumPy 2.5.4、SciPy 1.18.1、pandas 3.1.0、openpyxl 3.1.5 环境中完成了模块导入、附件读取、缓存汇总、小规模求解和 Excel 读写检查；这不代表全部实验已经重新计算。

在仓库根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行主程序：

```bash
python codes/main.py
```

主程序依次处理结构性检验、问题一至四和灵敏度分析。仓库附带的 JSON 通过程序内检查时会被复用；这次运行不等于重新完成全部计算。

忽略缓存、重新计算全部阶段：

```bash
python codes/main.py --force
```

完整重算包含网格、时间步和灵敏度分析，耗时随设备而异；会覆盖 `results/` 下对应的 Excel 和 JSON。运行中的临时数据写入 `_tmp/`。输入附件及原始结果模板不会被覆盖。

也可以单独运行某一问，例如：

```bash
python codes/problem1.py
```

问题四的部分对照读取问题三的结果。按主入口顺序运行即可。

扩展分析为可选项：

```bash
python codes/main.py --analysis-global --quick
```

上述扩展分析的输出位于 `results/enhanced_q4/`，其中包含该模块自身生成的图。本仓库未收录独立的论文绘图脚本。

## 主要结果

以下为随仓库提供的历史计算结果，来自 `results/all_results.json`：

| 指标 | 结果 |
| --- | --- |
| 问题一：1800 s 中心温度 | 33.5756 °C |
| 问题二：3 h 平均含水率 | 1.3825 kg/kg |
| 问题三：固定几何烘干时长 | 57.4745 h |
| 问题四：收缩模型烘干时长 | 51.0888 h |

问题三、四使用全场最大含水率低于 0.15 kg/kg 的终点判据。模型假设、适用范围及数值验证见论文。

## ANSYS 文件

`codes/ansys/` 保留提交材料中的 11 个 APDL 输入文件，覆盖问题一的三维温度与含水率、问题二和三的二维轴对称模型，以及问题四的七个含水率状态。需要另行安装有相应使用权限的 ANSYS MAPDL；原仿真环境记录为 ANSYS 2026 R1。

各输入文件应在独立工作目录运行，避免同名输出相互覆盖。问题四文件以给定时刻状态为初值继续推进 1 s，用于局部快照复核，不代表连续三维动边界全程求解。附带结果见 `results/ansys/`。

## 材料与授权

- 论文 PDF、Python 求解代码和主要计算结果来自原提交材料；赛题及附件按现有原文件保留。
- 整理版仅调整 Python 中的输入、输出目录及相应路径提示，计算公式和求解逻辑沿用原文件。
- 原创代码（`codes/`）采用 [MIT License](LICENSE)，可使用、修改和分发，需保留许可证要求的版权及许可声明。
- MIT 许可不扩展至赛题、赛题数据、论文或其他第三方材料；这些内容的权利和授权分别归其权利人。仓库附带赛题不构成对再分发权限的独立确认。

论文以 `paper.pdf` 为准。本次整理未重新计算全部结果，也未重新运行 ANSYS。
