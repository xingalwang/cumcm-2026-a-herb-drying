
from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
# MODEL_ROOT 可指定同样包含 problems/ 和 results/ 的仓库根目录。
_root_override = os.environ.get("MODEL_ROOT")
ROOT = (Path(_root_override).expanduser().resolve()
        if _root_override else CODE_DIR.parent)
INPUT_DATA = ROOT / "problems" / "附件"
RESULTS_DIR = ROOT / "results"
# 沿用求解器中的变量名；JSON 结果与 Excel 统一保存到 results/。
FIGDIR = RESULTS_DIR
TMPDIR = ROOT / "_tmp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMPDIR.mkdir(parents=True, exist_ok=True)

KELVIN_OFFSET = 273.15
INDEX_HEADER = "时间\\到药材中心的距离"


R0 = 0.02
R0_CM = 2.0
LENGTH_CM = 4.0
HALF_LENGTH_M = LENGTH_CM / 100.0 / 2.0
T0_C = 28.0
C0 = 2.55
C_THRESHOLD = 0.15
DECIMALS = 4
DURATION_DAYS_LOW = 2.0
DURATION_DAYS_HIGH = 3.0


# 附录 2
RHO_APP2 = 820.0
CP_APP2 = 2600.0
K_APP2 = 0.36
H_CONV = 25.0
HM_MASS = 8.0e-7
D0_APP2 = 7.0e-9
A_APP2 = 0.89


RHO_A3, RHO_B3 = 650.0, 128.0
CP_A3, CP_B3 = 1450.0, 2736.0
K_A3, K_B3 = 0.21, 0.38
D0_APP3 = 2.4e-3
A_APP3 = 0.45
B_ARRH3 = 3850.0


RHO_A4, RHO_B4 = 760.0, 90.0
CP_A4, CP_B4 = 1850.0, 2150.0
K_A4, K_B4 = 0.12, 0.20
D0_APP4 = 4.2e-4
A_APP4 = 0.30
B_ARRH4 = 3850.0


# 输出时刻和空间列
Q1_TABLE_TIMES_S = [100.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0]
Q1_TABLE_RADII_CM = [0.0, 0.5, 1.0, 1.5, 2.0]
Q1_SPAN_S = 1800.0
Q1_DT_OUT = 1.0
Q2_TABLE_TIMES_H = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
Q2_TABLE_RADII_CM = [0.0, 0.5, 1.0, 1.5, 2.0]
Q2_SPAN_S = 10800.0
Q2_DT_OUT = 1.0
Q3_DT_OUT = 60.0
Q3_TABLE_DT_H = 6.0
Q4_DT_OUT = 60.0
Q4_TABLE_DT_H = 6.0
DR_OUT_CM = 0.1
SHEETS_TC = ["温度", "水分浓度"]
SURFACE_COL_NAME = "药材表面"


N_CELLS = 200
DR_SOLVE = R0 / N_CELLS
OUT_STRIDE_SPACE = int(round((DR_OUT_CM / 100.0) / DR_SOLVE))
# 主计算步长
DT_SOLVE_Q12 = 0.25
DT_SOLVE_Q34 = 2.0
PICARD_TOL_C = 1.0e-10
PICARD_TOL_T = 1.0e-8
PICARD_MAX = 50
N_CONVERGENCE = (100, 200, 400, 800)
DT_CONVERGENCE_Q12 = (1.0, 0.5, 0.25, 0.125)
DT_CONVERGENCE_Q34 = (8.0, 4.0, 2.0, 1.0)
T_MAX_CAP_S = 120.0 * 3600.0
BISECT_TOL_S = 0.2
REBOUND_FACTOR = 1.05


INCLUDE_LATENT = False
MASS_FORM = "C"
AIR_EXTRAP_MODE = "hold"
AIR_EXTRAP_T_C = 50.0
AIR_EXTRAP_C = 0.05


SENS_T_AIR_C = (49.0, 50.0, 51.0)
SENS_C_AIR = (0.045, 0.05, 0.055)
SENS_H_FACTORS = (0.5, 1.0, 1.5)
SENS_HM_FACTORS = (0.5, 1.0, 1.5)
SENS_DT_SOLVE_S = 4.0
SENS_DT_OUT_S = 3600.0


AIR_XLSX = INPUT_DATA / "附件1.xlsx"
RADIUS_XLSX = INPUT_DATA / "附件2.xlsx"
# 输入表头
AIR_COLS = ["时间", "温度", "水分浓度"]
RADIUS_COLS = ["时间", "半径"]
AIR_T_END_S = 14400.0
RADIUS_T_END_S = 259200.0


FACTS = {
    "geometry": {"radius_cm": R0_CM, "length_cm": LENGTH_CM},
    "initial_conditions": {"temperature_C": T0_C, "moisture_kgkg": C0},
    "criteria": {"moisture_threshold_kgkg": C_THRESHOLD, "decimals": DECIMALS},
}
DATA_FACTS = {
    "air_file": AIR_XLSX.name,
    "radius_file": RADIUS_XLSX.name,
    "air_measured_until_s": AIR_T_END_S,
    "radius_measured_until_s": RADIUS_T_END_S,
}

assert abs(R0 - 0.02) < 1.0e-15
assert abs(C_THRESHOLD - 0.15) < 1.0e-15
assert abs(T0_C - 28.0) < 1.0e-12
assert abs(D0_APP2 - 7.0e-9) < 1.0e-24
assert abs(D0_APP3 - 2.4e-3) < 1.0e-18
assert abs(D0_APP4 - 4.2e-4) < 1.0e-19
assert OUT_STRIDE_SPACE == 10
assert abs(DR_SOLVE - 1.0e-4) < 1.0e-18
assert AIR_EXTRAP_C < C_THRESHOLD
for _out_dt, _solve_dt in (
    (Q1_DT_OUT, DT_SOLVE_Q12),
    (Q2_DT_OUT, DT_SOLVE_Q12),
    (Q3_DT_OUT, DT_SOLVE_Q34),
    (Q4_DT_OUT, DT_SOLVE_Q34),
):
    assert abs(_out_dt / _solve_dt - round(_out_dt / _solve_dt)) < 1.0e-12

__all__ = [
    name for name in dir()
    if name.isupper() or name in ("FACTS", "DATA_FACTS", "ROOT", "INPUT_DATA", "RESULTS_DIR", "FIGDIR", "TMPDIR")
]
