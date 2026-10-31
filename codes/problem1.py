
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from params import (C0, DECIMALS, DT_SOLVE_Q12, FIGDIR, H_CONV, HM_MASS, KELVIN_OFFSET,
                    N_CELLS, OUT_STRIDE_SPACE, Q1_DT_OUT, Q1_SPAN_S, Q1_TABLE_RADII_CM,
                    Q1_TABLE_TIMES_S, R0, SHEETS_TC, T0_C, TMPDIR, RESULTS_DIR,
                    N_CONVERGENCE, DT_CONVERGENCE_Q12)
from props import D_app2, props_app2
import inspect
from utils import (air_driving, convergence_ratios, duhamel_T, extract_output_nodes,
                   grid_weights, make_field_df, output_columns_cm, solve_drying,
                   volume_average, write_xlsx)

TAG = "problem_1"


def run():
    t0_wall = time.time()
    # 先算无量纲数，再推进温度和水分。
    rho, cp, k = props_app2()
    alpha = k / (rho * cp)
    Bi_T = H_CONV * R0 / k
    D_C0 = float(D_app2(C0))
    Bi_m = HM_MASS * R0 / D_C0
    out = {"problem": 1, "property_set": "appendix2", "coupling": "decoupled",
           "rho_kg_m3": rho, "cp_J_kgK": cp, "k_W_mK": k,
           "alpha_m2_s": alpha, "Bi_T": Bi_T, "D_at_C0_m2_s": D_C0, "Bi_m": Bi_m,
           "Fo_m_1800s": D_C0 * 1800.0 / R0 ** 2, "Le_alpha_over_D": alpha / D_C0,
           "diffusion_time_scale_s": R0 ** 2 / alpha}
    print(f"[Q1] alpha={alpha:.6e} Bi_T={Bi_T:.7f} D(C0)={D_C0:.6e} Bi_m={Bi_m:.6f} "
          f"Le={alpha / D_C0:.4f}", flush=True)


    sig = str(inspect.signature(D_app2))
    assert "T" not in sig, f"C-P1-2 违反：D_app2 签名 {sig} 含温度参数"
    out["D_app2_signature"] = sig


    # 主解
    print(f"[Q1] 主求解 N={N_CELLS} dt={DT_SOLVE_Q12} span={Q1_SPAN_S} s", flush=True)
    res = solve_drying("app2", Q1_SPAN_S, stop=("time", Q1_SPAN_S), dt_solve=DT_SOLVE_Q12,
                       dt_out=Q1_DT_OUT, N=N_CELLS, air_mode="linear",
                       C_init=C0, T_init=T0_C, record_fields=True)
    print(f"[Q1] 主求解完成 {res['n_steps']} 步 用时 {time.time() - t0_wall:.1f} s", flush=True)
    r_nodes, dr, V, A_in = grid_weights(R0, N_CELLS)
    T_field = extract_output_nodes(res["T_out"], OUT_STRIDE_SPACE)
    C_field = extract_output_nodes(res["C_out"], OUT_STRIDE_SPACE)
    t_out = res["t_out"]
    assert T_field.shape == (1800, 21), f"输出形状不符：{T_field.shape}"
    assert abs(t_out[0] - 1.0) < 1e-12 and abs(t_out[-1] - 1800.0) < 1e-12, "输出时刻不符"

    out["n_steps"] = res["n_steps"]
    out["dt_solve_s"] = DT_SOLVE_Q12
    out["dt_out_s"] = Q1_DT_OUT
    out["N_cells"] = N_CELLS
    out["surface_form"] = res["surface_form"]
    out["water_resid_max"] = res["water_resid_max"]
    out["n_picard_max"] = res["n_picard_max"]
    out["argmax_center_strict"] = bool(res["argmax_center_always"])
    out["argmax_center_within_tol"] = bool(res["argmax_center_within_tol"])
    out["C_hump_excess_max"] = res["C_hump_excess_max"]


    # 解耦时交换方程顺序，结果应一致。
    res_swap = solve_drying("app2", Q1_SPAN_S, stop=("time", Q1_SPAN_S),
                            dt_solve=DT_SOLVE_Q12, dt_out=Q1_SPAN_S, N=N_CELLS,
                            air_mode="linear", C_init=C0, T_init=T0_C,
                            solve_order="T_first", record_fields=False)
    dT_swap = float(np.max(np.abs(res_swap["T_final"] - res["T_final"])))
    dC_swap = float(np.max(np.abs(res_swap["C_final"] - res["C_final"])))
    out["decoupling_swap_order_max_dT_K"] = dT_swap
    out["decoupling_swap_order_max_dC"] = dC_swap
    assert dT_swap < 1e-12 and dC_swap < 1e-12,\
        f"P1-C1 解耦性失败：换序后 dT={dT_swap:.3e} dC={dC_swap:.3e}"
    print(f"[Q1] 换序一致性 dT={dT_swap:.3e} dC={dC_swap:.3e}", flush=True)


    # 用 Duhamel 解独立核对温度场。
    T_air_of, C_air_of = air_driving()
    idx_check = [0, 100, 200]
    T_duh = duhamel_T(r_nodes[idx_check], 1800.0, Bi_T, alpha, R0, T0_C, T_air_of,
                      n_terms=60, n_sub=7200)
    T_num = res["T_final"][idx_check]
    duh_err = np.abs(np.asarray(T_duh) - T_num)
    out["duhamel_check"] = {
        "r_cm": [0.0, 1.0, 2.0],
        "T_numeric_C": [float(x) for x in T_num],
        "T_duhamel_C": [float(x) for x in T_duh],
        "max_abs_err_K": float(np.max(duh_err))}
    print(f"[Q1] Duhamel 核对 num={[f'{x:.4f}' for x in T_num]} "
          f"duh={[f'{x:.4f}' for x in T_duh]} maxerr={np.max(duh_err):.4e}", flush=True)
    assert float(np.max(duh_err)) < 5e-2, f"Duhamel 核对超差 {np.max(duh_err):.4e} K"


    T18 = T_field[-1]
    C18 = C_field[-1]
    ref_T = {0.0: 33.5766, 1.0: 34.3653, 2.0: 36.7863}
    ref_C = {0.0: 2.5500, 1.0: 2.5383, 2.0: 1.5103}
    col_cm = output_columns_cm(21, 0.1)
    got_T = {c: float(T18[col_cm.index(c)]) for c in ref_T}
    got_C = {c: float(C18[col_cm.index(c)]) for c in ref_C}
    out["t1800_temperature_C"] = got_T
    out["t1800_moisture_kgkg"] = got_C
    out["t1800_ref_temperature_C"] = ref_T
    out["t1800_ref_moisture_kgkg"] = ref_C
    out["t1800_dev_temperature_K"] = {str(c): abs(got_T[c] - ref_T[c]) for c in ref_T}
    out["t1800_dev_moisture"] = {str(c): abs(got_C[c] - ref_C[c]) for c in ref_C}
    print(f"[Q1] t=1800s T: {got_T}", flush=True)
    print(f"[Q1] t=1800s C: {got_C}", flush=True)


    out["C_center_1800_rounded"] = round(got_C[0.0], DECIMALS)
    out["C_center_unchanged_to_4dp"] = bool(round(got_C[0.0], DECIMALS) == round(C0, DECIMALS))
    out["T_air_1800_C"] = float(T_air_of(1800.0))
    out["C_air_1800"] = float(C_air_of(1800.0))
    out["C_bar_1800"] = volume_average(res["C_out"][-1], V)
    out["T_center_history_sample"] = {
        str(int(tt)): float(res["T_out"][int(tt) - 1][0]) for tt in (100, 600, 1200, 1800)}


    # 收敛性
    print("[Q1] 网格收敛 N=100/200/400/800 ...", flush=True)
    gv_T, gv_C = [], []
    for Nc in N_CONVERGENCE:
        rc = solve_drying("app2", Q1_SPAN_S, stop=("time", Q1_SPAN_S), dt_solve=DT_SOLVE_Q12,
                          dt_out=Q1_SPAN_S, N=Nc, air_mode="linear", C_init=C0,
                          T_init=T0_C, record_fields=False)
        _, _, Vc, _ = grid_weights(R0, Nc)
        gv_T.append(float(rc["T_final"][0]))
        gv_C.append(volume_average(rc["C_final"], Vc))
        print(f"  N={Nc}: T(0,1800)={gv_T[-1]:.8f} Cbar={gv_C[-1]:.10f}", flush=True)
    print("[Q1] 步长收敛 dt=1/0.5/0.25/0.125 ...", flush=True)
    dv_T, dv_C = [], []
    for dtc in DT_CONVERGENCE_Q12:
        rc = solve_drying("app2", Q1_SPAN_S, stop=("time", Q1_SPAN_S), dt_solve=dtc,
                          dt_out=Q1_SPAN_S, N=N_CELLS, air_mode="linear", C_init=C0,
                          T_init=T0_C, record_fields=False)
        dv_T.append(float(rc["T_final"][0]))
        dv_C.append(volume_average(rc["C_final"], V))
        print(f"  dt={dtc}: T(0,1800)={dv_T[-1]:.8f} Cbar={dv_C[-1]:.10f}", flush=True)
    out["convergence"] = {
        "N_list": list(N_CONVERGENCE), "dt_list": list(DT_CONVERGENCE_Q12),
        "grid_T_center": convergence_ratios(gv_T), "grid_C_bar": convergence_ratios(gv_C),
        "dt_T_center": convergence_ratios(dv_T), "dt_C_bar": convergence_ratios(dv_C)}
    for kk in ("grid_T_center", "grid_C_bar", "dt_T_center", "dt_C_bar"):
        rc_ = out["convergence"][kk]["rel_change"][-1]
        assert abs(rc_) < 5e-3, f"{kk} 末档相对变化 {rc_:.3e} 未收敛"


    tab_ti = [int(round(x)) for x in Q1_TABLE_TIMES_S]
    tab_ci = [col_cm.index(round(x, 1)) for x in Q1_TABLE_RADII_CM]
    tabT = {str(tt): {str(Q1_TABLE_RADII_CM[j]): float(T_field[tt - 1][ci])
                      for j, ci in enumerate(tab_ci)} for tt in tab_ti}
    tabC = {str(tt): {str(Q1_TABLE_RADII_CM[j]): float(C_field[tt - 1][ci])
                      for j, ci in enumerate(tab_ci)} for tt in tab_ti}
    out["paper_table_temperature_C"] = tabT
    out["paper_table_moisture_kgkg"] = tabC

    # 写表并回读检查。
    dfT = make_field_df(t_out, col_cm, T_field)
    dfC = make_field_df(t_out, col_cm, C_field)
    xlsx = RESULTS_DIR / "result1.xlsx"
    write_xlsx(xlsx, {SHEETS_TC[0]: dfT, SHEETS_TC[1]: dfC})
    out["result_file"] = str(xlsx.name)
    out["result_shape"] = [int(dfT.shape[0]), int(dfT.shape[1])]
    print(f"[Q1] 写出 {xlsx.name} sheets={SHEETS_TC} shape={dfT.shape}", flush=True)


    chk = pd.read_excel(xlsx, sheet_name=SHEETS_TC[0])
    worst = 0.0
    for tt in tab_ti:
        for j, ci in enumerate(tab_ci):
            a = float(chk.iloc[tt - 1, ci + 1])
            b = round(tabT[str(tt)][str(Q1_TABLE_RADII_CM[j])], DECIMALS)
            worst = max(worst, abs(a - b))
    out["table_vs_file_max_abs_diff"] = worst
    assert worst == 0.0, f"表格与文件不一致 {worst}"

    out["wall_time_s"] = time.time() - t0_wall
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / f"{TAG}_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[Q1] -> results/{TAG}_results.json ({out['wall_time_s']:.1f} s)", flush=True)
    return out


if __name__ == "__main__":
    run()
