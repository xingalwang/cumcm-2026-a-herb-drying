
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from params import (C0, C_THRESHOLD, DECIMALS, DT_CONVERGENCE_Q12, DT_SOLVE_Q12, FIGDIR,
                    H_CONV, HM_MASS, KELVIN_OFFSET, N_CELLS, N_CONVERGENCE,
                    OUT_STRIDE_SPACE, Q2_DT_OUT, Q2_SPAN_S, Q2_TABLE_RADII_CM,
                    Q2_TABLE_TIMES_H, R0, SHEETS_TC, T0_C, RESULTS_DIR)
from props import D_app2, D_app3, props_app3, arrhenius
from utils import (air_driving, convergence_ratios, extract_output_nodes, grid_weights,
                   make_field_df, output_columns_cm, solve_drying, volume_average,
                   write_xlsx)

TAG = "problem_2"


def run():
    t0_wall = time.time()
    out = {"problem": 2, "property_set": "appendix3", "coupling": "two_way"}


    # 物性和温度因子检查
    r0_, c0_, k0_ = props_app3(C0)
    r_thr, c_thr, k_thr = props_app3(C_THRESHOLD)
    out["props_at_C0"] = {"rho": float(r0_), "cp": float(c0_), "k": float(k0_)}
    out["props_at_Cthr"] = {"rho": float(r_thr), "cp": float(c_thr), "k": float(k_thr)}
    out["cp_drop_pct"] = float((c0_ - c_thr) / c0_ * 100.0)
    out["k_drop_pct"] = float((k0_ - k_thr) / k0_ * 100.0)
    assert abs(float(c0_) - 3415.2958) < 1e-3, f"cp(2.55) 应为 3415.2958，得 {float(c0_):.4f}"
    assert abs(float(k0_) - 0.4829577) < 1e-6, f"k(2.55) 应为 0.4829577，得 {float(k0_):.7f}"
    out["capability_checklist_deviation"] = {
        "P2-C1(a)_expects_cp": 3416.6, "corrected_cp": float(c0_),
        "P2-C1(a)_expects_k": 0.4832, "corrected_k": float(k0_),
        "note": "CAPABILITY_CHECKLIST 携带 §0 修正前的上游值；第二项自变量应为 C/(C+1)=51/71"}
    print(f"[Q2] props@C0 rho={float(r0_):.4f} cp={float(c0_):.4f} k={float(k0_):.7f}", flush=True)


    f28 = float(arrhenius(T0_C + KELVIN_OFFSET, 3850.0))
    f50 = float(arrhenius(50.0 + KELVIN_OFFSET, 3850.0))
    out["arrhenius"] = {"f_28C": f28, "f_50C": f50, "amplification_28_to_50": f50 / f28,
                        "silent_bug_if_celsius": float(np.exp(-3850.0 / 50.0)),
                        "bound_lo": 1e-7, "bound_hi": 1e-5}
    assert 1e-7 <= f28 <= 1e-5 and 1e-7 <= f50 <= 1e-5, "Arrhenius 因子越界"
    D0_a3 = float(D_app3(C0, 50.0 + KELVIN_OFFSET))
    D0_a2 = float(D_app2(C0))
    out["D_app3_at_C0_50C"] = D0_a3
    out["D_ratio_app3_over_app2"] = D0_a3 / D0_a2
    print(f"[Q2] f(28C)={f28:.6e} f(50C)={f50:.6e} 放大 {f50 / f28:.4f} 倍; "
          f"D3/D2={D0_a3 / D0_a2:.4f}", flush=True)


    # 主解
    print(f"[Q2] 主求解 N={N_CELLS} dt={DT_SOLVE_Q12} span={Q2_SPAN_S} s "
          f"(43200 步，约 1 分钟)", flush=True)
    res = solve_drying("app3", Q2_SPAN_S, stop=("time", Q2_SPAN_S), dt_solve=DT_SOLVE_Q12,
                       dt_out=Q2_DT_OUT, N=N_CELLS, air_mode="linear",
                       C_init=C0, T_init=T0_C, record_fields=True,
                       progress=14400)
    print(f"[Q2] 主求解完成 {res['n_steps']} 步 Picard max={res['n_picard_max']} "
          f"用时 {time.time() - t0_wall:.1f} s", flush=True)
    r_nodes, dr, V, A_in = grid_weights(R0, N_CELLS)
    T_field = extract_output_nodes(res["T_out"], OUT_STRIDE_SPACE)
    C_field = extract_output_nodes(res["C_out"], OUT_STRIDE_SPACE)
    t_out = res["t_out"]
    assert T_field.shape == (10800, 21), f"输出形状不符：{T_field.shape}"
    out.update({"n_steps": res["n_steps"], "dt_solve_s": DT_SOLVE_Q12,
                "dt_out_s": Q2_DT_OUT, "N_cells": N_CELLS,
                "surface_form": res["surface_form"],
                "water_resid_max": res["water_resid_max"],
                "n_picard_max": res["n_picard_max"],
                "D_field_ratio_max": res["D_field_ratio_max"],
                "argmax_center_strict": bool(res["argmax_center_always"]),
                "argmax_center_within_tol": bool(res["argmax_center_within_tol"]),
                "C_hump_excess_max": res["C_hump_excess_max"]})

    col_cm = output_columns_cm(21, 0.1)
    ci = {c: col_cm.index(c) for c in (0.0, 0.5, 1.0, 1.5, 2.0)}
    out["t10800_temperature_C"] = {str(c): float(T_field[-1][i]) for c, i in ci.items()}
    out["t10800_moisture_kgkg"] = {str(c): float(C_field[-1][i]) for c, i in ci.items()}
    out["C_bar_10800"] = volume_average(res["C_out"][-1], V)
    out["C_bar_0"] = float(C0)
    T_air_of, C_air_of = air_driving()
    out["T_air_10800_C"] = float(T_air_of(10800.0))
    out["C_air_10800"] = float(C_air_of(10800.0))
    print(f"[Q2] t=3h  T: {out['t10800_temperature_C']}", flush=True)
    print(f"[Q2] t=3h  C: {out['t10800_moisture_kgkg']}  Cbar={out['C_bar_10800']:.6f}",
          flush=True)


    # 长程运行只作一致性检查。
    print("[Q2] P2-C2 长程对照 t_span=207360 s 的前 10800 s ...", flush=True)
    res_long = solve_drying("app3", 207360.0, stop=("time", 207360.0), dt_solve=DT_SOLVE_Q12,
                            dt_out=Q2_SPAN_S, N=N_CELLS, air_mode="linear",
                            C_init=C0, T_init=T0_C, record_fields=True, progress=100000)
    i10800 = int(np.argmin(np.abs(res_long["t_out"] - Q2_SPAN_S)))
    assert abs(res_long["t_out"][i10800] - Q2_SPAN_S) < 1e-9, "长程输出缺 10800 s 时层"
    dC_long = float(np.max(np.abs(res_long["C_out"][i10800] - res["C_out"][-1])))
    dT_long = float(np.max(np.abs(res_long["T_out"][i10800] - res["T_out"][-1])))
    out["P2C2_long_vs_short_max_dC"] = dC_long
    out["P2C2_long_vs_short_max_dT_K"] = dT_long
    out["P2C2_long_span_s"] = 207360.0
    assert dC_long < 1e-12 and dT_long < 1e-12,\
        f"P2-C2 失败：长程改变了短程解 dC={dC_long:.3e} dT={dT_long:.3e}"
    print(f"[Q2] P2-C2 长短程一致 dC={dC_long:.3e} dT={dT_long:.3e}", flush=True)


    # 比较 Picard 收敛和固定两轮。
    res_p2 = solve_drying("app3", Q2_SPAN_S, stop=("time", Q2_SPAN_S), dt_solve=DT_SOLVE_Q12,
                          dt_out=Q2_SPAN_S, N=N_CELLS, air_mode="linear", C_init=C0,
                          T_init=T0_C, picard_fixed=2, record_fields=False)
    _, _, Vp, _ = grid_weights(R0, N_CELLS)
    cb_conv = volume_average(res["C_out"][-1], V)
    cb_fix2 = volume_average(res_p2["C_final"], Vp)
    rel_pic = abs(cb_fix2 - cb_conv) / cb_conv
    out["PXC3_picard_invariance"] = {
        "C_bar_converged_1e-10": cb_conv, "C_bar_fixed_2_iters": cb_fix2,
        "rel_diff": rel_pic,
        "T_center_converged": float(res["T_out"][-1][0]),
        "T_center_fixed2": float(res_p2["T_final"][0])}
    assert rel_pic < 1e-4, f"PX-C3 失败：迭代次数改变了解 rel={rel_pic:.3e}（疑把迭代当推进）"
    print(f"[Q2] PX-C3 Picard 不变性 rel={rel_pic:.3e}", flush=True)


    # 收敛性
    print("[Q2] 网格收敛 N=100/200/400/800 ...", flush=True)
    gv_T, gv_C = [], []
    for Nc in N_CONVERGENCE:
        rc = solve_drying("app3", Q2_SPAN_S, stop=("time", Q2_SPAN_S), dt_solve=DT_SOLVE_Q12,
                          dt_out=Q2_SPAN_S, N=Nc, air_mode="linear", C_init=C0,
                          T_init=T0_C, record_fields=False)
        _, _, Vc, _ = grid_weights(R0, Nc)
        gv_T.append(float(rc["T_final"][0]))
        gv_C.append(volume_average(rc["C_final"], Vc))
        print(f"  N={Nc}: T(0,3h)={gv_T[-1]:.8f} Cbar={gv_C[-1]:.10f}", flush=True)
    print("[Q2] 步长收敛 dt=1/0.5/0.25/0.125 ...", flush=True)
    dv_T, dv_C = [], []
    for dtc in DT_CONVERGENCE_Q12:
        rc = solve_drying("app3", Q2_SPAN_S, stop=("time", Q2_SPAN_S), dt_solve=dtc,
                          dt_out=Q2_SPAN_S, N=N_CELLS, air_mode="linear", C_init=C0,
                          T_init=T0_C, record_fields=False)
        dv_T.append(float(rc["T_final"][0]))
        dv_C.append(volume_average(rc["C_final"], V))
        print(f"  dt={dtc}: T(0,3h)={dv_T[-1]:.8f} Cbar={dv_C[-1]:.10f}", flush=True)
    out["convergence"] = {
        "N_list": list(N_CONVERGENCE), "dt_list": list(DT_CONVERGENCE_Q12),
        "grid_T_center": convergence_ratios(gv_T), "grid_C_bar": convergence_ratios(gv_C),
        "dt_T_center": convergence_ratios(dv_T), "dt_C_bar": convergence_ratios(dv_C)}
    for kk in ("grid_T_center", "grid_C_bar", "dt_T_center", "dt_C_bar"):
        rc_ = out["convergence"][kk]["rel_change"][-1]
        assert abs(rc_) < 5e-3, f"{kk} 末档相对变化 {rc_:.3e} 未收敛"


    tab_ti = [int(round(x * 3600.0)) for x in Q2_TABLE_TIMES_H]
    tab_ci = [col_cm.index(round(x, 1)) for x in Q2_TABLE_RADII_CM]
    tabT = {str(tt): {str(Q2_TABLE_RADII_CM[j]): float(T_field[tt - 1][c])
                      for j, c in enumerate(tab_ci)} for tt in tab_ti}
    tabC = {str(tt): {str(Q2_TABLE_RADII_CM[j]): float(C_field[tt - 1][c])
                      for j, c in enumerate(tab_ci)} for tt in tab_ti}
    out["paper_table_times_s"] = tab_ti
    out["paper_table_temperature_C"] = tabT
    out["paper_table_moisture_kgkg"] = tabC

    # 写表并回读检查。
    dfT = make_field_df(t_out, col_cm, T_field)
    dfC = make_field_df(t_out, col_cm, C_field)
    xlsx = RESULTS_DIR / "result2.xlsx"
    write_xlsx(xlsx, {SHEETS_TC[0]: dfT, SHEETS_TC[1]: dfC})
    out["result_file"] = str(xlsx.name)
    out["result_shape"] = [int(dfT.shape[0]), int(dfT.shape[1])]
    print(f"[Q2] 写出 {xlsx.name} shape={dfT.shape}", flush=True)

    chk = pd.read_excel(xlsx, sheet_name=SHEETS_TC[1])
    worst = 0.0
    for tt in tab_ti:
        for j, c in enumerate(tab_ci):
            a = float(chk.iloc[tt - 1, c + 1])
            b = round(tabC[str(tt)][str(Q2_TABLE_RADII_CM[j])], DECIMALS)
            worst = max(worst, abs(a - b))
    out["table_vs_file_max_abs_diff"] = worst
    assert worst == 0.0, f"表格与文件不一致 {worst}"

    out["wall_time_s"] = time.time() - t0_wall
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / f"{TAG}_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[Q2] -> results/{TAG}_results.json ({out['wall_time_s']:.1f} s)", flush=True)
    return out


if __name__ == "__main__":
    run()
