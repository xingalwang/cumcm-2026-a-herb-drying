
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from params import (AIR_EXTRAP_C, AIR_EXTRAP_T_C, AIR_T_END_S, C0, C_THRESHOLD, DECIMALS,
                    DT_CONVERGENCE_Q34, DT_SOLVE_Q34, DURATION_DAYS_HIGH,
                    DURATION_DAYS_LOW, FIGDIR, N_CELLS, N_CONVERGENCE, OUT_STRIDE_SPACE,
                    Q3_DT_OUT, Q3_TABLE_DT_H, R0, T0_C, T_MAX_CAP_S, RESULTS_DIR)
from utils import (convergence_ratios, extract_output_nodes, grid_weights, make_field_df,
                   output_columns_cm, solve_drying, volume_average, write_xlsx)

TAG = "problem_3"


def run():
    t0_wall = time.time()
    out = {"problem": 3, "property_set": "appendix3", "coupling": "two_way",
           "solver_reused_from": "problem2 (identical solve_drying call, stop= first_passage)",
           "C_threshold_kgkg": C_THRESHOLD, "criterion": "max_r C(r,t) < C_thr (R5, 上确界)"}

    # 终点用 max(C) 的首次越阈值时刻。
    print(f"[Q3] 首达求解 N={N_CELLS} dt={DT_SOLVE_Q34} 上限 {T_MAX_CAP_S / 3600:.0f} h",
          flush=True)
    res = solve_drying("app3", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                       dt_solve=DT_SOLVE_Q34, dt_out=Q3_DT_OUT, N=N_CELLS,
                       air_mode="linear", C_init=C0, T_init=T0_C, record_fields=True,
                       rebound_check=True, progress=20000)
    assert res["t_end"] is not None,\
        f"上限 {T_MAX_CAP_S}s 内未达标，末态 maxC={float(np.max(res['C_final'])):.6f}"
    t_end_s = float(res["t_end"])
    t_end_h = t_end_s / 3600.0
    print(f"[Q3] t_end = {t_end_s:.4f} s = {t_end_h:.4f} h = {t_end_h / 24:.4f} d "
          f"({res['n_steps']} 步 {time.time() - t0_wall:.1f} s)", flush=True)

    r_nodes, dr, V, A_in = grid_weights(R0, N_CELLS)
    out.update({"t_end_s": t_end_s, "t_end_h": round(t_end_h, DECIMALS),
                "t_end_h_full": t_end_h, "t_end_days": t_end_h / 24.0,
                "bracket_s": [float(res["bracket"][0]), float(res["bracket"][1])],
                "bracket_len_s": float(res["bracket"][1] - res["bracket"][0]),
                "no_rebound_to_1.05t_end": bool(res["no_rebound"]),
                "argmax_center_strict": bool(res["argmax_center_always"]),
                "argmax_center_within_tol": bool(res["argmax_center_within_tol"]),
                "C_hump_excess_max": res["C_hump_excess_max"],
                "hump_tol": res["hump_tol"],
                "hump_first_t_s": res["hump_first_t_s"],
                "final_profile_max_diff": float(np.max(np.diff(res["C_final"]))),
                "n_steps": res["n_steps"], "dt_solve_s": DT_SOLVE_Q34,
                "dt_out_s": Q3_DT_OUT, "N_cells": N_CELLS,
                "surface_form": res["surface_form"],
                "water_resid_max": res["water_resid_max"],
                "n_picard_max": res["n_picard_max"],
                "D_field_ratio_max": res["D_field_ratio_max"]})



    assert res["argmax_center_within_tol"],\
        f"出现真实内部驼峰（高度 {res['C_hump_excess_max']:.3e} > {res['hump_tol']:.0e}）："\
        "须如实报告剖面非单调，不得改判据"
    assert float(np.max(np.diff(res["C_final"]))) < 0.0, "末态剖面非严格单调递减"
    assert res["no_rebound"], "1.05*t_end 内 g 出现回升：首达时刻需修正"


    # 用二分返回的状态，不取最近输出点。
    hist = res["hist"]
    ht = np.asarray(hist["t"], dtype=float)


    C_end = res["C_at_t_end"]
    T_end = res["T_at_t_end"]
    assert C_end is not None, "二分未返回 t_end 处状态"
    out["at_t_end"] = {"t_s": float(res["t_at_state_end"]),
                       "max_C": float(np.max(C_end)),
                       "C_bar": volume_average(C_end, V),
                       "C_center": float(C_end[0]), "C_surface": float(C_end[-1]),
                       "T_center_C": float(T_end[0]), "T_surface_C": float(T_end[-1]),
                       "g_at_t_end": float(np.max(C_end)) - C_THRESHOLD}
    assert out["at_t_end"]["max_C"] < C_THRESHOLD,\
        f"t_end 处 maxC={out['at_t_end']['max_C']:.8f} 未低于阈值 {C_THRESHOLD}"

    out["at_t_end"]["g_magnitude_ok"] = bool(-1e-4 < out["at_t_end"]["g_at_t_end"] < 0.0)

    cb = np.asarray(hist["C_bar"], dtype=float)
    below = np.where(cb < C_THRESHOLD)[0]
    t_bar_s = float(ht[below[0]]) if below.size else None
    out["wrong_criterion_C_bar_t_end_h"] = (t_bar_s / 3600.0) if t_bar_s else None
    out["wrong_criterion_C_bar_underestimates_by_h"] =\
        (t_end_h - t_bar_s / 3600.0) if t_bar_s else None
    print(f"[Q3] t_end 处 maxC={out['at_t_end']['max_C']:.6f} Cbar={out['at_t_end']['C_bar']:.6f}"
          f"；若误用 Cbar 判据 t_end={out['wrong_criterion_C_bar_t_end_h']} h", flush=True)


    out["duration_days_cross_check"] = {
        "computed_days": t_end_h / 24.0, "problem_statement_low": DURATION_DAYS_LOW,
        "problem_statement_high": DURATION_DAYS_HIGH,
        "inside_stated_range": bool(DURATION_DAYS_LOW <= t_end_h / 24.0 <= DURATION_DAYS_HIGH)}

    out["expectation_window_h"] = [30.7, 80.6]
    out["inside_expectation_window"] = bool(30.7 < t_end_h < 80.6)
    assert 30.0 < t_end_h < 90.0, f"t_end={t_end_h:.4f} h 越出 §13 预期区间"
    out["checklist_P3C2_window_h"] = [24.0, 72.0]
    out["inside_checklist_window"] = bool(24.0 <= t_end_h <= 72.0)


    out["air_extrapolation"] = {
        "measured_until_s": AIR_T_END_S, "hold_T_air_C": AIR_EXTRAP_T_C,
        "hold_C_air": AIR_EXTRAP_C,
        "extrapolated_fraction_pct": (1.0 - AIR_T_END_S / t_end_s) * 100.0}
    print(f"[Q3] 外推段占比 {out['air_extrapolation']['extrapolated_fraction_pct']:.4f}%",
          flush=True)


    dcb = -np.gradient(cb, ht)
    out["drying_rate"] = {
        "max_rate_per_s": float(np.max(dcb)), "t_at_max_rate_h": float(ht[np.argmax(dcb)] / 3600.0),
        "rate_at_1h_per_s": float(np.interp(3600.0, ht, dcb)),
        "rate_at_t_end_per_s": float(np.interp(t_end_s, ht, dcb)),
        "rate_at_12h_per_s": float(np.interp(12.0 * 3600.0, ht, dcb)),
        "C_bar_note": "体积平均 (2/R^2)int C r dr，用 §5.1 体积权，与守恒诊断同口径"}


    # 收敛性
    print("[Q3] t_end 网格收敛 N=100/200/400/800 ...", flush=True)
    gv = []
    for Nc in N_CONVERGENCE:
        rc = solve_drying("app3", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                          dt_solve=DT_SOLVE_Q34, dt_out=3600.0, N=Nc, air_mode="linear",
                          C_init=C0, T_init=T0_C, record_fields=False)
        gv.append(float(rc["t_end"]) / 3600.0)
        print(f"  N={Nc}: t_end={gv[-1]:.6f} h", flush=True)
    print("[Q3] t_end 步长收敛 dt=8/4/2/1 ...", flush=True)
    dv = []
    for dtc in DT_CONVERGENCE_Q34:
        rc = solve_drying("app3", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                          dt_solve=dtc, dt_out=3600.0, N=N_CELLS, air_mode="linear",
                          C_init=C0, T_init=T0_C, record_fields=False)
        dv.append(float(rc["t_end"]) / 3600.0)
        print(f"  dt={dtc}: t_end={dv[-1]:.6f} h", flush=True)
    out["convergence"] = {"N_list": list(N_CONVERGENCE), "dt_list": list(DT_CONVERGENCE_Q34),
                          "grid_t_end_h": convergence_ratios(gv),
                          "dt_t_end_h": convergence_ratios(dv)}
    for kk in ("grid_t_end_h", "dt_t_end_h"):
        rc_ = abs(out["convergence"][kk]["rel_change"][-1])
        out["convergence"][kk]["last_rel_change_pct"] = rc_ * 100.0
        assert rc_ < 5e-3, f"{kk} 末档相对变化 {rc_ * 100:.4f}% 超 0.5%"
    print(f"[Q3] 收敛：网格末档 {out['convergence']['grid_t_end_h']['last_rel_change_pct']:.4f}%，"
          f"步长末档 {out['convergence']['dt_t_end_h']['last_rel_change_pct']:.4f}%", flush=True)


    # 末行单独写入 t_end 状态。
    col_cm = output_columns_cm(21, 0.1)
    C_field = extract_output_nodes(res["C_out"], OUT_STRIDE_SPACE)
    t_out = res["t_out"]
    keep = t_out <= t_end_s + Q3_DT_OUT
    dfC = make_field_df(t_out[keep], col_cm, C_field[keep])
    xlsx = RESULTS_DIR / "result3.xlsx"
    write_xlsx(xlsx, {"水分浓度": dfC})
    out["result_file"] = str(xlsx.name)
    out["result_shape"] = [int(dfC.shape[0]), int(dfC.shape[1])]
    out["result_last_time_s"] = float(t_out[keep][-1])
    print(f"[Q3] 写出 {xlsx.name} shape={dfC.shape} 末行 t={out['result_last_time_s']:.0f} s",
          flush=True)


    tab_ci = [col_cm.index(c) for c in (0.0, 0.5, 1.0, 1.5, 2.0)]
    tab_t_h = list(np.arange(0.0, t_end_h, Q3_TABLE_DT_H))
    rows = []
    for th in tab_t_h:
        ts = th * 3600.0
        if ts <= 0.0:
            vals = [float(C0)] * len(tab_ci)
        else:
            j = int(np.argmin(np.abs(t_out - ts)))
            assert abs(t_out[j] - ts) < 1e-6, f"表格时刻 {ts} 不在输出层"
            vals = [float(C_field[j][c]) for c in tab_ci]
        rows.append({"t_h": float(th), "values": vals})



    rows.append({"t_h": t_end_h, "values": [float(C_end[c * OUT_STRIDE_SPACE])
                                           for c in tab_ci], "is_t_end": True})
    out["paper_table_radii_cm"] = [0.0, 0.5, 1.0, 1.5, 2.0]
    out["paper_table_rows"] = rows
    out["paper_table_last_row_max_C"] = float(np.max(rows[-1]["values"]))
    assert out["paper_table_last_row_max_C"] < C_THRESHOLD, "末行（t_end）maxC 未达标"

    _gap = abs(out["paper_table_last_row_max_C"] - out["at_t_end"]["max_C"])
    assert _gap < 1e-6, (f"末行 maxC={out['paper_table_last_row_max_C']:.8f} 与 "
                         f"at_t_end maxC={out['at_t_end']['max_C']:.8f} 差 {_gap:.2e}，"
                         "末行取的不是 t_end 处的场")

    chk = pd.read_excel(xlsx, sheet_name="水分浓度")
    j6 = int(np.argmin(np.abs(t_out[keep] - 6.0 * 3600.0)))
    worst = max(abs(float(chk.iloc[j6, c + 1]) - round(float(C_field[keep][j6][c]), DECIMALS))
                for c in tab_ci)
    out["table_vs_file_max_abs_diff"] = worst
    assert worst == 0.0, f"表格与文件不一致 {worst}"

    out["wall_time_s"] = time.time() - t0_wall
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / f"{TAG}_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[Q3] -> results/{TAG}_results.json ({out['wall_time_s']:.1f} s)", flush=True)
    return out


if __name__ == "__main__":
    run()
