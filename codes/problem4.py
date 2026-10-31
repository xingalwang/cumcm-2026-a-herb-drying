
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from params import (AIR_EXTRAP_C, AIR_EXTRAP_T_C, AIR_T_END_S, C0, C_THRESHOLD, DECIMALS,
                    DT_CONVERGENCE_Q34, DT_SOLVE_Q34, FIGDIR, HM_MASS, KELVIN_OFFSET,
                    N_CELLS, N_CONVERGENCE, OUT_STRIDE_SPACE, Q4_DT_OUT, Q4_TABLE_DT_H,
                    R0, RADIUS_T_END_S, SURFACE_COL_NAME, T0_C, T_MAX_CAP_S, RESULTS_DIR,
                    INDEX_HEADER)
from props import D_app3, D_app4, props_app4
from utils import (convergence_ratios, extract_output_nodes, grid_weights, load_radius,
                   output_columns_cm, radius_interpolant, solve_drying, volume_average,
                   write_xlsx)

TAG = "problem_4"


def run():
    t0_wall = time.time()
    out = {"problem": 4, "property_set": "appendix4", "coupling": "two_way",
           "geometry": "ALE moving boundary x=r/R(t), R from 附件2 (PCHIP)"}


    r0_, c0_, k0_ = props_app4(C0)
    out["props_at_C0"] = {"rho": float(r0_), "cp": float(c0_), "k": float(k0_)}
    assert abs(float(c0_) - 3394.3662) < 1e-3, f"cp4(2.55) 应 3394.3662，得 {float(c0_):.4f}"
    assert abs(float(k0_) - 0.2636620) < 1e-6, f"k4(2.55) 应 0.2636620，得 {float(k0_):.7f}"
    out["capability_checklist_deviation"] = {
        "P4-C2(c)_expects_cp": 3395.8, "corrected_cp": float(c0_),
        "P4-C2(c)_expects_k": 0.2634, "corrected_k": float(k0_),
        "note": "CAPABILITY_CHECKLIST 携带 §0 修正前的上游值；第二项自变量应为 C/(C+1)=51/71"}
    TK50 = 50.0 + KELVIN_OFFSET
    D4 = float(D_app4(C0, TK50)); D3 = float(D_app3(C0, TK50))
    out["D_app4_at_C0_50C"] = D4
    out["D_ratio_app4_over_app3"] = D4 / D3
    assert 0.15 <= D4 / D3 <= 0.25, f"P4-C2(d) D4/D3={D4 / D3:.5f} 越出 [0.15,0.25]"
    print(f"[Q4] props@C0 cp={float(c0_):.4f} k={float(k0_):.7f}; D4/D3={D4 / D3:.5f}", flush=True)


    # 半径数据进入 ALE 网格。
    R_of, Rdot_of, meta = radius_interpolant()
    assert abs(float(R_of(0.0)) - 0.02) < 1e-9, "R(0) 应为 2.000 cm"
    assert abs(float(R_of(RADIUS_T_END_S)) - 0.01198) < 1e-9, "R(259200) 应为 1.198 cm"
    tq = np.linspace(0.0, RADIUS_T_END_S, 200001)
    Rq = np.asarray(R_of(tq), dtype=float)
    assert np.all(np.diff(Rq) <= 1e-15), "PCHIP 插值不单调不增"
    assert Rq.min() >= 0.01198 - 1e-12 and Rq.max() <= 0.02 + 1e-12, "PCHIP 冲出实测界"
    Rdotq = np.asarray(Rdot_of(tq), dtype=float)
    Rdot_max = float(np.max(np.abs(Rdotq)))
    Pe_shr = Rdot_max * R0 / D4
    out["radius"] = {**meta, "ptp_cm": meta["ptp_cm"],
                     "monotone_nonincreasing": True,
                     "min_cm": float(Rq.min() * 100.0), "max_cm": float(Rq.max() * 100.0),
                     "Rdot_max_abs_m_s": Rdot_max,
                     "t_at_Rdot_max_s": float(tq[int(np.argmax(np.abs(Rdotq)))]),
                     "diffusion_velocity_D_over_R0_m_s": D4 / R0,
                     "Pe_shrink": Pe_shr,
                     "area_ratio_R0_over_Rend_sq": (0.02 / 0.01198) ** 2,
                     "shrink_pct": (0.02 - 0.01198) / 0.02 * 100.0}
    assert meta["ptp_cm"] > 0.5, "P4-C1(a) 收缩幅度须 >0.5 cm"
    assert Pe_shr > 1.0, f"Pe_shr={Pe_shr:.4f} 应 >1（收缩不是小扰动）"
    print(f"[Q4] Pe_shr={Pe_shr:.4f} |Rdot|max={Rdot_max:.6e} m/s 面积比="
          f"{out['radius']['area_ratio_R0_over_Rend_sq']:.4f}", flush=True)


    # 主解
    print(f"[Q4] ALE 主求解 N={N_CELLS} dt={DT_SOLVE_Q34}", flush=True)
    res = solve_drying("app4", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                       geometry=("ale", (R_of, Rdot_of)), dt_solve=DT_SOLVE_Q34,
                       dt_out=Q4_DT_OUT, N=N_CELLS, air_mode="linear", C_init=C0,
                       T_init=T0_C, record_fields=True, rebound_check=True, progress=20000)
    assert res["t_end"] is not None,\
        f"上限内未达标，末态 maxC={float(np.max(res['C_final'])):.6f}"
    t_end_s = float(res["t_end"]); t_end_h = t_end_s / 3600.0
    print(f"[Q4] t_end = {t_end_s:.4f} s = {t_end_h:.4f} h = {t_end_h / 24:.4f} d "
          f"({res['n_steps']} 步 {time.time() - t0_wall:.1f} s)", flush=True)

    C_end = res["C_at_t_end"]; T_end = res["T_at_t_end"]
    R_end_m = float(R_of(t_end_s))
    _, _, V_end, _ = grid_weights(R_end_m, N_CELLS)
    out.update({"t_end_s": t_end_s, "t_end_h": round(t_end_h, DECIMALS),
                "t_end_h_full": t_end_h, "t_end_days": t_end_h / 24.0,
                "R_at_t_end_cm": R_end_m * 100.0,
                "bracket_s": [float(res["bracket"][0]), float(res["bracket"][1])],
                "no_rebound_to_1.05t_end": bool(res["no_rebound"]),
                "argmax_center_strict": bool(res["argmax_center_always"]),
                "argmax_center_within_tol": bool(res["argmax_center_within_tol"]),
                "C_hump_excess_max": res["C_hump_excess_max"],
                "final_profile_max_diff": float(np.max(np.diff(res["C_final"]))),
                "motion_terms_cancel_resid_max": res["motion_resid_max"],
                "v_mesh_active_step_fraction": res["v_mesh_active_steps"] / max(res["n_steps"], 1),
                "n_steps": res["n_steps"], "dt_solve_s": DT_SOLVE_Q34,
                "dt_out_s": Q4_DT_OUT, "N_cells": N_CELLS,
                "surface_form": res["surface_form"],
                "water_resid_max": res["water_resid_max"],
                "n_picard_max": res["n_picard_max"],
                "D_field_ratio_max": res["D_field_ratio_max"],
                "at_t_end": {"max_C": float(np.max(C_end)),
                             "C_bar": volume_average(C_end, V_end),
                             "C_center": float(C_end[0]), "C_surface": float(C_end[-1]),
                             "T_center_C": float(T_end[0]),
                             "g_at_t_end": float(np.max(C_end)) - C_THRESHOLD}})
    assert res["argmax_center_within_tol"],\
        f"出现真实内部驼峰 {res['C_hump_excess_max']:.3e}：须如实报告"
    assert res["no_rebound"], "1.05*t_end 内 g 回升"
    assert out["at_t_end"]["max_C"] < C_THRESHOLD, "t_end 处 maxC 未达标"

    assert res["motion_resid_max"] < 1e-12,\
        f"运动项抵消残差 {res['motion_resid_max']:.3e} 未达 1e-12"





    # 检查网格运动项是否与实测收缩时序一致。
    span_integrated = res["n_steps"] * DT_SOLVE_Q34
    t_meas, R_meas = load_radius()
    seg_move = np.abs(np.diff(R_meas)) > 1e-15
    seg_span = np.clip(np.minimum(t_meas[1:], span_integrated) - t_meas[:-1], 0.0, None)

    denom = span_integrated
    frac_expected = float(seg_span[seg_move].sum() / denom)
    out["v_mesh_span_integrated_s"] = span_integrated
    frac_got = out["v_mesh_active_step_fraction"]
    out["v_mesh_active_fraction_expected_from_data"] = frac_expected
    out["v_mesh_active_fraction_abs_diff"] = abs(frac_got - frac_expected)
    assert abs(frac_got - frac_expected) < 1e-3,\
        f"P4-C1(b)：网格运动活跃占比 {frac_got:.6f} 与实测非平台时长占比 "\
        f"{frac_expected:.6f} 不符，收缩未按实测时序进入求解"

    R_at_end = float(R_of(t_end_s))
    out["R_at_t_end_cm"] = R_at_end * 100.0
    out["R_shrink_during_solve_cm"] = (R0 - R_at_end) * 100.0
    assert R0 - R_at_end > 5e-3, f"求解期内半径仅缩 {(R0 - R_at_end) * 100:.4f} cm"
    print(f"[Q4] 运动项抵消残差={res['motion_resid_max']:.3e} 活跃步占比="
          f"{frac_got:.6f}（实测非平台占比 {frac_expected:.6f}）"
          f" R(t_end)={R_at_end * 100:.4f} cm", flush=True)




    # 冻结半径作为对照。
    print("[Q4] P4-C1(c) 冻结半径对照 R≡2 cm（上限放宽到 3e6 s）...", flush=True)
    CAP_FROZEN_S = 3.0e6
    res_fix = solve_drying("app4", CAP_FROZEN_S, stop=("first_passage", C_THRESHOLD),
                           t_cap=CAP_FROZEN_S, geometry=("fixed", R0),
                           dt_solve=DT_SOLVE_Q34, dt_out=3600.0,
                           N=N_CELLS, air_mode="linear", C_init=C0, T_init=T0_C,
                           record_fields=False, rebound_check=False, progress=100000)
    assert res_fix["t_end"] is not None,\
        f"冻结半径算例在 {CAP_FROZEN_S:.0e} s 内未达标，末态 maxC="\
        f"{float(np.max(res_fix['C_final'])):.6f}"
    t_fix_h = float(res_fix["t_end"]) / 3600.0
    rel_shrink = abs(t_end_h - t_fix_h) / t_fix_h
    out["P4C1c_frozen_radius"] = {
        "t_end_frozen_R_h": t_fix_h, "t_end_ale_h": t_end_h,
        "rel_change": rel_shrink, "rel_change_pct": rel_shrink * 100.0,
        "shrink_shortens_drying": bool(t_end_h < t_fix_h),
        "cap_used_s": CAP_FROZEN_S,
        "note": "冻结半径下 t_end 超出 §6.3 的 120 h 时窗，说明收缩不是小扰动"
                "而是干燥能在 5 天内完成的必要条件"}
    assert rel_shrink > 0.03,\
        f"P4-C1(c) 失败：收缩只改变 t_end {rel_shrink * 100:.4f}%（<3%），收缩未进入求解"
    print(f"[Q4] 冻结 R: t_end={t_fix_h:.4f} h vs ALE {t_end_h:.4f} h "
          f"→ 相对变化 {rel_shrink * 100:.4f}%", flush=True)


    try:
        with open(FIGDIR / "problem_3_results.json", "r", encoding="utf-8") as fh:
            t3_h = float(json.load(fh)["t_end_h_full"])
    except Exception:
        t3_h = None
    out["vs_problem3"] = {
        "t_end_q3_h": t3_h, "t_end_q4_h": t_end_h,
        "ratio_q4_over_q3": (t_end_h / t3_h) if t3_h else None,
        "diff_h": (t_end_h - t3_h) if t3_h else None,
        "competing_mechanisms": {
            "D4_over_D3_at_C0_50C": D4 / D3,
            "geometric_enhancement_area_ratio": out["radius"]["area_ratio_R0_over_Rend_sq"]}}
    if t3_h:
        print(f"[Q4] vs 问题三：{t_end_h:.4f} h vs {t3_h:.4f} h "
              f"(比 {t_end_h / t3_h:.4f})", flush=True)


    out["P4C2e_no_inversion"] = {
        "radius_plateau_start_h": meta["plateau_start_s"] / 3600.0,
        "t_end_h": t_end_h,
        "gap_from_plateau_h": abs(t_end_h - meta["plateau_start_s"] / 3600.0),
        "note": "附件2 首次达 1.198 cm 的 67.0 h 与干燥终点无因果关系；"
                "t_end 由水分方程积分产生，附件2 只经 1/R(t)^2 进入几何因子"}
    out["radius_extrapolation"] = {
        "radius_data_covers_until_s": RADIUS_T_END_S,
        "t_end_beyond_radius_data": bool(t_end_s > RADIUS_T_END_S),
        "extrapolated_fraction_pct": max(0.0, (1.0 - RADIUS_T_END_S / t_end_s) * 100.0),
        "mode": "hold R=1.198 cm (禁止趋势外推：末段已进入平台，外推会给出负半径方向)"}
    out["air_extrapolation"] = {
        "measured_until_s": AIR_T_END_S, "hold_T_air_C": AIR_EXTRAP_T_C,
        "hold_C_air": AIR_EXTRAP_C,
        "extrapolated_fraction_pct": (1.0 - AIR_T_END_S / t_end_s) * 100.0}


    # 无收缩时，两种实现应退化为同一结果。
    print("[Q4] 拉格朗日对照：R≡2 cm 下 ALE 分支 vs 固定域分支 ...", flush=True)
    res_ale_frozen = solve_drying(
        "app4", 7200.0, stop=("time", 7200.0),
        geometry=("ale", (lambda tt: np.full_like(np.asarray(tt, dtype=float), R0),
                          lambda tt: np.zeros_like(np.asarray(tt, dtype=float)))),
        dt_solve=DT_SOLVE_Q34, dt_out=7200.0, N=N_CELLS, air_mode="linear",
        C_init=C0, T_init=T0_C, record_fields=False)
    res_fixed_short = solve_drying("app4", 7200.0, stop=("time", 7200.0),
                                   geometry=("fixed", R0), dt_solve=DT_SOLVE_Q34,
                                   dt_out=7200.0, N=N_CELLS, air_mode="linear",
                                   C_init=C0, T_init=T0_C, record_fields=False)
    d_impl = float(np.max(np.abs(res_ale_frozen["C_final"] - res_fixed_short["C_final"])))
    out["no_shrink_two_implementations_max_dC"] = d_impl
    assert d_impl < 1e-8, f"无收缩时两实现不一致 dC={d_impl:.3e}"
    print(f"[Q4] 无收缩两实现一致 dC={d_impl:.3e}", flush=True)


    rho_d0 = float(props_app4(C0)[0]) / (1.0 + C0)
    R_pred_end = R0 * np.sqrt(rho_d0 / (float(props_app4(C_THRESHOLD)[0]) / (1.0 + C_THRESHOLD)))
    out["dry_mass_consistency"] = {
        "rho_dry_at_C0_kg_m3": rho_d0,
        "rho_dry_at_Cthr_kg_m3": float(props_app4(C_THRESHOLD)[0]) / (1.0 + C_THRESHOLD),
        "R_predicted_end_cm": float(R_pred_end * 100.0),
        "R_measured_end_cm": 1.198,
        "rel_gap_pct": float(abs(R_pred_end * 100.0 - 1.198) / 1.198 * 100.0),
        "note": "拉格朗日模型由干物质守恒自行预测 R(t)，与实测差异是题面数据内在不一致"
                "（DATA_FACTS F1），非实现错误；ALE 主模型中 rho_d 被完全消去，不受影响"}
    print(f"[Q4] 干物质诊断：R_pred={R_pred_end * 100:.4f} cm vs 实测 1.198 cm "
          f"(差 {out['dry_mass_consistency']['rel_gap_pct']:.4f}%)", flush=True)


    print("[Q4] t_end 网格收敛 N=100/200/400/800 ...", flush=True)
    gv = []
    for Nc in N_CONVERGENCE:
        rc = solve_drying("app4", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                          geometry=("ale", (R_of, Rdot_of)), dt_solve=DT_SOLVE_Q34,
                          dt_out=3600.0, N=Nc, air_mode="linear", C_init=C0, T_init=T0_C,
                          record_fields=False)
        gv.append(float(rc["t_end"]) / 3600.0)
        print(f"  N={Nc}: t_end={gv[-1]:.6f} h", flush=True)
    print("[Q4] t_end 步长收敛 dt=8/4/2/1 ...", flush=True)
    dv = []
    for dtc in DT_CONVERGENCE_Q34:
        rc = solve_drying("app4", T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                          geometry=("ale", (R_of, Rdot_of)), dt_solve=dtc, dt_out=3600.0,
                          N=N_CELLS, air_mode="linear", C_init=C0, T_init=T0_C,
                          record_fields=False)
        dv.append(float(rc["t_end"]) / 3600.0)
        print(f"  dt={dtc}: t_end={dv[-1]:.6f} h", flush=True)
    out["convergence"] = {"N_list": list(N_CONVERGENCE), "dt_list": list(DT_CONVERGENCE_Q34),
                          "grid_t_end_h": convergence_ratios(gv),
                          "dt_t_end_h": convergence_ratios(dv)}
    for kk in ("grid_t_end_h", "dt_t_end_h"):
        rc_ = abs(out["convergence"][kk]["rel_change"][-1])
        out["convergence"][kk]["last_rel_change_pct"] = rc_ * 100.0
        assert rc_ < 5e-3, f"{kk} 末档相对变化 {rc_ * 100:.4f}% 超 0.5%"
    print(f"[Q4] 收敛：网格末档 {out['convergence']['grid_t_end_h']['last_rel_change_pct']:.4f}%，"
          f"步长末档 {out['convergence']['dt_t_end_h']['last_rel_change_pct']:.4f}%", flush=True)


    # 固定半径列从 ALE 坐标插值到物理半径。
    C_field_x = res["C_out"]
    t_out = res["t_out"]
    R_out = res["R_out"]
    keep = t_out <= t_end_s + Q4_DT_OUT
    t_keep, C_keep, R_keep = t_out[keep], C_field_x[keep], R_out[keep]
    col_cm = output_columns_cm(20, 0.1)
    n_rows = t_keep.shape[0]
    data = np.full((n_rows, len(col_cm)), np.nan)
    x_nodes = np.linspace(0.0, 1.0, N_CELLS + 1)
    for i in range(n_rows):
        Rc = float(R_keep[i]) * 100.0
        for j, rc_cm in enumerate(col_cm):
            if rc_cm > Rc + 1e-12:
                continue
            xj = (rc_cm / 100.0) / float(R_keep[i])
            kj = int(round(xj * N_CELLS))
            assert abs(kj / N_CELLS - xj) < 5e-3, "固定 r 列未落在 ALE 节点附近"
            data[i, j] = C_keep[i][min(kj, N_CELLS)]
    surf = np.array([C_keep[i][-1] for i in range(n_rows)], dtype=float)
    dfC = pd.DataFrame(data, columns=[str(c) for c in col_cm])
    dfC[SURFACE_COL_NAME] = surf
    dfC.insert(0, INDEX_HEADER, t_keep.astype(float))
    xlsx = RESULTS_DIR / "result4.xlsx"
    write_xlsx(xlsx, {"水分浓度": dfC})
    n_blank_last = int(np.sum(np.isnan(data[-1])))
    out["result_file"] = str(xlsx.name)
    out["result_shape"] = [int(dfC.shape[0]), int(dfC.shape[1])]
    out["result_last_time_s"] = float(t_keep[-1])
    out["result_last_column_name"] = SURFACE_COL_NAME
    out["blank_cells_total"] = int(np.sum(np.isnan(data)))
    out["blank_cols_in_last_row"] = n_blank_last
    out["R_at_last_row_cm"] = float(R_keep[-1]) * 100.0
    print(f"[Q4] 写出 {xlsx.name} shape={dfC.shape} 末列='{SURFACE_COL_NAME}' "
          f"末行留空 {n_blank_last} 列 (R={out['R_at_last_row_cm']:.4f} cm)", flush=True)


    tab_r_cm = [0.0, 0.5, 1.0, 1.5]
    rows = []
    for th in np.arange(0.0, t_end_h, Q4_TABLE_DT_H):
        ts = th * 3600.0
        Rc_cm = float(R_of(ts)) * 100.0
        if ts <= 0.0:
            vals = [float(C0)] * len(tab_r_cm); surf_v = float(C0)
        else:
            j = int(np.argmin(np.abs(t_keep - ts)))
            vals = []
            for rc_cm in tab_r_cm:
                if rc_cm > Rc_cm + 1e-12:
                    vals.append(None)
                else:
                    kj = int(round((rc_cm / 100.0) / float(R_keep[j]) * N_CELLS))
                    vals.append(float(C_keep[j][min(kj, N_CELLS)]))
            surf_v = float(C_keep[j][-1])
        rows.append({"t_h": float(th), "R_cm": Rc_cm, "values": vals, "surface": surf_v})
    R_te_cm = R_end_m * 100.0
    vals_te = []
    for rc_cm in tab_r_cm:
        if rc_cm > R_te_cm + 1e-12:
            vals_te.append(None)
        else:
            kj = int(round((rc_cm / 100.0) / R_end_m * N_CELLS))
            vals_te.append(float(C_end[min(kj, N_CELLS)]))
    rows.append({"t_h": t_end_h, "R_cm": R_te_cm, "values": vals_te,
                 "surface": float(C_end[-1]), "is_t_end": True})
    out["paper_table_radii_cm"] = tab_r_cm
    out["paper_table_last_column"] = SURFACE_COL_NAME
    out["paper_table_rows"] = rows
    present = [v for v in vals_te + [float(C_end[-1])] if v is not None]
    out["paper_table_last_row_max_C"] = float(max(present))
    assert out["paper_table_last_row_max_C"] < C_THRESHOLD, "末行（t_end）maxC 未达标"

    t_R15 = float(tq[int(np.argmax(Rq <= 0.015))])
    out["r_1p5cm_column_blank_from_h"] = t_R15 / 3600.0

    out["wall_time_s"] = time.time() - t0_wall
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / f"{TAG}_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[Q4] -> results/{TAG}_results.json ({out['wall_time_s']:.1f} s)", flush=True)
    return out


if __name__ == "__main__":
    run()
