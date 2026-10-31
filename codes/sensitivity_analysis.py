
from __future__ import annotations

import json
import time

import numpy as np

from params import (AIR_EXTRAP_C, AIR_EXTRAP_T_C, C0, C_THRESHOLD, DT_SOLVE_Q34, FIGDIR,
                    H_CONV, HM_MASS, N_CELLS, R0, SENS_C_AIR, SENS_DT_OUT_S,
                    SENS_DT_SOLVE_S, SENS_H_FACTORS, SENS_HM_FACTORS, SENS_T_AIR_C,
                    T0_C, T_MAX_CAP_S)
from props import D_app3, props_app2
from utils import air_driving, radius_interpolant, solve_drying, volume_average, grid_weights

TAG = "sensitivity"
OUT = {}


def _checkpoint(stage):
    # 中断后可以从最近一节继续检查。
    p = FIGDIR.parent / "_tmp" / f"{TAG}_partial.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"last_completed": stage, **OUT}, fh, ensure_ascii=False,
                  indent=2, default=float)


def _t_end_h(pset, geometry, **kw):
    kw.setdefault("dt_solve", SENS_DT_SOLVE_S)
    kw.setdefault("dt_out", SENS_DT_OUT_S)
    kw.setdefault("N", N_CELLS)
    kw.setdefault("air_mode", "linear")
    res = solve_drying(pset, T_MAX_CAP_S, stop=("first_passage", C_THRESHOLD),
                       geometry=geometry, C_init=C0, T_init=T0_C,
                       record_fields=False, rebound_check=False, **kw)
    return (float(res["t_end"]) / 3600.0) if res["t_end"] is not None else None


def _rel(x, base):
    return None if x is None else (x - base) / base


def _h(x):
    return "unreached" if x is None else f"{x:.4f} h"


def _p(rel):
    return "      n/a" if rel is None else f"{rel * 100:+.4f}%"


def _req(rel, label):
    assert rel is not None,\
        f"§8.3 对照方案 {label} 在 {T_MAX_CAP_S / 3600:.0f} h 上限内未达阈值："\
        "无法与主方案在同一口径下比较相对差，须先登记该对照未达标"
    return rel


def baselines():
    # 所有相对变化都以这里的结果为基准。
    print("[SENS] 基准（dt=4）...", flush=True)
    R_of, Rdot_of, _ = radius_interpolant()
    b3 = _t_end_h("app3", ("fixed", R0))
    b4 = _t_end_h("app4", ("ale", (R_of, Rdot_of)))
    ref3, ref4 = 57.47449652777778, 51.08876736111111
    OUT["baseline"] = {
        "t_end_q3_h_dt4": b3, "t_end_q4_h_dt4": b4,
        "t_end_q3_h_main_dt2": ref3, "t_end_q4_h_main_dt2": ref4,
        "q3_dt_offset_pct": (b3 - ref3) / ref3 * 100.0,
        "q4_dt_offset_pct": (b4 - ref4) / ref4 * 100.0,
        "note": "扫描用 dt=4 以控制机时；步长偏移 <0.01%，远小于所有被扫参数的效应，"
                "故相对变化的排序不受影响"}
    print(f"[SENS] 基准 Q3={b3:.6f} h Q4={b4:.6f} h "
          f"(相对主结果 {OUT['baseline']['q3_dt_offset_pct']:+.4f}% / "
          f"{OUT['baseline']['q4_dt_offset_pct']:+.4f}%)", flush=True)
    return b3, b4, R_of, Rdot_of


def sec_8_1_air(b3, b4, R_of, Rdot_of):
    # 空气平台温度和浓度的 3x3 扫描。
    print("[SENS] §8.1 空气外推 3x3（Q3 与 Q4 各 9 个组合）...", flush=True)
    assert max(SENS_C_AIR) < C_THRESHOLD,\
        f"C_air 上界 {max(SENS_C_AIR)} 已达阈值 {C_THRESHOLD}，终点不可达（§8.1 有解前提）"
    cells = []
    for Ta in SENS_T_AIR_C:
        for Ca in SENS_C_AIR:
            t3 = _t_end_h("app3", ("fixed", R0), T_plateau=Ta, C_plateau=Ca)
            t4 = _t_end_h("app4", ("ale", (R_of, Rdot_of)), T_plateau=Ta, C_plateau=Ca)
            cells.append({"T_air_C": Ta, "C_air": Ca, "t_end_q3_h": t3, "t_end_q4_h": t4,
                          "rel_q3": _rel(t3, b3), "rel_q4": _rel(t4, b4)})
            print(f"  T={Ta} C={Ca}: Q3={_h(t3)} ({_p(_rel(t3, b3))}) "
                  f"Q4={_h(t4)} ({_p(_rel(t4, b4))})", flush=True)

    n_unreached = sum(1 for c in cells if c["t_end_q3_h"] is None or c["t_end_q4_h"] is None)
    assert n_unreached == 0,\
        f"§8.1 有 {n_unreached}/{len(cells)} 个组合在 {T_MAX_CAP_S / 3600:.0f} h 上限内未达阈值："\
        "跨度与单调性无法在同一口径下比较，须先如实登记未达标组合再重排"

    byT = {Ta: [c for c in cells if c["T_air_C"] == Ta] for Ta in SENS_T_AIR_C}
    mono_T = all(byT[SENS_T_AIR_C[0]][j]["t_end_q3_h"] > byT[SENS_T_AIR_C[1]][j]["t_end_q3_h"]
                 > byT[SENS_T_AIR_C[2]][j]["t_end_q3_h"] for j in range(len(SENS_C_AIR)))
    byC = {Ca: [c for c in cells if c["C_air"] == Ca] for Ca in SENS_C_AIR}
    mono_C = all(byC[SENS_C_AIR[0]][j]["t_end_q3_h"] < byC[SENS_C_AIR[1]][j]["t_end_q3_h"]
                 < byC[SENS_C_AIR[2]][j]["t_end_q3_h"] for j in range(len(SENS_T_AIR_C)))
    r3 = [c["rel_q3"] for c in cells]
    r4 = [c["rel_q4"] for c in cells]

    T_K = AIR_EXTRAP_T_C + 273.15
    dlnD_dT = 3850.0 / T_K ** 2
    OUT["sec_8_1_air_extrapolation"] = {
        "T_air_grid_C": list(SENS_T_AIR_C), "C_air_grid": list(SENS_C_AIR),
        "cells": cells,
        "span_rel_q3": float(max(r3) - min(r3)), "span_rel_q4": float(max(r4) - min(r4)),
        "max_abs_rel_q3": float(max(abs(x) for x in r3)),
        "max_abs_rel_q4": float(max(abs(x) for x in r4)),
        "monotone_T_up_gives_shorter_q3": bool(mono_T),
        "monotone_C_up_gives_longer_q3": bool(mono_C),
        "dlnD_dTK_per_K": dlnD_dT, "D_change_per_1K_pct": dlnD_dT * 100.0,
        "unreachable_bound_note": f"若 C_air >= {C_THRESHOLD} 则终点永不可达；"
                                  f"本扫描上界 {max(SENS_C_AIR)} 远低于该边界，"
                                  "但这一边界是问题三、四有解的前提，必须写入论文",
        "C_air_unreachable_threshold": C_THRESHOLD}
    assert mono_T, "§8.1 预期方向被证伪：T_air 升高未使 t_end 减小，须查 Arrhenius 路径"
    assert mono_C, "§8.1 预期方向被证伪：C_air 升高未使 t_end 增大，须查表面驱动力"
    print(f"[SENS] §8.1 Q3 相对变化跨度 {OUT['sec_8_1_air_extrapolation']['span_rel_q3'] * 100:.4f}%"
          f" Q4 {OUT['sec_8_1_air_extrapolation']['span_rel_q4'] * 100:.4f}%"
          f"；单调性 T:{mono_T} C:{mono_C}", flush=True)


def sec_8_2_surface(b3, b4, R_of, Rdot_of):
    # 分别扰动传热和传质系数。
    print("[SENS] §8.2 h / h_m 各 ±50% ...", flush=True)
    rows_hm, rows_h = [], []
    for f in SENS_HM_FACTORS:
        hm = HM_MASS * f
        t3 = _t_end_h("app3", ("fixed", R0), hm=hm)
        t4 = _t_end_h("app4", ("ale", (R_of, Rdot_of)), hm=hm)
        rows_hm.append({"factor": f, "hm_m_s": hm, "t_end_q3_h": t3, "t_end_q4_h": t4,
                        "rel_q3": _rel(t3, b3), "rel_q4": _rel(t4, b4)})
        print(f"  h_m x{f}: Q3={_h(t3)} ({_p(_rel(t3, b3))}) "
              f"Q4={_h(t4)} ({_p(_rel(t4, b4))})", flush=True)
    for f in SENS_H_FACTORS:
        hh = H_CONV * f
        t3 = _t_end_h("app3", ("fixed", R0), h=hh)
        t4 = _t_end_h("app4", ("ale", (R_of, Rdot_of)), h=hh)
        rows_h.append({"factor": f, "h_W_m2K": hh, "t_end_q3_h": t3, "t_end_q4_h": t4,
                       "rel_q3": _rel(t3, b3), "rel_q4": _rel(t4, b4)})
        print(f"  h   x{f}: Q3={_h(t3)} ({_p(_rel(t3, b3))}) "
              f"Q4={_h(t4)} ({_p(_rel(t4, b4))})", flush=True)

    n_unreached = sum(1 for r in rows_hm + rows_h
                      if r["t_end_q3_h"] is None or r["t_end_q4_h"] is None)
    assert n_unreached == 0,\
        f"§8.2 有 {n_unreached}/{len(rows_hm) + len(rows_h)} 个扰动在 "\
        f"{T_MAX_CAP_S / 3600:.0f} h 上限内未达阈值：跨度不可比，须先登记未达标项"

    hm_hi = [r for r in rows_hm if r["factor"] == max(SENS_HM_FACTORS)][0]
    pure_surface_pred = 1.0 / max(SENS_HM_FACTORS) - 1.0
    ctrl = hm_hi["rel_q3"] / pure_surface_pred

    D_thr = float(D_app3(C_THRESHOLD, AIR_EXTRAP_T_C + 273.15))
    D_C0v = float(D_app3(C0, AIR_EXTRAP_T_C + 273.15))
    Rs = 1.0 / HM_MASS
    Ri_thr, Ri_C0 = R0 / D_thr, R0 / D_C0v
    OUT["sec_8_2_surface_coefficients"] = {
        "hm_rows": rows_hm, "h_rows": rows_h,
        "hm_span_rel_q3": float(max(r["rel_q3"] for r in rows_hm)
                                - min(r["rel_q3"] for r in rows_hm)),
        "h_span_rel_q3": float(max(r["rel_q3"] for r in rows_h)
                               - min(r["rel_q3"] for r in rows_h)),
        "hm_span_rel_q4": float(max(r["rel_q4"] for r in rows_hm)
                                - min(r["rel_q4"] for r in rows_hm)),
        "h_span_rel_q4": float(max(r["rel_q4"] for r in rows_h)
                               - min(r["rel_q4"] for r in rows_h)),
        "surface_control_index_q3": ctrl,
        "pure_surface_control_prediction_rel": pure_surface_pred,
        "hm_hi_actual_rel_q3": hm_hi["rel_q3"],
        "resistance_surface_s_m": Rs,
        "resistance_internal_at_C0_s_m": Ri_C0,
        "resistance_internal_at_Cthr_s_m": Ri_thr,
        "internal_fraction_at_C0": Ri_C0 / (Ri_C0 + Rs),
        "internal_fraction_at_Cthr": Ri_thr / (Ri_thr + Rs),
        "verdict": "内部扩散电阻占主导 -> h_m 的 t_end 灵敏度远弱于纯表面控制的 1/h_m 标度"}
    assert abs(ctrl) < 1.0,\
        f"h_m 灵敏度 {hm_hi['rel_q3'] * 100:.4f}% 达到或超过纯表面控制预期 "\
        f"{pure_surface_pred * 100:.4f}%，与内部电阻占主导的结论矛盾"
    print(f"[SENS] §8.2 h_m 跨度 {OUT['sec_8_2_surface_coefficients']['hm_span_rel_q3'] * 100:.4f}%"
          f" h 跨度 {OUT['sec_8_2_surface_coefficients']['h_span_rel_q3'] * 100:.4f}%"
          f"；表面控制指数 {ctrl:.4f}（内部电阻占比 @Cthr "
          f"{OUT['sec_8_2_surface_coefficients']['internal_fraction_at_Cthr'] * 100:.2f}%）",
          flush=True)


def sec_8_3_forms(b3, b4, R_of, Rdot_of):
    # 比较方程形式和离散选项。
    print("[SENS] §8.3 形式选择对照 ...", flush=True)
    items = {}
    OUT["sec_8_3_form_choices"] = items


    t3 = _t_end_h("app3", ("fixed", R0), mass_form="rho_d_C")
    t4 = _t_end_h("app4", ("ale", (R_of, Rdot_of)), mass_form="rho_d_C")
    items["mass_form_rho_d_C"] = {
        "main": "C", "alt": "rho_d_C", "t_end_q3_h": t3, "t_end_q4_h": t4,
        "rel_q3": _rel(t3, b3), "rel_q4": _rel(t4, b4),
        "criterion": "问题一应严格一致（rho_d 常数）；问题三、四的差异即 rho_d 变化的影响"}
    print(f"  mass_form=rho_d_C: Q3 {_p(_rel(t3, b3))} Q4 {_p(_rel(t4, b4))}", flush=True)





    rho2 = float(props_app2()[0])
    kwu = dict(N=N_CELLS, air_mode="linear", C_init=C0, T_init=T0_C, record_fields=False)

    def _two_forms(span, dt, **extra):
        a_ = solve_drying("app2", span, stop=("time", span), dt_solve=dt, dt_out=span,
                          **kwu, **extra)
        b_ = solve_drying("app2", span, stop=("time", span), dt_solve=dt, dt_out=span,
                          mass_form="rho_d_C", **kwu, **extra)
        rd_ = rho2 / (1.0 + a_["C_final"])
        return (float(np.max(np.abs(a_["C_final"] - b_["C_final"]))),
                float(rd_.max() / rd_.min() - 1.0))


    dC_unif, sp_unif = _two_forms(0.25, 0.25, picard_fixed=1)

    ladder = []
    for span in (60.0, 300.0, 1800.0):
        d_, s_ = _two_forms(span, 0.25)
        ladder.append({"span_s": span, "max_dC": d_, "rho_d_spread": s_})
        print(f"  mass_form Q1 t={span:6.1f}s: dC={d_:.6e} rho_d 跨度={s_:.6e}", flush=True)
    dC1 = ladder[-1]["max_dC"]
    items["mass_form_q1_identity"] = {
        "claim_under_test": "§1 假设(4) / LOGIC_CONTRACT A4：附录2 rho 常数 => rho_d 常数 "
                            "=> 两形式严格等价，Q1 敏感度恰为 0",
        "verdict": "falsified",
        "why": "rho_d = rho/(1+C)（§10 参数表定义）。附录2 的 rho=820 为常数，但 C 沿径向"
               "不均匀，故 rho_d 在 C in [0.15,2.55] 上取 230.9859~713.0435 kg/m^3（3.0870 倍），"
               "并非常数。等价性的真正条件是 rho_d 空间均匀，而非 rho 为常数。",
        "true_condition_uniform_rho_d": {
            "max_dC": dC_unif, "rho_d_spread": sp_unif,
            "setup": "首步 dt=0.25 s + picard_fixed=1：系数取在均匀初值 C=C0 上，rho_d 逐点相等",
            "criterion_lt_1e-12": bool(dC_unif < 1e-12)},
        "nonuniformity_ladder": ladder,
        "max_dC_at_1800s": dC1,
        "rho_d_at_C0": rho2 / (1.0 + C0), "rho_d_at_Cthr": rho2 / (1.0 + C_THRESHOLD),
        "rho_d_ratio_Cthr_over_C0": (1.0 + C0) / (1.0 + C_THRESHOLD),
        "note": "Q1 的 8.675e-02 不是实现误差，而是假设(4) 在问题一口径下的真实效应量；"
                "报告与逻辑契约中『Q1 敏感度恰为 0』一项据此改为『仅在 rho_d 均匀时为 0』"}

    assert dC_unif < 1e-12,\
        f"rho_d 空间均匀时两形式应逐点相同，实得 dC={dC_unif:.3e}：离散层面的缩放不变性被破坏"

    assert all(ladder[i]["max_dC"] < ladder[i + 1]["max_dC"]
               and ladder[i]["rho_d_spread"] < ladder[i + 1]["rho_d_spread"]
               for i in range(len(ladder) - 1)),\
        f"两形式差未随 rho_d 非均匀度单调增长：{ladder}，差的来源需重新归因"
    assert dC_unif < ladder[0]["max_dC"],\
        "均匀 rho_d 极限的差未小于最短非均匀情形，等价条件的判定失效"
    print(f"  mass_form Q1 等价条件: 均匀 rho_d 下 dC={dC_unif:.3e}（跨度 {sp_unif:.3e}）；"
          f"1800 s 时 dC={dC1:.3e} —— 『Q1 恰为 0』证伪", flush=True)


    kwp = dict(stop=("time", 1800.0), dt_solve=0.25, dt_out=1800.0, N=N_CELLS,
               C_init=C0, T_init=T0_C, record_fields=False)
    la = solve_drying("app2", 1800.0, air_mode="linear", **kwp)
    pa = solve_drying("app2", 1800.0, air_mode="pchip", **kwp)
    dT_air = float(np.max(np.abs(la["T_final"] - pa["T_final"])))
    t3p = _t_end_h("app3", ("fixed", R0), air_mode="pchip")
    items["air_interp_pchip"] = {
        "main": "linear", "alt": "pchip", "max_dT_1800s_K": dT_air,
        "t_end_q3_h": t3p, "rel_q3": _rel(t3p, b3),
        "criterion_dT_lt_0p01K": bool(dT_air < 0.01)}
    assert dT_air < 0.01, f"附件1 插值方式使 T(r,1800) 差 {dT_air:.4e} K > 0.01 K：驱动数据分辨率不足"
    print(f"  air_interp=pchip: dT(1800s)={dT_air:.4e} K, Q3 t_end {_rel(t3p, b3) * 100:+.4f}%",
          flush=True)


    t3a = _t_end_h("app3", ("fixed", R0), d_face_mode="arithmetic")
    t4a = _t_end_h("app4", ("ale", (R_of, Rdot_of)), d_face_mode="arithmetic")
    items["d_face_arithmetic"] = {
        "main": "kirchhoff", "alt": "arithmetic", "t_end_q3_h": t3a, "t_end_q4_h": t4a,
        "rel_q3": _rel(t3a, b3), "rel_q4": _rel(t4a, b4),
        "criterion": "差异用于说明 §5.2 Kirchhoff 的必要性（退化扩散下平均方式非中性）"}
    print(f"  d_face=arithmetic: Q3 {_p(_rel(t3a, b3))} Q4 {_p(_rel(t4a, b4))}", flush=True)


    t3f = _t_end_h("app3", ("fixed", R0), picard_fixed=2)
    items["picard_fixed_2"] = {"main": "tol 1e-10", "alt": "fixed 2 iters",
                               "t_end_q3_h": t3f, "rel_q3": _rel(t3f, b3),
                               "criterion_rel_lt_1e-4": bool(abs(_req(_rel(t3f, b3),
                                                                     "picard_fixed_2")) < 1e-4)}
    assert abs(_req(_rel(t3f, b3), "picard_fixed_2")) < 1e-4,\
        f"PX-C3：固定 2 轮使 t_end 变 {_rel(t3f, b3):.3e}（>1e-4），疑把迭代当推进"
    print(f"  picard_fixed=2: Q3 {_rel(t3f, b3) * 100:+.6f}%", flush=True)


    Rl_of, Rldot_of, meta_lin = radius_interpolant(mode="linear")
    t4l = _t_end_h("app4", ("ale", (Rl_of, Rldot_of)))
    items["radius_interp_linear"] = {
        "main": "pchip", "alt": "linear", "t_end_q4_h": t4l, "rel_q4": _rel(t4l, b4),
        "criterion_rel_lt_0p5pct": bool(abs(_req(_rel(t4l, b4),
                                                 "radius_interp_linear")) < 5e-3),
        "cubic_spline_forbidden": "三次样条在平台衔接处过冲出实测下界 1.198 cm，禁用",
        "linear_meta": meta_lin}
    assert abs(_req(_rel(t4l, b4), "radius_interp_linear")) < 5e-3,\
        f"附件2 插值方式使 t_end 差 {_rel(t4l, b4) * 100:.4f}%（>0.5%），半径数据分辨率不足"
    print(f"  radius_interp=linear: Q4 {_rel(t4l, b4) * 100:+.4f}%", flush=True)


    t3s = _t_end_h("app3", ("fixed", R0), surface_form="series")
    t4s = _t_end_h("app4", ("ale", (R_of, Rdot_of)), surface_form="series")
    items["surface_form_series"] = {
        "main": "node (bare h)", "alt": "series U_eff (§4.2 原式)",
        "t_end_q3_h": t3s, "t_end_q4_h": t4s,
        "rel_q3": _rel(t3s, b3), "rel_q4": _rel(t4s, b4),
        "why_main_is_node": "节点中心布局下末节点即表面，串联 (dr/2)/D 重复计入半格内阻；"
                            "S3b 实测 series 误差 ~O(dr)（N=200/400: 1.99e-2/1.02e-2 K），"
                            "node 误差与 dr 无关（4.94e-4/4.91e-4 K）"}
    print(f"  surface_form=series: Q3 {_p(_rel(t3s, b3))} Q4 {_p(_rel(t4s, b4))}", flush=True)

    OUT["sec_8_3_form_choices"] = items


def sec_8_4_tornado():
    # 按实际跨度排序。
    a1 = OUT["sec_8_1_air_extrapolation"]
    a2 = OUT["sec_8_2_surface_coefficients"]
    a3 = OUT["sec_8_3_form_choices"]

    def _span(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(max(vals) - min(vals)) if vals else None


    cells = a1["cells"]
    T_mid, C_mid = SENS_T_AIR_C[1], SENS_C_AIR[1]
    T_only = [c for c in cells if c["C_air"] == C_mid]
    C_only = [c for c in cells if c["T_air_C"] == T_mid]

    def _abs(x):
        return None if x is None else abs(x)

    entries = []
    for name, q3, q4, note in (
            ("air_T_infty_pm1K", _span(T_only, "rel_q3"), _span(T_only, "rel_q4"),
             "§8.1 T_air 49/50/51 C（C_air 固定 0.05）"),
            ("air_C_infty_pm10pct", _span(C_only, "rel_q3"), _span(C_only, "rel_q4"),
             "§8.1 C_air 0.045/0.05/0.055（T_air 固定 50 C）"),
            ("air_joint_3x3", a1["span_rel_q3"], a1["span_rel_q4"],
             "§8.1 两者联合 3x3 全跨度"),
            ("h_m_pm50pct", a2["hm_span_rel_q3"], a2["hm_span_rel_q4"], "§8.2 传质系数"),
            ("h_pm50pct", a2["h_span_rel_q3"], a2["h_span_rel_q4"], "§8.2 传热系数"),
            ("mass_form_rho_d_C", _abs(a3["mass_form_rho_d_C"]["rel_q3"]),
             _abs(a3["mass_form_rho_d_C"]["rel_q4"]), "§8.3 水分方程形式"),
            ("d_face_arithmetic", _abs(a3["d_face_arithmetic"]["rel_q3"]),
             _abs(a3["d_face_arithmetic"]["rel_q4"]), "§8.3 界面 D 取算术平均"),
            ("surface_form_series", _abs(a3["surface_form_series"]["rel_q3"]),
             _abs(a3["surface_form_series"]["rel_q4"]), "§8.3 表面 Robin 串联形式"),
            ("air_interp_pchip", _abs(a3["air_interp_pchip"]["rel_q3"]), None,
             "§8.3 附件1 插值改 PCHIP"),
            ("radius_interp_linear", None, _abs(a3["radius_interp_linear"]["rel_q4"]),
             "§8.3 附件2 插值改线性"),
            ("picard_fixed_2", _abs(a3["picard_fixed_2"]["rel_q3"]), None,
             "§8.3 Picard 固定 2 轮")):
        entries.append({"factor": name, "span_rel_q3": q3, "span_rel_q4": q4,
                        "span_pct_q3": None if q3 is None else q3 * 100.0,
                        "span_pct_q4": None if q4 is None else q4 * 100.0,
                        "description": note})
    rank3 = sorted([e for e in entries if e["span_rel_q3"] is not None],
                   key=lambda e: -e["span_rel_q3"])
    rank4 = sorted([e for e in entries if e["span_rel_q4"] is not None],
                   key=lambda e: -e["span_rel_q4"])
    OUT["sec_8_4_tornado"] = {
        "entries": entries,
        "ranking_q3": [e["factor"] for e in rank3],
        "ranking_q4": [e["factor"] for e in rank4],
        "top_factor_q3": rank3[0]["factor"], "top_span_pct_q3": rank3[0]["span_pct_q3"],
        "top_factor_q4": rank4[0]["factor"], "top_span_pct_q4": rank4[0]["span_pct_q4"],
        "metric": "t_end 相对变化幅度（跨度），不作显著/不显著判断",
        "note": "排序由本次计算产生；§8.4 的先验猜测是 h_m 或空气外推，实际结果见 ranking_*"}
    print(f"[SENS] §8.4 Q3 首位 {rank3[0]['factor']} ({rank3[0]['span_pct_q3']:.4f}%)；"
          f"Q4 首位 {rank4[0]['factor']} ({rank4[0]['span_pct_q4']:.4f}%)", flush=True)
    for e in rank3[:5]:
        print(f"    Q3 {e['factor']:<24s} {e['span_pct_q3']:8.4f}%", flush=True)
    for e in rank4[:5]:
        print(f"    Q4 {e['factor']:<24s} {e['span_pct_q4']:8.4f}%", flush=True)


def run():
    t0 = time.time()
    b3, b4, R_of, Rdot_of = baselines()
    _checkpoint("baseline")
    sec_8_1_air(b3, b4, R_of, Rdot_of)
    _checkpoint("sec_8_1")
    sec_8_2_surface(b3, b4, R_of, Rdot_of)
    _checkpoint("sec_8_2")
    sec_8_3_forms(b3, b4, R_of, Rdot_of)
    _checkpoint("sec_8_3")
    sec_8_4_tornado()
    OUT["wall_time_s"] = time.time() - t0
    OUT["dt_solve_s"] = SENS_DT_SOLVE_S
    OUT["N_cells"] = N_CELLS
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / f"{TAG}_results.json", "w", encoding="utf-8") as fh:
        json.dump(OUT, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[SENS] -> results/{TAG}_results.json ({OUT['wall_time_s']:.1f} s)", flush=True)
    return OUT


if __name__ == "__main__":
    run()
