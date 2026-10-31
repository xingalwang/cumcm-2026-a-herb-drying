
from __future__ import annotations

import json
import numpy as np

from params import (C0, C_THRESHOLD, DECIMALS, FIGDIR, H_CONV, HM_MASS, KELVIN_OFFSET,
                    K_APP2, CP_APP2, RHO_APP2, N_CELLS, R0, T0_C)
from props import (D_app2, D_app3, D_app4, Phi_app2, PROPERTY_SETS, UnitError,
                   props_app2, props_app3, props_app4, to_kelvin)
from utils import (D_face, U_eff_series, air_driving, bessel_Bn, bessel_series_T,
                   check_geometry_identity, grid_weights, radius_interpolant,
                   solve_drying, volume_average)

RESULTS = {}


# 统一记录测试结果。
def _rec(name, passed, detail):
    RESULTS[name] = {"pass": bool(passed), **detail}
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {detail}", flush=True)
    if not passed:
        raise AssertionError(f"{name} 未通过：{detail}")


def S1_geometry_weights():
    # 体积权和应回到 R^2/2。
    r, dr, V, A_in = grid_weights(1.0, 4)
    tgt = {"V0": 0.0078125, "V1": 0.0625, "V4": 0.1171875, "sum": 0.5}
    got = {"V0": float(V[0]), "V1": float(V[1]), "V4": float(V[-1]), "sum": float(V.sum())}
    err = max(abs(got[k] - tgt[k]) for k in tgt)
    rel = check_geometry_identity(R0, N_CELLS)
    _rec("S1_geometry_weights", err < 1e-15 and rel < 1e-14,
         {"got": got, "max_abs_err": err, "sum_identity_rel_err_R0N200": rel})


def S2_zero_flux_constant():
    # 零通量时常场不能自行变化。
    res = solve_drying("app3", 600.0, stop=("time", 600.0), dt_solve=2.0, dt_out=600.0,
                       N=50, air_mode="const", T_plateau=T0_C, C_plateau=C0,
                       C_init=C0, T_init=T0_C, record_fields=False)
    dC = float(np.max(np.abs(res["C_final"] - C0)))
    dT = float(np.max(np.abs(res["T_final"] - T0_C)))
    _rec("S2_zero_flux_constant", dC < 1e-12 and dT < 1e-12,
         {"max_dC": dC, "max_dT": dT})


def _bessel_err(N, surface_form):
    rho, cp, k = props_app2()
    alpha = k / (rho * cp)
    Bi = H_CONV * R0 / k
    res = solve_drying("app2", 1800.0, stop=("time", 1800.0), dt_solve=0.125, dt_out=1800.0,
                       N=N, air_mode="const", T_plateau=50.0, C_plateau=C0,
                       C_init=C0, T_init=T0_C, record_fields=True,
                       surface_form=surface_form)
    r, dr, V, A_in = grid_weights(R0, N)
    T_ana, lam, Cn = bessel_series_T(r, 1800.0, Bi, alpha, R0, T0_C, 50.0, n_terms=60)
    return {"max_abs_err_K": float(np.max(np.abs(res["T_final"] - T_ana))),
            "T_center_num": float(res["T_final"][0]), "T_center_ana": float(T_ana[0]),
            "T_surf_num": float(res["T_final"][-1]), "T_surf_ana": float(T_ana[-1]),
            "Bi": Bi, "alpha": alpha, "lam_1to5": [float(x) for x in lam[:5]]}


def S3_bessel_limit():
    # 与圆柱 Bessel 级数比较。
    d = _bessel_err(400, "node")
    sB, sC, lam5 = bessel_Bn(Bi_of := H_CONV * R0 / props_app2()[2], 40)
    d.update({"sum_Bn": sB, "sum_Cn": sC, "Bi_check": Bi_of})
    _rec("S3_bessel_limit", d["max_abs_err_K"] < 1e-3 and abs(sB - 1.0) < 1e-5, d)


def S3b_surface_form_order():
    ev = {}
    for form in ("node", "series"):
        for N in (200, 400):
            ev[f"{form}_N{N}"] = _bessel_err(N, form)
    e_node = [ev[f"node_N{N}"]["max_abs_err_K"] for N in (200, 400)]
    e_ser = [ev[f"series_N{N}"]["max_abs_err_K"] for N in (200, 400)]
    node_flat = abs(e_node[1] - e_node[0]) / e_node[0] < 0.05
    ser_halves = 1.6 < e_ser[0] / e_ser[1] < 2.4
    _rec("S3b_surface_form_order", node_flat and ser_halves and e_node[1] < e_ser[1] / 5.0,
         {"err_node_N200_N400": e_node, "err_series_N200_N400": e_ser,
          "node_rel_change": abs(e_node[1] - e_node[0]) / e_node[0],
          "series_ratio_N200_over_N400": e_ser[0] / e_ser[1],
          "T_surf_ana": ev["node_N400"]["T_surf_ana"],
          "T_surf_node_N400": ev["node_N400"]["T_surf_num"],
          "T_surf_series_N400": ev["series_N400"]["T_surf_num"],
          "verdict": "node-centered FV must use bare h; series U_eff is the cell-centered form"})


def S4_Ueff_limits():
    dr = R0 / N_CELLS
    u1 = float(U_eff_series(1e30, dr / 2.0, HM_MASS))
    u2 = float(U_eff_series(5e-9, dr / 2.0, 1e30))
    e1 = abs(u1 - HM_MASS) / HM_MASS
    e2 = abs(u2 - 5e-9 / (dr / 2.0)) / (5e-9 / (dr / 2.0))
    _rec("S4_Ueff_limits", e1 < 1e-10 and e2 < 1e-10,
         {"rel_err_hm_limit": e1, "rel_err_diff_limit": e2, "U_eff_hm": u1, "U_eff_D": u2})


def S5_kirchhoff_consistency():
    # 检查 Kirchhoff 势的差商。
    Ca, Cb = 0.08, 0.05
    ps = PROPERTY_SETS["app2"]
    Carr = np.array([Ca, Cb])
    dfc = float(D_face(Carr, ps)[0])

    ss = np.linspace(1e-9, Ca, 400001)
    Phi_num = float(np.trapezoid(D_app2(ss), ss))
    Phi_cf = float(Phi_app2(Ca))
    rel_phi = abs(Phi_num - Phi_cf) / abs(Phi_cf)

    eq = float(D_face(np.array([0.3, 0.3]), ps)[0])
    rel_eq = abs(eq - float(D_app2(0.3))) / float(D_app2(0.3))
    Da, Db = float(D_app2(Ca)), float(D_app2(Cb))
    harm = 2 * Da * Db / (Da + Db)
    arith = 0.5 * (Da + Db)
    _rec("S5_kirchhoff_consistency", rel_phi < 1e-6 and rel_eq < 1e-12,
         {"rel_err_Phi_vs_quad": rel_phi, "rel_err_equal_C_degeneracy": rel_eq,
          "D_face_kirchhoff": dfc, "harmonic": harm, "arithmetic": arith,
          "kirchhoff_over_harmonic": dfc / harm, "kirchhoff_over_arith": dfc / arith})


def S6_ale_rigid_scaling():
    R_of, Rdot_of, meta = radius_interpolant()
    res = solve_drying("app4", 7200.0, stop=("time", 7200.0), dt_solve=2.0, dt_out=7200.0,
                       geometry=("ale", (R_of, Rdot_of)), N=50, air_mode="const",
                       T_plateau=T0_C, C_plateau=C0, hm=0.0 + 1e-300, h=1e-300,
                       C_init=C0, T_init=T0_C, record_fields=False)
    dC = float(np.max(np.abs(res["C_final"] - C0)))
    frac_active = res["v_mesh_active_steps"] / max(res["n_steps"], 1)
    _rec("S6_ale_rigid_scaling",
         dC < 1e-10 and res["motion_resid_max"] < 1e-12 and frac_active > 0.5,
         {"max_dC": dC, "motion_cancel_resid_max": res["motion_resid_max"],
          "v_mesh_active_fraction": frac_active, "radius_meta": meta})


def S7_first_passage():
    # 同时测首达和回吸。



    R_small = 0.002
    kw = dict(stop=("first_passage", C_THRESHOLD), dt_out=60.0, N=100,
              geometry=("fixed", R_small), air_mode="const", T_plateau=50.0,
              C_plateau=0.05, hm=8e-5, C_init=0.5, T_init=50.0, record_fields=False)
    a = solve_drying("app3", 20000.0, dt_solve=4.0, **kw)
    b = solve_drying("app3", 20000.0, dt_solve=1.0, **kw)
    assert a["t_end"] is not None and b["t_end"] is not None, "S7 算例未达标，测试设计失效"
    rel = abs(a["t_end"] - b["t_end"]) / b["t_end"]

    import utils as U
    T_of, _ = U.air_driving(50.0, 0.05, mode="const")

    t_jump = 1.2 * a["t_end"]

    def C_air_rebound(tq):
        tq = np.asarray(tq, dtype=float)
        return np.where(tq < t_jump, 0.02, 0.60)

    step = U.make_stepper("app3", T_of, C_air_rebound, geometry=("fixed", R_small),
                          N=100, hm=8e-5)
    C, T, t = np.full(101, 0.5), np.full(101, 50.0), 0.0
    seen_below, seen_rebound = False, False
    while t < 3.0 * a["t_end"]:
        C, T, _ = step(C, T, t, 4.0)
        t += 4.0
        g = float(np.max(C)) - C_THRESHOLD
        if g < 0:
            seen_below = True
        elif seen_below:
            seen_rebound = True
            break
    _rec("S7_first_passage", rel < 5e-3 and a["bracket"] is not None and seen_rebound,
         {"t_end_dt4_s": a["t_end"], "t_end_dt1_s": b["t_end"], "rel_diff": rel,
          "bracket_len_s": a["bracket"][1] - a["bracket"][0],
          "rebound_case_detected": seen_rebound, "no_rebound_main": a["no_rebound"]})


def S8_unit_misuse_probe():
    # 摄氏度误传必须被拦截。
    raised = {}
    for name, fn in (("D_app3", D_app3), ("D_app4", D_app4)):
        try:
            fn(C0, 50.0)
            raised[name] = False
        except UnitError:
            raised[name] = True

    ok3 = float(D_app3(C0, to_kelvin(50.0)))
    ok4 = float(D_app4(C0, to_kelvin(50.0)))
    silent = float(2.4e-3 * np.exp(-0.45 / C0) * np.exp(-3850.0 / 50.0))
    _rec("S8_unit_misuse_probe", all(raised.values()) and 1e-9 < ok3 < 1e-7,
         {"raised": raised, "D_app3_K": ok3, "D_app4_K": ok4,
          "what_silent_bug_would_give": silent, "ratio_app4_app3": ok4 / ok3})


def main():
    print("=== 结构性测试 S1-S8（真实算例前置门）===", flush=True)
    S1_geometry_weights()
    S2_zero_flux_constant()
    S3_bessel_limit()
    S3b_surface_form_order()
    S4_Ueff_limits()
    S5_kirchhoff_consistency()
    S6_ale_rigid_scaling()
    S7_first_passage()
    S8_unit_misuse_probe()
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / "structural_tests.json", "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, ensure_ascii=False, indent=2, default=float)
    print(f"\n全部 {len(RESULTS)} 项结构性测试通过 -> results/structural_tests.json", flush=True)


if __name__ == "__main__":
    main()
