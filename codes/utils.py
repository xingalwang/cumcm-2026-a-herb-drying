
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros

import pandas as pd

from params import (AIR_COLS, AIR_EXTRAP_C, AIR_EXTRAP_T_C, AIR_T_END_S, AIR_XLSX,
                    BISECT_TOL_S, C0, C_THRESHOLD, DECIMALS, H_CONV, HM_MASS,
                    INDEX_HEADER, KELVIN_OFFSET, N_CELLS, PICARD_MAX, PICARD_TOL_C,
                    PICARD_TOL_T, R0, RADIUS_COLS, RADIUS_T_END_S, RADIUS_XLSX,
                    REBOUND_FACTOR, T0_C, T_MAX_CAP_S)
from props import PROPERTY_SETS, arrhenius, to_kelvin


# 网格权重
def grid_weights(R, N):
    dr = R / N
    r = np.arange(N + 1, dtype=float) * dr
    V = np.empty(N + 1, dtype=float)
    V[0] = dr ** 2 / 8
    V[1:N] = r[1:N] * dr
    V[-1] = (dr / 2) * (R - dr / 4)
    A_in = (np.arange(N, dtype=float) + 0.5) * dr
    return r, dr, V, A_in


def check_geometry_identity(R, N):
    _, _, V, _ = grid_weights(R, N)
    target = R ** 2 / 2
    return abs(V.sum() - target) / target


def volume_average(C, V):
    return float(np.dot(V, C) / V.sum())


# 界面扩散系数
def D_face(C, pset, T_face_K=None, mode="kirchhoff"):
    Cl, Cr = C[:-1], C[1:]
    if mode == "arithmetic":
        if pset["b"] is None:
            return np.maximum(0.5 * (pset["D"](Cl) + pset["D"](Cr)), 0.0)
        return np.maximum(0.5 * (pset["D"](Cl, T_face_K) + pset["D"](Cr, T_face_K)), 0.0)
    if mode != "kirchhoff":
        raise ValueError(f"未知界面 D 取法 {mode!r}，只允许 'kirchhoff' / 'arithmetic'")
    if pset["b"] is None:
        Phi_l = pset["Phi"](Cl)
        Phi_r = pset["Phi"](Cr)
        D_l = pset["D"](Cl)
    else:
        fT = arrhenius(T_face_K, pset["b"])
        Phi_l = pset["Phi"](Cl, fT)
        Phi_r = pset["Phi"](Cr, fT)
        D_l = pset["D"](Cl, T_face_K)
    dC = Cr - Cl
    out = np.where(np.abs(dC) > 0.0, (Phi_r - Phi_l) / np.where(dC == 0.0, 1.0, dC), D_l)
    return np.maximum(out, 0.0)


def k_face(k_nodes):
    return (k_nodes[:-1] + k_nodes[1:]) * 0.5


def U_eff_series(D_surf, half_thickness, h_ext):
    D_surf = np.maximum(np.asarray(D_surf, dtype=float), 1e-300)
    return 1.0 / (half_thickness / D_surf + 1.0 / h_ext)


def U_surface(Dif_surf, half_thickness, h_ext, form="node"):
    if form == "node":
        return float(h_ext)
    if form == "series":
        return float(U_eff_series(Dif_surf, half_thickness, h_ext))
    raise ValueError(f"未知表面形式 {form!r}，只允许 'node' / 'series'")


# 三对角线性系统
def assemble_solve(u_old, cap, V, A_in, Dif, dt, U_surf, u_inf, A_surf, source=None):
    n = u_old.size
    ab = np.zeros((3, n), dtype=float)
    diag = V * cap / dt
    rhs = diag * u_old
    ab[1, :] = diag
    coef = A_in * Dif
    ab[1, :-1] += coef
    ab[0, 1:] = -coef
    ab[1, 1:] += coef
    ab[2, :-1] = -coef
    ab[1, -1] += A_surf * U_surf
    rhs[-1] += A_surf * U_surf * u_inf
    if source is not None:
        rhs = rhs + V * source
    return solve_banded((1, 1), ab, rhs)



_AIR_CACHE = {}


# 附件 1
def load_air(mode="linear"):
    key = ("air", mode)
    if key in _AIR_CACHE:
        return _AIR_CACHE[key]
    df = pd.read_excel(AIR_XLSX, sheet_name=0)
    cols = list(df.columns)
    assert cols == AIR_COLS, f"附件1 列名不符：{cols} != {AIR_COLS}"
    t = df[AIR_COLS[0]].to_numpy(dtype=float)
    Ta = df[AIR_COLS[1]].to_numpy(dtype=float)
    Ca = df[AIR_COLS[2]].to_numpy(dtype=float)
    assert t.size == 241 and abs(t[-1] - AIR_T_END_S) < 1e-9, "附件1 行数/量程不符"
    assert np.all(np.diff(t) > 0), "附件1 时间列非严格递增"
    _AIR_CACHE[key] = (t, Ta, Ca)
    return t, Ta, Ca


def air_driving(T_plateau=AIR_EXTRAP_T_C, C_plateau=AIR_EXTRAP_C, mode="linear"):
    t, Ta, Ca = load_air()
    if mode == "linear":
        def T_of(tq):
            tq = np.asarray(tq, dtype=float)
            return np.where(tq <= AIR_T_END_S, np.interp(np.minimum(tq, AIR_T_END_S), t, Ta), T_plateau)

        def C_of(tq):
            tq = np.asarray(tq, dtype=float)
            return np.where(tq <= AIR_T_END_S, np.interp(np.minimum(tq, AIR_T_END_S), t, Ca), C_plateau)
    elif mode == "pchip":
        pT, pC = PchipInterpolator(t, Ta), PchipInterpolator(t, Ca)

        def T_of(tq):
            tq = np.asarray(tq, dtype=float)
            return np.where(tq <= AIR_T_END_S, pT(np.minimum(tq, AIR_T_END_S)), T_plateau)

        def C_of(tq):
            tq = np.asarray(tq, dtype=float)
            return np.where(tq <= AIR_T_END_S, pC(np.minimum(tq, AIR_T_END_S)), C_plateau)
    elif mode == "const":
        def T_of(tq):
            return np.full_like(np.asarray(tq, dtype=float), T_plateau)

        def C_of(tq):
            return np.full_like(np.asarray(tq, dtype=float), C_plateau)
    else:
        raise ValueError(f"未知空气插值模式 {mode}")
    return T_of, C_of



def load_radius():
    df = pd.read_excel(RADIUS_XLSX, sheet_name=0)
    cols = list(df.columns)
    assert cols == RADIUS_COLS, f"附件2 列名不符：{cols} != {RADIUS_COLS}"
    t = df[RADIUS_COLS[0]].to_numpy(dtype=float)
    R_cm = df[RADIUS_COLS[1]].to_numpy(dtype=float)
    assert t.size == 145 and abs(t[-1] - RADIUS_T_END_S) < 1e-9, "附件2 行数/量程不符"
    assert np.all(np.diff(R_cm) <= 1e-12), "附件2 半径应单调不增"
    return t, R_cm / 100.0


# 附件 2
def radius_interpolant(mode="pchip"):
    t, R = load_radius()
    if mode == "linear":
        slope = np.diff(R) / np.diff(t)

        class _Lin:
            def __call__(self, tq):
                return np.interp(tq, t, R)

            def derivative(self):
                def d(tq):
                    j = np.clip(np.searchsorted(t, np.asarray(tq, dtype=float),
                                                side="right") - 1, 0, slope.size - 1)
                    return slope[j]
                return d
        pch = _Lin()
        dpch = pch.derivative()
    elif mode == "pchip":
        pch = PchipInterpolator(t, R)
        dpch = pch.derivative()
    else:
        raise ValueError(f"未知半径插值模式 {mode!r}，只允许 'pchip' / 'linear'")
    R_end = float(R[-1])

    def R_of(tq):
        tq = np.asarray(tq, dtype=float)
        return np.where(tq >= RADIUS_T_END_S, R_end, pch(np.clip(tq, 0.0, RADIUS_T_END_S)))

    def Rdot_of(tq):
        tq = np.asarray(tq, dtype=float)
        return np.where(tq >= RADIUS_T_END_S, 0.0, dpch(np.clip(tq, 0.0, RADIUS_T_END_S)))

    meta = {"R_start_m": float(R[0]), "R_end_m": R_end,
            "ptp_cm": float((R.max() - R.min()) * 100.0),
            "max_abs_step_cm": float(np.max(np.abs(np.diff(R))) * 100.0),
            "plateau_start_s": float(t[np.argmax(R <= R_end + 1e-15)]),
            "mode": mode}
    return R_of, Rdot_of, meta



def _ale_motion_terms(C, r, R, Rdot, dr):
    dCdr = np.empty_like(C)
    dCdr[1:-1] = (C[2:] - C[:-2]) / (2.0 * dr)
    dCdr[0] = (C[1] - C[0]) / dr
    dCdr[-1] = (C[-1] - C[-2]) / dr
    v_mesh = r * Rdot / R
    v_solid = r * Rdot / R
    term_mesh_motion = -v_mesh * dCdr
    term_solid_advection = v_solid * dCdr
    return term_mesh_motion, term_solid_advection, float(np.max(np.abs(v_mesh)))


def _dry_density(C, props_fn, coupling):
    if coupling == "decoupled":
        rho = np.full_like(np.asarray(C, dtype=float), props_fn()[0])
    else:
        rho = np.asarray(props_fn(C)[0], dtype=float)
    return rho / (1.0 + np.asarray(C, dtype=float))


def make_stepper(pset_name, T_air_of, C_air_of, geometry=("fixed", R0), N=N_CELLS,
                 h=H_CONV, hm=HM_MASS, picard_tol_C=PICARD_TOL_C,
                 picard_tol_T=PICARD_TOL_T, picard_max=PICARD_MAX,
                 mass_form="C", picard_fixed=None, solve_order="C_first",
                 surface_form="node", d_face_mode="kirchhoff"):
    pset = PROPERTY_SETS[pset_name]
    coupling = pset["coupling"]
    props_fn, D_fn = pset["props"], pset["D"]
    geom_mode, geom_arg = geometry[0], geometry[1]
    if geom_mode == "ale":
        R_of, Rdot_of = geom_arg
    else:
        R_const = float(geom_arg)

    def step(C_old, T_old, t_old, dt):
        t_new = t_old + dt
        if geom_mode == "ale":
            R = float(R_of(t_new))
            Rdot = float(Rdot_of(t_new))
        else:
            R, Rdot = R_const, 0.0
        r, dr, V, A_in = grid_weights(R, N)
        A_surf = R
        Ta = float(T_air_of(t_new))
        Ca = float(C_air_of(t_new))

        if geom_mode == "ale":
            term_mesh_motion, term_solid_advection, v_max = _ale_motion_terms(C_old, r, R, Rdot, dr)
            motion_src = term_mesh_motion + term_solid_advection
            motion_resid = float(np.linalg.norm(motion_src))
        else:
            motion_src = None
            motion_resid, v_max = 0.0, 0.0

        C_iter = C_old.copy()
        T_iter = T_old.copy()
        n_it, dC_last, dT_last = 0, np.inf, np.inf
        resid_hist = []
        n_target = picard_max if picard_fixed is None else picard_fixed
        for n_it in range(1, n_target + 1):

            T_face_K = to_kelvin(0.5 * (T_iter[:-1] + T_iter[1:]))
            if coupling == "decoupled":
                Dfc = D_face(C_iter, pset, mode=d_face_mode)
                D_surf = float(D_fn(C_iter[-1]))
                rho_s, cp_s, k_s = props_fn()
                cap_T = np.full_like(T_iter, rho_s * cp_s)
                kfc = np.full(N, k_s)
                k_surf = k_s
            else:
                Dfc = D_face(C_iter, pset, T_face_K, mode=d_face_mode)
                D_surf = float(D_fn(C_iter[-1], to_kelvin(T_iter[-1])))
                rho_n, cp_n, k_n = props_fn(C_iter)
                cap_T = rho_n * cp_n
                kfc = k_face(k_n)
                k_surf = float(k_n[-1])
            U_m = U_surface(D_surf, dr / 2.0, hm, surface_form)
            U_t = U_surface(k_surf, dr / 2.0, h, surface_form)

            if mass_form == "rho_d_C":
                rd = _dry_density(C_iter, props_fn, coupling)
                cap_C = rd
                Dif_C = (rd[:-1] + rd[1:]) * 0.5 * Dfc / dr
                U_m_eff = U_m * float(rd[-1])
            else:
                cap_C = np.ones_like(C_iter)
                Dif_C = Dfc / dr
                U_m_eff = U_m

            def solve_C():



                return assemble_solve(C_old, cap_C, V, A_in, Dif_C, dt, U_m_eff, Ca,
                                      A_surf, source=motion_src)

            def solve_T():
                return assemble_solve(T_old, cap_T, V, A_in, kfc / dr, dt, U_t, Ta, A_surf)

            if solve_order == "C_first":
                C_new = solve_C()
                T_new = solve_T()
            else:
                T_new = solve_T()
                C_new = solve_C()

            dC_last = float(np.max(np.abs(C_new - C_iter)))
            dT_last = float(np.max(np.abs(T_new - T_iter)))
            resid_hist.append((dC_last, dT_last))
            C_iter, T_iter = C_new, T_new
            if picard_fixed is None and dC_last < picard_tol_C and dT_last < picard_tol_T:
                break

        info = {"n_picard": n_it, "dC": dC_last, "dT": dT_last, "R": R, "Rdot": Rdot,
                "U_m": U_m, "U_t": U_t, "Ta": Ta, "Ca": Ca, "V": V, "dr": dr, "r": r,
                "A_in": A_in, "A_surf": A_surf, "motion_resid": motion_resid,
                "v_mesh_max": v_max, "resid_hist": resid_hist,
                "surface_form": surface_form,
                "converged": (dC_last < picard_tol_C and dT_last < picard_tol_T)}
        return C_iter, T_iter, info

    return step


def solve_drying(pset_name, t_span, stop=("time", None), geometry=("fixed", R0),
                 dt_solve=0.25, dt_out=None, N=N_CELLS, air_mode="linear",
                 T_plateau=AIR_EXTRAP_T_C, C_plateau=AIR_EXTRAP_C, h=H_CONV, hm=HM_MASS,
                 C_init=C0, T_init=T0_C, mass_form="C", picard_fixed=None,
                 solve_order="C_first", record_fields=True, rebound_check=True,
                 surface_form="node", d_face_mode="kirchhoff", t_cap=T_MAX_CAP_S,
                 progress=None):
    T_air_of, C_air_of = air_driving(T_plateau, C_plateau, mode=air_mode)
    step = make_stepper(pset_name, T_air_of, C_air_of, geometry=geometry, N=N, h=h, hm=hm,
                        mass_form=mass_form, picard_fixed=picard_fixed,
                        solve_order=solve_order, surface_form=surface_form,
                        d_face_mode=d_face_mode)
    thr = None if stop[0] == "time" else float(stop[1])


    t_limit = float(t_span) if stop[0] == "time" else min(float(t_span), float(t_cap))
    if dt_out is None:
        dt_out = dt_solve
    stride = int(round(dt_out / dt_solve))
    assert abs(stride * dt_solve - dt_out) < 1e-12, "dt_out 必须被 dt_solve 整除"

    C = np.full(N + 1, float(C_init))
    T = np.full(N + 1, float(T_init))
    t = 0.0
    out_t, out_C, out_T, out_R = [], [], [], []
    hist = {"t": [0.0], "C_max": [float(np.max(C))], "C_bar": [], "T_center": [float(T[0])],
            "T_surface": [float(T[-1])], "C_center": [float(C[0])],
            "C_surface": [float(C[-1])], "n_picard": [], "argmax_ok": [],
            "water_resid": [], "R": [], "D_ratio": []}
    R_init = float(geometry[1][0](0.0)) if geometry[0] == "ale" else float(geometry[1])
    _, _, V0w, _ = grid_weights(R_init, N)
    hist["C_bar"].append(volume_average(C, V0w))
    hist["R"].append(R_init)
    n_pic_max, motion_resid_max, v_mesh_active = 0, 0.0, 0
    n_steps, argmax_all_center = 0, True
    hump_tol = 1e-12
    argmax_center_within_tol, C_hump_excess_max, hump_first_t = True, 0.0, None
    t_end, bracket, g_prev = None, None, float(np.max(C)) - (thr if thr else 0.0)
    state_end = None
    state_prev = (C.copy(), T.copy(), 0.0)
    D_field_ratio_max = 0.0
    water_resid_max = 0.0

    while t < t_limit - 1e-12:
        dt = min(dt_solve, t_limit - t)
        C_prev_arr, T_prev_arr, t_prev = C.copy(), T.copy(), t
        C, T, info = step(C_prev_arr, T_prev_arr, t, dt)
        t += dt
        n_steps += 1
        n_pic_max = max(n_pic_max, info["n_picard"])
        motion_resid_max = max(motion_resid_max, info["motion_resid"])
        if info["v_mesh_max"] > 0.0:
            v_mesh_active += 1

        Vv = info["V"]
        lhs = float(np.dot(Vv, C - C_prev_arr) / dt)
        rhs_flux = -info["A_surf"] * info["U_m"] * (float(C[-1]) - info["Ca"])
        scale = max(abs(lhs), abs(rhs_flux), 1e-300)
        wres = abs(lhs - rhs_flux) / scale
        water_resid_max = max(water_resid_max, wres)
        if np.min(C) > 0:
            D_field_ratio_max = max(D_field_ratio_max, float(np.max(C) / np.min(C)))
        Cmax = float(np.max(C))




        hump = Cmax - float(C[0])
        C_hump_excess_max = max(C_hump_excess_max, hump)
        if int(np.argmax(C)) != 0:
            argmax_all_center = False
            if hump > hump_tol:
                argmax_center_within_tol = False
                if hump_first_t is None:
                    hump_first_t = t

        if n_steps % stride == 0 and record_fields:
            out_t.append(t)
            out_C.append(C.copy())
            out_T.append(T.copy())
            out_R.append(info["R"])
        if n_steps % max(1, stride) == 0:
            hist["t"].append(t)
            hist["C_max"].append(Cmax)
            hist["C_bar"].append(volume_average(C, Vv))
            hist["T_center"].append(float(T[0]))
            hist["T_surface"].append(float(T[-1]))
            hist["C_center"].append(float(C[0]))
            hist["C_surface"].append(float(C[-1]))
            hist["n_picard"].append(info["n_picard"])
            hist["water_resid"].append(wres)
            hist["R"].append(info["R"])

        if thr is not None and t_end is None:
            g_now = Cmax - thr
            if g_prev >= 0.0 > g_now:
                bracket = (t_prev, t, g_prev, g_now)
                t_end, state_end = _bisect_first_passage(step, C_prev_arr, T_prev_arr,
                                                         t_prev, dt, thr)
                if not rebound_check:
                    break
            g_prev = g_now
        if thr is not None and t_end is not None and rebound_check:
            if t >= REBOUND_FACTOR * t_end:
                break
        if progress is not None and n_steps % progress == 0:
            print(f"    t={t:9.1f}s  maxC={Cmax:.6f}  Tc={T[0]:7.3f}  npic={info['n_picard']}",
                  flush=True)

    no_rebound = True
    if thr is not None and t_end is not None:
        tail = [c for tt, c in zip(hist["t"], hist["C_max"]) if tt >= t_end]
        no_rebound = bool(all(c < thr for c in tail)) if tail else True

    return {"t_out": np.asarray(out_t), "C_out": np.asarray(out_C) if out_C else np.zeros((0, N + 1)),
            "T_out": np.asarray(out_T) if out_T else np.zeros((0, N + 1)),
            "R_out": np.asarray(out_R), "C_final": C, "T_final": T, "t_final": t,
            "hist": hist, "t_end": t_end, "bracket": bracket, "no_rebound": no_rebound,
            "C_at_t_end": None if state_end is None else state_end[0],
            "T_at_t_end": None if state_end is None else state_end[1],
            "t_at_state_end": None if state_end is None else state_end[2],
            "n_steps": n_steps, "n_picard_max": n_pic_max,
            "motion_resid_max": motion_resid_max, "v_mesh_active_steps": v_mesh_active,
            "argmax_center_always": argmax_all_center,
            "argmax_center_within_tol": argmax_center_within_tol,
            "C_hump_excess_max": C_hump_excess_max, "hump_tol": hump_tol,
            "hump_first_t_s": hump_first_t, "water_resid_max": water_resid_max,
            "D_field_ratio_max": D_field_ratio_max, "N": N, "dt_solve": dt_solve,
            "dt_out": dt_out, "stride": stride, "surface_form": surface_form}


# 首达时刻的局部二分
def _bisect_first_passage(step, C0_arr, T0_arr, t0, dt, thr):
    lo, hi = 0.0, dt
    sub = max(dt / 16.0, 1e-6)
    state_hi = None
    while hi - lo > BISECT_TOL_S:
        mid = 0.5 * (lo + hi)
        C, T, tt = C0_arr.copy(), T0_arr.copy(), t0
        remaining = mid
        while remaining > 1e-12:
            h_step = min(sub, remaining)
            C, T, _ = step(C, T, tt, h_step)
            tt += h_step
            remaining -= h_step
        if float(np.max(C)) - thr < 0.0:
            hi = mid
            state_hi = (C.copy(), T.copy(), t0 + mid)
        else:
            lo = mid
    return t0 + 0.5 * (lo + hi), state_hi


# 解析解校核
def bessel_eigenvalues(Bi, n_terms=40):
    z0 = jn_zeros(0, n_terms + 2)
    f = lambda x: x * j1(x) - Bi * j0(x)
    roots = []
    lo = 1e-9
    for k in range(n_terms):
        hi = z0[k] - 1e-9
        if f(lo) * f(hi) < 0:
            roots.append(brentq(f, lo, hi, xtol=1e-14, rtol=1e-15))
        lo = z0[k] + 1e-9
    return np.asarray(roots[:n_terms])


def bessel_series_T(r, t, Bi, alpha, R, T0, T_inf, n_terms=40):
    lam = bessel_eigenvalues(Bi, n_terms)
    Fo = alpha * np.asarray(t, dtype=float) / R ** 2
    rr = np.asarray(r, dtype=float) / R
    Cn = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * j0(lam))
    theta = np.zeros(np.shape(rr), dtype=float)
    for n in range(lam.size):
        theta = theta + Cn[n] * j0(lam[n] * rr) * np.exp(-lam[n] ** 2 * Fo)
    return T_inf + (T0 - T_inf) * theta, lam, Cn


def bessel_Bn(Bi, n_terms=40):
    lam = bessel_eigenvalues(Bi, n_terms)
    Cn = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * j0(lam))
    Bn = 2.0 * Cn * j1(lam) / lam
    return float(np.sum(Bn)), float(np.sum(Cn)), lam



def duhamel_T(r, t_query, Bi, alpha, R, T0, T_air_of, n_terms=40, n_sub=4000):
    lam = bessel_eigenvalues(Bi, n_terms)
    Cn = 2.0 * Bi / ((lam ** 2 + Bi ** 2) * j0(lam))
    rr = np.atleast_1d(np.asarray(r, dtype=float)) / R
    basis = np.array([j0(l * rr) for l in lam])
    out = []
    for tq in np.atleast_1d(np.asarray(t_query, dtype=float)):
        s = np.linspace(0.0, tq, n_sub + 1)
        Ta = np.asarray(T_air_of(s), dtype=float)
        dTa = np.gradient(Ta, s, edge_order=2)

        Fo = alpha * (tq - s) / R ** 2
        decay = np.exp(-np.outer(lam ** 2, Fo))
        theta = np.tensordot(Cn[:, None] * basis, decay, axes=([0], [0]))
        g = 1.0 - theta
        conv = np.trapezoid(g * dTa[None, :], s, axis=1)
        theta_t0 = np.tensordot(Cn[:, None] * basis, np.exp(-lam ** 2 * alpha * tq / R ** 2),
                               axes=([0], [0]))
        out.append(T0 + (Ta[0] - T0) * (1.0 - theta_t0) + conv)
    res = np.asarray(out)
    return res[0] if res.shape[0] == 1 else res



def convergence_ratios(values):
    v = [float(x) for x in values]
    diffs = [v[i + 1] - v[i] for i in range(len(v) - 1)]
    ratio = [(diffs[i] / diffs[i + 1]) if abs(diffs[i + 1]) > 0 else float("inf")
             for i in range(len(diffs) - 1)]
    rel = [abs(diffs[i]) / max(abs(v[i + 1]), 1e-300) for i in range(len(diffs))]
    return {"values": v, "diffs": diffs, "ratio": ratio, "rel_change": rel}


# 输出列和表格
def output_columns_cm(n_cols=21, dr_cm=0.1):
    return [round(i * dr_cm, 1) for i in range(n_cols)]


def extract_output_nodes(field_rows, stride):
    return np.asarray(field_rows)[:, ::stride]


def write_xlsx(path, sheets, decimals=DECIMALS):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.round(decimals).to_excel(xw, sheet_name=name, index=False)
    return path


def make_field_df(times_s, cols_cm, values, index_header=INDEX_HEADER):
    df = pd.DataFrame(values, columns=[str(c) for c in cols_cm])
    df.insert(0, index_header, np.asarray(times_s, dtype=float))
    return df


def fmt(x, decimals=DECIMALS):
    return float(f"{float(x):.{decimals}f}")
