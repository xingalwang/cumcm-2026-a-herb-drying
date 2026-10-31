
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import numpy as np

from params import (
    C0,
    C_THRESHOLD,
    FIGDIR,
    H_CONV,
    HM_MASS,
    R0,
    T0_C,
)
from props import PROPERTY_SETS, to_kelvin
from utils import (
    D_face,
    air_driving,
    assemble_solve,
    k_face,
    radius_interpolant,
)


OUTDIR = FIGDIR / "enhanced_q4"
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURES = (
    "T_plateau_C",
    "C_plateau",
    "hm_factor",
    "eta_nonuniform",
    "gamma_timing",
)
BOUNDS = {
    "T_plateau_C": (49.0, 51.0),
    "C_plateau": (0.045, 0.055),
    "hm_factor": (0.5, 1.5),
    "eta_nonuniform": (-0.6, 0.6),
    "gamma_timing": (0.5, 2.0),
}

DEFAULTS = {
    "seed": 20260912,
    "samples": 128,
    "explain": 48,
    "background": 24,
    "N": 48,
    "dt": 120.0,
    "cap": 3.0e6,
}


# 结果写 JSON 前转成基础类型。
def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def write_json(name, value):
    path = OUTDIR / name
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def sample_inputs(rng, n):
    u = rng.random((n, len(FEATURES)))
    x = np.empty_like(u)
    for j, name in enumerate(FEATURES):
        lo, hi = BOUNDS[name]
        x[:, j] = lo + (hi - lo) * u[:, j]
    return x


def feature_dict(x):
    return {name: float(x[j]) for j, name in enumerate(FEATURES)}


def geometry(t, radius_of, eta, gamma, n, radius_end):
    # eta 控制径向不均匀，gamma 控制出现时机。
    R = float(radius_of(t))
    x = np.linspace(0.0, 1.0, n + 1)
    q = max(0.0, min(1.0, 1.0 - R / R0))
    q_end = max(1.0e-14, 1.0 - radius_end / R0)
    amp = (q / q_end) ** gamma if q > 0.0 else 0.0
    phi = x + eta * amp * x * (1.0 - x)
    jac = 1.0 + eta * amp * (1.0 - 2.0 * x)
    if np.min(jac) <= 0.0:
        raise ValueError("nonuniform mapping is not monotone")
    r = R * phi
    r[-1] = R
    faces = 0.5 * (r[:-1] + r[1:])
    V = np.empty(n + 1, dtype=float)
    V[0] = faces[0] ** 2 / 2.0
    V[1:n] = (faces[1:] ** 2 - faces[:-1] ** 2) / 2.0
    V[n] = (R**2 - faces[-1] ** 2) / 2.0
    return r, V, faces, R, jac


def make_solver(x, *, n, dt, cap_s):
    params = feature_dict(x)
    radius_of, _, radius_meta = radius_interpolant()
    radius_end = float(radius_meta["R_end_m"])
    T_air, C_air = air_driving(
        params["T_plateau_C"], params["C_plateau"], mode="linear"
    )
    pset = PROPERTY_SETS["app4"]
    props_fn, D_fn = pset["props"], pset["D"]
    hm = HM_MASS * params["hm_factor"]
    eta = params["eta_nonuniform"]
    gamma = params["gamma_timing"]

    def step(C_old, T_old, t_old, h):
        t_new = t_old + h
        r, V, faces, R, _ = geometry(
            t_new, radius_of, eta, gamma, n, radius_end
        )
        dr_face = np.diff(r)
        Ta = float(T_air(t_new))
        Ca = float(C_air(t_new))
        C_iter, T_iter = C_old.copy(), T_old.copy()
        for _ in range(6):
            T_face = to_kelvin(0.5 * (T_iter[:-1] + T_iter[1:]))
            D_face_now = D_face(C_iter, pset, T_face, mode="kirchhoff")
            D_surface = float(D_fn(C_iter[-1], to_kelvin(T_iter[-1])))
            rho, cp, conductivity = props_fn(C_iter)
            thermal_capacity = rho * cp
            conductivity_face = k_face(conductivity)
            C_new = assemble_solve(
                C_old,
                np.ones_like(C_old),
                V,
                faces,
                D_face_now / dr_face,
                h,
                hm,
                Ca,
                R,
            )
            T_new = assemble_solve(
                T_old,
                thermal_capacity,
                V,
                faces,
                conductivity_face / dr_face,
                h,
                H_CONV,
                Ta,
                R,
            )
            if (
                np.max(np.abs(C_new - C_iter)) < 1.0e-9
                and np.max(np.abs(T_new - T_iter)) < 1.0e-7
            ):
                C_iter, T_iter = C_new, T_new
                break
            C_iter, T_iter = C_new, T_new
        return C_iter, T_iter, R

    def solve():
        C = np.full(n + 1, C0, dtype=float)
        T = np.full(n + 1, T0_C, dtype=float)
        t = 0.0
        previous_g = C0 - C_THRESHOLD
        first_bracket = None
        while t < cap_s - 1.0e-12:
            h = min(dt, cap_s - t)
            C_before, T_before, t_before = C, T, t
            C, T, R = step(C_before, T_before, t, h)
            t += h
            g = float(np.max(C)) - C_THRESHOLD
            if previous_g >= 0.0 > g:
                first_bracket = (t_before, t)
                break
            previous_g = g
        if first_bracket is None:
            return None

        lo, hi = 0.0, first_bracket[1] - first_bracket[0]
        sub = max(dt / 8.0, 1.0)
        while hi - lo > 2.0:
            mid = 0.5 * (lo + hi)
            C_mid, T_mid = C_before.copy(), T_before.copy()
            tt = first_bracket[0]
            while tt < first_bracket[0] + mid - 1.0e-12:
                hh = min(sub, first_bracket[0] + mid - tt)
                C_mid, T_mid, _ = step(C_mid, T_mid, tt, hh)
                tt += hh
            if float(np.max(C_mid)) < C_THRESHOLD:
                hi = mid
            else:
                lo = mid
        t_end = first_bracket[0] + 0.5 * (lo + hi)
        return {
            "t_end_s": float(t_end),
            "t_end_h": float(t_end / 3600.0),
            "R_at_t_end_cm": float(radius_of(t_end) * 100.0),
            "eta": float(eta),
            "gamma": float(gamma),
            "N": int(n),
            "dt_solve_s": float(dt),
            "hm_factor": float(params["hm_factor"]),
        }

    return solve


def evaluate(x, **solver_options):
    result = make_solver(x, **solver_options)()
    row = feature_dict(x)
    if result is None:
        row.update({"status": "unreached", "t_end_h": None})
    else:
        row.update({"status": "reached", **result})
    return row


def geometry_tests(n):
    # 先检查坐标映射再跑随机样本。
    radius_of, _, meta = radius_interpolant()
    tests = []
    for eta in (-0.6, 0.0, 0.6):
        for gamma in (0.5, 1.0, 2.0):
            jac_min = 1.0
            boundary_error = 0.0
            for t in np.linspace(0.0, min(RADIUS_TEST_CAP, DEFAULTS["cap"]), 121):
                r, _, _, R, jac = geometry(
                    t, radius_of, eta, gamma, n, float(meta["R_end_m"])
                )
                jac_min = min(jac_min, float(np.min(jac)))
                boundary_error = max(boundary_error, abs(float(r[-1] - R)))
            tests.append(
                {
                    "eta": eta,
                    "gamma": gamma,
                    "min_mapping_jacobian": jac_min,
                    "max_boundary_error_m": boundary_error,
                    "monotone": bool(jac_min > 0.0),
                }
            )
    return {"radius_meta": meta, "tests": tests}


RADIUS_TEST_CAP = 259200.0


def run_monte_carlo(settings):
    # Monte Carlo 只对能达到阈值的样本统计。
    rng = np.random.default_rng(settings["seed"])
    X = sample_inputs(rng, settings["samples"])
    rows = []
    start = time.time()
    for i, x in enumerate(X):
        row = evaluate(
            x,
            n=settings["N"],
            dt=settings["dt"],
            cap_s=settings["cap"],
        )
        row["sample_id"] = i
        rows.append(row)
        if (i + 1) % max(1, settings["samples"] // 8) == 0:
            print(f"[global-q4] Monte Carlo {i + 1}/{settings['samples']}", flush=True)

    reached = [row for row in rows if row["t_end_h"] is not None]
    y = np.asarray([row["t_end_h"] for row in reached], dtype=float)
    if y.size == 0:
        raise RuntimeError("Monte Carlo 样本全部未在上限内达标")
    result = {
        "method": "seeded uniform Monte Carlo",
        "seed": settings["seed"],
        "n_samples": settings["samples"],
        "n_reached": int(y.size),
        "features": list(FEATURES),
        "bounds": BOUNDS,
        "solver": {
            "N": settings["N"],
            "dt_solve_s": settings["dt"],
            "t_cap_s": settings["cap"],
        },
        "summary": {
            "mean_h": float(np.mean(y)),
            "std_h": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
            "q05_h": float(np.quantile(y, 0.05)),
            "q50_h": float(np.quantile(y, 0.50)),
            "q95_h": float(np.quantile(y, 0.95)),
            "min_h": float(np.min(y)),
            "max_h": float(np.max(y)),
        },
        "rows": rows,
        "wall_time_s": time.time() - start,
    }
    write_json("monte_carlo_results.json", result)
    _write_rows_csv("monte_carlo_results.csv", rows)
    return X, y, result


def polynomial_design(X):
    X = np.asarray(X, dtype=float)
    columns = [np.ones(X.shape[0])]
    columns.extend(X[:, j] for j in range(X.shape[1]))
    columns.extend(X[:, j] ** 2 for j in range(X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            columns.append(X[:, i] * X[:, j])
    return np.column_stack(columns)


def fit_surrogate(X, y, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    split = max(8, int(0.8 * len(y)))
    train, test = order[:split], order[split:]
    x_min = np.array([BOUNDS[name][0] for name in FEATURES])
    x_max = np.array([BOUNDS[name][1] for name in FEATURES])
    scale = np.where(x_max > x_min, x_max - x_min, 1.0)
    X01 = (X - x_min) / scale
    Z = polynomial_design(X01)
    penalty = np.eye(Z.shape[1]) * 1.0e-8
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(Z[train].T @ Z[train] + penalty, Z[train].T @ y[train])
    pred_test = Z[test] @ coef if test.size else Z[train] @ coef
    truth = y[test] if test.size else y[train]
    residual = pred_test - truth
    ss_tot = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = None if ss_tot == 0.0 else 1.0 - float(np.sum(residual**2)) / ss_tot
    return {
        "coef": coef,
        "x_min": x_min,
        "scale": scale,
        "metrics": {
            "model": "second-order ridge surrogate",
            "train_n": int(train.size),
            "test_n": int(test.size),
            "test_r2": r2,
            "test_rmse_h": float(np.sqrt(np.mean(residual**2))),
            "test_mae_h": float(np.mean(np.abs(residual))),
        },
    }


def surrogate_predict(model, X):
    X01 = (np.asarray(X, dtype=float) - model["x_min"]) / model["scale"]
    return polynomial_design(X01) @ model["coef"]


def exact_interventional_shap(model, X_explain, X_background):
    # 对二阶代理模型枚举全部特征子集。
    p = len(FEATURES)
    factorial = math.factorial
    phi = np.zeros((len(X_explain), p), dtype=float)
    base = float(np.mean(surrogate_predict(model, X_background)))
    for row_id, x in enumerate(X_explain):
        values = {}
        for mask in range(1 << p):
            samples = X_background.copy()
            for j in range(p):
                if mask & (1 << j):
                    samples[:, j] = x[j]
            values[mask] = float(np.mean(surrogate_predict(model, samples)))
        for j in range(p):
            for mask in range(1 << p):
                if mask & (1 << j):
                    continue
                size = int(mask.bit_count())
                weight = factorial(size) * factorial(p - size - 1) / factorial(p)
                phi[row_id, j] += weight * (
                    values[mask | (1 << j)] - values[mask]
                )
    return phi, base


def run_shap(X, y, settings):
    model = fit_surrogate(X, y, settings["seed"] + 17)
    rng = np.random.default_rng(settings["seed"] + 55)
    n_exp = min(settings["explain"], len(y))
    n_bg = min(settings["background"], len(y))
    exp_idx = rng.choice(len(y), size=n_exp, replace=False)
    bg_idx = rng.choice(len(y), size=n_bg, replace=False)
    phi, base = exact_interventional_shap(model, X[exp_idx], X[bg_idx])
    pred = surrogate_predict(model, X[exp_idx])
    efficiency = pred - (base + np.sum(phi, axis=1))
    abs_mean = np.mean(np.abs(phi), axis=0)
    signed_mean = np.mean(phi, axis=0)
    order = np.argsort(-abs_mean)
    importance = [
        {
            "feature": FEATURES[int(j)],
            "mean_abs_shap_h": float(abs_mean[j]),
            "mean_signed_shap_h": float(signed_mean[j]),
            "rank": rank,
        }
        for rank, j in enumerate(order, 1)
    ]
    result = {
        "method": "exact interventional Shapley on second-order ridge surrogate",
        "interpretation": "归因对象是代理模型，不是原始 PDE 的因果效应",
        "features": list(FEATURES),
        "baseline_expected_t_end_h": base,
        "surrogate_metrics": model["metrics"],
        "n_explain": int(n_exp),
        "n_background": int(n_bg),
        "coalitions": 2 ** len(FEATURES),
        "global_importance": importance,
        "efficiency_residual": {
            "mean_abs_h": float(np.mean(np.abs(efficiency))),
            "max_abs_h": float(np.max(np.abs(efficiency))),
        },
        "rows": [
            {
                "sample_id": int(exp_idx[i]),
                "t_end_h": float(y[exp_idx[i]]),
                "shap_h": {
                    FEATURES[j]: float(phi[i, j]) for j in range(len(FEATURES))
                },
            }
            for i in range(n_exp)
        ],
    }
    write_json("shap_results.json", result)
    _write_rows_csv(
        "shap_results.csv",
        [
            {
                "sample_id": row["sample_id"],
                "t_end_h": row["t_end_h"],
                **row["shap_h"],
            }
            for row in result["rows"]
        ],
    )
    return result


def _write_rows_csv(name, rows):
    path = OUTDIR / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def try_plots(mc, shap):
    # 图形是附加输出，缺少 matplotlib 时不影响计算。
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        write_json("plot_status.json", {"available": False, "reason": str(exc)})
        return

    rows = [row for row in mc["rows"] if row["t_end_h"] is not None]
    y = np.asarray([row["t_end_h"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.hist(y, bins=12, color="#4472C4", edgecolor="white")
    ax.axvline(np.mean(y), color="#C00000", lw=1.8, label=f"mean={np.mean(y):.2f} h")
    ax.axvline(np.quantile(y, 0.05), color="#70AD47", ls="--", label="5%")
    ax.axvline(np.quantile(y, 0.95), color="#70AD47", ls="--", label="95%")
    ax.set_xlabel("Q4 first-passage time (h)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_mc_t_end_distribution.png", bbox_inches="tight")
    plt.close(fig)

    imp = shap["global_importance"]
    names = [row["feature"] for row in reversed(imp)]
    values = [row["mean_abs_shap_h"] for row in reversed(imp)]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    ax.barh(names, values, color="#ED7D31")
    ax.set_xlabel("Mean absolute Shapley contribution (h)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_shap_global_importance.png", bbox_inches="tight")
    plt.close(fig)
    write_json("plot_status.json", {"available": True})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Q4 global uncertainty analysis")
    parser.add_argument("--quick", action="store_true", help="少量样本快速检查")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    settings = dict(DEFAULTS)
    settings["seed"] = args.seed
    if args.samples is not None:
        settings["samples"] = max(4, args.samples)
    if args.quick:
        settings.update({"samples": min(settings["samples"], 16),
                         "explain": 8, "background": 8,
                         "N": 24, "dt": 240.0})

    start = time.time()
    geometry_result = geometry_tests(settings["N"])
    write_json(
        "nonuniform_geometry_tests.json",
        {
            "mapping": "r=R(t)[x+eta*a(t)*x*(1-x)]",
            "eta_zero_expected_exact_reduction": True,
            "all_pass": all(
                item["monotone"] and item["max_boundary_error_m"] < 1.0e-14
                for item in geometry_result["tests"]
            ),
            **geometry_result,
        },
    )
    X, y, mc = run_monte_carlo(settings)
    shap = run_shap(X, y, settings)
    if not args.no_plots:
        try_plots(mc, shap)
    summary = {
        "files": [
            "nonuniform_geometry_tests.json",
            "monte_carlo_results.json",
            "monte_carlo_results.csv",
            "shap_results.json",
            "shap_results.csv",
        ],
        "settings": settings,
        "wall_time_s": time.time() - start,
        "mc_summary": mc["summary"],
        "shap_ranking": shap["global_importance"],
    }
    write_json("enhanced_q4_summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
