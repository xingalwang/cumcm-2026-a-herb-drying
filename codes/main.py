
from __future__ import annotations

import argparse
import json
import os
import sys
import time


_HERE = os.path.dirname(os.path.abspath(__file__))
# 直接运行脚本时也能找到同目录模块。
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from params import C_THRESHOLD, FIGDIR


STAGES = [
    # 后一阶段会读取前一阶段的结果。
    ("structural_tests", "structural_tests", "结构性检验 S1–S8（解析极限与单位误用探针）"),
    ("problem_1", "problem1", "问题一：附录2 常物性解耦场，t≤1800 s"),
    ("problem_2", "problem2", "问题二：附录3 双向耦合，t≤3 h"),
    ("problem_3", "problem3", "问题三：附录3 固定几何，干燥终点首达 t_end"),
    ("problem_4", "problem4", "问题四：附录4 + ALE 移动边界，干燥终点首达 t_end"),
    ("sensitivity", "sensitivity_analysis", "灵敏度与鲁棒性 §8 + 龙卷排序"),
]


def _path(tag):
    # structural_tests 的文件名不同。
    if tag == "structural_tests":
        return FIGDIR / "structural_tests.json"
    return FIGDIR / f"{tag}_results.json"


def _verify(tag, d):
    def finite(x):
        return x is not None and np.all(np.isfinite(np.atleast_1d(np.asarray(x, float))))

    if tag == "structural_tests":
        if not d:
            return False, "结果为空"
        bad = [k for k, v in d.items() if isinstance(v, dict) and not v.get("pass")]
        return (not bad), (f"S1–S8 全部通过（{len(d)} 项）" if not bad else f"未通过 {bad}")

    if tag == "problem_1":
        for key in ("Bi_T", "Bi_m", "alpha_m2_s", "C_bar_1800"):
            if not finite(d.get(key)):
                return False, f"{key} 缺失或非有限"
        err = d.get("duhamel_check", {}).get("max_abs_err_K")
        if not finite(err):
            return False, "Duhamel 独立解校核缺失"

        if (d.get("decoupling_swap_order_max_dT_K") != 0.0
                or d.get("decoupling_swap_order_max_dC") != 0.0):
            return False, "解耦顺序互换未逐点一致"
        return True, f"Duhamel 独立解最大偏差 {float(err):.2e} K，解耦顺序互换严格一致"

    if tag == "problem_2":
        if not finite(d.get("C_bar_10800")):
            return False, "Cbar(3h) 缺失或非有限"
        rel = d.get("PXC3_picard_invariance", {}).get("rel_diff")
        if not finite(rel) or abs(float(rel)) >= 1e-4:
            return False, f"PX-C3 Picard 不变性未通过（rel_diff={rel}）"
        if d.get("P2C2_long_vs_short_max_dC") != 0.0:
            return False, "P2-C2 长程/短程解不一致"
        return True, f"Cbar(3h)={d['C_bar_10800']:.6f}，Picard 不变性 rel={float(rel):.2e}"

    if tag in ("problem_3", "problem_4"):
        t_end = d.get("t_end_s")
        mx = d.get("at_t_end", {}).get("max_C")
        if not finite(t_end) or not finite(mx):
            return False, "t_end 或 t_end 处 max_r C 缺失"

        g = float(mx) - C_THRESHOLD
        if g > 0:
            return False, f"t_end 处 max_r C={mx:.10f} 未过阈值（g={g:+.2e}）"
        if abs(g) > 1e-5:
            return False, f"t_end 处 g={g:+.2e} 超出二分容差量级"
        if not d.get("no_rebound_to_1.05t_end"):
            return False, "1.05·t_end 内 max_r C 出现回升，首达时刻不成立"
        return True, f"t_end={float(t_end) / 3600:.4f} h, g={g:+.2e}, 无回升"

    if tag == "sensitivity":
        tor = d.get("sec_8_4_tornado", {})
        if not tor.get("ranking_q3") or not tor.get("ranking_q4"):
            return False, "龙卷排序缺失"
        return True, f"Q3 首位 {tor['top_factor_q3']}, Q4 首位 {tor['top_factor_q4']}"

    return False, "未知阶段"


def _load(tag):
    p = _path(tag)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  缓存不可读（{exc}）→ 重算", flush=True)
        return None


def run_stage(tag, module, desc, force):
    print(f"\n{'=' * 72}\n[MAIN] {tag}: {desc}\n{'=' * 72}", flush=True)
    if not force:
        cached = _load(tag)
        if cached is not None:
            ok, why = _verify(tag, cached)
            if ok:
                print(f"[MAIN] 复用已验证缓存 → {why}", flush=True)
                return cached, True
            print(f"[MAIN] 缓存未通过复核（{why}）→ 重算", flush=True)

    mod = __import__(module)
    entry = getattr(mod, "run", None) or getattr(mod, "main")
    t0 = time.time()
    entry()
    d = _load(tag)
    if d is None:
        raise RuntimeError(f"{tag} 运行结束但未写出 {_path(tag).name}")
    ok, why = _verify(tag, d)
    if not ok:
        raise RuntimeError(f"{tag} 新结果未通过复核：{why}")
    print(f"[MAIN] {tag} 完成（{time.time() - t0:.1f} s）→ {why}", flush=True)
    return d, False


def summarize(results):
    q1, q2 = results["problem_1"], results["problem_2"]
    q3, q4 = results["problem_3"], results["problem_4"]
    t3, t4 = float(q3["t_end_s"]), float(q4["t_end_s"])
    mech = q4["vs_problem3"]["competing_mechanisms"]
    return {
        "problem_1": {"t_span_s": 1800.0,
                      "T_center_1800s_C": q1["t1800_temperature_C"]["0.0"],
                      "T_surface_1800s_C": q1["t1800_temperature_C"]["2.0"],
                      "C_center_1800s": q1["t1800_moisture_kgkg"]["0.0"],
                      "C_surface_1800s": q1["t1800_moisture_kgkg"]["2.0"],
                      "C_bar_1800s": q1["C_bar_1800"],
                      "Bi_T": q1["Bi_T"], "Bi_m": q1["Bi_m"],
                      "Le": q1["Le_alpha_over_D"]},
        "problem_2": {"t_span_s": 10800.0,
                      "T_center_3h_C": q2["t10800_temperature_C"]["0.0"],
                      "C_center_3h": q2["t10800_moisture_kgkg"]["0.0"],
                      "C_surface_3h": q2["t10800_moisture_kgkg"]["2.0"],
                      "C_bar_3h": q2["C_bar_10800"]},
        "problem_3": {"t_end_s": t3, "t_end_h": t3 / 3600.0, "t_end_d": t3 / 86400.0,
                      "max_C_at_t_end": q3["at_t_end"]["max_C"],
                      "C_bar_at_t_end": q3["at_t_end"]["C_bar"],
                      "wrong_criterion_C_bar_t_end_h": q3["wrong_criterion_C_bar_t_end_h"],
                      "geometry": "fixed R=2 cm", "property_set": "appendix3"},
        "problem_4": {"t_end_s": t4, "t_end_h": t4 / 3600.0, "t_end_d": t4 / 86400.0,
                      "max_C_at_t_end": q4["at_t_end"]["max_C"],
                      "C_bar_at_t_end": q4["at_t_end"]["C_bar"],
                      "R_final_cm": q4["R_at_t_end_cm"],
                      "frozen_radius_t_end_h":
                          q4["P4C1c_frozen_radius"]["t_end_frozen_R_h"],
                      "geometry": "ALE moving boundary (附件2)", "property_set": "appendix4"},
        "q4_vs_q3": {"t_end_ratio": t4 / t3,
                     "delta_h": (t4 - t3) / 3600.0,
                     "D4_over_D3_at_C0_50C": mech["D4_over_D3_at_C0_50C"],
                     "geometric_enhancement_area_ratio":
                         mech["geometric_enhancement_area_ratio"],
                     "verdict": "收缩带来的几何增强胜过附录4 更小的扩散系数"
                                if t4 < t3 else "附录4 更小的扩散系数占主导"},
        "threshold": C_THRESHOLD,
        "criterion": "干燥终点判据为 max_r C(r,t) < 0.15（全域上确界），非体积平均",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="药材烘干四问主调度")
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重算")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定阶段（tag），其余阶段仍从缓存读取以完成汇总")
    ap.add_argument("--analysis-global", action="store_true",
                    help="只运行问题四非均匀收缩、Monte Carlo 与 Shapley 分析")
    ap.add_argument("--quick", action="store_true",
                    help="用于 --analysis-global 的快速检查模式")
    args = ap.parse_args(argv)

    if args.analysis_global:
        from global_q4_analysis import main as run_global
        return run_global(["--quick"] if args.quick else [])

    t0 = time.time()
    results, reused = {}, []
    for tag, module, desc in STAGES:
        if args.only and tag not in args.only:
            cached = _load(tag)
            if cached is None:
                raise SystemExit(f"--only 跳过了 {tag} 但其结果文件不存在，无法汇总")
            ok, why = _verify(tag, cached)
            if not ok:
                raise SystemExit(f"--only 跳过的 {tag} 缓存未通过复核：{why}")
            results[tag], _ = cached, True
            reused.append(tag)
            print(f"[MAIN] {tag}: --only 跳过，复用缓存（{why}）", flush=True)
            continue
        d, was_reused = run_stage(tag, module, desc, args.force)
        results[tag] = d
        if was_reused:
            reused.append(tag)

    summary = summarize(results)
    all_out = {
        "meta": {"stages": [s[0] for s in STAGES], "reused_from_cache": reused,
                 "wall_time_s": time.time() - t0,
                 "python": sys.version.split()[0], "numpy": np.__version__},
        "summary": summary,
        "problem_1": results["problem_1"], "problem_2": results["problem_2"],
        "problem_3": results["problem_3"], "problem_4": results["problem_4"],
        "structural_tests": results["structural_tests"],
        "sensitivity": results["sensitivity"],
    }
    FIGDIR.mkdir(exist_ok=True)
    with open(FIGDIR / "all_results.json", "w", encoding="utf-8") as fh:
        json.dump(all_out, fh, ensure_ascii=False, indent=2, default=float)

    print(f"\n{'=' * 72}\n[MAIN] 汇总", flush=True)
    print(f"  问题一 T(0,1800s) = {summary['problem_1']['T_center_1800s_C']:.4f} °C，"
          f"Cbar = {summary['problem_1']['C_bar_1800s']:.6f}", flush=True)
    print(f"  问题二 Cbar(3h)   = {summary['problem_2']['C_bar_3h']:.6f}", flush=True)
    print(f"  问题三 t_end      = {summary['problem_3']['t_end_h']:.4f} h "
          f"({summary['problem_3']['t_end_d']:.4f} d)", flush=True)
    print(f"  问题四 t_end      = {summary['problem_4']['t_end_h']:.4f} h "
          f"({summary['problem_4']['t_end_d']:.4f} d)，R_end = "
          f"{summary['problem_4']['R_final_cm']:.4f} cm", flush=True)
    print(f"  Q4/Q3 = {summary['q4_vs_q3']['t_end_ratio']:.5f}"
          f"（{summary['q4_vs_q3']['delta_h']:+.4f} h）", flush=True)
    print(f"[MAIN] -> results/all_results.json（总 {all_out['meta']['wall_time_s']:.1f} s，"
          f"复用 {len(reused)}/{len(STAGES)} 阶段）", flush=True)
    return all_out


if __name__ == "__main__":
    main()
