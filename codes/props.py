
from __future__ import annotations

import numpy as np
from scipy.special import exp1

from params import (A_APP2, A_APP3, A_APP4, B_ARRH3, B_ARRH4, CP_A3, CP_A4, CP_APP2,
                    CP_B3, CP_B4, D0_APP2, D0_APP3, D0_APP4, K_A3, K_A4, K_APP2,
                    K_B3, K_B4, KELVIN_OFFSET, RHO_A3, RHO_A4, RHO_APP2, RHO_B3, RHO_B4)

T_K_LO, T_K_HI = 250.0, 400.0
ARRH_LO, ARRH_HI = 1e-7, 1e-5


class UnitError(ValueError):
    # 温度单位错时不要继续计算。
    pass


def to_kelvin(T_c):
    return np.asarray(T_c, dtype=float) + KELVIN_OFFSET


def _guard_kelvin(T_K):
    T_K = np.asarray(T_K, dtype=float)
    lo, hi = float(np.min(T_K)), float(np.max(T_K))
    if not (T_K_LO < lo and hi < T_K_HI):
        raise UnitError(
            f"温度实参必须为开尔文且落在 ({T_K_LO},{T_K_HI})，实测 [{lo:.4f},{hi:.4f}]；"
            "若传入摄氏度，exp(-3850/T) 会退化到 1e-34 量级并使 D 归零"
        )
    return T_K


def arrhenius(T_K, b):
    T_K = _guard_kelvin(T_K)
    f = np.exp(-b / T_K)
    fmin, fmax = float(np.min(f)), float(np.max(f))
    if not (ARRH_LO <= fmin and fmax <= ARRH_HI):
        raise UnitError(f"exp(-{b}/T_K) 量级异常：[{fmin:.6e},{fmax:.6e}] 越出 [{ARRH_LO:g},{ARRH_HI:g}]")
    return f


# 附录 2
def props_app2():
    return RHO_APP2, CP_APP2, K_APP2


def D_app2(C):
    C = np.asarray(C, dtype=float)
    return D0_APP2 * np.exp(-A_APP2 / np.maximum(C, 1e-300))


# 附录 3
def props_app3(C):
    C = np.asarray(C, dtype=float)
    w = C / (C + 1.0)
    rho = RHO_A3 + RHO_B3 * C
    cp = CP_A3 + CP_B3 * w
    k = K_A3 + K_B3 * w
    return rho, cp, k


def D_app3(C, T_K):
    C = np.asarray(C, dtype=float)
    return D0_APP3 * np.exp(-A_APP3 / np.maximum(C, 1e-300)) * arrhenius(T_K, B_ARRH3)


# 附录 4
def props_app4(C):
    C = np.asarray(C, dtype=float)
    w = C / (C + 1.0)
    rho = RHO_A4 + RHO_B4 * C
    cp = CP_A4 + CP_B4 * w
    k = K_A4 + K_B4 * w
    return rho, cp, k


def D_app4(C, T_K):
    C = np.asarray(C, dtype=float)
    return D0_APP4 * np.exp(-A_APP4 / np.maximum(C, 1e-300)) * arrhenius(T_K, B_ARRH4)


# Kirchhoff 势
def _phi_shape(C, a):
    C = np.asarray(C, dtype=float)
    out = np.zeros_like(C)
    m = C > 0.0
    if np.any(m):
        z = a / C[m]
        out[m] = C[m] * np.exp(-z) - a * exp1(z)
    return out


def Phi_app2(C):
    return D0_APP2 * _phi_shape(C, A_APP2)


def Phi_app3(C, fT):
    return D0_APP3 * _phi_shape(C, A_APP3) * np.asarray(fT, dtype=float)


def Phi_app4(C, fT):
    return D0_APP4 * _phi_shape(C, A_APP4) * np.asarray(fT, dtype=float)


PROPERTY_SETS = {
    "app2": {"props": props_app2, "D": D_app2, "Phi": Phi_app2, "a": A_APP2, "D0": D0_APP2,
             "b": None, "coupling": "decoupled"},
    "app3": {"props": props_app3, "D": D_app3, "Phi": Phi_app3, "a": A_APP3, "D0": D0_APP3,
             "b": B_ARRH3, "coupling": "two_way"},
    "app4": {"props": props_app4, "D": D_app4, "Phi": Phi_app4, "a": A_APP4, "D0": D0_APP4,
             "b": B_ARRH4, "coupling": "two_way"},
}


if __name__ == "__main__":
    import inspect
    from params import C0, KELVIN_OFFSET as KO
    print("D_app2 signature:", inspect.signature(D_app2))
    assert "T" not in str(inspect.signature(D_app2)), "C-P1-2: D_app2 签名不得含 T"
    print(f"props_app2 = {props_app2()}")
    print(f"D_app2({C0}) = {float(D_app2(C0)):.6e}")
    r3, c3, k3 = props_app3(C0)
    r4, c4, k4 = props_app4(C0)
    print(f"app3 @C0: rho={float(r3):.4f} cp={float(c3):.4f} k={float(k3):.7f}")
    print(f"app4 @C0: rho={float(r4):.4f} cp={float(c4):.4f} k={float(k4):.7f}")
    TK50 = 50.0 + KO
    d3 = float(D_app3(C0, TK50)); d4 = float(D_app4(C0, TK50))
    print(f"D_app3(C0,323.15)={d3:.6e}  D_app4={d4:.6e}  ratio={d4 / d3:.6f}")
    print("props self-check OK")
