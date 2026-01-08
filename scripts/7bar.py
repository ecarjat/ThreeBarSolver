#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt


def dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.hypot(d[0], d[1]))


def angle_at(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    va = a - p
    vb = b - p
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.dot(va, vb) / (na * nb))
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c))


@dataclass
class Config:
    Hext: float
    ax1: float  # fixed AX1 length = dist(1,3); with x3=0 => P3=(0, ax1) if +Y down

    # weights
    w_len: float = 250.0
    w_pose: float = 900.0
    w_soft: float = 60.0
    w_soft_ineq: float = 60.0
    w_reg: float = 1e-2
    w_TR: float = 0.01
    w_DR: float = 0.01
    gap: float = 0.002
    w_clear: float = 1500.0  # start here; increase if it still collapses
    w_above: float = 1200.0  # weight for point7 above point2 preference

    close_scale: float = 0.02  # meters tolerance for "similar"


# Rod list (for plotting + interpretation)
RODS = [
    ("TR1", 1, 2),
    ("TR2", 2, 7),
    ("TR_1_7", 1, 7),  # closure to keep TR rigid
    ("DR", 3, 4),
    ("AX1", 1, 3),  # fixed length via ax1
    ("AX2", 2, 4),
    ("CR1", 2, 5),
    ("CR2", 4, 6),
    ("AX3", 5, 6),
    ("CSR", 6, 7),
]


def seg_point_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    c = a + t * ab
    return float(np.linalg.norm(p - c))


def segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    # Standard orientation test
    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
            and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # general case
    if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
        return True

    # colinear special cases
    if abs(o1) < 1e-12 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-12 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-12 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-12 and on_segment(c, b, d):
        return True
    return False


def seg_seg_distance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        seg_point_distance(a, c, d),
        seg_point_distance(b, c, d),
        seg_point_distance(c, a, b),
        seg_point_distance(d, a, b),
    )


def pack_vars(P: Dict[int, np.ndarray], L: Dict[str, float]) -> np.ndarray:
    order_pts = [2, 4, 5, 6, 7]
    x = []
    for i in order_pts:
        x.extend([P[i][0], P[i][1]])
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        x.append(float(L[k]))
    return np.array(x, dtype=float)


def unpack_vars(
    x: np.ndarray, cfg: Config
) -> Tuple[Dict[int, np.ndarray], Dict[str, float]]:
    P: Dict[int, np.ndarray] = {}
    P[1] = np.array([0.0, 0.0], dtype=float)
    P[3] = np.array([0.0, cfg.ax1], dtype=float)  # +Y down

    order_pts = [2, 4, 5, 6, 7]
    idx = 0
    for i in order_pts:
        P[i] = np.array([x[idx], x[idx + 1]], dtype=float)
        idx += 2

    L = {}
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        L[k] = float(x[idx])
        idx += 1

    return P, L


def residuals(x: np.ndarray, cfg: Config) -> np.ndarray:
    P, L = unpack_vars(x, cfg)
    wheel = 0.5 * (P[5] + P[6])

    r = []

    # --- Hard rod constraints ---
    r.append((dist(P[1], P[2]) - L["LTR1"]) * cfg.w_len)  # TR1
    r.append((dist(P[2], P[7]) - L["LTR2"]) * cfg.w_len)  # TR2
    r.append((dist(P[1], P[7]) - L["L17"]) * cfg.w_len)  # TR closure 1-7

    r.append((dist(P[3], P[4]) - L["LDR"]) * cfg.w_len)  # DR

    r.append((dist(P[2], P[4]) - L["LAX2"]) * cfg.w_len)  # AX2
    r.append((dist(P[5], P[6]) - L["LAX3"]) * cfg.w_len)  # AX3

    r.append((dist(P[2], P[5]) - L["LCR1"]) * cfg.w_len)  # CR1
    r.append((dist(P[4], P[6]) - L["LCR2"]) * cfg.w_len)  # CR2

    r.append((dist(P[6], P[7]) - L["LCSR"]) * cfg.w_len)  # CSR

    # --- Pose constraints ---
    # wheel midpoint at x=0, y = y3 + Hext = ax1 + Hext
    r.append((wheel[0] - 0.0) * cfg.w_pose)
    r.append((wheel[1] - (cfg.ax1 + cfg.Hext)) * cfg.w_pose)

    # --- Soft "similar" constraints ---
    s = max(1e-6, cfg.close_scale)

    # TR1 ~ DR
    r.append(((L["LDR"] - L["LTR1"]) / s) * cfg.w_soft)

    # CR1 ~ CR2
    r.append(((L["LCR2"] - L["LCR1"]) / s) * cfg.w_soft)

    # AX2 ~ AX1, AX3 ~ AX1, AX2 ~ AX3   (AX1 fixed = cfg.ax1)
    r.append(((L["LAX2"] - cfg.ax1) / s) * cfg.w_soft)
    r.append(((L["LAX3"] - cfg.ax1) / s) * cfg.w_soft)
    r.append(((L["LAX3"] - L["LAX2"]) / s) * (0.5 * cfg.w_soft))

    # --- Soft inequalities ---
    # TR triangle inequality
    margin = 1e-3
    tri_violation = max(0.0, L["L17"] - (L["LTR1"] + L["LTR2"] - margin))
    r.append(tri_violation * cfg.w_soft_ineq)

    # CSR longer than CRs (encourage)
    cr_mean = 0.5 * (L["LCR1"] + L["LCR2"])
    csr_violation = max(0.0, 1.1 * cr_mean - L["LCSR"])
    r.append(csr_violation * cfg.w_soft_ineq)

    # ---- Thick-bar clearance: TR1(1-2) vs DR(3-4) ----
    d_req = 0.5 * (cfg.w_TR + cfg.w_DR) + cfg.gap  # = 0.012 m with your values
    d_seg = seg_seg_distance(P[1], P[2], P[3], P[4])

    violation = max(0.0, d_req - d_seg)
    r.append(violation * cfg.w_clear)


    # --- Positivity + mild regularization ---
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        if L[k] <= 0:
            r.append((abs(L[k]) + 1.0) * 1e6)
        else:
            r.append(cfg.w_reg * L[k])

    return np.array(r, dtype=float)


def solve(cfg: Config, x0: Optional[np.ndarray] = None) -> Dict:
    if x0 is None:
        P = {
            1: np.array([0.0, 0.0]),
            3: np.array([0.0, cfg.ax1]),
            2: np.array([0.06, 0.05]),
            4: np.array([0.12, cfg.ax1 + 0.05]),
            5: np.array([-0.05, cfg.ax1 + 0.80 * cfg.Hext]),
            6: np.array([0.05, cfg.ax1 + 0.80 * cfg.Hext]),
            7: np.array([0.10, 0.20]),
        }

        LTR1 = dist(P[1], P[2])
        LTR2 = dist(P[2], P[7])
        L17 = dist(P[1], P[7])

        LDR = LTR1 * 1.05
        LAX2 = cfg.ax1 * 1.02
        LAX3 = cfg.ax1 * 0.98

        LCR1 = dist(P[2], P[5])
        LCR2 = LCR1 * 1.02
        LCSR = max(1.2 * 0.5 * (LCR1 + LCR2), dist(P[6], P[7]))

        L = {
            "LTR1": LTR1,
            "LTR2": LTR2,
            "L17": L17,
            "LDR": LDR,
            "LAX2": LAX2,
            "LAX3": LAX3,
            "LCR1": LCR1,
            "LCR2": LCR2,
            "LCSR": LCSR,
        }
        x0 = pack_vars(P, L)

    res = least_squares(lambda v: residuals(v, cfg), x0, method="trf", max_nfev=80000)
    P, L = unpack_vars(res.x, cfg)
    wheel = 0.5 * (P[5] + P[6])
    TR_angle = angle_at(P[2], P[1], P[7])

    out = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "nfev": int(res.nfev),
        "cost": float(res.cost),
        "Hext": cfg.Hext,
        "ax1_fixed": cfg.ax1,
        "points": {
            str(i): {"x": float(P[i][0]), "y": float(P[i][1])} for i in sorted(P.keys())
        },
        "wheel_midpoint": {"x": float(wheel[0]), "y": float(wheel[1])},
        "lengths": {
            "TR1(1-2)": float(L["LTR1"]),
            "TR2(2-7)": float(L["LTR2"]),
            "TR_1_7": float(L["L17"]),
            "DR(3-4)": float(L["LDR"]),
            "AX1(1-3) fixed": float(cfg.ax1),
            "AX2(2-4)": float(L["LAX2"]),
            "AX3(5-6)": float(L["LAX3"]),
            "CR1(2-5)": float(L["LCR1"]),
            "CR2(4-6)": float(L["LCR2"]),
            "CSR(6-7)": float(L["LCSR"]),
        },
        "angles": {
            "TR1_TR2_at_point2_deg": float(TR_angle * 180.0 / math.pi),
            "TR1_TR2_at_point2_rad": float(TR_angle),
        },
    }
    return out


def plot_solution(
    sol: Dict, cfg: Config, out_path: Optional[str] = None, show: bool = False
):
    # rebuild points as numpy arrays
    P = {
        int(k): np.array([v["x"], v["y"]], dtype=float)
        for k, v in sol["points"].items()
    }
    wheel = np.array(
        [sol["wheel_midpoint"]["x"], sol["wheel_midpoint"]["y"]], dtype=float
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Draw rods
    for name, i, j in RODS:
        if i not in P or j not in P:
            continue
        xi, yi = P[i]
        xj, yj = P[j]
        ax.plot([xi, xj], [yi, yj], linewidth=2)
        # label rod near midpoint (optional, can be cluttered)
        mx, my = 0.5 * (xi + xj), 0.5 * (yi + yj)
        ax.text(mx, my, name, fontsize=8)

    # Draw points
    for i in sorted(P.keys()):
        x, y = P[i]
        ax.scatter([x], [y], s=40)
        ax.text(x, y, f"  {i}", fontsize=10, va="center")

    # Draw wheel midpoint
    ax.scatter([wheel[0]], [wheel[1]], s=60, marker="x")
    ax.text(wheel[0], wheel[1], "  wheel(mid 5-6)", fontsize=9, va="center")

    # Axes settings
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  (+Y down)")
    ax.set_title(f"Linkage solve: Hext={cfg.Hext:.3f} m, AX1={cfg.ax1:.3f} m")

    # Nice limits with padding
    xs = [p[0] for p in P.values()] + [wheel[0]]
    ys = [p[1] for p in P.values()] + [wheel[1]]
    pad = 0.15 * max(1e-3, (max(xs) - min(xs)), (max(ys) - min(ys)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.grid(True)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def solve_pose_given_lengths(
    cfg: Config, lengths: dict, ratio: float, x0: Optional[np.ndarray] = None
) -> Dict:
    """
    Solve joint coordinates for a given extension ratio using FIXED lengths.
    ratio=1.0 => full extension, ratio=0.8 => 80%, ratio=0.2 => 20%
    """
    target_y = cfg.ax1 + ratio * cfg.Hext

    # variable vector is only the free points: 2,4,5,6,7 each (x,y)
    # pack order: [x2,y2,x4,y4,x5,y5,x6,y6,x7,y7]
    def packP(P):
        order = [2, 4, 5, 6, 7]
        v = []
        for i in order:
            v += [P[i][0], P[i][1]]
        return np.array(v, dtype=float)

    def unpackP(v):
        P = {}
        P[1] = np.array([0.0, 0.0], dtype=float)
        P[3] = np.array([0.0, cfg.ax1], dtype=float)
        order = [2, 4, 5, 6, 7]
        k = 0
        for i in order:
            P[i] = np.array([v[k], v[k + 1]], dtype=float)
            k += 2
        return P

    # fixed lengths
    LTR1 = lengths["TR1(1-2)"]
    LTR2 = lengths["TR2(2-7)"]
    L17 = lengths["TR_1_7"]
    LDR = lengths["DR(3-4)"]
    LAX2 = lengths["AX2(2-4)"]
    LAX3 = lengths["AX3(5-6)"]
    LCR1 = lengths["CR1(2-5)"]
    LCR2 = lengths["CR2(4-6)"]
    LCSR = lengths["CSR(6-7)"]

    def res(v):
        P = unpackP(v)
        wheel = 0.5 * (P[5] + P[6])
        r = []

        # hard bar constraints (distance-length)
        r.append((dist(P[1], P[2]) - LTR1) * cfg.w_len)  # TR1
        r.append((dist(P[2], P[7]) - LTR2) * cfg.w_len)  # TR2
        r.append((dist(P[1], P[7]) - L17) * cfg.w_len)  # TR closure
        r.append((dist(P[3], P[4]) - LDR) * cfg.w_len)  # DR

        r.append((dist(P[2], P[4]) - LAX2) * cfg.w_len)  # AX2
        r.append((dist(P[5], P[6]) - LAX3) * cfg.w_len)  # AX3

        r.append((dist(P[2], P[5]) - LCR1) * cfg.w_len)  # CR1
        r.append((dist(P[4], P[6]) - LCR2) * cfg.w_len)  # CR2
        r.append((dist(P[6], P[7]) - LCSR) * cfg.w_len)  # CSR

        # pose constraint for this ratio
        r.append((wheel[0] - 0.0) * cfg.w_pose)
        r.append((wheel[1] - target_y) * cfg.w_pose)

        # tiny regularization to avoid wandering (optional)
        r.append(1e-3 * v[0])
        return np.array(r, dtype=float)

    if x0 is None:
        # seed: roughly vertical, with small x offsets
        Pseed = {
            2: np.array([0.06, 0.08]),
            4: np.array([0.10, cfg.ax1 + 0.10]),
            5: np.array([-0.03, target_y - 0.03]),
            6: np.array([0.03, target_y + 0.03]),
            7: np.array([0.10, 0.20]),
        }
        x0 = packP(Pseed)

    ls = least_squares(res, x0, method="trf", max_nfev=40000)
    P = unpackP(ls.x)
    wheel = 0.5 * (P[5] + P[6])

    TR_angle = angle_at(P[2], P[1], P[7])

    return {
        "ratio": ratio,
        "target_wheel_y": float(target_y),
        "success": bool(ls.success),
        "message": str(ls.message),
        "cost": float(ls.cost),
        "points": {
            str(i): {"x": float(P[i][0]), "y": float(P[i][1])} for i in sorted(P.keys())
        },
        "wheel_midpoint": {"x": float(wheel[0]), "y": float(wheel[1])},
        "angles": {
            "TR1_TR2_at_point2_deg": float(TR_angle * 180.0 / math.pi),
        },
    }


def plot_two_poses(
    pose_a: Dict,
    pose_b: Dict,
    title_a: str,
    title_b: str,
    out_path: Optional[str] = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    import numpy as np

    def draw(ax, pose: Dict, title: str):
        P = {int(k): np.array([v["x"], v["y"]]) for k, v in pose["points"].items()}
        wheel = np.array([pose["wheel_midpoint"]["x"], pose["wheel_midpoint"]["y"]])

        # rods
        for name, i, j in RODS:
            if i in P and j in P:
                ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], linewidth=2)
                mx, my = 0.5 * (P[i][0] + P[j][0]), 0.5 * (P[i][1] + P[j][1])
                ax.text(mx, my, name, fontsize=8)

        # points
        for i in sorted(P.keys()):
            ax.scatter([P[i][0]], [P[i][1]], s=40)
            ax.text(P[i][0], P[i][1], f"  {i}", fontsize=10, va="center")

        ax.scatter([wheel[0]], [wheel[1]], s=70, marker="x")
        ax.text(wheel[0], wheel[1], "  wheel", fontsize=9, va="center")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m) (+Y down)")
        ax.set_title(title)
        ax.grid(True)

        xs = [p[0] for p in P.values()] + [wheel[0]]
        ys = [p[1] for p in P.values()] + [wheel[1]]
        pad = 0.15 * max(1e-3, (max(xs) - min(xs)), (max(ys) - min(ys)))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.invert_yaxis()
        ax.invert_xaxis()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    draw(ax1, pose_a, title_a)
    draw(ax2, pose_b, title_b)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--Hext",
        type=float,
        required=True,
        help="Extended height from point 3 to wheel midpoint (m). +Y is down.",
    )
    ap.add_argument(
        "--ax1", type=float, required=True, help="Fixed AX1 length = dist(1,3) (m)."
    )
    ap.add_argument(
        "--close_scale",
        type=float,
        default=0.02,
        help="Tolerance (m) for 'similar' penalties.",
    )
    ap.add_argument("--out", type=str, default=None, help="Output JSON path.")
    ap.add_argument("--plot", type=str, default=None, help="Output plot PNG path.")
    ap.add_argument("--show", action="store_true", help="Show interactive plot window.")
    args = ap.parse_args()

    cfg = Config(Hext=args.Hext, ax1=args.ax1, close_scale=args.close_scale)
    sol = solve(cfg)

    # Use the designed lengths to solve two poses
    lengths = sol["lengths"]

    pose80 = solve_pose_given_lengths(cfg, lengths, ratio=0.80)
    pose20 = solve_pose_given_lengths(cfg, lengths, ratio=0.20)

    # Save/plot
    if args.plot or args.show:
        plot_two_poses(
            pose80,
            pose20,
            title_a=f"Pose @ 80% (wheel y={pose80['wheel_midpoint']['y']:.3f})",
            title_b=f"Pose @ 20% (wheel y={pose20['wheel_midpoint']['y']:.3f})",
            out_path=args.plot,
            show=args.show,
        )

    s = json.dumps(sol, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
        print(f"Wrote JSON: {args.out}")
    else:
        print(s)


if __name__ == "__main__":
    main()
