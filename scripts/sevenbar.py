#!/usr/bin/env python3

# Your solver is doing least-squares optimization to satisfy a set of distance constraints (rigid bars) 
# plus non-overlap and branch-selection constraints, either to design link lengths that work across poses, 
# or to compute the mechanism configuration for a driven input angle θ.

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares, differential_evolution

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


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class Config:
    Hext: float
    ax1: float  # fixed AX1 length = dist(1,3); with x3=0 => P3=(0, ax1) if +Y down

    # weights
    w_len: float = 250.0  # rod length constraints
    w_pose: float = 900.0  # wheel y-position constraint
    w_soft: float = 60.0  # similarity constraints (TR1~DR, CR1~CR2, etc.)
    w_soft_ineq: float = 60.0
    w_reg: float = 1e-2
    w_TR: float = 0.01  # TR bar thickness
    w_DR: float = 0.01  # DR bar thickness
    w_CR1: float = 0.01  # CR1 bar thickness
    w_CR2: float = 0.01  # CR2 bar thickness
    gap: float = 0.002
    w_clear: float = 1500.0  # clearance penalty weight
    w_above: float = 1200.0  # weight for point7 above point2 preference
    w_tr_orient: float = (
        1200.0  # weight for TR triangle orientation consistency across poses
    )
    tr_orient_margin: float = (
        1e-5  # small positive margin to avoid near-degenerate sign flips
    )
    close_scale: float = 0.02  # meters tolerance for "similar"
    ratios: Tuple[float, ...] = (1.0, 0.8, 0.2)
    above_margin: float = 0.008  # meters
    w_wheel_x: float = 3000.0  # penalize wheel midpoint x drift (single constraint per pose)

    # Multi-start and global optimization settings
    n_starts: int = 8  # number of random starts for multi-start optimization
    use_global_opt: bool = True  # use differential evolution first
    global_maxiter: int = 500  # max iterations for global optimizer
    global_popsize: int = 15  # population size for differential evolution
    global_tol: float = 1e-4  # tolerance for global optimizer


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


# ------------------------------
# Pose solve for animation (DR rotates about point 3)
# ------------------------------


def pack_pose_vars(P: Dict[int, np.ndarray]) -> np.ndarray:
    order = [2, 5, 6, 7]
    out = []
    for i in order:
        out.extend([float(P[i][0]), float(P[i][1])])
    return np.array(out, dtype=float)


def unpack_pose_vars(
    v: np.ndarray, cfg: Config, P4: np.ndarray
) -> Dict[int, np.ndarray]:
    P: Dict[int, np.ndarray] = {}
    P[1] = np.array([0.0, 0.0], dtype=float)
    P[3] = np.array([0.0, cfg.ax1], dtype=float)
    P[4] = np.array([float(P4[0]), float(P4[1])], dtype=float)
    order = [2, 5, 6, 7]
    k = 0
    for i in order:
        P[i] = np.array([v[k], v[k + 1]], dtype=float)
        k += 2
    return P


def pack_vars(
    P_by_ratio: Dict[float, Dict[int, np.ndarray]], L: Dict[str, float], cfg: Config
) -> np.ndarray:
    # Packs all free points (per ratio) and then lengths (shared)
    order_pts = [2, 4, 5, 6, 7]
    x = []
    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        for i in order_pts:
            x.extend([P[i][0], P[i][1]])
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        x.append(float(L[k]))
    return np.array(x, dtype=float)


def unpack_vars(
    x: np.ndarray, cfg: Config
) -> Tuple[Dict[float, Dict[int, np.ndarray]], Dict[str, float]]:
    order_pts = [2, 4, 5, 6, 7]
    n_pts = len(order_pts)
    n_ratios = len(cfg.ratios)
    idx = 0
    P_by_ratio: Dict[float, Dict[int, np.ndarray]] = {}
    for ratio in cfg.ratios:
        P: Dict[int, np.ndarray] = {}
        P[1] = np.array([0.0, 0.0], dtype=float)
        P[3] = np.array([0.0, cfg.ax1], dtype=float)
        for i in order_pts:
            P[i] = np.array([x[idx], x[idx + 1]], dtype=float)
            idx += 2
        P_by_ratio[ratio] = P
    L = {}
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        L[k] = float(x[idx])
        idx += 1
    return P_by_ratio, L


def residuals(x: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute residuals for the linkage design optimization.

    Constraint structure:
    1. Rod length constraints (kinematic feasibility)
    2. Wheel y-position constraints (target extension ratios)
    3. Wheel x-position constraint (SINGLE per pose - keep wheel on y-axis)
    4. Clearance constraints (avoid bar collisions)
    5. Geometric preferences (point 7 above point 2, TR orientation)
    6. Similarity constraints (TR1~DR, CR1~CR2, AX lengths similar)
    7. Inequality constraints (triangle inequality, CSR length)
    8. Positivity + regularization
    """
    P_by_ratio, L = unpack_vars(x, cfg)
    r = []

    # --- TR rigid-body orientation consistency across poses ---
    # Prevent mirrored solutions where triangle (1,2,7) flips between poses.
    ref_ratio = 1.0 if 1.0 in cfg.ratios else cfg.ratios[0]
    Pref = P_by_ratio[ref_ratio]
    ref_cross = cross2(Pref[1] - Pref[2], Pref[7] - Pref[2])
    ref_sign = 1.0 if ref_cross >= 0.0 else -1.0

    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        wheel = 0.5 * (P[5] + P[6])

        # === ROD LENGTH CONSTRAINTS (kinematic feasibility) ===
        r.append((dist(P[1], P[2]) - L["LTR1"]) * cfg.w_len)  # TR1
        r.append((dist(P[2], P[7]) - L["LTR2"]) * cfg.w_len)  # TR2
        r.append((dist(P[1], P[7]) - L["L17"]) * cfg.w_len)  # TR closure 1-7
        r.append((dist(P[3], P[4]) - L["LDR"]) * cfg.w_len)  # DR
        r.append((dist(P[2], P[4]) - L["LAX2"]) * cfg.w_len)  # AX2
        r.append((dist(P[5], P[6]) - L["LAX3"]) * cfg.w_len)  # AX3
        r.append((dist(P[2], P[5]) - L["LCR1"]) * cfg.w_len)  # CR1
        r.append((dist(P[4], P[6]) - L["LCR2"]) * cfg.w_len)  # CR2
        r.append((dist(P[6], P[7]) - L["LCSR"]) * cfg.w_len)  # CSR

        # === WHEEL POSITION CONSTRAINTS ===
        # Wheel y-position: target extension for this ratio
        r.append((wheel[1] - (cfg.ax1 + ratio * cfg.Hext)) * cfg.w_pose)
        # Wheel x-position: keep on y-axis (SINGLE constraint per pose)
        r.append(wheel[0] * cfg.w_wheel_x)

        # === CLEARANCE CONSTRAINTS ===
        # TR1(1-2) vs DR(3-4) clearance
        d_req = 0.5 * (cfg.w_TR + cfg.w_DR) + cfg.gap
        d_seg = seg_seg_distance(P[1], P[2], P[3], P[4])
        violation = max(0.0, d_req - d_seg)
        r.append(violation * cfg.w_clear)
        # CR1(2-5) vs CR2(4-6) clearance
        d_req_cr = 0.5 * (cfg.w_CR1 + cfg.w_CR2) + cfg.gap
        d_seg_cr = seg_seg_distance(P[2], P[5], P[4], P[6])
        violation_cr = max(0.0, d_req_cr - d_seg_cr)
        r.append(violation_cr * cfg.w_clear)

        # === GEOMETRIC PREFERENCES ===
        # Point 7 above point 2 (y7 <= y2 - margin; +Y down)
        above_violation = max(0.0, P[7][1] - P[2][1] + cfg.above_margin)
        r.append(above_violation * cfg.w_above)
        # TR triangle orientation consistent with reference pose
        tri_cross = cross2(P[1] - P[2], P[7] - P[2])
        orient_violation = max(0.0, cfg.tr_orient_margin - (ref_sign * tri_cross))
        r.append(orient_violation * cfg.w_tr_orient)

    # === SIMILARITY CONSTRAINTS (applied once, not per pose) ===
    s = max(1e-6, cfg.close_scale)
    # TR1 ~ DR (similar, not necessarily equal)
    r.append(((L["LDR"] - L["LTR1"]) / s) * cfg.w_soft)
    # CR1 ~ CR2 (similar, not necessarily equal)
    r.append(((L["LCR2"] - L["LCR1"]) / s) * cfg.w_soft)
    # AX2 ~ AX1, AX3 ~ AX1, AX2 ~ AX3 (AX1 fixed = cfg.ax1)
    r.append(((L["LAX2"] - cfg.ax1) / s) * cfg.w_soft)
    r.append(((L["LAX3"] - cfg.ax1) / s) * cfg.w_soft)
    r.append(((L["LAX3"] - L["LAX2"]) / s) * (0.5 * cfg.w_soft))

    # === INEQUALITY CONSTRAINTS ===
    margin = 1e-3
    # Triangle inequality for TR (L17 < LTR1 + LTR2)
    tri_violation = max(0.0, L["L17"] - (L["LTR1"] + L["LTR2"] - margin))
    r.append(tri_violation * cfg.w_soft_ineq)
    # CSR should be long enough
    cr_mean = 0.5 * (L["LCR1"] + L["LCR2"])
    csr_violation = max(0.0, 1.1 * cr_mean - L["LCSR"])
    r.append(csr_violation * cfg.w_soft_ineq)

    # === POSITIVITY + REGULARIZATION ===
    for k in ["LTR1", "LTR2", "L17", "LDR", "LAX2", "LAX3", "LCR1", "LCR2", "LCSR"]:
        if L[k] <= 0:
            r.append((abs(L[k]) + 1.0) * 1e6)
        else:
            r.append(cfg.w_reg * L[k])

    return np.array(r, dtype=float)


def generate_initial_seed(cfg: Config, seed_idx: int = 0) -> np.ndarray:
    """Generate an initial guess for the optimization.

    Different seed_idx values produce different starting configurations
    to enable multi-start optimization.
    """
    rng = np.random.default_rng(seed=seed_idx * 1337 + 42)

    # Base geometry parameters with some randomization
    scale = 0.8 + 0.4 * rng.random()  # 0.8 to 1.2 scale factor
    x_offset = (rng.random() - 0.5) * 0.1  # random x offset for points

    P_by_ratio: Dict[float, Dict[int, np.ndarray]] = {}
    for ratio in cfg.ratios:
        wheel_y = cfg.ax1 + ratio * cfg.Hext
        P: Dict[int, np.ndarray] = {}
        P[1] = np.array([0.0, 0.0], dtype=float)
        P[3] = np.array([0.0, cfg.ax1], dtype=float)
        # Randomize starting positions
        P[2] = np.array([0.06 * scale + x_offset, 0.05 * scale], dtype=float)
        P[4] = np.array([0.12 * scale + x_offset, cfg.ax1 + 0.05 * scale], dtype=float)
        # Keep wheel points symmetric around y-axis for better x=0 convergence
        wheel_spread = 0.03 * scale
        P[5] = np.array([0.0, wheel_y - wheel_spread], dtype=float)
        P[6] = np.array([0.0, wheel_y + wheel_spread], dtype=float)
        P[7] = np.array([0.10 * scale + x_offset, 0.02 * scale], dtype=float)
        P_by_ratio[ratio] = P

    # Use the 1.0 pose to seed lengths
    P1 = P_by_ratio[1.0] if 1.0 in cfg.ratios else P_by_ratio[cfg.ratios[0]]
    LTR1 = dist(P1[1], P1[2])
    LTR2 = dist(P1[2], P1[7])
    L17 = dist(P1[1], P1[7])
    LDR = LTR1 * (0.95 + 0.1 * rng.random())  # similar to TR1
    LAX2 = cfg.ax1 * (0.98 + 0.04 * rng.random())
    LAX3 = cfg.ax1 * (0.98 + 0.04 * rng.random())
    LCR1 = dist(P1[2], P1[5])
    LCR2 = LCR1 * (0.98 + 0.04 * rng.random())  # similar to CR1
    LCSR = max(1.2 * 0.5 * (LCR1 + LCR2), dist(P1[6], P1[7]))

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
    return pack_vars(P_by_ratio, L, cfg)


def get_variable_bounds(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Get bounds for all optimization variables.

    Returns (lower_bounds, upper_bounds) arrays.
    """
    order_pts = [2, 4, 5, 6, 7]
    n_pts = len(order_pts)
    n_ratios = len(cfg.ratios)
    n_lengths = 9  # LTR1, LTR2, L17, LDR, LAX2, LAX3, LCR1, LCR2, LCSR

    # Total variables: 2 coords per point, per ratio, plus lengths
    n_vars = n_pts * 2 * n_ratios + n_lengths

    # Geometric bounds based on Hext and ax1
    max_extent = max(cfg.Hext * 2, 1.0)

    lower = np.zeros(n_vars)
    upper = np.zeros(n_vars)

    idx = 0
    # Point coordinates
    for _ in cfg.ratios:
        for _ in order_pts:
            # x coordinate
            lower[idx] = -max_extent
            upper[idx] = max_extent
            idx += 1
            # y coordinate
            lower[idx] = -max_extent
            upper[idx] = cfg.ax1 + cfg.Hext * 1.5
            idx += 1

    # Length bounds
    min_length = 0.001
    max_length = max_extent * 1.5
    for _ in range(n_lengths):
        lower[idx] = min_length
        upper[idx] = max_length
        idx += 1

    return lower, upper


def objective_scalar(x: np.ndarray, cfg: Config) -> float:
    """Scalar objective for global optimization (sum of squared residuals)."""
    r = residuals(x, cfg)
    return float(np.sum(r ** 2))


def evaluate_solution_quality(x: np.ndarray, cfg: Config) -> Dict:
    """Evaluate the quality of a solution, focusing on wheel x-drift."""
    P_by_ratio, L = unpack_vars(x, cfg)

    max_wheel_x = 0.0
    total_wheel_x = 0.0
    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        wheel = 0.5 * (P[5] + P[6])
        max_wheel_x = max(max_wheel_x, abs(wheel[0]))
        total_wheel_x += abs(wheel[0])

    r = residuals(x, cfg)
    cost = float(np.sum(r ** 2))

    return {
        "cost": cost,
        "max_wheel_x": max_wheel_x,
        "mean_wheel_x": total_wheel_x / len(cfg.ratios),
    }


def solve_local(cfg: Config, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Run local least-squares optimization from a given starting point."""
    lower, upper = get_variable_bounds(cfg)
    res = least_squares(
        lambda v: residuals(v, cfg),
        x0,
        method="trf",
        bounds=(lower, upper),
        max_nfev=80000,
    )
    return res.x, float(res.cost), bool(res.success)


def solve_global(cfg: Config, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Run global optimization using differential evolution."""
    lower, upper = get_variable_bounds(cfg)
    bounds = list(zip(lower, upper))

    if verbose:
        print("Running global optimization (differential_evolution)...")
        print(f"  Population size: {cfg.global_popsize}")
        print(f"  Max iterations: {cfg.global_maxiter}")

    result = differential_evolution(
        lambda v: objective_scalar(v, cfg),
        bounds,
        maxiter=cfg.global_maxiter,
        popsize=cfg.global_popsize,
        tol=cfg.global_tol,
        seed=42,
        workers=1,  # Use 1 for reproducibility; set to -1 for parallel
        updating="deferred",
        polish=False,  # We'll do our own local refinement
    )

    if verbose:
        print(f"  Global optimization cost: {result.fun:.6e}")
        print(f"  Converged: {result.success}")

    return result.x, float(result.fun)


def solve_multistart(cfg: Config, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """Run multi-start local optimization from multiple random seeds."""
    best_x = None
    best_cost = float("inf")
    best_quality = None

    if verbose:
        print(f"Running multi-start optimization ({cfg.n_starts} starts)...")

    for i in range(cfg.n_starts):
        x0 = generate_initial_seed(cfg, seed_idx=i)
        x_opt, cost, success = solve_local(cfg, x0)

        quality = evaluate_solution_quality(x_opt, cfg)

        if verbose:
            print(
                f"  Start {i+1}/{cfg.n_starts}: cost={cost:.6e}, "
                f"max_wheel_x={quality['max_wheel_x']:.6f}, success={success}"
            )

        # Prefer solutions with lower wheel x-drift when costs are similar
        # Use a combined metric: cost + penalty for wheel x drift
        wheel_penalty = quality["max_wheel_x"] * cfg.w_wheel_x * 10
        combined_metric = cost + wheel_penalty

        if best_x is None or combined_metric < (
            best_cost + (best_quality["max_wheel_x"] * cfg.w_wheel_x * 10 if best_quality else 0)
        ):
            best_x = x_opt
            best_cost = cost
            best_quality = quality

    if verbose:
        print(f"  Best: cost={best_cost:.6e}, max_wheel_x={best_quality['max_wheel_x']:.6f}")

    return best_x, best_cost


def solve(cfg: Config, x0: Optional[np.ndarray] = None, verbose: bool = False) -> Dict:
    """
    Main solve function with two-stage optimization:
    1. Stage 1: Global optimization (differential_evolution) OR multi-start
    2. Stage 2: Local refinement (least_squares)

    If x0 is provided, skip global optimization and use x0 directly.
    """
    if x0 is not None:
        # Use provided starting point, just do local refinement
        if verbose:
            print("Using provided initial guess, running local refinement only...")
        x_opt, cost, success = solve_local(cfg, x0)
    else:
        # Two-stage optimization
        if cfg.use_global_opt:
            # Stage 1a: Global optimization
            x_global, cost_global = solve_global(cfg, verbose=verbose)

            # Stage 1b: Also run multi-start and compare
            x_multi, cost_multi = solve_multistart(cfg, verbose=verbose)

            # Pick the better starting point for local refinement
            if cost_multi < cost_global:
                x_stage1 = x_multi
                if verbose:
                    print("Multi-start found better solution than global opt")
            else:
                x_stage1 = x_global
                if verbose:
                    print("Global opt found better solution than multi-start")
        else:
            # Just use multi-start
            x_stage1, _ = solve_multistart(cfg, verbose=verbose)

        # Stage 2: Local refinement
        if verbose:
            print("Stage 2: Local refinement...")
        x_opt, cost, success = solve_local(cfg, x_stage1)

        if verbose:
            quality = evaluate_solution_quality(x_opt, cfg)
            print(f"Final: cost={cost:.6e}, max_wheel_x={quality['max_wheel_x']:.6f}")

    # Build output
    P_by_ratio, L = unpack_vars(x_opt, cfg)

    out = {
        "success": bool(success) if x0 is not None else True,
        "cost": float(cost),
        "Hext": cfg.Hext,
        "ax1_fixed": cfg.ax1,
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
        "poses": {},
    }

    # Compute quality metrics
    max_wheel_x = 0.0
    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        wheel = 0.5 * (P[5] + P[6])
        max_wheel_x = max(max_wheel_x, abs(wheel[0]))
        TR_angle = angle_at(P[2], P[1], P[7])
        out["poses"][str(ratio)] = {
            "points": {
                str(i): {"x": float(P[i][0]), "y": float(P[i][1])}
                for i in sorted(P.keys())
            },
            "wheel_midpoint": {"x": float(wheel[0]), "y": float(wheel[1])},
            "angles": {
                "TR1_TR2_at_point2_deg": float(TR_angle * 180.0 / math.pi),
                "TR1_TR2_at_point2_rad": float(TR_angle),
            },
        }

    out["quality"] = {
        "max_wheel_x_drift": max_wheel_x,
    }

    return out


# ------------------------------
# Pose solver for a given DR angle (theta about point 3)
# ------------------------------


def solve_pose_theta(
    cfg: Config,
    lengths: Dict[str, float],
    theta_rad: float,
    x0: Optional[np.ndarray] = None,
    enforce_above: bool = True,
    enforce_clearance: bool = True,
    enforce_tr_orient: bool = True,
    enforce_wheel_x: bool = True,
) -> Dict:
    """Solve a single pose given fixed link lengths and a DR rotation angle about point 3.

    DR is segment (3-4) of length DR(3-4). We set P4 = P3 + LDR*[cos(theta), sin(theta)]
    in the solver's internal coordinates (+Y down).

    Returns a dict with points, wheel_midpoint, angles, and a 'seed' vector you can reuse as x0.
    """

    # Map length names from the design output
    LTR1 = float(lengths["TR1(1-2)"])
    LTR2 = float(lengths["TR2(2-7)"])
    L17 = float(lengths["TR_1_7"])
    LDR = float(lengths["DR(3-4)"])
    LAX2 = float(lengths["AX2(2-4)"])
    LAX3 = float(lengths["AX3(5-6)"])
    LCR1 = float(lengths["CR1(2-5)"])
    LCR2 = float(lengths["CR2(4-6)"])
    LCSR = float(lengths["CSR(6-7)"])

    P3 = np.array([0.0, cfg.ax1], dtype=float)
    P4 = P3 + np.array([math.cos(theta_rad), math.sin(theta_rad)], dtype=float) * LDR

    # Seed guess
    if x0 is None:
        # Roughly place points near the designed geometry: keep wheel near mid of AX3
        wheel_y_guess = cfg.ax1 + 0.8 * cfg.Hext
        Pseed: Dict[int, np.ndarray] = {
            2: np.array([0.06, 0.05], dtype=float),
            5: np.array([-0.05, wheel_y_guess - 0.03], dtype=float),
            6: np.array([0.05, wheel_y_guess + 0.03], dtype=float),
            7: np.array([0.10, 0.02], dtype=float),
        }
        x0 = pack_pose_vars(Pseed)

    # Reference TR orientation sign from the initial guess (stabilizes branch)
    Pref = unpack_pose_vars(x0, cfg, P4)
    ref_cross = cross2(Pref[1] - Pref[2], Pref[7] - Pref[2])
    ref_sign = 1.0 if ref_cross >= 0.0 else -1.0

    def pose_res(v: np.ndarray) -> np.ndarray:
        P = unpack_pose_vars(v, cfg, P4)
        wheel = 0.5 * (P[5] + P[6])
        rr = []

        # Hard rod constraints (dist - length)
        rr.append((dist(P[1], P[2]) - LTR1) * cfg.w_len)  # TR1
        rr.append((dist(P[2], P[7]) - LTR2) * cfg.w_len)  # TR2
        rr.append((dist(P[1], P[7]) - L17) * cfg.w_len)  # TR closure

        rr.append((dist(P[2], P[4]) - LAX2) * cfg.w_len)  # AX2
        rr.append((dist(P[5], P[6]) - LAX3) * cfg.w_len)  # AX3

        rr.append((dist(P[2], P[5]) - LCR1) * cfg.w_len)  # CR1
        rr.append((dist(P[4], P[6]) - LCR2) * cfg.w_len)  # CR2

        rr.append((dist(P[6], P[7]) - LCSR) * cfg.w_len)  # CSR

        # Optional constraints
        if enforce_clearance:
            # TR1 vs DR
            d_req = 0.5 * (cfg.w_TR + cfg.w_DR) + cfg.gap
            d_seg = seg_seg_distance(P[1], P[2], P[3], P[4])
            rr.append(max(0.0, d_req - d_seg) * cfg.w_clear)

            # CR1 vs CR2
            d_req_cr = 0.5 * (cfg.w_CR1 + cfg.w_CR2) + cfg.gap
            d_seg_cr = seg_seg_distance(P[2], P[5], P[4], P[6])
            rr.append(max(0.0, d_req_cr - d_seg_cr) * cfg.w_clear)

        if enforce_above:
            rr.append(max(0.0, P[7][1] - P[2][1] + cfg.above_margin) * cfg.w_above)

        if enforce_tr_orient:
            tri_cross = cross2(P[1] - P[2], P[7] - P[2])
            rr.append(
                max(0.0, cfg.tr_orient_margin - (ref_sign * tri_cross))
                * cfg.w_tr_orient
            )

        if enforce_wheel_x:
            # Soft preference to keep wheel on y-axis (helps select correct branch)
            rr.append(wheel[0] * cfg.w_wheel_x)

        # Small regularization to keep things bounded
        rr.append(1e-3 * float(np.linalg.norm(v)))

        return np.array(rr, dtype=float)

    ls = least_squares(pose_res, x0, method="trf", max_nfev=20000)
    P = unpack_pose_vars(ls.x, cfg, P4)
    wheel = 0.5 * (P[5] + P[6])
    TR_angle = angle_at(P[2], P[1], P[7])

    out = {
        "success": bool(ls.success),
        "status": int(ls.status),
        "message": str(ls.message),
        "nfev": int(ls.nfev),
        "cost": float(ls.cost),
        "theta_rad": float(theta_rad),
        "theta_deg": float(theta_rad * 180.0 / math.pi),
        "points": {
            str(i): {"x": float(P[i][0]), "y": float(P[i][1])} for i in sorted(P.keys())
        },
        "wheel_midpoint": {"x": float(wheel[0]), "y": float(wheel[1])},
        "angles": {"TR1_TR2_at_point2_deg": float(TR_angle * 180.0 / math.pi)},
        "seed": ls.x.tolist(),
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


## solve_pose_given_lengths is no longer needed and removed.


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


# ------------------------------
# Plotting helper for a single pose
# ------------------------------


def plot_pose_matplotlib(ax, pose: Dict, title: str = ""):
    P = {
        int(k): np.array([v["x"], v["y"]], dtype=float)
        for k, v in pose["points"].items()
    }
    wheel = np.array(
        [pose["wheel_midpoint"]["x"], pose["wheel_midpoint"]["y"]], dtype=float
    )

    for name, i, j in RODS:
        if i in P and j in P:
            ax.plot([P[i][0], P[j][0]], [P[i][1], P[j][1]], linewidth=2)

    for i in sorted(P.keys()):
        ax.scatter([P[i][0]], [P[i][1]], s=40)
        ax.text(P[i][0], P[i][1], f"  {i}", fontsize=10, va="center")

    ax.scatter([wheel[0]], [wheel[1]], s=70, marker="x")
    ax.text(wheel[0], wheel[1], "  wheel", fontsize=9, va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m) (+Y down)")
    if title:
        ax.set_title(title)

    xs = [p[0] for p in P.values()] + [wheel[0]]
    ys = [p[1] for p in P.values()] + [wheel[1]]
    pad = 0.15 * max(1e-3, (max(xs) - min(xs)), (max(ys) - min(ys)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # Animation view: invert both axes
    ax.invert_xaxis()
    ax.invert_yaxis()


def theta_sweep_stats(
    cfg: Config, lengths: Dict[str, float], theta_min_deg=30.0, theta_max_deg=70.0, n=21
):
    thetas = np.linspace(theta_min_deg, theta_max_deg, n)
    seed = None
    xs = []
    for td in thetas:
        pose = solve_pose_theta(
            cfg,
            lengths,
            math.radians(float(td)),
            x0=None if seed is None else np.array(seed),
        )
        seed = pose["seed"]
        xs.append(pose["wheel_midpoint"]["x"])
    xs = np.array(xs, dtype=float)
    return {
        "theta_min_deg": theta_min_deg,
        "theta_max_deg": theta_max_deg,
        "n": n,
        "max_abs_xwheel": float(np.max(np.abs(xs))),
        "mean_abs_xwheel": float(np.mean(np.abs(xs))),
    }


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

    stats = theta_sweep_stats(cfg, sol["lengths"], 30, 70, 21)
    print(stats)

    # Extract poses for 80% and 20% extension
    pose80 = sol["poses"]["0.8"]
    pose20 = sol["poses"]["0.2"]

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
