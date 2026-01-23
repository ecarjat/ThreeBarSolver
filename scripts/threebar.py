#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.optimize import least_squares, differential_evolution


def dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.hypot(d[0], d[1]))


def compute_wheel(knee: np.ndarray, inner: np.ndarray, lk_wheel: float) -> np.ndarray:
    v = inner - knee
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return knee + np.array([0.0, lk_wheel], dtype=float)
    u = v / n
    return knee - u * lk_wheel


def bc_from_params(xbc: float, ybc: float, cfg: "Config") -> np.ndarray:
    """Map raw params to a Bc point, smoothly constrained inside bc_radius_max if set."""
    if cfg.bc_radius_max is None:
        return np.array([xbc, ybc], dtype=float)
    r_max = float(cfg.bc_radius_max)
    if r_max <= 0.0:
        return np.array([xbc, ybc], dtype=float)
    norm = float(math.hypot(xbc, ybc))
    if norm < 1e-12:
        return np.array([0.0, 0.0], dtype=float)
    # Smooth radial squash: r = r_max * tanh(norm / r_max), always < r_max.
    scale = (r_max * math.tanh(norm / r_max)) / norm
    return np.array([xbc * scale, ybc * scale], dtype=float)

# --- Angle wrapping utility ---
def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> bool:
    if (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps) and (
        min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    ):
        return abs(orientation(a, b, p)) < eps
    return False


def segments_intersect_strict(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, eps: float = 1e-9
) -> bool:
    """True if segments intersect in their interiors. Endpoint touching allowed."""
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    if abs(o1) < eps and on_segment(p1, p2, p3, eps) and not np.allclose(p3, p1) and not np.allclose(p3, p2):
        return True
    if abs(o2) < eps and on_segment(p1, p2, p4, eps) and not np.allclose(p4, p1) and not np.allclose(p4, p2):
        return True
    if abs(o3) < eps and on_segment(p3, p4, p1, eps) and not np.allclose(p1, p3) and not np.allclose(p1, p4):
        return True
    if abs(o4) < eps and on_segment(p3, p4, p2, eps) and not np.allclose(p2, p3) and not np.allclose(p2, p4):
        return True

    return False


def seg_point_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    c = a + t * ab
    return float(np.linalg.norm(p - c))


def seg_seg_distance(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> float:
    if segments_intersect_strict(a, b, c, d):
        return 0.0
    return min(
        seg_point_distance(a, c, d),
        seg_point_distance(b, c, d),
        seg_point_distance(c, a, b),
        seg_point_distance(d, a, b),
    )


def has_crossing(P_by_ratio: Dict[float, Dict[str, np.ndarray]], Bc: np.ndarray) -> bool:
    H = np.array([0.0, 0.0], dtype=float)
    for ratio in P_by_ratio:
        P = P_by_ratio[ratio]
        if segments_intersect_strict(H, P["K"], Bc, P["C"]):
            return True
    return False


@dataclass
class Config:
    Hcrouch: float
    Hext: float
    ratios: Tuple[float, ...] = (0.0, 0.5, 1.0)

    # weights
    w_len: float = 250.0
    w_pose: float = 900.0
    w_wheel_x: float = 2000.0
    w_wheel_x_mean: float = 200.0
    w_knee_above_wheel: float = 800.0
    knee_above_margin: float = 0.0
    w_below: float = 400.0
    w_reg: float = 1e-2

    # multi-start/global optimization
    n_starts: int = 8
    use_global_opt: bool = False
    global_maxiter: int = 400
    global_popsize: int = 15
    global_tol: float = 1e-4

    # jumping / transmission shaping (wheel vertical motion vs hip motor angle)
    w_jac_profile: float = 120.0
    w_jac_bounds: float = 120.0
    w_theta_span: float = 200.0
    w_theta_monotonic: float = 200.0

    # dy/dtheta targets (m/rad). For jumping you usually want low J early (high force)
    # and higher J late (higher takeoff speed).
    jac_start: float = 0.18
    jac_end: float = 0.38
    jac_min: float = 0.08
    jac_max: float = 0.6

    # encourage using a meaningful hip rotation range (rad) and avoid angle reversals
    theta_span_min: float = 0.9
    theta_step_min: float = 0.02

    # minimum allowed inner joint offset KC (m)
    kc_min: float = 0.05
    # maximum allowed inner joint offset KC (m)
    kc_max: Optional[float] = 0.1

    # keep C-W length similar to H-K (ratio bounds)
    cw_hk_ratio_min: Optional[float] = 0.9
    cw_hk_ratio_max: Optional[float] = 1.1
    w_cw_hk_ratio: float = 600.0

    # keep Bc-C length proportional to H-K (ratio bounds)
    lc_hk_ratio_min: Optional[float] = 1.1
    lc_hk_ratio_max: Optional[float] = 1.1
    w_lc_hk_ratio: float = 1200.0

    # enforce positive pin joint x (xbc >= xbc_min)
    xbc_min: Optional[float] = 0.0
    w_xbc_min: float = 800.0

    # bias Bc movement toward x instead of y (higher weight => stronger penalty)
    w_bc_x: float = 1e-2
    w_bc_y: float = 5e-2

    # keep Bc within a max radius from H (m)
    bc_radius_max: Optional[float] = 0.1
    w_bc_radius: float = 800.0

    # prevent crossing between H-K and Bc-C links
    cross_min: float = 0.001
    w_no_cross: float = 200000.0

    # optional: max hip motor speed at the joint (rad/s), used to estimate takeoff vertical speed
    omega_max: Optional[float] = 11.0
    

def validate_config(cfg: Config) -> None:
    if not cfg.ratios:
        raise ValueError("Invalid config: ratios must be a non-empty sequence.")
    if cfg.n_starts < 1:
        raise ValueError("Invalid config: n_starts must be >= 1.")
    if cfg.kc_max is not None and float(cfg.kc_min) > float(cfg.kc_max):
        raise ValueError(
            f"Invalid bounds: kc_min ({cfg.kc_min}) exceeds kc_max ({cfg.kc_max})."
        )
    if (
        cfg.cw_hk_ratio_min is not None
        and cfg.cw_hk_ratio_max is not None
        and float(cfg.cw_hk_ratio_min) > float(cfg.cw_hk_ratio_max)
    ):
        raise ValueError(
            "Invalid bounds: cw_hk_ratio_min exceeds cw_hk_ratio_max."
        )
    if (
        cfg.lc_hk_ratio_min is not None
        and cfg.lc_hk_ratio_max is not None
        and float(cfg.lc_hk_ratio_min) > float(cfg.lc_hk_ratio_max)
    ):
        raise ValueError(
            "Invalid bounds: lc_hk_ratio_min exceeds lc_hk_ratio_max."
        )
    if float(cfg.jac_min) > float(cfg.jac_max):
        raise ValueError("Invalid bounds: jac_min exceeds jac_max.")
    if cfg.bc_radius_max is not None and cfg.xbc_min is not None:
        if float(cfg.bc_radius_max) > 0.0 and float(cfg.xbc_min) > float(cfg.bc_radius_max):
            raise ValueError(
                "Invalid bounds: xbc_min exceeds bc_radius_max; "
                "no feasible pin joint location exists."
            )

def height_for_ratio(cfg: Config, ratio: float) -> float:
    return cfg.Hcrouch + ratio * (cfg.Hext - cfg.Hcrouch)


def pack_vars(
    P_by_ratio: Dict[float, Dict[str, np.ndarray]],
    params: Dict[str, float],
    cfg: Config,
) -> np.ndarray:
    order_pts = ["K", "C"]
    x = []
    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        for name in order_pts:
            x.extend([float(P[name][0]), float(P[name][1])])
    x.extend(
        [
            float(params["Lu"]),
            float(params["Lkw"]),
            float(params["Lkc"]),
            float(params["Lc"]),
            float(params["xbc"]),
            float(params["ybc"]),
        ]
    )
    return np.array(x, dtype=float)


def unpack_vars(
    x: np.ndarray, cfg: Config
) -> Tuple[Dict[float, Dict[str, np.ndarray]], Dict[str, float]]:
    P_by_ratio: Dict[float, Dict[str, np.ndarray]] = {}
    idx = 0
    for ratio in cfg.ratios:
        K = np.array([x[idx], x[idx + 1]], dtype=float)
        idx += 2
        C = np.array([x[idx], x[idx + 1]], dtype=float)
        idx += 2
        P_by_ratio[ratio] = {"K": K, "C": C}

    params = {
        "Lu": float(x[idx]),
        "Lkw": float(x[idx + 1]),
        "Lkc": float(x[idx + 2]),
        "Lc": float(x[idx + 3]),
        "xbc": float(x[idx + 4]),
        "ybc": float(x[idx + 5]),
    }
    return P_by_ratio, params


def build_bounds(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    validate_config(cfg)
    max_h = max(abs(cfg.Hcrouch), abs(cfg.Hext), 0.2)
    max_extent = max(2.5 * max_h, 1.0)
    min_len = 0.05 * max_h
    max_len = 3.0 * max_h

    n_ratios = len(cfg.ratios)
    n_vars = n_ratios * 4 + 6
    lower = np.full(n_vars, -max_extent, dtype=float)
    upper = np.full(n_vars, max_extent, dtype=float)

    len_start = n_ratios * 4
    for i in range(4):
        lower[len_start + i] = min_len
        upper[len_start + i] = max_len

    # Hard lower bound for Lkc (inner_joint_offset_KC)
    lkc_index = len_start + 2
    lower[lkc_index] = max(lower[lkc_index], float(cfg.kc_min))
    if lower[lkc_index] >= upper[lkc_index]:
        upper[lkc_index] = lower[lkc_index] + max(1e-3, 0.05 * lower[lkc_index])
    if cfg.kc_max is not None:
        upper[lkc_index] = min(upper[lkc_index], float(cfg.kc_max))
        if upper[lkc_index] <= lower[lkc_index]:
            upper[lkc_index] = lower[lkc_index] + max(1e-3, 0.05 * lower[lkc_index])

    xbc_index = len_start + 4
    ybc_index = len_start + 5
    # Hard lower bound for xbc (pin joint x) only when not using radius squash.
    if cfg.xbc_min is not None and cfg.bc_radius_max is None:
        lower[xbc_index] = max(lower[xbc_index], float(cfg.xbc_min))
        if lower[xbc_index] >= upper[xbc_index]:
            upper[xbc_index] = lower[xbc_index] + max(1e-3, 0.05 * lower[xbc_index])

    return lower, upper


def residuals(x: np.ndarray, cfg: Config) -> np.ndarray:
    P_by_ratio, params = unpack_vars(x, cfg)
    H = np.array([0.0, 0.0], dtype=float)
    Bc = bc_from_params(params["xbc"], params["ybc"], cfg)
    Lu = params["Lu"]
    Lkw = params["Lkw"]
    Lkc = params["Lkc"]
    Lc = params["Lc"]

    r: List[float] = []
    # Keep C-W length similar to H-K (ratio bounds)
    if cfg.cw_hk_ratio_min is not None or cfg.cw_hk_ratio_max is not None:
        hk_len = max(float(Lu), 1e-9)
        cw_len = float(Lkc + Lkw)
        ratio = cw_len / hk_len
        if cfg.cw_hk_ratio_min is not None:
            r.append(
                max(0.0, float(cfg.cw_hk_ratio_min) - ratio) * cfg.w_cw_hk_ratio
            )
        if cfg.cw_hk_ratio_max is not None:
            r.append(
                max(0.0, ratio - float(cfg.cw_hk_ratio_max)) * cfg.w_cw_hk_ratio
            )

    # Keep Bc-C length proportional to H-K (ratio bounds)
    if cfg.lc_hk_ratio_min is not None or cfg.lc_hk_ratio_max is not None:
        hk_len = max(float(Lu), 1e-9)
        ratio = float(Lc) / hk_len
        if cfg.lc_hk_ratio_min is not None:
            r.append(
                max(0.0, float(cfg.lc_hk_ratio_min) - ratio) * cfg.w_lc_hk_ratio
            )
        if cfg.lc_hk_ratio_max is not None:
            r.append(
                max(0.0, ratio - float(cfg.lc_hk_ratio_max)) * cfg.w_lc_hk_ratio
            )

    if cfg.xbc_min is not None:
        r.append(max(0.0, float(cfg.xbc_min) - float(Bc[0])) * cfg.w_xbc_min)
    wheels: List[np.ndarray] = []
    wheel_ys: List[float] = []
    hip_thetas: List[float] = []
    ratios_sorted = tuple(sorted(cfg.ratios))

    if cfg.bc_radius_max is not None:
        bc_r = float(np.linalg.norm(Bc - H))
        r.append(max(0.0, bc_r - float(cfg.bc_radius_max)) * cfg.w_bc_radius)

    for ratio in ratios_sorted:
        P = P_by_ratio[ratio]
        K = P["K"]
        C = P["C"]
        W = compute_wheel(K, C, Lkw)
        wheels.append(W)
        wheel_ys.append(float(W[1]))
        hip_thetas.append(float(math.atan2(K[1] - H[1], K[0] - H[0])))

        r.append((dist(H, K) - Lu) * cfg.w_len)
        r.append((dist(K, C) - Lkc) * cfg.w_len)
        r.append((dist(Bc, C) - Lc) * cfg.w_len)

        target_y = height_for_ratio(cfg, ratio)
        r.append((W[1] - target_y) * cfg.w_pose)

        knee_above_violation = max(0.0, K[1] - W[1] + cfg.knee_above_margin)
        r.append(knee_above_violation * cfg.w_knee_above_wheel)

        r.append(max(0.0, -K[1]) * cfg.w_below)
        r.append(max(0.0, -C[1]) * cfg.w_below)
        r.append(max(0.0, -W[1]) * cfg.w_below)

        d_cross = seg_seg_distance(H, K, Bc, C)
        cross_violation = max(0.0, cfg.cross_min - d_cross)
        cross_scale = cross_violation / max(cfg.cross_min, 1e-9)
        r.append(cross_scale * cfg.w_no_cross)

    mean_wx = float(sum(W[0] for W in wheels) / max(1, len(wheels)))
    for W in wheels:
        r.append((W[0] - mean_wx) * cfg.w_wheel_x)
    r.append(mean_wx * cfg.w_wheel_x_mean)

    # --- Jumping-oriented transmission shaping ---
    # Approximate J = dy/dtheta between samples, where theta is the hip motor angle.
    # Small J => high force (torque -> vertical GRF), large J => higher extension speed.
    if len(ratios_sorted) >= 2:
        # expected direction of motion in hip angle
        dtheta_total = wrap_pi(hip_thetas[-1] - hip_thetas[0])
        expected_sign = 1.0 if dtheta_total >= 0.0 else -1.0

        # Encourage a meaningful usable angle span
        theta_span = abs(dtheta_total)
        r.append(max(0.0, cfg.theta_span_min - theta_span) * cfg.w_theta_span)

        for i in range(len(ratios_sorted) - 1):
            r0 = float(ratios_sorted[i])
            r1 = float(ratios_sorted[i + 1])
            dy = float(wheel_ys[i + 1] - wheel_ys[i])
            dtheta = wrap_pi(hip_thetas[i + 1] - hip_thetas[i])

            # Penalize near-singular steps where motor angle barely changes
            small_dtheta_pen = max(0.0, 1e-4 - abs(dtheta)) * 1e6
            r.append(small_dtheta_pen)

            dtheta_safe = dtheta
            if abs(dtheta_safe) < 1e-4:
                dtheta_safe = math.copysign(1e-4, dtheta_safe if dtheta_safe != 0.0 else 1.0)

            # Monotonic hip motion (avoid angle reversals through the stroke)
            r.append(
                max(0.0, cfg.theta_step_min - (expected_sign * dtheta_safe))
                * cfg.w_theta_monotonic
            )

            J = dy / dtheta_safe  # m/rad
            t = 0.5 * (r0 + r1)
            J_target = cfg.jac_start + t * (cfg.jac_end - cfg.jac_start)

            # Track desired profile: high force early (low J), speed late (higher J)
            Jabs = abs(J)
            r.append((Jabs - J_target) * cfg.w_jac_profile)

            # Soft bounds to stay away from extreme MA / near-singular behavior
            r.append(max(0.0, cfg.jac_min - Jabs) * cfg.w_jac_bounds)
            r.append(max(0.0, Jabs - cfg.jac_max) * cfg.w_jac_bounds)

    for val in (Lu, Lkw, Lkc, Lc):
        if val <= 0:
            r.append((abs(val) + 1.0) * 1e6)
        else:
            r.append(val * cfg.w_reg)

    r.append(float(Bc[0]) * cfg.w_bc_x)
    r.append(float(Bc[1]) * cfg.w_bc_y)
    return np.array(r, dtype=float)


def generate_initial_seed(cfg: Config, seed_idx: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed_idx * 1337 + 11)
    scale = max(abs(cfg.Hcrouch), abs(cfg.Hext), 0.2)

    P_by_ratio: Dict[float, Dict[str, np.ndarray]] = {}
    for ratio in cfg.ratios:
        target_y = height_for_ratio(cfg, ratio)
        kx = (rng.random() - 0.5) * 0.2 * scale
        ky = target_y * (0.55 + 0.15 * rng.random())
        c_offset = (0.15 + 0.25 * rng.random()) * scale
        cx = kx + (rng.random() - 0.5) * 0.1 * scale
        cy = max(0.02 * scale, ky - c_offset)
        P_by_ratio[ratio] = {
            "K": np.array([kx, ky], dtype=float),
            "C": np.array([cx, cy], dtype=float),
        }

    ref_ratio = 1.0 if 1.0 in cfg.ratios else cfg.ratios[-1]
    Pref = P_by_ratio[ref_ratio]
    Kref = Pref["K"]
    Cref = Pref["C"]
    Wref = np.array([0.0, height_for_ratio(cfg, ref_ratio)], dtype=float)

    Lu = dist(np.array([0.0, 0.0], dtype=float), Kref)
    Lkc = dist(Kref, Cref)
    Lkw = max(0.1 * scale, dist(Kref, Wref))

    xbc = (rng.random() - 0.5) * 0.6 * scale
    ybc = (rng.random() - 0.3) * 0.6 * scale
    Bc = np.array([xbc, ybc], dtype=float)
    Lc = dist(Bc, Cref)

    params = {
        "Lu": Lu,
        "Lkw": Lkw,
        "Lkc": Lkc,
        "Lc": Lc,
        "xbc": xbc,
        "ybc": ybc,
    }
    return pack_vars(P_by_ratio, params, cfg)


def evaluate_solution_quality(x: np.ndarray, cfg: Config) -> Dict[str, float]:
    P_by_ratio, params = unpack_vars(x, cfg)
    Lkw = params["Lkw"]
    wheel_x: List[float] = []
    wheel_y: List[float] = []
    hip_theta: List[float] = []
    for ratio in cfg.ratios:
        K = P_by_ratio[ratio]["K"]
        C = P_by_ratio[ratio]["C"]
        W = compute_wheel(K, C, Lkw)
        wheel_x.append(float(W[0]))
        wheel_y.append(float(W[1]))
        hip_theta.append(float(math.atan2(K[1], K[0])))

    if not wheel_x:
        return {"max_wheel_x": 0.0, "wheel_x_pp": 0.0, "wheel_x_rms": 0.0}

    mean_x = sum(wheel_x) / len(wheel_x)
    rms_x = math.sqrt(sum((x - mean_x) ** 2 for x in wheel_x) / len(wheel_x))
    return {
        "max_wheel_x": max(abs(x) for x in wheel_x),
        "wheel_x_pp": max(wheel_x) - min(wheel_x),
        "wheel_x_rms": rms_x,
        "wheel_y_span": max(wheel_y) - min(wheel_y),
        "mean_wheel_x": mean_x,
        "theta_span": abs(wrap_pi(hip_theta[-1] - hip_theta[0])) if hip_theta else 0.0,
    }

# --- Compute jump report ---
def compute_jump_report(P_by_ratio: Dict[float, Dict[str, np.ndarray]], params: Dict[str, float], cfg: Config) -> Dict[str, float]:
    """Compute simple kinematic jump metrics based on samples.

    J = dy/dtheta where theta is the hip motor angle and y is wheel vertical position.
    """
    Lkw = float(params["Lkw"])
    H = np.array([0.0, 0.0], dtype=float)

    ratios_sorted = tuple(sorted(cfg.ratios))
    if len(ratios_sorted) < 2:
        return {
            "J_min": 0.0,
            "J_max": 0.0,
            "J_start": 0.0,
            "J_end": 0.0,
            "theta_span": 0.0,
            "y_dot_takeoff_est": 0.0,
        }

    wheel_ys: List[float] = []
    hip_thetas: List[float] = []
    for ratio in ratios_sorted:
        K = P_by_ratio[ratio]["K"]
        C = P_by_ratio[ratio]["C"]
        W = compute_wheel(K, C, Lkw)
        wheel_ys.append(float(W[1]))
        hip_thetas.append(float(math.atan2(K[1] - H[1], K[0] - H[0])))

    Js: List[float] = []
    for i in range(len(ratios_sorted) - 1):
        dy = float(wheel_ys[i + 1] - wheel_ys[i])
        dtheta = wrap_pi(hip_thetas[i + 1] - hip_thetas[i])
        if abs(dtheta) < 1e-9:
            continue
        Js.append(dy / dtheta)

    if not Js:
        J_min = J_max = J_start = J_end = 0.0
    else:
        J_min = float(min(Js))
        J_max = float(max(Js))
        J_start = float(Js[0])
        J_end = float(Js[-1])

    theta_span = float(abs(wrap_pi(hip_thetas[-1] - hip_thetas[0])))

    y_dot_takeoff_est = 0.0
    if cfg.omega_max is not None:
        try:
            y_dot_takeoff_est = float(J_end) * float(cfg.omega_max)
        except Exception:
            y_dot_takeoff_est = 0.0

    return {
        "J_min": J_min,
        "J_max": J_max,
        "J_start": J_start,
        "J_end": J_end,
        "theta_span": theta_span,
        "y_dot_takeoff_est": y_dot_takeoff_est,
    }

def solve_local(cfg: Config, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    lower, upper = build_bounds(cfg)
    eps = 1e-9
    x0 = np.minimum(np.maximum(x0, lower + eps), upper - eps)
    ls = least_squares(
        lambda v: residuals(v, cfg),
        x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=20000,
    )
    return ls.x, float(ls.cost), bool(ls.success)


def solve_global(cfg: Config, verbose: bool = False) -> Tuple[np.ndarray, float]:
    lower, upper = build_bounds(cfg)
    bounds = list(zip(lower.tolist(), upper.tolist()))

    def objective(v: np.ndarray) -> float:
        r = residuals(v, cfg)
        return float(np.dot(r, r))

    if verbose:
        print("Running differential evolution...")

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=cfg.global_maxiter,
        popsize=cfg.global_popsize,
        tol=cfg.global_tol,
        polish=False,
        updating="deferred",
    )
    return np.array(result.x, dtype=float), float(result.fun)


def solve_multistart(cfg: Config, verbose: bool = False) -> Tuple[np.ndarray, float]:
    best_x = None
    best_cost = float("inf")
    best_quality = None
    best_x_any = None
    best_cost_any = float("inf")

    if verbose:
        print(f"Running multi-start optimization ({cfg.n_starts} starts)...")

    for i in range(cfg.n_starts):
        x0 = generate_initial_seed(cfg, seed_idx=i)
        x_opt, cost, success = solve_local(cfg, x0)
        quality = evaluate_solution_quality(x_opt, cfg)

        if cost < best_cost_any:
            best_cost_any = cost
            best_x_any = x_opt

        if verbose:
            print(
                f"  Start {i+1}/{cfg.n_starts}: cost={cost:.6e}, "
                f"max_wheel_x={quality['max_wheel_x']:.6f}, success={success}"
            )

        P_by_ratio, params = unpack_vars(x_opt, cfg)
        Bc = bc_from_params(params["xbc"], params["ybc"], cfg)
        if has_crossing(P_by_ratio, Bc):
            if verbose:
                print("    Crossing detected; skipping solution.")
            continue

        score = cost + quality["max_wheel_x"] * cfg.w_wheel_x
        if best_x is None or score < (
            best_cost + (best_quality["max_wheel_x"] * cfg.w_wheel_x if best_quality else 0.0)
        ):
            best_x = x_opt
            best_cost = cost
            best_quality = quality

    if best_x is None:
        if verbose:
            print("  No non-crossing solution found; returning best available.")
        return best_x_any, best_cost_any

    if verbose and best_quality is not None:
        print(
            f"  Best: cost={best_cost:.6e}, max_wheel_x={best_quality['max_wheel_x']:.6f}"
        )

    return best_x, best_cost


def solve(cfg: Config, x0: Optional[np.ndarray] = None, verbose: bool = False) -> Dict:
    if x0 is not None:
        if verbose:
            print("Using provided initial guess, running local refinement only...")
        x_opt, cost, success = solve_local(cfg, x0)
    else:
        if cfg.use_global_opt:
            x_global, cost_global = solve_global(cfg, verbose=verbose)
            x_multi, cost_multi = solve_multistart(cfg, verbose=verbose)
            if cost_multi < cost_global:
                x_stage1 = x_multi
                if verbose:
                    print("Multi-start found better solution than global opt")
            else:
                x_stage1 = x_global
                if verbose:
                    print("Global opt found better solution than multi-start")
        else:
            x_stage1, _ = solve_multistart(cfg, verbose=verbose)

        if verbose:
            print("Local refinement...")
        x_opt, cost, success = solve_local(cfg, x_stage1)

    P_by_ratio, params = unpack_vars(x_opt, cfg)
    Lkw = params["Lkw"]
    Bc = bc_from_params(params["xbc"], params["ybc"], cfg)
    crossing = has_crossing(P_by_ratio, Bc)

    jump_report = compute_jump_report(P_by_ratio, params, cfg)

    out = {
        "success": bool(success) and not crossing,
        "cost": float(cost),
        "Hcrouch": cfg.Hcrouch,
        "Hext": cfg.Hext,
        "lengths": {
            "upper_leg_HK": float(params["Lu"]),
            "lower_leg_KW": float(params["Lkw"]),
            "link_BcC": float(params["Lc"]),
        },
        "pin_joint": {"x": float(Bc[0]), "y": float(Bc[1])},
        "inner_joint_offset_KC": float(params["Lkc"]),
        "jump_report": jump_report,
        "poses": {},
        "crossing": bool(crossing),
    }

    wheel_xs: List[float] = []
    wheel_ys: List[float] = []
    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        K = P["K"]
        C = P["C"]
        W = compute_wheel(K, C, Lkw)
        wheel_xs.append(float(W[0]))
        wheel_ys.append(float(W[1]))
        out["poses"][str(ratio)] = {
            "target_wheel_y": float(height_for_ratio(cfg, ratio)),
            "points": {
                "H": {"x": 0.0, "y": 0.0},
                "K": {"x": float(K[0]), "y": float(K[1])},
                "C": {"x": float(C[0]), "y": float(C[1])},
                "W": {"x": float(W[0]), "y": float(W[1])},
                "Bc": {"x": float(Bc[0]), "y": float(Bc[1])},
            },
        }

    if wheel_xs:
        mean_wx = sum(wheel_xs) / len(wheel_xs)
        rms_wx = math.sqrt(sum((x - mean_wx) ** 2 for x in wheel_xs) / len(wheel_xs))
        out["quality"] = {
            "max_wheel_x": max(abs(x) for x in wheel_xs),
            "wheel_x_pp": max(wheel_xs) - min(wheel_xs),
            "wheel_x_rms": rms_wx,
            "wheel_y_span": max(wheel_ys) - min(wheel_ys),
            "mean_wheel_x": mean_wx,
            "crossing": bool(crossing),
        }
    else:
        out["quality"] = {
            "max_wheel_x": 0.0,
            "wheel_x_pp": 0.0,
            "wheel_x_rms": 0.0,
            "wheel_y_span": 0.0,
            "mean_wheel_x": 0.0,
            "crossing": bool(crossing),
        }
    if crossing:
        out["message"] = "Crossing detected between H-K and Bc-C."
    if verbose:
        jr = out.get("jump_report", {})
        print(
            "Jump report: "
            f"J_start={jr.get('J_start', 0.0):.6f}, "
            f"J_end={jr.get('J_end', 0.0):.6f}, "
            f"J_min={jr.get('J_min', 0.0):.6f}, "
            f"J_max={jr.get('J_max', 0.0):.6f}, "
            f"theta_span={jr.get('theta_span', 0.0):.6f} rad, "
            f"y_dot_takeoff_est={jr.get('y_dot_takeoff_est', 0.0):.6f} m/s"
        )

    return out


def solve_pose_ratio(
    cfg: Config,
    *,
    lengths: Dict[str, float],
    pin_joint: Dict[str, float],
    inner_joint_offset_kc: float,
    ratio: float,
    x0: Optional[np.ndarray] = None,
) -> Dict:
    validate_config(cfg)
    if cfg.kc_max is not None and float(inner_joint_offset_kc) > float(cfg.kc_max):
        raise ValueError(
            f"Invalid inner_joint_offset_kc ({inner_joint_offset_kc}) exceeds kc_max ({cfg.kc_max})."
        )
    if float(inner_joint_offset_kc) < float(cfg.kc_min):
        raise ValueError(
            f"Invalid inner_joint_offset_kc ({inner_joint_offset_kc}) below kc_min ({cfg.kc_min})."
        )
    H = np.array([0.0, 0.0], dtype=float)
    Bc = bc_from_params(float(pin_joint["x"]), float(pin_joint["y"]), cfg)
    Lu = float(lengths["upper_leg_HK"])
    Lkw = float(lengths["lower_leg_KW"])
    Lkc = float(inner_joint_offset_kc)
    Lc = float(lengths["link_BcC"])

    target_y = height_for_ratio(cfg, ratio)

    if x0 is None:
        kx = 0.0
        ky = target_y * 0.6
        cx = 0.1 * Lu
        cy = ky - 0.4 * Lu
        x0 = np.array([kx, ky, cx, cy], dtype=float)

    def pose_res(v: np.ndarray) -> np.ndarray:
        K = np.array([v[0], v[1]], dtype=float)
        C = np.array([v[2], v[3]], dtype=float)
        W = compute_wheel(K, C, Lkw)

        r: List[float] = []
        r.append((dist(H, K) - Lu) * cfg.w_len)
        r.append((dist(K, C) - Lkc) * cfg.w_len)
        r.append((dist(Bc, C) - Lc) * cfg.w_len)

        r.append((W[1] - target_y) * cfg.w_pose)
        r.append(W[0] * cfg.w_wheel_x)

        knee_above_violation = max(0.0, K[1] - W[1] + cfg.knee_above_margin)
        r.append(knee_above_violation * cfg.w_knee_above_wheel)

        r.append(max(0.0, -K[1]) * cfg.w_below)
        r.append(max(0.0, -C[1]) * cfg.w_below)
        r.append(max(0.0, -W[1]) * cfg.w_below)

        d_cross = seg_seg_distance(H, K, Bc, C)
        cross_violation = max(0.0, cfg.cross_min - d_cross)
        cross_scale = cross_violation / max(cfg.cross_min, 1e-9)
        r.append(cross_scale * cfg.w_no_cross)

        r.append(1e-3 * float(np.linalg.norm(v)))
        return np.array(r, dtype=float)

    ls = least_squares(pose_res, x0, method="trf", max_nfev=20000)
    K = np.array([ls.x[0], ls.x[1]], dtype=float)
    C = np.array([ls.x[2], ls.x[3]], dtype=float)
    W = compute_wheel(K, C, Lkw)
    crossing = segments_intersect_strict(H, K, Bc, C)

    return {
        "success": bool(ls.success) and not crossing,
        "status": int(ls.status),
        "message": str(ls.message),
        "nfev": int(ls.nfev),
        "cost": float(ls.cost),
        "ratio": float(ratio),
        "target_wheel_y": float(target_y),
        "points": {
            "H": {"x": float(H[0]), "y": float(H[1])},
            "K": {"x": float(K[0]), "y": float(K[1])},
            "C": {"x": float(C[0]), "y": float(C[1])},
            "W": {"x": float(W[0]), "y": float(W[1])},
            "Bc": {"x": float(Bc[0]), "y": float(Bc[1])},
        },
        "crossing": bool(crossing),
        "seed": ls.x.tolist(),
    }


def parse_ratios(values: Optional[List[float]], samples: int) -> Tuple[float, ...]:
    if values:
        return tuple(sorted(set(values)))
    if samples < 2:
        return (0.0, 1.0)
    return tuple(np.linspace(0.0, 1.0, samples).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-bar solver for hip/knee/inner linkage (hip at (0,0), +Y down)"
    )

    parser.add_argument("--Hcrouch", type=float, required=True, help="crouched wheel height")
    parser.add_argument("--Hext", type=float, required=True, help="extended wheel height")
    parser.add_argument(
        "--kc-min",
        type=float,
        default=0.05,
        help="minimum inner joint offset KC (inner_joint_offset_KC) in meters",
    )
    parser.add_argument(
        "--kc-max",
        type=float,
        default=0.1,
        help="maximum inner joint offset KC (inner_joint_offset_KC) in meters (<=0 disables)",
    )
    parser.add_argument(
        "--cw-hk-ratio-min",
        type=float,
        default=0.9,
        help="minimum ratio (C-W)/(H-K). Set <=0 to disable.",
    )
    parser.add_argument(
        "--cw-hk-ratio-max",
        type=float,
        default=1.1,
        help="maximum ratio (C-W)/(H-K). Set <=0 to disable.",
    )
    parser.add_argument(
        "--lc-hk-ratio-min",
        type=float,
        default=1.1,
        help="minimum ratio (Bc-C)/(H-K). Set <=0 to disable.",
    )
    parser.add_argument(
        "--lc-hk-ratio-max",
        type=float,
        default=1.1,
        help="maximum ratio (Bc-C)/(H-K). Set <=0 to disable.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=11.0,
        help="optional max hip motor speed at the joint (rad/s) used to estimate takeoff vertical speed",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        default=None,
        help="ratios between crouch/ext (0..1). Overrides --samples",
    )
    parser.add_argument("--samples", type=int, default=3, help="number of samples if no ratios")
    parser.add_argument("--n-starts", type=int, default=8, help="multi-start count")
    parser.add_argument("--global", dest="use_global", action="store_true", help="use global opt")
    parser.add_argument("--no-global", dest="use_global", action="store_false")
    parser.set_defaults(use_global=False)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    ratios = parse_ratios(args.ratios, args.samples)
    cfg = Config(
        Hcrouch=args.Hcrouch,
        Hext=args.Hext,
        ratios=ratios,
        n_starts=args.n_starts,
        use_global_opt=args.use_global,
        kc_min=args.kc_min,
        kc_max=float(args.kc_max) if float(args.kc_max) > 0.0 else None,
        cw_hk_ratio_min=float(args.cw_hk_ratio_min)
        if float(args.cw_hk_ratio_min) > 0.0
        else None,
        cw_hk_ratio_max=float(args.cw_hk_ratio_max)
        if float(args.cw_hk_ratio_max) > 0.0
        else None,
        lc_hk_ratio_min=float(args.lc_hk_ratio_min)
        if float(args.lc_hk_ratio_min) > 0.0
        else None,
        lc_hk_ratio_max=float(args.lc_hk_ratio_max)
        if float(args.lc_hk_ratio_max) > 0.0
        else None,
        omega_max=args.omega_max,
    )

    sol = solve(cfg, verbose=args.verbose)
    print(json.dumps(sol, indent=2))


if __name__ == "__main__":
    main()
