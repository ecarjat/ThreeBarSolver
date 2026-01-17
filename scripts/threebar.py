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

    return lower, upper


def residuals(x: np.ndarray, cfg: Config) -> np.ndarray:
    P_by_ratio, params = unpack_vars(x, cfg)
    H = np.array([0.0, 0.0], dtype=float)
    Bc = np.array([params["xbc"], params["ybc"]], dtype=float)
    Lu = params["Lu"]
    Lkw = params["Lkw"]
    Lkc = params["Lkc"]
    Lc = params["Lc"]

    r: List[float] = []
    wheels: List[np.ndarray] = []

    for ratio in cfg.ratios:
        P = P_by_ratio[ratio]
        K = P["K"]
        C = P["C"]
        W = compute_wheel(K, C, Lkw)
        wheels.append(W)

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

    mean_wx = float(sum(W[0] for W in wheels) / max(1, len(wheels)))
    for W in wheels:
        r.append((W[0] - mean_wx) * cfg.w_wheel_x)
    r.append(mean_wx * cfg.w_wheel_x_mean)

    for val in (Lu, Lkw, Lkc, Lc):
        if val <= 0:
            r.append((abs(val) + 1.0) * 1e6)
        else:
            r.append(val * cfg.w_reg)

    r.append(cfg.w_reg * float(np.linalg.norm(Bc)))
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
    for ratio in cfg.ratios:
        K = P_by_ratio[ratio]["K"]
        C = P_by_ratio[ratio]["C"]
        W = compute_wheel(K, C, Lkw)
        wheel_x.append(float(W[0]))
        wheel_y.append(float(W[1]))

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
    }


def solve_local(cfg: Config, x0: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    lower, upper = build_bounds(cfg)
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

        score = cost + quality["max_wheel_x"] * cfg.w_wheel_x
        if best_x is None or score < (
            best_cost + (best_quality["max_wheel_x"] * cfg.w_wheel_x if best_quality else 0.0)
        ):
            best_x = x_opt
            best_cost = cost
            best_quality = quality

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
    Bc = np.array([params["xbc"], params["ybc"]], dtype=float)

    out = {
        "success": bool(success),
        "cost": float(cost),
        "Hcrouch": cfg.Hcrouch,
        "Hext": cfg.Hext,
        "lengths": {
            "upper_leg_HK": float(params["Lu"]),
            "lower_leg_KW": float(params["Lkw"]),
            "link_BcC": float(params["Lc"]),
        },
        "pin_joint": {"x": float(params["xbc"]), "y": float(params["ybc"])},
        "inner_joint_offset_KC": float(params["Lkc"]),
        "poses": {},
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
        }
    else:
        out["quality"] = {
            "max_wheel_x": 0.0,
            "wheel_x_pp": 0.0,
            "wheel_x_rms": 0.0,
            "wheel_y_span": 0.0,
            "mean_wheel_x": 0.0,
        }

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
    H = np.array([0.0, 0.0], dtype=float)
    Bc = np.array([float(pin_joint["x"]), float(pin_joint["y"])], dtype=float)
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

        r.append(1e-3 * float(np.linalg.norm(v)))
        return np.array(r, dtype=float)

    ls = least_squares(pose_res, x0, method="trf", max_nfev=20000)
    K = np.array([ls.x[0], ls.x[1]], dtype=float)
    C = np.array([ls.x[2], ls.x[3]], dtype=float)
    W = compute_wheel(K, C, Lkw)

    return {
        "success": bool(ls.success),
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
    )

    sol = solve(cfg, verbose=args.verbose)
    print(json.dumps(sol, indent=2))


if __name__ == "__main__":
    main()
