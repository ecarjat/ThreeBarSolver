import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .geometry import Point, circle_circle_intersections, segments_intersect_strict

@dataclass(frozen=True)
class Design:
    # Fixed in body frame
    Ll: float   # K->W (INPUT)
    Lu: float   # H->K
    Lkc: float  # K->C
    Lc: float   # Bc->C
    xbc: float  # Bc.x (typically <0)
    ybc: float  # Bc.y (must be >0)

@dataclass
class Pose:
    alpha_deg: float
    H: Point
    Bc: Point
    K: Point
    C: Point
    W: Point


def solve_pose_for_alpha(
    d: Design,
    alpha_deg: float,
    *,
    require_KCW_below_H: bool = True,
    collision_check: bool = True,
) -> Optional[Pose]:
    """
    Given design + hip angle alpha, compute K on circle about H, solve C from 2-circle intersection,
    then compute W using colinearity W-K-C with K between W and C.

    Constraints:
      - Bc.y > 0
      - (optional) K,C,W below hip line: y < 0
      - (optional) rods H-K and Bc-C do not intersect (excluding endpoints)
    """
    H: Point = (0.0, 0.0)
    if d.ybc <= 0:
        return None
    Bc: Point = (d.xbc, d.ybc)

    a = math.radians(alpha_deg)
    K: Point = (d.Lu * math.cos(a), d.Lu * math.sin(a))

    Cs = circle_circle_intersections(K, d.Lkc, Bc, d.Lc)
    if not Cs:
        return None

    best: Optional[Pose] = None
    best_score: Optional[float] = None

    for C in Cs:
        vx, vy = (C[0] - K[0], C[1] - K[1])
        vnorm = math.hypot(vx, vy)
        if vnorm < 1e-9:
            continue
        ux, uy = vx / vnorm, vy / vnorm

        # W-K-C colinear, K between W and C
        W: Point = (K[0] - d.Ll * ux, K[1] - d.Ll * uy)

        if require_KCW_below_H:
            if not (K[1] < 0.0 and C[1] < 0.0 and W[1] < 0.0):
                continue

        if collision_check:
            if segments_intersect_strict(H, K, Bc, C):
                continue

        # Choose the branch that tends to keep the wheel near x=0 (helps "verticality")
        score = abs(W[0]) + 0.01 * abs(W[1])
        if best_score is None or score < best_score:
            best_score = score
            best = Pose(alpha_deg=alpha_deg, H=H, Bc=Bc, K=K, C=C, W=W)

    return best


def simulate_design(
    d: Design,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_step_deg: float,
) -> List[Pose]:
    poses: List[Pose] = []
    steps = int((alpha_max_deg - alpha_min_deg) / alpha_step_deg) + 1
    for i in range(steps):
        alpha = alpha_min_deg + i * alpha_step_deg
        p = solve_pose_for_alpha(d, alpha, require_KCW_below_H=True, collision_check=True)
        if p is not None:
            poses.append(p)
    return poses
