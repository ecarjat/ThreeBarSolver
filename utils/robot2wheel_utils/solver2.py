import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

Point = Tuple[float, float]

# ----------------------------
# Basic geometry
# ----------------------------

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def orientation(a: Point, b: Point, c: Point) -> float:
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def on_segment(a: Point, b: Point, p: Point, eps=1e-9) -> bool:
    (ax, ay), (bx, by), (px, py) = a, b, p
    if min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps:
        return abs(orientation(a, b, p)) < eps
    return False

def segments_intersect_strict(p1: Point, p2: Point, p3: Point, p4: Point, eps=1e-9) -> bool:
    """True if segments intersect in interiors. Endpoint touching allowed."""
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    # Colinear overlaps treated as collision (excluding endpoints)
    if abs(o1) < eps and on_segment(p1, p2, p3, eps) and p3 not in (p1, p2): return True
    if abs(o2) < eps and on_segment(p1, p2, p4, eps) and p4 not in (p1, p2): return True
    if abs(o3) < eps and on_segment(p3, p4, p1, eps) and p1 not in (p3, p4): return True
    if abs(o4) < eps and on_segment(p3, p4, p2, eps) and p2 not in (p3, p4): return True
    return False

def circle_circle_intersections(c0: Point, r0: float, c1: Point, r1: float, eps=1e-9) -> List[Point]:
    """Return 0/1/2 intersection points of two circles."""
    x0, y0 = c0
    x1, y1 = c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d < eps: return []
    if d > r0 + r1 + eps: return []
    if d < abs(r0 - r1) - eps: return []

    a = (r0*r0 - r1*r1 + d*d) / (2*d)
    h2 = r0*r0 - a*a
    if h2 < 0:
        if h2 > -1e-9:
            h2 = 0.0
        else:
            return []
    h = math.sqrt(h2)

    x2 = x0 + a * dx / d
    y2 = y0 + a * dy / d

    rx = -dy * (h / d)
    ry =  dx * (h / d)

    p3 = (x2 + rx, y2 + ry)
    p4 = (x2 - rx, y2 - ry)

    if dist(p3, p4) < 1e-8:
        return [p3]
    return [p3, p4]

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def rms(xs: List[float]) -> float:
    if not xs: return 0.0
    m2 = sum(x*x for x in xs) / len(xs)
    return math.sqrt(m2)

def max_contiguous_span_deg(values: List[float], step_deg: float, tol: float = 1e-6) -> float:
    """Largest contiguous span in degrees given sampled points."""
    if not values:
        return 0.0
    values = sorted(values)
    best = 0.0
    start = values[0]
    prev = values[0]
    for v in values[1:]:
        if abs((v - prev) - step_deg) <= tol:
            prev = v
        else:
            best = max(best, prev - start)
            start = v
            prev = v
    best = max(best, prev - start)
    return best

# ----------------------------
# Model
# ----------------------------

@dataclass(frozen=True)
class Design:
    Ll: float     # K->W (INPUT)
    Lu: float     # H->K
    Lkc: float    # K->C
    Lc: float     # Bc->C
    xbc: float    # Bc.x (typically negative / behind hip)
    ybc: float    # Bc.y (must be > 0)

@dataclass
class Pose:
    alpha_deg: float
    H: Point
    Bc: Point
    K: Point
    C: Point
    W: Point

# ----------------------------
# Forward kinematics per alpha
# ----------------------------

def solve_pose_for_alpha(d: Design, alpha_deg: float) -> Optional[Pose]:
    H = (0.0, 0.0)
    Bc = (d.xbc, d.ybc)

    a = math.radians(alpha_deg)
    K = (d.Lu * math.cos(a), d.Lu * math.sin(a))

    # C from circle intersection around K and Bc
    Cs = circle_circle_intersections(K, d.Lkc, Bc, d.Lc)
    if not Cs:
        return None

    # pick the C that best satisfies "K,C,W below H" and avoids link intersection
    best = None
    best_score = None

    for C in Cs:
        # compute W as colinear extension opposite C from K
        vx, vy = (C[0] - K[0], C[1] - K[1])
        vnorm = math.hypot(vx, vy)
        if vnorm < 1e-9:
            continue
        ux, uy = vx / vnorm, vy / vnorm
        W = (K[0] - d.Ll * ux, K[1] - d.Ll * uy)

        # constraints:
        # - Bc above hip
        if d.ybc <= 0:
            continue
        # - K,C,W below hip line
        if not (K[1] < 0 and C[1] < 0 and W[1] < 0):
            continue
        # - no intersection between upper rod H-K and connector rod Bc-C
        if segments_intersect_strict(H, K, Bc, C):
            continue

        # prefer smaller |Wx| for quasi-verticality, and deeper wheel (more negative y) to avoid degeneracy
        score = abs(W[0]) + 0.01 * abs(W[1])
        if best_score is None or score < best_score:
            best_score = score
            best = Pose(alpha_deg=alpha_deg, H=H, Bc=Bc, K=K, C=C, W=W)

    return best

def simulate_design(
    d: Design,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_step_deg: float
) -> List[Pose]:
    poses = []
    for i in range(int((alpha_max_deg - alpha_min_deg) / alpha_step_deg) + 1):
        alpha = alpha_min_deg + i * alpha_step_deg
        p = solve_pose_for_alpha(d, alpha)
        if p is not None:
            poses.append(p)
    return poses

# ----------------------------
# Objective / scoring
# ----------------------------

def score_design(
    d: Design,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_step_deg: float,
) -> Optional[Dict]:
    poses = simulate_design(d, alpha_min_deg, alpha_max_deg, alpha_step_deg)
    if not poses:
        return None

    alphas = [p.alpha_deg for p in poses]
    span = max_contiguous_span_deg(alphas, alpha_step_deg)

    # Require a minimum usable span, otherwise reject
    if span < 30.0:
        return None

    wx = [p.W[0] for p in poses]
    wy = [p.W[1] for p in poses]

    wx_rms = rms(wx)
    wx_pp = (max(wx) - min(wx))  # peak-to-peak x drift
    # "quasi-linear vertical course": penalize x drift relative to y range
    y_range = max(wy) - min(wy)
    if y_range < 1e-6:
        return None
    straightness = wx_pp / abs(y_range)  # smaller is better

    # Preference: Lu close to Lkc
    match = abs(d.Lu - d.Lkc) / max(1.0, d.Ll)

    # Total score (lower is better)
    # You can tune these weights.
    w_span = 2.0      # reward span
    w_rms = 4.0       # penalize x RMS
    w_pp = 2.0        # penalize peak-to-peak x
    w_str = 6.0       # penalize "not straight"
    w_match = 1.0

    score = (
        -w_span * (span / (alpha_max_deg - alpha_min_deg)) +
        w_rms * (wx_rms / max(1.0, d.Ll)) +
        w_pp * (wx_pp / max(1.0, d.Ll)) +
        w_str * straightness +
        w_match * match
    )

    return {
        "score": score,
        "span_deg": span,
        "wx_rms": wx_rms,
        "wx_pp": wx_pp,
        "y_range": y_range,
        "straightness": straightness,
        "poses": poses
    }

# ----------------------------
# Optimizer: given Ll, find best (Lu, Lkc, Lc, xbc, ybc)
# ----------------------------

def optimize_from_Ll(
    Ll: float,
    *,
    alpha_min_deg: float = -160.0,
    alpha_max_deg: float = -20.0,
    alpha_step_deg: float = 2.0,
    n_random: int = 5000,
    seed: int = 1
) -> Tuple[Optional[Design], Optional[Dict]]:
    random.seed(seed)

    # --- Search ranges (scale these relative to Ll) ---
    # These are intentionally broad. Tighten once you get “good looking” candidates.
    Lu_min, Lu_max = 0.6*Ll, 2.0*Ll
    Lkc_min, Lkc_max = 0.4*Ll, 1.8*Ll
    Lc_min, Lc_max = 0.6*Ll, 2.4*Ll

    # Bc location: behind hip and above hip
    xbc_min, xbc_max = -1.2*Ll, -0.2*Ll
    ybc_min, ybc_max =  0.1*Ll,  0.9*Ll

    best_d = None
    best_m = None
    best_score = None

    for _ in range(n_random):
        Lu = random.uniform(Lu_min, Lu_max)
        Lkc = random.uniform(Lkc_min, Lkc_max)
        Lc  = random.uniform(Lc_min,  Lc_max)
        xbc = random.uniform(xbc_min, xbc_max)
        ybc = random.uniform(ybc_min, ybc_max)

        d = Design(Ll=Ll, Lu=Lu, Lkc=Lkc, Lc=Lc, xbc=xbc, ybc=ybc)
        m = score_design(d, alpha_min_deg, alpha_max_deg, alpha_step_deg)
        if m is None:
            continue

        if best_score is None or m["score"] < best_score:
            best_score = m["score"]
            best_d = d
            best_m = m

    # Optional: local refinement around best (small random perturbations)
    if best_d is not None:
        d0 = best_d
        for _ in range(1500):
            def jiggle(x, frac=0.08):
                return x * (1.0 + random.uniform(-frac, frac))

            d = Design(
                Ll=Ll,
                Lu=jiggle(d0.Lu),
                Lkc=jiggle(d0.Lkc),
                Lc=jiggle(d0.Lc),
                xbc=jiggle(d0.xbc, frac=0.10),
                ybc=jiggle(d0.ybc, frac=0.10),
            )
            # enforce Bc above hip and behind
            if d.ybc <= 0 or d.xbc >= 0:
                continue

            m = score_design(d, alpha_min_deg, alpha_max_deg, alpha_step_deg)
            if m is None:
                continue
            if m["score"] < best_score:
                best_score = m["score"]
                best_d = d
                best_m = m
                d0 = d

    return best_d, best_m

# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    Ll = 120.0  # <-- YOU set this
    d, m = optimize_from_Ll(Ll, n_random=8000, seed=5)

    if d is None:
        print("No solution found. Try increasing n_random, widening alpha range, or relaxing constraints.")
    else:
        print("Best design for Ll =", Ll)
        print(f"  H = (0,0)")
        print(f"  Bc = ({d.xbc:.2f}, {d.ybc:.2f})")
        print(f"  Lu  (H-K)  = {d.Lu:.2f}")
        print(f"  Lkc (K-C)  = {d.Lkc:.2f}")
        print(f"  Lc  (Bc-C) = {d.Lc:.2f}")
        print()
        print("Motion quality:")
        print(f"  contiguous alpha span = {m['span_deg']:.1f} deg")
        print(f"  Wx RMS                = {m['wx_rms']:.2f}")
        print(f"  Wx peak-to-peak        = {m['wx_pp']:.2f}")
        print(f"  Wy range               = {m['y_range']:.2f}")
        print(f"  straightness (Wx_pp/|Wy_range|) = {m['straightness']:.4f}")
        print(f"  score                  = {m['score']:.4f}")