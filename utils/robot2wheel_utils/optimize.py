import math
import random
from typing import Dict, Optional, Tuple

from .kinematics import Design
from .scoring import score_design


def optimize_from_Ll(
    Ll: float,
    *,
    alpha_min_deg: float = -160.0,
    alpha_max_deg: float = -20.0,
    alpha_step_deg: float = 2.0,
    n_random: int = 8000,
    refine_iters: int = 1500,
    seed: int = 2,
    # Bc bounds (compactness control)
    xbc_min_factor: float = -2.5,
    xbc_max_factor: float = -0.2,
    ybc_min_factor: float = 0.1,
    ybc_max_factor: float = 1.5,
    # targeted connector search
    target_bc: Optional[Tuple[float, float]] = None,
    target_bc_radius: float = 5.0,
    # scoring knobs
    min_span_deg: float = 30.0,
    w_bc: float = 0.0,   # set >0 to penalize far Bc
    show_progress: bool = False,
) -> Tuple[Optional[Design], Optional[Dict]]:
    """
    Random search + local refinement.
    Given Ll, optimise Lu, Lkc, Lc, xbc, ybc.

    If you want Bc closer to H:
      - tighten bounds via *_factor args
      - and/or set w_bc > 0
    """
    random.seed(seed)

    # Length ranges relative to Ll (broad priors; tighten later)
    Lu_min, Lu_max = 0.6 * Ll, 2.0 * Ll
    Lkc_min, Lkc_max = 0.4 * Ll, 1.8 * Ll
    Lc_min, Lc_max = 0.6 * Ll, 2.4 * Ll

    xbc_min, xbc_max = xbc_min_factor * Ll, xbc_max_factor * Ll
    ybc_min, ybc_max = ybc_min_factor * Ll, ybc_max_factor * Ll

    radius = max(0.0, target_bc_radius)
    if target_bc is not None:
        tx, ty = target_bc
        xbc_min = max(xbc_min, tx - radius)
        xbc_max = min(xbc_max, tx + radius)
        ybc_min = max(ybc_min, ty - radius)
        ybc_max = min(ybc_max, ty + radius)

    def clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(value, hi))

    best_d: Optional[Design] = None
    best_m: Optional[Dict] = None
    best_score: Optional[float] = None

    def sample_design() -> Design:
        Lu = random.uniform(Lu_min, Lu_max)
        Lkc = random.uniform(Lkc_min, Lkc_max)
        Lc = random.uniform(Lc_min, Lc_max)
        if target_bc is None or radius <= 0.0:
            xbc = random.uniform(xbc_min, xbc_max)
            ybc = random.uniform(ybc_min, ybc_max)
        else:
            xbc = clamp(random.uniform(target_bc[0] - radius, target_bc[0] + radius), xbc_min, xbc_max)
            ybc = clamp(random.uniform(target_bc[1] - radius, target_bc[1] + radius), ybc_min, ybc_max)
        return Design(Ll=Ll, Lu=Lu, Lkc=Lkc, Lc=Lc, xbc=xbc, ybc=ybc)

    # Global random search
    progress_interval = max(1, n_random // 20)
    for i in range(n_random):
        d = sample_design()
        m = score_design(
            d,
            alpha_min_deg=alpha_min_deg,
            alpha_max_deg=alpha_max_deg,
            alpha_step_deg=alpha_step_deg,
            min_span_deg=min_span_deg,
            w_bc=w_bc,
        )
        if m is None:
            continue
        if best_score is None or m["score"] < best_score:
            best_score = m["score"]
            best_d = d
            best_m = m
            if show_progress:
                print(
                    f"[Random search {i+1}/{n_random}] new best score {best_score:.4f} "
                    f"(Bc={best_d.xbc:.2f},{best_d.ybc:.2f})"
                )

        if show_progress and (i + 1) % progress_interval == 0:
            print(f"[Random search] {i+1}/{n_random} samples evaluated")

    # Local refinement around best
    if best_d is not None:
        d0 = best_d

        def jiggle(x: float, frac: float) -> float:
            return x * (1.0 + random.uniform(-frac, frac))

        def jiggle_bcx() -> float:
            if target_bc is None or radius <= 0.0:
                return jiggle(d0.xbc, 0.10)
            return clamp(random.uniform(target_bc[0] - radius, target_bc[0] + radius), xbc_min, xbc_max)

        def jiggle_bcy() -> float:
            if target_bc is None or radius <= 0.0:
                return jiggle(d0.ybc, 0.10)
            return clamp(random.uniform(target_bc[1] - radius, target_bc[1] + radius), ybc_min, ybc_max)

        refine_interval = max(1, refine_iters // 20)
        for j in range(refine_iters):
            d = Design(
                Ll=Ll,
                Lu=jiggle(d0.Lu, 0.08),
                Lkc=jiggle(d0.Lkc, 0.08),
                Lc=jiggle(d0.Lc, 0.08),
                xbc=jiggle_bcx(),
                ybc=jiggle_bcy(),
            )

            # Keep Bc above and behind
            if d.ybc <= 0 or d.xbc >= 0:
                continue

            # Keep within configured bounds
            if not (xbc_min <= d.xbc <= xbc_max and ybc_min <= d.ybc <= ybc_max):
                continue

            m = score_design(
                d,
                alpha_min_deg=alpha_min_deg,
                alpha_max_deg=alpha_max_deg,
                alpha_step_deg=alpha_step_deg,
                min_span_deg=min_span_deg,
                w_bc=w_bc,
            )
            if m is None:
                continue
            if best_score is None or m["score"] < best_score:
                best_score = m["score"]
                best_d = d
                best_m = m
                d0 = d
                if show_progress:
                    print(
                        f"[Refine {j+1}/{refine_iters}] new best score {best_score:.4f} "
                        f"(Bc={best_d.xbc:.2f},{best_d.ybc:.2f})"
                    )

            if show_progress and (j + 1) % refine_interval == 0:
                print(f"[Refine search] {j+1}/{refine_iters} iterations evaluated")

    return best_d, best_m
