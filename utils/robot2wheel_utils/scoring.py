import math
from typing import Dict, Optional

from .geometry import max_contiguous_span_deg, rms
from .kinematics import Design, simulate_design


def score_design(
    d: Design,
    *,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_step_deg: float,
    min_span_deg: float = 30.0,
    # weights
    w_span: float = 2.0,
    w_rms: float = 4.0,
    w_pp: float = 2.0,
    w_str: float = 6.0,
    w_match: float = 1.0,
    w_bc: float = 0.0,  # set >0 to prefer compact Bc
) -> Optional[Dict]:
    poses = simulate_design(d, alpha_min_deg, alpha_max_deg, alpha_step_deg)
    if not poses:
        return None

    alphas = [p.alpha_deg for p in poses]
    span = max_contiguous_span_deg(alphas, alpha_step_deg)

    if span < min_span_deg:
        return None

    wx = [p.W[0] for p in poses]
    wy = [p.W[1] for p in poses]

    wx_rms = rms(wx)
    wx_pp = max(wx) - min(wx)
    y_range = max(wy) - min(wy)
    if abs(y_range) < 1e-6:
        return None

    # "straightness": how much x drifts relative to y movement
    straightness = wx_pp / abs(y_range)

    # Preferences
    match = abs(d.Lu - d.Lkc) / max(1.0, d.Ll)
    bc_r = math.hypot(d.xbc, d.ybc)
    bc_cost = bc_r / max(1.0, d.Ll)

    # Score: lower is better
    score = (
        -w_span * (span / max(1e-6, (alpha_max_deg - alpha_min_deg))) +
        w_rms * (wx_rms / max(1.0, d.Ll)) +
        w_pp * (wx_pp / max(1.0, d.Ll)) +
        w_str * straightness +
        w_match * match +
        w_bc * bc_cost
    )

    return {
        "score": score,
        "span_deg": span,
        "wx_rms": wx_rms,
        "wx_pp": wx_pp,
        "y_range": y_range,
        "straightness": straightness,
        "match": match,
        "bc_r": bc_r,
        "poses": poses,
    }
