import math
from typing import List, Tuple

Point = Tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def orientation(a: Point, b: Point, c: Point) -> float:
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    (ax, ay), (bx, by), (px, py) = a, b, p
    if (min(ax, bx) - eps <= px <= max(ax, bx) + eps) and (min(ay, by) - eps <= py <= max(ay, by) + eps):
        return abs(orientation(a, b, p)) < eps
    return False


def segments_intersect_strict(p1: Point, p2: Point, p3: Point, p4: Point, eps: float = 1e-9) -> bool:
    """
    True if segments intersect in their interiors. Endpoint touching is allowed.
    Colinear interior overlaps are treated as intersection (collision).
    """
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    # Proper intersection
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True

    # Colinear overlaps (excluding endpoints)
    if abs(o1) < eps and on_segment(p1, p2, p3, eps) and p3 not in (p1, p2):
        return True
    if abs(o2) < eps and on_segment(p1, p2, p4, eps) and p4 not in (p1, p2):
        return True
    if abs(o3) < eps and on_segment(p3, p4, p1, eps) and p1 not in (p3, p4):
        return True
    if abs(o4) < eps and on_segment(p3, p4, p2, eps) and p2 not in (p3, p4):
        return True

    return False


def circle_circle_intersections(c0: Point, r0: float, c1: Point, r1: float, eps: float = 1e-9) -> List[Point]:
    """
    Return 0/1/2 intersection points of two circles.
    """
    x0, y0 = c0
    x1, y1 = c1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)

    if d < eps:
        return []
    if d > r0 + r1 + eps:
        return []
    if d < abs(r0 - r1) - eps:
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h2 = r0 * r0 - a * a

    if h2 < 0:
        if h2 > -1e-9:
            h2 = 0.0
        else:
            return []
    h = math.sqrt(h2)

    x2 = x0 + a * dx / d
    y2 = y0 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    p3 = (x2 + rx, y2 + ry)
    p4 = (x2 - rx, y2 - ry)

    if dist(p3, p4) < 1e-8:
        return [p3]
    return [p3, p4]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def rms(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def max_contiguous_span_deg(values: List[float], step_deg: float, tol: float = 1e-6) -> float:
    """
    Largest contiguous span in degrees given sampled points at step_deg.
    If values = [-120, -118, -116, -110, -108] and step=2 => returns 4.
    """
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