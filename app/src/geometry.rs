use crate::types::Point2D;
use std::f64::consts::PI;

/// Euclidean distance between two points
#[inline]
pub fn dist(a: &Point2D, b: &Point2D) -> f64 {
    (a - b).norm()
}

/// Wrap angle to [-pi, pi]
#[inline]
pub fn wrap_pi(angle: f64) -> f64 {
    let mut a = (angle + PI) % (2.0 * PI);
    if a < 0.0 {
        a += 2.0 * PI;
    }
    a - PI
}

/// Orientation test: returns signed area of triangle (a, b, c)
/// Positive = counter-clockwise, Negative = clockwise, Zero = collinear
#[inline]
pub fn orientation(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Check if point p lies on segment [a, b] (assuming near-collinearity)
#[inline]
pub fn on_segment(a: &Point2D, b: &Point2D, p: &Point2D, eps: f64) -> bool {
    let min_x = a.x.min(b.x) - eps;
    let max_x = a.x.max(b.x) + eps;
    let min_y = a.y.min(b.y) - eps;
    let max_y = a.y.max(b.y) + eps;

    p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y && orientation(a, b, p).abs() < eps
}

/// Check if two points are approximately equal
#[inline]
fn approx_eq(a: &Point2D, b: &Point2D, eps: f64) -> bool {
    (a - b).norm() < eps
}

/// Strict segment intersection (excludes endpoint touching)
/// Returns true if segments [p1,p2] and [p3,p4] cross in their interiors
pub fn segments_intersect_strict(
    p1: &Point2D,
    p2: &Point2D,
    p3: &Point2D,
    p4: &Point2D,
    eps: f64,
) -> bool {
    let o1 = orientation(p1, p2, p3);
    let o2 = orientation(p1, p2, p4);
    let o3 = orientation(p3, p4, p1);
    let o4 = orientation(p3, p4, p2);

    // General case: opposite orientations on both sides
    if o1 * o2 < -eps && o3 * o4 < -eps {
        return true;
    }

    // Collinear cases (excluding endpoints)
    if o1.abs() < eps
        && on_segment(p1, p2, p3, eps)
        && !approx_eq(p3, p1, eps)
        && !approx_eq(p3, p2, eps)
    {
        return true;
    }
    if o2.abs() < eps
        && on_segment(p1, p2, p4, eps)
        && !approx_eq(p4, p1, eps)
        && !approx_eq(p4, p2, eps)
    {
        return true;
    }
    if o3.abs() < eps
        && on_segment(p3, p4, p1, eps)
        && !approx_eq(p1, p3, eps)
        && !approx_eq(p1, p4, eps)
    {
        return true;
    }
    if o4.abs() < eps
        && on_segment(p3, p4, p2, eps)
        && !approx_eq(p2, p3, eps)
        && !approx_eq(p2, p4, eps)
    {
        return true;
    }

    false
}

/// Distance from point p to segment [a, b]
pub fn seg_point_distance(p: &Point2D, a: &Point2D, b: &Point2D) -> f64 {
    let ab = b - a;
    let denom = ab.dot(&ab);

    if denom < 1e-12 {
        return (p - a).norm();
    }

    let t = ((p - a).dot(&ab) / denom).clamp(0.0, 1.0);
    let closest = a + t * ab;
    (p - closest).norm()
}

/// Minimum distance between two segments [a, b] and [c, d]
pub fn seg_seg_distance(a: &Point2D, b: &Point2D, c: &Point2D, d: &Point2D) -> f64 {
    if segments_intersect_strict(a, b, c, d, 1e-9) {
        return 0.0;
    }

    [
        seg_point_distance(a, c, d),
        seg_point_distance(b, c, d),
        seg_point_distance(c, a, b),
        seg_point_distance(d, a, b),
    ]
    .into_iter()
    .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn test_dist() {
        let a = Vector2::new(0.0, 0.0);
        let b = Vector2::new(3.0, 4.0);
        assert!((dist(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_wrap_pi() {
        assert!((wrap_pi(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_pi(PI) - PI).abs() < 1e-10);
        assert!((wrap_pi(-PI) - (-PI)).abs() < 1e-10);
        assert!((wrap_pi(2.0 * PI) - 0.0).abs() < 1e-10);
        assert!((wrap_pi(3.0 * PI) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_segments_intersect() {
        let a = Vector2::new(0.0, 0.0);
        let b = Vector2::new(1.0, 1.0);
        let c = Vector2::new(0.0, 1.0);
        let d = Vector2::new(1.0, 0.0);

        assert!(segments_intersect_strict(&a, &b, &c, &d, 1e-9));
    }
}
