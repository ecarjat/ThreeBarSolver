use crate::config::Config;
use crate::geometry::{dist, segments_intersect_strict};
use crate::types::Point2D;
use nalgebra::Vector2;

/// Compute wheel position from knee and inner joint
/// The lower leg bar has three collinear points: W --- K --- C
/// K is in the middle, W and C are on opposite ends
/// W = K - normalize(C - K) * Lkw
pub fn compute_wheel(knee: &Point2D, inner: &Point2D, lk_wheel: f64) -> Point2D {
    let v = inner - knee; // direction from K toward C
    let n = v.norm();

    if n < 1e-9 {
        // Degenerate case: point straight down
        return knee + Vector2::new(0.0, lk_wheel);
    }

    let u = v / n;
    knee - u * lk_wheel // W is opposite direction from C
}

/// Map raw (xbc, ybc) parameters to Bc point with optional radius constraint
/// Uses smooth tanh squashing when bc_radius_max is set
pub fn bc_from_params(xbc: f64, ybc: f64, cfg: &Config) -> Point2D {
    match cfg.bc_radius_max {
        None => Vector2::new(xbc, ybc),
        Some(r_max) if r_max <= 0.0 => Vector2::new(xbc, ybc),
        Some(r_max) => {
            let norm = (xbc * xbc + ybc * ybc).sqrt();
            if norm < 1e-12 {
                Vector2::new(0.0, 0.0)
            } else {
                // Smooth radial squash: r = r_max * tanh(norm / r_max)
                let scale = (r_max * (norm / r_max).tanh()) / norm;
                Vector2::new(xbc * scale, ybc * scale)
            }
        }
    }
}

/// Compute target wheel height for a given ratio
#[inline]
pub fn height_for_ratio(cfg: &Config, ratio: f64) -> f64 {
    cfg.h_crouch + ratio * (cfg.h_ext - cfg.h_crouch)
}

fn circle_intersections(
    c0: &Point2D,
    r0: f64,
    c1: &Point2D,
    r1: f64,
) -> Option<(Point2D, Point2D)> {
    let d = dist(c0, c1);
    if d <= 1e-12 {
        return None;
    }

    if d > r0 + r1 + 1e-12 || d < (r0 - r1).abs() - 1e-12 {
        return None;
    }

    let a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d);
    let mut h_sq = r0 * r0 - a * a;
    if h_sq < 0.0 {
        if h_sq > -1e-12 {
            h_sq = 0.0;
        } else {
            return None;
        }
    }
    let h = h_sq.sqrt();

    let p2 = c0 + a * (c1 - c0) / d;
    let rx = -(c1.y - c0.y) * (h / d);
    let ry = (c1.x - c0.x) * (h / d);

    let p3 = Vector2::new(p2.x + rx, p2.y + ry);
    let p4 = Vector2::new(p2.x - rx, p2.y - ry);

    Some((p3, p4))
}

pub fn eval_pose_for_theta(
    theta: f64,
    bc: &Point2D,
    lu: f64,
    lkc: f64,
    lc: f64,
    lkw: f64,
    target_y: f64,
    preferred_c: Option<&Point2D>,
) -> Option<(Point2D, Point2D, Point2D, f64)> {
    let k = Vector2::new(lu * theta.cos(), lu * theta.sin());
    let (c1, c2) = circle_intersections(&k, lkc, bc, lc)?;

    let w1 = compute_wheel(&k, &c1, lkw);
    let f1 = w1.y - target_y;
    let w2 = compute_wheel(&k, &c2, lkw);
    let f2 = w2.y - target_y;

    let use_first = if let Some(pref) = preferred_c {
        (c1 - pref).norm() <= (c2 - pref).norm()
    } else {
        f1.abs() <= f2.abs()
    };

    if use_first {
        Some((k, c1, w1, f1))
    } else {
        Some((k, c2, w2, f2))
    }
}

/// Check if any pose has H-K / Bc-C crossing
pub fn has_crossing(poses: &[(f64, Point2D, Point2D)], bc: &Point2D) -> bool {
    let h = Vector2::new(0.0, 0.0);

    for (_, k, c) in poses {
        if segments_intersect_strict(&h, k, bc, c, 1e-9) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_wheel() {
        let knee = Vector2::new(0.0, 0.3);
        let inner = Vector2::new(0.0, 0.5); // C is below K (+Y is down)
        let lkw = 0.2;
        let wheel = compute_wheel(&knee, &inner, lkw);

        // Bar order: W --- K --- C
        // W is opposite direction from C relative to K
        // Since C is below K (C.y > K.y), W should be above K (W.y < K.y)
        assert!(wheel.y < knee.y);
        // Check distance K to W equals Lkw
        assert!(((wheel - knee).norm() - lkw).abs() < 1e-10);
    }

    #[test]
    fn test_height_for_ratio() {
        let cfg = Config::default();
        assert!((height_for_ratio(&cfg, 0.0) - cfg.h_crouch).abs() < 1e-10);
        assert!((height_for_ratio(&cfg, 1.0) - cfg.h_ext).abs() < 1e-10);
        assert!(
            (height_for_ratio(&cfg, 0.5) - (cfg.h_crouch + cfg.h_ext) / 2.0).abs() < 1e-10
        );
    }
}
