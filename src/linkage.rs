use crate::config::Config;
use crate::geometry::segments_intersect_strict;
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
