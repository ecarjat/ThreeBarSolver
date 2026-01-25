use crate::config::Config;
use crate::geometry::{seg_seg_distance, wrap_pi};
use crate::linkage::{bc_from_params, eval_pose_for_theta, height_for_ratio};
use crate::optimization::packing::unpack_vars;
use crate::types::Point2D;
use nalgebra::Vector2;

// Infeasibility penalty for constraint violations
const INFEASIBLE_PENALTY: f64 = 1e6;

// Geometric tolerances
const GEOMETRIC_EPS: f64 = 1e-9;

// Numerical differentiation
const NUMERICAL_DIFF_EPS: f64 = 1e-4;
const DALPHA_THRESHOLD: f64 = 1e-6;

// Near-singular theta step threshold
const DTHETA_MIN_ABS: f64 = 1e-4;

/// Compute knee angle (H-K-W) from joint positions.
/// Returns None if either segment is degenerate.
#[inline]
fn compute_knee_angle(h: &Point2D, k: &Point2D, w: &Point2D) -> Option<f64> {
    let vec_kh = h - k;
    let vec_kw = w - k;
    let len_kh = vec_kh.norm();
    let len_kw = vec_kw.norm();
    if len_kh > GEOMETRIC_EPS && len_kw > GEOMETRIC_EPS {
        let cos_angle = (vec_kh.dot(&vec_kw) / (len_kh * len_kw)).clamp(-1.0, 1.0);
        Some(cos_angle.acos())
    } else {
        None
    }
}

#[inline]
fn for_each_residual<F: FnMut(f64) -> bool>(x: &[f64], cfg: &Config, mut emit: F) {
    let vars = unpack_vars(x, cfg);
    let h = Vector2::new(0.0, 0.0);
    let bc = bc_from_params(vars.xbc, vars.ybc, cfg);

    // --- Link ratio constraints ---
    // CW/HK ratio
    if cfg.cw_hk_ratio_min.is_some() || cfg.cw_hk_ratio_max.is_some() {
        let hk_len = vars.lu.max(GEOMETRIC_EPS);
        let cw_len = vars.lkc + vars.lkw;
        let ratio = cw_len / hk_len;

        if let Some(min) = cfg.cw_hk_ratio_min {
            if !emit((min - ratio).max(0.0) * cfg.w_cw_hk_ratio) {
                return;
            }
        }
        if let Some(max) = cfg.cw_hk_ratio_max {
            if !emit((ratio - max).max(0.0) * cfg.w_cw_hk_ratio) {
                return;
            }
        }
    }

    // LC/HK ratio
    if cfg.lc_hk_ratio_min.is_some() || cfg.lc_hk_ratio_max.is_some() {
        let hk_len = vars.lu.max(GEOMETRIC_EPS);
        let ratio = vars.lc / hk_len;

        if let Some(min) = cfg.lc_hk_ratio_min {
            if !emit((min - ratio).max(0.0) * cfg.w_lc_hk_ratio) {
                return;
            }
        }
        if let Some(max) = cfg.lc_hk_ratio_max {
            if !emit((ratio - max).max(0.0) * cfg.w_lc_hk_ratio) {
                return;
            }
        }
    }

    // XBC minimum constraint
    if let Some(xbc_min) = cfg.xbc_min {
        if !emit((xbc_min - bc.x).max(0.0) * cfg.w_xbc_min) {
            return;
        }
    }

    // BC radius constraint
    if let Some(bc_radius_max) = cfg.bc_radius_max {
        let bc_r = (bc - h).norm();
        if !emit((bc_r - bc_radius_max).max(0.0) * cfg.w_bc_radius) {
            return;
        }
    }

    // --- Per-pose constraints ---
    let mut wheels = Vec::with_capacity(cfg.ratios.len());
    let mut wheel_ys = Vec::with_capacity(cfg.ratios.len());
    let mut hip_thetas = Vec::with_capacity(cfg.ratios.len());
    let mut preferred_c: Option<Vector2<f64>> = None;
    let mut infeasible = false;

    for &(idx, ratio) in &cfg.ratios_sorted {
        let theta = match vars.poses.get(&idx) {
            Some(&t) => t,
            None => {
                // Missing pose index - treat as infeasible
                infeasible = true;
                if !emit(INFEASIBLE_PENALTY) {
                    return;
                }
                continue;
            }
        };
        let target_y = height_for_ratio(cfg, ratio);
        let (k, c, w, f) = match eval_pose_for_theta(
            theta,
            &bc,
            vars.lu,
            vars.lkc,
            vars.lc,
            vars.lkw,
            target_y,
            preferred_c.as_ref(),
        ) {
            Some(result) => result,
            None => {
                infeasible = true;
                if !emit(INFEASIBLE_PENALTY) {
                    return;
                }
                continue;
            }
        };

        preferred_c = Some(c);
        wheels.push(w);
        wheel_ys.push(w.y);
        hip_thetas.push(theta);

        // Pose target
        if !emit(f * cfg.w_pose) {
            return;
        }

        // Knee above wheel
        let knee_violation = (k.y - w.y + cfg.knee_above_margin).max(0.0);
        if !emit(knee_violation * cfg.w_knee_above_wheel) {
            return;
        }

        // Below ground penalties
        if !emit((-k.y).max(0.0) * cfg.w_below) {
            return;
        }
        if !emit((-c.y).max(0.0) * cfg.w_below) {
            return;
        }
        if !emit((-w.y).max(0.0) * cfg.w_below) {
            return;
        }

        // Crossing prevention
        let d_cross = seg_seg_distance(&h, &k, &bc, &c);
        let cross_violation = (cfg.cross_min - d_cross).max(0.0);
        let cross_scale = cross_violation / cfg.cross_min.max(1e-9);
        if !emit(cross_scale * cfg.w_no_cross) {
            return;
        }

        // Knee angle constraint (angle H-K-W)
        let alpha = compute_knee_angle(&h, &k, &w);
        if let Some(angle_hkw) = alpha {
            let angle_violation = (angle_hkw - cfg.max_angle_hkw).max(0.0);
            if !emit(angle_violation * cfg.w_angle_hkw) {
                return;
            }
        }

        // Pose=1 theta target (alpha input)
        // Uses forward difference (one extra eval) instead of central difference (two extra evals)
        if (ratio - 1.0).abs() < GEOMETRIC_EPS && cfg.w_theta_pose1 > 0.0 {
            if let Some(alpha_current) = alpha {
                let alpha_target = cfg.alpha_pose1_target;
                let alpha_error = alpha_current - alpha_target;

                // Forward difference: compute alpha at theta + eps, reuse alpha_current
                let alpha_next = eval_pose_for_theta(
                    theta + NUMERICAL_DIFF_EPS,
                    &bc,
                    vars.lu,
                    vars.lkc,
                    vars.lc,
                    vars.lkw,
                    target_y,
                    Some(&c),
                )
                .and_then(|(k_n, _c_n, w_n, _)| compute_knee_angle(&h, &k_n, &w_n));

                let theta_error = if let Some(a1) = alpha_next {
                    let dalpha = a1 - alpha_current;
                    if dalpha.abs() > DALPHA_THRESHOLD {
                        // Forward difference: dalpha/dtheta ≈ (alpha_next - alpha_current) / eps
                        alpha_error / (dalpha / NUMERICAL_DIFF_EPS)
                    } else {
                        alpha_error
                    }
                } else {
                    alpha_error
                };

                if !emit(theta_error * cfg.w_theta_pose1) {
                    return;
                }
            }
        }
    }

    if infeasible {
        return;
    }

    // Wheel X alignment
    let n_wheels = wheels.len().max(1);
    let mean_wx: f64 = wheels.iter().map(|w| w.x).sum::<f64>() / n_wheels as f64;
    for w in &wheels {
        if !emit((w.x - mean_wx) * cfg.w_wheel_x) {
            return;
        }
    }
    if !emit(mean_wx * cfg.w_wheel_x_mean) {
        return;
    }

    // --- Jumping transmission shaping ---
    if cfg.ratios_sorted.len() >= 2 {
        let dtheta_total = wrap_pi(hip_thetas.last().unwrap() - hip_thetas.first().unwrap());
        let expected_sign = if dtheta_total >= 0.0 { 1.0 } else { -1.0 };
        let theta_span = dtheta_total.abs();

        if !emit((cfg.theta_span_min - theta_span).max(0.0) * cfg.w_theta_span) {
            return;
        }

        for i in 0..(cfg.ratios_sorted.len() - 1) {
            let (_, r0) = cfg.ratios_sorted[i];
            let (_, r1) = cfg.ratios_sorted[i + 1];
            let dy = wheel_ys[i + 1] - wheel_ys[i];
            let dtheta = wrap_pi(hip_thetas[i + 1] - hip_thetas[i]);

            // Near-singular penalty
            if !emit((DTHETA_MIN_ABS - dtheta.abs()).max(0.0) * INFEASIBLE_PENALTY) {
                return;
            }

            let dtheta_safe = if dtheta.abs() < DTHETA_MIN_ABS {
                if dtheta >= 0.0 { DTHETA_MIN_ABS } else { -DTHETA_MIN_ABS }
            } else {
                dtheta
            };

            // Monotonic hip motion
            if !emit(
                (cfg.theta_step_min - expected_sign * dtheta_safe).max(0.0) * cfg.w_theta_monotonic,
            ) {
                return;
            }

            // Jacobian profile
            let j = dy / dtheta_safe;
            let t = 0.5 * (r0 + r1);
            let j_target = cfg.jac_start + t * (cfg.jac_end - cfg.jac_start);
            let j_abs = j.abs();

            if !emit((j_abs - j_target) * cfg.w_jac_profile) {
                return;
            }
            if !emit((cfg.jac_min - j_abs).max(0.0) * cfg.w_jac_bounds) {
                return;
            }
            if !emit((j_abs - cfg.jac_max).max(0.0) * cfg.w_jac_bounds) {
                return;
            }
        }
    }

    // Regularization
    for val in [vars.lu, vars.lkw, vars.lkc, vars.lc] {
        if val <= 0.0 {
            if !emit((val.abs() + 1.0) * INFEASIBLE_PENALTY) {
                return;
            }
        } else if !emit(val * cfg.w_reg) {
            return;
        }
    }

    // Pin joint bias
    if !emit(bc.x * cfg.w_bc_x) {
        return;
    }
    let _ = emit(bc.y * cfg.w_bc_y);
}

/// Compute residual vector for the full optimization problem
/// Returns weighted constraint violations as a vector
#[allow(dead_code)]
pub fn residuals(x: &[f64], cfg: &Config) -> Vec<f64> {
    // Calculate exact capacity to avoid reallocations:
    // - Fixed constraints: up to 13 (link ratios, pin constraints, regularization, bias, mean_wx)
    // - Per-pose: up to 8 each (pose target, knee, 3 below-ground, crossing, angle, alpha)
    // - Wheel X alignment: n_poses
    // - Jumping (n>=2): 1 + 5*(n-1) = 5n - 4
    // Total max: 13 + 8n + n + 5n - 4 = 9 + 14n
    let n_poses = cfg.ratios.len();
    let capacity = 9 + 14 * n_poses;
    let mut r: Vec<f64> = Vec::with_capacity(capacity);
    for_each_residual(x, cfg, |val| {
        r.push(val);
        true
    });
    r
}

/// Compute cost (sum of squared residuals)
pub fn cost(x: &[f64], cfg: &Config) -> f64 {
    let mut sum = 0.0;
    for_each_residual(x, cfg, |val| {
        sum += val * val;
        true
    });
    sum
}

/// Compute cost with an upper bound; stops early if cost exceeds max_cost.
pub fn cost_bounded(x: &[f64], cfg: &Config, max_cost: f64) -> f64 {
    let mut sum = 0.0;
    for_each_residual(x, cfg, |val| {
        sum += val * val;
        sum <= max_cost
    });
    sum
}
