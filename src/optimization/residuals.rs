use crate::config::Config;
use crate::geometry::{seg_seg_distance, wrap_pi};
use crate::linkage::{bc_from_params, eval_pose_for_theta, height_for_ratio};
use crate::optimization::packing::unpack_vars;
use nalgebra::Vector2;

/// Compute residual vector for the full optimization problem
/// Returns weighted constraint violations as a vector
pub fn residuals(x: &[f64], cfg: &Config) -> Vec<f64> {
    let vars = unpack_vars(x, cfg);
    let h = Vector2::new(0.0, 0.0);
    let bc = bc_from_params(vars.xbc, vars.ybc, cfg);

    let mut r: Vec<f64> = Vec::with_capacity(100);

    // --- Link ratio constraints ---
    // CW/HK ratio
    if cfg.cw_hk_ratio_min.is_some() || cfg.cw_hk_ratio_max.is_some() {
        let hk_len = vars.lu.max(1e-9);
        let cw_len = vars.lkc + vars.lkw;
        let ratio = cw_len / hk_len;

        if let Some(min) = cfg.cw_hk_ratio_min {
            r.push((min - ratio).max(0.0) * cfg.w_cw_hk_ratio);
        }
        if let Some(max) = cfg.cw_hk_ratio_max {
            r.push((ratio - max).max(0.0) * cfg.w_cw_hk_ratio);
        }
    }

    // LC/HK ratio
    if cfg.lc_hk_ratio_min.is_some() || cfg.lc_hk_ratio_max.is_some() {
        let hk_len = vars.lu.max(1e-9);
        let ratio = vars.lc / hk_len;

        if let Some(min) = cfg.lc_hk_ratio_min {
            r.push((min - ratio).max(0.0) * cfg.w_lc_hk_ratio);
        }
        if let Some(max) = cfg.lc_hk_ratio_max {
            r.push((ratio - max).max(0.0) * cfg.w_lc_hk_ratio);
        }
    }

    // XBC minimum constraint
    if let Some(xbc_min) = cfg.xbc_min {
        r.push((xbc_min - bc.x).max(0.0) * cfg.w_xbc_min);
    }

    // BC radius constraint
    if let Some(bc_radius_max) = cfg.bc_radius_max {
        let bc_r = (bc - h).norm();
        r.push((bc_r - bc_radius_max).max(0.0) * cfg.w_bc_radius);
    }

    // --- Per-pose constraints ---
    let mut wheels = Vec::new();
    let mut wheel_ys = Vec::new();
    let mut hip_thetas = Vec::new();
    let mut preferred_c: Option<Vector2<f64>> = None;
    let mut infeasible = false;

    // Sort poses by ratio
    let mut ratios_sorted: Vec<_> = cfg.ratios.iter().copied().enumerate().collect();
    ratios_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (idx, ratio) in &ratios_sorted {
        let theta = *vars.poses.get(idx).unwrap();
        let target_y = height_for_ratio(cfg, *ratio);
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
                r.push(1e6);
                continue;
            }
        };

        preferred_c = Some(c.clone());
        wheels.push(w);
        wheel_ys.push(w.y);
        hip_thetas.push(theta);

        // Pose target
        r.push(f * cfg.w_pose);

        // Knee above wheel
        let knee_violation = (k.y - w.y + cfg.knee_above_margin).max(0.0);
        r.push(knee_violation * cfg.w_knee_above_wheel);

        // Below ground penalties
        r.push((-k.y).max(0.0) * cfg.w_below);
        r.push((-c.y).max(0.0) * cfg.w_below);
        r.push((-w.y).max(0.0) * cfg.w_below);

        // Crossing prevention
        let d_cross = seg_seg_distance(&h, &k, &bc, &c);
        let cross_violation = (cfg.cross_min - d_cross).max(0.0);
        let cross_scale = cross_violation / cfg.cross_min.max(1e-9);
        r.push(cross_scale * cfg.w_no_cross);

        // Knee angle constraint (angle H-K-W)
        let vec_kh = h - k;
        let vec_kw = w - k;
        let len_kh = vec_kh.norm();
        let len_kw = vec_kw.norm();
        let alpha = if len_kh > 1e-9 && len_kw > 1e-9 {
            let cos_angle = (vec_kh.dot(&vec_kw) / (len_kh * len_kw)).clamp(-1.0, 1.0);
            Some(cos_angle.acos())
        } else {
            None
        };
        if let Some(angle_hkw) = alpha {
            let angle_violation = (angle_hkw - cfg.max_angle_hkw).max(0.0);
            r.push(angle_violation * cfg.w_angle_hkw);
        }

        // Pose=1 theta target (alpha input)
        if (ratio - 1.0).abs() < 1e-9 && cfg.w_theta_pose1 > 0.0 {
            if let Some(alpha_current) = alpha {
                let alpha_target = cfg.alpha_pose1_target;
                let alpha_error = alpha_current - alpha_target;

                let eps = 1e-4;
                let alpha_prev = eval_pose_for_theta(
                    theta - eps,
                    &bc,
                    vars.lu,
                    vars.lkc,
                    vars.lc,
                    vars.lkw,
                    target_y,
                    Some(&c),
                )
                .map(|(k_p, _c_p, w_p, _)| {
                    let v_kh = h - k_p;
                    let v_kw = w_p - k_p;
                    let len_kh = v_kh.norm();
                    let len_kw = v_kw.norm();
                    if len_kh > 1e-9 && len_kw > 1e-9 {
                        let cos_angle =
                            (v_kh.dot(&v_kw) / (len_kh * len_kw)).clamp(-1.0, 1.0);
                        Some(cos_angle.acos())
                    } else {
                        None
                    }
                })
                .flatten();

                let alpha_next = eval_pose_for_theta(
                    theta + eps,
                    &bc,
                    vars.lu,
                    vars.lkc,
                    vars.lc,
                    vars.lkw,
                    target_y,
                    Some(&c),
                )
                .map(|(k_n, _c_n, w_n, _)| {
                    let v_kh = h - k_n;
                    let v_kw = w_n - k_n;
                    let len_kh = v_kh.norm();
                    let len_kw = v_kw.norm();
                    if len_kh > 1e-9 && len_kw > 1e-9 {
                        let cos_angle =
                            (v_kh.dot(&v_kw) / (len_kh * len_kw)).clamp(-1.0, 1.0);
                        Some(cos_angle.acos())
                    } else {
                        None
                    }
                })
                .flatten();

                let theta_error = if let (Some(a0), Some(a1)) = (alpha_prev, alpha_next) {
                    let dalpha = a1 - a0;
                    if dalpha.abs() > 1e-6 {
                        alpha_error / (dalpha / (2.0 * eps))
                    } else {
                        alpha_error
                    }
                } else {
                    alpha_error
                };

                r.push(theta_error * cfg.w_theta_pose1);
            }
        }
    }

    if infeasible {
        return r;
    }

    // Wheel X alignment
    let n_wheels = wheels.len().max(1);
    let mean_wx: f64 = wheels.iter().map(|w| w.x).sum::<f64>() / n_wheels as f64;
    for w in &wheels {
        r.push((w.x - mean_wx) * cfg.w_wheel_x);
    }
    r.push(mean_wx * cfg.w_wheel_x_mean);

    // --- Jumping transmission shaping ---
    if ratios_sorted.len() >= 2 {
        let dtheta_total = wrap_pi(hip_thetas.last().unwrap() - hip_thetas.first().unwrap());
        let expected_sign = if dtheta_total >= 0.0 { 1.0 } else { -1.0 };
        let theta_span = dtheta_total.abs();

        r.push((cfg.theta_span_min - theta_span).max(0.0) * cfg.w_theta_span);

        for i in 0..(ratios_sorted.len() - 1) {
            let (_, r0) = ratios_sorted[i];
            let (_, r1) = ratios_sorted[i + 1];
            let dy = wheel_ys[i + 1] - wheel_ys[i];
            let dtheta = wrap_pi(hip_thetas[i + 1] - hip_thetas[i]);

            // Near-singular penalty
            r.push((1e-4 - dtheta.abs()).max(0.0) * 1e6);

            let dtheta_safe = if dtheta.abs() < 1e-4 {
                if dtheta >= 0.0 { 1e-4 } else { -1e-4 }
            } else {
                dtheta
            };

            // Monotonic hip motion
            r.push(
                (cfg.theta_step_min - expected_sign * dtheta_safe).max(0.0) * cfg.w_theta_monotonic,
            );

            // Jacobian profile
            let j = dy / dtheta_safe;
            let t = 0.5 * (r0 + r1);
            let j_target = cfg.jac_start + t * (cfg.jac_end - cfg.jac_start);
            let j_abs = j.abs();

            r.push((j_abs - j_target) * cfg.w_jac_profile);
            r.push((cfg.jac_min - j_abs).max(0.0) * cfg.w_jac_bounds);
            r.push((j_abs - cfg.jac_max).max(0.0) * cfg.w_jac_bounds);
        }
    }

    // Regularization
    for val in [vars.lu, vars.lkw, vars.lkc, vars.lc] {
        if val <= 0.0 {
            r.push((val.abs() + 1.0) * 1e6);
        } else {
            r.push(val * cfg.w_reg);
        }
    }

    // Pin joint bias
    r.push(bc.x * cfg.w_bc_x);
    r.push(bc.y * cfg.w_bc_y);

    r
}

/// Compute cost (sum of squared residuals)
pub fn cost(x: &[f64], cfg: &Config) -> f64 {
    let r = residuals(x, cfg);
    r.iter().map(|v| v * v).sum()
}
