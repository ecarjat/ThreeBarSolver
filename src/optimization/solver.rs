use crate::config::Config;
use crate::geometry::{segments_intersect_strict, wrap_pi};
use crate::linkage::{bc_from_params, compute_wheel, eval_pose_for_theta, has_crossing, height_for_ratio};
use crate::optimization::bounds::build_bounds;
use crate::optimization::packing::{pack_vars, unpack_vars};
use crate::optimization::problem::ThreeBarProblem;
use crate::optimization::residuals::cost;
use crate::types::*;
use argmin::core::{Executor, IterState};
use argmin::solver::neldermead::NelderMead;
use nalgebra::Vector2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

const GLOBAL_INFEASIBLE_COST: f64 = 1.0e12;
const GLOBAL_POSE_TOL: f64 = 1.0e-4;

/// Generate initial seed for multi-start optimization
pub fn generate_initial_seed(cfg: &Config, seed_idx: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64((seed_idx * 1337 + 11) as u64);

    let scale = cfg.h_crouch.abs().max(cfg.h_ext.abs()).max(0.2);
    let mut poses = BTreeMap::new();

    let min_len = 0.05 * scale;
    let max_len = 3.0 * scale;
    let lu = rng.gen_range(min_len..max_len);
    let lkw = rng.gen_range(min_len..max_len);

    let mut lkc = rng.gen_range(min_len..max_len);
    lkc = lkc.max(cfg.kc_min);
    if let Some(kc_max) = cfg.kc_max {
        lkc = lkc.min(kc_max);
    }

    let lc = rng.gen_range(min_len..max_len);
    let xbc = (rng.gen::<f64>() - 0.5) * 0.6 * scale;
    let ybc = (rng.gen::<f64>() - 0.3) * 0.6 * scale;
    let pin_joint = PinJointLocation { x: xbc, y: ybc };
    let lengths = Lengths {
        upper_leg_hk: lu,
        lower_leg_kw: lkw,
        link_bc_c: lc,
    };

    for (i, ratio) in cfg.ratios.iter().enumerate() {
        let pose = solve_pose_ratio(cfg, &lengths, &pin_joint, lkc, *ratio, None);
        let theta = pose.points.k.y.atan2(pose.points.k.x);
        poses.insert(i, theta);
    }

    pack_vars(&poses, lu, lkw, lkc, lc, xbc, ybc, cfg)
}

fn is_feasible_global(x: &[f64], cfg: &Config, pose_tol: f64) -> bool {
    let vars = unpack_vars(x, cfg);
    let bc = bc_from_params(vars.xbc, vars.ybc, cfg);

    let mut preferred_c: Option<Vector2<f64>> = None;
    let mut poses_for_crossing = Vec::new();

    let mut ratios_sorted: Vec<_> = cfg.ratios.iter().copied().enumerate().collect();
    ratios_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (idx, ratio) in ratios_sorted {
        let theta = *vars.poses.get(&idx).unwrap_or(&0.0);
        let target_y = height_for_ratio(cfg, ratio);
        let (k, c, _w, f) = match eval_pose_for_theta(
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
            None => return false,
        };

        if !f.is_finite() || f.abs() > pose_tol {
            return false;
        }

        poses_for_crossing.push((ratio, k, c));
        preferred_c = Some(c);
    }

    !has_crossing(&poses_for_crossing, &bc)
}

fn global_cost(x: &[f64], cfg: &Config, pose_tol: f64) -> f64 {
    if !is_feasible_global(x, cfg, pose_tol) {
        return GLOBAL_INFEASIBLE_COST;
    }
    cost(x, cfg)
}

/// Local optimization using Nelder-Mead
pub fn solve_local(cfg: &Config, x0: Vec<f64>) -> (Vec<f64>, f64, bool) {
    let bounds = build_bounds(cfg);

    // Clamp to bounds
    let mut x0_clamped = x0;
    bounds.clamp(&mut x0_clamped);

    let problem = ThreeBarProblem::new(Arc::new(cfg.clone()));

    // Build simplex for Nelder-Mead
    let n = x0_clamped.len();
    let simplex: Vec<Vec<f64>> = (0..=n)
        .map(|i| {
            let mut p = x0_clamped.clone();
            if i > 0 {
                let delta = 0.05 * (bounds.upper[i - 1] - bounds.lower[i - 1]);
                p[i - 1] = (p[i - 1] + delta).min(bounds.upper[i - 1] - 1e-9);
            }
            p
        })
        .collect();

    let solver = match NelderMead::new(simplex).with_sd_tolerance(1e-8) {
        Ok(s) => s,
        Err(_) => {
            let cost = crate::optimization::residuals::cost(&x0_clamped, cfg);
            return (x0_clamped, cost, false);
        }
    };

    let result = Executor::new(problem, solver)
        .configure(|state: IterState<Vec<f64>, (), (), (), (), f64>| state.max_iters(20000))
        .run();

    match result {
        Ok(res) => {
            let state: &IterState<Vec<f64>, (), (), (), (), f64> = res.state();
            let best = state.best_param.clone().unwrap_or(x0_clamped);
            let cost = state.best_cost;
            (best, cost, true)
        }
        Err(_) => {
            // Return initial guess with high cost on failure
            let cost = crate::optimization::residuals::cost(&x0_clamped, cfg);
            (x0_clamped, cost, false)
        }
    }
}

/// Multi-start optimization with Rayon parallelization
pub fn solve_multistart(cfg: &Config) -> (Vec<f64>, f64) {
    let results: Vec<_> = (0..cfg.n_starts)
        .into_par_iter()
        .map(|i| {
            let x0 = generate_initial_seed(cfg, i);
            let (x_opt, cost, _success) = solve_local(cfg, x0);

            // Check for crossing
            let vars = unpack_vars(&x_opt, cfg);
            let bc = bc_from_params(vars.xbc, vars.ybc, cfg);
            let mut preferred_c: Option<Vector2<f64>> = None;
            let mut poses_for_crossing = Vec::new();
            let mut infeasible = false;
            let mut ratios_sorted: Vec<_> = cfg.ratios.iter().copied().enumerate().collect();
            ratios_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (idx, ratio) in ratios_sorted {
                let theta = *vars.poses.get(&idx).unwrap_or(&0.0);
                let target_y = height_for_ratio(cfg, ratio);
                if let Some((k, c, _w, _f)) = eval_pose_for_theta(
                    theta,
                    &bc,
                    vars.lu,
                    vars.lkc,
                    vars.lc,
                    vars.lkw,
                    target_y,
                    preferred_c.as_ref(),
                ) {
                    poses_for_crossing.push((ratio, k, c));
                    preferred_c = Some(c);
                } else {
                    infeasible = true;
                    break;
                }
            }

            let crossing = infeasible || has_crossing(&poses_for_crossing, &bc);

            (x_opt, cost, crossing)
        })
        .collect();

    // Handle empty results (shouldn't happen if n_starts >= 1, but be safe)
    if results.is_empty() {
        let fallback = generate_initial_seed(cfg, 0);
        let fallback_cost = cost(&fallback, cfg);
        return (fallback, fallback_cost);
    }

    // Compare costs safely, treating NaN as worse than any finite value
    let compare_costs = |a: &f64, b: &f64| -> std::cmp::Ordering {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaN is "worse"
            (false, true) => std::cmp::Ordering::Less,    // finite is "better"
            (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        }
    };

    // Find best non-crossing solution
    let best_non_crossing = results
        .iter()
        .filter(|(_, _, crossing)| !crossing)
        .min_by(|a, b| compare_costs(&a.1, &b.1));

    match best_non_crossing {
        Some((x, cost, _)) => (x.clone(), *cost),
        None => {
            // Fall back to best overall (results is guaranteed non-empty here)
            let best = results
                .iter()
                .min_by(|a, b| compare_costs(&a.1, &b.1))
                .expect("results should be non-empty");
            (best.0.clone(), best.1)
        }
    }
}

/// Global optimization using Differential Evolution (DE)
fn solve_global_de(cfg: &Config) -> (Vec<f64>, f64) {
    let bounds = build_bounds(cfg);
    let n = bounds.lower.len();
    let pop_size = cfg.global_popsize.max(4);
    let seed = cfg.global_seed.unwrap_or(1234);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let pose_tol = GLOBAL_POSE_TOL;

    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    let mut seed_idx = 0;
    let mut seed_attempts = 0;
    let max_seed_attempts = pop_size.saturating_mul(20).max(20);
    while pop.len() < pop_size && seed_attempts < max_seed_attempts {
        let mut x = generate_initial_seed(cfg, seed_idx);
        seed_idx += 1;
        seed_attempts += 1;
        bounds.clamp(&mut x);
        if is_feasible_global(&x, cfg, pose_tol) {
            pop.push(x);
        }
    }

    while pop.len() < pop_size {
        let mut tries = 0;
        let mut x = vec![0.0; n];
        loop {
            for i in 0..n {
                let lo = bounds.lower[i];
                let hi = bounds.upper[i];
                x[i] = lo + rng.gen::<f64>() * (hi - lo);
            }
            tries += 1;
            if is_feasible_global(&x, cfg, pose_tol) || tries >= 8 {
                break;
            }
        }
        pop.push(x);
    }

    let mut costs: Vec<f64> = pop.iter().map(|x| global_cost(x, cfg, pose_tol)).collect();
    let mut best_idx = costs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let f = 0.7;
    let cr = 0.9;

    for _ in 0..cfg.global_maxiter.max(1) {
        let prev_best = costs[best_idx];

        for i in 0..pop_size {
            let (a, b, c) = pick_three_distinct(i, pop_size, &mut rng);
            let mut mutant = vec![0.0; n];
            for j in 0..n {
                mutant[j] = pop[a][j] + f * (pop[b][j] - pop[c][j]);
            }

            let mut trial = pop[i].clone();
            let j_rand = rng.gen_range(0..n);
            for j in 0..n {
                if rng.gen::<f64>() < cr || j == j_rand {
                    trial[j] = mutant[j];
                }
            }

            bounds.clamp(&mut trial);
            let trial_cost = global_cost(&trial, cfg, pose_tol);
            if trial_cost < costs[i] {
                pop[i] = trial;
                costs[i] = trial_cost;
                if trial_cost < costs[best_idx] {
                    best_idx = i;
                }
            }
        }

        if (prev_best - costs[best_idx]).abs() < cfg.global_tol {
            break;
        }
    }

    (pop[best_idx].clone(), costs[best_idx])
}

fn pick_three_distinct(
    exclude: usize,
    n: usize,
    rng: &mut impl Rng,
) -> (usize, usize, usize) {
    let mut a = rng.gen_range(0..n);
    while a == exclude {
        a = rng.gen_range(0..n);
    }
    let mut b = rng.gen_range(0..n);
    while b == exclude || b == a {
        b = rng.gen_range(0..n);
    }
    let mut c = rng.gen_range(0..n);
    while c == exclude || c == a || c == b {
        c = rng.gen_range(0..n);
    }
    (a, b, c)
}

/// Main solve function
pub fn solve(cfg: &Config) -> Solution {
    let (x_stage1, _) = if cfg.use_global_opt {
        let (x, c) = solve_global_de(cfg);
        if c >= GLOBAL_INFEASIBLE_COST || !is_feasible_global(&x, cfg, GLOBAL_POSE_TOL) {
            solve_multistart(cfg)
        } else {
            (x, c)
        }
    } else {
        solve_multistart(cfg)
    };
    let (x_opt, cost, success) = solve_local(cfg, x_stage1);

    // Build solution from optimized variables
    build_solution(&x_opt, cost, success, cfg)
}

/// Build Solution struct from optimization result
fn build_solution(x: &[f64], cost: f64, success: bool, cfg: &Config) -> Solution {
    let vars = unpack_vars(x, cfg);
    let bc = bc_from_params(vars.xbc, vars.ybc, cfg);
    let h = Vector2::new(0.0, 0.0);

    // Build poses
    let mut poses = Vec::new();
    let mut wheel_xs = Vec::new();
    let mut wheel_ys = Vec::new();
    let mut hip_thetas = Vec::new();
    let mut preferred_c: Option<Vector2<f64>> = None;
    let mut infeasible = false;

    let mut ratios_sorted: Vec<_> = cfg.ratios.iter().copied().enumerate().collect();
    ratios_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (idx, ratio) in ratios_sorted {
        let theta = *vars.poses.get(&idx).unwrap_or(&0.0);
        let target_y = height_for_ratio(cfg, ratio);
        if let Some((k, c, w, _f)) = eval_pose_for_theta(
            theta,
            &bc,
            vars.lu,
            vars.lkc,
            vars.lc,
            vars.lkw,
            target_y,
            preferred_c.as_ref(),
        ) {
            wheel_xs.push(w.x);
            wheel_ys.push(w.y);
            hip_thetas.push(theta);

            let pose_crossing = segments_intersect_strict(&h, &k, &bc, &c, 1e-9);

            poses.push(Pose {
                ratio,
                target_wheel_y: target_y,
                points: PosePoints { h, k, c, w, bc },
                crossing: pose_crossing,
            });
            preferred_c = Some(c);
        } else {
            infeasible = true;
            let k = Vector2::new(vars.lu * theta.cos(), vars.lu * theta.sin());
            let c = bc;
            let w = compute_wheel(&k, &c, vars.lkw);
            wheel_xs.push(w.x);
            wheel_ys.push(w.y);
            hip_thetas.push(theta);

            poses.push(Pose {
                ratio,
                target_wheel_y: target_y,
                points: PosePoints { h, k, c, w, bc },
                crossing: false,
            });
        }
    }

    let poses_for_crossing: Vec<_> = poses
        .iter()
        .map(|pose| (pose.ratio, pose.points.k, pose.points.c))
        .collect();
    let crossing = has_crossing(&poses_for_crossing, &bc);

    // Quality metrics
    let n_wheels = wheel_xs.len().max(1);
    let mean_wx = wheel_xs.iter().sum::<f64>() / n_wheels as f64;
    let rms_wx =
        (wheel_xs.iter().map(|x| (x - mean_wx).powi(2)).sum::<f64>() / n_wheels as f64).sqrt();

    let quality = Quality {
        max_wheel_x: wheel_xs.iter().map(|x| x.abs()).fold(0.0, f64::max),
        wheel_x_pp: wheel_xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - wheel_xs.iter().cloned().fold(f64::INFINITY, f64::min),
        wheel_x_rms: rms_wx,
        wheel_y_span: wheel_ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - wheel_ys.iter().cloned().fold(f64::INFINITY, f64::min),
        mean_wheel_x: mean_wx,
        crossing,
    };

    // Jump report
    let jump_report = compute_jump_report(&wheel_ys, &hip_thetas, cfg);

    Solution {
        success: success && !crossing && !infeasible,
        cost,
        h_crouch: cfg.h_crouch,
        h_ext: cfg.h_ext,
        lengths: Lengths {
            upper_leg_hk: vars.lu,
            lower_leg_kw: vars.lkw,
            link_bc_c: vars.lc,
        },
        pin_joint: PinJointLocation { x: bc.x, y: bc.y },
        inner_joint_offset_kc: vars.lkc,
        jump_report,
        poses,
        quality,
        message: if infeasible {
            Some("Infeasible pose geometry".to_string())
        } else if crossing {
            Some("Crossing detected".to_string())
        } else {
            None
        },
    }
}

/// Compute jump kinematic report
fn compute_jump_report(wheel_ys: &[f64], hip_thetas: &[f64], cfg: &Config) -> JumpReport {
    if wheel_ys.len() < 2 {
        return JumpReport::default();
    }

    let mut js: Vec<f64> = Vec::new();

    for i in 0..(wheel_ys.len() - 1) {
        let dy = wheel_ys[i + 1] - wheel_ys[i];
        let dtheta = wrap_pi(hip_thetas[i + 1] - hip_thetas[i]);
        if dtheta.abs() >= 1e-9 {
            js.push(dy / dtheta);
        }
    }

    if js.is_empty() {
        return JumpReport::default();
    }

    let j_min = js.iter().cloned().fold(f64::INFINITY, f64::min);
    let j_max = js.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let j_start = js.first().cloned().unwrap_or(0.0);
    let j_end = js.last().cloned().unwrap_or(0.0);
    let theta_span = wrap_pi(hip_thetas.last().unwrap() - hip_thetas.first().unwrap()).abs();

    let y_dot_takeoff_est = cfg.omega_max.map(|omega| j_end * omega).unwrap_or(0.0);

    JumpReport {
        j_min,
        j_max,
        j_start,
        j_end,
        theta_span,
        y_dot_takeoff_est,
    }
}

/// Solve for a single pose given fixed linkage parameters
pub fn solve_pose_ratio(
    cfg: &Config,
    lengths: &Lengths,
    pin_joint: &PinJointLocation,
    inner_joint_offset_kc: f64,
    ratio: f64,
    x0: Option<&[f64]>,
) -> PoseSolveResult {
    let h = Vector2::new(0.0, 0.0);
    let bc = Vector2::new(pin_joint.x, pin_joint.y);
    let lu = lengths.upper_leg_hk;
    let lkw = lengths.lower_leg_kw;
    let lkc = inner_joint_offset_kc;
    let lc = lengths.link_bc_c;
    let target_y = height_for_ratio(cfg, ratio);

    let seed_kc = x0.and_then(|seed| {
        if seed.len() == 4 {
            Some((
                Vector2::new(seed[0], seed[1]),
                Vector2::new(seed[2], seed[3]),
            ))
        } else {
            None
        }
    });
    let mut preferred_c = seed_kc.map(|(_, c)| c);

    let steps = 720;
    let mut best: Option<(Vector2<f64>, Vector2<f64>, Vector2<f64>, f64)> = None;
    let mut prev: Option<(f64, Vector2<f64>, f64)> = None;
    let mut best_bracket: Option<(f64, f64, Vector2<f64>)> = None;

    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let theta = -PI + 2.0 * PI * t;
        if let Some((k, c, w, f)) =
            eval_pose_for_theta(theta, &bc, lu, lkc, lc, lkw, target_y, preferred_c.as_ref())
        {
            if best
                .as_ref()
                .map(|(_, _, _, bf)| f.abs() < bf.abs())
                .unwrap_or(true)
            {
                best = Some((k, c, w, f));
            }

            if let Some((prev_theta, prev_c, prev_f)) = prev.as_ref() {
                if prev_f.signum() != f.signum() {
                    best_bracket = Some((*prev_theta, theta, prev_c.clone()));
                }
            }

            prev = Some((theta, c.clone(), f));
            preferred_c = Some(c);
        }
    }

    let mut refined = None;
    if let Some((mut lo, mut hi, mut pref_c)) = best_bracket {
        let mut f_lo = eval_pose_for_theta(lo, &bc, lu, lkc, lc, lkw, target_y, Some(&pref_c))
            .map(|(_, c, _, f)| {
                pref_c = c;
                f
            })
            .unwrap_or(0.0);

        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if let Some((k, c, w, f_mid)) =
                eval_pose_for_theta(mid, &bc, lu, lkc, lc, lkw, target_y, Some(&pref_c))
            {
                if f_lo.signum() == f_mid.signum() {
                    lo = mid;
                    f_lo = f_mid;
                } else {
                    hi = mid;
                }
                pref_c = c;
                refined = Some((k, c, w, f_mid));
            } else {
                break;
            }
        }
    }

    let (k, c, w, f) = refined.or(best).unwrap_or((
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
        f64::INFINITY,
    ));

    let crossing = segments_intersect_strict(&h, &k, &bc, &c, 1e-9);
    let cost = f * f;
    let success = f.is_finite() && f.abs() < 1e-4 && !crossing;

    PoseSolveResult {
        success,
        cost,
        points: PosePoints { h, k, c, w, bc },
        crossing,
        seed: vec![k.x, k.y, c.x, c.y],
    }
}
