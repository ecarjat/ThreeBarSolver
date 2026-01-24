use crate::config::Config;
use crate::geometry::{dist, segments_intersect_strict, wrap_pi};
use crate::linkage::{bc_from_params, compute_wheel, has_crossing, height_for_ratio};
use crate::optimization::bounds::build_bounds;
use crate::optimization::packing::{pack_vars, unpack_vars};
use crate::optimization::problem::{PoseProblem, ThreeBarProblem};
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

/// Generate initial seed for multi-start optimization
pub fn generate_initial_seed(cfg: &Config, seed_idx: usize) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64((seed_idx * 1337 + 11) as u64);

    let scale = cfg.h_crouch.abs().max(cfg.h_ext.abs()).max(0.2);
    let mut poses = BTreeMap::new();

    for (i, ratio) in cfg.ratios.iter().enumerate() {
        let target_y = height_for_ratio(cfg, *ratio);
        let kx = (rng.gen::<f64>() - 0.5) * 0.2 * scale;
        let ky = target_y * (0.55 + 0.15 * rng.gen::<f64>());
        let c_offset = (0.15 + 0.25 * rng.gen::<f64>()) * scale;
        let cx = kx + (rng.gen::<f64>() - 0.5) * 0.1 * scale;
        let cy = (0.02 * scale).max(ky - c_offset);

        poses.insert(i, (Vector2::new(kx, ky), Vector2::new(cx, cy)));
    }

    // Reference pose for initial lengths
    let ref_idx = cfg
        .ratios
        .iter()
        .position(|&r| (r - 1.0).abs() < 1e-9)
        .unwrap_or(cfg.ratios.len() - 1);
    let ref_ratio = cfg.ratios[ref_idx];
    let (k_ref, c_ref) = poses.get(&ref_idx).unwrap();
    let w_ref = Vector2::new(0.0, height_for_ratio(cfg, ref_ratio));

    let h = Vector2::new(0.0, 0.0);
    let lu = dist(&h, k_ref);
    let lkc = dist(k_ref, c_ref);
    let lkw = (0.1 * scale).max(dist(k_ref, &w_ref));

    let xbc = (rng.gen::<f64>() - 0.5) * 0.6 * scale;
    let ybc = (rng.gen::<f64>() - 0.3) * 0.6 * scale;
    let bc = Vector2::new(xbc, ybc);
    let lc = dist(&bc, c_ref);

    pack_vars(&poses, lu, lkw, lkc, lc, xbc, ybc, cfg)
}

/// Local optimization using Nelder-Mead
pub fn solve_local(cfg: &Config, x0: Vec<f64>) -> (Vec<f64>, f64, bool) {
    let bounds = build_bounds(cfg);

    // Clamp to bounds
    let mut x0_clamped = x0;
    bounds.clamp(&mut x0_clamped);

    let problem = ThreeBarProblem::new(cfg.clone());

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
            let poses: Vec<_> = vars
                .poses
                .iter()
                .map(|(idx, (k, c))| (cfg.ratios[*idx], *k, *c))
                .collect();
            let crossing = has_crossing(&poses, &bc);

            (x_opt, cost, crossing)
        })
        .collect();

    // Find best non-crossing solution
    let best_non_crossing = results
        .iter()
        .filter(|(_, _, crossing)| !crossing)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    match best_non_crossing {
        Some((x, cost, _)) => (x.clone(), *cost),
        None => {
            // Fall back to best overall
            let best = results
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
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

    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(pop_size);
    for _ in 0..pop_size {
        let mut x = vec![0.0; n];
        for i in 0..n {
            let lo = bounds.lower[i];
            let hi = bounds.upper[i];
            x[i] = lo + rng.gen::<f64>() * (hi - lo);
        }
        pop.push(x);
    }

    let mut costs: Vec<f64> = pop.iter().map(|x| cost(x, cfg)).collect();
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
            let trial_cost = cost(&trial, cfg);
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
        solve_global_de(cfg)
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

    // Check crossing
    let poses_for_crossing: Vec<_> = vars
        .poses
        .iter()
        .map(|(idx, (k, c))| (cfg.ratios[*idx], *k, *c))
        .collect();
    let crossing = has_crossing(&poses_for_crossing, &bc);

    // Build poses
    let mut poses = Vec::new();
    let mut wheel_xs = Vec::new();
    let mut wheel_ys = Vec::new();
    let mut hip_thetas = Vec::new();

    for (idx, (k, c)) in &vars.poses {
        let ratio = cfg.ratios[*idx];
        let w = compute_wheel(k, c, vars.lkw);
        wheel_xs.push(w.x);
        wheel_ys.push(w.y);
        hip_thetas.push((k.y - h.y).atan2(k.x - h.x));

        let pose_crossing = segments_intersect_strict(&h, k, &bc, c, 1e-9);

        poses.push(Pose {
            ratio,
            target_wheel_y: height_for_ratio(cfg, ratio),
            points: PosePoints {
                h,
                k: *k,
                c: *c,
                w,
                bc,
            },
            crossing: pose_crossing,
        });
    }
    poses.sort_by(|a, b| a.ratio.partial_cmp(&b.ratio).unwrap());

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
        success: success && !crossing,
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
        message: if crossing {
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

    // Initial guess
    let x0_vec = match x0 {
        Some(seed) if seed.len() == 4 => seed.to_vec(),
        _ => {
            let kx = 0.0;
            let ky = target_y * 0.6;
            let cx = 0.1 * lu;
            let cy = ky - 0.4 * lu;
            vec![kx, ky, cx, cy]
        }
    };

    let problem = PoseProblem::new(cfg.clone(), bc, lu, lkc, lc, lkw, target_y);

    // Build simplex for Nelder-Mead
    let simplex: Vec<Vec<f64>> = (0..=4)
        .map(|i| {
            let mut p = x0_vec.clone();
            if i > 0 {
                p[i - 1] += 0.01;
            }
            p
        })
        .collect();

    let solver = match NelderMead::new(simplex).with_sd_tolerance(1e-10) {
        Ok(s) => s,
        Err(_) => {
            let cost = crate::optimization::residuals::pose_cost(
                &x0_vec, cfg, &bc, lu, lkc, lc, lkw, target_y,
            );
            return PoseSolveResult {
                success: false,
                cost,
                points: PosePoints {
                    h: Vector2::new(0.0, 0.0),
                    k: Vector2::new(x0_vec[0], x0_vec[1]),
                    c: Vector2::new(x0_vec[2], x0_vec[3]),
                    w: compute_wheel(
                        &Vector2::new(x0_vec[0], x0_vec[1]),
                        &Vector2::new(x0_vec[2], x0_vec[3]),
                        lkw,
                    ),
                    bc,
                },
                crossing: false,
                seed: x0_vec,
            };
        }
    };

    let result = Executor::new(problem, solver)
        .configure(|state: IterState<Vec<f64>, (), (), (), (), f64>| state.max_iters(20000))
        .run();

    let (best, cost, success) = match result {
        Ok(res) => {
            let state: &IterState<Vec<f64>, (), (), (), (), f64> = res.state();
            let best = state.best_param.clone().unwrap_or(x0_vec);
            let cost = state.best_cost;
            (best, cost, true)
        }
        Err(_) => {
            let cost = crate::optimization::residuals::pose_cost(
                &x0_vec, cfg, &bc, lu, lkc, lc, lkw, target_y,
            );
            (x0_vec, cost, false)
        }
    };

    let k = Vector2::new(best[0], best[1]);
    let c = Vector2::new(best[2], best[3]);
    let w = compute_wheel(&k, &c, lkw);
    let crossing = segments_intersect_strict(&h, &k, &bc, &c, 1e-9);

    PoseSolveResult {
        success: success && !crossing,
        cost,
        points: PosePoints { h, k, c, w, bc },
        crossing,
        seed: best,
    }
}
