use crate::app::state::AppState;
use crate::geometry::segments_intersect_strict;
use crate::linkage::{eval_pose_for_theta, height_for_ratio};
use egui::Ui;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Points};
use std::f64::consts::PI;
use std::env;

/// Render the linkage visualization plot
pub fn render_linkage_plot(ui: &mut Ui, state: &mut AppState) {
    if state.solution.is_none() {
        ui.label("Solve design first to view linkage.");
        return;
    }

    let solution = state.solution.as_ref().unwrap();

    // Pose slider
    ui.horizontal(|ui| {
        ui.label("Pose ratio:");
        ui.add(egui::Slider::new(&mut state.current_pose_ratio, 0.0..=1.0).step_by(0.01));
    });

    // Compute pose for current ratio (cached)
    let ratio = state.current_pose_ratio;
    let should_recompute = match state.cached_pose_ratio {
        Some(prev) => (prev - ratio).abs() > 1e-6,
        None => true,
    };

    if should_recompute {
        let pose = interpolate_pose_for_ratio(&state.config, solution, ratio);
        state.cached_pose_ratio = Some(ratio);
        state.cached_pose = Some(pose);
    }

    let pose = state.cached_pose.as_ref().expect("pose cache missing");
    let display_points = &pose.points;
    let display_crossing = pose.crossing;
    let trace_samples = 121;
    let w_trace = build_w_trace_points(&state.config, solution, trace_samples);

    // Convert to mm for display
    let scale = 1000.0;
    let h = [0.0, 0.0];
    let k = [display_points.k.x * scale, -display_points.k.y * scale]; // Negate Y for +Y up display
    let c = [display_points.c.x * scale, -display_points.c.y * scale];
    let w = [display_points.w.x * scale, -display_points.w.y * scale];
    let bc = [display_points.bc.x * scale, -display_points.bc.y * scale];

    let plot_bounds = compute_solution_bounds(
        &state.config,
        solution,
        Some(display_points),
        trace_samples,
        &w_trace,
    );
    state.plot_bounds = Some(plot_bounds);
    let should_set_bounds = true;

    // Plot
    let plot = Plot::new("linkage_plot")
        .width(400.0)
        .height(400.0)
        .data_aspect(1.0)
        .auto_bounds(true.into())
        .show_axes(true)
        .show_grid(true)
        .allow_zoom(true)
        .allow_drag(true);
    plot.show(ui, |plot_ui| {
        if should_set_bounds {
            plot_ui.set_plot_bounds(plot_bounds);
        }
            if w_trace.len() >= 2 {
                plot_ui.line(
                    Line::new(PlotPoints::from(w_trace))
                        .color(egui::Color32::from_rgb(0, 100, 0))
                        .width(2.0)
                        .name("W trace"),
                );
            }

            // H-K segment (upper leg) - blue
            plot_ui.line(
                Line::new(PlotPoints::from(vec![h, k]))
                    .color(egui::Color32::from_rgb(50, 100, 200))
                    .width(3.0)
                    .name("H-K (upper leg)"),
            );

            // K-C segment (inner joint link) - green
            plot_ui.line(
                Line::new(PlotPoints::from(vec![k, c]))
                    .color(egui::Color32::from_rgb(50, 180, 50))
                    .width(3.0)
                    .name("K-C (inner link)"),
            );

            // Bc-C segment (connecting rod) - red
            plot_ui.line(
                Line::new(PlotPoints::from(vec![bc, c]))
                    .color(egui::Color32::from_rgb(200, 50, 50))
                    .width(3.0)
                    .name("Bc-C (link)"),
            );

            // K-W segment (lower leg) - dark blue
            plot_ui.line(
                Line::new(PlotPoints::from(vec![k, w]))
                    .color(egui::Color32::from_rgb(30, 60, 150))
                    .width(3.0)
                    .name("K-W (lower leg)"),
            );

            // Points
            plot_ui.points(
                Points::new(vec![h])
                    .radius(8.0)
                    .color(egui::Color32::BLACK)
                    .name("H (hip)"),
            );
            plot_ui.points(
                Points::new(vec![k])
                    .radius(6.0)
                    .color(egui::Color32::from_rgb(100, 100, 255))
                    .name("K (knee)"),
            );
            plot_ui.points(
                Points::new(vec![c])
                    .radius(6.0)
                    .color(egui::Color32::from_rgb(100, 200, 100))
                    .name("C (inner joint)"),
            );
            plot_ui.points(
                Points::new(vec![w])
                    .radius(8.0)
                    .color(egui::Color32::DARK_GRAY)
                    .name("W (wheel)"),
            );
            plot_ui.points(
                Points::new(vec![bc])
                    .radius(6.0)
                    .color(egui::Color32::from_rgb(200, 100, 100))
                    .name("Bc (pin joint)"),
            );
    });

    if should_recompute && env::var("PLOT_DEBUG").ok().as_deref() == Some("1") {
        let min = plot_bounds.min();
        let max = plot_bounds.max();
        let bc_mm = [display_points.bc.x * scale, -display_points.bc.y * scale];
        let w_mm = [display_points.w.x * scale, -display_points.w.y * scale];
        let c_mm = [display_points.c.x * scale, -display_points.c.y * scale];
        eprintln!(
            "plot_debug ratio={:.3} bounds=([{:.2},{:.2}]..[{:.2},{:.2}]) Bc=({:.2},{:.2}) C=({:.2},{:.2}) W=({:.2},{:.2})",
            ratio,
            min[0],
            min[1],
            max[0],
            max[1],
            bc_mm[0],
            bc_mm[1],
            c_mm[0],
            c_mm[1],
            w_mm[0],
            w_mm[1]
        );
    }

    // Status directly under the diagram
    ui.add_space(6.0);
    if solution.success {
        ui.colored_label(
            egui::Color32::GREEN,
            format!("Solution found! Cost: {:.6e}", solution.cost),
        );
    } else {
        let message = solution
            .message
            .as_deref()
            .unwrap_or("Optimization failed");
        ui.colored_label(
            egui::Color32::RED,
            format!("{} Cost: {:.6e}", message, solution.cost),
        );
    }

    // Point coordinates table
    ui.add_space(10.0);
    ui.label("Point Coordinates (mm):");

    egui::Grid::new("point_coords")
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Point");
            ui.label("X");
            ui.label("Y");
            ui.end_row();

            let points = [
                ("H", display_points.h),
                ("K", display_points.k),
                ("C", display_points.c),
                ("W", display_points.w),
                ("Bc", display_points.bc),
            ];

            for (name, pt) in points {
                ui.label(name);
                ui.label(format!("{:.2}", pt.x * scale));
                ui.label(format!("{:.2}", -pt.y * scale));
                ui.end_row();
            }

            let kc_norm = (display_points.k - display_points.c).norm() * scale;
            ui.label("||K-C||");
            ui.label(format!("{:.2}", kc_norm));
            ui.label("-");
            ui.end_row();

            let theta = display_points.k.y.atan2(display_points.k.x);
            ui.label("Theta (deg)");
            ui.label(format!("{:.2}", theta.to_degrees()));
            ui.label("-");
            ui.end_row();

            let v_kh = display_points.h - display_points.k;
            let v_kc = display_points.c - display_points.k;
            let len_kh = v_kh.norm();
            let len_kc = v_kc.norm();
            let beta = if len_kh > 1e-9 && len_kc > 1e-9 {
                let cos_beta = (v_kh.dot(&v_kc) / (len_kh * len_kc)).clamp(-1.0, 1.0);
                Some(cos_beta.acos())
            } else {
                None
            };
            ui.label("Beta (deg)");
            ui.label(match beta {
                Some(b) => format!("{:.2}", b.to_degrees()),
                None => "-".to_string(),
            });
            ui.label("-");
            ui.end_row();

            let v_kw = display_points.w - display_points.k;
            let len_kw = v_kw.norm();
            let alpha = if len_kh > 1e-9 && len_kw > 1e-9 {
                let cos_alpha = (v_kh.dot(&v_kw) / (len_kh * len_kw)).clamp(-1.0, 1.0);
                Some(cos_alpha.acos())
            } else {
                None
            };
            ui.label("Alpha (deg)");
            ui.label(match alpha {
                Some(a) => format!("{:.2}", a.to_degrees()),
                None => "-".to_string(),
            });
            ui.label("-");
            ui.end_row();
        });

    if display_crossing {
        ui.add_space(5.0);
        ui.colored_label(
            egui::Color32::RED,
            "Warning: Crossing detected at this pose!",
        );
    }

    if !pose.success {
        ui.add_space(5.0);
        ui.colored_label(
            egui::Color32::DARK_RED,
            format!("Pose kinematics failed (cost: {:.3e})", pose.cost),
        );
    }
}

fn compute_solution_bounds(
    cfg: &crate::config::Config,
    solution: &crate::types::Solution,
    extra_pose: Option<&crate::types::PosePoints>,
    samples: usize,
    trace_points: &[[f64; 2]],
) -> PlotBounds {
    let scale = 1000.0;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    let mut extend = |x: f64, y: f64| {
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    };

    for pose in &solution.poses {
        let points = [pose.points.w, pose.points.c, pose.points.bc];
        for pt in points {
            let x = pt.x * scale;
            let y = -pt.y * scale;
            extend(x, y);
        }
    }

    // Always include the fixed pin joint
    extend(
        solution.pin_joint.x * scale,
        -solution.pin_joint.y * scale,
    );

    if samples >= 2 && !solution.poses.is_empty() {
        let min_ratio = solution
            .poses
            .iter()
            .map(|pose| pose.ratio)
            .fold(f64::INFINITY, f64::min);
        let max_ratio = solution
            .poses
            .iter()
            .map(|pose| pose.ratio)
            .fold(f64::NEG_INFINITY, f64::max);

        if min_ratio.is_finite() && max_ratio.is_finite() {
            let steps = samples.max(2);
            for i in 0..steps {
                let t = i as f64 / (steps - 1) as f64;
                let ratio = min_ratio + t * (max_ratio - min_ratio);
                let pose = interpolate_pose_for_ratio(cfg, solution, ratio);
                if pose.success {
                    let points = [pose.points.w, pose.points.c, pose.points.bc];
                    for pt in points {
                        let x = pt.x * scale;
                        let y = -pt.y * scale;
                        extend(x, y);
                    }
                }
            }
        }
    }

    if let Some(pose) = extra_pose {
        let points = [pose.w, pose.c, pose.bc];
        for pt in points {
            let x = pt.x * scale;
            let y = -pt.y * scale;
            extend(x, y);
        }
    }

    for pt in trace_points {
        extend(pt[0], pt[1]);
    }

    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return PlotBounds::new_symmetrical(100.0);
    }

    let width = (max_x - min_x).max(1.0);
    let height = (max_y - min_y).max(1.0);
    let pad = 0.1 * width.max(height);

    PlotBounds::from_min_max([min_x - pad, min_y - pad], [max_x + pad, max_y + pad])
}

fn interpolate_pose_for_ratio(
    cfg: &crate::config::Config,
    solution: &crate::types::Solution,
    ratio: f64,
 ) -> crate::types::PoseSolveResult {
    use crate::types::PoseSolveResult;
    use nalgebra::Vector2;

    let h = Vector2::new(0.0, 0.0);
    let bc = Vector2::new(solution.pin_joint.x, solution.pin_joint.y);
    let lu = solution.lengths.upper_leg_hk;
    let lkw = solution.lengths.lower_leg_kw;
    let lkc = solution.inner_joint_offset_kc;
    let lc = solution.lengths.link_bc_c;
    let target_y = height_for_ratio(cfg, ratio);

    let poses = &solution.poses;
    if poses.is_empty() {
        return PoseSolveResult {
            success: false,
            cost: f64::INFINITY,
            points: crate::types::PosePoints { h, k: h, c: h, w: h, bc },
            crossing: false,
            seed: Vec::new(),
        };
    }

    let (p0, p1, t) = if ratio <= poses[0].ratio {
        (&poses[0], &poses[0], 0.0)
    } else if ratio >= poses[poses.len() - 1].ratio {
        (&poses[poses.len() - 1], &poses[poses.len() - 1], 0.0)
    } else {
        let mut idx = 0;
        for i in 0..(poses.len() - 1) {
            if ratio >= poses[i].ratio && ratio <= poses[i + 1].ratio {
                idx = i;
                break;
            }
        }
        let r0 = poses[idx].ratio;
        let r1 = poses[idx + 1].ratio;
        let t = if (r1 - r0).abs() < 1e-12 {
            0.0
        } else {
            (ratio - r0) / (r1 - r0)
        };
        (&poses[idx], &poses[idx + 1], t)
    };

    let theta0 = p0.points.k.y.atan2(p0.points.k.x);
    let theta1 = p1.points.k.y.atan2(p1.points.k.x);
    let theta = if (p0.ratio - p1.ratio).abs() < 1e-12 {
        theta0
    } else {
        lerp_angle(theta0, theta1, t)
    };

    let preferred_c = p0.points.c * (1.0 - t) + p1.points.c * t;
    if let Some((k, c, w, f)) = eval_pose_for_theta(
        theta,
        &bc,
        lu,
        lkc,
        lc,
        lkw,
        target_y,
        Some(&preferred_c),
    ) {
        let crossing = segments_intersect_strict(&h, &k, &bc, &c, 1e-9);
        PoseSolveResult {
            success: f.is_finite(),
            cost: f * f,
            points: crate::types::PosePoints { h, k, c, w, bc },
            crossing,
            seed: vec![k.x, k.y, c.x, c.y],
        }
    } else {
        PoseSolveResult {
            success: false,
            cost: f64::INFINITY,
            points: crate::types::PosePoints {
                h,
                k: p0.points.k,
                c: p0.points.c,
                w: p0.points.w,
                bc,
            },
            crossing: p0.crossing,
            seed: Vec::new(),
        }
    }
}

fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    let mut delta = b - a;
    if delta > PI {
        delta -= 2.0 * PI;
    } else if delta < -PI {
        delta += 2.0 * PI;
    }
    a + t * delta
}

fn build_w_trace_points(
    cfg: &crate::config::Config,
    solution: &crate::types::Solution,
    samples: usize,
) -> Vec<[f64; 2]> {
    if solution.poses.is_empty() {
        return Vec::new();
    }

    let min_ratio = solution
        .poses
        .iter()
        .map(|pose| pose.ratio)
        .fold(f64::INFINITY, f64::min);
    let max_ratio = solution
        .poses
        .iter()
        .map(|pose| pose.ratio)
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_ratio.is_finite() || !max_ratio.is_finite() {
        return Vec::new();
    }

    let steps = samples.max(2);
    let mut points = Vec::with_capacity(steps);
    let scale = 1000.0;

    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        let ratio = min_ratio + t * (max_ratio - min_ratio);
        let pose = interpolate_pose_for_ratio(cfg, solution, ratio);
        if pose.success {
            let w = pose.points.w;
            points.push([w.x * scale, -w.y * scale]);
        }
    }

    points
}
