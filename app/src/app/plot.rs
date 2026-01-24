use crate::app::state::AppState;
use crate::optimization::solver::solve_pose_ratio;
use egui::Ui;
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Points};

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

    // Solve pose for current ratio (cached)
    let ratio = state.current_pose_ratio;
    let should_recompute = match state.cached_pose_ratio {
        Some(prev) => (prev - ratio).abs() > 1e-6,
        None => true,
    };

    if should_recompute {
        let pose = solve_pose_ratio(
            &state.config,
            &solution.lengths,
            &solution.pin_joint,
            solution.inner_joint_offset_kc,
            ratio as f64,
            state.pose_seed.as_deref(),
        );

        // Update seed for continuation
        state.pose_seed = Some(pose.seed.clone());
        state.cached_pose_ratio = Some(ratio);
        state.cached_pose = Some(pose);
    }

    let pose = state.cached_pose.as_ref().expect("pose cache missing");

    // Convert to mm for display
    let scale = 1000.0;
    let h = [0.0, 0.0];
    let k = [pose.points.k.x * scale, -pose.points.k.y * scale]; // Negate Y for +Y up display
    let c = [pose.points.c.x * scale, -pose.points.c.y * scale];
    let w = [pose.points.w.x * scale, -pose.points.w.y * scale];
    let bc = [pose.points.bc.x * scale, -pose.points.bc.y * scale];

    let apply_bounds = state.plot_bounds.is_none();
    if apply_bounds {
        state.plot_bounds = Some(compute_solution_bounds(solution));
    }
    let plot_bounds = state.plot_bounds.unwrap();

    // Plot
    Plot::new("linkage_plot")
        .width(400.0)
        .height(400.0)
        .data_aspect(1.0)
        .auto_bounds(false.into())
        .show_axes(true)
        .show_grid(true)
        .allow_zoom(true)
        .allow_drag(true)
        .show(ui, |plot_ui| {
            if apply_bounds {
                plot_ui.set_plot_bounds(plot_bounds);
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
                ("H", pose.points.h),
                ("K", pose.points.k),
                ("C", pose.points.c),
                ("W", pose.points.w),
                ("Bc", pose.points.bc),
            ];

            for (name, pt) in points {
                ui.label(name);
                ui.label(format!("{:.2}", pt.x * scale));
                ui.label(format!("{:.2}", pt.y * scale));
                ui.end_row();
            }

            let kc_norm = (pose.points.k - pose.points.c).norm() * scale;
            ui.label("||K-C||");
            ui.label(format!("{:.2}", kc_norm));
            ui.label("-");
            ui.end_row();
        });

    if pose.crossing {
        ui.add_space(5.0);
        ui.colored_label(
            egui::Color32::RED,
            "Warning: Crossing detected at this pose!",
        );
    }

    if !pose.success {
        ui.add_space(5.0);
        ui.colored_label(
            egui::Color32::YELLOW,
            format!("Pose solve failed (cost: {:.3e})", pose.cost),
        );
    }
}

fn compute_solution_bounds(solution: &crate::types::Solution) -> PlotBounds {
    let scale = 1000.0;
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for pose in &solution.poses {
        let points = [
            pose.points.h,
            pose.points.k,
            pose.points.c,
            pose.points.w,
            pose.points.bc,
        ];
        for pt in points {
            let x = pt.x * scale;
            let y = -pt.y * scale;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return PlotBounds::new_symmetrical(100.0);
    }

    let width = (max_x - min_x).max(1.0);
    let height = (max_y - min_y).max(1.0);
    let pad = 0.1 * width.max(height);

    PlotBounds::from_min_max([min_x - pad, min_y - pad], [max_x + pad, max_y + pad])
}
