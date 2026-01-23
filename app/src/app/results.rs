use crate::app::state::AppState;
use egui::Ui;

/// Render the results panel
pub fn render_results(ui: &mut Ui, state: &mut AppState) {
    ui.heading("Design Output");

    if state.solution.is_none() {
        ui.label("Click 'Solve 3-bar design' to compute.");
        return;
    }

    let solution = state.solution.as_ref().unwrap();

    // Status
    ui.add_space(5.0);
    if solution.success {
        ui.colored_label(egui::Color32::GREEN, "Solution found!");
    } else {
        ui.colored_label(
            egui::Color32::RED,
            solution
                .message
                .as_deref()
                .unwrap_or("Optimization failed"),
        );
    }

    ui.add_space(5.0);
    ui.label(format!("Cost: {:.6e}", solution.cost));

    ui.separator();

    // Lengths (in mm)
    ui.collapsing("Lengths", |ui| {
        egui::Grid::new("lengths_grid")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Upper leg (H-K):");
                ui.label(format!("{:.2} mm", solution.lengths.upper_leg_hk * 1000.0));
                ui.end_row();

                ui.label("Lower leg (K-W):");
                ui.label(format!("{:.2} mm", solution.lengths.lower_leg_kw * 1000.0));
                ui.end_row();

                ui.label("Link (Bc-C):");
                ui.label(format!("{:.2} mm", solution.lengths.link_bc_c * 1000.0));
                ui.end_row();

                ui.label("Inner offset (K-C):");
                ui.label(format!(
                    "{:.2} mm",
                    solution.inner_joint_offset_kc * 1000.0
                ));
                ui.end_row();
            });
    });

    // Pin joint
    ui.collapsing("Pin Joint", |ui| {
        ui.label(format!("X: {:.2} mm", solution.pin_joint.x * 1000.0));
        ui.label(format!("Y: {:.2} mm", solution.pin_joint.y * 1000.0));
    });

    // Quality metrics
    ui.collapsing("Quality Metrics", |ui| {
        egui::Grid::new("quality_grid")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Max wheel X:");
                ui.label(format!(
                    "{:.3} mm",
                    solution.quality.max_wheel_x * 1000.0
                ));
                ui.end_row();

                ui.label("Wheel X P-P:");
                ui.label(format!("{:.3} mm", solution.quality.wheel_x_pp * 1000.0));
                ui.end_row();

                ui.label("Wheel X RMS:");
                ui.label(format!("{:.3} mm", solution.quality.wheel_x_rms * 1000.0));
                ui.end_row();

                ui.label("Wheel Y span:");
                ui.label(format!(
                    "{:.2} mm",
                    solution.quality.wheel_y_span * 1000.0
                ));
                ui.end_row();

                ui.label("Mean wheel X:");
                ui.label(format!(
                    "{:.3} mm",
                    solution.quality.mean_wheel_x * 1000.0
                ));
                ui.end_row();

                ui.label("Crossing:");
                ui.label(if solution.quality.crossing {
                    "Yes"
                } else {
                    "No"
                });
                ui.end_row();
            });
    });

    // Jump report
    ui.collapsing("Jump Report", |ui| {
        egui::Grid::new("jump_grid")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("J_start (mm/rad):");
                ui.label(format!(
                    "{:.2}",
                    solution.jump_report.j_start * 1000.0
                ));
                ui.end_row();

                ui.label("J_end (mm/rad):");
                ui.label(format!("{:.2}", solution.jump_report.j_end * 1000.0));
                ui.end_row();

                ui.label("J_min (mm/rad):");
                ui.label(format!("{:.2}", solution.jump_report.j_min * 1000.0));
                ui.end_row();

                ui.label("J_max (mm/rad):");
                ui.label(format!("{:.2}", solution.jump_report.j_max * 1000.0));
                ui.end_row();

                ui.label("Theta span (rad):");
                ui.label(format!("{:.3}", solution.jump_report.theta_span));
                ui.end_row();

                ui.label("Est. takeoff (m/s):");
                ui.label(format!("{:.3}", solution.jump_report.y_dot_takeoff_est));
                ui.end_row();
            });
    });

    // Pose coordinates
    ui.collapsing("Pose Coordinates", |ui| {
        egui::Grid::new("pose_grid")
            .num_columns(5)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Ratio");
                ui.label("K (mm)");
                ui.label("C (mm)");
                ui.label("W (mm)");
                ui.label("Crossing");
                ui.end_row();

                for pose in &solution.poses {
                    ui.label(format!("{:.2}", pose.ratio));
                    ui.label(format!(
                        "({:.1}, {:.1})",
                        pose.points.k.x * 1000.0,
                        pose.points.k.y * 1000.0
                    ));
                    ui.label(format!(
                        "({:.1}, {:.1})",
                        pose.points.c.x * 1000.0,
                        pose.points.c.y * 1000.0
                    ));
                    ui.label(format!(
                        "({:.1}, {:.1})",
                        pose.points.w.x * 1000.0,
                        pose.points.w.y * 1000.0
                    ));
                    ui.label(if pose.crossing { "Yes" } else { "No" });
                    ui.end_row();
                }
            });
    });

    ui.separator();

    // JSON export button
    if ui.button("Copy JSON to clipboard").clicked() {
        if let Ok(json) = serde_json::to_string_pretty(&solution) {
            ui.output_mut(|o| o.copied_text = json);
        }
    }
}
