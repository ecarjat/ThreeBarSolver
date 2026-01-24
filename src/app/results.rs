use crate::app::state::AppState;
use egui::Ui;

/// Render the results panel
pub fn render_results(ui: &mut Ui, state: &mut AppState) {
    if state.solution.is_none() {
        ui.label("Click 'Solve 3-bar design' to compute.");
        return;
    }

    let solution = state.solution.as_ref().unwrap();

    // Lengths (in mm)
    egui::CollapsingHeader::new("Lengths")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(format!(
                "Upper leg (H-K): {:.2} mm",
                solution.lengths.upper_leg_hk * 1000.0
            ));
            ui.label(format!(
                "Lower leg (K-W): {:.2} mm",
                solution.lengths.lower_leg_kw * 1000.0
            ));
            ui.label(format!(
                "Link (Bc-C): {:.2} mm",
                solution.lengths.link_bc_c * 1000.0
            ));
            ui.label(format!(
                "Inner offset (K-C): {:.2} mm",
                solution.inner_joint_offset_kc * 1000.0
            ));
        });

    // Pin joint
    egui::CollapsingHeader::new("Pin Joint")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(format!("X: {:.2} mm", solution.pin_joint.x * 1000.0));
            ui.label(format!("Y: {:.2} mm", -solution.pin_joint.y * 1000.0));
        });

    // Quality metrics
    egui::CollapsingHeader::new("Quality Metrics")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(format!(
                "Max wheel X: {:.3} mm",
                solution.quality.max_wheel_x * 1000.0
            ));
            ui.label(format!(
                "Wheel X P-P: {:.3} mm",
                solution.quality.wheel_x_pp * 1000.0
            ));
            ui.label(format!(
                "Wheel X RMS: {:.3} mm",
                solution.quality.wheel_x_rms * 1000.0
            ));
            ui.label(format!(
                "Wheel Y span: {:.2} mm",
                solution.quality.wheel_y_span * 1000.0
            ));
            ui.label(format!(
                "Mean wheel X: {:.3} mm",
                solution.quality.mean_wheel_x * 1000.0
            ));
            ui.label(format!(
                "Crossing: {}",
                if solution.quality.crossing { "Yes" } else { "No" }
            ));
        });

    // Jump report
    egui::CollapsingHeader::new("Jump Report")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(format!(
                "J_start (mm/rad): {:.2}",
                solution.jump_report.j_start * 1000.0
            ));
            ui.label(format!(
                "J_end (mm/rad): {:.2}",
                solution.jump_report.j_end * 1000.0
            ));
            ui.label(format!(
                "J_min (mm/rad): {:.2}",
                solution.jump_report.j_min * 1000.0
            ));
            ui.label(format!(
                "J_max (mm/rad): {:.2}",
                solution.jump_report.j_max * 1000.0
            ));
            ui.label(format!(
                "Theta span (rad): {:.3}",
                solution.jump_report.theta_span
            ));
            ui.label(format!(
                "Est. takeoff (m/s): {:.3}",
                solution.jump_report.y_dot_takeoff_est
            ));
        });

    // Pose coordinates
    egui::CollapsingHeader::new("Pose Coordinates")
        .default_open(true)
        .show(ui, |ui| {
            for pose in &solution.poses {
                ui.label(format!(
                    "Ratio {:.2}: K ({:.1}, {:.1}) | C ({:.1}, {:.1}) | W ({:.1}, {:.1}) | Crossing {}",
                    pose.ratio,
                    pose.points.k.x * 1000.0,
                    -pose.points.k.y * 1000.0,
                    pose.points.c.x * 1000.0,
                    -pose.points.c.y * 1000.0,
                    pose.points.w.x * 1000.0,
                    -pose.points.w.y * 1000.0,
                    if pose.crossing { "Yes" } else { "No" }
                ));
            }
        });

    // JSON export button
    if ui.button("Copy JSON to clipboard").clicked() {
        if let Ok(json) = serde_json::to_string_pretty(&solution) {
            ui.output_mut(|o| o.copied_text = json);
        }
    }
}
