use crate::app::state::AppState;
use crate::optimization::solver::solve;
use egui::{DragValue, Ui};

/// Render the sidebar with all configuration controls
pub fn render_sidebar(ui: &mut Ui, state: &mut AppState) {
    ui.heading("3-Bar Linkage Designer");
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        // Geometry section
        ui.collapsing("Geometry", |ui| {
            ui.horizontal(|ui| {
                ui.label("Hcrouch (mm):");
                ui.add(
                    DragValue::new(&mut state.h_crouch_mm)
                        .speed(1.0)
                        .range(100.0..=500.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Hext (mm):");
                ui.add(
                    DragValue::new(&mut state.h_ext_mm)
                        .speed(1.0)
                        .range(300.0..=1000.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("kc_min (mm):");
                ui.add(
                    DragValue::new(&mut state.kc_min_mm)
                        .speed(1.0)
                        .range(10.0..=200.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("bc_radius_max (mm, 0=off):");
                ui.add(
                    DragValue::new(&mut state.bc_radius_max_mm)
                        .speed(1.0)
                        .range(0.0..=300.0),
                );
            });
        });

        // Link constraints section
        ui.collapsing("Link Constraints", |ui| {
            ui.horizontal(|ui| {
                ui.label("kc_max (mm, 0=off):");
                ui.add(
                    DragValue::new(&mut state.kc_max_mm)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("xbc_min (mm):");
                ui.add(
                    DragValue::new(&mut state.xbc_min_mm)
                        .speed(1.0)
                        .range(0.0..=100.0),
                );
            });

            // CW/HK ratio bounds
            let mut cw_min = state.config.cw_hk_ratio_min.unwrap_or(0.9) as f32;
            ui.horizontal(|ui| {
                ui.label("cw/hk ratio min:");
                if ui
                    .add(DragValue::new(&mut cw_min).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.cw_hk_ratio_min = if cw_min > 0.0 {
                        Some(cw_min as f64)
                    } else {
                        None
                    };
                }
            });

            let mut cw_max = state.config.cw_hk_ratio_max.unwrap_or(1.1) as f32;
            ui.horizontal(|ui| {
                ui.label("cw/hk ratio max:");
                if ui
                    .add(DragValue::new(&mut cw_max).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.cw_hk_ratio_max = if cw_max > 0.0 {
                        Some(cw_max as f64)
                    } else {
                        None
                    };
                }
            });

            // LC/HK ratio bounds
            let mut lc_min = state.config.lc_hk_ratio_min.unwrap_or(1.1) as f32;
            ui.horizontal(|ui| {
                ui.label("lc/hk ratio min:");
                if ui
                    .add(DragValue::new(&mut lc_min).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.lc_hk_ratio_min = if lc_min > 0.0 {
                        Some(lc_min as f64)
                    } else {
                        None
                    };
                }
            });

            let mut lc_max = state.config.lc_hk_ratio_max.unwrap_or(1.1) as f32;
            ui.horizontal(|ui| {
                ui.label("lc/hk ratio max:");
                if ui
                    .add(DragValue::new(&mut lc_max).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.lc_hk_ratio_max = if lc_max > 0.0 {
                        Some(lc_max as f64)
                    } else {
                        None
                    };
                }
            });

            let mut w_bc_x = state.config.w_bc_x as f32;
            ui.horizontal(|ui| {
                ui.label("w_bc_x:");
                if ui
                    .add(DragValue::new(&mut w_bc_x).speed(0.001).range(0.0..=1.0))
                    .changed()
                {
                    state.config.w_bc_x = w_bc_x as f64;
                }
            });

            let mut w_bc_y = state.config.w_bc_y as f32;
            ui.horizontal(|ui| {
                ui.label("w_bc_y:");
                if ui
                    .add(DragValue::new(&mut w_bc_y).speed(0.001).range(0.0..=1.0))
                    .changed()
                {
                    state.config.w_bc_y = w_bc_y as f64;
                }
            });

            // Knee angle constraint (H-K-W)
            ui.add_space(5.0);
            ui.label("Knee angle (H-K-W):");
            let mut max_angle_deg = state.config.max_angle_hkw.to_degrees() as f32;
            ui.horizontal(|ui| {
                ui.label("max_angle_hkw (deg):");
                if ui
                    .add(DragValue::new(&mut max_angle_deg).speed(1.0).range(90.0..=180.0))
                    .changed()
                {
                    state.config.max_angle_hkw = (max_angle_deg as f64).to_radians();
                }
            });
        });

        // Sampling section
        ui.collapsing("Sampling", |ui| {
            ui.label("Pose ratios:");
            ui.checkbox(&mut state.ratio_0, "0.0");
            ui.checkbox(&mut state.ratio_25, "0.25");
            ui.checkbox(&mut state.ratio_50, "0.5");
            ui.checkbox(&mut state.ratio_75, "0.75");
            ui.checkbox(&mut state.ratio_100, "1.0");
        });

        // Jumping profile section
        ui.collapsing("Jumping Profile", |ui| {
            ui.horizontal(|ui| {
                ui.label("jac_start (mm/rad):");
                ui.add(
                    DragValue::new(&mut state.jac_start_mm)
                        .speed(1.0)
                        .range(0.0..=500.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("jac_end (mm/rad):");
                ui.add(
                    DragValue::new(&mut state.jac_end_mm)
                        .speed(1.0)
                        .range(0.0..=500.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("jac_min (mm/rad):");
                ui.add(
                    DragValue::new(&mut state.jac_min_mm)
                        .speed(1.0)
                        .range(0.0..=200.0),
                );
            });

            ui.horizontal(|ui| {
                ui.label("jac_max (mm/rad):");
                ui.add(
                    DragValue::new(&mut state.jac_max_mm)
                        .speed(1.0)
                        .range(0.0..=1000.0),
                );
            });
        });

        // Theta constraints
        ui.collapsing("Theta Constraints", |ui| {
            let mut theta_span_min = state.config.theta_span_min as f32;
            ui.horizontal(|ui| {
                ui.label("theta_span_min (rad):");
                if ui
                    .add(
                        DragValue::new(&mut theta_span_min)
                            .speed(0.01)
                            .range(0.0..=3.14),
                    )
                    .changed()
                {
                    state.config.theta_span_min = theta_span_min as f64;
                }
            });

            let mut theta_step_min = state.config.theta_step_min as f32;
            ui.horizontal(|ui| {
                ui.label("theta_step_min (rad):");
                if ui
                    .add(
                        DragValue::new(&mut theta_step_min)
                            .speed(0.001)
                            .range(0.0..=0.5),
                    )
                    .changed()
                {
                    state.config.theta_step_min = theta_step_min as f64;
                }
            });
        });

        // Motor section
        ui.collapsing("Motor", |ui| {
            let mut omega = state.config.omega_max.unwrap_or(11.0) as f32;
            ui.horizontal(|ui| {
                ui.label("omega_max (rad/s, 0=off):");
                if ui
                    .add(DragValue::new(&mut omega).speed(0.1).range(0.0..=50.0))
                    .changed()
                {
                    state.config.omega_max = if omega > 0.0 {
                        Some(omega as f64)
                    } else {
                        None
                    };
                }
            });
        });

        // Optimization section
        ui.collapsing("Optimization", |ui| {
            let mut n_starts = state.config.n_starts as i32;
            ui.horizontal(|ui| {
                ui.label("n_starts:");
                if ui
                    .add(DragValue::new(&mut n_starts).speed(1).range(1..=32))
                    .changed()
                {
                    state.config.n_starts = n_starts as usize;
                }
            });

            ui.checkbox(&mut state.config.use_global_opt, "Use global optimization");
        });

        ui.separator();

        // Solve button
        let button_text = if state.is_solving {
            "Solving..."
        } else {
            "Solve 3-bar design"
        };

        ui.add_enabled_ui(!state.is_solving, |ui| {
            if ui.button(button_text).clicked() {
                state.sync_config_from_ui();
                state.solver_error = None;

                // Run solver synchronously (for simplicity)
                // In production, spawn a thread
                state.is_solving = true;
                match state.config.validate() {
                    Ok(()) => {
                        let solution = solve(&state.config);
                        state.solution = Some(solution);
                        state.pose_seed = None;
                        state.current_pose_ratio = 0.0;
                    }
                    Err(e) => {
                        state.solver_error = Some(e);
                    }
                }
                state.is_solving = false;
            }
        });

        if state.is_solving {
            ui.spinner();
        }

        if let Some(err) = &state.solver_error {
            ui.colored_label(egui::Color32::RED, err);
        }
    });
}
