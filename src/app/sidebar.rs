use crate::app::state::AppState;
use crate::optimization::solver::solve;
use egui::{DragValue, Ui};

/// Render the sidebar with all configuration controls
pub fn render_sidebar(ui: &mut Ui, state: &mut AppState) {
    ui.heading("3-Bar Linkage Designer");
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        let mut config_changed = false;

        // Geometry section
        ui.collapsing("Geometry", |ui| {
            ui.label("Target wheel heights:");

            ui.horizontal(|ui| {
                ui.label("Hcrouch (mm):")
                    .on_hover_text("Wheel height when leg is fully crouched (ratio=0)");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.h_crouch_mm)
                            .speed(1.0)
                            .range(100.0..=500.0),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("Hext (mm):")
                    .on_hover_text("Wheel height when leg is fully extended (ratio=1)");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.h_ext_mm)
                            .speed(1.0)
                            .range(300.0..=1000.0),
                    )
                    .changed();
            });
        });

        // Hard constraints section (must be satisfied - variable bounds)
        ui.collapsing("Hard Constraints", |ui| {
            ui.label("Strict bounds that cannot be violated.")
                .on_hover_text("Enforced via optimization variable bounds or geometric squashing");

            ui.add_space(5.0);
            ui.label("Inner joint offset (K-C):");

            ui.horizontal(|ui| {
                ui.label("kc_min (mm):")
                    .on_hover_text("Minimum distance from knee (K) to inner joint (C). Enforced as variable lower bound");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.kc_min_mm)
                            .speed(1.0)
                            .range(10.0..=200.0),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("kc_max (mm):")
                    .on_hover_text("Maximum distance from knee (K) to inner joint (C). Enforced as variable upper bound. 0 = no limit");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.kc_max_mm)
                            .speed(1.0)
                            .range(0.0..=200.0),
                    )
                    .changed();
            });

            ui.add_space(5.0);
            ui.label("Pin joint position:");

            ui.horizontal(|ui| {
                ui.label("bc_radius_max (mm):")
                    .on_hover_text("Maximum distance of pin joint (Bc) from hip. Enforced via tanh squashing. 0 = no limit");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.bc_radius_max_mm)
                            .speed(1.0)
                            .range(0.0..=300.0),
                    )
                    .changed();
            });
        });

        // Soft constraints section (weight-based optimization)
        ui.collapsing("Soft Constraints", |ui| {
            ui.label("Weighted penalties. Higher weight = stricter enforcement.")
                .on_hover_text("Soft constraints guide the optimizer but can be violated if other constraints conflict");

            ui.add_space(5.0);

            // Pin joint X minimum
            ui.label("Pin joint X minimum:");
            ui.horizontal(|ui| {
                ui.label("xbc_min (mm):")
                    .on_hover_text("Minimum X coordinate for pin joint (Bc)");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.xbc_min_mm)
                            .speed(1.0)
                            .range(0.0..=100.0),
                    )
                    .changed();
                ui.label("w:")
                    .on_hover_text("Weight for xbc_min constraint");
                let mut w = state.config.w_xbc_min as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_xbc_min = w as f64;
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Link length ratios (CW/HK):");

            // CW/HK ratio bounds
            let mut cw_min = state.config.cw_hk_ratio_min.unwrap_or(0.9) as f32;
            ui.horizontal(|ui| {
                ui.label("min:")
                    .on_hover_text("Min ratio of (Lkc + Lkw) / Lu. 0 = disabled");
                if ui
                    .add(DragValue::new(&mut cw_min).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.cw_hk_ratio_min = if cw_min > 0.0 {
                        Some(cw_min as f64)
                    } else {
                        None
                    };
                    config_changed = true;
                }
                let mut cw_max = state.config.cw_hk_ratio_max.unwrap_or(1.1) as f32;
                ui.label("max:")
                    .on_hover_text("Max ratio of (Lkc + Lkw) / Lu. 0 = disabled");
                if ui
                    .add(DragValue::new(&mut cw_max).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.cw_hk_ratio_max = if cw_max > 0.0 {
                        Some(cw_max as f64)
                    } else {
                        None
                    };
                    config_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("weight:")
                    .on_hover_text("Weight for CW/HK ratio constraint");
                let mut w = state.config.w_cw_hk_ratio as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_cw_hk_ratio = w as f64;
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Link length ratios (LC/HK):");

            // LC/HK ratio bounds
            let mut lc_min = state.config.lc_hk_ratio_min.unwrap_or(1.1) as f32;
            ui.horizontal(|ui| {
                ui.label("min:")
                    .on_hover_text("Min ratio of Lc / Lu. 0 = disabled");
                if ui
                    .add(DragValue::new(&mut lc_min).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.lc_hk_ratio_min = if lc_min > 0.0 {
                        Some(lc_min as f64)
                    } else {
                        None
                    };
                    config_changed = true;
                }
                let mut lc_max = state.config.lc_hk_ratio_max.unwrap_or(1.1) as f32;
                ui.label("max:")
                    .on_hover_text("Max ratio of Lc / Lu. 0 = disabled");
                if ui
                    .add(DragValue::new(&mut lc_max).speed(0.01).range(0.0..=2.0))
                    .changed()
                {
                    state.config.lc_hk_ratio_max = if lc_max > 0.0 {
                        Some(lc_max as f64)
                    } else {
                        None
                    };
                    config_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("weight:")
                    .on_hover_text("Weight for LC/HK ratio constraint");
                let mut w = state.config.w_lc_hk_ratio as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_lc_hk_ratio = w as f64;
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Wheel X bias:")
                .on_hover_text("Penalize mean wheel X offset from 0");

            ui.horizontal(|ui| {
                ui.label("w_wheel_x:")
                    .on_hover_text("Weight for aligning wheel X across poses");
                let mut w = state.config.w_wheel_x as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_wheel_x = w as f64;
                    config_changed = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("w_wheel_x_mean:")
                    .on_hover_text("Weight for keeping mean wheel X near 0");
                let mut w = state.config.w_wheel_x_mean as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_wheel_x_mean = w as f64;
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Pin joint bias:")
                .on_hover_text("Weights to keep pin joint near origin");

            ui.horizontal(|ui| {
                ui.label("w_bc_x:")
                    .on_hover_text("Weight for biasing pin joint X toward zero");
                let mut w_bc_x = state.config.w_bc_x as f32;
                if ui
                    .add(DragValue::new(&mut w_bc_x).speed(0.001).range(0.0..=1.0))
                    .changed()
                {
                    state.config.w_bc_x = w_bc_x as f64;
                    config_changed = true;
                }
                ui.label("w_bc_y:")
                    .on_hover_text("Weight for biasing pin joint Y toward zero");
                let mut w_bc_y = state.config.w_bc_y as f32;
                if ui
                    .add(DragValue::new(&mut w_bc_y).speed(0.001).range(0.0..=1.0))
                    .changed()
                {
                    state.config.w_bc_y = w_bc_y as f64;
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Knee angle (H-K-W):")
                .on_hover_text("Angle constraint at the knee joint");

            let mut max_angle_deg = state.config.max_angle_hkw.to_degrees() as f32;
            ui.horizontal(|ui| {
                ui.label("max_angle_hkw (deg):")
                    .on_hover_text("Maximum angle at knee between upper leg (H-K) and wheel direction (K-W). Penalized if exceeded");
                if ui
                    .add(DragValue::new(&mut max_angle_deg).speed(1.0).range(90.0..=180.0))
                    .changed()
                {
                    state.config.max_angle_hkw = (max_angle_deg as f64).to_radians();
                    config_changed = true;
                }
            });

            ui.add_space(5.0);
            ui.label("Pose=1 alpha target:")
                .on_hover_text("Target HKW angle at pose ratio = 1.0 (input in degrees)");

            ui.horizontal(|ui| {
                ui.label("alpha_target (deg):")
                    .on_hover_text("Desired HKW angle at pose=1; internally constrains theta");
                let mut alpha_deg = state.config.alpha_pose1_target.to_degrees() as f32;
                if ui
                    .add(DragValue::new(&mut alpha_deg).speed(1.0).range(0.0..=180.0))
                    .changed()
                {
                    state.config.alpha_pose1_target = (alpha_deg as f64).to_radians();
                    config_changed = true;
                }
            });

            ui.horizontal(|ui| {
                ui.label("w_theta_pose1:")
                    .on_hover_text("Weight for pose=1 theta target derived from alpha");
                let mut w = state.config.w_theta_pose1 as f32;
                if ui
                    .add(DragValue::new(&mut w).speed(10.0).range(0.0..=10000.0))
                    .changed()
                {
                    state.config.w_theta_pose1 = w as f64;
                    config_changed = true;
                }
            });
        });

        // Sampling section
        ui.collapsing("Sampling", |ui| {
            ui.label("Pose ratios:")
                .on_hover_text("Select which pose ratios to optimize. 0=crouched, 1=extended");
            config_changed |= ui
                .checkbox(&mut state.ratio_0, "0.0")
                .on_hover_text("Fully crouched pose")
                .changed();
            config_changed |= ui
                .checkbox(&mut state.ratio_25, "0.25")
                .on_hover_text("25% extended")
                .changed();
            config_changed |= ui
                .checkbox(&mut state.ratio_50, "0.5")
                .on_hover_text("Half extended")
                .changed();
            config_changed |= ui
                .checkbox(&mut state.ratio_75, "0.75")
                .on_hover_text("75% extended")
                .changed();
            config_changed |= ui
                .checkbox(&mut state.ratio_100, "1.0")
                .on_hover_text("Fully extended pose")
                .changed();
        });

        // Jumping profile section
        ui.collapsing("Jumping Profile", |ui| {
            ui.horizontal(|ui| {
                ui.label("jac_start (mm/rad):")
                    .on_hover_text("Target Jacobian (dy/dtheta) at start of stroke. Low value = slow initial extension");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.jac_start_mm)
                            .speed(1.0)
                            .range(0.0..=500.0),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("jac_end (mm/rad):")
                    .on_hover_text("Target Jacobian at end of stroke (takeoff). High value = fast extension at takeoff");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.jac_end_mm)
                            .speed(1.0)
                            .range(0.0..=500.0),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("jac_min (mm/rad):")
                    .on_hover_text("Minimum allowed Jacobian. Prevents near-singular configurations");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.jac_min_mm)
                            .speed(1.0)
                            .range(0.0..=200.0),
                    )
                    .changed();
            });

            ui.horizontal(|ui| {
                ui.label("jac_max (mm/rad):")
                    .on_hover_text("Maximum allowed Jacobian. Prevents excessive mechanical advantage");
                config_changed |= ui
                    .add(
                        DragValue::new(&mut state.jac_max_mm)
                            .speed(1.0)
                            .range(0.0..=1000.0),
                    )
                    .changed();
            });
        });

        // Theta constraints
        ui.collapsing("Theta Constraints", |ui| {
            let mut theta_span_min = state.config.theta_span_min.to_degrees() as f32;
            ui.horizontal(|ui| {
                ui.label("theta_span_min (deg):")
                    .on_hover_text("Minimum total hip angle rotation from crouch to extension. Ensures sufficient motor travel");
                if ui
                    .add(
                        DragValue::new(&mut theta_span_min)
                            .speed(1.0)
                            .range(0.0..=180.0),
                    )
                    .changed()
                {
                    state.config.theta_span_min = (theta_span_min as f64).to_radians();
                    config_changed = true;
                }
            });

            let mut theta_step_min = state.config.theta_step_min.to_degrees() as f32;
            ui.horizontal(|ui| {
                ui.label("theta_step_min (deg):")
                    .on_hover_text("Minimum angle change between adjacent poses. Prevents bunching of poses");
                if ui
                    .add(
                        DragValue::new(&mut theta_step_min)
                            .speed(0.1)
                            .range(0.0..=30.0),
                    )
                    .changed()
                {
                    state.config.theta_step_min = (theta_step_min as f64).to_radians();
                    config_changed = true;
                }
            });
        });

        // Motor section
        ui.collapsing("Motor", |ui| {
            let mut omega = state.config.omega_max.unwrap_or(11.0) as f32;
            ui.horizontal(|ui| {
                ui.label("omega_max (rad/s):")
                    .on_hover_text("Maximum motor angular velocity. Used to estimate takeoff velocity. 0 = disabled");
                if ui
                    .add(DragValue::new(&mut omega).speed(0.1).range(0.0..=50.0))
                    .changed()
                {
                    state.config.omega_max = if omega > 0.0 {
                        Some(omega as f64)
                    } else {
                        None
                    };
                    config_changed = true;
                }
            });
        });

        // Optimization section
        ui.collapsing("Optimization", |ui| {
            let mut n_starts = state.config.n_starts as i32;
            ui.horizontal(|ui| {
                ui.label("n_starts:")
                    .on_hover_text("Number of random starting points for multi-start optimization. More = better solutions but slower");
                if ui
                    .add(DragValue::new(&mut n_starts).speed(1).range(1..=128))
                    .changed()
                {
                    state.config.n_starts = n_starts as usize;
                    config_changed = true;
                }
            });

            config_changed |= ui
                .checkbox(&mut state.config.use_global_opt, "Use global optimization")
                .on_hover_text("Use Differential Evolution instead of multi-start. Better for complex problems but slower")
                .changed();
        });

        ui.separator();

        // Solve and Save buttons
        ui.horizontal(|ui| {
            let button_text = if state.is_solving {
                "Solving..."
            } else {
                "Solve"
            };

            ui.add_enabled_ui(!state.is_solving, |ui| {
                if ui
                    .button(button_text)
                    .on_hover_text("Run the optimizer to find linkage geometry")
                    .clicked()
                {
                    state.sync_config_from_ui();
                    state.solver_error = None;
                    state.save_message = None;

                    // Run solver synchronously (for simplicity)
                    // In production, spawn a thread
                    state.is_solving = true;
                    match state.config.validate() {
                        Ok(()) => {
                            let solution = solve(&state.config);
                            state.solution = Some(solution);
                            state.pose_seed = None;
                            state.current_pose_ratio = 0.0;
                            state.invalidate_pose_cache();
                        }
                        Err(e) => {
                            state.solver_error = Some(e);
                        }
                    }
                    state.is_solving = false;
                }
            });

            if ui
                .button("Save")
                .on_hover_text("Save current parameters to params.json")
                .clicked()
            {
                state.sync_config_from_ui();
                match state.save_config_to_file() {
                    Ok(()) => {
                        state.save_message = Some(("Saved to params.json".to_string(), true));
                    }
                    Err(e) => {
                        state.save_message = Some((e, false));
                    }
                }
            }
        });

        if state.is_solving {
            ui.spinner();
        }

        if let Some(err) = &state.solver_error {
            ui.colored_label(egui::Color32::RED, err);
        }

        // Show save message
        if let Some((msg, success)) = &state.save_message {
            let color = if *success {
                egui::Color32::GREEN
            } else {
                egui::Color32::RED
            };
            ui.colored_label(color, msg);
        }

        if config_changed {
            state.invalidate_pose_cache();
        }
    });
}
