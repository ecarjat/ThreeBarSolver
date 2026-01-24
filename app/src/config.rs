use serde::{Deserialize, Serialize};

/// Configuration for the 3-bar linkage optimizer
/// Maps directly from Python Config dataclass (~45 fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Geometry
    pub h_crouch: f64,    // Crouched wheel height (m)
    pub h_ext: f64,       // Extended wheel height (m)
    pub ratios: Vec<f64>, // Pose sample ratios (0.0 to 1.0)

    // Weights for residuals
    pub w_len: f64,              // Length constraint weight
    pub w_pose: f64,             // Pose target weight
    pub w_wheel_x: f64,          // Wheel X deviation weight
    pub w_wheel_x_mean: f64,     // Mean wheel X weight
    pub w_knee_above_wheel: f64, // Knee above wheel constraint
    pub knee_above_margin: f64,  // Margin for knee constraint
    pub w_below: f64,            // Below-ground penalty
    pub w_reg: f64,              // Regularization weight

    // Multi-start / Global optimization
    pub n_starts: usize,      // Number of random starts
    pub use_global_opt: bool, // Use differential evolution
    pub global_maxiter: usize,
    pub global_popsize: usize,
    pub global_tol: f64,
    pub global_seed: Option<u64>,

    // Jumping / transmission shaping
    pub w_jac_profile: f64,   // Jacobian profile weight
    pub w_jac_bounds: f64,    // Jacobian bounds weight
    pub w_theta_span: f64,    // Theta span weight
    pub w_theta_monotonic: f64, // Theta monotonicity weight
    pub jac_start: f64,       // Target J at start (m/rad)
    pub jac_end: f64,         // Target J at end (m/rad)
    pub jac_min: f64,         // Min allowed J
    pub jac_max: f64,         // Max allowed J

    // Theta constraints
    pub theta_span_min: f64, // Min hip angle span (rad)
    pub theta_step_min: f64, // Min step between poses (rad)

    // Link constraints
    pub kc_min: f64,                 // Min inner joint offset KC (m)
    pub kc_max: Option<f64>,         // Max inner joint offset KC (m)
    pub cw_hk_ratio_min: Option<f64>, // Min (C-W)/(H-K) ratio
    pub cw_hk_ratio_max: Option<f64>, // Max (C-W)/(H-K) ratio
    pub w_cw_hk_ratio: f64,          // CW/HK ratio weight
    pub lc_hk_ratio_min: Option<f64>, // Min (Bc-C)/(H-K) ratio
    pub lc_hk_ratio_max: Option<f64>, // Max (Bc-C)/(H-K) ratio
    pub w_lc_hk_ratio: f64,          // LC/HK ratio weight

    // Pin joint constraints
    pub xbc_min: Option<f64>, // Min pin joint X (m)
    pub w_xbc_min: f64,       // XBC min weight
    pub w_bc_x: f64,          // Pin joint X bias weight
    pub w_bc_y: f64,          // Pin joint Y bias weight
    pub bc_radius_max: Option<f64>, // Max pin joint radius (m)
    pub w_bc_radius: f64,     // BC radius weight

    // Crossing prevention
    pub cross_min: f64,  // Min segment distance (m)
    pub w_no_cross: f64, // Crossing penalty weight

    // Knee angle constraint (angle H-K-W)
    pub max_angle_hkw: f64, // Max knee angle in radians (default: 170 deg = ~2.967 rad)
    pub w_angle_hkw: f64,   // Weight for angle constraint

    // Motor constraints
    pub omega_max: Option<f64>, // Max motor speed (rad/s)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            h_crouch: 0.35,
            h_ext: 0.65,
            ratios: vec![0.0, 0.5, 1.0],
            w_len: 250.0,
            w_pose: 900.0,
            w_wheel_x: 2000.0,
            w_wheel_x_mean: 200.0,
            w_knee_above_wheel: 800.0,
            knee_above_margin: 0.0,
            w_below: 400.0,
            w_reg: 1e-2,
            n_starts: 8,
            use_global_opt: false,
            global_maxiter: 400,
            global_popsize: 15,
            global_tol: 1e-4,
            global_seed: Some(1234),
            w_jac_profile: 120.0,
            w_jac_bounds: 120.0,
            w_theta_span: 200.0,
            w_theta_monotonic: 200.0,
            jac_start: 0.18,
            jac_end: 0.38,
            jac_min: 0.08,
            jac_max: 0.6,
            theta_span_min: 0.9,
            theta_step_min: 0.02,
            kc_min: 0.05,
            kc_max: Some(0.1),
            cw_hk_ratio_min: Some(0.9),
            cw_hk_ratio_max: Some(1.1),
            w_cw_hk_ratio: 600.0,
            lc_hk_ratio_min: Some(1.1),
            lc_hk_ratio_max: Some(1.1),
            w_lc_hk_ratio: 1200.0,
            xbc_min: Some(0.0),
            w_xbc_min: 800.0,
            w_bc_x: 1e-2,
            w_bc_y: 5e-2,
            bc_radius_max: Some(0.1),
            w_bc_radius: 800.0,
            cross_min: 0.001,
            w_no_cross: 200000.0,
            max_angle_hkw: 170.0_f64.to_radians(), // 170 degrees in radians
            w_angle_hkw: 800.0,
            omega_max: Some(11.0),
        }
    }
}

impl Config {
    /// Validate configuration constraints
    pub fn validate(&self) -> Result<(), String> {
        if self.ratios.is_empty() {
            return Err("ratios must be non-empty".to_string());
        }
        for (idx, ratio) in self.ratios.iter().enumerate() {
            if !ratio.is_finite() {
                return Err(format!("ratios[{}] must be a finite number", idx));
            }
            if *ratio < 0.0 || *ratio > 1.0 {
                return Err(format!(
                    "ratios[{}] must be within [0.0, 1.0] (got {})",
                    idx, ratio
                ));
            }
        }
        if self.n_starts < 1 {
            return Err("n_starts must be >= 1".to_string());
        }
        if let Some(kc_max) = self.kc_max {
            if self.kc_min > kc_max {
                return Err(format!(
                    "kc_min ({}) exceeds kc_max ({})",
                    self.kc_min, kc_max
                ));
            }
        }
        if let (Some(min), Some(max)) = (self.cw_hk_ratio_min, self.cw_hk_ratio_max) {
            if min > max {
                return Err("cw_hk_ratio_min exceeds cw_hk_ratio_max".to_string());
            }
        }
        if let (Some(min), Some(max)) = (self.lc_hk_ratio_min, self.lc_hk_ratio_max) {
            if min > max {
                return Err("lc_hk_ratio_min exceeds lc_hk_ratio_max".to_string());
            }
        }
        if self.jac_min > self.jac_max {
            return Err("jac_min exceeds jac_max".to_string());
        }
        if let (Some(bc_radius_max), Some(xbc_min)) = (self.bc_radius_max, self.xbc_min) {
            if bc_radius_max > 0.0 && xbc_min > bc_radius_max {
                return Err("xbc_min exceeds bc_radius_max".to_string());
            }
        }
        Ok(())
    }
}
