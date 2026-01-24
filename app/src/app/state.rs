use crate::config::Config;
use crate::types::{PoseSolveResult, Solution};

/// Application state
#[derive(Default)]
pub struct AppState {
    pub config: Config,
    pub solution: Option<Solution>,
    pub pose_seed: Option<Vec<f64>>,
    pub current_pose_ratio: f32,
    pub is_solving: bool,
    pub solver_error: Option<String>,
    pub cached_pose_ratio: Option<f32>,
    pub cached_pose: Option<PoseSolveResult>,

    // UI state for mm/m conversion (displayed in mm)
    pub h_crouch_mm: f32,
    pub h_ext_mm: f32,
    pub kc_min_mm: f32,
    pub kc_max_mm: f32,
    pub bc_radius_max_mm: f32,
    pub xbc_min_mm: f32,
    pub jac_start_mm: f32,
    pub jac_end_mm: f32,
    pub jac_min_mm: f32,
    pub jac_max_mm: f32,

    // Ratio selection
    pub ratio_0: bool,
    pub ratio_25: bool,
    pub ratio_50: bool,
    pub ratio_75: bool,
    pub ratio_100: bool,
}

impl AppState {
    pub fn new() -> Self {
        let cfg = Config::default();
        Self {
            h_crouch_mm: (cfg.h_crouch * 1000.0) as f32,
            h_ext_mm: (cfg.h_ext * 1000.0) as f32,
            kc_min_mm: (cfg.kc_min * 1000.0) as f32,
            kc_max_mm: (cfg.kc_max.unwrap_or(0.1) * 1000.0) as f32,
            bc_radius_max_mm: (cfg.bc_radius_max.unwrap_or(0.1) * 1000.0) as f32,
            xbc_min_mm: (cfg.xbc_min.unwrap_or(0.0) * 1000.0) as f32,
            jac_start_mm: (cfg.jac_start * 1000.0) as f32,
            jac_end_mm: (cfg.jac_end * 1000.0) as f32,
            jac_min_mm: (cfg.jac_min * 1000.0) as f32,
            jac_max_mm: (cfg.jac_max * 1000.0) as f32,
            ratio_0: true,
            ratio_25: false,
            ratio_50: true,
            ratio_75: false,
            ratio_100: true,
            config: cfg,
            solution: None,
            pose_seed: None,
            current_pose_ratio: 0.0,
            is_solving: false,
            solver_error: None,
            cached_pose_ratio: None,
            cached_pose: None,
        }
    }

    /// Sync config from UI values
    pub fn sync_config_from_ui(&mut self) {
        self.config.h_crouch = (self.h_crouch_mm / 1000.0) as f64;
        self.config.h_ext = (self.h_ext_mm / 1000.0) as f64;
        self.config.kc_min = (self.kc_min_mm / 1000.0) as f64;
        self.config.kc_max = if self.kc_max_mm > 0.0 {
            Some((self.kc_max_mm / 1000.0) as f64)
        } else {
            None
        };
        self.config.bc_radius_max = if self.bc_radius_max_mm > 0.0 {
            Some((self.bc_radius_max_mm / 1000.0) as f64)
        } else {
            None
        };
        self.config.xbc_min = Some((self.xbc_min_mm / 1000.0) as f64);
        self.config.jac_start = (self.jac_start_mm / 1000.0) as f64;
        self.config.jac_end = (self.jac_end_mm / 1000.0) as f64;
        self.config.jac_min = (self.jac_min_mm / 1000.0) as f64;
        self.config.jac_max = (self.jac_max_mm / 1000.0) as f64;

        // Build ratios from checkboxes
        let mut ratios = Vec::new();
        if self.ratio_0 {
            ratios.push(0.0);
        }
        if self.ratio_25 {
            ratios.push(0.25);
        }
        if self.ratio_50 {
            ratios.push(0.5);
        }
        if self.ratio_75 {
            ratios.push(0.75);
        }
        if self.ratio_100 {
            ratios.push(1.0);
        }
        if ratios.is_empty() {
            ratios = vec![0.0, 0.5, 1.0];
        }
        self.config.ratios = ratios;
    }

    /// Clear cached pose data so it is recomputed on next render.
    pub fn invalidate_pose_cache(&mut self) {
        self.cached_pose_ratio = None;
        self.cached_pose = None;
    }
}
